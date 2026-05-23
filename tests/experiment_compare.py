"""
对比实验：涡格尾流 vs 涡粒子尾流 vs Theodorsen 解析解

三组求解器：
  1. Ring-wake: PteraSoftware 涡环面板尾涡 (baseline)
  2. VPM-wake:  涡粒子尾涡，前腿脱落 + kernel_factor 双向耦合
  3. Theory:    Theodorsen 解析解

对每个 reduced frequency k，对比 CL 时间历程、幅值比、相位差。
"""
import sys, os, time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")

from fluxvortex.particles import VortexParticleField

# ─── Theodorsen 解析解 ──────────────────────────────────────────
def theodorsen_Ck(k):
    if k < 1e-10: return 1.0 + 0j
    H0, H1 = hankel2(0, k), hankel2(1, k)
    return H1 / (H1 + 1j * H0)

def theodorsen_cl(k, h0c, omega, t, chord=1.0):
    U = omega * chord / (2 * k) if k > 1e-10 else 1e10
    h0 = h0c * chord
    hd = h0 * omega * np.cos(omega * t)
    hdd = -h0 * omega**2 * np.sin(omega * t)
    C = theodorsen_Ck(k)
    return np.real(np.pi * hdd * chord / (2 * U**2) - 2 * np.pi * C * hd / U)


# ─── Wing / Movement builders ───────────────────────────────────
def make_wing(chord=1.0, half_span=5.0, nc=10, ns=6):
    return ps.geometry.wing.Wing(
        name="Wing", symmetric=True,
        symmetryNormal_G=(0, 1, 0), symmetryPoint_G_Cg=(0, 0, 0),
        num_chordwise_panels=nc, chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=ns, spanwise_spacing="uniform", chord=chord,
                Lp_Wcsp_Lpp=(0, 0, 0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None, spanwise_spacing=None, chord=chord,
                Lp_Wcsp_Lpp=(0, half_span, 0),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75,
                control_surface_deflection=0.0),
        ])

def make_plunge(wing, h0, period, V=10.0):
    airplane = ps.geometry.airplane.Airplane(wings=[wing], name="P")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=0, beta=0)
    wcs_mov = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=wcs) for wcs in wing.wing_cross_sections]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=wcs_mov,
        ampLer_Gs_Cgs=(0, 0, h0), periodLer_Gs_Cgs=(0, 0, period),
        spacingLer_Gs_Cgs=("sine", "sine", "sine"), phaseLer_Gs_Cgs=(0, 0, 0))
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    return airplane, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=3, delta_time=period / 50)

def extract_cl(solver, mv):
    first = solver.unsteady_problem.first_results_step
    dt = mv.delta_time
    ts, cls = [], []
    for step in range(first, solver.num_steps):
        for ap in solver.steady_problems[step].airplanes:
            c = ap.forceCoefficients_W
            if c is not None:
                ts.append(step * dt)
                cls.append(-c[2])
    return np.array(ts), np.array(cls)


# ─── VPM 双向耦合求解器 (前腿脱落 + kernel_factor) ────────────
class VPMCoupledSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    双向耦合 VPM 求解器：VPM 粒子参与 wake-wing influence。
    kernel_factor 可调，用于校准实验。
    """
    def __init__(self, unsteady_problem, kernel_factor=1.0,
                 max_particles=50000, nu=0.0, rlxf=0.3,
                 stretch=False, free_wake=False):
        super().__init__(unsteady_problem)
        self._vpm = VortexParticleField(max_particles=max_particles, nu=nu, rlxf=rlxf)
        self._kf = kernel_factor
        self._stretch = stretch
        self._free_wake = free_wake

    # ── wake-wing influence: VPM 粒子 → 翼面 ──
    def _calculate_wake_wing_influences(self):
        if self._current_step == 0 or self._vpm.np == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels)
            return
        cp = self.stackCpp_GP1_CgP1
        U = self._vpm.induce_velocity_at(cp)
        self._currentStackWakeWingInfluences__E = np.einsum("ij,ij->i", U, self.stackUnitNormals_GP1)

    # ── wake 添加 + 粒子脱落 ──
    def _populate_next_airplanes_wake(self):
        if self._current_step > 0:
            self._shed()
            self._advect()
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed(self):
        strength = self._current_bound_vortex_strengths
        if strength is None:
            return
        op = self.current_operating_point
        Vvec = op.vCg__E * np.array([
            np.cos(np.radians(op.beta)) * np.cos(np.radians(op.alpha)),
            np.sin(np.radians(op.beta)),
            -np.cos(np.radians(op.beta)) * np.sin(np.radians(op.alpha)),
        ])
        V = np.linalg.norm(Vvec)
        if V < 1e-10: return
        infD = Vvec / V
        dt = self.delta_time
        dl = V * dt
        sigma = dl * 0.5
        te_off = infD * dl * 0.25

        pos_list, gam_list, sig_list = [], [], []
        idx = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]
                        g = strength[idx]; idx += 1
                        if not p.is_trailing_edge or abs(g) < 1e-15:
                            continue
                        bl = p.Blpp_GP1_CgP1
                        br = p.Brpp_GP1_CgP1
                        back = br - bl
                        pos_list.append(0.5 * (bl + br) + te_off)
                        gam_list.append(-back * g * self._kf)
                        sig_list.append(sigma)
        if pos_list:
            self._vpm.add_particles_batch(np.array(pos_list), np.array(gam_list), np.array(sig_list))

    def _advect(self):
        dt = self.delta_time
        op = self.current_operating_point
        V = op.vCg__E * np.array([
            np.cos(np.radians(op.beta)) * np.cos(np.radians(op.alpha)),
            np.sin(np.radians(op.beta)),
            -np.cos(np.radians(op.beta)) * np.sin(np.radians(op.alpha)),
        ])
        self._vpm.advect_rk3(dt, lambda X: np.broadcast_to(V, X.shape).copy(),
                              bound_velocity_func=None,
                              stretch=self._stretch, free_wake=self._free_wake)


# ─── 实验运行 ───────────────────────────────────────────────────
def run_experiment(k, h0c=0.1, V=10.0, chord=1.0, kf_values=None):
    omega = 2 * k * V / chord
    period = 2 * np.pi / omega
    h0 = h0c * chord

    if kf_values is None:
        kf_values = [0.5, 1.0, 1.587]

    print(f"\n{'='*70}")
    print(f"  k = {k:.2f}  |  period = {period:.4f}s  |  omega = {omega:.2f} rad/s")
    Ck = theodorsen_Ck(k)
    print(f"  C(k) = |{abs(Ck):.4f}| ∠{np.degrees(np.angle(Ck)):.1f}°")
    print(f"{'='*70}")

    # 1. Ring-wake baseline
    print(f"  Ring-wake ... ", end="", flush=True)
    w1 = make_wing(chord=chord)
    _, mv1 = make_plunge(w1, h0, period, V)
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    t0 = time.perf_counter()
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t_ring = time.perf_counter() - t0
    t_r, cl_r = extract_cl(sol1, mv1)
    print(f"{t_ring:.1f}s  steps={mv1.num_steps}  dt={mv1.delta_time:.5f}s")

    # 2. VPM-wake with different kernel_factors
    vpm_results = {}
    for kf in kf_values:
        label = f"VPM kf={kf:.3f}"
        print(f"  {label} ... ", end="", flush=True)
        w2 = make_wing(chord=chord)
        _, mv2 = make_plunge(w2, h0, period, V)
        prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
        sol2 = VPMCoupledSolver(prob2, kernel_factor=kf, max_particles=50000,
                                stretch=False, free_wake=False)
        try:
            t0 = time.perf_counter()
            sol2.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
            tv = time.perf_counter() - t0
            t_v, cl_v = extract_cl(sol2, mv2)
            np_count = sol2._vpm.np
            cl_max = np.max(np.abs(cl_v))

            if cl_max > 100:
                print(f"BLOWUP (|CL|_max={cl_max:.1e}, np={np_count})")
                vpm_results[kf] = {'status': 'BLOWUP', 'cl_max': cl_max}
            else:
                # Discard first 2 cycles
                t_trans = 2 * period
                n = min(np.sum(t_r > t_trans), np.sum(t_v > t_trans))
                mr = t_r > t_trans; mv_ = t_v > t_trans
                n = min(np.sum(mr), np.sum(mv_))

                ring_amp = (np.max(cl_r[mr]) - np.min(cl_r[mr])) / 2
                vpm_amp = (np.max(cl_v[mv_]) - np.min(cl_v[mv_])) / 2

                n_cmp = min(np.sum(mr), np.sum(mv_))
                corr = np.corrcoef(cl_r[mr][:n_cmp], cl_v[mv_][:n_cmp])[0, 1]
                rmse = np.sqrt(np.mean((cl_r[mr][:n_cmp] - cl_v[mv_][:n_cmp])**2))

                amp_ratio = vpm_amp / ring_amp if ring_amp > 1e-10 else 0
                print(f"{tv:.1f}s  amp={vpm_amp:.4f} ({amp_ratio:.1%} of ring) corr={corr:.4f} np={np_count}")
                vpm_results[kf] = {
                    'status': 'OK', 'amp': vpm_amp, 'amp_ratio': amp_ratio,
                    'corr': corr, 'rmse': rmse, 'cl_max': cl_max,
                    't': t_v, 'cl': cl_v, 'np': np_count
                }
        except Exception as e:
            print(f"ERROR: {e}")
            vpm_results[kf] = {'status': 'ERROR', 'msg': str(e)}

    # 3. Theodorsen theory
    cl_theo = theodorsen_cl(k, h0c, omega, t_r, chord)

    # ── Summary ──
    t_trans = 2 * period
    mr = t_r > t_trans
    ring_amp = (np.max(cl_r[mr]) - np.min(cl_r[mr])) / 2
    theo_amp = (np.max(cl_theo[mr]) - np.min(cl_theo[mr])) / 2

    print(f"\n  --- k={k:.2f} Summary ---")
    print(f"  {'Method':<20s} {'Amp':>8s} {'vs Ring':>8s} {'vs Theo':>8s} {'Corr':>6s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'Theodorsen':<20s} {theo_amp:8.4f} {'':>8s} {'':>8s} {'':>6s}")

    ring_theo_ratio = ring_amp / theo_amp if theo_amp > 1e-10 else 0
    ring_theo_corr = np.corrcoef(cl_r[mr], cl_theo[mr])[0, 1]
    print(f"  {'Ring-wake':<20s} {ring_amp:8.4f} {'1.000':>8s} {ring_theo_ratio:8.3f} {ring_theo_corr:6.3f}")

    for kf in kf_values:
        r = vpm_results[kf]
        if r['status'] == 'OK':
            vpm_theo = r['amp'] / theo_amp if theo_amp > 1e-10 else 0
            print(f"  {'VPM kf='+str(kf):<20s} {r['amp']:8.4f} {r['amp_ratio']:8.1%} {vpm_theo:8.3f} {r['corr']:6.3f}")
        else:
            print(f"  {'VPM kf='+str(kf):<20s} {r['status']:>8s}")

    return {
        'k': k, 't_ring': t_r, 'cl_ring': cl_r, 'cl_theo': cl_theo,
        'ring_amp': ring_amp, 'theo_amp': theo_amp, 'vpm': vpm_results,
    }


# ─── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("对比实验：涡格尾流 vs 涡粒子尾流 vs Theodorsen 解析解")
    print("NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles")
    print("VPM: front-leg shedding, prescribed wake, no stretch")
    print("=" * 70)

    # kernel_factor 候选值
    # 1.587 = 1/0.63 (纯前腿补偿，无高斯修正)
    # 1.0   = 无补偿 (仅前腿，欠估计)
    # 0.5   = 强欠估计
    kf_sweep = [0.5, 1.0, 1.587]

    all_results = []
    for k in [0.5, 0.2, 0.1]:
        r = run_experiment(k, kf_values=kf_sweep)
        all_results.append(r)

    # ── Grand summary ──
    print(f"\n{'='*70}")
    print("  Grand Summary")
    print(f"{'='*70}")
    print(f"  {'k':>5s} | {'Ring/Theo':>9s} | ", end="")
    for kf in kf_sweep:
        print(f"{'VPM kf='+format(kf,'.3f'):>14s} | ", end="")
    print()
    print(f"  {'─'*5}─┼─{'─'*9}─┼─" + "─"*16 + "┼─" * (len(kf_sweep)-1))

    for r in all_results:
        k = r['k']
        r_t = r['ring_amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
        print(f"  {k:5.2f} | {r_t:9.3f} | ", end="")
        for kf in kf_sweep:
            v = r['vpm'].get(kf, {})
            if v.get('status') == 'OK':
                ratio = v['amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
                print(f"{'amp='+format(ratio,'.3f'):>14s} | ", end="")
            else:
                print(f"{'['+v.get('status','?')+']':>14s} | ", end="")
        print()

    print(f"{'='*70}")
