"""
Hybrid Panel-Particle Wake: Near-field ring vortex panels + far-field VPM particles.

Architecture:
  - PteraSoftware UVLM solver handles bound vortex + wake ring panels as usual
  - After N_keep wake rows, old ring panels are converted to VPM particles
  - Wake-wing influence = near-field panels (parent class) + far-field particles (VPM)
  - VPM particles in far-field can use free_wake for rollup dynamics

This gives:
  - Near-field accuracy of ring vortex panels (92-93% vs Theodorsen)
  - Far-field capabilities of VPM (free wake rollup, vortex stretching)
  - No feedback instability (panels are inherently stable near the wing)
"""
import sys, os, time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")
from fluxvortex.particles import VortexParticleField


# ─── Theodorsen ───
def theo_Ck(k):
    if k < 1e-10: return 1.0+0j
    return hankel2(1,k)/(hankel2(1,k)+1j*hankel2(0,k))

def theo_cl(k, h0c, omega, t, c=1.0):
    U = omega*c/(2*k) if k>1e-10 else 1e10
    h0=h0c*c
    return np.real(np.pi*(-h0*omega**2*np.sin(omega*t))*c/(2*U**2)
                   -2*np.pi*theo_Ck(k)*h0*omega*np.cos(omega*t)/U)

# ─── Wing builders ───
def make_wing(chord=1.0, half_span=5.0, nc=10, ns=6):
    return ps.geometry.wing.Wing(
        name="Wing", symmetric=True,
        symmetryNormal_G=(0,1,0), symmetryPoint_G_Cg=(0,0,0),
        num_chordwise_panels=nc, chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=ns, spanwise_spacing="uniform", chord=chord,
                Lp_Wcsp_Lpp=(0,0,0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None, spanwise_spacing=None, chord=chord,
                Lp_Wcsp_Lpp=(0,half_span,0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0)])

def make_plunge(wing, h0, period, V=10.0):
    ap = ps.geometry.airplane.Airplane(wings=[wing], name="P")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=0, beta=0)
    wcs=[ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=w) for w in wing.wing_cross_sections]
    wm=ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=wcs,
        ampLer_Gs_Cgs=(0,0,h0), periodLer_Gs_Cgs=(0,0,period),
        spacingLer_Gs_Cgs=("sine","sine","sine"), phaseLer_Gs_Cgs=(0,0,0))
    am=ps.movements.airplane_movement.AirplaneMovement(base_airplane=ap, wing_movements=[wm])
    opm=ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    return ap, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=3, delta_time=period/50)

def extract_cl(sol, mv):
    first=sol.unsteady_problem.first_results_step; dt=mv.delta_time
    ts,cls=[],[]
    for step in range(first, sol.num_steps):
        for ap in sol.steady_problems[step].airplanes:
            c=ap.forceCoefficients_W
            if c is not None: ts.append(step*dt); cls.append(-c[2])
    return np.array(ts), np.array(cls)

def get_Vvec(op):
    return op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                  np.sin(np.radians(op.beta)),
                                  -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])


# ─── Hybrid Panel-Particle Solver ───────────────────────────────────
class HybridSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    Near-field: ring vortex panels (handled by parent class).
    Far-field: VPM particles (converted from old wake rows).

    Strategy:
      1. Parent sheds ring vortex panels normally each timestep
      2. After N_keep rows, old panels are converted to VPM particles
      3. Panel strengths of converted rows are zeroed (prevent double-counting)
      4. Wake-wing influence = near-field panels + far-field VPM particles

    PteraSoftware wake indexing: row 0 = newest (near TE), higher = older.
    So we keep rows [0..N_keep-1] as panels, convert rows [N_keep..] to particles.

    Parameters:
        n_keep: number of wake rows to keep as ring vortex panels.
                0 = pure VPM particles
                large = pure ring panels (like PteraSoftware default)
        kernel: VPM kernel ('gaussianerf' or 'winckelmans')
    """
    def __init__(self, prob, n_keep=5, kernel='gaussianerf', free_vpm=False, **kw):
        super().__init__(prob)
        self._n_keep = n_keep
        self._free_vpm = free_vpm
        self._vpm = VortexParticleField(max_particles=200000, nu=0, rlxf=0.3, kernel=kernel)
        self._step_count = 0

    def _calculate_wake_wing_influences(self):
        """Panel influence (parent, far-field rows already zeroed) + VPM particle influence."""
        super()._calculate_wake_wing_influences()

        if self._vpm.np > 0:
            panel_inf = self._currentStackWakeWingInfluences__E.copy()
            cp = self.stackCpp_GP1_CgP1
            U = self._vpm.induce_velocity_at(cp)
            vpm_inf = np.einsum("ij,ij->i", U, self.stackUnitNormals_GP1)
            self._currentStackWakeWingInfluences__E = panel_inf + vpm_inf

    def _populate_next_airplanes_wake(self):
        """Standard panel wake + convert old rows to particles + advect particles."""
        super()._populate_next_airplanes_wake()
        self._step_count += 1

        # Convert old wake rows to particles (modify next step's airplanes)
        next_step = self._current_step + 1
        if next_step < self.num_steps:
            next_airplanes = self.steady_problems[next_step].airplanes
            self._convert_old_wake_to_particles(next_airplanes)

        # Advect VPM particles
        if self._vpm.np > 0:
            op = self.current_operating_point
            Vvec = get_Vvec(op)
            self._vpm.advect_rk3(
                self.delta_time,
                lambda X: np.broadcast_to(Vvec, X.shape).copy(),
                stretch=False, free_wake=self._free_vpm)

    def _convert_old_wake_to_particles(self, airplanes):
        """
        Convert wake rows beyond N_keep to VPM particles.
        Zero out their panel strengths to prevent double-counting.

        Each ring vortex is decomposed into 4 particles (one per leg):
          Front: FR→FL, Left: FL→BL (trailing), Back: BL→BR, Right: BR→FR (trailing)
        """
        op = self.current_operating_point
        Vvec = get_Vvec(op)
        V = np.linalg.norm(Vvec)
        if V < 1e-10: return
        sigma = V * self.delta_time

        pos_list, gam_list, sig_list = [], [], []

        for airplane in airplanes:
            for wing in airplane.wings:
                wake_vortices = wing.wake_ring_vortices
                if wake_vortices is None or wake_vortices.shape[0] == 0:
                    continue

                nw_chord = wake_vortices.shape[0]
                nw_span = wake_vortices.shape[1]

                if nw_chord <= self._n_keep:
                    continue

                for i in range(self._n_keep, nw_chord):
                    for j in range(nw_span):
                        rv = wake_vortices[i, j]
                        if rv is None: continue

                        strength = rv.strength
                        if abs(strength) < 1e-15: continue

                        fr = rv.Frrvp_GP1_CgP1
                        fl = rv.Flrvp_GP1_CgP1
                        bl = rv.Blrvp_GP1_CgP1
                        br = rv.Brrvp_GP1_CgP1

                        # 4 particles per ring: one per leg
                        legs = [
                            (0.5 * (fr + fl), strength * (fl - fr)),   # front leg
                            (0.5 * (fl + bl), strength * (bl - fl)),   # left trailing leg
                            (0.5 * (bl + br), strength * (br - bl)),   # back leg
                            (0.5 * (br + fr), strength * (fr - br)),   # right trailing leg
                        ]

                        for pos, gam in legs:
                            if np.dot(gam, gam) > 1e-30:
                                pos_list.append(pos)
                                gam_list.append(gam)
                                sig_list.append(sigma)

                        rv.strength = 0.0

        if pos_list:
            self._vpm.add_particles_batch(
                np.array(pos_list), np.array(gam_list), np.array(sig_list))


# ─── Experiment runner ───
def run_k(k, h0c=0.1, V=10.0, chord=1.0, free_vpm=False):
    omega = 2*k*V/chord; period = 2*np.pi/omega; h0 = h0c*chord

    print(f"\n{'='*70}")
    print(f"  k={k:.2f}  period={period:.4f}s  C(k)=|{abs(theo_Ck(k)):.4f}|∠{np.degrees(np.angle(theo_Ck(k))):.1f}°")
    print(f"{'='*70}")

    # Ring baseline
    print(f"  Ring-wake ... ", end="", flush=True)
    w1 = make_wing(chord); _, mv1 = make_plunge(w1, h0, period, V)
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t_r, cl_r = extract_cl(sol1, mv1)
    print("done")

    strategies = [
        ("Ring (all panels)", 'ring', {}),
        ("Hybrid N=10", 'hybrid', {'n_keep': 10, 'free_vpm': free_vpm}),
        ("Hybrid N=20", 'hybrid', {'n_keep': 20, 'free_vpm': free_vpm}),
        ("VPM-only (N=0)", 'vpm', {'free_vpm': free_vpm}),
    ]

    results = {}
    for label, mode, kw in strategies:
        print(f"  {label:<25s} ... ", end="", flush=True)
        w2 = make_wing(chord); _, mv2 = make_plunge(w2, h0, period, V)
        prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
        try:
            if mode == 'ring':
                sol2 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob2)
                sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
            elif mode == 'hybrid':
                sol2 = HybridSolver(prob2, **kw)
                sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
            elif mode == 'vpm':
                sol2 = HybridSolver(prob2, n_keep=0, **kw)
                sol2.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)

            t_v, cl_v = extract_cl(sol2, mv2)
            cl_max = np.max(np.abs(cl_v))

            if cl_max > 100:
                print(f"BLOWUP (|CL|={cl_max:.1e})")
                results[label] = {'status': 'BLOWUP', 'cl_max': cl_max}
                continue

            t_trans = 2 * period
            mr = t_r > t_trans; mv_ = t_v > t_trans
            n = min(np.sum(mr), np.sum(mv_))
            ring_amp = (np.max(cl_r[mr]) - np.min(cl_r[mr])) / 2
            vpm_amp = (np.max(cl_v[mv_]) - np.min(cl_v[mv_])) / 2
            corr = np.corrcoef(cl_r[mr][:n], cl_v[mv_][:n])[0, 1]
            rmse = np.sqrt(np.mean((cl_r[mr][:n] - cl_v[mv_][:n]) ** 2))
            ratio = vpm_amp / ring_amp if ring_amp > 1e-10 else 0
            np_count = getattr(sol2, '_vpm', None)
            np_count = np_count.np if np_count else 0
            print(f"amp={vpm_amp:.4f} ({ratio:.1%}) corr={corr:.4f} np={np_count}")
            results[label] = {'status': 'OK', 'amp': vpm_amp, 'ratio': ratio,
                              'corr': corr, 'rmse': rmse, 'np': np_count}
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results[label] = {'status': 'ERROR', 'msg': str(e)}

    # Theodorsen
    t_trans = 2 * period; mr = t_r > t_trans
    cl_th = theo_cl(k, h0c, omega, t_r, chord)
    ring_amp = (np.max(cl_r[mr]) - np.min(cl_r[mr])) / 2
    theo_amp = (np.max(cl_th[mr]) - np.min(cl_th[mr])) / 2
    ring_theo = ring_amp / theo_amp if theo_amp > 1e-10 else 0
    ring_corr = np.corrcoef(cl_r[mr], cl_th[mr])[0, 1]

    print(f"\n  --- k={k:.2f} Summary ---")
    print(f"  {'Method':<25s} {'Amp':>8s} {'vs Ring':>8s} {'vs Theo':>8s} {'Corr':>6s} {'Np':>6s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    print(f"  {'Theodorsen':<25s} {theo_amp:8.4f}")
    print(f"  {'Ring-wake':<25s} {ring_amp:8.4f} {'100.0%':>8s} {ring_theo:8.3f} {ring_corr:6.3f}")
    for label, _, _ in strategies:
        r = results[label]
        if r['status'] == 'OK':
            vt = r['amp'] / theo_amp if theo_amp > 1e-10 else 0
            print(f"  {label:<25s} {r['amp']:8.4f} {r['ratio']:8.1%} {vt:8.3f} {r['corr']:6.3f} {r.get('np',0):6d}")
        else:
            print(f"  {label:<25s} {r['status']:>8s}")

    return {'k': k, 't_r': t_r, 'cl_r': cl_r, 'cl_th': cl_th,
            'ring_amp': ring_amp, 'theo_amp': theo_amp, 'vpm': results}


# ─── Main ───
if __name__ == '__main__':
    print("=" * 70)
    print("Hybrid Panel-Particle Wake: Free VPM Wake Test")
    print("NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles")
    print("VPM particles use FREE WAKE (self-induction enabled)")
    print("=" * 70)

    all_res = []
    for k in [0.5, 0.2, 0.1]:
        r = run_k(k, free_vpm=True)
        all_res.append(('FREE', r))

    # Grand summary
    strategy_labels = ["Ring (all panels)", "Hybrid N=10", "Hybrid N=20", "VPM-only (N=0)"]

    print(f"\n{'='*70}")
    print("  Grand Summary: Free VPM Wake")
    print(f"{'='*70}")
    for tag, r in all_res:
        k = r['k']
        rt = r['ring_amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
        print(f"  [{tag}] k={k:.2f} | Ring/Theo={rt:.3f}")
        for label in strategy_labels:
            v = r['vpm'].get(label, {})
            if v.get('status') == 'OK':
                vt = v['amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
                print(f"    {label:<25s} Theo={vt:.3f} Ring={v['ratio']:.1%} c={v['corr']:.3f} np={v.get('np',0)}")
            else:
                print(f"    {label:<25s} {v.get('status','?')}")
    print(f"{'='*70}")
