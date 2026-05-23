"""
Strategy D revisited: Line vortex segment wake with proper singularity protection.

Each TE panel sheds one line segment (spanwise front leg) per timestep.
Segments are advected with freestream (prescribed wake).
Wake-wing influence uses the SAME Biot-Savart formula as PteraSoftware's ring vortices,
so the kernel is identical — only the geometry differs (individual legs vs complete rings).

Key fix over previous Strategy D:
  - Vectorized NumPy instead of Python loops
  - Proper core radius: rc² = rc0² + lamb*(squire*|Γ|+nu)*age
  - Initial rc0 proportional to panel size (dl * factor)
  - Multiple shedding modes: front-leg only, front+side legs, full ring
"""
import sys, os, time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")

# ─── Constants matching PteraSoftware ───
_FOUR_PI = 4.0 * np.pi
_EPS = 2.22e-16
_TOL = 1e-10
_LAMB = 1.25643  # = 4 * pi * 0.1 (Lamb-Oseen constant)
_SQUIRE = 1e-4   # Squire's instability term

_CHUNK = 256


# ─── Vectorized line vortex segment Biot-Savart ─────────────────────
def velocity_from_line_segments(targets, starts, ends, strengths,
                                rc0s=None, ages=None, nu=0.0):
    """
    Biot-Savart velocity from finite line vortex segments.
    Exact same formula as PteraSoftware's _collapsed_velocities_from_line_vortices.

    targets:    (M, 3)
    starts:     (N, 3)
    ends:       (N, 3)
    strengths:  (N,)
    rc0s:       (N,) initial core radii
    ages:       (N,) segment ages
    """
    M = targets.shape[0]
    N = starts.shape[0]
    U = np.zeros((M, 3))
    if N == 0:
        return U

    if rc0s is None:
        rc0s = np.full(N, 1e-7)
    if ages is None:
        ages = np.zeros(N)

    for i0 in range(0, M, _CHUNK):
        i1 = min(i0 + _CHUNK, M)
        tgt = targets[i0:i1]  # (Bt, 3)
        Bt = tgt.shape[0]

        for j0 in range(0, N, _CHUNK):
            j1 = min(j0 + _CHUNK, N)
            ss = starts[j0:j1]      # (Bs, 3)
            se = ends[j0:j1]        # (Bs, 3)
            sg = strengths[j0:j1]   # (Bs,)
            sr = rc0s[j0:j1]        # (Bs,)
            sa = ages[j0:j1]        # (Bs,)

            # r0 = end - start (filament vector)
            r0 = se[None, :, :] - ss[None, :, :]   # (1, Bs, 3) broadcast
            r0_sq = np.sum(r0 ** 2, axis=-1)         # (1, Bs)
            r0_len = np.sqrt(r0_sq)

            # r1 = start - target, r2 = end - target
            r1 = ss[None, :, :] - tgt[:, None, :]   # (Bt, Bs, 3)
            r2 = se[None, :, :] - tgt[:, None, :]   # (Bt, Bs, 3)

            # r3 = cross(r1, r2)
            r3 = np.cross(r1, r2)                    # (Bt, Bs, 3)
            r3_sq = np.sum(r3 ** 2, axis=-1)         # (Bt, Bs)

            r1_len = np.sqrt(np.sum(r1 ** 2, axis=-1))  # (Bt, Bs)
            r2_len = np.sqrt(np.sum(r2 ** 2, axis=-1))  # (Bt, Bs)

            # Core radius: rc² = rc0² + lamb*(squire*|Γ|+nu)*age
            rc_sq = sr[None, :] ** 2 + _LAMB * (_SQUIRE * np.abs(sg[None, :]) + nu) * sa[None, :]

            # ── Masks for degenerate cases ──
            r0_len_safe = np.maximum(r0_len, _EPS)
            ok_filament = r0_len > _EPS  # (1, Bs)

            r0_tol = r0_len_safe * _TOL
            ok_endpoints = (r1_len > r0_tol) & (r2_len > r0_tol)  # (Bt, Bs)

            r1r2 = r1_len * r2_len
            ok_cross = r3_sq > (_TOL * r1r2) ** 2  # (Bt, Bs)

            ok = ok_filament & ok_endpoints & ok_cross  # (Bt, Bs)
            if not np.any(ok):
                continue

            # ── Biot-Savart coefficient ──
            r1r2_safe = np.where(ok, r1r2, 1.0)
            r3_sq_safe = np.where(ok, r3_sq, 1.0)
            r0_sq_x_rc = r0_sq * rc_sq  # (1, Bs) regularization term

            c3 = np.sum(r1 * r2, axis=-1)  # dot(r1, r2)  # (Bt, Bs)
            r1_plus_r2 = r1_len + r2_len

            denom = r1r2_safe * (r3_sq_safe + r0_sq_x_rc)
            denom = np.where(ok & (denom > _EPS), denom, 1.0)

            c1 = sg[None, :] / _FOUR_PI  # (1, Bs)
            c4 = c1 * r1_plus_r2 * (r1r2_safe - c3) / denom  # (Bt, Bs)

            # velocity contribution: c4 * r3
            vel = c4[:, :, None] * r3  # (Bt, Bs, 3)
            vel *= ok[:, :, None]  # zero out degenerate cases

            U[i0:i1] += np.sum(vel, axis=1)

    return U


# ─── Line segment field ──────────────────────────────────────────────
class LineSegmentField:
    """Stores and manages line vortex segments for the wake."""

    def __init__(self, max_segments=200000, nu=0.0):
        self.max_segments = max_segments
        self.nu = nu
        self.ns = 0  # current segment count

        self._starts = np.zeros((max_segments, 3))
        self._ends = np.zeros((max_segments, 3))
        self._strengths = np.zeros(max_segments)
        self._rc0s = np.zeros(max_segments)
        self._ages = np.zeros(max_segments)

    def add_segments_batch(self, starts, ends, strengths, rc0s):
        n = starts.shape[0]
        if self.ns + n > self.max_segments:
            raise RuntimeError(f"Segment field overflow ({self.ns}+{n} > {self.max_segments})")
        sl = slice(self.ns, self.ns + n)
        self._starts[sl] = starts
        self._ends[sl] = ends
        self._strengths[sl] = strengths
        self._rc0s[sl] = rc0s
        self._ages[sl] = 0.0
        self.ns += n

    def induce_velocity_at(self, targets):
        if self.ns == 0:
            return np.zeros_like(targets)
        return velocity_from_line_segments(
            targets,
            self._starts[:self.ns],
            self._ends[:self.ns],
            self._strengths[:self.ns],
            self._rc0s[:self.ns],
            self._ages[:self.ns],
            self.nu,
        )

    def advect(self, dt, V_inf):
        """Advect all segments with freestream and age them."""
        if self.ns == 0:
            return
        shift = V_inf * dt
        self._starts[:self.ns] += shift
        self._ends[:self.ns] += shift
        self._ages[:self.ns] += dt


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

def get_inf(op, dt):
    Vvec = op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                  np.sin(np.radians(op.beta)),
                                  -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
    V = np.linalg.norm(Vvec)
    return Vvec, V, Vvec/max(V,1e-10), V*dt


# ─── Solver: Line segment wake (front-leg only) ──────────────────────
class LineSegSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """
    Two-way coupling with line vortex segments instead of Gaussian particles.
    Each TE panel sheds one spanwise line segment (front leg of wake ring).
    Uses the exact same Biot-Savart formula as PteraSoftware's ring vortices.
    """
    def __init__(self, prob, rc0_factor=0.01, shed_mode='front', **kw):
        super().__init__(prob)
        self._field = LineSegmentField(max_segments=200000, nu=0.0)
        self._rc0_factor = rc0_factor
        self._shed_mode = shed_mode  # 'front', 'front+side', 'full_ring'

    def _calculate_wake_wing_influences(self):
        if self._current_step == 0 or self._field.ns == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels)
            return
        cp = self.stackCpp_GP1_CgP1
        U = self._field.induce_velocity_at(cp)
        self._currentStackWakeWingInfluences__E = np.einsum("ij,ij->i", U, self.stackUnitNormals_GP1)

    def _populate_next_airplanes_wake(self):
        if self._current_step > 0:
            self._shed()
            self._advect()
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed(self):
        strength = self._current_bound_vortex_strengths
        if strength is None: return
        op = self.current_operating_point
        Vvec, V, infD, dl = get_inf(op, self.delta_time)
        if V < 1e-10: return

        rc0 = dl * self._rc0_factor

        starts_list, ends_list, str_list, rc_list = [], [], [], []
        idx = 0

        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                prev_te_data = None

                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]
                        g = strength[idx]; idx += 1
                        if not p.is_trailing_edge or abs(g) < 1e-15: continue

                        bl = p.Blpp_GP1_CgP1  # back-left
                        br = p.Brpp_GP1_CgP1  # back-right

                        if self._shed_mode == 'front':
                            # Front leg: BL → BR, strength = -gamma
                            starts_list.append(bl)
                            ends_list.append(br)
                            str_list.append(-g)
                            rc_list.append(rc0)

                        elif self._shed_mode == 'front+side':
                            # Front leg
                            starts_list.append(bl)
                            ends_list.append(br)
                            str_list.append(-g)
                            rc_list.append(rc0)

                            # Side legs (streamwise) connecting to previous row
                            if prev_te_data is not None:
                                prev_bl, prev_br = prev_te_data
                                # Left side: prev_BL → BL
                                starts_list.append(prev_bl)
                                ends_list.append(bl)
                                str_list.append(g)
                                rc_list.append(rc0)
                                # Right side: BR → prev_BR
                                starts_list.append(br)
                                ends_list.append(prev_br)
                                str_list.append(g)
                                rc_list.append(rc0)

                            prev_te_data = (bl.copy(), br.copy())

                        elif self._shed_mode == 'full_ring':
                            # Front leg at TE
                            starts_list.append(bl)
                            ends_list.append(br)
                            str_list.append(-g)
                            rc_list.append(rc0)

                            # Back leg at TE+dl (opposite direction)
                            bl_b = bl + infD * dl
                            br_b = br + infD * dl
                            starts_list.append(br_b)
                            ends_list.append(bl_b)
                            str_list.append(-g)  # same sign as front (ring topology)
                            rc_list.append(rc0)

                            # Left side: BL → BL_b
                            starts_list.append(bl)
                            ends_list.append(bl_b)
                            str_list.append(g)
                            rc_list.append(rc0)

                            # Right side: BR_b → BR
                            starts_list.append(br_b)
                            ends_list.append(br)
                            str_list.append(g)
                            rc_list.append(rc0)

        if starts_list:
            self._field.add_segments_batch(
                np.array(starts_list), np.array(ends_list),
                np.array(str_list), np.array(rc_list))

    def _advect(self):
        op = self.current_operating_point
        Vvec, V, infD, dl = get_inf(op, self.delta_time)
        self._field.advect(self.delta_time, Vvec)


# ─── Experiment runner ───
def run_k(k, h0c=0.1, V=10.0, chord=1.0):
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
    print(f"done")

    strategies = [
        # Front-leg only, different rc0
        ("front rc0=0.01dl", 'front', {'rc0_factor': 0.01}),
        ("front rc0=0.1dl",  'front', {'rc0_factor': 0.1}),
        ("front rc0=0.5dl",  'front', {'rc0_factor': 0.5}),
        # Front + side legs
        ("front+side rc0=0.1dl", 'front+side', {'rc0_factor': 0.1}),
        # Full ring (4 segments)
        ("full_ring rc0=0.1dl",  'full_ring',  {'rc0_factor': 0.1}),
        ("full_ring rc0=0.5dl",  'full_ring',  {'rc0_factor': 0.5}),
    ]

    results = {}
    for label, mode, kw in strategies:
        print(f"  {label:<25s} ... ", end="", flush=True)
        w2 = make_wing(chord); _, mv2 = make_plunge(w2, h0, period, V)
        prob2 = ps.problems.UnsteadyProblem(movement=mv2, only_final_results=False)
        try:
            sol2 = LineSegSolver(prob2, shed_mode=mode, **kw)
            sol2.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
            t_v, cl_v = extract_cl(sol2, mv2)
            ns_count = sol2._field.ns
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
            print(f"amp={vpm_amp:.4f} ({ratio:.1%}) corr={corr:.4f} ns={ns_count}")
            results[label] = {'status': 'OK', 'amp': vpm_amp, 'ratio': ratio,
                              'corr': corr, 'rmse': rmse, 'ns': ns_count}
        except Exception as e:
            print(f"ERROR: {e}")
            results[label] = {'status': 'ERROR', 'msg': str(e)}

    # Theodorsen
    t_trans = 2 * period; mr = t_r > t_trans
    cl_th = theo_cl(k, h0c, omega, t_r, chord)
    ring_amp = (np.max(cl_r[mr]) - np.min(cl_r[mr])) / 2
    theo_amp = (np.max(cl_th[mr]) - np.min(cl_th[mr])) / 2
    ring_theo = ring_amp / theo_amp if theo_amp > 1e-10 else 0
    ring_corr = np.corrcoef(cl_r[mr], cl_th[mr])[0, 1]

    print(f"\n  --- k={k:.2f} Summary ---")
    print(f"  {'Method':<25s} {'Amp':>8s} {'vs Ring':>8s} {'vs Theo':>8s} {'Corr':>6s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'Theodorsen':<25s} {theo_amp:8.4f}")
    print(f"  {'Ring-wake':<25s} {ring_amp:8.4f} {'100.0%':>8s} {ring_theo:8.3f} {ring_corr:6.3f}")
    for label, _, _ in strategies:
        r = results[label]
        if r['status'] == 'OK':
            vt = r['amp'] / theo_amp if theo_amp > 1e-10 else 0
            print(f"  {label:<25s} {r['amp']:8.4f} {r['ratio']:8.1%} {vt:8.3f} {r['corr']:6.3f}")
        else:
            print(f"  {label:<25s} {r['status']:>8s}")

    return {'k': k, 't_r': t_r, 'cl_r': cl_r, 'cl_th': cl_th,
            'ring_amp': ring_amp, 'theo_amp': theo_amp, 'vpm': results}


# ─── Main ───
if __name__ == '__main__':
    print("=" * 70)
    print("Line vortex segment wake: Strategy D revisited")
    print("NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles")
    print("Same Biot-Savart kernel as PteraSoftware ring vortices")
    print("=" * 70)

    all_res = []
    for k in [0.5, 0.2, 0.1]:
        r = run_k(k)
        all_res.append(r)

    # Grand summary
    strategy_labels = [l for l, _, _ in [
        ("front rc0=0.01dl", 'front', {'rc0_factor': 0.01}),
        ("front rc0=0.1dl",  'front', {'rc0_factor': 0.1}),
        ("front rc0=0.5dl",  'front', {'rc0_factor': 0.5}),
        ("front+side rc0=0.1dl", 'front+side', {'rc0_factor': 0.1}),
        ("full_ring rc0=0.1dl",  'full_ring',  {'rc0_factor': 0.1}),
        ("full_ring rc0=0.5dl",  'full_ring',  {'rc0_factor': 0.5}),
    ]]

    print(f"\n{'='*70}")
    print("  Grand Summary (3 cycles, discard first 2)")
    print(f"{'='*70}")
    header = f"  {'k':>5s} | {'Ring/Theo':>9s} |"
    for s in strategy_labels:
        header += f" {s[:15]:>15s} |"
    print(header)
    sep = f"  {'─'*5}─┼─{'─'*9}─┼─" + "─"*17 + "┼─" * (len(strategy_labels) - 1)
    print(sep)

    for r in all_res:
        k = r['k']
        rt = r['ring_amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
        line = f"  {k:5.2f} | {rt:9.3f} |"
        for label in strategy_labels:
            v = r['vpm'].get(label, {})
            if v.get('status') == 'OK':
                vt = v['amp'] / r['theo_amp'] if r['theo_amp'] > 1e-10 else 0
                line += f" {vt:6.3f} c={v['corr']:.2f} |"
            else:
                line += f" {'[' + v.get('status', '?') + ']':>15s} |"
        print(line)
    print(f"{'='*70}")
