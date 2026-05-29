"""Goland Wing Flutter — UVLM + PD Micro-Beam Bond (Zheng 2022).

Full coupling: unsteady UVLM aerodynamics + PD micro-beam bond structure.
PD beam uses Velocity-Verlet (real velocity state variable) with subcycling
to meet CFL constraints (dt_beam ≈ 0.0002s < dt_crit ≈ 0.00026s).

Designed as drop-in comparison to benchmark_goland.py (UVLM + BeamFE).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import time
from fluxvortex.pd_beam import PDBeam


def build_goland_wing(V_inf, dt=0.003, num_chords=100, alpha=2.0):
    chord = 1.8288; semi_span = 6.096
    airplane = ps.geometry.airplane.Airplane(
        wings=[ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=8, chord=chord,
                    airfoil=ps.geometry.airfoil.Airfoil(name='naca0012', n_points_per_side=200),
                    spanwise_spacing='uniform'),
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=None, chord=chord,
                    Lp_Wcsp_Lpp=(0.0, semi_span, 0.0),
                    airfoil=ps.geometry.airfoil.Airfoil(name='naca0012', n_points_per_side=200),
                    spanwise_spacing=None),
            ],
            name='Wing', symmetric=False, num_chordwise_panels=4,
            chordwise_spacing='uniform',
        )],
    )
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V_inf, alpha=alpha, beta=0.0, nu=15.06e-6)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    wcsms = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(base_wing_cross_section=wcs)
             for wcs in airplane.wings[0].wing_cross_sections]
    wm = ps.movements.wing_movement.WingMovement(base_wing=airplane.wings[0],
                                                   wing_cross_section_movements=wcsms)
    am = ps.movements.airplane_movement.AirplaneMovement(base_airplane=airplane, wing_movements=[wm])
    mv = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_chords=num_chords, delta_time=dt)
    return mv


class AeroSolverPD(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """UVLM + PD micro-beam bond solver (subclassed for _calculate_loads override)."""

    def __init__(self, unsteady_problem, beam, relaxation=1.0, x_ea_chord=0.33,
                 beam_dt=0.0002):
        super().__init__(unsteady_problem)
        self._beam = beam
        self._relaxation = relaxation
        self._x_ea_chord = x_ea_chord
        self._beam_dt = beam_dt
        self._prev_w = None
        self._prev_theta = None
        self.tip_w_history = []
        self.tip_theta_history = []

    def run(self, **kwargs):
        self.steady_problems = list(self.steady_problems)
        super().run(**kwargs)

    def _calculate_loads(self):
        super()._calculate_loads()
        if self._current_step >= 1 and self._current_step < self.num_steps - 1:
            self._pd_coupling()

    def _pd_coupling(self):
        beam = self._beam
        dt_uvlm = self.delta_time

        # 1. Extract per-strip lift and moment from UVLM panels
        yf, lf, mf = [], [], []
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                for j in range(ns):
                    sl, sm, sy, n = 0.0, 0.0, 0.0, 0
                    for i in range(nc):
                        p = wing.panels[i, j]
                        f = getattr(p, 'forces_GP1', None)
                        if f is not None:
                            sl += -f[2]; sm += p.Cpp_GP1_CgP1[0] * (-f[2])
                            if n == 0: sy = p.Cpp_GP1_CgP1[1]
                            n += 1
                    if n > 0 and abs(sy) > 1e-6:
                        yf.append(sy); lf.append(sl); mf.append(sm)

        if len(yf) == 0:
            self.tip_w_history.append(0.0); self.tip_theta_history.append(0.0)
            return

        yf = np.array(yf); lf = np.array(lf); mf = np.array(mf)

        # 2. Distribute forces to beam nodes
        F_beam = np.zeros(3 * beam.nnodes)
        for k in range(len(yf)):
            y = yf[k]
            idx = np.searchsorted(beam.y_nodes, y, side='right') - 1
            idx = max(0, min(idx, beam.nnodes - 2))
            Le = beam.y_nodes[idx+1] - beam.y_nodes[idx]
            xi = max(0.0, min((y - beam.y_nodes[idx]) / Le, 1.0))
            F_beam[3*idx]     += lf[k] * (1 - xi)
            F_beam[3*(idx+1)] += lf[k] * xi
            F_beam[3*idx+2]     += mf[k] * (1 - xi)
            F_beam[3*(idx+1)+2] += mf[k] * xi

        # 3. Step PD beam with subcycling
        n_sub = max(1, int(dt_uvlm / self._beam_dt))
        dt_sub = dt_uvlm / n_sub
        for _ in range(n_sub):
            beam.step(F_beam, dt_sub)

        # 4. Get deformations with relaxation
        w_new, theta_new = beam.get_nodal_displacements()
        if self._prev_w is not None:
            w_new = self._relaxation * w_new + (1 - self._relaxation) * self._prev_w
            theta_new = self._relaxation * theta_new + (1 - self._relaxation) * self._prev_theta
        self._prev_w = w_new.copy()
        self._prev_theta = theta_new.copy()

        self.tip_w_history.append(w_new[-1])
        self.tip_theta_history.append(theta_new[-1])

        # 5. Deform panel vertices for next step
        self._deform_panels(self._current_step + 1, w_new, theta_new)

    def _deform_panels(self, step, w, theta):
        problem = self.steady_problems[step]
        beam_y = self._beam.y_nodes
        for airplane in problem.airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                x_le = panels[0, 0].Frpp_GP1_CgP1[0]
                x_te = panels[nc-1, 0].Brpp_GP1_CgP1[0]
                chord = x_te - x_le
                x_ea = x_le + self._x_ea_chord * chord
                for i in range(nc):
                    for j in range(ns):
                        p = panels[i, j]
                        yp = abs(p.Cpp_GP1_CgP1[1])
                        wz = np.interp(yp, beam_y, w)
                        tz = np.interp(yp, beam_y, theta)
                        st, ct_m1 = np.sin(tz), np.cos(tz) - 1.0
                        for attr in ['_Frpp_GP1_CgP1', '_Flpp_GP1_CgP1',
                                     '_Blpp_GP1_CgP1', '_Brpp_GP1_CgP1']:
                            try:
                                v = getattr(p, attr)
                                if v is not None and v.flags.writeable:
                                    x_rel = v[0] - x_ea
                                    v[2] += wz + x_rel * st
                                    v[0] += x_rel * ct_m1
                            except (AttributeError, ValueError): pass
                        if p.ring_vortex is not None:
                            rv = p.ring_vortex
                            for attr in ['_Frrvp_GP1_CgP1', '_Flrvp_GP1_CgP1',
                                         '_Blrvp_GP1_CgP1', '_Brrvp_GP1_CgP1']:
                                try:
                                    v = getattr(rv, attr)
                                    if v is not None and v.flags.writeable:
                                        x_rel = v[0] - x_ea
                                        v[2] += wz + x_rel * st
                                        v[0] += x_rel * ct_m1
                                except (AttributeError, ValueError): pass
                        try:
                            if p._Cpp_GP1_CgP1 is not None and p._Cpp_GP1_CgP1.flags.writeable:
                                x_rel = p._Cpp_GP1_CgP1[0] - x_ea
                                p._Cpp_GP1_CgP1[2] += wz + x_rel * st
                        except (AttributeError, ValueError): pass


def envelope_growth(signal, dt):
    if len(signal) < 10: return 0.0
    a = np.abs(signal); peaks = []
    for i in range(1, len(a)-1):
        if a[i] > a[i-1] and a[i] > a[i+1]: peaks.append((i*dt, a[i]))
    if len(peaks) < 3: return 0.0
    tp = np.array([p[0] for p in peaks])
    ap = np.maximum(np.array([p[1] for p in peaks]), 1e-15)
    if len(tp) > 4: la = np.log(ap[1:]); tf = tp[1:]
    else: la = np.log(ap); tf = tp
    if len(tf) >= 2: return np.polyfit(tf, la, 1)[0]
    return 0.0


if __name__ == '__main__':
    ps.set_up_logging(level="Warning")
    chord = 1.8288; semi_span = 6.096

    beam_params = {
        'length': semi_span, 'n_elements': 8,
        'EI': 9.773e6, 'GJ': 0.988e6,
        'm_per_length': 35.72,
        'Ip': 35.72 * (chord**2) / 24,
        'x_ea_cg': 0.10 * chord,
        'structural_damping': 0.005,
    }
    beam_dt = 0.0001  # PD beam CFL: dt_crit ≈ 0.00026s, stay well below
    dt_uvlm = 0.003

    print("=" * 70)
    print("Goland Wing Flutter — UVLM + PD Micro-Beam Bond (Zheng 2022)")
    print(f"  UVLM dt={dt_uvlm}s, beam dt={beam_dt}s ({dt_uvlm/beam_dt:.0f} substeps)")
    print(f"  Beam: n_elements={beam_params['n_elements']}, EI={beam_params['EI']:.2e}, "
          f"GJ={beam_params['GJ']:.2e}")
    print("=" * 70)

    velocities = [80, 100, 110, 120, 130, 135, 140, 145, 150, 160, 180]
    results = []

    for V in velocities:
        print(f"  V={V:3d} m/s ... ", end="", flush=True)
        try:
            mv = build_goland_wing(V, dt=dt_uvlm)
            prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
            beam = PDBeam(**beam_params)
            # Initial perturbation — distributed spanwise (quadratic w, linear theta)
            # Using distributed perturbation reduces local curvature at tip,
            # which would otherwise cause overflow in explicit Velocity-Verlet
            y_norm = beam.y_nodes / semi_span
            beam.d[0::3] = 0.01 * y_norm**2       # w: quadratic (first bending mode approx)
            beam.d[2::3] = np.radians(1.0) * y_norm  # theta: linear (first torsion mode approx)
            beam.d[0:3] = 0.0; beam.v[0:3] = 0.0  # root BC

            solver = AeroSolverPD(prob, beam, relaxation=1.0, beam_dt=beam_dt)
            t0 = time.time()
            solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
            t1 = time.time()

            tw = np.array(solver.tip_w_history)
            tth = np.array(solver.tip_theta_history)
            sig_w = envelope_growth(tw, dt_uvlm)
            sig_th = envelope_growth(tth, dt_uvlm)
            status = "FLUTTER" if sig_w > 0 else "stable"
            print(f"{status} (sig_w={sig_w:+.3f}, sig_th={sig_th:+.3f}, {t1-t0:.0f}s)")
            results.append({'V': V, 'sig_w': sig_w, 'sig_th': sig_th, 'status': status})
        except Exception as e:
            print(f"error: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"{'V (m/s)':>10s} {'sig_w':>10s} {'sig_th':>10s} {'Status':>10s}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        print(f"{r['V']:10d} {r['sig_w']:+10.4f} {r['sig_th']:+10.4f} {r['status']:>10s}")

    # Find flutter speed
    flutter_V = None
    for i in range(len(results)-1):
        if results[i]['sig_w'] < 0 and results[i+1]['sig_w'] > 0:
            s0, s1 = results[i]['sig_w'], results[i+1]['sig_w']
            V0, V1 = results[i]['V'], results[i+1]['V']
            flutter_V = V0 - s0*(V1-V0)/(s1-s0)
            break

    if flutter_V:
        err = abs(flutter_V - 137)/137*100
        print(f"\n  Flutter speed: {flutter_V:.1f} m/s (ref: 137, error: {err:.1f}%)")
    else:
        print(f"\n  No flutter transition found")
    print(f"{'='*70}")
