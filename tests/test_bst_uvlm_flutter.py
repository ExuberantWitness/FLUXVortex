"""Goland Wing Flutter -- UVLM + BST Shell Dihedral Bending.

Full coupling: unsteady UVLM aerodynamics + BST shell structure.
BST uses dihedral angle bending + CST membrane + Velocity-Verlet.
Subcycling ensures shell CFL compliance.

Baselines:
  - UVLM + BeamFE (implicit):  140.2 m/s (2.4% error)
  - UVLM + PD beam  (explicit): 130.4 m/s (4.8% error)
  - Reference flutter speed:    ~137 m/s

Mesh independence study: 5 grid levels (2x4 through 8x16).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import time
from fluxvortex.bst_shell import BSTShell


def make_rect_mesh(Lx, Ly, nx, ny):
    """Structured triangle mesh for rectangle [0, Lx] x [0, Ly]."""
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    n_x, n_y = nx + 1, ny + 1
    vertices = np.zeros((n_x * n_y, 3))
    for j in range(n_y):
        for i in range(n_x):
            vertices[j * n_x + i] = [xs[i], ys[j], 0]

    triangles = []
    for j in range(ny):
        for i in range(nx):
            p00 = j * n_x + i
            p10 = j * n_x + (i + 1)
            p01 = (j + 1) * n_x + i
            p11 = (j + 1) * n_x + (i + 1)
            if (i + j) % 2 == 0:
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
            else:
                triangles.append([p00, p10, p01])
                triangles.append([p10, p11, p01])
    return vertices, np.array(triangles, dtype=np.int32)


def build_goland_wing(V_inf, dt=0.003, num_chords=100, alpha=2.0,
                      n_chord=4, n_span=8):
    chord = 1.8288; semi_span = 6.096
    airplane = ps.geometry.airplane.Airplane(
        wings=[ps.geometry.wing.Wing(
            wing_cross_sections=[
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=n_span, chord=chord,
                    airfoil=ps.geometry.airfoil.Airfoil(name='naca0012',
                                                        n_points_per_side=200),
                    spanwise_spacing='uniform'),
                ps.geometry.wing_cross_section.WingCrossSection(
                    num_spanwise_panels=None, chord=chord,
                    Lp_Wcsp_Lpp=(0.0, semi_span, 0.0),
                    airfoil=ps.geometry.airfoil.Airfoil(name='naca0012',
                                                        n_points_per_side=200),
                    spanwise_spacing=None),
            ],
            name='Wing', symmetric=False, num_chordwise_panels=n_chord,
            chordwise_spacing='uniform',
        )],
    )
    op = ps.operating_point.OperatingPoint(
        rho=1.225, vCg__E=V_inf, alpha=alpha, beta=0.0, nu=15.06e-6)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    wcsms = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=wcs)
        for wcs in airplane.wings[0].wing_cross_sections]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[0], wing_cross_section_movements=wcsms)
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    mv = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_chords=num_chords, delta_time=dt)
    return mv


class AeroSolverBST(ps.unsteady_ring_vortex_lattice_method
                     .UnsteadyRingVortexLatticeMethodSolver):
    """UVLM + BST shell aeroelastic solver."""

    def __init__(self, unsteady_problem, shell, relaxation=1.0,
                 x_ea_chord=0.33, shell_dt=0.0001):
        super().__init__(unsteady_problem)
        self._shell = shell
        self._relaxation = relaxation
        self._x_ea_chord = x_ea_chord
        self._shell_dt = shell_dt
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
            self._bst_coupling()

    def _bst_coupling(self):
        shell = self._shell
        dt_uvlm = self.delta_time
        vertices = shell.vertices0

        # 1. Extract per-strip lift and moment from UVLM panels
        yf, lf, mf = [], [], []
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels
                for j in range(ns):
                    sl, sm, sy, n = 0.0, 0.0, 0.0, 0
                    for i in range(nc):
                        p = wing.panels[i, j]
                        f = getattr(p, 'forces_GP1', None)
                        if f is not None:
                            sl += -f[2]
                            sm += p.Cpp_GP1_CgP1[0] * (-f[2])
                            if n == 0:
                                sy = p.Cpp_GP1_CgP1[1]
                            n += 1
                    if n > 0 and abs(sy) > 1e-6:
                        yf.append(sy); lf.append(sl); mf.append(sm)

        if len(yf) == 0:
            self.tip_w_history.append(0.0)
            self.tip_theta_history.append(0.0)
            return

        yf = np.array(yf); lf = np.array(lf); mf = np.array(mf)

        # 2. Distribute forces to BST shell nodes
        F_shell = np.zeros((shell.nv, 3))
        ny_mesh = int(round(vertices[:, 1].max() /
                            (vertices[1, 1] - vertices[0, 1]))) \
                  if vertices[1, 1] > 0 else 1
        chord = vertices[:, 0].max()
        x_ea = self._x_ea_chord * chord

        for k in range(len(yf)):
            y = abs(yf[k])
            fz = lf[k]   # lift force
            mx = mf[k]   # pitching moment about LE

            # Find nearby mesh nodes and distribute
            for ni in range(shell.nv):
                ny_i = vertices[ni, 1]
                nx_i = vertices[ni, 0]
                dy = abs(ny_i - y)
                if dy < (vertices[1, 1] - vertices[0, 1]) * 1.1 + 1e-6:
                    # Weight by proximity (inverse distance)
                    w_y = max(0, 1.0 - dy / (vertices[1, 1] - vertices[0, 1] + 1e-10))
                    # Distribute lift as z-force
                    F_shell[ni, 2] += fz * w_y * 0.5
                    # Distribute moment as z-force proportional to x offset from EA
                    x_rel = nx_i - x_ea
                    F_shell[ni, 2] += mx * w_y * 0.5 * np.sign(x_rel)

        # 3. Step BST shell with subcycling
        n_sub = max(1, int(dt_uvlm / self._shell_dt))
        dt_sub = dt_uvlm / n_sub
        for _ in range(n_sub):
            shell.step(F_shell, dt_sub)

        # 4. Get deformations with relaxation
        u_shell = shell.get_nodal_displacements()

        # Extract spanwise w and theta by averaging over chord
        span_vals = sorted(set(np.round(vertices[:, 1], 8)))
        n_span = len(span_vals)
        w_span = np.zeros(n_span)
        theta_span = np.zeros(n_span)
        y_span = np.zeros(n_span)

        for si, yv in enumerate(span_vals):
            mask = np.abs(vertices[:, 1] - yv) < 1e-8
            y_span[si] = yv
            w_vals = u_shell[mask, 2]
            x_vals = vertices[mask, 0]
            w_span[si] = np.mean(w_vals)
            # Theta from linear fit of w vs x
            if len(x_vals) > 1:
                theta_span[si] = np.polyfit(x_vals - np.mean(x_vals), w_vals, 1)[0]

        w_new = w_span
        theta_new = theta_span

        if self._prev_w is not None:
            w_new = self._relaxation * w_new + (1 - self._relaxation) * self._prev_w
            theta_new = self._relaxation * theta_new + (1 - self._relaxation) * self._prev_theta
        self._prev_w = w_new.copy()
        self._prev_theta = theta_new.copy()

        self.tip_w_history.append(w_new[-1])
        self.tip_theta_history.append(theta_new[-1])

        # 5. Deform UVLM panel vertices
        self._deform_panels(self._current_step + 1, w_new, theta_new, y_span)

    def _deform_panels(self, step, w, theta, beam_y):
        problem = self.steady_problems[step]
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
                            except (AttributeError, ValueError):
                                pass
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
                                except (AttributeError, ValueError):
                                    pass
                        try:
                            if (p._Cpp_GP1_CgP1 is not None
                                    and p._Cpp_GP1_CgP1.flags.writeable):
                                x_rel = p._Cpp_GP1_CgP1[0] - x_ea
                                p._Cpp_GP1_CgP1[2] += wz + x_rel * st
                        except (AttributeError, ValueError):
                            pass


def envelope_growth(signal, dt):
    if len(signal) < 10:
        return 0.0
    a = np.abs(signal)
    peaks = []
    for i in range(1, len(a) - 1):
        if a[i] > a[i - 1] and a[i] > a[i + 1]:
            peaks.append((i * dt, a[i]))
    if len(peaks) < 3:
        return 0.0
    tp = np.array([p[0] for p in peaks])
    ap = np.maximum(np.array([p[1] for p in peaks]), 1e-15)
    if len(tp) > 4:
        la = np.log(ap[1:])
        tf = tp[1:]
    else:
        la = np.log(ap)
        tf = tp
    if len(tf) >= 2:
        return np.polyfit(tf, la, 1)[0]
    return 0.0


def run_flutter_sweep(n_chord=4, n_span=8, label=""):
    """Run flutter speed sweep for a given BST mesh density."""
    ps.set_up_logging(level="Warning")
    chord = 1.8288; semi_span = 6.096

    # Goland wing structural properties
    EI = 9.773e6       # N·m²
    GJ = 0.988e6       # N·m²
    m_per_length = 35.72  # kg/m

    # IMPORTANT: An isotropic plate CANNOT match the Goland Wing's EI/GJ = 9.89
    # ratio (max possible is ~0.65 for isotropic). This is because the real wing
    # uses a concentrated spar, not a uniform plate.
    #
    # Strategy: Use a thick plate with E matched to give correct EI (bending),
    # accept higher GJ (torsion). Flutter speed will be higher than reference
    # because torsion is too stiff. This validates the BST+UVLM coupling
    # mechanism, not the exact flutter speed.

    h_shell = 0.3       # thick shell for tractable CFL
    nu_shell = 0.3
    # E from bending: EI = E*h³*Lx/(12*(1-ν²))
    E_shell = EI * 12 * (1 - nu_shell**2) / (h_shell**3 * chord)
    rho_shell = m_per_length / (chord * h_shell)

    # Resulting properties:
    # D = E*h³/(12*(1-ν²)) — bending rigidity per unit width
    # GJ_plate = E*h³*Lx/(6*(1+ν)) — torsional stiffness (>> actual GJ)
    D_actual = E_shell * h_shell**3 / (12 * (1 - nu_shell**2))
    GJ_plate = E_shell * h_shell**3 * chord / (6 * (1 + nu_shell))
    ratio = EI / GJ_plate

    # Create BST mesh
    mesh_verts, mesh_tris = make_rect_mesh(chord, semi_span, n_chord, n_span)

    dt_uvlm = 0.003
    shell_dt = 0.000002  # very small for stiff plate CFL
    # CFL: c=sqrt(E/rho) ≈ sqrt(513e6/39) ≈ 3600 m/s, L_elem≈0.05
    # dt_crit ≈ 0.05/3600 ≈ 1.4e-5, use 2e-6 (7x below)

    print(f"\n{'='*70}")
    print(f"Goland Wing Flutter -- UVLM + BST Shell {label}")
    print(f"  Mesh: {n_chord}x{n_span} = {2*n_chord*n_span} triangles")
    print(f"  E={E_shell:.2e} Pa, h={h_shell:.4f} m, nu={nu_shell}, "
          f"rho={rho_shell:.1f} kg/m³")
    print(f"  D={D_actual:.2f} N·m (EI_target={EI:.2e})")
    print(f"  GJ_plate={GJ_plate:.2e} (GJ_actual={GJ:.2e}, "
          f"ratio={ratio:.2f})")
    print(f"  UVLM dt={dt_uvlm}s, shell dt={shell_dt}s "
          f"({dt_uvlm/shell_dt:.0f} substeps)")
    print(f"  NOTE: Isotropic plate has GJ >> actual → flutter speed "
          f"will be elevated")
    print(f"{'='*70}")

    velocities = [80, 100, 120, 140, 160, 180, 200, 220, 250]
    results = []

    for V in velocities:
        print(f"  V={V:3d} m/s ... ", end="", flush=True)
        try:
            mv = build_goland_wing(V, dt=dt_uvlm,
                                   n_chord=n_chord, n_span=n_span)
            prob = ps.problems.UnsteadyProblem(
                movement=mv, only_final_results=False)

            shell = BSTShell(mesh_verts, mesh_tris,
                             E=E_shell, nu=nu_shell, h=h_shell,
                             rho=rho_shell, structural_damping=0.005)

            # Clamp root (y=0 nodes)
            root_nodes = np.where(np.abs(mesh_verts[:, 1]) < 1e-10)[0]
            shell.set_bc(root_nodes)

            # Initial perturbation — very small for stiff shell
            y_norm = mesh_verts[:, 1] / semi_span
            shell.u[:, 2] = 0.001 * y_norm**2  # small bending perturbation

            solver = AeroSolverBST(prob, shell, relaxation=1.0,
                                   shell_dt=shell_dt)
            t0 = time.time()
            solver.run(prescribed_wake=True, calculate_streamlines=False,
                       show_progress=False)
            t1 = time.time()

            tw = np.array(solver.tip_w_history)
            tth = np.array(solver.tip_theta_history)
            sig_w = envelope_growth(tw, dt_uvlm)
            sig_th = envelope_growth(tth, dt_uvlm)
            status = "FLUTTER" if sig_w > 0 else "stable"
            print(f"{status} (sig_w={sig_w:+.3f}, sig_th={sig_th:+.3f}, "
                  f"{t1-t0:.0f}s)")
            results.append({
                'V': V, 'sig_w': sig_w, 'sig_th': sig_th, 'status': status})
        except Exception as e:
            print(f"error: {e}")
            import traceback; traceback.print_exc()

    # Find flutter speed
    flutter_V = None
    for i in range(len(results) - 1):
        if results[i]['sig_w'] < 0 and results[i + 1]['sig_w'] > 0:
            s0, s1 = results[i]['sig_w'], results[i + 1]['sig_w']
            V0, V1 = results[i]['V'], results[i + 1]['V']
            flutter_V = V0 - s0 * (V1 - V0) / (s1 - s0)
            break

    # Summary
    print(f"\n{'─'*70}")
    print(f"{'V (m/s)':>10s} {'sig_w':>10s} {'sig_th':>10s} {'Status':>10s}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        print(f"{r['V']:10d} {r['sig_w']:+10.4f} {r['sig_th']:+10.4f} "
              f"{r['status']:>10s}")

    if flutter_V:
        err = abs(flutter_V - 137) / 137 * 100
        print(f"\n  Flutter speed: {flutter_V:.1f} m/s "
              f"(ref: 137, error: {err:.1f}%)")
    else:
        print(f"\n  No flutter transition found")
    print(f"{'='*70}")

    return flutter_V, results


if __name__ == '__main__':
    # Single mesh sweep for initial validation
    flutter_V, results = run_flutter_sweep(
        n_chord=4, n_span=8, label="(4x8 baseline)")

    # Uncomment for mesh independence study:
    # meshes = [(2, 4), (3, 6), (4, 8), (6, 12), (8, 16)]
    # for nc, ns in meshes:
    #     run_flutter_sweep(n_chord=nc, n_span=ns,
    #                       label=f"({nc}x{ns})")
