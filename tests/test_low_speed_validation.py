"""Low-speed (<50 m/s) aeroelastic validation for BST IBM + torsion shell.

Validates:
  1. Natural frequencies (bending ~7.88 Hz, torsion ~81.17 Hz)
  2. Aeroelastic equilibrium at V=30 m/s (compare tip deflection with beam theory)
  3. Dynamic response: impulse decay rate and oscillation frequency
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import time
from fluxvortex.bst_shell import BSTShell
from fluxvortex.bst_implicit_gpu import BSTImplicitGPU


def make_rect_mesh(Lx, Ly, nx, ny):
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


class AeroSolverImplicit(ps.unsteady_ring_vortex_lattice_method
                          .UnsteadyRingVortexLatticeMethodSolver):
    def __init__(self, unsteady_problem, shell, implicit_solver,
                 relaxation=1.0, x_ea_chord=0.33):
        super().__init__(unsteady_problem)
        self._shell = shell
        self._implicit = implicit_solver
        self._relaxation = relaxation
        self._x_ea_chord = x_ea_chord
        self._prev_w = None
        self._prev_theta = None
        self.tip_w_history = []
        self.tip_theta_history = []
        self.newton_iters = []
        self.strip_forces_history = []

    def run(self, **kwargs):
        self.steady_problems = list(self.steady_problems)
        super().run(**kwargs)

    def _calculate_loads(self):
        super()._calculate_loads()
        if self._current_step >= 1 and self._current_step < self.num_steps - 1:
            self._implicit_coupling()

    def _implicit_coupling(self):
        shell = self._shell
        dt_uvlm = self.delta_time
        vertices = shell.vertices0

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

        F_shell = np.zeros((shell.nv, 3))
        chord_val = vertices[:, 0].max()
        x_ea = self._x_ea_chord * chord_val
        dy_mesh = vertices[1, 1] - vertices[0, 1] if vertices[1, 1] > 0 else 1.0

        total_lift = 0.0
        total_moment = 0.0

        for k in range(len(yf)):
            y = abs(yf[k])
            fz = lf[k]
            mx = mf[k]
            strip_mask = np.abs(vertices[:, 1] - y) < dy_mesh * 0.6
            strip_idx = np.where(strip_mask)[0]
            if len(strip_idx) == 0:
                continue
            total_lift += fz
            total_moment += mx

            # Corrected moment about elastic axis
            x_mean_strip = np.mean(vertices[strip_idx, 0])
            mx_corrected = mx - fz * x_mean_strip

            F_shell[strip_idx, 2] += fz / len(strip_idx)
            for ni in strip_idx:
                x_rel = vertices[ni, 0] - x_ea
                x_rels = vertices[strip_idx, 0] - x_ea
                sum_x2 = np.sum(x_rels**2)
                if sum_x2 > 1e-20:
                    F_shell[ni, 2] += mx_corrected * x_rel / sum_x2

        self.strip_forces_history.append((total_lift, total_moment))

        n_iter, r_norm = self._implicit.step(F_shell, dt_uvlm)
        self.newton_iters.append(n_iter)

        u_shell = shell.get_nodal_displacements()
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
        self._deform_panels(self._current_step + 1, w_new, theta_new, y_span)

    def _deform_panels(self, step, w, theta, beam_y):
        problem = self.steady_problems[step]
        for airplane in problem.airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                x_le = panels[0, 0].Frpp_GP1_CgP1[0]
                x_te = panels[nc-1, 0].Brpp_GP1_CgP1[0]
                chord_val = x_te - x_le
                x_ea = x_le + self._x_ea_chord * chord_val

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


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

ps.set_up_logging(level="Warning")
chord = 1.8288; semi_span = 6.096
EI = 9.773e6; GJ = 0.988e6
m_per_length = 35.72
h_shell = 0.3; nu_xy = 0.3
Ey = EI * 12.0 * (1.0 - nu_xy**2) / (h_shell**3 * chord)
G_xy = GJ * 3.0 / (h_shell**3 * chord)
Ex = Ey
rho_shell = m_per_length / (chord * h_shell)
n_chord = 4; n_span = 8
dt_uvlm = 0.003
mesh_verts, mesh_tris = make_rect_mesh(chord, semi_span, n_chord, n_span)

# Theoretical references
print("=" * 70)
print("LOW-SPEED AEROELASTIC VALIDATION")
print("=" * 70)

# Natural frequencies
f1_bend = (1.8751**2 / (2 * np.pi * semi_span**2)) * np.sqrt(EI / m_per_length)
f1_tors_beam = (1.0 / (4 * semi_span)) * np.sqrt(GJ / 0.252)  # beam I_theta

# Static deflection under uniform load q
# For cantilever with uniform load: delta_tip = q*L^4/(8*EI)
# q = 0.5 * rho * V^2 * chord * CL, CL = 2*pi*alpha
V_test = 30.0
alpha_rad = 2.0 * np.pi / 180
CL = 2 * np.pi * alpha_rad
q_aero = 0.5 * 1.225 * V_test**2 * chord * CL  # N/m
delta_tip_theory = q_aero * semi_span**4 / (8 * EI)

print(f"\nMesh: {n_chord}x{n_span}")
print(f"V_test = {V_test} m/s, alpha = 2 deg")
print(f"CL = {CL:.4f}, q_aero = {q_aero:.2f} N/m")
print(f"Expected: f_bend = {f1_bend:.2f} Hz")
print(f"Expected: delta_tip = {delta_tip_theory*1000:.3f} mm (beam theory)")

# ── Test 1: Natural frequencies (structural only) ──
print(f"\n--- TEST 1: Natural Frequencies ---")
shell = BSTShell(mesh_verts, mesh_tris,
                 h=h_shell, rho=rho_shell,
                 Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy,
                 structural_damping=0.0)
root = np.where(np.abs(mesh_verts[:, 1]) < 1e-10)[0]
shell.set_bc(root)
shell._precompute_ibm()

# Eigenvalue analysis for z-DOF
free = shell.mass > 0
Q_ff = shell._Q[np.ix_(free, free)]
M_inv_sqrt = np.diag(1.0 / np.sqrt(shell.mass[free]))
A = M_inv_sqrt @ Q_ff @ M_inv_sqrt
eigvals = np.linalg.eigvalsh(A)
pos_eigs = eigvals[eigvals > 1e-3]
freqs = np.sqrt(pos_eigs) / (2 * np.pi)
freqs = np.sort(freqs)

print(f"  First 5 frequencies (Hz): {freqs[:5].round(2)}")
print(f"  f1 = {freqs[0]:.2f} Hz (expected {f1_bend:.2f}, error {abs(freqs[0]-f1_bend)/f1_bend*100:.1f}%)")

# ── Test 2: Low-speed aeroelastic equilibrium ──
print(f"\n--- TEST 2: Aeroelastic Equilibrium at V={V_test} m/s ---")
for V in [30, 50]:
    shell = BSTShell(mesh_verts, mesh_tris,
                     h=h_shell, rho=rho_shell,
                     Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy,
                     structural_damping=1.0)
    shell.set_bc(root)
    y_norm = mesh_verts[:, 1] / semi_span
    shell.u[:, 2] = 1e-5 * y_norm**2

    implicit = BSTImplicitGPU(shell, scheme='newmark', k_strategy='fd_direct')
    nc_wake = 30
    mv = build_goland_wing(V, dt=dt_uvlm, n_chord=n_chord, n_span=n_span,
                           num_chords=nc_wake)
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

    solver = AeroSolverImplicit(prob, shell, implicit, relaxation=0.3)
    t0 = time.time()
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1 = time.time()

    tw = np.array(solver.tip_w_history)
    tth = np.array(solver.tip_theta_history)

    # Steady-state values (last 20% of simulation)
    n_ss = max(1, len(tw) // 5)
    w_ss = np.mean(tw[-n_ss:])
    th_ss = np.mean(tth[-n_ss:])
    avg_newton = np.mean(solver.newton_iters) if solver.newton_iters else 0

    # Total lift from strip forces
    total_lift_N = 0.0
    if solver.strip_forces_history:
        last_fifth = solver.strip_forces_history[-n_ss:]
        total_lift_N = np.mean([f[0] for f in last_fifth])

    # Theoretical comparisons
    q_v = 0.5 * 1.225 * V**2 * chord * CL  # N/m uniform load
    delta_uniform = q_v * semi_span**4 / (8 * EI)
    delta_elliptical = 2.0 / 3.0 * delta_uniform  # elliptical load correction
    L_theory = q_v * semi_span  # total lift (uniform)

    abs_w = abs(w_ss) * 1000
    print(f"  V={V:2d} m/s (wake={nc_wake}, {t1-t0:.0f}s):")
    print(f"    |tip_w| = {abs_w:.3f} mm  (uniform theory={delta_uniform*1000:.3f}, "
          f"elliptical={delta_elliptical*1000:.3f})")
    print(f"    theta   = {th_ss:+.6f} rad")
    print(f"    Lift    = {total_lift_N:.1f} N  (theory={L_theory:.1f} N, "
          f"ratio={total_lift_N/L_theory:.3f})")
    print(f"    Newton  ~{avg_newton:.1f} iters")

# ── Test 3: Dynamic response (impulse) ──
print(f"\n--- TEST 3: Dynamic Response at V={V_test} m/s ---")
# Use low damping to see oscillation
shell = BSTShell(mesh_verts, mesh_tris,
                 h=h_shell, rho=rho_shell,
                 Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy,
                 structural_damping=0.02)
shell.set_bc(root)

# Initial perturbation: quadratic bending
y_norm = mesh_verts[:, 1] / semi_span
shell.u[:, 2] = 5e-4 * y_norm**2  # 0.5 mm at tip

implicit = BSTImplicitGPU(shell, scheme='newmark', k_strategy='fd_direct')
mv = build_goland_wing(V_test, dt=dt_uvlm, n_chord=n_chord, n_span=n_span,
                        num_chords=20)
prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

solver = AeroSolverImplicit(prob, shell, implicit, relaxation=0.3)
t0 = time.time()
solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
t1 = time.time()

tw = np.array(solver.tip_w_history)
tth = np.array(solver.tip_theta_history)

print(f"  Simulation time: {t1-t0:.0f}s")
print(f"  Tip w range: [{tw.min()*1000:.3f}, {tw.max()*1000:.3f}] mm")
print(f"  Tip theta range: [{tth.min()*1000:.4f}, {tth.max()*1000:.4f}] mrad")

# FFT of tip response
if len(tw) > 50:
    from scipy.signal import welch
    nperseg = min(len(tw), 256)
    f_w, Pxx_w = welch(tw, fs=1/dt_uvlm, nperseg=nperseg)
    f_th, Pxx_th = welch(tth, fs=1/dt_uvlm, nperseg=nperseg)

    peak_w = f_w[np.argmax(Pxx_w)]
    peak_th = f_th[np.argmax(Pxx_th)]

    print(f"\n  FFT peak frequencies:")
    print(f"    Bending:  {peak_w:.2f} Hz (expected {f1_bend:.2f})")
    print(f"    Torsion:  {peak_th:.2f} Hz")

    # Decay analysis
    # Half-life of oscillation
    abs_tw = np.abs(tw)
    max_amp = np.max(abs_tw[:len(abs_tw)//4])  # peak in first quarter
    # Find when amplitude drops to half
    for i in range(len(abs_tw)//4, len(abs_tw)):
        if abs_tw[i] < max_amp / 2:
            t_half = i * dt_uvlm
            print(f"    Half-life: {t_half:.3f} s ({i} steps)")
            break
    else:
        print(f"    No half-life found in simulation range")

print(f"\n{'='*70}")
print(f"VALIDATION SUMMARY")
print(f"{'='*70}")
