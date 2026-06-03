"""Low-speed solver comparison sweep: V=10,20,30,40,50 m/s.

Compares Euler/Newmark/GenAlpha × fd_direct/jfnk/ibm_precond.
Outputs: tip deflection, torsion angle, Newton iterations, wall time.
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
from test_low_speed_validation import (
    make_rect_mesh, build_goland_wing, AeroSolverImplicit
)

ps.set_up_logging(level="Warning")

# ── Goland Wing parameters ──
chord = 1.8288; semi_span = 6.096
EI = 9.773e6; GJ = 0.988e6
h = 0.3; nu = 0.3
Ey = EI * 12.0 * (1.0 - nu**2) / (h**3 * chord)
Gxy = GJ * 3.0 / (h**3 * chord)
Ex = Ey
rho = 35.72 / (chord * h)
n_chord = 4; n_span = 8
dt_uvlm = 0.003
mesh_verts, mesh_tris = make_rect_mesh(chord, semi_span, n_chord, n_span)
root = np.where(np.abs(mesh_verts[:, 1]) < 1e-10)[0]

# Theoretical references
f1_theory = (1.8751**2 / (2 * np.pi * semi_span**2)) * np.sqrt(EI / 35.72)
alpha_rad = 2.0 * np.pi / 180
CL = 2 * np.pi * alpha_rad

print("=" * 80)
print("LOW-SPEED SOLVER COMPARISON SWEEP")
print("=" * 80)
print(f"Mesh: {n_chord}x{n_span}, dt={dt_uvlm}")
print(f"f1_theory = {f1_theory:.3f} Hz")
print()

# ── Solver configurations ──
configs = [
    ('Euler-fd',        'euler',     'fd_direct'),
    ('Euler-jfnk',      'euler',     'jfnk'),
    ('Newmark-fd',      'newmark',   'fd_direct'),
    ('Newmark-jfnk',    'newmark',   'jfnk'),
    ('Newmark-ibm',     'newmark',   'ibm_precond'),
    ('GenAlpha-fd',     'gen_alpha', 'fd_direct'),
    ('GenAlpha-jfnk',   'gen_alpha', 'jfnk'),
]

speeds = [10, 20, 30, 40, 50]

# ── Header ──
header = f"{'Config':<20s}"
for V in speeds:
    header += f" | V={V:2d}"
header += f" | {'Avg Newton':>10s} {'Total':>6s}"
print(header)
print("-" * len(header))

# ── Run sweep ──
results = {}
for name, scheme, k_strat in configs:
    row = f"{name:<20s}"
    total_newton = 0
    total_time = 0
    all_ok = True

    for V in speeds:
        try:
            shell = BSTShell(mesh_verts, mesh_tris, h=h, rho=rho,
                             Ex=Ex, Ey=Ey, nu_xy=nu, G_xy=Gxy,
                             structural_damping=1.0)
            shell.set_bc(root)
            y_norm = mesh_verts[:, 1] / semi_span
            shell.u[:, 2] = 1e-5 * y_norm**2

            implicit = BSTImplicitGPU(shell, scheme=scheme, k_strategy=k_strat)
            nc_wake = max(10, int(20 * 30 / max(V, 1)))
            nc_wake = min(nc_wake, 30)

            mv = build_goland_wing(V, dt=dt_uvlm, n_chord=n_chord, n_span=n_span,
                                   num_chords=nc_wake)
            prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
            solver = AeroSolverImplicit(prob, shell, implicit, relaxation=0.3)

            t0 = time.time()
            solver.run(prescribed_wake=True, calculate_streamlines=False,
                       show_progress=False)
            t1 = time.time()

            tw = np.array(solver.tip_w_history)
            tth = np.array(solver.tip_theta_history)
            n_ss = max(1, len(tw) // 5)
            w_ss = np.mean(tw[-n_ss:])
            th_ss = np.mean(tth[-n_ss:])
            avg_newton = np.mean(solver.newton_iters) if solver.newton_iters else 0

            if np.isfinite(w_ss) and abs(w_ss) < 1.0:
                row += f" {abs(w_ss)*1000:7.2f}"
                total_newton += avg_newton
            else:
                row += "   DIVERGE"
                all_ok = False

            total_time += t1 - t0

        except Exception as e:
            row += f"   ERR"
            all_ok = False

    avg_n = total_newton / len(speeds) if all_ok else 0
    row += f" {avg_n:10.1f} {total_time:5.0f}s"
    print(row)

# ── Theory row ──
theory_row = f"{'Theory(uniform)':<20s}"
for V in speeds:
    q_v = 0.5 * 1.225 * V**2 * chord * CL
    delta_v = q_v * semi_span**4 / (8 * EI)
    theory_row += f" {delta_v*1000:7.2f}"
theory_row += f" {'':>10s} {'':>6s}"
print(theory_row)

theory_ell = f"{'Theory(elliptical)':<20s}"
for V in speeds:
    q_v = 0.5 * 1.225 * V**2 * chord * CL
    delta_v = 2.0/3.0 * q_v * semi_span**4 / (8 * EI)
    theory_ell += f" {delta_v*1000:7.2f}"
theory_ell += f" {'':>10s} {'':>6s}"
print(theory_ell)

print()
print("Units: |tip_w| in mm. Theory uses beam cantilever formula with uniform/elliptical load.")
