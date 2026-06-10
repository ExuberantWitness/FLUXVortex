"""Structural-only Newmark validation: mesh convergence + MATLAB comparison.

Runs MATLAB-matched structural-only Newmark (trapezoidal rule, membrane-not-averaged
corrector) on increasing mesh sizes. Compares tip w* at t*=0.2025 against MATLAB FSI.

The key diagnostic: does the structural solver alone (no fluid coupling) produce
similar tip displacements to MATLAB's full FSI result? If the FSI result is much
smaller, fluid coupling (aerodynamic damping + added mass) accounts for the difference.
"""
import sys, os, time as time_mod
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.ancf_shell import ANCFShell


def build_shell(nx, ny, thickness=1e-3, rho=3125.0, E=3.125e6, nu=0.3):
    L, W = 1.0, 1.0
    x_vec = np.linspace(0, L, nx + 1)
    y_vec = np.linspace(0, W, ny + 1)

    nn = (nx + 1) * (ny + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = j * (nx + 1) + i
            nodes[idx, 0] = x_vec[i]
            nodes[idx, 1] = y_vec[j]

    ne = nx * ny
    quads = np.zeros((ne, 4), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            elem = j * nx + i
            quads[elem] = [
                j * (nx + 1) + i,
                j * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + i,
            ]

    shell = ANCFShell(
        nodes, quads, h=thickness, rho=rho,
        Ex=E, Ey=E, nu_xy=nu, mode='full', n_gauss=2,
    )
    le_nodes = [j * (nx + 1) for j in range(ny + 1)]
    shell.set_bc(le_nodes, fix_slopes=True)
    return shell


def yamano_pulse_force(shell, params, t):
    L = params['Length']
    V_inf = params['V_inf']
    q_in_norm_max = 0.5
    force_density_ref = params['rho_fluid'] * V_inf**2 / params['thickness']
    t_nondim = t * V_inf / L

    if t_nondim < 0.2:
        q_in_norm = q_in_norm_max * np.sin(np.pi * t_nondim / 0.2)
    else:
        q_in_norm = 0.0

    F_body = np.array([0.0, 0.0, -q_in_norm * force_density_ref])
    return shell.distributed_load(F_body)


def main():
    np.set_printoptions(precision=6, linewidth=100)

    params = {
        'Length': 1.0, 'thickness': 1e-3,
        'rho_fluid': 1.225, 'V_inf': 10.0,
    }

    # MATLAB d_t=0.0015 is NONDIMENSIONAL
    # dt_dim = d_t * L / V_inf = 0.0015 * 1 / 10 = 0.00015s
    L, V_inf = params['Length'], params['V_inf']
    dt = 0.0015 * L / V_inf  # = 0.00015 s
    # Target: t* = 0.2025 → t_dim = 0.02025s → 135 steps
    n_steps = 135

    for (nx, ny, rho_s, E_s) in [(5, 3, 3125.0, 3.125e6), (8, 5, 3125.0, 3.125e6)]:
        print(f"\n=== Mesh: {nx}×{ny} ===")
        t0 = time_mod.time()
        shell = build_shell(nx, ny, params['thickness'], rho_s, E_s)
        build_t = time_mod.time() - t0
        print(f"  Build: {build_t:.1f}s, DOF={shell.ndof}, ne={shell.ne}")

        tip_idx = (shell.nn - 1) * 9 + 2  # last node, z-dof

        t_start = time_mod.time()
        for step in range(n_steps):
            t = step * dt
            F_ext = yamano_pulse_force(shell, params, t)
            shell.step_newmark(F_ext, dt, alpha_v=0.5)

            if step < 3 or step % 45 == 0:
                w = shell.q[tip_idx]
                t_nd = t * params['V_inf'] / params['Length']
                w_nd = w / params['Length']
                print(f"  step {step:4d}, t*={t_nd:.4f}, w={w:.6e}m, w*={w_nd:.6e}")

        elapsed = time_mod.time() - t_start
        w_final = shell.q[tip_idx]
        t_final_nd = n_steps * dt * params['V_inf'] / params['Length']
        w_final_nd = w_final / params['Length']
        print(f"  {n_steps} steps in {elapsed:.1f}s ({elapsed/n_steps*1000:.1f}ms/step)")
        print(f"  Final: t*={t_final_nd:.4f}, w*={w_final_nd:.6f}")

    # 15×10 mesh - same dt as above
    print(f"\n=== Mesh: 15×10 ===")
    t0 = time_mod.time()
    shell = build_shell(15, 10, params['thickness'], 3125.0, 3.125e6)
    build_t = time_mod.time() - t0
    print(f"  Build: {build_t:.1f}s, DOF={shell.ndof}, ne={shell.ne}")

    tip_idx = (shell.nn - 1) * 9 + 2
    t_start = time_mod.time()
    for step in range(n_steps):
        t = step * dt
        F_ext = yamano_pulse_force(shell, params, t)
        shell.step_newmark(F_ext, dt, alpha_v=0.5)
        if step < 3 or step % 45 == 0:
            w_nd = shell.q[tip_idx] / params['Length']
            t_nd = t * params['V_inf'] / params['Length']
            print(f"  step {step:4d}, t*={t_nd:.4f}, w*={w_nd:.6e}")

    elapsed = time_mod.time() - t_start
    w_final = shell.q[tip_idx]
    t_final_nd = n_steps * dt * params['V_inf'] / params['Length']
    w_final_nd = w_final / params['Length']
    print(f"  {n_steps} steps in {elapsed:.1f}s ({elapsed/n_steps*1000:.1f}ms/step)")
    print(f"\n  MATLAB FSI ref:    w* = -0.001489 at t*=0.1995")
    print(f"  Python struct-only: w* = {w_final_nd:.6f} at t*={t_final_nd:.4f}")
    ratio = abs(w_final_nd / -0.001489) if abs(w_final_nd) > 1e-12 else 0
    print(f"  Ratio Python/MATLAB: {ratio:.2f}x")


if __name__ == '__main__':
    main()
