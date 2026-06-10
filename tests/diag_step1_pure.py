"""Minimal step-1 test: pure ANCFShell.step_newmark(F_pulse) — no FSI machinery.

Goal: at step 1 with flat plate and gamma=0, the aero contribution is 0 and
M_added=0 (per __init__ semantics + time-interp at alpha=1/34 of 0-0). So
shell.step_newmark(F_pulse, dt) should be functionally identical to the full
strong-coupling step. This test bypasses the entire StandaloneHybridSolver and
calls shell.step_newmark directly.

If THIS step matches MATLAB bit-exactly, then the FSI machinery is to blame
(M_added timing, aero force at step 1, gamma_pred, etc).

If THIS step still has 17% slope error, then the bug is in shell.step_newmark
even though M, K, F are bit-exact (e.g. integer indexing of free DOFs).
"""
import os, sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from run_standalone_yamano import yamano_params, build_yamano_shell


def matlab_to_python_dof_perm(Nx, Ny):
    nn = (Nx + 1) * (Ny + 1)
    node_perm = np.empty(nn, dtype=np.int64)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            node_perm[j * (Nx + 1) + i] = i * (Ny + 1) + j
    dof_perm = np.empty(9 * nn, dtype=np.int64)
    for k_p in range(nn):
        k_m = node_perm[k_p]
        for d in range(9):
            dof_perm[9 * k_p + d] = 9 * k_m + d
    return dof_perm


def main():
    Nx, Ny = 15, 10
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    V_inf = params['V_inf']; L = params['Length']
    dt = 2e-4
    T_dur = 0.2 * L / V_inf

    # Pulse force at end-of-step (matches MATLAB q_in_norm(time) at t=d_t)
    q_norm_t1 = 0.5 * np.sin(np.pi * dt / T_dur)
    f_density = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0.0, 0.0, +f_density]))
    F_step1 = pulse * q_norm_t1

    print(f"[pure] q_in_norm(dt) = {q_norm_t1:.6e}")
    print(f"[pure] |F_step1|_max = {np.max(np.abs(F_step1)):.4e}")

    q_py_0 = shell.q.copy()
    dq_py_0 = shell.dq.copy()

    # ONE Newmark step, no aero, no M_added
    shell._M_added = None      # explicit: no added mass
    shell.step_newmark(F_step1, dt)
    q_py_1 = shell.q.copy()
    dq_py_1 = shell.dq.copy()

    # MATLAB fixture
    fx = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat',
                 squeeze_me=True, struct_as_record=False)
    hX = np.asarray(fx['h_X_vec'])
    N_q = (Nx + 1) * (Ny + 1) * 9
    q_ml_0_nondim = hX[:N_q, 0]
    q_ml_1_nondim = hX[:N_q, 1]
    # L=1 so nondim = dim numerically
    q_ml_0 = q_ml_0_nondim * L
    q_ml_1 = q_ml_1_nondim * L

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    q_ml_0_py = q_ml_0[dof_perm]
    q_ml_1_py = q_ml_1[dof_perm]

    # Initial state sanity
    d0 = q_py_0 - q_ml_0_py
    print(f"\n[init] |q_py_0 - q_ml_0|_max = {np.max(np.abs(d0)):.3e}  (should be ~0)")

    # Compare step 1
    dq_diff = q_py_1 - q_ml_1_py
    ml_amp = np.abs(q_ml_1_py - q_ml_0_py)
    py_amp = np.abs(q_py_1 - q_py_0)
    print(f"\n=== q_1 comparison (pure Newmark, no FSI) ===")
    print(f"  |q_py_1 - q_ref|_max = {np.max(py_amp):.4e}")
    print(f"  |q_ml_1 - q_ref|_max = {np.max(ml_amp):.4e}")
    print(f"  |q_py_1 - q_ml_1|_max = {np.max(np.abs(dq_diff)):.4e}")
    print(f"  |q_py_1 - q_ml_1|_F   = {np.linalg.norm(dq_diff):.4e}")
    print(f"  rel |F| / |MATLAB disp|: "
          f"{np.linalg.norm(dq_diff)/np.linalg.norm(q_ml_1_py - q_ml_0_py):.3e}")

    # Tip z
    tip_node_py = (Ny // 2) * (Nx + 1) + Nx
    z_dof_py = tip_node_py * 9 + 2
    print(f"\n  tip z (py)    = {q_py_1[z_dof_py]:+.5e}")
    print(f"  tip z (ml)    = {q_ml_1_py[z_dof_py]:+.5e}  "
          f"ratio = {q_py_1[z_dof_py]/q_ml_1_py[z_dof_py]:.4f}")
    print(f"  tip dxrz (py) = {q_py_1[z_dof_py + 3]:+.5e}")
    print(f"  tip dxrz (ml) = {q_ml_1_py[z_dof_py + 3]:+.5e}  "
          f"ratio = {q_py_1[z_dof_py+3]/(q_ml_1_py[z_dof_py+3]+1e-30):.4f}")

    # Per-DOF-type breakdown
    print("\n  Per-DOF-type RMS error / RMS amplitude:")
    kinds = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z']
    for d in range(9):
        idx = np.arange(d, len(q_py_1), 9)
        rms_err = np.sqrt(np.mean(dq_diff[idx]**2))
        rms_ml  = np.sqrt(np.mean((q_ml_1_py[idx] - q_ml_0_py[idx])**2))
        rel = rms_err / (rms_ml + 1e-30)
        print(f"    {kinds[d]:9s}  rms_err={rms_err:.4e}  rms_amp_ml={rms_ml:.4e}  rel={rel:.3e}")


if __name__ == "__main__":
    main()
