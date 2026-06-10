"""Clean step-1 diagnostic: where does Python diverge from MATLAB at step 1?

Known facts (from compare_K_at_qref.py):
  - M_global and K_global are BIT-EXACT after applying the dimensional scaling
    (K_ml * 122.5 = K_py,  M_ml * 1.225 = M_py)

So if step 1 already differs, the divergence is in:
  - Pulse force application
  - M_added (added mass) at step 1
  - Aero contribution (should be ~0 at step 1: gamma=0 at flat plate)
  - PASS A / PASS B logic
  - Or the Newmark internals (force averaging, BC application)

This script:
  1. Builds Python solver in same config as smoke_mf2_vec1_wired (relaxation=1.0)
  2. Runs ONE full FSI step
  3. Loads MATLAB hX[:, 1] from fixture (state after 1 d_t step)
  4. Applies inverse unit scaling to MATLAB q (dimensionless → meters)
  5. Permutes MATLAB DOF ordering to Python's
  6. Prints per-DOF diff stats and top differing nodes/DOFs
"""
import os, sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell


# Reuse permutation from compare_K_at_qref
def matlab_to_python_dof_perm(Nx, Ny):
    nn = (Nx + 1) * (Ny + 1)
    node_perm = np.empty(nn, dtype=np.int64)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            k_p = j * (Nx + 1) + i
            k_m = i * (Ny + 1) + j
            node_perm[k_p] = k_m
    dof_perm = np.empty(9 * nn, dtype=np.int64)
    for k_p in range(nn):
        k_m = node_perm[k_p]
        for d in range(9):
            dof_perm[9 * k_p + d] = 9 * k_m + d
    return dof_perm


def dof_label(dof_idx, Nx, Ny):
    node = dof_idx // 9
    d = dof_idx % 9
    j = node // (Nx + 1); i = node % (Nx + 1)
    kind = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z'][d]
    return f"node(i={i:2d},j={j:2d})/{kind}"


def main():
    Nx, Ny = 15, 10
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    V_inf = params['V_inf']; L = params['Length']
    dt_struct = 2e-4
    T_dur = 0.2 * L / V_inf

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct, uvlm_dt_ratio=34,
        integrator='implicit',
        relaxation=1.0, newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, core_radius=1e-6,
        coupling='strong',
    )
    f_density_ref = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0.0, 0.0, +0.5 * f_density_ref]))
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    # Run ONE step
    q_py_0 = shell.q.copy()
    dq_py_0 = shell.dq.copy()
    solver.run(1, print_every=0)
    q_py_1 = shell.q.copy()
    dq_py_1 = shell.dq.copy()

    # Load MATLAB
    fx_path = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat'
    fx = loadmat(fx_path, squeeze_me=True, struct_as_record=False)
    hX = np.asarray(fx['h_X_vec'])
    N_q = (Nx + 1) * (Ny + 1) * 9
    assert hX.shape[0] == 2 * N_q, f"hX rows {hX.shape[0]} != 2*N_q {2*N_q}"

    # MATLAB d_t is NON-DIMENSIONAL (~2e-3), Python dt_struct is dimensional (2e-4 s).
    # Physical-time-per-step is equal: 2e-3 * L/V_inf = 2e-4 s = dt_struct.
    # So MATLAB step k ↔ Python step k (1:1).
    print(f"MATLAB d_t (nondim) = {float(fx['d_t']):.2e}  Python dt_struct (dim) = {dt_struct:.2e}")
    print(f"  Both correspond to physical {dt_struct:.2e} s per step → 1:1 step correspondence")

    q_ml_0_nondim = hX[:N_q, 0]
    q_ml_1_nondim = hX[:N_q, 1]
    dq_ml_1_nondim = hX[N_q:, 1]

    # Convert nondim → dim: q* = r/L → q = q* * L. With L=1 they numerically agree.
    # dt_q* = dt_r / V_inf → dt_q = dt_q* * V_inf
    q_ml_0 = q_ml_0_nondim * L
    q_ml_1 = q_ml_1_nondim * L
    dq_ml_1 = dq_ml_1_nondim * V_inf

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    q_ml_0_py = q_ml_0[dof_perm]
    q_ml_1_py = q_ml_1[dof_perm]
    dq_ml_1_py = dq_ml_1[dof_perm]

    # Sanity: q_py_0 should equal q_ml_0_py
    d0 = q_py_0 - q_ml_0_py
    print(f"\n[init] |q_py_0 - q_ml_0|_max = {np.max(np.abs(d0)):.3e}  (should be ~0)")
    if np.max(np.abs(d0)) > 1e-12:
        print("  ⚠ initial state differs — fix DOF perm or scaling first")

    # --- Compare q_1 ---
    dq_diff = q_py_1 - q_ml_1_py
    abs_dq = np.abs(dq_diff)
    # Relative diff per DOF (skip ~0 entries to avoid blowup)
    ml_amp = np.abs(q_ml_1_py - q_ml_0_py)  # MATLAB displacement amplitude
    py_amp = np.abs(q_py_1 - q_py_0)
    print(f"\n=== q_1 comparison (after 1 step) ===")
    print(f"  |q_py_1 - q_ref|_max = {np.max(py_amp):.4e}")
    print(f"  |q_ml_1 - q_ref|_max = {np.max(ml_amp):.4e}")
    print(f"  |q_py_1 - q_ml_1|_max = {np.max(abs_dq):.4e}")
    print(f"  |q_py_1 - q_ml_1|_F   = {np.linalg.norm(dq_diff):.4e}")
    print(f"  |q_py_1 - q_ml_1|_F / |q_ml_1 - q_ref|_F = "
          f"{np.linalg.norm(dq_diff)/np.linalg.norm(q_ml_1_py - q_ml_0_py):.3e}")

    # Tip displacement diagnostic
    # Trailing edge tip, span middle (matches plot_results.m: tip_node = Nx*(Ny+1) + Ny/2 + 1 (1-indexed))
    tip_node_ml_1idx = Nx * (Ny + 1) + Ny // 2 + 1
    z_dof_ml = (tip_node_ml_1idx - 1) * 9 + 2
    # In Python ordering same physical tip node
    tip_node_py = (Ny // 2) * (Nx + 1) + Nx  # j=Ny//2, i=Nx (trailing edge)
    z_dof_py = tip_node_py * 9 + 2
    print(f"\n  tip z DOF:  py={z_dof_py}  ml={z_dof_ml}  (ml→py perm: {dof_perm[z_dof_py]})")
    print(f"  q_py_1 tip_z  = {q_py_1[z_dof_py]:+.5e}")
    print(f"  q_ml_1 tip_z  = {q_ml_1_py[z_dof_py]:+.5e}  (ratio py/ml = {q_py_1[z_dof_py]/q_ml_1_py[z_dof_py]:.4f})")
    print(f"  q_py_1 tip dx_rz = {q_py_1[z_dof_py + 3]:+.5e}")
    print(f"  q_ml_1 tip dx_rz = {q_ml_1_py[z_dof_py + 3]:+.5e}  "
          f"(ratio py/ml = {q_py_1[z_dof_py+3]/(q_ml_1_py[z_dof_py+3]+1e-30):.4f})")

    # Top differing DOFs
    print("\n  Top 15 differing DOFs (|q_py - q_ml|):")
    for fi in np.argsort(abs_dq)[::-1][:15]:
        rel = abs_dq[fi] / (np.abs(q_ml_1_py[fi] - q_ml_0_py[fi]) + 1e-30)
        print(f"    [{fi:4d}] {dof_label(fi,Nx,Ny):24s} py={q_py_1[fi]:+.4e}  ml={q_ml_1_py[fi]:+.4e}  "
              f"d={dq_diff[fi]:+.3e}  rel_to_disp={rel:.2f}")

    # Per-DOF-kind breakdown of error
    print("\n  Per-DOF-type RMS error / RMS amplitude (MATLAB):")
    kinds = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z']
    for d in range(9):
        idx = np.arange(d, len(q_py_1), 9)
        rms_err = np.sqrt(np.mean(dq_diff[idx]**2))
        rms_ml  = np.sqrt(np.mean((q_ml_1_py[idx] - q_ml_0_py[idx])**2))
        rel = rms_err / (rms_ml + 1e-30)
        print(f"    {kinds[d]:9s}  rms_err={rms_err:.4e}  rms_amp_ml={rms_ml:.4e}  rel={rel:.3e}")


if __name__ == "__main__":
    main()
