"""Step-1 with stage-0 (predictor) ONLY — no Qe averaging correction.

At step 1 with q_n=q_ref (flat plate), Qe_n=0 and Qk_n=0. So stage 0 reduces to:
  A1 X_p1 = A2 X_n + dt [0; F_pulse]
which uses ONLY M, K, F (all bit-exact). If THIS gives 16% slope error too,
the bug is in scipy.sparse.linalg.spsolve or system assembly. If it's bit-exact,
the bug is in stage-1 Q_bend(q_p1) computation.
"""
import os, sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse import eye as speye, bmat as spbmat, csc_matrix as spcsc
from scipy.sparse.linalg import spsolve

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

    # Pulse at end-of-step
    q_norm_t1 = 0.5 * np.sin(np.pi * dt / T_dur)
    f_density = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0.0, 0.0, +f_density]))
    F = pulse * q_norm_t1

    bc = np.array(sorted(shell._bc_dofs), dtype=np.int32)
    free = np.setdiff1d(np.arange(shell.ndof), bc)
    nf = len(free)

    q_n = shell.q.copy()
    dq_n = shell.dq.copy()

    _, Kt = shell._internal_forces_and_tangent(q_n)
    Kt_ff = Kt[np.ix_(free, free)].tocsc()
    M_ff = shell.M[np.ix_(free, free)].tocsc()

    # Build Newmark operator (matches NewmarkSolver.step)
    alpha_v = 0.5
    c_damp = 2.0

    I_sp = speye(nf, format='csc')
    O_sp = spcsc((nf, nf))
    D_bot_left = c_damp * dt / 2.0 * Kt_ff
    D_mat = spbmat([[I_sp, O_sp], [D_bot_left, M_ff]], format='csc')
    X2_mat = spbmat([[O_sp, I_sp], [O_sp, O_sp]], format='csc')
    A1 = D_mat - alpha_v * dt * X2_mat
    A2 = D_mat + (1.0 - alpha_v) * dt * X2_mat

    X_n_free = np.concatenate([q_n[free], dq_n[free]])
    A2Xn = A2 @ X_n_free

    # Stage 0: Qe_n = 0 at q_ref
    Q_global = F.copy()    # Qe_n=0 at flat plate
    rhs0 = np.zeros(2 * nf)
    rhs0[nf:] = Q_global[free]

    X_p1_free = spsolve(A1, A2Xn) + dt * spsolve(A1, rhs0)

    q_p1 = q_n.copy()
    q_p1[free] = X_p1_free[:nf]
    dq_p1 = dq_n.copy()
    dq_p1[free] = X_p1_free[nf:]

    # Compare to MATLAB step 1 (which is AFTER stage 1)
    fx = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat',
                 squeeze_me=True, struct_as_record=False)
    hX = np.asarray(fx['h_X_vec'])
    N_q = (Nx + 1) * (Ny + 1) * 9
    q_ml_0 = hX[:N_q, 0] * L
    q_ml_1 = hX[:N_q, 1] * L

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    q_ml_0_py = q_ml_0[dof_perm]
    q_ml_1_py = q_ml_1[dof_perm]

    print(f"=== Stage-0 ONLY (no Qe correction) vs MATLAB stage-1 (full) ===")
    diff = q_p1 - q_ml_1_py
    print(f"  |stage0 - ml|_max = {np.max(np.abs(diff)):.4e}")
    print(f"  |stage0 - ml|_F   = {np.linalg.norm(diff):.4e}")

    tip_node_py = (Ny // 2) * (Nx + 1) + Nx
    z_dof_py = tip_node_py * 9 + 2
    print(f"  tip z (py stage0)    = {q_p1[z_dof_py]:+.5e}")
    print(f"  tip z (ml)           = {q_ml_1_py[z_dof_py]:+.5e}")

    # First-interior-node slope check
    node1_j5_py = 5 * (Nx + 1) + 1
    dxrz_dof = node1_j5_py * 9 + 5
    print(f"\n  node(i=1,j=5) dx_rz (py stage0) = {q_p1[dxrz_dof]:+.5e}")
    print(f"  node(i=1,j=5) dx_rz (ml)        = {q_ml_1_py[dxrz_dof]:+.5e}")
    print(f"  ratio py/ml = {q_p1[dxrz_dof]/q_ml_1_py[dxrz_dof]:.4f}")

    # Per-DOF-type
    print("\n  Per-DOF-type RMS error / RMS amplitude:")
    kinds = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z']
    for d in range(9):
        idx = np.arange(d, len(q_p1), 9)
        rms_err = np.sqrt(np.mean((q_p1[idx] - q_ml_1_py[idx])**2))
        rms_ml  = np.sqrt(np.mean((q_ml_1_py[idx] - q_ml_0_py[idx])**2))
        rel = rms_err / (rms_ml + 1e-30)
        print(f"    {kinds[d]:9s}  rms_err={rms_err:.4e}  rms_amp_ml={rms_ml:.4e}  rel={rel:.3e}")


if __name__ == "__main__":
    main()
