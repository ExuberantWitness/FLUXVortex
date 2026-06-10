"""Side-by-side numerical comparison of Python and MATLAB Newmark step 1 outputs.

Reads MATLAB's dump of q_p1 (stage 0) + q_new (stage 1) from step1_qp1.mat, and
computes Python's equivalents in matching non-dim form. Prints unit-stripped
numerical values entry-by-entry for the first few interesting DOFs.
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
    L = params['Length']; V_inf = params['V_inf']
    rho_f = params['rho_fluid']
    h = params['thickness']
    dt_dim = 2e-4              # Python time step (seconds)
    dt_nondim = dt_dim * V_inf / L     # = 2e-3 (matches MATLAB d_t)
    T_dur_dim = 0.2 * L / V_inf

    print(f"Python dt_dim = {dt_dim}  → dt_nondim = {dt_nondim}")
    print(f"L={L}, V_inf={V_inf}, rho_f={rho_f}, h={h}")

    # Build Python Newmark step 1 in dim units
    f_density = rho_f * V_inf**2 / h
    pulse_dim = shell.distributed_load(np.array([0.0, 0.0, +f_density]))   # dim N
    q_norm_t = 0.5 * np.sin(np.pi * dt_dim / T_dur_dim)
    F_dim = pulse_dim * q_norm_t

    bc = np.array(sorted(shell._bc_dofs), dtype=np.int32)
    free = np.setdiff1d(np.arange(shell.ndof), bc)
    nf = len(free)
    M_ff_dim = shell.M[np.ix_(free, free)].tocsc()
    _, Kt_dim = shell._internal_forces_and_tangent(shell.q)
    K_ff_dim = Kt_dim[np.ix_(free, free)].tocsc()

    # Newmark stage 0
    alpha_v = 0.5; c_damp = 2.0
    I_sp = speye(nf, format='csc')
    O_sp = spcsc((nf, nf))
    D_mat = spbmat([[I_sp, O_sp], [c_damp*dt_dim/2.0 * K_ff_dim, M_ff_dim]], format='csc')
    X2_mat = spbmat([[O_sp, I_sp], [O_sp, O_sp]], format='csc')
    A1 = D_mat - alpha_v * dt_dim * X2_mat
    A2 = D_mat + (1 - alpha_v) * dt_dim * X2_mat

    q_n_dim = shell.q.copy()
    dq_n_dim = shell.dq.copy()
    X_n_free = np.concatenate([q_n_dim[free], dq_n_dim[free]])
    A2Xn = A2 @ X_n_free
    rhs0 = np.zeros(2*nf)
    rhs0[nf:] = F_dim[free]
    X_p1_free = spsolve(A1, A2Xn) + dt_dim * spsolve(A1, rhs0)

    q_p1_py_dim = q_n_dim.copy()
    q_p1_py_dim[free] = X_p1_free[:nf]
    dq_p1_py_dim = dq_n_dim.copy()
    dq_p1_py_dim[free] = X_p1_free[nf:]

    # Convert to non-dim for comparison
    # q*_p1 = q_p1_dim / L, dt_q*_p1 = dq_p1_dim * L / V → wait no:
    # q_nondim = r / L. So q*_p1 = q_p1_dim / L.
    # dt_q_nondim = dr/(dt_nondim) = (dr/dt_dim) * (dt_dim/dt_nondim) = dq_dim * (L/V_inf)
    q_p1_py_nondim  = q_p1_py_dim / L
    dq_p1_py_nondim = dq_p1_py_dim * (L / V_inf)

    # Load MATLAB
    ml = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/step1_qp1.mat',
                 squeeze_me=True, struct_as_record=False)
    q_p1_ml_nondim  = np.asarray(ml['q_p1'], dtype=float).ravel()
    dq_p1_ml_nondim = np.asarray(ml['dq_p1'], dtype=float).ravel()
    q_ref_ml        = np.asarray(ml['q_vec'], dtype=float).ravel()

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    q_p1_ml_py_nondim  = q_p1_ml_nondim[dof_perm]
    dq_p1_ml_py_nondim = dq_p1_ml_nondim[dof_perm]
    q_ref_ml_py        = q_ref_ml[dof_perm]

    print(f"\nNon-dim displacement δq = q_p1 - q_ref:")
    delta_py = q_p1_py_nondim - q_ref_ml_py
    delta_ml = q_p1_ml_py_nondim - q_ref_ml_py
    print(f"  |δq_py|_max = {np.max(np.abs(delta_py)):.4e}")
    print(f"  |δq_ml|_max = {np.max(np.abs(delta_ml)):.4e}")
    print(f"  ratio (max) = {np.max(np.abs(delta_py)) / np.max(np.abs(delta_ml)):.4f}")

    # Tip Z
    tip_node_py = (Ny // 2) * (Nx + 1) + Nx
    z_dof = tip_node_py * 9 + 2
    print(f"\nTip Z (non-dim):")
    print(f"  py = {q_p1_py_nondim[z_dof]:+.5e}")
    print(f"  ml = {q_p1_ml_py_nondim[z_dof]:+.5e}")
    print(f"  ratio = {q_p1_py_nondim[z_dof] / q_p1_ml_py_nondim[z_dof]:.4f}")

    # dq tip Z (non-dim)
    print(f"\nTip dt_Z (non-dim velocity):")
    print(f"  py = {dq_p1_py_nondim[z_dof]:+.5e}")
    print(f"  ml = {dq_p1_ml_py_nondim[z_dof]:+.5e}")
    if abs(dq_p1_ml_py_nondim[z_dof]) > 1e-20:
        print(f"  ratio = {dq_p1_py_nondim[z_dof] / dq_p1_ml_py_nondim[z_dof]:.4f}")


if __name__ == "__main__":
    main()
