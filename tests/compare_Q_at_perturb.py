"""Compare Python's _internal_forces_separated with MATLAB's Qe, Qk at a small
z-perturbation away from the flat-plate reference state.

Loads the MATLAB dump (Q_at_perturb.mat) and computes Python's Q_mem, Q_bend at
the SAME q_perturb, then checks per-DOF residuals.
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


def dof_label(idx, Nx, Ny):
    node = idx // 9
    d = idx % 9
    j = node // (Nx + 1); i = node % (Nx + 1)
    kind = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z'][d]
    return f"node(i={i:2d},j={j:2d})/{kind}"


def main():
    fx = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/Q_at_perturb.mat',
                 squeeze_me=True, struct_as_record=False)
    Nx, Ny = int(fx['Nx']), int(fx['Ny'])
    q_ref_ml      = np.asarray(fx['q_ref'], dtype=float).ravel()
    q_perturb_ml  = np.asarray(fx['q_perturb'], dtype=float).ravel()
    Qe_ml = np.asarray(fx['Qe_global'], dtype=float).ravel()    # non-dim membrane
    Qk_ml = np.asarray(fx['Qk_global'], dtype=float).ravel()    # non-dim bending

    # Unit scale: non-dim force → dim force = F_scale  (=ρ_f V_inf^2 L^2)
    rho_f = 1.225; V_inf = 10.0; L = 1.0
    F_scale = rho_f * V_inf**2 * L**2
    Qe_ml_dim = Qe_ml * F_scale
    Qk_ml_dim = Qk_ml * F_scale

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    q_perturb_ml_py = q_perturb_ml[dof_perm]
    q_ref_ml_py     = q_ref_ml[dof_perm]
    Qe_ml_py        = Qe_ml_dim[dof_perm]
    Qk_ml_py        = Qk_ml_dim[dof_perm]

    # Build Python shell, set q to perturbation
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)

    # Sanity: q_ref agreement
    print(f"|Python q (shell.q at init) - q_ref_ml_py|_max = "
          f"{np.max(np.abs(shell.q - q_ref_ml_py)):.3e}")

    # Compute Python Q_mem, Q_bend at MATLAB q_perturb
    Q_mem_py, Q_bend_py = shell._internal_forces_separated(q_perturb_ml_py)

    # Compare membrane
    dm = Q_mem_py - Qe_ml_py
    print(f"\n=== Q_mem (membrane internal force at q_perturb) ===")
    print(f"  |Qmem_py|_F = {np.linalg.norm(Q_mem_py):.4e}")
    print(f"  |Qmem_ml|_F = {np.linalg.norm(Qe_ml_py):.4e}")
    print(f"  |diff|_max  = {np.max(np.abs(dm)):.4e}")
    print(f"  |diff|_F    = {np.linalg.norm(dm):.4e}")
    print(f"  rel |F|     = {np.linalg.norm(dm)/(np.linalg.norm(Qe_ml_py)+1e-30):.3e}")

    # Compare bending
    db = Q_bend_py - Qk_ml_py
    print(f"\n=== Q_bend (bending internal force at q_perturb) ===")
    print(f"  |Qbend_py|_F = {np.linalg.norm(Q_bend_py):.4e}")
    print(f"  |Qbend_ml|_F = {np.linalg.norm(Qk_ml_py):.4e}")
    print(f"  |diff|_max   = {np.max(np.abs(db)):.4e}")
    print(f"  |diff|_F     = {np.linalg.norm(db):.4e}")
    print(f"  rel |F|      = {np.linalg.norm(db)/(np.linalg.norm(Qk_ml_py)+1e-30):.3e}")

    # Per-DOF-type RMS
    print("\nPer-DOF-type Q_bend comparison:")
    kinds = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z']
    for d in range(9):
        idx = np.arange(d, len(Q_bend_py), 9)
        py = Q_bend_py[idx]; ml = Qk_ml_py[idx]
        nz_d = np.linalg.norm(py - ml)
        nz_ml = np.linalg.norm(ml)
        print(f"  {kinds[d]:9s}  |py|={np.linalg.norm(py):.4e}  |ml|={nz_ml:.4e}  "
              f"|d|={nz_d:.4e}  ratio_py/ml={(np.linalg.norm(py)/(nz_ml+1e-30)):.4f}")

    # Top differing entries
    abs_d = np.abs(db)
    print("\nTop 10 differing Q_bend entries:")
    for fi in np.argsort(abs_d)[::-1][:10]:
        print(f"  [{fi:4d}] {dof_label(fi,Nx,Ny):24s}  "
              f"py={Q_bend_py[fi]:+.4e}  ml={Qk_ml_py[fi]:+.4e}  d={db[fi]:+.3e}  "
              f"ratio={Q_bend_py[fi]/(Qk_ml_py[fi]+1e-30):.3f}")


if __name__ == "__main__":
    main()
