"""Compare MATLAB and Python tangent stiffness K_global at the reference state.

Loads fixtures/K_at_qref.mat (dumped by dump_K_at_qref.m), builds the same Yamano
shell in Python with identical mesh + material, computes Python's Kt at the
reference flat-plate state, applies the node-ordering permutation
(MATLAB i-outer/j-inner -> Python j-outer/i-inner), and compares:

  - M_global       (mass)
  - dq_Qe_global   (membrane + bending tangent K)
  - Qe_global      (membrane internal force)
  - Qk_global      (bending  internal force)

For each, prints Frobenius residual, max entry diff, and top differing entries.
"""
import os, sys, numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.ancf_shell import ANCFShell
from run_standalone_yamano import yamano_params, build_yamano_shell


FIXTURE = os.path.join(os.path.dirname(__file__), '..',
                       'FSI_by_FEM_and_UVLM/single_sheet/fixtures/K_at_qref.mat')


def matlab_to_python_dof_perm(Nx, Ny):
    """Return a 1-D index array `p` such that for any global vector v_matlab,
    v_python = v_matlab[p]  reorders to Python's j-outer/i-inner DOF layout.

    MATLAB node k_m (0-indexed): i_m = k_m // (Ny+1), j_m = k_m %  (Ny+1)
    Python  node k_p:            j_p = k_p // (Nx+1), i_p = k_p %  (Nx+1)
    Same physical (i, j) gives k_p = j*(Nx+1) + i, k_m = i*(Ny+1) + j.

    DOFs: index 9*node + d   for d in 0..8.
    """
    nn = (Nx + 1) * (Ny + 1)
    # node_perm[k_p] = matlab node k_m for the same physical node
    node_perm = np.empty(nn, dtype=np.int64)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            k_p = j * (Nx + 1) + i
            k_m = i * (Ny + 1) + j
            node_perm[k_p] = k_m
    # DOF perm
    dof_perm = np.empty(9 * nn, dtype=np.int64)
    for k_p in range(nn):
        k_m = node_perm[k_p]
        for d in range(9):
            dof_perm[9 * k_p + d] = 9 * k_m + d
    return dof_perm


def reorder_full(A_ml, dof_perm):
    """Reorder a MATLAB-indexed full matrix to Python ordering: A_py[i,j] = A_ml[p[i], p[j]]."""
    return A_ml[np.ix_(dof_perm, dof_perm)]


def reorder_vec(v_ml, dof_perm):
    return v_ml[dof_perm]


def summarize_diff(name, ml, py, k_top=10):
    """Print summary stats and top-k differing entries."""
    d = ml - py
    fro = np.linalg.norm(d)
    fmax = np.max(np.abs(d))
    rel = fro / (np.linalg.norm(ml) + 1e-30)
    print(f"\n--- {name} ---")
    print(f"  |ml|_F = {np.linalg.norm(ml):.6e}   |py|_F = {np.linalg.norm(py):.6e}")
    print(f"  |ml - py|_F = {fro:.6e}   max|ml-py| = {fmax:.6e}   rel |F| = {rel:.3e}")
    if d.ndim == 2:
        flat = np.abs(d).ravel()
        idx_sorted = np.argsort(flat)[::-1][:k_top]
        n = d.shape[0]
        print(f"  top {k_top} differing entries (i, j, ml, py, diff):")
        for k, fi in enumerate(idx_sorted):
            i, j = fi // n, fi % n
            print(f"    [{i:4d},{j:4d}]  ml={ml[i,j]:+.6e}  py={py[i,j]:+.6e}  d={d[i,j]:+.6e}")
    else:
        idx_sorted = np.argsort(np.abs(d))[::-1][:k_top]
        print(f"  top {k_top} differing entries (i, ml, py, diff):")
        for fi in idx_sorted:
            print(f"    [{fi:4d}]  ml={ml[fi]:+.6e}  py={py[fi]:+.6e}  d={d[fi]:+.6e}")


def dof_label(dof_idx, Nx, Ny):
    """Decode a Python DOF index → (node, kind, comp) for human reading."""
    node = dof_idx // 9
    d = dof_idx % 9
    j = node // (Nx + 1); i = node % (Nx + 1)
    kind = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z'][d]
    return f"node({i},{j})/{kind}"


def main():
    print(f"[compare] Loading {FIXTURE}")
    m = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    Nx = int(m['Nx']); Ny = int(m['Ny'])
    print(f"[compare] Nx={Nx} Ny={Ny}, N_q_all={int(m['N_q_all'])}")

    M_ml      = np.asarray(m['M_global'].toarray() if hasattr(m['M_global'], 'toarray') else m['M_global'])
    Kmem_ml   = np.asarray(m['dq_Qe_mem_global'].toarray() if hasattr(m['dq_Qe_mem_global'], 'toarray') else m['dq_Qe_mem_global'])
    Kbend_ml  = np.asarray(m['dq_Qk_global'].toarray() if hasattr(m['dq_Qk_global'], 'toarray') else m['dq_Qk_global'])
    Qe_ml     = np.asarray(m['Qe_global'], dtype=float).ravel()
    Qk_ml     = np.asarray(m['Qk_global'], dtype=float).ravel()

    K_ml = Kmem_ml + Kbend_ml

    # --- MATLAB is in non-dimensional units, Python in dimensional SI.
    # Non-dim scales (with L=1, V=10, rho_f=1.225):
    #   mass scale     [m]  = rho_f * L^3                       = 1.225
    #   length scale       = L                                  = 1.0
    #   time scale         = L/V_inf                            = 0.1
    #   force scale  [F]   = rho_f * V_inf^2 * L^2              = 122.5
    #   K = F/L            = rho_f * V_inf^2 * L                = 122.5
    #   M = F * t^2 / L    = rho_f * L^2                        = 1.225
    # (q is r/L, but L=1 so q_dim numerically equals q_nondim)
    rho_f = 1.225; V_inf = 10.0; L = 1.0
    scale_M = rho_f * L**2          # 1.225
    scale_K = rho_f * V_inf**2 * L  # 122.5
    scale_F = rho_f * V_inf**2 * L**2  # 122.5  (force, for Qe/Qk)
    print(f"\n[compare] unit scales:  M*={scale_M}  K*={scale_K}  F*={scale_F}")
    M_ml    *= scale_M
    Kmem_ml *= scale_K
    Kbend_ml*= scale_K
    K_ml    *= scale_K
    Qe_ml   *= scale_F
    Qk_ml   *= scale_F

    # --- Build Python shell with same parameters ---
    print("\n[compare] Building Python shell (15x10 Yamano)")
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    print(f"[compare] Python shell: nn={shell.nn}, ne={shell.ne}, ndof={shell.ndof}")

    # --- Permutation MATLAB -> Python ordering ---
    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    M_ml_py    = reorder_full(M_ml,    dof_perm)
    K_ml_py    = reorder_full(K_ml,    dof_perm)
    Kmem_ml_py = reorder_full(Kmem_ml, dof_perm)
    Kbend_ml_py= reorder_full(Kbend_ml,dof_perm)
    Qe_ml_py   = reorder_vec(Qe_ml,    dof_perm)
    Qk_ml_py   = reorder_vec(Qk_ml,    dof_perm)

    # --- Python state at reference (flat plate, unit slopes) ---
    q_py = np.zeros(shell.ndof)
    for k in range(shell.nn):
        q_py[9*k + 0] = shell.nodes[k, 0]
        q_py[9*k + 1] = shell.nodes[k, 1]
        q_py[9*k + 2] = 0.0
        q_py[9*k + 3] = 1.0   # dx_r = [1,0,0]
        q_py[9*k + 6] = 0.0   # dy_r = [0,1,0]
        q_py[9*k + 7] = 1.0

    # --- Mass ---
    M_py = shell.M.toarray() if hasattr(shell.M, 'toarray') else np.asarray(shell.M)
    summarize_diff("M_global", M_ml_py, M_py)

    # --- Internal forces (Qe, Qk separated) ---
    Q_mem_py, Q_bend_py = shell._internal_forces_separated(q_py)
    summarize_diff("Qe_global (membrane)", Qe_ml_py, np.asarray(Q_mem_py).ravel())
    summarize_diff("Qk_global (bending)",  Qk_ml_py, np.asarray(Q_bend_py).ravel())

    # --- Tangent stiffness ---
    _, Kt_py_sp = shell._internal_forces_and_tangent(q_py)
    Kt_py = Kt_py_sp.toarray() if hasattr(Kt_py_sp, 'toarray') else np.asarray(Kt_py_sp)
    summarize_diff("dq_Qe (total K = mem + bend)", K_ml_py, Kt_py)

    # --- Symmetry diagnostic ---
    sym_ml = np.max(np.abs(K_ml_py - K_ml_py.T))
    sym_py = np.max(np.abs(Kt_py - Kt_py.T))
    print(f"\n[compare] |K_ml - K_ml^T|_inf = {sym_ml:.3e}  "
          f"|K_py - K_py^T|_inf = {sym_py:.3e}")

    # --- Decode top differing DOF labels ---
    D = K_ml_py - Kt_py
    flat = np.abs(D).ravel()
    n = D.shape[0]
    print("\n[compare] Top 15 K differences with DOF labels:")
    for fi in np.argsort(flat)[::-1][:15]:
        i, j = fi // n, fi % n
        print(f"  [{i:4d},{j:4d}]  {dof_label(i,Nx,Ny):28s}  vs  {dof_label(j,Nx,Ny):28s}  "
              f"ml={K_ml_py[i,j]:+.4e}  py={Kt_py[i,j]:+.4e}  d={D[i,j]:+.4e}")


if __name__ == "__main__":
    main()
