"""Compare F_pulse (the consistent nodal force from the uniform z-pulse) between
Python's `shell.distributed_load` and MATLAB's `Qf_time_global * q_in_norm`,
at slope DOFs specifically — to test whether the distributed-load projection
onto Hermitian slope basis functions differs.
"""
import os, sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.ancf_shell import ANCFShell
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
    V_inf = params['V_inf']; L = params['Length']; rho_f = params['rho_fluid']; h = params['thickness']

    # Python pulse (raw distributed_load at peak amplitude q_in_norm=0.5)
    f_density_ref = rho_f * V_inf**2 / h          # N/m³
    pulse_py_peak = shell.distributed_load(np.array([0.0, 0.0, +0.5 * f_density_ref]))

    # Load MATLAB Qf_time_global (non-dim force; multiply by force scale to get dim)
    fx = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat',
                 squeeze_me=True, struct_as_record=False)
    if 'Qf_time_global' in fx:
        Qf_time_ml_nondim = np.asarray(fx['Qf_time_global'], dtype=float).ravel()
    else:
        print("⚠ Qf_time_global not in fixture — falling back to Qf_global (should be 0 for Yamano)")
        Qf_time_ml_nondim = np.asarray(fx['Qf_global'], dtype=float).ravel()

    F_scale = rho_f * V_inf**2 * L**2     # 122.5 N per non-dim unit
    pulse_ml_peak = 0.5 * Qf_time_ml_nondim * F_scale     # dim, at q_in_norm peak=0.5

    dof_perm = matlab_to_python_dof_perm(Nx, Ny)
    pulse_ml_peak_py = pulse_ml_peak[dof_perm]

    diff = pulse_py_peak - pulse_ml_peak_py
    print(f"|pulse_py - pulse_ml|_max = {np.max(np.abs(diff)):.4e}")
    print(f"|pulse_py - pulse_ml|_F   = {np.linalg.norm(diff):.4e}")
    print(f"|pulse_ml|_F              = {np.linalg.norm(pulse_ml_peak_py):.4e}")
    print(f"rel |F|                   = "
          f"{np.linalg.norm(diff)/(np.linalg.norm(pulse_ml_peak_py)+1e-30):.3e}")

    # Per-DOF-type RMS error
    print("\nPer-DOF-type:")
    kinds = ['r_x','r_y','r_z','dx_r_x','dx_r_y','dx_r_z','dy_r_x','dy_r_y','dy_r_z']
    for d in range(9):
        idx = np.arange(d, len(pulse_py_peak), 9)
        py = pulse_py_peak[idx]; ml = pulse_ml_peak_py[idx]
        nz_py = np.linalg.norm(py); nz_ml = np.linalg.norm(ml)
        nz_d  = np.linalg.norm(py - ml)
        print(f"  {kinds[d]:9s}  |py|={nz_py:.4e}  |ml|={nz_ml:.4e}  |d|={nz_d:.4e}  "
              f"ratio_py/ml={(nz_py/(nz_ml+1e-30)):.4f}")

    # Top 10 differing DOFs
    abs_d = np.abs(diff)
    print("\nTop 10 differing DOFs:")
    for fi in np.argsort(abs_d)[::-1][:10]:
        node_p = fi // 9; d = fi % 9
        j = node_p // (Nx + 1); i = node_p % (Nx + 1)
        kind = kinds[d]
        ratio = pulse_py_peak[fi] / (pulse_ml_peak_py[fi] + 1e-30)
        print(f"  dof {fi:4d}  node(i={i:2d},j={j:2d})/{kind:9s}  "
              f"py={pulse_py_peak[fi]:+.4e}  ml={pulse_ml_peak_py[fi]:+.4e}  "
              f"d={diff[fi]:+.3e}  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
