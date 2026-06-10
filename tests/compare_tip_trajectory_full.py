"""Step-by-step tip-trajectory comparison: Python vs MATLAB h_X_vec.

MATLAB h_X_vec is sampled every structural step (d_t = 2e-3 nondim = 2e-4 s),
the SAME cadence as Python's dt_struct. So we can compare tip_w[k] directly at
every step and see exactly where (and how) the divergence accumulates.

Outputs a table: step | t* | py tip_z | ml tip_z | ratio | abs-err
and flags the first step where |ratio-1| exceeds 1%, 5%, 10%.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell
from load_matlab_fixture import MatlabFixture


def matlab_tip_dof(Nx, Ny):
    """MATLAB trailing tip node (i=Nx, j=Ny/2), z-DOF, 0-indexed.
    MATLAB i-outer/j-inner: node = i*(Ny+1) + j (0-indexed)."""
    i, j = Nx, Ny // 2
    node = i * (Ny + 1) + j
    return node * 9 + 2


def main():
    params = yamano_params()
    Nx, Ny = 15, 10
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    V_inf = params['V_inf']; L = params['Length']
    dt_struct = 2e-4
    n_steps = 110

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct, uvlm_dt_ratio=34,
        integrator='implicit', relaxation=1.0,
        newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, core_radius=1e-6,
        coupling='strong')

    T_dur = 0.2 * L / V_inf
    f_density = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0.0, 0.0, +0.5 * f_density]))
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    solver.run(n_steps, print_every=0)
    # Python records tip AFTER each step (no initial entry). Prepend initial 0 so
    # tip_py[0] = initial state, aligning index-for-index with MATLAB h_X_vec[:,0].
    tip_py = np.concatenate([[0.0], np.array(solver.tip_w_history)])

    # MATLAB trajectory (h_X_vec[:,0] = initial state)
    fx = MatlabFixture('FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat')
    hX = np.asarray(fx._raw['h_X_vec'])
    zdof = matlab_tip_dof(Nx, Ny)
    tip_ml = hX[zdof, :]                       # nondim = dimensional (L=1)

    n = min(len(tip_py), tip_ml.shape[0])
    print(f"  Python tip history len={len(tip_py)}, MATLAB len={tip_ml.shape[0]}")
    print(f"  Comparing {n} steps. zdof={zdof}")
    print()
    print(f"{'step':>4} {'t*':>7} {'py_tip':>12} {'ml_tip':>12} {'ratio':>8} {'abs_err':>11}")
    flags = {0.01: None, 0.05: None, 0.10: None}
    for k in range(0, n, 5):
        py = tip_py[k]; ml = tip_ml[k]
        tstar = k * dt_struct * V_inf / L
        ratio = py / ml if abs(ml) > 1e-12 else float('nan')
        err = py - ml
        print(f"{k:4d} {tstar:7.4f} {py:+.5e} {ml:+.5e} {ratio:8.4f} {err:+.4e}")
    # First-divergence flags (check every step)
    for k in range(1, n):
        ml = tip_ml[k]
        if abs(ml) < 1e-9:
            continue
        r = abs(tip_py[k] / ml - 1.0)
        for thr in flags:
            if flags[thr] is None and r > thr:
                flags[thr] = k
    print()
    for thr in sorted(flags):
        ks = flags[thr]
        ts = ks * dt_struct * V_inf / L if ks else None
        print(f"  first step exceeding {int(thr*100)}% error: "
              f"{ks} (t*={ts:.4f})" if ks else
              f"  never exceeds {int(thr*100)}% error")


if __name__ == '__main__':
    main()
