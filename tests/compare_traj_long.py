"""Long-time Python<->MATLAB tip-trajectory + stability comparison.

Runs Python to the same end time as the MATLAB reference (traj_long_t<T>.mat),
compares z_tip at intervals, and reports the divergence growth + a stability
check (does Python track MATLAB's growth, stay bounded, blow up, or decay?).
"""
import os, sys, argparse, time as time_mod
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', type=float, default=1.0, help='t* end time')
    ap.add_argument('--save', default=None, help='optional .npz to save python tip')
    args = ap.parse_args()

    ref_path = (f'FSI_by_FEM_and_UVLM/single_sheet/fixtures/'
                f'traj_long_t{args.end:.1f}.mat')
    m = loadmat(ref_path, squeeze_me=True, struct_as_record=False)
    z_ml = np.asarray(m['z_tip']).ravel()
    time_m = np.asarray(m['time_m']).ravel()
    Nx, Ny = int(m['Nx']), int(m['Ny'])

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    V_inf = params['V_inf']; L = params['Length']
    dt_struct = 2e-4
    n_steps = int(round(args.end * L / V_inf / dt_struct))  # t*_end / (V/L) / dt

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

    print(f"[long] running Python {n_steps} steps to t*={args.end} ...")
    t0 = time_mod.time()
    solver.run(n_steps, print_every=0)
    print(f"[long] python done in {time_mod.time()-t0:.0f}s")

    tip_py = np.concatenate([[0.0], np.array(solver.tip_w_history)])
    if args.save:
        np.savez(args.save, tip_py=tip_py, z_ml=z_ml, time_m=time_m)

    n = min(len(tip_py), len(z_ml))
    print(f"\n{'t*':>7} {'py':>12} {'ml':>12} {'ratio':>8} {'abs_err':>11}")
    for k in range(0, n, max(1, n // 25)):
        tstar = k * dt_struct * V_inf / L
        py, ml = tip_py[k], z_ml[k]
        r = py / ml if abs(ml) > 1e-12 else float('nan')
        print(f"{tstar:7.3f} {py:+.5e} {ml:+.5e} {r:8.4f} {py-ml:+.4e}")

    # Stability summary
    print("\n=== stability / fidelity summary ===")
    print(f"  MATLAB:  z_tip end = {z_ml[n-1]:+.4e}, max|z| = {np.max(np.abs(z_ml[:n])):.4e}")
    print(f"  Python:  z_tip end = {tip_py[n-1]:+.4e}, max|z| = {np.max(np.abs(tip_py[:n])):.4e}")
    # growth: ratio of |z| in last quarter vs second quarter (envelope growth)
    q = n // 4
    ml_growth = np.max(np.abs(z_ml[3*q:n])) / (np.max(np.abs(z_ml[q:2*q])) + 1e-30)
    py_growth = np.max(np.abs(tip_py[3*q:n])) / (np.max(np.abs(tip_py[q:2*q])) + 1e-30)
    print(f"  envelope growth (last-qtr/2nd-qtr):  MATLAB={ml_growth:.3f}  Python={py_growth:.3f}")
    print(f"  final ratio Python/MATLAB = {tip_py[n-1]/z_ml[n-1]:.4f}")
    if not np.all(np.isfinite(tip_py[:n])):
        print("  [!] Python produced NaN/Inf -> UNSTABLE")


if __name__ == '__main__':
    main()
