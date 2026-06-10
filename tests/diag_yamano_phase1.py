"""Diagnostic: run Yamano benchmark just past the reference point t*=0.1995 and
print w* there. Quantifies how Phase 1+2 (gradient fixes + Mf1 added-mass)
position us relative to MATLAB target w* = -0.001489.

Run: python tests/diag_yamano_phase1.py [--small]
"""
import os, sys, time as time_mod, argparse
import numpy as np
import functools

# Force flushing on every print so we see progress in real time
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

from run_standalone_yamano import yamano_params, build_yamano_shell


TARGET_TSTAR = 0.1995
TARGET_W_STAR = -0.001489


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--small', action='store_true', help='5x3 grid for fast sanity check')
    args = ap.parse_args()

    nx_grid = 5 if args.small else 15
    ny_grid = 3 if args.small else 10

    print(f"[diag] Building Yamano shell ({nx_grid}x{ny_grid})...")
    params = yamano_params()
    shell, x_vec, y_vec, le_nodes = build_yamano_shell(params, nx=nx_grid, ny=ny_grid)
    print(f"[diag] Shell built: nn={shell.nn}, ne={shell.ne}, ndof={shell.ndof}")

    V_inf = params['V_inf']
    L = params['Length']
    dx = L / nx_grid
    dt_uvlm = dx / V_inf
    dt_struct = dt_uvlm / 45

    # Target time (real seconds)
    t_target = TARGET_TSTAR * L / V_inf
    n_steps_target = int(np.ceil(t_target / dt_struct))
    n_steps_run = n_steps_target + 30  # small overshoot
    print(f"\nDiagnostic: run {n_steps_run} struct steps (target t*={TARGET_TSTAR})")
    print(f"  dt_struct = {dt_struct:.4e} s")
    print(f"  t at step {n_steps_target} = {n_steps_target * dt_struct:.4e} s "
          f"(t* = {n_steps_target * dt_struct * V_inf / L:.4f})")

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct,
        uvlm_dt_ratio=45,
        integrator='implicit',
        relaxation=0.95,
        newton_tol=1e-4,
        max_newton=30,
        max_particles=20000,
        wake_truncation=5.5,
        core_radius=1e-6,
        coupling='strong',
    )

    # Yamano pulse
    T_dur = 0.2 * L / V_inf
    f_density_ref = params['rho_fluid'] * V_inf**2 / params['thickness']
    F_body_peak = np.array([0.0, 0.0, -0.5 * f_density_ref])
    pulse = shell.distributed_load(F_body_peak)
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    t0 = time_mod.time()
    solver.run(n_steps_run, print_every=20)
    print(f"\nRuntime: {time_mod.time() - t0:.1f}s")

    # Resolve closest sample to TARGET_TSTAR
    tip_w = np.array(solver.tip_w_history)
    t_axis = np.arange(len(tip_w)) * dt_struct
    tstar_axis = t_axis * V_inf / L
    k = int(np.argmin(np.abs(tstar_axis - TARGET_TSTAR)))

    w_dim = tip_w[k]
    w_star = w_dim / L

    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC RESULT @ t* = {tstar_axis[k]:.4f}")
    print(f"  tip w (dim) = {w_dim:+.6e} m")
    print(f"  tip w*       = {w_star:+.6e}")
    print(f"  MATLAB ref   = {TARGET_W_STAR:+.6e}")
    print(f"  ratio        = {w_star / TARGET_W_STAR:+.3f}")
    print(f"  abs error    = {abs(w_star - TARGET_W_STAR):.3e}")
    print("=" * 60)

    # Surrounding context
    print("\nNeighborhood (every 5th sample):")
    for kk in range(max(0, k - 20), min(len(tip_w), k + 20), 5):
        marker = "  <-- target" if kk == k else ""
        print(f"  t*={tstar_axis[kk]:.4f}  w*={tip_w[kk]/L:+.4e}{marker}")


if __name__ == "__main__":
    main()
