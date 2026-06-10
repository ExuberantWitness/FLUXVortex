"""Smoke test: confirm production _uvlm_step wires Mf2_vec1 correctly.

Runs ~100 struct steps of Yamano (small grid), prints Mf2_vec1 magnitude and
tip_w trajectory. Should run quickly and produce non-zero Mf2_vec1.
"""
import os, sys, time as time_mod
import numpy as np
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

from run_standalone_yamano import yamano_params, build_yamano_shell


def main():
    print("[smoke] Setup …")
    params = yamano_params()
    nx, ny = 15, 10  # Yamano benchmark grid
    shell, x_vec, y_vec, le_nodes = build_yamano_shell(params, nx=nx, ny=ny)
    print(f"[smoke] Shell built: nn={shell.nn}, ne={shell.ne}, ndof={shell.ndof}")

    V_inf = params['V_inf']; L = params['Length']
    # MATLAB exact match: d_t_nondim = 2e-3 → dt_struct = d_t * L / V_inf = 2e-4 s
    # dt_wake_per_dt = 34 (MATLAB ceil((1/15)/2e-3))
    dt_struct = 2e-4
    n_steps = 110   # 110 steps × 2e-4 = 0.022s → t* = 0.22

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct,
        uvlm_dt_ratio=34,    # MATLAB dt_wake_per_dt
        integrator='implicit',
        relaxation=1.0, newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, core_radius=1e-6,
        coupling='strong',
    )

    T_dur = 0.2 * L / V_inf
    f_density_ref = params['rho_fluid'] * V_inf**2 / params['thickness']
    # MATLAB q_in_vec = [0, 0, +1] (positive z), q_in_norm peak = 0.5
    F_body_peak = np.array([0.0, 0.0, +0.5 * f_density_ref])
    pulse = shell.distributed_load(F_body_peak)
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    print(f"[smoke] Run {n_steps} struct steps …")
    t0 = time_mod.time()
    solver.run(n_steps, print_every=15)
    print(f"[smoke] Done in {time_mod.time() - t0:.1f}s")

    print()
    print(f"[smoke] Final tip_w  = {solver.tip_w_history[-1]:.4e} m")
    print(f"[smoke] Mf2_vec1 |max| = {np.max(np.abs(solver.uvlm.Mf2_vec1)):.4e}")
    print(f"[smoke] Mf2_vec1 rms   = {np.sqrt(np.mean(solver.uvlm.Mf2_vec1**2)):.4e}")
    print(f"[smoke] forces_no_vstruct |max| = "
          f"{np.max(np.abs(solver.uvlm.forces_no_vstruct)):.4e}")
    print(f"[smoke] wake rows tracked = {len(solver.uvlm.wake_vertices)}")

    # Yamano comparison at t* = 0.1995
    tip_w = np.array(solver.tip_w_history)
    t_axis = np.arange(len(tip_w)) * dt_struct
    tstar = t_axis * V_inf / L
    k = int(np.argmin(np.abs(tstar - 0.1995)))
    w_star = tip_w[k] / L
    print(f"\n[smoke] === Yamano comparison ===")
    print(f"[smoke]   t* = {tstar[k]:.4f}  tip w = {tip_w[k]:+.4e} m  w* = {w_star:+.4e}")
    print(f"[smoke]   MATLAB ref      w* = -1.489e-03")
    print(f"[smoke]   ratio Python/ML = {w_star / -1.489e-3:+.3f}")


if __name__ == "__main__":
    main()
