"""Pulse-only diagnostic: disable UVLM aero, run structure under pulse + Mf1
added-mass. If tip_w at t*=0.1995 is far from Yamano -1.489e-3, the bug is in
the STRUCTURAL setup (mass, stiffness, pulse scaling). If close, the bug is
in the aerodynamic forcing.
"""
import os, sys, time as time_mod, functools
import numpy as np
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

from run_standalone_yamano import yamano_params, build_yamano_shell


def main():
    params = yamano_params()
    nx, ny = 6, 4
    shell, x_vec, y_vec, le_nodes = build_yamano_shell(params, nx=nx, ny=ny)
    V_inf = params['V_inf']; L = params['Length']
    dt_uvlm = (L / nx) / V_inf
    dt_struct = dt_uvlm / 45
    n_steps = 90

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct, uvlm_dt_ratio=45,
        integrator='implicit', relaxation=0.95,
        newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, core_radius=1e-6,
        coupling='weak',
    )

    # Disable aero entirely by stubbing out UVLM step methods
    def _noop(*args, **kwargs):
        pass
    solver._uvlm_step = _noop
    solver._uvlm_step_initial = _noop
    # Also zero out forces_no_vstruct so _load_transfer sees zeros
    solver.uvlm.forces_no_vstruct[:] = 0.0
    solver.uvlm.forces[:] = 0.0

    # Yamano pulse
    T_dur = 0.2 * L / V_inf
    f_density_ref = params['rho_fluid'] * V_inf**2 / params['thickness']
    F_body_peak = np.array([0.0, 0.0, -0.5 * f_density_ref])
    pulse = shell.distributed_load(F_body_peak)
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    t0 = time_mod.time()
    solver.run(n_steps, print_every=20)
    print(f"\nRuntime: {time_mod.time() - t0:.1f}s")

    tip_w = np.array(solver.tip_w_history)
    t_axis = np.arange(len(tip_w)) * dt_struct
    tstar = t_axis * V_inf / L
    k = int(np.argmin(np.abs(tstar - 0.1995)))
    w_star = tip_w[k] / L
    print()
    print("=== Pulse-only result ===")
    print(f"  t* = {tstar[k]:.4f}  tip w = {tip_w[k]:+.4e} m  w* = {w_star:+.4e}")
    print(f"  MATLAB ref      w* = -1.489e-03 (with aero)")
    print(f"  ratio Python_pulseonly / ML_with_aero = {w_star / -1.489e-3:+.3f}")
    print()
    if abs(w_star / -1.489e-3) > 1.5:
        print("  ⚠ Pulse-only already >> Yamano ref → structural/pulse setup is the bug")
    else:
        print("  ✓ Pulse-only matches Yamano scale → aero forces are the bug source")


if __name__ == "__main__":
    main()
