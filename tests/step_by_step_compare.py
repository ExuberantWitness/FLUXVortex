"""Step-by-step Python vs MATLAB comparison.

Runs Python Yamano simulation, records full ANCF state (q, dq, F_aero, F_pulse)
at every structural step. Then compares against MATLAB's h_X_vec trajectory at
common t* values to locate the first divergence point.

MATLAB d_t = 2e-3, 151 steps → t* ∈ [0, 0.3]
Python dt_struct = (1/15)/10/45 ≈ 1.481e-4, 160 steps → t ∈ [0, 0.0237]
Ratio Python/MATLAB step rate = 13.5 (one MATLAB step ≈ 13.5 Python steps).
"""
import os, sys, time as time_mod, functools
import numpy as np
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

from run_standalone_yamano import yamano_params, build_yamano_shell
from load_matlab_fixture import MatlabFixture


def run_python_with_recording(n_steps=140, nx=15, ny=10):
    """Run Python smoke and record full q, dq at every step."""
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    V_inf = params['V_inf']; L = params['Length']
    dt_struct = (L/nx)/V_inf/45

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0.0, 0.0]),
        rho_fluid=params['rho_fluid'],
        structural_dt=dt_struct, uvlm_dt_ratio=45,
        integrator='implicit', relaxation=0.95,
        newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, coupling='strong')

    T_dur = 0.2 * L / V_inf
    f_dens = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0., 0., +0.5*f_dens]))   # +z to match MATLAB
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    # Monkey-patch _record_history to record q AND dq
    q_history = []
    dq_history = []
    original_record = solver._record_history
    def patched_record():
        q_history.append(solver.shell.q.copy())
        dq_history.append(solver.shell.dq.copy())
        original_record()
    solver._record_history = patched_record

    print(f"[py] Running {n_steps} struct steps to t={n_steps*dt_struct:.4f}s, "
          f"t*={n_steps*dt_struct*V_inf/L:.4f}...")
    t0 = time_mod.time()
    solver.run(n_steps, print_every=30)
    print(f"[py] Done in {time_mod.time()-t0:.0f}s, recorded {len(q_history)} steps")

    return {
        'q': np.array(q_history),       # (n_steps, ndof)
        'dq': np.array(dq_history),
        'dt_struct': dt_struct,
        'V_inf': V_inf,
        'L': L,
        'tip_w': np.array(solver.tip_w_history),
    }


def compare_step_by_step(py_data, fx_path, n_checkpoints=10):
    """Compare Python and MATLAB q_vec at common t*."""
    fx = MatlabFixture(fx_path)
    hX = np.asarray(fx._raw['h_X_vec'])  # (3168, 151) = (2*N_q_all, n_time)
    d_t_ml = 2e-3
    n_ml_steps = hX.shape[1]
    Nx, Ny = 15, 10
    N_q_all = (Nx+1)*(Ny+1)*9   # 1584

    # Extract q and dq from h_X_vec (first half is q, second half is dq)
    q_ml = hX[:N_q_all, :]
    dq_ml = hX[N_q_all:, :]

    # Python time axis
    n_py = len(py_data['q'])
    dt_struct = py_data['dt_struct']
    V_inf = py_data['V_inf']; L = py_data['L']
    t_py = np.arange(n_py) * dt_struct * V_inf / L  # t*
    t_ml = np.arange(n_ml_steps) * d_t_ml            # t* (already nondim since U_in=1)

    print(f"\n══ Step-by-step comparison ══")
    print(f"{'t*':>7} {'tip_z_ml':>11} {'tip_w_py':>11} {'ratio':>8} "
          f"{'|q_diff|_max':>14} {'|dq_diff|_max':>14}")

    # Pick t* checkpoints
    checkpoints = np.linspace(t_py[0], min(t_py[-1], t_ml[-1]) * 0.95, n_checkpoints + 1)[1:]
    for tk in checkpoints:
        k_py = int(np.argmin(np.abs(t_py - tk)))
        k_ml = int(np.argmin(np.abs(t_ml - tk)))
        # tip at node 175, z dof = 1532
        tip_ml = q_ml[1532, k_ml]
        tip_py = py_data['tip_w'][k_py]
        ratio = tip_py / tip_ml if abs(tip_ml) > 1e-12 else float('nan')

        # Full q diff
        q_diff = np.abs(py_data['q'][k_py] - q_ml[:, k_ml])
        dq_diff = np.abs(py_data['dq'][k_py] - dq_ml[:, k_ml])
        print(f"{tk:>7.4f} {tip_ml:>+11.3e} {tip_py:>+11.3e} {ratio:>+8.3f} "
              f"{q_diff.max():>14.3e} {dq_diff.max():>14.3e}")


def main():
    py_data = run_python_with_recording(n_steps=140)
    np.savez('/tmp/py_step_data.npz', **py_data)
    print("[py] Saved trajectory to /tmp/py_step_data.npz\n")
    fx_path = '/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat'
    compare_step_by_step(py_data, fx_path, n_checkpoints=10)


if __name__ == "__main__":
    main()
