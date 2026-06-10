"""Step-by-step full FSI comparison: Python vs MATLAB.

Full strong-coupling FSI (NO isolation of structure or aero), MATLAB-matched
dt and uvlm_ratio. Per-step comparison of q, dq using h_X_vec from MATLAB.

Node mapping by physical position (Python vs MATLAB use different orderings).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell
from load_matlab_fixture import MatlabFixture


def build_node_mapping(shell, fixture):
    """Build Python→MATLAB node index map by matching physical positions.

    Both initialize nodes at uniform grid. Python ordering: i*Ny + j for (i,j) chord-span.
    MATLAB ordering: typically column-major or different. Use rc_vec or initial q to recover positions.
    """
    n_nodes = shell.nn
    py_pos = shell.nodes.copy()      # (n_nodes, 3)
    # MATLAB initial node positions: stored as first 3 DOFs per node in q_vec at t=0
    hX = np.asarray(fixture._raw['h_X_vec'])
    q_init_ml = hX[:1584, 0]   # (1584,) — but actually all zeros at t=0 if initialized that way
    # Use the coordinates struct if available, otherwise reconstruct from x_vec, y_vec
    coords = fixture._raw.get('coordinates')
    if coords is not None:
        coords_ml = np.asarray(coords)
        if coords_ml.ndim == 2 and coords_ml.shape[1] == 2:
            # Add z=0
            ml_pos = np.column_stack([coords_ml, np.zeros(len(coords_ml))])
        else:
            ml_pos = coords_ml.reshape(-1, 3)
    else:
        # Reconstruct: MATLAB iterates i (chord) outer, j (span) inner per Plate_Mesh
        # For Yamano 15x10: x_vec = (0:15)/15, y_vec = (0:10)/10
        # MATLAB likely uses x_vec first index (chord) outer
        x_vec_ml = np.linspace(0, 1, 16)
        y_vec_ml = np.linspace(0, 1, 11)
        ml_pos = np.zeros((176, 3))
        idx = 0
        for j in range(11):     # span outer per MATLAB
            for i in range(16):  # chord inner per MATLAB
                ml_pos[idx] = [x_vec_ml[i], y_vec_ml[j], 0]
                idx += 1

    # Match by closest position
    py_to_ml = np.zeros(n_nodes, dtype=np.int32)
    for n in range(n_nodes):
        dists = np.sum((ml_pos - py_pos[n])**2, axis=1)
        py_to_ml[n] = np.argmin(dists)
    return py_to_ml, ml_pos


def map_python_q_to_matlab(q_py, py_to_ml):
    """Map Python q_vec to MATLAB DOF ordering."""
    n_nodes = len(py_to_ml)
    q_ml_ordered = np.zeros(n_nodes * 9)
    for n_py in range(n_nodes):
        n_ml = py_to_ml[n_py]
        q_ml_ordered[n_ml*9:(n_ml+1)*9] = q_py[n_py*9:(n_ml+1)*9]
    return q_ml_ordered


def main():
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    V_inf = params['V_inf']; L = params['Length']

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0., 0.]),
        rho_fluid=params['rho_fluid'],
        structural_dt=2e-4,   # MATLAB-matched
        uvlm_dt_ratio=34,     # MATLAB-matched
        integrator='implicit',
        relaxation=0.95, newton_tol=1e-8, max_newton=30,
        max_particles=5000, wake_truncation=5.5,
        coupling='strong')

    T_dur = 0.2 * L / V_inf
    f_dens = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0., 0., +0.5*f_dens]))
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    # Build node mapping
    fx_path = 'FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat'
    fx = MatlabFixture(fx_path)
    py_to_ml, ml_pos = build_node_mapping(shell, fx)
    print(f'Node mapping: Python→MATLAB. Sample: py[0]→ml[{py_to_ml[0]}], py[175]→ml[{py_to_ml[175]}]')
    print(f'Python node 0 pos: {shell.nodes[0]}, MATLAB node {py_to_ml[0]} pos: {ml_pos[py_to_ml[0]]}')
    print(f'Python node 175 pos: {shell.nodes[175]}, MATLAB node {py_to_ml[175]} pos: {ml_pos[py_to_ml[175]]}')

    hX = np.asarray(fx._raw['h_X_vec'])
    N_q_all = 1584

    print(f'\\n{"step":>4} {"t*":>6} {"|q-q_ml|_max":>14} {"|dq-dq_ml|_max":>16} {"py_tip":>12} {"ml_tip":>12} {"ratio":>7}')
    for step in range(1, 16):
        solver.run(1, print_every=0)
        py_q = solver.shell.q
        py_dq = solver.shell.dq

        # Map Python q, dq to MATLAB ordering
        q_py_ml_ord = np.zeros(N_q_all)
        dq_py_ml_ord = np.zeros(N_q_all)
        for n_py in range(shell.nn):
            n_ml = py_to_ml[n_py]
            q_py_ml_ord[n_ml*9:(n_ml+1)*9] = py_q[n_py*9:(n_py+1)*9]
            dq_py_ml_ord[n_ml*9:(n_ml+1)*9] = py_dq[n_py*9:(n_py+1)*9]

        # Read MATLAB q, dq at this step
        q_ml = hX[:N_q_all, step]
        dq_ml = hX[N_q_all:, step]

        # Compute differences
        q_diff = np.abs(q_py_ml_ord - q_ml)
        dq_diff = np.abs(dq_py_ml_ord - dq_ml)

        # Tip z-DOF (MATLAB node 175, dof 1532)
        # But py_to_ml might map differently, find the corresponding Python z-DOF for ml node 175
        py_node_for_ml175 = np.where(py_to_ml == 175)[0]
        if len(py_node_for_ml175) > 0:
            py_node_idx = py_node_for_ml175[0]
            py_tip = py_q[py_node_idx * 9 + 2]
        else:
            py_tip = float('nan')
        ml_tip = q_ml[1532]
        ratio = py_tip / ml_tip if abs(ml_tip) > 1e-15 else 0
        print(f'{step:>4} {step*2e-3:>6.3f} {q_diff.max():>14.4e} {dq_diff.max():>16.4e} {py_tip:>+12.4e} {ml_tip:>+12.4e} {ratio:>7.4f}')


if __name__ == '__main__':
    main()
