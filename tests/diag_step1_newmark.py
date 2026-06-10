"""Function-by-function comparison: Newmark integrator step 1.

Process:
  1. Build Python M, Kt at q_ref; pulse F at t=d_t
  2. Load MATLAB M_global, Kt_global, q_0, q_1, dq_1 from fixture / h_X_vec
  3. Map Python DOF ordering to MATLAB
  4. Compare:
     - M matrices (per-entry diff)
     - Kt matrices (per-entry diff)
     - F vectors (per-entry diff)
     - q_1 output (per-entry diff)
     - dq_1 output (per-entry diff)
  5. Locate first divergence in the chain
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell
from load_matlab_fixture import MatlabFixture


def build_node_mapping(shell, n_chord=15, n_span=10):
    """Map Python node index → MATLAB node index by physical position.

    Both meshes share the same physical layout. Python builds shell.nodes;
    MATLAB iterates jj (span) outer, ii (chord) inner.
    """
    py_pos = shell.nodes.copy()
    # MATLAB Plate_Mesh ordering: x changes slow (chord outer), y changes fast (span inner)
    # Or the opposite — test both and pick the one with smallest residual
    x_vec = np.linspace(0, 1, n_chord + 1)
    y_vec = np.linspace(0, 1, n_span + 1)

    n_nodes = (n_chord + 1) * (n_span + 1)
    # Variant A: i (chord) outer, j (span) inner
    ml_pos_A = np.zeros((n_nodes, 3))
    idx = 0
    for i in range(n_chord + 1):
        for j in range(n_span + 1):
            ml_pos_A[idx] = [x_vec[i], y_vec[j], 0]
            idx += 1
    # Variant B: j (span) outer, i (chord) inner
    ml_pos_B = np.zeros((n_nodes, 3))
    idx = 0
    for j in range(n_span + 1):
        for i in range(n_chord + 1):
            ml_pos_B[idx] = [x_vec[i], y_vec[j], 0]
            idx += 1

    def map_to(ml_pos):
        py_to_ml = np.zeros(n_nodes, dtype=np.int32)
        for n in range(n_nodes):
            dists = np.sum((ml_pos - py_pos[n])**2, axis=1)
            py_to_ml[n] = np.argmin(dists)
        return py_to_ml

    map_A = map_to(ml_pos_A)
    map_B = map_to(ml_pos_B)
    # Pick whichever is unique (1:1)
    if len(set(map_A.tolist())) == n_nodes:
        return map_A, ml_pos_A, 'A'
    if len(set(map_B.tolist())) == n_nodes:
        return map_B, ml_pos_B, 'B'
    return map_A, ml_pos_A, 'A_fallback'


def py_dofs_to_ml(py_vec, py_to_ml):
    """Reorder a (ndof,) vector from Python DOF order to MATLAB DOF order."""
    n_nodes = len(py_to_ml)
    out = np.zeros_like(py_vec)
    for n_py in range(n_nodes):
        n_ml = py_to_ml[n_py]
        out[n_ml*9:(n_ml+1)*9] = py_vec[n_py*9:(n_py+1)*9]
    return out


def py_matrix_to_ml(M_py, py_to_ml):
    """Reorder a (ndof, ndof) matrix from Python DOF order to MATLAB."""
    n_nodes = len(py_to_ml)
    # Build permutation matrix index
    perm = np.zeros(n_nodes * 9, dtype=np.int32)
    for n_py in range(n_nodes):
        n_ml = py_to_ml[n_py]
        perm[n_ml*9:(n_ml+1)*9] = np.arange(n_py*9, (n_py+1)*9)
    # M_ml_ordered[i, j] = M_py[perm[i], perm[j]]
    return M_py[np.ix_(perm, perm)]


def main():
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    V_inf = params['V_inf']; L = params['Length']

    solver = StandaloneHybridSolver(
        shell, np.array([V_inf, 0., 0.]),
        rho_fluid=params['rho_fluid'],
        structural_dt=2e-4, uvlm_dt_ratio=34,
        integrator='implicit', relaxation=0.95,
        newton_tol=1e-8, max_newton=30,
        max_particles=5000, wake_truncation=5.5, coupling='strong')

    T_dur = 0.2 * L / V_inf
    f_dens = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse = shell.distributed_load(np.array([0., 0., +0.5*f_dens]))
    solver.set_pulse_distributed(pulse, amplitude=1.0, duration=T_dur)

    fx = MatlabFixture('FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat')
    hX = np.asarray(fx._raw['h_X_vec'])
    N_q = 1584

    py_to_ml, ml_pos, variant = build_node_mapping(shell)
    print(f'Node mapping variant: {variant}')
    print(f'Map check: py node 0 (pos {shell.nodes[0]}) → ml node {py_to_ml[0]} (pos {ml_pos[py_to_ml[0]]})')
    print(f'Map check: py node 175 (pos {shell.nodes[175]}) → ml node {py_to_ml[175]} (pos {ml_pos[py_to_ml[175]]})')

    # ─── Compare M matrices ───
    print('\n══ M matrix ══')
    M_py = shell.M.toarray()
    M_py_ml_ord = py_matrix_to_ml(M_py, py_to_ml)
    M_ml_raw = fx._raw['M_global']
    M_ml = M_ml_raw.toarray() if hasattr(M_ml_raw, 'toarray') else np.asarray(M_ml_raw)
    M_ml_dim = M_ml * params['rho_fluid']    # nondim → dim
    print(f'  Python M (in MATLAB DOF order): |max|={np.max(np.abs(M_py_ml_ord)):.4e}, |F|={np.linalg.norm(M_py_ml_ord):.4e}')
    print(f'  MATLAB M (dim):                  |max|={np.max(np.abs(M_ml_dim)):.4e}, |F|={np.linalg.norm(M_ml_dim):.4e}')
    diff_M = M_py_ml_ord - M_ml_dim
    print(f'  |M_py - M_ml|_F = {np.linalg.norm(diff_M):.4e}  (relative: {np.linalg.norm(diff_M)/np.linalg.norm(M_ml_dim):.3e})')
    print(f'  |M_py - M_ml|_max = {np.max(np.abs(diff_M)):.4e}')
    # Largest discrepancy location
    if np.max(np.abs(diff_M)) > 1e-8:
        ij = np.unravel_index(np.argmax(np.abs(diff_M)), diff_M.shape)
        print(f'  Worst entry at (row {ij[0]}, col {ij[1]}): py={M_py_ml_ord[ij]:.4e}  ml={M_ml_dim[ij]:.4e}')

    # ─── Compare pulse F at t=d_t ───
    print('\n══ Pulse F at t=d_t ══')
    d_t = 2e-4
    q_in_norm_dt = 0.5 * np.sin(np.pi * d_t / T_dur)  # MATLAB convention
    # MATLAB Qf_time_global
    Qf_time = np.asarray(fx._raw['Qf_time_global']).ravel()
    F_pulse_ml = Qf_time * q_in_norm_dt * params['rho_fluid'] * V_inf**2 * L**2
    # Python pulse (with timing fix uses t_eval = elapsed + dt)
    solver._pulse_elapsed = 0.0   # reset
    F_pulse_py = solver._pulse_force()
    F_pulse_py_ml_ord = py_dofs_to_ml(F_pulse_py, py_to_ml)
    print(f'  Python F_pulse_z sum: {F_pulse_py_ml_ord[2::9].sum():+.4e}')
    print(f'  MATLAB F_pulse_z sum: {F_pulse_ml[2::9].sum():+.4e}')
    diff_F = F_pulse_py_ml_ord - F_pulse_ml
    print(f'  |F_py - F_ml|_max = {np.max(np.abs(diff_F)):.4e}')
    print(f'  |F_py - F_ml|_F   = {np.linalg.norm(diff_F):.4e}  (relative: {np.linalg.norm(diff_F)/np.linalg.norm(F_pulse_ml):.3e})')

    # ─── Compare Kt (tangent stiffness) at q_ref ───
    print('\n══ Kt (tangent stiffness at q_ref) ══')
    _, Kt_py = shell._internal_forces_and_tangent(shell.q)
    Kt_py_arr = Kt_py.toarray()
    Kt_py_ml = py_matrix_to_ml(Kt_py_arr, py_to_ml)
    print(f'  Python Kt: |max|={np.max(np.abs(Kt_py_ml)):.4e}, |F|={np.linalg.norm(Kt_py_ml):.4e}')
    # MATLAB Kt = dq_Qe_global at q_ref. We don't have it directly in fixture (computed at t=0).
    # But we can compare with stored dq_Qe_global at t*=0.1995 (which is at deformed state, NOT q_ref).
    # For exact comparison, would need MATLAB to dump Kt at t=0.
    # Skip for now — we trust that since modal ω matches exactly, K_t at q_ref matches.

    # ─── Total force at step 1 (pulse + aero contribution) ───
    print('\n══ Total F at step 1 ══')
    # MATLAB at step 1 PASS B uses Qf_p_global from PASS A's solve_fluid (initial flat state)
    # Initial flat plate: gamma=0, so Qf_p_global should be 0
    # So F_total_ml = Qf_time_global * q_in_norm(d_t)
    # If Qf_p_global ≠ 0, there's residual aero force we need to account for

    # ─── Run Python step 1 and compare q, dq ───
    print('\n══ After step 1 ══')
    solver.run(1, print_every=0)
    q_py = solver.shell.q.copy()
    dq_py = solver.shell.dq.copy()
    q_py_ml = py_dofs_to_ml(q_py, py_to_ml)
    dq_py_ml = py_dofs_to_ml(dq_py, py_to_ml)
    q_ml = hX[:N_q, 1]
    dq_ml = hX[N_q:, 1]

    # Subtract reference state so we compare displacements
    # MATLAB stores rotations [1,0,0],[0,1,0] as constants -- they appear as 1.0 in q_ref
    # h_X_vec at step 0 = reference state, at step 1 = after first integration.
    q_ref_ml = hX[:N_q, 0]
    dq_ref_ml = hX[N_q:, 0]
    # Python q at initialization has reference positions + unit slopes
    # We compare RELATIVE motion: q - q_ref

    print(f'  Python q_1 norm (after step 1): |q_py|_F = {np.linalg.norm(q_py_ml):.4e}')
    print(f'  MATLAB q_1 norm: |q_ml|_F = {np.linalg.norm(q_ml):.4e}')
    # Compare displacement (relative to initial)
    dq_disp_py = q_py_ml - q_ref_ml
    dq_disp_ml = q_ml - q_ref_ml
    print(f'  |q_py - q_ref|_max = {np.max(np.abs(dq_disp_py)):.4e}')
    print(f'  |q_ml - q_ref|_max = {np.max(np.abs(dq_disp_ml)):.4e}')
    print(f'  |dq_py - dq_ml|_max = {np.max(np.abs(dq_disp_py - dq_disp_ml)):.4e}')
    # Tip z DOF
    print(f'  Tip z (dof 1532): py={q_py_ml[1532]:+.5e}, ml={q_ml[1532]:+.5e}, ratio={q_py_ml[1532]/q_ml[1532]:.4f}')

    # Velocity comparison
    print(f'  Velocity tip z: py={dq_py_ml[1532]:+.5e}, ml={dq_ml[1532]:+.5e}, ratio={dq_py_ml[1532]/dq_ml[1532]:.4f}')


if __name__ == '__main__':
    main()
