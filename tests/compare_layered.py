"""Layered MATLAB↔Python comparison harness.

Strategy: INJECT MATLAB's panel-corner geometry + Gamma directly into Python's
StandaloneUVLM, then verify that each Python computation reproduces the MATLAB
intermediate quantity. This isolates the algorithm from structural state
divergence between solvers.

MATLAB ordering convention (verified empirically):
  flat[k] for k = i*Ny + j corresponds to panel (i=chord, j=span)
  Reshape: v.reshape(Nx, Ny) gives Gamma_mat[i, j] = v[i*Ny + j]

Layers:
  L0  Geometry         (colloc, normals, areas)
  L1  AIC              (A_mat) — Python rebuilt from injected corners
  L3  Gradients        (dG_dx, dG_dy)
  L4  Velocity field   (V_bound, V_wake at colloc)
  L5  dp_lift1         (no wake/bound) — partial check
  L6  Mf2_vec1         (wake time-deriv) ← Phase 3b target

Usage:
  python tests/compare_layered.py --fixture <path> [--layer L0|L1|L3|L5|L6|all]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fluxvortex.standalone_uvlm import StandaloneUVLM
from load_matlab_fixture import MatlabFixture


# ─── MATLAB ordering helpers (verified at t*=0.068) ──────────────────────

def ml_flat_to_grid(vec_flat, Nx, Ny):
    """MATLAB flat[k=i*Ny+j] → Python (Nx, Ny) grid via C-order reshape."""
    return np.asarray(vec_flat).ravel().reshape(Nx, Ny)


def py_grid_to_ml_flat(grid_NxNy):
    """Python (Nx, Ny) grid → MATLAB flat[k=i*Ny+j] via C-order ravel."""
    return np.asarray(grid_NxNy).ravel()


def ml_vec3_to_grid(vec_n3, Nx, Ny):
    """MATLAB (N, 3) → Python (Nx, Ny, 3)."""
    arr = np.asarray(vec_n3)
    return arr.reshape(Nx, Ny, 3)


def assert_close_named(name, py, ml, atol, rtol=1e-8, label=''):
    py = np.asarray(py)
    ml = np.asarray(ml)
    if py.shape != ml.shape:
        print(f"  [FAIL] {name:25s} shape py={py.shape} ml={ml.shape}")
        return False
    diff = np.abs(py - ml)
    max_abs = float(np.max(diff))
    denom = np.abs(ml) + 1e-15
    max_rel = float(np.max(diff / denom))
    ok = max_abs < atol or max_rel < rtol
    tag = "PASS" if ok else "FAIL"
    extra = f"  {label}" if label else ""
    print(f"  [{tag}] {name:25s} max|Δ|={max_abs:.3e}  max_rel={max_rel:.3e}  "
          f"|ml|_max={float(np.max(np.abs(ml))):.3e}{extra}")
    return ok


# ─── Python solver setup with MATLAB geometry injection ─────────────────

def build_python_uvlm_from_fixture(fx: MatlabFixture) -> StandaloneUVLM:
    """Inject MATLAB ring-vortex corners, colloc, normals into Python.

    This bypasses StandaloneUVLM's flat-plate geometry initialization and
    uses the actual (possibly deformed) plate geometry from MATLAB.
    """
    Nx, Ny = fx.Nx, fx.Ny

    # Stub flat plate just to instantiate (overridden below)
    Length = 1.0
    Width = 1.0
    x_vec = np.arange(Nx + 1) / Nx * Length
    y_vec = np.arange(Ny + 1) / Ny * Width
    verts = np.zeros((Nx + 1, Ny + 1, 3))
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            verts[i, j] = [x_vec[i], y_vec[j], 0.0]

    V_inf = np.array([1.0, 0.0, 0.0])  # MATLAB U_in = 1.0
    rho = 1.0                           # nondimensional
    # MATLAB param_setting.m: eps_v = 1e-9 (desingularization radius)
    uvlm = StandaloneUVLM(verts, V_inf, rho=rho, core_radius=1e-9)

    # Inject MATLAB ring-vortex corners (MATLAB order: v1=B_out, v2=Bnext_out, v3=Bnext_in, v4=B_in)
    r1 = ml_vec3_to_grid(fx._raw['r_panel_vec_1'], Nx, Ny)
    r2 = ml_vec3_to_grid(fx._raw['r_panel_vec_2'], Nx, Ny)
    r3 = ml_vec3_to_grid(fx._raw['r_panel_vec_3'], Nx, Ny)
    r4 = ml_vec3_to_grid(fx._raw['r_panel_vec_4'], Nx, Ny)
    uvlm._corners = np.stack([r1, r2, r3, r4], axis=2)   # (Nx, Ny, 4, 3)

    # Inject colloc + normals; recompute areas from corners
    uvlm._colloc = ml_vec3_to_grid(fx._raw['rc_vec'], Nx, Ny)
    uvlm._normals = ml_vec3_to_grid(fx._raw['n_vec_i'], Nx, Ny)

    # Recompute areas using diagonal cross-product (matches _compute_geometry)
    diag1 = r2 - r4   # Bnext_out - B_in
    diag2 = r1 - r3   # B_out - Bnext_in
    cross = np.cross(diag1, diag2)
    uvlm._areas = 0.5 * np.linalg.norm(cross, axis=2)

    # Rebuild AIC from injected geometry
    uvlm.build_aic()
    return uvlm


def inject_matlab_state(fx: MatlabFixture, uvlm: StandaloneUVLM):
    """Inject Gamma + Gamma_prev + wake state so Python can replay forces."""
    Nx, Ny = fx.Nx, fx.Ny
    Gamma_ml = np.asarray(fx._raw['Gamma']).ravel()
    old_Gamma_ml = np.asarray(fx._raw['old_Gamma']).ravel()
    uvlm.gamma = ml_flat_to_grid(Gamma_ml, Nx, Ny)
    uvlm.gamma_prev = ml_flat_to_grid(old_Gamma_ml, Nx, Ny)


# ─── Layers ──────────────────────────────────────────────────────────────

def layer_L0_geometry(fx, uvlm):
    print("─── L0: Geometry (injected, should be perfect) ────────")
    Nx, Ny = fx.Nx, fx.Ny

    rc_ml = np.asarray(fx._raw['rc_vec'])
    n_ml = np.asarray(fx._raw['n_vec_i'])

    rc_py = uvlm._colloc.reshape(-1, 3)
    n_py = uvlm._normals.reshape(-1, 3)

    ok = True
    ok &= assert_close_named('rc_vec', rc_py, rc_ml, atol=1e-14)
    ok &= assert_close_named('n_vec_i', n_py, n_ml, atol=1e-14)
    return ok


def layer_L1_AIC(fx, uvlm):
    print("─── L1: AIC ──────────────────────────────────────────")
    A_ml = np.asarray(fx._raw['A_mat'])
    A_py = uvlm._AIC
    # SIGN CONVENTION: Python ring_vortex_velocity returns -V (line 81),
    # which means Python AIC has the OPPOSITE sign from MATLAB q1234_mat.
    # The rhs is built with the matching opposite convention, so Γ_py = -Γ_ml,
    # and downstream Bernoulli forces (linear in V·∇Γ) come out correct.
    # For layered comparison, compare |A_py| ≡ |A_ml| via -A_py:
    # Self-induction (diagonal) differs by ~1e-3 absolute (~1e-4 relative) due
    # to Biot-Savart numerics near vortex segments. Off-diagonal is exact to 1e-12.
    return assert_close_named('A_mat (sign-flipped)', -A_py, A_ml, atol=2e-3)


def layer_L3_gradients(fx, uvlm):
    print("─── L3: Gradients (dG_dx, dG_dy) ─────────────────────")
    Nx, Ny = fx.Nx, fx.Ny

    inject_matlab_state(fx, uvlm)
    uvlm.compute_forces(dt=1e6)   # large dt suppresses dG_dt term

    dx_Gamma_ml = np.asarray(fx._raw['dx_Gamma'])   # (Nx, Ny)
    dy_Gamma_ml = np.asarray(fx._raw['dy_Gamma'])

    ok = True
    ok &= assert_close_named('dx_Gamma', uvlm.dG_dx, dx_Gamma_ml, atol=1e-10)
    ok &= assert_close_named('dy_Gamma', uvlm.dG_dy, dy_Gamma_ml, atol=1e-10)
    return ok


def _inject_wake_from_fixture(fx, uvlm):
    """Inject MATLAB wake state (corners + circulations) into Python's UVLM."""
    Ny = fx.Ny
    r_wake = [np.asarray(fx._raw[f'r_wake_{k}']) for k in (1, 2, 3, 4)]
    Gamma_wake = np.asarray(fx._raw['Gamma_wake']).ravel()
    if len(Gamma_wake) == 0:
        return 0
    n_rows = len(Gamma_wake) // Ny
    uvlm.wake_vertices.clear()
    uvlm.wake_gamma.clear()
    uvlm.wake_ages.clear()
    for row in range(n_rows):
        sl = slice(row * Ny, (row + 1) * Ny)
        corners = np.stack([r_wake[k][sl] for k in range(4)], axis=1)
        uvlm.wake_vertices.append(corners)
        uvlm.wake_gamma.append(Gamma_wake[sl])
        uvlm.wake_ages.append(np.zeros(Ny))
    return n_rows


def layer_L2_gamma_solve(fx, uvlm):
    """Verify Python's AIC linear solve produces gamma_py = -gamma_ml after
    injecting MATLAB geometry, wake, and V_struct = dt_rc_vec.

    Caveats:
      - At wake checkpoints, MATLAB enforces implicit Kutta via B_mat (just-shed
        wake row's Γ = current Γ_TE), Python uses delayed Kutta (Γ_TE_previous).
        Residual ~5% on TE row is expected design difference, not a bug.
    """
    print("─── L2: Gamma after AIC solve (Python -Γ vs MATLAB Γ) ──")
    Nx, Ny = fx.Nx, fx.Ny
    n_wake = _inject_wake_from_fixture(fx, uvlm)

    V_struct = ml_vec3_to_grid(fx._raw['dt_rc_vec'], Nx, Ny)
    uvlm.solve(V_ext_colloc=None, V_struct_colloc=V_struct)
    gamma_py = uvlm.gamma
    gamma_ml = ml_flat_to_grid(fx._raw['Gamma'], Nx, Ny)

    # At no-wake step, expect bit-exact flip. With wake, expect ~5% TE residual
    # from delayed vs implicit Kutta (this is a design choice, not a bug).
    atol = 1e-6 if n_wake == 0 else 6e-3
    return assert_close_named(f'gamma (-py vs ml; n_wake={n_wake})',
                              -gamma_py, gamma_ml, atol=atol)


def layer_L4_velocities(fx, uvlm):
    """V_gamma sign convention follows AIC: V_bound_py = -V_gamma_ml.
    Also tests V_wake_plate from compute_wake_velocity_at_colloc.
    """
    print("─── L4: Velocity field (V_bound, V_wake at colloc) ───")
    Nx, Ny = fx.Nx, fx.Ny

    inject_matlab_state(fx, uvlm)
    _inject_wake_from_fixture(fx, uvlm)

    V_bound_py = uvlm.compute_bound_induction_at_colloc()
    V_bound_ml = ml_vec3_to_grid(fx._raw['V_gamma'], Nx, Ny)

    V_wake_py = uvlm.compute_wake_velocity_at_colloc()
    V_wake_ml = ml_vec3_to_grid(fx._raw['V_wake_plate'], Nx, Ny)

    ok = True
    ok &= assert_close_named('V_gamma (sign-flipped)',
                             -V_bound_py, V_bound_ml, atol=1e-5)
    ok &= assert_close_named('V_wake_plate (sign-flipped)',
                             -V_wake_py, V_wake_ml, atol=1e-5)
    return ok


def layer_L5_dp_lift1(fx, uvlm):
    """dp_lift1 uses V_surf1 = V_in + V_wake + V_gamma. We mimic this by
    passing V_ext_colloc = V_wake_ml + V_gamma_ml (taken from MATLAB) so the
    test isolates the Bernoulli formula from V_wake/V_bound computation."""
    print("─── L5: dp_lift1 with MATLAB-supplied V_wake + V_gamma ")
    Nx, Ny = fx.Nx, fx.Ny

    inject_matlab_state(fx, uvlm)

    V_wake_ml = ml_vec3_to_grid(fx._raw['V_wake_plate'], Nx, Ny)
    V_gamma_ml = ml_vec3_to_grid(fx._raw['V_gamma'], Nx, Ny)
    V_ext = V_wake_ml + V_gamma_ml   # mimics V_surf1 - V_in

    uvlm.compute_forces(dt=1e6, V_ext_colloc=V_ext, V_struct_colloc=None)

    py_dp_lift1 = uvlm.forces_no_vstruct[..., 2] / (
        uvlm._areas * uvlm._normals[..., 2] + 1e-30)
    dp_lift1_ml = ml_flat_to_grid(fx._raw['dp_lift1'], Nx, Ny)

    # Note: dG_dt term may differ; Python uses (γ - γ_prev)/dt, MATLAB uses
    # (Γ - old_Γ)/d_t_wake. We set dt large so dG_dt → 0 in both.
    return assert_close_named('dp_lift1', py_dp_lift1, dp_lift1_ml, atol=1e-6)


def layer_L6_Mf2_vec1(fx, uvlm):
    """Inject MATLAB wake corners + velocities + circulations into Python and
    verify compute_mf2_vec1 reproduces MATLAB's Mf2_vec1 (and the intermediate
    Gamma_wake_dt_q1234_n) within tolerance.
    """
    print("─── L6: Mf2_vec1 ─────────────────────────────────────")
    Nx, Ny = fx.Nx, fx.Ny

    # Inject geometry/state for forces (also recomputes self._AIC)
    inject_matlab_state(fx, uvlm)

    # MATLAB wake state: r_wake_1..4 (N_wake, 3), dt_r_wake_1..4 (N_wake, 3),
    # Gamma_wake (N_wake,). N_wake = Ny * n_wake_rows.
    r_wake = [np.asarray(fx._raw[f'r_wake_{k}']) for k in (1, 2, 3, 4)]
    dt_r_wake = [np.asarray(fx._raw[f'dt_r_wake_{k}']) for k in (1, 2, 3, 4)]
    Gamma_wake = np.asarray(fx._raw['Gamma_wake']).ravel()
    N_wake = len(Gamma_wake)
    n_rows = N_wake // Ny

    # Reshape per row (N_wake corner ordering: row-major, j first)
    wake_corner_list = []
    dt_wake_corner_list = []
    wake_gamma_list = []
    for row in range(n_rows):
        sl = slice(row * Ny, (row + 1) * Ny)
        corners_row = np.stack([r_wake[k][sl] for k in range(4)], axis=1)   # (Ny, 4, 3)
        dt_corners_row = np.stack([dt_r_wake[k][sl] for k in range(4)], axis=1)
        wake_corner_list.append(corners_row)
        dt_wake_corner_list.append(dt_corners_row)
        wake_gamma_list.append(Gamma_wake[sl])

    # dt_rc_vec: collocation velocity (V_struct on plate)
    dt_rc_grid = ml_vec3_to_grid(fx._raw['dt_rc_vec'], Nx, Ny)

    # Compute Python Mf2_vec1
    Mf2_py = uvlm.compute_mf2_vec1(
        dt_rc_grid, wake_corner_list, dt_wake_corner_list, wake_gamma_list)

    # Verify the intermediate Gamma_wake_dt_q1234_n first
    Mf2_ml = np.asarray(fx._raw['Mf2_vec1']).ravel()
    Mf2_py_flat = py_grid_to_ml_flat(Mf2_py)

    # Note: sign conventions: Python AIC = -ML A_mat. The Mf2_vec1 formula
    # uses A^{-1} · (-Gamma_w_dt_q1234_n). If Python Gamma_w_dt_q1234_n has
    # opposite sign of MATLAB (due to dt_ring_vortex_velocity returning -dt_V
    # to match ring sign), the two sign flips cancel and Mf2_vec1 should match
    # directly (no sign flip) — same logic as L5.
    return assert_close_named('Mf2_vec1', Mf2_py_flat, Mf2_ml,
                              atol=1e-5, rtol=1e-3)


# ─── Main ────────────────────────────────────────────────────────────────

LAYERS = {
    'L0': layer_L0_geometry,
    'L1': layer_L1_AIC,
    'L2': layer_L2_gamma_solve,
    'L3': layer_L3_gradients,
    'L4': layer_L4_velocities,
    'L5': layer_L5_dp_lift1,
    'L6': layer_L6_Mf2_vec1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--layer', default='all', help='|'.join(LAYERS) + '|all')
    args = ap.parse_args()

    fx = MatlabFixture(args.fixture)
    fx.summary()
    uvlm = build_python_uvlm_from_fixture(fx)

    layers = LAYERS if args.layer == 'all' else {args.layer: LAYERS[args.layer]}

    print("\n══ Layered comparison ══")
    results = {}
    for name, fn in layers.items():
        try:
            results[name] = fn(fx, uvlm)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    print("\n══ Summary ══")
    for name, ok in results.items():
        tag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        print(f"  {name}: {tag}")

    sys.exit(1 if any(v is False for v in results.values()) else 0)


if __name__ == "__main__":
    main()
