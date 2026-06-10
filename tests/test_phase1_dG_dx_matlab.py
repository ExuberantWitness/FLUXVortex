"""Phase 1 verification: chordwise/spanwise gradient now matches MATLAB.

MATLAB convention (calc_fluid_force.m:36-43):
  dx_Γ[0,j] = Γ[0,j]/Δx                       # Ghommem 2011 p.138
  dx_Γ[i,j] = (Γ[i,j] - Γ[i-1,j])/Δx          # backward difference, i > 0

  dy_Γ on per-panel Γ with zero padding:
    dy_Γ[i,0]    =  Γ[i,0]/Δy
    dy_Γ[i,-1]   = -Γ[i,-1]/Δy
    dy_Γ[i,mid]  = (Γ[i,j+1] - Γ[i,j-1])/(2Δy)

We construct a flat plate, prescribe Γ, and recover dG_dx from the
chord-component of the per-panel force F_z / (ρ * U * A).
"""
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fluxvortex.standalone_uvlm import StandaloneUVLM


def _flat_plate(nc, ns, chord=1.0, span=1.0):
    verts = np.zeros((nc + 1, ns + 1, 3))
    for i in range(nc + 1):
        for j in range(ns + 1):
            verts[i, j] = [i * chord / nc, j * span / ns, 0.0]
    return verts


def _panel_tau_x_norm(uvlm, i, j):
    """Extract τ_x norm exactly as compute_forces does (handles TE shortened ring)."""
    c = uvlm._corners[i, j]
    r21 = c[1] - c[0]
    r34 = c[2] - c[3]
    return float(np.linalg.norm((r21 + r34) / 2))


def test_chordwise_gradient_matches_matlab_difference():
    nc, ns = 4, 1
    chord, span = 1.0, 1.0

    uvlm = StandaloneUVLM(_flat_plate(nc, ns, chord, span),
                          V_inf=np.array([1.0, 0.0, 0.0]), rho=1.0)
    uvlm.build_aic()

    gamma = np.array([[0.3], [0.7], [1.2], [1.5]])
    uvlm.gamma = gamma.copy()
    uvlm.gamma_prev = gamma.copy()  # dG_dt = 0
    uvlm.compute_forces(dt=1e6)

    # F_z = ρ * (V_inf · τ_x) * dG_dx * area * n_z
    F_z_per_area = uvlm.forces[:, 0, 2] / uvlm._areas[:, 0]

    dx_per_panel = np.array([_panel_tau_x_norm(uvlm, i, 0) for i in range(nc)])
    expected = np.array([
        gamma[0, 0] / dx_per_panel[0],                       # first row: Γ_0/Δx
        (gamma[1, 0] - gamma[0, 0]) / dx_per_panel[1],
        (gamma[2, 0] - gamma[1, 0]) / dx_per_panel[2],
        (gamma[3, 0] - gamma[2, 0]) / dx_per_panel[3],
    ])

    np.testing.assert_allclose(F_z_per_area, expected, atol=1e-12,
                               err_msg="dG_dx does not match MATLAB backward diff")


def test_spanwise_gradient_uses_per_panel_not_cumulative():
    """Set a chord-uniform Γ field so dG_dx=0 (except first row).
    Force chord component τ_x · V_inf = V_inf. Force span component vanishes
    since τ_y · V_inf = 0. We probe dG_dy through dp_lift2 instead."""
    nc, ns = 1, 4
    chord, span = 1.0, 1.0

    uvlm = StandaloneUVLM(_flat_plate(nc, ns, chord, span),
                          V_inf=np.array([1.0, 0.0, 0.0]), rho=1.0)
    uvlm.build_aic()

    gamma = np.array([[0.5, 1.0, 1.5, 2.0]])
    uvlm.gamma = gamma.copy()
    uvlm.gamma_prev = gamma.copy()
    uvlm.compute_forces(dt=1e6)

    # dp_lift2 = ρ * (τ_x * dG_dx + τ_y * dG_dy); τ_y = [0,1,0], so y-component = dG_dy
    dG_dy = uvlm.dp_lift2[0, :, 1]

    def _tau_y_norm(j):
        c = uvlm._corners[0, j]
        return float(np.linalg.norm(((c[0] - c[3]) + (c[1] - c[2])) / 2))

    dy_per_panel = np.array([_tau_y_norm(j) for j in range(ns)])
    expected = np.array([
        gamma[0, 0] / dy_per_panel[0],
        (gamma[0, 2] - gamma[0, 0]) / (2 * dy_per_panel[1]),
        (gamma[0, 3] - gamma[0, 1]) / (2 * dy_per_panel[2]),
        -gamma[0, 3] / dy_per_panel[3],
    ])

    np.testing.assert_allclose(dG_dy, expected, atol=1e-12,
                               err_msg="dG_dy does not match MATLAB per-panel central diff")


def test_uniform_chord_force_concentrated_at_LE():
    """Critical regression check: with uniform Γ over chord, MATLAB Bernoulli
    gives nonzero force ONLY at the leading-edge panel (since dx_Γ[i>0] = 0).
    The OLD γ[i]/Δx formula would have made every panel nonzero."""
    nc, ns = 5, 1
    chord, span = 1.0, 1.0

    uvlm = StandaloneUVLM(_flat_plate(nc, ns, chord, span),
                          V_inf=np.array([1.0, 0.0, 0.0]), rho=1.0)
    uvlm.build_aic()

    Gamma_uniform = 1.0
    uvlm.gamma = np.full((nc, ns), Gamma_uniform)
    uvlm.gamma_prev = uvlm.gamma.copy()
    uvlm.compute_forces(dt=1e6)

    F_z = uvlm.forces[:, 0, 2]
    dx_LE = _panel_tau_x_norm(uvlm, 0, 0)
    F_LE_expected = 1.0 * 1.0 * (Gamma_uniform / dx_LE) * uvlm._areas[0, 0] * 1.0

    assert abs(F_z[0] - F_LE_expected) < 1e-12, \
        f"LE force {F_z[0]} should equal ρ*U*(Γ/Δx)*A = {F_LE_expected}"
    for i in range(1, nc):
        assert abs(F_z[i]) < 1e-12, \
            f"Panel {i} should have zero force with uniform Γ (got {F_z[i]})"


if __name__ == "__main__":
    test_chordwise_gradient_matches_matlab_difference()
    print("PASS: chordwise gradient matches MATLAB backward difference")
    test_spanwise_gradient_uses_per_panel_not_cumulative()
    print("PASS: spanwise gradient uses per-panel Γ (not cumulative)")
    test_uniform_chord_force_concentrated_at_LE()
    print("PASS: uniform-Γ force concentrated at leading edge (MATLAB convention)")
    print("\nPhase 1 verified: gradients now match MATLAB calc_fluid_force.m formulas.")
