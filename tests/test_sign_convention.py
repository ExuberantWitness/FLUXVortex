"""Test: does adding MATLAB negation -(q1+q2+q3+q4) fix sign convention?

MATLAB's generate_panel.m: q1234_mat = -(q1 + q2 + q3 + q4)
Our code: V = q1 + q2 + q3 + q4 (no negation)

This test checks force sign and magnitude with/without negation.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.standalone_uvlm import StandaloneUVLM, ring_vortex_velocity, _vortex_segment_velocity


def main():
    np.set_printoptions(precision=6, linewidth=120)

    # Build a simple 5x3 UVLM mesh
    nx, ny = 5, 3
    L, W = 1.0, 1.0
    x_vec = np.linspace(0, L, nx + 1)
    y_vec = np.linspace(0, W, ny + 1)
    vertices = np.zeros((nx + 1, ny + 1, 3))
    for i in range(nx + 1):
        for j in range(ny + 1):
            vertices[i, j] = [x_vec[i], y_vec[j], 0.0]

    alpha_deg = 1.0
    alpha = np.radians(alpha_deg)
    V_inf = 10.0
    V_inf_vec = np.array([V_inf * np.cos(alpha), 0.0, -V_inf * np.sin(alpha)])
    rho = 1.225

    # Standard UVLM (no negation)
    uvlm_std = StandaloneUVLM(vertices, V_inf_vec, rho=rho, core_radius=1e-6)
    uvlm_std.build_aic()
    print(f"Standard AIC diag range: [{np.diag(uvlm_std._AIC).min():.4f}, {np.diag(uvlm_std._AIC).max():.4f}]")

    V_struct = np.zeros((nx, ny, 3))
    uvlm_std.solve(V_ext_colloc=None, V_struct_colloc=V_struct)
    gamma_std = uvlm_std.gamma.copy()
    print(f"Standard gamma sum: {gamma_std.sum():.6f}")
    print(f"Standard gamma range: [{gamma_std.min():.6f}, {gamma_std.max():.6f}]")

    V_bound = uvlm_std.compute_bound_induction_at_colloc()
    uvlm_std.compute_forces(0.02, V_ext_colloc=V_bound, V_struct_colloc=V_struct)
    Fz_std = uvlm_std.forces[:,:,2].sum()
    print(f"Standard forces Fz total: {Fz_std:.4f} N (should be >0 for +AoA)")

    # Now build a VERSION 2 with negation matching MATLAB
    # Patch: negate the AIC and ring velocity
    uvlm_neg = StandaloneUVLM(vertices, V_inf_vec, rho=rho, core_radius=1e-6)
    uvlm_neg.build_aic()

    # Negate AIC to match MATLAB: q1234_mat = -(q1+q2+q3+q4)
    uvlm_neg._AIC = -uvlm_neg._AIC
    print(f"\nNegated AIC diag range: [{np.diag(uvlm_neg._AIC).min():.4f}, {np.diag(uvlm_neg._AIC).max():.4f}]")

    uvlm_neg.solve(V_ext_colloc=None, V_struct_colloc=V_struct)
    gamma_neg = uvlm_neg.gamma.copy()
    print(f"Negated gamma sum: {gamma_neg.sum():.6f}")
    print(f"Negated gamma range: [{gamma_neg.min():.6f}, {gamma_neg.max():.6f}]")
    print(f"Ratio gamma_neg/gamma_std: {gamma_neg.sum()/gamma_std.sum():.4f}")

    # For the negated version, we also need to negate ring_vortex_velocity for
    # force computations. The AIC is negated, so the gamma solution is negated.
    # But for the force computation, we need the ring velocity to also be
    # negated (or the pressure force to use the correct sign convention).

    # The key question: does the force computation work correctly with
    # the negated AIC + gamma?

    # Recompute forces with negated EIC
    V_bound_neg = np.zeros_like(V_bound)
    nc, ns = nx, ny
    pts_flat = uvlm_neg._colloc.reshape(-1, 3)
    V_bound_flat = np.zeros((nc * ns, 3))
    for bi in range(nc):
        for bj in range(ns):
            g = gamma_neg[bi, bj]
            if abs(g) < 1e-15:
                continue
            # Use NEGATED ring velocity (MATLAB convention)
            from fluxvortex.standalone_uvlm import ring_vortex_velocity
            V_bound_flat -= ring_vortex_velocity(pts_flat, uvlm_neg._corners[bi, bj],
                                                g, uvlm_neg._core_radius)
    V_bound_neg = V_bound_flat.reshape(nc, ns, 3)

    # Now compute forces. The force formula should use:
    # dp_lift = ρ * V_surf·τ_x * (dΦ/dx or dγ/dx)
    # The dp_lift2 = ρ*(τ_x*dΦ/dx + τ_y*dΦ/dy) or ρ*(τ_x*dγ/dx + τ_y*dγ/dy)

    # For the force computation, we need to the correct gamma and induced velocity
    # Let's use the negated solver and manually force-compute with both gradients
    uvlm_neg.compute_forces(0.02, V_ext_colloc=V_bound_neg, V_struct_colloc=V_struct)
    Fz_neg = uvlm_neg.forces[:,:,2].sum()
    print(f"Negated forces Fz total: {Fz_neg:.4f} N (should be >0 for +AoA)")

    # ── Now check: What does dΦ/dx vs dγ/dx give with negated gamma? ──
    print("\n" + "=" * 70)
    print("GRADIENT COMPARISON WITH NEGATED GAMMA")
    print("=" * 70)

    gamma = gamma_neg  # Use negated gamma (should be positive for +AoA)
    dx = L / nx

    # dΦ/dx = γ/dx
    dPhi_dx = gamma / dx

    # dγ/dx = diff(gamma)/dx
    dgamma_dx = np.zeros_like(gamma)
    dgamma_dx[0, :] = gamma[0, :] / dx
    dgamma_dx[1:, :] = (gamma[1:, :] - gamma[:-1, :]) / dx

    # Bernoulli pressure
    V_dot_tau_x = V_inf * np.cos(alpha)
    dp_phi = rho * V_dot_tau_x * dPhi_dx
    dp_gamma = rho * V_dot_tau_x * dgamma_dx

    area = dx * (W / ny)
    Fz_phi = np.sum(dp_phi * area)
    Fz_gamma = np.sum(dp_gamma * area)

    print(f"dΦ/dx (γ/dx):  Fz = {Fz_phi:.4f} N")
    print(f"dγ/dx (diff γ): Fz = {Fz_gamma:.4f} N")
    print(f"Ratio dΦ/dγ: {Fz_phi/(Fz_gamma+1e-15):.2f}x")

    # KJ theorem
    Gamma_bound_TE = np.sum(np.cumsum(gamma, axis=0)[-1, :])
    dy = W / ny
    Fz_KJ = rho * V_inf * Gamma_bound_TE * dy
    print(f"KJ theorem:     Fz = {Fz_KJ:.4f} N")
    print(f"Match dΦ/KJ: {abs(Fz_phi/Fz_KJ - 1):.4f}")
    print(f"Match dγ/KJ: {abs(Fz_gamma/Fz_KJ - 1):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standard UVLM (no negation): gamma={gamma_std.sum():.4f}, Fz={Fz_std:.4f} N")
    print(f"Negated UVLM (MATLAB):       gamma={gamma_neg.sum():.4f}, Fz={Fz_neg:.4f} N")
    print(f"Expected (KJ):               Fz={Fz_KJ:.4f} N (upward for +AoA)")

    if Fz_std < 0:
        print("\n*** STANDARD UVLM HAS WRONG SIGN! Force is downward for +AoA ***")
    if Fz_neg > 0:
        print("*** NEGATED UVLM HAS CORRECT SIGN! Force is upward for +AoA ***")


if __name__ == '__main__':
    main()
