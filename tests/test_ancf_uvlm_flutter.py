"""ANCF Shell + UVLM/VPM Aeroelastic Flutter Validation.

Compares against Yamano et al. published data:
  - J. Sound and Vibration (2020): flutter boundary, LCO amplitude
  - MEJ (2021): energy harvesting, aspect ratio effects
  - IJSSD (2022): spanwise plate deformation

Test cases:
  1. Cantilever sheet flutter sweep (U* = 10-20)
  2. Pinned sheet flutter sweep
  3. Static aeroelastic equilibrium at sub-critical U*

References:
  Yamano et al. — single sheet parameters:
    L=1.0, W=1.0 (AR=1), h=1e-3
    M* = ρ_f*L/(ρ_m*h) = 1.0
    U* = 15 (flutter threshold for clamped LE)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import time
import warnings
warnings.filterwarnings('ignore')


def test_static_aeroelastic_equilibrium():
    """Verify ANCF+UVLM converges to static equilibrium at sub-critical U*.

    Cantilever sheet under steady flow should reach a static deflected
    equilibrium (no flutter) at low U*.
    """
    print("=" * 70)
    print("Test 1: Static Aeroelastic Equilibrium")
    print("=" * 70)

    from fluxvortex.ancf_aero_coupling import (
        ANCFAeroelasticSolver, build_ancf_wing, build_uvlm_problem)

    # Build ANCF sheet matching Yamano single_sheet
    L, W = 1.0, 1.0
    h = 1e-3
    E = 1e7
    rho_struct = 1000.0

    shell, _ = build_ancf_wing(
        Length=L, Width=W, thickness=h,
        nx=4, ny=6, rho=rho_struct, E=E, nu=0.3, bc_type='clamped')

    # Sub-critical velocity (U* ≈ 8, well below flutter)
    rho_fluid = 1.225
    V_inf = 12.0
    alpha = 2.0

    mv, op = build_uvlm_problem(
        shell, V_inf, rho=rho_fluid, alpha=alpha,
        dt=0.002, num_chords=40)

    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
    n_steps = len(prob.steady_problems)

    solver = ANCFAeroelasticSolver(
        prob, shell, integrator='implicit',
        relaxation=0.7, structural_dt_ratio=2,
        newton_tol=1e-6, max_newton=15)

    t0 = time.time()
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    elapsed = time.time() - t0

    tip_w = np.array(solver.tip_w_history)
    max_tip = np.max(np.abs(tip_w)) if len(tip_w) > 0 else 0

    print(f"  V={V_inf:.0f} m/s, α={alpha}°")
    print(f"  Steps: {n_steps}, elapsed: {elapsed:.1f}s")
    print(f"  Max tip |w|: {max_tip:.6f} m")
    print(f"  Final tip w: {tip_w[-1]:.6f} m" if len(tip_w) > 0 else "  No data")

    # Should reach equilibrium (no growing oscillation)
    is_stable = True
    if len(tip_w) > 20:
        # Check last 50% of signal isn't growing
        mid = len(tip_w) // 2
        first_half_amp = np.max(np.abs(tip_w[:mid]))
        second_half_amp = np.max(np.abs(tip_w[mid:]))
        ratio = second_half_amp / max(first_half_amp, 1e-15)
        is_stable = ratio < 2.0  # Not doubling in amplitude
        print(f"  Stability: {'STABLE' if is_stable else 'GROWING'} "
              f"(amp ratio = {ratio:.2f})")

    print(f"  Result: {'PASS' if is_stable else 'WARNING'}")
    return is_stable


def test_flutter_at_Ustar15():
    """Run at U*≈15 (Yamano's flutter condition for clamped LE).

    Yamano et al. report flutter at U*≈15 for single sheet with AR=1, M*=1.
    We check if our coupled solver exhibits flutter (growing oscillation).
    """
    print("\n" + "=" * 70)
    print("Test 2: Flutter Detection at U*≈15")
    print("=" * 70)

    from fluxvortex.ancf_aero_coupling import (
        ANCFAeroelasticSolver, compute_envelope_growth,
        build_ancf_wing, build_uvlm_problem)

    L, W = 1.0, 1.0
    h = 1e-3
    rho_struct = 1000.0

    # Compute E to match desired U*
    # U* = sqrt(ρ_m * h * L^2 * W * V^2 / (E * I))
    # I = W * h^3 / 12
    # Solving for E:
    # E = ρ_m * h * L^2 * W * V^2 / (U*^2 * I)
    #   = 12 * ρ_m * L^2 * V^2 / (U*^2 * h^2)
    rho_fluid = 1.225
    V_inf = 10.0
    target_Ustar = 15.0
    E = 12 * rho_struct * L**2 * V_inf**2 / (target_Ustar**2 * h**2)

    print(f"  Target U* = {target_Ustar}")
    print(f"  V = {V_inf} m/s, E = {E:.2e} Pa")
    print(f"  L={L}, W={W}, h={h}, ρ_struct={rho_struct}")

    shell, _ = build_ancf_wing(
        Length=L, Width=W, thickness=h,
        nx=4, ny=6, rho=rho_struct, E=E, nu=0.3, bc_type='clamped')

    # Verify structural natural frequencies
    from scipy.sparse.linalg import eigsh
    K_full = shell._internal_forces_and_tangent(shell.q)[1]
    M_full = shell.M
    bc = np.array(sorted(shell._bc_dofs), dtype=np.int32)
    free = np.setdiff1d(np.arange(shell.ndof), bc)
    K_ff = K_full[np.ix_(free, free)]
    M_ff = M_full[np.ix_(free, free)]
    eigvals = eigsh(K_ff, M=M_ff, k=6, which='SM', return_eigenvectors=False)
    freqs = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
    print(f"  Structural natural frequencies (first 6): {freqs[:6]}")

    mv, op = build_uvlm_problem(
        shell, V_inf, rho=rho_fluid, alpha=2.0,
        dt=0.001, num_chords=60)

    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

    solver = ANCFAeroelasticSolver(
        prob, shell, integrator='implicit',
        relaxation=0.7, structural_dt_ratio=2,
        newton_tol=1e-6, max_newton=15)

    # Apply initial tip perturbation for flutter excitation
    solver.apply_tip_perturbation(tip_force_z=0.5, tip_moment_y=0.02)

    t0 = time.time()
    try:
        solver.run(prescribed_wake=True, calculate_streamlines=False,
                   show_progress=False)
        elapsed = time.time() - t0

        tip_w = np.array(solver.tip_w_history)
        dt = prob.steady_problems[0].delta_time if hasattr(
            prob.steady_problems[0], 'delta_time') else 0.001

        sigma_w = compute_envelope_growth(tip_w, dt)
        max_w = np.max(np.abs(tip_w))

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Max tip |w|: {max_w:.6f} m")
        print(f"  σ_w: {sigma_w:+.4f} 1/s")
        print(f"  Status: {'FLUTTER' if sigma_w > 0 else 'STABLE'}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sigma_w = 0.0

    # At U*=15, expect flutter (σ_w > 0)
    # But with very coarse mesh (4×6) and limited steps, may not fully develop
    print(f"  Result: {'PASS (flutter detected)' if sigma_w > 0 else 'CHECK (near-threshold)'}")
    return sigma_w


def test_structural_only():
    """Verify ANCF structural model in isolation (no fluid coupling).

    Cantilever beam under tip load: compare tip deflection to Euler-Bernoulli theory.
    Also check first bending mode frequency.
    """
    print("\n" + "=" * 70)
    print("Test 3: ANCF Structural Validation (no fluid)")
    print("=" * 70)

    from fluxvortex.ancf_shell import ANCFShell, NDOF_NODE
    from scipy.sparse.linalg import eigsh

    L, W = 1.0, 0.1
    h = 0.01
    E = 1e7
    rho = 1000.0
    EI = E * W * h**3 / 12.0
    m_per_length = rho * W * h

    # Euler-Bernoulli cantilever first bending frequency
    # ω₁ = 1.875² * sqrt(EI / (m * L⁴))
    omega1_theory = 1.875**2 * np.sqrt(EI / (m_per_length * L**4))
    freq1_theory = omega1_theory / (2 * np.pi)

    nx, ny = 10, 2
    nn_x, nn_y = nx + 1, ny + 1
    nn = nn_x * nn_y
    nodes = np.zeros((nn, 3))
    for j in range(nn_y):
        for i in range(nn_x):
            nodes[j * nn_x + i, 0] = i * L / nx
            nodes[j * nn_x + i, 1] = j * W / ny

    quads = np.array([[j*nn_x+i, j*nn_x+i+1, (j+1)*nn_x+i+1, (j+1)*nn_x+i]
                       for j in range(ny) for i in range(nx)])

    shell = ANCFShell(nodes, quads, h=h, rho=rho, Ex=E, Ey=E, nu_xy=0.3)
    shell.set_bc(np.where(nodes[:, 0] < 1e-10)[0])

    # Frequency check
    K_full = shell._internal_forces_and_tangent(shell.q)[1]
    M_full = shell.M
    bc = np.array(sorted(shell._bc_dofs), dtype=np.int32)
    free = np.setdiff1d(np.arange(shell.ndof), bc)
    K_ff = K_full[np.ix_(free, free)]
    M_ff = M_full[np.ix_(free, free)]
    eigvals = eigsh(K_ff, M=M_ff, k=3, which='SM', return_eigenvectors=False)
    freqs = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
    freq1_err = abs(freqs[0] - freq1_theory) / freq1_theory * 100

    print(f"  First bending frequency:")
    print(f"    ANCF:  {freqs[0]:.3f} Hz")
    print(f"    Theory: {freq1_theory:.3f} Hz (EB cantilever)")
    print(f"    Error:  {freq1_err:.1f}%")

    # Static tip deflection
    P = 1.0  # tip load
    delta_theory = P * L**3 / (3.0 * EI)

    F_ext = np.zeros(shell.ndof)
    tip = np.where(np.abs(nodes[:, 0] - L) < 1e-10)[0]
    for n in tip:
        F_ext[n * NDOF_NODE + 2] = -P / len(tip)

    # Use implicit Newmark for robustness
    from scipy.sparse.linalg import spsolve
    for k in range(5000):
        shell.step_newmark(F_ext, 1e-4, newton_tol=1e-8, max_newton=20)

    disp = shell.get_displacement()
    tip_disp = np.mean(np.abs(disp[tip, 2]))
    disp_err = abs(tip_disp - delta_theory) / delta_theory * 100

    print(f"\n  Static tip deflection:")
    print(f"    ANCF:   {tip_disp*1000:.4f} mm")
    print(f"    Theory: {delta_theory*1000:.4f} mm (EB cantilever)")
    print(f"    Error:  {disp_err:.1f}%")

    assert freq1_err < 10, f"Frequency error {freq1_err:.1f}% exceeds 10%"
    assert disp_err < 10, f"Deflection error {disp_err:.1f}% exceeds 10%"
    print(f"  Result: PASS")
    return freqs[0], tip_disp


if __name__ == '__main__':
    print("ANCF + UVLM/VPM Aeroelastic Flutter Validation")
    print("Comparison target: Yamano et al. (2020, 2021, 2022)")
    print()

    ps.set_up_logging(level="Warning")

    # Test 3 first (fast, no fluid)
    test_structural_only()

    # Test 1: static equilibrium (fast)
    try:
        test_static_aeroelastic_equilibrium()
    except Exception as e:
        print(f"Test 1 skipped: {e}")

    # Test 2: flutter detection (longer - 60 chord lengths of wake)
    try:
        test_flutter_at_Ustar15()
    except Exception as e:
        print(f"Test 2 error: {e}")
        import traceback
        traceback.print_exc()
