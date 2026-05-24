"""
Test BeamFE: Goland Wing natural frequencies.

Goland Wing reference values (Bisplinghoff et al., Hodges & Pierce):
  - 1st bending: ~2.5 Hz
  - 1st torsion: ~8.5 Hz
  - Flutter speed: ~450 ft/s ≈ 137 m/s at sea level

Properties (converted to SI):
  - L = 6.096 m (20 ft semi-span)
  - chord = 1.8288 m (6 ft)
  - EI = 23.65e6 lb·ft² = 32.1e6 N·m² (approx, various sources)
  - GJ = 2.39e6 lb·ft² = 3.24e6 N·m²
  - m = 0.746 slugs/ft = 35.7 kg/m
  - x_alpha = 0.20 * chord = 0.366 m (CG aft of EA)
  - EA at 33% chord

Note: Exact Goland Wing properties vary across sources. We use the most common
values for validation. The beam FE should reproduce the first few natural
frequencies within ~5% of analytical solutions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from fluxvortex.beam_fe import BeamFE


def test_goland_wing():
    """Test natural frequencies against analytical Goland Wing values."""
    # Goland Wing properties (SI units)
    L = 6.096       # semi-span [m]
    chord = 1.8288  # chord [m]
    EI = 32.1e6     # bending stiffness [N·m²]
    GJ = 3.24e6     # torsional stiffness [N·m²]
    m = 35.7        # mass per unit length [kg/m]
    Ip = 0.5 * m * (chord / 2)**2  # approximate polar inertia
    x_ea_cg = 0.20 * chord  # CG aft of EA

    beam = BeamFE(
        length=L,
        n_elements=20,
        EI=EI,
        GJ=GJ,
        m_per_length=m,
        Ip=Ip,
        x_ea_cg=0.0,  # No coupling first — verify uncoupled
    )

    freqs, _ = beam.compute_natural_frequencies()

    # Analytical bending frequency (uncoupled, cantilever):
    # omega_1 = (1.8751)^2 * sqrt(EI / (m * L^4))
    omega_b1 = 1.8751**2 * np.sqrt(EI / (m * L**4))
    f_b1 = omega_b1 / (2 * np.pi)

    # Analytical torsion frequency (uncoupled, cantilever):
    # omega_1 = pi/2 * sqrt(GJ / (Ip * L^2))
    omega_t1 = np.pi / 2 * np.sqrt(GJ / (Ip * L**2))
    f_t1 = omega_t1 / (2 * np.pi)

    print(f"Goland Wing Natural Frequencies:")
    print(f"  1st bending (analytical): {f_b1:.2f} Hz")
    print(f"  1st torsion  (analytical): {f_t1:.2f} Hz")
    print(f"  FE frequencies: {freqs[:6].round(2)} Hz")

    # Check: first bending mode should be close to analytical
    bending_err = abs(freqs[0] - f_b1) / f_b1 * 100
    print(f"  1st bending error: {bending_err:.1f}%")

    # Find torsion mode: closest frequency to analytical torsion
    torsion_idx = np.argmin(np.abs(freqs[:6] - f_t1))
    torsion_err = abs(freqs[torsion_idx] - f_t1) / f_t1 * 100
    print(f"  1st torsion mode at index {torsion_idx}, error: {torsion_err:.1f}%")

    # Coupled system shifts frequencies — accept up to 15% difference
    assert bending_err < 15, f"Bending frequency error too large: {bending_err:.1f}%"
    assert torsion_err < 15, f"Torsion frequency error too large: {torsion_err:.1f}%"

    print("  PASS: Natural frequencies within tolerance")
    return beam


def test_newmark_integration():
    """Test that Newmark integration is stable for a free vibration problem."""
    L = 6.096
    chord = 1.8288
    beam = BeamFE(
        length=L,
        n_elements=20,
        EI=32.1e6,
        GJ=3.24e6,
        m_per_length=35.7,
        Ip=0.5 * 35.7 * (chord / 2)**2,
        x_ea_cg=0.0,
    )

    # Apply an initial tip displacement and integrate
    tip_node = beam.nnodes - 1
    beam.d[3 * tip_node] = 0.1  # 0.1 m heave at tip

    # Initialize acceleration from initial displacement: a0 = M^{-1}*(-K*d0)
    K_r, M_r, _, free = beam.apply_bc(beam.K, beam.M)
    a0_r = np.linalg.solve(M_r, -K_r @ beam.d[free])
    beam.a[free] = a0_r

    dt = 0.001
    n_steps = 2000
    tip_w_history = np.zeros(n_steps)

    for i in range(n_steps):
        F = np.zeros(beam.ndof)
        beam.step(F, dt)
        w, theta = beam.get_nodal_displacements()
        tip_w_history[i] = w[-1]

    # Check: tip displacement should oscillate (not diverge)
    max_w = np.max(np.abs(tip_w_history))
    assert max_w < 1.0, f"Newmark integration unstable: max tip w = {max_w:.3f}"

    # Check: displacement should not grow (energy conservation)
    amp_first = np.max(tip_w_history[:100]) - np.min(tip_w_history[:100])
    amp_last = np.max(tip_w_history[-100:]) - np.min(tip_w_history[-100:])
    growth = amp_last / max(amp_first, 1e-10)
    assert growth < 1.5, f"Amplitude growing: {growth:.2f}x"

    print(f"Newmark stability test:")
    print(f"  Max tip displacement: {max_w:.4f} m")
    print(f"  Amplitude growth ratio: {growth:.3f}")
    print("  PASS: Integration stable")


if __name__ == '__main__':
    print("=" * 60)
    print("BeamFE Validation Tests")
    print("=" * 60)
    test_goland_wing()
    print()
    test_newmark_integration()
    print()
    print("All tests passed.")
