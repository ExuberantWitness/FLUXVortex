"""Goland Wing flutter with FLUX-V5M (Q16 structure + LEV/TEV + predictor/corrector).

Classic benchmark (Goland & Luke 1948): cantilevered rectangular wing,
bending-torsion coupled flutter. Reference values:
  - Goland-Luke strip-theory analytic: ~137 m/s
  - Our lagged beam FE + Ptera UVLM: 140.2 m/s (verified)
  - 3D UVLM codes (SHARPy etc.): 163-169 m/s

Method:
  1. Build Q16 MITC16/EAS shell model of the Goland wing (clamped root)
  2. Apply initial tip perturbation (heave + twist)
  3. Run strong predictor/corrector FSI with native FLUX-V5M aero
     (LEV + joint TEV + free wake, all in one transaction)
  4. Measure envelope growth rate σ from tip displacement
  5. Flutter = velocity where σ crosses zero

Goland wing physical parameters (from benchmark_goland.py / beam_fe.py):
  chord = 1.8288 m, semi_span = 6.096 m (AR = 6.67)
  EI = 9.773e6 N·m², GJ = 0.988e6 N·m²
  m = 35.72 kg/m, I_α = m·c²/24
  x_ea_cg = 0.10·c (CG ahead... actually positive = aft of EA)
  structural damping ζ = 0.005

For the Q16 shell, we convert beam properties to plate/shell equivalents:
  thickness h: from EI = E·h³/12 per unit width → E·I_total = E·c·h³/12
  → h = (12·EI/(E·c))^(1/3)
  E = aluminum ≈ chosen so that h is reasonable

Actually the Goland wing is typically modeled as a beam. The Q16 shell
needs: E, ν, ρ_s, thickness. We derive these from EI/GJ/m:
  h = (12·EI/(E·c))^(1/3)  [from bending stiffness match]
  ρ_s = m/(c·h)             [from mass match]
  G = GJ·c/(h³/3) → E = 2G(1+ν) [from torsion match, check consistency]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))
sys.path.insert(0, str(REPO / "platform/warp_vpm"))

from fluxvortex.q16_ancf_mesh import make_rectangular_q16_mesh
from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh
from fluxvortex.q16_boundary_constraints import make_clamped_q16_nodes
from fluxvortex.warp_fsi import config as wsi_config
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    Q16NativeV5MSurface, Q16NativeV5MSolver, NativeV5MConfig,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native_fsi import (
    Q16NativeV5MFSIStepper, Q16NativeV5MFSIOwner,
)

# Goland wing parameters (frozen, matching benchmark_goland.py)
CHORD = 1.8288          # m
SEMI_SPAN = 6.096       # m
EI = 9.773e6            # N·m² (bending)
GJ = 0.988e6            # N·m² (torsion)
MASS_PER_LENGTH = 35.72 # kg/m
X_EA_CG = 0.10 * CHORD  # m
STRUCT_DAMPING = 0.005  # ζ
RHO_AIR = 1.225         # kg/m³
NU_AIR = 15.06e-6       # m²/s

# Derived Q16 shell parameters
# EI = E·c·h³/12 → for a rectangular section of width c
# We pick E and solve for h. Use E such that h is a realistic value.
# Try E = 7.0e10 (aluminum):
E_SHELL = 7.0e10
H_SHELL = (12.0 * EI / (E_SHELL * CHORD)) ** (1.0 / 3.0)
RHO_SHELL = MASS_PER_LENGTH / (CHORD * H_SHELL)
POISSON = 0.3

# Check torsion consistency: GJ = G·c·h³/3 (open thin-walled, c>>h)
G_SHELL = E_SHELL / (2.0 * (1.0 + POISSON))
GJ_CHECK = G_SHELL * CHORD * H_SHELL**3 / 3.0

# Grid
Q16_CHORD_ELEMENTS = 4     # chordwise macro elements
Q16_SPAN_ELEMENTS = 8      # spanwise macro elements (finer in span for bending)
AERO_CHORD_PANELS = 8
AERO_SPAN_PANELS = 16

# Time
DT = 0.003                # structural dt [s]
AERO_STEPS_PER = 4        # structural steps per aero step → aero_dt = 4*0.003
N_OUTER_STEPS = 60        # ~60 aero steps × 4 struct × 0.003s = 0.72s sim
                                # at 10Hz oscillation → ~7 periods, enough for envelope


def build_goland_q16(V_inf):
    """Build Q16 Goland wing model clamped at root (y=0)."""
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=Q16_CHORD_ELEMENTS,
        spanwise_element_count=Q16_SPAN_ELEMENTS,
        chord=CHORD,
        span=SEMI_SPAN,
        thickness=H_SHELL,
    )
    model = Q16MITC16EASMesh(
        mesh,
        young_modulus=E_SHELL,
        poisson_ratio=POISSON,
        density=RHO_SHELL,
    )
    # Clamp root: all nodes at y=0 (spanwise coordinate = 0)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(CHORD, SEMI_SPAN)
    root_nodes = np.flatnonzero(
        np.abs(mesh.reference_rows[:, 1]) <= tolerance)
    root_nodes = np.ascontiguousarray(root_nodes, dtype=np.int64)
    constraints = make_clamped_q16_nodes(mesh, root_nodes)
    return mesh, model, constraints, root_nodes


def apply_tip_perturbation(mesh, model, state_flat):
    """Apply initial tip perturbation: heave + twist at tip nodes.

    state_flat is the 1-D reference_state (node_count * 6).
    """
    tolerance = 1e-8
    tip_mask = np.abs(mesh.reference_rows[:, 1] - SEMI_SPAN) <= tolerance

    tip_w = 0.005  # m (10× smaller: Goland is a stiff wing, not a sheet)
    tip_twist = np.radians(0.2)  # rad

    for node in np.flatnonzero(tip_mask):
        x = mesh.reference_rows[node, 0]
        state_flat[node * 6 + 2] = tip_w + tip_twist * (x - CHORD / 2.0)
    return state_flat


def run_flutter_point(V_inf, quiet=False):
    """Run one velocity point, return envelope growth rate."""
    mesh, model, boundary, root_nodes = build_goland_q16(V_inf)

    structural = Q16CudaNewmarkStepper(
        model, boundary,
        device=wsi_config.config.DEVICE
        if hasattr(wsi_config, 'config') else 'cuda:0',
        newton_tolerance=2.0e-4,
        max_newton_iterations=256,
        # Refresh tangent every 8 iterations (Goland stiffness needs fresher
        # tangent than Yamano's default 48 — stalls at 1e-4 with 48)
        reference_dense_refresh_after=8,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        nonsymmetric_solver="reference_dense",
    )

    aero_dt = AERO_STEPS_PER * DT
    settings = NativeV5MConfig(
        chordwise_panels=AERO_CHORD_PANELS,
        spanwise_panels=AERO_SPAN_PANELS,
        density=RHO_AIR,
        freestream=V_inf,
        aerodynamic_dt=aero_dt,
        lesp_crit=0.11,
        wake_max_rows=96,
        # DVM settings for the Goland chord (avoid frozen-overlap violation
        # from the small default target spacing scaled to a much larger chord)
        dvm_smoothing_radius_chord=0.04,
        dvm_target_spacing_chord=0.04 / 2.5,
    )
    surface = Q16NativeV5MSurface(
        mesh,
        q16_chordwise_elements=Q16_CHORD_ELEMENTS,
        q16_spanwise_elements=Q16_SPAN_ELEMENTS,
        aerodynamic_chordwise_panels=AERO_CHORD_PANELS,
        aerodynamic_spanwise_panels=AERO_SPAN_PANELS,
        device="cuda:0",
    )
    aero = Q16NativeV5MSolver(surface, settings)

    # Initial state: equilibrium geometry + velocity impulse at tip
    # (velocity perturbation avoids the huge internal force from a
    #  displacement perturbation on a stiff wing)
    state_np = mesh.reference_state.copy()
    vel_np = np.zeros_like(state_np)
    tolerance_v = 1e-8
    tip_mask = np.abs(mesh.reference_rows[:, 1] - SEMI_SPAN) <= tolerance_v
    tip_velocity = 0.1  # m/s downward at tip
    for node in np.flatnonzero(tip_mask):
        vel_np[node * 6 + 2] = tip_velocity

    device = "cuda:0"
    state = wp.from_torch(
        torch.as_tensor(np.ascontiguousarray(state_np[None, :]),
                        dtype=torch.float64, device=device),
        dtype=wp.float64)
    vel = wp.from_torch(
        torch.as_tensor(np.ascontiguousarray(vel_np[None, :]),
                        dtype=torch.float64, device=device),
        dtype=wp.float64)

    owner = Q16NativeV5MFSIOwner.initialize(aero, state, vel, wp.zeros_like(state))
    coupling = Q16NativeV5MFSIStepper(
        structural, aero,
        coupling_tolerance=5.0e-7,
        max_coupling_iterations=10,
        relaxation=0.7,
    )

    # Run N_OUTER_STEPS aero steps
    tip_hist = []
    tip_idx = None
    tolerance = 256.0 * np.finfo(np.float64).eps * max(CHORD, SEMI_SPAN)
    tip_nodes = np.flatnonzero(
        np.abs(mesh.reference_rows[:, 1] - SEMI_SPAN) <= tolerance)
    mid_tip = tip_nodes[len(tip_nodes) // 2]

    t0 = time.perf_counter()
    for step in range(N_OUTER_STEPS):
        # No prescribed force (free vibration after perturbation)
        zero_force = wp.zeros((1, mesh.reference_state.size),
                              dtype=wp.float64, device=device)
        result = coupling.advance(
            owner,
            delta_time=aero_dt,
            prescribed_forces=(zero_force,) * AERO_STEPS_PER,
            load_betas=tuple((i + 1) / AERO_STEPS_PER
                             for i in range(AERO_STEPS_PER)),
        )
        # Record tip z-displacement
        q_t = wp.to_torch(owner.state).cpu().numpy().flatten()
        tip_hist.append(q_t[mid_tip * 6 + 2])  # z-dof of mid-tip node

        if not quiet and step % 10 == 0:
            print(f"  step {step}/{N_OUTER_STEPS}: tip_z={tip_hist[-1]:+.5f}",
                  flush=True)

    elapsed = time.perf_counter() - t0

    # Envelope growth rate
    tip = np.array(tip_hist)
    dt_total = aero_dt
    sigma = compute_envelope_growth(tip, dt_total)

    return {
        "V": V_inf,
        "sigma": sigma,
        "max_w": float(np.max(np.abs(tip))),
        "n_steps": N_OUTER_STEPS,
        "elapsed": elapsed,
        "tip_history": tip.tolist(),
    }


def compute_envelope_growth(signal, dt):
    """Exponential growth rate from peak envelope (same as benchmark_goland)."""
    if len(signal) < 10:
        return 0.0
    abs_signal = np.abs(signal)
    peaks = []
    for i in range(1, len(abs_signal) - 1):
        if abs_signal[i] > abs_signal[i - 1] and abs_signal[i] > abs_signal[i + 1]:
            peaks.append((i * dt, abs_signal[i]))
    if len(peaks) < 3:
        return 0.0
    t_peaks = np.array([p[0] for p in peaks])
    a_peaks = np.maximum(np.array([p[1] for p in peaks]), 1e-15)
    if len(t_peaks) > 4:
        log_a = np.log(a_peaks[1:])
        t_fit = t_peaks[1:]
    else:
        log_a = np.log(a_peaks)
        t_fit = t_peaks
    if len(t_fit) >= 2:
        coeffs = np.polyfit(t_fit, log_a, 1)
        return coeffs[0]
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocities", type=float, nargs="+",
                        default=[100, 200, 300, 400, 500, 700])
    parser.add_argument("--output", type=Path, default=Path(
                        "/tmp/goland_q16_fluxv5m_sweep.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Goland Wing Flutter — Q16 + FLUX-V5M (LEV/TEV + PC)")
    print("=" * 60)
    print(f"  Q16: {Q16_CHORD_ELEMENTS}×{Q16_SPAN_ELEMENTS} macro elements")
    print(f"  Aero: {AERO_CHORD_PANELS}×{AERO_SPAN_PANELS} panels")
    print(f"  h={H_SHELL:.4f}m, ρ_s={RHO_SHELL:.1f}kg/m³, E={E_SHELL:.2e}")
    print(f"  GJ check: target={GJ:.3e}, shell={GJ_CHECK:.3e}, "
          f"ratio={GJ_CHECK/GJ:.3f}")
    print(f"  dt={DT}, aero_dt={AERO_STEPS_PER*DT}, "
          f"steps={N_OUTER_STEPS} ({N_OUTER_STEPS*AERO_STEPS_PER*DT:.3f}s)")
    print(f"  References: 137 (Goland-Luke), 140.2 (our lagged), "
          f"163-169 (SHARPy 3D)")
    print()

    results = []
    for V in args.velocities:
        print(f"V = {V:.0f} m/s ...", flush=True)
        try:
            r = run_flutter_point(V)
            status = "FLUTTER" if r["sigma"] > 0 else "stable"
            print(f"  → {status} (σ={r['sigma']:+.4f}/s, "
                  f"max_w={r['max_w']:.4f}m, {r['elapsed']:.1f}s)", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  → ERROR: {e}", flush=True)
            results.append({"V": V, "error": str(e)})

    # Find flutter speed (zero crossing of sigma)
    valid = [(r["V"], r["sigma"]) for r in results if "sigma" in r]
    if len(valid) >= 2:
        vf = None
        for i in range(len(valid) - 1):
            v1, s1 = valid[i]
            v2, s2 = valid[i + 1]
            if s1 < 0 and s2 > 0:
                vf = v1 + (0 - s1) / (s2 - s1) * (v2 - v1)
                break
        if vf:
            print(f"\nFlutter speed: V_f = {vf:.1f} m/s")
        else:
            print("\nFlutter speed: NOT FOUND in swept range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
