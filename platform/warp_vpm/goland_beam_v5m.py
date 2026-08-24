"""Goland flutter: Beam FE + FLUX-V5M pressure via AeroelasticSolver override.

Builds on the verified AeroelasticSolver (beam+Ptera lagged, 140.2 m/s) but
overrides the force extraction to use the FLUX-V5M pressure decomposition:
  lift_station = ρ · V · Σ_Γ  (Kutta-Joukowski from bound circulation)
instead of Ptera's panel-force summation. This tests whether the FLUX-V5M
circulation-based pressure gives the correct aerodynamic damping.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))

from fluxvortex.aeroelastic_solver import AeroelasticSolver

# build_goland_wing is in tests/benchmark_goland.py
sys.path.insert(0, str(REPO / "tests"))
from benchmark_goland import build_goland_wing

CHORD = 1.8288      # m
SEMI_SPAN = 6.096   # m


class FluxV5MPressureSolver(AeroelasticSolver):
    """Pterra verified forces + FLUX-V5M added-mass via mass-matrix modification.

    Instead of applying the added-mass force as a lagged external load
    (which introduces artificial damping), the FLUX-V5M Mf1 physics is
    implemented as an effective mass increase: m_eff = m_struct + m_added
    where m_added = ρ·π·(c/2)² is the classical flat-plate added mass per
    unit span for heave acceleration.

    This is the correct implicit formulation — no artificial damping,
    the natural frequency shifts down as the effective inertia increases.
    """

    def __init__(self, unsteady_problem, beam_params, relaxation=1.0,
                 x_ea_chord=0.33, added_mass_factor=1.0):
        # Modify the mass per unit length to include the added mass
        rho = unsteady_problem.movement.operating_point_movement.\
            base_operating_point.rho
        c = beam_params.get("length", 6.096)  # use span for now; chord below
        # Get chord from the wing geometry
        try:
            wing = unsteady_problem.movement.airplane_movements[0].\
                base_airplane.wings[0]
            chord = wing.wing_cross_sections[0].chord
        except (AttributeError, IndexError):
            chord = 1.8288

        m_add = rho * np.pi * (chord / 2)**2 * added_mass_factor
        beam_params_v5m = dict(beam_params)
        beam_params_v5m["m_per_length"] = (
            beam_params["m_per_length"] + m_add)
        # Also increase torsional inertia proportionally (flat plate:
        # I_add ≈ m_add·c²/24 matching the structural Ip convention)
        beam_params_v5m["Ip"] = (
            beam_params["Ip"] + m_add * chord**2 / 24)

        super().__init__(unsteady_problem, beam_params_v5m,
                         relaxation, x_ea_chord)


SEMI_SPAN_REF = SEMI_SPAN


def compute_envelope_growth(signal, dt):
    if len(signal) < 10:
        return 0.0
    abs_s = np.abs(signal)
    peaks = [(i * dt, abs_s[i]) for i in range(1, len(abs_s) - 1)
             if abs_s[i] > abs_s[i-1] and abs_s[i] > abs_s[i+1]]
    if len(peaks) < 3:
        return 0.0
    t_p = np.array([p[0] for p in peaks])
    a_p = np.maximum(np.array([p[1] for p in peaks]), 1e-15)
    if len(t_p) > 4:
        log_a, t_fit = np.log(a_p[1:]), t_p[1:]
    else:
        log_a, t_fit = np.log(a_p), t_p
    if len(t_fit) >= 2:
        return np.polyfit(t_fit, log_a, 1)[0]
    return 0.0


def run_at_velocity(V, dt=0.003, num_chords=100):
    mv, op = build_goland_wing(V, dt=dt, num_chords=num_chords)
    import pterasoftware as ps
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

    solver = FluxV5MPressureSolver(prob, beam_params={
        "length": SEMI_SPAN, "n_elements": 8,
        "EI": 9.773e6, "GJ": 0.988e6,
        "m_per_length": 35.72,
        "Ip": 35.72 * CHORD**2 / 24,
        "x_ea_cg": 0.10 * CHORD,
        "structural_damping": 0.005,
    }, relaxation=1.0)

    tip_node = solver.beam.nnodes - 1
    solver.beam.d[3 * tip_node] = 0.05
    solver.beam.d[3 * tip_node + 2] = np.radians(2.0)
    K_r, M_r, _, free = solver.beam.apply_bc(solver.beam.K, solver.beam.M)
    solver.beam.a[free] = np.linalg.solve(M_r, -K_r @ solver.beam.d[free])

    t0 = time.time()
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    elapsed = time.time() - t0

    tip_w = np.array(solver.tip_w_history)
    tip_theta = np.array(solver.tip_theta_history)
    sigma_w = compute_envelope_growth(tip_w, dt)
    sigma_theta = compute_envelope_growth(tip_theta, dt)
    return {
        "V": V, "sigma": max(sigma_w, sigma_theta),
        "sigma_w": sigma_w, "sigma_theta": sigma_theta,
        "max_w": float(np.max(np.abs(tip_w))) if len(tip_w) else 0,
        "elapsed": elapsed, "n_history": len(tip_w),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocities", type=float, nargs="+",
                        default=[130, 140, 145, 150, 160])
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/goland_beam_v5m.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Goland: Pterra forces + FLUX-V5M Mf1 added-mass")
    print("=" * 60)
    print("  Refs: 137 (Goland-Luke), 140.2 (lagged Pterra)")
    print()

    results = []
    for V in args.velocities:
        print(f"V = {V:.0f} m/s ...", flush=True)
        try:
            r = run_at_velocity(V)
            status = "FLUTTER" if r["sigma_w"] > 0 else "stable"
            print(f"  → {status} (σ_w={r['sigma_w']:+.3f}, "
                  f"σ_θ={r['sigma_theta']:+.3f}, {r['elapsed']:.1f}s)",
                  flush=True)
            results.append(r)
        except Exception as e:
            print(f"  → ERROR: {e}", flush=True)
            results.append({"V": V, "error": str(e)})

    valid = [(r["V"], r["sigma_w"]) for r in results if "sigma_w" in r]
    if len(valid) >= 2:
        for i in range(len(valid) - 1):
            v1, s1 = valid[i]
            v2, s2 = valid[i + 1]
            if s1 < 0 and s2 > 0:
                vf = v1 + (0 - s1) / (s2 - s1) * (v2 - v1)
                print(f"\nFlutter speed: V_f = {vf:.1f} m/s")
                break

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

