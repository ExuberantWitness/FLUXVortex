"""W1 — PteraSoftware baseline for the simplified rectangular flapping case.

Case (deliberately congruent with our rectangular flat-plate UVLM):
  - single (non-symmetric) rectangular wing: chord 1.5 m, semi-span 6.0 m
  - naca0012 (symmetric -> flat camber line == flat plate for UVLM)
  - uniform chordwise & spanwise spacing, 6 x 8 panels
  - flapping: 15 deg sine rotation about the x-axis through the root LE, 1 Hz
  - freestream 10 m/s, alpha = 1 deg, rho = 1.225; 3 cycles
Runs prescribed-wake and free-wake variants; saves per-step force histories.

Output: flap_arena/out/ptera_baseline.npz
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

if not hasattr(np, "trapz"):          # numpy >= 2.0
    np.trapz = np.trapezoid

import pterasoftware as ps  # noqa: E402

ps.set_up_logging(level="Warning")

CHORD = 1.5
SPAN = 6.0
NC, NS = 6, 8
AMP_DEG = 15.0
PERIOD = 1.0
V_INF = 10.0
ALPHA = 1.0
RHO = 1.225
CYCLES = 3


def build_movement():
    airplane = ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=[
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=NS,
                        chord=CHORD,
                        Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                        spanwise_spacing="uniform",
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca0012", n_points_per_side=400),
                    ),
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=None,
                        chord=CHORD,
                        Lp_Wcsp_Lpp=(0.0, SPAN, 0.0),
                        angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                        spanwise_spacing=None,
                        airfoil=ps.geometry.airfoil.Airfoil(
                            name="naca0012", n_points_per_side=400),
                    ),
                ],
                name="Rect Wing",
                Ler_Gs_Cgs=(0.0, 0.0, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
                symmetric=False,
                num_chordwise_panels=NC,
                chordwise_spacing="uniform",
            ),
        ],
    )
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[0],
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wcs)
            for wcs in airplane.wings[0].wing_cross_sections
        ],
        ampAngles_Gs_to_Wn_ixyz=(AMP_DEG, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(PERIOD, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
    )
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    op = ps.operating_point.OperatingPoint(
        rho=RHO, vCg__E=V_INF, alpha=ALPHA, beta=0.0, nu=15.06e-6)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    return ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=CYCLES)


def run(prescribed: bool):
    mv = build_movement()
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
    sol = ps.unsteady_ring_vortex_lattice_method.\
        UnsteadyRingVortexLatticeMethodSolver(prob)
    t0 = time.time()
    sol.run(prescribed_wake=prescribed, calculate_streamlines=False)
    wall = time.time() - t0
    dt = mv.delta_time
    n = sol.num_steps
    F = np.full((n, 3), np.nan)
    C = np.full((n, 3), np.nan)
    for k in range(n):
        ap = sol.steady_problems[k].airplanes[0]
        if ap.forces_W is not None:
            F[k] = np.asarray(ap.forces_W).ravel()
            C[k] = np.asarray(ap.forceCoefficients_W).ravel()
    print(f"[ptera {'presc' if prescribed else 'free'}] steps={n} dt={dt:.5f}"
          f" wall={wall:.1f}s  CL(last)={-C[-1, 2]:.4f}", flush=True)
    return dt, F, C, wall


def main():
    os.makedirs("flap_arena/out", exist_ok=True)
    dt_p, F_p, C_p, w_p = run(prescribed=True)
    dt_f, F_f, C_f, w_f = run(prescribed=False)
    np.savez("flap_arena/out/ptera_baseline.npz",
             chord=CHORD, span=SPAN, nc=NC, ns=NS, amp_deg=AMP_DEG,
             period=PERIOD, v_inf=V_INF, alpha=ALPHA, rho=RHO,
             cycles=CYCLES, dt_presc=dt_p, dt_free=dt_f,
             F_presc=F_p, C_presc=C_p, F_free=F_f, C_free=C_f,
             wall_presc=w_p, wall_free=w_f)
    print("saved flap_arena/out/ptera_baseline.npz")


if __name__ == "__main__":
    main()
