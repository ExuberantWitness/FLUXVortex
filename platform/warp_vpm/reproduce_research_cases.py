#!/usr/bin/env python3
"""Published flapping-wing cases on the rigid native FLUX-V5M path (note A24).

Six papers, two groups.  Without a leading-edge vortex: Kumar et al. (arXiv 2309.14832) and
Kumar et al. (Sadhana 51:103, 2026), whose solver is of the same kind as this one, a ring
lattice with a trailing-edge wake and no separation model.  With one: Hirato et al. (J.
Aircraft, doi:10.2514/1.C035124), whose leading-edge sheet this solver implements under
``FLUXV_LEV_MODEL=hirato``; Faure et al. (2023); Li et al. (JFM 972, A30); Martinez-Carmena
et al. (AIAA), the last three with two-dimensional point-vortex methods.

The solver, its defaults and the other runners are not touched.  ``--lev`` selects the
leading-edge model of a run and sets the environment switch BEFORE the solver module is
imported (the switches are read at import):

    off      crit = 1e3              every strip attached: the lattice and the trailing-edge
                                     wake alone, which is what the two papers without a
                                     leading-edge vortex compute.  The leading-edge solve of
                                     the default model decides its active set from the suction
                                     parameter against the critical value and does not read
                                     FLUXV_LESP_MASK_OFF, so the critical value is what is
                                     moved (the flag is set as well, for the legacy machinery)
    hirato   the default              the ring-chain sheet solved with the bound circulation
    legacy   FLUXV_LEV_MODEL=legacy  the two-dimensional source bank and its particles

What differs from the production settings, and is written to the manifest: the particles are
not culled (the production lifetime of 64 steps would remove the leading-edge vortex in these
cases), the wake keeps about 1.25 cycles of rows, and the critical suction parameter is the
one each paper states.

Per step the run records the world force and moment, the coefficients, the attached suction
parameter, the separation counts, the leading-edge strength and the solver's own residuals.  At
the snapshot steps it stores the whole state (bound rings, wake, leading-edge chains, particles,
loads) and, unless ``--no-fields``, the velocity on planes fixed in the wing.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \\
        python3 platform/warp_vpm/reproduce_research_cases.py --case K2-a00 --lev off \\
            --output-dir artifacts/.../research_cases/K2/K2-a00/off

Exit status: 0 completed, 2 a coefficient left the range of ``--max-coefficient``, 3 a
non-finite value.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

AIR, WATER = 1.2, 1000.0
#: damping of the wake and the sheet is not a thing here; both papers state 1 ms (Kumar et al.)
KUMAR_CHORD, KUMAR_HALF_SPAN, KUMAR_SPEED = 0.1, 0.3, 11.0
#: the wake keeps this many cycles of rows: the wake pictures of Kumar et al. span about a cycle
WAKE_CYCLES = 1.25

LOG = logging.getLogger("research_case")
#: the critical suction parameter of a run without a leading-edge vortex: no strip ever reaches it
NO_SEPARATION_CRIT = 1.0e3


@dataclass(frozen=True)
class ResearchCase:
    """One published case, with what the solver needs and what the paper reports."""

    case_id: str
    paper: str
    kind: str                       # "hinged" or "rigid"
    chord_m: float
    span_m: float                   # tip to tip
    speed_m_s: float
    density_kg_m3: float
    area_m2: float                  # what the force coefficients are divided by
    delta_time_s: float
    period_steps: int               # steps per cycle, 0 for a motion without a period
    default_steps: int
    average_cycles: float           # cycles the mean coefficients are taken over, 0 for none
    wake_rows: int
    lesp_crit: float
    spanwise_panels: int
    chordwise_panels: int = 8
    per_unit_span: bool = False     # CL = F / (q c): the two-dimensional convention
    particle_capacity: int = 524_288
    parameters: dict = field(default_factory=dict)
    snapshots: tuple[int, ...] = ()  # empty: the last cycle at quarters
    planes: tuple[str, ...] = ("midspan",)
    note: str = ""


def _kumar(case_id: str, paper: str, k: float, alpha_deg: float, delta0_deg: float, cycles: int, note: str):
    """Root flapping at a fixed geometric angle of attack: delta(t) = delta0 + 30 deg sin(omega t)."""

    omega = 2.0 * k * KUMAR_SPEED / KUMAR_CHORD            # k = omega c / (2 U)
    period = 2.0 * math.pi / omega
    steps = int(round(period / 1.0e-3))                    # the papers' 1 ms, a whole number per cycle
    return ResearchCase(
        case_id=case_id, paper=paper, kind="hinged",
        chord_m=KUMAR_CHORD, span_m=2.0 * KUMAR_HALF_SPAN, speed_m_s=KUMAR_SPEED,
        density_kg_m3=AIR,
        # the force is the whole wing's (both halves), so S is the whole wing's area: this is
        # the semi-wing's lift over the semi-wing's area.  The papers do not define S.
        area_m2=2.0 * KUMAR_CHORD * KUMAR_HALF_SPAN,
        delta_time_s=period / steps, period_steps=steps, default_steps=steps * cycles,
        average_cycles=min(3.0, float(cycles)),
        wake_rows=int(math.ceil(WAKE_CYCLES * steps)), lesp_crit=0.11, spanwise_panels=34,
        parameters={
            "alpha_deg": alpha_deg, "delta0_deg": delta0_deg, "delta1_deg": 30.0,
            "omega_rad_s": omega, "k": k, "frequency_hz": omega / (2.0 * math.pi),
        },
        planes=("cross", "midspan"),
        note=note,
    )


PAPER_K1 = "Kumar, Padhee, Samanta: Aerodynamic performance and flow mechanism of 3D flapping wing using discrete vortex method (arXiv 2309.14832)"
PAPER_K2 = "Kumar, Padhee, Samanta: Computational study of aerodynamic characteristics in flapping MAVs at different angles of attack (Sadhana 51:103, 2026)"
PAPER_H = "Hirato, Shen, Gopalarathnam, Edwards: Vortex-Sheet Representation of Leading-Edge Vortex Shedding from Finite Wings (J. Aircraft, doi:10.2514/1.C035124), case 1"
PAPER_F = "Faure, Roncin, Viaud, Simonet, Daridon: Flapping wing propulsion: comparison between discrete vortex method and other models (2023), inflexible wing"
PAPER_L = "Li, Feng, Wang, Liang: Lift generation mechanism of the leading-edge vortex for an unsteady plate (JFM 972, A30), alpha_eff,max = 24 deg"
PAPER_M = "Martinez-Carmena, He, Mulleners, Ramesh: Modulation of the Leading-Edge Vortex Shedding Rate in Discrete-Vortex Methods (AIAA), case II, as an aspect-ratio-16 plate"


def _hirato() -> ResearchCase:
    chord, speed = 1.0, 0.3                 # chord 1 m (Fig. 7); Re = 20,000 in air
    start = 0.862                           # puts 24.15 deg at t* = 1.710, as the figures are labelled
    knots = [
        [0.0, -5.0], [start, -5.0],
        [start + math.radians(50.0) / 0.6, 45.0],   # K = 0.3: 0.6 rad per unit of convective time
        [99.0, 45.0],
    ]
    return ResearchCase(
        case_id="H", paper=PAPER_H, kind="rigid",
        chord_m=chord, span_m=6.0 * chord, speed_m_s=speed, density_kg_m3=AIR,
        area_m2=6.0 * chord * chord,
        delta_time_s=0.01 * chord / speed, period_steps=0, default_steps=300,
        average_cycles=0.0, wake_rows=300, lesp_crit=0.27, spanwise_panels=48,
        parameters={"law": "ramp", "pivot": 0.25, "knots": knots, "knot_unit": "deg"},
        snapshots=(171, 204, 250, 300),
        note="pure pitch about the root quarter chord, Fig. 9: -5 deg, K = 0.3 ramp to 45 deg, held",
    )


def _faure() -> ResearchCase:
    chord, speed, k_g, h_over_c, steps, cycles = 0.1, 0.3, 1.82, 0.175, 128, 3
    omega = k_g * speed / chord            # kG = pi f c / U = omega c / U
    return ResearchCase(
        case_id="F", paper=PAPER_F, kind="rigid",
        chord_m=chord, span_m=6.0 * chord, speed_m_s=speed, density_kg_m3=WATER,
        area_m2=6.0 * chord * chord,
        delta_time_s=(2.0 * math.pi / omega) / steps, period_steps=steps,
        default_steps=steps * cycles, average_cycles=1.0,
        wake_rows=steps * cycles, lesp_crit=0.25, spanwise_panels=48, per_unit_span=True,
        parameters={
            "law": "heave_pitch", "heave_amplitude_m": h_over_c * chord, "omega_rad_s": omega,
            "pivot": 0.25, "kG": k_g, "reynolds": 30_000,
        },
        snapshots=(256, 288, 320, 352, 384),
        planes=("midspan", "span6"),
        note="pure heave, h = 0.175 c cos(omega t), AR 6 (the root wall of the paper), NACA 0012 as a flat plate",
    )


def _li() -> ResearchCase:
    chord, span, speed, k, st, alpha_eff, steps = 0.12, 0.6, 0.2, 0.3, 0.04, 24.0, 1024
    omega = 2.0 * k * speed / chord        # k = pi f c / U
    heave = st * math.pi * speed / omega    # St = 2 f h / U
    plunge = math.degrees(math.atan(omega * heave / speed))
    return ResearchCase(
        case_id="L", paper=PAPER_L, kind="rigid",
        chord_m=chord, span_m=span, speed_m_s=speed, density_kg_m3=WATER,
        area_m2=chord * span,
        delta_time_s=(2.0 * math.pi / omega) / steps, period_steps=steps,
        default_steps=steps // 2, average_cycles=0.0,
        wake_rows=steps // 2, lesp_crit=0.11, spanwise_panels=40,
        parameters={
            "law": "heave_pitch", "heave_amplitude_m": heave, "omega_rad_s": omega,
            "pitch_amplitude_rad": math.radians(alpha_eff - plunge),
            "phase_heave_rad": math.pi, "phase_pitch_rad": -0.5 * math.pi,
            "pivot": 0.0, "alpha_eff_max_deg": alpha_eff, "plunge_amplitude_deg": plunge,
            "reynolds": 24_000,
        },
        snapshots=(256, 410, 512),
        note="half a cycle: the plate at -h cos(omega t), alpha_geo = alpha_max sin(omega t), pivot on the leading edge",
    )


def _martinez() -> ResearchCase:
    chord, speed, hold = 0.1, 0.1, 6.0 - (2.0 + math.radians(45.0) / 0.4)
    return ResearchCase(
        case_id="M", paper=PAPER_M, kind="rigid",
        chord_m=chord, span_m=16.0 * chord, speed_m_s=speed, density_kg_m3=WATER,
        area_m2=16.0 * chord * chord,
        delta_time_s=0.01 * chord / speed, period_steps=0, default_steps=800,
        # three chords of wake: the two-dimensional wake of the paper is longer, and a wake of
        # 800 rows on 64 strips costs more per step than the other cases together
        average_cycles=0.0, wake_rows=300, lesp_crit=0.29, spanwise_panels=32,
        parameters={
            "law": "smoothed_ramp", "pivot": 0.25, "rate": 0.2, "smoothing": 11.0,
            "start": 2.0, "maximum_deg": 45.0, "hold": hold, "reynolds": 10_000,
        },
        snapshots=(350, 400, 500, 600, 700, 800),
        note="ramp, hold and return about the quarter chord, K = 0.2, 45 deg; the paper is two-dimensional",
    )


CASES: dict[str, ResearchCase] = {}


def _register() -> None:
    cycles = {0.1: 5, 0.2: 5}
    for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        key = f"{k:.1f}"
        CASES[f"K1-k{key}"] = _kumar(
            f"K1-k{key}", PAPER_K1, k, 2.0, -30.0, cycles.get(k, 10),
            "rigid, symmetric: delta1 = 30 deg, delta0 = -30 deg, alpha = 2 deg, AR 3",
        )
    CASES["K1-k0.3-d0"] = _kumar(
        "K1-k0.3-d0", PAPER_K1, 0.3, 2.0, 0.0, 10,
        "the same with delta0 = 0 deg, which is what the dashed delta of the paper's Fig. 5 shows",
    )
    for alpha in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        key = f"{alpha:02.0f}"
        CASES[f"K2-a{key}"] = _kumar(
            f"K2-a{key}", PAPER_K2, 0.3, alpha, 0.0, 10,
            "rigid: delta1 = 30 deg, delta0 = 0 deg (the captions), k = 0.3",
        )
    for maker in (_hirato, _faure, _li, _martinez):
        case = maker()
        CASES[case.case_id] = case


_register()


# ── Kinematics ────────────────────────────────────────────────────────────

def make_kinematics(case: ResearchCase, device: str):
    from fluxvortex.cases.research_kinematics import (
        HingedFlapKinematics,
        heave_pitch_law,
        pitch_profile_law,
        pitch_ramp_law,
        rigid_case_kinematics,
        smoothed_ramp_profile,
    )

    law = None
    if case.kind == "hinged":
        p = case.parameters
        return HingedFlapKinematics(
            chord_m=case.chord_m, half_span_m=0.5 * case.span_m, alpha_deg=p["alpha_deg"],
            delta0_deg=p["delta0_deg"], delta1_deg=p["delta1_deg"], omega_rad_s=p["omega_rad_s"],
            start_ramp_s=2.0 * math.pi / p["omega_rad_s"],       # one period, from rest
            chordwise_panels=case.chordwise_panels, spanwise_panels=case.spanwise_panels,
            device=device,
        ), None
    p = case.parameters
    if p["law"] == "heave_pitch":
        law = heave_pitch_law(
            omega_rad_s=p["omega_rad_s"], heave_amplitude_m=p.get("heave_amplitude_m", 0.0),
            pitch_amplitude_rad=p.get("pitch_amplitude_rad", 0.0),
            phase_heave_rad=p.get("phase_heave_rad", 0.0),
            phase_pitch_rad=p.get("phase_pitch_rad", 0.0), device=device,
        )
    elif p["law"] == "ramp":
        law = pitch_ramp_law(knots=p["knots"], speed=case.speed_m_s, chord=case.chord_m, device=device)
    elif p["law"] == "smoothed_ramp":
        law = pitch_profile_law(
            profile=smoothed_ramp_profile(
                rate=p["rate"], smoothing=p["smoothing"], start=p["start"],
                maximum_deg=p["maximum_deg"], hold=p["hold"],
            ),
            speed=case.speed_m_s, chord=case.chord_m, device=device,
        )
    else:
        raise KeyError(f"unknown law {p['law']!r}")
    return rigid_case_kinematics(
        law=law, chord_m=case.chord_m, span_m=case.span_m, pivot_fraction_chord=p["pivot"],
        chordwise_panels=case.chordwise_panels, spanwise_panels=case.spanwise_panels,
        device=device, surface_id=f"paper_{case.case_id}",
    ), law


def pose(case: ResearchCase, law, time: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the wing's reference frame is at ``time``: its origin and its axes (as columns)."""

    from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics

    dtype = torch.float64
    if law is None:
        return torch.zeros(3, device=device, dtype=dtype), torch.eye(3, device=device, dtype=dtype)
    pos, quat, _, _ = law(time)
    return pos, PrescribedRigidSurfaceKinematics._quaternion_to_matrix(quat)


# ── Sampling planes ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Plane:
    name: str
    origin: torch.Tensor    # (3,)
    e1: torch.Tensor        # (3,) first in-plane axis
    e2: torch.Tensor        # (3,) second in-plane axis
    a1: torch.Tensor        # (n1,) coordinates along e1
    a2: torch.Tensor        # (n2,) coordinates along e2
    names: tuple[str, str] = ("x", "z")   # what e1 and e2 are in the wing's own axes


def build_planes(case: ResearchCase, which: tuple[str, ...], origin, axes, device: str) -> list[Plane]:
    chord, dtype = case.chord_m, torch.float64
    x_axis, y_axis, z_axis = axes[:, 0], axes[:, 1], axes[:, 2]
    station = (
        0.25 * case.span_m if case.kind == "hinged"
        else 0.5 * case.span_m / case.spanwise_panels    # the middle of a strip, never on a leg
    )
    planes: list[Plane] = []
    for name in which:
        if name == "midspan":
            planes.append(Plane(
                "midspan", origin + station * y_axis, x_axis, z_axis,
                torch.linspace(-0.5 * chord, 4.5 * chord, 181, device=device, dtype=dtype),
                torch.linspace(-1.25 * chord, 1.25 * chord, 91, device=device, dtype=dtype),
                names=("x", "z"),
            ))
        elif name == "span6":
            # Faure et al., Fig. 17: six chordwise planes of the semi-wing
            for index, over_chord in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 2.85)):
                planes.append(Plane(
                    f"span{index}", origin + over_chord * chord * y_axis, x_axis, z_axis,
                    torch.linspace(-0.5 * chord, 1.5 * chord, 81, device=device, dtype=dtype),
                    torch.linspace(-0.5 * chord, 0.5 * chord, 41, device=device, dtype=dtype),
                    names=("x", "z"),
                ))
        elif name == "cross":
            # Kumar et al., Fig. 4b: the cross-flow plane 0.3 c behind the trailing edge
            planes.append(Plane(
                "cross", origin + 1.3 * chord * x_axis, y_axis, z_axis,
                torch.linspace(-0.4, 0.4, 41, device=device, dtype=dtype),
                torch.linspace(-0.4, 0.1, 26, device=device, dtype=dtype),
                names=("y", "z"),
            ))
        else:
            raise KeyError(f"unknown plane {name!r}")
    return planes


def metrics_from_history(case: ResearchCase, arguments, average_cycles: float | None = None) -> dict:
    """The summary of a finished run from its own history.csv: what a run writes at its end.

    A run that completed its steps and then failed in its bookkeeping leaves ``history.csv``
    behind; this rebuilds ``metrics.json`` from it without touching the solver.
    """

    out = Path(arguments.output_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    with (out / "history.csv").open() as handle:
        reader = csv.reader(handle)
        next(reader)
        history = [
            [float(value) if value != "" else math.nan for value in row] for row in reader
        ]
    period = manifest.get("period_steps") or 0
    cycles = case.average_cycles if average_cycles is None else average_cycles
    average_steps = int(round(cycles * period)) if period else 0
    dynamic = 0.5 * case.density_kg_m3 * case.speed_m_s ** 2
    metrics: dict[str, Any] = {
        "case": case.case_id, "paper": case.paper, "lev": arguments.lev,
        "status": "completed", "steps_done": len(history), "steps": manifest["steps"],
        "rescored_from_history": True,
        "effective_lesp_crit": manifest.get("effective_lesp_crit"),
        "snapshots": manifest.get("snapshot_steps"),
    }
    if average_steps and len(history) >= average_steps:
        window = history[len(history) - average_steps:]
        metrics["mean"] = {name: float(np.mean([row[COLUMN[name]] for row in window]))
                           for name in ("CL", "CT", "CD", "CM")}
        metrics["mean"]["cycles"] = average_steps / period
        metrics["extrema"] = {
            "CL_min": min(row[COLUMN["CL"]] for row in window),
            "CL_max": max(row[COLUMN["CL"]] for row in window),
            "CT_min": min(row[COLUMN["CT"]] for row in window),
            "CT_max": max(row[COLUMN["CT"]] for row in window),
        }
        if period:
            metrics["cycle_means"] = [
                {"cycle": index + 1, **{name: float(np.mean([row[COLUMN[name]] for row in history[index * period:(index + 1) * period]]))
                                        for name in ("CL", "CT", "CD")}}
                for index in range(len(history) // period)
            ]
    else:
        metrics["extrema"] = {
            "CL_min": min(row[COLUMN["CL"]] for row in history),
            "CL_max": max(row[COLUMN["CL"]] for row in history),
        }
    for name in ("lesp_pre_max_abs", "kelvin_residual", "neumann_residual"):
        values = [row[COLUMN[name]] for row in history if not math.isnan(row[COLUMN[name]])]
        metrics[f"max_{name}"] = max(values, default=float("nan"))
    metrics["max_particles"] = max(row[COLUMN["particles"]] for row in history)
    metrics["max_wake_rows"] = max(row[COLUMN["wake_rows"]] for row in history)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    LOG.info("rescored %s from %d history rows: %s", out, len(history), json.dumps(metrics.get("mean", {})))
    print(json.dumps(metrics, indent=2, default=str))
    return metrics



# ── The state at a snapshot ───────────────────────────────────────────────

def solution_velocity(solver, trial, geometry, points: torch.Tensor, gamma_load: torch.Tensor):
    """The solution velocity at ``points``: the terms the load refinement integrates with.

    ``RigidNativeV5MSolver._refine_load_with_filament_velocity`` builds its filament velocities
    from the freestream, the bound sheet, the stored wake with its core, the particles, the
    leading-edge chains and the pseudovortex rings.  This is the same sum at other points.
    """

    value = solver.v_inf[None, :].expand(points.shape[0], -1) + solver._ring_velocity(
        points, geometry.rings, gamma_load, rough=False
    )
    if trial.wake_rings.shape[0]:
        value = value + solver._ring_velocity(
            points, trial.wake_rings, trial.wake_gamma, rough=False,
            core_radius=solver._core_radius(geometry), free=True,
        )
    if trial.particle_field.n:
        value = value + trial.particle_field.velocity_at_cuda(points)
    if solver._lev_field is not None:
        value = value + solver._ring_velocity(points, *solver._lev_field, rough=False, free=True)
        value = value + solver._ring_velocity(points, *solver._lev_pseudovortex, rough=False)
    return value


def lesp_of_strip(geometry, gamma: torch.Tensor, span_panels: int, speed: float) -> torch.Tensor:
    """The suction parameter of every strip, as the solver defines it (Hirato et al., Eq. 6)."""

    from fluxvortex.warp_fsi.q16_flux_v5m_native import LESP_FACTOR

    chord = 0.5 * (
        torch.linalg.vector_norm(geometry.trailing_edge[:-1] - geometry.leading_edge[:-1], dim=1)
        + torch.linalg.vector_norm(geometry.trailing_edge[1:] - geometry.leading_edge[1:], dim=1)
    )
    first_row = geometry.rings[:span_panels]
    first_dx = 0.5 * (
        torch.linalg.vector_norm(first_row[:, 3] - first_row[:, 0], dim=1)
        + torch.linalg.vector_norm(first_row[:, 2] - first_row[:, 1], dim=1)
    )
    theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / chord, -1.0, 1.0))
    scale = chord * speed * (theta + torch.sin(theta)) / LESP_FACTOR
    return -gamma[:span_panels] / scale


def host(tensor) -> np.ndarray:
    return tensor.detach().contiguous().cpu().numpy()


def save_snapshot(
    path: Path, case: ResearchCase, step: int, time: float, origin, axes, solver, trial,
    proposal, planes: list[Plane],
) -> None:
    from fluxvortex.warp_fsi import lev_hirato

    geometry = solver.surface.evaluate(None, None)
    ns, nc = case.spanwise_panels, case.chordwise_panels
    extra = solver._lev_bound_extra
    gamma_load = trial.gamma_bound if extra is None else trial.gamma_bound + extra
    arrays: dict[str, Any] = {
        "step": np.int64(step), "time": np.float64(time),
        "chord_m": np.float64(case.chord_m), "speed_m_s": np.float64(case.speed_m_s),
        "density_kg_m3": np.float64(case.density_kg_m3), "area_m2": np.float64(case.area_m2),
        "per_unit_span": np.bool_(case.per_unit_span),
        "spanwise_panels": np.int64(ns), "chordwise_panels": np.int64(nc),
        "pose_origin": host(origin), "pose_axes": host(axes),
        "bound_rings": host(geometry.rings), "bound_ring_velocity": host(geometry.ring_velocity),
        "collocation": host(geometry.collocation), "normals": host(geometry.normals),
        "areas": host(geometry.areas),
        "leading_edge": host(geometry.leading_edge), "trailing_edge": host(geometry.trailing_edge),
        "gamma_bound": host(trial.gamma_bound), "gamma_load": host(gamma_load),
        "lesp_strip": host(lesp_of_strip(geometry, gamma_load, ns, case.speed_m_s)),
        "wake_rings": host(trial.wake_rings), "wake_gamma": host(trial.wake_gamma),
        "panel_forces": host(proposal.load.panel_forces),
        "total_force": host(proposal.load.total_force),
        "total_moment": host(proposal.load.total_moment),
    }
    if trial.lev_chain is not None:
        chain = trial.lev_chain
        arrays.update({
            "lev_rings": host(chain.rings), "lev_gamma": host(chain.gamma),
            "lev_count": host(chain.count), "lev_joined": host(chain.joined),
        })
        if solver._lev_pseudovortex is not None:
            arrays["pseudovortex_rings"] = host(solver._lev_pseudovortex[0])
            arrays["pseudovortex_gamma"] = host(solver._lev_pseudovortex[1])
    field = trial.particle_field
    if field.n:
        count = int(field.n)
        arrays.update({
            "particle_pos": host(field.positions_cuda[:count]),
            "particle_gamma": host(field.gammas_cuda[:count]),
            "particle_sigma": host(field.sigmas_cuda[:count]),
            "particle_circul": host(field.circul[:count]),
            "particle_birth_step": host(field.birth_step[:count]),
            "particle_source_strip": host(field.source_strip[:count]),
        })
    else:
        arrays.update({
            "particle_pos": np.zeros((0, 3)), "particle_gamma": np.zeros((0, 3)),
            "particle_sigma": np.zeros((0,)), "particle_circul": np.zeros((0,)),
            "particle_birth_step": np.zeros((0,), dtype=np.int64),
            "particle_source_strip": np.zeros((0,), dtype=np.int64),
        })
    for index, plane in enumerate(planes):
        points = (
            plane.origin[None, None, :]
            + plane.a1[None, :, None] * plane.e1[None, None, :]
            + plane.a2[:, None, None] * plane.e2[None, None, :]
        ).reshape(-1, 3)
        velocity = solution_velocity(solver, trial, geometry, points, gamma_load)
        in_body = velocity @ axes
        prefix = f"field{index}_"
        arrays.update({
            prefix + "name": np.str_(plane.name),
            prefix + "origin": host(plane.origin), prefix + "e1": host(plane.e1),
            prefix + "e2": host(plane.e2), prefix + "a1": host(plane.a1),
            prefix + "a2": host(plane.a2),
            prefix + "axis1": np.str_(plane.names[0]), prefix + "axis2": np.str_(plane.names[1]),
            prefix + "velocity_body": host(in_body.reshape(plane.a2.shape[0], plane.a1.shape[0], 3)),
        })
    np.savez_compressed(path, **arrays)


# ── The run ───────────────────────────────────────────────────────────────

HISTORY_COLUMNS = (
    "step", "time", "t_star", "t_over_T", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
    "CL", "CT", "CD", "CM", "mid_cl", "pose_z",
    "lesp_pre_max_abs", "separated_strips", "lev_releases", "lev_strength_sum",
    "lev_circulation_sum", "particles", "wake_rows", "kelvin_residual", "neumann_residual",
    "lesp_pin_max_abs", "detached_strips",
)
COLUMN = {name: index for index, name in enumerate(HISTORY_COLUMNS)}


def default_snapshots(case: ResearchCase, total_steps: int) -> tuple[int, ...]:
    if case.snapshots:
        return tuple(step for step in case.snapshots if step <= total_steps)
    if not case.period_steps:
        return (total_steps,)
    start = total_steps - case.period_steps
    return tuple(start + int(round(fraction * case.period_steps)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0))


def write_history(out: Path, history: list[list[float]]) -> None:
    """The history so far, rewritten at every report: a run that is stopped keeps its record."""

    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HISTORY_COLUMNS)
        for row in history:
            writer.writerow(["" if isinstance(value, float) and math.isnan(value) else value for value in row])


def run_case(case: ResearchCase, arguments) -> int:
    from fluxvortex.aero.v5m.stepper import RigidV5MStepper
    from fluxvortex.warp_fsi import q16_flux_v5m_native as native
    from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
        RIGID_NATIVE_V5M_CONTRACT,
        RigidNativeV5MSolver,
        RigidV5MSurface,
    )

    device = arguments.device
    if not torch.cuda.is_available():
        raise RuntimeError("the research cases run on CUDA only")
    out = Path(arguments.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    nc, ns = case.chordwise_panels, case.spanwise_panels
    if arguments.span is not None:
        # a diagnostic: the same motion on a wing of another span (the papers do not say whether
        # their span is the full wing's or the semi-wing's)
        case = replace(case, span_m=float(arguments.span), area_m2=float(arguments.span) * case.chord_m)
    total_steps = int(arguments.steps or case.default_steps)
    dt = case.delta_time_s
    wake_rows = int(arguments.wake_rows or case.wake_rows)
    period = case.period_steps if arguments.steps_per_cycle is None else int(arguments.steps_per_cycle)
    if period and arguments.steps_per_cycle is not None:
        # a finer step than the paper's on a periodic case: the same motion, more steps per cycle
        refine = period / case.period_steps
        dt = 2.0 * math.pi / case.parameters["omega_rad_s"] / period
        wake_rows = int(math.ceil(case.wake_rows * refine))
        total_steps = int(arguments.steps or round(case.default_steps * refine))
    average_cycles = case.average_cycles if arguments.average_cycles is None else arguments.average_cycles
    average_steps = int(round(average_cycles * period)) if period else 0
    snapshots = set(int(value) for value in (
        arguments.snapshot_steps.split(",") if arguments.snapshot_steps
        else default_snapshots(case, total_steps)
    ))
    planes_selected = (
        tuple(arguments.planes.split(",")) if arguments.planes
        else (() if arguments.no_fields else case.planes)
    )
    dynamic = float(case.density_kg_m3 * case.speed_m_s ** 2 / 2.0)
    scale = dynamic * case.chord_m if case.per_unit_span else dynamic * case.area_m2
    moment_scale = dynamic * case.chord_m ** 2 if case.per_unit_span else dynamic * case.area_m2 * case.chord_m
    strip_area = case.chord_m * case.span_m / ns
    strip_scale = dynamic * strip_area

    kin, law = make_kinematics(case, device)
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=device)
    settings = native.NativeV5MConfig(
        chordwise_panels=nc, spanwise_panels=ns,
        density=case.density_kg_m3, freestream=case.speed_m_s, aerodynamic_dt=dt,
        wake_max_rows=wake_rows, particle_capacity=case.particle_capacity,
        wake_free_rows=int(arguments.wake_free_rows or 0),
        particle_max_age_steps=0,             # no cull: the leading-edge vortex lives as long as the run
        dvm_target_spacing_chord=0.018,
        lesp_crit_override=NO_SEPARATION_CRIT if arguments.lev == "off" else case.lesp_crit,
        wake_history_mode="bound_rate", device=device,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=device)
    switches = {key: value for key, value in os.environ.items() if key.startswith("FLUXV_")}
    manifest = {
        "contract": RIGID_NATIVE_V5M_CONTRACT,
        "case": case.case_id, "paper": case.paper, "note": case.note,
        "kind": case.kind, "parameters": case.parameters, "lev": arguments.lev,
        "chord_m": case.chord_m, "speed_m_s": case.speed_m_s, "span_m": case.span_m,
        "effective_lesp_crit": settings.effective_lesp_crit,
        "paper_lesp_crit": case.lesp_crit,
        "grid": {"chordwise": nc, "spanwise": ns},
        "delta_time_s": dt, "period_steps": period, "steps": total_steps,
        "average_cycles": average_cycles, "average_steps": average_steps,
        "wake_max_rows": wake_rows, "particle_capacity": case.particle_capacity,
        "particle_max_age_steps": 0, "dvm_target_spacing_chord": 0.018,
        "snapshot_steps": sorted(snapshots), "planes": list(planes_selected),
        "coefficients": {
            "scale": scale, "moment_scale": moment_scale,
            "CL": "Fz / scale", "CT": "-Fx / scale", "CD": "Fx / scale",
            "CM": "My about the world origin / moment_scale (positive: leading edge up)",
            "mid_cl": "force of the strip next to the centreline / (q c dy)",
        },
        "environment": switches,
        "kernel": {"eps_v": native.__dict__.get("_EPS_V"), "te_wake": native.__dict__.get("_TE_WAKE_CHOICE"),
                   "lev_model": native.__dict__.get("_LEV_CHOICE"), "lesp_mask_off": native.__dict__.get("_LESP_MASK_OFF")},
        "torch": torch.__version__, "device": torch.cuda.get_device_name(torch.device(device)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    LOG.info("case %s, %s, lev %s: %d steps of %.6g s, grid %dx%d, %d wake rows, crit %.4g",
             case.case_id, case.paper, arguments.lev, total_steps, dt, nc, ns, wake_rows,
             settings.effective_lesp_crit)
    LOG.info("switches %s", json.dumps(switches, sort_keys=True))

    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    owner = stepper.initialize((kin.evaluate(0.0),))
    history: list[list[float]] = []
    status = "completed"
    report_every = arguments.report_every or max(1, total_steps // 10)
    for step in range(1, total_steps + 1):
        time_now = step * dt
        frame = kin.evaluate(time_now)
        proposal = stepper.propose(owner, (frame,), dt)
        trial = proposal.trial_state
        force, moment = proposal.load.total_force, proposal.load.total_moment
        if not bool(torch.isfinite(force).all().item()) or not bool(torch.isfinite(moment).all().item()):
            status = "non_finite"
            LOG.info("step %d: non-finite force or moment; stop", step)
            break
        fx, fy, fz = (float(value) for value in force)
        mx, my, mz = (float(value) for value in moment)
        cl, ct, cd, cm = fz / scale, -fx / scale, fx / scale, my / moment_scale
        t_star = case.speed_m_s * time_now / case.chord_m
        if abs(cl) > arguments.max_coefficient or abs(ct) > arguments.max_coefficient:
            status = "diverged"
            LOG.info("step %d: coefficient out of range (CL %.4g, CT %.4g); stop", step, cl, ct)
            break
        origin, axes = pose(case, law, time_now, device)
        if step in snapshots:
            planes = build_planes(case, planes_selected, origin, axes, device)
            save_snapshot(
                out / f"snapshot_{step:05d}.npz", case, step, time_now, origin, axes, solver,
                trial, proposal, planes,
            )
            LOG.info("snapshot %d: t* = %.4f, CL %.4f, CT %.4f, %d particles, %d wake rows",
                     step, t_star, cl, ct, int(trial.particle_field.n),
                     int(trial.wake_gamma.numel() // ns))
        mid_force = proposal.load.panel_forces.reshape(nc, ns, 3)[:, ns // 2].sum(dim=0)
        chain = trial.lev_chain
        released = trial.lev_strip_released_total
        circulation = (
            float(chain.gamma.sum().item()) if chain is not None
            else float(released.sum().item()) if released is not None else 0.0
        )
        stepper.commit(owner, proposal)
        diag = owner.state.diagnostics[-1]
        record = diag.get("lev_hirato")
        strength_sum = (
            float(record["strength_sum"]) if record
            else float(diag.get("lev_strip_released_total_sum", float("nan")))
        )
        history.append([
            step, time_now, t_star, step / period if period else float("nan"),
            fx, fy, fz, mx, my, mz, cl, ct, cd, cm,
            float(mid_force[2].item()) / strip_scale, float(origin[2].item()),
            float(diag["lesp_pre_max_abs"]), int(diag["separated_strip_count"]),
            int(diag["lev_release_count"]), strength_sum, circulation,
            int(trial.particle_field.n), int(trial.wake_gamma.numel() // ns),
            float(diag.get("kelvin_identity_residual", float("nan"))),
            float(diag.get("full_surface_neumann_max_abs", float("nan"))),
            float(diag.get("lesp_pin_max_abs", float("nan"))),
            int((record or {}).get("detached_strips", 0)),
        ])
        if step % report_every == 0 or step == total_steps:
            write_history(out, history)
            elapsed = time.perf_counter() - started
            LOG.info(
                "step %d/%d t*=%.3f CL=%.4f CT=%.4f CM=%.4f lesp=%.4f strips=%d particles=%d wake=%d %.3fs/step",
                step, total_steps, t_star, cl, ct, cm, history[-1][COLUMN["lesp_pre_max_abs"]],
                history[-1][COLUMN["separated_strips"]], history[-1][COLUMN["particles"]],
                history[-1][COLUMN["wake_rows"]], elapsed / step,
            )
    torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(torch.device(device)) / (1024.0 ** 2)

    write_history(out, history)
    steps_done = len(history)
    metrics: dict[str, Any] = {
        "case": case.case_id, "paper": case.paper, "lev": arguments.lev, "status": status,
        "steps_done": steps_done, "steps": total_steps, "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, steps_done), "gpu_peak_mib": peak_mib,
        "effective_lesp_crit": settings.effective_lesp_crit,
        "snapshots": sorted(snapshots),
    }
    if average_steps and steps_done >= average_steps:
        window = history[steps_done - average_steps:]
        metrics["mean"] = {
            name: float(np.mean([row[COLUMN[name]] for row in window]))
            for name in ("CL", "CT", "CD", "CM")
        }
        metrics["mean"]["cycles"] = average_steps / period
        cl_window = [row[COLUMN["CL"]] for row in window]
        metrics["extrema"] = {
            "CL_min": min(cl_window), "CL_max": max(cl_window),
            "CT_min": min(row[COLUMN["CT"]] for row in window),
            "CT_max": max(row[COLUMN["CT"]] for row in window),
        }
        # the cycle means, to see whether the last cycles repeat
        metrics["cycle_means"] = [
            {
                "cycle": index + 1,
                **{
                    name: float(np.mean([row[COLUMN[name]] for row in history[index * period:(index + 1) * period]]))
                    for name in ("CL", "CT", "CD")
                },
            }
            for index in range(steps_done // period)
        ]
    else:
        metrics["extrema"] = {
            name: float(fn(row[COLUMN["CL"]] for row in history))
            for name, fn in (("CL_min", min), ("CL_max", max))
        } if history else {}

    def largest(name: str) -> float:
        values = [row[COLUMN[name]] for row in history if not math.isnan(row[COLUMN[name]])]
        return max(values, default=float("nan"))

    metrics["max_lesp_pre"] = largest("lesp_pre_max_abs")
    metrics["max_particles"] = max((row[COLUMN["particles"]] for row in history), default=0)
    metrics["max_wake_rows"] = max((row[COLUMN["wake_rows"]] for row in history), default=0)
    metrics["max_kelvin_residual"] = largest("kelvin_residual")
    metrics["max_neumann_residual"] = largest("neumann_residual")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    LOG.info("done: %s after %d steps, %.1f s, %.3f s/step, peak %.0f MiB; means %s",
             status, steps_done, elapsed, elapsed / max(1, steps_done), peak_mib,
             json.dumps(metrics.get("mean", {})))
    LOG.removeHandler(handler)
    handler.close()
    return {"completed": 0, "diverged": 2, "non_finite": 3}[status]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--rescore-only", action="store_true",
                        help="rebuild metrics.json from an existing history.csv; no solver")
    parser.add_argument("--lev", choices=("off", "hirato", "hirato_v2", "legacy"), default="off")
    parser.add_argument("--steps", type=int, default=None, help="total steps (default: the case's own)")
    parser.add_argument("--steps-per-cycle", type=int, default=None, help="periodic cases: a finer step than the paper's")
    parser.add_argument("--average-cycles", type=float, default=None)
    parser.add_argument("--wake-rows", type=int, default=None)
    parser.add_argument("--snapshot-steps", default=None, help="comma-separated step numbers")
    parser.add_argument("--planes", default=None, help="comma-separated: midspan, span6, cross")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--max-coefficient", type=float, default=50.0)
    parser.add_argument("--wake-core", type=float, default=None,
                        help="FLUXV_FREE_WAKE_CORE_FRACTION of the run (default: the solver's 0.1)")
    parser.add_argument("--span", type=float, default=None, help="the wing's span, for the span diagnostic")
    parser.add_argument("--wake-free-rows", type=int, default=None,
                        help="rows of the wake left free to convect; the rest are frozen (a stability diagnostic)")
    parser.add_argument("--lev-detach", action="store_true",
                        help="FLUXV_LEV_DETACH=on: the sheet stops being fed when its own induction "
                             "at the leading edge acts with the freestream")
    parser.add_argument("--lev-shed-law", choices=("pin", "flux"), default=None,
                        help="FLUXV_LEV_SHED_LAW: how the leading-edge sheet decides its strength")
    parser.add_argument("--shed-distance", type=float, default=None,
                        help="FLUXV_SHED_DISTANCE_FRACTION: the shed-vortex distance in U_inf dt (Kumar et al.)")
    parser.add_argument("--te-wake", choices=("v3", "v4", "legacy"), default=None,
                        help="FLUXV_TE_WAKE: the trailing-edge wake scheme")
    parser.add_argument("--te-split-beta", type=float, default=None,
                        help="FLUXV_TE_SPLIT_BETA: connected TE split fraction under --te-wake v4 (default 0.3)")
    parser.add_argument("--convection-v4", action="store_true",
                        help="FLUXV_CONVECTION_V4=1: synchronized t^{n-1} unique-vertex wake/LEV convection")
    parser.add_argument("--ring-kernel", choices=("legacy", "v3"), default=None,
                        help="FLUXV_RING_KERNEL: ring Biot-Savart kernel")
    parser.add_argument("--wake-core-panels", type=float, default=None,
                        help="FLUXV_WAKE_CORE_PANELS: v3 core radius in chordwise panels")
    parser.add_argument("--wake-core-collocation", action="store_true",
                        help="FLUXV_WAKE_CORE_AT_COLLOCATION=1: apply v3 wake core at wing collocation points")
    parser.add_argument("--wing-slip-wall", type=float, default=None,
                        help="FLUXV_WING_SLIP_WALL: wing-impenetrability slip-wall clearance in panel dx")
    parser.add_argument("--squire-core", type=float, default=None,
                        help="FLUXV_SQUIRE_CORE: Squire viscous-turbulent core spreading coefficient")
    parser.add_argument("--lesp-urel", action="store_true",
                        help="FLUXV_LESP_UREL=1: Galilean-invariant relative LE speed in LESP")
    parser.add_argument("--lev-sep-suction", action="store_true",
                        help="FLUXV_LEV_SEP_SUCTION=1: LESP-limited Eq. (20) suction on separated strips in KJ")
    parser.add_argument("--hirato-pressure-loads", action="store_true",
                        help="FLUXV_HIRATO_PRESSURE_LOADS=1: Hirato Eqs. (16)-(20) pressure + suction loads")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="FLUXV_RING_MIXED_PRECISION=1: coalesced FP32 cored Biot-Savart kernel")
    parser.add_argument("--fast-rigid-loads", action="store_true",
                        help="FLUXV_FAST_RIGID_LOADS=1: skip redundant bound_rate rigid load passes")
    parser.add_argument("--wake-core-steps", type=float, default=None,
                        help="FLUXV_FREE_WAKE_CORE_STEPS: the core in U_inf dt, as Li et al. and Hirato et al. state it")
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    # the queue passes arguments without a shell, so an empty value may arrive quoted
    for name in ("snapshot_steps", "planes"):
        value = getattr(arguments, name)
        if value is not None and value.strip().strip("'\"") == "":
            setattr(arguments, name, None)
    # the solver reads its switches when it is imported, so they are set first
    os.environ.pop("FLUXV_LESP_MASK_OFF", None)
    os.environ["FLUXV_LEV_MODEL"] = (
        arguments.lev if arguments.lev in {"hirato", "hirato_v2"} else "legacy"
    )
    if arguments.lev == "hirato_v2":
        # the repaired sheet needs one core radius for the shared edges (note A24, B2b)
        os.environ["FLUXV_RING_KERNEL"] = "v3"
        os.environ.setdefault("FLUXV_LEV_MAX_ROWS", "512")
    if arguments.lev == "off":
        # every strip attached: the leading-edge model is never asked to shed anything, and the
        # legacy one is the only one that can be combined with the legacy trailing-edge wake
        os.environ["FLUXV_LESP_MASK_OFF"] = "1"
    if arguments.wake_core is not None:
        os.environ["FLUXV_FREE_WAKE_CORE_FRACTION"] = str(arguments.wake_core)
    if arguments.wake_core_steps is not None:
        os.environ["FLUXV_FREE_WAKE_CORE_STEPS"] = str(arguments.wake_core_steps)
    if arguments.shed_distance is not None:
        os.environ["FLUXV_SHED_DISTANCE_FRACTION"] = str(arguments.shed_distance)
    if arguments.lev_shed_law is not None:
        os.environ["FLUXV_LEV_SHED_LAW"] = arguments.lev_shed_law
    if arguments.lev_detach:
        os.environ["FLUXV_LEV_DETACH"] = "on"
        os.environ["FLUXV_LEV_SEPARATED_STATE"] = "on"
    if arguments.te_wake is not None:
        os.environ["FLUXV_TE_WAKE"] = arguments.te_wake
    if arguments.te_split_beta is not None:
        os.environ["FLUXV_TE_SPLIT_BETA"] = str(arguments.te_split_beta)
    if arguments.convection_v4:
        os.environ["FLUXV_CONVECTION_V4"] = "1"
    if arguments.ring_kernel is not None:
        os.environ["FLUXV_RING_KERNEL"] = arguments.ring_kernel
    if arguments.wake_core_panels is not None:
        os.environ["FLUXV_WAKE_CORE_PANELS"] = str(arguments.wake_core_panels)
    if arguments.wake_core_collocation:
        os.environ["FLUXV_WAKE_CORE_AT_COLLOCATION"] = "1"
    if arguments.wing_slip_wall is not None:
        os.environ["FLUXV_WING_SLIP_WALL"] = str(arguments.wing_slip_wall)
    if arguments.squire_core is not None:
        os.environ["FLUXV_SQUIRE_CORE"] = str(arguments.squire_core)
    if arguments.lesp_urel:
        os.environ["FLUXV_LESP_UREL"] = "1"
    if arguments.lev_sep_suction:
        os.environ["FLUXV_LEV_SEP_SUCTION"] = "1"
    if arguments.hirato_pressure_loads:
        os.environ["FLUXV_HIRATO_PRESSURE_LOADS"] = "1"
    if arguments.mixed_precision:
        os.environ["FLUXV_RING_MIXED_PRECISION"] = "1"
    if arguments.fast_rigid_loads:
        os.environ["FLUXV_FAST_RIGID_LOADS"] = "1"
    case = CASES[arguments.case]
    if arguments.rescore_only:
        metrics_from_history(case, arguments)
        return 0
    return run_case(case, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
