#!/usr/bin/env python3
"""Canonical guard for pressure coupling on an actual-thickness body.

The exact potential flow around a circular cylinder with prescribed
circulation separates linearly at the velocity-potential level.  Bernoulli
does not separate at the pressure level because it contains the cross term
between the non-circulatory and circulatory surface velocities.  This guard
freezes that representation fact before any RoboEagle source-panel solver is
implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PLATFORM = Path(__file__).resolve().parent
CASES = PLATFORM / "docs" / "diag" / "thick_body_pressure_coupling_cases.yaml"
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "thick_body_pressure_coupling_results.json"
)


class ThickBodyPressureCouplingError(ValueError):
    """Invalid canonical cylinder or pressure-coupling configuration."""


@dataclass(frozen=True)
class CylinderPressureCoupling:
    theta: np.ndarray
    normal: np.ndarray
    noncirculatory_velocity: np.ndarray
    circulation_velocity: np.ndarray
    total_pressure_coefficient: np.ndarray
    pressure_level_addition: np.ndarray
    bernoulli_cross_term: np.ndarray
    total_force_coefficient: np.ndarray
    pressure_addition_force_coefficient: np.ndarray
    gauge_shifted_force_coefficient: np.ndarray
    expected_lift_coefficient: float


def _pressure_force_coefficient(
    theta: np.ndarray,
    normal: np.ndarray,
    pressure_coefficient: np.ndarray,
) -> np.ndarray:
    """Integrate pressure force using diameter ``2R`` as reference chord."""
    if (
        theta.ndim != 1
        or normal.shape != (len(theta), 2)
        or pressure_coefficient.shape != (len(theta),)
        or not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(pressure_coefficient))
    ):
        raise ThickBodyPressureCouplingError(
            "theta, normal, and pressure must be finite paired arrays"
        )
    spacing = 2.0 * np.pi / len(theta)
    return -0.5 * spacing * np.einsum(
        "i,ij->j", pressure_coefficient, normal
    )


def circular_cylinder_pressure_coupling(
    *,
    radius: float,
    freestream_speed: float,
    circulation_velocity_ratio: float,
    azimuth_nodes: int,
    pressure_gauge: float = 0.731,
) -> CylinderPressureCoupling:
    """Return the exact cylinder pressure and a deliberately invalid split.

    ``circulation_velocity_ratio`` is
    ``k = Gamma/(2*pi*R*U)``.  The surface tangential velocity is

    ``v_theta/U = -2*sin(theta) + k``.

    The invalid pressure-level split keeps the non-circulatory pressure and
    the quadratic pressure of the circulation mode, but omits their Bernoulli
    cross term.
    """
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(freestream_speed)
        or freestream_speed <= 0.0
        or not np.isfinite(circulation_velocity_ratio)
        or not isinstance(azimuth_nodes, int)
        or azimuth_nodes < 16
        or azimuth_nodes % 4 != 0
        or not np.isfinite(pressure_gauge)
    ):
        raise ThickBodyPressureCouplingError(
            "invalid cylinder, flow, circulation, grid, or gauge"
        )

    theta = 2.0 * np.pi * np.arange(azimuth_nodes) / azimuth_nodes
    normal = np.column_stack((np.cos(theta), np.sin(theta)))
    noncirculatory = -2.0 * freestream_speed * np.sin(theta)
    circulatory = np.full(
        azimuth_nodes,
        circulation_velocity_ratio * freestream_speed,
    )
    total_velocity = noncirculatory + circulatory
    total_cp = 1.0 - (total_velocity / freestream_speed) ** 2
    noncirculatory_cp = 1.0 - (
        noncirculatory / freestream_speed
    ) ** 2
    circulation_pressure_increment = -(
        circulatory / freestream_speed
    ) ** 2
    pressure_addition = (
        noncirculatory_cp + circulation_pressure_increment
    )
    cross_term = (
        -2.0
        * noncirculatory
        * circulatory
        / freestream_speed**2
    )
    total_force = _pressure_force_coefficient(theta, normal, total_cp)
    addition_force = _pressure_force_coefficient(
        theta, normal, pressure_addition
    )
    gauge_force = _pressure_force_coefficient(
        theta, normal, total_cp + pressure_gauge
    )
    expected_lift = -2.0 * np.pi * circulation_velocity_ratio
    return CylinderPressureCoupling(
        theta=theta,
        normal=normal,
        noncirculatory_velocity=noncirculatory,
        circulation_velocity=circulatory,
        total_pressure_coefficient=total_cp,
        pressure_level_addition=pressure_addition,
        bernoulli_cross_term=cross_term,
        total_force_coefficient=total_force,
        pressure_addition_force_coefficient=addition_force,
        gauge_shifted_force_coefficient=gauge_force,
        expected_lift_coefficient=expected_lift,
    )


def evaluate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    canonical = contract["canonical_problem"]
    thresholds = contract["thresholds"]
    radius = float(canonical["radius"])
    speed = float(canonical["freestream_speed"])
    nodes = int(canonical["azimuth_nodes"])
    cases: dict[str, Any] = {}
    maxima = {
        "pressure_identity_error": 0.0,
        "cross_term_identity_error": 0.0,
        "force_coefficient_error": 0.0,
        "zero_drag_error": 0.0,
        "gauge_force_change": 0.0,
    }
    minimum_lift_lost = np.inf
    for ratio_value in canonical["circulation_velocity_ratios"]:
        ratio = float(ratio_value)
        result = circular_cylinder_pressure_coupling(
            radius=radius,
            freestream_speed=speed,
            circulation_velocity_ratio=ratio,
            azimuth_nodes=nodes,
        )
        closed_form = (
            1.0
            - (
                -2.0 * np.sin(result.theta) + ratio
            ) ** 2
        )
        pressure_error = float(
            np.max(
                np.abs(result.total_pressure_coefficient - closed_form),
                initial=0.0,
            )
        )
        cross_error = float(
            np.max(
                np.abs(
                    result.total_pressure_coefficient
                    - result.pressure_level_addition
                    - result.bernoulli_cross_term
                ),
                initial=0.0,
            )
        )
        force_error = abs(
            float(result.total_force_coefficient[1])
            - result.expected_lift_coefficient
        )
        zero_drag = abs(float(result.total_force_coefficient[0]))
        gauge_change = float(
            np.max(
                np.abs(
                    result.gauge_shifted_force_coefficient
                    - result.total_force_coefficient
                ),
                initial=0.0,
            )
        )
        lift_lost = abs(
            float(result.total_force_coefficient[1])
            - float(result.pressure_addition_force_coefficient[1])
        )
        maxima["pressure_identity_error"] = max(
            maxima["pressure_identity_error"], pressure_error
        )
        maxima["cross_term_identity_error"] = max(
            maxima["cross_term_identity_error"], cross_error
        )
        maxima["force_coefficient_error"] = max(
            maxima["force_coefficient_error"], force_error
        )
        maxima["zero_drag_error"] = max(
            maxima["zero_drag_error"], zero_drag
        )
        maxima["gauge_force_change"] = max(
            maxima["gauge_force_change"], gauge_change
        )
        minimum_lift_lost = min(minimum_lift_lost, lift_lost)
        cases[f"k={ratio:.2f}"] = {
            "circulation_velocity_ratio": ratio,
            "max_pressure_nonadditivity": float(
                np.max(
                    np.abs(
                        result.total_pressure_coefficient
                        - result.pressure_level_addition
                    ),
                    initial=0.0,
                )
            ),
            "expected_max_pressure_nonadditivity": 4.0 * abs(ratio),
            "total_force_coefficient": (
                result.total_force_coefficient.tolist()
            ),
            "expected_lift_coefficient": result.expected_lift_coefficient,
            "pressure_addition_force_coefficient": (
                result.pressure_addition_force_coefficient.tolist()
            ),
            "lift_lost_by_pressure_addition": lift_lost,
        }

    checks = {
        "total_pressure_matches_closed_form": (
            maxima["pressure_identity_error"]
            <= float(thresholds["max_pressure_identity_error"])
        ),
        "nonadditive_cross_term_matches_closed_form": (
            maxima["cross_term_identity_error"]
            <= float(thresholds["max_cross_term_identity_error"])
        ),
        "total_lift_matches_kutta_joukowski": (
            maxima["force_coefficient_error"]
            <= float(thresholds["max_force_coefficient_error"])
        ),
        "pressure_level_addition_loses_lift": (
            minimum_lift_lost
            > 100.0 * float(thresholds["max_force_coefficient_error"])
        ),
        "total_inviscid_drag_is_zero": (
            maxima["zero_drag_error"]
            <= float(thresholds["max_zero_drag_error"])
        ),
        "uniform_pressure_gauge_does_not_change_force": (
            maxima["gauge_force_change"]
            <= float(thresholds["max_gauge_force_change"])
        ),
    }
    return {
        "version": 1,
        "claim": contract["claim"],
        "canonical_problem": canonical["name"],
        "case_count": len(cases),
        "cases": cases,
        "maxima": maxima,
        "minimum_lift_lost_by_pressure_addition": minimum_lift_lost,
        "checks": checks,
        "all_pass": all(checks.values()),
        "decision": contract["required_outcome"],
        "scope_limit": contract["scope_limit"],
        "production_formula_changed": False,
    }


def main() -> int:
    contract = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    result = evaluate_contract(contract)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
