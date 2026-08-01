"""Frozen S3ai-v2 history enumeration and accounting.

This module contains no solver call.  It makes the 31 preregistered histories
and their shared tangent/zero/mixed-cube identities explicit so a runner
cannot silently omit, duplicate, or relabel a numerical family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class ReachablePressureCaseError(ValueError):
    """The frozen history enumeration is inconsistent."""


@dataclass(frozen=True)
class TangentConfiguration:
    """One positive epsilon magnitude, timestep, and coupled quadrature."""

    name: str
    epsilon: float
    timestep: float
    quadrature_order: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or not np.isfinite(self.epsilon)
            or self.epsilon <= 0.0
            or not np.isfinite(self.timestep)
            or self.timestep <= 0.0
            or not isinstance(self.quadrature_order, int)
            or self.quadrature_order < 2
        ):
            raise ReachablePressureCaseError(
                "invalid tangent configuration"
            )


@dataclass(frozen=True)
class HistoryCase:
    """One fresh S3e march including its zero compatible pre-step."""

    name: str
    configuration: str
    epsilon_signed: float
    timestep: float
    quadrature_order: int
    role: Literal["nominal_tangent", "zero_reference", "fresh_repeat"]

    def __post_init__(self) -> None:
        exact = 1.0 / float(self.timestep)
        if (
            not self.name
            or not self.configuration
            or not np.isfinite(self.epsilon_signed)
            or not np.isfinite(self.timestep)
            or self.timestep <= 0.0
            or abs(exact - round(exact))
            > 64.0 * np.finfo(float).eps * max(abs(exact), 1.0)
            or not isinstance(self.quadrature_order, int)
            or self.quadrature_order < 2
            or self.role
            not in {"nominal_tangent", "zero_reference", "fresh_repeat"}
        ):
            raise ReachablePressureCaseError("invalid history case")

    @property
    def measurement_steps(self) -> int:
        return int(round(1.0 / self.timestep))

    @property
    def marcher_steps(self) -> int:
        """Measurement steps plus exactly one ``[-dt,0]`` pre-step."""

        return self.measurement_steps + 1

    @property
    def half_full_solves(self) -> int:
        return 2 * self.marcher_steps

    @property
    def observed_stages(self) -> int:
        """Pre-step full plus every measured half/full stage."""

        return 1 + 2 * self.measurement_steps


def frozen_tangent_configurations() -> dict[str, TangentConfiguration]:
    """Return the seven axis and four mixed-cube configurations."""

    return {
        "A": TangentConfiguration("A", 0.0025, 0.0625, 12),
        "E2": TangentConfiguration("E2", 0.005, 0.0625, 12),
        "E4": TangentConfiguration("E4", 0.01, 0.0625, 12),
        "DT2": TangentConfiguration("DT2", 0.0025, 0.125, 12),
        "DT4": TangentConfiguration("DT4", 0.0025, 0.25, 12),
        "Q10": TangentConfiguration("Q10", 0.0025, 0.0625, 10),
        "Q8": TangentConfiguration("Q8", 0.0025, 0.0625, 8),
        "E2_DT2": TangentConfiguration(
            "E2_DT2", 0.005, 0.125, 12
        ),
        "E2_Q10": TangentConfiguration(
            "E2_Q10", 0.005, 0.0625, 10
        ),
        "DT2_Q10": TangentConfiguration(
            "DT2_Q10", 0.0025, 0.125, 10
        ),
        "E2_DT2_Q10": TangentConfiguration(
            "E2_DT2_Q10", 0.005, 0.125, 10
        ),
    }


def mixed_cube_configuration_names() -> dict[str, str]:
    """Map epsilon/timestep/quadrature bits to frozen configuration names."""

    return {
        "000": "A",
        "100": "E2",
        "010": "DT2",
        "001": "Q10",
        "110": "E2_DT2",
        "101": "E2_Q10",
        "011": "DT2_Q10",
        "111": "E2_DT2_Q10",
    }


def frozen_history_cases() -> tuple[HistoryCase, ...]:
    """Return all 31 fresh histories in deterministic order."""

    configurations = frozen_tangent_configurations()
    result: list[HistoryCase] = []
    for name, configuration in configurations.items():
        for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
            result.append(
                HistoryCase(
                    name=f"{name}_{suffix}",
                    configuration=name,
                    epsilon_signed=sign * configuration.epsilon,
                    timestep=configuration.timestep,
                    quadrature_order=configuration.quadrature_order,
                    role="nominal_tangent",
                )
            )

    zero_configurations = (
        ("Z_A", 0.0625, 12),
        ("Z_DT2", 0.125, 12),
        ("Z_DT4", 0.25, 12),
        ("Z_Q10", 0.0625, 10),
        ("Z_Q8", 0.0625, 8),
        ("Z_DT2_Q10", 0.125, 10),
    )
    for name, timestep, quadrature in zero_configurations:
        result.append(
            HistoryCase(
                name=name,
                configuration=name,
                epsilon_signed=0.0,
                timestep=timestep,
                quadrature_order=quadrature,
                role="zero_reference",
            )
        )

    anchor = configurations["A"]
    for epsilon, suffix in (
        (anchor.epsilon, "plus"),
        (-anchor.epsilon, "minus"),
        (0.0, "zero"),
    ):
        result.append(
            HistoryCase(
                name=f"REPEAT_A_{suffix}",
                configuration="REPEAT_A",
                epsilon_signed=epsilon,
                timestep=anchor.timestep,
                quadrature_order=anchor.quadrature_order,
                role="fresh_repeat",
            )
        )

    cases = tuple(result)
    if len(cases) != 31 or len({case.name for case in cases}) != 31:
        raise ReachablePressureCaseError(
            "frozen S3ai-v2 enumeration must contain 31 unique names"
        )
    return cases


def frozen_execution_accounting() -> dict[str, int | dict[str, int]]:
    """Return preregistered path, step, solve, and observer counts."""

    cases = frozen_history_cases()
    timestep_counts: dict[str, int] = {}
    for timestep in (0.0625, 0.125, 0.25):
        timestep_counts[str(timestep)] = sum(
            case.timestep == timestep for case in cases
        )
    return {
        "histories": len(cases),
        "nominal_signed_histories": sum(
            case.role == "nominal_tangent" for case in cases
        ),
        "zero_histories": sum(
            case.role == "zero_reference" for case in cases
        ),
        "fresh_repeat_histories": sum(
            case.role == "fresh_repeat" for case in cases
        ),
        "measurement_steps": sum(
            case.measurement_steps for case in cases
        ),
        "compatible_presteps": len(cases),
        "marcher_steps": sum(case.marcher_steps for case in cases),
        "half_full_solves": sum(
            case.half_full_solves for case in cases
        ),
        "observed_stages": sum(
            case.observed_stages for case in cases
        ),
        "histories_by_timestep": timestep_counts,
    }

