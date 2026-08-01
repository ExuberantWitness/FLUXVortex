"""No-force algebraic ledger for the N1--DDE global conservation solve.

This module freezes only the *assembly semantics* of a future
predictor--corrector.  Every unknown has circulation units (m^2/s).  Equation
blocks retain their physical residual units and are normalized solely by
``U_ref`` or ``U_ref*L_ref`` before a least-squares solve.  There are no
per-block weights, regularizers, smoothing terms, pressure targets, or force
targets.

LESP is deliberately absent from the allowed equation blocks.  It may open
or close a shedding topology before this system is assembled, but it cannot
be used here as a stripwise amplitude equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


class GlobalConservationError(ValueError):
    """Invalid, rank-deficient, or forbidden conservation system."""


_BLOCK_DIMENSIONS = {
    "no_penetration": "velocity",
    "trace_continuity": "circulation",
    "vorticity_compatibility": "velocity",
    "material_kelvin": "circulation",
    "kutta_interface": "circulation",
    "mirror_symmetry": "circulation",
    "free_edge": "circulation",
}


def _finite(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise GlobalConservationError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise GlobalConservationError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class ConservationEquationBlock:
    """One named physical residual block."""

    name: str
    matrix: np.ndarray
    rhs: np.ndarray
    residual_dimension: str


@dataclass(frozen=True)
class ConservationBlockReport:
    rows: int
    residual_dimension: str
    physical_scale: float
    max_abs_residual: float
    max_abs_normalized_residual: float


@dataclass(frozen=True)
class GlobalConservationSolution:
    values: np.ndarray
    values_by_name: Mapping[str, float]
    rank: int
    unknown_count: int
    equation_count: int
    condition_number: float
    max_abs_normalized_residual: float
    block_reports: Mapping[str, ConservationBlockReport]
    passed: bool


class GlobalConservationSystem:
    """Assemble and solve named conservation equations without tuneable weights."""

    def __init__(
        self,
        unknown_names,
        *,
        velocity_reference: float,
        length_reference: float,
    ) -> None:
        names = tuple(unknown_names)
        if not names or any(
            not isinstance(name, str) or not name.strip()
            for name in names
        ):
            raise GlobalConservationError(
                "unknown_names must contain non-empty strings"
            )
        names = tuple(name.strip() for name in names)
        if len(set(names)) != len(names):
            raise GlobalConservationError("unknown_names must be unique")
        if (
            not np.isfinite(velocity_reference)
            or velocity_reference <= 0.0
            or not np.isfinite(length_reference)
            or length_reference <= 0.0
        ):
            raise GlobalConservationError(
                "velocity_reference and length_reference must be positive"
            )
        self.unknown_names = names
        self.velocity_reference = float(velocity_reference)
        self.length_reference = float(length_reference)
        self._blocks: dict[str, ConservationEquationBlock] = {}

    @property
    def circulation_reference(self) -> float:
        return self.velocity_reference * self.length_reference

    def add_block(self, name: str, matrix, rhs) -> None:
        """Add one physical equation family.

        The fixed name determines the residual dimension.  In particular,
        names containing an LESP amplitude constraint are not accepted.
        """
        if not isinstance(name, str):
            raise GlobalConservationError("block name must be a string")
        canonical = name.strip()
        if canonical not in _BLOCK_DIMENSIONS:
            raise GlobalConservationError(
                f"unknown/forbidden conservation block {canonical!r}"
            )
        if canonical in self._blocks:
            raise GlobalConservationError(
                f"duplicate conservation block {canonical!r}"
            )
        operator = _finite(f"{canonical}.matrix", matrix, ndim=2)
        target = _finite(f"{canonical}.rhs", rhs, ndim=1)
        if operator.shape != (len(target), len(self.unknown_names)):
            raise GlobalConservationError(
                f"{canonical} matrix must have shape "
                f"({len(target)},{len(self.unknown_names)})"
            )
        if len(target) == 0:
            raise GlobalConservationError(
                f"{canonical} must contain at least one equation"
            )
        self._blocks[canonical] = ConservationEquationBlock(
            name=canonical,
            matrix=operator.copy(),
            rhs=target.copy(),
            residual_dimension=_BLOCK_DIMENSIONS[canonical],
        )

    def _scale(self, block: ConservationEquationBlock) -> float:
        if block.residual_dimension == "velocity":
            return self.velocity_reference
        return self.circulation_reference

    def solve(
        self,
        *,
        normalized_tolerance: float = 1.0e-10,
        require_full_column_rank: bool = True,
    ) -> GlobalConservationSolution:
        if not self._blocks:
            raise GlobalConservationError(
                "at least one conservation block is required"
            )
        if (
            not np.isfinite(normalized_tolerance)
            or normalized_tolerance < 0.0
        ):
            raise GlobalConservationError(
                "normalized_tolerance must be finite and non-negative"
            )
        matrices = []
        targets = []
        for block in self._blocks.values():
            scale = self._scale(block)
            matrices.append(block.matrix / scale)
            targets.append(block.rhs / scale)
        matrix = np.vstack(matrices)
        rhs = np.concatenate(targets)
        values, _, rank, singular = np.linalg.lstsq(
            matrix,
            rhs,
            rcond=None,
        )
        unknown_count = len(self.unknown_names)
        if require_full_column_rank and rank != unknown_count:
            raise GlobalConservationError(
                f"global conservation system has rank {rank}, "
                f"expected {unknown_count}"
            )
        if singular.size == 0 or singular[-1] <= 0.0:
            condition = np.inf
        else:
            condition = float(singular[0] / singular[-1])
        reports = {}
        maximum = 0.0
        for name, block in self._blocks.items():
            residual = block.matrix @ values - block.rhs
            scale = self._scale(block)
            physical = float(
                np.max(np.abs(residual), initial=0.0)
            )
            normalized = physical / scale
            maximum = max(maximum, normalized)
            reports[name] = ConservationBlockReport(
                rows=len(block.rhs),
                residual_dimension=block.residual_dimension,
                physical_scale=scale,
                max_abs_residual=physical,
                max_abs_normalized_residual=normalized,
            )
        return GlobalConservationSolution(
            values=values,
            values_by_name=MappingProxyType(
                dict(zip(self.unknown_names, values))
            ),
            rank=int(rank),
            unknown_count=unknown_count,
            equation_count=len(rhs),
            condition_number=condition,
            max_abs_normalized_residual=maximum,
            block_reports=MappingProxyType(reports),
            passed=bool(
                rank == unknown_count
                and maximum <= normalized_tolerance
            ),
        )

