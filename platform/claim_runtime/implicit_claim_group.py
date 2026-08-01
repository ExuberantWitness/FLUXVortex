"""Named block-Newton solver for an implicit group inside the claim DAG.

The outer ClaimGraph remains acyclic.  A physically simultaneous N1--N2.6--N3
stage can later be represented by one atomic component whose internal state
and residual blocks are solved here.  This numerical module has no aerodynamic
closure and accepts no force target or LESP-amplitude residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np


class ImplicitClaimGroupError(RuntimeError):
    """Invalid, forbidden, rank-deficient or non-convergent implicit group."""


_FORBIDDEN_TOKENS = (
    "target_lift",
    "target_thrust",
    "force_target",
    "target_force",
    "lesp_amplitude",
    "regularization",
    "smoothing",
)


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ImplicitClaimGroupError(
            "state and residual block names must be non-empty strings"
        )
    canonical = name.strip()
    lower = canonical.lower()
    if any(token in lower for token in _FORBIDDEN_TOKENS):
        raise ImplicitClaimGroupError(
            f"forbidden implicit block name {canonical!r}"
        )
    return canonical


def _block_layout(name: str, values) -> Mapping[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ImplicitClaimGroupError(f"{name} must be a non-empty mapping")
    output = {}
    for raw_name, raw_size in values.items():
        block_name = _validate_name(raw_name)
        if block_name in output:
            raise ImplicitClaimGroupError(
                f"duplicate {name} block {block_name!r}"
            )
        if (
            not isinstance(raw_size, (int, np.integer))
            or int(raw_size) <= 0
        ):
            raise ImplicitClaimGroupError(
                f"{name} block sizes must be positive integers"
            )
        output[block_name] = int(raw_size)
    return MappingProxyType(output)


def _scales(layout: Mapping[str, int], values) -> Mapping[str, float]:
    if set(values) != set(layout):
        raise ImplicitClaimGroupError(
            "one physical scale is required for every residual block"
        )
    output = {}
    for name in layout:
        scale = float(values[name])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ImplicitClaimGroupError(
                f"residual scale for {name!r} must be positive and finite"
            )
        output[name] = scale
    return MappingProxyType(output)


@dataclass(frozen=True)
class ImplicitBlockResidual:
    size: int
    physical_scale: float
    max_absolute_residual: float
    max_normalized_residual: float


@dataclass(frozen=True)
class ImplicitIterationReport:
    iteration: int
    step_length: float
    jacobian_rank: int
    jacobian_condition_number: float
    max_normalized_residual: float
    block_residuals: Mapping[str, ImplicitBlockResidual]


@dataclass(frozen=True)
class ImplicitClaimGroupSolution:
    state: Mapping[str, np.ndarray]
    iterations: tuple[ImplicitIterationReport, ...]
    update_count: int
    max_normalized_residual: float
    passed: bool


class ImplicitClaimGroup:
    """Solve named nonlinear residual blocks with analytic block Jacobians."""

    def __init__(
        self,
        *,
        state_block_sizes: Mapping[str, int],
        residual_block_sizes: Mapping[str, int],
        residual_physical_scales: Mapping[str, float],
    ) -> None:
        self.state_block_sizes = _block_layout(
            "state_block_sizes",
            state_block_sizes,
        )
        self.residual_block_sizes = _block_layout(
            "residual_block_sizes",
            residual_block_sizes,
        )
        self.residual_physical_scales = _scales(
            self.residual_block_sizes,
            residual_physical_scales,
        )
        self._state_slices = self._slices(self.state_block_sizes)
        self._residual_slices = self._slices(self.residual_block_sizes)
        self.state_size = sum(self.state_block_sizes.values())
        self.residual_size = sum(self.residual_block_sizes.values())
        if self.residual_size < self.state_size:
            raise ImplicitClaimGroupError(
                "implicit group has fewer residual equations than states"
            )

    @staticmethod
    def _slices(layout: Mapping[str, int]) -> Mapping[str, slice]:
        output = {}
        start = 0
        for name, size in layout.items():
            output[name] = slice(start, start+size)
            start += size
        return MappingProxyType(output)

    def _pack_state(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(state) != set(self.state_block_sizes):
            raise ImplicitClaimGroupError(
                "initial/candidate state block names do not match the layout"
            )
        packed = np.empty(self.state_size, dtype=float)
        for name, size in self.state_block_sizes.items():
            value = np.asarray(state[name], dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ImplicitClaimGroupError(
                    f"state block {name!r} must have shape ({size},)"
                )
            packed[self._state_slices[name]] = value
        return packed

    def _unpack_state(self, packed: np.ndarray) -> Mapping[str, np.ndarray]:
        return MappingProxyType({
            name: packed[block_slice].copy()
            for name, block_slice in self._state_slices.items()
        })

    def _residual(
        self,
        packed_state: np.ndarray,
        function: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
    ) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
        state = self._unpack_state(packed_state)
        raw = function(state)
        if not isinstance(raw, Mapping) or set(raw) != set(
            self.residual_block_sizes
        ):
            raise ImplicitClaimGroupError(
                "residual function returned the wrong named blocks"
            )
        normalized = np.empty(self.residual_size, dtype=float)
        physical = {}
        for name, size in self.residual_block_sizes.items():
            value = np.asarray(raw[name], dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ImplicitClaimGroupError(
                    f"residual block {name!r} must have shape ({size},)"
                )
            physical[name] = value.copy()
            normalized[self._residual_slices[name]] = (
                value/self.residual_physical_scales[name]
            )
        return normalized, MappingProxyType(physical)

    def _jacobian(
        self,
        state: Mapping[str, np.ndarray],
        function: Callable[
            [Mapping[str, np.ndarray]],
            Mapping[tuple[str, str], np.ndarray],
        ],
    ) -> np.ndarray:
        raw = function(state)
        expected = {
            (residual, state_name)
            for residual in self.residual_block_sizes
            for state_name in self.state_block_sizes
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ImplicitClaimGroupError(
                "jacobian function must return every (residual,state) block"
            )
        matrix = np.empty(
            (self.residual_size, self.state_size),
            dtype=float,
        )
        for residual_name, residual_size in self.residual_block_sizes.items():
            row = self._residual_slices[residual_name]
            scale = self.residual_physical_scales[residual_name]
            for state_name, state_size in self.state_block_sizes.items():
                column = self._state_slices[state_name]
                value = np.asarray(
                    raw[(residual_name, state_name)],
                    dtype=float,
                )
                if (
                    value.shape != (residual_size, state_size)
                    or not np.all(np.isfinite(value))
                ):
                    raise ImplicitClaimGroupError(
                        f"jacobian block {(residual_name, state_name)} "
                        f"must have shape ({residual_size},{state_size})"
                    )
                matrix[row, column] = value/scale
        return matrix

    def _report(
        self,
        *,
        iteration: int,
        step_length: float,
        normalized: np.ndarray,
        physical: Mapping[str, np.ndarray],
        rank: int,
        condition: float,
    ) -> ImplicitIterationReport:
        blocks = {}
        for name, value in physical.items():
            maximum = float(np.max(np.abs(value), initial=0.0))
            blocks[name] = ImplicitBlockResidual(
                size=len(value),
                physical_scale=self.residual_physical_scales[name],
                max_absolute_residual=maximum,
                max_normalized_residual=(
                    maximum/self.residual_physical_scales[name]
                ),
            )
        return ImplicitIterationReport(
            iteration=iteration,
            step_length=float(step_length),
            jacobian_rank=int(rank),
            jacobian_condition_number=float(condition),
            max_normalized_residual=float(
                np.max(np.abs(normalized), initial=0.0)
            ),
            block_residuals=MappingProxyType(blocks),
        )

    def solve(
        self,
        *,
        initial_state: Mapping[str, np.ndarray],
        residual_function: Callable[
            [Mapping[str, np.ndarray]],
            Mapping[str, np.ndarray],
        ],
        jacobian_function: Callable[
            [Mapping[str, np.ndarray]],
            Mapping[tuple[str, str], np.ndarray],
        ],
        normalized_tolerance: float = 1.0e-10,
        maximum_updates: int = 20,
        maximum_backtracks: int = 16,
        minimum_step_length: float = 2.0**-16,
    ) -> ImplicitClaimGroupSolution:
        if (
            not np.isfinite(normalized_tolerance)
            or normalized_tolerance < 0.0
        ):
            raise ImplicitClaimGroupError(
                "normalized_tolerance must be finite and non-negative"
            )
        if maximum_updates <= 0 or maximum_backtracks < 0:
            raise ImplicitClaimGroupError(
                "invalid Newton update/backtracking limit"
            )
        if (
            not np.isfinite(minimum_step_length)
            or minimum_step_length <= 0.0
            or minimum_step_length > 1.0
        ):
            raise ImplicitClaimGroupError(
                "minimum_step_length must lie in (0,1]"
            )
        packed = self._pack_state(initial_state)
        normalized, physical = self._residual(packed, residual_function)
        reports = []
        for update in range(maximum_updates+1):
            maximum = float(
                np.max(np.abs(normalized), initial=0.0)
            )
            if maximum <= normalized_tolerance:
                reports.append(self._report(
                    iteration=update,
                    step_length=0.0,
                    normalized=normalized,
                    physical=physical,
                    rank=self.state_size,
                    condition=1.0,
                ))
                return ImplicitClaimGroupSolution(
                    state=self._unpack_state(packed),
                    iterations=tuple(reports),
                    update_count=update,
                    max_normalized_residual=maximum,
                    passed=True,
                )
            if update == maximum_updates:
                break
            state = self._unpack_state(packed)
            jacobian = self._jacobian(state, jacobian_function)
            _, singular, rank, _ = np.linalg.lstsq(
                jacobian,
                -normalized,
                rcond=None,
            )
            rank = int(rank)
            if rank != self.state_size:
                raise ImplicitClaimGroupError(
                    f"implicit Jacobian rank {rank}, expected {self.state_size}"
                )
            singular_values = np.linalg.svd(
                jacobian,
                compute_uv=False,
            )
            condition = (
                np.inf
                if singular_values[-1] <= 0.0
                else float(singular_values[0]/singular_values[-1])
            )
            delta, _, _, _ = np.linalg.lstsq(
                jacobian,
                -normalized,
                rcond=None,
            )
            accepted = False
            step_length = 1.0
            for _ in range(maximum_backtracks+1):
                candidate = packed+step_length*delta
                candidate_normalized, candidate_physical = self._residual(
                    candidate,
                    residual_function,
                )
                candidate_maximum = float(np.max(
                    np.abs(candidate_normalized),
                    initial=0.0,
                ))
                if candidate_maximum < maximum:
                    accepted = True
                    break
                step_length *= 0.5
                if step_length < minimum_step_length:
                    break
            reports.append(self._report(
                iteration=update,
                step_length=step_length if accepted else 0.0,
                normalized=normalized,
                physical=physical,
                rank=rank,
                condition=condition,
            ))
            if not accepted:
                raise ImplicitClaimGroupError(
                    "backtracking failed to reduce every-step maximum residual"
                )
            packed = candidate
            normalized = candidate_normalized
            physical = candidate_physical
        raise ImplicitClaimGroupError(
            f"implicit group did not converge in {maximum_updates} updates"
        )

