"""Chronological single-force adapter for the FluxV v5b shared wake.

The adapter owns exactly one :class:`HiratoLiveShadow` and advances it once
per structured geometry sample.  At each step it snapshots the previous bound
and LEV strengths, obtains the live report (including the pre-convection
TE/LE material sheets), and calls :func:`fluxv_v5b_surface_force` exactly once.
No LDVM, polar, impulse, or baseline force is added afterwards.

This is a synthetic-sequence integration surface, not a paper benchmark
authorization.  Any chronological, equation, material-circulation, or force
ledger failure raises before a result is returned.  Paper scoring remains
``blocked_not_scored`` until independent geometry/kinematics adapters and
convergence gates are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from claim_runtime.hirato_equations import HiratoEquationError
from claim_runtime.hirato_live_shadow import (
    HiratoLiveShadow,
    HiratoLiveStepReport,
    build_bound_lattice,
)

from .fluxv_v5b_force import (
    FluxVV5BForceError,
    FluxVV5BForceLedger,
    fluxv_v5b_surface_force,
)
from .fluxv_v5b_shared_wake import FluxVV5BSharedWakeConfig


SEQUENCE_STATUS = "synthetic_sequence_only"
PAPER_SCORING_STATUS = "blocked_not_scored"
FORCE_OWNER = "fluxv_v5b_force.fluxv_v5b_surface_force"


class FluxVV5BSequenceError(RuntimeError):
    """The chronological shared-wake/force transaction failed closed."""


def _readonly(value: Any, *, dtype: Any = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite_vector(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class FluxVV5BSequenceConfig:
    """Physical inputs shared by every chronological force observation."""

    shared_wake: FluxVV5BSharedWakeConfig
    density: float
    moment_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.shared_wake, FluxVV5BSharedWakeConfig):
            raise TypeError("shared_wake must be FluxVV5BSharedWakeConfig")
        density = float(self.density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("density must be finite and positive")
        origin = _finite_vector("moment_origin", self.moment_origin, (3,))
        object.__setattr__(self, "density", density)
        object.__setattr__(
            self,
            "moment_origin",
            tuple(float(value) for value in origin),
        )

    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "shared_wake": self.shared_wake.manifest(),
                "density_kg_m3": self.density,
                "moment_origin_m": list(self.moment_origin),
                "force_owner": FORCE_OWNER,
                "force_evaluations_per_step": 1,
                "second_force_source": "forbidden",
                "chronology": (
                    "snapshot previous bound/LEV; one live-shadow step; one "
                    "pre-convection surface-pressure force; then state commit"
                ),
                "status": SEQUENCE_STATUS,
                "paper_scoring_status": PAPER_SCORING_STATUS,
                "observation_fit": "none",
            }
        )


@dataclass(frozen=True)
class FluxVV5BChronologicalStep:
    """Immutable witness for one committed live-shadow/force transaction."""

    step: int
    phase: float
    previous_bound_gamma: np.ndarray
    previous_gamma_lev: np.ndarray
    pre_convection_report: HiratoLiveStepReport
    force_ledger: FluxVV5BForceLedger
    eq9_max_abs_residual: float
    lesp_max_abs_residual: float
    convection_max_abs_residual: float
    material_gamma_max_abs_change: float
    pristine_no_lev: bool
    no_lev_exact_reduction_passed: bool


@dataclass(frozen=True)
class FluxVV5BSequenceGuards:
    expected_steps: int
    committed_steps: int
    force_ledger_count: int
    max_eq9_residual: float
    max_lesp_residual: float
    max_convection_residual: float
    max_material_gamma_change: float
    max_velocity_ledger_residual: float
    max_pressure_ledger_residual: float
    max_force_channel_residual: float
    no_lev_steps: int
    no_lev_exact_reduction_passed: bool
    all_finite: bool
    unique_force_owner: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.committed_steps == self.expected_steps
            and self.force_ledger_count == self.expected_steps
            and self.max_eq9_residual <= 1.0e-12
            and self.max_lesp_residual <= 1.0e-12
            and self.max_convection_residual <= 1.0e-12
            and self.max_material_gamma_change == 0.0
            and self.max_velocity_ledger_residual <= 1.0e-12
            and self.max_pressure_ledger_residual <= 1.0e-12
            and self.max_force_channel_residual <= 1.0e-12
            and self.no_lev_exact_reduction_passed
            and self.all_finite
            and self.unique_force_owner
        )


@dataclass(frozen=True)
class FluxVV5BSequenceResult:
    """Phase-resolved force/moment history and its chronological witnesses."""

    phase: np.ndarray
    force_history_n: np.ndarray
    moment_history_nm: np.ndarray
    steps: tuple[FluxVV5BChronologicalStep, ...]
    guards: FluxVV5BSequenceGuards
    manifest: Mapping[str, Any]
    status: str = SEQUENCE_STATUS
    paper_scoring_status: str = PAPER_SCORING_STATUS


def _material_gamma(snapshot: Any, prefix: str) -> dict[tuple[str, int], float]:
    return {
        (prefix, int(ring_id)): float(gamma)
        for ring_id, gamma in zip(snapshot.ring_id, snapshot.gamma, strict=True)
    }


def _retained_gamma_change(
    previous: Mapping[tuple[str, int], float],
    current: Mapping[tuple[str, int], float],
) -> tuple[float, list[str]]:
    missing = [key for key in previous if key not in current]
    changes = [
        abs(float(current[key] - value))
        for key, value in previous.items()
        if key in current
    ]
    return max(changes, default=0.0), [
        f"{prefix}:{ring_id}" for prefix, ring_id in sorted(missing)
    ]


def _validate_histories(
    config: FluxVV5BSequenceConfig,
    corners_history: Any,
    corner_velocity_history: Any,
    phase: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = config.shared_wake
    expected_tail = (shared.nc + 1, shared.ns + 1, 3)
    corners = np.asarray(corners_history, dtype=float)
    velocity = np.asarray(corner_velocity_history, dtype=float)
    if corners.ndim != 4 or corners.shape[1:] != expected_tail or corners.shape[0] < 1:
        raise ValueError(
            "corners_history must have shape "
            f"(steps,{expected_tail[0]},{expected_tail[1]},3)"
        )
    if velocity.shape != corners.shape:
        raise ValueError("corner_velocity_history must exactly match corners_history")
    if np.any(~np.isfinite(corners)) or np.any(~np.isfinite(velocity)):
        raise ValueError("structured geometry histories must be finite")
    steps = corners.shape[0]
    if phase is None:
        phase_array = np.arange(steps, dtype=float) / steps
    else:
        phase_array = np.asarray(phase, dtype=float)
        if phase_array.shape != (steps,) or np.any(~np.isfinite(phase_array)):
            raise ValueError("phase must be a finite one-dimensional step history")
        if np.any((phase_array < 0.0) | (phase_array >= 1.0)):
            raise ValueError("phase samples must lie in [0,1)")
        if np.any(np.diff(phase_array) <= 0.0):
            raise ValueError("phase samples must be strictly increasing")
    return corners, velocity, phase_array


def run_fluxv_v5b_force_sequence(
    config: FluxVV5BSequenceConfig,
    corners_history: Any,
    corner_velocity_history: Any,
    *,
    phase: Any | None = None,
) -> FluxVV5BSequenceResult:
    """Run one fail-closed chronological shared-wake/force sequence."""

    if not isinstance(config, FluxVV5BSequenceConfig):
        raise TypeError("config must be FluxVV5BSequenceConfig")
    corners, velocity, phase_array = _validate_histories(
        config,
        corners_history,
        corner_velocity_history,
        phase,
    )
    shared = config.shared_wake
    shadow = HiratoLiveShadow(
        nc=shared.nc,
        ns=shared.ns,
        u_infinity=np.asarray(shared.u_infinity),
        dt=shared.dt,
        lesp_crit=shared.lesp_crit,
        core_radius=shared.core_radius,
        mirror_symmetry=shared.mirror_symmetry,
        velocity_backend=shared.velocity_backend,
        warp_device=shared.warp_device,
    )
    previous_bound = np.zeros(shared.nc * shared.ns, dtype=float)
    previous_lev = np.zeros(shared.ns, dtype=float)
    material_reference: dict[tuple[str, int], float] = {}
    committed: list[FluxVV5BChronologicalStep] = []

    try:
        for step, (geometry, geometry_velocity, phase_value) in enumerate(
            zip(corners, velocity, phase_array, strict=True)
        ):
            # Snapshot the exact Eq.17/Eq.9 history before any current-step solve.
            previous_bound_witness = previous_bound.copy()
            previous_lev_witness = previous_lev.copy()
            before = {
                **_material_gamma(shadow.tev.snapshot(), "tev"),
                **_material_gamma(shadow.lev.snapshot(), "lev"),
            }
            prior_change, prior_missing = _retained_gamma_change(
                material_reference,
                before,
            )
            if prior_change != 0.0 or prior_missing:
                raise FluxVV5BSequenceError(
                    "material circulation changed between chronological steps"
                )

            lattice = build_bound_lattice(
                geometry,
                geometry_velocity,
                nc=shared.nc,
                ns=shared.ns,
            )
            report = shadow.step(
                step=step,
                corners=geometry,
                corner_velocity=geometry_velocity,
            )
            pre_convection = {
                **_material_gamma(report.tev_pre_convection, "tev"),
                **_material_gamma(report.lev_pre_convection, "lev"),
            }
            post_convection = {
                **_material_gamma(report.tev_post_convection, "tev"),
                **_material_gamma(report.lev_post_convection, "lev"),
            }
            birth_change, birth_missing = _retained_gamma_change(
                before,
                pre_convection,
            )
            convection_change, convection_missing = _retained_gamma_change(
                pre_convection,
                post_convection,
            )
            material_change = max(prior_change, birth_change, convection_change)
            if material_change != 0.0 or birth_missing or convection_missing:
                raise FluxVV5BSequenceError(
                    "material circulation changed during birth/remesh/convection"
                )

            eq9_residual = float(np.max(np.abs(report.eq9_residual), initial=0.0))
            lesp_residual = float(
                np.max(np.abs(report.lesp_active_residual), initial=0.0)
            )
            convection_residual = float(report.convection_ledger_max_abs_residual)
            if (
                eq9_residual > 1.0e-12
                or lesp_residual > 1.0e-12
                or convection_residual > 1.0e-12
            ):
                raise FluxVV5BSequenceError(
                    "shared-wake equation or convection ledger did not close"
                )

            # Sole force evaluation.  It consumes pre-convection TE/LE sheets
            # from this report and the saved previous states above.
            ledger = fluxv_v5b_surface_force(
                density=config.density,
                dt=shared.dt,
                nc=shared.nc,
                ns=shared.ns,
                lattice=lattice,
                report=report,
                previous_bound_gamma=previous_bound_witness,
                previous_gamma_lev=previous_lev_witness,
                u_infinity=np.asarray(shared.u_infinity),
                core_radius=shared.core_radius,
                mirror_symmetry=shared.mirror_symmetry,
                moment_origin=np.asarray(config.moment_origin),
            )
            if not ledger.guards.passed:
                raise FluxVV5BSequenceError("single-owner force ledger failed")
            pristine_no_lev = bool(
                len(report.lev_pre_convection.rings) == 0
                and not np.any(report.gamma_lev)
                and not np.any(previous_lev_witness)
            )
            reduction_passed = bool(
                not pristine_no_lev
                or (
                    ledger.guards.no_lev_exact_reduction_required
                    and ledger.guards.no_lev_exact_reduction_passed
                )
            )
            if not reduction_passed:
                raise FluxVV5BSequenceError("pristine no-LEV reduction failed")

            committed.append(
                FluxVV5BChronologicalStep(
                    step=step,
                    phase=float(phase_value),
                    previous_bound_gamma=_readonly(previous_bound_witness),
                    previous_gamma_lev=_readonly(previous_lev_witness),
                    pre_convection_report=report,
                    force_ledger=ledger,
                    eq9_max_abs_residual=eq9_residual,
                    lesp_max_abs_residual=lesp_residual,
                    convection_max_abs_residual=convection_residual,
                    material_gamma_max_abs_change=material_change,
                    pristine_no_lev=pristine_no_lev,
                    no_lev_exact_reduction_passed=reduction_passed,
                )
            )
            previous_bound = np.asarray(report.bound_gamma, dtype=float).copy()
            previous_lev = np.asarray(report.gamma_lev, dtype=float).copy()
            material_reference = post_convection
    except (HiratoEquationError, FluxVV5BForceError, np.linalg.LinAlgError) as error:
        raise FluxVV5BSequenceError(
            f"chronological v5b transaction failed at step {len(committed)}: {error}"
        ) from error

    force_history = np.stack([item.force_ledger.total_force for item in committed])
    moment_history = np.stack([item.force_ledger.total_moment for item in committed])
    force_guards = [item.force_ledger.guards for item in committed]
    no_lev_steps = [item for item in committed if item.pristine_no_lev]
    guards = FluxVV5BSequenceGuards(
        expected_steps=len(phase_array),
        committed_steps=len(committed),
        force_ledger_count=len(committed),
        max_eq9_residual=max(
            (item.eq9_max_abs_residual for item in committed),
            default=0.0,
        ),
        max_lesp_residual=max(
            (item.lesp_max_abs_residual for item in committed),
            default=0.0,
        ),
        max_convection_residual=max(
            (item.convection_max_abs_residual for item in committed),
            default=0.0,
        ),
        max_material_gamma_change=max(
            (item.material_gamma_max_abs_change for item in committed),
            default=0.0,
        ),
        max_velocity_ledger_residual=max(
            (guard.velocity_ledger_residual for guard in force_guards),
            default=0.0,
        ),
        max_pressure_ledger_residual=max(
            (guard.pressure_ledger_residual for guard in force_guards),
            default=0.0,
        ),
        max_force_channel_residual=max(
            (guard.force_channel_residual for guard in force_guards),
            default=0.0,
        ),
        no_lev_steps=len(no_lev_steps),
        no_lev_exact_reduction_passed=all(
            item.no_lev_exact_reduction_passed for item in no_lev_steps
        ),
        all_finite=bool(
            np.all(np.isfinite(force_history)) and np.all(np.isfinite(moment_history))
        ),
        unique_force_owner=True,
    )
    if not guards.passed:
        raise FluxVV5BSequenceError(f"sequence guards failed: {guards}")
    return FluxVV5BSequenceResult(
        phase=_readonly(phase_array),
        force_history_n=_readonly(force_history),
        moment_history_nm=_readonly(moment_history),
        steps=tuple(committed),
        guards=guards,
        manifest=config.manifest(),
    )


__all__ = [
    "FORCE_OWNER",
    "PAPER_SCORING_STATUS",
    "SEQUENCE_STATUS",
    "FluxVV5BChronologicalStep",
    "FluxVV5BSequenceConfig",
    "FluxVV5BSequenceError",
    "FluxVV5BSequenceGuards",
    "FluxVV5BSequenceResult",
    "run_fluxv_v5b_force_sequence",
]
