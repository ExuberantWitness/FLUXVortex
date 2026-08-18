"""Isolated no-force shared TE/LE material-wake core for FluxV v5b.

This module wraps the existing equation-audited Hirato live shadow.  It lets
the same bound AIC see both trailing- and leading-edge material wake rings,
but deliberately exports neither pressure nor aerodynamic force.  Until a
single conservative pressure/force owner is implemented, all benchmark
scoring through this module is fail-closed as ``blocked_not_scored``.

The pristine dispatcher is intentionally simple: if no shared-wake/LEV event
is requested it calls and directly returns the parent callback *before* a
configuration or wake state is constructed.  This preserves the exact parent
object, including identity and floating-point bit patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import numpy as np

from claim_runtime.hirato_equations import HiratoEquationError
from claim_runtime.hirato_live_shadow import (
    HiratoLiveShadow,
    build_bound_lattice,
)


FORCE_COUPLING = "not_implemented"
SCORING_STATUS = "blocked_not_scored"
CORE_STATUS = "no_force_diagnostic_only"

_T = TypeVar("_T")


@dataclass(frozen=True)
class FluxVV5BSharedWakeConfig:
    """Source/numerics configuration for the isolated Hirato state machine."""

    nc: int
    ns: int
    u_infinity: tuple[float, float, float]
    dt: float
    lesp_crit: float
    core_radius: float
    mirror_symmetry: bool = False
    velocity_backend: str = "numpy"
    warp_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.nc <= 0 or self.ns <= 0:
            raise ValueError("nc and ns must be positive")
        velocity = np.asarray(self.u_infinity, dtype=float)
        if velocity.shape != (3,) or np.any(~np.isfinite(velocity)):
            raise ValueError("u_infinity must be a finite three-vector")
        if np.linalg.norm(velocity) <= 0.0:
            raise ValueError("u_infinity must be nonzero")
        object.__setattr__(
            self,
            "u_infinity",
            tuple(float(value) for value in velocity),
        )
        for name in ("dt", "lesp_crit", "core_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.velocity_backend not in ("numpy", "warp"):
            raise ValueError("velocity_backend must be numpy or warp")
        if not isinstance(self.warp_device, str) or not self.warp_device:
            raise ValueError("warp_device must be a nonempty string")

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "equation_core": "claim_runtime.hirato_live_shadow",
            "equations": ["Hirato Eq.6", "Eq.7", "Eq.9", "Eq.24"],
            "bound_aic": "frozen N1-compatible no-through solve",
            "wake_owners": ["material_TEV", "material_LEV"],
            "force_coupling": FORCE_COUPLING,
            "pressure_coupling": FORCE_COUPLING,
            "status": CORE_STATUS,
            "scoring_status": SCORING_STATUS,
            "observation_fit": "none",
        }


def _snapshot_gamma(snapshot: Any, prefix: str) -> dict[tuple[str, int], float]:
    return {
        (prefix, int(ring_id)): float(gamma)
        for ring_id, gamma in zip(snapshot.ring_id, snapshot.gamma, strict=True)
    }


def _compare_material_gamma(
    before: dict[tuple[str, int], float],
    after: dict[tuple[str, int], float],
) -> dict[str, Any]:
    missing = sorted(key for key in before if key not in after)
    changes = {
        key: float(after[key] - value) for key, value in before.items() if key in after
    }
    max_abs = max((abs(value) for value in changes.values()), default=0.0)
    return {
        "max_abs_change": float(max_abs),
        "missing_ids": [f"{prefix}:{ring_id}" for prefix, ring_id in missing],
        "retained_count": int(len(changes)),
        "passed": not missing and max_abs == 0.0,
    }


def _maximum_abs(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _birth_observables(
    report: Any,
    lattice: Any,
    *,
    step: int,
    u_infinity_speed: float,
    dt: float,
) -> dict[str, Any]:
    tev_mask = report.tev_pre_convection.birth_step == step
    lev_mask = report.lev_pre_convection.birth_step == step
    tev_rings = report.tev_pre_convection.rings[tev_mask]
    tev_strips = report.tev_pre_convection.strip[tev_mask]
    tev_gamma = report.tev_pre_convection.gamma[tev_mask]
    lev_rings = report.lev_pre_convection.rings[lev_mask]
    lev_strips = report.lev_pre_convection.strip[lev_mask]
    lev_gamma = report.lev_pre_convection.gamma[lev_mask]

    tev_ratio: list[float] = []
    for ring, strip in zip(tev_rings, tev_strips, strict=True):
        edge = lattice.trailing_edges[int(strip)]
        tev_ratio.extend(
            np.linalg.norm(ring[:2] - edge, axis=1) / (u_infinity_speed * dt)
        )
    lev_ratio: list[float] = []
    for ring, strip in zip(lev_rings, lev_strips, strict=True):
        edge = lattice.leading_edges[int(strip)]
        lev_ratio.extend(
            (
                np.linalg.norm(
                    ring[[3, 2]] - edge,
                    axis=1,
                )
                / (u_infinity_speed * dt)
            )
        )

    normalized_gamma: list[float] = []
    for gamma, strip in zip(tev_gamma, tev_strips, strict=True):
        normalized_gamma.append(
            abs(float(gamma)) / (u_infinity_speed * lattice.chord[int(strip)])
        )
    for gamma, strip in zip(lev_gamma, lev_strips, strict=True):
        normalized_gamma.append(
            abs(float(gamma)) / (u_infinity_speed * lattice.chord[int(strip)])
        )
    all_gamma = np.concatenate((tev_gamma, lev_gamma))
    return {
        "new_tev_gamma": tev_gamma.copy(),
        "new_lev_gamma": lev_gamma.copy(),
        "new_tev_count": int(tev_gamma.size),
        "new_lev_count": int(lev_gamma.size),
        "birth_gamma_max_abs": _maximum_abs(all_gamma),
        "birth_gamma_over_Uc_max_abs": max(normalized_gamma, default=0.0),
        "tev_birth_track_ratio_over_Udt": np.asarray(tev_ratio, dtype=float),
        "tev_birth_track_ratio_max_abs_error_from_0p3": (
            _maximum_abs(np.asarray(tev_ratio) - 0.3) if tev_ratio else 0.0
        ),
        "lev_birth_displacement_over_Udt": np.asarray(lev_ratio, dtype=float),
        "birth_limit_scope": (
            "single-resolution observables only; use birth_limit_diagnostic "
            "across distinct dt before promotion"
        ),
    }


class FluxVV5BSharedWakeCore:
    """No-force secondary core using one bound solve and shared TE/LE induction."""

    def __init__(self, config: FluxVV5BSharedWakeConfig):
        if not isinstance(config, FluxVV5BSharedWakeConfig):
            raise TypeError("config must be FluxVV5BSharedWakeConfig")
        self.config = config
        self._shadow = HiratoLiveShadow(
            nc=config.nc,
            ns=config.ns,
            u_infinity=np.asarray(config.u_infinity),
            dt=config.dt,
            lesp_crit=config.lesp_crit,
            core_radius=config.core_radius,
            mirror_symmetry=config.mirror_symmetry,
            velocity_backend=config.velocity_backend,
            warp_device=config.warp_device,
        )
        self._next_step = 0
        self._material_reference: dict[tuple[str, int], float] = {}

    def _current_material_gamma(self) -> dict[tuple[str, int], float]:
        return {
            **_snapshot_gamma(self._shadow.tev.snapshot(), "tev"),
            **_snapshot_gamma(self._shadow.lev.snapshot(), "lev"),
        }

    def step(
        self,
        corners: Any,
        corner_velocity: Any,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Advance one chronological shared-wake step and return only diagnostics."""

        step_id = self._next_step if step is None else int(step)
        if step_id != self._next_step:
            raise HiratoEquationError(
                f"expected v5b step {self._next_step}, got {step_id}"
            )
        lattice = build_bound_lattice(
            corners,
            corner_velocity,
            nc=self.config.nc,
            ns=self.config.ns,
        )
        before = self._current_material_gamma()
        historical_check = _compare_material_gamma(
            self._material_reference,
            before,
        )
        report = self._shadow.step(
            step=step_id,
            corners=corners,
            corner_velocity=corner_velocity,
        )
        pre_convection = {
            **_snapshot_gamma(report.tev_pre_convection, "tev"),
            **_snapshot_gamma(report.lev_pre_convection, "lev"),
        }
        after = {
            **_snapshot_gamma(report.tev_post_convection, "tev"),
            **_snapshot_gamma(report.lev_post_convection, "lev"),
        }
        retention_check = _compare_material_gamma(before, pre_convection)
        convection_check = _compare_material_gamma(pre_convection, after)
        material_max = max(
            historical_check["max_abs_change"],
            retention_check["max_abs_change"],
            convection_check["max_abs_change"],
        )
        missing = sorted(
            {
                *historical_check["missing_ids"],
                *retention_check["missing_ids"],
                *convection_check["missing_ids"],
            }
        )
        material_passed = (
            historical_check["passed"]
            and retention_check["passed"]
            and convection_check["passed"]
        )
        self._material_reference = after.copy()
        self._next_step += 1

        eq9_max = _maximum_abs(report.eq9_residual)
        lesp_max = _maximum_abs(report.lesp_active_residual)
        birth = _birth_observables(
            report,
            lattice,
            step=step_id,
            u_infinity_speed=float(np.linalg.norm(np.asarray(self.config.u_infinity))),
            dt=self.config.dt,
        )
        return {
            "step": step_id,
            "active": report.active.copy(),
            "new_sheet": report.new_sheet.copy(),
            "a0_event": report.a0_event.copy(),
            "a0_post": report.a0_post.copy(),
            "bound_gamma": report.bound_gamma.copy(),
            "gamma_tev_eq9": report.gamma_tev.copy(),
            "gamma_lev_lesp": report.gamma_lev.copy(),
            "eq9_residual": report.eq9_residual.copy(),
            "eq9_max_abs_residual": eq9_max,
            "kelvin_residual": report.eq9_residual.copy(),
            "kelvin_max_abs_residual": eq9_max,
            "kelvin_scope": (
                "Hirato Eq.9 per-strip TE birth identity; not a claimed common "
                "UVLM/LDVM circulation system"
            ),
            "lesp_active_residual": report.lesp_active_residual.copy(),
            "lesp_max_abs_residual": lesp_max,
            "lesp_condition_number": report.lesp_condition_number,
            "material_gamma_max_abs_change": float(material_max),
            "material_gamma_missing_ids": missing,
            "material_gamma_immutable": material_passed,
            "material_gamma_checks": {
                "between_steps": historical_check,
                "birth_and_remesh": retention_check,
                "eq24_convection": convection_check,
            },
            "convection_ledger_max_abs_residual": (
                report.convection_ledger_max_abs_residual
            ),
            "tev_ring_count": int(len(report.tev_post_convection.rings)),
            "lev_ring_count": int(len(report.lev_post_convection.rings)),
            "tev_bootstrap_vertices": report.tev_bootstrap_vertices,
            "lev_bootstrap_vertices": report.lev_bootstrap_vertices,
            **birth,
            "force_coupling": FORCE_COUPLING,
            "status": CORE_STATUS,
            "scoring_status": SCORING_STATUS,
        }

    def run_sequence(
        self,
        corners_history: Iterable[Any],
        corner_velocity_history: Iterable[Any],
    ) -> dict[str, Any]:
        """Advance aligned geometry/velocity histories with no force output."""

        corners = list(corners_history)
        velocity = list(corner_velocity_history)
        if not corners or len(corners) != len(velocity):
            raise ValueError("aligned nonempty corner histories are required")
        reports = [
            self.step(geometry, rate)
            for geometry, rate in zip(corners, velocity, strict=True)
        ]
        return {
            "steps": reports,
            "step_count": len(reports),
            "max_eq9_residual": max(item["eq9_max_abs_residual"] for item in reports),
            "max_kelvin_residual": max(
                item["kelvin_max_abs_residual"] for item in reports
            ),
            "max_lesp_residual": max(item["lesp_max_abs_residual"] for item in reports),
            "max_material_gamma_change": max(
                item["material_gamma_max_abs_change"] for item in reports
            ),
            "material_gamma_immutable": all(
                item["material_gamma_immutable"] for item in reports
            ),
            "birth_gamma_max_abs": max(item["birth_gamma_max_abs"] for item in reports),
            "force_coupling": FORCE_COUPLING,
            "status": CORE_STATUS,
            "scoring_status": SCORING_STATUS,
            "config": self.config.manifest(),
        }


def dispatch_v5b_or_parent(
    enable_shared_wake: bool,
    parent_callback: Callable[[], _T],
    *,
    config: FluxVV5BSharedWakeConfig | None = None,
    corners_history: Iterable[Any] | None = None,
    corner_velocity_history: Iterable[Any] | None = None,
) -> _T | dict[str, Any]:
    """Return the pristine parent directly unless shared-wake work is requested."""

    if not isinstance(enable_shared_wake, (bool, np.bool_)):
        raise TypeError("enable_shared_wake must be boolean")
    if not callable(parent_callback):
        raise TypeError("parent_callback must be callable")
    if not bool(enable_shared_wake):
        # Do not validate config/history or create a secondary state here.
        return parent_callback()
    if config is None or corners_history is None or corner_velocity_history is None:
        raise ValueError("enabled shared wake requires config and aligned histories")
    core = FluxVV5BSharedWakeCore(config)
    return core.run_sequence(corners_history, corner_velocity_history)


def birth_limit_diagnostic(
    delta_time_s: Iterable[float],
    birth_gamma_max_abs_m2_s: Iterable[float],
) -> dict[str, Any]:
    """Fit the nonzero material-birth limit ``|Gamma_birth| ~ dt**p``.

    At least three distinct, positive refinement levels are required.  The
    routine is a diagnostic only: it reports whether the fitted exponent is
    positive and every consecutive local order is positive, but it does not
    authorize pressure/force coupling.
    """

    dt = np.asarray(tuple(delta_time_s), dtype=float)
    gamma = np.asarray(tuple(birth_gamma_max_abs_m2_s), dtype=float)
    if dt.ndim != 1 or gamma.shape != dt.shape or dt.size < 3:
        raise ValueError("at least three aligned birth-refinement levels are required")
    if (
        np.any(~np.isfinite(dt))
        or np.any(~np.isfinite(gamma))
        or np.any(dt <= 0.0)
        or np.any(gamma <= 0.0)
        or len(np.unique(dt)) != dt.size
    ):
        raise ValueError("birth-limit levels must be positive, finite, and distinct")
    order = np.argsort(dt)[::-1]
    dt = dt[order]
    gamma = gamma[order]
    x = np.log(dt)
    y = np.log(gamma)
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    if rank != 2:
        raise ValueError("birth-limit log-OLS design is rank deficient")
    fitted = design @ coefficients
    residual = y - fitted
    centered = y - np.mean(y)
    residual_sum_squares = float(residual @ residual)
    total_sum_squares = float(centered @ centered)
    local_orders = np.diff(y) / np.diff(x)
    exponent = float(coefficients[0])
    tends_to_zero = exponent > 0.0 and bool(np.all(local_orders > 0.0))
    return {
        "delta_time_s_coarse_to_fine": dt,
        "birth_gamma_max_abs_m2_s_coarse_to_fine": gamma,
        "slope_p": exponent,
        "intercept_log_C": float(coefficients[1]),
        "local_orders": local_orders,
        "design_rank": int(rank),
        "singular_values": singular_values,
        "residual_sum_squares": residual_sum_squares,
        "r_squared": (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else 1.0
        ),
        "tends_to_zero": tends_to_zero,
        "force_coupling": FORCE_COUPLING,
        "scoring_status": SCORING_STATUS,
    }


__all__ = [
    "CORE_STATUS",
    "FORCE_COUPLING",
    "SCORING_STATUS",
    "FluxVV5BSharedWakeConfig",
    "FluxVV5BSharedWakeCore",
    "birth_limit_diagnostic",
    "dispatch_v5b_or_parent",
]
