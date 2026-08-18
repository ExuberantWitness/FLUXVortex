"""FluxV v5a's mutually exclusive equilibrium/transient force ledger.

The retained UVLM is the sole finite-wing and non-circulatory load owner.
This module adds two *coefficient* residuals and converts them to strip forces
with one, and only one, multiplication by local ``q S``:

``UVLM + direct equilibrium residual + high-passed paired-LDVM residual``.

The physical transient path deliberately keeps the two LDVM load axes in the
wing frame while filtering.  The frozen v4b finite-span gains first aggregate
the paired component discrepancy into projected normal and suction
coefficients.  Those two constant linear operations commute with the common
convective high-pass.  The time-dependent incidence rotation into lift and
drag happens only *after* filtering.  This is the two-state ``m_N, m_S``
implementation refinement of the proposal's ``m_N, m_D`` shorthand; filtering
lift/drag directly would not commute with a time-varying incidence.

There are no paper names, case identifiers, observation residuals, or fitted
branches in this module.  The compatibility helper at the end operates on
already projected, integrated coefficient histories for exploratory runners;
it is not the canonical strip-local physical path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .ldvm_uvlm_correction import project_ldvm_delta_to_finite_wing


PHYSICAL_COMPONENTS = ("normal", "suction")
LOAD_COMPONENTS = ("lift", "drag")
V5A_MODES = ("full", "equilibrium_only", "transient_only", "off")


@dataclass(frozen=True)
class FluxVV5AParameters:
    """The single new v5a time-scale assumption and its numerical floor."""

    lambda_tau: float = 1.0
    velocity_floor_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        if np.isnan(self.lambda_tau) or self.lambda_tau < 0.0:
            raise ValueError("lambda_tau must be nonnegative or positive infinity")
        if not np.isfinite(self.velocity_floor_fraction):
            raise ValueError("velocity floor fraction must be finite")
        if self.velocity_floor_fraction <= 0.0:
            raise ValueError("velocity floor fraction must be positive")

    def manifest(self) -> dict[str, float | str]:
        out: dict[str, float | str] = asdict(self)
        out.update(
            time_scale_source=(
                "FluxV v5a preregistration: one local convective time; "
                "lambda_tau=0.5/2.0 are sensitivity values only"
            ),
            state_contract=(
                "two wing-frame states m_N,m_S; frozen constant finite-span "
                "component gains commute with the high-pass, while the "
                "time-dependent alpha rotation is applied afterwards"
            ),
            observation_fit="none",
        )
        return out


DEFAULT_V5A_PARAMETERS = FluxVV5AParameters()


def _finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _component_history(name: str, value: Any) -> np.ndarray:
    history = _finite(name, value)
    if history.ndim < 2 or history.shape[0] < 1 or history.shape[-1] < 1:
        raise ValueError(
            f"{name} must be time-first with a nonempty final component axis"
        )
    return history


def _broadcast_to_history(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = _finite(name, value)
    try:
        return np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(f"{name} cannot broadcast to history shape {shape}") from error


def equilibrium_section_residual(
    normal_equilibrium_coefficient: Any,
    normal_attached_coefficient: Any,
    drag_equilibrium_coefficient: Any,
    drag_baseline_coefficient: Any,
) -> dict[str, Any]:
    """Return the direct two-dimensional equilibrium coefficient residual.

    No aspect ratio, LDVM gain, circulation projection, dynamic pressure, or
    strip area enters this operation.  Area integration is performed once by
    :func:`assemble_strip_force_ledger`.
    """

    cn_eq, cn_att, cd_eq, cd_base = np.broadcast_arrays(
        _finite("normal equilibrium coefficient", normal_equilibrium_coefficient),
        _finite("normal attached coefficient", normal_attached_coefficient),
        _finite("drag equilibrium coefficient", drag_equilibrium_coefficient),
        _finite("drag baseline coefficient", drag_baseline_coefficient),
    )
    delta_normal = cn_eq - cn_att
    delta_drag = cd_eq - cd_base
    coefficients = np.stack((delta_normal, delta_drag), axis=-1)
    return {
        "coefficient": coefficients,
        "delta_normal_coefficient": delta_normal,
        "delta_drag_coefficient": delta_drag,
        "component_order": LOAD_COMPONENTS,
        "spatial_mapping": "direct_section_to_strip_no_ldvm_projection",
        "units": "dimensionless_coefficient_before_qS",
    }


def project_ldvm_pair_components(
    section_pair: Mapping[str, Any],
    *,
    aspect_ratio: float,
) -> dict[str, Any]:
    """Apply frozen v4b component gains to a paired LDVM discrepancy.

    The returned coefficient axes are wing-frame normal and Ramesh axial
    suction.  Incidence rotation is intentionally deferred until after the
    convective filter.  The implementation calls the frozen v4b projector to
    make its gain contract auditable, then reconstructs its pre-rotation axes
    from the component ledgers returned by that projector.
    """

    try:
        delta = section_pair["delta"]
        cnc = _finite("paired delta CNc", delta["CNc"])
        cnnc = _finite("paired delta CNnc", delta["CNnc"])
        cn_nonl = _finite("paired delta CNnonl", delta["CNnonl"])
        cs = _finite("paired delta CSf", delta["CSf"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "section_pair must contain paired LDVM CNc/CNnc/CNnonl/CSf deltas"
        ) from error
    if not (cnc.shape == cnnc.shape == cn_nonl.shape == cs.shape):
        raise ValueError("paired LDVM component histories must have identical shapes")

    zero_incidence = np.zeros_like(cnc)
    projection = project_ldvm_delta_to_finite_wing(
        cnc,
        cnnc,
        cn_nonl,
        cs,
        zero_incidence,
        aspect_ratio=aspect_ratio,
    )
    normal = (
        np.asarray(projection["projected_delta_CNc"], dtype=float)
        + np.asarray(projection["projected_delta_CNnc"], dtype=float)
        + np.asarray(projection["projected_delta_CNnonl"], dtype=float)
    )
    suction = np.asarray(projection["projected_delta_CS"], dtype=float)
    coefficients = np.stack((normal, suction), axis=-1)
    return {
        "coefficient_ns": coefficients,
        "projected_delta_CN": normal,
        "projected_delta_CS": suction,
        "projected_component_ledgers": {
            key: np.asarray(projection[key], dtype=float)
            for key in (
                "projected_delta_CNc",
                "projected_delta_CNnc",
                "projected_delta_CNnonl",
                "projected_delta_CS",
            )
        },
        "gains": {
            key: float(projection[key])
            for key in (
                "normal_gain",
                "added_mass_gain",
                "nonlinear_normal_gain",
                "axial_suction_gain",
            )
        },
        "component_order": PHYSICAL_COMPONENTS,
        "units": "dimensionless_coefficient_before_high_pass_and_qS",
        "projection_semantics": (
            "frozen v4b component gains; alpha rotation deferred until after "
            "the v5a convective high-pass"
        ),
    }


def convective_increment(
    relative_speed_m_s: Any,
    delta_time_s: Any,
    chord_m: Any,
    *,
    reference_speed_m_s: float,
    velocity_floor_fraction: float = 1.0e-6,
) -> np.ndarray:
    """Return ``delta_chi = max(|V|, floor Uref) dt / c`` as an array."""

    if not np.isfinite(reference_speed_m_s) or reference_speed_m_s <= 0.0:
        raise ValueError("reference speed must be finite and positive")
    if not np.isfinite(velocity_floor_fraction) or velocity_floor_fraction <= 0.0:
        raise ValueError("velocity floor fraction must be finite and positive")
    speed, delta_time, chord = np.broadcast_arrays(
        _finite("relative speed", relative_speed_m_s),
        _finite("delta time", delta_time_s),
        _finite("chord", chord_m),
    )
    if np.any(delta_time <= 0.0):
        raise ValueError("delta time must be positive")
    if np.any(chord <= 0.0):
        raise ValueError("chord must be positive")
    floor = velocity_floor_fraction * reference_speed_m_s
    return np.maximum(np.abs(speed), floor) * delta_time / chord


def convective_high_pass(
    raw_discrepancy_coefficient: Any,
    *,
    delta_chi: Any,
    lambda_tau: float = 1.0,
    initial_state: Any | None = None,
) -> dict[str, Any]:
    """High-pass a time-first dimensionless discrepancy history.

    The residual at sample ``n`` is ``r[n] - m[n]`` and the state then advances
    with the exact exponential gain.  ``lambda_tau=0`` is implemented as its
    exact limiting branch (zero transient residual); positive infinity keeps
    the initial state frozen and therefore recovers the raw discrepancy when
    ``m0=0``.
    """

    raw = _component_history("raw discrepancy", raw_discrepancy_coefficient)
    chi = _broadcast_to_history("delta_chi", delta_chi, raw.shape[:-1])
    if np.any(chi < 0.0):
        raise ValueError("delta_chi must be nonnegative")
    if np.isnan(lambda_tau) or lambda_tau < 0.0:
        raise ValueError("lambda_tau must be nonnegative or positive infinity")

    state_shape = raw.shape[1:]
    if initial_state is None:
        state = np.zeros(state_shape, dtype=float)
    else:
        state = _finite("initial state", initial_state)
        if state.shape != state_shape:
            raise ValueError(f"initial state must have shape {state_shape}")
        state = state.copy()

    before = np.empty_like(raw)
    after = np.empty_like(raw)
    transient = np.empty_like(raw)
    gain = np.empty(raw.shape[:-1], dtype=float)

    if lambda_tau == 0.0:
        # Exact singular limit: the low-pass tracks each sample instantly.
        before[...] = raw
        after[...] = raw
        transient.fill(0.0)
        gain.fill(1.0)
        state = raw[-1].copy()
    else:
        for index in range(raw.shape[0]):
            before[index] = state
            transient[index] = raw[index] - state
            if np.isposinf(lambda_tau):
                sample_gain = np.zeros_like(chi[index])
            else:
                sample_gain = -np.expm1(-chi[index] / lambda_tau)
            gain[index] = sample_gain
            state = state + sample_gain[..., None] * transient[index]
            after[index] = state

    if not np.all(np.isfinite(transient + before + after)):
        raise FloatingPointError("convective high-pass produced non-finite values")
    return {
        "raw_coefficient": raw.copy(),
        "low_pass_state_before": before,
        "low_pass_state_after": after,
        "transient_coefficient": transient,
        "final_state": state.copy(),
        "delta_chi": chi.copy(),
        "update_gain": gain,
        "lambda_tau": float(lambda_tau),
        "units": "dimensionless_coefficient",
    }


def resolve_normal_suction_to_lift_drag(
    normal_suction_coefficient: Any,
    alpha_rad: Any,
) -> dict[str, Any]:
    """Rotate filtered ``(CN, CS)`` into ``(CL, CD)`` at instantaneous alpha."""

    coefficient = _component_history(
        "normal/suction coefficient", normal_suction_coefficient
    )
    if coefficient.shape[-1] != 2:
        raise ValueError("normal/suction history must have exactly two components")
    alpha = _broadcast_to_history("alpha", alpha_rad, coefficient.shape[:-1])
    normal = coefficient[..., 0]
    suction = coefficient[..., 1]
    lift = normal * np.cos(alpha) + suction * np.sin(alpha)
    drag = normal * np.sin(alpha) - suction * np.cos(alpha)
    load_axis = np.stack((lift, drag), axis=-1)
    return {
        "coefficient_ld": load_axis,
        "delta_CL": lift,
        "delta_CD": drag,
        "alpha_rad": alpha,
        "component_order": LOAD_COMPONENTS,
        "rotation_semantics": "instantaneous alpha rotation after high-pass",
    }


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "equilibrium": "equilibrium_only",
        "transient": "transient_only",
        "module_off": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in V5A_MODES:
        raise ValueError(f"mode must be one of {V5A_MODES}")
    return normalized


def assemble_strip_force_ledger(
    uvlm_force_n: Any,
    *,
    dynamic_pressure_pa: Any,
    strip_area_m2: Any,
    equilibrium_residual_coefficient: Any,
    transient_residual_coefficient: Any,
    mode: str = "full",
) -> dict[str, Any]:
    """Close the v5a strip force ledger with a single local ``q S`` product.

    All force/coefficient histories are time-first and use the same final
    ``(lift, drag)`` component order.  The coefficient providers are forbidden
    from pre-multiplying pressure or area by this unit contract.
    """

    baseline = _component_history("UVLM force", uvlm_force_n)
    if baseline.shape[-1] != 2:
        raise ValueError("UVLM force must have exactly lift and drag components")
    equilibrium = _component_history(
        "equilibrium residual coefficient", equilibrium_residual_coefficient
    )
    transient = _component_history(
        "transient residual coefficient", transient_residual_coefficient
    )
    if equilibrium.shape != baseline.shape or transient.shape != baseline.shape:
        raise ValueError("all force-ledger histories must have identical shapes")
    q = _broadcast_to_history(
        "dynamic pressure", dynamic_pressure_pa, baseline.shape[:-1]
    )
    area = _broadcast_to_history("strip area", strip_area_m2, baseline.shape[:-1])
    if np.any(q < 0.0):
        raise ValueError("dynamic pressure cannot be negative")
    if np.any(area <= 0.0):
        raise ValueError("strip area must be positive")

    normalized_mode = _normalize_mode(mode)
    q_area = q * area
    zero_force = np.zeros_like(baseline)
    if normalized_mode == "off":
        equilibrium_force = zero_force.copy()
        transient_force = zero_force.copy()
        # Preserve exact/bitwise UVLM reduction rather than recomputing it.
        total = baseline.copy()
    else:
        equilibrium_force = (
            q_area[..., None] * equilibrium
            if normalized_mode in ("full", "equilibrium_only")
            else zero_force.copy()
        )
        transient_force = (
            q_area[..., None] * transient
            if normalized_mode in ("full", "transient_only")
            else zero_force.copy()
        )
        if not np.any(equilibrium_force) and not np.any(transient_force):
            # The attached/no-separation limit is an exact implementation
            # identity, including the sign bit of a zero-valued UVLM entry.
            total = baseline.copy()
        else:
            total = baseline + equilibrium_force + transient_force

    recomputed = baseline + equilibrium_force + transient_force
    closure = total - recomputed
    absolute_closure = float(np.max(np.abs(closure)))
    scale = max(float(np.max(np.abs(total))), 1.0e-300)
    return {
        "total_force_n": total,
        "uvlm_force_n": baseline.copy(),
        "equilibrium_force_n": equilibrium_force,
        "transient_force_n": transient_force,
        "equilibrium_residual_coefficient": equilibrium.copy(),
        "transient_residual_coefficient": transient.copy(),
        "dynamic_pressure_pa": q.copy(),
        "strip_area_m2": area.copy(),
        "qS_n": q_area,
        "ledger_residual_n": closure,
        "ledger_max_abs_residual_n": absolute_closure,
        "ledger_relative_residual": absolute_closure / scale,
        "qS_multiplication_count": 1,
        "mode": normalized_mode,
        "component_order": LOAD_COMPONENTS,
        "ownership": {
            "uvlm": "finite-wing circulation, induced drag, added mass",
            "equilibrium": "direct section-to-strip saturation/profile residual",
            "transient": "convective high-pass paired-LDVM discrepancy only",
        },
    }


def periodic_convergence_diagnostic(
    low_pass_state_after: Any,
    load_history: Any,
    *,
    steps_per_cycle: int,
    tolerance: float = 1.0e-4,
    required_consecutive_passes: int = 1,
) -> dict[str, Any]:
    """Audit cycle-end state and cycle-mean load convergence.

    A pass compares two consecutive complete cycles.  Set
    ``required_consecutive_passes=2`` when a runner requires two successful
    transitions (three complete cycles) before scoring.
    """

    state = _component_history("low-pass state", low_pass_state_after)
    load = _component_history("load history", load_history)
    if state.shape[0] != load.shape[0]:
        raise ValueError("state and load histories must have the same time length")
    if steps_per_cycle < 2:
        raise ValueError("steps_per_cycle must be at least two")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if required_consecutive_passes < 1:
        raise ValueError("required_consecutive_passes must be positive")
    cycles = state.shape[0] // steps_per_cycle
    if cycles < 2:
        return {
            "passed": False,
            "reason": "fewer_than_two_complete_cycles",
            "complete_cycles": int(cycles),
            "tolerance": float(tolerance),
            "transitions": [],
        }

    transitions: list[dict[str, float | int | bool]] = []
    pass_streak = 0
    for cycle in range(1, cycles):
        previous_slice = slice((cycle - 1) * steps_per_cycle, cycle * steps_per_cycle)
        current_slice = slice(cycle * steps_per_cycle, (cycle + 1) * steps_per_cycle)
        previous_state = state[cycle * steps_per_cycle - 1]
        current_state = state[(cycle + 1) * steps_per_cycle - 1]
        previous_mean = np.mean(load[previous_slice], axis=0)
        current_mean = np.mean(load[current_slice], axis=0)
        state_scale = max(
            float(np.max(np.abs(previous_state))),
            float(np.max(np.abs(current_state))),
            1.0e-12,
        )
        load_scale = max(
            float(np.max(np.abs(previous_mean))),
            float(np.max(np.abs(current_mean))),
            1.0e-12,
        )
        state_change = float(np.max(np.abs(current_state - previous_state)))
        load_change = float(np.max(np.abs(current_mean - previous_mean)))
        state_relative = state_change / state_scale
        load_relative = load_change / load_scale
        passed = state_relative < tolerance and load_relative < tolerance
        pass_streak = pass_streak + 1 if passed else 0
        transitions.append(
            {
                "from_cycle": int(cycle - 1),
                "to_cycle": int(cycle),
                "state_max_abs_change": state_change,
                "load_mean_max_abs_change": load_change,
                "state_relative_change": state_relative,
                "load_mean_relative_change": load_relative,
                "passed": passed,
            }
        )
    return {
        "passed": pass_streak >= required_consecutive_passes,
        "complete_cycles": int(cycles),
        "ignored_tail_steps": int(state.shape[0] - cycles * steps_per_cycle),
        "tolerance": float(tolerance),
        "required_consecutive_passes": int(required_consecutive_passes),
        "final_pass_streak": int(pass_streak),
        "transitions": transitions,
    }


def _load_axis_history(value: Any, name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        try:
            return np.stack(
                (
                    _finite(f"{name} CL", value["CL"]),
                    _finite(f"{name} CD", value["CD"]),
                ),
                axis=-1,
            )
        except KeyError as error:
            raise ValueError(f"{name} mapping must contain CL and CD") from error
    history = _component_history(name, value)
    if history.shape[-1] != 2:
        raise ValueError(f"{name} must have exactly CL and CD components")
    return history


def apply_fluxv_v5a_ledger(
    baseline: Any,
    equilibrium: Any,
    ldvm_delta: Any,
    delta_chi: Any,
    lambda_tau: float = 1.0,
    mode: str = "full",
) -> dict[str, Any]:
    """Compatibility adapter for projected, integrated CL/CD histories.

    ``ldvm_delta`` must already be the frozen-P_LDVM projected *dimensionless*
    paired discrepancy.  Canonical strip-local runs should instead use
    :func:`project_ldvm_pair_components`, :func:`convective_high_pass`,
    :func:`resolve_normal_suction_to_lift_drag`, and
    :func:`assemble_strip_force_ledger` in that order.
    """

    base = _load_axis_history(baseline, "baseline")
    equilibrium_delta = _load_axis_history(equilibrium, "equilibrium")
    raw_ldvm = _load_axis_history(ldvm_delta, "ldvm delta")
    if equilibrium_delta.shape != base.shape or raw_ldvm.shape != base.shape:
        raise ValueError("compatibility histories must have identical shapes")
    filtered = convective_high_pass(
        raw_ldvm,
        delta_chi=delta_chi,
        lambda_tau=lambda_tau,
    )
    normalized_mode = _normalize_mode(mode)
    equilibrium_owned = (
        equilibrium_delta
        if normalized_mode in ("full", "equilibrium_only")
        else np.zeros_like(base)
    )
    transient_owned = (
        np.asarray(filtered["transient_coefficient"], dtype=float)
        if normalized_mode in ("full", "transient_only")
        else np.zeros_like(base)
    )
    total = (
        base.copy()
        if normalized_mode == "off"
        else base + equilibrium_owned + transient_owned
    )
    closure = total - (base + equilibrium_owned + transient_owned)
    return {
        "CL": total[..., 0],
        "CD": total[..., 1],
        "equilibrium_CL": equilibrium_owned[..., 0],
        "equilibrium_CD": equilibrium_owned[..., 1],
        "raw_ldvm_CL": raw_ldvm[..., 0],
        "raw_ldvm_CD": raw_ldvm[..., 1],
        "state_CL": np.asarray(filtered["low_pass_state_after"])[..., 0],
        "state_CD": np.asarray(filtered["low_pass_state_after"])[..., 1],
        "transient_CL": transient_owned[..., 0],
        "transient_CD": transient_owned[..., 1],
        "ledger_residual_CL": closure[..., 0],
        "ledger_residual_CD": closure[..., 1],
        "mode": normalized_mode,
        "lambda_tau": float(lambda_tau),
        "delta_chi": np.asarray(filtered["delta_chi"], dtype=float),
        "compatibility_scope": "projected_integrated_proxy",
        "canonical_strip_gate": "blocked",
        "semantics": (
            "coefficient-level compatibility adapter; no qS is applied here and "
            "it cannot establish the canonical strip-local v5a claim"
        ),
    }


__all__ = [
    "DEFAULT_V5A_PARAMETERS",
    "FluxVV5AParameters",
    "LOAD_COMPONENTS",
    "PHYSICAL_COMPONENTS",
    "V5A_MODES",
    "apply_fluxv_v5a_ledger",
    "assemble_strip_force_ledger",
    "convective_high_pass",
    "convective_increment",
    "equilibrium_section_residual",
    "periodic_convergence_diagnostic",
    "project_ldvm_pair_components",
    "resolve_normal_suction_to_lift_drag",
]
