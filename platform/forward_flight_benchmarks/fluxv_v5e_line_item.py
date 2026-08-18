"""Isolated ULLT-to-UVLM unsteady line-item replacement shadow.

This module replaces exactly one UVLM load-ledger entry.  The caller supplies
the old total strip force and its Kutta--Joukowski component separately.  When
the shadow is enabled, the old ``dGamma/dt`` force is removed in full and the
new strip force is

``F_new = F_KJ + Delta F_(phi-Gamma) + F_AM,kin``.

For every time and strip, the one-state closure is frozen as

``Gamma_eq = (F_KJ dot e_L) / (rho V_perp ds)``,
``Delta t_tilde = 2 V_perp dt / c``,
``y_Gamma = 2 Gamma_eq / (c 2 pi)``,
``x_n = exp(-1.25 Delta t_tilde) x_(n-1)``
``      + 0.5 (1-exp(-1.25 Delta t_tilde)) y_Gamma,n``, and
``y_phi = 2.5 y_Gamma - 3 x``.

The corresponding lift-direction mismatch is

``Delta F_(phi-Gamma) = 0.5 rho c 2 pi V_perp``
``                        (y_phi-y_Gamma) ds e_L``.

``e_L`` is an explicit unit Kutta--Joukowski lift direction.  It is not the
wing surface normal: those directions generally differ at high incidence.

No polar, separation selector, LDVM/LEV force, case identity, or observation
enters the closure.  The kinematic added-mass history is an independent input,
and its required provenance enum freezes Izraelevitz Eqs. (35)--(39), the
published finite-AR interpolation, and exclusive replacement ownership.

``initial_state`` is the state immediately before the first supplied sample.
A periodic caller can warm up causally by repeatedly calling this function and
passing the returned ``final_state`` into the next call.  The module does not
wrap the supplied history or inspect future samples.  Each row uses a
right-end zero-order hold: row ``n`` is held over the interval ending at sample
``n``; ``x_history[n]`` and the reported force are the post-update values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


SECTION_LIFT_SLOPE_PER_RAD = 2.0 * np.pi
STATE_DECAY_PER_T_TILDE = 1.25
STATE_EQUILIBRIUM_GAIN = 0.5
PHI_DIRECT_GAIN = 2.5
PHI_STATE_GAIN = -3.0


class KinematicAddedMassProvenance(str, Enum):
    """Only mechanically admissible added-mass provenance for this shadow."""

    IZRAELEVITZ_EQS35_39_AR_INTERPOLATED_EXCLUSIVE = (
        "izraelevitz2017_eqs35_39_ar3_0p85_ar6_0p95_"
        "linear_clamped_exclusive_replacement_v1"
    )


CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE = (
    KinematicAddedMassProvenance.IZRAELEVITZ_EQS35_39_AR_INTERPOLATED_EXCLUSIVE
)


@dataclass(frozen=True)
class ULLTUVLMLineItemParameters:
    """Frozen coefficients of the v5e line-item shadow."""

    section_lift_slope_per_rad: float = SECTION_LIFT_SLOPE_PER_RAD
    state_decay_per_t_tilde: float = STATE_DECAY_PER_T_TILDE
    state_equilibrium_gain: float = STATE_EQUILIBRIUM_GAIN
    phi_direct_gain: float = PHI_DIRECT_GAIN
    phi_state_gain: float = PHI_STATE_GAIN

    def __post_init__(self) -> None:
        expected = (
            (self.section_lift_slope_per_rad, SECTION_LIFT_SLOPE_PER_RAD),
            (self.state_decay_per_t_tilde, STATE_DECAY_PER_T_TILDE),
            (self.state_equilibrium_gain, STATE_EQUILIBRIUM_GAIN),
            (self.phi_direct_gain, PHI_DIRECT_GAIN),
            (self.phi_state_gain, PHI_STATE_GAIN),
        )
        if any(not np.isfinite(value) for value, _ in expected):
            raise ValueError("v5e line-item parameters must be finite")
        if any(value != frozen for value, frozen in expected):
            raise ValueError(
                "v5e line-item coefficients are source-frozen; custom values "
                "require another model identity"
            )

    def manifest(self) -> dict[str, float | str]:
        """Return a machine-readable formula and ownership declaration."""

        out: dict[str, float | str] = asdict(self)
        out.update(
            model_identity="FluxV-v5e-ULLT-UVLM-line-item-shadow",
            state_source=(
                "Izraelevitz, Zhu, and Triantafyllou (2017), "
                "one-state phi/Gamma closure"
            ),
            gamma_rule="Gamma_eq=(F_KJ dot e_L)/(rho*V_perp*ds)",
            lift_direction_identity=(
                "e_L is the explicit unit Kutta--Joukowski lift direction; "
                "it is not inferred from the surface normal"
            ),
            convective_clock="Delta t_tilde=2*V_perp*dt/c",
            state_update=(
                "x_n=exp(-1.25*Delta t_tilde)*x_(n-1) + "
                "0.5*(1-exp(-1.25*Delta t_tilde))*y_Gamma,n"
            ),
            output_rule="y_phi=2.5*y_Gamma-3*x",
            time_discretization=(
                "right-end ZOH: row n inputs are held over (t_n-dt,t_n]; "
                "x_history[n] and force[n] are post-update at t_n"
            ),
            force_rule=(
                "F_new=F_KJ+DeltaF_(phi-Gamma)+F_AM,kin; "
                "old F_dGamma is removed completely"
            ),
            added_mass_ownership=(
                "caller supplies force plus the required frozen provenance enum; "
                "the term is inserted exactly once"
            ),
            excluded_terms="LDVM/LEV, polar, separation, and owner selectors",
            observation_access="none",
            observation_fit="none",
            case_or_paper_branch="none",
        )
        return out


DEFAULT_LINE_ITEM_PARAMETERS = ULLTUVLMLineItemParameters()


def _added_mass_provenance_manifest(
    provenance: KinematicAddedMassProvenance,
) -> dict[str, bool | str]:
    """Return the frozen source and exclusive-ownership contract."""

    if provenance is not CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE:
        raise ValueError(
            "kinematic_added_mass_provenance must be the frozen "
            "KinematicAddedMassProvenance enum member; missing, raw-string, "
            "or unknown provenance is not admissible"
        )
    return {
        "tag": provenance.value,
        "source": "Izraelevitz et al. (2017), Eqs. (35)-(39)",
        "aspect_ratio_rule": (
            "K_AM=0.85 at AR=3 and 0.95 at AR=6; linear interpolation "
            "on [3,6] with endpoint clamp"
        ),
        "exclusive_replacement": True,
        "force_ownership": (
            "F_AM,kin is supplied once and replaces, rather than augments, "
            "the removed UVLM F_dGamma line item"
        ),
        "canonical_tag_accepted": True,
    }


def _force_history(value: Any, name: str) -> np.ndarray:
    force = np.asarray(value)
    if force.ndim != 3 or force.shape[0] < 1 or force.shape[1] < 1:
        raise ValueError(f"{name} must have shape (time>=1, strip>=1, 3)")
    if force.shape[2] != 3:
        raise ValueError(f"{name} must have shape (time>=1, strip>=1, 3)")
    if not np.issubdtype(force.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if np.any(~np.isfinite(force)):
        raise ValueError(f"{name} must be finite")
    return force


def _matching_force_history(
    value: Any, topology: tuple[int, int, int], name: str
) -> np.ndarray:
    force = np.asarray(value, dtype=float)
    if force.shape != topology:
        raise ValueError(f"{name} must match baseline force topology {topology}")
    if np.any(~np.isfinite(force)):
        raise ValueError(f"{name} must be finite")
    return force


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    scalar = float(array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _strip_scalar_history(
    value: Any,
    *,
    time_count: int,
    strip_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        history = np.full((time_count, strip_count), float(array), dtype=float)
    elif array.shape == (strip_count,):
        history = np.broadcast_to(array[None, :], (time_count, strip_count)).copy()
    elif array.shape == (time_count, strip_count):
        history = array
    else:
        raise ValueError(
            f"{name} must be scalar or have shape (strip,) or (time, strip)"
        )
    if np.any(~np.isfinite(history)) or np.any(history <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return history


def _lift_direction_history(
    value: Any, *, time_count: int, strip_count: int
) -> np.ndarray:
    directions = np.asarray(value, dtype=float)
    if directions.shape == (strip_count, 3):
        history = np.broadcast_to(
            directions[None, :, :], (time_count, strip_count, 3)
        ).copy()
    elif directions.shape == (time_count, strip_count, 3):
        history = directions
    else:
        raise ValueError(
            "strip_lift_direction must have shape (strip, 3) or " "(time, strip, 3)"
        )
    if np.any(~np.isfinite(history)):
        raise ValueError("strip_lift_direction must be finite")
    norms = np.linalg.norm(history, axis=2)
    if np.any(~np.isfinite(norms)) or not np.allclose(
        norms, 1.0, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("strip_lift_direction vectors must be unit length")
    return history


def _initial_state(value: Any, strip_count: int) -> np.ndarray:
    if value is None:
        return np.zeros(strip_count, dtype=float)
    state = np.asarray(value, dtype=float)
    if state.ndim == 0:
        state = np.full(strip_count, float(state), dtype=float)
    elif state.shape != (strip_count,):
        raise ValueError("initial_state must be scalar or have shape (strip,)")
    if np.any(~np.isfinite(state)):
        raise ValueError("initial_state must be finite")
    return state.copy()


def _disabled_result(
    baseline: np.ndarray, parameters: ULLTUVLMLineItemParameters
) -> dict[str, Any]:
    # No enabled-only input is converted, shaped, or numerically inspected in
    # this branch.  Copying preserves the baseline dtype and every finite bit.
    unchanged = baseline.copy()
    manifest = parameters.manifest()
    manifest["kinematic_added_mass_provenance_tag"] = "not_evaluated_disabled"
    return {
        "new_force_history": unchanged,
        "span_summed_force_history": np.sum(unchanged, axis=1),
        "components": {
            "f_kj_history": None,
            "removed_old_fd_gamma_force_history": None,
            "delta_phi_gamma_force_history": None,
            "kinematic_added_mass_force_history": None,
            "strip_lift_direction_history": None,
        },
        "state": {
            "gamma_eq_history": None,
            "delta_t_tilde_history": None,
            "y_gamma_history": None,
            "x_history": None,
            "y_phi_history": None,
            "initial_state": None,
            "final_state": None,
        },
        "ledger": {
            "old_fd_gamma_replaced_completely": False,
            "module_off_max_abs_residual": 0.0,
            "new_force_closure_max_abs_residual": 0.0,
            "replacement_identity_max_abs_residual": 0.0,
        },
        "parameters": manifest,
        "provenance": {
            "kinematic_added_mass": {
                "tag": "not_evaluated_disabled",
                "canonical_tag_accepted": False,
            }
        },
        "diagnostics": {
            "enabled": False,
            "status": "not_evaluated_disabled_exact_baseline",
            "state_causal": None,
            "state_lookahead_samples": None,
            "added_mass_causality": "not_evaluated_disabled",
            "overall_causality": "not_established_by_line_item_core",
            "state_updated": False,
        },
        "model_contract": {
            "shadow_only": True,
            "canonical_eligible": False,
            "force_owner": "disabled; baseline UVLM total force",
            "observation_access": "none",
        },
    }


def ullt_to_uvlm_line_item_shadow(
    baseline_total_force_history: Any,
    *,
    f_kj_history: Any,
    strip_lift_direction: Any,
    chord: Any,
    strip_width: Any,
    v_perp: Any,
    density: Any,
    delta_time: Any,
    kinematic_added_mass_force_history: Any,
    kinematic_added_mass_provenance: KinematicAddedMassProvenance,
    initial_state: Any = None,
    enabled: bool = True,
    parameters: ULLTUVLMLineItemParameters = DEFAULT_LINE_ITEM_PARAMETERS,
) -> dict[str, Any]:
    """Replace the old UVLM ``dGamma/dt`` strip-force line item.

    All force histories contain one three-vector per time and strip.  Thus the
    word ``total`` in ``baseline_total_force_history`` denotes the old
    per-strip total ``F_KJ + F_dGamma``; it is not a pre-summed airplane force.
    ``strip_lift_direction`` is the explicit unit Kutta--Joukowski lift
    direction, not a surface normal.  It, ``chord``, and ``strip_width`` may be
    static strip data or time histories.  ``v_perp`` is the positive local
    speed magnitude used by both the Kutta--Joukowski identity and semi-chord
    clock.  Row ``n`` uses a right-end ZOH over the interval ending at row
    ``n`` and reports the post-update state and force.

    The added-mass force is accepted only with the single frozen provenance
    enum.  That tag asserts Izraelevitz Eqs. (35)--(39), AR interpolation from
    the 0.85/0.95 landmarks, and exclusive replacement ownership.

    Disabled execution validates and copies only the baseline history.  It
    deliberately does not evaluate any enabled-only input, including the
    initial state.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be Boolean")
    baseline = _force_history(
        baseline_total_force_history, "baseline_total_force_history"
    )
    if not enabled:
        return _disabled_result(baseline, parameters)

    time_count, strip_count, _ = baseline.shape
    topology = (time_count, strip_count, 3)
    f_kj = _matching_force_history(f_kj_history, topology, "f_kj_history")
    added_mass = _matching_force_history(
        kinematic_added_mass_force_history,
        topology,
        "kinematic_added_mass_force_history",
    )
    added_mass_provenance = _added_mass_provenance_manifest(
        kinematic_added_mass_provenance
    )
    lift_directions = _lift_direction_history(
        strip_lift_direction, time_count=time_count, strip_count=strip_count
    )
    chord_history = _strip_scalar_history(
        chord, time_count=time_count, strip_count=strip_count, name="chord"
    )
    width_history = _strip_scalar_history(
        strip_width,
        time_count=time_count,
        strip_count=strip_count,
        name="strip_width",
    )
    speed_history = _strip_scalar_history(
        v_perp, time_count=time_count, strip_count=strip_count, name="v_perp"
    )
    rho = _scalar(density, "density")
    dt = _scalar(delta_time, "delta_time")
    state = _initial_state(initial_state, strip_count)
    supplied_initial_state = state.copy()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kj_lift_force = np.sum(f_kj * lift_directions, axis=2)
        gamma_eq = kj_lift_force / (rho * speed_history * width_history)
        delta_t_tilde = 2.0 * speed_history * dt / chord_history
        y_gamma = (
            2.0 * gamma_eq / (chord_history * parameters.section_lift_slope_per_rad)
        )
    for value, name in (
        (gamma_eq, "Gamma_eq"),
        (delta_t_tilde, "Delta t_tilde"),
        (y_gamma, "y_Gamma"),
    ):
        if np.any(~np.isfinite(value)):
            raise ValueError(f"computed {name} history is not finite")
    if np.any(delta_t_tilde <= 0.0):
        raise ValueError("computed Delta t_tilde history must be positive")

    state_history = np.empty_like(y_gamma)
    y_phi = np.empty_like(y_gamma)
    for time_index in range(time_count):
        step = delta_t_tilde[time_index]
        decay = np.exp(-parameters.state_decay_per_t_tilde * step)
        # -expm1 preserves the positive assimilation gain at small steps.
        gain = -np.expm1(-parameters.state_decay_per_t_tilde * step)
        with np.errstate(over="ignore", invalid="ignore"):
            state = decay * state + (
                parameters.state_equilibrium_gain * gain * y_gamma[time_index]
            )
            current_y_phi = (
                parameters.phi_direct_gain * y_gamma[time_index]
                + parameters.phi_state_gain * state
            )
        if np.any(~np.isfinite(state)) or np.any(~np.isfinite(current_y_phi)):
            raise ValueError("one-state closure produced a nonfinite state")
        state_history[time_index] = state
        y_phi[time_index] = current_y_phi

    with np.errstate(over="ignore", invalid="ignore"):
        mismatch_scalar = (
            0.5
            * rho
            * chord_history
            * parameters.section_lift_slope_per_rad
            * speed_history
            * (y_phi - y_gamma)
            * width_history
        )
        delta_phi_gamma = mismatch_scalar[:, :, None] * lift_directions
        old_fd_gamma = baseline.astype(float, copy=False) - f_kj
        reconstructed_new = (f_kj + delta_phi_gamma) + added_mass
        new_force = reconstructed_new.copy()
        replacement_increment = (delta_phi_gamma + added_mass) - old_fd_gamma
        replacement_residual = (
            new_force - baseline.astype(float, copy=False)
        ) - replacement_increment
        closure_residual = new_force - reconstructed_new
        kj_identity_residual = (
            rho * speed_history * width_history * gamma_eq - kj_lift_force
        )
        delta_direction_residual = delta_phi_gamma - (
            np.sum(delta_phi_gamma * lift_directions, axis=2)[:, :, None]
            * lift_directions
        )
        span_summed = np.sum(new_force, axis=1)
    for value, name in (
        (delta_phi_gamma, "Delta F_(phi-Gamma)"),
        (old_fd_gamma, "old F_dGamma"),
        (new_force, "new force"),
        (replacement_residual, "replacement ledger residual"),
        (kj_identity_residual, "KJ lift-direction identity residual"),
        (delta_direction_residual, "lift-direction force residual"),
        (span_summed, "span-summed force"),
    ):
        if np.any(~np.isfinite(value)):
            raise ValueError(f"computed {name} is not finite")

    manifest = parameters.manifest()
    manifest["kinematic_added_mass_provenance_tag"] = added_mass_provenance["tag"]
    return {
        "new_force_history": new_force,
        "span_summed_force_history": span_summed,
        "components": {
            "f_kj_history": f_kj,
            "removed_old_fd_gamma_force_history": old_fd_gamma,
            "delta_phi_gamma_force_history": delta_phi_gamma,
            "kinematic_added_mass_force_history": added_mass,
            "strip_lift_direction_history": lift_directions,
        },
        "state": {
            "gamma_eq_history": gamma_eq,
            "delta_t_tilde_history": delta_t_tilde,
            "y_gamma_history": y_gamma,
            "x_history": state_history,
            "y_phi_history": y_phi,
            "initial_state": supplied_initial_state,
            "final_state": state.copy(),
        },
        "ledger": {
            "old_fd_gamma_replaced_completely": True,
            "module_off_max_abs_residual": None,
            "new_force_closure_max_abs_residual": float(
                np.max(np.abs(closure_residual))
            ),
            "replacement_identity_max_abs_residual": float(
                np.max(np.abs(replacement_residual))
            ),
            "kj_lift_direction_identity_max_abs_residual": float(
                np.max(np.abs(kj_identity_residual))
            ),
            "delta_lift_direction_max_abs_residual": float(
                np.max(np.abs(delta_direction_residual))
            ),
        },
        "parameters": manifest,
        "provenance": {"kinematic_added_mass": added_mass_provenance},
        "diagnostics": {
            "enabled": True,
            "status": "ullt_uvlm_line_item_shadow_evaluated",
            "state_causal": True,
            "state_lookahead_samples": 0,
            "added_mass_causality": "caller-provenance; not established by core",
            "overall_causality": "not_established_by_line_item_core",
            "state_updated": True,
            "time_discretization": "right_end_zero_order_hold",
        },
        "model_contract": {
            "shadow_only": True,
            "canonical_eligible": False,
            "force_owner": (
                "F_KJ + DeltaF_(phi-Gamma) + caller-supplied F_AM,kin; "
                "old F_dGamma removed completely"
            ),
            "observation_access": "none",
        },
    }


__all__ = [
    "CANONICAL_KINEMATIC_ADDED_MASS_PROVENANCE",
    "DEFAULT_LINE_ITEM_PARAMETERS",
    "KinematicAddedMassProvenance",
    "PHI_DIRECT_GAIN",
    "PHI_STATE_GAIN",
    "SECTION_LIFT_SLOPE_PER_RAD",
    "STATE_DECAY_PER_T_TILDE",
    "STATE_EQUILIBRIUM_GAIN",
    "ULLTUVLMLineItemParameters",
    "ullt_to_uvlm_line_item_shadow",
]
