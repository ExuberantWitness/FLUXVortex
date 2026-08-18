"""Reference-point-safe profile-drag ledger for FluxV v5c.

The Figure-14 benchmark adds Scherer's source-specified ``Cd0=0.057`` to an
otherwise inviscid load history.  That profile term is a kinematic line item:
it must be evaluated at the paper's three-quarter-chord reference and it must
be owned exactly once.  Earlier frozen v4b histories used a quarter-chord
kinematic reference.  This module rebases only that line item,

``F_target = F_owned + Cd0 * (F_unit,target - F_unit,owned)``,

without changing UVLM circulation, LDVM increments, or any other force owner.
It intentionally contains no paper/case switch and accepts no observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProfileDragReference:
    """Declared ownership of one constant-profile-drag ledger line."""

    coefficient: float
    fraction_chord: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("profile-drag coefficient must be finite and nonnegative")
        if (
            not np.isfinite(self.fraction_chord)
            or not 0.0 <= self.fraction_chord <= 1.0
        ):
            raise ValueError("profile-drag reference must lie on the chord")


def _reference_fraction(kinematics: dict[str, Any]) -> float:
    try:
        value = float(
            kinematics["parameters"]["section_velocity_reference_fraction_chord"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "kinematic ledger lacks a declared chord-reference fraction"
        ) from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("invalid kinematic chord-reference fraction")
    return value


def _aligned_phase(
    first: dict[str, Any],
    second: dict[str, Any],
) -> np.ndarray:
    phase = np.asarray(first["phase"], dtype=float)
    other = np.asarray(second["phase"], dtype=float)
    if phase.ndim != 1 or phase.size < 2 or not np.all(np.isfinite(phase)):
        raise ValueError("profile-drag phase must be a finite one-dimensional cycle")
    if phase.shape != other.shape or not np.allclose(
        phase, other, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("owned and target profile-drag phases are not aligned")
    return phase


def profile_drag_reference_delta(
    owned_kinematics: dict[str, Any],
    target_kinematics: dict[str, Any],
    *,
    ownership: ProfileDragReference,
    target_fraction_chord: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Return the sole correction needed to move an owned ``Cd0`` ledger.

    Both kinematic inputs are outputs of ``movement_polar_residual`` on the
    same phase grid.  The returned force is in the benchmark output axes:
    ``+z`` is lift, ``+x`` is drag, and therefore ``CT=-CD``.
    """

    phase = _aligned_phase(owned_kinematics, target_kinematics)
    owned_fraction = _reference_fraction(owned_kinematics)
    target_fraction = _reference_fraction(target_kinematics)
    if not np.isclose(owned_fraction, ownership.fraction_chord, atol=1.0e-15, rtol=0.0):
        raise ValueError("owned kinematics do not match declared ledger ownership")
    if not np.isclose(target_fraction, target_fraction_chord, atol=1.0e-15, rtol=0.0):
        raise ValueError("target kinematics do not match requested reference")

    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    if not np.isfinite(q_area) or q_area <= 0.0:
        raise ValueError("reference dynamic pressure times area must be positive")

    keys = (
        "unit_profile_drag_lift_n",
        "unit_profile_drag_drag_n",
    )
    arrays: list[np.ndarray] = []
    for key in keys:
        owned = np.asarray(owned_kinematics[key], dtype=float)
        target = np.asarray(target_kinematics[key], dtype=float)
        if owned.shape != phase.shape or target.shape != phase.shape:
            raise ValueError("profile-drag force history does not match phase")
        if not np.all(np.isfinite(owned)) or not np.all(np.isfinite(target)):
            raise ValueError("profile-drag force history contains non-finite values")
        arrays.append(ownership.coefficient * (target - owned))
    delta_lift, delta_drag = arrays
    delta_cl = delta_lift / q_area
    delta_cd = delta_drag / q_area
    delta_ct = -delta_cd
    return {
        "phase": phase.copy(),
        "delta_lift_n": delta_lift,
        "delta_drag_n": delta_drag,
        "delta_thrust_n": -delta_drag,
        "delta_CL": delta_cl,
        "delta_CD": delta_cd,
        "delta_CT": delta_ct,
        "mean_delta_lift_n": float(np.mean(delta_lift)),
        "mean_delta_drag_n": float(np.mean(delta_drag)),
        "mean_delta_thrust_n": -float(np.mean(delta_drag)),
        "mean_delta_CL": float(np.mean(delta_cl)),
        "mean_delta_CD": float(np.mean(delta_cd)),
        "mean_delta_CT": float(np.mean(delta_ct)),
        "profile_drag_coefficient": ownership.coefficient,
        "owned_reference_fraction_chord": owned_fraction,
        "target_reference_fraction_chord": target_fraction,
        "semantics": (
            "source-defined constant profile-drag reference rebase: "
            "Cd0*(unit_target-unit_owned), with no UVLM/LDVM change"
        ),
    }


def local_velocity_reference_delta(
    owned_kinematics: dict[str, Any],
    target_kinematics: dict[str, Any],
    *,
    ownership: ProfileDragReference,
    target_fraction_chord: float,
    replace_polar_residual: bool,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Return the complete local-reference correction for one load branch.

    A branch that owns only source ``Cd0`` sets ``replace_polar_residual=False``.
    A branch that also owns the full-angle polar residual sets it to ``True``;
    in that case the old residual is removed before the target residual is
    inserted.  This is replacement bookkeeping, never a second polar owner.
    """

    profile = profile_drag_reference_delta(
        owned_kinematics,
        target_kinematics,
        ownership=ownership,
        target_fraction_chord=target_fraction_chord,
        rho_kg_m3=rho_kg_m3,
        freestream_m_s=freestream_m_s,
        area_m2=area_m2,
    )
    phase = np.asarray(profile["phase"], dtype=float)
    delta_lift = np.asarray(profile["delta_lift_n"], dtype=float).copy()
    delta_drag = np.asarray(profile["delta_drag_n"], dtype=float).copy()
    polar_delta_lift = np.zeros_like(phase)
    polar_delta_drag = np.zeros_like(phase)
    if replace_polar_residual:
        for key, target in (
            ("delta_lift_n", polar_delta_lift),
            ("delta_drag_n", polar_delta_drag),
        ):
            owned = np.asarray(owned_kinematics[key], dtype=float)
            new = np.asarray(target_kinematics[key], dtype=float)
            if owned.shape != phase.shape or new.shape != phase.shape:
                raise ValueError("polar residual history does not match phase")
            if not np.all(np.isfinite(owned)) or not np.all(np.isfinite(new)):
                raise ValueError("polar residual history contains non-finite values")
            target[:] = new - owned
        delta_lift += polar_delta_lift
        delta_drag += polar_delta_drag

    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    delta_cl = delta_lift / q_area
    delta_cd = delta_drag / q_area
    out = dict(profile)
    out.update(
        delta_lift_n=delta_lift,
        delta_drag_n=delta_drag,
        delta_thrust_n=-delta_drag,
        delta_CL=delta_cl,
        delta_CD=delta_cd,
        delta_CT=-delta_cd,
        mean_delta_lift_n=float(np.mean(delta_lift)),
        mean_delta_drag_n=float(np.mean(delta_drag)),
        mean_delta_thrust_n=-float(np.mean(delta_drag)),
        mean_delta_CL=float(np.mean(delta_cl)),
        mean_delta_CD=float(np.mean(delta_cd)),
        mean_delta_CT=-float(np.mean(delta_cd)),
        polar_reference_replaced=bool(replace_polar_residual),
        polar_reference_delta_lift_n=polar_delta_lift,
        polar_reference_delta_drag_n=polar_delta_drag,
        semantics=(
            "reference replacement: target polar minus owned polar when enabled, "
            "plus Cd0*(unit_target-unit_owned); no second load owner"
        ),
    )
    return out


def rebase_constant_profile_drag_reference(
    history: dict[str, Any],
    owned_kinematics: dict[str, Any],
    target_kinematics: dict[str, Any],
    *,
    ownership: ProfileDragReference,
    target_fraction_chord: float,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Rebase exactly one already-owned constant profile-drag contribution."""

    phase = np.asarray(history["phase"], dtype=float)
    if phase.shape != np.asarray(owned_kinematics["phase"]).shape or not np.allclose(
        phase, owned_kinematics["phase"], atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("load history and owned profile-drag phases are not aligned")

    prior_reference = history.get("profile_drag_reference_fraction_chord")
    if prior_reference is not None and not np.isclose(
        float(prior_reference), ownership.fraction_chord, atol=1.0e-15, rtol=0.0
    ):
        raise ValueError("load history already declares a different profile reference")
    prior_coefficient = history.get("profile_drag_coefficient")
    if prior_coefficient is not None and not np.isclose(
        float(prior_coefficient), ownership.coefficient, atol=1.0e-15, rtol=0.0
    ):
        raise ValueError(
            "load history already declares a different profile coefficient"
        )

    delta = profile_drag_reference_delta(
        owned_kinematics,
        target_kinematics,
        ownership=ownership,
        target_fraction_chord=target_fraction_chord,
        rho_kg_m3=rho_kg_m3,
        freestream_m_s=freestream_m_s,
        area_m2=area_m2,
    )
    out = dict(history)

    # The disabled ledger is a strict identity path, including declared means.
    if ownership.coefficient == 0.0 or (
        np.array_equal(delta["delta_lift_n"], np.zeros_like(phase))
        and np.array_equal(delta["delta_drag_n"], np.zeros_like(phase))
    ):
        out.update(
            profile_drag_coefficient=ownership.coefficient,
            profile_drag_reference_fraction_chord=float(target_fraction_chord),
            profile_drag_rebase_applied=False,
            profile_drag_rebase_semantics=delta["semantics"],
        )
        return out

    lift = np.asarray(history["lift_n"], dtype=float) + delta["delta_lift_n"]
    if "drag_n" in history:
        base_drag = np.asarray(history["drag_n"], dtype=float)
    else:
        base_drag = -np.asarray(history["thrust_n"], dtype=float)
    drag = base_drag + delta["delta_drag_n"]
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    out.update(
        phase=phase.copy(),
        lift_n=lift,
        drag_n=drag,
        thrust_n=-drag,
        CL=lift / q_area,
        CD=drag / q_area,
        CT=-drag / q_area,
        mean_lift_n=float(np.mean(lift)),
        mean_drag_n=float(np.mean(drag)),
        mean_thrust_n=-float(np.mean(drag)),
        mean_CL=float(np.mean(lift) / q_area),
        mean_CD=float(np.mean(drag) / q_area),
        mean_CT=-float(np.mean(drag) / q_area),
        profile_drag_coefficient=ownership.coefficient,
        profile_drag_reference_fraction_chord=float(target_fraction_chord),
        profile_drag_rebase_applied=True,
        profile_drag_rebase_delta_lift_n=delta["delta_lift_n"],
        profile_drag_rebase_delta_drag_n=delta["delta_drag_n"],
        profile_drag_rebase_semantics=delta["semantics"],
    )
    return out


def rebase_local_velocity_reference(
    history: dict[str, Any],
    owned_kinematics: dict[str, Any],
    target_kinematics: dict[str, Any],
    *,
    ownership: ProfileDragReference,
    target_fraction_chord: float,
    replace_polar_residual: bool,
    rho_kg_m3: float,
    freestream_m_s: float,
    area_m2: float,
) -> dict[str, Any]:
    """Rebase a profile-only or polar-plus-profile local load branch."""

    # Reuse the strict ownership, reference, and phase guards.
    base = rebase_constant_profile_drag_reference(
        history,
        owned_kinematics,
        target_kinematics,
        ownership=ownership,
        target_fraction_chord=target_fraction_chord,
        rho_kg_m3=rho_kg_m3,
        freestream_m_s=freestream_m_s,
        area_m2=area_m2,
    )
    if not replace_polar_residual:
        base["polar_reference_replaced"] = False
        return base

    complete = local_velocity_reference_delta(
        owned_kinematics,
        target_kinematics,
        ownership=ownership,
        target_fraction_chord=target_fraction_chord,
        replace_polar_residual=True,
        rho_kg_m3=rho_kg_m3,
        freestream_m_s=freestream_m_s,
        area_m2=area_m2,
    )
    # ``base`` already contains the profile correction.  Add only the polar
    # replacement here so each line item remains singly owned.
    polar_lift = np.asarray(complete["polar_reference_delta_lift_n"], dtype=float)
    polar_drag = np.asarray(complete["polar_reference_delta_drag_n"], dtype=float)
    if np.array_equal(polar_lift, np.zeros_like(polar_lift)) and np.array_equal(
        polar_drag, np.zeros_like(polar_drag)
    ):
        base["polar_reference_replaced"] = False
        return base
    lift = np.asarray(base["lift_n"], dtype=float) + polar_lift
    drag = np.asarray(base["drag_n"], dtype=float) + polar_drag
    q_area = 0.5 * rho_kg_m3 * freestream_m_s**2 * area_m2
    base.update(
        lift_n=lift,
        drag_n=drag,
        thrust_n=-drag,
        CL=lift / q_area,
        CD=drag / q_area,
        CT=-drag / q_area,
        mean_lift_n=float(np.mean(lift)),
        mean_drag_n=float(np.mean(drag)),
        mean_thrust_n=-float(np.mean(drag)),
        mean_CL=float(np.mean(lift) / q_area),
        mean_CD=float(np.mean(drag) / q_area),
        mean_CT=-float(np.mean(drag) / q_area),
        polar_reference_replaced=True,
        polar_reference_delta_lift_n=polar_lift,
        polar_reference_delta_drag_n=polar_drag,
        local_reference_rebase_semantics=complete["semantics"],
    )
    return base
