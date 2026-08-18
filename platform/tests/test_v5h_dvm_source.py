from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
import os
import re
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h-dvm-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h-dvm-mpl-cache")

import numpy as np
import pytest

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    ALLOWED_THRESHOLD_SOURCE_ROLES,
    CANONICAL_BLOCKER,
    DVMSourceEvent,
    DVMSourcePlacement,
    EVENT_CHAIN_DOMAIN,
    SOURCE_BACKEND_ID,
    SOURCE_INTERFACE_ID,
    SOURCE_PLACEMENT_SCHEMA_ID,
    V5hDVMSource,
    validate_dvm_source_event,
)


def _threshold(
    *,
    section_family: str = "generic 2-percent thin flat plate",
    source: str = "Ramesh LDVM v2.5 published source input",
    source_role: str = "published_model_parameter",
) -> LESPThreshold:
    return LESPThreshold(
        value=0.18,
        section_family=section_family,
        reynolds=30_000.0,
        source=source,
        source_role=source_role,
    )


def _source(
    *,
    physical_section_id: str = "wing-left:section-004",
    physical_strip_id: str = "wing-left:strip-004",
    settings: LDVMSectionSettings | None = None,
    threshold: LESPThreshold | None = None,
    geometry_identity: str = "explicit zero-camber flat-plate surrogate",
) -> V5hDVMSource:
    return V5hDVMSource(
        physical_section_id=physical_section_id,
        physical_strip_id=physical_strip_id,
        geometry_identity=geometry_identity,
        reference_speed_m_per_s=3.0,
        reference_chord_m=0.25,
        zero_camber_surrogate=True,
        delta_time_convective=0.02,
        pivot_fraction_chord=0.25,
        threshold=threshold or _threshold(),
        settings=settings or LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )


class _ExplodingInput:
    def __float__(self) -> float:
        raise AssertionError("disabled source inspected a numeric input")

    def __array__(self) -> np.ndarray:
        raise AssertionError("disabled source converted an input")


def _freeze(value: Any) -> Any:
    """Deep deterministic snapshot for the backend's mutable state."""

    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _all_manifest_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_manifest_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(_all_manifest_keys(item))
    return keys


def _assert_half_step_placement(
    placement: DVMSourcePlacement, delta_time_convective: float
) -> None:
    edge = placement.edge_anchor_position_over_chord_backend_world
    birth = placement.birth_position_over_chord_backend_world
    displacement = placement.birth_displacement_from_edge_over_chord_backend_world
    q_birth = placement.q_birth_over_u_backend_world
    q_kinematic = placement.q_kinematic_over_u_backend_world
    q_old_wake = placement.q_old_wake_over_u_backend_world
    q_provisional = placement.q_provisional_tev_over_u_backend_world
    assert edge is not None and birth is not None
    assert displacement is not None and q_birth is not None
    assert q_kinematic is not None and q_old_wake is not None
    assert q_provisional is not None
    np.testing.assert_array_equal(
        q_birth,
        np.asarray(q_kinematic) + np.asarray(q_old_wake) + np.asarray(q_provisional),
    )
    np.testing.assert_allclose(
        birth,
        np.asarray(edge) + 0.5 * np.asarray(q_birth) * delta_time_convective,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_array_equal(displacement, np.asarray(birth) - np.asarray(edge))


def _direct_vatistas2(
    target: tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    core: float,
) -> tuple[float, float]:
    if x.size == 0:
        return (0.0, 0.0)
    rx = target[0] - x
    ry = target[1] - y
    r2 = rx * rx + ry * ry
    factor = gamma / (2.0 * np.pi) / np.sqrt(r2 * r2 + core**4)
    return (float(np.sum(factor * -ry)), float(np.sum(factor * rx)))


@pytest.mark.parametrize(
    ("section_family", "source", "source_role"),
    [
        ("unknown", "published source", "published_model_parameter"),
        ("thin flat plate", "TBD", "published_model_parameter"),
        ("thin flat plate", "published source", "unspecified"),
        ("thin flat plate default", "published source", "published_model_parameter"),
    ],
)
def test_unknown_section_or_threshold_provenance_fails_closed(
    section_family: str,
    source: str,
    source_role: str,
) -> None:
    with pytest.raises(ValueError, match="unknown/default provenance"):
        _source(
            threshold=_threshold(
                section_family=section_family,
                source=source,
                source_role=source_role,
            )
        )


@pytest.mark.parametrize(
    "source_role",
    [
        "target_force_fit",
        "observation_fit",
        "paper-parameter mapping hypothesis",
        "published flat-plate transfer hypothesis; no Baik force fit",
    ],
)
def test_threshold_source_role_is_a_strict_source_safe_allowlist(
    source_role: str,
) -> None:
    assert "published_model_parameter" in ALLOWED_THRESHOLD_SOURCE_ROLES
    with pytest.raises(ValueError, match="not source-safe"):
        _source(threshold=_threshold(source_role=source_role))


def test_threshold_requires_real_section_reynolds_and_source_contract() -> None:
    kwargs = dict(
        physical_section_id="wing-left:section-004",
        physical_strip_id="wing-left:strip-004",
        geometry_identity="explicit zero-camber flat-plate surrogate",
        reference_speed_m_per_s=3.0,
        reference_chord_m=0.25,
        zero_camber_surrogate=True,
        delta_time_convective=0.02,
        pivot_fraction_chord=0.25,
    )
    with pytest.raises(ValueError, match="explicit LESPThreshold"):
        V5hDVMSource(threshold=None, **kwargs)  # type: ignore[arg-type]
    boolean_threshold = LESPThreshold(
        value=True,  # type: ignore[arg-type]
        section_family="thin flat plate",
        reynolds=30_000.0,
        source="published source",
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        V5hDVMSource(threshold=boolean_threshold, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_speed_m_per_s", True),
        ("reference_chord_m", np.inf),
        ("delta_time_convective", 0.0),
        ("pivot_fraction_chord", 1.1),
    ],
)
def test_constructor_scalars_are_strict(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "physical_section_id": "wing-left:section-004",
        "physical_strip_id": "wing-left:strip-004",
        "geometry_identity": "explicit zero-camber flat-plate surrogate",
        "reference_speed_m_per_s": 3.0,
        "reference_chord_m": 0.25,
        "zero_camber_surrogate": True,
        "delta_time_convective": 0.02,
        "pivot_fraction_chord": 0.25,
        "threshold": _threshold(),
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        V5hDVMSource(**kwargs)  # type: ignore[arg-type]


def test_settings_integer_fields_cannot_be_silently_truncated() -> None:
    settings = LDVMSectionSettings(ndiv=24.5, naterm=8, max_wake_steps=48)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings.ndiv must be an integer"):
        _source(settings=settings)


def test_physical_section_and_strip_lineage_are_required_and_distinct() -> None:
    with pytest.raises(ValueError, match="physical_section_id must be explicit"):
        _source(physical_section_id="")
    with pytest.raises(ValueError, match="must be distinct"):
        _source(
            physical_section_id="wing-left:strip-004",
            physical_strip_id="wing-left:strip-004",
        )


def test_multi_strip_lineage_ids_do_not_alias() -> None:
    left = _source(physical_strip_id="wing-left:strip-004")
    right = _source(physical_strip_id="wing-left:strip-005")
    event_left = left.step(0.0, 0.0, 0.0)
    event_right = right.step(0.0, 0.0, 0.0)

    assert event_left.lineage.physical_strip_id != event_right.lineage.physical_strip_id
    assert (
        event_left.lineage.section_lineage_id != event_right.lineage.section_lineage_id
    )
    assert (
        event_left.lineage.newborn_tev_source_id
        != event_right.lineage.newborn_tev_source_id
    )


def test_zero_camber_is_explicit_and_cannot_claim_sd7003_geometry() -> None:
    source = _source()
    assert source.provenance.geometry_role == "explicit_zero_camber_surrogate"
    assert len(source.provenance.geometry_hash_sha256) == 64

    with pytest.raises(ValueError, match="non-flat section_family"):
        _source(
            threshold=_threshold(section_family="SD7003 airfoil"),
            geometry_identity="explicit zero-camber surrogate for SD7003",
        )
    with pytest.raises(ValueError, match="affirmatively say flat plate"):
        _source(geometry_identity="SD7003")
    with pytest.raises(ValueError, match="non-flat section_family"):
        _source(
            threshold=_threshold(section_family="non-flat SD7003 airfoil"),
            geometry_identity="explicit zero-camber flat-plate surrogate",
        )
    with pytest.raises(ValueError, match="non-flat section_family"):
        _source(
            threshold=_threshold(section_family="not a flat plate"),
            geometry_identity="explicit zero-camber flat-plate surrogate",
        )


def test_paired_camber_geometry_is_applied_and_hashed() -> None:
    theta = np.linspace(0.0, np.pi, 24)
    zc = 0.02 * np.sin(theta)
    dzc = 0.04 * np.cos(theta)
    source = V5hDVMSource(
        physical_section_id="wing-left:section-004",
        physical_strip_id="wing-left:strip-004",
        geometry_identity="audited SD7003 mean-camber spline revision 1",
        reference_speed_m_per_s=3.0,
        reference_chord_m=0.25,
        zero_camber_surrogate=False,
        camber_z_over_chord=zc,
        camber_slope=dzc,
        delta_time_convective=0.02,
        pivot_fraction_chord=0.25,
        threshold=_threshold(section_family="SD7003 airfoil"),
        settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48),
    )
    assert source.provenance.geometry_role.startswith("explicit_paired")
    np.testing.assert_array_equal(source._model.zc, zc)
    np.testing.assert_array_equal(source._model.dzc, dzc)
    assert source._model.source_parity is True

    with pytest.raises(ValueError, match="paired camber"):
        V5hDVMSource(
            physical_section_id="wing-left:section-004",
            physical_strip_id="wing-left:strip-004",
            geometry_identity="audited SD7003 mean-camber spline revision 1",
            reference_speed_m_per_s=3.0,
            reference_chord_m=0.25,
            zero_camber_surrogate=False,
            camber_z_over_chord=zc,
            delta_time_convective=0.02,
            pivot_fraction_chord=0.25,
            threshold=_threshold(section_family="SD7003 airfoil"),
            settings=LDVMSectionSettings(ndiv=24, naterm=8),
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        [True, *([0.0] * 23)],
        ["0.0", *([0.0] * 23)],
    ],
)
def test_paired_camber_rejects_boolean_and_numeric_string_values(
    bad_value: list[object],
) -> None:
    with pytest.raises(ValueError, match="finite real values"):
        V5hDVMSource(
            physical_section_id="wing-left:section-004",
            physical_strip_id="wing-left:strip-004",
            geometry_identity="audited SD7003 mean-camber spline revision 1",
            reference_speed_m_per_s=3.0,
            reference_chord_m=0.25,
            zero_camber_surrogate=False,
            camber_z_over_chord=bad_value,
            camber_slope=np.zeros(24),
            delta_time_convective=0.02,
            pivot_fraction_chord=0.25,
            threshold=_threshold(section_family="SD7003 airfoil"),
            settings=LDVMSectionSettings(ndiv=24, naterm=8),
        )


@pytest.mark.parametrize(
    ("speed", "chord", "dt", "core_ratio"),
    [
        (1.0e308, 1.0e308, 0.02, 1.3),
        (1.0e-300, 1.0e-300, 0.02, 1.3),
        (3.0, 0.25, 1.0e308, 1.0e308),
        (3.0, 0.25, 1.0e-300, 1.0e-300),
    ],
)
def test_derived_scales_fail_closed_on_overflow_or_underflow(
    speed: float,
    chord: float,
    dt: float,
    core_ratio: float,
) -> None:
    settings = LDVMSectionSettings(
        ndiv=24,
        naterm=8,
        max_wake_steps=48,
        core_radius_chord=None,
        core_radius_time_step_ratio=core_ratio,
    )
    with pytest.raises(ValueError, match="scale|core radius"):
        V5hDVMSource(
            physical_section_id="wing-left:section-004",
            physical_strip_id="wing-left:strip-004",
            geometry_identity="explicit zero-camber flat-plate surrogate",
            reference_speed_m_per_s=speed,
            reference_chord_m=chord,
            zero_camber_surrogate=True,
            delta_time_convective=dt,
            pivot_fraction_chord=0.25,
            threshold=_threshold(),
            settings=settings,
        )


def test_disabled_is_deeply_input_blind_and_does_not_advance_state() -> None:
    source = _source()
    model_snapshot = _freeze(source._model.__dict__)
    source_snapshot = (
        source.step_count,
        source._last_step_active,
        source._ever_lev_shed,
        source._poisoned,
    )
    event = source.step(
        _ExplodingInput(),
        _ExplodingInput(),
        _ExplodingInput(),
        enabled=False,
        delta_time_convective=_ExplodingInput(),
    )

    assert event.enabled is False
    assert event.status == "not_evaluated_disabled"
    assert event.delta_time_convective is None
    assert event.a0_pre is None and event.a0_post is None
    assert event.lesp_signed_target is None
    assert event.lesp_constraint_residual is None
    assert event.lev_birth_mode == "disabled"
    assert event.lesp_active is False and event.restart is False
    assert event.gamma_lev_new_over_u_c == 0.0
    assert not np.signbit(event.gamma_lev_new_over_u_c)
    assert event.gamma_tev_new_solved_over_u_c == 0.0
    assert event.gamma_tev_new_persisted_over_u_c == 0.0
    assert event.kelvin_residual_over_u_c == 0.0 and event.kelvin_ledger is None
    assert event.lev_birth_position_over_chord_backend_world is None
    assert event.tev_birth_position_over_chord_backend_world is None
    for family, placement in (
        ("lev", event.lev_placement),
        ("tev", event.tev_placement),
    ):
        assert placement == DVMSourcePlacement(
            schema_id=SOURCE_PLACEMENT_SCHEMA_ID,
            vortex_family=family,
            placement_mode="disabled",
            edge_anchor_position_over_chord_backend_world=None,
            birth_position_over_chord_backend_world=None,
            birth_displacement_from_edge_over_chord_backend_world=None,
            q_birth_over_u_backend_world=None,
            q_kinematic_over_u_backend_world=None,
            q_old_wake_over_u_backend_world=None,
            q_provisional_tev_over_u_backend_world=None,
            continuous_parent_source_id=None,
            continuous_parent_position_over_chord_backend_world=None,
            used_for_topology_eligible=False,
        )
    assert event.lineage.source_step_index is None
    assert event.lineage.newborn_tev_source_id is None
    assert event.lineage.persistent_history_exported is False
    assert model_snapshot == _freeze(source._model.__dict__)
    assert source_snapshot == (
        source.step_count,
        source._last_step_active,
        source._ever_lev_shed,
        source._poisoned,
    )


def test_first_solved_tev_is_explicitly_zeroed_for_persistence() -> None:
    source = _source()
    event = source.step(np.deg2rad(5.0), 0.0, 0.0)
    ledger = event.kelvin_ledger
    assert ledger is not None
    assert ledger.first_tev_zeroed is True
    assert np.isfinite(ledger.gamma_tev_new_te_only_provisional)
    assert event.gamma_tev_new_solved_over_u_c != 0.0
    assert event.gamma_tev_new_persisted_over_u_c == 0.0
    assert source._model.tg[-1] == 0.0
    assert ledger.tev_solved_to_persisted_delta == -event.gamma_tev_new_solved_over_u_c
    assert "zeroed" in event.lineage.newborn_tev_role


def test_zero_attached_step_is_exact_no_source_reduction() -> None:
    source = _source()
    event = source.step(0.0, 0.0, 0.0)

    assert event.enabled is True
    assert event.lesp_active is False
    assert event.lev_birth_mode == "none"
    assert event.restart is False
    assert event.a0_pre == event.a0_post == 0.0
    assert event.lesp_signed_target == event.a0_pre
    assert event.lesp_constraint_residual == 0.0
    assert event.gamma_lev_new_over_u_c == 0.0
    assert not np.signbit(event.gamma_lev_new_over_u_c)
    assert event.gamma_tev_new_solved_over_u_c == 0.0
    assert event.gamma_tev_new_persisted_over_u_c == 0.0
    assert event.kelvin_ledger is not None
    assert event.kelvin_ledger.gamma_tev_new_te_only_provisional == 0.0
    assert event.kelvin_residual_over_u_c == 0.0
    assert event.lev_birth_position_over_chord_backend_world is None
    assert event.tev_birth_position_over_chord_backend_world is not None
    assert event.lineage.persistent_tev_count_before == 0
    assert event.lineage.persistent_tev_count_after == 1


def test_enabled_no_shed_advances_attached_te_and_uses_per_step_dt() -> None:
    source = _source()
    first = source.step(np.deg2rad(5.0), 0.0, 0.0, delta_time_convective=0.019)
    second = source.step(np.deg2rad(5.0), 0.0, 0.0)

    assert first.lesp_active is False
    assert first.a0_pre == first.a0_post
    assert first.delta_time_convective == 0.019
    assert first.gamma_tev_new_persisted_over_u_c == 0.0
    assert (
        second.gamma_tev_new_persisted_over_u_c == second.gamma_tev_new_solved_over_u_c
    )
    assert second.gamma_tev_new_persisted_over_u_c != 0.0
    assert source.step_count == 2
    assert len(source._model.tg) == 2


def test_birth_positions_are_captured_from_actual_backend_calls() -> None:
    source = _source()
    actual_calls: list[tuple[float, float]] = []
    original_wcol = source._model._wcol

    def audited_wcol(px: float, py: float) -> np.ndarray:
        actual_calls.append((float(px), float(py)))
        return original_wcol(px, py)

    source._model._wcol = audited_wcol
    event = source.step(np.deg2rad(35.0), 0.0, 0.0)

    assert event.lesp_active is True
    assert len(actual_calls) == 2
    assert event.tev_birth_position_over_chord_backend_world == actual_calls[0]
    assert event.lev_birth_position_over_chord_backend_world == actual_calls[1]
    assert event.tev_birth_position_over_chord_backend_world != (
        source._model.tx[-1],
        source._model.ty[-1],
    )
    assert event.lev_birth_position_over_chord_backend_world != (
        source._model.lx[-1],
        source._model.ly[-1],
    )
    assert event.gamma_tev_new_persisted_over_u_c == source._model.tg[-1]
    assert event.gamma_lev_new_over_u_c == source._model.lg[-1]
    assert event.tev_placement.birth_position_over_chord_backend_world == (
        event.tev_birth_position_over_chord_backend_world
    )
    assert event.lev_placement.birth_position_over_chord_backend_world == (
        event.lev_birth_position_over_chord_backend_world
    )
    assert event.tev_placement.edge_anchor_position_over_chord_backend_world == (
        tuple(float(value) for value in source._model._world(source._model.c))
    )
    assert event.lev_placement.edge_anchor_position_over_chord_backend_world == (
        tuple(float(value) for value in source._model._world(0.0))
    )


def test_step_one_and_two_placements_are_attested_without_topology_aliasing() -> None:
    source = _source()
    first = source.step(np.deg2rad(35.0), 0.0, 0.0)
    first_tev_current_point = (source._model.tx[-1], source._model.ty[-1])
    first_lev_current_point = (source._model.lx[-1], source._model.ly[-1])
    second = source.step(np.deg2rad(35.0), 0.0, 0.0)

    assert first.tev_placement.placement_mode == "first"
    assert first.lev_placement.placement_mode == "first"
    assert first.tev_placement.used_for_topology_eligible is False
    assert first.lev_placement.used_for_topology_eligible is True
    _assert_half_step_placement(first.tev_placement, first.delta_time_convective)
    _assert_half_step_placement(first.lev_placement, first.delta_time_convective)

    for family, placement, parent_id, parent_point in (
        (
            "tev",
            second.tev_placement,
            first.lineage.newborn_tev_source_id,
            first_tev_current_point,
        ),
        (
            "lev",
            second.lev_placement,
            first.lineage.newborn_lev_source_id,
            first_lev_current_point,
        ),
    ):
        assert placement.schema_id == SOURCE_PLACEMENT_SCHEMA_ID
        assert placement.vortex_family == family
        assert placement.placement_mode == "continuous"
        assert placement.q_birth_over_u_backend_world is None
        assert placement.q_kinematic_over_u_backend_world is None
        assert placement.q_old_wake_over_u_backend_world is None
        assert placement.q_provisional_tev_over_u_backend_world is None
        assert placement.continuous_parent_source_id == parent_id
        assert (
            placement.continuous_parent_position_over_chord_backend_world
            == parent_point
        )
        assert placement.used_for_topology_eligible is False
        edge = np.asarray(
            placement.edge_anchor_position_over_chord_backend_world, dtype=float
        )
        parent = np.asarray(parent_point, dtype=float)
        birth = np.asarray(placement.birth_position_over_chord_backend_world)
        np.testing.assert_allclose(
            birth, edge + (parent - edge) / 3.0, rtol=0.0, atol=2.0e-15
        )


def test_inactive_and_restart_lev_placements_are_unambiguous() -> None:
    source = _source()
    source.step(np.deg2rad(35.0), 0.0, 0.0)
    source.step(np.deg2rad(35.0), 0.0, 0.0)
    inactive = source.step(0.0, 0.0, 0.0)
    restart = source.step(np.deg2rad(35.0), 0.0, 0.0)

    inactive_placement = inactive.lev_placement
    assert inactive_placement.placement_mode == "inactive"
    assert inactive_placement.edge_anchor_position_over_chord_backend_world is not None
    assert inactive_placement.birth_position_over_chord_backend_world is None
    assert (
        inactive_placement.birth_displacement_from_edge_over_chord_backend_world is None
    )
    assert inactive_placement.q_birth_over_u_backend_world is None
    assert inactive_placement.q_kinematic_over_u_backend_world is None
    assert inactive_placement.q_old_wake_over_u_backend_world is None
    assert inactive_placement.q_provisional_tev_over_u_backend_world is None
    assert inactive_placement.continuous_parent_source_id is None
    assert inactive_placement.used_for_topology_eligible is False

    restart_placement = restart.lev_placement
    assert restart_placement.placement_mode == "restart"
    assert restart_placement.continuous_parent_source_id is None
    assert restart_placement.continuous_parent_position_over_chord_backend_world is None
    assert restart_placement.used_for_topology_eligible is True
    _assert_half_step_placement(restart_placement, restart.delta_time_convective)


def test_first_placement_components_use_te_only_provisional_strength() -> None:
    source = _source()
    event = source.step(
        np.deg2rad(35.0),
        0.0,
        0.0,
        delta_time_convective=0.013,
    )
    placement = event.lev_placement
    ledger = event.kelvin_ledger
    assert ledger is not None
    assert placement.placement_mode == "first"
    assert placement.q_kinematic_over_u_backend_world == (1.0, 0.0)
    assert placement.q_old_wake_over_u_backend_world == (0.0, 0.0)
    assert ledger.gamma_tev_new_te_only_provisional != ledger.gamma_tev_new_solved

    edge = placement.edge_anchor_position_over_chord_backend_world
    tev_birth = event.tev_birth_position_over_chord_backend_world
    assert edge is not None and tev_birth is not None
    expected_provisional = _direct_vatistas2(
        edge,
        np.asarray([tev_birth[0]]),
        np.asarray([tev_birth[1]]),
        np.asarray([ledger.gamma_tev_new_te_only_provisional]),
        source.provenance.resolved_core_radius_chord,
    )
    np.testing.assert_array_equal(
        placement.q_provisional_tev_over_u_backend_world,
        expected_provisional,
    )
    _assert_half_step_placement(placement, 0.013)


def test_restart_components_recompute_frozen_old_wake_with_variable_dt() -> None:
    source = _source()
    source.step(np.deg2rad(35.0), 0.0, 0.0, delta_time_convective=0.011)
    source.step(np.deg2rad(35.0), 0.0, 0.0, delta_time_convective=0.017)
    source.step(0.0, 0.0, 0.0, delta_time_convective=0.019)
    old_tev = (
        np.asarray(source._model.tx, dtype=float).copy(),
        np.asarray(source._model.ty, dtype=float).copy(),
        np.asarray(source._model.tg, dtype=float).copy(),
    )
    old_lev = (
        np.asarray(source._model.lx, dtype=float).copy(),
        np.asarray(source._model.ly, dtype=float).copy(),
        np.asarray(source._model.lg, dtype=float).copy(),
    )

    event = source.step(
        np.deg2rad(35.0),
        0.12,
        0.03,
        delta_time_convective=0.027,
    )
    placement = event.lev_placement
    ledger = event.kelvin_ledger
    edge = placement.edge_anchor_position_over_chord_backend_world
    tev_birth = event.tev_birth_position_over_chord_backend_world
    assert ledger is not None and edge is not None and tev_birth is not None
    assert placement.placement_mode == "restart"
    expected_kinematic = (
        1.0 - 0.25 * np.sin(np.deg2rad(35.0)) * 0.12,
        -0.03 - 0.25 * np.cos(np.deg2rad(35.0)) * 0.12,
    )
    np.testing.assert_array_equal(
        placement.q_kinematic_over_u_backend_world,
        expected_kinematic,
    )
    expected_old_tev = _direct_vatistas2(
        edge, *old_tev, source.provenance.resolved_core_radius_chord
    )
    expected_old_lev = _direct_vatistas2(
        edge, *old_lev, source.provenance.resolved_core_radius_chord
    )
    expected_old = (
        expected_old_tev[0] + expected_old_lev[0],
        expected_old_tev[1] + expected_old_lev[1],
    )
    expected_provisional = _direct_vatistas2(
        edge,
        np.asarray([tev_birth[0]]),
        np.asarray([tev_birth[1]]),
        np.asarray([ledger.gamma_tev_new_te_only_provisional]),
        source.provenance.resolved_core_radius_chord,
    )
    np.testing.assert_array_equal(
        placement.q_old_wake_over_u_backend_world, expected_old
    )
    np.testing.assert_array_equal(
        placement.q_provisional_tev_over_u_backend_world,
        expected_provisional,
    )
    _assert_half_step_placement(placement, 0.027)


def test_signed_lev_placement_velocity_tracks_positive_and_negative_incidence() -> None:
    positive_source = _source(physical_strip_id="wing-left:strip-positive")
    negative_source = _source(physical_strip_id="wing-left:strip-negative")
    positive = positive_source.step(np.deg2rad(35.0), 0.0, 0.0)
    negative = negative_source.step(np.deg2rad(-35.0), 0.0, 0.0)

    positive_q = positive.lev_placement.q_birth_over_u_backend_world
    negative_q = negative.lev_placement.q_birth_over_u_backend_world
    assert positive_q is not None and negative_q is not None
    assert positive_q[0] == pytest.approx(negative_q[0], abs=2.0e-15)
    assert positive_q[1] == pytest.approx(-negative_q[1], abs=2.0e-15)
    assert positive_q[1] < 0.0 < negative_q[1]
    _assert_half_step_placement(positive.lev_placement, positive.delta_time_convective)
    _assert_half_step_placement(negative.lev_placement, negative.delta_time_convective)

    positive_source.step(0.0, 0.0, 0.0)
    negative_source.step(0.0, 0.0, 0.0)
    positive_restart = positive_source.step(np.deg2rad(35.0), 0.0, 0.0)
    negative_restart = negative_source.step(np.deg2rad(-35.0), 0.0, 0.0)
    positive_restart_q = positive_restart.lev_placement.q_birth_over_u_backend_world
    negative_restart_q = negative_restart.lev_placement.q_birth_over_u_backend_world
    assert positive_restart_q is not None and negative_restart_q is not None
    assert positive_restart.lev_placement.placement_mode == "restart"
    assert negative_restart.lev_placement.placement_mode == "restart"
    assert positive_restart_q[0] == pytest.approx(negative_restart_q[0], abs=2.0e-15)
    assert positive_restart_q[1] == pytest.approx(-negative_restart_q[1], abs=2.0e-15)


def test_first_continuous_and_restart_shedding_are_distinct() -> None:
    source = _source()
    first = source.step(np.deg2rad(35.0), 0.0, 0.0)
    continuous = source.step(np.deg2rad(35.0), 0.0, 0.0)
    attached = source.step(0.0, 0.0, 0.0)
    restart = source.step(np.deg2rad(35.0), 0.0, 0.0)

    assert first.lev_birth_mode == "first" and first.restart is False
    assert continuous.lev_birth_mode == "continuous" and continuous.restart is False
    assert attached.lev_birth_mode == "none" and attached.restart is False
    assert restart.lev_birth_mode == "restart" and restart.restart is True
    assert (
        first.lineage.newborn_lev_source_id != continuous.lineage.newborn_lev_source_id
    )
    assert restart.lineage.parent_state_step_index == 3


def test_positive_and_negative_lesp_targets_are_signed_and_recomputable() -> None:
    positive = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    negative = _source(physical_strip_id="wing-left:strip-005").step(
        np.deg2rad(-35.0), 0.0, 0.0
    )

    for event in (positive, negative):
        assert event.lesp_active is True
        assert event.lesp_signed_target == np.sign(event.a0_pre) * event.lesp_critical
        assert event.lesp_constraint_residual == pytest.approx(
            event.a0_post - event.lesp_signed_target,
            abs=1.0e-15,
        )
        assert abs(event.lesp_constraint_residual) < 2.0e-14
    assert positive.lesp_signed_target == -negative.lesp_signed_target


def test_kelvin_ledger_is_independently_recomputable() -> None:
    source = _source()
    events = [
        source.step(np.deg2rad(5.0), 0.0, 0.0),
        source.step(np.deg2rad(35.0), 0.0, 0.0),
        source.step(np.deg2rad(35.0), 0.0, 0.0),
        source.step(0.0, 0.0, 0.0),
    ]
    for event in events:
        ledger = event.kelvin_ledger
        assert ledger is not None
        solve_residual = (
            -ledger.gamma_bound_post
            + ledger.gamma_old_tev_persisted
            + ledger.gamma_old_lev_persisted
            + ledger.gamma_deleted_before
            + ledger.gamma_tev_new_solved
            + ledger.gamma_lev_new_solved
        )
        persistence_residual = (
            ledger.gamma_tev_persisted_after
            + ledger.gamma_lev_persisted_after
            + ledger.gamma_deleted_after
            - (
                ledger.gamma_old_tev_persisted
                + ledger.gamma_old_lev_persisted
                + ledger.gamma_deleted_before
                + ledger.gamma_tev_new_persisted
                + ledger.gamma_lev_new_persisted
            )
        )
        assert ledger.gamma_deleted_delta == pytest.approx(
            ledger.gamma_deleted_after - ledger.gamma_deleted_before,
            abs=1.0e-16,
        )
        assert ledger.kelvin_solve_residual == pytest.approx(
            solve_residual, abs=1.0e-16
        )
        assert ledger.persistence_residual == pytest.approx(
            persistence_residual, abs=1.0e-16
        )
        assert event.kelvin_residual_over_u_c == ledger.kelvin_solve_residual
        if not event.lesp_active:
            assert (
                ledger.gamma_tev_new_te_only_provisional == ledger.gamma_tev_new_solved
            )
        assert abs(solve_residual) < 2.0e-13
        assert abs(persistence_residual) < 2.0e-13


def test_w2_handoff_public_tev_strength_remains_final_coupled_solution() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    ledger = event.kelvin_ledger
    assert ledger is not None and event.lesp_active
    assert ledger.gamma_tev_new_te_only_provisional != ledger.gamma_tev_new_solved
    # The W2 source aggregate consumes this public field.  Provisional TEV is
    # placement evidence only; circulation injection remains the coupled solve.
    assert event.gamma_tev_new_solved_over_u_c == ledger.gamma_tev_new_solved
    assert (
        event.gamma_tev_new_solved_over_u_c != ledger.gamma_tev_new_te_only_provisional
    )


def test_trimmed_histories_close_deleted_circulation_ledger() -> None:
    source = _source(settings=LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=8))
    events = [source.step(np.deg2rad(35.0), 0.0, 0.0) for _ in range(11)]
    trimmed = [
        event
        for event in events
        if event.kelvin_ledger is not None
        and event.kelvin_ledger.gamma_deleted_delta != 0.0
    ]

    assert trimmed
    assert events[-1].lineage.persistent_tev_count_after == 8
    assert events[-1].lineage.persistent_lev_count_after == 8
    for event in events[8:]:
        ledger = event.kelvin_ledger
        assert ledger is not None
        assert ledger.gamma_deleted_after == pytest.approx(
            ledger.gamma_deleted_before + ledger.gamma_deleted_delta,
            abs=1.0e-16,
        )
        assert abs(ledger.persistence_residual) < 2.0e-13


def test_units_sign_frame_and_scaling_are_explicit() -> None:
    provenance = _source().provenance
    assert provenance.circulation_units == "Gamma/(U_ref*c_ref)"
    assert provenance.circulation_scale_u_times_c_m2_per_s == 0.75
    assert provenance.position_units == "(x/c_ref,z/c_ref)"
    assert provenance.position_scale_chord_m == 0.25
    assert "x downstream" in provenance.position_frame
    assert "counter-clockwise" in provenance.circulation_sign
    assert "0.5*U*dt" in provenance.tev_birth_law
    assert "same-step provisional TEV" in provenance.lev_birth_law
    assert "pre-convection" in provenance.birth_time_layer
    assert "no span coordinate" in provenance.dimensionalization_limitations


def test_event_manifest_recursively_has_no_load_or_case_output() -> None:
    source = _source()
    event = source.step(np.deg2rad(35.0), 0.0, 0.0)
    field_names = {
        field.name.casefold()
        for schema in (DVMSourceEvent, DVMSourcePlacement)
        for field in fields(schema)
    }
    forbidden_exact = {
        "cl",
        "cd",
        "cn",
        "cs",
        "lift",
        "drag",
        "force",
        "load",
        "moment",
        "normal",
        "axial",
        "suction",
        "impulse",
        "polar",
        "correction",
        "case_id",
    }
    assert field_names.isdisjoint(forbidden_exact)
    assert "birth_mode" not in field_names

    manifest = event.manifest()
    keys = _all_manifest_keys(manifest)
    for key in keys:
        tokens = set(re.findall(r"[a-z0-9]+", key.casefold()))
        assert tokens.isdisjoint(forbidden_exact), key
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    assert manifest["provenance"]["canonical"] is False
    assert "D0" in CANONICAL_BLOCKER
    assert "174 LEVs" in manifest["provenance"]["bottom_model_parity"]
    assert manifest["provenance"]["source_parity"] is True
    assert manifest["provenance"]["source_solver"] == "clean_linear"
    assert "sole load owner" in manifest["provenance"]["ownership_scope"]
    assert manifest["provenance"]["observation_access"] == "none"
    assert manifest["provenance"]["target_case_branch"] == "none"
    assert manifest["lineage"]["persistent_history_exported"] is False
    assert SOURCE_INTERFACE_ID.endswith("-v3")
    assert SOURCE_PLACEMENT_SCHEMA_ID.endswith("-v3")
    assert SOURCE_BACKEND_ID.endswith("-provisional-tev-v3")
    assert EVENT_CHAIN_DOMAIN.endswith("-v3")
    assert manifest["lev_placement"]["schema_id"] == SOURCE_PLACEMENT_SCHEMA_ID
    assert manifest["tev_placement"]["schema_id"] == SOURCE_PLACEMENT_SCHEMA_ID
    assert set(manifest["lev_placement"]).issuperset(
        {
            "q_kinematic_over_u_backend_world",
            "q_old_wake_over_u_backend_world",
            "q_provisional_tev_over_u_backend_world",
        }
    )
    assert "gamma_tev_new_te_only_provisional" in manifest["kelvin_ledger"]


def test_live_event_attestation_rejects_even_an_unchanged_dataclass_clone() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    assert validate_dvm_source_event(event) is event
    assert len(event.producer_manifest_sha256) == 64

    cloned = replace(event)
    with pytest.raises(ValueError, match="not a directly produced live source object"):
        validate_dvm_source_event(cloned)


def test_event_and_nested_schemas_reject_extra_fields_and_method_shadowing() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    ledger = event.kelvin_ledger
    assert ledger is not None
    payloads = (
        event,
        event.lev_placement,
        event.tev_placement,
        event.lineage,
        event.provenance,
        ledger,
    )
    for payload in payloads:
        assert not hasattr(payload, "__dict__")
        with pytest.raises(AttributeError):
            object.__setattr__(payload, "load", 42.0)
        assert not hasattr(payload, "load")

    with pytest.raises(AttributeError):
        object.__setattr__(event, "manifest", lambda: {"lift": 999.0})
    assert validate_dvm_source_event(event) is event
    assert "lift" not in json.dumps(event.manifest()).casefold()


def test_placement_tampering_and_nan_fail_before_live_identity_check() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    placement = event.lev_placement
    assert placement.q_birth_over_u_backend_world is not None
    assert placement.birth_displacement_from_edge_over_chord_backend_world is not None

    bad_schema = replace(placement, schema_id="fluxv-v5h-dvm-source-placement-v1")
    with pytest.raises(ValueError, match="placement schema is not pinned"):
        validate_dvm_source_event(replace(event, lev_placement=bad_schema))

    bad_family = replace(placement, vortex_family="tev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vortex family is inconsistent"):
        validate_dvm_source_event(replace(event, lev_placement=bad_family))

    bad_q = replace(
        placement,
        q_birth_over_u_backend_world=(
            np.nan,
            placement.q_birth_over_u_backend_world[1],
        ),
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        validate_dvm_source_event(replace(event, lev_placement=bad_q))

    finite_q_tamper = replace(
        placement,
        q_birth_over_u_backend_world=(
            np.nextafter(placement.q_birth_over_u_backend_world[0], np.inf),
            placement.q_birth_over_u_backend_world[1],
        ),
    )
    with pytest.raises(ValueError, match="q_birth is not the exact component sum"):
        validate_dvm_source_event(replace(event, lev_placement=finite_q_tamper))

    displacement = placement.birth_displacement_from_edge_over_chord_backend_world
    bad_displacement = replace(
        placement,
        birth_displacement_from_edge_over_chord_backend_world=(
            np.nextafter(displacement[0], np.inf),
            displacement[1],
        ),
    )
    with pytest.raises(ValueError, match="displacement is not reproducible"):
        validate_dvm_source_event(replace(event, lev_placement=bad_displacement))

    topology_alias = replace(placement, used_for_topology_eligible=False)
    with pytest.raises(ValueError, match="topology eligibility is inconsistent"):
        validate_dvm_source_event(replace(event, lev_placement=topology_alias))


def test_component_missing_nextafter_nan_and_compensation_are_rejected() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    placement = event.lev_placement
    q_birth = placement.q_birth_over_u_backend_world
    q_kinematic = placement.q_kinematic_over_u_backend_world
    q_old = placement.q_old_wake_over_u_backend_world
    q_provisional = placement.q_provisional_tev_over_u_backend_world
    assert q_birth is not None and q_kinematic is not None
    assert q_old is not None and q_provisional is not None

    missing = replace(placement, q_old_wake_over_u_backend_world=None)
    with pytest.raises(ValueError, match="two-component finite real pair"):
        validate_dvm_source_event(replace(event, lev_placement=missing))

    nan_component = replace(
        placement,
        q_provisional_tev_over_u_backend_world=(np.nan, q_provisional[1]),
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        validate_dvm_source_event(replace(event, lev_placement=nan_component))

    nextafter_component = replace(
        placement,
        q_kinematic_over_u_backend_world=(
            np.nextafter(q_kinematic[0], np.inf),
            q_kinematic[1],
        ),
    )
    with pytest.raises(ValueError, match="exact component sum"):
        validate_dvm_source_event(replace(event, lev_placement=nextafter_component))

    compensated_old_x = q_old[0] + 0.125
    compensated_provisional_x = float(q_birth[0] - (q_kinematic[0] + compensated_old_x))
    compensated = replace(
        placement,
        q_old_wake_over_u_backend_world=(compensated_old_x, q_old[1]),
        q_provisional_tev_over_u_backend_world=(
            compensated_provisional_x,
            q_provisional[1],
        ),
    )
    assert (
        compensated.q_kinematic_over_u_backend_world[0]
        + compensated.q_old_wake_over_u_backend_world[0]
        + compensated.q_provisional_tev_over_u_backend_world[0]
        == q_birth[0]
    )
    with pytest.raises(ValueError, match="manifest digest does not match"):
        validate_dvm_source_event(replace(event, lev_placement=compensated))


def test_continuous_placement_rejects_all_first_restart_velocity_fields() -> None:
    source = _source()
    source.step(np.deg2rad(35.0), 0.0, 0.0)
    event = source.step(np.deg2rad(35.0), 0.0, 0.0)
    placement = event.lev_placement
    assert placement.placement_mode == "continuous"
    forged = replace(
        placement,
        q_kinematic_over_u_backend_world=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="cannot carry velocity evidence"):
        validate_dvm_source_event(replace(event, lev_placement=forged))


def test_semantically_coherent_nested_placement_tamper_breaks_event_digest() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    placement = event.lev_placement
    edge = placement.edge_anchor_position_over_chord_backend_world
    birth = placement.birth_position_over_chord_backend_world
    assert edge is not None and birth is not None
    shift = 0.125
    translated_edge = (edge[0] + shift, edge[1])
    translated_birth = (birth[0] + shift, birth[1])

    # Translate edge and birth together: the placement equations remain
    # self-consistent, so the nested producer digest must catch the mutation.
    object.__setattr__(
        placement,
        "edge_anchor_position_over_chord_backend_world",
        translated_edge,
    )
    object.__setattr__(
        placement,
        "birth_position_over_chord_backend_world",
        translated_birth,
    )
    object.__setattr__(
        event,
        "lev_birth_position_over_chord_backend_world",
        translated_birth,
    )
    with pytest.raises(ValueError, match="manifest digest does not match"):
        validate_dvm_source_event(event)


def test_continuous_parent_is_audit_only_and_tamper_attested() -> None:
    source = _source()
    first = source.step(np.deg2rad(35.0), 0.0, 0.0)
    second = source.step(np.deg2rad(35.0), 0.0, 0.0)
    placement = second.lev_placement
    assert placement.placement_mode == "continuous"
    assert placement.continuous_parent_source_id == first.lineage.newborn_lev_source_id
    assert placement.used_for_topology_eligible is False

    bad_parent = replace(
        placement,
        continuous_parent_source_id=second.lineage.newborn_lev_source_id,
    )
    with pytest.raises(ValueError, match="parent source identity is inconsistent"):
        validate_dvm_source_event(replace(second, lev_placement=bad_parent))

    topology_alias = replace(placement, used_for_topology_eligible=True)
    with pytest.raises(ValueError, match="topology eligibility is inconsistent"):
        validate_dvm_source_event(replace(second, lev_placement=topology_alias))

    old_birth = second.lev_birth_position_over_chord_backend_world
    assert old_birth is not None
    mismatched_legacy = replace(
        second,
        lev_birth_position_over_chord_backend_world=(
            np.nextafter(old_birth[0], np.inf),
            old_birth[1],
        ),
    )
    with pytest.raises(ValueError, match="legacy birth positions disagree"):
        validate_dvm_source_event(mismatched_legacy)


def test_enabled_event_chain_is_parent_linked_and_disabled_does_not_advance() -> None:
    source = _source()
    first = source.step(np.deg2rad(35.0), 0.0, 0.0)
    disabled = source.step(_ExplodingInput(), _ExplodingInput(), enabled=False)
    second = source.step(np.deg2rad(35.0), 0.0, 0.0)

    assert len(first.parent_event_manifest_sha256) == 64
    assert disabled.parent_event_manifest_sha256 == first.producer_manifest_sha256
    assert second.parent_event_manifest_sha256 == first.producer_manifest_sha256
    assert second.parent_event_manifest_sha256 != second.producer_manifest_sha256
    assert validate_dvm_source_event(disabled) is disabled
    assert validate_dvm_source_event(second) is second


def test_missing_backend_source_diagnostics_poison_state_fail_closed() -> None:
    source = _source()
    original_step = source._model.step
    source._model.step = lambda *_args, **_kwargs: {"lesp": 0.0}
    with pytest.raises(RuntimeError, match="diagnostics are incomplete"):
        source.step(0.0, 0.0, 0.0)
    source._model.step = original_step
    with pytest.raises(RuntimeError, match="fail-closed"):
        source.step(0.0, 0.0, 0.0)


def test_backend_diagnostic_types_are_strict_and_poison_on_mutation() -> None:
    source = _source()
    original_step = source._model.step

    def corrupt_count(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_step(*args, **kwargs)
        result["n_tev"] = float(result["n_tev"])
        return result

    source._model.step = corrupt_count
    with pytest.raises(ValueError, match="backend n_tev must be an integer"):
        source.step(0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="fail-closed"):
        source.step(0.0, 0.0, 0.0)


def test_final_coupled_tev_cannot_masquerade_as_birth_provisional() -> None:
    source = _source()
    original_step = source._model.step

    def substitute_final(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_step(*args, **kwargs)
        result["tev_strength_te_only_provisional"] = result["tev_strength_solved"]
        return result

    source._model.step = substitute_final
    with pytest.raises(RuntimeError, match="independently recomputed velocity"):
        source.step(np.deg2rad(35.0), 0.0, 0.0)
    with pytest.raises(RuntimeError, match="fail-closed"):
        source.step(np.deg2rad(35.0), 0.0, 0.0)


@pytest.mark.parametrize("tamper", ["dt", "edge", "tev"])
def test_live_dt_edge_and_tev_birth_tampering_fail_closed(tamper: str) -> None:
    source = _source()
    original_step = source._model.step

    def corrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_step(*args, **kwargs)
        if tamper == "dt":
            result["dt_i"] = np.nextafter(result["dt_i"], np.inf)
        elif tamper == "edge":
            edge = source._model.v5h_edge_anchors["lev"]
            source._model.v5h_edge_anchors["lev"] = (
                np.nextafter(edge[0], np.inf),
                edge[1],
            )
        else:
            tev = source._model.v5h_birth_columns[0]
            source._model.v5h_birth_columns[0] = (
                np.nextafter(tev[0], np.inf),
                tev[1],
            )
        return result

    source._model.step = corrupt
    expected = {
        "dt": "different source time step",
        "edge": "edge anchor drifted",
        "tev": "independently recomputed velocity",
    }[tamper]
    with pytest.raises(RuntimeError, match=expected):
        source.step(np.deg2rad(35.0), 0.0, 0.0)


def test_live_core_tampering_is_rejected_before_backend_advance() -> None:
    source = _source()
    source._model.rc = np.nextafter(source._model.rc, np.inf)
    with pytest.raises(RuntimeError, match="backend rc drifted"):
        source.step(np.deg2rad(35.0), 0.0, 0.0)
    assert source.step_count == 0

    post_source = _source(physical_strip_id="wing-left:strip-core-post")
    original_step = post_source._model.step

    def corrupt_after(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_step(*args, **kwargs)
        post_source._model.rc = np.nextafter(post_source._model.rc, np.inf)
        return result

    post_source._model.step = corrupt_after
    with pytest.raises(RuntimeError, match="backend rc drifted"):
        post_source.step(np.deg2rad(35.0), 0.0, 0.0)
    assert post_source.step_count == 1
    with pytest.raises(RuntimeError, match="fail-closed"):
        post_source.step(np.deg2rad(35.0), 0.0, 0.0)


def test_provisional_ledger_nan_and_nextafter_are_attested() -> None:
    event = _source().step(np.deg2rad(35.0), 0.0, 0.0)
    ledger = event.kelvin_ledger
    assert ledger is not None
    with pytest.raises(ValueError, match="finite real scalar"):
        validate_dvm_source_event(
            replace(
                event,
                kelvin_ledger=replace(
                    ledger,
                    gamma_tev_new_te_only_provisional=np.nan,
                ),
            )
        )
    with pytest.raises(ValueError, match="manifest digest does not match"):
        validate_dvm_source_event(
            replace(
                event,
                kelvin_ledger=replace(
                    ledger,
                    gamma_tev_new_te_only_provisional=np.nextafter(
                        ledger.gamma_tev_new_te_only_provisional,
                        np.inf,
                    ),
                ),
            )
        )


def test_invalid_enabled_inputs_fail_before_backend_state_advances() -> None:
    source = _source()
    with pytest.raises(ValueError, match="enabled must be Boolean"):
        source.step(0.0, 0.0, 0.0, enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha_rad"):
        source.step(np.nan, 0.0, 0.0)
    with pytest.raises(ValueError, match="delta_time_convective"):
        source.step(0.0, 0.0, 0.0, delta_time_convective=False)
    assert source.step_count == 0
    event = source.step(0.0, 0.0, 0.0)
    assert event.lineage.source_step_index == 1
