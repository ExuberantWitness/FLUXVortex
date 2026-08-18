"""M0/M1 tests for the FluxV v5h9 live-boundary net-state owner.

The v5h8 append-only cloud is used only as a gross mechanical oracle.  These
tests require v5h9 to collapse the newest coincident incidence transaction
before the state can become physical: counter particles, newly redeposited
upstream particles, and clone rows may never survive in the committed owner
state.  The first panel's true outer upstream boundary remains physical.

No Ptera, load, feedback, target-data, or remesh implementation is imported.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
import gc
import importlib.util
from math import fsum
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable
import weakref

import numpy as np
from numpy.typing import NDArray
import pytest

from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import reformulated_vpm_rhs


FloatArray = NDArray[np.float64]
_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "forward_flight_benchmarks"


def _isolated_module(name: str, filename: str) -> Any:
    path = _BENCHMARK_DIR / filename
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to load isolated module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_MODULES_BEFORE_IMPORT = frozenset(sys.modules)
sheet = _isolated_module(
    "_fluxv_v5h8_for_v5h9_owner_tests",
    "fluxv_v5h8_incremental_sheet.py",
)
owner_api = _isolated_module(
    "_fluxv_v5h9_live_boundary_owner_under_test",
    "fluxv_v5h9_live_boundary_owner.py",
)
edge_bridge = owner_api._EDGE_MODULE
_MODULES_ADDED_BY_IMPORT = frozenset(sys.modules).difference(_MODULES_BEFORE_IMPORT)


SPAN_M = 0.60
PANEL_STEP_M = 0.04
DELTA_GAMMA_M2_PER_S = 0.016
SMOOTHING_RADIUS_M = 0.085
TARGET_SPACING_M = 0.02
DELTA_TIME_S = 0.02
PARTICLE_CAP = 1_000
OWNER_ID = "v5h9:test:live-boundary-owner"
WING_ID = "wing:manufactured:right"
EPSILON_MULTIPLIER = 64.0
PROBES_M = np.asarray(
    (
        (0.05, 0.30, 0.30),
        (0.10, 0.15, 0.40),
        (0.20, 0.45, 0.50),
        (-0.08, 0.52, 0.18),
    ),
    dtype=np.float64,
)


def _first_panel() -> Any:
    return sheet.make_panel(
        (0.0, 0.0, 0.0),
        (0.0, SPAN_M, 0.0),
        (PANEL_STEP_M, 0.0, 0.0),
        (PANEL_STEP_M, SPAN_M, 0.0),
        DELTA_GAMMA_M2_PER_S,
        release_index=1,
    )


def _first_gross_state() -> Any:
    return sheet.start_incremental_sheet(
        _first_panel(),
        SMOOTHING_RADIUS_M,
        TARGET_SPACING_M,
        particle_cap=PARTICLE_CAP,
    ).state


def _bootstrap(*, particle_cap: int = PARTICLE_CAP, owner_id: str = OWNER_ID) -> Any:
    return owner_api.bootstrap_live_boundary_owner(
        _first_gross_state(),
        particle_cap=particle_cap,
        owner_id=owner_id,
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )


def _append_gross(gross: Any, release_index: int) -> Any:
    return sheet.append_live_basis_panel(
        gross,
        (release_index * PANEL_STEP_M, 0.0, 0.0),
        (release_index * PANEL_STEP_M, SPAN_M, 0.0),
        release_index * DELTA_GAMMA_M2_PER_S,
        particle_cap=PARTICLE_CAP,
    ).state


def _propose(owner: Any, gross: Any, *, proposal_id: str | None = None) -> Any:
    release_index = len(gross.panels)
    return owner_api.propose_live_boundary_update(
        owner,
        owner.state,
        gross,
        (release_index - 1) * DELTA_GAMMA_M2_PER_S,
        release_index * DELTA_GAMMA_M2_PER_S,
        proposal_id=proposal_id or f"proposal:release:{release_index}",
        wing_id=WING_ID,
        source_time_s=release_index * DELTA_TIME_S,
    )


def _readonly(array: Any) -> FloatArray:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    immutable = immutable.reshape(contiguous.shape)
    immutable.setflags(write=False)
    return immutable


def _state_bytes(state: Any) -> tuple[bytes, bytes, bytes]:
    return (state.positions.tobytes(), state.gamma.tobytes(), state.sigma.tobytes())


def _owner_fingerprint(owner: Any) -> tuple[Any, ...]:
    state = owner.state
    return (
        repr(owner),
        owner.owner_id,
        owner.particle_cap,
        owner.epoch,
        owner.wing_id,
        owner.source_time_s,
        _state_bytes(state),
        state.particle_ids,
        state.lineage,
        state.panels,
        state.live_boundary_indices,
        state.release_index,
        state.circulation_m2_s,
        state.state_epoch,
        state.state_sha256,
        owner.events,
        tuple(id(event) for event in owner.events),
    )


def _impulse(positions: FloatArray, gamma: FloatArray) -> FloatArray:
    cross = np.cross(positions, gamma)
    return np.asarray(
        [0.5 * fsum(float(value) for value in cross[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _tolerance(reference: Any) -> float:
    values = np.asarray(reference, dtype=np.float64)
    scale = 1.0 if values.size == 0 else max(1.0, float(np.max(np.abs(values))))
    return EPSILON_MULTIPLIER * np.finfo(np.float64).eps * scale


def _manual_canonical_snapshot(gross: Any) -> Any:
    """Independent test reducer; deliberately does not call either owner module."""

    count = int(gross.positions.shape[0])
    contributions: dict[int, list[FloatArray]] = {
        index: [np.asarray(gross.gamma[index], dtype=np.float64)]
        for index in range(count)
    }
    removed: set[int] = set()
    for pair in gross.clone_pairs:
        assert type(pair) is tuple and len(pair) == 2
        old_index, clone_index = pair
        assert type(old_index) is int and type(clone_index) is int
        assert clone_index not in removed
        np.testing.assert_array_equal(
            gross.positions[old_index], gross.positions[clone_index]
        )
        assert gross.sigma[old_index] == gross.sigma[clone_index]
        contributions[old_index].append(
            np.asarray(gross.gamma[clone_index], dtype=np.float64)
        )
        removed.add(clone_index)

    kept = tuple(index for index in range(count) if index not in removed)
    gamma = np.asarray(
        [
            [
                fsum(float(term[axis]) for term in contributions[index])
                for axis in range(3)
            ]
            for index in kept
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        positions=np.asarray(gross.positions[list(kept)], dtype=np.float64),
        gamma=gamma,
        sigma=np.asarray(gross.sigma[list(kept)], dtype=np.float64),
        particle_ids=tuple(gross.particle_ids[index] for index in kept),
        lineage=tuple(gross.lineage[index] for index in kept),
    )


def _assert_readonly_particle_state(state: Any) -> None:
    for array, shape_tail in (
        (state.positions, (3,)),
        (state.gamma, (3,)),
        (state.sigma, ()),
    ):
        assert type(array) is np.ndarray
        assert array.dtype == np.dtype(np.float64)
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        assert np.all(np.isfinite(array))
        assert array.shape[1:] == shape_tail
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert state.positions.shape == state.gamma.shape
    assert state.sigma.shape == (state.positions.shape[0],)
    assert np.all(state.sigma > 0.0)
    assert len(state.particle_ids) == state.positions.shape[0]
    assert len(state.lineage) == state.positions.shape[0]
    assert len(set(state.particle_ids)) == len(state.particle_ids)


def _assert_zero_gross_rows(value: Any) -> None:
    assert value.clone_count == 0
    assert value.counter_particle_count == 0
    assert value.fresh_upstream_particle_count == 0
    roles = tuple(record.role for record in value.lineage)
    assert "inherited_upstream_counter" not in roles
    # The first panel's upstream is a real outer sheet boundary.  Later
    # releases must update the inherited live slice and may not add another
    # fresh-upstream row.
    assert all(
        record.release_index == 1
        for record in value.lineage
        if record.role == "fresh_upstream"
    )


def _assert_physical_equivalence(state: Any, gross: Any) -> None:
    expected = _manual_canonical_snapshot(gross)
    np.testing.assert_array_equal(state.positions, expected.positions)
    np.testing.assert_allclose(
        state.gamma,
        expected.gamma,
        rtol=0.0,
        atol=_tolerance(expected.gamma),
    )
    np.testing.assert_array_equal(state.sigma, expected.sigma)
    assert state.particle_ids == expected.particle_ids

    actual_field = direct_gaussian_erf_velocity_jacobian(
        state.positions,
        state.gamma,
        state.sigma,
        target_positions=PROBES_M,
    )
    expected_field = direct_gaussian_erf_velocity_jacobian(
        expected.positions,
        expected.gamma,
        expected.sigma,
        target_positions=PROBES_M,
    )
    np.testing.assert_allclose(
        actual_field.velocity,
        expected_field.velocity,
        rtol=0.0,
        atol=_tolerance(expected_field.velocity),
    )
    np.testing.assert_allclose(
        actual_field.jacobian,
        expected_field.jacobian,
        rtol=0.0,
        atol=_tolerance(expected_field.jacobian),
    )
    np.testing.assert_allclose(
        _impulse(state.positions, state.gamma),
        _impulse(expected.positions, expected.gamma),
        rtol=0.0,
        atol=_tolerance(_impulse(expected.positions, expected.gamma)),
    )

    actual_self = direct_gaussian_erf_velocity_jacobian(
        state.positions,
        state.gamma,
        state.sigma,
    )
    expected_self = direct_gaussian_erf_velocity_jacobian(
        expected.positions,
        expected.gamma,
        expected.sigma,
    )
    actual_rhs = reformulated_vpm_rhs(
        state.gamma,
        state.sigma,
        actual_self.jacobian,
    )
    expected_rhs = reformulated_vpm_rhs(
        expected.gamma,
        expected.sigma,
        expected_self.jacobian,
    )
    for actual, reference in zip(actual_rhs, expected_rhs, strict=True):
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=0.0,
            atol=_tolerance(reference),
        )


def _latest_clone_indices(candidate: Any) -> tuple[int, ...]:
    release_index = len(candidate.panels)
    return tuple(
        clone_index
        for _, clone_index in candidate.clone_pairs
        if candidate.lineage[clone_index].release_index == release_index
    )


def _fresh_suffix_index(candidate: Any) -> int:
    release_index = len(candidate.panels)
    return next(
        index
        for index, record in enumerate(candidate.lineage)
        if record.release_index == release_index and record.role == "fresh_downstream"
    )


def _incidence_signature(incidences: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            incidence.ring_id,
            incidence.traversal_index,
            incidence.source_start_id,
            incidence.source_end_id,
            incidence.canonical_sign,
            incidence.ring_circulation,
            incidence.signed_circulation,
        )
        for incidence in incidences
    )


def _nested_incidence_mutation(
    candidate: Any, mode: str
) -> tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...]]:
    index = _fresh_suffix_index(candidate)
    particle_ids = list(candidate.particle_ids)
    lineage = list(candidate.lineage)
    record = lineage[index]
    source = record.source_lineage
    assert source is not None
    incidence = source.ring_incidences[0]

    if mode == "value":
        incidences = (
            replace(
                incidence,
                traversal_index=(incidence.traversal_index + 1) % 4,
            ),
        )
    elif mode == "order":
        other = next(
            other_record.source_lineage.ring_incidences[0]
            for other_record in candidate.lineage
            if other_record.release_index == len(candidate.panels)
            and other_record.source_lineage is not None
            and other_record.source_edge != record.source_edge
        )
        incidences = (other, incidence)
    elif mode == "ulp":
        ring_circulation = float(np.nextafter(incidence.ring_circulation, np.inf))
        incidences = (
            replace(
                incidence,
                ring_circulation=ring_circulation,
                signed_circulation=float(incidence.canonical_sign * ring_circulation),
            ),
        )
    elif mode == "signature":
        incidences = source.ring_incidences
    else:
        raise AssertionError(f"unknown incidence mutation: {mode}")

    if mode != "signature":
        lineage[index] = replace(
            record,
            source_lineage=replace(source, ring_incidences=incidences),
        )
        return candidate.particle_ids, tuple(lineage)

    encoded_id = list(record.particle_id)
    signature = _incidence_signature(incidences)[0]
    encoded_id[5] = (signature[1:] + signature[:1],)
    forged_id = tuple(encoded_id)
    particle_ids[index] = forged_id
    forged_source = replace(
        source,
        particle_id=forged_id,
        ring_incidences=incidences,
    )
    lineage[index] = replace(
        record,
        particle_id=forged_id,
        source_lineage=forged_source,
    )
    return tuple(particle_ids), tuple(lineage)


def _assert_rejected_transactionally(
    owner: Any,
    operation: Callable[[], Any],
) -> Any | None:
    before = _owner_fingerprint(owner)
    try:
        proposal = operation()
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        proposal = None
    else:
        assert proposal.status == "remesh_required"
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.commit_live_boundary_update(owner, proposal)
    assert _owner_fingerprint(owner) == before
    return proposal


def _different_sha256(value: str) -> str:
    replacement = "1" if value[0] == "0" else "0"
    return replacement + value[1:]


def _proposal_field_mutations(proposal: Any, candidate: Any) -> dict[str, object]:
    changed_indices = proposal.changed_indices[1:] + proposal.changed_indices[:1]
    if changed_indices == proposal.changed_indices:
        changed_indices = proposal.changed_indices + proposal.changed_indices[:1]
    return {
        "status": "remesh_required",
        "proposal_id": f"{proposal.proposal_id}:forged",
        # bool compares equal to epoch one; exact-type matching must reject it.
        "parent_epoch": True,
        "changed_indices": changed_indices,
        "planned_particle_count": proposal.planned_particle_count + 1,
        "appended_particle_count": proposal.appended_particle_count + 1,
        "first_mismatch": "forged",
        "gamma_previous_m2_s": float(
            np.nextafter(proposal.gamma_previous_m2_s, np.inf)
        ),
        "gamma_current_m2_s": float(np.nextafter(proposal.gamma_current_m2_s, np.inf)),
        "gamma_scale": float(np.nextafter(proposal.gamma_scale, np.inf)),
        "wing_id": f"{proposal.wing_id}:forged",
        # Equal numeric value with a distinct exact scalar type.
        "source_time_s": np.float64(proposal.source_time_s),
        "parent_state_sha256": _different_sha256(proposal.parent_state_sha256),
        "candidate_state_sha256": _different_sha256(proposal.candidate_state_sha256),
        "proposal_sha256": _different_sha256(proposal.proposal_sha256),
        # An equal-value dataclass clone is not the issued candidate capability.
        "candidate_state": replace(candidate),
        "clone_count": proposal.clone_count + 1,
        "counter_particle_count": proposal.counter_particle_count + 1,
        "fresh_upstream_particle_count": (proposal.fresh_upstream_particle_count + 1),
        "parent_owner_sha256": _different_sha256(proposal.parent_owner_sha256),
        "oracle_source_sha256": _different_sha256(proposal.oracle_source_sha256),
        "candidate_identity": proposal.candidate_identity + 1,
    }


def test_isolated_import_has_no_ptera_benchmark_package_or_load_surface() -> None:
    assert not any(
        name == "pterasoftware" or name.startswith("pterasoftware.")
        for name in _MODULES_ADDED_BY_IMPORT
    )


def test_first_binding_rejects_spoofed_transitive_deposition_callable() -> None:
    local_sheet = _isolated_module(
        "_fluxv_v5h8_transitive_spoof_probe",
        "fluxv_v5h8_incremental_sheet.py",
    )
    first_panel = local_sheet.make_panel(
        (0.0, 0.0, 0.0),
        (0.0, SPAN_M, 0.0),
        (PANEL_STEP_M, 0.0, 0.0),
        (PANEL_STEP_M, SPAN_M, 0.0),
        DELTA_GAMMA_M2_PER_S,
        release_index=1,
    )
    first = local_sheet.start_incremental_sheet(
        first_panel,
        SMOOTHING_RADIUS_M,
        TARGET_SPACING_M,
        particle_cap=PARTICLE_CAP,
    ).state
    original_deposit = local_sheet.deposit_edge_graph_prescribed_sigma_and_spacing
    spoof_calls = 0

    def forwarding_spoof(*args: Any, **kwargs: Any) -> Any:
        nonlocal spoof_calls
        spoof_calls += 1
        deposited = original_deposit(*args, **kwargs)
        forged_gamma = np.array(deposited.gamma, copy=True)
        forged_gamma[:, 0] += 1.0
        return replace(deposited, gamma=_readonly(forged_gamma))

    forwarding_spoof.__module__ = "fluxvortex.rvpm_edge_bridge"
    forwarding_spoof.__qualname__ = "deposit_edge_graph_prescribed_sigma_and_spacing"
    local_sheet.deposit_edge_graph_prescribed_sigma_and_spacing = forwarding_spoof
    forged_candidate = local_sheet.append_live_basis_panel(
        first,
        (2.0 * PANEL_STEP_M, 0.0, 0.0),
        (2.0 * PANEL_STEP_M, SPAN_M, 0.0),
        2.0 * DELTA_GAMMA_M2_PER_S,
        particle_cap=PARTICLE_CAP,
    ).state
    assert spoof_calls == 1

    local_owner = _isolated_module(
        "_fluxv_v5h9_transitive_spoof_probe",
        "fluxv_v5h9_live_boundary_owner.py",
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        local_owner.bootstrap_live_boundary_owner(
            first,
            particle_cap=PARTICLE_CAP,
            owner_id="v5h9:test:transitive-spoof",
            wing_id=WING_ID,
            source_time_s=DELTA_TIME_S,
        )
    assert spoof_calls == 1

    local_sheet.deposit_edge_graph_prescribed_sigma_and_spacing = original_deposit
    clean_owner = local_owner.bootstrap_live_boundary_owner(
        first,
        particle_cap=PARTICLE_CAP,
        owner_id="v5h9:test:transitive-clean-retry",
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )
    rejected = local_owner.propose_live_boundary_update(
        clean_owner,
        clean_owner.state,
        forged_candidate,
        DELTA_GAMMA_M2_PER_S,
        2.0 * DELTA_GAMMA_M2_PER_S,
        proposal_id="forged-candidate-after-clean-binding",
        wing_id=WING_ID,
        source_time_s=2.0 * DELTA_TIME_S,
    )
    assert rejected.status == "remesh_required"
    assert rejected.first_mismatch == "gamma"

    clean_candidate = local_sheet.append_live_basis_panel(
        first,
        (2.0 * PANEL_STEP_M, 0.0, 0.0),
        (2.0 * PANEL_STEP_M, SPAN_M, 0.0),
        2.0 * DELTA_GAMMA_M2_PER_S,
        particle_cap=PARTICLE_CAP,
    ).state
    proposal = local_owner.propose_live_boundary_update(
        clean_owner,
        clean_owner.state,
        clean_candidate,
        DELTA_GAMMA_M2_PER_S,
        2.0 * DELTA_GAMMA_M2_PER_S,
        proposal_id="clean-after-transitive-spoof",
        wing_id=WING_ID,
        source_time_s=2.0 * DELTA_TIME_S,
    )
    result = local_owner.commit_live_boundary_update(clean_owner, proposal)
    _assert_physical_equivalence(result.state, clean_candidate)


def test_edge_builtin_fallback_shadow_is_rejected_before_call() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-module-float-shadow")
    candidate = _append_gross(_first_gross_state(), 2)
    calls = 0

    def forged_float(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("shadowed edge float was executed")

    assert "float" not in edge_bridge.__dict__
    edge_bridge.__dict__["float"] = forged_float
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-module-float-shadow")
    finally:
        edge_bridge.__dict__.pop("float", None)
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-module-float-shadow-clean-retry",
    )
    edge_bridge.__dict__["float"] = forged_float
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.commit_live_boundary_update(owner, proposal)
    finally:
        edge_bridge.__dict__.pop("float", None)
    assert calls == 0

    # The failed provenance check consumed neither the proposal nor owner.
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_builtin_shadow_present_before_owner_binding_is_rejected() -> None:
    calls = 0

    def forged_float(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("pre-binding edge float shadow was executed")

    forged_module_name = "_fluxv_v5h9_prebinding_edge_shadow"
    assert "float" not in edge_bridge.__dict__
    edge_bridge.__dict__["float"] = forged_float
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _isolated_module(
                forged_module_name,
                "fluxv_v5h9_live_boundary_owner.py",
            )
    finally:
        edge_bridge.__dict__.pop("float", None)
        sys.modules.pop(forged_module_name, None)
    assert calls == 0

    clean_owner_api = _isolated_module(
        "_fluxv_v5h9_after_prebinding_edge_shadow",
        "fluxv_v5h9_live_boundary_owner.py",
    )
    clean_owner = clean_owner_api.bootstrap_live_boundary_owner(
        _first_gross_state(),
        particle_cap=PARTICLE_CAP,
        owner_id="v5h9:test:prebinding-shadow-clean-retry",
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )
    assert clean_owner_api.validate_live_boundary_owner(clean_owner) is clean_owner


def test_prebinding_root_lookup_never_invokes_edge_module_getattr() -> None:
    calls = 0

    def forged_getattr(name: str) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"prebinding module __getattr__ executed for {name}")

    module_name = "_fluxv_v5h9_prebinding_missing_root"
    missing = object()
    original_getattr = edge_bridge.__dict__.get("__getattr__", missing)
    original_root = edge_bridge.__dict__.pop("canonical_edge_key")
    edge_bridge.__dict__["__getattr__"] = forged_getattr
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _isolated_module(module_name, "fluxv_v5h9_live_boundary_owner.py")
    finally:
        edge_bridge.__dict__["canonical_edge_key"] = original_root
        if original_getattr is missing:
            edge_bridge.__dict__.pop("__getattr__", None)
        else:
            edge_bridge.__dict__["__getattr__"] = original_getattr
        sys.modules.pop(module_name, None)
    assert calls == 0

    owner = _bootstrap(owner_id="v5h9:test:prebinding-root-clean-retry")
    assert owner_api.validate_live_boundary_owner(owner) is owner


def test_edge_builtin_mapping_drift_is_rejected_before_call() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-builtins-drift")
    candidate = _append_gross(_first_gross_state(), 2)
    builtins_mapping = edge_bridge._finite_real.__builtins__
    assert type(builtins_mapping) is dict
    original_float = builtins_mapping["float"]
    calls = 0

    def forged_float(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("changed edge builtin was executed")

    caught = False
    builtins_mapping["float"] = forged_float
    try:
        try:
            _propose(owner, candidate, proposal_id="edge-builtins-drift")
        except (TypeError, ValueError, RuntimeError):
            caught = True
    finally:
        builtins_mapping["float"] = original_float
    assert caught
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-builtins-drift-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_private_helper_identity_drift_is_rejected_before_call() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-helper-identity")
    candidate = _append_gross(_first_gross_state(), 2)
    original = edge_bridge._finite_real
    calls = 0

    def forged_helper(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("changed edge helper was executed")

    edge_bridge.__dict__["_finite_real"] = forged_helper
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-helper-identity")
    finally:
        edge_bridge.__dict__["_finite_real"] = original
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-helper-identity-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_module_getattr_is_never_used_for_provenance_lookup() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-module-getattr")
    candidate = _append_gross(_first_gross_state(), 2)
    original_helper = edge_bridge.__dict__.pop("_finite_real")
    missing = object()
    original_getattr = edge_bridge.__dict__.get("__getattr__", missing)
    calls = 0

    def forged_getattr(name: str) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"edge module __getattr__ executed for {name}")

    edge_bridge.__dict__["__getattr__"] = forged_getattr
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-module-getattr")
    finally:
        edge_bridge.__dict__["_finite_real"] = original_helper
        if original_getattr is missing:
            edge_bridge.__dict__.pop("__getattr__", None)
        else:
            edge_bridge.__dict__["__getattr__"] = original_getattr
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-module-getattr-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_private_helper_code_drift_is_rejected_before_call() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-helper-code")
    candidate = _append_gross(_first_gross_state(), 2)
    helper = edge_bridge._finite_real
    original_code = helper.__code__
    calls = 0

    def forged_helper(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("changed edge helper code was executed")

    helper.__code__ = forged_helper.__code__
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-helper-code")
    finally:
        helper.__code__ = original_code
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-helper-code-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_callable_kwdefault_drift_is_rejected_and_retryable() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-kwdefault")
    candidate = _append_gross(_first_gross_state(), 2)
    function = edge_bridge.deposit_edge_graph_prescribed_sigma_and_spacing
    kwdefaults = function.__kwdefaults__
    assert type(kwdefaults) is dict
    original = kwdefaults["minimum_overlap"]
    kwdefaults["minimum_overlap"] = float(np.nextafter(original, np.inf))
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-kwdefault")
    finally:
        kwdefaults["minimum_overlap"] = original

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-kwdefault-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


@pytest.mark.parametrize(
    ("class_name", "descriptor_name"),
    (("EdgeLedger", "retained"), ("EdgeGraph", "retained_edges")),
)
def test_edge_class_descriptor_drift_is_rejected_before_call(
    class_name: str, descriptor_name: str
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:edge-class-descriptor:{class_name}")
    candidate = _append_gross(_first_gross_state(), 2)
    class_type = edge_bridge.__dict__[class_name]
    original = class_type.__dict__[descriptor_name]
    calls = 0

    def forged_retained(instance: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("changed edge class descriptor was executed")

    setattr(class_type, descriptor_name, property(forged_retained))
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(
                owner,
                candidate,
                proposal_id=f"edge-class-descriptor:{class_name}",
            )
    finally:
        setattr(class_type, descriptor_name, original)
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id=f"edge-class-descriptor-clean-retry:{class_name}",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_class_callable_global_closure_is_sealed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-class-global-closure")
    candidate = _append_gross(_first_gross_state(), 2)
    function = edge_bridge.EdgeGraph.__dict__["__setattr__"]
    builtins_mapping = function.__builtins__
    original_super = builtins_mapping["super"]
    calls = 0

    def forged_super(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("changed class callable global was executed")

    caught = False
    builtins_mapping["super"] = forged_super
    try:
        try:
            _propose(owner, candidate, proposal_id="edge-class-global-closure")
        except (TypeError, ValueError, RuntimeError):
            caught = True
    finally:
        builtins_mapping["super"] = original_super
    assert caught
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-class-global-closure-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_prebinding_class_builtin_shadow_is_rejected_before_call() -> None:
    calls = 0

    def forged_super(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("prebinding class builtin shadow was executed")

    module_name = "_fluxv_v5h9_prebinding_class_builtin"
    assert "super" not in edge_bridge.__dict__
    edge_bridge.__dict__["super"] = forged_super
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _isolated_module(module_name, "fluxv_v5h9_live_boundary_owner.py")
    finally:
        edge_bridge.__dict__.pop("super", None)
        sys.modules.pop(module_name, None)
    assert calls == 0

    owner = _bootstrap(owner_id="v5h9:test:prebinding-class-global-clean-retry")
    assert owner_api.validate_live_boundary_owner(owner) is owner


def test_edge_property_accessor_code_is_sealed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-property-code")
    candidate = _append_gross(_first_gross_state(), 2)
    getter = edge_bridge.EdgeGraph.__dict__["retained_edges"].fget
    original_code = getter.__code__

    def forged_getter(instance: object) -> object:
        raise AssertionError("changed edge property getter was executed")

    getter.__code__ = forged_getter.__code__
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-property-code")
    finally:
        getter.__code__ = original_code

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-property-code-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_prebinding_edge_property_code_must_match_authoritative_source() -> None:
    getter = edge_bridge.EdgeGraph.__dict__["retained_edges"].fget
    original_code = getter.__code__
    module_name = "_fluxv_v5h9_prebinding_property_code"

    def forged_getter(instance: object) -> object:
        raise AssertionError("prebinding changed property getter was executed")

    getter.__code__ = forged_getter.__code__
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _isolated_module(module_name, "fluxv_v5h9_live_boundary_owner.py")
    finally:
        getter.__code__ = original_code
        sys.modules.pop(module_name, None)

    owner = _bootstrap(owner_id="v5h9:test:prebinding-property-clean-retry")
    assert owner_api.validate_live_boundary_owner(owner) is owner


def test_edge_numpy_attribute_drift_is_rejected_before_call() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-numpy-attribute")
    candidate = _append_gross(_first_gross_state(), 2)
    namespace = edge_bridge.np.linalg.__dict__
    original = namespace["norm"]
    calls = 0

    def forged_norm(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise AssertionError("changed NumPy dependency was executed")

    namespace["norm"] = forged_norm
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-numpy-attribute")
    finally:
        namespace["norm"] = original
    assert calls == 0

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-numpy-attribute-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_source_path_drift_is_rejected_and_retryable() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-source-path")
    candidate = _append_gross(_first_gross_state(), 2)
    original_file = edge_bridge.__file__
    edge_bridge.__file__ = str(Path(original_file).with_name("forged_edge.py"))
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-source-path")
    finally:
        edge_bridge.__file__ = original_file

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-source-path-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_edge_module_identity_drift_is_rejected_and_retryable() -> None:
    owner = _bootstrap(owner_id="v5h9:test:edge-module-identity")
    candidate = _append_gross(_first_gross_state(), 2)
    module_name = edge_bridge.__name__
    original = sys.modules[module_name]
    sys.modules[module_name] = SimpleNamespace(__name__=module_name)
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            _propose(owner, candidate, proposal_id="edge-module-identity")
    finally:
        sys.modules[module_name] = original

    proposal = _propose(
        owner,
        candidate,
        proposal_id="edge-module-identity-clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_import_closure_still_excludes_ptera_target_and_load_surfaces() -> None:
    assert "forward_flight_benchmarks" not in _MODULES_ADDED_BY_IMPORT
    assert not any(
        forbidden in name.lower()
        for name in _MODULES_ADDED_BY_IMPORT
        for forbidden in ("target", "ptera", "load_adapter")
    )


def test_bootstrap_constructs_a_frozen_net_owner_without_gross_rows() -> None:
    gross = _first_gross_state()
    owner = _bootstrap()
    state = owner.state

    assert owner.owner_id == OWNER_ID
    assert owner.particle_cap == PARTICLE_CAP
    assert owner.wing_id == WING_ID
    assert owner.source_time_s == DELTA_TIME_S
    assert owner.epoch == state.state_epoch
    assert owner.events == ()
    assert state.release_index == 1
    assert state.circulation_m2_s == DELTA_GAMMA_M2_PER_S
    assert state.live_boundary_indices
    assert isinstance(state.state_sha256, str) and len(state.state_sha256) == 64
    _assert_readonly_particle_state(state)
    _assert_zero_gross_rows(state)
    _assert_physical_equivalence(state, gross)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        state.release_index = 99
    with pytest.raises((FrozenInstanceError, AttributeError)):
        owner.epoch = 99


def test_bootstrap_accepts_only_the_uncloned_first_release() -> None:
    release_two = _append_gross(_first_gross_state(), 2)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.bootstrap_live_boundary_owner(
            release_two,
            particle_cap=PARTICLE_CAP,
            owner_id="v5h9:test:invalid-release-two-bootstrap",
            wing_id=WING_ID,
            source_time_s=DELTA_TIME_S,
        )


def _forge_release_one_derived_field(gross: Any, field: str) -> Any:
    if field == "positions":
        positions = np.array(gross.positions, copy=True)
        positions[0, 2] = 0.125
        return replace(gross, positions=_readonly(positions))
    if field == "gamma":
        gamma = np.array(gross.gamma, copy=True)
        gamma[0, 0] = np.nextafter(gamma[0, 0], np.inf)
        return replace(gross, gamma=_readonly(gamma))
    if field == "sigma":
        sigma = np.array(gross.sigma, copy=True)
        sigma[0] = np.nextafter(sigma[0], np.inf)
        return replace(gross, sigma=_readonly(sigma))
    lineage = list(gross.lineage)
    record = lineage[0]
    source = record.source_lineage
    assert source is not None
    if field == "particle_id":
        particle_ids = list(gross.particle_ids)
        forged_id = particle_ids[0] + ("forged",)
        particle_ids[0] = forged_id
        source = replace(source, particle_id=forged_id)
        lineage[0] = replace(
            record,
            particle_id=forged_id,
            source_lineage=source,
        )
        return replace(
            gross,
            particle_ids=tuple(particle_ids),
            lineage=tuple(lineage),
        )
    if field == "lineage":
        forged_index = (source.subdivision_index + 1) % source.subdivision_count
        assert forged_index != source.subdivision_index
        lineage[0] = replace(
            record,
            source_lineage=replace(source, subdivision_index=forged_index),
        )
        return replace(gross, lineage=tuple(lineage))
    raise AssertionError(f"unknown release-one field: {field}")


@pytest.mark.parametrize(
    "field",
    ("positions", "gamma", "sigma", "particle_id", "lineage"),
)
def test_bootstrap_redeposits_and_rejects_self_consistent_release_one_forgery(
    field: str,
) -> None:
    clean = _first_gross_state()
    forged = _forge_release_one_derived_field(clean, field)
    # This is the precise audit gap: the gross container remains self-consistent
    # according to v5h8, but its derived deposition is no longer authoritative.
    assert sheet._validate_state(forged) is forged
    owner_id = f"v5h9:test:release-one-redeposit:{field}"

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.bootstrap_live_boundary_owner(
            forged,
            particle_cap=PARTICLE_CAP,
            owner_id=owner_id,
            wing_id=WING_ID,
            source_time_s=DELTA_TIME_S,
        )

    # A rejected genesis must not consume the process-global owner-id claim.
    owner = owner_api.bootstrap_live_boundary_owner(
        clean,
        particle_cap=PARTICLE_CAP,
        owner_id=owner_id,
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )
    assert owner_api.validate_live_boundary_owner(owner) is owner


def test_owner_numpy_attribute_drift_is_zero_call_and_transactionally_retryable() -> (
    None
):
    clean_gross = _first_gross_state()
    original_array_equal = np.array_equal
    calls = 0

    def forged_array_equal(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        return True

    def assert_rejected_before_call(operation: Callable[[], Any]) -> None:
        nonlocal calls
        before = calls
        caught = False
        np.__dict__["array_equal"] = forged_array_equal
        try:
            try:
                operation()
            except (TypeError, ValueError, RuntimeError):
                caught = True
        finally:
            np.__dict__["array_equal"] = original_array_equal
        assert caught
        assert calls == before

    owner_id = "v5h9:test:owner-numpy-array-equal"
    assert_rejected_before_call(
        lambda: owner_api.bootstrap_live_boundary_owner(
            clean_gross,
            particle_cap=PARTICLE_CAP,
            owner_id=owner_id,
            wing_id=WING_ID,
            source_time_s=DELTA_TIME_S,
        )
    )
    owner = owner_api.bootstrap_live_boundary_owner(
        clean_gross,
        particle_cap=PARTICLE_CAP,
        owner_id=owner_id,
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )

    bad_candidate = _append_gross(clean_gross, 2)
    bad_positions = np.array(bad_candidate.positions, copy=True)
    bad_positions[-1, 2] = 0.125
    bad_candidate = replace(bad_candidate, positions=_readonly(bad_positions))
    assert sheet._validate_state(bad_candidate) is bad_candidate
    assert_rejected_before_call(
        lambda: _propose(
            owner,
            bad_candidate,
            proposal_id="owner-array-equal-bad-suffix",
        )
    )
    remesh = _propose(
        owner,
        bad_candidate,
        proposal_id="owner-array-equal-bad-suffix-clean-retry",
    )
    assert remesh.status == "remesh_required"
    assert remesh.first_mismatch == "positions"

    clean_candidate = _append_gross(clean_gross, 2)
    proposal = _propose(
        owner,
        clean_candidate,
        proposal_id="owner-array-equal-commit-retry",
    )
    assert_rejected_before_call(
        lambda: owner_api.commit_live_boundary_update(owner, proposal)
    )
    result = owner_api.commit_live_boundary_update(owner, proposal)

    assert_rejected_before_call(
        lambda: owner_api.validate_live_boundary_owner(result.owner)
    )
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner
    _assert_physical_equivalence(result.state, clean_candidate)


def test_releases_one_to_four_reduce_v5h8_gross_oracle_and_preserve_owner() -> None:
    gross = _first_gross_state()
    owner = _bootstrap(owner_id="v5h9:test:release-sequence")
    _assert_physical_equivalence(owner.state, gross)

    for release_index in range(2, 5):
        parent_owner = owner
        parent_state = parent_owner.state
        parent_count = parent_state.positions.shape[0]
        parent_boundary = parent_state.live_boundary_indices
        parent_events = parent_owner.events
        parent_fingerprint = _owner_fingerprint(parent_owner)
        gross = _append_gross(gross, release_index)

        proposal = _propose(parent_owner, gross)
        assert proposal.status == "compatible"
        assert proposal.parent_epoch == parent_owner.epoch
        assert proposal.changed_indices == parent_boundary
        assert proposal.planned_particle_count <= parent_owner.particle_cap
        assert proposal.appended_particle_count == (
            proposal.planned_particle_count - parent_count
        )
        assert proposal.clone_count == 0
        assert proposal.counter_particle_count == 0
        assert proposal.fresh_upstream_particle_count == 0
        assert isinstance(proposal.proposal_sha256, str)
        assert len(proposal.proposal_sha256) == 64

        result = owner_api.commit_live_boundary_update(parent_owner, proposal)
        owner = result.owner
        state = result.state
        assert owner.state is state
        assert owner.events[-1] is result.event
        assert owner.events[:-1] == parent_events
        assert _owner_fingerprint(parent_owner) == parent_fingerprint
        assert owner.epoch == parent_owner.epoch + 1
        assert state.state_epoch == owner.epoch
        assert state.release_index == release_index
        assert state.circulation_m2_s == release_index * DELTA_GAMMA_M2_PER_S
        assert state.positions.shape[0] == proposal.planned_particle_count
        _assert_readonly_particle_state(state)
        _assert_zero_gross_rows(state)
        _assert_physical_equivalence(state, gross)

        np.testing.assert_array_equal(
            state.positions[:parent_count], parent_state.positions
        )
        np.testing.assert_array_equal(state.sigma[:parent_count], parent_state.sigma)
        assert state.particle_ids[:parent_count] == parent_state.particle_ids
        assert state.lineage[:parent_count] == parent_state.lineage
        assert all(
            after is before
            for after, before in zip(
                state.lineage[:parent_count], parent_state.lineage, strict=True
            )
        )

        boundary_set = set(parent_boundary)
        unchanged = tuple(
            index for index in range(parent_count) if index not in boundary_set
        )
        np.testing.assert_array_equal(
            state.gamma[list(unchanged)], parent_state.gamma[list(unchanged)]
        )
        expected_scale = 1.0 - release_index / (release_index - 1.0)
        np.testing.assert_allclose(
            state.gamma[list(parent_boundary)],
            expected_scale * parent_state.gamma[list(parent_boundary)],
            rtol=0.0,
            atol=_tolerance(parent_state.gamma[list(parent_boundary)]),
        )

        assert result.event.changed_indices == parent_boundary
        assert result.event.clone_count == 0
        assert result.event.counter_particle_count == 0
        assert result.event.fresh_upstream_particle_count == 0
        assert result.event.parent_state_sha256 == parent_state.state_sha256
        assert result.event.committed_state_sha256 == state.state_sha256


def test_event_prefix_is_immutable_and_append_only() -> None:
    gross = _first_gross_state()
    owner = _bootstrap(owner_id="v5h9:test:event-prefix")
    historical: list[Any] = []

    for release_index in range(2, 5):
        gross = _append_gross(gross, release_index)
        old_owner = owner
        old_events = owner.events
        result = owner_api.commit_live_boundary_update(
            owner,
            _propose(
                owner,
                gross,
                proposal_id=f"event-prefix:{release_index}",
            ),
        )
        owner = result.owner
        assert old_owner.events == old_events
        assert owner.events[: len(old_events)] == old_events
        assert all(
            owner.events[index] is event for index, event in enumerate(historical)
        )
        historical.append(result.event)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        historical[0].proposal_id = "tampered"
    assert owner.events == tuple(historical)


def test_owner_never_calls_v5h8_collapse_to_construct_or_validate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _first_gross_state()
    candidate = _append_gross(first, 2)
    collapse_calls = 0

    def forbidden_collapse(*args: Any, **kwargs: Any) -> Any:
        nonlocal collapse_calls
        collapse_calls += 1
        raise AssertionError("v5h9 physical owner called the v5h8 collapse oracle")

    monkeypatch.setattr(sheet, "collapse_live_basis_pairs", forbidden_collapse)
    owner = owner_api.bootstrap_live_boundary_owner(
        first,
        particle_cap=PARTICLE_CAP,
        owner_id="v5h9:test:no-production-collapse",
        wing_id=WING_ID,
        source_time_s=DELTA_TIME_S,
    )
    proposal = _propose(owner, candidate, proposal_id="no-collapse:release:2")
    result = owner_api.commit_live_boundary_update(owner, proposal)
    assert collapse_calls == 0
    _assert_physical_equivalence(result.state, candidate)


def test_independent_oracle_detects_a_wrong_live_slice_update() -> None:
    first = _first_gross_state()
    owner = _bootstrap(owner_id="v5h9:test:oracle-sensitivity")
    candidate = _append_gross(first, 2)
    result = owner_api.commit_live_boundary_update(owner, _propose(owner, candidate))
    expected = _manual_canonical_snapshot(candidate)

    wrong_gamma = np.array(result.state.gamma, copy=True)
    wrong_gamma[list(owner.state.live_boundary_indices)] *= 0.9
    expected_field = direct_gaussian_erf_velocity_jacobian(
        expected.positions,
        expected.gamma,
        expected.sigma,
        target_positions=PROBES_M,
    )
    wrong_field = direct_gaussian_erf_velocity_jacobian(
        result.state.positions,
        wrong_gamma,
        result.state.sigma,
        target_positions=PROBES_M,
    )
    assert float(
        np.max(np.abs(wrong_field.velocity - expected_field.velocity))
    ) > 100.0 * _tolerance(expected_field.velocity)
    assert float(
        np.max(np.abs(wrong_field.jacobian - expected_field.jacobian))
    ) > 100.0 * _tolerance(expected_field.jacobian)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("gamma_scale", 123.0),
        ("operator_order", "gross_then_relax"),
        ("parent_write_count", 9),
    ),
)
def test_event_and_owner_replace_forgery_is_not_live(
    field: str,
    replacement: object,
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:event-forgery:{field}")
    candidate = _append_gross(_first_gross_state(), 2)
    result = owner_api.commit_live_boundary_update(owner, _propose(owner, candidate))
    forged_event = replace(result.event, **{field: replacement})
    forged_owner = replace(
        result.owner,
        events=result.owner.events[:-1] + (forged_event,),
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.validate_live_boundary_owner(forged_owner)
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner


def test_live_event_and_owner_content_are_bound_to_private_attestations() -> None:
    owner = _bootstrap(owner_id="v5h9:test:live-attestation-content")
    candidate = _append_gross(_first_gross_state(), 2)
    result = owner_api.commit_live_boundary_update(owner, _propose(owner, candidate))
    event = result.event

    original_scale = event.gamma_scale
    object.__setattr__(event, "gamma_scale", 321.0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.validate_live_boundary_owner(result.owner)
    object.__setattr__(event, "gamma_scale", original_scale)
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner

    original_wing = result.owner.wing_id
    object.__setattr__(result.owner, "wing_id", "wing:forged")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.validate_live_boundary_owner(result.owner)
    object.__setattr__(result.owner, "wing_id", original_wing)
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.validate_live_boundary_owner(replace(result.owner))


def test_live_state_digest_chain_cannot_be_publicly_resealed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:state-chain-reseal")
    candidate = _append_gross(_first_gross_state(), 2)
    result = owner_api.commit_live_boundary_update(owner, _propose(owner, candidate))
    state = result.state
    event = result.event
    original_positions = state.positions
    original_state_sha256 = state.state_sha256
    original_committed_sha256 = event.committed_state_sha256
    original_event_sha256 = event.event_sha256
    original_owner_sha256 = result.owner.owner_sha256
    forged_positions = np.array(state.positions, copy=True)
    forged_positions[0, 0] = np.nextafter(forged_positions[0, 0], np.inf)

    try:
        object.__setattr__(state, "positions", _readonly(forged_positions))
        object.__setattr__(state, "state_sha256", owner_api._state_digest(state))
        object.__setattr__(
            event,
            "committed_state_sha256",
            state.state_sha256,
        )
        object.__setattr__(event, "event_sha256", owner_api._event_digest(event))
        object.__setattr__(
            result.owner,
            "owner_sha256",
            owner_api._owner_digest(result.owner),
        )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.validate_live_boundary_owner(result.owner)
    finally:
        object.__setattr__(state, "positions", original_positions)
        object.__setattr__(state, "state_sha256", original_state_sha256)
        object.__setattr__(
            event,
            "committed_state_sha256",
            original_committed_sha256,
        )
        object.__setattr__(event, "event_sha256", original_event_sha256)
        object.__setattr__(
            result.owner,
            "owner_sha256",
            original_owner_sha256,
        )
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner


def test_live_event_digest_chain_cannot_be_publicly_resealed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:event-chain-reseal")
    candidate = _append_gross(_first_gross_state(), 2)
    result = owner_api.commit_live_boundary_update(owner, _propose(owner, candidate))
    event = result.event
    original_proposal_id = event.proposal_id
    original_event_sha256 = event.event_sha256
    original_owner_sha256 = result.owner.owner_sha256

    try:
        object.__setattr__(event, "proposal_id", "proposal:forged-event")
        object.__setattr__(event, "event_sha256", owner_api._event_digest(event))
        object.__setattr__(
            result.owner,
            "owner_sha256",
            owner_api._owner_digest(result.owner),
        )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.validate_live_boundary_owner(result.owner)
    finally:
        object.__setattr__(event, "proposal_id", original_proposal_id)
        object.__setattr__(event, "event_sha256", original_event_sha256)
        object.__setattr__(
            result.owner,
            "owner_sha256",
            original_owner_sha256,
        )
    assert owner_api.validate_live_boundary_owner(result.owner) is result.owner


def test_live_owner_cannot_be_publicly_resealed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:owner-reseal")
    original_cap = owner.particle_cap
    original_owner_sha256 = owner.owner_sha256

    try:
        object.__setattr__(owner, "particle_cap", original_cap + 1)
        object.__setattr__(owner, "owner_sha256", owner_api._owner_digest(owner))
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.validate_live_boundary_owner(owner)
    finally:
        object.__setattr__(owner, "particle_cap", original_cap)
        object.__setattr__(owner, "owner_sha256", original_owner_sha256)
    assert owner_api.validate_live_boundary_owner(owner) is owner


@pytest.mark.parametrize(
    "factory",
    (
        lambda gross: replace(
            gross,
            positions=_readonly(
                np.where(
                    np.indices(gross.positions.shape)[0] == 0,
                    np.nan,
                    gross.positions,
                )
            ),
        ),
        lambda gross: replace(
            gross,
            gamma=_readonly(
                np.where(
                    np.indices(gross.gamma.shape)[0] == 0,
                    np.inf,
                    gross.gamma,
                )
            ),
        ),
        lambda gross: replace(
            gross,
            sigma=_readonly(
                np.where(np.arange(len(gross.sigma)) == 0, np.nan, gross.sigma)
            ),
        ),
    ),
)
def test_nonfinite_candidate_is_rejected_without_owner_or_event_mutation(
    factory: Callable[[Any], Any],
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:nonfinite:{id(factory)}")
    candidate = factory(_append_gross(_first_gross_state(), 2))
    _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="nonfinite",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )


@pytest.mark.parametrize("gamma_previous", (0.0, np.nan, np.inf, True, "0.016"))
def test_invalid_previous_circulation_fails_before_transaction(
    gamma_previous: object,
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:bad-gamma:{gamma_previous!r}")
    candidate = _append_gross(_first_gross_state(), 2)
    _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            candidate,
            gamma_previous,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="bad-previous-gamma",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )


@pytest.mark.parametrize(
    ("gamma_previous", "gamma_current"),
    (
        (
            np.nextafter(DELTA_GAMMA_M2_PER_S, np.inf),
            2.0 * DELTA_GAMMA_M2_PER_S,
        ),
        (
            DELTA_GAMMA_M2_PER_S,
            np.nextafter(2.0 * DELTA_GAMMA_M2_PER_S, np.inf),
        ),
        (DELTA_GAMMA_M2_PER_S, 0.0),
        (DELTA_GAMMA_M2_PER_S, np.nan),
        (DELTA_GAMMA_M2_PER_S, np.inf),
        (DELTA_GAMMA_M2_PER_S, True),
        (DELTA_GAMMA_M2_PER_S, "0.032"),
    ),
)
def test_circulation_arguments_must_exactly_match_parent_and_candidate_panels(
    gamma_previous: object,
    gamma_current: object,
) -> None:
    owner = _bootstrap(
        owner_id=f"v5h9:test:circulation-argument:{gamma_previous!r}:{gamma_current!r}"
    )
    candidate = _append_gross(_first_gross_state(), 2)
    _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            candidate,
            gamma_previous,
            gamma_current,
            proposal_id="circulation-argument-mismatch",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )


@pytest.mark.parametrize(
    "candidate_circulation",
    (
        DELTA_GAMMA_M2_PER_S,
        -2.0 * DELTA_GAMMA_M2_PER_S,
        np.nextafter(0.0, 1.0),
    ),
)
def test_plateau_sign_reversal_and_near_zero_schedule_cannot_use_local_update(
    candidate_circulation: float,
) -> None:
    owner = _bootstrap(
        owner_id=f"v5h9:test:circulation-lifecycle:{candidate_circulation!r}"
    )
    candidate = _append_gross(_first_gross_state(), 2)
    forged_panel = replace(candidate.panels[-1], circulation_m2_s=candidate_circulation)
    forged = replace(
        candidate,
        panels=candidate.panels[:-1] + (forged_panel,),
    )
    _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            forged,
            DELTA_GAMMA_M2_PER_S,
            candidate_circulation,
            proposal_id="unsupported-circulation-lifecycle",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )


@pytest.mark.parametrize("field", ("positions", "sigma", "gamma"))
def test_one_ulp_latest_clone_mismatch_routes_fail_closed(field: str) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:ulp:{field}")
    candidate = _append_gross(_first_gross_state(), 2)
    clone_index = _latest_clone_indices(candidate)[0]
    values = np.array(getattr(candidate, field), copy=True)
    if field == "sigma":
        values[clone_index] = np.nextafter(values[clone_index], np.inf)
    else:
        component = int(np.flatnonzero(values[clone_index] != 0.0)[0])
        values[clone_index, component] = np.nextafter(
            values[clone_index, component], np.inf
        )
    forged = replace(candidate, **{field: _readonly(values)})
    proposal = _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            forged,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id=f"ulp:{field}",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )
    assert proposal is not None
    assert proposal.first_mismatch in {field, f"clone_{field}", "candidate_state"}


def test_count_order_and_particle_id_mismatch_are_never_welded() -> None:
    candidate = _append_gross(_first_gross_state(), 2)
    latest_pairs = tuple(
        pair
        for pair in candidate.clone_pairs
        if candidate.lineage[pair[1]].release_index == 2
    )
    assert len(latest_pairs) > 1
    forged_candidates = (
        replace(candidate, clone_pairs=candidate.clone_pairs[:-1]),
        replace(
            candidate,
            clone_pairs=(latest_pairs[1], latest_pairs[0]) + latest_pairs[2:],
        ),
        replace(
            candidate,
            particle_ids=(
                candidate.particle_ids[: latest_pairs[0][1]]
                + (("forged", "particle", "id"),)
                + candidate.particle_ids[latest_pairs[0][1] + 1 :]
            ),
        ),
    )
    for mismatch, forged in zip(
        ("count", "order", "particle_id"), forged_candidates, strict=True
    ):
        owner = _bootstrap(owner_id=f"v5h9:test:mismatch:{mismatch}")
        proposal = _assert_rejected_transactionally(
            owner,
            lambda forged=forged, mismatch=mismatch: owner_api.propose_live_boundary_update(
                owner,
                owner.state,
                forged,
                DELTA_GAMMA_M2_PER_S,
                2.0 * DELTA_GAMMA_M2_PER_S,
                proposal_id=f"mismatch:{mismatch}",
                wing_id=WING_ID,
                source_time_s=2.0 * DELTA_TIME_S,
            ),
        )
        assert proposal is not None
        assert proposal.first_mismatch is not None


@pytest.mark.parametrize("field", ("positions", "gamma", "sigma"))
def test_one_ulp_genuinely_new_suffix_tamper_requires_remesh(field: str) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:fresh-suffix:{field}")
    candidate = _append_gross(_first_gross_state(), 2)
    fresh_index = next(
        index
        for index, record in enumerate(candidate.lineage)
        if record.release_index == 2
        and record.role in {"fresh_left_tip", "fresh_right_tip", "fresh_downstream"}
    )
    values = np.array(getattr(candidate, field), copy=True)
    if field == "sigma":
        values[fresh_index] = np.nextafter(values[fresh_index], np.inf)
    else:
        nonzero = np.flatnonzero(values[fresh_index] != 0.0)
        component = int(nonzero[0]) if nonzero.size else 0
        values[fresh_index, component] = np.nextafter(
            values[fresh_index, component],
            np.inf,
        )
    forged = replace(candidate, **{field: _readonly(values)})
    proposal = owner_api.propose_live_boundary_update(
        owner,
        owner.state,
        forged,
        DELTA_GAMMA_M2_PER_S,
        2.0 * DELTA_GAMMA_M2_PER_S,
        proposal_id=f"fresh-suffix:{field}",
        wing_id=WING_ID,
        source_time_s=2.0 * DELTA_TIME_S,
    )
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch in {field, f"fresh_{field}", "candidate_state"}
    assert owner.events == ()


@pytest.mark.parametrize("mode", ("value", "order", "ulp", "signature"))
def test_exact_edge_incidence_tamper_requires_remesh_and_clean_same_object_retry(
    mode: str,
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:exact-incidence:{mode}")
    candidate = _append_gross(_first_gross_state(), 2)
    original_ids = candidate.particle_ids
    original_lineage = candidate.lineage
    forged_ids, forged_lineage = _nested_incidence_mutation(candidate, mode)

    try:
        object.__setattr__(candidate, "particle_ids", forged_ids)
        object.__setattr__(candidate, "lineage", forged_lineage)
        assert sheet._validate_state(candidate) is candidate
        proposal = _assert_rejected_transactionally(
            owner,
            lambda: _propose(
                owner,
                candidate,
                proposal_id=f"exact-incidence:{mode}",
            ),
        )
        assert proposal is not None
        expected_mismatch = "particle_id" if mode == "signature" else "lineage"
        assert proposal.first_mismatch == expected_mismatch
    finally:
        object.__setattr__(candidate, "particle_ids", original_ids)
        object.__setattr__(candidate, "lineage", original_lineage)

    clean_proposal = _propose(
        owner,
        candidate,
        proposal_id=f"exact-incidence:{mode}:clean-retry",
    )
    result = owner_api.commit_live_boundary_update(owner, clean_proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_spoofed_incidence_never_runs_equality_and_commit_rechecks_issued_suffix() -> (
    None
):
    owner = _bootstrap(owner_id="v5h9:test:foreign-incidence")
    candidate = _append_gross(_first_gross_state(), 2)
    index = _fresh_suffix_index(candidate)
    original_lineage = candidate.lineage
    record = original_lineage[index]
    source = record.source_lineage
    assert source is not None
    incidence = source.ring_incidences[0]
    equality_calls = 0

    @dataclass(frozen=True, eq=False)
    class ForeignEdgeIncidence:
        ring_id: Any
        traversal_index: int
        source_start_id: Any
        source_end_id: Any
        canonical_sign: int
        ring_circulation: float
        signed_circulation: float

        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            return True

    ForeignEdgeIncidence.__module__ = "fluxvortex.rvpm_edge_bridge"
    ForeignEdgeIncidence.__qualname__ = "EdgeIncidence"
    foreign = ForeignEdgeIncidence(
        **{field.name: getattr(incidence, field.name) for field in fields(incidence)}
    )
    forged_lineage = list(original_lineage)
    forged_lineage[index] = replace(
        record,
        source_lineage=replace(source, ring_incidences=(foreign,)),
    )
    forged_lineage_tuple = tuple(forged_lineage)
    clean_digest = owner_api._gross_digest(candidate)

    try:
        object.__setattr__(candidate, "lineage", forged_lineage_tuple)
        assert sheet._validate_state(candidate) is candidate
        assert owner_api._gross_digest(candidate) == clean_digest
        rejected = _assert_rejected_transactionally(
            owner,
            lambda: _propose(
                owner,
                candidate,
                proposal_id="foreign-incidence:proposal",
            ),
        )
        assert rejected is not None
        assert rejected.first_mismatch == "lineage"
        assert equality_calls == 0
    finally:
        object.__setattr__(candidate, "lineage", original_lineage)

    proposal = _propose(
        owner,
        candidate,
        proposal_id="foreign-incidence:commit",
    )
    before = _owner_fingerprint(owner)
    try:
        object.__setattr__(candidate, "lineage", forged_lineage_tuple)
        assert owner_api._gross_digest(candidate) == clean_digest
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            owner_api.commit_live_boundary_update(owner, proposal)
        assert _owner_fingerprint(owner) == before
        assert equality_calls == 0
    finally:
        object.__setattr__(candidate, "lineage", original_lineage)

    # The failed commit consumed neither issuance nor owner generation.
    result = owner_api.commit_live_boundary_update(owner, proposal)
    committed = next(
        item
        for item in result.state.lineage
        if item.release_index == 2 and item.particle_id == record.particle_id
    )
    assert committed is not record
    assert committed.source_lineage is not source
    assert committed.source_lineage.ring_incidences[0] is not incidence
    assert equality_calls == 0
    _assert_physical_equivalence(result.state, candidate)


@pytest.mark.parametrize(
    ("wing_id", "source_time_s"),
    (
        ("wing:wrong", 2.0 * DELTA_TIME_S),
        (WING_ID, DELTA_TIME_S),
        (WING_ID, np.nan),
        (WING_ID, np.inf),
    ),
)
def test_wing_and_source_time_mismatch_are_transactional(
    wing_id: str,
    source_time_s: float,
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:clock:{wing_id}:{source_time_s}")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="clock-wing-mismatch",
            wing_id=wing_id,
            source_time_s=source_time_s,
        ),
    )
    if proposal is not None:
        assert proposal.first_mismatch in {"wing_id", "source_time_s"}


def test_wrong_types_and_equal_but_nonidentical_parent_fail_closed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:identity")
    candidate = _append_gross(_first_gross_state(), 2)
    forged_parent = replace(owner.state)

    operations = (
        lambda: owner_api.propose_live_boundary_update(
            object(),
            owner.state,
            candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="wrong-owner",
        ),
        lambda: owner_api.propose_live_boundary_update(
            owner,
            forged_parent,
            candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="wrong-parent-identity",
        ),
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            object(),
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="wrong-candidate",
        ),
        lambda: owner_api.commit_live_boundary_update(owner, object()),
    )
    for operation in operations:
        _assert_rejected_transactionally(owner, operation)


def test_proposal_is_exact_once_and_cannot_be_replayed() -> None:
    owner = _bootstrap(owner_id="v5h9:test:replay")
    gross = _append_gross(_first_gross_state(), 2)
    proposal = _propose(owner, gross, proposal_id="replay:release:2")
    result = owner_api.commit_live_boundary_update(owner, proposal)
    committed_fingerprint = _owner_fingerprint(result.owner)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError), match="replay|stale|commit"
    ):
        owner_api.commit_live_boundary_update(owner, proposal)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError), match="replay|stale|epoch"
    ):
        owner_api.commit_live_boundary_update(result.owner, proposal)
    assert _owner_fingerprint(result.owner) == committed_fingerprint


def test_parent_generation_is_exact_once_across_sibling_proposals() -> None:
    owner = _bootstrap(owner_id="v5h9:test:owner-generation-exact-once")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal_a = _propose(owner, candidate, proposal_id="sibling:a")
    proposal_b = _propose(owner, candidate, proposal_id="sibling:b")
    parent_fingerprint = _owner_fingerprint(owner)

    committed = owner_api.commit_live_boundary_update(owner, proposal_a)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError), match="stale|consum|epoch"
    ):
        owner_api.commit_live_boundary_update(owner, proposal_b)
    assert _owner_fingerprint(owner) == parent_fingerprint
    assert owner_api.validate_live_boundary_owner(committed.owner) is committed.owner
    assert len(committed.owner.events) == 1


@pytest.mark.parametrize("commit_once", (False, True))
def test_owner_id_claim_survives_gc_and_cannot_restart_genesis(
    commit_once: bool,
) -> None:
    owner_id = f"v5h9:test:owner-id-gc-tombstone:{commit_once}"
    owner = _bootstrap(owner_id=owner_id)
    latest_owner = owner
    if commit_once:
        candidate = _append_gross(_first_gross_state(), 2)
        committed = owner_api.commit_live_boundary_update(
            owner,
            _propose(owner, candidate, proposal_id="owner-id-gc-generation-two"),
        )
        latest_owner = committed.owner
        del committed
    latest_reference = weakref.ref(latest_owner)
    del latest_owner
    del owner
    gc.collect()

    latest = latest_reference()
    assert latest is not None
    assert latest.epoch == (2 if commit_once else 1)
    with pytest.raises((TypeError, ValueError, RuntimeError), match="owner_id|current"):
        owner_api.bootstrap_live_boundary_owner(
            _first_gross_state(),
            particle_cap=PARTICLE_CAP,
            owner_id=owner_id,
            wing_id=WING_ID,
            source_time_s=3.0 * DELTA_TIME_S,
        )
    assert owner_api.validate_live_boundary_owner(latest) is latest


@pytest.mark.parametrize(
    "field",
    tuple(field.name for field in fields(owner_api.UpdateProposal)),
)
def test_every_bound_proposal_field_is_rechecked_and_clean_retry_survives(
    field: str,
) -> None:
    owner = _bootstrap(owner_id=f"v5h9:test:proposal-field-binding:{field}")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _propose(
        owner,
        candidate,
        proposal_id=f"proposal-field-binding:{field}",
    )
    mutations = _proposal_field_mutations(proposal, candidate)
    assert set(mutations) == {
        proposal_field.name for proposal_field in fields(owner_api.UpdateProposal)
    }
    original = getattr(proposal, field)
    original_sha256 = proposal.proposal_sha256
    parent_fingerprint = _owner_fingerprint(owner)

    object.__setattr__(proposal, field, mutations[field])
    if field != "proposal_sha256":
        # A caller-visible digest helper is not an issuance authority.
        object.__setattr__(
            proposal,
            "proposal_sha256",
            owner_api._proposal_digest(proposal),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, proposal)
    assert _owner_fingerprint(owner) == parent_fingerprint

    object.__setattr__(proposal, field, original)
    object.__setattr__(proposal, "proposal_sha256", original_sha256)
    result = owner_api.commit_live_boundary_update(owner, proposal)
    assert result.event.proposal_id == proposal.proposal_id
    assert result.event.gamma_scale == proposal.gamma_scale
    _assert_physical_equivalence(result.state, candidate)


def test_combined_proposal_reseal_cannot_forge_event_or_counters() -> None:
    owner = _bootstrap(owner_id="v5h9:test:proposal-combined-reseal")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _propose(owner, candidate, proposal_id="proposal:issued")
    fields_to_forge = {
        "proposal_id": "proposal:forged",
        "wing_id": "wing:forged",
        "appended_particle_count": proposal.appended_particle_count + 7,
        "first_mismatch": "forged",
        "clone_count": 3,
        "counter_particle_count": 4,
        "fresh_upstream_particle_count": 5,
    }
    originals = {name: getattr(proposal, name) for name in fields_to_forge}
    original_sha256 = proposal.proposal_sha256
    parent_fingerprint = _owner_fingerprint(owner)

    for name, value in fields_to_forge.items():
        object.__setattr__(proposal, name, value)
    object.__setattr__(
        proposal,
        "proposal_sha256",
        owner_api._proposal_digest(proposal),
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, proposal)
    assert _owner_fingerprint(owner) == parent_fingerprint
    assert owner.events == ()

    for name, value in originals.items():
        object.__setattr__(proposal, name, value)
    object.__setattr__(proposal, "proposal_sha256", original_sha256)
    result = owner_api.commit_live_boundary_update(owner, proposal)
    assert result.event.proposal_id == "proposal:issued"
    assert result.event.clone_count == 0
    assert result.event.counter_particle_count == 0
    assert result.event.fresh_upstream_particle_count == 0
    _assert_physical_equivalence(result.state, candidate)


def test_replaced_and_resealed_proposal_is_not_the_issued_capability() -> None:
    owner = _bootstrap(owner_id="v5h9:test:proposal-replace-reseal")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _propose(owner, candidate, proposal_id="proposal:original")
    forged = replace(proposal, proposal_id="proposal:replaced")
    object.__setattr__(
        forged,
        "proposal_sha256",
        owner_api._proposal_digest(forged),
    )
    parent_fingerprint = _owner_fingerprint(owner)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, forged)
    assert _owner_fingerprint(owner) == parent_fingerprint

    result = owner_api.commit_live_boundary_update(owner, proposal)
    assert result.event.proposal_id == "proposal:original"
    _assert_physical_equivalence(result.state, candidate)


def test_foreign_proposal_field_type_is_rejected_without_comparison() -> None:
    owner = _bootstrap(owner_id="v5h9:test:proposal-foreign-field-type")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _propose(owner, candidate, proposal_id="proposal:foreign-type")
    original_sha256 = proposal.proposal_sha256
    comparison_calls = 0

    class HostileDigest:
        def __eq__(self, other: object) -> bool:
            nonlocal comparison_calls
            comparison_calls += 1
            raise AssertionError("foreign proposal field comparator executed")

        def __ne__(self, other: object) -> bool:
            nonlocal comparison_calls
            comparison_calls += 1
            raise AssertionError("foreign proposal field comparator executed")

    object.__setattr__(proposal, "proposal_sha256", HostileDigest())
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, proposal)
    assert comparison_calls == 0

    object.__setattr__(proposal, "proposal_sha256", original_sha256)
    result = owner_api.commit_live_boundary_update(owner, proposal)
    assert result.event.proposal_id == "proposal:foreign-type"
    _assert_physical_equivalence(result.state, candidate)


def test_post_proposal_oracle_callable_changes_are_not_executed_by_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _bootstrap(owner_id="v5h9:test:post-proposal-binding")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = _propose(owner, candidate, proposal_id="post-proposal-binding")
    validator_calls = 0
    collapse_calls = 0

    def forbidden_validator(*args: Any, **kwargs: Any) -> Any:
        nonlocal validator_calls
        validator_calls += 1
        raise AssertionError("commit re-entered mutable v5h8 validation")

    def forbidden_collapse(*args: Any, **kwargs: Any) -> Any:
        nonlocal collapse_calls
        collapse_calls += 1
        raise AssertionError("commit called mutable v5h8 collapse")

    original_validator = sheet._validate_state
    original_collapse = sheet.collapse_live_basis_pairs
    monkeypatch.setattr(sheet, "_validate_state", forbidden_validator)
    monkeypatch.setattr(sheet, "collapse_live_basis_pairs", forbidden_collapse)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, proposal)
    assert validator_calls == 0
    assert collapse_calls == 0
    monkeypatch.setattr(sheet, "_validate_state", original_validator)
    monkeypatch.setattr(sheet, "collapse_live_basis_pairs", original_collapse)
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_preproposal_validator_callable_change_is_rejected_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _bootstrap(owner_id="v5h9:test:pre-proposal-binding")
    candidate = _append_gross(_first_gross_state(), 2)
    calls = 0

    def forged_validator(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return args[0]

    original = sheet._validate_state
    monkeypatch.setattr(sheet, "_validate_state", forged_validator)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _propose(owner, candidate, proposal_id="forged-validator")
    assert calls == 0
    monkeypatch.setattr(sheet, "_validate_state", original)

    proposal = _propose(owner, candidate, proposal_id="forged-validator")
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)


def test_release_time_must_match_the_frozen_release_cadence() -> None:
    owner = _bootstrap(owner_id="v5h9:test:exact-release-time")
    candidate = _append_gross(_first_gross_state(), 2)
    proposal = owner_api.propose_live_boundary_update(
        owner,
        owner.state,
        candidate,
        DELTA_GAMMA_M2_PER_S,
        2.0 * DELTA_GAMMA_M2_PER_S,
        proposal_id="arbitrary-future-time",
        wing_id=WING_ID,
        source_time_s=999.0,
    )
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch == "source_time_s"
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.commit_live_boundary_update(owner, proposal)
    assert owner.events == ()


def test_particle_cap_is_checked_before_committed_net_materialization() -> None:
    first = _first_gross_state()
    candidate = _append_gross(first, 2)
    expected_count = candidate.positions.shape[0] - len(candidate.clone_pairs)
    assert expected_count > first.positions.shape[0]
    owner = _bootstrap(
        particle_cap=expected_count - 1,
        owner_id="v5h9:test:preallocation-cap",
    )

    proposal = _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="over-cap",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )
    assert proposal is not None
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch == "particle_cap"
    assert proposal.planned_particle_count == expected_count
    assert owner.events == ()


def test_failed_proposal_allows_a_clean_retry_equal_to_fresh_oracle() -> None:
    owner = _bootstrap(owner_id="v5h9:test:clean-retry")
    candidate = _append_gross(_first_gross_state(), 2)
    bad_positions = candidate.positions.copy()
    clone_index = _latest_clone_indices(candidate)[0]
    component = int(np.flatnonzero(bad_positions[clone_index] != 0.0)[0])
    bad_positions[clone_index, component] = np.nextafter(
        bad_positions[clone_index, component], np.inf
    )
    bad_candidate = replace(candidate, positions=_readonly(bad_positions))

    _assert_rejected_transactionally(
        owner,
        lambda: owner_api.propose_live_boundary_update(
            owner,
            owner.state,
            bad_candidate,
            DELTA_GAMMA_M2_PER_S,
            2.0 * DELTA_GAMMA_M2_PER_S,
            proposal_id="retry:same-id",
            wing_id=WING_ID,
            source_time_s=2.0 * DELTA_TIME_S,
        ),
    )

    proposal = owner_api.propose_live_boundary_update(
        owner,
        owner.state,
        candidate,
        DELTA_GAMMA_M2_PER_S,
        2.0 * DELTA_GAMMA_M2_PER_S,
        proposal_id="retry:same-id",
        wing_id=WING_ID,
        source_time_s=2.0 * DELTA_TIME_S,
    )
    assert proposal.status == "compatible"
    result = owner_api.commit_live_boundary_update(owner, proposal)
    _assert_physical_equivalence(result.state, candidate)
    assert len(result.owner.events) == 1
    assert result.event.proposal_id == "retry:same-id"


@pytest.mark.parametrize(
    ("state", "particle_cap", "owner_id", "wing_id", "source_time_s"),
    (
        (object(), PARTICLE_CAP, "owner", WING_ID, DELTA_TIME_S),
        (None, True, "owner", WING_ID, DELTA_TIME_S),
        (None, 0, "owner", WING_ID, DELTA_TIME_S),
        (None, PARTICLE_CAP, "", WING_ID, DELTA_TIME_S),
        (None, PARTICLE_CAP, "owner", "", DELTA_TIME_S),
        (None, PARTICLE_CAP, "owner", WING_ID, np.nan),
    ),
)
def test_bad_constructor_inputs_fail_closed(
    state: object,
    particle_cap: object,
    owner_id: object,
    wing_id: object,
    source_time_s: object,
) -> None:
    candidate = _first_gross_state() if state is None else state
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owner_api.bootstrap_live_boundary_owner(
            candidate,
            particle_cap=particle_cap,
            owner_id=owner_id,
            wing_id=wing_id,
            source_time_s=source_time_s,
        )
