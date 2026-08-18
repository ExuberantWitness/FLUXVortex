"""Mechanical and hostile tests for the v5h10 global release-row owner."""

from __future__ import annotations

from dataclasses import replace
from itertools import count
from math import fsum
from weakref import WeakKeyDictionary

import numpy as np
import pytest

import fluxvortex.rvpm_edge_bridge as edge_bridge
from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DirectedRing,
    EdgeIncidence,
    assemble_ring_edge_graph,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from fluxvortex.rvpm_transport import make_particle_state

import forward_flight_benchmarks.fluxv_v5h10_row_owner as row_owner
from forward_flight_benchmarks.fluxv_v5h4_ptera_rvpm_transport import (
    FrozenExternalField,
    lsrk3_step_with_external_field,
)
from forward_flight_benchmarks.fluxv_v5h10_row_owner import (
    ReleaseRow,
    ReleaseRowOwner,
    RowParticleLineage,
    advance_release_row_transport_parent,
    attest_release_row_common_transport,
    begin_release_row_common_transport,
    bootstrap_release_row_owner,
    commit_release_row_update,
    make_release_row,
    propose_release_row_update,
    release_row_transport_digest,
    validate_current_release_row_owner,
    validate_release_row_common_transport,
    validate_release_row_owner,
    validate_release_row_transport_attestation,
)


SIGMA = 0.25
SPACING = 0.10
ROOT_TIME = 1.0
DT = 0.5
_OWNER_NUMBER = count()


def _owner_id(label: str) -> str:
    return f"v5h10-test:{label}:{next(_OWNER_NUMBER)}"


def _row(
    cell_count: int,
    release: int,
    *,
    sign: float = 1.0,
    source_time_s: float | None = None,
    sheet_id: str = "straight-row",
) -> ReleaseRow:
    y = np.linspace(-1.2, 1.2, cell_count + 1)
    upstream = np.column_stack(
        (
            np.full(cell_count + 1, 0.4 * (release - 1)),
            y,
            np.zeros(cell_count + 1),
        )
    )
    downstream = upstream.copy()
    downstream[:, 0] += 0.4
    base = 0.75 + 0.04 * np.arange(cell_count)
    circulation = sign * (1.0 + 0.30 * (release - 1)) * base
    return make_release_row(
        upstream,
        downstream,
        circulation,
        release_index=release,
        source_time_s=(
            ROOT_TIME + (release - 1) * DT if source_time_s is None else source_time_s
        ),
        sheet_id=sheet_id,
    )


def _bootstrap(cell_count: int, *, cap: int = 100_000) -> ReleaseRowOwner:
    return bootstrap_release_row_owner(
        _row(cell_count, 1),
        smoothing_radius_m=SIGMA,
        target_spacing_m=SPACING,
        release_dt_s=DT,
        particle_cap=cap,
        owner_id=_owner_id(f"{cell_count}-cell"),
    )


def _identity_transport(
    owner: ReleaseRowOwner, *, source_step_index: int
) -> ReleaseRowOwner:
    state = owner.state
    end_time = state.rows[-1].source_time_s + owner.release_dt_s
    _, attestation, digest = _prepare_transport_handoff(
        owner,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        source_step_index=source_step_index,
    )
    return advance_release_row_transport_parent(
        owner,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )


def _prepare_transport_handoff(
    owner: ReleaseRowOwner,
    positions: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    nodes: np.ndarray,
    *,
    source_step_index: int,
    transported_material_tracers: np.ndarray | None = None,
    transported_live_material_sigma: np.ndarray | None = None,
):
    state = owner.state
    session = begin_release_row_common_transport(owner, state)
    assert validate_release_row_common_transport(session) is session
    flat = tuple(
        index for indices in session.live_particle_indices_by_cell for index in indices
    )
    if transported_material_tracers is None:
        transported_material_tracers = np.vstack(
            (positions[np.asarray(flat, dtype=np.int64)], nodes)
        )
    if transported_live_material_sigma is None:
        transported_live_material_sigma = sigma[np.asarray(flat, dtype=np.int64)]
    end_time = state.rows[-1].source_time_s + owner.release_dt_s
    attestation = attest_release_row_common_transport(
        session,
        positions,
        gamma,
        sigma,
        transported_material_tracers,
        transported_live_material_sigma,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    assert validate_release_row_transport_attestation(attestation) is attestation
    digest = release_row_transport_digest(
        state,
        positions,
        gamma,
        sigma,
        nodes,
        common_transport_attestation=attestation,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    return session, attestation, digest


def _direct_snapshot(rows: tuple[ReleaseRow, ...]):
    first = rows[0]
    nodes = [
        BridgeNode(
            f"v5h10:{first.sheet_id}:plane:0:span:{span}",
            tuple(point),
        )
        for span, point in enumerate(first.upstream_nodes)
    ]
    for row in rows:
        nodes.extend(
            BridgeNode(
                f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{span}",
                tuple(point),
            )
            for span, point in enumerate(row.downstream_nodes)
        )
    rings = []
    for row in rows:
        for cell, circulation in enumerate(row.circulation_m2_s):
            rings.append(
                DirectedRing(
                    ring_id=(
                        f"v5h10:{row.sheet_id}:release:"
                        f"{row.release_index}:cell:{cell}"
                    ),
                    node_ids=(
                        f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell}",
                        f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell + 1}",
                        f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{cell + 1}",
                        f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{cell}",
                    ),
                    circulation=float(circulation),
                )
            )
    graph = assemble_ring_edge_graph(tuple(nodes), tuple(rings))
    return deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=SIGMA,
        target_spacing=SPACING,
        step=len(rows),
    )


def _single_row_snapshot(row: ReleaseRow):
    nodes = []
    for span, point in enumerate(row.upstream_nodes):
        nodes.append(
            BridgeNode(
                f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{span}",
                tuple(point),
            )
        )
    for span, point in enumerate(row.downstream_nodes):
        nodes.append(
            BridgeNode(
                f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{span}",
                tuple(point),
            )
        )
    rings = tuple(
        DirectedRing(
            ring_id=f"v5h10:{row.sheet_id}:release:{row.release_index}:cell:{cell}",
            node_ids=(
                f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell}",
                f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell + 1}",
                f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{cell + 1}",
                f"v5h10:{row.sheet_id}:plane:{row.release_index}:span:{cell}",
            ),
            circulation=float(circulation),
        )
        for cell, circulation in enumerate(row.circulation_m2_s)
    )
    graph = assemble_ring_edge_graph(tuple(nodes), rings)
    deposited = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=SIGMA,
        target_spacing=SPACING,
        step=row.release_index,
    )
    upstream_keys = tuple(
        edge_bridge.canonical_edge_key(
            f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell}",
            f"v5h10:{row.sheet_id}:plane:{row.release_index - 1}:span:{cell + 1}",
        )
        for cell in range(row.circulation_m2_s.size)
    )
    kept = tuple(
        index
        for index, record in enumerate(deposited.lineage)
        if record.source_edge not in set(upstream_keys)
    )
    edge_by_key = {edge.key: edge for edge in graph.retained_edges}
    return deposited, kept, upstream_keys, edge_by_key


def _impulse(positions: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    cross = np.cross(positions, gamma)
    return np.asarray(
        [0.5 * fsum(float(item) for item in cross[:, axis]) for axis in range(3)]
    )


def _assert_mechanics(owner: ReleaseRowOwner) -> None:
    direct = _direct_snapshot(owner.state.rows)
    probes = np.asarray(
        (
            (-0.31, -0.83, 0.37),
            (0.73, 0.19, -0.42),
            (1.67, 0.91, 0.58),
        ),
        dtype=np.float64,
    )
    actual = direct_gaussian_erf_velocity_jacobian(
        owner.state.positions,
        owner.state.gamma,
        owner.state.sigma,
        target_positions=probes,
    )
    expected = direct_gaussian_erf_velocity_jacobian(
        direct.positions,
        direct.gamma,
        direct.sigma,
        target_positions=probes,
    )
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        actual.velocity, expected.velocity, rtol=0.0, atol=128 * eps
    )
    np.testing.assert_allclose(
        actual.jacobian, expected.jacobian, rtol=0.0, atol=256 * eps
    )
    np.testing.assert_allclose(
        _impulse(owner.state.positions, owner.state.gamma),
        _impulse(direct.positions, direct.gamma),
        rtol=0.0,
        atol=128 * eps,
    )


def _assert_arrays_mechanics(
    owner: ReleaseRowOwner,
    positions: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
) -> None:
    probes = np.asarray(
        ((-0.22, -0.77, 0.41), (0.91, 0.23, -0.38), (1.74, 0.84, 0.63)),
        dtype=np.float64,
    )
    actual = direct_gaussian_erf_velocity_jacobian(
        owner.state.positions,
        owner.state.gamma,
        owner.state.sigma,
        target_positions=probes,
    )
    expected = direct_gaussian_erf_velocity_jacobian(
        positions,
        gamma,
        sigma,
        target_positions=probes,
    )
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        actual.velocity, expected.velocity, rtol=0.0, atol=256 * eps
    )
    np.testing.assert_allclose(
        actual.jacobian, expected.jacobian, rtol=0.0, atol=512 * eps
    )
    np.testing.assert_allclose(
        _impulse(owner.state.positions, owner.state.gamma),
        _impulse(positions, gamma),
        rtol=0.0,
        atol=256 * eps,
    )


def _nonlinear_external_field(points: np.ndarray) -> FrozenExternalField:
    targets = np.asarray(points, dtype=np.float64)
    x = targets[:, 0]
    y = targets[:, 1]
    z = targets[:, 2]
    velocity = np.column_stack(
        (
            0.08 + 0.04 * y * y + 0.01 * z,
            0.03 * x * y - 0.02 * z,
            0.05 * y + 0.02 * x * x,
        )
    )
    jacobian = np.zeros((targets.shape[0], 3, 3), dtype=np.float64)
    jacobian[:, 0, 1] = 0.08 * y
    jacobian[:, 0, 2] = 0.01
    jacobian[:, 1, 0] = 0.03 * y
    jacobian[:, 1, 1] = 0.03 * x
    jacobian[:, 1, 2] = -0.02
    jacobian[:, 2, 0] = 0.04 * x
    jacobian[:, 2, 1] = 0.05
    return FrozenExternalField(velocity=velocity, jacobian=jacobian)


def _lsrk3_common_transport(
    owner: ReleaseRowOwner,
    *,
    source_step_index: int,
):
    session = begin_release_row_common_transport(owner, owner.state)
    physical = make_particle_state(
        owner.state.positions, owner.state.gamma, owner.state.sigma
    )
    result = lsrk3_step_with_external_field(
        physical,
        owner.release_dt_s,
        external_field=_nonlinear_external_field,
        baseline_freestream_velocity_gp1_m_per_s=np.zeros(3),
        enabled=True,
    )
    tracers = np.array(session.material_tracer_positions, copy=True)
    storage = np.zeros_like(tracers)
    for stage in result.stages:
        self_velocity = direct_gaussian_erf_velocity_jacobian(
            stage.pre.positions,
            stage.pre.gamma,
            stage.pre.sigma,
            target_positions=tracers,
        ).velocity
        external_velocity = _nonlinear_external_field(tracers).velocity
        storage = stage.a * storage + owner.release_dt_s * (
            self_velocity + external_velocity
        )
        tracers = tracers + stage.b * storage
    flat = tuple(
        index for indices in session.live_particle_indices_by_cell for index in indices
    )
    flat_array = np.asarray(flat, dtype=np.int64)
    np.testing.assert_array_equal(
        tracers[: session.frontier_node_offset],
        result.final_state.positions[flat_array],
    )
    live_sigma = result.final_state.sigma[flat_array]
    end_time = owner.state.rows[-1].source_time_s + owner.release_dt_s
    attestation = attest_release_row_common_transport(
        session,
        result.final_state.positions,
        result.final_state.gamma,
        result.final_state.sigma,
        tracers,
        live_sigma,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    nodes = attestation.transported_live_boundary_nodes
    digest = release_row_transport_digest(
        owner.state,
        result.final_state.positions,
        result.final_state.gamma,
        result.final_state.sigma,
        nodes,
        common_transport_attestation=attestation,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    advanced = advance_release_row_transport_parent(
        owner,
        owner.state,
        result.final_state.positions,
        result.final_state.gamma,
        result.final_state.sigma,
        nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=source_step_index,
        transport_end_time_s=end_time,
        transport_epoch=owner.epoch,
    )
    return advanced, result, session, attestation, tracers


def _assert_cloud_mechanics(
    owner: ReleaseRowOwner,
    positions: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
) -> None:
    probes = np.asarray(((-0.37, -0.88, 0.29), (0.82, 0.16, -0.46), (1.51, 0.93, 0.61)))
    actual = direct_gaussian_erf_velocity_jacobian(
        owner.state.positions,
        owner.state.gamma,
        owner.state.sigma,
        target_positions=probes,
    )
    expected = direct_gaussian_erf_velocity_jacobian(
        positions,
        gamma,
        sigma,
        target_positions=probes,
    )
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        actual.velocity, expected.velocity, rtol=0, atol=512 * eps
    )
    np.testing.assert_allclose(
        actual.jacobian, expected.jacobian, rtol=0, atol=1024 * eps
    )
    np.testing.assert_allclose(
        _impulse(owner.state.positions, owner.state.gamma),
        _impulse(positions, gamma),
        rtol=0,
        atol=512 * eps,
    )


def test_eight_cell_three_release_lsrk3_material_basis_matches_clone_collapse() -> None:
    transport_dt = 0.01
    owner = bootstrap_release_row_owner(
        _row(8, 1),
        smoothing_radius_m=SIGMA,
        target_spacing_m=SPACING,
        release_dt_s=transport_dt,
        particle_cap=100_000,
        owner_id=_owner_id("8-cell-lsrk-material-basis"),
    )
    for release, source_step in ((2, 4), (3, 5)):
        transported, lsrk, session, attestation, tracers = _lsrk3_common_transport(
            owner, source_step_index=source_step
        )
        assert len(lsrk.stages) == 3
        assert not np.array_equal(transported.state.positions, owner.state.positions)
        assert not np.array_equal(transported.state.gamma, owner.state.gamma)
        assert not np.array_equal(transported.state.sigma, owner.state.sigma)
        assert not np.array_equal(
            transported.state.live_boundary_nodes, owner.state.live_boundary_nodes
        )
        with pytest.raises(RuntimeError, match="stale|not live"):
            validate_release_row_common_transport(session)
        with pytest.raises(RuntimeError, match="stale|not live"):
            validate_release_row_transport_attestation(attestation)
        assert tracers.shape[0] == (
            session.frontier_node_offset
            + transported.state.live_boundary_nodes.shape[0]
        )
        assert session.live_particle_ids == tuple(
            owner.state.particle_ids[index]
            for indices in owner.state.live_boundary_indices_by_cell
            for index in indices
        )

        # A nonlinear same-stage transport curves at least one internal
        # material support away from the straight endpoint chord.  This is a
        # compatible state and must not be projected back to chord midpoints.
        straight_supports = []
        for cell, indices in enumerate(transported.state.live_boundary_indices_by_cell):
            start = transported.state.live_boundary_nodes[cell]
            end = transported.state.live_boundary_nodes[cell + 1]
            segment = (end - start) / len(indices)
            straight_supports.extend(
                start + (subdivision + 0.5) * segment
                for subdivision in range(len(indices))
            )
        live_flat = tuple(
            index
            for indices in transported.state.live_boundary_indices_by_cell
            for index in indices
        )
        assert not np.array_equal(
            transported.state.positions[np.asarray(live_flat, dtype=np.int64)],
            np.asarray(straight_supports),
        )

        upstream = transported.state.live_boundary_nodes
        downstream = np.array(upstream, copy=True)
        downstream += np.asarray((0.4, 0.0, 0.0))
        circulation = _row(8, release).circulation_m2_s
        candidate = make_release_row(
            upstream,
            downstream,
            circulation,
            release_index=release,
            source_time_s=ROOT_TIME + (release - 1) * transport_dt,
            sheet_id="straight-row",
        )
        deposited, kept, upstream_keys, edge_by_key = _single_row_snapshot(candidate)
        expected_gamma = np.array(transported.state.gamma, copy=True)
        clone_positions = []
        clone_gamma = []
        clone_sigma = []
        for cell, key in enumerate(upstream_keys):
            edge = edge_by_key[key]
            c_new = fsum(incidence.signed_circulation for incidence in edge.incidences)
            old_indices = transported.state.live_boundary_indices_by_cell[cell]
            c_old_cell = None
            for index in old_indices:
                record = transported.state.lineage[index]
                c_old = fsum(
                    incidence.signed_circulation
                    for incidence in record.birth_incidences + record.update_incidences
                )
                if c_old_cell is None:
                    c_old_cell = c_old
                assert c_old == c_old_cell and c_old != 0.0
                delta = c_new * (transported.state.gamma[index] / c_old)
                expected_gamma[index] += delta
                clone_positions.append(transported.state.positions[index])
                clone_gamma.append(delta)
                clone_sigma.append(transported.state.sigma[index])

        kept_array = np.asarray(kept, dtype=np.int64)
        expected_positions = np.vstack(
            (transported.state.positions, deposited.positions[kept_array])
        )
        expected_gamma = np.vstack((expected_gamma, deposited.gamma[kept_array]))
        expected_sigma = np.concatenate(
            (transported.state.sigma, deposited.sigma[kept_array])
        )
        explicit_positions = np.vstack(
            (
                transported.state.positions,
                np.asarray(clone_positions),
                deposited.positions[kept_array],
            )
        )
        explicit_gamma = np.vstack(
            (
                transported.state.gamma,
                np.asarray(clone_gamma),
                deposited.gamma[kept_array],
            )
        )
        explicit_sigma = np.concatenate(
            (
                transported.state.sigma,
                np.asarray(clone_sigma),
                deposited.sigma[kept_array],
            )
        )

        proposal = propose_release_row_update(
            transported, candidate, proposal_id=f"lsrk-material-{release}"
        )
        assert proposal.status == "compatible"
        committed = commit_release_row_update(transported, proposal)
        assert committed.committed
        owner = committed.owner
        np.testing.assert_array_equal(owner.state.positions, expected_positions)
        np.testing.assert_array_equal(owner.state.gamma, expected_gamma)
        np.testing.assert_array_equal(owner.state.sigma, expected_sigma)
        _assert_arrays_mechanics(
            owner, expected_positions, expected_gamma, expected_sigma
        )
        _assert_cloud_mechanics(
            owner, explicit_positions, explicit_gamma, explicit_sigma
        )
        assert owner.state.clone_count == 0
        assert owner.state.counter_particle_count == 0
        assert owner.state.fresh_upstream_particle_count == 0
    assert owner.epoch == 3 and len(owner.state.rows) == 3


@pytest.mark.parametrize("cell_count", (2, 8))
def test_three_release_global_row_matches_direct_full_graph_mechanics(
    cell_count: int,
) -> None:
    owner = _bootstrap(cell_count)
    _assert_mechanics(owner)
    previous_ids = owner.state.particle_ids
    for release in (2, 3):
        owner = _identity_transport(owner, source_step_index=release + 2)
        transported_positions = owner.state.positions.tobytes()
        transported_sigma = owner.state.sigma.tobytes()
        proposal = propose_release_row_update(
            owner, _row(cell_count, release), proposal_id=f"release-{release}"
        )
        assert proposal.status == "compatible"
        result = commit_release_row_update(owner, proposal)
        assert result.committed and result.event is not None
        owner = result.owner
        assert (
            owner.state.positions[: len(previous_ids)].tobytes()
            == transported_positions
        )
        assert owner.state.sigma[: len(previous_ids)].tobytes() == transported_sigma
        assert owner.state.particle_ids[: len(previous_ids)] == previous_ids
        assert owner.state.clone_count == 0
        assert owner.state.counter_particle_count == 0
        assert owner.state.fresh_upstream_particle_count == 0
        assert not any(
            "clone" in record.role
            or "counter" in record.role
            or "fresh_upstream" in record.role
            for record in owner.state.lineage
        )
        assert owner.events[-1].global_graph_build_count == 1
        _assert_mechanics(owner)
        previous_ids = owner.state.particle_ids


def test_shared_cell_edges_have_one_global_particle_basis_and_exact_incidence() -> None:
    owner = _bootstrap(8)
    records_by_basis: dict[tuple[object, int], RowParticleLineage] = {}
    shared_count = 0
    for record in owner.state.lineage:
        key = (record.source_edge, record.subdivision_index)
        assert key not in records_by_basis
        records_by_basis[key] = record
        assert all(type(item) is EdgeIncidence for item in record.birth_incidences)
        if len(record.birth_incidences) == 2:
            shared_count += 1
            assert {item.canonical_sign for item in record.birth_incidences} == {-1, 1}
    assert shared_count > 0


def test_exact_types_readonly_arrays_stable_ids_and_live_attestation() -> None:
    owner = _bootstrap(2)
    validated = validate_release_row_owner(owner)
    assert validated is owner
    for array in (
        owner.state.positions,
        owner.state.gamma,
        owner.state.sigma,
        owner.state.live_boundary_nodes,
    ):
        assert type(array) is np.ndarray
        assert array.dtype == np.dtype(np.float64)
        assert array.flags.c_contiguous and not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(RuntimeError, match="live issued"):
        validate_release_row_owner(replace(owner))
    assert all(type(record) is RowParticleLineage for record in owner.state.lineage)


def test_next_release_requires_transported_parent_and_remesh_is_zero_commit() -> None:
    owner = _bootstrap(2)
    proposal = propose_release_row_update(owner, _row(2, 2), proposal_id="too-early")
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch == "transport_required"
    result = commit_release_row_update(owner, proposal)
    assert not result.committed
    assert (
        result.owner is owner and result.state is owner.state and result.event is None
    )
    assert validate_current_release_row_owner(owner) is owner


def test_current_validator_rejects_stale_commit_after_transport_advance() -> None:
    first = _identity_transport(_bootstrap(2), source_step_index=4)
    proposal = propose_release_row_update(
        first, _row(2, 2), proposal_id="current-validator-release-2"
    )
    committed = commit_release_row_update(first, proposal)
    assert committed.committed
    pre_transport = committed.owner
    assert validate_current_release_row_owner(pre_transport) is pre_transport
    state = pre_transport.state
    end = state.rows[-1].source_time_s + DT
    _, attestation, digest = _prepare_transport_handoff(
        pre_transport,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        source_step_index=5,
    )
    current = advance_release_row_transport_parent(
        pre_transport,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=5,
        transport_end_time_s=end,
        transport_epoch=2,
    )
    # Historical audit remains possible, but live coupling must reject this
    # stale RowCommitResult owner after transport advances the generation.
    assert validate_release_row_owner(pre_transport) is pre_transport
    with pytest.raises(RuntimeError, match="stale"):
        validate_current_release_row_owner(pre_transport)
    assert validate_current_release_row_owner(current) is current
    assert current.state.phase == "post_transport"


def test_nonzero_transported_frontier_binds_next_row_and_stale_birth_nodes_stop() -> (
    None
):
    owner = _bootstrap(2)
    state = owner.state
    shift = np.asarray((0.07, -0.013, 0.021))
    positions = np.asarray(state.positions) + shift
    gamma = np.asarray(state.gamma) * 1.03
    sigma = np.asarray(state.sigma) * 1.01
    nodes = np.asarray(state.live_boundary_nodes) + shift
    end = state.rows[-1].source_time_s + DT
    _, attestation, digest = _prepare_transport_handoff(
        owner,
        positions,
        gamma,
        sigma,
        nodes,
        source_step_index=4,
    )
    advanced = advance_release_row_transport_parent(
        owner,
        state,
        positions,
        gamma,
        sigma,
        nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=4,
        transport_end_time_s=end,
        transport_epoch=1,
    )
    stale = propose_release_row_update(
        advanced, _row(2, 2), proposal_id="stale-birth-frontier"
    )
    assert stale.status == "remesh_required" and stale.first_mismatch == "support"
    downstream = np.array(nodes, copy=True)
    downstream[:, 0] += 0.4
    current = _row(2, 2).circulation_m2_s
    transported_row = make_release_row(
        nodes,
        downstream,
        current,
        release_index=2,
        source_time_s=ROOT_TIME + DT,
        sheet_id="straight-row",
    )
    proposal = propose_release_row_update(
        advanced, transported_row, proposal_id="transported-frontier"
    )
    assert proposal.status == "compatible"
    result = commit_release_row_update(advanced, proposal)
    assert result.committed
    prefix = state.positions.shape[0]
    np.testing.assert_array_equal(result.state.positions[:prefix], positions)
    np.testing.assert_array_equal(result.state.sigma[:prefix], sigma)
    assert result.state.row_parent_transport_digests == (None, digest)
    assert result.event is not None
    assert result.event.parent_transport_digest == digest
    assert validate_release_row_owner(result.owner) is result.owner


@pytest.mark.parametrize("mismatch", ("x-0.47", "one-ulp"))
def test_raw_or_inconsistent_material_transport_cannot_attest_and_retries_cleanly(
    mismatch: str,
) -> None:
    owner = _bootstrap(2)
    state = owner.state
    session = begin_release_row_common_transport(owner, state)
    end_time = state.rows[-1].source_time_s + DT
    with pytest.raises(TypeError, match="common_transport_attestation"):
        release_row_transport_digest(
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )

    inconsistent = np.array(session.material_tracer_positions, copy=True)
    if mismatch == "x-0.47":
        inconsistent[0, 0] = 0.47
    else:
        inconsistent[0, 0] = np.nextafter(inconsistent[0, 0], np.inf)
    flat = tuple(
        index for indices in session.live_particle_indices_by_cell for index in indices
    )
    flat_array = np.asarray(flat, dtype=np.int64)
    with pytest.raises(ValueError, match="live supports.*common-transport"):
        attest_release_row_common_transport(
            session,
            state.positions,
            state.gamma,
            state.sigma,
            inconsistent,
            state.sigma[flat_array],
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )
    assert validate_release_row_common_transport(session) is session

    attestation = attest_release_row_common_transport(
        session,
        state.positions,
        state.gamma,
        state.sigma,
        session.material_tracer_positions,
        state.sigma[flat_array],
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    with pytest.raises(RuntimeError, match="already attested"):
        attest_release_row_common_transport(
            session,
            state.positions,
            state.gamma,
            state.sigma,
            session.material_tracer_positions,
            state.sigma[flat_array],
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )
    digest = release_row_transport_digest(
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    advanced = advance_release_row_transport_parent(
        owner,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    assert validate_current_release_row_owner(advanced) is advanced


def test_transport_digest_rejects_one_ulp_time_epoch_and_replay() -> None:
    owner = _bootstrap(2)
    state = owner.state
    end_time = state.rows[-1].source_time_s + DT
    _, attestation, digest = _prepare_transport_handoff(
        owner,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        source_step_index=4,
    )
    changed = np.array(state.positions, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    with pytest.raises(ValueError, match="attestation|digest"):
        advance_release_row_transport_parent(
            owner,
            state,
            changed,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            parent_transport_digest=digest,
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )
    with pytest.raises(ValueError, match="end time"):
        advance_release_row_transport_parent(
            owner,
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            parent_transport_digest=digest,
            source_step_index=4,
            transport_end_time_s=np.nextafter(end_time, np.inf),
            transport_epoch=1,
        )
    with pytest.raises(ValueError, match="epoch"):
        advance_release_row_transport_parent(
            owner,
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            parent_transport_digest=digest,
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=2,
        )
    advanced = advance_release_row_transport_parent(
        owner,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    assert advanced.state.phase == "post_transport"
    with pytest.raises(RuntimeError, match="stale"):
        validate_current_release_row_owner(owner)
    assert validate_current_release_row_owner(advanced) is advanced
    assert advanced.state.particle_ids == state.particle_ids
    assert advanced.state.lineage == state.lineage
    with pytest.raises(RuntimeError, match="stale"):
        advance_release_row_transport_parent(
            owner,
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            parent_transport_digest=digest,
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )


@pytest.mark.parametrize(
    ("mutator", "reason"),
    (
        (
            lambda row: make_release_row(
                np.asarray(row.upstream_nodes),
                row.downstream_nodes,
                -row.circulation_m2_s,
                release_index=row.release_index,
                source_time_s=row.source_time_s,
                sheet_id=row.sheet_id,
            ),
            "circulation_sign",
        ),
        (
            lambda row: make_release_row(
                row.upstream_nodes,
                row.downstream_nodes,
                row.circulation_m2_s,
                release_index=row.release_index,
                source_time_s=np.nextafter(row.source_time_s, np.inf),
                sheet_id=row.sheet_id,
            ),
            "source_time_s",
        ),
        (
            lambda row: make_release_row(
                row.upstream_nodes,
                row.downstream_nodes,
                row.circulation_m2_s,
                release_index=row.release_index,
                source_time_s=row.source_time_s,
                sheet_id="other-sheet",
            ),
            "sheet_id",
        ),
    ),
)
def test_sign_time_and_sheet_mismatch_fail_closed(mutator, reason: str) -> None:
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    proposal = propose_release_row_update(
        owner, mutator(_row(2, 2)), proposal_id=f"bad-{reason}"
    )
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch == reason
    assert commit_release_row_update(owner, proposal).owner is owner


def test_one_ulp_support_and_per_cell_alias_fail_closed() -> None:
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    clean = _row(2, 2)
    support = np.array(clean.upstream_nodes, copy=True)
    support[1, 1] = np.nextafter(support[1, 1], np.inf)
    one_ulp = make_release_row(
        support,
        clean.downstream_nodes,
        clean.circulation_m2_s,
        release_index=2,
        source_time_s=clean.source_time_s,
        sheet_id=clean.sheet_id,
    )
    proposal = propose_release_row_update(owner, one_ulp, proposal_id="one-ulp")
    assert proposal.status == "remesh_required" and proposal.first_mismatch == "support"
    aliased = np.array(clean.upstream_nodes, copy=True)
    aliased[1] = aliased[0]
    with pytest.raises(ValueError):
        make_release_row(
            aliased,
            clean.downstream_nodes,
            clean.circulation_m2_s,
            release_index=2,
            source_time_s=clean.source_time_s,
            sheet_id=clean.sheet_id,
        )


def test_sibling_fork_proposal_copy_and_exact_once_are_rejected() -> None:
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    row = _row(2, 2)
    first = propose_release_row_update(owner, row, proposal_id="first")
    sibling = propose_release_row_update(owner, row, proposal_id="sibling")
    with pytest.raises(RuntimeError, match="forged|not live"):
        commit_release_row_update(owner, replace(first))
    result = commit_release_row_update(owner, first)
    assert result.committed
    with pytest.raises(RuntimeError, match="stale"):
        commit_release_row_update(owner, sibling)
    with pytest.raises(RuntimeError, match="stale"):
        commit_release_row_update(owner, first)


def test_private_additive_plan_drift_rejects_before_commit_and_retries_cleanly() -> (
    None
):
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    proposal = propose_release_row_update(
        owner, _row(2, 2), proposal_id="additive-plan-drift"
    )
    plan = row_owner._LIVE_PROPOSALS[proposal]
    first = plan.upstream_additions[0]
    changed = np.array(first[1], copy=True)
    changed[0] = np.nextafter(changed[0], np.inf)
    forged_first = (first[0], changed, *first[2:])
    row_owner._LIVE_PROPOSALS[proposal] = replace(
        plan,
        upstream_additions=(forged_first,) + plan.upstream_additions[1:],
    )
    with pytest.raises(RuntimeError, match="additive incidence plan changed"):
        commit_release_row_update(owner, proposal)
    assert validate_current_release_row_owner(owner) is owner
    row_owner._LIVE_PROPOSALS[proposal] = plan
    committed = commit_release_row_update(owner, proposal)
    assert committed.committed
    assert validate_current_release_row_owner(committed.owner) is committed.owner


class _FailOnceWeakRegistry(WeakKeyDictionary):
    def __init__(self, original: WeakKeyDictionary):
        super().__init__()
        for key, value in original.items():
            super().__setitem__(key, value)
        self.failed = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected publication failure")


def test_common_transport_session_and_attestation_exact_once_with_clean_retry(
    monkeypatch,
) -> None:
    owner = _bootstrap(2)
    state = owner.state
    original_sessions = row_owner._LIVE_COMMON_TRANSPORTS
    monkeypatch.setattr(
        row_owner,
        "_LIVE_COMMON_TRANSPORTS",
        _FailOnceWeakRegistry(original_sessions),
    )
    with pytest.raises(RuntimeError, match="injected"):
        begin_release_row_common_transport(owner, state)
    assert validate_current_release_row_owner(owner) is owner
    monkeypatch.setattr(row_owner, "_LIVE_COMMON_TRANSPORTS", original_sessions)
    session = begin_release_row_common_transport(owner, state)
    with pytest.raises(RuntimeError, match="already issued"):
        begin_release_row_common_transport(owner, state)

    flat = tuple(
        index for indices in session.live_particle_indices_by_cell for index in indices
    )
    flat_array = np.asarray(flat, dtype=np.int64)
    end_time = state.rows[-1].source_time_s + DT
    original_attestations = row_owner._LIVE_TRANSPORT_ATTESTATIONS
    monkeypatch.setattr(
        row_owner,
        "_LIVE_TRANSPORT_ATTESTATIONS",
        _FailOnceWeakRegistry(original_attestations),
    )
    with pytest.raises(RuntimeError, match="injected"):
        attest_release_row_common_transport(
            session,
            state.positions,
            state.gamma,
            state.sigma,
            session.material_tracer_positions,
            state.sigma[flat_array],
            source_step_index=4,
            transport_end_time_s=end_time,
            transport_epoch=1,
        )
    assert validate_release_row_common_transport(session) is session
    monkeypatch.setattr(
        row_owner, "_LIVE_TRANSPORT_ATTESTATIONS", original_attestations
    )
    attestation = attest_release_row_common_transport(
        session,
        state.positions,
        state.gamma,
        state.sigma,
        session.material_tracer_positions,
        state.sigma[flat_array],
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    assert validate_release_row_transport_attestation(attestation) is attestation
    digest = release_row_transport_digest(
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    advanced = advance_release_row_transport_parent(
        owner,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=4,
        transport_end_time_s=end_time,
        transport_epoch=1,
    )
    assert validate_current_release_row_owner(advanced) is advanced
    with pytest.raises(RuntimeError, match="stale|not live"):
        validate_release_row_common_transport(session)
    with pytest.raises(RuntimeError, match="stale|not live"):
        validate_release_row_transport_attestation(attestation)


def test_transport_publication_rollback_and_clean_retry(monkeypatch) -> None:
    owner = _bootstrap(2)
    state = owner.state
    end = state.rows[-1].source_time_s + DT
    session, attestation, digest = _prepare_transport_handoff(
        owner,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        source_step_index=4,
    )
    original = row_owner._LIVE_TRANSPORT_EVENTS
    failing = _FailOnceWeakRegistry(original)
    monkeypatch.setattr(row_owner, "_LIVE_TRANSPORT_EVENTS", failing)
    with pytest.raises(RuntimeError, match="injected"):
        advance_release_row_transport_parent(
            owner,
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            parent_transport_digest=digest,
            source_step_index=4,
            transport_end_time_s=end,
            transport_epoch=1,
        )
    assert validate_current_release_row_owner(owner) is owner
    assert validate_release_row_common_transport(session) is session
    assert validate_release_row_transport_attestation(attestation) is attestation
    monkeypatch.setattr(row_owner, "_LIVE_TRANSPORT_EVENTS", original)
    advanced = advance_release_row_transport_parent(
        owner,
        state,
        state.positions,
        state.gamma,
        state.sigma,
        state.live_boundary_nodes,
        common_transport_attestation=attestation,
        parent_transport_digest=digest,
        source_step_index=4,
        transport_end_time_s=end,
        transport_epoch=1,
    )
    assert advanced.state.phase == "post_transport"
    with pytest.raises(RuntimeError, match="stale"):
        validate_current_release_row_owner(owner)
    assert validate_current_release_row_owner(advanced) is advanced


def test_commit_publication_rollback_and_clean_retry(monkeypatch) -> None:
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    proposal = propose_release_row_update(owner, _row(2, 2), proposal_id="rollback")
    original = row_owner._LIVE_EVENTS
    failing = _FailOnceWeakRegistry(original)
    monkeypatch.setattr(row_owner, "_LIVE_EVENTS", failing)
    with pytest.raises(RuntimeError, match="injected"):
        commit_release_row_update(owner, proposal)
    assert validate_current_release_row_owner(owner) is owner
    monkeypatch.setattr(row_owner, "_LIVE_EVENTS", original)
    committed = commit_release_row_update(owner, proposal)
    assert committed.committed
    with pytest.raises(RuntimeError, match="stale"):
        validate_current_release_row_owner(owner)
    assert validate_current_release_row_owner(committed.owner) is committed.owner


def test_particle_cap_precedes_particle_materialization(monkeypatch) -> None:
    sizing_owner = _bootstrap(2)
    exact_cap = sizing_owner.state.positions.shape[0]
    capped = _bootstrap(2, cap=exact_cap)
    capped = _identity_transport(capped, source_step_index=4)
    calls = 0

    def bomb(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("particle materialization ran before cap gate")

    monkeypatch.setattr(row_owner, "_deposit_plans", bomb)
    with pytest.raises(RuntimeError, match="before materialization"):
        bootstrap_release_row_owner(
            _row(8, 1),
            smoothing_radius_m=SIGMA,
            target_spacing_m=SPACING,
            release_dt_s=DT,
            particle_cap=1,
            owner_id=_owner_id("tiny-cap"),
        )
    assert calls == 0

    proposal = propose_release_row_update(capped, _row(2, 2), proposal_id="propose-cap")
    assert proposal.status == "remesh_required"
    assert proposal.first_mismatch == "particle_cap"
    assert calls == 0


def test_runtime_edge_helper_drift_rejected_before_commit_and_clean_retry(
    monkeypatch,
) -> None:
    owner = _bootstrap(2)
    original = edge_bridge._stable_id
    monkeypatch.setattr(edge_bridge, "_stable_id", lambda name, value: value)
    with pytest.raises(RuntimeError, match="runtime global"):
        validate_release_row_owner(owner)
    monkeypatch.setattr(edge_bridge, "_stable_id", original)
    assert validate_release_row_owner(owner) is owner


def test_runtime_edge_module_leaf_drift_runs_zero_leaf_calls_and_clean_retry(
    monkeypatch,
) -> None:
    owner = _identity_transport(_bootstrap(2), source_step_index=4)
    candidate = _row(2, 2)
    original = edge_bridge.np.subtract
    calls = 0

    def forged_subtract(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(edge_bridge.np, "subtract", forged_subtract)
    with pytest.raises(RuntimeError, match="module attribute"):
        propose_release_row_update(owner, candidate, proposal_id="drifted-leaf")
    assert calls == 0
    assert validate_release_row_owner.__module__ == row_owner.__name__
    monkeypatch.setattr(edge_bridge.np, "subtract", original)
    proposal = propose_release_row_update(
        owner, candidate, proposal_id="clean-leaf-retry"
    )
    assert proposal.status == "compatible"
    assert commit_release_row_update(owner, proposal).committed


def test_source_has_no_ptera_target_load_feedback_or_private_v5h9_oracle() -> None:
    source = row_owner.Path(row_owner.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "import pterasoftware",
        "ground_truth",
        "observation_csv",
        "load_adapter",
        "fluxv_v5h9_live_boundary_owner",
        "collapse_live_basis_pairs",
    ):
        assert forbidden not in source
    assert row_owner.DEPENDENCY_SHA256 == (
        ("fluxvortex.rvpm_edge_bridge", row_owner._EDGE_SOURCE_SHA256),
    )
