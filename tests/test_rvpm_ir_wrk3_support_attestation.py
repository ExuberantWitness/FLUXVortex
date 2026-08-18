from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from fluxvortex.rvpm_ir_wrk3 import (
    IRWRK3Field,
    IRWRK3StepResult,
    IRWRK3TracerField,
    ir_wrk3_step_with_external_field,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
    validate_ir_wrk3_result,
)
from fluxvortex.rvpm_transport import ParticleState, make_particle_state


CELL_COUNT = 8
SUPPORTS_PER_CELL = 2
MATERIAL_SUPPORT_COUNT = CELL_COUNT * SUPPORTS_PER_CELL
FRONTIER_NODE_COUNT = 9
FRONTIER_NODE_OFFSET = MATERIAL_SUPPORT_COUNT
TOTAL_TRACER_COUNT = MATERIAL_SUPPORT_COUNT + FRONTIER_NODE_COUNT
PARENT_TOKEN = "v5h11-b2-synthetic-support-parent-v1"
DELTA_TIME = 0.03125
TRANSPORT_SUBSTEPS = 3

AFFINE_JACOBIAN = np.asarray(
    ((0.17, -0.08, 0.03), (0.06, -0.12, 0.04), (-0.02, 0.05, -0.05)),
    dtype=np.float64,
)
AFFINE_TRANSLATION = np.asarray((0.07, -0.025, 0.04), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class B2SupportLayout:
    cell_count: int
    supports_per_cell: int
    cell_ids: tuple[str, ...]
    physical_particle_ids: tuple[str, ...]
    material_tracer_ids: tuple[str, ...]
    material_to_physical_indices: tuple[int, ...]
    material_tracer_indices: tuple[int, ...]
    cell_support_offsets: tuple[int, ...]
    material_support_count: int
    frontier_node_offset: int
    frontier_node_count: int
    frontier_node_ids: tuple[str, ...]
    total_tracer_count: int
    layout_sha256: str


@dataclass(frozen=True, slots=True)
class B2SupportAttestation:
    layout_sha256: str
    parent_token: str
    delta_time_hex: str
    transport_substep_count: int
    transport_stage_count: int
    initial_source_state_sha256: str
    initial_material_positions_sha256: str
    initial_frontier_positions_sha256: str
    stage_schedule: tuple[tuple[int, int, str, str], ...]
    stage_source_state_sha256s: tuple[str, ...]
    stage_post_state_sha256s: tuple[str, ...]
    stage_trace_sha256s: tuple[str, ...]
    stage_material_pre_sha256s: tuple[str, ...]
    stage_material_post_sha256s: tuple[str, ...]
    stage_frontier_pre_sha256s: tuple[str, ...]
    stage_frontier_post_sha256s: tuple[str, ...]
    stage_chain_sha256: str
    final_state_sha256: str
    result_sha256: str
    final_material_positions: np.ndarray
    final_material_sigma: np.ndarray
    final_frontier_positions: np.ndarray
    attestation_sha256: str


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frozen_float64(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("support array has an invalid shape or non-finite value")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _validate_frozen_float64(
    name: str, value: object, shape: tuple[int, ...]
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or value.flags.writeable
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name} is not an exact frozen float64 array")
    ancestor: object = value
    seen: set[int] = set()
    while type(ancestor) is np.ndarray:
        if id(ancestor) in seen:
            raise ValueError(f"{name} has a cyclic ownership tree")
        seen.add(id(ancestor))
        if ancestor.flags.writeable or not ancestor.flags.c_contiguous:
            raise ValueError(f"{name} has a mutable array ownership tree")
        ancestor = ancestor.base
    if type(ancestor) is not bytes:
        raise ValueError(f"{name} is not backed by exact immutable bytes")
    return value


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(b"v5h11-b2-support-array-v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _same_array_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _layout_digest(layout: B2SupportLayout) -> str:
    payload = {
        "domain": "v5h11-b2-support-layout-v1",
        "cell_count": layout.cell_count,
        "supports_per_cell": layout.supports_per_cell,
        "cell_ids": layout.cell_ids,
        "physical_particle_ids": layout.physical_particle_ids,
        "material_tracer_ids": layout.material_tracer_ids,
        "material_to_physical_indices": layout.material_to_physical_indices,
        "material_tracer_indices": layout.material_tracer_indices,
        "cell_support_offsets": layout.cell_support_offsets,
        "material_support_count": layout.material_support_count,
        "frontier_node_offset": layout.frontier_node_offset,
        "frontier_node_count": layout.frontier_node_count,
        "frontier_node_ids": layout.frontier_node_ids,
        "total_tracer_count": layout.total_tracer_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()


def _reseal_layout(layout: B2SupportLayout) -> B2SupportLayout:
    return replace(layout, layout_sha256=_layout_digest(layout))


def _make_layout() -> B2SupportLayout:
    cell_ids = tuple(f"cell:{cell}" for cell in range(CELL_COUNT))
    physical_ids = tuple(
        f"physical:{cell}:{support}"
        for cell in range(CELL_COUNT)
        for support in range(SUPPORTS_PER_CELL)
    )
    material_ids = tuple(
        f"material:{cell}:{support}"
        for cell in range(CELL_COUNT)
        for support in range(SUPPORTS_PER_CELL)
    )
    layout = B2SupportLayout(
        cell_count=CELL_COUNT,
        supports_per_cell=SUPPORTS_PER_CELL,
        cell_ids=cell_ids,
        physical_particle_ids=physical_ids,
        material_tracer_ids=material_ids,
        material_to_physical_indices=tuple(range(MATERIAL_SUPPORT_COUNT)),
        material_tracer_indices=tuple(range(MATERIAL_SUPPORT_COUNT)),
        cell_support_offsets=tuple(
            SUPPORTS_PER_CELL * cell for cell in range(CELL_COUNT + 1)
        ),
        material_support_count=MATERIAL_SUPPORT_COUNT,
        frontier_node_offset=FRONTIER_NODE_OFFSET,
        frontier_node_count=FRONTIER_NODE_COUNT,
        frontier_node_ids=tuple(
            f"frontier-node:{node}" for node in range(FRONTIER_NODE_COUNT)
        ),
        total_tracer_count=TOTAL_TRACER_COUNT,
        layout_sha256="",
    )
    return _reseal_layout(layout)


def validate_b2_support_layout(layout: B2SupportLayout) -> B2SupportLayout:
    if type(layout) is not B2SupportLayout:
        raise ValueError("layout must be an exact B2SupportLayout")
    scalar_fields = (
        layout.cell_count,
        layout.supports_per_cell,
        layout.material_support_count,
        layout.frontier_node_offset,
        layout.frontier_node_count,
        layout.total_tracer_count,
    )
    if any(type(value) is not int for value in scalar_fields):
        raise ValueError("layout counts and offsets must be exact integers")
    tuple_fields = (
        layout.cell_ids,
        layout.physical_particle_ids,
        layout.material_tracer_ids,
        layout.material_to_physical_indices,
        layout.material_tracer_indices,
        layout.cell_support_offsets,
        layout.frontier_node_ids,
    )
    if any(type(value) is not tuple for value in tuple_fields):
        raise ValueError("layout collections must be exact tuples")

    expected = _make_layout()
    if layout.cell_count != CELL_COUNT or layout.supports_per_cell != SUPPORTS_PER_CELL:
        raise ValueError("layout cell/support count drift")
    if layout.cell_ids != expected.cell_ids:
        raise ValueError("layout cell ID drift")
    if layout.physical_particle_ids != expected.physical_particle_ids:
        raise ValueError("layout physical ID drift")
    if layout.material_tracer_ids != expected.material_tracer_ids:
        raise ValueError("layout material ID drift")
    if layout.material_to_physical_indices != tuple(range(MATERIAL_SUPPORT_COUNT)):
        raise ValueError("layout material-to-physical index drift")
    if layout.material_tracer_indices != tuple(range(MATERIAL_SUPPORT_COUNT)):
        raise ValueError("layout material tracer index drift")
    if layout.cell_support_offsets != tuple(range(0, MATERIAL_SUPPORT_COUNT + 1, 2)):
        raise ValueError("layout support offset drift")
    if layout.material_support_count != MATERIAL_SUPPORT_COUNT:
        raise ValueError("layout material support count drift")
    if layout.frontier_node_offset != FRONTIER_NODE_OFFSET:
        raise ValueError("layout frontier offset drift")
    if (
        layout.frontier_node_count != FRONTIER_NODE_COUNT
        or layout.frontier_node_ids != expected.frontier_node_ids
    ):
        raise ValueError("layout must contain the ordered 9-node frontier")
    if layout.total_tracer_count != TOTAL_TRACER_COUNT:
        raise ValueError("layout total tracer count drift")
    if not _is_sha256(layout.layout_sha256):
        raise ValueError("layout digest is malformed")
    if layout.layout_sha256 != _layout_digest(layout):
        raise ValueError("layout digest mismatch")
    return layout


def _attestation_digest(attestation: B2SupportAttestation) -> str:
    payload = {
        "domain": "v5h11-b2-support-attestation-v1",
        "layout_sha256": attestation.layout_sha256,
        "parent_token": attestation.parent_token,
        "delta_time_hex": attestation.delta_time_hex,
        "transport_substep_count": attestation.transport_substep_count,
        "transport_stage_count": attestation.transport_stage_count,
        "initial_source_state_sha256": attestation.initial_source_state_sha256,
        "initial_material_positions_sha256": (
            attestation.initial_material_positions_sha256
        ),
        "initial_frontier_positions_sha256": (
            attestation.initial_frontier_positions_sha256
        ),
        "stage_schedule": attestation.stage_schedule,
        "stage_source_state_sha256s": attestation.stage_source_state_sha256s,
        "stage_post_state_sha256s": attestation.stage_post_state_sha256s,
        "stage_trace_sha256s": attestation.stage_trace_sha256s,
        "stage_material_pre_sha256s": attestation.stage_material_pre_sha256s,
        "stage_material_post_sha256s": attestation.stage_material_post_sha256s,
        "stage_frontier_pre_sha256s": attestation.stage_frontier_pre_sha256s,
        "stage_frontier_post_sha256s": attestation.stage_frontier_post_sha256s,
        "stage_chain_sha256": attestation.stage_chain_sha256,
        "final_state_sha256": attestation.final_state_sha256,
        "result_sha256": attestation.result_sha256,
        "final_material_positions_sha256": _array_sha256(
            attestation.final_material_positions
        ),
        "final_material_sigma_sha256": _array_sha256(attestation.final_material_sigma),
        "final_frontier_positions_sha256": _array_sha256(
            attestation.final_frontier_positions
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()


def _reseal_attestation(
    attestation: B2SupportAttestation,
) -> B2SupportAttestation:
    return replace(attestation, attestation_sha256=_attestation_digest(attestation))


def _support_slices(
    layout: B2SupportLayout, tracer_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    material = tracer_positions[np.asarray(layout.material_tracer_indices)]
    frontier = tracer_positions[layout.frontier_node_offset :]
    return material, frontier


def _make_attestation(
    layout: B2SupportLayout, result: IRWRK3StepResult
) -> B2SupportAttestation:
    validate_b2_support_layout(layout)
    validate_ir_wrk3_result(result)
    stages = result.stages
    first_material, first_frontier = _support_slices(layout, stages[0].tracer_pre)
    final_material, final_frontier = _support_slices(
        layout, result.final_tracer_positions
    )
    attestation = B2SupportAttestation(
        layout_sha256=layout.layout_sha256,
        parent_token=result.parent_token,
        delta_time_hex=result.delta_time.hex(),
        transport_substep_count=result.counters.substep_count,
        transport_stage_count=len(stages),
        initial_source_state_sha256=stages[0].source_state_sha256,
        initial_material_positions_sha256=_array_sha256(first_material),
        initial_frontier_positions_sha256=_array_sha256(first_frontier),
        stage_schedule=tuple(
            (stage.substep, stage.stage, stage.a.hex(), stage.b.hex())
            for stage in stages
        ),
        stage_source_state_sha256s=tuple(stage.source_state_sha256 for stage in stages),
        stage_post_state_sha256s=tuple(stage.post_state_sha256 for stage in stages),
        stage_trace_sha256s=tuple(stage.trace_sha256 for stage in stages),
        stage_material_pre_sha256s=tuple(
            _array_sha256(_support_slices(layout, stage.tracer_pre)[0])
            for stage in stages
        ),
        stage_material_post_sha256s=tuple(
            _array_sha256(_support_slices(layout, stage.tracer_post)[0])
            for stage in stages
        ),
        stage_frontier_pre_sha256s=tuple(
            _array_sha256(_support_slices(layout, stage.tracer_pre)[1])
            for stage in stages
        ),
        stage_frontier_post_sha256s=tuple(
            _array_sha256(_support_slices(layout, stage.tracer_post)[1])
            for stage in stages
        ),
        stage_chain_sha256=result.stage_chain_sha256,
        final_state_sha256=stages[-1].post_state_sha256,
        result_sha256=result.result_sha256,
        final_material_positions=_frozen_float64(
            final_material, (MATERIAL_SUPPORT_COUNT, 3)
        ),
        final_material_sigma=_frozen_float64(
            result.final_state.sigma, (MATERIAL_SUPPORT_COUNT,)
        ),
        final_frontier_positions=_frozen_float64(
            final_frontier, (FRONTIER_NODE_COUNT, 3)
        ),
        attestation_sha256="",
    )
    return _reseal_attestation(attestation)


def _exact_zero_storage(value: np.ndarray) -> bool:
    return value.tobytes(order="C") == np.zeros(value.shape, dtype=np.float64).tobytes(
        order="C"
    )


def validate_b2_support_attestation(
    layout: B2SupportLayout,
    result: IRWRK3StepResult,
    attestation: B2SupportAttestation,
) -> B2SupportAttestation:
    validate_b2_support_layout(layout)
    validate_ir_wrk3_result(result)
    if type(attestation) is not B2SupportAttestation:
        raise ValueError("attestation must be an exact B2SupportAttestation")

    exact_ints = (
        attestation.transport_substep_count,
        attestation.transport_stage_count,
    )
    if any(type(value) is not int for value in exact_ints):
        raise ValueError("attestation counts must be exact integers")
    exact_strings = (
        attestation.layout_sha256,
        attestation.parent_token,
        attestation.delta_time_hex,
        attestation.initial_source_state_sha256,
        attestation.initial_material_positions_sha256,
        attestation.initial_frontier_positions_sha256,
        attestation.stage_chain_sha256,
        attestation.final_state_sha256,
        attestation.result_sha256,
        attestation.attestation_sha256,
    )
    if any(type(value) is not str for value in exact_strings):
        raise ValueError("attestation scalar tree is not exact")
    tuple_fields = (
        attestation.stage_schedule,
        attestation.stage_source_state_sha256s,
        attestation.stage_post_state_sha256s,
        attestation.stage_trace_sha256s,
        attestation.stage_material_pre_sha256s,
        attestation.stage_material_post_sha256s,
        attestation.stage_frontier_pre_sha256s,
        attestation.stage_frontier_post_sha256s,
    )
    if any(type(value) is not tuple for value in tuple_fields):
        raise ValueError("attestation stage collections must be exact tuples")

    stage_count = 3 * result.counters.substep_count
    if attestation.transport_substep_count != result.counters.substep_count:
        raise ValueError("attestation substep count mismatch")
    if (
        attestation.transport_stage_count != stage_count
        or len(result.stages) != stage_count
    ):
        raise ValueError("attestation stage count mismatch")
    if any(len(value) != stage_count for value in tuple_fields):
        raise ValueError("attestation stage tuple count mismatch")
    for coordinate in attestation.stage_schedule:
        if (
            type(coordinate) is not tuple
            or len(coordinate) != 4
            or type(coordinate[0]) is not int
            or type(coordinate[1]) is not int
            or type(coordinate[2]) is not str
            or type(coordinate[3]) is not str
        ):
            raise ValueError("attestation stage schedule tree is not exact")
    for hashes in tuple_fields[1:]:
        if any(not _is_sha256(value) for value in hashes):
            raise ValueError("attestation stage digest tree is malformed")

    if result.final_state.positions.shape != (MATERIAL_SUPPORT_COUNT, 3):
        raise ValueError("physical material support count mismatch")
    if result.final_tracer_positions.shape != (TOTAL_TRACER_COUNT, 3):
        raise ValueError("material/frontier tracer count mismatch")
    if attestation.layout_sha256 != layout.layout_sha256:
        raise ValueError("attestation layout binding mismatch")
    if (
        attestation.parent_token != result.parent_token
        or result.parent_token != PARENT_TOKEN
    ):
        raise ValueError("attestation parent binding mismatch")
    if attestation.delta_time_hex != result.delta_time.hex():
        raise ValueError("attestation delta-time binding mismatch")

    expected_schedule: list[tuple[int, int, str, str]] = []
    expected_source_hashes: list[str] = []
    expected_post_hashes: list[str] = []
    expected_trace_hashes: list[str] = []
    expected_material_pre_hashes: list[str] = []
    expected_material_post_hashes: list[str] = []
    expected_frontier_pre_hashes: list[str] = []
    expected_frontier_post_hashes: list[str] = []
    expected_a = (0.0, -5.0 / 9.0, -153.0 / 128.0)
    expected_b = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
    physical_indices = np.asarray(layout.material_to_physical_indices)

    for index, stage in enumerate(result.stages):
        substep = index // 3 + 1
        stage_number = index % 3 + 1
        if stage.substep != substep or stage.stage != stage_number:
            raise ValueError("result stage coordinate drift")
        if (
            stage.a != expected_a[stage_number - 1]
            or stage.b != expected_b[stage_number - 1]
        ):
            raise ValueError("result RK coefficient drift")
        if not (
            stage.source_state_sha256
            == stage.field.source_state_sha256
            == stage.tracer_field.source_state_sha256
        ):
            raise ValueError("physical/tracer source binding mismatch")
        if not (
            stage.field.parent_token
            == stage.tracer_field.parent_token
            == result.parent_token
        ):
            raise ValueError("physical/tracer parent binding mismatch")

        material_pre, frontier_pre = _support_slices(layout, stage.tracer_pre)
        material_post, frontier_post = _support_slices(layout, stage.tracer_post)
        if not _same_array_bytes(material_pre, stage.pre.positions[physical_indices]):
            raise ValueError("stage-pre material support mapping mismatch")
        if not _same_array_bytes(material_post, stage.post.positions[physical_indices]):
            raise ValueError("stage-post material support mapping mismatch")
        if stage_number == 1 and not (
            _exact_zero_storage(stage.position_storage_pre)
            and _exact_zero_storage(stage.gamma_storage_pre)
            and _exact_zero_storage(stage.tracer_storage_pre)
        ):
            raise ValueError("stage-1 physical/tracer storage did not reset exactly")

        expected_schedule.append((substep, stage_number, stage.a.hex(), stage.b.hex()))
        expected_source_hashes.append(stage.source_state_sha256)
        expected_post_hashes.append(stage.post_state_sha256)
        expected_trace_hashes.append(stage.trace_sha256)
        expected_material_pre_hashes.append(_array_sha256(material_pre))
        expected_material_post_hashes.append(_array_sha256(material_post))
        expected_frontier_pre_hashes.append(_array_sha256(frontier_pre))
        expected_frontier_post_hashes.append(_array_sha256(frontier_post))

    expected_stage_tuples = (
        tuple(expected_schedule),
        tuple(expected_source_hashes),
        tuple(expected_post_hashes),
        tuple(expected_trace_hashes),
        tuple(expected_material_pre_hashes),
        tuple(expected_material_post_hashes),
        tuple(expected_frontier_pre_hashes),
        tuple(expected_frontier_post_hashes),
    )
    if tuple_fields != expected_stage_tuples:
        raise ValueError("attestation stage source/RK/support binding mismatch")
    first_material, first_frontier = _support_slices(
        layout, result.stages[0].tracer_pre
    )
    if attestation.initial_source_state_sha256 != result.stages[0].source_state_sha256:
        raise ValueError("attestation initial source binding mismatch")
    if attestation.initial_material_positions_sha256 != _array_sha256(first_material):
        raise ValueError("attestation initial material binding mismatch")
    if attestation.initial_frontier_positions_sha256 != _array_sha256(first_frontier):
        raise ValueError("attestation initial frontier binding mismatch")
    if attestation.stage_chain_sha256 != result.stage_chain_sha256:
        raise ValueError("attestation stage-chain binding mismatch")
    if attestation.final_state_sha256 != result.stages[-1].post_state_sha256:
        raise ValueError("attestation final-state binding mismatch")
    if attestation.result_sha256 != result.result_sha256:
        raise ValueError("attestation result binding mismatch")

    final_material = _validate_frozen_float64(
        "final_material_positions",
        attestation.final_material_positions,
        (MATERIAL_SUPPORT_COUNT, 3),
    )
    final_sigma = _validate_frozen_float64(
        "final_material_sigma",
        attestation.final_material_sigma,
        (MATERIAL_SUPPORT_COUNT,),
    )
    final_frontier = _validate_frozen_float64(
        "final_frontier_positions",
        attestation.final_frontier_positions,
        (FRONTIER_NODE_COUNT, 3),
    )
    result_material, result_frontier = _support_slices(
        layout, result.final_tracer_positions
    )
    if not (
        _same_array_bytes(final_material, result_material)
        and _same_array_bytes(final_material, result.final_state.positions)
    ):
        raise ValueError("final material support positions are not exact")
    if not _same_array_bytes(final_sigma, result.final_state.sigma):
        raise ValueError("final material support sigma is not exact")
    if not _same_array_bytes(final_frontier, result_frontier):
        raise ValueError("final ordered 9-node frontier is not exact")

    if not _is_sha256(attestation.attestation_sha256):
        raise ValueError("attestation digest is malformed")
    if attestation.attestation_sha256 != _attestation_digest(attestation):
        raise ValueError("attestation digest mismatch")
    return attestation


def _fixture() -> tuple[ParticleState, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    gamma: list[tuple[float, float, float]] = []
    sigma: list[float] = []
    for cell in range(CELL_COUNT):
        base_x = -0.31 + 0.085 * cell
        for support in range(SUPPORTS_PER_CELL):
            sign = -1.0 if support == 0 else 1.0
            positions.append(
                (base_x + 0.011 * sign, -0.035 + 0.009 * cell, 0.014 * sign)
            )
            gamma.append(
                (
                    0.018 + 0.001 * cell,
                    sign * (0.009 + 0.0004 * cell),
                    -0.012 + 0.0007 * cell,
                )
            )
            sigma.append(0.075 + 0.0015 * cell + 0.0005 * support)
    state = make_particle_state(positions, gamma, sigma)
    frontier = np.asarray(
        tuple(
            (
                -0.34 + 0.085 * node,
                0.155 + 0.004 * node,
                0.018 * np.sin(0.3 * node),
            )
            for node in range(FRONTIER_NODE_COUNT)
        ),
        dtype=np.float64,
    )
    tracer_pack = np.vstack((state.positions, frontier))
    return state, tracer_pack


def _run_support_case(
    layout: B2SupportLayout,
    initial: ParticleState,
    tracer_pack: np.ndarray,
    *,
    calls: list[str] | None = None,
) -> tuple[IRWRK3StepResult, B2SupportAttestation]:
    validate_b2_support_layout(layout)
    tracer_array = np.asarray(tracer_pack)
    if tracer_array.dtype != np.dtype(np.float64) or tracer_array.shape != (
        TOTAL_TRACER_COUNT,
        3,
    ):
        raise ValueError("initial support tracer pack has an invalid exact tree")
    initial_material, _ = _support_slices(layout, tracer_array)
    initial_physical = initial.positions[
        np.asarray(layout.material_to_physical_indices)
    ]
    if not _same_array_bytes(initial_material, initial_physical):
        raise ValueError("initial material support mismatch before field")

    latest_field: IRWRK3Field | None = None

    def physical(state: ParticleState) -> IRWRK3Field:
        nonlocal latest_field
        if calls is not None:
            calls.append("physical")
        velocity = state.positions @ AFFINE_JACOBIAN.T + AFFINE_TRANSLATION
        jacobian = np.repeat(AFFINE_JACOBIAN[None, :, :], len(state.sigma), axis=0)
        latest_field = make_ir_wrk3_field(
            state, velocity, jacobian, parent_token=PARENT_TOKEN
        )
        return latest_field

    def tracer(
        state: ParticleState, tracer_pre: np.ndarray, parent_token: str
    ) -> IRWRK3TracerField:
        if calls is not None:
            calls.append("tracer")
        if latest_field is None:
            raise RuntimeError("tracer field ran before physical field")
        material_pre, frontier_pre = _support_slices(layout, tracer_pre)
        if not _same_array_bytes(material_pre, state.positions):
            raise RuntimeError("material tracer did not use the physical stage-pre")
        if (
            latest_field.source_state_sha256
            != make_ir_wrk3_field(
                state,
                latest_field.velocity,
                latest_field.jacobian,
                parent_token=parent_token,
            ).source_state_sha256
        ):
            raise RuntimeError("tracer field used a stale physical source")
        frontier_velocity = frontier_pre @ AFFINE_JACOBIAN.T + AFFINE_TRANSLATION
        velocity = np.vstack((latest_field.velocity, frontier_velocity))
        return make_ir_wrk3_tracer_field(
            state, tracer_pre, velocity, parent_token=parent_token
        )

    result = ir_wrk3_step_with_external_field(
        initial,
        DELTA_TIME,
        physical,
        transport_substeps=TRANSPORT_SUBSTEPS,
        tracer_positions=tracer_array,
        tracer_field_evaluator=tracer,
        parent_token=PARENT_TOKEN,
    )
    attestation = _make_attestation(layout, result)
    validate_b2_support_attestation(layout, result, attestation)
    return result, attestation


def test_b2_support_layout_stage_mapping_and_two_run_determinism() -> None:
    layout = _make_layout()
    initial, tracer_pack = _fixture()
    initial_bytes = (
        initial.positions.tobytes(order="C"),
        initial.gamma.tobytes(order="C"),
        initial.sigma.tobytes(order="C"),
        tracer_pack.tobytes(order="C"),
    )
    first, first_attestation = _run_support_case(layout, initial, tracer_pack)
    second, second_attestation = _run_support_case(layout, initial, tracer_pack)

    assert validate_b2_support_layout(layout) is layout
    assert (
        validate_b2_support_attestation(layout, first, first_attestation)
        is first_attestation
    )
    assert (
        validate_b2_support_attestation(layout, second, second_attestation)
        is second_attestation
    )
    assert first.result_sha256 == second.result_sha256
    assert first.stage_chain_sha256 == second.stage_chain_sha256
    assert first_attestation.attestation_sha256 == second_attestation.attestation_sha256
    assert _same_array_bytes(
        first_attestation.final_material_positions,
        second_attestation.final_material_positions,
    )
    assert _same_array_bytes(
        first_attestation.final_material_sigma,
        second_attestation.final_material_sigma,
    )
    assert _same_array_bytes(
        first_attestation.final_frontier_positions,
        second_attestation.final_frontier_positions,
    )
    assert first.counters.invariant_reference_freeze_count == 1
    assert first.counters.stage_count == 3 * TRANSPORT_SUBSTEPS
    assert first.counters.tracer_storage_reset_count == TRANSPORT_SUBSTEPS
    assert first.counters.sigma_storage_update_count == 0
    assert first.counters.relaxation_call_count == 0
    for stage in first.stages:
        assert _same_array_bytes(
            stage.tracer_pre[:MATERIAL_SUPPORT_COUNT], stage.pre.positions
        )
        assert _same_array_bytes(
            stage.tracer_post[:MATERIAL_SUPPORT_COUNT], stage.post.positions
        )
    assert initial_bytes == (
        initial.positions.tobytes(order="C"),
        initial.gamma.tobytes(order="C"),
        initial.sigma.tobytes(order="C"),
        tracer_pack.tobytes(order="C"),
    )


def test_b2_support_layout_rejects_index_id_count_offset_and_frontier_drift() -> None:
    layout = _make_layout()
    swapped_indices = (1, 0, *layout.material_to_physical_indices[2:])
    wrong_material_ids = (
        "material:forged:0",
        *layout.material_tracer_ids[1:],
    )
    frontier_ten = (*layout.frontier_node_ids, "frontier-node:9")
    attacks = (
        replace(layout, material_to_physical_indices=swapped_indices),
        replace(layout, material_tracer_ids=wrong_material_ids),
        replace(layout, material_support_count=MATERIAL_SUPPORT_COUNT - 1),
        replace(layout, cell_support_offsets=(0, 1, *layout.cell_support_offsets[2:])),
        replace(layout, frontier_node_offset=FRONTIER_NODE_OFFSET - 1),
        replace(
            layout,
            frontier_node_count=8,
            frontier_node_ids=layout.frontier_node_ids[:8],
            total_tracer_count=24,
        ),
        replace(
            layout,
            frontier_node_count=10,
            frontier_node_ids=frontier_ten,
            total_tracer_count=26,
        ),
    )
    for attack in attacks:
        with pytest.raises(ValueError, match="index|ID|count|offset|frontier"):
            validate_b2_support_layout(_reseal_layout(attack))

    with pytest.raises(ValueError, match="exact tuples"):
        validate_b2_support_layout(
            replace(layout, cell_ids=list(layout.cell_ids))  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="digest"):
        validate_b2_support_layout(replace(layout, layout_sha256="0" * 64))


def test_initial_material_ulp_mismatch_stops_before_field_and_clean_retry() -> None:
    layout = _make_layout()
    initial, tracer_pack = _fixture()
    mismatched = tracer_pack.copy()
    mismatched[0, 0] = np.nextafter(mismatched[0, 0], np.inf)
    calls: list[str] = []
    with pytest.raises(ValueError, match="before field"):
        _run_support_case(layout, initial, mismatched, calls=calls)
    assert calls == []

    result, attestation = _run_support_case(layout, initial, tracer_pack, calls=calls)
    assert calls == ["physical", "tracer"] * (3 * TRANSPORT_SUBSTEPS)
    assert validate_b2_support_attestation(layout, result, attestation) is attestation


def test_final_support_ulp_reorder_splice_and_binding_attacks_fail_closed() -> None:
    layout = _make_layout()
    initial, tracer_pack = _fixture()
    result, attestation = _run_support_case(layout, initial, tracer_pack)

    ulp_material = np.array(attestation.final_material_positions, copy=True)
    ulp_material[3, 1] = np.nextafter(ulp_material[3, 1], np.inf)
    ulp_attack = _reseal_attestation(
        replace(
            attestation,
            final_material_positions=_frozen_float64(
                ulp_material, (MATERIAL_SUPPORT_COUNT, 3)
            ),
        )
    )
    with pytest.raises(ValueError, match="final material"):
        validate_b2_support_attestation(layout, result, ulp_attack)

    reordered_frontier = np.array(attestation.final_frontier_positions[::-1], copy=True)
    reorder_attack = _reseal_attestation(
        replace(
            attestation,
            final_frontier_positions=_frozen_float64(
                reordered_frontier, (FRONTIER_NODE_COUNT, 3)
            ),
        )
    )
    with pytest.raises(ValueError, match="ordered 9-node frontier"):
        validate_b2_support_attestation(layout, result, reorder_attack)

    alternate_pack = np.array(tracer_pack, copy=True)
    alternate_pack[-1, 2] += 0.02
    _, alternate_attestation = _run_support_case(layout, initial, alternate_pack)
    spliced_frontier = np.array(attestation.final_frontier_positions, copy=True)
    spliced_frontier[-1] = alternate_attestation.final_frontier_positions[-1]
    splice_attack = _reseal_attestation(
        replace(
            attestation,
            final_frontier_positions=_frozen_float64(
                spliced_frontier, (FRONTIER_NODE_COUNT, 3)
            ),
        )
    )
    with pytest.raises(ValueError, match="ordered 9-node frontier"):
        validate_b2_support_attestation(layout, result, splice_attack)

    binding_attacks = (
        replace(attestation, parent_token="wrong-parent"),
        replace(
            attestation,
            stage_source_state_sha256s=(
                "0" * 64,
                *attestation.stage_source_state_sha256s[1:],
            ),
        ),
        replace(
            attestation,
            stage_schedule=(
                (1, 1, (1.0).hex(), attestation.stage_schedule[0][3]),
                *attestation.stage_schedule[1:],
            ),
        ),
        replace(attestation, stage_chain_sha256="0" * 64),
    )
    for attack in binding_attacks:
        with pytest.raises(ValueError, match="parent|source/RK|stage-chain"):
            validate_b2_support_attestation(layout, result, _reseal_attestation(attack))

    mutable_final = np.array(attestation.final_material_positions, copy=True)
    with pytest.raises(ValueError, match="exact frozen"):
        validate_b2_support_attestation(
            layout,
            result,
            replace(attestation, final_material_positions=mutable_final),
        )
    with pytest.raises(ValueError, match="exact tuples"):
        validate_b2_support_attestation(
            layout,
            result,
            replace(
                attestation,
                stage_schedule=list(attestation.stage_schedule),  # type: ignore[arg-type]
            ),
        )

    retry_result, retry_attestation = _run_support_case(layout, initial, tracer_pack)
    assert retry_result.result_sha256 == result.result_sha256
    assert retry_attestation.attestation_sha256 == attestation.attestation_sha256
    assert (
        validate_b2_support_attestation(layout, retry_result, retry_attestation)
        is retry_attestation
    )
