from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
from pterasoftware import _aerodynamics_functions

from claim_runtime.hirato_equations import HiratoEquationError
from forward_flight_benchmarks.fluxv_v5f_material_state import (
    PTERA_REVERSE_RING_TRAVERSAL,
    PteraMaterialLEVSnapshot,
    PteraNativeMaterialLEVState,
    ptera_to_shadow_forward_rings,
    shadow_forward_to_ptera_rings,
)


def _edges() -> tuple[np.ndarray, np.ndarray]:
    leading = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    first_aft = leading.copy()
    first_aft[..., 0] = 0.3
    return leading, first_aft


def _assert_snapshots_identical(
    left: PteraMaterialLEVSnapshot, right: PteraMaterialLEVSnapshot
) -> None:
    for field in fields(PteraMaterialLEVSnapshot):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value, strict=True)
        else:
            assert left_value == right_value


def _independent_forward_segment_velocity(
    point: np.ndarray, ring_forward: np.ndarray, gamma: float
) -> np.ndarray:
    """Unregularized finite-segment field along 0->1->2->3->0."""

    result = np.zeros(3)
    closed = np.concatenate((ring_forward, ring_forward[:1]), axis=0)
    for start, end in zip(closed[:-1], closed[1:], strict=True):
        r_start = point - start
        r_end = point - end
        segment = end - start
        cross = np.cross(r_start, r_end)
        result += (
            gamma
            * cross
            / (4.0 * np.pi * np.dot(cross, cross))
            * np.dot(
                segment,
                r_start / np.linalg.norm(r_start) - r_end / np.linalg.norm(r_end),
            )
        )
    return result


def test_public_ring_reversal_preserves_physical_gamma_in_ptera_kernel() -> None:
    shadow_ring = np.array(
        [
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ]
    )
    gamma = 0.37
    point = np.array([[0.1, -0.2, 0.8]])
    ptera_ring = shadow_forward_to_ptera_rings(shadow_ring)
    np.testing.assert_array_equal(
        ptera_ring, shadow_ring[:, PTERA_REVERSE_RING_TRAVERSAL]
    )
    np.testing.assert_array_equal(
        ptera_to_shadow_forward_rings(ptera_ring), shadow_ring
    )

    singularity_counts = np.zeros(4, dtype=np.int64)
    actual = _aerodynamics_functions.expanded_velocities_from_ring_vortices(
        stackP_GP1_CgP1=point,
        stackBrrvp_GP1_CgP1=ptera_ring[:, 2],
        stackFrrvp_GP1_CgP1=ptera_ring[:, 1],
        stackFlrvp_GP1_CgP1=ptera_ring[:, 0],
        stackBlrvp_GP1_CgP1=ptera_ring[:, 3],
        strengths=np.array([gamma]),
        r_c0s=np.array([0.0]),
        singularity_counts=singularity_counts,
        ages=None,
        nu=0.0,
    )
    expected = _independent_forward_segment_velocity(point[0], shadow_ring[0], gamma)
    np.testing.assert_allclose(actual[0, 0], expected, atol=3.0e-15, rtol=3.0e-15)
    assert float(np.dot(actual[0, 0], expected)) > 0.0

    unreversed = _aerodynamics_functions.expanded_velocities_from_ring_vortices(
        stackP_GP1_CgP1=point,
        stackBrrvp_GP1_CgP1=shadow_ring[:, 2],
        stackFrrvp_GP1_CgP1=shadow_ring[:, 1],
        stackFlrvp_GP1_CgP1=shadow_ring[:, 0],
        stackBlrvp_GP1_CgP1=shadow_ring[:, 3],
        strengths=np.array([gamma]),
        r_c0s=np.array([0.0]),
        singularity_counts=np.zeros(4, dtype=np.int64),
        ages=None,
        nu=0.0,
    )
    np.testing.assert_allclose(unreversed[0, 0], -expected, atol=3.0e-15, rtol=3.0e-15)
    np.testing.assert_array_equal(singularity_counts, np.zeros(4, dtype=np.int64))


def test_first_shed_is_transactional_and_exposes_created_ids_and_strips() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    before = state.snapshot()
    proposal = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    )
    _assert_snapshots_identical(state.snapshot(), before)
    np.testing.assert_array_equal(proposal.created_ids, [0])
    np.testing.assert_array_equal(proposal.active_strip_indices, [0])
    expected_shadow = np.stack(
        (leading[0, 0], leading[0, 1], first_aft[0, 1], first_aft[0, 0])
    )[None, ...]
    np.testing.assert_array_equal(
        proposal.newborn_ptera_rings_gp1_m,
        expected_shadow[:, PTERA_REVERSE_RING_TRAVERSAL],
    )

    candidate_zero = proposal.candidate_material_view()
    np.testing.assert_array_equal(candidate_zero.gamma_m2_s, [0.0])
    candidate_solved = proposal.candidate_material_view(
        created_strengths_m2_s=np.array([0.2])
    )
    np.testing.assert_array_equal(candidate_solved.gamma_m2_s, [0.2])
    assert state.material_view().gamma_m2_s.size == 0

    proposal.commit(np.array([0.2]))
    view = state.material_view()
    np.testing.assert_array_equal(view.gamma_m2_s, [0.2])
    np.testing.assert_array_equal(view.strip, [0])
    np.testing.assert_array_equal(view.birth_step, [0])
    np.testing.assert_array_equal(view.age_steps, [0])
    np.testing.assert_array_equal(state.indices_for_strip(0), [0])
    np.testing.assert_array_equal(state.ring_ids_for_strip(0), [0])
    np.testing.assert_array_equal(state.indices_for_strip(1), np.empty(0, dtype=int))
    with pytest.raises(RuntimeError, match="already been finalized"):
        proposal.candidate_material_view()


def test_atomic_commit_and_convection_publishes_only_after_validation() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    proposal = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    )
    before = state.snapshot()
    with pytest.raises(HiratoEquationError, match="must match"):
        proposal.commit_and_convect(
            np.array([0.2]),
            np.zeros((0, 4, 3)),
            dt_s=0.1,
        )
    _assert_snapshots_identical(state.snapshot(), before)
    assert not proposal.is_finalized

    velocity = np.full((1, 4, 3), 0.25)
    snapshot, report = proposal.commit_and_convect(
        np.array([0.2]),
        velocity,
        dt_s=0.1,
    )
    np.testing.assert_array_equal(snapshot.gamma_m2_s, [0.2])
    np.testing.assert_allclose(
        snapshot.ptera_rings_gp1_m,
        proposal.newborn_ptera_rings_gp1_m + 0.025,
    )
    np.testing.assert_allclose(report.displacement_gp1_m, 0.025)
    assert snapshot.last_committed_step == 0
    assert snapshot.last_convected_step == 0
    assert proposal.is_finalized


def test_atomic_empty_step_advances_without_creating_material() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    proposal = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([False, False]),
        first_aft_edges_gp1_m=first_aft,
    )
    snapshot, report = proposal.commit_and_convect(
        np.empty(0),
        np.empty((0, 4, 3)),
        dt_s=0.1,
    )
    assert snapshot.gamma_m2_s.size == 0
    assert snapshot.last_committed_step == 0
    assert snapshot.last_convected_step == 0
    assert report.displacement_gp1_m.shape == (0, 4, 3)


def test_continuation_applies_only_eq7_geometry_and_preserves_old_gamma_bits() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=0)
    before = state.snapshot()
    old_gamma_bytes = before.gamma_m2_s.tobytes()

    proposal = state.propose_shed(
        step=1,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
    )
    np.testing.assert_array_equal(proposal.created_ids, [1])
    proposal.commit_by_strip(np.array([0.35, 0.0]))
    after = state.snapshot()
    assert after.gamma_m2_s[:1].tobytes() == old_gamma_bytes
    np.testing.assert_array_equal(after.gamma_m2_s, [0.2, 0.35])
    np.testing.assert_array_equal(after.ring_id, [0, 1])
    np.testing.assert_array_equal(state.ring_ids_for_strip(0), [0, 1])

    expected_split = leading[0].copy()
    expected_split[..., 0] = 0.1
    np.testing.assert_allclose(
        after.shadow_rings_forward_gp1_m[0, :2], expected_split, atol=0.0
    )
    np.testing.assert_allclose(
        after.shadow_rings_forward_gp1_m[1, [3, 2]], expected_split, atol=2.0e-16
    )
    np.testing.assert_array_equal(
        state.material_view(reference_step=3).age_steps, [3, 2]
    )


def test_deactivation_creates_nothing_and_leaves_material_arrays_bitwise() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=0)
    material_before = state.material_view(reference_step=1)

    proposal = state.propose_shed(
        step=1,
        leading_edges_gp1_m=leading,
        active=np.array([False, False]),
    )
    np.testing.assert_array_equal(proposal.created_ids, np.empty(0, dtype=int))
    assert proposal.newborn_ptera_rings_gp1_m.shape == (0, 4, 3)
    proposal.commit(np.empty(0))
    material_after = state.material_view(reference_step=1)
    for name in (
        "rings_gp1_m",
        "gamma_m2_s",
        "strip",
        "birth_step",
        "ring_id",
        "sheet_id",
    ):
        np.testing.assert_array_equal(
            getattr(material_after, name), getattr(material_before, name)
        )


def test_reactivation_can_start_a_detached_sheet_without_remeshing_old_ring() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=0)
    state.propose_shed(
        step=1,
        leading_edges_gp1_m=leading,
        active=np.array([False, False]),
    ).commit([])
    state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=1)
    old = state.snapshot()

    restarted_aft = first_aft.copy()
    restarted_aft[..., 0] = 0.15
    proposal = state.propose_shed(
        step=2,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        new_sheet=np.array([True, False]),
        first_aft_edges_gp1_m=restarted_aft,
    )
    proposal.commit([0.4])
    after = state.snapshot()
    np.testing.assert_array_equal(
        after.shadow_rings_forward_gp1_m[0], old.shadow_rings_forward_gp1_m[0]
    )
    assert after.sheet_id[1] != after.sheet_id[0]
    np.testing.assert_array_equal(after.ring_id, [0, 1])
    np.testing.assert_allclose(
        after.shadow_rings_forward_gp1_m[1, 2:, 0], 0.15, atol=0.0
    )


def test_aborted_or_failed_proposal_cannot_mutate_original_state() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    before = state.snapshot()
    proposal = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, True]),
        first_aft_edges_gp1_m=first_aft,
    )
    with pytest.raises(HiratoEquationError, match="finite"):
        proposal.commit([0.2, np.nan])
    _assert_snapshots_identical(state.snapshot(), before)
    proposal.abort()
    _assert_snapshots_identical(state.snapshot(), before)
    with pytest.raises(RuntimeError, match="finalized"):
        proposal.commit([0.2, -0.1])


def test_missing_first_placement_failure_rolls_back_original_state() -> None:
    leading, _ = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    before = state.snapshot()
    with pytest.raises(HiratoEquationError, match="first_aft_edges"):
        state.propose_shed(
            step=0,
            leading_edges_gp1_m=leading,
            active=np.array([True, False]),
        )
    _assert_snapshots_identical(state.snapshot(), before)


def test_stale_parallel_proposal_fails_without_overwriting_committed_state() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    first = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    )
    stale = state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([False, True]),
        first_aft_edges_gp1_m=first_aft,
    )
    first.commit([0.2])
    committed = state.snapshot()
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit([-0.1])
    _assert_snapshots_identical(state.snapshot(), committed)


def test_eq24_convection_maps_ptera_vertices_and_bootstraps_birth() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    before = state.material_view().rings_gp1_m
    gamma_bytes = state.material_view().gamma_m2_s.tobytes()
    velocity = np.arange(12, dtype=float).reshape(1, 4, 3) / 10.0
    report = state.convect_eq24(velocity, dt_s=0.25, step=0)
    np.testing.assert_array_equal(report.bootstrap_vertex, np.ones((1, 4), dtype=bool))
    np.testing.assert_allclose(report.displacement_gp1_m, 0.25 * velocity)
    np.testing.assert_allclose(
        state.material_view().rings_gp1_m, before + 0.25 * velocity
    )
    assert state.material_view().gamma_m2_s.tobytes() == gamma_bytes


def test_eq7_remesh_bootstraps_only_affected_old_front_and_newborn_vertices() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    state.convect_eq24(np.full((1, 4, 3), 2.0), dt_s=0.1, step=0)
    state.propose_shed(
        step=1,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
    ).commit([0.35])
    gamma_before = state.material_view().gamma_m2_s.copy()
    report = state.convect_eq24(np.full((2, 4, 3), 4.0), dt_s=0.1, step=1)
    expected_bootstrap = np.array(
        [[True, True, False, False], [True, True, True, True]]
    )
    np.testing.assert_array_equal(report.bootstrap_vertex, expected_bootstrap)
    np.testing.assert_allclose(report.displacement_gp1_m[0, :2], 0.4)
    np.testing.assert_allclose(report.displacement_gp1_m[0, 2:], 0.3)
    np.testing.assert_allclose(report.displacement_gp1_m[1], 0.4)
    np.testing.assert_array_equal(state.material_view().gamma_m2_s, gamma_before)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": 1}, "advance by one"),
        ({"step": 0.0}, "integer"),
        ({"active": np.array([1, 0])}, "boolean"),
        ({"leading_edges_gp1_m": np.full((2, 2, 3), np.nan)}, "finite"),
        (
            {"new_sheet": np.array([True, False]), "active": np.array([False, False])},
            "active strips",
        ),
    ],
)
def test_proposal_inputs_fail_closed(kwargs: dict[str, object], match: str) -> None:
    leading, first_aft = _edges()
    arguments: dict[str, object] = {
        "step": 0,
        "leading_edges_gp1_m": leading,
        "active": np.array([False, False]),
        "first_aft_edges_gp1_m": first_aft,
    }
    arguments.update(kwargs)
    with pytest.raises(HiratoEquationError, match=match):
        PteraNativeMaterialLEVState(ns=2).propose_shed(**arguments)


def test_step_and_convection_sequence_fail_closed() -> None:
    leading, first_aft = _edges()
    state = PteraNativeMaterialLEVState(ns=2)
    state.propose_shed(
        step=0,
        leading_edges_gp1_m=leading,
        active=np.array([True, False]),
        first_aft_edges_gp1_m=first_aft,
    ).commit([0.2])
    with pytest.raises(HiratoEquationError, match="must be convected"):
        state.propose_shed(
            step=1,
            leading_edges_gp1_m=leading,
            active=np.array([False, False]),
        )
    with pytest.raises(HiratoEquationError, match="match the material-ring shape"):
        state.convect_eq24(np.zeros((2, 4, 3)), dt_s=0.1, step=0)
    before = state.snapshot()
    with pytest.raises(HiratoEquationError, match="finite"):
        state.convect_eq24(np.full((1, 4, 3), np.nan), dt_s=0.1, step=0)
    _assert_snapshots_identical(state.snapshot(), before)
    state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=0)
    with pytest.raises(HiratoEquationError, match="already"):
        state.convect_eq24(np.zeros((1, 4, 3)), dt_s=0.1, step=0)


def test_lookup_and_reference_age_validation_fail_closed() -> None:
    state = PteraNativeMaterialLEVState(ns=2)
    with pytest.raises(HiratoEquationError, match="integer"):
        state.indices_for_strip(0.0)
    with pytest.raises(HiratoEquationError, match=r"\[0, 2\)"):
        state.indices_for_strip(2)
    with pytest.raises(HiratoEquationError, match="non-negative integer"):
        state.material_view(reference_step=0.5)
