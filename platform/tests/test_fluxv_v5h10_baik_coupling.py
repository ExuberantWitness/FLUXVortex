from __future__ import annotations

from itertools import count
import os
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h10-coupling-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h10-coupling-mpl")

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.rvpm_transport import ParticleState
import forward_flight_benchmarks.fluxv_v5h10_baik_coupling as coupling_module
from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES, build_baik_movement
from forward_flight_benchmarks.fluxv_v5h10_baik_coupling import (
    ACTIVE_BIRTH_MODES,
    ACTIVE_PTERA_STEPS,
    ACTIVE_SOURCE_STEPS,
    CASE_ID,
    COUPLING_INTERFACE_ID,
    FORCE_SCORING_STATUS,
    OBSERVATION_ACCESS,
    SURFACE_FORCE_OWNER,
    BaikW2CouplingConfig,
    GlobalRowCommitRequest,
    GlobalRowPteraRVPMCouplingSolver,
    make_fluxv_v5h10_baik_w2_solver,
)
from forward_flight_benchmarks.fluxv_v5h3_native_feedback import (
    SURFACE_LOAD_OWNER,
    NativePteraRVPMFeedbackSolver,
)
from forward_flight_benchmarks.fluxv_v5h10_row_owner import (
    RowCommitResult,
    advance_release_row_transport_parent,
    attest_release_row_common_transport,
    begin_release_row_common_transport,
    bootstrap_release_row_owner,
    commit_release_row_update,
    make_release_row,
    propose_release_row_update,
    release_row_transport_digest,
    validate_current_release_row_owner,
)


_OWNER_IDS = count()


class _MockCommitter:
    def __init__(self, *, explode_at: int | None = None) -> None:
        self.requests: list[GlobalRowCommitRequest] = []
        self.commits: list[RowCommitResult] = []
        self.explode_at = explode_at

    def __call__(self, request: GlobalRowCommitRequest) -> RowCommitResult:
        if self.explode_at == len(self.requests):
            raise RuntimeError("injected row commit failure")
        self.requests.append(request)
        release = len(self.commits) + 1
        if request.previous_owner is None:
            upstream = np.column_stack(
                (np.full(9, 0.10), np.linspace(0.0, 0.6, 9), np.zeros(9))
            )
        else:
            upstream = request.previous_owner.state.live_boundary_nodes.copy()
        downstream = upstream.copy()
        downstream[:, 0] += 0.04
        row = make_release_row(
            upstream,
            downstream,
            np.full(8, 1.0e-10 * (1.0 + 0.01 * release)),
            release_index=release,
            source_time_s=request.source_time_s,
            sheet_id="baik-w2-test-row",
        )
        if request.previous_owner is None:
            owner = bootstrap_release_row_owner(
                row,
                smoothing_radius_m=0.10,
                target_spacing_m=0.04,
                release_dt_s=0.11125,
                particle_cap=1000,
                owner_id=f"v5h10-coupling-test:{next(_OWNER_IDS)}",
            )
            commit = RowCommitResult(
                committed=True,
                status="compatible",
                owner=owner,
                state=owner.state,
                event=None,
                first_mismatch=None,
            )
        else:
            proposal = propose_release_row_update(
                request.previous_owner,
                row,
                proposal_id=f"coupling-release-{release}",
            )
            if proposal.status != "compatible":
                raise RuntimeError(f"row requires remesh: {proposal.first_mismatch}")
            commit = commit_release_row_update(request.previous_owner, proposal)
        self.commits.append(commit)
        return commit


def _problem() -> ps.problems.UnsteadyProblem:
    movement, metadata = build_baik_movement(BAIK_2012_CASES[CASE_ID], "smoke")
    assert metadata["grid_chord_span"] == [2, 8]
    assert movement.delta_time == 0.11125
    return ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )


def _solver(
    monkeypatch: pytest.MonkeyPatch,
    committer: Any,
) -> GlobalRowPteraRVPMCouplingSolver:
    del monkeypatch
    solver = make_fluxv_v5h10_baik_w2_solver(
        _problem(),
        row_committer=committer,
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    assert type(solver) is GlobalRowPteraRVPMCouplingSolver
    return solver


def _run(
    monkeypatch: pytest.MonkeyPatch,
    committer: Any | None = None,
) -> tuple[GlobalRowPteraRVPMCouplingSolver, Any]:
    callback = _MockCommitter() if committer is None else committer
    solver = _solver(monkeypatch, callback)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver, callback


def test_frozen_config_rejects_case_numerics_and_gate_tuning() -> None:
    assert BaikW2CouplingConfig() is not None
    with pytest.raises(ValueError, match="preregistered and frozen"):
        BaikW2CouplingConfig(case_id="W1")
    with pytest.raises(ValueError, match="preregistered and frozen"):
        BaikW2CouplingConfig(spanwise_panels=7)
    with pytest.raises(ValueError, match="cannot relax"):
        BaikW2CouplingConfig(no_penetration_atol=2.0e-12)
    with pytest.raises(ValueError, match="outside the preregistered"):
        BaikW2CouplingConfig(relative_epsilon=1.0e-3)


def test_readonly_copy_never_aliases_or_changes_parent_array_flags() -> None:
    parent = np.arange(6, dtype=np.float64).reshape(2, 3)
    parent_bytes = parent.tobytes(order="C")
    parent_writeable = parent.flags.writeable
    frozen = coupling_module._readonly_float64("parent", parent, ndim=2)
    assert not np.shares_memory(frozen, parent)
    assert frozen.flags.c_contiguous
    assert frozen.flags.writeable is False
    assert parent.tobytes(order="C") == parent_bytes
    assert parent.flags.writeable is parent_writeable is True


def test_real_ptera_runs_pre_active_history_then_three_lazy_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, committer = _run(monkeypatch)
    assert solver.v5h10_slice_complete
    assert [request.ptera_step_index for request in committer.requests] == list(
        ACTIVE_PTERA_STEPS
    )
    assert ACTIVE_PTERA_STEPS == (3, 4, 5)
    assert [request.source_step_index for request in committer.requests] == list(
        ACTIVE_SOURCE_STEPS
    )
    assert [
        (request.source_step_index, request.ptera_step_index)
        for request in committer.requests
    ] == [(4, 3), (5, 4), (6, 5)]
    assert [request.expected_birth_mode for request in committer.requests] == list(
        ACTIVE_BIRTH_MODES
    )
    assert [request.source_time_s for request in committer.requests] == [
        step * 0.11125 for step in ACTIVE_PTERA_STEPS
    ]
    assert committer.requests[0].previous_owner is None
    assert committer.requests[0].transported_parent is None
    for index in (1, 2):
        request = committer.requests[index]
        assert request.previous_owner is not None
        assert (
            request.previous_owner.owner_sha256
            == solver.v5h10_raw_step_records[index - 1].advanced_owner_sha256
        )
        assert type(request.transported_parent) is ParticleState
        assert request.transported_parent_sha256 == (
            solver.v5h10_raw_step_records[index - 1].transported_arrays_sha256
        )

    records = solver.v5h10_raw_step_records
    assert len(records) == 3
    for index, record in enumerate(records):
        assert record.interface_id == COUPLING_INTERFACE_ID
        assert record.case_id == CASE_ID
        assert record.ptera_step_index == ACTIVE_PTERA_STEPS[index]
        assert record.source_step_index == ACTIVE_SOURCE_STEPS[index]
        assert record.birth_mode == ACTIVE_BIRTH_MODES[index]
        assert record.collocation_evaluation_count == 1
        assert record.load_batch_evaluation_count == 4
        assert record.native_load_call_count == 1
        assert record.transport_call_count == 1
        assert record.transport_stage_count == 3
        assert len(record.common_transport_sha256) == 64
        assert len(record.common_transport_attestation_sha256) == 64
        assert len(record.transported_material_tracers_sha256) == 64
        assert record.material_tracer_count > 9
        assert record.material_support_tracer_count > 0
        assert record.frontier_node_tracer_count == 9
        assert record.material_tracer_count == (
            record.material_support_tracer_count + record.frontier_node_tracer_count
        )
        assert record.common_transport_self_field_call_count == 3
        assert record.common_transport_ptera_center_call_count == 3
        assert record.common_transport_stage_count == 3
        assert record.frontier_self_field_call_count == 3
        assert record.frontier_ptera_center_call_count == 3
        assert record.frontier_transport_stage_count == 3
        assert record.transported_live_boundary_nodes.shape == (9, 3)
        assert record.transported_live_boundary_nodes.flags.writeable is False
        assert record.ptera_parent_state_unchanged
        assert (
            record.ptera_parent_sha256_before_transport
            == record.ptera_parent_sha256_after_transport
            == record.ptera_parent_sha256_after_raw_record
        )
        assert record.ptera_parent_state_unchanged_after_raw_record
        assert record.observation_access == OBSERVATION_ACCESS
        assert record.surface_load_owner == SURFACE_FORCE_OWNER == SURFACE_LOAD_OWNER
        assert record.source_load_owner == "forbidden"
        assert record.force_scoring_status == FORCE_SCORING_STATUS
        assert record.feedback_report.no_penetration_max_abs <= 1.0e-12
        assert len(record.feedback_report.load_evaluations) == 4
        assert len(record.transport_result.stages) == 3
        assert record.ptera_forces_w.flags.writeable is False
        assert record.ptera_force_coefficients_w.flags.writeable is False
        assert record.ptera_moments_w_cgp1.flags.writeable is False
        assert record.ptera_moment_coefficients_w.flags.writeable is False
        assert len(record.ptera_forces_w_sha256) == 64
        assert len(record.ptera_force_coefficients_w_sha256) == 64
        assert len(record.ptera_moments_w_cgp1_sha256) == 64
        assert len(record.ptera_moment_coefficients_w_sha256) == 64
        assert len(record.ptera_panel_ids) == 16
        assert record.ptera_panel_ids[0] == "airplane:0/wing:0/chord:0/span:0"
        assert record.ptera_panel_ids[-1] == "airplane:0/wing:0/chord:1/span:7"
        assert record.ptera_panel_forces_w.shape == (16, 3)
        assert record.ptera_panel_moments_w_cgp1.shape == (16, 3)
        assert record.ptera_panel_forces_w.flags.writeable is False
        assert record.ptera_panel_moments_w_cgp1.flags.writeable is False
        assert record.ptera_panel_force_sum_w.flags.writeable is False
        assert record.ptera_panel_moment_sum_w_cgp1.flags.writeable is False
        assert len(record.ptera_panel_forces_w_sha256) == 64
        assert len(record.ptera_panel_moments_w_cgp1_sha256) == 64
        assert np.array_equal(
            record.ptera_panel_force_sum_w,
            np.sum(record.ptera_panel_forces_w, axis=0),
        )
        assert np.array_equal(
            record.ptera_panel_moment_sum_w_cgp1,
            np.sum(record.ptera_panel_moments_w_cgp1, axis=0),
        )
        assert (
            record.ptera_panel_force_sum_max_abs_residual
            <= record.ptera_panel_force_sum_atol
        )
        assert (
            record.ptera_panel_moment_sum_max_abs_residual
            <= record.ptera_panel_moment_sum_atol
        )
        assert np.all(np.isfinite(record.ptera_forces_w))
        assert np.all(np.isfinite(record.ptera_force_coefficients_w))
    airplane = solver.current_airplanes[0]
    for parent in (
        airplane.forces_W,
        airplane.forceCoefficients_W,
        airplane.moments_W_CgP1,
        airplane.momentCoefficients_W_CgP1,
    ):
        assert parent.flags.writeable
    for panel in solver.panels:
        for parent in (
            panel.forces_GP1,
            panel.moments_GP1_CgP1,
            panel.forces_W,
            panel.moments_W_CgP1,
        ):
            assert parent.flags.writeable


def test_second_row_commit_failure_is_fail_stop_without_duplicate_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committer = _MockCommitter(explode_at=1)
    solver = _solver(monkeypatch, committer)
    with pytest.raises(RuntimeError, match="injected row commit failure"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert len(solver.v5h10_raw_step_records) == 1
    assert solver.v5h10_raw_step_records[0].transport_call_count == 1
    assert not solver.v5h10_slice_complete
    with pytest.raises(RuntimeError, match="poisoned"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )


def test_valid_row_commit_precedes_every_active_parent_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _MockCommitter()
    committed_steps: set[int] = set()
    parent_wake_steps: list[int] = []

    def callback(request: GlobalRowCommitRequest) -> RowCommitResult:
        result = base(request)
        committed_steps.add(request.ptera_step_index)
        return result

    original_parent_wake = NativePteraRVPMFeedbackSolver._calculate_wake_wing_influences

    def guarded_parent_wake(self: Any) -> Any:
        step = int(self._current_step)
        if step in ACTIVE_PTERA_STEPS:
            assert step in committed_steps
            parent_wake_steps.append(step)
        return original_parent_wake(self)

    monkeypatch.setattr(
        NativePteraRVPMFeedbackSolver,
        "_calculate_wake_wing_influences",
        guarded_parent_wake,
    )
    solver = _solver(monkeypatch, callback)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    assert parent_wake_steps == list(ACTIVE_PTERA_STEPS)
    assert solver.v5h10_slice_complete


def test_foreign_commit_schema_stops_before_feedback_or_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _solver(monkeypatch, lambda request: object())
    with pytest.raises(TypeError, match="foreign commit-result schema"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert solver.v5h3_feedback_step_reports == []
    assert solver.v5h10_raw_step_records == []


def test_callback_returning_consumed_stale_owner_stops_before_active_ptera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _MockCommitter()
    advanced: list[Any] = []

    def stale_callback(request: GlobalRowCommitRequest) -> RowCommitResult:
        result = base(request)
        owner = result.owner
        state = owner.state
        end_time = state.rows[-1].source_time_s + owner.release_dt_s
        common_transport = begin_release_row_common_transport(owner, state)
        flat_live_indices = tuple(
            index
            for indices in common_transport.live_particle_indices_by_cell
            for index in indices
        )
        attestation = attest_release_row_common_transport(
            common_transport,
            state.positions,
            state.gamma,
            state.sigma,
            common_transport.material_tracer_positions,
            state.sigma[np.asarray(flat_live_indices, dtype=np.int64)],
            source_step_index=request.source_step_index,
            transport_end_time_s=end_time,
            transport_epoch=owner.epoch,
        )
        digest = release_row_transport_digest(
            state,
            state.positions,
            state.gamma,
            state.sigma,
            state.live_boundary_nodes,
            common_transport_attestation=attestation,
            source_step_index=request.source_step_index,
            transport_end_time_s=end_time,
            transport_epoch=owner.epoch,
        )
        advanced.append(
            advance_release_row_transport_parent(
                owner,
                state,
                state.positions,
                state.gamma,
                state.sigma,
                state.live_boundary_nodes,
                common_transport_attestation=attestation,
                parent_transport_digest=digest,
                source_step_index=request.source_step_index,
                transport_end_time_s=end_time,
                transport_epoch=owner.epoch,
            )
        )
        return result

    counters = {"parent_wake": 0, "field": 0, "active_load": 0, "transport": 0}
    original_parent_wake = NativePteraRVPMFeedbackSolver._calculate_wake_wing_influences
    original_field = GlobalRowPteraRVPMCouplingSolver._field_velocity
    original_load = GlobalRowPteraRVPMCouplingSolver._calculate_loads
    original_transport = GlobalRowPteraRVPMCouplingSolver._transport_active_row

    def counted_parent_wake(self: Any) -> Any:
        if int(self._current_step) in ACTIVE_PTERA_STEPS:
            counters["parent_wake"] += 1
        return original_parent_wake(self)

    def counted_field(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters["field"] += 1
        return original_field(self, *args, **kwargs)

    def counted_load(self: Any) -> Any:
        if int(self._current_step) in ACTIVE_PTERA_STEPS:
            counters["active_load"] += 1
        return original_load(self)

    def counted_transport(self: Any) -> Any:
        counters["transport"] += 1
        return original_transport(self)

    monkeypatch.setattr(
        NativePteraRVPMFeedbackSolver,
        "_calculate_wake_wing_influences",
        counted_parent_wake,
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver, "_field_velocity", counted_field
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver, "_calculate_loads", counted_load
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver,
        "_transport_active_row",
        counted_transport,
    )
    solver = _solver(monkeypatch, stale_callback)
    with pytest.raises(RuntimeError, match="stale or already consumed"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert len(advanced) == 1
    assert counters == {
        "parent_wake": 0,
        "field": 0,
        "active_load": 0,
        "transport": 0,
    }
    assert solver.v5h3_feedback_step_reports == []
    assert solver.v5h10_raw_step_records == []


def test_remesh_required_is_a_hard_stop_before_second_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _MockCommitter()
    blocked_step = ACTIVE_PTERA_STEPS[1]
    counters = {"parent_wake": 0, "field": 0, "active_load": 0, "transport": 0}
    original_parent_wake = NativePteraRVPMFeedbackSolver._calculate_wake_wing_influences
    original_field = GlobalRowPteraRVPMCouplingSolver._field_velocity
    original_load = GlobalRowPteraRVPMCouplingSolver._calculate_loads
    original_transport = GlobalRowPteraRVPMCouplingSolver._transport_active_row

    def counted_parent_wake(self: Any) -> Any:
        if int(self._current_step) == blocked_step:
            counters["parent_wake"] += 1
        return original_parent_wake(self)

    def counted_field(self: Any, *args: Any, **kwargs: Any) -> Any:
        if int(self._current_step) == blocked_step:
            counters["field"] += 1
        return original_field(self, *args, **kwargs)

    def counted_load(self: Any) -> Any:
        if int(self._current_step) == blocked_step:
            counters["active_load"] += 1
        return original_load(self)

    def counted_transport(self: Any) -> Any:
        if int(self._current_step) == blocked_step:
            counters["transport"] += 1
        return original_transport(self)

    monkeypatch.setattr(
        NativePteraRVPMFeedbackSolver,
        "_calculate_wake_wing_influences",
        counted_parent_wake,
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver, "_field_velocity", counted_field
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver, "_calculate_loads", counted_load
    )
    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver,
        "_transport_active_row",
        counted_transport,
    )

    def callback(request: GlobalRowCommitRequest) -> RowCommitResult:
        if request.previous_owner is None:
            return first(request)
        upstream = request.previous_owner.state.live_boundary_nodes[:-1].copy()
        downstream = upstream.copy()
        downstream[:, 0] += 0.04
        row = make_release_row(
            upstream,
            downstream,
            np.full(7, 1.02e-10),
            release_index=2,
            source_time_s=request.source_time_s,
            sheet_id="baik-w2-test-row",
        )
        proposal = propose_release_row_update(
            request.previous_owner,
            row,
            proposal_id="injected-remesh",
        )
        assert proposal.status == "remesh_required"
        return commit_release_row_update(request.previous_owner, proposal)

    solver = _solver(monkeypatch, callback)
    with pytest.raises(RuntimeError, match="did not produce a compatible commit"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert len(solver.v5h3_feedback_step_reports) == 1
    assert len(solver.v5h10_raw_step_records) == 1
    assert counters == {
        "parent_wake": 0,
        "field": 0,
        "active_load": 0,
        "transport": 0,
    }
    assert not solver.v5h10_slice_complete


def test_parent_hash_change_during_transport_stops_before_raw_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _solver(monkeypatch, _MockCommitter())
    original = coupling_module.ptera_parent_state_sha256
    calls = 0

    def changed_hash(value: object) -> str:
        nonlocal calls
        calls += 1
        digest = original(value)
        return digest if calls % 2 else "f" * 64

    monkeypatch.setattr(coupling_module, "ptera_parent_state_sha256", changed_hash)
    with pytest.raises(RuntimeError, match="mutated native Ptera parent"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert len(solver.v5h3_feedback_step_reports) == 1
    assert solver.v5h10_raw_step_records == []
    assert not solver.v5h10_slice_complete


def test_panel_load_gate_fails_before_transport_owner_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committer = _MockCommitter()
    solver = _solver(monkeypatch, committer)
    original_transport = GlobalRowPteraRVPMCouplingSolver._transport_active_row

    def corrupt_panel_load(self: Any) -> Any:
        result = original_transport(self)
        self.panels[0].forces_W[0] += 1.0
        return result

    monkeypatch.setattr(
        GlobalRowPteraRVPMCouplingSolver,
        "_transport_active_row",
        corrupt_panel_load,
    )
    with pytest.raises(RuntimeError, match="panel forces do not sum"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
    assert len(committer.commits) == 1
    owner = committer.commits[0].owner
    assert validate_current_release_row_owner(owner) is owner
    assert owner.state.phase == "post_commit_pre_transport"
    assert owner.transport_events == ()
    assert solver.v5h10_raw_step_records == []


def test_bounded_slice_rejects_streamlines_and_duplicate_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _solver(monkeypatch, _MockCommitter())
    with pytest.raises(ValueError, match="forbids streamline"):
        solver.run(calculate_streamlines=True)
    solver.run(
        prescribed_wake=True,
        calculate_streamlines=False,
        show_progress=False,
    )
    with pytest.raises(RuntimeError, match="cannot be run twice"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
