"""No-GT focused tests for the v5h15 B3 coupling transaction."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import count
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h15-coupling-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h15-coupling-mpl")

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.rvpm_ir_wrk3_fd_adapter import make_frozen_parent_velocity

import forward_flight_benchmarks.fluxv_v5h15_baik_coupling as coupling
from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES, build_baik_movement
from forward_flight_benchmarks.fluxv_v5h3_native_feedback import (
    NativePteraRVPMFeedbackSolver,
)
from forward_flight_benchmarks.fluxv_v5h10_row_owner import (
    RowCommitResult,
    bootstrap_release_row_owner,
    commit_release_row_update,
    make_release_row,
    propose_release_row_update,
    validate_current_release_row_owner,
)
from forward_flight_benchmarks.fluxv_v5h15_baik_coupling import (
    ACTIVE_BIRTH_MODES,
    ACTIVE_PTERA_STEPS,
    ACTIVE_SOURCE_STEPS,
    FORMAL_TRANSPORT_SUBSTEPS,
    V5H15BaikCouplingConfig,
    V5H15GlobalRowCommitRequest,
    V5H15RowCommitEnvelope,
    make_fluxv_v5h15_baik_w2_solver,
    make_v5h15_layer_load_ledger,
    make_v5h15_source_kelvin_evidence,
    transport_v5h15_committed_layer,
    validate_v5h15_layer_load_ledger,
    validate_v5h15_layer_result,
)


_OWNER_INDEX = count()
_PARENT_SHA = "1" * 64
_PARENT_TOKEN = "synthetic-frozen-parent"
_DT = 1.0e-5


class _ParentHash:
    __slots__ = ("value",)

    def __init__(self, value: str = _PARENT_SHA) -> None:
        self.value = value

    def __call__(self) -> str:
        return self.value


class _ZeroFrozenParent:
    __slots__ = ("calls", "fail_on", "parent_sha", "parent_token")

    def __init__(self, *, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.parent_sha = _PARENT_SHA
        self.parent_token = _PARENT_TOKEN

    def __call__(self, targets: np.ndarray):
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("synthetic parent failure")
        return make_frozen_parent_velocity(
            targets,
            np.zeros_like(targets),
            parent_token=self.parent_token,
            parent_state_sha256=self.parent_sha,
        )


class _RuntimeRebindingParent(_ZeroFrozenParent):
    __slots__ = ("target_name", "replacement")

    def __init__(self, target_name: str, replacement: object) -> None:
        super().__init__()
        self.target_name = target_name
        self.replacement = replacement

    def __call__(self, targets: np.ndarray):
        if self.calls == 0:
            setattr(coupling, self.target_name, self.replacement)
        return super().__call__(targets)


class _NativeMockCommitter:
    __slots__ = ("commits", "requests")

    def __init__(self) -> None:
        self.requests: list[V5H15GlobalRowCommitRequest] = []
        self.commits: list[RowCommitResult] = []

    def __call__(self, request: V5H15GlobalRowCommitRequest) -> V5H15RowCommitEnvelope:
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
            sheet_id="baik-w2-v5h15-synthetic-row",
        )
        if request.previous_owner is None:
            owner = bootstrap_release_row_owner(
                row,
                smoothing_radius_m=0.10,
                target_spacing_m=0.04,
                release_dt_s=0.11125,
                particle_cap=1000,
                owner_id=f"v5h15-native-synthetic:{next(_OWNER_INDEX)}",
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
                proposal_id=f"v5h15-native-release-{release}",
            )
            if proposal.status != "compatible":
                raise RuntimeError(
                    f"synthetic row requires remesh: {proposal.first_mismatch}"
                )
            commit = commit_release_row_update(request.previous_owner, proposal)
        self.commits.append(commit)
        kelvin = make_v5h15_source_kelvin_evidence(
            source_step_index=request.source_step_index,
            row_owner_sha256=commit.owner.owner_sha256,
            source_event_sha256=f"{request.source_step_index:064x}",
            kelvin_ledger_sha256=f"{release + 100:064x}",
            residual_m2_s=0.0,
        )
        return V5H15RowCommitEnvelope(
            commit_result=commit,
            source_kelvin_evidence=kelvin,
        )


def _native_problem() -> ps.problems.UnsteadyProblem:
    movement, metadata = build_baik_movement(BAIK_2012_CASES["W2"], "smoke")
    assert metadata["grid_chord_span"] == [2, 8]
    assert movement.delta_time == 0.11125
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _native_config() -> V5H15BaikCouplingConfig:
    return V5H15BaikCouplingConfig(
        transport_substeps=16,
        test_mode=True,
        delta_time_s=0.11125,
        active_layer_limit=1,
    )


def _owner():
    y = np.array((-0.15, 0.0, 0.15), dtype=np.float64)
    upstream = np.column_stack((np.zeros(3), y, np.zeros(3)))
    downstream = upstream.copy()
    downstream[:, 0] = 0.12
    row = make_release_row(
        upstream,
        downstream,
        np.array((0.025, 0.031), dtype=np.float64),
        release_index=1,
        source_time_s=1.0,
        sheet_id="v5h15-synthetic-sheet",
    )
    return bootstrap_release_row_owner(
        row,
        smoothing_radius_m=0.05,
        target_spacing_m=0.02,
        release_dt_s=_DT,
        particle_cap=10_000,
        owner_id=f"v5h15-synthetic:{next(_OWNER_INDEX)}",
    )


def _load_ledger(*, forces_source: np.ndarray | None = None):
    panel_forces = (
        np.array(((1.0, 0.1, -0.2), (2.0, -0.1, 0.3)), dtype=np.float64)
        if forces_source is None
        else forces_source
    )
    panel_moments = np.array(((0.1, 0.2, 0.3), (0.4, -0.2, 0.1)), dtype=np.float64)
    return make_v5h15_layer_load_ledger(
        ptera_step_index=3,
        parent_state_sha256=_PARENT_SHA,
        reference_chord_m=1.0,
        panel_ids=("panel:0", "panel:1"),
        forces_w=np.sum(panel_forces, axis=0),
        force_coefficients_w=np.array((0.2, 0.0, 0.1)),
        moments_w_cgp1=np.sum(panel_moments, axis=0),
        moment_coefficients_w=np.array((0.01, 0.02, 0.03)),
        panel_forces_w=panel_forces,
        panel_moments_w_cgp1=panel_moments,
        no_penetration_residual=np.zeros(2),
    )


def _config(n: int = 1) -> V5H15BaikCouplingConfig:
    return V5H15BaikCouplingConfig(
        transport_substeps=n,
        test_mode=True,
        delta_time_s=_DT,
    )


def _kelvin(owner, *, residual: float = 0.0):
    return make_v5h15_source_kelvin_evidence(
        source_step_index=4,
        row_owner_sha256=owner.owner_sha256,
        source_event_sha256="2" * 64,
        kelvin_ledger_sha256="3" * 64,
        residual_m2_s=residual,
    )


def test_config_separates_formal_matrix_diagnostic_smoke_and_synthetic_small_n() -> (
    None
):
    with pytest.raises(ValueError, match="exactly one"):
        V5H15BaikCouplingConfig(transport_substeps=64)
    with pytest.raises(ValueError, match="exactly one"):
        V5H15BaikCouplingConfig(
            transport_substeps=32,
            formal_matrix=True,
            diagnostic_smoke=True,
        )
    with pytest.raises(ValueError, match="formal B3 N"):
        V5H15BaikCouplingConfig(transport_substeps=1, formal_matrix=True)
    with pytest.raises(ValueError, match="small-test cap"):
        V5H15BaikCouplingConfig(transport_substeps=32, test_mode=True)
    with pytest.raises(ValueError, match="diagnostic smoke requires N=32"):
        V5H15BaikCouplingConfig(
            transport_substeps=64,
            diagnostic_smoke=True,
            active_layer_limit=1,
        )
    with pytest.raises(ValueError, match="diagnostic smoke requires one active layer"):
        V5H15BaikCouplingConfig(
            transport_substeps=32,
            diagnostic_smoke=True,
        )
    for n in FORMAL_TRANSPORT_SUBSTEPS:
        config = V5H15BaikCouplingConfig(
            transport_substeps=n,
            formal_matrix=True,
        )
        assert config.scope == "formal"
    assert (
        V5H15BaikCouplingConfig(transport_substeps=64, formal_matrix=True).role
        == "candidate"
    )
    diagnostic = V5H15BaikCouplingConfig(
        transport_substeps=32,
        diagnostic_smoke=True,
        active_layer_limit=1,
    )
    assert diagnostic.scope == "diagnostic_smoke"
    assert diagnostic.role == "diagnostic_smoke"
    assert _config().scope == "synthetic"
    assert _config().role == "synthetic"


def test_native_one_layer_orders_commit_before_parent_and_closes_exact_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(zip(ACTIVE_PTERA_STEPS, ACTIVE_SOURCE_STEPS, strict=True)) == (
        (3, 4),
        (4, 5),
        (5, 6),
    )
    assert ACTIVE_BIRTH_MODES == ("first", "continuous", "continuous")
    committer = _NativeMockCommitter()
    events: list[tuple[str, int]] = []
    original_parent_wake = NativePteraRVPMFeedbackSolver._calculate_wake_wing_influences

    def guarded_parent_wake(self: object) -> object:
        step = int(self._current_step)  # type: ignore[attr-defined]
        if step in ACTIVE_PTERA_STEPS:
            assert committer.requests[-1].ptera_step_index == step
            assert self._v5h15_active_owner is committer.commits[-1].owner  # type: ignore[attr-defined]
            events.append(("parent", step))
        return original_parent_wake(self)  # type: ignore[arg-type]

    original_call = committer.__call__

    class _OrderedCommitter:
        def __call__(
            self, request: V5H15GlobalRowCommitRequest
        ) -> V5H15RowCommitEnvelope:
            events.append(("commit", request.ptera_step_index))
            return original_call(request)

    monkeypatch.setattr(
        NativePteraRVPMFeedbackSolver,
        "_calculate_wake_wing_influences",
        guarded_parent_wake,
    )
    solver = make_fluxv_v5h15_baik_w2_solver(
        _native_problem(),
        row_committer=_OrderedCommitter(),
        config=_native_config(),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    assert events == [("commit", 3), ("parent", 3)]
    assert solver.v5h15_slice_complete
    assert len(solver.v5h15_layer_results) == 1
    result = solver.v5h15_layer_results[0]
    assert validate_v5h15_layer_result(result) is result
    assert result.scope == "synthetic"
    assert (result.ptera_step_index, result.source_step_index) == (3, 4)
    assert result.load_ledger.collocation_evaluation_count == 1
    assert result.load_ledger.load_batch_evaluation_count == 4
    assert result.load_ledger.native_load_call_count == 1
    assert result.load_ledger.no_penetration_max_abs <= 1.0e-12
    assert result.counters.direct_field_call_count == 96
    assert result.counters.ptera_center_call_count == 96
    assert result.counters.ptera_offset_call_count == 288
    assert result.counters.transport_stage_count == 48
    assert result.advanced_owner.state.phase == "post_transport"


def test_native_transport_failure_keeps_owner_current_and_no_success_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committer = _NativeMockCommitter()

    def fail_native_parent(self: object, targets: np.ndarray) -> object:
        del self, targets
        raise RuntimeError("synthetic native transport failure")

    monkeypatch.setattr(
        coupling._NativePteraParentVelocity,
        "__call__",
        fail_native_parent,
    )
    solver = make_fluxv_v5h15_baik_w2_solver(
        _native_problem(),
        row_committer=committer,
        config=_native_config(),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    with pytest.raises(RuntimeError, match="physical_field"):
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
    assert solver.v5h15_layer_results == []
    assert len(solver.v5h3_feedback_step_reports) == 1
    assert not solver.v5h15_slice_complete
    with pytest.raises(RuntimeError, match="poisoned"):
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )


def test_load_ledger_is_detached_readonly_and_recomputed() -> None:
    source = np.array(((1.0, 0.1, -0.2), (2.0, -0.1, 0.3)), dtype=np.float64)
    ledger = _load_ledger(forces_source=source)
    frozen = ledger.panel_forces_w.copy()
    source[:] = 99.0
    assert np.array_equal(ledger.panel_forces_w, frozen)
    assert not ledger.panel_forces_w.flags.writeable
    assert validate_v5h15_layer_load_ledger(ledger) is ledger
    with pytest.raises(ValueError, match="no-penetration"):
        validate_v5h15_layer_load_ledger(replace(ledger, no_penetration_max_abs=1.0))
    with pytest.raises(ValueError, match="native counts"):
        validate_v5h15_layer_load_ledger(replace(ledger, load_batch_evaluation_count=3))


def test_small_n_layer_closes_stream_fd_support_owner_and_scope() -> None:
    owner = _owner()
    parent = _ZeroFrozenParent()
    result = transport_v5h15_committed_layer(
        owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=parent,
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert validate_v5h15_layer_result(result) is result
    assert result.scope == "synthetic"
    assert result.substep_role == "synthetic"
    assert result.stream_result.counters.retained_stage_array_count == 0
    assert len(result.stream_result.stages) == 3
    assert result.counters.direct_field_call_count == 6
    assert result.counters.ptera_center_call_count == 6
    assert result.counters.ptera_offset_call_count == 18
    assert result.counters.transport_stage_count == 3
    assert result.counters.invariant_reference_freeze_count == 1
    assert result.counters.sigma_storage_update_count == 0
    assert result.counters.relaxation_call_count == 0
    assert result.fd_call_ledger.physical_evaluation_count == 3
    assert result.fd_call_ledger.tracer_evaluation_count == 3
    assert parent.calls == 24
    assert result.support_envelope.exact_support_match
    assert (
        result.support_envelope.transport_attestation_sha256
        == result.transport_attestation_sha256
    )
    assert result.advanced_owner.state.phase == "post_transport"
    assert np.array_equal(
        result.advanced_owner.state.positions,
        result.stream_result.final_state.positions,
    )
    with pytest.raises(ValueError, match="child digest"):
        validate_v5h15_layer_result(replace(result, fd_ledger_sha256="0" * 64))


def test_parent_failure_leaves_row_owner_unadvanced_and_returns_no_result() -> None:
    owner = _owner()
    parent = _ZeroFrozenParent(fail_on=2)
    with pytest.raises(RuntimeError, match="physical_field"):
        transport_v5h15_committed_layer(
            owner,
            config=_config(1),
            ptera_step_index=3,
            source_step_index=4,
            load_ledger=_load_ledger(),
            source_kelvin_evidence=_kelvin(owner),
            parent_velocity_evaluator=parent,
            parent_state_sha256_getter=_ParentHash(),
            parent_token=_PARENT_TOKEN,
            galilean_velocity_gp1_m_per_s=np.zeros(3),
        )
    assert validate_current_release_row_owner(owner) is owner
    assert owner.state.phase == "post_commit_pre_transport"
    assert owner.transport_events == ()
    retry = transport_v5h15_committed_layer(
        owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert validate_v5h15_layer_result(retry) is retry


def test_layer_result_copy_is_not_live_issued() -> None:
    owner = _owner()
    result = transport_v5h15_committed_layer(
        owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    with pytest.raises(ValueError, match="exact live issued"):
        validate_v5h15_layer_result(replace(result))


@pytest.mark.parametrize(
    ("target_name", "replacement"),
    (
        ("_execute_layer_result_transaction", lambda *args, **kwargs: None),
        ("V5H15LayerResult", object),
        ("_layer_payload", lambda result: {}),
    ),
)
def test_callback_runtime_rebind_stops_before_advance_and_clean_retry(
    target_name: str,
    replacement: object,
) -> None:
    owner = _owner()
    original = getattr(coupling, target_name)
    try:
        with pytest.raises(RuntimeError, match="binding drift|registry function drift"):
            transport_v5h15_committed_layer(
                owner,
                config=_config(1),
                ptera_step_index=3,
                source_step_index=4,
                load_ledger=_load_ledger(),
                source_kelvin_evidence=_kelvin(owner),
                parent_velocity_evaluator=_RuntimeRebindingParent(
                    target_name, replacement
                ),
                parent_state_sha256_getter=_ParentHash(),
                parent_token=_PARENT_TOKEN,
                galilean_velocity_gp1_m_per_s=np.zeros(3),
            )
    finally:
        setattr(coupling, target_name, original)
    assert validate_current_release_row_owner(owner) is owner
    assert owner.state.phase == "post_commit_pre_transport"
    assert owner.transport_events == ()
    retry = transport_v5h15_committed_layer(
        owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert validate_v5h15_layer_result(retry) is retry


def test_external_manual_publish_lacks_the_private_issuance_role() -> None:
    owner = _owner()
    result = transport_v5h15_committed_layer(
        owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert not hasattr(coupling, "_LAYER_RESULT_ISSUANCE_ROLE")
    assert not hasattr(coupling, "_reserve_layer_result")
    assert not hasattr(coupling, "_publish_layer_result")
    assert not hasattr(coupling, "_cancel_layer_result_reservation")
    assert not hasattr(coupling, "_transport_v5h15_committed_layer_impl")
    assert not hasattr(coupling, "_make_live_layer_result_registry")
    assert transport_v5h15_committed_layer.__defaults__ is None
    assert transport_v5h15_committed_layer.__kwdefaults__ is None
    assert transport_v5h15_committed_layer.__closure__ is None
    transaction = coupling._execute_layer_result_transaction
    transaction_freevars = transaction.__code__.co_freevars
    assert "issuance_role" not in transaction_freevars
    assert "reserve" not in transaction_freevars
    assert "publish" not in transaction_freevars
    registry_before = coupling._snapshot_live_layer_result_registry()
    with pytest.raises(AttributeError):
        getattr(coupling, "_reserve_layer_result")
    with pytest.raises(AttributeError):
        getattr(coupling, "_publish_layer_result")
    coupling._assert_live_layer_result_registry_unchanged(registry_before)
    with pytest.raises(ValueError, match="exact live issued"):
        validate_v5h15_layer_result(replace(result))
    coupling._assert_live_layer_result_registry_unchanged(registry_before)

    transaction_cells = dict(
        zip(transaction_freevars, transaction.__closure__ or (), strict=True)
    )
    registry = transaction_cells["registry"].cell_contents
    semantic_registry = transaction_cells["semantic_registry"].cell_contents
    compute_seal = transaction_cells["compute_seal"].cell_contents
    seal_cell = transaction_cells["seal"]
    original_seal = seal_cell.cell_contents
    original_registry = dict(registry)
    original_semantic_registry = dict(semantic_registry)
    forged = replace(result)
    try:
        registry[id(forged)] = (forged, forged.result_sha256)
        seal_cell.cell_contents = compute_seal()
        with pytest.raises(RuntimeError, match="semantic registry cardinality drift"):
            coupling._snapshot_live_layer_result_registry()
    finally:
        registry.clear()
        registry.update(original_registry)
        semantic_registry.clear()
        semantic_registry.update(original_semantic_registry)
        seal_cell.cell_contents = original_seal
    coupling._assert_live_layer_result_registry_unchanged(registry_before)

    retry_owner = _owner()
    retry = transport_v5h15_committed_layer(
        retry_owner,
        config=_config(1),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(retry_owner),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert validate_v5h15_layer_result(retry) is retry


def test_graded_birth_window_synthetic_layer_uses_effective_substeps() -> None:
    owner = _owner()
    parent = _ZeroFrozenParent()
    graded_config = replace(_config(5), birth_window_refinement=True)
    assert coupling.graded_substep_delta_times(1.0e-5, 5) == ((1.0e-5 / 20.0,) * 20)
    result = transport_v5h15_committed_layer(
        owner,
        config=graded_config,
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner),
        parent_velocity_evaluator=parent,
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert validate_v5h15_layer_result(result) is result
    assert result.transport_substeps == 5 - 5 + 5 * 4 == 20
    assert len(result.stream_result.stages) == 60
    assert result.counters.transport_stage_count == 60
    fine = 1.0e-5 / 20.0
    per_substep = [
        result.stream_result.stages[3 * j].substep_delta_time for j in range(20)
    ]
    assert per_substep == [fine] * 20
    assert result.stream_result.delta_time == math.fsum(per_substep)

    graded6 = replace(_config(6), birth_window_refinement=True)
    assert coupling.graded_substep_delta_times(1.0e-5, 6) == (
        (1.0e-5 / 24.0,) * 20 + (1.0e-5 / 6.0,)
    )
    owner6 = _owner()
    result6 = transport_v5h15_committed_layer(
        owner6,
        config=graded6,
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=_load_ledger(),
        source_kelvin_evidence=_kelvin(owner6),
        parent_velocity_evaluator=_ZeroFrozenParent(),
        parent_state_sha256_getter=_ParentHash(),
        parent_token=_PARENT_TOKEN,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )
    assert result6.transport_substeps == 21
    per_substep6 = [
        result6.stream_result.stages[3 * j].substep_delta_time for j in range(21)
    ]
    assert per_substep6 == [(1.0e-5 / 24.0)] * 20 + [(1.0e-5 / 6.0)]

    with pytest.raises(ValueError, match="birth-window refinement requires"):
        replace(_config(3), birth_window_refinement=True)
