from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h10-runner-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h10-runner-mpl")

import numpy as np
import pytest

import forward_flight_benchmarks.fluxv_v5h10_baik_coupling as coupling_module
import forward_flight_benchmarks.fluxv_v5h10_row_owner as row_owner_module
import forward_flight_benchmarks.run_fluxv_v5h10_baik_w2 as runner_module
from forward_flight_benchmarks.baik2012 import BAIK_2012_CASES
from forward_flight_benchmarks.fluxv_v5h10_baik_coupling import (
    ACTIVE_BIRTH_MODES,
    ACTIVE_PTERA_STEPS,
    ACTIVE_SOURCE_STEPS,
)
from forward_flight_benchmarks.run_fluxv_v5h10_baik_w2 import (
    CASE_ID,
    MAX_SOURCE_KELVIN_M2_PER_S,
    ROW_SMOOTHING_RADIUS_M,
    ROW_TARGET_SPACING_M,
    SOURCE_CELL_COUNT,
    FROZEN_ACTIVE_NEW_LEV_OVER_UC,
    FROZEN_ACTIVE_PTERA_STEPS,
    FROZEN_ACTIVE_PERSISTED_LEV_OVER_UC,
    PASS_ARTIFACT_FILES,
    RawSliceStopped,
    _array_sha256,
    _assert_no_penetration,
    _assert_raw_records,
    _assert_runtime_dependency_manifest,
    _dimensional_cumulative_lev,
    _first_birth_nodes,
    _leading_edge_nodes_from_solver,
    _panel_load_arrays,
    _particle_state_sha256,
    _publish_directory_noreplace,
    _require_sha256,
    _transport_stage_evidence,
    _validate_fixed_inertial_dvm_axes,
    _write_pass_bundle,
    build_w2_source_events,
    run_raw_slice,
)


@pytest.fixture(scope="module")
def source_events():
    return build_w2_source_events()


def test_canonical_source_is_exact_six_by_eight_and_active_only_at_four_to_six(
    source_events,
) -> None:
    assert len(source_events) == 6
    assert ACTIVE_PTERA_STEPS == FROZEN_ACTIVE_PTERA_STEPS == (3, 4, 5)
    assert ACTIVE_SOURCE_STEPS == (4, 5, 6)
    assert tuple(source - 1 for source in ACTIVE_SOURCE_STEPS) == ACTIVE_PTERA_STEPS
    assert all(len(row) == SOURCE_CELL_COUNT for row in source_events)
    assert tuple(row[0].lev_birth_mode for row in source_events) == (
        "none",
        "none",
        "none",
        *ACTIVE_BIRTH_MODES,
    )
    scale = BAIK_2012_CASES[CASE_ID].freestream_m_s * BAIK_2012_CASES[CASE_ID].chord_m
    for source_step, row in enumerate(source_events, start=1):
        for event in row:
            assert event.lineage.source_step_index == source_step
            assert abs(event.kelvin_residual_over_u_c) * scale <= (
                MAX_SOURCE_KELVIN_M2_PER_S
            )
            assert event.provenance.observation_access == "none"


def test_cumulative_persisted_lev_circulation_has_frozen_lifecycle(
    source_events,
) -> None:
    active = tuple(
        _dimensional_cumulative_lev(source_events[index]) for index in (3, 4, 5)
    )
    assert all(values.shape == (8,) for values in active)
    assert all(np.all(np.signbit(values)) for values in active)
    assert np.all(np.abs(active[1]) > np.abs(active[0]))
    assert np.all(np.abs(active[2]) > np.abs(active[1]))
    assert ROW_SMOOTHING_RADIUS_M == 0.02 * BAIK_2012_CASES[CASE_ID].chord_m
    assert ROW_TARGET_SPACING_M == ROW_SMOOTHING_RADIUS_M / 2.125
    for active_index, row_index in enumerate((3, 4, 5)):
        assert all(
            event.gamma_lev_new_over_u_c == FROZEN_ACTIVE_NEW_LEV_OVER_UC[active_index]
            for event in source_events[row_index]
        )
        assert all(
            event.kelvin_ledger is not None
            and event.kelvin_ledger.gamma_lev_persisted_after
            == FROZEN_ACTIVE_PERSISTED_LEV_OVER_UC[active_index]
            for event in source_events[row_index]
        )


def _fake_solver(*, broken_anchor: bool = False):
    panels = np.empty((2, 8), dtype=object)
    y = np.linspace(0.0, 0.6, 9)
    for chord in range(2):
        for span in range(8):
            left = np.array((0.01 * chord, y[span], 0.02))
            right = np.array((0.01 * chord, y[span + 1], 0.02))
            if broken_anchor and chord == 0 and span == 3:
                left[1] += 1.0e-12
            panels[chord, span] = SimpleNamespace(
                is_leading_edge=(chord == 0),
                is_trailing_edge=(chord == 1),
                Flpp_GP1_CgP1=left,
                Frpp_GP1_CgP1=right,
                Blpp_GP1_CgP1=left + np.array((0.066, 0.0, 0.0)),
                Brpp_GP1_CgP1=right + np.array((0.066, 0.0, 0.0)),
            )
    wing = SimpleNamespace(panels=panels)
    return SimpleNamespace(current_airplanes=(SimpleNamespace(wings=(wing,)),))


def test_first_row_uses_closed_live_ptera_anchors_and_fixed_inertial_birth_map(
    source_events,
) -> None:
    leading = _leading_edge_nodes_from_solver(_fake_solver())
    downstream = _first_birth_nodes(leading, source_events[3])
    assert leading.shape == downstream.shape == (9, 3)
    offsets = downstream - leading
    assert np.allclose(offsets, offsets[:1], rtol=0.0, atol=1.0e-18)
    assert np.all(offsets[:, 1] == 0.0)
    assert np.all(np.linalg.norm(offsets, axis=1) > 0.0)


def test_nonclosing_ptera_leading_edge_is_a_hard_stop() -> None:
    with pytest.raises(RuntimeError, match="anchors do not close"):
        _leading_edge_nodes_from_solver(_fake_solver(broken_anchor=True))


def test_fixed_inertial_axes_match_dvm_and_reject_a_second_chord_rotation(
    source_events,
) -> None:
    events = source_events[3]
    case = BAIK_2012_CASES[CASE_ID]
    lev = events[0].lev_placement.edge_anchor_position_over_chord_backend_world
    tev = events[0].tev_placement.edge_anchor_position_over_chord_backend_world
    assert lev is not None and tev is not None
    chord_vector = case.chord_m * np.array((tev[0] - lev[0], 0.0, tev[1] - lev[1]))
    leading = np.column_stack(
        (np.full(9, 0.3), np.linspace(0.0, 0.6, 9), np.full(9, -0.2))
    )
    trailing = leading + chord_vector
    _validate_fixed_inertial_dvm_axes(leading, trailing, events)

    angle = np.deg2rad(7.0)
    double_rotated = np.array(
        (
            chord_vector[0] * np.cos(angle) + chord_vector[2] * np.sin(angle),
            0.0,
            -chord_vector[0] * np.sin(angle) + chord_vector[2] * np.cos(angle),
        )
    )
    with pytest.raises(RuntimeError, match="fixed inertial DVM axes disagree"):
        _validate_fixed_inertial_dvm_axes(leading, leading + double_rotated, events)


def test_unfrozen_dependencies_are_deterministically_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "FROZEN_ROW_OWNER_SHA256", None)
    monkeypatch.setattr(runner_module, "FROZEN_COUPLING_SHA256", None)
    with pytest.raises(RuntimeError, match="not audit-frozen"):
        _assert_runtime_dependency_manifest()


@dataclass(frozen=True)
class _FakeCommitEvent:
    changed_indices: tuple[int, ...]
    event_sha256: str


@dataclass(frozen=True)
class _FakeTransportEvent:
    transported_state_sha256: str


def _fake_bundle_owner():
    digest = "a" * 64
    particle_ids = tuple(("particle", index) for index in range(6))
    lineage = tuple(
        SimpleNamespace(birth_release_index=1 + index // 2) for index in range(6)
    )
    state = SimpleNamespace(
        particle_ids=particle_ids,
        lineage=lineage,
        rows=tuple(SimpleNamespace(row_sha256=digest) for _ in range(3)),
    )
    return SimpleNamespace(
        state=state,
        events=(
            _FakeCommitEvent((0,), digest),
            _FakeCommitEvent((1,), digest),
        ),
        transport_events=tuple(_FakeTransportEvent(digest) for _ in range(3)),
    )


def _fake_transport_result(particle_count: int):
    gamma = np.full((particle_count, 3), 2.0e-8)
    sigma = np.full(particle_count, ROW_SMOOTHING_RADIUS_M)
    stages = []
    for stage_index in range(1, 4):
        pre = SimpleNamespace(
            positions=np.full((particle_count, 3), float(stage_index - 1)),
            gamma=gamma,
            sigma=sigma,
        )
        post = SimpleNamespace(
            positions=np.full((particle_count, 3), float(stage_index)),
            gamma=gamma,
            sigma=sigma,
        )
        target = pre.positions.copy()
        velocity = np.full((particle_count, 3), 0.01 * stage_index)
        jacobian = np.zeros((particle_count, 3, 3))
        stages.append(
            SimpleNamespace(
                stage=stage_index,
                pre=pre,
                post=post,
                ptera_field=SimpleNamespace(
                    target_positions_gp1_m=target,
                    velocity_gp1_m_per_s=velocity,
                    jacobian_per_s=jacobian,
                    center_call_count=1,
                    finite_difference_call_count=6,
                    target_sha256=_array_sha256(target),
                    velocity_sha256=_array_sha256(velocity),
                    jacobian_sha256=_array_sha256(jacobian),
                ),
            )
        )
    return SimpleNamespace(
        stages=tuple(stages),
        final_state=stages[-1].post,
        ptera_center_call_count=3,
        ptera_finite_difference_call_count=18,
    )


def _fake_bundle_records():
    digest = "b" * 64
    records = []
    panel_ids = tuple(
        f"airplane:0/wing:0/chord:{chord}/span:{span}"
        for chord in range(2)
        for span in range(8)
    )
    panel_forces = np.arange(48, dtype=float).reshape(16, 3) * 1.0e-6
    panel_moments = panel_forces * 0.01
    for ptera_step, source_step, mode in zip(
        ACTIVE_PTERA_STEPS, ACTIVE_SOURCE_STEPS, ACTIVE_BIRTH_MODES, strict=True
    ):
        particle_count = 100 + ptera_step
        coefficients = np.array((0.2, 0.0, -0.7))
        force_atol = float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(panel_forces))))
        )
        moment_atol = float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(panel_moments))))
        )
        records.append(
            SimpleNamespace(
                ptera_step_index=ptera_step,
                source_step_index=source_step,
                source_time_s=ptera_step * BAIK_2012_CASES[CASE_ID].period_s / 32,
                birth_mode=mode,
                ptera_force_coefficients_w=coefficients,
                ptera_forces_w=np.sum(panel_forces, axis=0),
                ptera_moments_w_cgp1=np.sum(panel_moments, axis=0),
                ptera_panel_ids=panel_ids,
                ptera_panel_forces_w=panel_forces,
                ptera_panel_moments_w_cgp1=panel_moments,
                ptera_panel_forces_w_sha256=_array_sha256(panel_forces),
                ptera_panel_moments_w_cgp1_sha256=_array_sha256(panel_moments),
                ptera_panel_force_sum_w=np.sum(panel_forces, axis=0),
                ptera_panel_moment_sum_w_cgp1=np.sum(panel_moments, axis=0),
                ptera_panel_force_sum_max_abs_residual=0.0,
                ptera_panel_moment_sum_max_abs_residual=0.0,
                ptera_panel_force_sum_atol=force_atol,
                ptera_panel_moment_sum_atol=moment_atol,
                particle_count=particle_count,
                material_tracer_count=30,
                material_support_tracer_count=21,
                frontier_node_tracer_count=9,
                row_commit_sha256=digest,
                row_state_sha256=digest,
                transported_parent_sha256=None,
                transported_state_sha256=digest,
                transported_arrays_sha256=digest,
                transport_parent_digest=digest,
                common_transport_sha256=digest,
                common_transport_attestation_sha256=digest,
                transported_material_tracers_sha256=digest,
                advanced_owner_sha256=digest,
                feedback_report=SimpleNamespace(
                    feedback_report_sha256=digest,
                    no_penetration_max_abs=0.0,
                ),
                ptera_parent_sha256_before_transport=digest,
                ptera_parent_sha256_after_transport=digest,
                ptera_parent_sha256_after_raw_record=digest,
                collocation_evaluation_count=1,
                load_batch_evaluation_count=4,
                native_load_call_count=1,
                transport_call_count=1,
                transport_stage_count=3,
                common_transport_self_field_call_count=3,
                common_transport_ptera_center_call_count=3,
                common_transport_stage_count=3,
                transport_result=_fake_transport_result(particle_count),
                surface_load_owner="ptera-native-kj-plus-unsteady",
                source_load_owner="forbidden",
            )
        )
    return tuple(records)


def test_fake_pass_writer_publishes_exact_nine_file_closed_bundle(
    tmp_path: Path, source_events
) -> None:
    output = tmp_path / "fake-pass"
    output.mkdir()
    summary = _write_pass_bundle(
        output,
        events=source_events,
        records=_fake_bundle_records(),
        final_owner=_fake_bundle_owner(),
        movement_metadata={"grid_chord_span": [2, 8]},
        dependency_sha256={"row_owner": "c" * 64, "coupling": "d" * 64},
        run_provenance={"run_uuid": "fake", "output_path": str(output)},
    )
    assert set(path.name for path in output.iterdir()) == set(PASS_ARTIFACT_FILES)
    assert summary["status"] == "GO_v5h10_baik_w2_three_layer_raw_mechanics_only"
    assert summary["paper_accuracy_claim"] is False
    assert summary["full_cycle_complete"] is False
    assert summary["production_decision"] == "blocked"
    checksums = (output / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) == 8
    assert not any("SHA256SUMS" in line for line in checksums)
    load_lines = (output / "raw_loads.csv").read_text().splitlines()
    assert len(load_lines) == 1 + 3 * 17
    with (output / "raw_steps.csv").open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    assert len(raw_rows) == 3
    fake_records = _fake_bundle_records()
    for raw_row, record in zip(raw_rows, fake_records, strict=True):
        stage_evidence = _transport_stage_evidence(record)
        for stage in stage_evidence:
            prefix = f"stage_{stage['stage']}_"
            assert (
                raw_row[prefix + "pre_particle_state_sha256"]
                == stage["pre_particle_state_sha256"]
            )
            assert (
                raw_row[prefix + "post_particle_state_sha256"]
                == stage["post_particle_state_sha256"]
            )
            assert (
                raw_row[prefix + "ptera_target_sha256"] == stage["ptera_target_sha256"]
            )
            assert (
                raw_row[prefix + "ptera_velocity_sha256"]
                == stage["ptera_velocity_sha256"]
            )
            assert (
                raw_row[prefix + "ptera_jacobian_sha256"]
                == stage["ptera_jacobian_sha256"]
            )
            assert raw_row[prefix + "ptera_center_call_count"] == "1"
            assert raw_row[prefix + "ptera_finite_difference_call_count"] == "6"
    payload = json.loads((output / "summary.json").read_text())
    assert payload == summary


def test_existing_output_is_refused_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_raw_slice(output, invocation_argv=("runner",))
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output.iterdir()) == [sentinel]


def test_atomic_publish_rejects_racing_empty_directory_without_replacement(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".artifact.staging-race"
    staging.mkdir()
    staged_file = staging / "summary.json"
    staged_file.write_text("staged", encoding="utf-8")
    raced_output = tmp_path / "artifact"
    raced_output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish_directory_noreplace(staging, raced_output)

    assert raced_output.is_dir()
    assert list(raced_output.iterdir()) == []
    assert staged_file.read_text(encoding="utf-8") == "staged"


def test_two_concurrent_atomic_publishers_cannot_replace_each_other(
    tmp_path: Path,
) -> None:
    staging_dirs = (tmp_path / ".staging-a", tmp_path / ".staging-b")
    for marker, staging in zip(("a", "b"), staging_dirs, strict=True):
        staging.mkdir()
        (staging / "marker.txt").write_text(marker, encoding="utf-8")
    destination = tmp_path / "artifact"
    barrier = Barrier(2)

    def publish(staging: Path) -> str:
        barrier.wait()
        try:
            _publish_directory_noreplace(staging, destination)
        except FileExistsError:
            return "blocked"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(publish, staging_dirs))

    assert sorted(outcomes) == ["blocked", "published"]
    winner = (destination / "marker.txt").read_text(encoding="utf-8")
    assert winner in {"a", "b"}
    loser = staging_dirs[1] if winner == "a" else staging_dirs[0]
    assert (loser / "marker.txt").read_text(encoding="utf-8") != winner


def test_dependency_hash_stop_is_isolated_checksummed_and_never_starts_ptera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = runner_module._runtime_dependency_sha256()
    monkeypatch.setattr(runner_module, "FROZEN_ROW_OWNER_SHA256", actual["row_owner"])
    monkeypatch.setattr(runner_module, "FROZEN_COUPLING_SHA256", actual["coupling"])
    frozen_source = {
        name: actual[name] for name in runner_module.FROZEN_SOURCE_DEPENDENCY_SHA256
    }
    frozen_source["v5h_dvm_source.py"] = "0" * 64
    monkeypatch.setattr(runner_module, "FROZEN_SOURCE_DEPENDENCY_SHA256", frozen_source)

    def forbid_source_or_ptera_start():
        pytest.fail("source/Ptera execution started after dependency hash failure")

    monkeypatch.setattr(
        runner_module, "build_w2_source_events", forbid_source_or_ptera_start
    )
    output = tmp_path / "blocked"
    with pytest.raises(RawSliceStopped) as caught:
        run_raw_slice(output, invocation_argv=("runner", "--output-dir", str(output)))
    assert "source dependency differs" in str(caught.value.original_error)
    stop_dir = caught.value.stop_dir
    assert not output.exists()
    assert set(path.name for path in stop_dir.iterdir()) == {
        "summary.json",
        "run_manifest.json",
        "run.log",
        "SHA256SUMS",
    }
    assert not list(stop_dir.glob("*.csv"))
    assert json.loads((stop_dir / "summary.json").read_text())["status"] == "STOP"
    checksum_lines = (stop_dir / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 3
    assert not any("SHA256SUMS" in line for line in checksum_lines)
    for line in checksum_lines:
        digest, name = line.split("  ", maxsplit=1)
        assert digest == runner_module._sha256_file(stop_dir / name)


def _copy_namespace(value: SimpleNamespace, **updates: object) -> SimpleNamespace:
    fields = vars(value).copy()
    fields.update(updates)
    return SimpleNamespace(**fields)


def test_panel_load_revalidation_rejects_forged_panel_total_and_ledger() -> None:
    record = _fake_bundle_records()[0]
    _panel_load_arrays(record)

    forged_panel = record.ptera_panel_forces_w.copy()
    forged_panel[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="panel-sum ledger"):
        _panel_load_arrays(
            _copy_namespace(
                record,
                ptera_panel_forces_w=forged_panel,
                ptera_panel_forces_w_sha256=_array_sha256(forged_panel),
                ptera_panel_force_sum_w=np.sum(forged_panel, axis=0),
            )
        )

    forged_total = record.ptera_forces_w.copy()
    forged_total[2] += 1.0
    with pytest.raises(RuntimeError, match="panel-sum ledger"):
        _panel_load_arrays(_copy_namespace(record, ptera_forces_w=forged_total))

    with pytest.raises(RuntimeError, match="panel-sum ledger"):
        _panel_load_arrays(
            _copy_namespace(
                record,
                ptera_panel_force_sum_max_abs_residual=np.nextafter(0.0, 1.0),
            )
        )
    with pytest.raises(RuntimeError, match="panel-sum ledger"):
        _panel_load_arrays(
            _copy_namespace(
                record,
                ptera_panel_force_sum_atol=record.ptera_panel_force_sum_atol * 2.0,
            )
        )


def test_wrong_panel_hash_inf_and_nonpositive_sigma_are_hard_failures() -> None:
    record = _fake_bundle_records()[0]
    with pytest.raises(RuntimeError, match="panel-load hash"):
        _panel_load_arrays(
            _copy_namespace(record, ptera_panel_forces_w_sha256="0" * 64)
        )

    infinite_panel = record.ptera_panel_forces_w.copy()
    infinite_panel[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        _panel_load_arrays(
            _copy_namespace(
                record,
                ptera_panel_forces_w=infinite_panel,
                ptera_panel_forces_w_sha256=_array_sha256(infinite_panel),
            )
        )

    final = record.transport_result.final_state
    zero_sigma = _copy_namespace(final, sigma=np.zeros(record.particle_count))
    with pytest.raises(RuntimeError, match="non-positive sigma"):
        _particle_state_sha256(zero_sigma, label="attack")
    infinite_sigma = _copy_namespace(
        final, sigma=np.full(record.particle_count, np.inf)
    )
    with pytest.raises(RuntimeError, match="non-finite/non-positive sigma"):
        _particle_state_sha256(infinite_sigma, label="attack")


def test_stage_evidence_rejects_wrong_field_hash_inf_and_cross_stage_swap() -> None:
    record = _fake_bundle_records()[0]
    assert len(_transport_stage_evidence(record)) == 3
    stages = record.transport_result.stages

    wrong_hash_field = _copy_namespace(stages[0].ptera_field, jacobian_sha256="0" * 64)
    wrong_hash_stage = _copy_namespace(stages[0], ptera_field=wrong_hash_field)
    wrong_hash_transport = _copy_namespace(
        record.transport_result, stages=(wrong_hash_stage, *stages[1:])
    )
    with pytest.raises(RuntimeError, match="finite-difference stage"):
        _transport_stage_evidence(
            _copy_namespace(record, transport_result=wrong_hash_transport)
        )

    infinite_velocity = stages[0].ptera_field.velocity_gp1_m_per_s.copy()
    infinite_velocity[0, 0] = np.inf
    infinite_field = _copy_namespace(
        stages[0].ptera_field,
        velocity_gp1_m_per_s=infinite_velocity,
        velocity_sha256=_array_sha256(infinite_velocity),
    )
    infinite_stage = _copy_namespace(stages[0], ptera_field=infinite_field)
    infinite_transport = _copy_namespace(
        record.transport_result, stages=(infinite_stage, *stages[1:])
    )
    with pytest.raises(RuntimeError, match="finite-difference stage"):
        _transport_stage_evidence(
            _copy_namespace(record, transport_result=infinite_transport)
        )

    swapped_stage_two = _copy_namespace(stages[1], pre=stages[0].pre)
    swapped_transport = _copy_namespace(
        record.transport_result,
        stages=(stages[0], swapped_stage_two, stages[2]),
    )
    with pytest.raises(RuntimeError, match="finite-difference stage"):
        _transport_stage_evidence(
            _copy_namespace(record, transport_result=swapped_transport)
        )


def test_coupling_particle_hash_and_row_owner_transport_hash_are_distinct_domains() -> (
    None
):
    record = _fake_bundle_records()[0]
    final = record.transport_result.final_state
    coupling_digest = coupling_module._particle_state_sha256(final)
    runner_digest = _particle_state_sha256(final, label="domain regression")
    row_owner_digest = row_owner_module._digest(
        "fluxv-v5h10-transported-arrays-v1",
        final.positions,
        final.gamma,
        final.sigma,
    )

    assert runner_digest == coupling_digest
    assert row_owner_digest != coupling_digest
    stage_evidence = _transport_stage_evidence(record)
    assert stage_evidence[-1]["post_particle_state_sha256"] == coupling_digest


def test_nan_no_penetration_and_bad_hash_are_hard_failures() -> None:
    feedback = SimpleNamespace(
        no_penetration_residual=np.full(16, np.nan),
        no_penetration_max_abs=np.nan,
    )
    with pytest.raises(RuntimeError, match="no-penetration"):
        _assert_no_penetration(feedback)
    with pytest.raises(RuntimeError, match="invalid SHA-256"):
        _require_sha256("row", "not-a-hash")


def test_wrong_clock_and_foreign_raw_schema_fail_before_owner_use() -> None:
    wrong_clock = tuple(
        SimpleNamespace(
            ptera_step_index=step,
            source_step_index=source,
            birth_mode=mode,
        )
        for step, source, mode in zip(
            (4, 5, 6), ACTIVE_SOURCE_STEPS, ACTIVE_BIRTH_MODES, strict=True
        )
    )
    with pytest.raises(RuntimeError, match="Ptera step sequence"):
        _assert_raw_records(wrong_clock, SimpleNamespace(), SimpleNamespace())

    foreign = tuple(
        SimpleNamespace(
            ptera_step_index=step,
            source_step_index=source,
            birth_mode=mode,
        )
        for step, source, mode in zip(
            ACTIVE_PTERA_STEPS,
            ACTIVE_SOURCE_STEPS,
            ACTIVE_BIRTH_MODES,
            strict=True,
        )
    )
    with pytest.raises(TypeError, match="foreign schema"):
        _assert_raw_records(foreign, SimpleNamespace(), SimpleNamespace())
