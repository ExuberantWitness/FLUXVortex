"""Synthetic-only tests for the v5h13 Baik-W2 formal executor boundary."""

from __future__ import annotations

from hashlib import sha256
import math
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


_EXECUTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "forward_flight_benchmarks"
    / "fluxv_v5h13_baik_w2_executor.py"
)
_PLATFORM_PATH = _EXECUTOR_PATH.parents[1]
_FFB_PATH = _EXECUTOR_PATH.parent
_RUNNER_PATH = _FFB_PATH / "run_fluxv_v5h13_baik_w2.py"
_SPEC = importlib.util.spec_from_file_location(
    "synthetic_fluxv_v5h13_baik_w2_executor", _EXECUTOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
executor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = executor
_SPEC.loader.exec_module(executor)


_H = "0" * 64


_AUDITED_MODULE_PREFIXES = (
    "fluxvortex",
    "forward_flight_benchmarks",
    "pterasoftware",
    "ldvm_fourier",
    "flap_ldvm",
)


def _is_audited_module_name(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in _AUDITED_MODULE_PREFIXES
    )


@pytest.fixture
def pristine_audited_modules():
    """Run one executor-protocol test as if no project/Ptera module was loaded.

    Dependency-origin attestation must observe the pristine process state the
    audited formal entry would run in.  Suites that legitimately import the
    real coupling stack elsewhere in a joint pytest session must not leak into
    these tests; the executor attestation gate itself is never relaxed.
    """

    snapshot = dict(sys.modules)
    for name in tuple(sys.modules):
        if _is_audited_module_name(name):
            del sys.modules[name]
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if _is_audited_module_name(name) and name not in snapshot:
                del sys.modules[name]
        for name, module in snapshot.items():
            sys.modules[name] = module


@pytest.fixture
def restore_runtime_modules_after():
    """Drop runtime modules one real-runtime test imported, keeping the session clean."""

    snapshot = dict(sys.modules)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name not in snapshot:
                del sys.modules[name]
        for name, module in snapshot.items():
            sys.modules[name] = module


def _load_source_only_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, ModuleType]:
    package_name = "synthetic_executor_source_fixture"
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(_FFB_PATH)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.syspath_prepend(str(_PLATFORM_PATH))
    monkeypatch.delitem(sys.modules, "ldvm_fourier", raising=False)
    monkeypatch.delitem(sys.modules, "flap_ldvm", raising=False)

    loaded: dict[str, ModuleType] = {}
    for short_name in ("ldvm_uvlm_correction", "v5h_dvm_source"):
        module_name = f"{package_name}.{short_name}"
        path = _FFB_PATH / f"{short_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        loaded[short_name] = module
    runtime = SimpleNamespace(
        correction=loaded["ldvm_uvlm_correction"],
        source=loaded["v5h_dvm_source"],
    )
    return runtime, loaded["v5h_dvm_source"]


def _load_runner_protocol(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "synthetic_executor_runner_protocol"
    spec = importlib.util.spec_from_file_location(module_name, _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _Sink:
    pass


class _Stop(RuntimeError):
    def __init__(self, code: str, coordinate: object, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.coordinate = coordinate


def _leaf_maps(tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    names = set(executor.MODULE_LEAVES) | set(executor.DISTRIBUTION_LEAVES)
    for name in sorted(names):
        path = tmp_path / f"{name}.leaf"
        payload = f"synthetic leaf {name}\n".encode("ascii")
        path.write_bytes(payload)
        paths[name] = str(path.resolve())
        hashes[name] = sha256(payload).hexdigest()
    return paths, hashes


def _api(tmp_path: Path, **changes: object) -> SimpleNamespace:
    paths, hashes = _leaf_maps(tmp_path)
    values: dict[str, object] = {
        "schema_id": executor.EXECUTOR_API_SCHEMA_ID,
        "formal_levels": executor.FORMAL_LEVELS,
        "artifact_sink_type": _Sink,
        "stop_constructor": _Stop,
        "source_event_fields": executor.SOURCE_EVENT_FIELDS,
        "stage_fields": executor.STAGE_FIELDS,
        "frontier_node_ids": tuple(f"frontier-node:{index}" for index in range(9)),
        "fixed_probes_gp1_m": (
            (0.05, 0.30, 0.30),
            (0.10, 0.15, 0.40),
            (0.20, 0.45, 0.50),
        ),
        "dependency_leaf_file_path": paths,
        "dependency_leaf_file_sha256": hashes,
        "dependency_runtime_module_file_path": {},
        "dependency_runtime_module_file_sha256": {},
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _evidence(payload: bytes) -> SimpleNamespace:
    payload_sha = sha256(payload).hexdigest()
    evidence_sha = sha256(
        executor.STAGE_EVIDENCE_SCHEMA.encode("utf-8")
        + b"\0"
        + payload_sha.encode("ascii")
    ).hexdigest()
    return SimpleNamespace(
        schema=executor.STAGE_EVIDENCE_SCHEMA,
        payload=payload,
        payload_sha256=payload_sha,
        evidence_sha256=evidence_sha,
    )


def _stage_record(
    *,
    substep: int = 1,
    stage: int = 1,
    payload: bytes | None = None,
    parent_token: str | None = None,
    previous_chain_sha256: str = executor.STAGE_CHAIN_GENESIS,
    chain_sha256: str = "4" * 64,
) -> SimpleNamespace:
    normalized = float(executor.INVARIANT_NORMALIZED_GATE / 2.0)
    body = {
        "fd_physical_evaluation_sha256": "1" * 64,
        "fd_tracer_evaluation_sha256": "2" * 64,
        "h_convective_over_sigma": (0.2).hex(),
        "h_jacobian_frobenius": (0.3).hex(),
        "source_state_sha256": "3" * 64,
        "stage": stage,
        "substep": substep,
    }
    encoded = (
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
        if payload is None
        else payload
    )
    values = {
        "substep": substep,
        "stage": stage,
        "substep_delta_time": executor.graded_substep_delta_time(
            executor.FORMAL_LEVELS[0], substep
        ),
        "a": executor.RK_A[stage - 1],
        "b": executor.RK_B[stage - 1],
        "source_state_sha256": "3" * 64,
        "pre_state_sha256": "4" * 64,
        "post_state_sha256": "5" * 64,
        "tracer_pre_sha256": "6" * 64,
        "tracer_post_sha256": "7" * 64,
        "velocity_sha256": "8" * 64,
        "jacobian_sha256": "9" * 64,
        "gamma_rate_sha256": "a" * 64,
        "tracer_velocity_sha256": "b" * 64,
        "invariant_residual_sha256": "c" * 64,
        "invariant_residual_over_slog_max": normalized,
        "position_storage_pre_sha256": "d" * 64,
        "gamma_storage_pre_sha256": "e" * 64,
        "tracer_storage_pre_sha256": "f" * 64,
        "position_storage_post_sha256": "0" * 64,
        "gamma_storage_post_sha256": "1" * 64,
        "tracer_storage_post_sha256": "2" * 64,
        "record_sha256": "3" * 64,
        "previous_chain_sha256": previous_chain_sha256,
        "chain_sha256": chain_sha256,
        "evidence": _evidence(encoded),
    }
    if parent_token is not None:
        values["parent_token"] = parent_token
    return SimpleNamespace(**values)


def _layer_result() -> SimpleNamespace:
    physical = SimpleNamespace(
        evaluation_index=1,
        target_kind="physical",
        source_state_sha256="3" * 64,
        evaluation_sha256="1" * 64,
    )
    tracer = SimpleNamespace(
        evaluation_index=2,
        target_kind="tracer",
        source_state_sha256="3" * 64,
        evaluation_sha256="2" * 64,
    )
    return SimpleNamespace(
        transport_substeps=47,
        fd_call_ledger=SimpleNamespace(evaluations=(physical, tracer)),
        ptera_parent_sha256_before_transport="5" * 64,
        ptera_parent_sha256_after_transport="5" * 64,
    )


def _disposable_summary(**changes: object) -> object:
    values: dict[str, object] = {
        "schema_id": executor.DISPOSABLE_SMOKE_SUMMARY_SCHEMA_ID,
        "case_id": "W2",
        "scope": "diagnostic_smoke",
        "transport_substeps": 47,
        "active_layer_limit": 1,
        "layer": 1,
        "source_step_index": 4,
        "ptera_step_index": 3,
        "source_event_sha256": "0" * 64,
        "source_cell_manifest_sha256": "1" * 64,
        "source_prehistory_manifest_sha256": "2" * 64,
        "layer_result_sha256": "3" * 64,
        "stream_result_sha256": "4" * 64,
        "stream_stage_chain_sha256": "5" * 64,
        "fd_ledger_sha256": "6" * 64,
        "load_ledger_sha256": "7" * 64,
        "source_kelvin_evidence_sha256": "8" * 64,
        "row_owner_before_sha256": "9" * 64,
        "advanced_owner_sha256": "a" * 64,
        "advanced_state_sha256": "b" * 64,
        "ptera_parent_sha256_before": "c" * 64,
        "ptera_parent_sha256_after": "c" * 64,
        "particle_count": 24,
        "material_tracer_count": 33,
        "material_support_tracer_count": 24,
        "frontier_node_tracer_count": 9,
        "transport_stage_count": 141,
        "direct_field_call_count": 282,
        "ptera_center_call_count": 282,
        "ptera_offset_call_count": 846,
        "fd_physical_evaluation_count": 141,
        "fd_tracer_evaluation_count": 141,
        "fd_evaluator_call_count": 1128,
        "max_invariant_residual_over_slog": 1.0e-14,
        "max_h_jacobian_frobenius": 0.25,
        "max_h_convective_over_sigma": 0.125,
        "sigma_min_m": 1.0e-3,
        "sigma_max_m": 2.0e-3,
        "no_penetration_max_abs": 1.0e-14,
        "kelvin_residual_max_abs_m2_s": 1.0e-15,
        "observation_access": "none",
        "force_scoring_status": "blocked_no_gt_inner_mechanics_only",
        "artifact_persistence": "none",
        "summary_sha256": "",
    }
    values.update(changes)
    draft = executor.DisposableLayer1SmokeSummary(**values)
    return executor.DisposableLayer1SmokeSummary(
        **{
            **values,
            "summary_sha256": executor._disposable_smoke_summary_sha256(draft),
        }
    )


@pytest.mark.usefixtures("pristine_audited_modules")
def test_factory_validates_dynamic_api_without_importing_runner(tmp_path: Path) -> None:
    product = executor.build_fluxv_v5h13_w2_executor(_api(tmp_path))
    assert type(product) is executor.FormalCouplingExecutor
    product.attest_dependency_origins()
    product.attest_dependency_origins()
    source = _EXECUTOR_PATH.read_text(encoding="utf-8")
    assert "run_fluxv_v5h13_baik_w2" not in source
    with pytest.raises(executor.ExecutorContractError, match="stage schema drift"):
        executor.build_fluxv_v5h13_w2_executor(
            _api(tmp_path, stage_fields=executor.STAGE_FIELDS[:-1])
        )


@pytest.mark.usefixtures("pristine_audited_modules")
def test_executor_checks_levels_sink_and_uses_fresh_level_calls(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    runtime = SimpleNamespace()

    def runtime_loader(contract: object) -> object:
        assert contract.leaf_paths
        return runtime

    def level_runner(
        observed_runtime: object,
        contract: object,
        level: int,
        sink: object,
    ) -> None:
        assert observed_runtime is runtime
        seen.append((level, id(sink)))

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=runtime_loader,
        level_runner=level_runner,
    )
    sink = _Sink()
    product.run_formal_matrix(levels=(47, 79, 143), sink=sink)
    assert seen == [(47, id(sink)), (79, id(sink)), (143, id(sink))]
    with pytest.raises(executor.ExecutorContractError, match="exact graded level set"):
        product.run_formal_matrix(levels=(47, 79, 79), sink=sink)
    with pytest.raises(executor.ExecutorContractError, match="foreign sink"):
        product.run_formal_matrix(levels=(47, 79, 143), sink=object())


@pytest.mark.usefixtures("pristine_audited_modules")
def test_public_disposable_smoke_uses_verified_loader_and_returns_exact_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object()
    seen: list[object] = []
    expected = _disposable_summary()

    def runtime_loader(contract: object) -> object:
        assert contract.leaf_paths
        seen.append("loader")
        return runtime

    def smoke_runner(observed_runtime: object, contract: object) -> object:
        seen.append(("smoke", observed_runtime, contract))
        return expected

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=runtime_loader,
        smoke_runner=smoke_runner,
    )
    original_preflight = executor._preflight_all_leaf_bytes
    original_origins = executor._attest_loaded_runtime_modules

    def tracked_preflight(contract: object) -> None:
        seen.append("preflight")
        original_preflight(contract)

    def tracked_origins(contract: object, *, require_complete: bool) -> None:
        assert require_complete is False
        seen.append("origins")
        original_origins(contract, require_complete=require_complete)

    monkeypatch.setattr(executor, "_preflight_all_leaf_bytes", tracked_preflight)
    monkeypatch.setattr(executor, "_attest_loaded_runtime_modules", tracked_origins)
    assert product.run_disposable_n32_layer1_smoke() is expected
    assert seen[0] == "loader"
    assert seen[1][0] == "smoke" and seen[1][1] is runtime
    assert seen[2:] == []

    def failed_smoke(observed_runtime: object, contract: object) -> object:
        raise FloatingPointError("synthetic diagnostic failure")

    failed = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=runtime_loader,
        smoke_runner=failed_smoke,
    )
    seen.clear()
    with pytest.raises(FloatingPointError, match="diagnostic failure"):
        failed.run_disposable_n32_layer1_smoke()
    assert seen == ["loader"]


@pytest.mark.usefixtures("pristine_audited_modules")
def test_disposable_smoke_rejects_runner_validator_and_attester_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _disposable_summary()
    runtime = object()

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: runtime,
        smoke_runner=executor._run_disposable_n32_layer1_smoke,
    )
    monkeypatch.setattr(
        executor,
        "_run_disposable_n32_layer1_smoke",
        lambda observed_runtime, contract: expected,
    )
    with pytest.raises(executor.DependencyBindingError, match="runner was rebound"):
        product.run_disposable_n32_layer1_smoke()

    monkeypatch.undo()
    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: runtime,
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    monkeypatch.setattr(
        executor, "validate_disposable_layer1_smoke_summary", lambda value: value
    )
    with pytest.raises(executor.DependencyBindingError, match="validator was rebound"):
        product.run_disposable_n32_layer1_smoke()

    monkeypatch.undo()
    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: runtime,
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    monkeypatch.setattr(
        executor.FormalCouplingExecutor,
        "attest_dependency_origins",
        lambda self: None,
    )
    with pytest.raises(executor.DependencyBindingError, match="attester was rebound"):
        product.run_disposable_n32_layer1_smoke()


@pytest.mark.usefixtures("pristine_audited_modules")
def test_disposable_smoke_rejects_validator_code_and_hash_helper_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _disposable_summary()
    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    original_code = executor.validate_disposable_layer1_smoke_summary.__code__
    try:
        executor.validate_disposable_layer1_smoke_summary.__code__ = (
            lambda value, helper=None: value
        ).__code__
        with pytest.raises(executor.DependencyBindingError, match="summary_validator"):
            product.run_disposable_n32_layer1_smoke()
    finally:
        executor.validate_disposable_layer1_smoke_summary.__code__ = original_code

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    monkeypatch.setattr(
        executor,
        "_disposable_smoke_summary_sha256",
        lambda value: "0" * 64,
    )
    with pytest.raises(executor.DependencyBindingError, match="summary_sha256"):
        product.run_disposable_n32_layer1_smoke()


@pytest.mark.usefixtures("pristine_audited_modules")
def test_disposable_smoke_rejects_second_order_verifier_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: _disposable_summary(),
    )
    monkeypatch.setattr(
        executor.FormalCouplingExecutor,
        "_assert_smoke_callable_bindings",
        lambda self: None,
    )
    monkeypatch.setattr(executor, "_assert_callable_binding", lambda *args: None)
    monkeypatch.setattr(
        executor, "validate_disposable_layer1_smoke_summary", lambda value: value
    )
    product._smoke_runner = lambda observed_runtime, contract: sentinel
    product._smoke_validator = lambda value: value
    with pytest.raises(executor.DependencyBindingError, match="callable drift"):
        product.run_disposable_n32_layer1_smoke()

    monkeypatch.undo()
    clean = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: _disposable_summary(),
    )
    assert isinstance(
        clean.run_disposable_n32_layer1_smoke(),
        executor.DisposableLayer1SmokeSummary,
    )


@pytest.mark.usefixtures("pristine_audited_modules")
def test_disposable_smoke_rejects_verifier_code_defaults_and_loader_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _disposable_summary()

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    original_assert_code = executor._assert_callable_binding.__code__
    try:
        executor._assert_callable_binding.__code__ = (
            lambda name, value, frozen: None
        ).__code__
        with pytest.raises(executor.DependencyBindingError, match="verifier drift"):
            product.run_disposable_n32_layer1_smoke()
    finally:
        executor._assert_callable_binding.__code__ = original_assert_code

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    original_defaults = executor.validate_disposable_layer1_smoke_summary.__defaults__
    try:
        executor.validate_disposable_layer1_smoke_summary.__defaults__ = (
            lambda value: "0" * 64,
        )
        with pytest.raises(executor.DependencyBindingError, match="summary_validator"):
            product.run_disposable_n32_layer1_smoke()
    finally:
        executor.validate_disposable_layer1_smoke_summary.__defaults__ = (
            original_defaults
        )

    def hostile_loader(contract: object) -> object:
        monkeypatch.setattr(
            executor,
            "_make_disposable_layer1_smoke_summary",
            lambda **kwargs: expected,
        )
        return object()

    product = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=hostile_loader,
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    with pytest.raises(executor.DependencyBindingError, match="summary_builder"):
        product.run_disposable_n32_layer1_smoke()

    monkeypatch.undo()
    clean = executor.FormalCouplingExecutor(
        _api(tmp_path),
        runtime_loader=lambda contract: object(),
        smoke_runner=lambda observed_runtime, contract: expected,
    )
    assert clean.run_disposable_n32_layer1_smoke() is expected


def test_disposable_summary_validator_rejects_count_parent_and_hash_drift() -> None:
    summary = _disposable_summary()
    assert executor.validate_disposable_layer1_smoke_summary(summary) is summary
    for changed in (
        _disposable_summary(transport_stage_count=95),
        _disposable_summary(ptera_parent_sha256_after="d" * 64),
        _disposable_summary(max_h_convective_over_sigma=0.5000000000000001),
    ):
        with pytest.raises(executor.ExecutorContractError):
            executor.validate_disposable_layer1_smoke_summary(changed)


@pytest.mark.usefixtures("restore_runtime_modules_after")
def test_real_level_lifecycle_builds_fresh_bounded_solver_and_emits_three_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    movements: list[object] = []
    problems: list[object] = []
    committers: list[object] = []
    configs: list[dict[str, object]] = []
    solvers: list[object] = []
    emitted: list[tuple[object, object]] = []

    def build_events(runtime: object) -> object:
        value = object()
        events.append(value)
        return value

    def build_movement(pterasoftware: object) -> object:
        value = object()
        movements.append(value)
        return value

    class Problem:
        def __init__(self, *, movement: object, only_final_results: bool) -> None:
            assert movement is movements[-1]
            assert only_final_results is False
            problems.append(self)

    class Committer:
        def __init__(self, runtime: object, observed_events: object, *, level: int):
            assert observed_events is events[-1]
            self.level = level
            self.captures = tuple(
                SimpleNamespace(layer=layer, level=level) for layer in (1, 2, 3)
            )
            self.bound_solver: object | None = None
            committers.append(self)

        def bind_solver(self, solver: object) -> None:
            assert self.bound_solver is None
            self.bound_solver = solver

    class Config:
        def __init__(self, **values: object) -> None:
            configs.append(dict(values))

    class Solver:
        def __init__(self, committer: Committer) -> None:
            self.v5h13_layer_results = tuple(object() for _ in range(3))
            self.committer = committer
            self.run_kwargs: dict[str, object] | None = None
            solvers.append(self)

        def run(self, **values: object) -> None:
            self.run_kwargs = dict(values)

    def make_solver(
        problem: object,
        *,
        row_committer: Committer,
        config: Config,
        **values: object,
    ) -> Solver:
        assert problem is problems[-1]
        assert config is not None
        assert values == {
            "max_particles": executor.ROW_PARTICLE_CAP,
            "stretch": False,
            "free_wake": False,
        }
        return Solver(row_committer)

    monkeypatch.setattr(executor, "build_w2_source_events", build_events)
    monkeypatch.setattr(executor, "build_w2_movement", build_movement)
    monkeypatch.setattr(executor, "BaikW2RowCommitter", Committer)
    monkeypatch.setattr(
        executor,
        "emit_completed_layer",
        lambda **values: emitted.append((values["result"], values["capture"])),
    )
    coupling = SimpleNamespace(
        V5H13BaikCouplingConfig=Config,
        make_fluxv_v5h13_baik_w2_solver=make_solver,
        validate_v5h13_layer_result=lambda value: value,
    )
    runtime = SimpleNamespace(
        coupling=coupling,
        pterasoftware=SimpleNamespace(
            problems=SimpleNamespace(UnsteadyProblem=Problem)
        ),
        reference=SimpleNamespace(direct_gaussian_erf_velocity_jacobian=object()),
    )
    contract = executor._validated_api(_api(tmp_path))
    sink = object()

    for level in (47, 79):
        executor._run_formal_level(runtime, contract, level, sink)

    assert len(events) == len(movements) == len(problems) == len(committers) == 2
    assert committers[0] is not committers[1]
    assert [item.level for item in committers] == [47, 79]
    assert [item.bound_solver for item in committers] == solvers
    assert configs == [
        {
            "transport_substeps": executor.nominal_transport_substeps(level),
            "formal_matrix": True,
            "test_mode": False,
            "particle_cap": executor.ROW_PARTICLE_CAP,
            "birth_window_refinement": True,
        }
        for level in (47, 79)
    ]
    assert all(
        item.run_kwargs
        == {
            "prescribed_wake": True,
            "calculate_streamlines": False,
            "show_progress": False,
        }
        for item in solvers
    )
    assert emitted == [
        (result, capture)
        for solver, committer in zip(solvers, committers, strict=True)
        for result, capture in zip(
            solver.v5h13_layer_results, committer.captures, strict=True
        )
    ]


@pytest.mark.usefixtures("restore_runtime_modules_after")
def test_disposable_lifecycle_builds_real_components_for_one_n32_layer_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    movements: list[object] = []
    problems: list[object] = []
    committers: list[object] = []
    configs: list[dict[str, object]] = []
    solvers: list[object] = []
    validated: list[object] = []

    def build_events(runtime: object) -> object:
        value = object()
        events.append(value)
        return value

    def build_movement(pterasoftware: object) -> object:
        value = object()
        movements.append(value)
        return value

    class Problem:
        def __init__(self, *, movement: object, only_final_results: bool) -> None:
            assert movement is movements[-1]
            assert only_final_results is False
            problems.append(self)

    class Committer:
        def __init__(self, runtime: object, observed_events: object, *, level: int):
            assert observed_events is events[-1]
            assert level == 47
            self.level = level
            self.captures = [
                SimpleNamespace(
                    level=47,
                    layer=1,
                    source_step_index=4,
                    ptera_step_index=3,
                    source_cell_manifest_sha256="1" * 64,
                    source_prehistory_manifest_sha256="2" * 64,
                    committed_owner_sha256="9" * 64,
                    particle_ids=tuple(f"particle:{i}" for i in range(24)),
                    material_tracer_ids=tuple(f"material:{i}" for i in range(24)),
                    source_row=lambda: {"event_sha256": "0" * 64},
                )
            ]
            self.bound_solver: object | None = None
            committers.append(self)

        def bind_solver(self, solver: object) -> None:
            assert self.bound_solver is None
            self.bound_solver = solver

    class Config:
        def __init__(self, **values: object) -> None:
            configs.append(dict(values))

    result = SimpleNamespace(
        scope="diagnostic_smoke",
        substep_role="diagnostic_smoke",
        transport_substeps=47,
        ptera_step_index=3,
        source_step_index=4,
        row_owner_before_sha256="9" * 64,
        advanced_owner_sha256="a" * 64,
        advanced_state_sha256="b" * 64,
        result_sha256="3" * 64,
        stream_result_sha256="4" * 64,
        stream_stage_chain_sha256="5" * 64,
        fd_ledger_sha256="6" * 64,
        load_ledger_sha256="7" * 64,
        source_kelvin_evidence_sha256="8" * 64,
        ptera_parent_sha256_before_transport="c" * 64,
        ptera_parent_sha256_after_transport="c" * 64,
        observation_access="none",
        force_scoring_status="blocked_no_gt_inner_mechanics_only",
        stream_result=SimpleNamespace(
            stages=tuple(
                SimpleNamespace(invariant_residual_over_slog_max=1.0e-14)
                for _ in range(141)
            ),
            final_state=SimpleNamespace(
                positions=[[0.0, 0.0, 0.0]] * 24,
                gamma=[[0.0, 0.0, 0.0]] * 24,
                sigma=[1.0e-3] * 12 + [2.0e-3] * 12,
            ),
            final_tracer_positions=[object()] * 33,
        ),
        fd_call_ledger=SimpleNamespace(
            physical_evaluation_count=141,
            tracer_evaluation_count=141,
            center_call_count=282,
            offset_call_count=846,
            evaluator_call_count=1128,
        ),
        load_ledger=SimpleNamespace(no_penetration_max_abs=1.0e-14),
        source_kelvin_evidence=SimpleNamespace(residual_m2_s=1.0e-15),
        stability_envelope=SimpleNamespace(
            stage_count=141,
            max_h_jacobian_frobenius=0.25,
            max_h_convective_over_sigma=0.125,
        ),
        support_envelope=SimpleNamespace(support_count=24, frontier_count=9),
        counters=SimpleNamespace(
            transport_stage_count=141,
            direct_field_call_count=282,
            ptera_center_call_count=282,
            ptera_offset_call_count=846,
        ),
    )

    class Solver:
        def __init__(self, committer: Committer) -> None:
            self.v5h13_layer_results = (result,)
            self.committer = committer
            self.run_kwargs: dict[str, object] | None = None
            solvers.append(self)

        def run(self, **values: object) -> None:
            self.run_kwargs = dict(values)

    def make_solver(
        problem: object,
        *,
        row_committer: Committer,
        config: Config,
        **values: object,
    ) -> Solver:
        assert problem is problems[-1]
        assert config is not None
        assert values == {
            "max_particles": executor.ROW_PARTICLE_CAP,
            "stretch": False,
            "free_wake": False,
        }
        return Solver(row_committer)

    monkeypatch.setattr(executor, "build_w2_source_events", build_events)
    monkeypatch.setattr(executor, "build_w2_movement", build_movement)
    monkeypatch.setattr(executor, "BaikW2RowCommitter", Committer)

    def validate_result(value: object) -> object:
        validated.append(value)
        return value

    coupling = SimpleNamespace(
        V5H13BaikCouplingConfig=Config,
        make_fluxv_v5h13_baik_w2_solver=make_solver,
        validate_v5h13_layer_result=validate_result,
    )
    runtime = SimpleNamespace(
        coupling=coupling,
        pterasoftware=SimpleNamespace(
            problems=SimpleNamespace(UnsteadyProblem=Problem)
        ),
    )
    contract = executor._validated_api(_api(tmp_path))

    summary = executor._run_disposable_n32_layer1_smoke(runtime, contract)

    assert len(events) == len(movements) == len(problems) == len(committers) == 1
    assert committers[0].bound_solver is solvers[0]
    assert validated == [result]
    assert configs == [
        {
            "transport_substeps": 32,
            "formal_matrix": False,
            "test_mode": False,
            "diagnostic_smoke": True,
            "birth_window_refinement": True,
            "active_layer_limit": 1,
            "particle_cap": executor.ROW_PARTICLE_CAP,
        }
    ]
    assert solvers[0].run_kwargs == {
        "prescribed_wake": True,
        "calculate_streamlines": False,
        "show_progress": False,
    }
    assert summary.scope == "diagnostic_smoke"
    assert summary.layer_result_sha256 == result.result_sha256
    assert summary.transport_stage_count == 141
    assert summary.direct_field_call_count == 282
    assert summary.ptera_center_call_count == 282
    assert summary.ptera_offset_call_count == 846
    assert summary.artifact_persistence == "none"

    result.stream_result.final_state.positions = [[0.0, 0.0, 0.0]] * 23
    with pytest.raises(executor.ExecutorContractError, match="endpoint state"):
        executor._make_disposable_layer1_smoke_summary(
            runtime=runtime,
            contract=contract,
            result=result,
            capture=committers[0].captures[0],
        )


def test_imported_module_path_and_sha_are_both_manifest_bound(tmp_path: Path) -> None:
    contract = executor._validated_api(_api(tmp_path))
    leaf = "rvpm_ir_wrk3_v5h13_stream"
    expected = Path(contract.leaf_paths[leaf])
    module = ModuleType("synthetic_expected")
    module.__file__ = str(expected)
    executor._verify_imported_module(contract, leaf, module)

    substitute = tmp_path / "same_name_substitute" / expected.name
    substitute.parent.mkdir()
    substitute.write_bytes(expected.read_bytes())
    module.__file__ = str(substitute)
    with pytest.raises(executor.DependencyBindingError, match="path differs"):
        executor._verify_imported_module(contract, leaf, module)

    module.__file__ = str(expected)
    expected.write_text("drift\n", encoding="ascii")
    with pytest.raises(executor.DependencyBindingError, match="changed"):
        executor._verify_imported_module(contract, leaf, module)


@pytest.mark.usefixtures("pristine_audited_modules")
def test_top_level_ldvm_runtime_origins_reject_same_name_substitutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths: dict[str, str] = {}
    runtime_hashes: dict[str, str] = {}
    loaded: dict[str, ModuleType] = {}
    for module_name in ("ldvm_fourier", "flap_ldvm"):
        path = tmp_path / "audited" / f"{module_name}.py"
        path.parent.mkdir(exist_ok=True)
        payload = f"AUDITED_NAME = {module_name!r}\n".encode("ascii")
        path.write_bytes(payload)
        runtime_paths[module_name] = str(path.resolve())
        runtime_hashes[module_name] = sha256(payload).hexdigest()
        module = ModuleType(module_name)
        module.__file__ = str(path.resolve())
        monkeypatch.setitem(sys.modules, module_name, module)
        loaded[module_name] = module

    contract = executor._validated_api(
        _api(
            tmp_path,
            dependency_runtime_module_file_path=runtime_paths,
            dependency_runtime_module_file_sha256=runtime_hashes,
        )
    )
    executor._attest_loaded_runtime_modules(contract, require_complete=True)

    substitute = tmp_path / "same_name_substitute" / "flap_ldvm.py"
    substitute.parent.mkdir()
    substitute.write_bytes(Path(runtime_paths["flap_ldvm"]).read_bytes())
    loaded["flap_ldvm"].__file__ = str(substitute.resolve())
    with pytest.raises(executor.DependencyBindingError, match="origin differs"):
        executor._attest_loaded_runtime_modules(contract, require_complete=True)


@pytest.mark.usefixtures("restore_runtime_modules_after")
def test_real_source_only_prehistory_closes_runner_v3_lineage_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, source_module = _load_source_only_runtime(monkeypatch)
    runner = _load_runner_protocol(monkeypatch)
    events = executor.build_w2_source_events(runtime)
    assert len(events) == 6 and all(len(step) == 8 for step in events)
    assert source_module.SOURCE_INTERFACE_ID == executor.SOURCE_INTERFACE_ID
    assert source_module.SOURCE_PLACEMENT_SCHEMA_ID == (
        executor.SOURCE_PLACEMENT_SCHEMA_ID
    )
    assert source_module.SOURCE_BACKEND_ID == executor.SOURCE_BACKEND_ID
    assert source_module.EVENT_CHAIN_DOMAIN == executor.SOURCE_EVENT_CHAIN_DOMAIN

    prehistory_payload, prehistory_sha = executor._freeze_source_prehistory(
        events[:3], layer=1
    )
    prehistory = json.loads(prehistory_payload)
    assert [step[0]["lineage"]["source_step_index"] for step in prehistory] == [
        1,
        2,
        3,
    ]
    for layer in (2, 3):
        payload, digest = executor._freeze_source_prehistory((), layer=layer)
        assert payload == b"[]"
        assert digest == executor._payload_digest(
            executor.SOURCE_PREHISTORY_MANIFEST_DOMAIN, []
        )

    active = executor.aggregate_source_events(
        events[3],
        level=47,
        source_step_index=4,
        ptera_step_index=3,
        source_time_s=3.0 * executor.DELTA_TIME_S,
    )
    active_manifests = json.loads(active.cell_manifest_payload)
    step_one = prehistory[0][0]
    step_three = prehistory[2][0]
    step_four = active_manifests[0]
    lineage_id = step_four["lineage"]["section_lineage_id"]
    genesis = sha256(
        runner._json_bytes([runner.SOURCE_EVENT_CHAIN_DOMAIN, lineage_id])
    ).hexdigest()
    assert step_one["parent_event_manifest_sha256"] == genesis
    assert (
        step_four["parent_event_manifest_sha256"]
        == step_three["producer_manifest_sha256"]
    )
    assert step_four["parent_event_manifest_sha256"] != genesis

    particle_ids: tuple[str, ...] = ()
    material_ids: tuple[str, ...] = ()
    metadata = {
        "source_cell_manifest_sha256": active.cell_manifest_sha256,
        "source_cell_manifests": active_manifests,
        "source_prehistory_manifest_sha256": prehistory_sha,
        "source_prehistory_manifests": prehistory,
        "source_kelvin_ledger_sha256": active.kelvin_ledger_sha256,
        "source_kelvin_evidence_sha256": "a" * 64,
        "particle_id_sequence_sha256": executor._payload_digest(
            "fluxv-v5h13-particle-id-sequence-v1", list(particle_ids)
        ),
        "material_tracer_id_sequence_sha256": executor._payload_digest(
            "fluxv-v5h13-material-id-sequence-v1", list(material_ids)
        ),
        "frontier_start_positions_sha256": "b" * 64,
        "fixed_probe_contract": [list(point) for point in runner.FIXED_PROBES_GP1_M],
    }
    trajectory = {
        "layer": 1,
        "particle_ids": particle_ids,
        "material_tracer_ids": material_ids,
        "metadata": metadata,
    }
    runner._validate_trajectory_metadata(trajectory)
    runner._validate_source_manifest_binding(dict(active.row), trajectory, None)

    forged_metadata = json.loads(runner._json_bytes(metadata))
    forged_step_three = forged_metadata["source_prehistory_manifests"][2][0]
    forged_step_three["parent_event_manifest_sha256"] = genesis
    unsigned = dict(forged_step_three)
    del unsigned["producer_manifest_sha256"]
    forged_step_three["producer_manifest_sha256"] = sha256(
        runner.SOURCE_EVENT_DIGEST_PREFIX + runner._json_bytes(unsigned)
    ).hexdigest()
    forged_metadata["source_prehistory_manifest_sha256"] = runner._payload_sha256(
        executor.SOURCE_PREHISTORY_MANIFEST_DOMAIN,
        forged_metadata["source_prehistory_manifests"],
    )
    forged_trajectory = {**trajectory, "metadata": forged_metadata}
    runner._validate_trajectory_metadata(forged_trajectory)
    with pytest.raises(ValueError, match="prehistory chain|exact prehistory parent"):
        runner._validate_source_manifest_binding(
            dict(active.row), forged_trajectory, None
        )


def test_stage_conversion_binds_strict_payload_fd_pair_and_normalized_gate() -> None:
    row, compact = executor.completed_stage_row(
        _stage_record(), result=_layer_result(), layer=1
    )
    assert set(row) == set(executor.STAGE_FIELDS)
    assert row["fd_physical_evaluation_sha256"] == "1" * 64
    assert row["fd_tracer_evaluation_sha256"] == "2" * 64
    assert row["direct_field_call_count"] == 2
    assert row["ptera_center_call_count"] == 2
    assert row["ptera_offset_call_count"] == 6
    assert compact == {
        "invariant_residual_over_slog_max": (executor.INVARIANT_NORMALIZED_GATE / 2.0),
        "h_jacobian_frobenius": 0.3,
        "h_convective_over_sigma": 0.2,
    }


@pytest.mark.parametrize(
    "payload, message",
    (
        (
            b'{"fd_physical_evaluation_sha256":"'
            + b"1" * 64
            + b'","fd_physical_evaluation_sha256":"'
            + b"1" * 64
            + b'"}',
            "duplicate JSON key",
        ),
        (b'{"foreign":1}', "field set drift"),
    ),
)
def test_stage_evidence_rejects_duplicate_or_incomplete_json(
    payload: bytes, message: str
) -> None:
    with pytest.raises((ValueError, executor.ExecutorContractError), match=message):
        executor.parse_stage_evidence(_stage_record(payload=payload))


def test_stage_conversion_rejects_fd_order_or_source_drift() -> None:
    result = _layer_result()
    result.fd_call_ledger.evaluations[0].target_kind = "tracer"
    with pytest.raises(executor.ExecutorContractError, match="FD evaluation pair"):
        executor.completed_stage_row(_stage_record(), result=result, layer=1)


def test_failed_stage_row_contains_no_invented_scientific_evidence() -> None:
    cause = FloatingPointError("synthetic observer stop")
    stopped = SimpleNamespace(
        substep=2,
        stage=1,
        completed_stage_chain_sha256="a" * 64,
        original_cause=cause,
    )
    row, compact = executor.failed_stage_row(stopped, level=47, layer=1)
    assert row["status"] == "failed"
    assert row["substep"] == 2 and row["stage"] == 1
    assert row["previous_chain_sha256"] == row["chain_sha256"] == "a" * 64
    assert row["fd_physical_evaluation_sha256"] is None
    assert row["direct_field_call_count"] is None
    assert row["failure_type"] == "FloatingPointError"
    assert compact == {
        "invariant_residual_over_slog_max": None,
        "h_jacobian_frobenius": None,
        "h_convective_over_sigma": None,
    }


def test_stream_stop_routes_completed_prefix_and_failed_coordinate_separately(
    tmp_path: Path,
) -> None:
    contract = executor._validated_api(_api(tmp_path))
    parent_sha = "d" * 64
    parent_token = f"v5h13:ptera-step:4:parent:{parent_sha}"
    records: list[SimpleNamespace] = []
    previous = executor.STAGE_CHAIN_GENESIS
    for stage in (1, 2, 3):
        chain = str(stage) * 64
        records.append(
            _stage_record(
                stage=stage,
                parent_token=parent_token,
                previous_chain_sha256=previous,
                chain_sha256=chain,
            )
        )
        previous = chain
    cause = FloatingPointError("observer evidence drift")
    stopped = RuntimeError("synthetic stream stop")
    stopped.completed_stages = tuple(records)
    stopped.completed_stage_count = 3
    stopped.completed_stage_chain_sha256 = previous
    stopped.stage_began = True
    stopped.substep = 2
    stopped.stage = 1
    stopped.failure_phase = "observer"
    stopped.original_cause = cause
    source_row = {name: None for name in executor.SOURCE_EVENT_FIELDS}
    capture = SimpleNamespace(
        level=47,
        layer=2,
        source_step_index=5,
        ptera_step_index=4,
        source_row=lambda: source_row,
    )
    runtime = SimpleNamespace(
        ptera_transport=SimpleNamespace(
            ptera_parent_state_sha256=lambda solver: parent_sha
        )
    )

    class Sink:
        def __init__(self) -> None:
            self.sources: list[object] = []
            self.completed: list[tuple[object, object]] = []
            self.failed: list[object] = []

        def add_source_event(self, row: object) -> None:
            self.sources.append(row)

        def add_transport_stage_from_compact_evidence(
            self, row: object, compact: object
        ) -> None:
            assert row["status"] == "completed"
            self.completed.append((row, compact))

        def add_failed_transport_stage(self, row: object) -> None:
            assert row["status"] == "failed"
            self.failed.append(row)

    sink = Sink()
    with pytest.raises(_Stop) as raised:
        executor._emit_stream_stop(
            runtime=runtime,
            contract=contract,
            sink=sink,
            solver=object(),
            error=stopped,
            capture=capture,
        )
    assert sink.sources == [source_row]
    assert [(row["substep"], row["stage"]) for row, _ in sink.completed] == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert len(sink.failed) == 1
    assert sink.failed[0]["substep"] == 2 and sink.failed[0]["stage"] == 1
    assert sink.failed[0]["invariant_residual_over_slog_max"] is None
    assert raised.value.code == "ir_wrk3_stream_stopped"
    assert raised.value.coordinate == {
        "transport_substeps": 47,
        "layer": 2,
        "source_step_index": 5,
        "ptera_step_index": 4,
        "substep": 2,
        "stage": 1,
        "phase": "stream:observer",
        "stage_began": True,
    }


def test_generic_solver_failure_preserves_completed_layer_and_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = executor._validated_api(_api(tmp_path))
    completed_result = object()
    completed_capture = SimpleNamespace(layer=1)
    failing_source = {name: None for name in executor.SOURCE_EVENT_FIELDS}
    failing_source.update(
        {
            "transport_substeps": 47,
            "source_step_index": 5,
            "ptera_step_index": 4,
            "source_time_s": 0.2,
            "cell_count": 8,
            "status": "completed",
        }
    )
    failing_capture = SimpleNamespace(
        source_step_index=5,
        ptera_step_index=4,
        source_row=lambda: failing_source,
    )
    emitted: list[tuple[object, object]] = []

    def fake_emit_completed_layer(**kwargs: object) -> None:
        emitted.append((kwargs["result"], kwargs["capture"]))

    monkeypatch.setattr(executor, "emit_completed_layer", fake_emit_completed_layer)
    runtime = SimpleNamespace(
        coupling=SimpleNamespace(validate_v5h13_layer_result=lambda value: value),
        reference=SimpleNamespace(direct_gaussian_erf_velocity_jacobian=object()),
    )

    class Sink:
        def __init__(self) -> None:
            self.sources: list[object] = []

        def add_source_event(self, row: object) -> None:
            self.sources.append(row)

    sink = Sink()
    with pytest.raises(_Stop) as raised:
        executor._emit_generic_solver_stop(
            runtime=runtime,
            contract=contract,
            sink=sink,
            level=47,
            error=RuntimeError("after layer one"),
            results=(completed_result,),
            captures=(completed_capture, failing_capture),
        )
    assert emitted == [(completed_result, completed_capture)]
    assert sink.sources == [failing_source]
    assert raised.value.code == "formal_solver_stopped"
    assert raised.value.coordinate == {
        "transport_substeps": 47,
        "layer": None,
        "source_step_index": 5,
        "ptera_step_index": 4,
        "substep": None,
        "stage": None,
        "phase": "solver:non_stream_failure",
        "stage_began": False,
    }


OBSERVER_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "fd_physical_evaluation_sha256",
        "fd_tracer_evaluation_sha256",
        "h_convective_over_sigma",
        "h_jacobian_frobenius",
        "source_state_sha256",
        "stage",
        "substep",
    }
)


def _actual_coupling_result() -> object:
    """Build one real synthetic N=1 layer result from the frozen V5H13 coupling."""

    from itertools import count

    import numpy as np

    from fluxvortex.rvpm_ir_wrk3_fd_adapter import make_frozen_parent_velocity
    from forward_flight_benchmarks.fluxv_v5h10_row_owner import (
        bootstrap_release_row_owner,
        make_release_row,
    )
    from forward_flight_benchmarks.fluxv_v5h13_baik_coupling import (
        V5H13BaikCouplingConfig,
        make_v5h13_layer_load_ledger,
        make_v5h13_source_kelvin_evidence,
        transport_v5h13_committed_layer,
    )

    owner_seed = count()
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
        sheet_id="v5h13-cross-module-sheet",
    )
    owner = bootstrap_release_row_owner(
        row,
        smoothing_radius_m=0.05,
        target_spacing_m=0.02,
        release_dt_s=1.0e-5,
        particle_cap=10_000,
        owner_id=f"v5h13-cross-module:{next(owner_seed)}",
    )
    parent_sha = "1" * 64
    parent_token = "synthetic-frozen-parent"

    def parent(targets: object) -> object:
        return make_frozen_parent_velocity(
            targets,
            np.zeros_like(targets),
            parent_token=parent_token,
            parent_state_sha256=parent_sha,
        )

    panel_forces = np.array(((1.0, 0.1, -0.2), (2.0, -0.1, 0.3)), dtype=np.float64)
    panel_moments = np.array(((0.1, 0.2, 0.3), (0.4, -0.2, 0.1)), dtype=np.float64)
    ledger = make_v5h13_layer_load_ledger(
        ptera_step_index=3,
        parent_state_sha256=parent_sha,
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
    kelvin = make_v5h13_source_kelvin_evidence(
        source_step_index=4,
        row_owner_sha256=owner.owner_sha256,
        source_event_sha256="2" * 64,
        kelvin_ledger_sha256="3" * 64,
        residual_m2_s=0.0,
    )
    return transport_v5h13_committed_layer(
        owner,
        config=V5H13BaikCouplingConfig(
            transport_substeps=1,
            test_mode=True,
            delta_time_s=1.0e-5,
        ),
        ptera_step_index=3,
        source_step_index=4,
        load_ledger=ledger,
        source_kelvin_evidence=kelvin,
        parent_velocity_evaluator=parent,
        parent_state_sha256_getter=lambda: parent_sha,
        parent_token=parent_token,
        galilean_velocity_gp1_m_per_s=np.zeros(3),
    )


@pytest.mark.usefixtures("restore_runtime_modules_after")
def test_actual_coupling_record_carries_exact_seven_observer_fields_and_parses() -> (
    None
):
    result = _actual_coupling_result()
    from forward_flight_benchmarks.fluxv_v5h13_baik_coupling import (
        validate_v5h13_layer_result,
    )
    from fluxvortex.rvpm_ir_wrk3_v5h13_stream import validate_ir_wrk3_stream_result

    assert validate_v5h13_layer_result(result) is result
    stream = result.stream_result
    assert validate_ir_wrk3_stream_result(stream) is stream
    record = stream.stages[0]
    evidence = record.evidence
    assert evidence.schema == executor.STAGE_EVIDENCE_SCHEMA
    decoded = json.loads(evidence.payload.decode("ascii"))
    assert set(decoded) == OBSERVER_PAYLOAD_FIELDS
    normalized = record.invariant_residual_over_slog_max
    assert type(normalized) is float
    parsed = executor.parse_stage_evidence(record)
    assert parsed["invariant_residual_over_slog_max"] is normalized
    assert (
        parsed["fd_physical_evaluation_sha256"]
        == decoded["fd_physical_evaluation_sha256"]
    )


def _full_level_records() -> tuple[object, ...]:
    return tuple(
        _stage_record(
            substep=substep, stage=stage, chain_sha256=f"{substep:03d}{stage}"
        )
        for substep in range(1, 48)
        for stage in (1, 2, 3)
    )


def _emit_fake_result(records: tuple[object, ...]) -> SimpleNamespace:
    evaluations = tuple(
        SimpleNamespace(
            evaluation_index=index + 1,
            target_kind="physical" if index % 2 == 0 else "tracer",
            source_state_sha256="3" * 64,
            evaluation_sha256="1" * 64 if index % 2 == 0 else "2" * 64,
        )
        for index in range(2 * len(records))
    )
    return SimpleNamespace(
        transport_substeps=47,
        source_step_index=4,
        ptera_step_index=3,
        row_owner_before_sha256="6" * 64,
        row_state_before_sha256="7" * 64,
        stream_result=SimpleNamespace(stages=tuple(records)),
        fd_call_ledger=SimpleNamespace(evaluations=evaluations),
        ptera_parent_sha256_before_transport="5" * 64,
        ptera_parent_sha256_after_transport="5" * 64,
    )


def _emit_capture() -> SimpleNamespace:
    source_row = {"synthetic": "source"}
    return SimpleNamespace(
        level=47,
        layer=1,
        source_step_index=4,
        ptera_step_index=3,
        committed_owner_sha256="6" * 64,
        committed_state_sha256="7" * 64,
        source_row=lambda: source_row,
    )


class _OrderSink:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.failed: list[object] = []

    def add_source_event(self, row: object) -> None:
        self.calls.append(("source", row))

    def add_transport_stage_from_compact_evidence(
        self, row: object, compact: object
    ) -> None:
        self.calls.append(("stage", row["substep"], row["stage"]))

    def add_failed_transport_stage(self, row: object) -> None:
        self.failed.append(row)

    def commit_completed_layer(self, **kwargs: object) -> None:
        self.calls.append(("commit",))


def _patch_emit_row_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor, "trajectory_array_record", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(executor, "raw_step_row", lambda *args, **kwargs: object())
    monkeypatch.setattr(executor, "owner_event_row", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        executor, "particle_count_row", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(executor, "raw_load_rows", lambda *args, **kwargs: ())


def test_emit_completed_layer_appends_source_before_first_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = executor._validated_api(_api(tmp_path))
    records = _full_level_records()
    capture = _emit_capture()
    sink = _OrderSink()
    _patch_emit_row_helpers(monkeypatch)
    executor.emit_completed_layer(
        sink=sink,
        result=_emit_fake_result(records),
        capture=capture,
        contract=contract,
        direct_evaluator=object(),
    )
    stage_calls = [call for call in sink.calls if call[0] == "stage"]
    assert sink.calls[0] == ("source", capture.source_row())
    assert len(stage_calls) == 141
    assert stage_calls[0] == ("stage", 1, 1)
    assert stage_calls[-1] == ("stage", 47, 3)
    assert sink.calls[-1] == ("commit",)
    assert sink.calls.index(("commit",)) > sink.calls.index(
        ("source", capture.source_row())
    )


@pytest.mark.parametrize(
    ("broken_index", "expected_stage"),
    ((0, 1), (1, 2)),
)
def test_conversion_failure_raises_exact_unbegun_next_stage_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken_index: int,
    expected_stage: int,
) -> None:
    contract = executor._validated_api(_api(tmp_path))
    broken_payload = json.dumps(
        {"tampered": True}, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    records = _full_level_records()
    broken_substep, broken_stage = divmod(broken_index, 3)
    records = tuple(
        (
            _stage_record(
                substep=record.substep,
                stage=record.stage,
                payload=broken_payload,
            )
            if (record.substep, record.stage) == (broken_substep + 1, broken_stage + 1)
            else record
        )
        for record in records
    )
    capture = _emit_capture()
    sink = _OrderSink()
    _patch_emit_row_helpers(monkeypatch)
    with pytest.raises(_Stop) as raised:
        executor.emit_completed_layer(
            sink=sink,
            result=_emit_fake_result(records),
            capture=capture,
            contract=contract,
            direct_evaluator=object(),
        )
    assert raised.value.code == "stage_evidence_conversion_error"
    assert raised.value.coordinate == {
        "transport_substeps": 47,
        "layer": 1,
        "source_step_index": 4,
        "ptera_step_index": 3,
        "substep": 1,
        "stage": expected_stage,
        "phase": "artifact_stage_conversion",
        "stage_began": False,
    }
    source_calls = [call for call in sink.calls if call[0] == "source"]
    stage_calls = [call for call in sink.calls if call[0] == "stage"]
    assert source_calls == [("source", capture.source_row())]
    assert len(stage_calls) == broken_index
    assert sink.failed == []


def _with_record_normalized(record: object, value: object) -> object:
    record.invariant_residual_over_slog_max = value
    return record


@pytest.mark.parametrize(
    "value",
    (
        None,
        0,
        float("nan"),
        float("inf"),
        -0.0 + float("-1e-300"),
        math.nextafter(float(executor.INVARIANT_NORMALIZED_GATE), math.inf),
    ),
)
def test_record_owned_normalized_invariant_rejects_hostile_values(
    value: object,
) -> None:
    record = _with_record_normalized(_stage_record(), value)
    with pytest.raises(executor.ExecutorContractError):
        executor.parse_stage_evidence(record)


def test_record_owned_normalized_invariant_accepts_exact_gate_and_is_bitwise_copied() -> (
    None
):
    gate = float(executor.INVARIANT_NORMALIZED_GATE)
    record = _with_record_normalized(_stage_record(), gate)
    parsed = executor.parse_stage_evidence(record)
    row, compact = executor.completed_stage_row(record, result=_layer_result(), layer=1)
    assert parsed["invariant_residual_over_slog_max"] is gate
    assert row["invariant_residual_over_slog_max"] is gate
    assert compact["invariant_residual_over_slog_max"] is gate


def test_extra_eighth_observer_field_is_rejected_and_payload_bytes_are_binding() -> (
    None
):
    record = _stage_record()
    body = json.loads(record.evidence.payload.decode("ascii"))
    body[
        "invariant_residual_over_slog_max"
    ] = record.invariant_residual_over_slog_max.hex()
    legacy_payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(executor.ExecutorContractError, match="field set drift"):
        executor.parse_stage_evidence(_stage_record(payload=legacy_payload))
    tampered = dict(record.evidence.__dict__)
    tampered["payload"] = legacy_payload
    tampered["payload_sha256"] = sha256(legacy_payload).hexdigest()
    tampered["evidence_sha256"] = sha256(
        executor.STAGE_EVIDENCE_SCHEMA.encode("utf-8")
        + b"\0"
        + tampered["payload_sha256"].encode("ascii")
    ).hexdigest()
    record.evidence = SimpleNamespace(**tampered)
    with pytest.raises(executor.ExecutorContractError, match="field set drift"):
        executor.parse_stage_evidence(record)


def test_source_append_failure_stops_before_any_stage_or_layer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = executor._validated_api(_api(tmp_path))
    records = _full_level_records()
    sink = _OrderSink()

    def fail_source(row: object) -> None:
        sink.calls.append(("source", row))
        raise ValueError("synthetic source append failure")

    sink.add_source_event = fail_source
    _patch_emit_row_helpers(monkeypatch)
    with pytest.raises(ValueError, match="source append failure"):
        executor.emit_completed_layer(
            sink=sink,
            result=_emit_fake_result(records),
            capture=_emit_capture(),
            contract=contract,
            direct_evaluator=object(),
        )
    assert [call[0] for call in sink.calls] == ["source"]
    assert sink.failed == []
