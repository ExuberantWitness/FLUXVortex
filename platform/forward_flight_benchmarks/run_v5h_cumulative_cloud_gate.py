"""Run the observation-free v5h cumulative-cloud mechanical gate.

The numerical core in this runner exercises exact cumulative release ownership
only.  It does not call a Ptera solver, compute loads, inspect target data, or
produce paper scores.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import util as importlib_util
from importlib.metadata import PackageNotFoundError, version
import json
from math import ceil, fsum, radians
from numbers import Integral
import os
from pathlib import Path
import platform as runtime_platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from fluxvortex.rvpm_edge_bridge import (
    FROZEN_OVERLAP_LAMBDA,
    ShadowBridgeResult,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from forward_flight_benchmarks.run_v5h_dvm_node_fixed_core_gate import (
    _bridge_metrics,
    _cell_strength_ownership_passed,
    _geometry_nodes,
    _map_node_layer,
    _node_facts,
    _placement_cells,
    _placement_metrics,
    _ribbon_cells,
    _ribbon_half_step_matches_placement,
    _ribbon_handoff_integrity,
    _source_sets,
    _step_sources,
)
from forward_flight_benchmarks.v5h_cumulative_cloud_transport import (
    CumulativeCloudTransportReport,
    attest_cumulative_ribbon_handoff,
    materialize_cumulative_particle_state,
    transport_accumulated_particle_cloud,
    validate_cumulative_cloud_transport_report,
)
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    NodeLocalDVMPlacementAdapter,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    NodeOwnedDVMRibbonShadow,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    DVMSourceEvent,
    V5hDVMSource,
    validate_dvm_source_event,
)
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    materialize_transported_particle_state,
    transport_passive_node_frontiers,
    validate_passive_frontier_transport_report,
)


FloatArray = NDArray[np.float64]
GateGeometry = Literal["straight", "taper", "twist"]

RUN_ID = "20260815_fluxv_v5h_cumulative_cloud_gate"
RUN_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-gate-v2"
RUN_TIER = "auxiliary/dev"
RAW_REFINEMENT_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-raw-refinement-v1"
RECOMPUTED_GATE_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-recomputed-gates-v1"
SEMANTIC_MANIFEST_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-semantic-manifest-v1"
RUN_PROVENANCE_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-run-provenance-v1"
DECLARED_SOURCE_MANIFEST_SCHEMA_ID = (
    "fluxv-v5h-cumulative-cloud-declared-source-manifest-v1"
)
RESULT_MANIFEST_SCHEMA_ID = "fluxv-v5h-cumulative-cloud-result-manifest-v1"

RAW_REFINEMENT_ARTIFACT = "raw_refinement.json"
RECOMPUTED_GATE_ARTIFACT = "recomputed_gates.json"
SEMANTIC_MANIFEST_ARTIFACT = "semantic_manifest.json"
SEMANTIC_PAYLOAD_FILENAMES = frozenset(
    {
        "README.md",
        "metrics.json",
        RAW_REFINEMENT_ARTIFACT,
        RECOMPUTED_GATE_ARTIFACT,
        "source_manifest.json",
        "summary.json",
    }
)
ARTIFACT_FILENAMES = frozenset(
    {
        *SEMANTIC_PAYLOAD_FILENAMES,
        SEMANTIC_MANIFEST_ARTIFACT,
        "run.log",
        "run_manifest.json",
        "result_manifest.json",
        "SHA256SUMS",
    }
)

FORBIDDEN_TARGET_PATH_TOKENS = (
    "baik",
    "yang",
    "izraelevitz",
    "ground_truth",
    "observation",
)
TARGET_DATA_SUFFIXES = frozenset(
    {".csv", ".json", ".npy", ".npz", ".pdf", ".png", ".txt", ".xlsx"}
)
FORBIDDEN_SURFACE_KEY_TOKENS = frozenset(
    {"force", "load", "lift", "drag", "pressure", "score"}
)
REPRODUCIBILITY_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "TZ",
)
PROVENANCE_PACKAGES = (
    "fluxvortex",
    "matplotlib",
    "numba",
    "numpy",
    "pterasoftware",
    "scipy",
)
CASE_GEOMETRIES: tuple[GateGeometry, ...] = ("straight", "taper", "twist")
TARGET_SPACINGS_M = (0.04, 0.02, 0.01, 0.005)
TRANSPORT_SUBSTEPS = (1, 2, 4)
PHYSICAL_RELEASE_DT_S = 0.02
SIGMA_BIRTH_M = 0.085
FREESTREAM_GP1_M_PER_S = (2.0, 0.0, 0.0)
FIRST_ALPHA_RAD = radians(35.0)
CONTINUOUS_ALPHA_RAD = radians(55.0)
M1_CONTINUOUS_ALPHA_RAD = radians(45.0)
FIXED_PROBES_GP1_M = np.asarray(
    (
        (0.05, 0.075, 0.35),
        (0.10, 0.225, 0.45),
        (-0.05, 0.375, 0.40),
        (0.15, 0.525, 0.50),
        (0.20, 0.300, 0.65),
    ),
    dtype=np.float64,
)

MAX_PARTICLES_PER_RELEASE = 512
MAX_CUMULATIVE_PARTICLES = 1536
MAX_CONFIGURATIONS = 36
MAX_TRANSPORT_SUBSTEPS = 4
M1_SMOKE_RELEASE_COUNT = 4
M3_RELEASE_COUNT = 3

MAX_SOURCE_KELVIN_M2_PER_S = 1.0e-10
MAX_RIBBON_RESIDUAL = 1.0e-12
MAX_PARTICLE_INVARIANT_REL = 1.0e-6
MAX_FINE_TIME_RELATIVE_DIFFERENCE = 1.0e-6
MIN_TIME_REFINEMENT_RATIO = 1.5
MAX_FINE_H_RELATIVE_DIFFERENCE = 0.01
MIN_H_REFINEMENT_RATIO = 1.5
MIN_NONDEGENERATE_RELATIVE_DIFFERENCE = 1.0e-15

# This is the entire allowed dependency on the frozen v1 runner.  Every entry
# is observation-free and limited to generic geometry, source construction,
# node placement, ribbon construction, or mechanical diagnostics.
V1_PRIVATE_HELPER_WHITELIST = frozenset(
    {
        "_bridge_metrics",
        "_cell_strength_ownership_passed",
        "_geometry_nodes",
        "_map_node_layer",
        "_node_facts",
        "_placement_cells",
        "_placement_metrics",
        "_ribbon_cells",
        "_ribbon_half_step_matches_placement",
        "_ribbon_handoff_integrity",
        "_source_sets",
        "_step_sources",
    }
)


_RUNTIME_AUDIT_LOCAL = threading.local()
_RUNTIME_AUDIT_HOOK_LOCK = threading.Lock()
_RUNTIME_AUDIT_HOOK_INSTALLED = False


def _runtime_open_audit_hook(event: str, args: tuple[object, ...]) -> None:
    if event != "open" or not args:
        return
    tracker = getattr(_RUNTIME_AUDIT_LOCAL, "tracker", None)
    if tracker is not None:
        tracker.record_open(args)


def _install_runtime_open_audit_hook() -> None:
    global _RUNTIME_AUDIT_HOOK_INSTALLED
    with _RUNTIME_AUDIT_HOOK_LOCK:
        if not _RUNTIME_AUDIT_HOOK_INSTALLED:
            sys.addaudithook(_runtime_open_audit_hook)
            _RUNTIME_AUDIT_HOOK_INSTALLED = True


class _RuntimeBoundaryInstrumentation(
    AbstractContextManager["_RuntimeBoundaryInstrumentation"]
):
    """Measure target-data reads and known surface-solver entry-point calls."""

    def __init__(self) -> None:
        self._patches: list[tuple[ModuleType, str, object]] = []
        self._target_observation_paths: set[str] = set()
        self._guarded_call_counts: dict[str, int] = {}
        self._guarded_symbols: list[str] = []

    @staticmethod
    def _is_read_access(mode: object, flags: object) -> bool:
        if isinstance(mode, str):
            return "r" in mode or "+" in mode
        if isinstance(flags, int):
            return not bool(flags & os.O_WRONLY)
        return True

    def record_open(self, args: tuple[object, ...]) -> None:
        path_value = args[0]
        try:
            path = os.fsdecode(os.fspath(path_value))
        except TypeError:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        if not self._is_read_access(mode, flags):
            return
        normalized = str(Path(path).expanduser().resolve(strict=False))
        folded = normalized.casefold()
        if Path(normalized).suffix.casefold() in TARGET_DATA_SUFFIXES and any(
            token in folded for token in FORBIDDEN_TARGET_PATH_TOKENS
        ):
            self._target_observation_paths.add(normalized)

    def _guard_module_symbol(
        self,
        module: ModuleType,
        symbol: str,
        category: str,
    ) -> None:
        if not hasattr(module, symbol):
            return
        original = getattr(module, symbol)
        if not callable(original):
            return
        qualified = f"{module.__name__}.{symbol}"
        self._guarded_call_counts.setdefault(category, 0)

        def forbidden_call(*args: object, **kwargs: object) -> None:
            del args, kwargs
            self._guarded_call_counts[category] += 1
            raise RuntimeError(f"forbidden runtime boundary call: {qualified}")

        self._patches.append((module, symbol, original))
        setattr(module, symbol, forbidden_call)
        self._guarded_symbols.append(qualified)

    def __enter__(self) -> "_RuntimeBoundaryInstrumentation":
        if getattr(_RUNTIME_AUDIT_LOCAL, "tracker", None) is not None:
            raise RuntimeError("runtime boundary instrumentation cannot be nested")
        _install_runtime_open_audit_hook()
        for module_name, module in sorted(sys.modules.items()):
            if not isinstance(module, ModuleType):
                continue
            folded_module = module_name.casefold()
            target_module = (
                any(token in folded_module for token in ("ptera", "baik2012"))
                or module_name == "forward_flight_benchmarks"
            )
            if not target_module:
                continue
            for symbol in sorted(vars(module)):
                folded_symbol = symbol.casefold()
                if folded_symbol.startswith("build_"):
                    self._guard_module_symbol(module, symbol, "target_builder")
                elif folded_symbol in {"run_baik_old_fluxv", "run_model"} or (
                    "solver" in folded_symbol
                    and (
                        folded_symbol.startswith("run")
                        or folded_symbol.endswith("solver")
                    )
                ):
                    self._guard_module_symbol(module, symbol, "ptera_or_target_solver")
        _RUNTIME_AUDIT_LOCAL.tracker = self
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        _RUNTIME_AUDIT_LOCAL.tracker = None
        for module, symbol, original in reversed(self._patches):
            setattr(module, symbol, original)

    def evidence(self) -> dict[str, Any]:
        target_paths = sorted(self._target_observation_paths)
        return {
            "instrumentation": (
                "python_audit_open_events_plus_guarded_known_target_entry_points"
            ),
            "open_audit_active_during_gate": True,
            "guarded_symbols": sorted(self._guarded_symbols),
            "guarded_call_counts": dict(sorted(self._guarded_call_counts.items())),
            "target_observation_read_count": len(target_paths),
            "target_observation_read_paths": target_paths,
            "passed": bool(
                not target_paths and not any(self._guarded_call_counts.values())
            ),
        }


def _direct_forbidden_imports() -> list[str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    return sorted(
        module
        for module in modules
        if any(
            token in module.casefold()
            for token in ("baik", "yang", "izraelevitz", "ptera")
        )
    )


def _surface_quantity_key_paths(value: object, path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            tokens = set(key_text.casefold().replace("-", "_").split("_"))
            child_path = f"{path}.{key_text}"
            if tokens.intersection(FORBIDDEN_SURFACE_KEY_TOKENS):
                matches.append(child_path)
            matches.extend(_surface_quantity_key_paths(item, child_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            matches.extend(_surface_quantity_key_paths(item, f"{path}[{index}]"))
    return matches


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("strict JSON artifact must contain a top-level object")
    return value


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_payload_sha256(domain: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Pre-registered controls and bounded resource caps for M0--M3."""

    span_cells: int = 4
    target_spacings_m: tuple[float, ...] = TARGET_SPACINGS_M
    transport_substeps: tuple[int, ...] = TRANSPORT_SUBSTEPS
    m1_smoke_release_count: int = M1_SMOKE_RELEASE_COUNT
    m3_release_count: int = M3_RELEASE_COUNT
    particles_per_release_cap: int = MAX_PARTICLES_PER_RELEASE
    cumulative_particle_cap: int = MAX_CUMULATIVE_PARTICLES
    configuration_cap: int = MAX_CONFIGURATIONS
    transport_substeps_cap: int = MAX_TRANSPORT_SUBSTEPS


@dataclass(frozen=True, slots=True)
class _ConfigurationResult:
    summary: dict[str, Any]
    raw: dict[str, Any]
    frontier_minus_latest_birth: FloatArray
    fixed_probe_induced_velocity: FloatArray
    particle_count: int
    realized_spacing_max_m: float


def _validate_config(config: GateConfig) -> None:
    if type(config) is not GateConfig:
        raise ValueError("config must be the frozen GateConfig schema")
    expected = GateConfig()
    if config != expected:
        raise ValueError(
            "the cumulative-cloud grid, release counts, and resource caps are frozen"
        )
    configuration_count = (
        len(CASE_GEOMETRIES)
        * len(config.target_spacings_m)
        * len(config.transport_substeps)
    )
    if configuration_count != MAX_CONFIGURATIONS:
        raise ValueError("the frozen M3 grid must contain exactly 36 configurations")
    if max(config.transport_substeps) > config.transport_substeps_cap:
        raise ValueError("transport substeps exceed the runner cap")
    if SIGMA_BIRTH_M / max(config.target_spacings_m) < FROZEN_OVERLAP_LAMBDA:
        raise ValueError("coarsest h violates the frozen minimum overlap")


def _stable_id_key(value: object) -> tuple[str, int | str]:
    if isinstance(value, bool):
        raise ValueError("Boolean IDs are forbidden")
    if isinstance(value, Integral):
        return "integer", int(value)
    if isinstance(value, str) and value:
        return "string", value
    raise ValueError("IDs must be explicit integers or nonempty strings")


def _finite_array(name: str, value: object, *, ndim: int | None = None) -> FloatArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _array_sha256(*values: np.ndarray) -> str:
    digest = sha256(b"fluxv-v5h-cumulative-gate-array-bundle-v1\0")
    for value in values:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left_array = _finite_array("left observable", left)
    right_array = _finite_array("right observable", right)
    if left_array.shape != right_array.shape:
        return float("inf")
    scale = max(1.0e-15, float(np.linalg.norm(right_array)))
    return float(np.linalg.norm(left_array - right_array) / scale)


def _stable_vector_sum(values: np.ndarray) -> FloatArray:
    array = _finite_array("circulation vectors", values, ndim=2)
    if array.shape[1:] != (3,):
        raise ValueError("circulation vectors must have shape (n, 3)")
    return np.asarray(
        [fsum(float(item) for item in array[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _predicted_release_particle_count(graph: object, spacing_m: float) -> int:
    predicted = 0
    for edge in tuple(getattr(graph, "retained_edges")):
        start = _finite_array("edge start", edge.start_position, ndim=1)
        end = _finite_array("edge end", edge.end_position, ndim=1)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("edge endpoints must be length-3 vectors")
        length = float(np.linalg.norm(end - start))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("retained edge length must be positive and finite")
        predicted += max(1, ceil(length / spacing_m))
        if predicted > MAX_PARTICLES_PER_RELEASE:
            raise ValueError("predicted release exceeds the 512-particle cap")
    return predicted


def _step_sources_pattern(
    sources: Sequence[V5hDVMSource],
    active: Sequence[bool],
) -> tuple[DVMSourceEvent, ...]:
    if len(sources) != len(active):
        raise ValueError("source/activity pattern lengths disagree")
    events = tuple(
        source.step(FIRST_ALPHA_RAD if is_active else 0.0, 0.0, 0.0)
        for source, is_active in zip(sources, active, strict=True)
    )
    if any(validate_dvm_source_event(event) is not event for event in events):
        raise RuntimeError("pattern source attestation changed event identity")
    return events


def _source_layer_passed(
    events: Sequence[DVMSourceEvent],
    *,
    expected_step: int,
    expected_mode: str,
) -> bool:
    expected_active = expected_mode != "inactive"
    expected_source_mode = "none" if expected_mode == "inactive" else expected_mode
    return bool(
        events
        and all(
            event.lineage.source_step_index == expected_step
            and bool(event.lesp_active) is expected_active
            and event.lev_birth_mode == expected_source_mode
            and abs(float(event.kelvin_residual_over_u_c))
            * float(event.provenance.circulation_scale_u_times_c_m2_per_s)
            <= MAX_SOURCE_KELVIN_M2_PER_S
            and event.provenance.observation_access == "none"
            and event.provenance.target_case_branch == "none"
            and not event.provenance.canonical
            for event in events
        )
    )


def _report_snapshot(report: CumulativeCloudTransportReport) -> tuple[Any, ...]:
    cloud = report.transported_particle_cloud
    return (
        report.report_sha256,
        cloud.cloud_sha256,
        cloud.positions_gp1_m,
        cloud.gamma_vector_m3_per_s,
        cloud.sigma_m,
        cloud.particle_ids,
        cloud.lineage,
        cloud.release_slices,
        report.transport_trace,
    )


def _frontier_observable(
    report: CumulativeCloudTransportReport,
    ribbon: object,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    births = {
        _stable_id_key(birth.node_id): _finite_array(
            "latest birth position", birth.birth_position_gp1_m, ndim=1
        )
        for birth in getattr(ribbon, "node_births")
        if birth.active and birth.birth_position_gp1_m is not None
    }
    facts = {
        _stable_id_key(fact.node_id): _finite_array(
            "frontier position", fact.advected_position_gp1_m, ndim=1
        )
        for fact in report.facts
    }
    if set(births) != set(facts):
        raise RuntimeError("frontier facts do not exactly cover latest active births")
    keys = sorted(births)
    if not keys:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    birth_positions = np.ascontiguousarray(
        np.vstack([births[key] for key in keys]), dtype=np.float64
    )
    frontier_positions = np.ascontiguousarray(
        np.vstack([facts[key] for key in keys]), dtype=np.float64
    )
    return (
        frontier_positions - birth_positions,
        frontier_positions,
        birth_positions,
    )


def _continuous_one_third_passed(
    previous: CumulativeCloudTransportReport | None,
    placement: object,
    ribbon: object,
) -> tuple[bool, float]:
    if previous is None:
        return False, float("inf")
    facts = {_stable_id_key(fact.node_id): fact for fact in previous.facts}
    births = {
        _stable_id_key(birth.node_id): birth
        for birth in getattr(ribbon, "node_births")
        if birth.active
    }
    residuals: list[float] = []
    exact = True
    for ledger in getattr(placement, "node_ledgers"):
        key = _stable_id_key(ledger.node_id)
        fact = facts.get(key)
        birth = births.get(key)
        if fact is None or birth is None or birth.birth_position_gp1_m is None:
            return False, float("inf")
        anchor = _finite_array("continuous anchor", ledger.lev_edge_anchor_gp1_m)
        frontier = _finite_array(
            "continuous parent frontier", fact.advected_position_gp1_m
        )
        expected = anchor + (frontier - anchor) / 3.0
        actual = _finite_array("continuous birth", birth.birth_position_gp1_m)
        residuals.append(float(np.linalg.norm(actual - expected)))
        exact = bool(exact and np.array_equal(actual, expected))
    return exact and bool(residuals), max(residuals, default=float("inf"))


def _transport_invariants(
    previous_state: object | None,
    bridge: ShadowBridgeResult | None,
    transported_state: object,
) -> dict[str, Any]:
    old_gamma = (
        np.empty((0, 3), dtype=np.float64)
        if previous_state is None
        else _finite_array("previous gamma", previous_state.gamma, ndim=2)
    )
    old_sigma = (
        np.empty((0,), dtype=np.float64)
        if previous_state is None
        else _finite_array("previous sigma", previous_state.sigma, ndim=1)
    )
    new_gamma = (
        np.empty((0, 3), dtype=np.float64)
        if bridge is None
        else _finite_array("new gamma", bridge.gamma, ndim=2)
    )
    new_sigma = (
        np.empty((0,), dtype=np.float64)
        if bridge is None
        else _finite_array("new sigma", bridge.sigma, ndim=1)
    )
    initial_gamma = np.concatenate((old_gamma, new_gamma), axis=0)
    initial_sigma = np.concatenate((old_sigma, new_sigma), axis=0)
    final_positions = _finite_array(
        "transported positions", transported_state.positions, ndim=2
    )
    final_gamma = _finite_array("transported gamma", transported_state.gamma, ndim=2)
    final_sigma = _finite_array("transported sigma", transported_state.sigma, ndim=1)
    if not (
        initial_gamma.shape == final_gamma.shape
        and initial_sigma.shape == final_sigma.shape
        and final_positions.shape == final_gamma.shape
    ):
        raise RuntimeError("transport invariant arrays disagree in shape")
    initial_invariant = np.linalg.norm(initial_gamma, axis=1) * initial_sigma**2
    final_invariant = np.linalg.norm(final_gamma, axis=1) * final_sigma**2
    nonzero = initial_invariant > 0.0
    invariant_relative = float(
        np.max(
            np.abs(final_invariant[nonzero] - initial_invariant[nonzero])
            / initial_invariant[nonzero],
            initial=0.0,
        )
    )
    vector_scale = max(
        1.0e-30,
        fsum(float(np.linalg.norm(row)) for row in initial_gamma),
    )
    vector_drift_abs = float(
        np.linalg.norm(
            _stable_vector_sum(final_gamma) - _stable_vector_sum(initial_gamma)
        )
    )
    vector_drift_rel = vector_drift_abs / vector_scale
    positive_finite = bool(
        final_sigma.size
        and np.all(final_sigma > 0.0)
        and np.all(np.isfinite(final_positions))
        and np.all(np.isfinite(final_gamma))
    )
    return {
        "particle_invariant_relative_drift_max": invariant_relative,
        "global_vector_change_abs_m3_per_s": vector_drift_abs,
        "global_vector_change_relative": vector_drift_rel,
        "global_vector_change_gate_eligible": False,
        "global_vector_change_role": (
            "stretching diagnostic only; deposited edge/vector conservation is "
            "gated in edge_metrics"
        ),
        "positive_finite_sigma": positive_finite,
        "passed": bool(
            positive_finite and invariant_relative <= MAX_PARTICLE_INVARIANT_REL
        ),
    }


def _run_release_sequence(
    geometry: GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
    transport_substeps: int,
    alphas_rad: Sequence[float],
    expected_modes: Sequence[str],
    wing_id: str,
) -> _ConfigurationResult:
    """Run one fresh source/placement/ribbon stack through a release sequence."""

    if len(alphas_rad) != len(expected_modes) or not alphas_rad:
        raise ValueError(
            "release alphas and expected modes must be nonempty and aligned"
        )
    if transport_substeps < 1 or transport_substeps > MAX_TRANSPORT_SUBSTEPS:
        raise ValueError("transport substeps exceed the runner cap")
    if not np.isfinite(target_spacing_m) or target_spacing_m <= 0.0:
        raise ValueError("target spacing must be positive and finite")

    nodes = _geometry_nodes(geometry, span_cells)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
    ribbon_mapper = NodeOwnedDVMRibbonShadow(wing_id=wing_id, source_family="lev")
    previous: CumulativeCloudTransportReport | None = None
    active_release_steps: list[int] = []
    release_summaries: list[dict[str, Any]] = []
    release_raw: list[dict[str, Any]] = []
    additive_count = 0

    final_frontier_delta = np.empty((0, 3), dtype=np.float64)
    final_probe_velocity = np.empty((0, 3), dtype=np.float64)
    final_state: object | None = None
    final_report: CumulativeCloudTransportReport | None = None
    final_realized_spacing = 0.0

    for step, (alpha, expected_mode) in enumerate(
        zip(alphas_rad, expected_modes, strict=True), start=1
    ):
        source_time_s = (step - 1) * PHYSICAL_RELEASE_DT_S
        node_events = _step_sources(node_sources, float(alpha))
        cell_events = _step_sources(cell_sources, float(alpha))
        placement = _map_node_layer(
            placement_adapter,
            wing_id,
            geometry,
            nodes,
            node_events,
            cell_events,
            source_time_s=source_time_s,
        )
        previous_for_ribbon = previous if expected_mode == "continuous" else None
        ribbon = ribbon_mapper.map_step(
            _ribbon_cells(geometry, nodes, cell_events),
            placement.kinematics,
            delta_time_s=PHYSICAL_RELEASE_DT_S,
            transport_enabled=True,
            source_time_s=source_time_s,
            frontier_transport_report=previous_for_ribbon,
            node_placement_result=placement,
        )

        previous_snapshot = None if previous is None else _report_snapshot(previous)
        previous_state = (
            None
            if previous is None
            else materialize_cumulative_particle_state(previous)
        )
        previous_count = 0 if previous is None else previous.total_particle_count
        bridge: ShadowBridgeResult | None = None
        edge_metrics: dict[str, Any]
        if ribbon.edge_graph is None:
            if expected_mode != "inactive":
                raise RuntimeError("active release produced no diagnostic edge graph")
            predicted_count = 0
            edge_metrics = {
                "active": False,
                "particle_count": 0,
                "realized_spacing_max_m": None,
                "passed": True,
            }
        else:
            predicted_count = _predicted_release_particle_count(
                ribbon.edge_graph, target_spacing_m
            )
            if previous_count + predicted_count > MAX_CUMULATIVE_PARTICLES:
                raise ValueError("predicted cumulative cloud exceeds the 1536 cap")
            bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
                ribbon.edge_graph,
                smoothing_radius=SIGMA_BIRTH_M,
                target_spacing=target_spacing_m,
                step=step,
            )
            if bridge.positions.shape[0] > MAX_PARTICLES_PER_RELEASE:
                raise RuntimeError("diagnostic release exceeds the particle cap")
            edge_metrics = {
                "active": True,
                **_bridge_metrics(
                    bridge,
                    requested_spacing_m=target_spacing_m,
                    smoothing_radius_m=SIGMA_BIRTH_M,
                ),
            }
            final_realized_spacing = max(
                final_realized_spacing,
                float(edge_metrics["realized_spacing_max_m"]),
            )

        handoff = attest_cumulative_ribbon_handoff(
            ribbon_mapper,
            ribbon,
            wing_id=wing_id,
            source_time_s=source_time_s,
            previous_report=previous,
        )
        report = transport_accumulated_particle_cloud(
            handoff,
            smoothing_radius_m=SIGMA_BIRTH_M,
            deposition_target_spacing_m=target_spacing_m,
            transport_end_time_s=source_time_s + PHYSICAL_RELEASE_DT_S,
            transport_substeps=transport_substeps,
            freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
        )
        if validate_cumulative_cloud_transport_report(report) is not report:
            raise RuntimeError("cumulative report attestation changed identity")
        state = materialize_cumulative_particle_state(report)
        if report.total_particle_count > MAX_CUMULATIVE_PARTICLES:
            raise RuntimeError("materialized cloud exceeds the cumulative cap")

        active_layer = expected_mode != "inactive"
        if active_layer:
            active_release_steps.append(step)
            additive_count += predicted_count
        cloud = report.transported_particle_cloud
        prefix_identity_passed = bool(
            previous is None
            or (
                cloud.particle_ids[:previous_count]
                == previous.transported_particle_cloud.particle_ids
                and cloud.lineage[:previous_count]
                == previous.transported_particle_cloud.lineage
            )
        )
        previous_immutable_passed = bool(
            previous is None or _report_snapshot(previous) == previous_snapshot
        )
        new_release_identity_passed = True
        if bridge is not None:
            latest_slice = cloud.release_slices[-1]
            new_release_identity_passed = bool(
                tuple(bridge.particle_ids)
                == cloud.particle_ids[
                    latest_slice.start_index : latest_slice.stop_index
                ]
                and tuple(bridge.lineage)
                == cloud.lineage[latest_slice.start_index : latest_slice.stop_index]
            )
        slice_steps = tuple(item.source_step_index for item in cloud.release_slices)
        slice_ledger_passed = bool(
            slice_steps == tuple(active_release_steps)
            and tuple(item.release_index for item in cloud.release_slices)
            == tuple(range(1, len(active_release_steps) + 1))
            and sum(item.particle_count for item in cloud.release_slices)
            == report.total_particle_count
        )
        placement_metrics = _placement_metrics(placement, expected_mode=expected_mode)
        handoff_metrics = _ribbon_handoff_integrity(ribbon, placement)
        source_passed = bool(
            _source_layer_passed(
                node_events,
                expected_step=step,
                expected_mode=expected_mode,
            )
            and _source_layer_passed(
                cell_events,
                expected_step=step,
                expected_mode=expected_mode,
            )
        )
        if expected_mode in {"first", "restart"}:
            placement_birth_passed = _ribbon_half_step_matches_placement(
                ribbon, placement
            )
            continuous_passed = True
            continuous_residual = 0.0
        elif expected_mode == "continuous":
            placement_birth_passed = True
            continuous_passed, continuous_residual = _continuous_one_third_passed(
                previous, placement, ribbon
            )
        else:
            placement_birth_passed = ribbon.edge_graph is None
            continuous_passed = True
            continuous_residual = 0.0
        strength_owner_passed = bool(
            not active_layer or _cell_strength_ownership_passed(ribbon, cell_events)
        )
        diagnostics = ribbon.diagnostics
        ribbon_passed = bool(
            diagnostics.seam_count == 0
            and diagnostics.incidence_residual <= MAX_RIBBON_RESIDUAL
            and diagnostics.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
            and diagnostics.feedback_call_count == 0
            and handoff_metrics["passed"]
        )
        transport_invariants = _transport_invariants(previous_state, bridge, state)
        transport_counters = {
            "deposition_call_count": report.deposition_call_count,
            "lsrk3_call_count": report.lsrk3_call_count,
            "lsrk3_stage_count": report.lsrk3_stage_count,
            "stage_pre_field_call_count": report.stage_pre_field_call_count,
            "combined_stage_particle_counts": report.combined_stage_particle_counts,
            "sort_count": report.sort_count,
            "weld_count": report.weld_count,
            "delete_count": report.delete_count,
            "cancel_count": report.cancel_count,
            "remesh_count": report.remesh_count,
            "feedback_call_count": report.feedback_call_count,
            "parent_write_count": report.parent_write_count,
            "surface_channel_write_count": report.load_write_count,
        }
        counters_passed = bool(
            report.deposition_call_count == (1 if active_layer else 0)
            and report.lsrk3_call_count == transport_substeps
            and report.lsrk3_stage_count == 3 * transport_substeps
            and report.stage_pre_field_call_count == 3 * transport_substeps
            and all(
                count == report.total_particle_count
                for count in report.combined_stage_particle_counts
            )
            and not any(
                transport_counters[name]
                for name in (
                    "sort_count",
                    "weld_count",
                    "delete_count",
                    "cancel_count",
                    "remesh_count",
                    "feedback_call_count",
                    "parent_write_count",
                    "surface_channel_write_count",
                )
            )
        )
        frontier_delta, frontier_positions, birth_positions = _frontier_observable(
            report, ribbon
        )
        probe_velocity = direct_gaussian_erf_velocity_jacobian(
            state.positions,
            state.gamma,
            state.sigma,
            target_positions=FIXED_PROBES_GP1_M,
        ).velocity
        release_passed = bool(
            source_passed
            and placement_metrics["passed"]
            and placement_birth_passed
            and continuous_passed
            and strength_owner_passed
            and ribbon_passed
            and edge_metrics["passed"]
            and transport_invariants["passed"]
            and prefix_identity_passed
            and previous_immutable_passed
            and new_release_identity_passed
            and slice_ledger_passed
            and report.total_particle_count == additive_count
            and report.exact_append_passed
            and report.one_combined_field_passed
            and report.stage_pre_replay_passed
            and counters_passed
            and report.observation_access == "none"
            and report.target_case_branch == "none"
        )
        release_summaries.append(
            {
                "source_step_index": step,
                "source_time_s": source_time_s,
                "incidence_rad": float(alpha),
                "incidence_deg": float(np.rad2deg(alpha)),
                "expected_mode": expected_mode,
                "node_modes": [birth.mode for birth in ribbon.node_births],
                "previous_particle_count": report.previous_particle_count,
                "new_particle_count": report.new_particle_count,
                "total_particle_count": report.total_particle_count,
                "predicted_new_particle_count": report.predicted_new_particle_count,
                "active_release_steps": list(active_release_steps),
                "release_slice_steps": list(slice_steps),
                "source_passed": source_passed,
                "placement": placement_metrics,
                "placement_birth_passed": placement_birth_passed,
                "continuous_one_third_passed": continuous_passed,
                "continuous_one_third_residual_max_m": continuous_residual,
                "ribbon_handoff": handoff_metrics,
                "cell_strength_owner_passed": strength_owner_passed,
                "ribbon_passed": ribbon_passed,
                "edge_metrics": edge_metrics,
                "transport_invariants": transport_invariants,
                "prefix_identity_passed": prefix_identity_passed,
                "previous_report_immutable_passed": previous_immutable_passed,
                "new_release_identity_passed": new_release_identity_passed,
                "release_slice_ledger_passed": slice_ledger_passed,
                "report_sha256": report.report_sha256,
                "cloud_sha256": cloud.cloud_sha256,
                "edge_bridge_artifact_sha256": report.edge_bridge_artifact_sha256,
                "transport_trace_sha256": report.transport_trace_sha256,
                "transport_counters": transport_counters,
                "counters_passed": counters_passed,
                "passed": release_passed,
            }
        )
        release_raw.append(
            {
                "source_step_index": step,
                "positions_gp1_m": state.positions.copy(),
                "gamma_vector_m3_per_s": state.gamma.copy(),
                "sigma_m": state.sigma.copy(),
                "frontier_positions_gp1_m": frontier_positions.copy(),
                "latest_birth_positions_gp1_m": birth_positions.copy(),
                "frontier_minus_latest_birth_gp1_m": frontier_delta.copy(),
                "fixed_probe_induced_velocity_gp1_m_per_s": (probe_velocity.copy()),
                "frontier_fact_identity": tuple(
                    row[:-1] for row in _fact_physical_identity(report)
                ),
                "particle_ids": cloud.particle_ids,
                "lineage": cloud.lineage,
                "release_slices": cloud.release_slices,
                "cloud_sha256": cloud.cloud_sha256,
                "report_sha256": report.report_sha256,
                "parent_ribbon_digest_sha256": (report.parent_ribbon_digest_sha256),
                "current_ribbon_digest_sha256": (report.current_ribbon_digest_sha256),
                "handoff_sha256": report.handoff_sha256,
                "parent_report_sha256": report.parent_report_sha256,
                "parent_cloud_digest_before_append_sha256": (
                    report.parent_cloud_digest_before_append_sha256
                ),
                "deposited_new_release_digest_sha256": (
                    report.deposited_new_release_digest_sha256
                ),
                "appended_cloud_digest_before_transport_sha256": (
                    report.appended_cloud_digest_before_transport_sha256
                ),
                "transported_cloud_digest_after_sha256": (
                    report.transported_cloud_digest_after_sha256
                ),
                "edge_bridge_artifact_sha256": report.edge_bridge_artifact_sha256,
                "source_time_s": source_time_s,
                "transport_start_time_s": report.transport_start_time_s,
                "transport_end_time_s": report.transport_end_time_s,
                "transport_substeps": report.transport_substeps,
                "previous_particle_count": report.previous_particle_count,
                "new_particle_count": report.new_particle_count,
                "predicted_new_particle_count": report.predicted_new_particle_count,
                "total_particle_count": report.total_particle_count,
                "exact_append_passed": report.exact_append_passed,
                "one_combined_field_passed": report.one_combined_field_passed,
                "stage_pre_replay_passed": report.stage_pre_replay_passed,
                "exact_append_prefix_max_abs": report.exact_append_prefix_max_abs,
                "stage_pre_replay_max_abs": report.stage_pre_replay_max_abs,
                "transport_trace": report.transport_trace,
                "transport_trace_sha256": report.transport_trace_sha256,
                "transport_counters": transport_counters,
                "edge_metrics": edge_metrics,
                "transport_invariants": transport_invariants,
                "measured_mechanical_attestations": {
                    "source_passed": source_passed,
                    "placement_passed": placement_metrics["passed"],
                    "placement_birth_passed": placement_birth_passed,
                    "continuous_one_third_passed": continuous_passed,
                    "ribbon_handoff_passed": handoff_metrics["passed"],
                    "cell_strength_owner_passed": strength_owner_passed,
                    "ribbon_passed": ribbon_passed,
                    "prefix_identity_passed": prefix_identity_passed,
                    "previous_report_immutable_passed": previous_immutable_passed,
                    "new_release_identity_passed": new_release_identity_passed,
                    "release_slice_ledger_passed": slice_ledger_passed,
                    "report_exact_append_passed": report.exact_append_passed,
                    "report_one_combined_field_passed": (
                        report.one_combined_field_passed
                    ),
                    "report_stage_pre_replay_passed": (report.stage_pre_replay_passed),
                    "counters_passed": counters_passed,
                    "observation_access_none": report.observation_access == "none",
                    "target_case_branch_none": report.target_case_branch == "none",
                },
            }
        )
        previous = report
        final_state = state
        final_report = report
        final_frontier_delta = frontier_delta
        final_probe_velocity = probe_velocity

    if final_state is None or final_report is None:
        raise RuntimeError("release sequence produced no final state")
    final_cloud = final_report.transported_particle_cloud
    summary = {
        "geometry": geometry,
        "span_cells": span_cells,
        "target_spacing_m": target_spacing_m,
        "transport_substeps": transport_substeps,
        "smoothing_radius_m": SIGMA_BIRTH_M,
        "release_count": len(alphas_rad),
        "active_release_count": len(active_release_steps),
        "active_release_steps": active_release_steps,
        "final_particle_count": final_report.total_particle_count,
        "realized_spacing_max_m": final_realized_spacing,
        "final_cloud_sha256": final_cloud.cloud_sha256,
        "final_report_sha256": final_report.report_sha256,
        "final_state_sha256": _array_sha256(
            final_state.positions,
            final_state.gamma,
            final_state.sigma,
            final_frontier_delta,
            final_probe_velocity,
        ),
        "releases": release_summaries,
        "passed": bool(
            release_summaries and all(row["passed"] for row in release_summaries)
        ),
    }
    raw = {
        "positions_gp1_m": final_state.positions.copy(),
        "gamma_vector_m3_per_s": final_state.gamma.copy(),
        "sigma_m": final_state.sigma.copy(),
        "frontier_minus_latest_birth_gp1_m": final_frontier_delta.copy(),
        "fixed_probe_induced_velocity_gp1_m_per_s": final_probe_velocity.copy(),
        "particle_ids": final_cloud.particle_ids,
        "lineage": final_cloud.lineage,
        "release_slices": final_cloud.release_slices,
        "cloud_sha256": final_cloud.cloud_sha256,
        "report_sha256": final_report.report_sha256,
        "transport_trace": final_report.transport_trace,
        "transport_trace_sha256": final_report.transport_trace_sha256,
        "release_raw": release_raw,
    }
    return _ConfigurationResult(
        summary=summary,
        raw=raw,
        frontier_minus_latest_birth=final_frontier_delta,
        fixed_probe_induced_velocity=final_probe_velocity,
        particle_count=final_report.total_particle_count,
        realized_spacing_max_m=final_realized_spacing,
    )


def _run_configuration(
    geometry: GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
    transport_substeps: int,
) -> _ConfigurationResult:
    """Run one fresh M3 configuration with exactly three active releases."""

    return _run_release_sequence(
        geometry,
        span_cells=span_cells,
        target_spacing_m=target_spacing_m,
        transport_substeps=transport_substeps,
        alphas_rad=(FIRST_ALPHA_RAD, CONTINUOUS_ALPHA_RAD, CONTINUOUS_ALPHA_RAD),
        expected_modes=("first", "continuous", "continuous"),
        wing_id=(
            f"cumulative-m3:{geometry}:h={target_spacing_m.hex()}:"
            f"substeps={transport_substeps}:wing"
        ),
    )


def _ordered_fact_positions(report: object) -> FloatArray:
    facts = sorted(
        getattr(report, "facts"), key=lambda item: _stable_id_key(item.node_id)
    )
    return np.ascontiguousarray(
        np.asarray([fact.advected_position_gp1_m for fact in facts], dtype=np.float64)
    )


def _fact_physical_identity(report: object) -> tuple[tuple[Any, ...], ...]:
    facts = sorted(
        getattr(report, "facts"), key=lambda item: _stable_id_key(item.node_id)
    )
    return tuple(
        (
            fact.node_id,
            fact.lineage_epoch,
            fact.parent_frontier_id,
            fact.parent_frontier_digest_sha256,
            fact.parent_birth_step_index,
            fact.for_source_step_index,
            fact.advected_position_gp1_m,
        )
        for fact in facts
    )


def _run_m0_parity() -> dict[str, Any]:
    """Require cumulative v2 to reduce bitwise to passive-frontier v1."""

    geometry: GateGeometry = "straight"
    wing_id = "cumulative-m0:straight:wing"
    nodes = _geometry_nodes(geometry, 2)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
    ribbon_mapper = NodeOwnedDVMRibbonShadow(wing_id=wing_id, source_family="lev")
    node_events = _step_sources(node_sources, FIRST_ALPHA_RAD)
    cell_events = _step_sources(cell_sources, FIRST_ALPHA_RAD)
    placement = _map_node_layer(
        placement_adapter,
        wing_id,
        geometry,
        nodes,
        node_events,
        cell_events,
        source_time_s=0.0,
    )
    ribbon = ribbon_mapper.map_step(
        _ribbon_cells(geometry, nodes, cell_events),
        placement.kinematics,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
        source_time_s=0.0,
        node_placement_result=placement,
    )
    if ribbon.edge_graph is None:
        raise RuntimeError("M0 passive v1 first release produced no edge graph")
    predicted = _predicted_release_particle_count(
        ribbon.edge_graph, TARGET_SPACINGS_M[0]
    )
    v1_bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
        ribbon.edge_graph,
        smoothing_radius=SIGMA_BIRTH_M,
        target_spacing=TARGET_SPACINGS_M[0],
        step=1,
    )
    v1_report = transport_passive_node_frontiers(
        ribbon,
        v1_bridge,
        wing_id=wing_id,
        transport_start_time_s=0.0,
        transport_end_time_s=PHYSICAL_RELEASE_DT_S,
        transport_substeps=1,
        deposition_target_spacing_m=TARGET_SPACINGS_M[0],
        freestream_velocity_gp1_m_per_s=FREESTREAM_GP1_M_PER_S,
    )
    if validate_passive_frontier_transport_report(v1_report) is not v1_report:
        raise RuntimeError("M0 passive report attestation changed identity")
    v1_state = materialize_transported_particle_state(v1_report)
    v1_edge_metrics = _bridge_metrics(
        v1_bridge,
        requested_spacing_m=TARGET_SPACINGS_M[0],
        smoothing_radius_m=SIGMA_BIRTH_M,
    )

    v2 = _run_release_sequence(
        geometry,
        span_cells=2,
        target_spacing_m=TARGET_SPACINGS_M[0],
        transport_substeps=1,
        alphas_rad=(FIRST_ALPHA_RAD,),
        expected_modes=("first",),
        wing_id=wing_id,
    )
    v2_release = v2.raw["release_raw"][0]
    positions_equal = np.array_equal(v1_state.positions, v2_release["positions_gp1_m"])
    gamma_equal = np.array_equal(v1_state.gamma, v2_release["gamma_vector_m3_per_s"])
    sigma_equal = np.array_equal(v1_state.sigma, v2_release["sigma_m"])
    frontier_equal = np.array_equal(
        _ordered_fact_positions(v1_report),
        v2_release["frontier_positions_gp1_m"],
    )
    particle_ids_equal = bool(
        v1_report.transported_particle_cloud.particle_ids == v2_release["particle_ids"]
    )
    lineage_equal = bool(
        v1_report.transported_particle_cloud.lineage == v2_release["lineage"]
    )
    # Report producer IDs intentionally differ between v1 and v2.  The physical
    # fact identity fields that must reduce exactly are stored in the raw layer.
    v1_fact_identity = tuple(row[:-1] for row in _fact_physical_identity(v1_report))
    v2_fact_identity = tuple(v2_release["frontier_fact_identity"])
    fact_identity_equal = v1_fact_identity == v2_fact_identity
    physical_bitwise_parity = bool(
        positions_equal
        and gamma_equal
        and sigma_equal
        and frontier_equal
        and particle_ids_equal
        and lineage_equal
        and fact_identity_equal
    )
    return {
        "summary": {
            "geometry": geometry,
            "span_cells": 2,
            "release_count": 1,
            "predicted_particle_count": predicted,
            "positions_bitwise_equal": positions_equal,
            "gamma_bitwise_equal": gamma_equal,
            "sigma_bitwise_equal": sigma_equal,
            "frontier_bitwise_equal": frontier_equal,
            "particle_ids_equal": particle_ids_equal,
            "lineage_equal": lineage_equal,
            "frontier_fact_identity_equal": fact_identity_equal,
            "v1_edge_metrics": v1_edge_metrics,
            "v2_sequence": v2.summary,
            "passed": bool(
                physical_bitwise_parity
                and v1_edge_metrics["passed"]
                and v2.summary["passed"]
            ),
        },
        "raw_arrays": {
            "v1": {
                "positions_gp1_m": v1_state.positions.copy(),
                "gamma_vector_m3_per_s": v1_state.gamma.copy(),
                "sigma_m": v1_state.sigma.copy(),
                "frontier_positions_gp1_m": _ordered_fact_positions(v1_report),
                "particle_ids": v1_report.transported_particle_cloud.particle_ids,
                "lineage": v1_report.transported_particle_cloud.lineage,
                "report_sha256": v1_report.report_sha256,
            },
            "v2": v2.raw,
        },
    }


def _run_m1_four_release_smoke() -> dict[str, Any]:
    result = _run_release_sequence(
        "straight",
        span_cells=2,
        target_spacing_m=TARGET_SPACINGS_M[0],
        transport_substeps=1,
        alphas_rad=(
            FIRST_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
        ),
        expected_modes=("first", "continuous", "continuous", "continuous"),
        wing_id="cumulative-m1:straight:wing",
    )
    releases = result.summary["releases"]
    three_release_passed = bool(
        len(releases) >= 3
        and releases[2]["passed"]
        and releases[2]["release_slice_steps"] == [1, 2, 3]
    )
    four_release_passed = bool(
        len(releases) == M1_SMOKE_RELEASE_COUNT
        and releases[3]["passed"]
        and releases[3]["release_slice_steps"] == [1, 2, 3, 4]
    )
    return {
        "summary": {
            **result.summary,
            "three_release_blocker_closed": three_release_passed,
            "four_release_not_hard_coded": four_release_passed,
            "passed": bool(
                result.summary["passed"]
                and three_release_passed
                and four_release_passed
            ),
        },
        "raw_arrays": result.raw,
    }


def _run_partial_activity_pre_gates() -> dict[str, Any]:
    """Require underresolved split/shrink/grow boundaries to fail closed."""

    geometry: GateGeometry = "straight"
    patterns = {
        "split": ((True, True, False, True, False), (True, False, True, False)),
        "shrink": ((True, True, True, False, False), (True, True, False, False)),
        "grow": ((False, True, True, True, False), (False, True, True, False)),
    }
    rows: list[dict[str, Any]] = []
    for name, (node_activity, cell_activity) in patterns.items():
        wing_id = f"cumulative-m2-partial:{name}:wing"
        nodes = _geometry_nodes(geometry, 4)
        node_sources, cell_sources = _source_sets(geometry, nodes)
        node_events = _step_sources_pattern(node_sources, node_activity)
        cell_events = _step_sources_pattern(cell_sources, cell_activity)
        facts = _node_facts(
            nodes,
            node_events,
            wing_id=wing_id,
            source_time_s=0.0,
        )
        cells = _placement_cells(geometry, facts, cell_events)
        adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
        before = adapter.state_manifest()
        exception_type: str | None = None
        exception_message: str | None = None
        try:
            adapter.map_step(cells, delta_time_s=PHYSICAL_RELEASE_DT_S)
        except ValueError as error:
            exception_type = type(error).__name__
            exception_message = str(error)
        unchanged = adapter.state_manifest() == before
        rejected = bool(
            exception_message is not None
            and "mixed activity/mode" in exception_message
            and unchanged
        )
        rows.append(
            {
                "pattern": name,
                "node_activity": list(node_activity),
                "cell_activity": list(cell_activity),
                "expected_gate": "underresolved_activity_boundary_rejected",
                "exception_type": exception_type,
                "exception_message": exception_message,
                "adapter_state_unchanged": unchanged,
                "passed": rejected,
            }
        )
    return {
        "policy": "no interpolation across an activity boundary",
        "patterns": rows,
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def _run_m2_lifecycle() -> dict[str, Any]:
    result = _run_release_sequence(
        "straight",
        span_cells=2,
        target_spacing_m=TARGET_SPACINGS_M[0],
        transport_substeps=1,
        alphas_rad=(
            FIRST_ALPHA_RAD,
            0.0,
            CONTINUOUS_ALPHA_RAD,
            CONTINUOUS_ALPHA_RAD,
        ),
        expected_modes=("first", "inactive", "restart", "continuous"),
        wing_id="cumulative-m2-lifecycle:straight:wing",
    )
    releases = result.summary["releases"]
    lifecycle_count_passed = bool(
        len(releases) == 4
        and releases[1]["new_particle_count"] == 0
        and releases[1]["total_particle_count"] == releases[0]["total_particle_count"]
        and releases[2]["total_particle_count"]
        == releases[1]["total_particle_count"] + releases[2]["new_particle_count"]
        and releases[3]["total_particle_count"]
        == releases[2]["total_particle_count"] + releases[3]["new_particle_count"]
        and releases[3]["release_slice_steps"] == [1, 3, 4]
    )
    partial = _run_partial_activity_pre_gates()
    raw_with_partial = dict(result.raw)
    raw_with_partial["partial_activity_pre_gates"] = partial
    return {
        "summary": {
            **result.summary,
            "sequence": ["first", "inactive", "restart", "continuous"],
            "inactive_old_cloud_advanced_without_phantom_release": (
                lifecycle_count_passed
            ),
            "partial_activity_pre_gates": partial,
            "passed": bool(
                result.summary["passed"]
                and lifecycle_count_passed
                and partial["passed"]
            ),
        },
        "raw_arrays": raw_with_partial,
    }


def _refinement_family(
    differences: Sequence[float],
    *,
    fine_limit: float,
    minimum_ratio: float,
    require_nondegenerate: bool,
) -> dict[str, Any]:
    values = tuple(float(value) for value in differences)
    finite = bool(
        len(values) >= 2
        and all(np.isfinite(value) and value >= 0.0 for value in values)
    )
    nondegenerate = bool(
        finite
        and all(value > MIN_NONDEGENERATE_RELATIVE_DIFFERENCE for value in values)
    )
    ratios = tuple(
        coarse / fine
        if np.isfinite(coarse) and np.isfinite(fine) and fine > 0.0
        else float("nan")
        for coarse, fine in zip(values[:-1], values[1:])
    )
    monotone = bool(
        finite and all(fine < coarse for coarse, fine in zip(values[:-1], values[1:]))
    )
    ratio_passed = bool(
        ratios
        and all(np.isfinite(value) and value >= minimum_ratio for value in ratios)
    )
    passed = bool(
        finite
        and monotone
        and ratio_passed
        and values[-1] <= fine_limit
        and (nondegenerate or not require_nondegenerate)
    )
    return {
        "consecutive_relative_l2": list(values),
        "coarse_to_fine_ratios": [
            value if np.isfinite(value) else None for value in ratios
        ],
        "finite": finite,
        "nondegenerate": nondegenerate,
        "nondegenerate_required": require_nondegenerate,
        "strictly_monotone": monotone,
        "minimum_reduction_ratio": minimum_ratio,
        "fine_relative_l2_limit": fine_limit,
        "passed": passed,
    }


def _time_refinement_gate(
    rows: Sequence[_ConfigurationResult],
) -> dict[str, Any]:
    if len(rows) != len(TRANSPORT_SUBSTEPS):
        raise ValueError("time refinement requires the frozen three resolutions")
    frontier = [
        _relative_l2(
            left.frontier_minus_latest_birth,
            right.frontier_minus_latest_birth,
        )
        for left, right in zip(rows[:-1], rows[1:])
    ]
    probes = [
        _relative_l2(
            left.fixed_probe_induced_velocity,
            right.fixed_probe_induced_velocity,
        )
        for left, right in zip(rows[:-1], rows[1:])
    ]
    frontier_gate = _refinement_family(
        frontier,
        fine_limit=MAX_FINE_TIME_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_TIME_REFINEMENT_RATIO,
        require_nondegenerate=True,
    )
    probe_gate = _refinement_family(
        probes,
        fine_limit=MAX_FINE_TIME_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_TIME_REFINEMENT_RATIO,
        require_nondegenerate=True,
    )
    return {
        "transport_substeps": [row.summary["transport_substeps"] for row in rows],
        "frontier_minus_latest_birth": frontier_gate,
        "fixed_probe_induced_velocity": probe_gate,
        "passed": bool(frontier_gate["passed"] and probe_gate["passed"]),
    }


def _h_refinement_gate(rows: Sequence[_ConfigurationResult]) -> dict[str, Any]:
    if len(rows) != len(TARGET_SPACINGS_M):
        raise ValueError("h refinement requires the frozen four spacings")
    frontier = [
        _relative_l2(
            left.frontier_minus_latest_birth,
            right.frontier_minus_latest_birth,
        )
        for left, right in zip(rows[:-1], rows[1:])
    ]
    probes = [
        _relative_l2(
            left.fixed_probe_induced_velocity,
            right.fixed_probe_induced_velocity,
        )
        for left, right in zip(rows[:-1], rows[1:])
    ]
    frontier_gate = _refinement_family(
        frontier,
        fine_limit=MAX_FINE_H_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_H_REFINEMENT_RATIO,
        require_nondegenerate=True,
    )
    probe_gate = _refinement_family(
        probes,
        fine_limit=MAX_FINE_H_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_H_REFINEMENT_RATIO,
        require_nondegenerate=True,
    )
    particle_counts = [row.particle_count for row in rows]
    realized_spacing = [row.realized_spacing_max_m for row in rows]
    particle_count_up = all(
        coarse < fine for coarse, fine in zip(particle_counts[:-1], particle_counts[1:])
    )
    realized_h_down = all(
        coarse > fine
        for coarse, fine in zip(realized_spacing[:-1], realized_spacing[1:])
    )
    return {
        "target_spacings_m": [row.summary["target_spacing_m"] for row in rows],
        "particle_counts": particle_counts,
        "realized_spacing_max_m": realized_spacing,
        "particle_count_strictly_increases": particle_count_up,
        "realized_h_strictly_decreases": realized_h_down,
        "frontier_minus_latest_birth": frontier_gate,
        "fixed_probe_induced_velocity": probe_gate,
        "passed": bool(
            particle_count_up
            and realized_h_down
            and frontier_gate["passed"]
            and probe_gate["passed"]
        ),
    }


def _run_m3(config: GateConfig) -> dict[str, Any]:
    configurations: list[_ConfigurationResult] = []
    geometry_summaries: list[dict[str, Any]] = []
    for geometry in CASE_GEOMETRIES:
        by_spacing: list[list[_ConfigurationResult]] = []
        for spacing in config.target_spacings_m:
            time_rows = [
                _run_configuration(
                    geometry,
                    span_cells=config.span_cells,
                    target_spacing_m=spacing,
                    transport_substeps=substeps,
                )
                for substeps in config.transport_substeps
            ]
            configurations.extend(time_rows)
            by_spacing.append(time_rows)
        time_gates = [
            {
                "target_spacing_m": spacing,
                **_time_refinement_gate(rows),
            }
            for spacing, rows in zip(config.target_spacings_m, by_spacing, strict=True)
        ]
        finest_time_rows = [rows[-1] for rows in by_spacing]
        h_gate = _h_refinement_gate(finest_time_rows)
        configuration_mechanics = bool(
            all(row.summary["passed"] for rows in by_spacing for row in rows)
        )
        geometry_summaries.append(
            {
                "geometry": geometry,
                "configuration_count": sum(len(rows) for rows in by_spacing),
                "configurations": [row.summary for rows in by_spacing for row in rows],
                "time_refinement_by_h": time_gates,
                "h_refinement_at_finest_time": h_gate,
                "configuration_mechanics_passed": configuration_mechanics,
                "time_refinement_passed": all(gate["passed"] for gate in time_gates),
                "h_refinement_passed": h_gate["passed"],
                "passed": bool(
                    configuration_mechanics
                    and all(gate["passed"] for gate in time_gates)
                    and h_gate["passed"]
                ),
            }
        )
    if len(configurations) != MAX_CONFIGURATIONS:
        raise RuntimeError("M3 did not execute exactly 36 configurations")
    return {
        "summary": {
            "geometry_count": len(CASE_GEOMETRIES),
            "configuration_count": len(configurations),
            "release_count_per_configuration": M3_RELEASE_COUNT,
            "geometries": geometry_summaries,
            "passed": bool(
                geometry_summaries and all(row["passed"] for row in geometry_summaries)
            ),
        },
        "raw_arrays": [
            {
                "geometry": row.summary["geometry"],
                "target_spacing_m": row.summary["target_spacing_m"],
                "transport_substeps": row.summary["transport_substeps"],
                **row.raw,
            }
            for row in configurations
        ],
    }


def run_minimal_smoke() -> dict[str, Any]:
    """Run the bounded M0/M1 straight two-cell cumulative smoke."""

    m0 = _run_m0_parity()
    m1 = _run_m1_four_release_smoke()
    m2 = _run_m2_lifecycle()
    passed = bool(
        m0["summary"]["passed"] and m1["summary"]["passed"] and m2["summary"]["passed"]
    )
    return {
        "schema_id": f"{RUN_SCHEMA_ID}:minimal-smoke",
        "summary": {
            "m0_v1_physical_bitwise_parity": m0["summary"],
            "m1_three_and_four_release": m1["summary"],
            "m2_lifecycle_and_partial_activity": m2["summary"],
            "passed": passed,
        },
        "raw_arrays": {
            "m0_v1_physical_bitwise_parity": m0["raw_arrays"],
            "m1_three_and_four_release": m1["raw_arrays"],
            "m2_lifecycle": m2["raw_arrays"],
        },
        "passed": passed,
    }


def _run_cumulative_cloud_gate_impl(
    config: GateConfig,
    *,
    minimal_smoke_only: bool,
) -> dict[str, Any]:
    """Run the numerical gate; the public wrapper adds measured boundaries."""

    _validate_config(config)
    source_hashes_before = _source_hashes()
    smoke = run_minimal_smoke()
    m3: dict[str, Any] | None = None
    if smoke["passed"] and not minimal_smoke_only:
        m3 = _run_m3(config)
    source_hashes_after = _source_hashes()
    source_snapshot_stable = source_hashes_before == source_hashes_after
    coverage_passed = bool(
        m3 is not None and m3["summary"]["configuration_count"] == MAX_CONFIGURATIONS
    )
    m3_passed = bool(m3 is not None and m3["summary"]["passed"])
    passed = bool(
        smoke["passed"]
        and coverage_passed
        and m3_passed
        and source_snapshot_stable
        and not minimal_smoke_only
    )
    gate_summary = {
        "pre_gates_passed": bool(smoke["passed"]),
        "configuration_coverage_passed": coverage_passed,
        "m3_mechanics_and_refinement_passed": m3_passed,
        "declared_source_snapshot_stable_passed": source_snapshot_stable,
        "runtime_boundary_passed": True,
        "passed": passed,
        "stop_required": not passed,
    }
    result = {
        "schema_id": RUN_SCHEMA_ID,
        "run_id": RUN_ID,
        "tier": RUN_TIER,
        "status": "go_m3_mechanics_only" if passed else "stop",
        "passed": passed,
        "canonical_eligible": False,
        "summary": {
            "scope": "non_target_cumulative_cloud_m0_to_m3_numerical_core",
            "execution_mode": (
                "minimal_smoke_artifact_validation"
                if minimal_smoke_only
                else "full_frozen_36_configuration_gate"
            ),
            "observation_access": "none",
            "target_case_branch": "none",
            "surface_solver_call_count": 0,
            "surface_channel_call_count": 0,
            "paper_comparison_call_count": 0,
            "config": {
                "geometries": list(CASE_GEOMETRIES),
                "span_cells": config.span_cells,
                "target_spacings_m": list(config.target_spacings_m),
                "transport_substeps": list(config.transport_substeps),
                "m1_smoke_release_count": config.m1_smoke_release_count,
                "m3_release_count": config.m3_release_count,
                "sigma_birth_m": SIGMA_BIRTH_M,
                "physical_release_dt_s": PHYSICAL_RELEASE_DT_S,
                "freestream_velocity_gp1_m_per_s": list(FREESTREAM_GP1_M_PER_S),
                "particles_per_release_cap": config.particles_per_release_cap,
                "cumulative_particle_cap": config.cumulative_particle_cap,
                "configuration_cap": config.configuration_cap,
                "transport_substeps_cap": config.transport_substeps_cap,
            },
            "thresholds": {
                "fine_time_relative_l2": MAX_FINE_TIME_RELATIVE_DIFFERENCE,
                "minimum_time_reduction_ratio": MIN_TIME_REFINEMENT_RATIO,
                "fine_h_relative_l2": MAX_FINE_H_RELATIVE_DIFFERENCE,
                "minimum_h_reduction_ratio": MIN_H_REFINEMENT_RATIO,
                "minimum_nondegenerate_relative_l2": (
                    MIN_NONDEGENERATE_RELATIVE_DIFFERENCE
                ),
            },
            "pre_gates": smoke["summary"],
            "m3": None if m3 is None else m3["summary"],
            "gate_summary": gate_summary,
            "m4_artifact_writer_implemented": True,
            "passed": passed,
        },
        "raw_arrays": {
            "pre_gates": smoke["raw_arrays"],
            "m3": None if m3 is None else m3["raw_arrays"],
        },
        "declared_semantic_source_sha256": source_hashes_after,
    }
    return result


def _source_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    platform_root = repo_root / "platform"
    benchmark_root = platform_root / "forward_flight_benchmarks"
    return {
        "runner": Path(__file__).resolve(),
        "v1_fixed_core_runner": (
            benchmark_root / "run_v5h_dvm_node_fixed_core_gate.py"
        ),
        "benchmark_package_init": benchmark_root / "__init__.py",
        "cumulative_cloud_core": benchmark_root / "v5h_cumulative_cloud_transport.py",
        "passive_frontier_core": benchmark_root / "v5h_passive_frontier_transport.py",
        "dvm_node_ribbon": benchmark_root / "v5h_dvm_node_ribbon.py",
        "dvm_source": benchmark_root / "v5h_dvm_source.py",
        "dvm_node_placement": benchmark_root / "v5h_dvm_node_placement.py",
        "ldvm_section_contract": benchmark_root / "ldvm_uvlm_correction.py",
        "ldvm_fourier": platform_root / "ldvm_fourier.py",
        "ldvm_induction_kernel": platform_root / "flap_ldvm.py",
        "rvpm_edge_bridge": repo_root / "src/fluxvortex/rvpm_edge_bridge.py",
        "rvpm_reference": repo_root / "src/fluxvortex/rvpm_reference.py",
        "rvpm_transport": repo_root / "src/fluxvortex/rvpm_transport.py",
        "package_contract": repo_root / "pyproject.toml",
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[2]
    result: dict[str, dict[str, str]] = {}
    for role, path in _source_paths().items():
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(repo_root):
            raise FileNotFoundError(f"source closure path is unavailable: {path}")
        result[role] = {
            "path": resolved.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(resolved),
        }
    return result


def _optional_package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _validated_utc(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must carry the UTC offset")
    return value


def _validated_run_uuid(value: str | None) -> str:
    text = str(uuid4()) if value is None else str(value)
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as error:
        raise ValueError("run_uuid must be a valid UUID") from error
    if str(parsed) != text:
        raise ValueError("run_uuid must use canonical UUID text")
    return text


def _git_command(repo_root: Path, *args: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return False, message
    return True, completed.stdout.rstrip("\n")


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    revision_ok, revision = _git_command(repo_root, "rev-parse", "HEAD")
    branch_ok, branch = _git_command(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status_ok, status = _git_command(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    status_lines = status.splitlines() if status_ok and status else []
    return {
        "available": bool(revision_ok and branch_ok and status_ok),
        "revision": revision if revision_ok else None,
        "branch": branch if branch_ok else None,
        "dirty": bool(status_lines) if status_ok else None,
        "porcelain_v1": status_lines if status_ok else None,
        "porcelain_v1_sha256": (
            sha256((status + "\n").encode("utf-8")).hexdigest() if status_ok else None
        ),
        "errors": {
            key: item
            for key, ok, item in (
                ("revision", revision_ok, revision),
                ("branch", branch_ok, branch),
                ("status", status_ok, status),
            )
            if not ok
        },
    }


def _process_argv() -> list[str]:
    proc_cmdline = Path("/proc/self/cmdline")
    if proc_cmdline.is_file():
        raw = proc_cmdline.read_bytes().split(b"\0")
        values = [os.fsdecode(value) for value in raw if value]
        if values:
            return values
    return [sys.executable, *sys.argv]


def _observed_repo_module_snapshot(repo_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for module_name, module in sorted(sys.modules.items()):
        path_value = getattr(module, "__file__", None)
        if not isinstance(path_value, str):
            continue
        path = Path(path_value).resolve(strict=False)
        if path.suffix == ".pyc":
            try:
                source_candidate = Path(importlib_util.source_from_cache(str(path)))
            except (ValueError, NotImplementedError):
                source_candidate = path
            if source_candidate.is_file():
                path = source_candidate.resolve()
        if not path.is_file() or not path.is_relative_to(repo_root):
            continue
        relative = path.relative_to(repo_root).as_posix()
        key = (module_name, relative)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "module": module_name,
                "path": relative,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "scope": (
            "observed sys.modules repository snapshot at provenance capture; "
            "not claimed as a complete import trace"
        ),
        "files": rows,
    }


def capture_run_provenance(
    *,
    output_dir: Path,
    numerical_started_utc: str,
    numerical_completed_utc: str,
    run_uuid: str | None = None,
) -> dict[str, Any]:
    """Capture invocation-specific evidence outside deterministic semantics."""

    repo_root = Path(__file__).resolve().parents[2]
    process_argv = _process_argv()
    started_utc = _validated_utc("numerical_started_utc", numerical_started_utc)
    completed_utc = _validated_utc("numerical_completed_utc", numerical_completed_utc)
    return {
        "schema_id": RUN_PROVENANCE_SCHEMA_ID,
        "run_uuid": _validated_run_uuid(run_uuid),
        "numerical_started_utc": started_utc,
        "numerical_completed_utc": completed_utc,
        "provenance_captured_utc": _utc_now(),
        "process_argv": process_argv,
        "process_command": shlex.join(process_argv),
        "python_sys_argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "output_dir": str(Path(output_dir).resolve(strict=False)),
        "environment": {
            "python_executable": sys.executable,
            "python_version": runtime_platform.python_version(),
            "python_implementation": runtime_platform.python_implementation(),
            "platform": runtime_platform.platform(),
            "machine": runtime_platform.machine(),
            "byteorder": sys.byteorder,
            "variables": {
                key: os.environ[key]
                for key in REPRODUCIBILITY_ENV_KEYS
                if key in os.environ
            },
            "packages": {
                package: _optional_package_version(package)
                for package in PROVENANCE_PACKAGES
            },
        },
        "git": _git_snapshot(repo_root),
        "observed_repo_modules": _observed_repo_module_snapshot(repo_root),
    }


def _sum_named_integer(value: object, name: str) -> int:
    total = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == name:
                if isinstance(item, bool) or not isinstance(item, Integral):
                    raise ValueError(f"runtime counter {name!r} is not an integer")
                total += int(item)
            else:
                total += _sum_named_integer(item, name)
    elif isinstance(value, (list, tuple)):
        total += sum(_sum_named_integer(item, name) for item in value)
    return total


def run_cumulative_cloud_gate(
    config: GateConfig = GateConfig(),
    *,
    minimal_smoke_only: bool = False,
) -> dict[str, Any]:
    """Run M0--M3 with measured observation and surface-call boundaries."""

    if not isinstance(minimal_smoke_only, bool):
        raise ValueError("minimal_smoke_only must be Boolean")
    instrumentation = _RuntimeBoundaryInstrumentation()
    with instrumentation:
        result = _run_cumulative_cloud_gate_impl(
            config,
            minimal_smoke_only=minimal_smoke_only,
        )
    instrumented = instrumentation.evidence()
    direct_forbidden_imports = _direct_forbidden_imports()
    surface_quantity_paths = _surface_quantity_key_paths(result)
    measured_component_counts = {
        name: _sum_named_integer(result, name)
        for name in (
            "feedback_call_count",
            "parent_write_count",
            "surface_channel_write_count",
        )
    }
    runtime_boundary_passed = bool(
        instrumented["passed"]
        and not direct_forbidden_imports
        and not surface_quantity_paths
        and not any(measured_component_counts.values())
    )
    runtime_evidence = {
        "package_init_eager_target_or_ptera_definitions_loaded": any(
            module_name in sys.modules
            for module_name in (
                "forward_flight_benchmarks.baik2012",
                "forward_flight_benchmarks.ptera_adapter",
                "pterasoftware",
            )
        ),
        "scope_note": (
            "package initialisation may eagerly import target or Ptera definitions; "
            "the gate proves zero guarded solver/builder invocation and zero "
            "target-data read during the measured numerical interval"
        ),
        "instrumented_access": instrumented,
        "direct_forbidden_imports": direct_forbidden_imports,
        "surface_quantity_key_paths": surface_quantity_paths,
        "measured_component_counts": measured_component_counts,
        "passed": runtime_boundary_passed,
    }
    result["runtime_boundary_evidence"] = runtime_evidence
    gates = result["summary"]["gate_summary"]
    gates["runtime_boundary_passed"] = runtime_boundary_passed
    gates["passed"] = bool(gates["passed"] and runtime_boundary_passed)
    gates["stop_required"] = not gates["passed"]
    result["summary"]["passed"] = gates["passed"]
    result["passed"] = gates["passed"]
    result["status"] = "go_m3_mechanics_only" if gates["passed"] else "stop"
    if _surface_quantity_key_paths(result):
        raise RuntimeError("result schema exposed a forbidden surface quantity key")
    return result


_RAW_ARRAY_ENCODING = "float64-little-endian-json-v1"


def _raw_array_record(value: object) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError("raw evidence arrays must be finite")
    return {
        "encoding": _RAW_ARRAY_ENCODING,
        "dtype": "<f8",
        "shape": list(array.shape),
        "values": array.tolist(),
        "sha256": _array_sha256(array),
    }


def _raw_encode(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _raw_array_record(value)
    if isinstance(value, np.generic):
        return _raw_encode(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "record_type": type(value).__name__,
            **{
                item.name: _raw_encode(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("raw evidence mappings require string keys")
        return {key: _raw_encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_raw_encode(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("raw evidence contains a non-finite scalar")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported raw evidence value: {type(value).__name__}")


def _array_from_raw_record(name: str, record: object) -> FloatArray:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} raw array record must be an object")
    if set(record) != {"encoding", "dtype", "shape", "values", "sha256"}:
        raise ValueError(f"{name} raw array fields are incomplete or unknown")
    if record.get("encoding") != _RAW_ARRAY_ENCODING or record.get("dtype") != "<f8":
        raise ValueError(f"{name} raw array encoding is unsupported")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise ValueError(f"{name} raw array shape is invalid")
    array = np.ascontiguousarray(np.asarray(record.get("values"), dtype="<f8"))
    expected_shape = tuple(shape)
    if array.shape != expected_shape and array.size == 0 and 0 in expected_shape:
        array = np.ascontiguousarray(array.reshape(expected_shape))
    if array.shape != expected_shape:
        raise ValueError(f"{name} raw array shape disagrees with values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} raw array contains non-finite values")
    expected = record.get("sha256")
    if not isinstance(expected, str) or _array_sha256(array) != expected:
        raise ValueError(f"{name} raw array SHA-256 mismatch")
    return array


def _decode_raw_value(name: str, value: object) -> object:
    if isinstance(value, Mapping):
        if value.get("encoding") == _RAW_ARRAY_ENCODING:
            return _array_from_raw_record(name, value)
        return {
            key: _decode_raw_value(f"{name}.{key}", item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _decode_raw_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} raw scalar is non-finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"{name} contains an unsupported raw value")


def _configuration_summary_index(
    semantic_result: Mapping[str, Any],
) -> dict[tuple[str, float, int], Mapping[str, Any]]:
    numerical = semantic_result.get("summary")
    if not isinstance(numerical, Mapping):
        raise ValueError("semantic result numerical summary is invalid")
    m3 = numerical.get("m3")
    if m3 is None:
        return {}
    if not isinstance(m3, Mapping) or not isinstance(m3.get("geometries"), list):
        raise ValueError("semantic result M3 summary is invalid")
    result: dict[tuple[str, float, int], Mapping[str, Any]] = {}
    for geometry_row in m3["geometries"]:
        if not isinstance(geometry_row, Mapping):
            raise ValueError("M3 geometry row must be an object")
        configurations = geometry_row.get("configurations")
        if not isinstance(configurations, list):
            raise ValueError("M3 configurations must be a list")
        for row in configurations:
            if not isinstance(row, Mapping):
                raise ValueError("M3 configuration row must be an object")
            geometry = row.get("geometry")
            spacing = row.get("target_spacing_m")
            substeps = row.get("transport_substeps")
            if geometry not in CASE_GEOMETRIES:
                raise ValueError("M3 configuration geometry is invalid")
            if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
                raise ValueError("M3 configuration spacing is invalid")
            if isinstance(substeps, bool) or not isinstance(substeps, int):
                raise ValueError("M3 configuration substeps are invalid")
            key = str(geometry), float(spacing), int(substeps)
            if key in result:
                raise ValueError("M3 configuration key is duplicated")
            result[key] = row
    return result


def _raw_configuration_key(value: Mapping[str, Any]) -> tuple[str, float, int]:
    geometry = value.get("geometry")
    spacing = value.get("target_spacing_m")
    substeps = value.get("transport_substeps")
    if geometry not in CASE_GEOMETRIES:
        raise ValueError("raw M3 geometry is invalid")
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
        raise ValueError("raw M3 target spacing is invalid")
    if isinstance(substeps, bool) or not isinstance(substeps, int):
        raise ValueError("raw M3 transport substeps are invalid")
    return str(geometry), float(spacing), int(substeps)


def _make_raw_refinement_evidence(
    semantic_result: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    numerical = semantic_result.get("summary")
    if not isinstance(numerical, Mapping):
        raise ValueError("semantic result numerical summary is invalid")
    pre_summary = numerical.get("pre_gates")
    pre_raw = raw_arrays.get("pre_gates")
    if not isinstance(pre_summary, Mapping) or not isinstance(pre_raw, Mapping):
        raise ValueError("pre-gate semantic/raw evidence is missing")
    summary_index = _configuration_summary_index(semantic_result)
    m3_raw = raw_arrays.get("m3")
    if m3_raw is None:
        raw_rows: list[Mapping[str, Any]] = []
    elif isinstance(m3_raw, list) and all(isinstance(item, Mapping) for item in m3_raw):
        raw_rows = list(m3_raw)
    else:
        raise ValueError("M3 raw arrays must be a list or null")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, float, int]] = set()
    for raw_row in raw_rows:
        key = _raw_configuration_key(raw_row)
        if key in seen or key not in summary_index:
            raise ValueError("M3 raw configuration key is invalid or duplicated")
        seen.add(key)
        summary_row = summary_index[key]
        evidence = {
            name: item
            for name, item in raw_row.items()
            if name not in {"geometry", "target_spacing_m", "transport_substeps"}
        }
        records.append(
            {
                "geometry": key[0],
                "target_spacing_m": key[1],
                "transport_substeps": key[2],
                "configuration_summary_sha256": _json_payload_sha256(
                    "fluxv-v5h-cumulative-configuration-summary-v1", summary_row
                ),
                "evidence": _raw_encode(evidence),
            }
        )
    if seen != set(summary_index):
        raise ValueError("M3 raw evidence does not cover its semantic summary")
    return {
        "schema_id": RAW_REFINEMENT_SCHEMA_ID,
        "run_id": RUN_ID,
        "execution_mode": numerical.get("execution_mode"),
        "array_encoding": (
            "strict JSON binary64 round-trip values with dtype, shape, and "
            "canonical little-endian array SHA-256"
        ),
        "pre_gates_summary_sha256": _json_payload_sha256(
            "fluxv-v5h-cumulative-pre-gates-summary-v1", pre_summary
        ),
        "pre_gates": _raw_encode(pre_raw),
        "m3_configurations": records,
    }


def _semantic_result(
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = dict(result)
    raw_arrays = semantic.pop("raw_arrays", None)
    if not isinstance(raw_arrays, Mapping):
        raise ValueError("run result has no raw array evidence")
    raw = _make_raw_refinement_evidence(semantic, raw_arrays)
    return semantic, raw


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trace_sha256_from_raw(value: object) -> str:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("raw transport trace is invalid")
    digest = sha256(b"fluxv-v5h-cumulative-transport-trace-v2\0")
    for event in value:
        digest.update(event.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _slice_steps_and_count(value: object) -> tuple[list[int], int]:
    if not isinstance(value, list):
        raise ValueError("release slice ledger must be a list")
    steps: list[int] = []
    cursor = 0
    for release_index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("release slice row must be an object")
        if item.get("record_type") != "ReleaseSliceLedger":
            raise ValueError("release slice row type is invalid")
        start = item.get("start_index")
        stop = item.get("stop_index")
        count = item.get("particle_count")
        source_step = item.get("source_step_index")
        if any(
            isinstance(number, bool) or not isinstance(number, int)
            for number in (start, stop, count, source_step)
        ):
            raise ValueError("release slice integer field is invalid")
        if (
            item.get("release_index") != release_index
            or start != cursor
            or stop != start + count
            or count <= 0
            or source_step <= 0
            or item.get("smoothing_radius_m") != SIGMA_BIRTH_M
            or not _valid_sha256(item.get("parent_ribbon_digest_sha256"))
            or not _valid_sha256(item.get("deposited_cloud_digest_sha256"))
            or not _valid_sha256(item.get("particle_ids_sha256"))
            or not _valid_sha256(item.get("lineage_sha256"))
        ):
            raise ValueError("release slice ledger is inconsistent")
        steps.append(int(source_step))
        cursor = int(stop)
    return steps, cursor


def _edge_metrics_raw_passed(
    value: object,
    *,
    expected_active: bool,
    expected_particle_count: int,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not expected_active:
        return bool(
            value.get("active") is False
            and value.get("particle_count") == 0
            and value.get("realized_spacing_max_m") is None
            and value.get("passed") is True
        )
    realization = value.get("edge_realization")
    numeric_names = (
        "realized_spacing_min_m",
        "realized_spacing_max_m",
        "realized_overlap_min",
        "realized_overlap_max",
        "max_edge_conservation_abs",
        "global_conservation_abs",
        "incidence_residual",
        "edge_reconstruction_residual",
    )
    if not isinstance(realization, list) or not realization:
        return False
    if any(
        isinstance(value.get(name), bool)
        or not isinstance(value.get(name), (int, float))
        or not np.isfinite(float(value[name]))
        for name in numeric_names
    ):
        return False
    requested = value.get("requested_target_spacing_m")
    smoothing = value.get("smoothing_radius_m")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not np.isfinite(float(requested))
        or isinstance(smoothing, bool)
        or not isinstance(smoothing, (int, float))
    ):
        return False
    realized_counts = 0
    realized_max = 0.0
    realized_min = float("inf")
    overlap_min = float("inf")
    for row in realization:
        if not isinstance(row, Mapping):
            return False
        count = row.get("subdivision_count")
        spacing = row.get("realized_spacing_m")
        overlap = row.get("realized_overlap")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not np.isfinite(float(spacing))
            or float(spacing) <= 0.0
            or isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not np.isfinite(float(overlap))
        ):
            return False
        realized_counts += count
        realized_max = max(realized_max, float(spacing))
        realized_min = min(realized_min, float(spacing))
        overlap_min = min(overlap_min, float(overlap))
        if not np.isclose(
            float(overlap),
            float(smoothing) / float(spacing),
            rtol=0.0,
            atol=2.0e-15,
        ):
            return False
    return bool(
        value.get("active") is True
        and value.get("particle_count") == expected_particle_count
        and value.get("retained_edge_count") == len(realization)
        and value.get("edge_subdivision_counts")
        == [row["subdivision_count"] for row in realization]
        and realized_counts == expected_particle_count
        and float(smoothing) == SIGMA_BIRTH_M
        and float(requested) > 0.0
        and value.get("fixed_physical_sigma") is True
        and value.get("finite") is True
        and np.isclose(
            float(value["realized_spacing_min_m"]),
            realized_min,
            rtol=0.0,
            atol=2.0e-15,
        )
        and np.isclose(
            float(value["realized_spacing_max_m"]),
            realized_max,
            rtol=0.0,
            atol=2.0e-15,
        )
        and np.isclose(
            float(value["realized_overlap_min"]),
            overlap_min,
            rtol=0.0,
            atol=2.0e-15,
        )
        and realized_max <= float(requested) * (1.0 + 1.0e-12)
        and overlap_min >= FROZEN_OVERLAP_LAMBDA * (1.0 - 1.0e-12)
        and float(value["max_edge_conservation_abs"]) <= 1.0e-14
        and float(value["global_conservation_abs"]) <= 1.0e-14
        and float(value["incidence_residual"]) <= MAX_RIBBON_RESIDUAL
        and float(value["edge_reconstruction_residual"]) <= MAX_RIBBON_RESIDUAL
        and value.get("passed") is True
    )


def _transport_invariants_raw_passed(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    drift = value.get("particle_invariant_relative_drift_max")
    vector_abs = value.get("global_vector_change_abs_m3_per_s")
    vector_rel = value.get("global_vector_change_relative")
    return bool(
        not isinstance(drift, bool)
        and isinstance(drift, (int, float))
        and np.isfinite(float(drift))
        and float(drift) <= MAX_PARTICLE_INVARIANT_REL
        and not isinstance(vector_abs, bool)
        and isinstance(vector_abs, (int, float))
        and np.isfinite(float(vector_abs))
        and not isinstance(vector_rel, bool)
        and isinstance(vector_rel, (int, float))
        and np.isfinite(float(vector_rel))
        and value.get("global_vector_change_gate_eligible") is False
        and value.get("positive_finite_sigma") is True
        and value.get("passed") is True
    )


def _release_raw_passed(
    raw: Mapping[str, Any],
    reported: Mapping[str, Any],
    *,
    expected_substeps: int,
    expected_spacing_m: float,
    expected_step: int,
    expected_mode: str,
    expected_incidence_rad: float,
    expected_node_count: int,
) -> bool:
    required_arrays = {
        name: raw.get(name)
        for name in (
            "positions_gp1_m",
            "gamma_vector_m3_per_s",
            "sigma_m",
            "frontier_positions_gp1_m",
            "latest_birth_positions_gp1_m",
            "frontier_minus_latest_birth_gp1_m",
            "fixed_probe_induced_velocity_gp1_m_per_s",
        )
    }
    if not all(isinstance(value, np.ndarray) for value in required_arrays.values()):
        return False
    positions = required_arrays["positions_gp1_m"]
    gamma = required_arrays["gamma_vector_m3_per_s"]
    sigma = required_arrays["sigma_m"]
    frontier = required_arrays["frontier_positions_gp1_m"]
    births = required_arrays["latest_birth_positions_gp1_m"]
    delta = required_arrays["frontier_minus_latest_birth_gp1_m"]
    probes = required_arrays["fixed_probe_induced_velocity_gp1_m_per_s"]
    if not all(isinstance(item, np.ndarray) for item in required_arrays.values()):
        return False
    total = raw.get("total_particle_count")
    previous = raw.get("previous_particle_count")
    new = raw.get("new_particle_count")
    predicted = raw.get("predicted_new_particle_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (total, previous, new, predicted)
    ):
        return False
    counters = raw.get("transport_counters")
    trace = raw.get("transport_trace")
    slices = raw.get("release_slices")
    particle_ids = raw.get("particle_ids")
    lineage = raw.get("lineage")
    fact_identity = raw.get("frontier_fact_identity")
    if (
        not isinstance(counters, Mapping)
        or not isinstance(particle_ids, list)
        or not isinstance(lineage, list)
        or not isinstance(fact_identity, list)
    ):
        return False
    try:
        slice_steps, slice_count = _slice_steps_and_count(slices)
        trace_digest = _trace_sha256_from_raw(trace)
    except ValueError:
        return False
    active = expected_mode != "inactive"
    zero_counter_names = (
        "sort_count",
        "weld_count",
        "delete_count",
        "cancel_count",
        "remesh_count",
        "feedback_call_count",
        "parent_write_count",
        "surface_channel_write_count",
    )
    combined_counts = counters.get("combined_stage_particle_counts")
    reported_slice_steps = reported.get("release_slice_steps")
    digest_names = (
        "report_sha256",
        "cloud_sha256",
        "transport_trace_sha256",
        "edge_bridge_artifact_sha256",
        "current_ribbon_digest_sha256",
        "handoff_sha256",
        "appended_cloud_digest_before_transport_sha256",
        "transported_cloud_digest_after_sha256",
    )
    trace_prefix_counts = {
        "deposition_call_count": sum(
            item.startswith("deposit_prescribed_sigma_spacing:") for item in trace
        ),
        "lsrk3_call_count": sum(
            item.startswith("combined_lsrk3_call:") for item in trace
        ),
        "lsrk3_stage_count": sum(
            item.startswith("combined_lsrk3_stage:") for item in trace
        ),
        "stage_pre_field_call_count": sum(
            item.startswith("frontier_stage_pre_field:") for item in trace
        ),
        "sort_count": sum(item.startswith("particle_sort:") for item in trace),
        "weld_count": sum(item.startswith("particle_weld:") for item in trace),
        "delete_count": sum(item.startswith("particle_delete:") for item in trace),
        "cancel_count": sum(item.startswith("particle_cancel:") for item in trace),
        "remesh_count": sum(item.startswith("particle_remesh:") for item in trace),
        "feedback_call_count": sum(
            item.startswith("feedback_write:") for item in trace
        ),
        "parent_write_count": sum(item.startswith("parent_write:") for item in trace),
        "surface_channel_write_count": sum(
            item.startswith("load_write:") for item in trace
        ),
    }
    measured_attestations = raw.get("measured_mechanical_attestations")
    reported_placement = reported.get("placement")
    reported_handoff = reported.get("ribbon_handoff")
    if not isinstance(reported_placement, Mapping) or not isinstance(
        reported_handoff, Mapping
    ):
        return False
    expected_attestation_keys = {
        "source_passed",
        "placement_passed",
        "placement_birth_passed",
        "continuous_one_third_passed",
        "ribbon_handoff_passed",
        "cell_strength_owner_passed",
        "ribbon_passed",
        "prefix_identity_passed",
        "previous_report_immutable_passed",
        "new_release_identity_passed",
        "release_slice_ledger_passed",
        "report_exact_append_passed",
        "report_one_combined_field_passed",
        "report_stage_pre_replay_passed",
        "counters_passed",
        "observation_access_none",
        "target_case_branch_none",
    }
    reported_attestation_values = {
        "source_passed": reported.get("source_passed"),
        "placement_passed": reported_placement.get("passed"),
        "placement_birth_passed": reported.get("placement_birth_passed"),
        "continuous_one_third_passed": reported.get("continuous_one_third_passed"),
        "ribbon_handoff_passed": reported_handoff.get("passed"),
        "cell_strength_owner_passed": reported.get("cell_strength_owner_passed"),
        "ribbon_passed": reported.get("ribbon_passed"),
        "prefix_identity_passed": reported.get("prefix_identity_passed"),
        "previous_report_immutable_passed": reported.get(
            "previous_report_immutable_passed"
        ),
        "new_release_identity_passed": reported.get("new_release_identity_passed"),
        "release_slice_ledger_passed": reported.get("release_slice_ledger_passed"),
        "report_exact_append_passed": True,
        "report_one_combined_field_passed": True,
        "report_stage_pre_replay_passed": True,
        "counters_passed": reported.get("counters_passed"),
        "observation_access_none": True,
        "target_case_branch_none": True,
    }
    measured_attestations_passed = bool(
        isinstance(measured_attestations, Mapping)
        and set(measured_attestations) == expected_attestation_keys
        and measured_attestations == reported_attestation_values
        and all(value is True for value in measured_attestations.values())
    )
    edge_passed = _edge_metrics_raw_passed(
        raw.get("edge_metrics"),
        expected_active=active,
        expected_particle_count=new,
    )
    invariant_passed = _transport_invariants_raw_passed(raw.get("transport_invariants"))
    expected_frontier_count = expected_node_count if active else 0
    fact_identity_passed = bool(
        len(fact_identity) == expected_frontier_count
        and all(isinstance(item, list) and len(item) == 6 for item in fact_identity)
        and len({_json_text(item[0]) for item in fact_identity})
        == expected_frontier_count
        and all(item[5] == expected_step + 1 for item in fact_identity)
        and all(
            isinstance(item[4], int)
            and not isinstance(item[4], bool)
            and 1 <= item[4] <= expected_step
            for item in fact_identity
        )
    )
    return bool(
        positions.shape == (total, 3)
        and gamma.shape == (total, 3)
        and sigma.shape == (total,)
        and frontier.shape == (expected_frontier_count, 3)
        and births.shape == frontier.shape
        and delta.shape == frontier.shape
        and np.array_equal(delta, frontier - births)
        and fact_identity_passed
        and probes.shape == FIXED_PROBES_GP1_M.shape
        and np.all(np.isfinite(sigma))
        and np.all(sigma > 0.0)
        and previous + new == total
        and ((new > 0) is active)
        and predicted == new
        and len(particle_ids) == total
        and len(lineage) == total
        and len({_json_text(item) for item in particle_ids}) == total
        and slice_count == total
        and slice_steps == reported_slice_steps
        and (
            (active and slice_steps and slice_steps[-1] == expected_step)
            or (not active and expected_step not in slice_steps)
        )
        and raw.get("source_step_index") == expected_step
        and reported.get("source_step_index") == expected_step
        and reported.get("expected_mode") == expected_mode
        and raw.get("source_time_s") == (expected_step - 1) * PHYSICAL_RELEASE_DT_S
        and reported.get("source_time_s") == (expected_step - 1) * PHYSICAL_RELEASE_DT_S
        and reported.get("incidence_rad") == expected_incidence_rad
        and raw.get("transport_start_time_s")
        == (expected_step - 1) * PHYSICAL_RELEASE_DT_S
        and raw.get("transport_end_time_s") == expected_step * PHYSICAL_RELEASE_DT_S
        and raw.get("transport_substeps") == expected_substeps
        and counters.get("deposition_call_count") == (1 if active else 0)
        and counters.get("lsrk3_call_count") == expected_substeps
        and counters.get("lsrk3_stage_count") == 3 * expected_substeps
        and counters.get("stage_pre_field_call_count") == 3 * expected_substeps
        and isinstance(combined_counts, list)
        and len(combined_counts) == 3 * expected_substeps
        and all(count == total for count in combined_counts)
        and all(counters.get(name) == 0 for name in zero_counter_names)
        and all(
            counters.get(name) == count for name, count in trace_prefix_counts.items()
        )
        and edge_passed
        and invariant_passed
        and measured_attestations_passed
        and raw.get("edge_metrics") == reported.get("edge_metrics")
        and raw.get("transport_invariants") == reported.get("transport_invariants")
        and reported.get("passed") is True
        and reported.get("counters_passed") is True
        and reported.get("ribbon_passed") is True
        and reported.get("source_passed") is True
        and reported.get("cell_strength_owner_passed") is True
        and reported.get("release_slice_ledger_passed") is True
        and reported.get("target_spacing_m", expected_spacing_m) == expected_spacing_m
        and all(
            item.get("deposition_target_spacing_m") == expected_spacing_m
            for item in slices
        )
        and raw.get("exact_append_passed") is True
        and raw.get("one_combined_field_passed") is True
        and raw.get("stage_pre_replay_passed") is True
        and raw.get("exact_append_prefix_max_abs") == 0.0
        and raw.get("stage_pre_replay_max_abs") == 0.0
        and trace_digest == raw.get("transport_trace_sha256")
        and raw.get("transported_cloud_digest_after_sha256") == raw.get("cloud_sha256")
        and all(_valid_sha256(raw.get(name)) for name in digest_names)
        and reported.get("previous_particle_count") == previous
        and reported.get("new_particle_count") == new
        and reported.get("total_particle_count") == total
        and reported.get("predicted_new_particle_count") == predicted
        and reported.get("transport_trace_sha256") == raw.get("transport_trace_sha256")
        and reported.get("report_sha256") == raw.get("report_sha256")
        and reported.get("cloud_sha256") == raw.get("cloud_sha256")
    )


def _pre_gate_recomputation(
    numerical: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    pre_summary = numerical.get("pre_gates")
    if not isinstance(pre_summary, Mapping):
        raise ValueError("reported pre-gate summary is invalid")
    expected_summary_sha = _json_payload_sha256(
        "fluxv-v5h-cumulative-pre-gates-summary-v1", pre_summary
    )
    if raw.get("pre_gates_summary_sha256") != expected_summary_sha:
        raise ValueError("pre-gate raw evidence is not bound to its summary")
    decoded = _decode_raw_value("pre_gates", raw.get("pre_gates"))
    if not isinstance(decoded, Mapping):
        raise ValueError("decoded pre-gate evidence is invalid")

    m0_summary = pre_summary.get("m0_v1_physical_bitwise_parity")
    m1_summary = pre_summary.get("m1_three_and_four_release")
    m2_summary = pre_summary.get("m2_lifecycle_and_partial_activity")
    m0_raw = decoded.get("m0_v1_physical_bitwise_parity")
    m1_raw = decoded.get("m1_three_and_four_release")
    m2_raw = decoded.get("m2_lifecycle")
    if not all(
        isinstance(item, Mapping)
        for item in (m0_summary, m1_summary, m2_summary, m0_raw, m1_raw, m2_raw)
    ):
        raise ValueError("pre-gate summary/raw sections are incomplete")

    v1 = m0_raw.get("v1")
    v2 = m0_raw.get("v2")
    if not isinstance(v1, Mapping) or not isinstance(v2, Mapping):
        raise ValueError("M0 v1/v2 evidence is invalid")
    v2_release_rows = v2.get("release_raw")
    v2_reported_releases = m0_summary.get("v2_sequence", {}).get("releases")
    if (
        not isinstance(v2_release_rows, list)
        or len(v2_release_rows) != 1
        or not isinstance(v2_reported_releases, list)
        or len(v2_reported_releases) != 1
        or not isinstance(v2_release_rows[0], Mapping)
        or not isinstance(v2_reported_releases[0], Mapping)
    ):
        raise ValueError("M0 release evidence is invalid")
    m0_array_pairs = (
        ("positions_gp1_m", "positions_gp1_m"),
        ("gamma_vector_m3_per_s", "gamma_vector_m3_per_s"),
        ("sigma_m", "sigma_m"),
    )
    m0_arrays_equal = all(
        isinstance(v1.get(v1_name), np.ndarray)
        and isinstance(v2.get(v2_name), np.ndarray)
        and np.array_equal(v1[v1_name], v2[v2_name])
        for v1_name, v2_name in m0_array_pairs
    )
    v1_frontier = v1.get("frontier_positions_gp1_m")
    v2_frontier = v2_release_rows[0].get("frontier_positions_gp1_m")
    m0_frontier_equal = bool(
        isinstance(v1_frontier, np.ndarray)
        and isinstance(v2_frontier, np.ndarray)
        and np.array_equal(v1_frontier, v2_frontier)
    )
    m0_identity_equal = bool(
        v1.get("particle_ids") == v2.get("particle_ids")
        and v1.get("lineage") == v2.get("lineage")
    )
    m0_release_passed = _release_raw_passed(
        v2_release_rows[0],
        v2_reported_releases[0],
        expected_substeps=1,
        expected_spacing_m=TARGET_SPACINGS_M[0],
        expected_step=1,
        expected_mode="first",
        expected_incidence_rad=FIRST_ALPHA_RAD,
        expected_node_count=3,
    )
    v1_edge_metrics = m0_summary.get("v1_edge_metrics")
    if isinstance(v1_edge_metrics, Mapping):
        v1_edge_with_activity = {"active": True, **v1_edge_metrics}
        v1_edge_derived = _edge_metrics_raw_passed(
            v1_edge_with_activity,
            expected_active=True,
            expected_particle_count=int(m0_summary.get("predicted_particle_count", -1)),
        )
    else:
        v1_edge_derived = False
    m0_derived = bool(
        m0_arrays_equal
        and m0_frontier_equal
        and m0_identity_equal
        and m0_release_passed
        and v1_edge_derived
    )

    def release_sequence_evidence(
        raw_section: Mapping[str, Any],
        summary_section: Mapping[str, Any],
        expected_steps: list[int],
        expected_modes: list[str],
        expected_alphas: list[float],
    ) -> tuple[bool, list[Mapping[str, Any]]]:
        raw_releases = raw_section.get("release_raw")
        reported_releases = summary_section.get("releases")
        if (
            not isinstance(raw_releases, list)
            or not isinstance(reported_releases, list)
            or len(raw_releases) != len(expected_steps)
            or len(reported_releases) != len(expected_steps)
            or not all(isinstance(item, Mapping) for item in raw_releases)
            or not all(isinstance(item, Mapping) for item in reported_releases)
        ):
            return False, []
        passed = all(
            _release_raw_passed(
                raw_release,
                reported_release,
                expected_substeps=1,
                expected_spacing_m=TARGET_SPACINGS_M[0],
                expected_step=step,
                expected_mode=mode,
                expected_incidence_rad=alpha,
                expected_node_count=3,
            )
            for step, mode, alpha, raw_release, reported_release in zip(
                expected_steps,
                expected_modes,
                expected_alphas,
                raw_releases,
                reported_releases,
                strict=True,
            )
        )
        return bool(passed), list(raw_releases)

    m1_sequence_passed, m1_releases = release_sequence_evidence(
        m1_raw,
        m1_summary,
        [1, 2, 3, 4],
        ["first", "continuous", "continuous", "continuous"],
        [
            FIRST_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
            M1_CONTINUOUS_ALPHA_RAD,
        ],
    )
    m1_counts = [row.get("total_particle_count") for row in m1_releases]
    m1_derived = bool(
        m1_sequence_passed
        and len(m1_counts) == 4
        and all(
            isinstance(count, int) and not isinstance(count, bool)
            for count in m1_counts
        )
        and all(left < right for left, right in zip(m1_counts[:-1], m1_counts[1:]))
        and m1_releases[2].get("release_slices")[-1].get("source_step_index") == 3
        and m1_releases[3].get("release_slices")[-1].get("source_step_index") == 4
    )

    m2_sequence_passed, m2_releases = release_sequence_evidence(
        m2_raw,
        m2_summary,
        [1, 2, 3, 4],
        ["first", "inactive", "restart", "continuous"],
        [FIRST_ALPHA_RAD, 0.0, CONTINUOUS_ALPHA_RAD, CONTINUOUS_ALPHA_RAD],
    )
    if len(m2_releases) == 4:
        m2_totals = [row.get("total_particle_count") for row in m2_releases]
        m2_new = [row.get("new_particle_count") for row in m2_releases]
        try:
            m2_final_steps, _ = _slice_steps_and_count(
                m2_releases[-1].get("release_slices")
            )
        except ValueError:
            m2_final_steps = []
    else:
        m2_totals = []
        m2_new = []
        m2_final_steps = []
    partial = m2_summary.get("partial_activity_pre_gates")
    raw_partial = m2_raw.get("partial_activity_pre_gates")
    expected_partial_patterns = {"split", "shrink", "grow"}
    if isinstance(raw_partial, Mapping) and isinstance(
        raw_partial.get("patterns"), list
    ):
        partial_rows = raw_partial["patterns"]
        partial_names = {
            row.get("pattern") for row in partial_rows if isinstance(row, Mapping)
        }
        partial_raw_passed = bool(
            len(partial_rows) == 3
            and partial_names == expected_partial_patterns
            and raw_partial.get("policy")
            == "no interpolation across an activity boundary"
            and all(
                isinstance(row, Mapping)
                and row.get("expected_gate")
                == "underresolved_activity_boundary_rejected"
                and row.get("exception_type") == "ValueError"
                and isinstance(row.get("exception_message"), str)
                and "mixed activity/mode" in row["exception_message"]
                and row.get("adapter_state_unchanged") is True
                for row in partial_rows
            )
        )
    else:
        partial_raw_passed = False
    m2_derived = bool(
        m2_sequence_passed
        and len(m2_totals) == 4
        and m2_new[1] == 0
        and m2_totals[1] == m2_totals[0]
        and m2_totals[2] == m2_totals[1] + m2_new[2]
        and m2_totals[3] == m2_totals[2] + m2_new[3]
        and m2_final_steps == [1, 3, 4]
        and isinstance(partial, Mapping)
        and partial_raw_passed
        and raw_partial == partial
    )
    raw_derived_passed = bool(m0_derived and m1_derived and m2_derived)
    reported_match = bool(
        m0_summary.get("passed") is m0_derived
        and m1_summary.get("passed") is m1_derived
        and m2_summary.get("passed") is m2_derived
        and pre_summary.get("passed") is raw_derived_passed
    )
    return {
        "m0_v1_physical_bitwise_parity_passed": m0_derived,
        "m1_three_and_four_release_passed": m1_derived,
        "m2_lifecycle_and_partial_activity_passed": m2_derived,
        "reported_values_match_raw_recomputation": reported_match,
        "passed": bool(raw_derived_passed and reported_match),
    }


def _configuration_from_raw_record(
    raw_record: Mapping[str, Any],
    summary_row: Mapping[str, Any],
) -> tuple[_ConfigurationResult, bool]:
    expected_summary_sha = _json_payload_sha256(
        "fluxv-v5h-cumulative-configuration-summary-v1", summary_row
    )
    if raw_record.get("configuration_summary_sha256") != expected_summary_sha:
        raise ValueError("M3 raw configuration is not bound to its summary")
    evidence = _decode_raw_value("M3.evidence", raw_record.get("evidence"))
    if not isinstance(evidence, Mapping):
        raise ValueError("decoded M3 evidence is invalid")
    frontier = evidence.get("frontier_minus_latest_birth_gp1_m")
    probes = evidence.get("fixed_probe_induced_velocity_gp1_m_per_s")
    positions = evidence.get("positions_gp1_m")
    gamma = evidence.get("gamma_vector_m3_per_s")
    sigma = evidence.get("sigma_m")
    if not all(
        isinstance(item, np.ndarray)
        for item in (frontier, probes, positions, gamma, sigma)
    ):
        raise ValueError("M3 final raw arrays are incomplete")
    total = summary_row.get("final_particle_count")
    realized = summary_row.get("realized_spacing_max_m")
    releases = evidence.get("release_raw")
    reported_releases = summary_row.get("releases")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or isinstance(realized, bool)
        or not isinstance(realized, (int, float))
        or not isinstance(releases, list)
        or not isinstance(reported_releases, list)
        or len(releases) != M3_RELEASE_COUNT
        or len(reported_releases) != M3_RELEASE_COUNT
        or not all(isinstance(item, Mapping) for item in releases)
        or not all(isinstance(item, Mapping) for item in reported_releases)
    ):
        raise ValueError("M3 release evidence is incomplete")
    substeps = int(summary_row["transport_substeps"])
    expected_modes = ("first", "continuous", "continuous")
    expected_alphas = (FIRST_ALPHA_RAD, CONTINUOUS_ALPHA_RAD, CONTINUOUS_ALPHA_RAD)
    release_mechanics = all(
        _release_raw_passed(
            raw_release,
            reported_release,
            expected_substeps=substeps,
            expected_spacing_m=float(summary_row["target_spacing_m"]),
            expected_step=step,
            expected_mode=mode,
            expected_incidence_rad=alpha,
            expected_node_count=GateConfig().span_cells + 1,
        )
        for step, mode, alpha, raw_release, reported_release in zip(
            range(1, M3_RELEASE_COUNT + 1),
            expected_modes,
            expected_alphas,
            releases,
            reported_releases,
            strict=True,
        )
    )
    final_release = releases[-1]
    final_arrays_match = bool(
        np.array_equal(positions, final_release["positions_gp1_m"])
        and np.array_equal(gamma, final_release["gamma_vector_m3_per_s"])
        and np.array_equal(sigma, final_release["sigma_m"])
        and np.array_equal(frontier, final_release["frontier_minus_latest_birth_gp1_m"])
        and np.array_equal(
            probes, final_release["fixed_probe_induced_velocity_gp1_m_per_s"]
        )
    )
    mechanics_passed = bool(
        release_mechanics
        and final_arrays_match
        and positions.shape == (total, 3)
        and gamma.shape == (total, 3)
        and sigma.shape == (total,)
        and np.all(np.isfinite(sigma))
        and np.all(sigma > 0.0)
        and evidence.get("cloud_sha256") == summary_row.get("final_cloud_sha256")
        and evidence.get("report_sha256") == summary_row.get("final_report_sha256")
    )
    return (
        _ConfigurationResult(
            summary=dict(summary_row),
            raw=dict(evidence),
            frontier_minus_latest_birth=np.ascontiguousarray(frontier),
            fixed_probe_induced_velocity=np.ascontiguousarray(probes),
            particle_count=total,
            realized_spacing_max_m=float(realized),
        ),
        mechanics_passed,
    )


def _runtime_boundary_from_evidence(value: object) -> tuple[bool, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime boundary evidence must be an object")
    instrumented = value.get("instrumented_access")
    measured_counts = value.get("measured_component_counts")
    direct_imports = value.get("direct_forbidden_imports")
    surface_paths = value.get("surface_quantity_key_paths")
    if (
        not isinstance(instrumented, Mapping)
        or not isinstance(measured_counts, Mapping)
        or not isinstance(direct_imports, list)
        or not isinstance(surface_paths, list)
    ):
        raise ValueError("runtime boundary evidence is structurally incomplete")
    guarded_counts = instrumented.get("guarded_call_counts")
    guarded_symbols = instrumented.get("guarded_symbols")
    target_paths = instrumented.get("target_observation_read_paths")
    eager_definitions = value.get(
        "package_init_eager_target_or_ptera_definitions_loaded"
    )
    if (
        not isinstance(guarded_counts, Mapping)
        or not isinstance(guarded_symbols, list)
        or not isinstance(target_paths, list)
        or not isinstance(eager_definitions, bool)
    ):
        raise ValueError("instrumented runtime evidence is incomplete")
    expected_measured_keys = {
        "feedback_call_count",
        "parent_write_count",
        "surface_channel_write_count",
    }
    measured_valid = bool(
        set(measured_counts) == expected_measured_keys
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count == 0
            for count in measured_counts.values()
        )
    )
    guarded_valid = bool(
        all(
            isinstance(count, int) and not isinstance(count, bool) and count == 0
            for count in guarded_counts.values()
        )
        and all(isinstance(symbol, str) and symbol for symbol in guarded_symbols)
        and (bool(guarded_counts) is bool(guarded_symbols))
        and (not eager_definitions or bool(guarded_symbols))
    )
    target_count = instrumented.get("target_observation_read_count")
    derived = bool(
        instrumented.get("open_audit_active_during_gate") is True
        and guarded_valid
        and isinstance(target_count, int)
        and not isinstance(target_count, bool)
        and target_count == 0
        and target_paths == []
        and direct_imports == []
        and surface_paths == []
        and measured_valid
    )
    reported_consistent = bool(
        instrumented.get("passed") is derived and value.get("passed") is derived
    )
    return derived, reported_consistent


def _recompute_gates(
    semantic_result: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if semantic_result.get("schema_id") != RUN_SCHEMA_ID:
        raise ValueError("semantic result schema is unsupported")
    if semantic_result.get("run_id") != RUN_ID:
        raise ValueError("semantic result run ID is unsupported")
    if raw.get("schema_id") != RAW_REFINEMENT_SCHEMA_ID:
        raise ValueError("raw refinement schema is unsupported")
    if raw.get("run_id") != RUN_ID:
        raise ValueError("raw refinement run ID is unsupported")
    numerical = semantic_result.get("summary")
    if not isinstance(numerical, Mapping):
        raise ValueError("semantic numerical summary is invalid")
    config = numerical.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("semantic gate config is invalid")
    _validate_config(
        GateConfig(
            span_cells=config.get("span_cells"),
            target_spacings_m=tuple(config.get("target_spacings_m", ())),
            transport_substeps=tuple(config.get("transport_substeps", ())),
            m1_smoke_release_count=config.get("m1_smoke_release_count"),
            m3_release_count=config.get("m3_release_count"),
            particles_per_release_cap=config.get("particles_per_release_cap"),
            cumulative_particle_cap=config.get("cumulative_particle_cap"),
            configuration_cap=config.get("configuration_cap"),
            transport_substeps_cap=config.get("transport_substeps_cap"),
        )
    )
    pre = _pre_gate_recomputation(numerical, raw)
    summary_index = _configuration_summary_index(semantic_result)
    raw_rows = raw.get("m3_configurations")
    if not isinstance(raw_rows, list):
        raise ValueError("raw M3 configurations must be a list")
    rows: dict[tuple[str, float, int], _ConfigurationResult] = {}
    mechanics: dict[tuple[str, float, int], bool] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("raw M3 configuration row must be an object")
        key = _raw_configuration_key(raw_row)
        if key in rows or key not in summary_index:
            raise ValueError("raw M3 configuration key is invalid or duplicated")
        rows[key], mechanics[key] = _configuration_from_raw_record(
            raw_row, summary_index[key]
        )
    if set(rows) != set(summary_index):
        raise ValueError("raw M3 configurations do not cover their summary")
    reported_match = all(
        summary_index[key].get("passed") is mechanics[key] for key in rows
    )

    execution_mode = numerical.get("execution_mode")
    if execution_mode == "minimal_smoke_artifact_validation":
        expected_keys: set[tuple[str, float, int]] = set()
        if numerical.get("m3") is not None:
            raise ValueError("minimal-smoke artifact unexpectedly reports M3")
    elif execution_mode == "full_frozen_36_configuration_gate":
        expected_keys = {
            (geometry, spacing, substeps)
            for geometry in CASE_GEOMETRIES
            for spacing in TARGET_SPACINGS_M
            for substeps in TRANSPORT_SUBSTEPS
        }
    else:
        raise ValueError("semantic execution mode is unsupported")
    raw_coverage_valid = bool(
        set(rows) == expected_keys and len(rows) == len(expected_keys)
    )
    if not raw_coverage_valid:
        raise ValueError("raw/summary configuration coverage violates execution mode")
    coverage_passed = bool(
        execution_mode == "full_frozen_36_configuration_gate"
        and len(rows) == MAX_CONFIGURATIONS
    )

    recomputed_geometries: list[dict[str, Any]] = []
    if expected_keys:
        m3_summary = numerical.get("m3")
        if not isinstance(m3_summary, Mapping):
            raise ValueError("full gate has no M3 summary")
        reported_geometries = {
            row["geometry"]: row for row in m3_summary.get("geometries", [])
        }
        if set(reported_geometries) != set(CASE_GEOMETRIES):
            raise ValueError("M3 geometries do not cover the frozen family")
        for geometry in CASE_GEOMETRIES:
            time_gates = []
            for spacing in TARGET_SPACINGS_M:
                family = [
                    rows[(geometry, spacing, step)] for step in TRANSPORT_SUBSTEPS
                ]
                time_gates.append(
                    {
                        "target_spacing_m": spacing,
                        **_time_refinement_gate(family),
                    }
                )
            h_family = [
                rows[(geometry, spacing, TRANSPORT_SUBSTEPS[-1])]
                for spacing in TARGET_SPACINGS_M
            ]
            h_gate = _h_refinement_gate(h_family)
            configuration_mechanics = all(
                mechanics[(geometry, spacing, step)]
                for spacing in TARGET_SPACINGS_M
                for step in TRANSPORT_SUBSTEPS
            )
            geometry_passed = bool(
                configuration_mechanics
                and all(gate["passed"] for gate in time_gates)
                and h_gate["passed"]
            )
            reported = reported_geometries[geometry]
            reported_match = bool(
                reported_match
                and reported.get("time_refinement_by_h") == time_gates
                and reported.get("h_refinement_at_finest_time") == h_gate
                and reported.get("configuration_mechanics_passed")
                is configuration_mechanics
                and reported.get("time_refinement_passed")
                is all(gate["passed"] for gate in time_gates)
                and reported.get("h_refinement_passed") is h_gate["passed"]
                and reported.get("passed") is geometry_passed
            )
            recomputed_geometries.append(
                {
                    "geometry": geometry,
                    "configuration_mechanics_passed": configuration_mechanics,
                    "time_refinement_by_h": time_gates,
                    "h_refinement_at_finest_time": h_gate,
                    "passed": geometry_passed,
                }
            )
        m3_passed = bool(
            recomputed_geometries
            and all(row["passed"] for row in recomputed_geometries)
        )
        reported_match = bool(
            reported_match
            and m3_summary.get("configuration_count") == MAX_CONFIGURATIONS
            and m3_summary.get("passed") is m3_passed
        )
    else:
        m3_passed = False

    reported_gate = numerical.get("gate_summary")
    runtime_evidence = semantic_result.get("runtime_boundary_evidence")
    if not isinstance(reported_gate, Mapping) or not isinstance(
        runtime_evidence, Mapping
    ):
        raise ValueError("reported top gate/runtime evidence is invalid")
    source_stable = bool(reported_gate.get("declared_source_snapshot_stable_passed"))
    runtime_passed, runtime_reported_consistent = _runtime_boundary_from_evidence(
        runtime_evidence
    )
    top_passed = bool(
        pre["passed"]
        and coverage_passed
        and m3_passed
        and source_stable
        and runtime_passed
        and execution_mode == "full_frozen_36_configuration_gate"
    )
    recomputed_gate = {
        "pre_gates_passed": bool(pre["passed"]),
        "configuration_coverage_passed": coverage_passed,
        "m3_mechanics_and_refinement_passed": m3_passed,
        "declared_source_snapshot_stable_passed": source_stable,
        "runtime_boundary_passed": runtime_passed,
        "passed": top_passed,
        "stop_required": not top_passed,
    }
    reported_match = bool(
        reported_match
        and runtime_reported_consistent
        and reported_gate == recomputed_gate
        and numerical.get("passed") is top_passed
        and semantic_result.get("passed") is top_passed
        and semantic_result.get("status")
        == ("go_m3_mechanics_only" if top_passed else "stop")
    )
    return {
        "schema_id": RECOMPUTED_GATE_SCHEMA_ID,
        "run_id": RUN_ID,
        "execution_mode": execution_mode,
        "derivation_scope": (
            "configuration coverage, pre-gate array parity/lifecycle, time and h "
            "refinement, release counters/slices/traces/digests, and the top gate "
            "are reconstructed from strict raw evidence; source/placement/ribbon "
            "attestations, the partial-boundary exception rows, runtime "
            "instrumentation, and within-run source stability are measured, "
            "hash-bound semantic inputs rather than independently replayed here"
        ),
        "raw_refinement_sha256": _json_payload_sha256(
            "fluxv-v5h-cumulative-raw-refinement-v1", raw
        ),
        "reported_summary_sha256": _json_payload_sha256(
            "fluxv-v5h-cumulative-reported-summary-v1", semantic_result
        ),
        "pre_gates": pre,
        "configuration_count": len(rows),
        "configuration_coverage_passed": coverage_passed,
        "geometries": recomputed_geometries,
        "gate_summary": recomputed_gate,
        "reported_values_match_raw_recomputation": reported_match,
        "passed": bool(top_passed and reported_match),
    }


def recompute_gates_from_artifacts(output_dir: Path) -> dict[str, Any]:
    """Reload disk evidence and reconstruct every disk-recomputable gate."""

    output = Path(output_dir)
    summary = _load_strict_json(output / "summary.json")
    raw = _load_strict_json(output / RAW_REFINEMENT_ARTIFACT)
    return _recompute_gates(summary, raw)


def _artifact_readme(semantic_result: Mapping[str, Any]) -> str:
    passed = bool(semantic_result.get("passed"))
    execution_mode = semantic_result.get("summary", {}).get("execution_mode")
    verdict = "GO for bounded M0--M3 mechanics" if passed else "STOP"
    return (
        "# FluxV v5h cumulative-cloud mechanical gate\n\n"
        f"Verdict: **{verdict}**.\n\n"
        f"Execution mode: `{execution_mode}`. This is an observation-free, "
        "noncanonical mechanical artifact. Node-owned ribbon topology controls "
        "release ownership; one cumulative rVPM field advances all old and new "
        "particles. Ptera remains the external surface-aerodynamics owner and "
        "was not invoked by this gate.\n\n"
        "A STOP is a complete result, not an incomplete run. The frozen "
        "thresholds are not relaxed: in particular, a refinement ratio below "
        "the registered minimum remains a hard failure. No target-paper "
        "comparison is performed.\n\n"
        "`raw_refinement.json` stores pre-gate and per-configuration arrays with "
        "dtype, shape, values, and SHA-256, plus release slices, traces, counters, "
        "and producer digests. `recomputed_gates.json` is reconstructed from the "
        "strictly reloaded disk evidence. The seven deterministic semantic files "
        "are separated from UUID, clocks, argv, paths, git state, environment, "
        "package versions, and observed modules in the run-specific files.\n"
    )


def _semantic_manifest_core(files: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_id": SEMANTIC_MANIFEST_SCHEMA_ID,
        "run_id": RUN_ID,
        "files": dict(sorted(files.items())),
    }


def verify_semantic_manifest(output_dir: Path) -> str:
    """Verify deterministic semantic payloads and return their common digest."""

    output = Path(output_dir)
    manifest = _load_strict_json(output / SEMANTIC_MANIFEST_ARTIFACT)
    if manifest.get("schema_id") != SEMANTIC_MANIFEST_SCHEMA_ID:
        raise ValueError("semantic manifest schema is unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != SEMANTIC_PAYLOAD_FILENAMES:
        raise ValueError("semantic manifest does not contain the exact payload set")
    for name, expected in files.items():
        if not isinstance(name, str) or not _valid_sha256(expected):
            raise ValueError("semantic manifest file row is invalid")
        if _sha256_file(output / name) != expected:
            raise ValueError(f"semantic payload SHA-256 mismatch: {name}")
    expected_digest = _json_payload_sha256(
        "fluxv-v5h-cumulative-semantic-result-v1",
        _semantic_manifest_core(files),
    )
    if manifest.get("semantic_result_sha256") != expected_digest:
        raise ValueError("semantic result SHA-256 is inconsistent")
    return expected_digest


def _validate_run_provenance(
    provenance: Mapping[str, Any],
    output: Path,
) -> None:
    required = {
        "schema_id",
        "run_uuid",
        "numerical_started_utc",
        "numerical_completed_utc",
        "provenance_captured_utc",
        "process_argv",
        "process_command",
        "python_sys_argv",
        "cwd",
        "output_dir",
        "environment",
        "git",
        "observed_repo_modules",
    }
    if set(provenance) != required:
        raise ValueError("run provenance fields are incomplete or unknown")
    if provenance.get("schema_id") != RUN_PROVENANCE_SCHEMA_ID:
        raise ValueError("run provenance schema is unsupported")
    run_uuid = provenance.get("run_uuid")
    if not isinstance(run_uuid, str) or _validated_run_uuid(run_uuid) != run_uuid:
        raise ValueError("run provenance UUID is invalid")
    started = datetime.fromisoformat(
        _validated_utc(
            "numerical_started_utc", provenance["numerical_started_utc"]
        ).replace("Z", "+00:00")
    )
    completed = datetime.fromisoformat(
        _validated_utc(
            "numerical_completed_utc", provenance["numerical_completed_utc"]
        ).replace("Z", "+00:00")
    )
    captured = datetime.fromisoformat(
        _validated_utc(
            "provenance_captured_utc", provenance["provenance_captured_utc"]
        ).replace("Z", "+00:00")
    )
    process_argv = provenance.get("process_argv")
    if (
        not started <= completed <= captured
        or not isinstance(process_argv, list)
        or not process_argv
        or any(not isinstance(item, str) for item in process_argv)
        or provenance.get("process_command") != shlex.join(process_argv)
        or not isinstance(provenance.get("python_sys_argv"), list)
        or not isinstance(provenance.get("environment"), Mapping)
        or not isinstance(provenance.get("git"), Mapping)
        or not isinstance(provenance.get("observed_repo_modules"), Mapping)
    ):
        raise ValueError("run provenance structural evidence is invalid")
    if provenance.get("output_dir") != str(output.resolve(strict=False)):
        raise ValueError("run provenance output directory is not the write target")


def _source_manifest(
    declared_sources: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": DECLARED_SOURCE_MANIFEST_SCHEMA_ID,
        "run_id": RUN_ID,
        "closure_scope": (
            "declared numerical and artifact source closure: cumulative runner, "
            "read-only v1 helper runner, cumulative/passive cores, DVM ribbon, "
            "source and placement adapters, edge/reference/transport kernels, "
            "LDVM section/Fourier/induction dependencies, package initialiser, "
            "and pyproject contract"
        ),
        "runtime_import_closure_complete": False,
        "observed_runtime_modules_location": (
            "run_manifest.json.provenance.observed_repo_modules"
        ),
        "verified_against_current_files_at_artifact_write": True,
        "files": dict(declared_sources),
    }


def _validate_staged_bundle(
    staging: Path,
    expected_recomputed: Mapping[str, Any],
    expected_semantic_digest: str,
) -> None:
    actual_names = {path.name for path in staging.iterdir() if path.is_file()}
    if actual_names != ARTIFACT_FILENAMES:
        raise RuntimeError("staged artifact bundle does not contain exactly 11 files")
    for path in staging.glob("*.json"):
        _load_strict_json(path)
    result_manifest = _load_strict_json(staging / "result_manifest.json")
    if result_manifest.get("schema_id") != RESULT_MANIFEST_SCHEMA_ID:
        raise RuntimeError("result manifest schema is unsupported")
    files = result_manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("result manifest hash closure is invalid")
    expected_result_files = ARTIFACT_FILENAMES - {
        "result_manifest.json",
        "SHA256SUMS",
    }
    if set(files) != expected_result_files:
        raise RuntimeError("result manifest file closure is incomplete")
    for name, expected in files.items():
        if not _valid_sha256(expected) or _sha256_file(staging / name) != expected:
            raise RuntimeError(f"result manifest SHA-256 mismatch: {name}")
    checksum_rows: dict[str, str] = {}
    for line in (staging / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or parts[1] in checksum_rows:
            raise RuntimeError("SHA256SUMS row is invalid or duplicated")
        checksum_rows[parts[1]] = parts[0]
    expected_checksum_files = ARTIFACT_FILENAMES - {"SHA256SUMS"}
    if set(checksum_rows) != expected_checksum_files or any(
        not _valid_sha256(digest) or _sha256_file(staging / name) != digest
        for name, digest in checksum_rows.items()
    ):
        raise RuntimeError("external SHA256SUMS closure is inconsistent")
    reloaded = recompute_gates_from_artifacts(staging)
    if reloaded != expected_recomputed:
        raise RuntimeError("on-disk gate recomputation is inconsistent")
    if verify_semantic_manifest(staging) != expected_semantic_digest:
        raise RuntimeError("semantic manifest verification is inconsistent")


def write_run_artifacts(
    result: dict[str, Any],
    output_dir: Path,
    *,
    run_provenance: Mapping[str, Any],
) -> Path:
    """Atomically write one strict 11-file GO or STOP evidence bundle."""

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if not output.name:
        raise ValueError("output directory must have an explicit final component")
    _validate_run_provenance(run_provenance, output)
    if _surface_quantity_key_paths(result):
        raise ValueError("run result contains a forbidden surface quantity key")
    semantic_result, raw_refinement = _semantic_result(result)
    declared_sources = semantic_result.get("declared_semantic_source_sha256")
    if declared_sources != _source_hashes():
        raise RuntimeError("declared source hashes are stale at artifact write")
    recomputed = _recompute_gates(semantic_result, raw_refinement)
    if not recomputed["reported_values_match_raw_recomputation"]:
        raise RuntimeError("reported gates disagree with strict raw recomputation")
    if bool(semantic_result.get("passed")) != bool(recomputed["passed"]):
        raise RuntimeError("reported top verdict disagrees with recomputation")

    source_manifest = _source_manifest(declared_sources)
    numerical = semantic_result["summary"]
    metrics = {
        "schema_id": "fluxv-v5h-cumulative-cloud-metrics-v1",
        "run_id": RUN_ID,
        "execution_mode": numerical["execution_mode"],
        "pre_gates": numerical["pre_gates"],
        "m3": numerical["m3"],
        "gate_summary": numerical["gate_summary"],
    }
    semantic_payloads: dict[str, bytes] = {
        "summary.json": _json_text(semantic_result).encode("utf-8"),
        "metrics.json": _json_text(metrics).encode("utf-8"),
        RAW_REFINEMENT_ARTIFACT: _json_text(raw_refinement).encode("utf-8"),
        RECOMPUTED_GATE_ARTIFACT: _json_text(recomputed).encode("utf-8"),
        "source_manifest.json": _json_text(source_manifest).encode("utf-8"),
        "README.md": _artifact_readme(semantic_result).encode("utf-8"),
    }
    semantic_files = {
        name: sha256(content).hexdigest()
        for name, content in sorted(semantic_payloads.items())
    }
    semantic_core = _semantic_manifest_core(semantic_files)
    semantic_result_sha256 = _json_payload_sha256(
        "fluxv-v5h-cumulative-semantic-result-v1", semantic_core
    )
    semantic_manifest = {
        **semantic_core,
        "semantic_result_sha256": semantic_result_sha256,
        "deterministic_file_count_including_manifest": 7,
        "scope": (
            "six non-self-referential payload hashes plus this deterministic "
            "manifest; invocation provenance and elapsed time are excluded"
        ),
        "excluded_run_specific_files": [
            "run.log",
            "run_manifest.json",
            "result_manifest.json",
            "SHA256SUMS",
        ],
    }
    semantic_payloads[SEMANTIC_MANIFEST_ARTIFACT] = _json_text(
        semantic_manifest
    ).encode("utf-8")

    run_manifest = {
        "schema_id": "fluxv-v5h-cumulative-cloud-run-manifest-v1",
        "run_id": RUN_ID,
        "tier": RUN_TIER,
        "status": semantic_result["status"],
        "semantic_result_sha256": semantic_result_sha256,
        "config": numerical["config"],
        "provenance": dict(run_provenance),
        "runtime_boundary_evidence": semantic_result["runtime_boundary_evidence"],
        "declared_source_scope": source_manifest["closure_scope"],
        "observation_access": "none",
        "target_case_branch": "none",
    }
    run_uuid = str(run_provenance["run_uuid"])
    log_text = (
        f"run_id={RUN_ID}\n"
        f"run_uuid={run_uuid}\n"
        f"tier={RUN_TIER}\n"
        f"execution_mode={numerical['execution_mode']}\n"
        f"status={semantic_result['status']}\n"
        f"passed={str(bool(semantic_result['passed'])).lower()}\n"
        f"configuration_count={recomputed['configuration_count']}\n"
        f"numerical_started_utc={run_provenance['numerical_started_utc']}\n"
        f"numerical_completed_utc={run_provenance['numerical_completed_utc']}\n"
        f"output_dir={run_provenance['output_dir']}\n"
        f"semantic_result_sha256={semantic_result_sha256}\n"
        "observation_access=none\n"
        "target_comparison=prohibited\n"
    )
    payloads: dict[str, bytes] = {
        **semantic_payloads,
        "run_manifest.json": _json_text(run_manifest).encode("utf-8"),
        "run.log": log_text.encode("utf-8"),
    }
    result_manifest = {
        "schema_id": RESULT_MANIFEST_SCHEMA_ID,
        "run_id": RUN_ID,
        "run_uuid": run_uuid,
        "status": semantic_result["status"],
        "semantic_result_sha256": semantic_result_sha256,
        "files": {
            name: sha256(content).hexdigest()
            for name, content in sorted(payloads.items())
        },
    }
    payloads["result_manifest.json"] = _json_text(result_manifest).encode("utf-8")
    checksums = "".join(
        f"{sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(payloads.items())
    )
    payloads["SHA256SUMS"] = checksums.encode("ascii")
    if set(payloads) != ARTIFACT_FILENAMES:
        raise RuntimeError("artifact assembly did not produce exactly 11 files")
    if declared_sources != _source_hashes():
        raise RuntimeError("declared source hashes changed during artifact assembly")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        _validate_staged_bundle(staging, recomputed, semantic_result_sha256)
        if output.exists():
            raise FileExistsError(f"refusing to replace output directory: {output}")
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--minimal-smoke",
        action="store_true",
        help=(
            "exercise M0--M2 plus the artifact pipeline only; emits a complete "
            "STOP bundle without running the frozen 36-configuration M3 grid"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    numerical_started_utc = _utc_now()
    result = run_cumulative_cloud_gate(minimal_smoke_only=args.minimal_smoke)
    numerical_completed_utc = _utc_now()
    provenance = capture_run_provenance(
        output_dir=args.output_dir,
        numerical_started_utc=numerical_started_utc,
        numerical_completed_utc=numerical_completed_utc,
    )
    output = write_run_artifacts(
        result,
        args.output_dir,
        run_provenance=provenance,
    )
    print(
        json.dumps(
            {"artifact": str(output), "status": result["status"]},
            allow_nan=False,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


__all__ = [
    "GateConfig",
    "capture_run_provenance",
    "recompute_gates_from_artifacts",
    "run_cumulative_cloud_gate",
    "run_minimal_smoke",
    "verify_semantic_manifest",
    "write_run_artifacts",
]


if __name__ == "__main__":
    main()
