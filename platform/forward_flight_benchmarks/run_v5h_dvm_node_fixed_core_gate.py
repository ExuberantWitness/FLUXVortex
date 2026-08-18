"""Run the observation-free v5h node-DVM fixed-core auxiliary gate.

This runner is deliberately narrower than a FluxV performance experiment.  A
cell-centre LDVM owns the circulation of each ribbon cell, while an independent
LDVM at every shared span node supplies geometry-only placement evidence.  The
node facts are consumed by the attested node-placement adapter; this file never
interpolates sectional births or manufactures a node velocity.

The physical smoothing radius is registered before the run as
``lambda * U * dt / p``.  Midpoint deposition spacing is refined independently
at that fixed radius.  A small core scan is reported as an ineligible static
diagnostic and cannot select or change the registered radius.

Only the audited single-cloud/two-release v1 surface is exercised.  No surface
aerodynamic channel, parent write, feedback path, target observation, or paper
score is present.  A cumulative-cloud v2 remains a hard blocker.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
from importlib import util as importlib_util
from importlib.metadata import PackageNotFoundError, version
import json
from math import cos, fsum, radians, sin
import os
from pathlib import Path
import platform as runtime_platform
import shlex
import subprocess
import sys
import threading
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from fluxvortex.rvpm_edge_bridge import (
    DIAGNOSTIC_SHADOW_OWNER,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    ShadowBridgeResult,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.v5h_dvm_node_placement import (
    DVMNodePlacementCell,
    DVMNodePlacementResult,
    GP1NodeSectionFact,
    NodeLocalDVMPlacementAdapter,
    validate_live_dvm_node_placement_result,
)
from forward_flight_benchmarks.v5h_dvm_node_ribbon import (
    DVMPlaneToGP1Map,
    DVMRibbonShadowResult,
    DVMSpanCellSource,
    NodeOwnedDVMRibbonShadow,
    validate_live_dvm_ribbon_shadow_result,
)
from forward_flight_benchmarks.v5h_dvm_source import (
    DVMSourceEvent,
    V5hDVMSource,
    validate_dvm_source_event,
)
from forward_flight_benchmarks.v5h_passive_frontier_transport import (
    PASSIVE_FRONTIER_CONTINUATION_SCOPE,
    PassiveFrontierTransportReport,
    materialize_transported_particle_state,
    transport_passive_node_frontiers,
    validate_passive_frontier_transport_report,
)


FloatArray = NDArray[np.float64]
GateGeometry = Literal["straight", "taper", "twist"]

RUN_ID = "20260815_fluxv_v5h_dvm_node_birth_fixed_core_gate"
RUN_SCHEMA_ID = "fluxv-v5h-dvm-node-fixed-core-gate-v1"
RUN_TIER = "auxiliary/dev"
CASE_GEOMETRIES: tuple[GateGeometry, ...] = ("straight", "taper", "twist")

RAW_REFINEMENT_SCHEMA_ID = "fluxv-v5h-fixed-core-raw-refinement-v1"
RECOMPUTED_GATE_SCHEMA_ID = "fluxv-v5h-fixed-core-recomputed-gates-v1"
SEMANTIC_MANIFEST_SCHEMA_ID = "fluxv-v5h-fixed-core-semantic-manifest-v1"
RUN_PROVENANCE_SCHEMA_ID = "fluxv-v5h-fixed-core-run-provenance-v1"
DECLARED_SOURCE_MANIFEST_SCHEMA_ID = (
    "fluxv-v5h-fixed-core-declared-semantic-source-manifest-v2"
)
RESULT_MANIFEST_SCHEMA_ID = "fluxv-v5h-fixed-core-result-manifest-v2"

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
    {"force", "load", "lift", "drag", "pressure", "polar", "score"}
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

REFERENCE_SPEED_M_PER_S = 2.0
ROOT_CHORD_M = 0.20
SPAN_M = 0.60
PHYSICAL_RELEASE_DT_S = 0.02
SOURCE_PARTICLES_PER_RELEASE = 1
PIVOT_FRACTION_CHORD = 0.25
FIRST_ALPHA_RAD = radians(35.0)
CONTINUOUS_ALPHA_RAD = radians(55.0)
LESP_CRITICAL = 0.18
REYNOLDS = 30_000.0
SOURCE_SETTINGS = LDVMSectionSettings(ndiv=24, naterm=8, max_wake_steps=48)
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

SOURCE_TRANSFER_SPACING_M = (
    REFERENCE_SPEED_M_PER_S * PHYSICAL_RELEASE_DT_S / SOURCE_PARTICLES_PER_RELEASE
)
SOURCE_TRANSFER_SIGMA_M = FROZEN_OVERLAP_LAMBDA * SOURCE_TRANSFER_SPACING_M

DEFAULT_SPAN_CELLS = 4
DEFAULT_TARGET_SPACINGS_M = (0.04, 0.02, 0.01, 0.005)
DEFAULT_TRANSPORT_SUBSTEPS = (1, 2, 4)
CORE_SCAN_MULTIPLIERS = (0.5, 1.0, 2.0)

MAX_SOURCE_KELVIN_M2_PER_S = 1.0e-10
MAX_RIBBON_RESIDUAL = 1.0e-12
MAX_VECTOR_ABS = 1.0e-14
MAX_VECTOR_REL = 1.0e-6
MAX_PARTICLE_INVARIANT_REL = 1.0e-6
MIN_TEMPORAL_TWO_LAYER_OVERLAP = 2.0
MAX_FINE_TIME_RELATIVE_DIFFERENCE = 1.0e-6
MIN_TIME_REFINEMENT_RATIO = 1.5
MAX_FINE_QUADRATURE_RELATIVE_DIFFERENCE = 0.01
MIN_QUADRATURE_REFINEMENT_RATIO = 1.5

LIMITATIONS = (
    "observation-free, noncanonical, two-release mechanical v1 only; "
    "node-local DVM sources are geometry-only; cell-centre DVM sources are "
    "circulation-only; Ptera/FluxV remains the external unique surface owner; "
    "no parent mutation, feedback, target branch, or performance claim; "
    "cumulative-cloud v2 and paper scoring remain blocked"
)


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Pre-registered controls for the fixed-core auxiliary gate."""

    span_cells: int = DEFAULT_SPAN_CELLS
    target_spacings_m: tuple[float, ...] = DEFAULT_TARGET_SPACINGS_M
    transport_substeps: tuple[int, ...] = DEFAULT_TRANSPORT_SUBSTEPS


@dataclass(frozen=True, slots=True)
class _NodeGeometry:
    node_id: str
    leading_edge_anchor_gp1_m: tuple[float, float, float]
    chord_m: float
    x_axis_gp1: tuple[float, float, float]
    z_axis_gp1: tuple[float, float, float]
    span_axis_gp1: tuple[float, float, float]
    patch_id: str
    frame_id: str


@dataclass(frozen=True, slots=True)
class _ConfigurationResult:
    summary: dict[str, Any]
    state_vector: FloatArray
    frontier_positions: FloatArray
    probe_velocity: FloatArray


class _ExplodingInput:
    """Sentinel proving that disabled producers do not inspect input values."""

    def __iter__(self) -> Any:
        raise AssertionError("disabled producer iterated an input")

    def __float__(self) -> float:
        raise AssertionError("disabled producer converted an input")

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError("disabled producer inspected an input attribute")


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
    """Measure forbidden target calls and target-data reads during one gate."""

    def __init__(self) -> None:
        self._patches: list[tuple[ModuleType, str, object]] = []
        self._read_paths: set[str] = set()
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
        self._read_paths.add(normalized)
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
        for module_name in (
            "forward_flight_benchmarks",
            "forward_flight_benchmarks.baik2012",
            "forward_flight_benchmarks.ptera_adapter",
        ):
            module = sys.modules.get(module_name)
            if not isinstance(module, ModuleType):
                continue
            for symbol in sorted(vars(module)):
                if symbol.startswith("build_"):
                    self._guard_module_symbol(module, symbol, "target_builder")
            for symbol in ("run_baik_old_fluxv", "run_model"):
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


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = sha256(b"fluxv-v5h-fixed-core-array-bundle-v1\0")
    for array_like in arrays:
        array = np.ascontiguousarray(np.asarray(array_like, dtype="<f8"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_payload_sha256(domain: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


def _raw_array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError("raw refinement arrays must be finite")
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "values": array.tolist(),
        "sha256": _array_sha256(array),
    }


def _array_from_raw_record(name: str, record: object) -> FloatArray:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} raw array record must be an object")
    if record.get("dtype") != "<f8":
        raise ValueError(f"{name} raw array dtype must be '<f8'")
    shape_value = record.get("shape")
    if not isinstance(shape_value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape_value
    ):
        raise ValueError(f"{name} raw array shape is invalid")
    array = np.ascontiguousarray(np.asarray(record.get("values"), dtype="<f8"))
    if array.shape != tuple(shape_value):
        raise ValueError(f"{name} raw array shape disagrees with its values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} raw array is non-finite")
    expected = record.get("sha256")
    if not isinstance(expected, str) or _array_sha256(array) != expected:
        raise ValueError(f"{name} raw array SHA-256 mismatch")
    return array


def _raw_configuration_key(summary: Mapping[str, Any]) -> tuple[str, float, int]:
    geometry = summary.get("geometry")
    spacing = summary.get("target_spacing_m")
    substeps = summary.get("transport_substeps")
    if geometry not in CASE_GEOMETRIES:
        raise ValueError("raw refinement geometry is invalid")
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
        raise ValueError("raw refinement target spacing is invalid")
    if isinstance(substeps, bool) or not isinstance(substeps, int):
        raise ValueError("raw refinement transport substeps are invalid")
    return str(geometry), float(spacing), int(substeps)


def _raw_configuration_record(result: _ConfigurationResult) -> dict[str, Any]:
    geometry, spacing, substeps = _raw_configuration_key(result.summary)
    return {
        "geometry": geometry,
        "target_spacing_m": spacing,
        "transport_substeps": substeps,
        "particle_count": int(result.summary["first_bridge"]["particle_count"]),
        "realized_spacing_max_m": float(
            result.summary["first_bridge"]["realized_spacing_max_m"]
        ),
        "configuration_summary_sha256": _json_payload_sha256(
            "fluxv-v5h-configuration-summary-v1",
            result.summary,
        ),
        "state_vector": _raw_array_record(result.state_vector),
        "frontier_positions": _raw_array_record(result.frontier_positions),
        "probe_velocity": _raw_array_record(result.probe_velocity),
    }


def _raw_record_key(record: Mapping[str, Any]) -> tuple[str, float, int]:
    return _raw_configuration_key(record)


def _make_raw_refinement_evidence(
    configurations: Sequence[_ConfigurationResult],
    replay: _ConfigurationResult | None,
) -> dict[str, Any]:
    records = [_raw_configuration_record(row) for row in configurations]
    keys = [_raw_record_key(row) for row in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("raw refinement configurations are not uniquely keyed")
    return {
        "schema_id": RAW_REFINEMENT_SCHEMA_ID,
        "run_id": RUN_ID,
        "array_encoding": (
            "strict JSON binary64 round-trip values; dtype/shape and canonical "
            "little-endian array SHA-256 included"
        ),
        "configurations": records,
        "deterministic_replay": (
            None if replay is None else _raw_configuration_record(replay)
        ),
    }


def _configuration_summary_index(
    summary: Mapping[str, Any],
) -> dict[tuple[str, float, int], Mapping[str, Any]]:
    result: dict[tuple[str, float, int], Mapping[str, Any]] = {}
    geometries = summary.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("summary geometries must be a list")
    for geometry in geometries:
        if not isinstance(geometry, Mapping):
            raise ValueError("summary geometry row must be an object")
        configurations = geometry.get("configurations")
        if not isinstance(configurations, list):
            raise ValueError("summary configurations must be a list")
        for row in configurations:
            if not isinstance(row, Mapping):
                raise ValueError("summary configuration row must be an object")
            key = _raw_configuration_key(row)
            if key in result:
                raise ValueError("summary configuration key is duplicated")
            result[key] = row
    return result


def _configuration_from_raw_record(
    record: Mapping[str, Any],
    summary_row: Mapping[str, Any],
) -> _ConfigurationResult:
    if _json_payload_sha256(
        "fluxv-v5h-configuration-summary-v1",
        summary_row,
    ) != record.get("configuration_summary_sha256"):
        raise ValueError("raw configuration is not bound to its summary row")
    particle_count = record.get("particle_count")
    realized_spacing = record.get("realized_spacing_max_m")
    if particle_count != summary_row["first_bridge"]["particle_count"]:
        raise ValueError("raw particle count disagrees with summary")
    if realized_spacing != summary_row["first_bridge"]["realized_spacing_max_m"]:
        raise ValueError("raw realized spacing disagrees with summary")
    state = _array_from_raw_record("state_vector", record.get("state_vector"))
    frontier = _array_from_raw_record(
        "frontier_positions", record.get("frontier_positions")
    )
    probes = _array_from_raw_record("probe_velocity", record.get("probe_velocity"))
    if state.ndim != 1 or frontier.ndim != 2 or probes.ndim != 2:
        raise ValueError("raw refinement array dimensions are invalid")
    return _ConfigurationResult(
        summary=dict(summary_row),
        state_vector=state,
        frontier_positions=frontier,
        probe_velocity=probes,
    )


def _recompute_refinement_gates(
    summary: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if raw.get("schema_id") != RAW_REFINEMENT_SCHEMA_ID:
        raise ValueError("raw refinement schema is unsupported")
    if raw.get("run_id") != summary.get("run_id"):
        raise ValueError("raw refinement run ID disagrees with summary")
    summary_index = _configuration_summary_index(summary)
    raw_rows = raw.get("configurations")
    if not isinstance(raw_rows, list):
        raise ValueError("raw refinement configurations must be a list")
    rows: dict[tuple[str, float, int], _ConfigurationResult] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("raw refinement configuration must be an object")
        key = _raw_record_key(raw_row)
        if key in rows or key not in summary_index:
            raise ValueError(
                "raw refinement configuration key is invalid or duplicated"
            )
        rows[key] = _configuration_from_raw_record(raw_row, summary_index[key])
    if set(rows) != set(summary_index):
        raise ValueError("raw refinement does not cover every summary configuration")

    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("summary config must be an object")
    configured_geometries = tuple(config.get("geometries", ()))
    geometry_names = tuple(geometry["geometry"] for geometry in summary["geometries"])
    spacing_values = config.get("target_spacings_m", ())
    substep_values = config.get("transport_substeps", ())
    if not isinstance(spacing_values, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in spacing_values
    ):
        raise ValueError("summary target spacings are invalid")
    if not isinstance(substep_values, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in substep_values
    ):
        raise ValueError("summary transport substeps are invalid")
    spacings = tuple(float(value) for value in spacing_values)
    substeps = tuple(substep_values)
    if not configured_geometries or not spacings or not substeps:
        raise ValueError("summary refinement config is incomplete")
    smoke = summary.get("minimal_smoke")
    if not isinstance(smoke, Mapping):
        raise ValueError("summary minimal smoke is invalid")
    empty_stopped_run = bool(not geometry_names and not smoke.get("passed"))
    if configured_geometries != CASE_GEOMETRIES or (
        geometry_names != CASE_GEOMETRIES and not empty_stopped_run
    ):
        raise ValueError("reported geometries do not exactly cover the frozen config")
    span_cells = config.get("span_cells")
    if isinstance(span_cells, bool) or not isinstance(span_cells, int):
        raise ValueError("summary span-cell count is invalid")
    _validate_config(
        GateConfig(
            span_cells=span_cells,
            target_spacings_m=spacings,
            transport_substeps=substeps,
        )
    )
    expected_keys = (
        set()
        if empty_stopped_run
        else {
            (geometry, spacing, step)
            for geometry in CASE_GEOMETRIES
            for spacing in spacings
            for step in substeps
        }
    )
    if set(summary_index) != expected_keys:
        raise ValueError("reported configurations do not exactly cover the frozen grid")

    reported_geometries = {
        geometry["geometry"]: geometry for geometry in summary["geometries"]
    }
    recomputed_geometries: list[dict[str, Any]] = []
    gate_geometries: list[dict[str, Any]] = []
    reported_match = True
    for geometry in geometry_names:
        time_gates = []
        for spacing in spacings:
            family = [rows[(geometry, spacing, step)] for step in substeps]
            time_gates.append(
                {
                    "requested_target_spacing_m": spacing,
                    **_time_refinement_gate(family),
                }
            )
        quadrature_rows = [
            rows[(geometry, spacing, substeps[-1])] for spacing in spacings
        ]
        quadrature = _quadrature_refinement_gate(quadrature_rows)
        reported = reported_geometries[geometry]
        reported_match = bool(
            reported_match
            and time_gates == reported["time_refinement_by_requested_spacing"]
            and quadrature == reported["fixed_core_quadrature_at_finest_time"]
        )
        recomputed_geometries.append(
            {
                "geometry": geometry,
                "time_refinement_by_requested_spacing": time_gates,
                "fixed_core_quadrature_at_finest_time": quadrature,
            }
        )
        gate_geometry = dict(reported)
        gate_geometry["time_refinement_passed"] = all(
            gate["passed"] for gate in time_gates
        )
        gate_geometry["fixed_core_quadrature_passed"] = quadrature["passed"]
        gate_geometries.append(gate_geometry)

    replay_raw = raw.get("deterministic_replay")
    if replay_raw is None and not rows:
        deterministic_replay = False
    elif isinstance(replay_raw, Mapping):
        replay_key = _raw_record_key(replay_raw)
        expected_replay_key = (CASE_GEOMETRIES[0], spacings[0], substeps[0])
        if replay_key != expected_replay_key:
            raise ValueError("raw deterministic replay is not the frozen reference")
        replay = _configuration_from_raw_record(replay_raw, summary_index[replay_key])
        reference = rows[replay_key]
        deterministic_replay = bool(
            replay.summary == reference.summary
            and np.array_equal(replay.state_vector, reference.state_vector)
            and np.array_equal(replay.frontier_positions, reference.frontier_positions)
            and np.array_equal(replay.probe_velocity, reference.probe_velocity)
        )
    else:
        raise ValueError("raw deterministic replay record is missing")

    reported_gate = summary.get("gate_summary")
    if not isinstance(reported_gate, Mapping):
        raise ValueError("summary gate_summary must be an object")
    recomputed_gate = _gate_summary(
        smoke=dict(summary["minimal_smoke"]),
        geometries=gate_geometries,
        deterministic_replay_passed=deterministic_replay,
        declared_source_snapshot_stable_passed=bool(
            reported_gate["declared_source_snapshot_stable_passed"]
        ),
        runtime_boundary_passed=bool(reported_gate["runtime_boundary_passed"]),
    )
    reported_match = bool(reported_match and recomputed_gate == reported_gate)
    return {
        "schema_id": RECOMPUTED_GATE_SCHEMA_ID,
        "run_id": summary["run_id"],
        "derivation_scope": (
            "time/quadrature L2 differences, refinement ratios, threshold gates, "
            "and deterministic replay are recomputed from raw arrays; mechanical "
            "and runtime-boundary booleans are copied from the reported summary"
        ),
        "raw_derived_gate_components": [
            "deterministic_replay_passed",
            "fixed_core_quadrature_passed",
            "time_refinement_passed",
        ],
        "reported_non_refinement_components": [
            "configuration_mechanics_passed",
            "declared_source_snapshot_stable_passed",
            "global_exact_once_passed",
            "minimal_smoke_passed",
            "node_handoff_passed",
            "patch_binding_passed",
            "restart_half_step_passed",
            "runtime_boundary_passed",
        ],
        "raw_refinement_sha256": _json_payload_sha256(
            "fluxv-v5h-raw-refinement-payload-v1", raw
        ),
        "reported_summary_sha256": _json_payload_sha256(
            "fluxv-v5h-reported-summary-payload-v1", summary
        ),
        "geometries": recomputed_geometries,
        "deterministic_replay_passed": deterministic_replay,
        "gate_summary": recomputed_gate,
        "reported_values_match_raw_recomputation": reported_match,
        "passed": bool(recomputed_gate["passed"] and reported_match),
    }


def recompute_refinement_gates_from_artifacts(output_dir: Path) -> dict[str, Any]:
    """Strictly reload raw arrays and independently reconstruct every gate."""

    output = Path(output_dir)
    summary = _load_strict_json(output / "summary.json")
    raw = _load_strict_json(output / RAW_REFINEMENT_ARTIFACT)
    return _recompute_refinement_gates(summary, raw)


def _stable_vector_sum(values: np.ndarray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("stable vector sum requires an (n, 3) array")
    return np.asarray(
        [fsum(float(value) for value in array[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return float("nan")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        return float("nan")
    scale = max(1.0e-15, float(np.linalg.norm(right_array)))
    return float(np.linalg.norm(left_array - right_array) / scale)


def _validate_config(config: GateConfig) -> None:
    if isinstance(config.span_cells, bool) or not isinstance(config.span_cells, int):
        raise ValueError("span_cells must be an integer")
    if config.span_cells < 2:
        raise ValueError("span_cells must be at least two")
    if len(config.target_spacings_m) < 3:
        raise ValueError("fixed-core quadrature requires at least three h levels")
    spacings = tuple(float(value) for value in config.target_spacings_m)
    if not all(np.isfinite(value) and value > 0.0 for value in spacings):
        raise ValueError("target spacings must be positive and finite")
    if not all(left > right for left, right in zip(spacings[:-1], spacings[1:])):
        raise ValueError("target spacings must be strictly refined")
    if spacings[0] > SOURCE_TRANSFER_SIGMA_M / FROZEN_OVERLAP_LAMBDA:
        raise ValueError("coarsest target spacing violates the minimum overlap")
    if len(config.transport_substeps) < 3:
        raise ValueError("time refinement requires at least three levels")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in config.transport_substeps
    ):
        raise ValueError("transport substeps must be positive integers")
    if not all(
        left < right
        for left, right in zip(
            config.transport_substeps[:-1], config.transport_substeps[1:]
        )
    ):
        raise ValueError("transport substeps must be strictly increasing")
    finest_spacing = spacings[-1]
    minimum_core = SOURCE_TRANSFER_SIGMA_M * min(CORE_SCAN_MULTIPLIERS)
    if finest_spacing > minimum_core / FROZEN_OVERLAP_LAMBDA:
        raise ValueError("finest h cannot support the pre-registered core scan")


def _chord_and_twist(geometry: GateGeometry, eta: float) -> tuple[float, float]:
    if geometry == "straight":
        return ROOT_CHORD_M, 0.0
    if geometry == "taper":
        return ROOT_CHORD_M * (1.0 - 0.4 * eta), 0.0
    if geometry == "twist":
        return ROOT_CHORD_M, radians(20.0) * eta
    raise ValueError(f"unsupported non-target geometry {geometry!r}")


def _chord_direction(twist_rad: float) -> FloatArray:
    return np.asarray((cos(twist_rad), 0.0, -sin(twist_rad)), dtype=np.float64)


def _unit(name: str, value: np.ndarray) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not np.all(np.isfinite(array)) or norm <= 0.0:
        raise ValueError(f"{name} must define a finite nonzero vector")
    return array / norm


def _geometry_nodes(
    geometry: GateGeometry,
    span_cells: int,
) -> tuple[_NodeGeometry, ...]:
    anchors: list[FloatArray] = []
    chords: list[float] = []
    twists: list[float] = []
    for node_index in range(span_cells + 1):
        eta = node_index / span_cells
        chord, twist = _chord_and_twist(geometry, eta)
        pivot = np.asarray((0.0, SPAN_M * eta, 0.0), dtype=np.float64)
        anchor = pivot - PIVOT_FRACTION_CHORD * chord * _chord_direction(twist)
        anchors.append(anchor)
        chords.append(chord)
        twists.append(twist)

    result: list[_NodeGeometry] = []
    for node_index, (anchor, chord, twist) in enumerate(
        zip(anchors, chords, twists, strict=True)
    ):
        if node_index == 0:
            tangent = anchors[1] - anchors[0]
        elif node_index == span_cells:
            tangent = anchors[-1] - anchors[-2]
        else:
            tangent = anchors[node_index + 1] - anchors[node_index - 1]
        span_axis = _unit("node span tangent", tangent)
        chord_guess = _chord_direction(twist)
        x_axis = _unit(
            "node chord axis",
            chord_guess - float(np.dot(chord_guess, span_axis)) * span_axis,
        )
        z_axis = _unit("node normal axis", np.cross(span_axis, x_axis))
        result.append(
            _NodeGeometry(
                node_id=f"{geometry}:node:{node_index}",
                leading_edge_anchor_gp1_m=tuple(float(item) for item in anchor),
                chord_m=chord,
                x_axis_gp1=tuple(float(item) for item in x_axis),
                z_axis_gp1=tuple(float(item) for item in z_axis),
                span_axis_gp1=tuple(float(item) for item in span_axis),
                patch_id=f"non-target:{geometry}:continuous-wing-patch",
                frame_id=f"non-target:{geometry}:gp1-wing-frame",
            )
        )
    return tuple(result)


def _cell_plane_map(
    geometry: GateGeometry,
    cell_index: int,
    left: _NodeGeometry,
    right: _NodeGeometry,
) -> DVMPlaneToGP1Map:
    left_anchor = np.asarray(left.leading_edge_anchor_gp1_m, dtype=np.float64)
    right_anchor = np.asarray(right.leading_edge_anchor_gp1_m, dtype=np.float64)
    span_axis = _unit("cell span axis", right_anchor - left_anchor)
    # Reconstruct the cell frame from its two already-frozen node frames.
    chord_guess = np.asarray(left.x_axis_gp1) + np.asarray(right.x_axis_gp1)
    x_axis = _unit(
        "cell chord axis",
        chord_guess - float(np.dot(chord_guess, span_axis)) * span_axis,
    )
    z_axis = _unit("cell normal axis", np.cross(span_axis, x_axis))
    return DVMPlaneToGP1Map(
        origin_gp1_m=0.5 * (left_anchor + right_anchor),
        x_axis_gp1=x_axis,
        z_axis_gp1=z_axis,
        positive_circulation_axis_gp1=span_axis,
        circulation_to_ring_traversal_sign=1,
        provenance=(
            f"generic non-target {geometry} cell {cell_index}; explicit "
            "orthonormal DVM x-z to GP1 map"
        ),
    )


def _new_source(
    *,
    geometry: GateGeometry,
    role: Literal["node_geometry", "cell_circulation"],
    index: int,
    chord_m: float,
) -> V5hDVMSource:
    convective_dt = PHYSICAL_RELEASE_DT_S * REFERENCE_SPEED_M_PER_S / chord_m
    identity = f"fixed-core:{geometry}:{role}:{index}"
    return V5hDVMSource(
        physical_section_id=f"{identity}:section",
        physical_strip_id=f"{identity}:strip",
        geometry_identity="generic explicit zero camber flat plate",
        reference_speed_m_per_s=REFERENCE_SPEED_M_PER_S,
        reference_chord_m=chord_m,
        zero_camber_surrogate=True,
        delta_time_convective=convective_dt,
        pivot_fraction_chord=PIVOT_FRACTION_CHORD,
        threshold=LESPThreshold(
            value=LESP_CRITICAL,
            section_family="generic thin flat plate",
            reynolds=REYNOLDS,
            source="Ramesh LDVM v2.5 published source input Lcrit=0.18",
            source_role="published_source_input",
        ),
        settings=SOURCE_SETTINGS,
    )


def _source_sets(
    geometry: GateGeometry,
    nodes: tuple[_NodeGeometry, ...],
) -> tuple[tuple[V5hDVMSource, ...], tuple[V5hDVMSource, ...]]:
    node_sources = tuple(
        _new_source(
            geometry=geometry,
            role="node_geometry",
            index=index,
            chord_m=node.chord_m,
        )
        for index, node in enumerate(nodes)
    )
    cell_sources = tuple(
        _new_source(
            geometry=geometry,
            role="cell_circulation",
            index=index,
            chord_m=0.5 * (nodes[index].chord_m + nodes[index + 1].chord_m),
        )
        for index in range(len(nodes) - 1)
    )
    return node_sources, cell_sources


def _step_sources(
    sources: Sequence[V5hDVMSource],
    alpha_rad: float,
) -> tuple[DVMSourceEvent, ...]:
    events = tuple(source.step(alpha_rad, 0.0, 0.0) for source in sources)
    for event in events:
        if validate_dvm_source_event(event) is not event:
            raise RuntimeError("DVM source attestation changed event identity")
    return events


def _scaled_kelvin(event: DVMSourceEvent) -> float:
    return abs(float(event.kelvin_residual_over_u_c)) * float(
        event.provenance.circulation_scale_u_times_c_m2_per_s
    )


def _source_events_pass(
    node_events: Sequence[DVMSourceEvent],
    cell_events: Sequence[DVMSourceEvent],
    *,
    expected_step: int,
    expected_mode: str,
) -> bool:
    events = tuple(node_events) + tuple(cell_events)
    return bool(
        events
        and all(
            event.lineage.source_step_index == expected_step
            and event.lesp_active
            and event.lev_birth_mode == expected_mode
            and _scaled_kelvin(event) <= MAX_SOURCE_KELVIN_M2_PER_S
            and event.provenance.observation_access == "none"
            and event.provenance.target_case_branch == "none"
            and not event.provenance.canonical
            for event in events
        )
    )


def _node_facts(
    nodes: Sequence[_NodeGeometry],
    events: Sequence[DVMSourceEvent],
    *,
    wing_id: str,
    source_time_s: float,
) -> tuple[GP1NodeSectionFact, ...]:
    if len(nodes) != len(events):
        raise ValueError("node geometry and node-event counts disagree")
    facts: list[GP1NodeSectionFact] = []
    for node, event in zip(nodes, events, strict=True):
        step = event.lineage.source_step_index
        if step is None:
            raise ValueError("enabled node event has no source step")
        lev_edge = event.lev_placement.edge_anchor_position_over_chord_backend_world
        tev_edge = event.tev_placement.edge_anchor_position_over_chord_backend_world
        if lev_edge is None or tev_edge is None:
            raise ValueError("enabled node event has no same-layer LE/TE anchors")
        delta = np.asarray(tev_edge, dtype=np.float64) - np.asarray(
            lev_edge, dtype=np.float64
        )
        lev_anchor = np.asarray(node.leading_edge_anchor_gp1_m, dtype=np.float64)
        x_axis = np.asarray(node.x_axis_gp1, dtype=np.float64)
        z_axis = np.asarray(node.z_axis_gp1, dtype=np.float64)
        span_axis = np.cross(x_axis, z_axis)
        tev_anchor = lev_anchor + node.chord_m * (delta[0] * x_axis + delta[1] * z_axis)
        facts.append(
            GP1NodeSectionFact(
                wing_id=wing_id,
                node_id=node.node_id,
                source_step_index=step,
                source_time_s=source_time_s,
                event=event,
                lev_edge_anchor_gp1_m=tuple(float(item) for item in lev_anchor),
                tev_edge_anchor_gp1_m=tuple(float(item) for item in tev_anchor),
                reference_chord_m=node.chord_m,
                reference_speed_m_per_s=REFERENCE_SPEED_M_PER_S,
                dvm_x_axis_gp1=tuple(float(item) for item in x_axis),
                dvm_z_axis_gp1=tuple(float(item) for item in z_axis),
                positive_span_axis_gp1=tuple(float(item) for item in span_axis),
                topology_patch_id=node.patch_id,
                coordinate_frame_id=node.frame_id,
                node_lineage_id=event.lineage.section_lineage_id,
                geometry_token=event.provenance.geometry_hash_sha256,
            )
        )
    return tuple(facts)


def _placement_cells(
    geometry: GateGeometry,
    node_facts: Sequence[GP1NodeSectionFact],
    cell_events: Sequence[DVMSourceEvent],
) -> tuple[DVMNodePlacementCell, ...]:
    if len(node_facts) != len(cell_events) + 1:
        raise ValueError("node facts do not define one open cell chain")
    return tuple(
        DVMNodePlacementCell(
            cell_id=f"{geometry}:cell:{index}",
            left_node_fact=node_facts[index],
            right_node_fact=node_facts[index + 1],
            cell_source_event=cell_events[index],
        )
        for index in range(len(cell_events))
    )


def _map_node_layer(
    adapter: NodeLocalDVMPlacementAdapter,
    wing_id: str,
    geometry: GateGeometry,
    nodes: Sequence[_NodeGeometry],
    node_events: Sequence[DVMSourceEvent],
    cell_events: Sequence[DVMSourceEvent],
    *,
    source_time_s: float,
) -> DVMNodePlacementResult:
    facts = _node_facts(
        nodes,
        node_events,
        wing_id=wing_id,
        source_time_s=source_time_s,
    )
    result = adapter.map_step(
        _placement_cells(geometry, facts, cell_events),
        delta_time_s=PHYSICAL_RELEASE_DT_S,
    )
    if (
        validate_live_dvm_node_placement_result(result, expected_wing_id=wing_id)
        is not result
    ):
        raise RuntimeError("node-placement attestation changed result identity")
    return result


def _ribbon_cells(
    geometry: GateGeometry,
    nodes: Sequence[_NodeGeometry],
    cell_events: Sequence[DVMSourceEvent],
) -> tuple[DVMSpanCellSource, ...]:
    if len(nodes) != len(cell_events) + 1:
        raise ValueError("cell events do not cover the node chain")
    return tuple(
        DVMSpanCellSource(
            cell_id=f"{geometry}:cell:{index}",
            left_node_id=nodes[index].node_id,
            right_node_id=nodes[index + 1].node_id,
            event=event,
            plane_to_gp1=_cell_plane_map(
                geometry,
                index,
                nodes[index],
                nodes[index + 1],
            ),
        )
        for index, event in enumerate(cell_events)
    )


def _placement_metrics(
    result: DVMNodePlacementResult,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    ledgers = result.node_ledgers
    coverage = result.cell_coverage
    expected_active = expected_mode != "inactive"
    coverage_passed = bool(
        coverage
        and all(
            row.active is expected_active
            and row.endpoint_coverage_complete
            and row.shared_fact_identity_verified
            and row.placement_mode == expected_mode
            for row in coverage
        )
    )
    if expected_mode in {"first", "restart"}:
        topology_passed = bool(
            ledgers
            and all(
                row.active
                and row.placement_mode == expected_mode
                and row.topology_owner == "node_local_dvm_relative_birth"
                and row.edge_velocity_used_by_ribbon
                and not row.dvm_absolute_birth_used_for_topology
                and row.mapped_relative_birth_gp1_m is not None
                and row.half_step_reconstruction_residual_m <= 1.0e-12
                for row in ledgers
            )
        )
    elif expected_mode == "continuous":
        topology_passed = bool(
            ledgers
            and all(
                row.active
                and row.placement_mode == "continuous"
                and row.topology_owner == "attested_rvpm_frontier"
                and not row.edge_velocity_used_by_ribbon
                and not row.dvm_absolute_birth_used_for_topology
                and row.edge_velocity_gp1_m_per_s == (0.0, 0.0, 0.0)
                and row.mapped_relative_birth_gp1_m is None
                for row in ledgers
            )
        )
    elif expected_mode == "inactive":
        topology_passed = bool(
            ledgers
            and all(
                not row.active
                and row.placement_mode == "inactive"
                and row.topology_owner == "none"
                and not row.edge_velocity_used_by_ribbon
                and not row.dvm_absolute_birth_used_for_topology
                for row in ledgers
            )
        )
    else:
        raise ValueError(f"unsupported expected placement mode {expected_mode!r}")
    owner_passed = bool(
        result.node_geometry_role == "node-local-dvm-geometry-only"
        and result.exclusive_strength_owner == "cell-center-dvm"
        and result.exclusive_surface_owner == "ptera"
        and result.feedback_velocity is None
        and result.feedback_call_count == 0
    )
    return {
        "wing_id": result.wing_id,
        "producer_manifest_sha256": result.producer_manifest_sha256,
        "node_count": len(ledgers),
        "cell_count": len(coverage),
        "modes": [row.placement_mode for row in ledgers],
        "topology_owners": [row.topology_owner for row in ledgers],
        "event_manifest_sha256": [row.source_event_manifest_sha256 for row in ledgers],
        "node_fact_manifest_sha256": [row.node_fact_manifest_sha256 for row in ledgers],
        "half_step_reconstruction_residual_max_m": max(
            (row.half_step_reconstruction_residual_m for row in ledgers),
            default=0.0,
        ),
        "cell_source_coverage_passed": coverage_passed,
        "active_cell_endpoint_coverage_passed": coverage_passed,
        "topology_ownership_passed": topology_passed,
        "geometry_only_owner_passed": owner_passed,
        "passed": bool(coverage_passed and topology_passed and owner_passed),
    }


def _ribbon_half_step_matches_placement(
    ribbon: Any,
    placement: DVMNodePlacementResult,
) -> bool:
    expected = {
        ledger.node_id: ledger.mapped_relative_birth_gp1_m
        for ledger in placement.node_ledgers
        if ledger.active
    }
    actual = {
        birth.node_id: birth.birth_position_gp1_m
        for birth in ribbon.node_births
        if birth.active
    }
    if set(expected) != set(actual) or any(
        value is None for value in expected.values()
    ):
        return False
    return all(
        float(
            np.linalg.norm(
                np.asarray(actual[node_id], dtype=np.float64)
                - np.asarray(expected[node_id], dtype=np.float64)
            )
        )
        <= 1.0e-12
        for node_id in expected
    )


def _cell_strength_ownership_passed(
    ribbon: Any,
    cell_events: Sequence[DVMSourceEvent],
) -> bool:
    expected = {
        event.lineage.physical_strip_id: event.lineage.newborn_lev_source_id
        for event in cell_events
    }
    return bool(
        len(ribbon.cell_ledgers) == len(expected)
        and all(
            ledger.physical_strip_id in expected
            and ledger.source_lineage_id == expected[ledger.physical_strip_id]
            for ledger in ribbon.cell_ledgers
        )
    )


def _ribbon_handoff_integrity(
    ribbon: DVMRibbonShadowResult,
    placement: DVMNodePlacementResult,
) -> dict[str, Any]:
    """Measure the live placement/patch binding and atomic exact-once commit."""

    diagnostics = ribbon.diagnostics
    expected_patch_ids = sorted(
        {row.topology_patch_id for row in placement.cell_coverage}
    )
    expected_frame_ids = sorted(
        {row.coordinate_frame_id for row in placement.cell_coverage}
    )
    direct_live_result = bool(validate_live_dvm_ribbon_shadow_result(ribbon) is ribbon)
    placement_manifest_match = bool(
        diagnostics.node_placement_bound
        and diagnostics.node_placement_manifest_sha256
        == placement.producer_manifest_sha256
    )
    topology_patch_match = sorted(diagnostics.topology_patch_ids) == expected_patch_ids
    coordinate_frame_match = (
        sorted(diagnostics.coordinate_frame_ids) == expected_frame_ids
    )
    patch_binding_passed = bool(
        diagnostics.patch_binding_passed
        and placement_manifest_match
        and topology_patch_match
        and coordinate_frame_match
    )
    # ``map_step`` returns a live-attested result only from the same locked
    # commit section that registers every live source and placement globally.
    # A zero local reuse count plus that live attestation is therefore runtime
    # evidence that this handoff traversed the global exact-once commit.
    global_exact_once_commit_passed = bool(
        direct_live_result and diagnostics.source_reuse_count == 0
    )
    return {
        "node_placement_bound": bool(diagnostics.node_placement_bound),
        "node_placement_manifest_match": placement_manifest_match,
        "topology_patch_ids": list(diagnostics.topology_patch_ids),
        "coordinate_frame_ids": list(diagnostics.coordinate_frame_ids),
        "topology_patch_match": topology_patch_match,
        "coordinate_frame_match": coordinate_frame_match,
        "patch_binding_passed": patch_binding_passed,
        "direct_live_result_attested": direct_live_result,
        "source_reuse_count": int(diagnostics.source_reuse_count),
        "global_exact_once_commit_passed": global_exact_once_commit_passed,
        "passed": bool(patch_binding_passed and global_exact_once_commit_passed),
    }


def _probe_global_exact_once_rejection(
    *,
    wing_id: str,
    cells: Sequence[DVMSpanCellSource],
    placement: DVMNodePlacementResult,
    source_time_s: float,
) -> dict[str, Any]:
    """Attempt a cross-mapper replay of one live first layer and require rejection."""

    try:
        NodeOwnedDVMRibbonShadow(
            wing_id=wing_id,
            source_family="lev",
        ).map_step(
            cells,
            placement.kinematics,
            delta_time_s=PHYSICAL_RELEASE_DT_S,
            transport_enabled=True,
            source_time_s=source_time_s,
            node_placement_result=placement,
        )
    except ValueError as error:
        message = str(error)
        passed = bool(
            "already consumed" in message
            and "another ribbon mapper" in message
            and ("live DVM event" in message or "live node placement" in message)
        )
        return {
            "attempted": True,
            "exception_type": type(error).__name__,
            "exception_message": message,
            "passed": passed,
        }
    return {
        "attempted": True,
        "exception_type": None,
        "exception_message": None,
        "passed": False,
    }


def _bridge_metrics(
    bridge: ShadowBridgeResult,
    *,
    requested_spacing_m: float,
    smoothing_radius_m: float,
) -> dict[str, Any]:
    if bridge.edge_graph is None:
        raise RuntimeError("enabled fixed-core deposition lost its edge graph")
    edges = {edge.key: edge for edge in bridge.edge_graph.retained_edges}
    realized_by_edge: dict[object, tuple[int, float, float]] = {}
    for lineage in bridge.lineage:
        edge = edges[lineage.source_edge]
        start = np.asarray(edge.start_position, dtype=np.float64)
        end = np.asarray(edge.end_position, dtype=np.float64)
        count = int(lineage.subdivision_count)
        realized_spacing = float(np.linalg.norm(end - start) / count)
        realized_overlap = smoothing_radius_m / realized_spacing
        prior = realized_by_edge.setdefault(
            lineage.source_edge,
            (count, realized_spacing, realized_overlap),
        )
        if prior != (count, realized_spacing, realized_overlap):
            raise RuntimeError("one edge carries inconsistent deposition lineage")
    if set(realized_by_edge) != set(edges):
        raise RuntimeError("deposition lineage does not cover every retained edge")

    edge_realization = [
        {
            "edge_key": list(edge.key),
            "subdivision_count": realized_by_edge[edge.key][0],
            "realized_spacing_m": realized_by_edge[edge.key][1],
            "realized_overlap": realized_by_edge[edge.key][2],
        }
        for edge in bridge.edge_graph.retained_edges
    ]
    counts = [item["subdivision_count"] for item in edge_realization]
    realized_spacings = [item["realized_spacing_m"] for item in edge_realization]
    realized_overlaps = [item["realized_overlap"] for item in edge_realization]
    finite = bool(
        np.all(np.isfinite(bridge.positions))
        and np.all(np.isfinite(bridge.gamma))
        and np.all(np.isfinite(bridge.sigma))
        and bridge.sigma.size > 0
        and np.all(bridge.sigma > 0.0)
    )
    diagnostics = bridge.diagnostics
    passed = bool(
        finite
        and np.all(bridge.sigma == smoothing_radius_m)
        and max(realized_spacings) <= requested_spacing_m * (1.0 + 1.0e-12)
        and min(realized_overlaps) >= FROZEN_OVERLAP_LAMBDA * (1.0 - 1.0e-12)
        and diagnostics.max_edge_conservation_abs <= MAX_VECTOR_ABS
        and diagnostics.global_conservation_abs <= MAX_VECTOR_ABS
        and diagnostics.incidence_residual <= MAX_RIBBON_RESIDUAL
        and diagnostics.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
        and diagnostics.nonfinite_count == 0
        and diagnostics.owner_conflict_count == 0
        and diagnostics.feedback_call_count == 0
        and bridge.feedback_velocity is None
    )
    return {
        "requested_target_spacing_m": requested_spacing_m,
        "smoothing_radius_m": smoothing_radius_m,
        "particle_count": int(bridge.positions.shape[0]),
        "retained_edge_count": len(edges),
        "edge_subdivision_counts": counts,
        "edge_realization": edge_realization,
        "realized_spacing_min_m": min(realized_spacings),
        "realized_spacing_max_m": max(realized_spacings),
        "realized_overlap_min": min(realized_overlaps),
        "realized_overlap_max": max(realized_overlaps),
        "max_edge_conservation_abs": diagnostics.max_edge_conservation_abs,
        "global_conservation_abs": diagnostics.global_conservation_abs,
        "incidence_residual": diagnostics.incidence_residual,
        "edge_reconstruction_residual": diagnostics.edge_reconstruction_residual,
        "fixed_physical_sigma": bool(
            bridge.sigma.size and np.all(bridge.sigma == smoothing_radius_m)
        ),
        "finite": finite,
        "passed": passed,
    }


def _transport_invariants(
    first_bridge: ShadowBridgeResult,
    report: PassiveFrontierTransportReport,
) -> dict[str, Any]:
    transported = materialize_transported_particle_state(report)
    initial_invariant = (
        np.linalg.norm(first_bridge.gamma, axis=1) * first_bridge.sigma**2
    )
    final_invariant = np.linalg.norm(transported.gamma, axis=1) * transported.sigma**2
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
        fsum(float(np.linalg.norm(row)) for row in first_bridge.gamma),
    )
    vector_drift_abs = float(np.linalg.norm(_stable_vector_sum(transported.gamma)))
    vector_drift_rel = vector_drift_abs / vector_scale
    finite = bool(
        np.all(np.isfinite(transported.positions))
        and np.all(np.isfinite(transported.gamma))
        and np.all(np.isfinite(transported.sigma))
        and np.all(transported.sigma > 0.0)
    )
    return {
        "particle_count": int(transported.positions.shape[0]),
        "particle_invariant_relative_drift_max": invariant_relative,
        "global_vector_drift_abs_m3_per_s": vector_drift_abs,
        "global_vector_drift_relative": vector_drift_rel,
        "finite": finite,
        "passed": bool(
            finite
            and invariant_relative <= MAX_PARTICLE_INVARIANT_REL
            and vector_drift_rel <= MAX_VECTOR_REL
        ),
    }


def _temporal_layer_overlap(
    transported_positions: np.ndarray,
    transported_sigma: np.ndarray,
    transported_lineage: Sequence[Any],
    second_bridge: ShadowBridgeResult,
) -> dict[str, Any]:
    def cell_keys(lineage: Any) -> frozenset[str]:
        keys = {
            str(incidence.ring_id).rsplit(":step:", maxsplit=1)[0]
            for incidence in lineage.ring_incidences
        }
        if not keys:
            raise RuntimeError("retained particle has no source-cell incidence")
        return frozenset(keys)

    old_keys = tuple(cell_keys(item) for item in transported_lineage)
    new_keys = tuple(cell_keys(item) for item in second_bridge.lineage)
    max_distance = 0.0
    min_overlap = float("inf")
    missing = 0
    for index, keys in enumerate(new_keys):
        candidates = [
            old_index
            for old_index, old_group in enumerate(old_keys)
            if keys.intersection(old_group)
        ]
        if not candidates:
            missing += 1
            continue
        distances = np.linalg.norm(
            transported_positions[candidates] - second_bridge.positions[index],
            axis=1,
        )
        local = int(np.argmin(distances))
        distance = float(distances[local])
        old_index = candidates[local]
        max_distance = max(max_distance, distance)
        if distance > 0.0:
            min_overlap = min(
                min_overlap,
                min(
                    float(transported_sigma[old_index]),
                    float(second_bridge.sigma[index]),
                )
                / distance,
            )
    passed = bool(
        missing == 0
        and max_distance
        <= SOURCE_TRANSFER_SIGMA_M / MIN_TEMPORAL_TWO_LAYER_OVERLAP * (1.0 + 1.0e-12)
        and min_overlap >= MIN_TEMPORAL_TWO_LAYER_OVERLAP * (1.0 - 1.0e-12)
    )
    return {
        "missing_same_cell_group_count": missing,
        "source_transfer_spacing_m": SOURCE_TRANSFER_SPACING_M,
        "max_nearest_same_cell_distance_m": max_distance,
        "max_distance_over_source_transfer_spacing": (
            max_distance / SOURCE_TRANSFER_SPACING_M
        ),
        "minimum_two_layer_overlap": (
            min_overlap if np.isfinite(min_overlap) else None
        ),
        "minimum_two_layer_overlap_limit": MIN_TEMPORAL_TWO_LAYER_OVERLAP,
        "passed": passed,
    }


def _refinement_family(
    differences: Sequence[float],
    *,
    fine_limit: float,
    minimum_ratio: float,
) -> dict[str, Any]:
    values = tuple(float(value) for value in differences)
    finite = bool(
        values and all(np.isfinite(value) and value >= 0.0 for value in values)
    )
    ratios: list[float | None] = []
    for coarse, fine in zip(values[:-1], values[1:]):
        ratio = (
            None
            if not np.isfinite(coarse) or not np.isfinite(fine) or fine <= 1.0e-14
            else coarse / fine
        )
        ratios.append(ratio if ratio is None or np.isfinite(ratio) else None)
    monotone = bool(
        finite
        and all(
            fine <= coarse * (1.0 + 1.0e-12)
            for coarse, fine in zip(values[:-1], values[1:])
        )
    )
    reduction = bool(
        finite
        and all(
            fine <= 1.0e-14 or coarse / fine >= minimum_ratio
            for coarse, fine in zip(values[:-1], values[1:])
        )
    )
    passed = bool(finite and monotone and reduction and values[-1] <= fine_limit)
    return {
        "consecutive_relative_l2": [
            value if np.isfinite(value) else None for value in values
        ],
        "coarse_to_fine_ratios": ratios,
        "finite": finite,
        "monotone": monotone,
        "minimum_reduction_ratio": minimum_ratio,
        "fine_relative_l2_limit": fine_limit,
        "passed": passed,
    }


def _time_refinement_gate(rows: Sequence[_ConfigurationResult]) -> dict[str, Any]:
    differences = [
        _relative_l2(left.state_vector, right.state_vector)
        for left, right in zip(rows[:-1], rows[1:])
    ]
    return _refinement_family(
        differences,
        fine_limit=MAX_FINE_TIME_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_TIME_REFINEMENT_RATIO,
    )


def _quadrature_refinement_gate(
    rows: Sequence[_ConfigurationResult],
) -> dict[str, Any]:
    frontier = [
        _relative_l2(left.frontier_positions, right.frontier_positions)
        for left, right in zip(rows[:-1], rows[1:])
    ]
    probes = [
        _relative_l2(left.probe_velocity, right.probe_velocity)
        for left, right in zip(rows[:-1], rows[1:])
    ]
    frontier_gate = _refinement_family(
        frontier,
        fine_limit=MAX_FINE_QUADRATURE_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_QUADRATURE_REFINEMENT_RATIO,
    )
    probe_gate = _refinement_family(
        probes,
        fine_limit=MAX_FINE_QUADRATURE_RELATIVE_DIFFERENCE,
        minimum_ratio=MIN_QUADRATURE_REFINEMENT_RATIO,
    )
    particle_counts = [
        int(row.summary["first_bridge"]["particle_count"]) for row in rows
    ]
    realized_h = [
        float(row.summary["first_bridge"]["realized_spacing_max_m"]) for row in rows
    ]
    observable_refinement = bool(
        all(
            left < right
            for left, right in zip(particle_counts[:-1], particle_counts[1:])
        )
        and all(left > right for left, right in zip(realized_h[:-1], realized_h[1:]))
    )
    return {
        "requested_target_spacings_m": [
            float(row.summary["target_spacing_m"]) for row in rows
        ],
        "particle_counts": particle_counts,
        "realized_spacing_max_m": realized_h,
        "observable_ceil_count_refinement": observable_refinement,
        "frontier_positions": frontier_gate,
        "fixed_probe_velocity": probe_gate,
        "passed": bool(
            observable_refinement and frontier_gate["passed"] and probe_gate["passed"]
        ),
    }


def _run_configuration(
    geometry: GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
    smoothing_radius_m: float,
    transport_substeps: int,
) -> _ConfigurationResult:
    nodes = _geometry_nodes(geometry, span_cells)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    wing_id = f"fixed-core:{geometry}:wing"
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
    ribbon_mapper = NodeOwnedDVMRibbonShadow(
        wing_id=wing_id,
        source_family="lev",
    )

    first_node_events = _step_sources(node_sources, FIRST_ALPHA_RAD)
    first_cell_events = _step_sources(cell_sources, FIRST_ALPHA_RAD)
    first_placement = _map_node_layer(
        placement_adapter,
        wing_id,
        geometry,
        nodes,
        first_node_events,
        first_cell_events,
        source_time_s=0.0,
    )
    first_cells = _ribbon_cells(geometry, nodes, first_cell_events)
    first = ribbon_mapper.map_step(
        first_cells,
        first_placement.kinematics,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
        source_time_s=0.0,
        node_placement_result=first_placement,
    )
    if first.edge_graph is None:
        raise RuntimeError("first active source layer produced no ribbon graph")
    global_exact_once_probe = _probe_global_exact_once_rejection(
        wing_id=wing_id,
        cells=first_cells,
        placement=first_placement,
        source_time_s=0.0,
    )
    first_bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
        first.edge_graph,
        smoothing_radius=smoothing_radius_m,
        target_spacing=target_spacing_m,
        step=1,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
    )
    report = transport_passive_node_frontiers(
        first,
        first_bridge,
        wing_id=wing_id,
        transport_start_time_s=0.0,
        transport_end_time_s=PHYSICAL_RELEASE_DT_S,
        transport_substeps=transport_substeps,
        deposition_target_spacing_m=target_spacing_m,
        freestream_velocity_gp1_m_per_s=(REFERENCE_SPEED_M_PER_S, 0.0, 0.0),
    )
    if validate_passive_frontier_transport_report(report) is not report:
        raise RuntimeError("frontier transport attestation changed report identity")
    transported = materialize_transported_particle_state(report)

    second_node_events = _step_sources(node_sources, CONTINUOUS_ALPHA_RAD)
    second_cell_events = _step_sources(cell_sources, CONTINUOUS_ALPHA_RAD)
    second_placement = _map_node_layer(
        placement_adapter,
        wing_id,
        geometry,
        nodes,
        second_node_events,
        second_cell_events,
        source_time_s=PHYSICAL_RELEASE_DT_S,
    )
    second = ribbon_mapper.map_step(
        _ribbon_cells(geometry, nodes, second_cell_events),
        second_placement.kinematics,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
        source_time_s=PHYSICAL_RELEASE_DT_S,
        frontier_transport_report=report,
        node_placement_result=second_placement,
    )
    if second.edge_graph is None:
        raise RuntimeError("continuous source layer produced no ribbon graph")
    second_bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
        second.edge_graph,
        smoothing_radius=smoothing_radius_m,
        target_spacing=target_spacing_m,
        step=2,
        physical_owner=RING_PHYSICAL_OWNER,
        owner_state=DIAGNOSTIC_SHADOW_OWNER,
    )

    first_placement_metrics = _placement_metrics(first_placement, expected_mode="first")
    second_placement_metrics = _placement_metrics(
        second_placement, expected_mode="continuous"
    )
    first_handoff_integrity = _ribbon_handoff_integrity(first, first_placement)
    second_handoff_integrity = _ribbon_handoff_integrity(second, second_placement)
    first_half_step_passed = _ribbon_half_step_matches_placement(first, first_placement)
    frontier_by_node = {fact.node_id: fact for fact in report.facts}
    second_by_node = {birth.node_id: birth for birth in second.node_births}
    continuous_residuals: list[float] = []
    continuous_exact = True
    for ledger in second_placement.node_ledgers:
        fact = frontier_by_node.get(ledger.node_id)
        birth = second_by_node.get(ledger.node_id)
        if fact is None or birth is None or birth.birth_position_gp1_m is None:
            continuous_exact = False
            continue
        anchor = np.asarray(ledger.lev_edge_anchor_gp1_m, dtype=np.float64)
        frontier = np.asarray(fact.advected_position_gp1_m, dtype=np.float64)
        expected = anchor + (frontier - anchor) / 3.0
        actual = np.asarray(birth.birth_position_gp1_m, dtype=np.float64)
        residual = float(np.linalg.norm(actual - expected))
        continuous_residuals.append(residual)
        continuous_exact &= bool(np.array_equal(actual, expected))
    continuous_frontier_only_passed = bool(
        continuous_exact
        and len(continuous_residuals) == len(nodes)
        and second.diagnostics.transport_advance_count == len(nodes)
        and all(
            ledger.topology_owner == "attested_rvpm_frontier"
            and not ledger.edge_velocity_used_by_ribbon
            and not ledger.dvm_absolute_birth_used_for_topology
            for ledger in second_placement.node_ledgers
        )
    )

    first_bridge_metrics = _bridge_metrics(
        first_bridge,
        requested_spacing_m=target_spacing_m,
        smoothing_radius_m=smoothing_radius_m,
    )
    second_bridge_metrics = _bridge_metrics(
        second_bridge,
        requested_spacing_m=target_spacing_m,
        smoothing_radius_m=smoothing_radius_m,
    )
    transport_metrics = _transport_invariants(first_bridge, report)
    overlap = _temporal_layer_overlap(
        transported.positions,
        transported.sigma,
        report.transported_particle_cloud.lineage,
        second_bridge,
    )
    frontier_positions = np.asarray(
        [fact.advected_position_gp1_m for fact in report.facts],
        dtype=np.float64,
    )
    probe_velocity = direct_gaussian_erf_velocity_jacobian(
        transported.positions,
        transported.gamma,
        transported.sigma,
        target_positions=FIXED_PROBES_GP1_M,
    ).velocity
    second_birth_positions = np.asarray(
        [
            birth.birth_position_gp1_m
            for birth in second.node_births
            if birth.birth_position_gp1_m is not None
        ],
        dtype=np.float64,
    )
    state_vector = np.concatenate(
        (
            transported.positions.ravel(),
            transported.gamma.ravel(),
            transported.sigma.ravel(),
            frontier_positions.ravel(),
            second_birth_positions.ravel(),
            probe_velocity.ravel(),
        )
    )

    source_passed = bool(
        _source_events_pass(
            first_node_events,
            first_cell_events,
            expected_step=1,
            expected_mode="first",
        )
        and _source_events_pass(
            second_node_events,
            second_cell_events,
            expected_step=2,
            expected_mode="continuous",
        )
        and all(
            second_event.parent_event_manifest_sha256
            == first_event.producer_manifest_sha256
            for first_event, second_event in zip(
                first_node_events + first_cell_events,
                second_node_events + second_cell_events,
                strict=True,
            )
        )
    )
    cell_source_ownership = bool(
        _cell_strength_ownership_passed(first, first_cell_events)
        and _cell_strength_ownership_passed(second, second_cell_events)
    )
    first_diag = first.diagnostics
    second_diag = second.diagnostics
    ribbon_passed = bool(
        first_diag.first_node_count == len(nodes)
        and first_diag.continuous_node_count == 0
        and second_diag.continuous_node_count == len(nodes)
        and first_diag.seam_count == 0
        and second_diag.seam_count == 0
        and first_diag.incidence_residual <= MAX_RIBBON_RESIDUAL
        and second_diag.incidence_residual <= MAX_RIBBON_RESIDUAL
        and first_diag.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
        and second_diag.edge_reconstruction_residual <= MAX_RIBBON_RESIDUAL
        and first_diag.feedback_call_count == 0
        and second_diag.feedback_call_count == 0
        and first_handoff_integrity["passed"]
        and second_handoff_integrity["passed"]
        and global_exact_once_probe["passed"]
    )
    report_isolated = bool(
        report.feedback_call_count == 0
        and report.parent_write_count == 0
        and report.load_write_count == 0
        and report.observation_access == "none"
        and report.target_case_branch == "none"
    )
    node_placement_passed = bool(
        first_placement_metrics["passed"]
        and second_placement_metrics["passed"]
        and first_half_step_passed
    )
    node_handoff_fields = sorted(
        f"kinematics.{item.name}"
        for item in fields(type(first_placement.kinematics[0]))
    )
    forbidden_node_strength_fields = [
        name
        for name in node_handoff_fields
        if set(name.casefold().replace("-", "_").split("_")).intersection(
            {"circulation", "force", "gamma", "load", "pressure", "strength"}
        )
    ]
    isolation_evidence = {
        "first_ribbon_feedback_call_count": int(first_diag.feedback_call_count),
        "second_ribbon_feedback_call_count": int(second_diag.feedback_call_count),
        "transport_feedback_call_count": int(report.feedback_call_count),
        "transport_parent_write_count": int(report.parent_write_count),
        "transport_surface_channel_write_count": int(report.load_write_count),
        "transport_observation_access": report.observation_access,
        "transport_target_case_branch": report.target_case_branch,
    }
    passed = bool(
        source_passed
        and node_placement_passed
        and continuous_frontier_only_passed
        and cell_source_ownership
        and ribbon_passed
        and first_bridge_metrics["passed"]
        and second_bridge_metrics["passed"]
        and transport_metrics["passed"]
        and overlap["passed"]
        and report_isolated
        and np.all(np.isfinite(state_vector))
    )
    summary = {
        "geometry": geometry,
        "span_cells": span_cells,
        "target_spacing_m": target_spacing_m,
        "smoothing_radius_m": smoothing_radius_m,
        "transport_substeps": transport_substeps,
        "source_transfer_spacing_m": SOURCE_TRANSFER_SPACING_M,
        "node_source_role": "geometry_only",
        "cell_source_role": "circulation_only",
        "node_handoff_schema_fields": node_handoff_fields,
        "forbidden_node_strength_fields": forbidden_node_strength_fields,
        "node_circulation_consumed_count": len(forbidden_node_strength_fields),
        "first_modes": [birth.mode for birth in first.node_births],
        "second_modes": [birth.mode for birth in second.node_births],
        "source_kelvin_max_m2_per_s": max(
            _scaled_kelvin(event)
            for event in (
                first_node_events
                + first_cell_events
                + second_node_events
                + second_cell_events
            )
        ),
        "first_node_placement": first_placement_metrics,
        "second_node_placement": second_placement_metrics,
        "cell_source_coverage_passed": bool(
            first_placement_metrics["cell_source_coverage_passed"]
            and second_placement_metrics["cell_source_coverage_passed"]
        ),
        "first_half_step_passed": first_half_step_passed,
        "first_ribbon_handoff_integrity": first_handoff_integrity,
        "second_ribbon_handoff_integrity": second_handoff_integrity,
        "patch_binding_passed": bool(
            first_handoff_integrity["patch_binding_passed"]
            and second_handoff_integrity["patch_binding_passed"]
        ),
        "global_exact_once_passed": bool(
            first_handoff_integrity["global_exact_once_commit_passed"]
            and second_handoff_integrity["global_exact_once_commit_passed"]
            and global_exact_once_probe["passed"]
        ),
        "global_exact_once_rejection_probe": global_exact_once_probe,
        "continuous_one_third_residual_max_m": (
            max(continuous_residuals) if continuous_residuals else None
        ),
        "continuous_frontier_only_passed": continuous_frontier_only_passed,
        "cell_source_ownership_passed": cell_source_ownership,
        "first_bridge": first_bridge_metrics,
        "second_bridge": second_bridge_metrics,
        "transported_state": transport_metrics,
        "temporal_layer_overlap": overlap,
        "frontier_report_sha256": report.report_sha256,
        "transported_cloud_sha256": report.transported_cloud_digest_after_sha256,
        "state_sha256": _array_sha256(
            transported.positions,
            transported.gamma,
            transported.sigma,
            frontier_positions,
            second_birth_positions,
            probe_velocity,
        ),
        "source_passed": source_passed,
        "node_placement_passed": node_placement_passed,
        "ribbon_passed": ribbon_passed,
        "isolation_evidence": isolation_evidence,
        "report_isolated": report_isolated,
        "passed": passed,
    }
    return _ConfigurationResult(
        summary=summary,
        state_vector=state_vector,
        frontier_positions=frontier_positions,
        probe_velocity=probe_velocity,
    )


def _run_restart_audit(
    geometry: GateGeometry,
    span_cells: int,
) -> dict[str, Any]:
    nodes = _geometry_nodes(geometry, span_cells)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    wing_id = f"restart:{geometry}:wing"
    placement_adapter = NodeLocalDVMPlacementAdapter(wing_id=wing_id)
    ribbon_mapper = NodeOwnedDVMRibbonShadow(
        wing_id=wing_id,
        source_family="lev",
    )
    layers: list[dict[str, Any]] = []
    final_half_step = False
    global_exact_once_probe: dict[str, Any] | None = None
    for step, (alpha, mode) in enumerate(
        (
            (FIRST_ALPHA_RAD, "first"),
            (0.0, "inactive"),
            (CONTINUOUS_ALPHA_RAD, "restart"),
        ),
        start=1,
    ):
        node_events = _step_sources(node_sources, alpha)
        cell_events = _step_sources(cell_sources, alpha)
        placement = _map_node_layer(
            placement_adapter,
            wing_id,
            geometry,
            nodes,
            node_events,
            cell_events,
            source_time_s=(step - 1) * PHYSICAL_RELEASE_DT_S,
        )
        ribbon_cells = _ribbon_cells(geometry, nodes, cell_events)
        ribbon = ribbon_mapper.map_step(
            ribbon_cells,
            placement.kinematics,
            delta_time_s=PHYSICAL_RELEASE_DT_S,
            transport_enabled=True,
            source_time_s=(step - 1) * PHYSICAL_RELEASE_DT_S,
            node_placement_result=placement,
        )
        metrics = _placement_metrics(placement, expected_mode=mode)
        handoff_integrity = _ribbon_handoff_integrity(ribbon, placement)
        if step == 1:
            global_exact_once_probe = _probe_global_exact_once_rejection(
                wing_id=wing_id,
                cells=ribbon_cells,
                placement=placement,
                source_time_s=0.0,
            )
        half_step = (
            _ribbon_half_step_matches_placement(ribbon, placement)
            if mode in {"first", "restart"}
            else ribbon.edge_graph is None
        )
        if mode == "restart":
            final_half_step = half_step
        layers.append(
            {
                "source_step_index": step,
                "mode": mode,
                "placement_manifest_sha256": placement.producer_manifest_sha256,
                "node_modes": [row.placement_mode for row in placement.node_ledgers],
                "ribbon_modes": [birth.mode for birth in ribbon.node_births],
                "half_step_or_inactive_match": half_step,
                "ribbon_handoff_integrity": handoff_integrity,
                "patch_binding_passed": handoff_integrity["patch_binding_passed"],
                "global_exact_once_passed": handoff_integrity[
                    "global_exact_once_commit_passed"
                ],
                "global_exact_once_rejection_probe": (
                    global_exact_once_probe if step == 1 else None
                ),
                "passed": bool(
                    metrics["passed"] and half_step and handoff_integrity["passed"]
                ),
            }
        )
    patch_binding_passed = all(layer["patch_binding_passed"] for layer in layers)
    global_exact_once_passed = all(
        layer["global_exact_once_passed"] for layer in layers
    ) and bool(global_exact_once_probe and global_exact_once_probe["passed"])
    return {
        "sequence": ["first", "inactive", "restart"],
        "active_release_count": 2,
        "cumulative_cloud_merge_performed": False,
        "layers": layers,
        "restart_half_step_passed": final_half_step,
        "patch_binding_passed": patch_binding_passed,
        "global_exact_once_passed": global_exact_once_passed,
        "global_exact_once_rejection_probe": global_exact_once_probe,
        "passed": bool(
            final_half_step
            and patch_binding_passed
            and global_exact_once_passed
            and all(layer["passed"] for layer in layers)
        ),
    }


def _run_core_scan(
    geometry: GateGeometry,
    *,
    span_cells: int,
    target_spacing_m: float,
) -> dict[str, Any]:
    nodes = _geometry_nodes(geometry, span_cells)
    node_sources, cell_sources = _source_sets(geometry, nodes)
    node_events = _step_sources(node_sources, FIRST_ALPHA_RAD)
    cell_events = _step_sources(cell_sources, FIRST_ALPHA_RAD)
    wing_id = f"core-scan:{geometry}:wing"
    placement = _map_node_layer(
        NodeLocalDVMPlacementAdapter(wing_id=wing_id),
        wing_id,
        geometry,
        nodes,
        node_events,
        cell_events,
        source_time_s=0.0,
    )
    ribbon_cells = _ribbon_cells(geometry, nodes, cell_events)
    ribbon = NodeOwnedDVMRibbonShadow(
        wing_id=wing_id,
        source_family="lev",
    ).map_step(
        ribbon_cells,
        placement.kinematics,
        delta_time_s=PHYSICAL_RELEASE_DT_S,
        transport_enabled=True,
        source_time_s=0.0,
        node_placement_result=placement,
    )
    if ribbon.edge_graph is None:
        raise RuntimeError("core scan first layer produced no ribbon graph")
    global_exact_once_probe = _probe_global_exact_once_rejection(
        wing_id=wing_id,
        cells=ribbon_cells,
        placement=placement,
        source_time_s=0.0,
    )
    rows: list[dict[str, Any]] = []
    for multiplier in CORE_SCAN_MULTIPLIERS:
        sigma = SOURCE_TRANSFER_SIGMA_M * multiplier
        bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
            ribbon.edge_graph,
            smoothing_radius=sigma,
            target_spacing=target_spacing_m,
            step=1,
            physical_owner=RING_PHYSICAL_OWNER,
            owner_state=DIAGNOSTIC_SHADOW_OWNER,
        )
        bridge_metrics = _bridge_metrics(
            bridge,
            requested_spacing_m=target_spacing_m,
            smoothing_radius_m=sigma,
        )
        probes = direct_gaussian_erf_velocity_jacobian(
            bridge.positions,
            bridge.gamma,
            bridge.sigma,
            target_positions=FIXED_PROBES_GP1_M,
        ).velocity
        rows.append(
            {
                "sigma_multiplier": multiplier,
                "smoothing_radius_m": sigma,
                "bridge": bridge_metrics,
                "static_probe_velocity_l2_m_per_s": float(np.linalg.norm(probes)),
                "static_probe_velocity_sha256": _array_sha256(probes),
            }
        )
    handoff_integrity = _ribbon_handoff_integrity(ribbon, placement)
    return {
        "diagnostic_surface": "static_first_release_fixed_h_probe_only",
        "gate_eligible": False,
        "selection_role": "diagnostic_only_no_selection",
        "registered_sigma_m_before_scan": SOURCE_TRANSFER_SIGMA_M,
        "requested_target_spacing_m": target_spacing_m,
        "ribbon_handoff_integrity": handoff_integrity,
        "patch_binding_passed": handoff_integrity["patch_binding_passed"],
        "global_exact_once_passed": handoff_integrity["global_exact_once_commit_passed"]
        and global_exact_once_probe["passed"],
        "global_exact_once_rejection_probe": global_exact_once_probe,
        "rows": rows,
    }


def run_minimal_smoke() -> dict[str, Any]:
    """Run PLAN section 9.3's bounded straight two-cell first-release smoke."""

    geometry: GateGeometry = "straight"
    rows: list[dict[str, Any]] = []
    for incidence_sign in (1, -1):
        nodes = _geometry_nodes(geometry, 2)
        node_sources, cell_sources = _source_sets(geometry, nodes)
        node_events = _step_sources(node_sources, incidence_sign * FIRST_ALPHA_RAD)
        cell_events = _step_sources(cell_sources, incidence_sign * FIRST_ALPHA_RAD)
        wing_id = f"smoke:{geometry}:{incidence_sign}:wing"
        placement = _map_node_layer(
            NodeLocalDVMPlacementAdapter(wing_id=wing_id),
            wing_id,
            geometry,
            nodes,
            node_events,
            cell_events,
            source_time_s=0.0,
        )
        ribbon_cells = _ribbon_cells(geometry, nodes, cell_events)
        ribbon = NodeOwnedDVMRibbonShadow(
            wing_id=wing_id,
            source_family="lev",
        ).map_step(
            ribbon_cells,
            placement.kinematics,
            delta_time_s=PHYSICAL_RELEASE_DT_S,
            transport_enabled=True,
            source_time_s=0.0,
            node_placement_result=placement,
        )
        if ribbon.edge_graph is None:
            raise RuntimeError("minimal first release produced no ribbon graph")
        global_exact_once_probe = _probe_global_exact_once_rejection(
            wing_id=wing_id,
            cells=ribbon_cells,
            placement=placement,
            source_time_s=0.0,
        )
        bridge = deposit_edge_graph_prescribed_sigma_and_spacing(
            ribbon.edge_graph,
            smoothing_radius=SOURCE_TRANSFER_SIGMA_M,
            target_spacing=SOURCE_TRANSFER_SPACING_M,
            step=1,
            physical_owner=RING_PHYSICAL_OWNER,
            owner_state=DIAGNOSTIC_SHADOW_OWNER,
        )
        placement_metrics = _placement_metrics(placement, expected_mode="first")
        bridge_metrics = _bridge_metrics(
            bridge,
            requested_spacing_m=SOURCE_TRANSFER_SPACING_M,
            smoothing_radius_m=SOURCE_TRANSFER_SIGMA_M,
        )
        direct_attestation = all(
            validate_dvm_source_event(event) is event
            for event in node_events + cell_events
        )
        half_step = _ribbon_half_step_matches_placement(ribbon, placement)
        strength_owner = _cell_strength_ownership_passed(ribbon, cell_events)
        handoff_integrity = _ribbon_handoff_integrity(ribbon, placement)
        passed = bool(
            direct_attestation
            and placement_metrics["passed"]
            and half_step
            and strength_owner
            and bridge_metrics["passed"]
            and ribbon.diagnostics.seam_count == 0
            and handoff_integrity["passed"]
            and global_exact_once_probe["passed"]
        )
        rows.append(
            {
                "incidence_sign": incidence_sign,
                "direct_event_attestation_passed": direct_attestation,
                "node_placement_manifest_sha256": (placement.producer_manifest_sha256),
                "node_placement_passed": placement_metrics["passed"],
                "cell_source_coverage_passed": placement_metrics[
                    "cell_source_coverage_passed"
                ],
                "first_half_step_passed": half_step,
                "cell_source_ownership_passed": strength_owner,
                "ribbon_handoff_integrity": handoff_integrity,
                "patch_binding_passed": handoff_integrity["patch_binding_passed"],
                "global_exact_once_passed": handoff_integrity[
                    "global_exact_once_commit_passed"
                ]
                and global_exact_once_probe["passed"],
                "global_exact_once_rejection_probe": global_exact_once_probe,
                "particle_count": bridge_metrics["particle_count"],
                "state_sha256": _array_sha256(
                    bridge.positions, bridge.gamma, bridge.sigma
                ),
                "passed": passed,
            }
        )

    disabled_source = _new_source(
        geometry="straight",
        role="node_geometry",
        index=999,
        chord_m=ROOT_CHORD_M,
    )
    step_before = disabled_source.step_count
    disabled_event = disabled_source.step(
        _ExplodingInput(),
        _ExplodingInput(),
        _ExplodingInput(),
        enabled=False,
        delta_time_convective=_ExplodingInput(),
    )
    disabled_adapter = NodeLocalDVMPlacementAdapter(wing_id="smoke:disabled:wing")
    disabled_placement = disabled_adapter.map_step(
        _ExplodingInput(),  # type: ignore[arg-type]
        delta_time_s=_ExplodingInput(),
        enabled=False,
    )
    disabled_ribbon = NodeOwnedDVMRibbonShadow(
        wing_id="smoke:disabled:wing",
        source_family="lev",
    ).map_step(
        _ExplodingInput(),  # type: ignore[arg-type]
        _ExplodingInput(),  # type: ignore[arg-type]
        delta_time_s=_ExplodingInput(),
        enabled=False,
        transport_enabled=_ExplodingInput(),  # type: ignore[arg-type]
        source_time_s=_ExplodingInput(),
        frontier_transport_report=_ExplodingInput(),  # type: ignore[arg-type]
    )
    disabled_input_blind = bool(
        not disabled_event.enabled
        and disabled_source.step_count == step_before
        and validate_dvm_source_event(disabled_event) is disabled_event
        and not disabled_placement.enabled
        and validate_live_dvm_node_placement_result(
            disabled_placement,
            expected_wing_id="smoke:disabled:wing",
        )
        is disabled_placement
        and not disabled_ribbon.diagnostics.enabled
        and disabled_ribbon.edge_graph is None
    )
    return {
        "geometry": geometry,
        "span_cells": 2,
        "release_scope": "first_only",
        "incidence_signs": rows,
        "disabled_input_blind_passed": disabled_input_blind,
        "patch_binding_passed": bool(
            rows and all(row["patch_binding_passed"] for row in rows)
        ),
        "global_exact_once_passed": bool(
            rows and all(row["global_exact_once_passed"] for row in rows)
        ),
        "passed": bool(disabled_input_blind and all(row["passed"] for row in rows)),
    }


def _source_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    platform_root = repo_root / "platform"
    benchmark_root = platform_root / "forward_flight_benchmarks"
    return {
        "runner": Path(__file__).resolve(),
        "runner_test": platform_root / "tests/test_run_v5h_dvm_node_fixed_core_gate.py",
        "benchmark_package_init": benchmark_root / "__init__.py",
        "dvm_node_placement": benchmark_root / "v5h_dvm_node_placement.py",
        "dvm_source": benchmark_root / "v5h_dvm_source.py",
        "dvm_node_ribbon": benchmark_root / "v5h_dvm_node_ribbon.py",
        "passive_frontier_transport": benchmark_root
        / "v5h_passive_frontier_transport.py",
        "ldvm_fourier": platform_root / "ldvm_fourier.py",
        "ldvm_induction_kernel": platform_root / "flap_ldvm.py",
        "ldvm_section_contract": benchmark_root / "ldvm_uvlm_correction.py",
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


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"required runtime distribution is unavailable: {distribution}"
        ) from exc


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
            key: value
            for key, ok, value in (
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
    """Capture run-specific provenance without contaminating semantic hashes."""

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


def _gate_summary(
    *,
    smoke: dict[str, Any],
    geometries: Sequence[dict[str, Any]],
    deterministic_replay_passed: bool,
    declared_source_snapshot_stable_passed: bool,
    runtime_boundary_passed: bool,
) -> dict[str, Any]:
    configuration_mechanics = bool(
        geometries
        and all(geometry["configuration_mechanics_passed"] for geometry in geometries)
    )
    node_handoff = bool(
        smoke["passed"]
        and geometries
        and all(geometry["node_handoff_passed"] for geometry in geometries)
    )
    restart = bool(
        geometries
        and all(geometry["restart_audit"]["passed"] for geometry in geometries)
    )
    time_refinement = bool(
        geometries
        and all(geometry["time_refinement_passed"] for geometry in geometries)
    )
    quadrature = bool(
        geometries
        and all(geometry["fixed_core_quadrature_passed"] for geometry in geometries)
    )
    patch_binding = bool(
        smoke["patch_binding_passed"]
        and geometries
        and all(geometry["patch_binding_passed"] for geometry in geometries)
    )
    global_exact_once = bool(
        smoke["global_exact_once_passed"]
        and geometries
        and all(geometry["global_exact_once_passed"] for geometry in geometries)
    )
    passed = bool(
        smoke["passed"]
        and configuration_mechanics
        and node_handoff
        and restart
        and time_refinement
        and quadrature
        and patch_binding
        and global_exact_once
        and deterministic_replay_passed
        and declared_source_snapshot_stable_passed
        and runtime_boundary_passed
    )
    return {
        "minimal_smoke_passed": bool(smoke["passed"]),
        "node_handoff_passed": node_handoff,
        "restart_half_step_passed": restart,
        "configuration_mechanics_passed": configuration_mechanics,
        "time_refinement_passed": time_refinement,
        "fixed_core_quadrature_passed": quadrature,
        "patch_binding_passed": patch_binding,
        "global_exact_once_passed": global_exact_once,
        "deterministic_replay_passed": deterministic_replay_passed,
        "declared_source_snapshot_stable_passed": (
            declared_source_snapshot_stable_passed
        ),
        "runtime_boundary_passed": runtime_boundary_passed,
        "core_scan_gate_eligible": False,
        "passed": passed,
        "stop_required": not passed,
    }


def _run_fixed_core_gate_impl(config: GateConfig) -> dict[str, Any]:
    """Run the numerical gate; the public wrapper adds measured boundaries."""

    _validate_config(config)
    source_hashes_before = _source_hashes()
    smoke = run_minimal_smoke()
    all_configuration_results: list[_ConfigurationResult] = []
    replay: _ConfigurationResult | None = None
    if not smoke["passed"]:
        geometries: list[dict[str, Any]] = []
        deterministic_replay_passed = False
    else:
        geometries = []
        first_reference: _ConfigurationResult | None = None
        for geometry in CASE_GEOMETRIES:
            restart_audit = _run_restart_audit(geometry, config.span_cells)
            by_spacing: list[list[_ConfigurationResult]] = []
            for target_spacing in config.target_spacings_m:
                time_rows = [
                    _run_configuration(
                        geometry,
                        span_cells=config.span_cells,
                        target_spacing_m=target_spacing,
                        smoothing_radius_m=SOURCE_TRANSFER_SIGMA_M,
                        transport_substeps=substeps,
                    )
                    for substeps in config.transport_substeps
                ]
                by_spacing.append(time_rows)
                all_configuration_results.extend(time_rows)
            if geometry == CASE_GEOMETRIES[0]:
                first_reference = by_spacing[0][0]
            time_gates = [_time_refinement_gate(rows) for rows in by_spacing]
            quadrature_rows = [rows[-1] for rows in by_spacing]
            quadrature_gate = _quadrature_refinement_gate(quadrature_rows)
            core_scan = _run_core_scan(
                geometry,
                span_cells=config.span_cells,
                target_spacing_m=config.target_spacings_m[-1],
            )
            configuration_mechanics_passed = all(
                row.summary["passed"] for rows in by_spacing for row in rows
            )
            node_handoff_passed = all(
                row.summary["node_placement_passed"]
                and row.summary["continuous_frontier_only_passed"]
                for rows in by_spacing
                for row in rows
            )
            patch_binding_passed = bool(
                restart_audit["patch_binding_passed"]
                and core_scan["patch_binding_passed"]
                and all(
                    row.summary["patch_binding_passed"]
                    for rows in by_spacing
                    for row in rows
                )
            )
            global_exact_once_passed = bool(
                restart_audit["global_exact_once_passed"]
                and core_scan["global_exact_once_passed"]
                and all(
                    row.summary["global_exact_once_passed"]
                    for rows in by_spacing
                    for row in rows
                )
            )
            geometry_result = {
                "geometry": geometry,
                "restart_audit": restart_audit,
                "configurations": [row.summary for rows in by_spacing for row in rows],
                "time_refinement_by_requested_spacing": [
                    {
                        "requested_target_spacing_m": target_spacing,
                        **gate,
                    }
                    for target_spacing, gate in zip(
                        config.target_spacings_m, time_gates, strict=True
                    )
                ],
                "fixed_core_quadrature_at_finest_time": quadrature_gate,
                "core_scan_diagnostic": core_scan,
                "configuration_mechanics_passed": configuration_mechanics_passed,
                "node_handoff_passed": node_handoff_passed,
                "patch_binding_passed": patch_binding_passed,
                "global_exact_once_passed": global_exact_once_passed,
                "time_refinement_passed": all(gate["passed"] for gate in time_gates),
                "fixed_core_quadrature_passed": quadrature_gate["passed"],
                "passed": bool(
                    restart_audit["passed"]
                    and configuration_mechanics_passed
                    and node_handoff_passed
                    and patch_binding_passed
                    and global_exact_once_passed
                    and all(gate["passed"] for gate in time_gates)
                    and quadrature_gate["passed"]
                ),
            }
            geometries.append(geometry_result)

        if first_reference is None:
            raise RuntimeError("full gate produced no replay reference")
        replay = _run_configuration(
            CASE_GEOMETRIES[0],
            span_cells=config.span_cells,
            target_spacing_m=config.target_spacings_m[0],
            smoothing_radius_m=SOURCE_TRANSFER_SIGMA_M,
            transport_substeps=config.transport_substeps[0],
        )
        deterministic_replay_passed = bool(
            replay.summary == first_reference.summary
            and np.array_equal(replay.state_vector, first_reference.state_vector)
            and np.array_equal(
                replay.frontier_positions, first_reference.frontier_positions
            )
            and np.array_equal(replay.probe_velocity, first_reference.probe_velocity)
        )

    source_hashes_after = _source_hashes()
    declared_source_snapshot_stable_passed = source_hashes_before == source_hashes_after
    gates = _gate_summary(
        smoke=smoke,
        geometries=geometries,
        deterministic_replay_passed=deterministic_replay_passed,
        declared_source_snapshot_stable_passed=(declared_source_snapshot_stable_passed),
        runtime_boundary_passed=True,
    )
    configuration_rows = [
        row for geometry in geometries for row in geometry["configurations"]
    ]
    measured_node_circulation_count = sum(
        int(row["node_circulation_consumed_count"]) for row in configuration_rows
    )
    measured_feedback_count = sum(
        int(row["isolation_evidence"][key])
        for row in configuration_rows
        for key in (
            "first_ribbon_feedback_call_count",
            "second_ribbon_feedback_call_count",
            "transport_feedback_call_count",
        )
    )
    measured_parent_write_count = sum(
        int(row["isolation_evidence"]["transport_parent_write_count"])
        for row in configuration_rows
    )
    measured_load_write_count = sum(
        int(row["isolation_evidence"]["transport_surface_channel_write_count"])
        for row in configuration_rows
    )
    result = {
        "schema_id": RUN_SCHEMA_ID,
        "run_id": RUN_ID,
        "tier": RUN_TIER,
        "status": "go_v1_only" if gates["passed"] else "stop",
        "passed": gates["passed"],
        "scope": "non_target_node_dvm_two_release_fixed_core_mechanical_v1",
        "observation_access": "none",
        "target_case_branch": "none",
        "canonical_eligible": False,
        "ownership": {
            "surface_aerodynamics_owner": "Ptera_FluxV_external_not_called",
            "node_local_dvm_role": "geometry_only",
            "cell_centre_dvm_role": "circulation_only",
            "node_circulation_consumed_count": measured_node_circulation_count,
            "post_birth_transport_owner": "rVPM",
            "continuous_topology_owner": "attested_rvpm_frontier",
            "parent_write_count": measured_parent_write_count,
            "feedback_call_count": measured_feedback_count,
            "surface_channel_write_count": measured_load_write_count,
            "evidence_kind": "aggregated_live_component_diagnostics",
        },
        "config": {
            "geometries": list(CASE_GEOMETRIES),
            "span_cells": config.span_cells,
            "target_spacings_m": list(config.target_spacings_m),
            "transport_substeps": list(config.transport_substeps),
            "reference_speed_m_per_s": REFERENCE_SPEED_M_PER_S,
            "physical_release_dt_s": PHYSICAL_RELEASE_DT_S,
            "first_incidence_deg": 35.0,
            "continuous_incidence_deg": 55.0,
            "restart_audit_incidence_deg": [35.0, 0.0, 55.0],
            "source_particles_per_release": SOURCE_PARTICLES_PER_RELEASE,
            "source_transfer_spacing_m": SOURCE_TRANSFER_SPACING_M,
            "overlap_lambda": FROZEN_OVERLAP_LAMBDA,
            "fixed_physical_sigma_by_family_m": {"lev": SOURCE_TRANSFER_SIGMA_M},
            "sigma_preregistration_law": "lambda_times_U_times_dt_over_p",
            "core_scan_multipliers": list(CORE_SCAN_MULTIPLIERS),
            "core_scan_selection_role": "diagnostic_only_no_selection",
        },
        "thresholds": {
            "source_kelvin_m2_per_s": MAX_SOURCE_KELVIN_M2_PER_S,
            "ribbon_residual": MAX_RIBBON_RESIDUAL,
            "vector_absolute": MAX_VECTOR_ABS,
            "vector_relative": MAX_VECTOR_REL,
            "particle_invariant_relative": MAX_PARTICLE_INVARIANT_REL,
            "minimum_temporal_two_layer_overlap": (MIN_TEMPORAL_TWO_LAYER_OVERLAP),
            "fine_time_relative_difference": MAX_FINE_TIME_RELATIVE_DIFFERENCE,
            "minimum_time_refinement_ratio": MIN_TIME_REFINEMENT_RATIO,
            "fine_quadrature_relative_difference": (
                MAX_FINE_QUADRATURE_RELATIVE_DIFFERENCE
            ),
            "minimum_quadrature_refinement_ratio": (MIN_QUADRATURE_REFINEMENT_RATIO),
        },
        "minimal_smoke": smoke,
        "geometries": geometries,
        "gate_summary": gates,
        "limitations": LIMITATIONS,
        "blocked": {
            "cumulative_cloud_v2": (
                "blocked until exact-once multi-release merge is implemented and audited"
            ),
            "third_release_transport": PASSIVE_FRONTIER_CONTINUATION_SCOPE,
            "paper_scoring": "prohibited by this auxiliary mechanical gate",
        },
        "declared_semantic_source_sha256": source_hashes_after,
        "raw_refinement_evidence": _make_raw_refinement_evidence(
            all_configuration_results,
            replay,
        ),
    }
    _json_text(result)
    return result


def run_fixed_core_gate(config: GateConfig = GateConfig()) -> dict[str, Any]:
    """Run the complete gate with measured observation/call boundaries."""

    instrumentation = _RuntimeBoundaryInstrumentation()
    with instrumentation:
        result = _run_fixed_core_gate_impl(config)
    instrumented = instrumentation.evidence()
    direct_forbidden_imports = _direct_forbidden_imports()
    surface_quantity_paths = _surface_quantity_key_paths(result)
    forbidden_node_fields = sorted(
        {
            field_name
            for geometry in result["geometries"]
            for row in geometry["configurations"]
            for field_name in row["forbidden_node_strength_fields"]
        }
    )
    measured_component_counts = {
        key: int(result["ownership"][key])
        for key in (
            "feedback_call_count",
            "node_circulation_consumed_count",
            "parent_write_count",
            "surface_channel_write_count",
        )
    }
    runtime_boundary_passed = bool(
        instrumented["passed"]
        and not direct_forbidden_imports
        and not surface_quantity_paths
        and not forbidden_node_fields
        and not any(measured_component_counts.values())
    )
    result["runtime_boundary_evidence"] = {
        "package_init_eager_target_or_ptera_definitions_loaded": any(
            module_name in sys.modules
            for module_name in (
                "forward_flight_benchmarks.baik2012",
                "forward_flight_benchmarks.ptera_adapter",
                "pterasoftware",
            )
        ),
        "instrumented_access": instrumented,
        "direct_forbidden_imports": direct_forbidden_imports,
        "surface_quantity_key_paths": surface_quantity_paths,
        "forbidden_node_handoff_strength_fields": forbidden_node_fields,
        "measured_component_counts": measured_component_counts,
        "passed": runtime_boundary_passed,
    }
    gates = result["gate_summary"]
    gates["runtime_boundary_passed"] = runtime_boundary_passed
    gates["passed"] = bool(gates["passed"] and runtime_boundary_passed)
    gates["stop_required"] = not gates["passed"]
    result["passed"] = gates["passed"]
    result["status"] = "go_v1_only" if gates["passed"] else "stop"
    _json_text(result)
    return result


def _artifact_readme(result: dict[str, Any]) -> str:
    verdict = "GO for bounded v1 mechanics" if result["passed"] else "STOP"
    return (
        "# v5h node-DVM fixed-core auxiliary gate\n\n"
        f"Verdict: **{verdict}**.\n\n"
        "This is an observation-free, noncanonical mechanical artifact. "
        "Shared-node DVM instances supply geometry only; cell-centre DVM "
        "instances supply ribbon circulation only. The physical smoothing "
        "radius was fixed before quadrature refinement, while requested and "
        "realized edge spacings were recorded separately.\n\n"
        "The reported core scan is diagnostic-only and cannot select the "
        "registered smoothing radius. Cumulative-cloud v2 and all paper "
        "scoring remain blocked regardless of this bounded v1 verdict.\n\n"
        "`raw_refinement.json` contains every state/frontier/probe array needed "
        "to reconstruct the refinement metrics. `semantic_manifest.json` "
        "excludes run UUID, timestamps, paths, and other invocation provenance "
        "so independent runs can be compared by one deterministic digest.\n"
    )


def _semantic_result(
    result: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = dict(result)
    raw = semantic.pop("raw_refinement_evidence", None)
    if not isinstance(raw, dict):
        raise ValueError("run result has no raw refinement evidence")
    return semantic, raw


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
    if not isinstance(files, dict) or not files:
        raise ValueError("semantic manifest contains no files")
    if set(files) != SEMANTIC_PAYLOAD_FILENAMES:
        raise ValueError("semantic manifest does not contain the exact payload set")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("semantic manifest file row is invalid")
        if _sha256_file(output / name) != expected:
            raise ValueError(f"semantic payload SHA-256 mismatch: {name}")
    expected_digest = _json_payload_sha256(
        "fluxv-v5h-semantic-result-v1",
        _semantic_manifest_core(files),
    )
    if manifest.get("semantic_result_sha256") != expected_digest:
        raise ValueError("semantic result SHA-256 is inconsistent")
    return expected_digest


def write_run_artifacts(
    result: dict[str, Any],
    output_dir: Path,
    *,
    run_provenance: Mapping[str, Any],
) -> Path:
    """Write strict semantic and run-specific payloads into one new directory."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    required_provenance_fields = {
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
    if set(run_provenance) != required_provenance_fields:
        raise ValueError("run provenance fields are incomplete or unknown")
    if run_provenance.get("schema_id") != RUN_PROVENANCE_SCHEMA_ID:
        raise ValueError("run provenance schema is unsupported")
    run_uuid = run_provenance.get("run_uuid")
    if not isinstance(run_uuid, str) or _validated_run_uuid(run_uuid) != run_uuid:
        raise ValueError("run provenance UUID is invalid")
    started = datetime.fromisoformat(
        _validated_utc(
            "numerical_started_utc", run_provenance["numerical_started_utc"]
        ).replace("Z", "+00:00")
    )
    completed = datetime.fromisoformat(
        _validated_utc(
            "numerical_completed_utc", run_provenance["numerical_completed_utc"]
        ).replace("Z", "+00:00")
    )
    captured = datetime.fromisoformat(
        _validated_utc(
            "provenance_captured_utc", run_provenance["provenance_captured_utc"]
        ).replace("Z", "+00:00")
    )
    if not started <= completed <= captured:
        raise ValueError("run provenance UTC ordering is invalid")
    process_argv = run_provenance.get("process_argv")
    if (
        not isinstance(process_argv, list)
        or not process_argv
        or any(not isinstance(item, str) for item in process_argv)
        or run_provenance.get("process_command") != shlex.join(process_argv)
        or not isinstance(run_provenance.get("python_sys_argv"), list)
        or not isinstance(run_provenance.get("environment"), Mapping)
        or not isinstance(run_provenance.get("git"), Mapping)
        or not isinstance(run_provenance.get("observed_repo_modules"), Mapping)
    ):
        raise ValueError("run provenance structural evidence is incomplete")
    if run_provenance.get("output_dir") != str(output.resolve(strict=False)):
        raise ValueError("run provenance output directory is not the write target")

    summary, raw_refinement = _semantic_result(result)
    declared_sources = summary.get("declared_semantic_source_sha256")
    if declared_sources != _source_hashes():
        raise RuntimeError("declared source hashes are stale at artifact write")
    recomputed = _recompute_refinement_gates(summary, raw_refinement)
    if not recomputed["reported_values_match_raw_recomputation"]:
        raise RuntimeError("reported refinement gates disagree with raw arrays")

    source_manifest = {
        "schema_id": DECLARED_SOURCE_MANIFEST_SCHEMA_ID,
        "run_id": RUN_ID,
        "closure_scope": (
            "declared semantic gate sources checked before and after the numerical "
            "run; this is not claimed as a complete runtime import closure"
        ),
        "runtime_import_closure_complete": False,
        "observed_runtime_modules_location": (
            "run_manifest.json.provenance.observed_repo_modules"
        ),
        "verified_against_current_files_at_artifact_write": True,
        "files": declared_sources,
    }
    metrics = {
        "schema_id": "fluxv-v5h-fixed-core-per-configuration-metrics-v2",
        "run_id": RUN_ID,
        "minimal_smoke": summary["minimal_smoke"],
        "geometries": summary["geometries"],
        "gate_summary": summary["gate_summary"],
    }

    semantic_payloads: dict[str, bytes] = {
        "summary.json": _json_text(summary).encode("utf-8"),
        "metrics.json": _json_text(metrics).encode("utf-8"),
        RAW_REFINEMENT_ARTIFACT: _json_text(raw_refinement).encode("utf-8"),
        RECOMPUTED_GATE_ARTIFACT: _json_text(recomputed).encode("utf-8"),
        "source_manifest.json": _json_text(source_manifest).encode("utf-8"),
        "README.md": _artifact_readme(summary).encode("utf-8"),
    }
    semantic_files = {
        name: sha256(content).hexdigest()
        for name, content in sorted(semantic_payloads.items())
    }
    semantic_core = _semantic_manifest_core(semantic_files)
    semantic_result_sha256 = _json_payload_sha256(
        "fluxv-v5h-semantic-result-v1", semantic_core
    )
    semantic_manifest = {
        **semantic_core,
        "semantic_result_sha256": semantic_result_sha256,
        "scope": (
            "deterministic semantic payloads only; run UUID, clocks, argv, paths, "
            "git state, environment, run log, and outer result hashes are excluded"
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
        "schema_id": "fluxv-v5h-fixed-core-run-manifest-v2",
        "run_id": RUN_ID,
        "tier": RUN_TIER,
        "semantic_result_sha256": semantic_result_sha256,
        "config": summary["config"],
        "provenance": dict(run_provenance),
        "runtime_boundary_evidence": summary["runtime_boundary_evidence"],
        "declared_source_scope": source_manifest["closure_scope"],
        "observation_access": "none",
        "target_case_branch": "none",
    }
    run_uuid = str(run_provenance["run_uuid"])
    log_text = (
        f"run_id={RUN_ID}\n"
        f"run_uuid={run_uuid}\n"
        f"tier={RUN_TIER}\n"
        f"status={summary['status']}\n"
        f"passed={str(bool(summary['passed'])).lower()}\n"
        f"geometry_count={len(summary['geometries'])}\n"
        f"numerical_started_utc={run_provenance['numerical_started_utc']}\n"
        f"numerical_completed_utc={run_provenance['numerical_completed_utc']}\n"
        f"output_dir={run_provenance['output_dir']}\n"
        f"semantic_result_sha256={semantic_result_sha256}\n"
        "observation_access=none\n"
        "core_scan_role=diagnostic_only_no_selection\n"
        "cumulative_cloud_v2=blocked\n"
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

    if declared_sources != _source_hashes():
        raise RuntimeError("declared source hashes changed during artifact assembly")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (output / name).write_bytes(content)
    for path in output.glob("*.json"):
        _load_strict_json(path)
    for name, expected_digest in result_manifest["files"].items():
        if _sha256_file(output / name) != expected_digest:
            raise RuntimeError(f"result hash closure failed for {name}")
    checksum_rows = {
        name: digest
        for digest, name in (
            line.split("  ", maxsplit=1)
            for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        )
    }
    if checksum_rows != {
        name: _sha256_file(output / name) for name in payloads if name != "SHA256SUMS"
    }:
        raise RuntimeError("external SHA256SUMS closure is inconsistent")
    reloaded_recomputation = recompute_refinement_gates_from_artifacts(output)
    if reloaded_recomputation != recomputed:
        raise RuntimeError("on-disk raw refinement recomputation is inconsistent")
    if verify_semantic_manifest(output) != semantic_result_sha256:
        raise RuntimeError("semantic result verification is inconsistent")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    numerical_started_utc = _utc_now()
    result = run_fixed_core_gate()
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


if __name__ == "__main__":
    main()
