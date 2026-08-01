"""Isolated, resumable Fig. 17/18/19 runner for research candidates.

This runner deliberately does not read measured force values.  It imports only
the canonical solver-condition contract from :mod:`fig171819_benchmark`, calls
``gpu_run_twist`` once per condition, and checkpoints into a candidate-owned
directory.  It never writes the frozen production sweep files.

Examples
--------
Formal 32-condition representative screen::

    python platform/lb_sweep_candidate.py \
      --candidate-id n3_spatial_pressure_v0 \
      --closure n3_spatial_pressure_v0 \
      --scope representative32

Fast three-condition plumbing check::

    python platform/lb_sweep_candidate.py \
      --candidate-id n3_spatial_pressure_v0 \
      --closure n3_spatial_pressure_v0 \
      --scope smoke3 --quick

N3-only spatial edge-pressure shadow (retains every V4.1 production
channel except for the explicitly substituted N3 force)::

    python platform/lb_sweep_candidate.py \
      --candidate-id n3_spatial_edge_pressure_v1_shadow \
      --closure n3_spatial_edge_pressure_v1_shadow \
      --scope smoke3 --quick

Resume the exact same run identity::

    python platform/lb_sweep_candidate.py \
      --candidate-id n3_spatial_pressure_v0 \
      --closure n3_spatial_pressure_v0 \
      --scope representative32 \
      --resume platform/docs/candidates/n3_spatial_pressure_v0/runs/20260729_190000
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
DOCS = PLATFORM / "docs"
CANDIDATE_ROOT = DOCS / "candidates"
for _path in (ROOT, PLATFORM):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fig171819_benchmark as benchmark  # noqa: E402


Condition = benchmark.Condition
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_TIMESTAMP = re.compile(r"^[0-9]{8}_[0-9]{6}$")
N3_ONLY_CLOSURE = "n3_spatial_edge_pressure_v1_shadow"
N3_ONLY_CLAIM = "N3.1j5"
_RUN_LOCK_NAME = ".campaign.lock"
_N3_ONLY_REQUIRED_GUARDS = frozenset(
    {
        "aero_output_invariance",
        "cycle_reduction",
        "force_ledger",
        "n3_spatial_n3only",
        "unclassified_force",
        "unclassified_physical_force",
    }
)

PRODUCTION_GRID = {
    "nc": 12,
    "ns": 16,
    "n_cycle": 4,
    "steps_per_cycle": "spc_of(U,f,nc)",
    "wake_rows": "steps_per_cycle",
}
QUICK_GRID = {
    "nc": 4,
    "ns": 8,
    "n_cycle": 2,
    "steps_per_cycle": 60,
    "wake_rows": 60,
}

_CLOSURE_MODEL_OVERRIDES: Mapping[str, Mapping[str, Any]] = {
    # The spatial-pressure closure owns the only LEV state and books its
    # effect through one panel-pressure difference.  H16 is reused only for
    # the common geometry/kinematics/N2/N6 inputs; all of its historical
    # parallel LEV force paths must be disabled explicitly.
    "n3_spatial_pressure_v0": {
        "lev_shed_mode": "none",
        "lev_impulse": False,
        "lev_vnf": False,
        "fp_lev": False,
        "real_lev": False,
        "lev_merge": False,
        "part_lev": False,
        "vortex": False,
        "dstall": False,
        "les_suction": False,
    },
    # This shadow closure must enter gpu_run_twist with the unmodified H16
    # production inputs.  In particular, do not inherit any of the v0
    # overrides above: N1 leading-edge suction and every other V4.1
    # production channel remain active, and only N3 is substituted in-call.
    N3_ONLY_CLOSURE: {},
}

_CLOSURE_MODEL_ARG_ALLOWLIST: Mapping[str, frozenset[str]] = {
    # The only preregistered v1 parameter family is the q16 -> q24
    # quadrature check.  Every aerodynamic flag and physical constant is
    # intentionally unavailable through the generic escape hatch.
    N3_ONLY_CLOSURE: frozenset({"spatial_p2_quadrature"}),
}
_N3_ONLY_QUADRATURE_FAMILY = frozenset({16, 24})

# A nested screen: three spanwise-load-sensitive twist points are contained in
# the 32-point screen, which in turn is contained in the confirmed contract.
SMOKE3: tuple[Condition, ...] = (
    (8.0, benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ, 0.0, 5.0),
    (8.0, benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ, 22.5, 5.0),
    (8.0, benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ, 45.0, 5.0),
)


def _sorted_conditions(conditions: Iterable[Condition]) -> tuple[Condition, ...]:
    return tuple(
        sorted(
            {tuple(float(value) for value in condition) for condition in conditions},
            key=lambda item: tuple(float(value) for value in item),
        )
    )


REPRESENTATIVE32 = _sorted_conditions(
    # One complete Fig. 17 line at the benchmark's declared Fig. 19(c,d)
    # conditional frequency.  It also supplies the corresponding U=8
    # Fig. 18(c,d) and AoA=5 Fig. 19(c,d) traces without inventing another
    # condition convention inside this runner.
    (
        (
            8.0,
            benchmark.FIG19_CD_FIXED_FREQUENCY_ASSUMPTION_HZ,
            twist,
            5.0,
        )
        for twist in benchmark.TWS
    )
)
REPRESENTATIVE32 = _sorted_conditions(
    REPRESENTATIVE32
    # Fig. 18(a,b), two non-central wind-speed frequency lines.
    + tuple(
        (U, freq, 22.5, 5.0)
        for U in (6.0, 10.0)
        for freq in benchmark.FS
    )
    # Fig. 19(a,b), low/high AoA frequency lines.
    + tuple(
        (8.0, freq, 22.5, aoa)
        for aoa in (0.0, 15.0)
        for freq in benchmark.FS
    )
)

SCOPE_CONDITIONS: Mapping[str, tuple[Condition, ...]] = {
    "smoke3": SMOKE3,
    "representative32": REPRESENTATIVE32,
    "confirmed151": _sorted_conditions(
        benchmark.CONDITIONS_BY_EVIDENCE_SCOPE[benchmark.EVIDENCE_CONFIRMED]
    ),
    "conditional184": _sorted_conditions(benchmark.CONDITIONS),
}
_EXPECTED_COUNTS = {
    "smoke3": 3,
    "representative32": 32,
    "confirmed151": 151,
    "conditional184": 184,
}
for _scope, _expected in _EXPECTED_COUNTS.items():
    if len(SCOPE_CONDITIONS[_scope]) != _expected:
        raise AssertionError(
            f"candidate scope drift: {_scope} has "
            f"{len(SCOPE_CONDITIONS[_scope])}, expected {_expected}"
        )
if not set(SMOKE3) <= set(REPRESENTATIVE32):
    raise AssertionError("candidate scope drift: smoke3 is not nested in representative32")
if not set(REPRESENTATIVE32) <= set(SCOPE_CONDITIONS["confirmed151"]):
    raise AssertionError(
        "candidate scope drift: representative32 is not nested in confirmed151"
    )


def condition_key(condition: Condition) -> str:
    return benchmark.condition_key(condition)


def spc_of(U: float, freq: float, nc: int) -> int:
    """Production time-step contract at arbitrary explicitly selected ``nc``."""

    return max(60, int(round(15.0 * U * nc / freq / 60.0)) * 60)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _local_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprinted_paths(candidate_id: str) -> tuple[Path, ...]:
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError(
            "candidate id must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    candidate_docs = CANDIDATE_ROOT / candidate_id
    explicit = (
        Path(__file__).resolve(),
        candidate_docs / "PLAN.md",
        candidate_docs / "EXECUTION.md",
        candidate_docs / "DATA_EXPOSURE_ADDENDUM.md",
        PLATFORM / "_v2_robo.py",
        PLATFORM / "_v2_repro_nc12.py",
        PLATFORM / "fig171819_benchmark.py",
        PLATFORM / "score_n3_shadow_gates.py",
        PLATFORM / "plot_candidate_overlay.py",
        PLATFORM / "diff_uvlm_unsteady_gpu.py",
        PLATFORM / "diff_uvlm_unsteady.py",
        PLATFORM / "flap_flight_validate.py",
        PLATFORM / "diff_coupled_unsteady.py",
        PLATFORM / "diff_struct_design.py",
        PLATFORM / "diff_coupled_fsi.py",
        PLATFORM / "diff_vlm.py",
        PLATFORM / "_v2_robogeom.py",
        PLATFORM / "airfoil_geometry.py",
        PLATFORM / "lb_dyn.py",
        PLATFORM / "lb_static.py",
        PLATFORM / "cd_table.py",
        ROOT / "src" / "fluxvortex" / "__init__.py",
        ROOT / "src" / "fluxvortex" / "ancf_shell.py",
        ROOT / "src" / "fluxvortex" / "warp_fsi" / "__init__.py",
        ROOT / "src" / "fluxvortex" / "warp_fsi" / "config.py",
        ROOT / "src" / "fluxvortex" / "warp_fsi" / "batched_solver.py",
    )
    paths = (
        tuple(explicit)
        + tuple(sorted((PLATFORM / "claim_runtime").glob("*.py")))
        + tuple(sorted((PLATFORM / "claim_nodes").glob("*.yaml")))
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        display = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required campaign identity source is missing: {display}"
        )
    return paths


def implementation_fingerprints(
    candidate_id: str = N3_ONLY_CLOSURE,
) -> dict[str, str]:
    """Fingerprint every source that can change this closure or run contract."""

    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in _fingerprinted_paths(candidate_id)
    }


def runtime_environment_identity() -> dict[str, Any]:
    """Resolve the exact numerical runtime used by the solver.

    Importing the shared config here intentionally freezes dtype/device before
    the run manifest is written.  Resume and seed identities must match this
    mapping byte-for-byte, including the physical GPU UUID when CUDA is used.
    """

    import numpy as np
    import warp as wp
    from fluxvortex.warp_fsi import config as flux_config

    wp.init()
    device = wp.get_device(flux_config.DEVICE)
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "fluxv_dtype_env": os.environ.get("FLUXV_DTYPE"),
        "fluxv_dtype_resolved": flux_config.dtype_name(),
        "fluxv_device_env": os.environ.get("FLUXV_DEVICE"),
        "fluxv_device_resolved": str(flux_config.DEVICE),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_alias": str(device.alias),
        "device_name": str(device.name),
        "device_uuid": (
            str(device.uuid) if getattr(device, "uuid", None) else None
        ),
        "device_arch": (
            int(device.arch)
            if getattr(device, "arch", None) is not None
            else None
        ),
        "device_is_cuda": bool(device.is_cuda),
    }


def _git_capture(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_identity() -> dict[str, Any]:
    status = _git_capture("status", "--porcelain=v1")
    return {
        "commit": _git_capture("rev-parse", "HEAD"),
        "branch": _git_capture("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _parse_model_arg(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--model-arg must be KEY=JSON_VALUE")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key or not key.isidentifier():
        raise argparse.ArgumentTypeError(f"invalid model argument name {key!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


_MANAGED_CALL_KEYS = {
    "U",
    "aoa_deg",
    "freq",
    "twist_amp_deg",
    "closure",
    "nc",
    "ns",
    "n_cycle",
    "steps_per_cycle",
    "wake_rows",
}


def _model_args(
    items: Sequence[tuple[str, Any]],
    *,
    closure: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in _MANAGED_CALL_KEYS:
            raise ValueError(
                f"{key!r} is runner-managed; use its dedicated command-line option"
            )
        if key in result:
            raise ValueError(f"duplicate --model-arg {key!r}")
        result[key] = value
    allowed = _CLOSURE_MODEL_ARG_ALLOWLIST.get(closure)
    if allowed is not None:
        forbidden = sorted(set(result) - allowed)
        if forbidden:
            raise ValueError(
                f"{closure} forbids model arguments: "
                + ", ".join(repr(key) for key in forbidden)
            )
    if closure == N3_ONLY_CLOSURE and "spatial_p2_quadrature" in result:
        order = result["spatial_p2_quadrature"]
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order not in _N3_ONLY_QUADRATURE_FAMILY
        ):
            raise ValueError(
                "n3_spatial_edge_pressure_v1_shadow permits only the "
                "preregistered spatial_p2_quadrature values 16 or 24"
            )
    return result


def _candidate_dir(candidate_id: str, timestamp: str) -> Path:
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError(
            "candidate id must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    if not _SAFE_TIMESTAMP.fullmatch(timestamp):
        raise ValueError("timestamp must have YYYYMMDD_HHMMSS form")
    path = (CANDIDATE_ROOT / candidate_id / "runs" / timestamp).resolve()
    path.relative_to(CANDIDATE_ROOT.resolve())
    return path


def _resume_dir(path: str | Path, candidate_id: str) -> Path:
    candidate_parent = (CANDIDATE_ROOT / candidate_id / "runs").resolve()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    resolved.relative_to(candidate_parent)
    return resolved


class _RunDirectoryLock:
    """Non-blocking process lock for one durable campaign directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / _RUN_LOCK_NAME
        self._handle = None

    def __enter__(self) -> "_RunDirectoryLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"campaign run directory is already locked: {self.run_dir}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": _utc_now(),
                    "python_executable": str(Path(sys.executable).resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


class _CampaignDriftError(RuntimeError):
    """One condition completed under a source/runtime identity that drifted."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


_SCOPE_IDENTITY_KEYS = {"scope", "condition_count", "condition_keys"}


def _seed_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model/grid identity that must match across nested scopes."""

    return {
        key: value
        for key, value in identity.items()
        if key not in _SCOPE_IDENTITY_KEYS
    }


def _load_seed_run(
    path: str | Path,
    *,
    candidate_id: str,
    target_identity: Mapping[str, Any],
    target_conditions: Sequence[Condition],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_dir = _resume_dir(path, candidate_id)
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"seed run directory not found: {seed_dir}")
    with _RunDirectoryLock(seed_dir):
        config_path = seed_dir / "config.json"
        results_path = seed_dir / "candidate_results.json"
        status_path = seed_dir / "status.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"seed config not found: {config_path}")
        if not results_path.is_file():
            raise FileNotFoundError(f"seed results not found: {results_path}")
        if not status_path.is_file():
            raise FileNotFoundError(f"seed status not found: {status_path}")
        config = _read_json(config_path)
        source_identity = config.get("run_identity")
        if not isinstance(source_identity, Mapping):
            raise ValueError(f"invalid seed run identity: {config_path}")
        if _seed_identity(source_identity) != _seed_identity(target_identity):
            raise ValueError(
                "seed run model/grid identity differs from the requested run"
            )
        status = _read_json(status_path)
        expected_count = int(source_identity.get("condition_count", -1))
        if (
            not isinstance(status, Mapping)
            or status.get("status") != "complete"
            or status.get("completed_valid") != expected_count
            or status.get("failed", 0) != 0
        ):
            raise ValueError(
                "seed run must be a complete, failure-free formal scope"
            )
        raw_results = _read_json(results_path)
        if not isinstance(raw_results, Mapping):
            raise ValueError(f"invalid seed result mapping: {results_path}")
        closure = str(target_identity.get("closure", ""))
        source_keys = source_identity.get("condition_keys")
        if (
            not isinstance(source_keys, list)
            or len(source_keys) != expected_count
            or len(set(source_keys)) != expected_count
            or not all(isinstance(key, str) and key for key in source_keys)
        ):
            raise ValueError(
                "seed run identity has an invalid formal condition key set"
            )
        invalid_source = [
            key
            for key in source_keys
            if (
                key not in raw_results
                or not _valid_result(raw_results[key], closure=closure)
            )
        ]
        if invalid_source:
            raise ValueError(
                "seed run contains invalid formal-scope records: "
                + ", ".join(invalid_source)
            )
        target_keys = {
            condition_key(condition) for condition in target_conditions
        }
        invalid_overlap = [
            key
            for key in sorted(target_keys & set(raw_results))
            if not _valid_result(raw_results[key], closure=closure)
        ]
        if invalid_overlap:
            raise ValueError(
                "seed run contains invalid overlapping records: "
                + ", ".join(invalid_overlap)
            )
        copied = {
            key: value
            for key, value in raw_results.items()
            if key in target_keys
        }
        provenance = {
            "run_directory": str(seed_dir),
            "source_scope": source_identity.get("scope"),
            "copied_valid_condition_count": len(copied),
            "config_sha256": _sha256_file(config_path),
            "results_sha256": _sha256_file(results_path),
            "status_sha256": _sha256_file(status_path),
            "source_status": status.get("status"),
        }
    return copied, provenance


def _grid_from_args(args: argparse.Namespace) -> dict[str, Any]:
    defaults = QUICK_GRID if args.quick else PRODUCTION_GRID
    return {
        "mode": "quick" if args.quick else "production",
        "nc": args.nc if args.nc is not None else defaults["nc"],
        "ns": args.ns if args.ns is not None else defaults["ns"],
        "n_cycle": (
            args.n_cycle if args.n_cycle is not None else defaults["n_cycle"]
        ),
        "steps_per_cycle": (
            args.steps_per_cycle
            if args.steps_per_cycle is not None
            else defaults["steps_per_cycle"]
        ),
        "wake_rows": (
            args.wake_rows
            if args.wake_rows is not None
            else defaults["wake_rows"]
        ),
    }


def _resolved_step_grid(
    grid: Mapping[str, Any], condition: Condition
) -> tuple[int, int]:
    U, freq, _twist, _aoa = condition
    raw_spc = grid["steps_per_cycle"]
    steps_per_cycle = (
        spc_of(U, freq, int(grid["nc"]))
        if raw_spc == "spc_of(U,f,nc)"
        else int(raw_spc)
    )
    raw_wake = grid["wake_rows"]
    wake_rows = (
        steps_per_cycle if raw_wake == "steps_per_cycle" else int(raw_wake)
    )
    return steps_per_cycle, wake_rows


def _base_model_config() -> dict[str, Any]:
    # Imported lazily so scope/manifest tests do not initialize Warp.
    from _v2_repro_nc12 import CFG_PRESETS

    return dict(
        CFG_PRESETS["H16"],
        fsep_lag=False,
        cosine_chord="le",
        les_sep="plateau_fn",
        d_para=0.5,
        attached_drag="uiuc",
        geo_stall=False,
        flap_amp_deg=22.5,
        twist_phase_deg=90.0,
    )


def _run_identity(
    *,
    candidate_id: str,
    closure: str,
    scope: str,
    grid: Mapping[str, Any],
    model_args: Mapping[str, Any],
    resolved_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "closure": closure,
        "scope": scope,
        "condition_count": len(SCOPE_CONDITIONS[scope]),
        "condition_keys": [
            condition_key(condition) for condition in SCOPE_CONDITIONS[scope]
        ],
        "grid": dict(grid),
        "base_model_profile": "CFG_PRESETS.H16+lb_sweep118.production_overrides",
        "model_args": dict(model_args),
        "resolved_model_config_before_closure_profile": dict(
            resolved_model_config
        ),
        "kinematics": {
            "flap_amp_deg": 22.5,
            "twist_amp_rule": "nominal_twist_deg/2",
            "twist_phase_deg": 90.0,
        },
        "measurement_values_read_by_runner": False,
        "benchmark_condition_source": "fig171819_benchmark.py",
        "implementation_sha256": implementation_fingerprints(candidate_id),
        "runtime_environment": runtime_environment_identity(),
    }


def _result_validation_errors(
    value: Any,
    *,
    closure: str = N3_ONLY_CLOSURE,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["record is not a mapping"]
    try:
        if not (
            math.isfinite(float(value["L"]))
            and math.isfinite(float(value["T"]))
        ):
            errors.append("L/T are non-finite")
    except (KeyError, TypeError, ValueError):
        errors.append("L/T are missing or invalid")
    guards = value.get("claim_guards")
    if not isinstance(guards, Mapping) or not guards:
        errors.append("claim_guards must be a non-empty mapping")
    else:
        malformed_or_failed = [
            str(name)
            for name, payload in guards.items()
            if (
                not isinstance(payload, Mapping)
                or payload.get("passed") is not True
            )
        ]
        if malformed_or_failed:
            errors.append(
                "claim guards are malformed or failed: "
                + ", ".join(sorted(malformed_or_failed))
            )
    if closure == N3_ONLY_CLOSURE:
        if isinstance(guards, Mapping):
            missing_guards = sorted(
                _N3_ONLY_REQUIRED_GUARDS - set(guards)
            )
            if missing_guards:
                errors.append(
                    "required claim guards are missing: "
                    + ", ".join(missing_guards)
                )
        counterfactual_keys = (
            "L_wind_v41_counterfactual",
            "T_wind_v41_counterfactual",
        )
        present = [key in value for key in counterfactual_keys]
        if not all(present):
            errors.append(
                "complete same-call V4.1 counterfactual is required"
            )
        else:
            try:
                if not all(
                    math.isfinite(float(value[key]))
                    for key in counterfactual_keys
                ):
                    errors.append(
                        "same-call V4.1 counterfactual is non-finite"
                    )
            except (TypeError, ValueError):
                errors.append(
                    "same-call V4.1 counterfactual is invalid"
                )
        diagnostic = value.get("n3_spatial_n3only")
        if not isinstance(diagnostic, Mapping):
            errors.append("n3_spatial_n3only diagnostic is required")
        else:
            if diagnostic.get("closure") != N3_ONLY_CLOSURE:
                errors.append("n3_spatial_n3only closure identity mismatch")
            if diagnostic.get("claim_node") != N3_ONLY_CLAIM:
                errors.append("n3_spatial_n3only claim identity mismatch")
        manifest = value.get("claim_manifest")
        if not isinstance(manifest, Mapping):
            errors.append("claim_manifest is required")
        else:
            if manifest.get("closure") != N3_ONLY_CLOSURE:
                errors.append("claim_manifest closure identity mismatch")
            internal_stages = manifest.get("internal_stages")
            if (
                not isinstance(internal_stages, list)
                or len(internal_stages) != 1
                or not isinstance(internal_stages[0], Mapping)
            ):
                errors.append(
                    "claim_manifest must contain exactly one internal stage"
                )
            else:
                stage = internal_stages[0]
                expected_binding = {
                    "id": N3_ONLY_CLAIM,
                    "runtime_owner": "N3",
                    "runtime_binding": "internal_stage",
                }
                mismatches = [
                    f"{key}={stage.get(key)!r}"
                    for key, expected in expected_binding.items()
                    if stage.get(key) != expected
                ]
                if mismatches:
                    errors.append(
                        "claim_manifest internal-stage binding mismatch: "
                        + ", ".join(mismatches)
                    )
    return errors


def _valid_result(
    value: Any,
    *,
    closure: str = N3_ONLY_CLOSURE,
) -> bool:
    return not _result_validation_errors(value, closure=closure)


def _validate_checkpoint_results(
    results: Any,
    *,
    conditions: Sequence[Condition],
    closure: str,
) -> dict[str, Any]:
    """Reject malformed persisted records instead of silently reusing them."""

    if not isinstance(results, Mapping):
        raise ValueError("candidate checkpoint must be a result mapping")
    checked = dict(results)
    for condition in conditions:
        key = condition_key(condition)
        if key not in checked:
            continue
        value = checked[key]
        if isinstance(value, Mapping) and "fail" in value:
            continue
        errors = _result_validation_errors(value, closure=closure)
        if errors:
            raise ValueError(
                f"invalid persisted record {key}: " + "; ".join(errors)
            )
    return checked


def _finite_force_values(output: Mapping[str, Any]) -> tuple[float, float]:
    """Extract solver forces while refusing non-finite campaign records."""

    lift = float(output["L_wind"])
    thrust = float(output["T_wind"])
    if not math.isfinite(lift) or not math.isfinite(thrust):
        raise FloatingPointError(
            f"non-finite solver force: L_wind={lift!r}, T_wind={thrust!r}"
        )
    return lift, thrust


def _finite_counterfactual_force_values(
    output: Mapping[str, Any],
    *,
    required: bool = False,
) -> tuple[float, float] | None:
    """Extract the matched V4.1 counterfactual as one atomic force pair.

    Older candidates do not expose this pair and remain supported.  A solver
    that exposes only one channel, however, has produced an incomplete
    same-call counterfactual and must not enter a campaign checkpoint.
    """

    lift_key = "L_wind_v41_counterfactual"
    thrust_key = "T_wind_v41_counterfactual"
    present = (lift_key in output, thrust_key in output)
    if not any(present):
        if required:
            raise KeyError(
                "complete same-call V4.1 counterfactual is required"
            )
        return None
    if not all(present):
        missing = thrust_key if present[0] else lift_key
        raise KeyError(
            "incomplete same-call V4.1 counterfactual; "
            f"missing {missing!r}"
        )
    lift = float(output[lift_key])
    thrust = float(output[thrust_key])
    if not math.isfinite(lift) or not math.isfinite(thrust):
        raise FloatingPointError(
            "non-finite same-call V4.1 counterfactual force: "
            f"{lift_key}={lift!r}, {thrust_key}={thrust!r}"
        )
    return lift, thrust


def _candidate_record(
    output: Mapping[str, Any],
    *,
    wall_seconds: float,
    steps_per_cycle: int,
    wake_rows: int,
    closure: str | None = None,
) -> dict[str, Any]:
    """Build one finite, JSON-serializable candidate checkpoint record."""

    lift, thrust = _finite_force_values(output)
    record: dict[str, Any] = {
        "L": lift,
        "T": thrust,
        "wall_seconds": float(wall_seconds),
        "steps_per_cycle": int(steps_per_cycle),
        "wake_rows": int(wake_rows),
    }
    counterfactual = _finite_counterfactual_force_values(
        output,
        required=closure == N3_ONLY_CLOSURE,
    )
    if counterfactual is not None:
        (
            record["L_wind_v41_counterfactual"],
            record["T_wind_v41_counterfactual"],
        ) = counterfactual
    for key in (
        "claim_manifest",
        "claim_guards",
        "n3_spatial_p2",
        "n3_spatial_n3only",
    ):
        if key in output:
            record[key] = output[key]
    # Refuse NaN/Inf and non-serializable diagnostics before the record enters
    # the atomic campaign checkpoint.
    json.dumps(record, allow_nan=False)
    if closure is not None:
        errors = _result_validation_errors(record, closure=closure)
        if errors:
            raise ValueError(
                "candidate record failed strict campaign schema: "
                + "; ".join(errors)
            )
    return record


def _final_status(
    *,
    valid_count: int,
    selected_count: int,
    max_conditions: int | None,
) -> str:
    if valid_count != selected_count:
        return "completed_with_failures"
    if max_conditions is not None:
        return "incomplete_debug_prefix"
    return "complete"


def run(args: argparse.Namespace) -> Path:
    model_args = _model_args(args.model_arg, closure=args.closure)
    grid = _grid_from_args(args)
    base_model_config = _base_model_config()
    base_model_config.update(_CLOSURE_MODEL_OVERRIDES.get(args.closure, {}))
    base_model_config.update(model_args)
    identity = _run_identity(
        candidate_id=args.candidate_id,
        closure=args.closure,
        scope=args.scope,
        grid=grid,
        model_args=model_args,
        resolved_model_config=base_model_config,
    )
    conditions = SCOPE_CONDITIONS[args.scope]
    if args.max_conditions is not None:
        conditions = conditions[: args.max_conditions]

    seed_results: dict[str, Any] = {}
    seed_provenance: dict[str, Any] | None = None
    if args.seed_run:
        seed_results, seed_provenance = _load_seed_run(
            args.seed_run,
            candidate_id=args.candidate_id,
            target_identity=identity,
            target_conditions=conditions,
        )

    if args.resume:
        run_dir = _resume_dir(args.resume, args.candidate_id)
    else:
        timestamp = args.timestamp or _local_timestamp()
        run_dir = _candidate_dir(args.candidate_id, timestamp)
        run_dir.mkdir(parents=True, exist_ok=False)
    with _RunDirectoryLock(run_dir):
        config_path = run_dir / "config.json"
        if args.resume:
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"resume config not found: {config_path}"
                )
            existing_config = _read_json(config_path)
            if existing_config.get("run_identity") != identity:
                raise ValueError(
                    "resume identity differs from config.json; "
                    "start a new run directory"
                )
        else:
            _write_json_atomic(
                config_path,
                {
                    "schema_version": 2,
                    "created_at": _utc_now(),
                    "run_identity": identity,
                    "git_at_creation": git_identity(),
                    "command": sys.argv,
                    "seed_run": seed_provenance,
                    "lock_file": _RUN_LOCK_NAME,
                },
            )
        return _run_locked(
            args=args,
            run_dir=run_dir,
            conditions=conditions,
            grid=grid,
            base_model_config=base_model_config,
            identity=identity,
            seed_results=seed_results,
        )


def _run_locked(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    conditions: Sequence[Condition],
    grid: Mapping[str, Any],
    base_model_config: Mapping[str, Any],
    identity: Mapping[str, Any],
    seed_results: Mapping[str, Any],
) -> Path:
    """Execute one campaign while the caller holds its directory lock."""

    results_path = run_dir / "candidate_results.json"
    status_path = run_dir / "status.json"
    raw_results = (
        _read_json(results_path)
        if results_path.is_file()
        else dict(seed_results)
    )
    results = _validate_checkpoint_results(
        raw_results,
        conditions=conditions,
        closure=args.closure,
    )

    if runtime_environment_identity() != identity["runtime_environment"]:
        raise RuntimeError(
            "candidate runtime environment changed before execution"
        )

    if args.dry_run:
        _write_json_atomic(
            status_path,
            {
                "schema_version": 2,
                "status": "dry_run",
                "updated_at": _utc_now(),
                "run_directory": str(run_dir),
                "selected_condition_count": len(conditions),
                "grid": dict(grid),
                "runtime_environment": identity["runtime_environment"],
            },
        )
        for condition in conditions:
            steps, wake = _resolved_step_grid(grid, condition)
            print(
                condition_key(condition),
                f"spc={steps}",
                f"wake_rows={wake}",
            )
        return run_dir

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    base = dict(base_model_config)
    started = time.time()

    def checkpoint(status: str, **extra: Any) -> None:
        _write_json_atomic(results_path, results)
        n_ok = sum(
            _valid_result(
                results.get(condition_key(condition)),
                closure=args.closure,
            )
            for condition in conditions
        )
        n_fail = sum(
            isinstance(results.get(condition_key(condition)), Mapping)
            and "fail" in results[condition_key(condition)]
            for condition in conditions
        )
        _write_json_atomic(
            status_path,
            {
                "schema_version": 2,
                "status": status,
                "updated_at": _utc_now(),
                "run_directory": str(run_dir),
                "scope": args.scope,
                "selected_condition_count": len(conditions),
                "completed_valid": n_ok,
                "failed": n_fail,
                "remaining": len(conditions) - n_ok - n_fail,
                "elapsed_seconds_this_invocation": time.time() - started,
                "git_at_invocation": git_identity(),
                "runtime_environment": identity["runtime_environment"],
                **extra,
            },
        )

    checkpoint("running")
    interruption_reason = "KeyboardInterrupt"
    previous_signal_handlers: dict[int, Any] = {}

    def interrupt_from_signal(signum: int, _frame: Any) -> None:
        nonlocal interruption_reason
        interruption_reason = signal.Signals(signum).name
        raise KeyboardInterrupt(interruption_reason)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt_from_signal)
    try:
        for index, condition in enumerate(conditions, start=1):
            key = condition_key(condition)
            if (
                _valid_result(
                    results.get(key),
                    closure=args.closure,
                )
                and not args.force
            ):
                continue
            if (
                implementation_fingerprints(args.candidate_id)
                != identity["implementation_sha256"]
            ):
                checkpoint(
                    "aborted_source_drift",
                    last_condition_key=key,
                    reason=(
                        "candidate implementation changed after run identity "
                        "was frozen; start a new run under the new identity"
                    ),
                )
                raise RuntimeError(
                    "candidate implementation changed during the campaign"
                )
            if (
                runtime_environment_identity()
                != identity["runtime_environment"]
            ):
                checkpoint(
                    "aborted_environment_drift",
                    last_condition_key=key,
                    reason=(
                        "dtype/device/Warp runtime changed after run "
                        "identity was frozen"
                    ),
                )
                raise RuntimeError(
                    "candidate runtime environment changed during campaign"
                )
            U, freq, twist, aoa = condition
            steps, wake = _resolved_step_grid(grid, condition)
            call = dict(base)
            call.update(
                U=U,
                aoa_deg=aoa,
                freq=freq,
                twist_amp_deg=twist / 2.0,
                closure=args.closure,
                nc=int(grid["nc"]),
                ns=int(grid["ns"]),
                n_cycle=int(grid["n_cycle"]),
                steps_per_cycle=steps,
                wake_rows=wake,
            )
            t0 = time.time()
            try:
                output = gpu_run_twist(**call)
                if (
                    implementation_fingerprints(args.candidate_id)
                    != identity["implementation_sha256"]
                ):
                    raise _CampaignDriftError(
                        "aborted_source_drift",
                        (
                            "candidate implementation changed while the "
                            "solver condition was executing; its output was "
                            "not accepted"
                        ),
                    )
                if (
                    runtime_environment_identity()
                    != identity["runtime_environment"]
                ):
                    raise _CampaignDriftError(
                        "aborted_environment_drift",
                        (
                            "dtype/device/Warp runtime changed while the "
                            "solver condition was executing; its output was "
                            "not accepted"
                        ),
                    )
                record = _candidate_record(
                    output,
                    wall_seconds=time.time() - t0,
                    steps_per_cycle=steps,
                    wake_rows=wake,
                    closure=args.closure,
                )
                results[key] = record
                print(
                    f"[{index}/{len(conditions)}] {key}: "
                    f"L={record['L']:+.3f} T={record['T']:+.3f} "
                    f"({record['wall_seconds']:.1f}s)",
                    flush=True,
                )
                checkpoint("running", last_condition_key=key)
            except _CampaignDriftError as exc:
                checkpoint(
                    exc.status,
                    last_condition_key=key,
                    reason=exc.reason,
                )
                raise RuntimeError(exc.reason) from exc
            except Exception as exc:
                results[key] = {
                    "fail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "wall_seconds": time.time() - t0,
                    "steps_per_cycle": steps,
                    "wake_rows": wake,
                }
                print(
                    f"[{index}/{len(conditions)}] {key}: "
                    f"FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )
                checkpoint(
                    "failed_fast",
                    last_condition_key=key,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"candidate campaign failed at {key}"
                ) from exc
    except KeyboardInterrupt:
        checkpoint("interrupted", reason=interruption_reason)
        raise
    finally:
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)

    n_valid = sum(
        _valid_result(
            results.get(condition_key(condition)),
            closure=args.closure,
        )
        for condition in conditions
    )
    final_status = _final_status(
        valid_count=n_valid,
        selected_count=len(conditions),
        max_conditions=args.max_conditions,
    )
    checkpoint(final_status)
    print(
        f"{final_status}: {n_valid}/{len(conditions)} -> {run_dir}",
        flush=True,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--closure", required=True)
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPE_CONDITIONS),
        default="smoke3",
    )
    create = parser.add_mutually_exclusive_group()
    create.add_argument("--timestamp", help="new run timestamp, YYYYMMDD_HHMMSS")
    create.add_argument("--resume", help="existing candidate run directory")
    parser.add_argument(
        "--seed-run",
        help=(
            "copy valid overlapping records from a model/grid-identical "
            "candidate run before executing this new nested scope"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use nc=4, ns=8, 2 cycles, 60 steps/cycle and wake rows",
    )
    parser.add_argument("--nc", type=int)
    parser.add_argument("--ns", type=int)
    parser.add_argument("--n-cycle", type=int)
    parser.add_argument("--steps-per-cycle", type=int)
    parser.add_argument("--wake-rows", type=int)
    parser.add_argument(
        "--model-arg",
        action="append",
        type=_parse_model_arg,
        default=[],
        metavar="KEY=JSON_VALUE",
        help="explicit non-runner gpu_run_twist argument; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-conditions",
        type=int,
        help="debug-only prefix limit; recorded as an incomplete run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute existing condition records inside this candidate run only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("nc", "ns", "n_cycle", "steps_per_cycle", "wake_rows"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_conditions is not None and args.max_conditions <= 0:
        parser.error("--max-conditions must be positive")
    if args.resume and args.seed_run:
        parser.error("--resume and --seed-run cannot be used together")
    try:
        run_dir = run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    status = _read_json(run_dir / "status.json").get("status")
    if status == "completed_with_failures":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
