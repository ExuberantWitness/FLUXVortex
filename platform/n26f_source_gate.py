#!/usr/bin/env python3
"""Deterministic scoring for the N2.6f Basilisk source-validation gate.

This module scores only the independent NACA0015 source case.  It does not
read RoboEagle/Fig17--19 targets and it never modifies a model force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SHA256 = (
    "8ff0282a4bfa67473a46f67aea768f27b6db91a44d24296c036f5c701c0acb86"
)
INSTRUMENTED_SOURCE_SHA256 = (
    "31a1bc1d63a534b3bee7433235375ffaf0af84bb163522bfe3c72a240cb4d349"
)
INSTRUMENTED_DIFF_SHA256 = (
    "735f5bc46e95a5dddcd860e61291d98666a71891d4bac05881b73c2a8c1a485a"
)
QCC_SHA256 = (
    "cb12457e2aed3bb76e7bfc852fb95249477cb56fd83a606029f23667593b8ac3"
)
REFERENCE_SHA256 = {
    "CD": "bb068e3880b2f142527739cd8a425df84dbabeb7f33f789b7217eeda1ff0b022",
    "CL": "77da2650d58ded6f049b0cc879e2d2e8069ab3029698f8f5a8759e5564c1ff6a",
}
FROZEN_ASSET_SHA256 = {
    "basilisk_tarball": (
        "fe6b4b5821517d792c58f0413ea6de4b5bd6b1d337578bb5ceb2fa6f07f8f193"
    ),
    "official_source": SOURCE_SHA256,
    "myembed.h": "e8aa667e61c097a66ae1ba4bc291da8505f6c48da2be6c00a7db26e6ec922a29",
    "mycentered.h": "96a17c3a508522962232e81c342a05d8a8071b7de3e5f4da4483ba9101f6dd60",
    "myembed-moving.h": (
        "df4c4e16fd8dfea6d61e4a6a34cbed37ec05647d64d8ebc134e262372844a006"
    ),
    "myperfs.h": "48f4bbd2567aa7c11575fce8b8b5ca202af542ac0f5c5739c66409d56fcc07ec",
    "myembed-tree-moving.h": (
        "8710a71ba3af8949415dfaeea3cf15bcc1dfb511c33fd9a4cfcdfb4da44f4e71"
    ),
    "myquadratic.h": (
        "c9680c2c664c19dff4236d75cc80d771cfa62d7a935bd975f1ee435f2ccb911b"
    ),
    "mytimestep.h": (
        "dc5896f61405846e09e317fcb53aab33a46358445deb5aac570b461e285c9ce8"
    ),
    "myviscosity-embed.h": (
        "7e2a3eddc11d9b4d9518d39d9d5e330ad5c48c96f3167ee8b1d35ca965313c19"
    ),
    "mypoisson.h": "3a8c1b982fad94a0b01fad6335acc5e113529fd52a6f5767d9ce321365062138",
    "CD_reference": REFERENCE_SHA256["CD"],
    "CL_reference": REFERENCE_SHA256["CL"],
    "fig18_reference": (
        "319d83ae36b27439e98fae69dbb33e67f88eaa4dffb2ae0d2b3e3db969541888"
    ),
    "fig19_reference": (
        "1b2acafd06fa35f00f65673d3d1ada9fcdbd66124a788a718de463a537c01363"
    ),
}

# Frozen before looking at the completed source response.
ANGLE_MIN_DEG = 0.5
ANGLE_MAX_DEG = 55.0
SOURCE_RNRMSE_LIMIT = 0.10
SOURCE_PEAK_ANGLE_LIMIT_DEG = 3.0
CAUCHY_RELATIVE_L2_LIMIT = 0.03
CAUCHY_PEAK_CHANGE_LIMIT = 0.03
GEOMETRY_CLOSURE_LIMIT = 1.0e-8
TRACTION_CLOSURE_LIMIT = 0.01
REFERENCE_SUPPORT_DEG = {
    "CD": (0.4730174926739976, 54.98539927034809),
    "CL": (0.020766621629590087, 54.500844765657654),
}


@dataclass(frozen=True)
class Curve:
    angle_deg: np.ndarray
    cd: np.ndarray
    cl: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"{path}: expected schema {expected_schema}")
    return payload


def _validate_record(record: Any, role: str) -> dict[str, Any]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"invalid file record for {role}")
    actual = _file_record(Path(record["path"]))
    for field in ("sha256", "size"):
        if record.get(field) != actual[field]:
            raise ValueError(f"{role} {field} does not match actual file")
    return actual


def _validate_asset_manifest(
    manifest_path: Path,
    source_path: Path,
    cd_reference: Path,
    cl_reference: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "fluxv.n26f.asset_manifest.v1")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("asset manifest files must be an object")
    required_roles = set(FROZEN_ASSET_SHA256) | {"qcc", "executable"}
    missing = sorted(required_roles - set(files))
    if missing:
        raise ValueError(f"asset manifest is missing roles: {missing}")

    validated: dict[str, Any] = {}
    for role in sorted(required_roles):
        validated[role] = _validate_record(files[role], role)
        expected = FROZEN_ASSET_SHA256.get(role)
        if expected is not None and validated[role]["sha256"] != expected:
            raise ValueError(f"frozen asset SHA drift for {role}")

    expected_paths = {
        "official_source": source_path,
        "CD_reference": cd_reference,
        "CL_reference": cl_reference,
    }
    for role, expected_path in expected_paths.items():
        if Path(validated[role]["path"]) != expected_path.resolve(strict=True):
            raise ValueError(f"{role} argument is not the manifested asset")

    compile_record = manifest.get("compile")
    if (
        not isinstance(compile_record, dict)
        or compile_record.get("return_code") != 0
        or not compile_record.get("command")
        or not compile_record.get("cwd")
    ):
        raise ValueError("asset manifest lacks a successful compile receipt")
    return {
        "path": str(manifest_path.resolve(strict=True)),
        "sha256": _sha256(manifest_path),
        "files": validated,
        "compile": compile_record,
        "pass": True,
    }


def _validate_run_receipt(
    receipt_path: Path,
    log_path: Path,
    asset_validation: dict[str, Any],
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "fluxv.n26f.run_receipt.v1")
    if receipt.get("return_code") != 0:
        raise ValueError("formal source process return code is not zero")
    if not receipt.get("command"):
        raise ValueError("run receipt has no command")
    cwd = Path(str(receipt.get("cwd", ""))).resolve(strict=True)
    if cwd != log_path.parent.resolve(strict=True):
        raise ValueError("run receipt cwd does not match log directory")
    start_epoch = float(receipt.get("start_epoch"))
    end_epoch = float(receipt.get("end_epoch"))
    if not (math.isfinite(start_epoch) and math.isfinite(end_epoch)):
        raise ValueError("run receipt epochs are non-finite")
    if end_epoch <= start_epoch:
        raise ValueError("run receipt end is not after start")

    log_record = _validate_record(receipt.get("log"), "run log")
    if Path(log_record["path"]) != log_path.resolve(strict=True):
        raise ValueError("run receipt log path does not match --log")
    executable_record = _validate_record(
        receipt.get("executable"), "run executable"
    )
    manifested_executable = asset_validation["files"]["executable"]
    if (
        executable_record["path"] != manifested_executable["path"]
        or executable_record["sha256"] != manifested_executable["sha256"]
    ):
        raise ValueError("run executable does not match asset manifest")

    output_records = receipt.get("outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("run receipt has no output inventory")
    validated_outputs: dict[str, Any] = {}
    for index, record in enumerate(output_records):
        actual = _validate_record(record, f"run output {index}")
        if actual["mtime"] < start_epoch - 1.0:
            raise ValueError(f"run output predates this run: {actual['path']}")
        validated_outputs[actual["path"]] = actual
    return {
        "path": str(receipt_path.resolve(strict=True)),
        "sha256": _sha256(receipt_path),
        "return_code": 0,
        "command": receipt["command"],
        "cwd": str(cwd),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "log": log_record,
        "executable": executable_record,
        "outputs": validated_outputs,
        "pass": True,
    }


def make_run_receipt(
    run_directory: Path,
    log_path: Path,
    executable_path: Path,
    command: str,
    start_epoch: float,
    end_epoch: float,
    return_code: int,
    output_path: Path,
) -> dict[str, Any]:
    run_directory = run_directory.resolve(strict=True)
    log_path = log_path.resolve(strict=True)
    executable_path = executable_path.resolve(strict=True)
    if log_path.parent != run_directory:
        raise ValueError("receipt log is not directly inside the run directory")
    if output_path.resolve() in {log_path, executable_path}:
        raise ValueError("receipt output overlaps immutable evidence")
    if not (
        math.isfinite(start_epoch)
        and math.isfinite(end_epoch)
        and end_epoch > start_epoch
    ):
        raise ValueError("invalid receipt epochs")
    if not command.strip():
        raise ValueError("empty run command")

    outputs: list[dict[str, Any]] = []
    for path in sorted(run_directory.iterdir()):
        if not path.is_file() or path.resolve() == executable_path:
            continue
        stat = path.stat()
        if stat.st_mtime < start_epoch - 1.0 or stat.st_mtime > end_epoch + 1.0:
            continue
        outputs.append(_file_record(path))
    if not outputs:
        raise ValueError("no fresh run outputs found")
    receipt = {
        "schema": "fluxv.n26f.run_receipt.v1",
        "command": command,
        "cwd": str(run_directory),
        "return_code": int(return_code),
        "start_epoch": float(start_epoch),
        "end_epoch": float(end_epoch),
        "log": _file_record(log_path),
        "executable": _file_record(executable_path),
        "outputs": outputs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _read_log(path: Path) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    angle_tokens: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 16:
                raise ValueError(
                    f"{path}:{line_number}: expected 16 columns, got {len(fields)}"
                )
            try:
                row = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric log row"
                ) from exc
            rows.append(row)
            angle_tokens.append(fields[13])
    if not rows:
        raise ValueError(f"no 16-column numeric rows in {path}")
    return np.asarray(rows, dtype=np.float64), angle_tokens


def _read_numeric_table(path: Path, columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != columns:
                raise ValueError(
                    f"{path}:{line_number}: expected {columns} columns, "
                    f"got {len(fields)}"
                )
            try:
                row = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric row"
                ) from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"empty numeric table: {path}")
    data = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite value in {path}")
    return data


def load_instrumented_steps(
    step_path: Path, dt_cap: float
) -> tuple[np.ndarray, dict[str, Any]]:
    data = _read_numeric_table(step_path, 11)
    if int(data[0, 0]) != 0 or not np.array_equal(
        np.diff(data[:, 0]), np.ones(len(data) - 1)
    ):
        raise ValueError("instrumented step index is not contiguous from zero")
    if np.any(np.diff(data[:, 1]) <= 0.0) or np.any(data[:, 2] <= 0.0):
        raise ValueError("instrumented step time/dt is invalid")
    expected_dphys = np.minimum(data[:, 8], data[:, 9])
    if not np.array_equal(data[:, 10], expected_dphys):
        raise ValueError("d_phys is not the exact min(d_CFL,d_embed)")
    # The sidecar time is the completed-step end time, while d_phys was
    # sampled at that step's start.  The frozen replacement axis used the
    # historical end-time mask below.  Report the strict start-time mask and
    # the exact cross-boundary count so this known protocol deviation cannot
    # be hidden or silently redefined after observing CL/CD.
    post = data[:, 1] >= 0.2
    strict_post = data[:, 1] - data[:, 2] >= 0.2
    if not np.any(post):
        raise ValueError("instrumented run does not reach pitch start")
    active = dt_cap <= data[post, 10]
    diagnostics = {
        "rows": int(len(data)),
        "dt_cap": float(dt_cap),
        "post_start_rows": int(post.sum()),
        "strict_start_time_rows": int(strict_post.sum()),
        "cross_boundary_rows": int(np.count_nonzero(post & ~strict_post)),
        "selection_semantics": (
            "frozen end-time mask t_end>=0.2; d_phys belongs to step start; "
            "strict start-time diagnostics reported separately"
        ),
        "post_start_d_phys": {
            "min": float(data[post, 10].min()),
            "median": float(np.median(data[post, 10])),
            "max": float(data[post, 10].max()),
        },
        "post_start_actual_dt": {
            "min": float(data[post, 2].min()),
            "median": float(np.median(data[post, 2])),
            "max": float(data[post, 2].max()),
        },
        "cap_active_fraction": float(np.mean(active)),
    }
    return data, diagnostics


def _derive_time_axis_payload(step_path: Path) -> dict[str, Any]:
    """Purely derive the frozen replacement axis from a raw step sidecar."""

    # This operation deliberately consumes only the numerical-control sidecar
    # before any CL/CD/reference score is computed.
    data = _read_numeric_table(step_path, 11)
    expected_dphys = np.minimum(data[:, 8], data[:, 9])
    if not np.array_equal(data[:, 10], expected_dphys):
        raise ValueError("d_phys is not the exact min(d_CFL,d_embed)")
    post = data[:, 1] >= 0.2
    if not np.any(post):
        raise ValueError("step sidecar does not reach pitch start")
    d50 = float(np.median(data[post, 10]))
    if not math.isfinite(d50) or d50 <= 0.0:
        raise ValueError("derived d50 is not positive and finite")
    return {
        "schema": "fluxv.n26f.time_axis_freeze.v1",
        "scope": "SOURCE_NUMERICAL_CONTROL_ONLY",
        "claim_promotion_authorized": False,
        "source_step": _file_record(step_path),
        "selection": "median post-start d_phys; no CL/CD/reference/target read",
        "post_start_rows": int(post.sum()),
        "d50": d50,
        "frozen_dt_caps": [d50, d50 / 2.0, d50 / 4.0],
        "final_pair": [d50 / 2.0, d50 / 4.0],
        "cap_active_fraction_min": 0.5,
        "median_actual_dt_ratio_min": 1.5,
    }


def derive_time_axis(step_path: Path, output_path: Path) -> dict[str, Any]:
    result = _derive_time_axis_payload(step_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _validate_time_axis_freeze(
    freeze_path: Path, source_step_path: Path
) -> dict[str, Any]:
    """Re-derive every decision-bearing field from immutable raw evidence."""

    freeze = _load_json(freeze_path, "fluxv.n26f.time_axis_freeze.v1")
    expected = _derive_time_axis_payload(source_step_path)
    recorded_step = _validate_record(
        freeze.get("source_step"), "time-axis source step"
    )
    current_step = _file_record(source_step_path)
    if (
        recorded_step["path"] != current_step["path"]
        or recorded_step["sha256"] != current_step["sha256"]
    ):
        raise ValueError("time-axis freeze source step is not the supplied raw step")

    decision_fields = (
        "scope",
        "claim_promotion_authorized",
        "selection",
        "post_start_rows",
        "d50",
        "frozen_dt_caps",
        "final_pair",
        "cap_active_fraction_min",
        "median_actual_dt_ratio_min",
    )
    for field in decision_fields:
        if freeze.get(field) != expected[field]:
            raise ValueError(
                f"time-axis freeze {field} was not re-derived from raw d_phys"
            )
    return {
        "path": str(freeze_path.resolve(strict=True)),
        "sha256": _sha256(freeze_path),
        "source_step": current_step,
        "rederived": expected,
        "pass": True,
    }


def _traction_checkpoint(
    step_data: np.ndarray,
    traction_path: Path,
    metadata_path: Path,
    cp_path: Path,
) -> dict[str, Any]:
    traction = _read_numeric_table(traction_path, 16)
    metadata = _read_numeric_table(metadata_path, 4)
    if len(metadata) != 1:
        raise ValueError(f"expected one metadata row: {metadata_path}")
    cp = _read_numeric_table(cp_path, 4)
    if len(cp) != len(traction):
        raise ValueError("traction and official cpout fragment counts differ")

    iteration = int(metadata[0, 1])
    pid_value = int(metadata[0, 0])
    if metadata[0, 0] != pid_value or pid_value != 0:
        raise ValueError("S2 source gate is frozen to one serial PID 0")
    if not np.array_equal(traction[:, 0], np.zeros(len(traction))):
        raise ValueError("traction table contains a nonzero or non-integer PID")
    if metadata[0, 1] != iteration:
        raise ValueError("non-integer traction metadata iteration")
    matches = np.flatnonzero(step_data[:, 0] == iteration)
    if len(matches) != 1:
        raise ValueError("traction metadata does not select one step row")
    step = step_data[int(matches[0])]
    if step[1] != metadata[0, 2]:
        raise ValueError("traction metadata time does not exactly match step sidecar")

    normal = traction[:, 3:5]
    ds = traction[:, 5]
    if np.any(ds <= 0.0):
        raise ValueError("non-positive embedded fragment length")
    perimeter = float(ds.sum())
    geometry_vector = np.sum(normal * ds[:, None], axis=0)
    geometry_closure = float(np.linalg.norm(geometry_vector) / perimeter)

    pressure = traction[:, 6]
    mua = traction[:, 7]
    dudn = traction[:, 8:10]
    d_fp_stored = traction[:, 10:12]
    d_fmu_stored = traction[:, 12:14]
    tau_stored = traction[:, 14]
    sigma_mu_n_stored = traction[:, 15]

    # Recompute every exported traction channel from primitive source fields.
    # This is deliberately independent of the C-side dFp/dFmu/tau columns so
    # that a column-order or formula drift cannot self-cancel in the ledger.
    d_fp = ds[:, None] * pressure[:, None] * normal
    d_fmu = np.empty_like(d_fmu_stored)
    d_fmu[:, 0] = -ds * mua * (
        dudn[:, 0] * (normal[:, 0] ** 2 + 1.0)
        + dudn[:, 1] * normal[:, 0] * normal[:, 1]
    )
    d_fmu[:, 1] = -ds * mua * (
        dudn[:, 1] * (normal[:, 1] ** 2 + 1.0)
        + dudn[:, 0] * normal[:, 1] * normal[:, 0]
    )
    tau_recomputed = (
        -normal[:, 1] * d_fmu[:, 0] + normal[:, 0] * d_fmu[:, 1]
    ) / ds
    sigma_mu_n_recomputed = np.sum(normal * d_fmu, axis=1) / ds
    primitive_recompute = {
        "dFp_max_abs_residual": float(np.max(np.abs(d_fp - d_fp_stored))),
        "dFmu_max_abs_residual": float(
            np.max(np.abs(d_fmu - d_fmu_stored))
        ),
        "tau_w_max_abs_residual": float(
            np.max(np.abs(tau_recomputed - tau_stored))
        ),
        "sigma_mu_n_max_abs_residual": float(
            np.max(np.abs(sigma_mu_n_recomputed - sigma_mu_n_stored))
        ),
        "exact": bool(
            np.array_equal(d_fp, d_fp_stored)
            and np.array_equal(d_fmu, d_fmu_stored)
            and np.array_equal(tau_recomputed, tau_stored)
            and np.array_equal(sigma_mu_n_recomputed, sigma_mu_n_stored)
        ),
    }

    fp_sum = np.sum(d_fp, axis=0)
    fmu_sum = np.sum(d_fmu, axis=0)
    fp_embed = step[4:6]
    fmu_embed = step[6:8]
    total_embed = fp_embed + fmu_embed
    total_fragment = fp_sum + fmu_sum
    pressure_denominator = max(float(np.linalg.norm(fp_embed)), 1.0e-14)
    viscous_denominator = max(float(np.linalg.norm(fmu_embed)), 1.0e-14)
    total_denominator = max(float(np.linalg.norm(total_embed)), 1.0e-14)
    pressure_force_closure = float(
        np.linalg.norm(fp_sum - fp_embed) / pressure_denominator
    )
    viscous_force_closure = float(
        np.linalg.norm(fmu_sum - fmu_embed) / viscous_denominator
    )
    force_closure = float(
        np.linalg.norm(total_fragment - total_embed) / total_denominator
    )

    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    reconstructed = ds[:, None] * (
        traction[:, 14, None] * tangent
        + traction[:, 15, None] * normal
    )
    viscous_scale = max(float(np.linalg.norm(d_fmu)), 1.0e-14)
    decomposition_residual = float(
        np.linalg.norm(reconstructed - d_fmu) / viscous_scale
    )
    unit_normal_error = float(np.max(np.abs(np.linalg.norm(normal, axis=1) - 1.0)))
    return {
        "metadata": {
            "pid": pid_value,
            "iteration": iteration,
            "time": float(metadata[0, 2]),
            "angle_deg": float(metadata[0, 3]),
        },
        "files": {
            "traction": _file_record(traction_path),
            "metadata": _file_record(metadata_path),
            "cp": _file_record(cp_path),
        },
        "fragment_count": int(len(traction)),
        "cp_fragment_count": int(len(cp)),
        "perimeter": perimeter,
        "unit_normal_max_abs_error": unit_normal_error,
        "geometry_closure": geometry_closure,
        "geometry_closure_limit": GEOMETRY_CLOSURE_LIMIT,
        "force": {
            "fragment_pressure": fp_sum.tolist(),
            "embed_pressure": fp_embed.tolist(),
            "pressure_abs_residual": float(np.linalg.norm(fp_sum - fp_embed)),
            "pressure_relative_closure": pressure_force_closure,
            "fragment_full_viscous": fmu_sum.tolist(),
            "embed_full_viscous": fmu_embed.tolist(),
            "full_viscous_abs_residual": float(
                np.linalg.norm(fmu_sum - fmu_embed)
            ),
            "full_viscous_relative_closure": viscous_force_closure,
            "fragment_total": total_fragment.tolist(),
            "embed_total": total_embed.tolist(),
            "relative_total_closure": force_closure,
            "relative_total_closure_limit": TRACTION_CLOSURE_LIMIT,
        },
        "primitive_recompute": primitive_recompute,
        "viscous_decomposition_relative_residual": decomposition_residual,
        "pressure_l1": float(np.sum(np.linalg.norm(d_fp, axis=1))),
        "full_viscous_l1": float(np.sum(np.linalg.norm(d_fmu, axis=1))),
        "tangential_viscous_l1": float(np.sum(np.abs(traction[:, 14]) * ds)),
        "normal_viscous_l1": float(np.sum(np.abs(traction[:, 15]) * ds)),
        "pass": bool(
            geometry_closure <= GEOMETRY_CLOSURE_LIMIT
            and primitive_recompute["exact"]
            and pressure_force_closure <= TRACTION_CLOSURE_LIMIT
            and viscous_force_closure <= TRACTION_CLOSURE_LIMIT
            and force_closure <= TRACTION_CLOSURE_LIMIT
            and decomposition_residual <= 1.0e-12
            and unit_normal_error <= 1.0e-12
        ),
    }


def score_instrumented_run(
    run_directory: Path,
    dt_cap: float,
    lmax: int,
    output_path: Path | None,
) -> dict[str, Any]:
    run_directory = run_directory.resolve(strict=True)
    step_path = run_directory / "n26f-step-pid-0"
    step_data, time_diagnostics = load_instrumented_steps(step_path, dt_cap)
    checkpoints: dict[str, Any] = {}
    for angle in (44, 54):
        checkpoints[str(angle)] = _traction_checkpoint(
            step_data,
            run_directory / f"traction-angle-{angle}-pid-0",
            run_directory / f"traction-meta-angle-{angle}-pid-0",
            run_directory / f"cp-angle-{angle}-pid-0",
        )
    result = {
        "schema": "fluxv.n26f.instrumented_run.v1",
        "scope": "N2.6f1.S2_RUN_COMPONENT",
        "claim_promotion_authorized": False,
        "run_directory": str(run_directory),
        "configuration": {"lmax": int(lmax), "dt_cap": float(dt_cap)},
        "step_file": _file_record(step_path),
        "time_diagnostics": time_diagnostics,
        "checkpoints": checkpoints,
        "traction_pass": bool(
            all(checkpoint["pass"] for checkpoint in checkpoints.values())
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


S2_RUN_LAYOUT = {
    "space_l13": ("l13_dt0p01", "n26f1_l13_dt0p01", 13, 0.01),
    "space_l14": ("l14_dt0p01", "n26f1_l14_dt0p01", 14, 0.01),
    # The first l15 process was terminated externally with SIGTERM before
    # either checkpoint.  Its immutable partial directory is retained; the
    # preregistered configuration is rerun from zero in this fresh directory.
    "space_l15": ("l15_dt0p01", "n26f1_l15_dt0p01", 15, 0.01),
    "time_orig_0p02": ("l14_dt0p02", "n26f1_l14_dt0p02", 14, 0.02),
    "time_orig_0p01": ("l14_dt0p01", "n26f1_l14_dt0p01", 14, 0.01),
    "time_orig_0p005": ("l14_dt0p005", "n26f1_l14_dt0p005", 14, 0.005),
    "time_new_d50": ("l14_d50", "n26f1_l14_d50", 14, None),
    "time_new_half": ("l14_d50_half", "n26f1_l14_d50_half", 14, None),
    "time_new_quarter": (
        "l14_d50_quarter",
        "n26f1_l14_d50_quarter",
        14,
        None,
    ),
}


def _s2_dynamic_caps(time_axis_freeze: Path) -> tuple[dict[str, float], dict[str, Any]]:
    freeze = _load_json(time_axis_freeze, "fluxv.n26f.time_axis_freeze.v1")
    new_caps = freeze["frozen_dt_caps"]
    if not (
        isinstance(new_caps, list)
        and len(new_caps) == 3
        and all(math.isfinite(float(value)) and float(value) > 0 for value in new_caps)
    ):
        raise ValueError("invalid frozen replacement time axis")
    return (
        {
            "time_new_d50": float(new_caps[0]),
            "time_new_half": float(new_caps[1]),
            "time_new_quarter": float(new_caps[2]),
        },
        freeze,
    )


def _s2_executable_configs(time_axis_freeze: Path) -> dict[str, dict[str, Any]]:
    dynamic_caps, _ = _s2_dynamic_caps(time_axis_freeze)
    configs: dict[str, dict[str, Any]] = {}
    for label, (_, executable_name, lmax, fixed_cap) in S2_RUN_LAYOUT.items():
        config = {
            "lmax": int(lmax),
            "dt_cap": float(dynamic_caps.get(label, fixed_cap)),
        }
        previous = configs.setdefault(executable_name, config)
        if previous != config:
            raise ValueError(f"conflicting S2 build configuration for {executable_name}")
    return configs


def make_s2_repro_build(
    qcc: Path,
    instrumented_source: Path,
    parent_source: Path,
    source_diff: Path,
    time_axis_freeze: Path,
    output_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Rebuild every unique S2 executable and freeze the source-to-binary chain."""

    qcc = qcc.resolve(strict=True)
    instrumented_source = instrumented_source.resolve(strict=True)
    parent_source = parent_source.resolve(strict=True)
    source_diff = source_diff.resolve(strict=True)
    time_axis_freeze = time_axis_freeze.resolve(strict=True)
    output_root = output_root.resolve()
    output_path = output_path.resolve()
    if qcc.is_dir() or not os.access(qcc, os.X_OK):
        raise ValueError("qcc is not an executable file")
    if _sha256(qcc) != QCC_SHA256:
        raise ValueError("qcc identity drift")
    if _sha256(parent_source) != SOURCE_SHA256:
        raise ValueError("S2 parent source identity drift")
    if _sha256(instrumented_source) != INSTRUMENTED_SOURCE_SHA256:
        raise ValueError("instrumented source identity drift")
    if _sha256(source_diff) != INSTRUMENTED_DIFF_SHA256:
        raise ValueError("instrumented source diff identity drift")
    if output_root.exists():
        raise ValueError("repro-build output root already exists")
    if output_path.parent != output_root:
        raise ValueError("repro-build manifest must be directly inside output root")
    output_root.mkdir(parents=True, exist_ok=False)

    patch_executable = Path("/usr/bin/patch").resolve(strict=True)
    reconstructed = output_root / "reconstructed-instrumented.c"
    patch_command = [
        str(patch_executable),
        "--batch",
        "--silent",
        f"--output={reconstructed}",
        str(parent_source),
        str(source_diff),
    ]
    patch_run = subprocess.run(
        patch_command,
        text=True,
        capture_output=True,
        check=False,
    )
    if patch_run.returncode != 0:
        raise ValueError(
            "parent-to-instrumented patch reconstruction failed: "
            + patch_run.stderr.strip()
        )
    if _sha256(reconstructed) != INSTRUMENTED_SOURCE_SHA256:
        raise ValueError("source diff does not reconstruct instrumented source")

    header_records: dict[str, Any] = {}
    for name in (
        "myembed.h",
        "mycentered.h",
        "myembed-moving.h",
        "myperfs.h",
        "myembed-tree-moving.h",
        "myquadratic.h",
        "mytimestep.h",
        "myviscosity-embed.h",
        "mypoisson.h",
    ):
        # The source case includes "../my*.h".  qcc also emits guarded
        # preprocessing copies beside the C file; those generated copies are
        # not the authoritative build inputs and must not be manifested.
        header = instrumented_source.parent.parent / name
        record = _file_record(header)
        if record["sha256"] != FROZEN_ASSET_SHA256[name]:
            raise ValueError(f"frozen instrumented build header drift: {name}")
        header_records[name] = record

    env = os.environ.copy()
    env["BASILISK"] = str(qcc.parent)
    builds: dict[str, Any] = {}
    for executable_name, config in sorted(
        _s2_executable_configs(time_axis_freeze).items()
    ):
        output_executable = output_root / executable_name
        lmax_token = str(config["lmax"])
        dt_cap_token = format(config["dt_cap"], ".17g")
        command = [
            str(qcc),
            "-O2",
            "-Wall",
            "-autolink",
            f"-DN26F_LMAX={lmax_token}",
            f"-DN26F_DT_CAP={dt_cap_token}",
            instrumented_source.name,
            "-o",
            str(output_executable),
            "-lfb_tiny",
            "-lm",
        ]
        completed = subprocess.run(
            command,
            cwd=instrumented_source.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"repro build failed for {executable_name}: "
                + completed.stderr.strip()
            )
        builds[executable_name] = {
            "configuration": config,
            "macro_tokens": {
                "N26F_LMAX": lmax_token,
                "N26F_DT_CAP": dt_cap_token,
            },
            "command": command,
            "cwd": str(instrumented_source.parent),
            "return_code": 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "executable": _file_record(output_executable),
        }

    manifest = {
        "schema": "fluxv.n26f.s2_repro_build.v1",
        "scope": "SOURCE_TO_BINARY_IDENTITY_ONLY",
        "claim_promotion_authorized": False,
        "qcc": _file_record(qcc),
        "patch_executable": _file_record(patch_executable),
        "parent_source": _file_record(parent_source),
        "source_diff": _file_record(source_diff),
        "instrumented_source": _file_record(instrumented_source),
        "reconstructed_source": _file_record(reconstructed),
        "time_axis_freeze": _file_record(time_axis_freeze),
        "headers": header_records,
        "patch_command": patch_command,
        "patch_stdout": patch_run.stdout,
        "patch_stderr": patch_run.stderr,
        "builds": builds,
    }
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_s2_repro_build(
    manifest_path: Path,
    parent_source: Path,
    instrumented_source: Path,
    source_diff: Path,
    time_axis_freeze: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "fluxv.n26f.s2_repro_build.v1")
    expected_records = {
        "parent_source": (parent_source, SOURCE_SHA256),
        "instrumented_source": (
            instrumented_source,
            INSTRUMENTED_SOURCE_SHA256,
        ),
        "source_diff": (source_diff, INSTRUMENTED_DIFF_SHA256),
        "time_axis_freeze": (time_axis_freeze, None),
    }
    validated: dict[str, Any] = {}
    for role, (expected_path, expected_sha) in expected_records.items():
        record = _validate_record(manifest.get(role), f"S2 build {role}")
        if Path(record["path"]) != expected_path.resolve(strict=True):
            raise ValueError(f"S2 build {role} path drift")
        if expected_sha is not None and record["sha256"] != expected_sha:
            raise ValueError(f"S2 build {role} SHA drift")
        validated[role] = record
    qcc_record = _validate_record(manifest.get("qcc"), "S2 build qcc")
    if qcc_record["sha256"] != QCC_SHA256:
        raise ValueError("S2 build qcc SHA drift")
    _validate_record(manifest.get("patch_executable"), "S2 build patch executable")
    reconstructed = _validate_record(
        manifest.get("reconstructed_source"), "S2 reconstructed source"
    )
    if reconstructed["sha256"] != INSTRUMENTED_SOURCE_SHA256:
        raise ValueError("S2 reconstructed source SHA drift")

    headers = manifest.get("headers")
    if not isinstance(headers, dict):
        raise ValueError("S2 build header records are missing")
    for name, expected_sha in FROZEN_ASSET_SHA256.items():
        if name not in {
            "myembed.h",
            "mycentered.h",
            "myembed-moving.h",
            "myperfs.h",
            "myembed-tree-moving.h",
            "myquadratic.h",
            "mytimestep.h",
            "myviscosity-embed.h",
            "mypoisson.h",
        }:
            continue
        record = _validate_record(headers.get(name), f"S2 build header {name}")
        if record["sha256"] != expected_sha:
            raise ValueError(f"S2 build header SHA drift: {name}")

    expected_configs = _s2_executable_configs(time_axis_freeze)
    builds = manifest.get("builds")
    if not isinstance(builds, dict) or set(builds) != set(expected_configs):
        raise ValueError("S2 reproducible build inventory drift")
    build_records: dict[str, Any] = {}
    for executable_name, expected_config in expected_configs.items():
        build = builds[executable_name]
        if not isinstance(build, dict) or build.get("return_code") != 0:
            raise ValueError(f"S2 reproducible build failed: {executable_name}")
        if build.get("configuration") != expected_config:
            raise ValueError(f"S2 reproducible build config drift: {executable_name}")
        expected_macros = {
            "N26F_LMAX": str(expected_config["lmax"]),
            "N26F_DT_CAP": format(expected_config["dt_cap"], ".17g"),
        }
        if build.get("macro_tokens") != expected_macros:
            raise ValueError(f"S2 reproducible build macro drift: {executable_name}")
        executable_record = _validate_record(
            build.get("executable"), f"S2 reproducible binary {executable_name}"
        )
        expected_command = [
            qcc_record["path"],
            "-O2",
            "-Wall",
            "-autolink",
            f"-DN26F_LMAX={expected_macros['N26F_LMAX']}",
            f"-DN26F_DT_CAP={expected_macros['N26F_DT_CAP']}",
            instrumented_source.name,
            "-o",
            executable_record["path"],
            "-lfb_tiny",
            "-lm",
        ]
        if build.get("command") != expected_command:
            raise ValueError(f"S2 reproducible build command drift: {executable_name}")
        if Path(str(build.get("cwd", ""))).resolve(strict=True) != (
            instrumented_source.parent.resolve(strict=True)
        ):
            raise ValueError(f"S2 reproducible build cwd drift: {executable_name}")
        build_records[executable_name] = executable_record
    return {
        "path": str(manifest_path.resolve(strict=True)),
        "sha256": _sha256(manifest_path),
        "builds": build_records,
        "pass": True,
    }


def _validated_s1_authority_path(record: Any, role: str) -> Path:
    """Resolve a path/SHA authority record embedded in the frozen S1 score."""

    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
    ):
        raise ValueError(f"S1 score has no valid {role} path/SHA authority")
    path = Path(record["path"]).resolve(strict=True)
    if str(path) != record["path"]:
        raise ValueError(f"S1 score {role} path is not canonical")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"S1 score {role} SHA does not match the actual file")
    return path


def _validate_s1_official_log(
    s1_score_path: Path, official_log: Path
) -> dict[str, Any]:
    """Recompute the S1 gate and bind its actual receipt/log to neutrality."""

    score = _load_json(s1_score_path, "fluxv.n26f.s1_formal_response.v1")
    if not (
        score.get("formal_response_pass")
        and score.get("identity_pass")
        and score.get("required_outputs_pass")
    ):
        raise ValueError("S1 formal response evidence is not a complete PASS")
    score_log = Path(str(score.get("log", ""))).resolve(strict=True)
    receipt = score.get("run_receipt_validation")
    if not isinstance(receipt, dict) or not receipt.get("pass"):
        raise ValueError("S1 score has no successful run-receipt validation")

    # Do not trust only the receipt fields copied into the derived score.  Read
    # the actual receipt whose path/SHA the score froze, then bind both log
    # records to the official neutrality input.
    receipt_path = _validated_s1_authority_path(receipt, "run receipt")
    actual_receipt = _load_json(receipt_path, "fluxv.n26f.run_receipt.v1")
    if actual_receipt.get("return_code") != 0:
        raise ValueError("actual S1 run receipt does not record successful completion")
    score_receipt_log = _validate_record(receipt.get("log"), "S1 score receipt log")
    actual_receipt_log = _validate_record(
        actual_receipt.get("log"), "actual S1 receipt log"
    )
    official_record = _file_record(official_log)
    bound_paths = {
        score_log,
        Path(score_receipt_log["path"]),
        Path(actual_receipt_log["path"]),
        Path(official_record["path"]),
    }
    bound_shas = {
        _sha256(score_log),
        score_receipt_log["sha256"],
        actual_receipt_log["sha256"],
        official_record["sha256"],
    }
    if len(bound_paths) != 1 or len(bound_shas) != 1:
        raise ValueError(
            "official neutrality log is not the log bound by the "
            "S1 score and actual receipt"
        )

    # The frozen score contains every raw path needed by score_source().  A
    # read-only recomputation is therefore safer than trusting copied PASS
    # booleans or metrics.  The comparison intentionally excludes only the
    # derived comparison plot, which is not a numerical gate.
    asset_validation = score.get("asset_manifest_validation")
    if not isinstance(asset_validation, dict) or not isinstance(
        asset_validation.get("files"), dict
    ):
        raise ValueError("S1 score has no asset records for raw recomputation")
    source_path = _validated_s1_authority_path(
        asset_validation["files"].get("official_source"), "source"
    )
    cd_reference = _validated_s1_authority_path(
        asset_validation["files"].get("CD_reference"), "CD reference"
    )
    cl_reference = _validated_s1_authority_path(
        asset_validation["files"].get("CL_reference"), "CL reference"
    )
    asset_manifest_path = _validated_s1_authority_path(
        asset_validation, "asset manifest"
    )
    recomputed = score_source(
        Path(official_record["path"]),
        source_path,
        cd_reference,
        cl_reference,
        asset_manifest_path,
        receipt_path,
        None,
        None,
    )
    decision_fields = (
        "scope",
        "claim_promotion_authorized",
        "log",
        "asset_manifest_validation",
        "run_receipt_validation",
        "identity_pass",
        "diagnostics",
        "required_outputs",
        "required_outputs_pass",
        "metric_definition",
        "metrics",
        "formal_response_pass",
    )
    drifted = [
        field for field in decision_fields if score.get(field) != recomputed.get(field)
    ]
    # The historical score retained caller-relative identity paths.  Their path
    # spelling is not a gate: the asset manifest above supplies the canonical
    # files.  The decision-bearing identity hashes must nevertheless reproduce
    # exactly.
    identity_fields = ("sha256", "expected_sha256")
    stored_identities = score.get("identity")
    recomputed_identities = recomputed.get("identity")
    if not isinstance(stored_identities, dict) or not isinstance(
        recomputed_identities, dict
    ):
        drifted.append("identity")
    else:
        for role in ("source", "CD_reference", "CL_reference"):
            stored_identity = stored_identities.get(role)
            recomputed_identity = recomputed_identities.get(role)
            if (
                not isinstance(stored_identity, dict)
                or not isinstance(recomputed_identity, dict)
                or any(
                    stored_identity.get(field) != recomputed_identity.get(field)
                    for field in identity_fields
                )
            ):
                drifted.append(f"identity.{role}")
    if drifted:
        raise ValueError(
            "S1 stored score differs from raw-evidence recomputation: "
            + ", ".join(drifted)
        )
    return {
        "s1_score": _file_record(s1_score_path),
        "actual_receipt": _file_record(receipt_path),
        "official_log": official_record,
        "score_receipt_log": score_receipt_log,
        "actual_receipt_log": actual_receipt_log,
        "raw_gate_recomputation": {
            "decision_fields_exact": [
                *decision_fields,
                "identity.*.sha256",
                "identity.*.expected_sha256",
            ],
            "identity_pass": recomputed["identity_pass"],
            "required_outputs_pass": recomputed["required_outputs_pass"],
            "diagnostics_complete": recomputed["diagnostics"]["complete"],
            "metrics": recomputed["metrics"],
            "formal_response_pass": recomputed["formal_response_pass"],
            "not_recomputed": {
                "visual_topology": (
                    "the S1 score freezes required image identity/existence but "
                    "contains no preregistered machine-readable topology classifier"
                )
            },
            "pass": True,
        },
        "pass": True,
    }


def make_s2_manifest(
    root: Path,
    official_log: Path,
    s1_score: Path,
    instrumented_source: Path,
    parent_source: Path,
    source_diff: Path,
    time_axis_freeze: Path,
    repro_build_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output_path = output_path.resolve()
    immutable_inputs = {
        path.resolve(strict=True)
        for path in (
            official_log,
            s1_score,
            instrumented_source,
            parent_source,
            source_diff,
            time_axis_freeze,
            repro_build_manifest,
        )
    }
    if output_path in immutable_inputs:
        raise ValueError("S2 manifest output overlaps immutable input evidence")
    s1_binding = _validate_s1_official_log(s1_score, official_log)
    time_axis_validation = _validate_time_axis_freeze(
        time_axis_freeze, root / S2_RUN_LAYOUT["space_l14"][0] / "n26f-step-pid-0"
    )
    dynamic_caps, _ = _s2_dynamic_caps(time_axis_freeze)
    build_validation = _validate_s2_repro_build(
        repro_build_manifest,
        parent_source,
        instrumented_source,
        source_diff,
        time_axis_freeze,
    )

    runs: dict[str, Any] = {}
    for label, (directory_name, executable_name, lmax, fixed_cap) in S2_RUN_LAYOUT.items():
        run_directory = root / directory_name
        dt_cap = dynamic_caps.get(label, fixed_cap)
        receipt_path = run_directory / "run_receipt.json"
        score_path = run_directory / "instrumented_score.json"
        receipt = _load_json(receipt_path, "fluxv.n26f.run_receipt.v1")
        if receipt.get("return_code") != 0:
            raise ValueError(f"{label} process did not return zero")
        score = _load_json(score_path, "fluxv.n26f.instrumented_run.v1")
        if score.get("configuration") != {"lmax": lmax, "dt_cap": dt_cap}:
            raise ValueError(f"{label} score configuration drift")
        executable_record = _file_record(run_directory / executable_name)
        reproducible_record = build_validation["builds"][executable_name]
        if executable_record["sha256"] != reproducible_record["sha256"]:
            raise ValueError(f"{label} binary is not reproducible from frozen source")
        runs[label] = {
            "run_directory": str(run_directory),
            "lmax": lmax,
            "dt_cap": dt_cap,
            "executable": executable_record,
            "log": _file_record(run_directory / "log"),
            "step": _file_record(run_directory / "n26f-step-pid-0"),
            "score": _file_record(score_path),
            "receipt": _file_record(receipt_path),
        }

    manifest = {
        "schema": "fluxv.n26f.s2_family_manifest.v1",
        "scope": "SOURCE_VALIDATION_ONLY",
        "official_formal_log": _file_record(official_log),
        "s1_formal_score": _file_record(s1_score),
        "instrumented_source": _file_record(instrumented_source),
        "parent_source": _file_record(parent_source),
        "source_diff": _file_record(source_diff),
        "time_axis_freeze": _file_record(time_axis_freeze),
        "repro_build_manifest": _file_record(repro_build_manifest),
        "s1_official_log_binding": s1_binding,
        "time_axis_rederivation": time_axis_validation,
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validated_manifest_path(record: Any, role: str) -> Path:
    actual = _validate_record(record, role)
    return Path(actual["path"])


def _validate_s2_run_receipt(
    receipt_path: Path,
    run_directory: Path,
    executable_path: Path,
    required_outputs: list[Path],
    expected_command: str,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "fluxv.n26f.run_receipt.v1")
    if receipt.get("return_code") != 0 or not receipt.get("command"):
        raise ValueError("S2 run receipt does not record successful completion")
    command = str(receipt["command"])
    if command != expected_command:
        raise ValueError("S2 run receipt command differs from the frozen command")
    cwd = Path(str(receipt.get("cwd", ""))).resolve(strict=True)
    if cwd != run_directory.resolve(strict=True):
        raise ValueError("S2 run receipt cwd drift")
    start_epoch = float(receipt.get("start_epoch"))
    end_epoch = float(receipt.get("end_epoch"))
    if not (
        math.isfinite(start_epoch)
        and math.isfinite(end_epoch)
        and end_epoch > start_epoch
    ):
        raise ValueError("S2 run receipt epochs are invalid")

    executable = _validate_record(receipt.get("executable"), "S2 run executable")
    current_executable = _file_record(executable_path)
    if (
        executable["path"] != current_executable["path"]
        or executable["sha256"] != current_executable["sha256"]
    ):
        raise ValueError("S2 run receipt executable drift")
    log = _validate_record(receipt.get("log"), "S2 run log")
    expected_log = (run_directory / "log").resolve(strict=True)
    if Path(log["path"]) != expected_log:
        raise ValueError("S2 run receipt log path drift")

    output_records = receipt.get("outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("S2 run receipt has no output inventory")
    inventory: dict[Path, dict[str, Any]] = {}
    for index, record in enumerate(output_records):
        actual = _validate_record(record, f"S2 run output {index}")
        path = Path(actual["path"])
        if path.parent != run_directory.resolve(strict=True):
            raise ValueError("S2 run receipt inventories an external path")
        if actual["mtime"] < start_epoch - 1.0 or actual["mtime"] > end_epoch + 1.0:
            raise ValueError(f"S2 run output lies outside receipt epoch: {path}")
        inventory[path] = actual
    for required in required_outputs:
        resolved = required.resolve(strict=True)
        if resolved not in inventory:
            raise ValueError(f"S2 receipt omits required output: {resolved.name}")
        current = _file_record(resolved)
        recorded = inventory[resolved]
        if current["sha256"] != recorded["sha256"]:
            raise ValueError(f"S2 receipt output drift: {resolved.name}")
    return {
        "path": str(receipt_path.resolve(strict=True)),
        "sha256": _sha256(receipt_path),
        "command": command,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "executable": executable,
        "outputs": inventory,
        "pass": True,
    }


def _c_percent_g(values: np.ndarray) -> np.ndarray:
    """Round values as the source's C ``%g`` (default six significant digits)."""

    return np.asarray(
        [float(format(float(value), ".6g")) for value in values],
        dtype=np.float64,
    )


def _validate_step_log_binding(log_path: Path, step: np.ndarray) -> dict[str, Any]:
    """Prove the 11-column sidecar and official 16-column log are one run."""

    log, _ = _read_log(log_path)
    if len(log) != len(step):
        raise ValueError("instrumented step/log row counts differ")
    checks = {
        "iteration_exact": bool(np.array_equal(log[:, 0], step[:, 0])),
        "normalized_time_exact_after_source_format": bool(
            np.array_equal(log[:, 1], _c_percent_g(step[:, 1] - 0.2))
        ),
        "dt_exact_after_source_format": bool(
            np.array_equal(log[:, 2], _c_percent_g(step[:, 2]))
        ),
        "theta_exact_after_source_format": bool(
            np.array_equal(log[:, 13], _c_percent_g(step[:, 3]))
        ),
        "cd_exact_after_source_format": bool(
            np.array_equal(
                log[:, 14], _c_percent_g(2.0 * (step[:, 4] + step[:, 6]))
            )
        ),
        "cl_exact_after_source_format": bool(
            np.array_equal(
                log[:, 15], _c_percent_g(2.0 * (step[:, 5] + step[:, 7]))
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"instrumented step/log semantic mismatch: {failed}")
    return {"rows": int(len(log)), "checks": checks, "pass": True}


def _load_s2_run(manifest: dict[str, Any], label: str) -> dict[str, Any]:
    record = manifest["runs"].get(label)
    if not isinstance(record, dict):
        raise ValueError(f"missing S2 run {label}")
    expected_layout = S2_RUN_LAYOUT[label]
    expected_lmax = expected_layout[2]
    if int(record.get("lmax")) != expected_lmax:
        raise ValueError(f"{label} lmax drift")
    paths = {
        key: _validated_manifest_path(record[key], f"{label}:{key}")
        for key in ("executable", "log", "step", "score", "receipt")
    }
    expected_config = {
        "lmax": expected_lmax,
        "dt_cap": float(record["dt_cap"]),
    }
    stored_score = _load_json(paths["score"], "fluxv.n26f.instrumented_run.v1")
    if stored_score.get("configuration") != expected_config:
        raise ValueError(f"{label} score/manifest configuration mismatch")
    recomputed_score = score_instrumented_run(
        Path(record["run_directory"]),
        expected_config["dt_cap"],
        expected_config["lmax"],
        None,
    )
    if stored_score != recomputed_score:
        raise ValueError(f"{label} stored score does not match current raw evidence")
    required_outputs = [
        paths["log"],
        paths["step"],
        *(
            Path(checkpoint["files"][role]["path"])
            for checkpoint in recomputed_score["checkpoints"].values()
            for role in ("traction", "metadata", "cp")
        ),
    ]
    receipt = _validate_s2_run_receipt(
        paths["receipt"],
        Path(record["run_directory"]),
        paths["executable"],
        required_outputs,
        (
            "/usr/bin/time -v -f exit_status=%x -o runtime.txt "
            f"./{paths['executable'].name} >stdout.log 2>log"
            if label == "space_l15"
            else "/usr/bin/time -v -o runtime.txt "
            f"./{paths['executable'].name} >stdout.log 2>log"
        ),
    )
    curve, curve_diagnostics = load_curve(paths["log"])
    step, time_diagnostics = load_instrumented_steps(
        paths["step"], float(record["dt_cap"])
    )
    step_log_binding = _validate_step_log_binding(paths["log"], step)
    return {
        "record": record,
        "paths": paths,
        "receipt": receipt,
        "score": recomputed_score,
        "curve": curve,
        "curve_diagnostics": curve_diagnostics,
        "step": step,
        "time_diagnostics": time_diagnostics,
        "step_log_binding": step_log_binding,
    }


def _neutrality_metric(official_log: Path, instrumented_log: Path) -> dict[str, Any]:
    official, _ = _read_log(official_log)
    instrumented, _ = _read_log(instrumented_log)
    same_shape = official.shape == instrumented.shape
    if not same_shape:
        return {
            "same_shape": False,
            "official_shape": list(official.shape),
            "instrumented_shape": list(instrumented.shape),
            "pass": False,
        }
    identity_columns = (0, 1, 2, 13)
    force_columns = (14, 15)
    identity_exact = bool(
        np.array_equal(official[:, identity_columns], instrumented[:, identity_columns])
    )
    force_difference = official[:, force_columns] - instrumented[:, force_columns]
    force_denominator = float(np.linalg.norm(official[:, force_columns]))
    force_relative_l2 = (
        float(np.linalg.norm(force_difference) / force_denominator)
        if force_denominator > 0.0
        else math.inf
    )
    force_max_abs = float(np.max(np.abs(force_difference)))
    return {
        "same_shape": True,
        "identity_columns_i_tau_dt_theta_exact": identity_exact,
        "force_relative_l2": force_relative_l2,
        "force_max_abs": force_max_abs,
        "full_log_byte_sha_equal": _sha256(official_log) == _sha256(instrumented_log),
        "pass": bool(identity_exact and force_relative_l2 <= 1.0e-12),
    }


def score_s2_family(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.resolve() == manifest_path.resolve(strict=True):
        raise ValueError("S2 family output overlaps its manifest input")
    manifest = _load_json(manifest_path, "fluxv.n26f.s2_family_manifest.v1")
    official_log = _validated_manifest_path(
        manifest["official_formal_log"], "official formal log"
    )
    s1_score_path = _validated_manifest_path(
        manifest["s1_formal_score"], "S1 formal score"
    )
    instrumented_source = _validated_manifest_path(
        manifest["instrumented_source"], "instrumented source"
    )
    parent_source = _validated_manifest_path(
        manifest["parent_source"], "parent source"
    )
    source_diff = _validated_manifest_path(manifest["source_diff"], "source diff")
    freeze_path = _validated_manifest_path(
        manifest["time_axis_freeze"], "time-axis freeze"
    )
    repro_build_path = _validated_manifest_path(
        manifest["repro_build_manifest"], "S2 reproducible build manifest"
    )
    if _sha256(parent_source) != SOURCE_SHA256:
        raise ValueError("S2 parent source identity drift")
    if _sha256(instrumented_source) != INSTRUMENTED_SOURCE_SHA256:
        raise ValueError("S2 instrumented source identity drift")
    if _sha256(source_diff) != INSTRUMENTED_DIFF_SHA256:
        raise ValueError("S2 source diff identity drift")
    build_validation = _validate_s2_repro_build(
        repro_build_path,
        parent_source,
        instrumented_source,
        source_diff,
        freeze_path,
    )
    s1_binding = _validate_s1_official_log(s1_score_path, official_log)
    freeze = _load_json(freeze_path, "fluxv.n26f.time_axis_freeze.v1")

    runs = {label: _load_s2_run(manifest, label) for label in S2_RUN_LAYOUT}
    time_axis_validation = _validate_time_axis_freeze(
        freeze_path, runs["space_l14"]["paths"]["step"]
    )
    for label, run in runs.items():
        executable_name = S2_RUN_LAYOUT[label][1]
        if (
            _sha256(run["paths"]["executable"])
            != build_validation["builds"][executable_name]["sha256"]
        ):
            raise ValueError(f"{label} executable differs from reproducible build")
    expected_static_caps = {
        "space_l13": 0.01,
        "space_l14": 0.01,
        "space_l15": 0.01,
        "time_orig_0p02": 0.02,
        "time_orig_0p01": 0.01,
        "time_orig_0p005": 0.005,
    }
    for label, expected in expected_static_caps.items():
        if float(runs[label]["record"]["dt_cap"]) != expected:
            raise ValueError(f"{label} DT cap drift")
    frozen_caps = [float(value) for value in freeze["frozen_dt_caps"]]
    for label, expected in zip(
        ("time_new_d50", "time_new_half", "time_new_quarter"), frozen_caps
    ):
        if float(runs[label]["record"]["dt_cap"]) != expected:
            raise ValueError(f"{label} frozen DT cap drift")
    if runs["time_orig_0p01"]["paths"]["step"] != runs["space_l14"]["paths"]["step"]:
        raise ValueError("space_l14 and time_orig_0p01 are not the same run")

    space_13_14 = compare_cauchy(
        runs["space_l13"]["paths"]["log"],
        runs["space_l14"]["paths"]["log"],
        0.01,
        0.01,
        None,
    )
    space_14_15 = compare_cauchy(
        runs["space_l14"]["paths"]["log"],
        runs["space_l15"]["paths"]["log"],
        0.01,
        0.01,
        None,
    )
    space_pass = bool(space_14_15["curve_metric_pass"])

    orig_medium = runs["time_orig_0p01"]["time_diagnostics"]
    orig_fine = runs["time_orig_0p005"]["time_diagnostics"]
    original_dt_ratio = (
        orig_medium["post_start_actual_dt"]["median"]
        / orig_fine["post_start_actual_dt"]["median"]
    )
    original_degenerate = bool(
        orig_medium["cap_active_fraction"] < 0.5
        or orig_fine["cap_active_fraction"] < 0.5
        or original_dt_ratio < 1.5
    )
    original_time_curve = compare_cauchy(
        runs["time_orig_0p01"]["paths"]["log"],
        runs["time_orig_0p005"]["paths"]["log"],
        0.01,
        0.005,
        None,
    )

    new_medium = runs["time_new_half"]["time_diagnostics"]
    new_fine = runs["time_new_quarter"]["time_diagnostics"]
    new_dt_ratio = (
        new_medium["post_start_actual_dt"]["median"]
        / new_fine["post_start_actual_dt"]["median"]
    )
    replacement_curve = compare_cauchy(
        runs["time_new_half"]["paths"]["log"],
        runs["time_new_quarter"]["paths"]["log"],
        frozen_caps[1],
        frozen_caps[2],
        None,
    )
    replacement_axis_pass = bool(
        new_medium["cap_active_fraction"] >= 0.5
        and new_fine["cap_active_fraction"] >= 0.5
        and new_dt_ratio >= 1.5
        and replacement_curve["curve_metric_pass"]
    )
    time_pass = (
        replacement_axis_pass
        if original_degenerate
        else bool(original_time_curve["curve_metric_pass"])
    )

    neutrality = _neutrality_metric(
        official_log, runs["space_l15"]["paths"]["log"]
    )
    finest_traction = runs["space_l15"]["score"]
    traction_pass = bool(finest_traction.get("traction_pass"))
    geometry_progression = {
        label: {
            angle: runs[label]["score"]["checkpoints"][angle]["geometry_closure"]
            for angle in ("44", "54")
        }
        for label in ("space_l13", "space_l14", "space_l15")
    }
    complete_pass = all(
        run["curve_diagnostics"]["complete"] for run in runs.values()
    )
    s2_pass = bool(
        complete_pass
        and space_pass
        and time_pass
        and neutrality["pass"]
        and traction_pass
    )
    result = {
        "schema": "fluxv.n26f.s2_family_result.v1",
        "scope": "N2.6f1.S2_SOURCE_NUMERICS_AND_TRACTION",
        "manifest": _file_record(manifest_path),
        "s1_formal_response_pass": True,
        "s1_official_log_binding": s1_binding,
        "time_axis_rederivation": time_axis_validation,
        "step_log_bindings": {
            label: run["step_log_binding"] for label, run in runs.items()
        },
        "all_runs_complete": complete_pass,
        "instrumentation_neutrality": neutrality,
        "space": {
            "l13_vs_l14": space_13_14,
            "l14_vs_l15": space_14_15,
            "pass": space_pass,
        },
        "time": {
            "original_axis": {
                "curve": original_time_curve,
                "medium_cap_active_fraction": orig_medium["cap_active_fraction"],
                "fine_cap_active_fraction": orig_fine["cap_active_fraction"],
                "median_actual_dt_ratio": original_dt_ratio,
                "status": (
                    "AXIS-DEGENERATE" if original_degenerate else "ACTIVE"
                ),
            },
            "replacement_axis": {
                "freeze": freeze,
                "curve": replacement_curve,
                "medium_cap_active_fraction": new_medium["cap_active_fraction"],
                "fine_cap_active_fraction": new_fine["cap_active_fraction"],
                "median_actual_dt_ratio": new_dt_ratio,
                "pass": replacement_axis_pass,
            },
            "pass": time_pass,
        },
        "traction": {
            "finest_l15": finest_traction,
            "geometry_progression": geometry_progression,
            "pass": traction_pass,
        },
        "s2_pass": s2_pass,
        "n26f2_target_observation_authorized": s2_pass,
        "claim_promotion_authorized": s2_pass,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _mean_exact_angle_groups(
    angle: np.ndarray,
    angle_tokens: list[str],
    cd: np.ndarray,
    cl: np.ndarray,
) -> Curve:
    """Average only adjacent, print-identical post-start pitch angles."""

    if len(angle) == 0:
        raise ValueError("empty post-start response")
    if np.any(np.diff(angle) < -1.0e-10):
        raise ValueError("post-start pitch angle is not monotone")

    grouped_angle: list[float] = []
    grouped_cd: list[float] = []
    grouped_cl: list[float] = []
    start = 0
    for index in range(1, len(angle) + 1):
        if index == len(angle) or angle_tokens[index] != angle_tokens[start]:
            grouped_angle.append(float(angle[start]))
            grouped_cd.append(float(np.mean(cd[start:index])))
            grouped_cl.append(float(np.mean(cl[start:index])))
            start = index
    out = Curve(
        np.asarray(grouped_angle),
        np.asarray(grouped_cd),
        np.asarray(grouped_cl),
    )
    if np.any(np.diff(out.angle_deg) <= 0.0):
        raise ValueError("angle collapse did not produce a strict coordinate")
    return out


def load_curve(log_path: Path) -> tuple[Curve, dict[str, Any]]:
    rows, angle_tokens = _read_log(log_path)
    finite = np.isfinite(rows)
    all_finite = bool(finite.all())
    if not all_finite:
        bad = np.argwhere(~finite)[0].tolist()
        raise ValueError(f"non-finite log value at row/column {bad}")

    # Official columns: normalized time=1, dt=2, pitch angle=13, CD=14, CL=15.
    time_normalized = rows[:, 1]
    dt = rows[:, 2]
    angle = rows[:, 13]
    integer_columns = (0, 3, 4, 5, 6, 7, 8)
    if np.any(rows[:, integer_columns] < 0.0) or not np.array_equal(
        rows[:, integer_columns], np.rint(rows[:, integer_columns])
    ):
        raise ValueError("iteration/multigrid integer columns are invalid")
    if int(rows[0, 0]) != 0 or not np.array_equal(
        np.diff(rows[:, 0]), np.ones(len(rows) - 1)
    ):
        raise ValueError("iteration index does not start at zero and increment by one")
    if np.any(np.diff(time_normalized) <= 0.0):
        raise ValueError("normalized time is not strictly increasing")
    if np.any(dt <= 0.0):
        raise ValueError("non-positive timestep")
    post_mask = time_normalized >= 0.0
    if not np.any(post_mask):
        raise ValueError("source run never reached pitch start")

    post = rows[post_mask]
    post_tokens = [
        token for token, selected in zip(angle_tokens, post_mask) if bool(selected)
    ]
    curve = _mean_exact_angle_groups(
        post[:, 13], post_tokens, post[:, 14], post[:, 15]
    )
    post_dt = post[:, 2]
    diagnostics = {
        "numeric_rows": int(len(rows)),
        "all_nonempty_rows_parsed": True,
        "all_finite": all_finite,
        "normalized_time_min": float(time_normalized.min()),
        "normalized_time_max": float(time_normalized.max()),
        "angle_deg_min": float(angle.min()),
        "angle_deg_max": float(angle.max()),
        "post_start_rows": int(post_mask.sum()),
        "post_start_dt": {
            "min": float(post_dt.min()),
            "median": float(np.median(post_dt)),
            "max": float(post_dt.max()),
        },
        "complete": bool(
            time_normalized[-1] >= 1.85 - max(float(dt[-1]), 5.0e-6)
            and angle.max() >= ANGLE_MAX_DEG
        ),
    }
    return curve, diagnostics


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"invalid two-column reference: {path}")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"reference angle is not strictly increasing: {path}")
    return data[:, 0], data[:, 1]


def _curve_knots(
    first_x: np.ndarray,
    second_x: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    if (
        first_x.min() > lower
        or second_x.min() > lower
        or first_x.max() < upper
        or second_x.max() < upper
    ):
        raise ValueError(
            f"curve support does not cover frozen interval [{lower}, {upper}]"
        )
    knots = np.concatenate(
        (
            np.asarray([lower, upper]),
            first_x[(first_x > lower) & (first_x < upper)],
            second_x[(second_x > lower) & (second_x < upper)],
        )
    )
    return np.unique(knots)


def _integral_linear_square(knots: np.ndarray, values: np.ndarray) -> float:
    width = np.diff(knots)
    left = values[:-1]
    right = values[1:]
    return float(np.sum(width * (left * left + left * right + right * right) / 3.0))


def _piecewise_peak(
    x: np.ndarray, y: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    knots = np.concatenate(
        (
            np.asarray([lower]),
            x[(x > lower) & (x < upper)],
            np.asarray([upper]),
        )
    )
    values = np.interp(knots, x, y)
    index = int(np.argmax(values))
    return float(knots[index]), float(values[index])


def _reference_metric(
    model: Curve, reference_path: Path, field: str
) -> dict[str, Any]:
    ref_angle, ref_value = _read_reference(reference_path)
    lower, upper = REFERENCE_SUPPORT_DEG[field]
    if not (ref_angle.min() == lower and ref_angle.max() == upper):
        raise ValueError(f"{field} reference support identity drift")
    model_field = model.cd if field == "CD" else model.cl
    knots = _curve_knots(model.angle_deg, ref_angle, lower, upper)
    model_value = np.interp(knots, model.angle_deg, model_field)
    reference_value = np.interp(knots, ref_angle, ref_value)
    error_integral = _integral_linear_square(knots, model_value - reference_value)
    rmse = math.sqrt(error_integral / (upper - lower))
    ref_range = float(np.ptp(ref_value))
    if ref_range <= 0.0:
        raise ValueError(f"zero {field} reference range")
    rnrmse = rmse / ref_range

    # A piecewise-linear peak is attained at one of its own knots.  np.argmax
    # deterministically selects the smaller angle if equal maxima occur.
    model_peak_angle, model_peak_value = _piecewise_peak(
        model.angle_deg, model_field, lower, upper
    )
    ref_peak_angle, ref_peak_value = _piecewise_peak(
        ref_angle, ref_value, lower, upper
    )
    peak_angle_error = abs(model_peak_angle - ref_peak_angle)
    return {
        "reference_points": int(len(ref_angle)),
        "support_deg": [lower, upper],
        "union_knots": int(len(knots)),
        "rmse": rmse,
        "reference_range": ref_range,
        "range_normalized_rmse": rnrmse,
        "model_peak_angle_deg": model_peak_angle,
        "reference_peak_angle_deg": ref_peak_angle,
        "peak_angle_error_deg": peak_angle_error,
        "model_peak_value": model_peak_value,
        "reference_peak_value": ref_peak_value,
        "pass": bool(
            rnrmse <= SOURCE_RNRMSE_LIMIT
            and peak_angle_error <= SOURCE_PEAK_ANGLE_LIMIT_DEG
        ),
    }


def _plot_reference(
    model: Curve,
    reference_paths: dict[str, Path],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for axis, field, color in zip(axes, ("CD", "CL"), ("tab:blue", "tab:red")):
        ref_angle, ref_value = _read_reference(reference_paths[field])
        model_field = model.cd if field == "CD" else model.cl
        lower, upper = REFERENCE_SUPPORT_DEG[field]
        mask = (
            (model.angle_deg >= lower)
            & (model.angle_deg <= upper)
        )
        axis.plot(
            model.angle_deg[mask],
            model_field[mask],
            color=color,
            linewidth=1.4,
            label="Basilisk formal source",
        )
        axis.scatter(
            ref_angle,
            ref_value,
            s=13,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label="Schneiders et al. digitization",
        )
        axis.set(
            xlabel=r"pitch angle $\theta$ (deg)",
            ylabel=field,
            xlim=(lower, upper),
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def score_source(
    log_path: Path,
    source_path: Path,
    cd_reference: Path,
    cl_reference: Path,
    asset_manifest_path: Path,
    run_receipt_path: Path,
    output_path: Path | None,
    plot_path: Path | None,
) -> dict[str, Any]:
    input_paths = {
        path.resolve(strict=True)
        for path in (
            log_path,
            source_path,
            cd_reference,
            cl_reference,
            asset_manifest_path,
            run_receipt_path,
        )
    }
    derived_paths = [path for path in (output_path, plot_path) if path is not None]
    if len({path.resolve() for path in derived_paths}) != len(derived_paths):
        raise ValueError("output JSON and plot paths must be distinct")
    for path in derived_paths:
        if path.resolve() in input_paths:
            raise ValueError("derived output path overlaps immutable evidence")

    asset_validation = _validate_asset_manifest(
        asset_manifest_path, source_path, cd_reference, cl_reference
    )
    run_validation = _validate_run_receipt(
        run_receipt_path, log_path, asset_validation
    )
    identities = {
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "expected_sha256": SOURCE_SHA256,
        },
        "CD_reference": {
            "path": str(cd_reference),
            "sha256": _sha256(cd_reference),
            "expected_sha256": REFERENCE_SHA256["CD"],
        },
        "CL_reference": {
            "path": str(cl_reference),
            "sha256": _sha256(cl_reference),
            "expected_sha256": REFERENCE_SHA256["CL"],
        },
    }
    identity_pass = all(
        entry["sha256"] == entry["expected_sha256"] for entry in identities.values()
    )
    curve, diagnostics = load_curve(log_path)
    references = {"CD": cd_reference, "CL": cl_reference}
    metrics = {
        field: _reference_metric(curve, reference_path, field)
        for field, reference_path in references.items()
    }
    run_directory = log_path.parent
    required_patterns = (
        "cp-angle-44-pid-*",
        "cp-angle-54-pid-*",
        "omega-zoom-angle-44.png",
        "omega-zoom-angle-54.png",
    )
    output_evidence: dict[str, list[dict[str, Any]]] = {}
    output_pass = True
    for pattern in required_patterns:
        paths = sorted(
            path.resolve()
            for path in run_directory.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
        records = [_file_record(path) for path in paths]
        output_evidence[pattern] = records
        inventory_paths = run_validation["outputs"]
        output_pass = (
            output_pass
            and bool(records)
            and all(record["path"] in inventory_paths for record in records)
        )
    result = {
        "schema": "fluxv.n26f.s1_formal_response.v1",
        "scope": "N2.6f1.S1_RESPONSE_ONLY",
        "claim_promotion_authorized": False,
        "log": str(log_path.resolve(strict=True)),
        "asset_manifest_validation": asset_validation,
        "run_receipt_validation": run_validation,
        "identity": identities,
        "identity_pass": identity_pass,
        "diagnostics": diagnostics,
        "required_outputs": output_evidence,
        "required_outputs_pass": output_pass,
        "metric_definition": {
            "source_rmse": (
                "exact squared-error integral between unsmoothed piecewise-"
                "linear model/reference curves on their full frozen support; "
                "root mean integral divided by reference max-minus-min"
            ),
            "peak_angle": (
                "first argmax on each curve's own piecewise-linear knots "
                "within the frozen support"
            ),
            "range_normalized_rmse_limit": SOURCE_RNRMSE_LIMIT,
            "peak_angle_error_limit_deg": SOURCE_PEAK_ANGLE_LIMIT_DEG,
        },
        "metrics": metrics,
    }
    result["formal_response_pass"] = bool(
        asset_validation["pass"]
        and run_validation["pass"]
        and identity_pass
        and diagnostics["all_finite"]
        and diagnostics["complete"]
        and output_pass
        and all(metric["pass"] for metric in metrics.values())
    )
    if plot_path is not None:
        _plot_reference(curve, references, plot_path)
        result["comparison_plot"] = _file_record(plot_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def compare_cauchy(
    coarse_log: Path,
    fine_log: Path,
    coarse_dt_cap: float | None,
    fine_dt_cap: float | None,
    output_path: Path | None,
) -> dict[str, Any]:
    coarse, coarse_diag = load_curve(coarse_log)
    fine, fine_diag = load_curve(fine_log)
    metrics: dict[str, Any] = {}
    for field in ("CD", "CL"):
        lower, upper = REFERENCE_SUPPORT_DEG[field]
        coarse_field = coarse.cd if field == "CD" else coarse.cl
        fine_field = fine.cd if field == "CD" else fine.cl
        knots = _curve_knots(coarse.angle_deg, fine.angle_deg, lower, upper)
        coarse_value = np.interp(knots, coarse.angle_deg, coarse_field)
        fine_value = np.interp(knots, fine.angle_deg, fine_field)
        numerator = _integral_linear_square(knots, coarse_value - fine_value)
        denominator = _integral_linear_square(knots, fine_value)
        if denominator <= 0.0:
            raise ValueError(f"zero fine {field} norm")
        relative_l2 = math.sqrt(numerator / denominator)
        coarse_peak_angle, coarse_peak = _piecewise_peak(
            coarse.angle_deg, coarse_field, lower, upper
        )
        fine_peak_angle, fine_peak = _piecewise_peak(
            fine.angle_deg, fine_field, lower, upper
        )
        if fine_peak == 0.0:
            raise ValueError(f"zero fine {field} peak")
        peak_denominator = abs(fine_peak)
        peak_change = abs(coarse_peak - fine_peak) / peak_denominator
        metrics[field] = {
            "support_deg": [lower, upper],
            "union_knots": int(len(knots)),
            "relative_l2": relative_l2,
            "peak_relative_change": peak_change,
            "coarse_peak": coarse_peak,
            "fine_peak": fine_peak,
            "coarse_peak_angle_deg": coarse_peak_angle,
            "fine_peak_angle_deg": fine_peak_angle,
            "peak_angle_drift_deg": abs(coarse_peak_angle - fine_peak_angle),
            "pass": bool(
                relative_l2 <= CAUCHY_RELATIVE_L2_LIMIT
                and peak_change <= CAUCHY_PEAK_CHANGE_LIMIT
            ),
        }

    result = {
        "schema": "fluxv.n26f.cauchy_metric_kernel.v1",
        "scope": "METRIC_KERNEL_ONLY",
        "protocol_status": "PROTOCOL_INCOMPLETE",
        "claim_promotion_authorized": False,
        "coarse_log": str(coarse_log),
        "fine_log": str(fine_log),
        "reported_dt_caps_not_validated": {
            "coarse": coarse_dt_cap,
            "fine": fine_dt_cap,
        },
        "support_deg": REFERENCE_SUPPORT_DEG,
        "definition": {
            "relative_l2": (
                "sqrt(integral((coarse-fine)^2)/integral(fine^2)) "
                "using exact piecewise-linear union-knot quadrature"
            ),
            "peak_relative_change": "|max(coarse)-max(fine)|/|max(fine)|",
            "relative_l2_limit": CAUCHY_RELATIVE_L2_LIMIT,
            "peak_relative_change_limit": CAUCHY_PEAK_CHANGE_LIMIT,
        },
        "coarse_diagnostics": coarse_diag,
        "fine_diagnostics": fine_diag,
        "metrics": metrics,
        "curve_metric_pass": bool(
            coarse_diag["complete"]
            and fine_diag["complete"]
            and all(metric["pass"] for metric in metrics.values())
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def _source_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("source", help="score one formal source run")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cd-reference", type=Path, required=True)
    parser.add_argument("--cl-reference", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)


def _cauchy_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cauchy", help="compare two source-only runs")
    parser.add_argument("--coarse-log", type=Path, required=True)
    parser.add_argument("--fine-log", type=Path, required=True)
    parser.add_argument("--coarse-dt-cap", type=float)
    parser.add_argument("--fine-dt-cap", type=float)
    parser.add_argument("--output", type=Path, required=True)


def _receipt_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "receipt", help="inventory one completed source-only process"
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--command-line", required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument("--return-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _instrumented_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "instrumented", help="score one S2 instrumentation/traction component"
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--dt-cap", type=float, required=True)
    parser.add_argument("--lmax", type=int, choices=(13, 14, 15), required=True)
    parser.add_argument("--output", type=Path, required=True)


def _time_axis_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "derive-time-axis",
        help="freeze a non-degenerate source-only time axis from d_phys",
    )
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _s2_repro_build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "s2-repro-build",
        help="rebuild S2 binaries and freeze the source/macro identity chain",
    )
    parser.add_argument("--qcc", type=Path, required=True)
    parser.add_argument("--instrumented-source", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--source-diff", type=Path, required=True)
    parser.add_argument("--time-axis-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _s2_manifest_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "make-s2-manifest",
        help="bind all completed S2 runs to immutable evidence records",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-log", type=Path, required=True)
    parser.add_argument("--s1-score", type=Path, required=True)
    parser.add_argument("--instrumented-source", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--source-diff", type=Path, required=True)
    parser.add_argument("--time-axis-freeze", type=Path, required=True)
    parser.add_argument("--repro-build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _s2_family_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "s2-family", help="score the complete frozen S2 source-validation family"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _source_parser(subparsers)
    _cauchy_parser(subparsers)
    _receipt_parser(subparsers)
    _instrumented_parser(subparsers)
    _time_axis_parser(subparsers)
    _s2_repro_build_parser(subparsers)
    _s2_manifest_parser(subparsers)
    _s2_family_parser(subparsers)
    args = parser.parse_args()

    try:
        if args.command == "source":
            result = score_source(
                args.log,
                args.source,
                args.cd_reference,
                args.cl_reference,
                args.asset_manifest,
                args.run_receipt,
                args.output,
                args.plot,
            )
            return_code = 0 if result["formal_response_pass"] else 2
        elif args.command == "cauchy":
            result = compare_cauchy(
                args.coarse_log,
                args.fine_log,
                args.coarse_dt_cap,
                args.fine_dt_cap,
                args.output,
            )
            # This command is deliberately not a claim gate until fixed-axis
            # receipts, d_phys semantics and the traction ledger are wired.
            return_code = 3
        elif args.command == "receipt":
            result = make_run_receipt(
                args.run_directory,
                args.log,
                args.executable,
                args.command_line,
                args.start_epoch,
                args.end_epoch,
                args.return_code,
                args.output,
            )
            result = {
                "schema": "fluxv.n26f.receipt_creation.v1",
                "receipt": _file_record(args.output),
                "recorded_process_return_code": result["return_code"],
                "claim_promotion_authorized": False,
            }
            return_code = 0
        elif args.command == "instrumented":
            result = score_instrumented_run(
                args.run_directory,
                args.dt_cap,
                args.lmax,
                args.output,
            )
            return_code = 0 if result["traction_pass"] else 2
        elif args.command == "derive-time-axis":
            result = derive_time_axis(args.step, args.output)
            return_code = 0
        elif args.command == "s2-repro-build":
            result = make_s2_repro_build(
                args.qcc,
                args.instrumented_source,
                args.parent_source,
                args.source_diff,
                args.time_axis_freeze,
                args.output_root,
                args.output,
            )
            return_code = 0
        elif args.command == "make-s2-manifest":
            result = make_s2_manifest(
                args.root,
                args.official_log,
                args.s1_score,
                args.instrumented_source,
                args.parent_source,
                args.source_diff,
                args.time_axis_freeze,
                args.repro_build_manifest,
                args.output,
            )
            return_code = 0
        elif args.command == "s2-family":
            result = score_s2_family(args.manifest, args.output)
            return_code = 0 if result["s2_pass"] else 2
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = {
            "schema": "fluxv.n26f.protocol_error.v1",
            "scope": str(args.command),
            "reason_code": type(exc).__name__,
            "message": str(exc),
            "claim_promotion_authorized": False,
        }
        output_path = getattr(args, "output", None)
        if output_path is not None:
            inputs = {
                Path(value).resolve()
                for name, value in vars(args).items()
                if name != "output" and isinstance(value, Path)
            }
            if output_path.resolve() not in inputs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
        return_code = 4
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
