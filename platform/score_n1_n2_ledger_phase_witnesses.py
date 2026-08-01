"""Deterministically score the preregistered N1/N2 G0 witness campaign.

This module contains no aerodynamic solver call.  It implements the decision
matrix frozen in ``n1_n2_ledger_phase_witness_prereg_20260729.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402
import run_n1_n2_ledger_phase_witnesses as witness  # noqa: E402


SCHEMA = "n1-n2-ledger-identifiability-score-v1"
TAU_F_N = 0.15
TAU_CONTRAST_N = 0.30
NUMERIC_ZERO = 1.0e-12
COLLINEAR_CONDITION_LIMIT = 20.0
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
FIG16_SOURCE_SHA256 = (
    "15fa067119743efee1c509aeb1657fb16393fb74b9db905f8d7a09dcc8fe9072"
)
PREREG_SHA256 = (
    "e0c6e0a41a221a42a34a6185919a1ad7d56103adca6fd1f71800b0a6ef3479be"
)

N1_YAML = PLATFORM / "claim_nodes" / "n1_uvlm.yaml"
N2_YAML = PLATFORM / "claim_nodes" / "n2_kirchhoff.yaml"

RAW_STEPS = witness.G0_STEPS_PER_CYCLE
RAW_NPAN = witness.G0_NC * witness.G0_NS
RAW_NS = witness.G0_NS

# The recorder contract is intentionally exact.  A source change that adds or
# removes a field must update the preregistration and this scorer before it can
# vote in a claim decision.
RAW_EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "step": (RAW_STEPS,),
    "last_cycle_step": (RAW_STEPS,),
    "cycle_index": (RAW_STEPS,),
    "time_s": (RAW_STEPS,),
    "dt_s": (RAW_STEPS,),
    "phase_solver_rad": (RAW_STEPS,),
    "phase_paper_rad": (RAW_STEPS,),
    "theta_rad": (RAW_STEPS,),
    "theta_dot_rad_s": (RAW_STEPS,),
    "psi_rad": (RAW_STEPS, RAW_NS),
    "psi_dot_rad_s": (RAW_STEPS, RAW_NS),
    "y_ref_edge_m": (RAW_STEPS, RAW_NS + 1),
    "y_ref_center_m": (RAW_STEPS, RAW_NS),
    "eta_ref": (RAW_STEPS, RAW_NS),
    "snapshot_phase": (RAW_STEPS,),
    "nc": (RAW_STEPS,),
    "ns": (RAW_STEPS,),
    "n1.collocation_points_m": (RAW_STEPS, RAW_NPAN, 3),
    "n1.panel_normals_body": (RAW_STEPS, RAW_NPAN, 3),
    "n1.panel_area_m2": (RAW_STEPS, RAW_NPAN),
    "n1.bernoulli_local_velocity_m_s": (RAW_STEPS, RAW_NPAN, 3),
    "n1.collocation_wall_velocity_m_s": (RAW_STEPS, RAW_NPAN, 3),
    "n1.bound_gamma_m2_s": (RAW_STEPS, RAW_NPAN),
    "n1.dGdx_m_s": (RAW_STEPS, RAW_NPAN),
    "n1.dGdy_m_s": (RAW_STEPS, RAW_NPAN),
    "n1.dGdt_m2_s2": (RAW_STEPS, RAW_NPAN),
    "n1.pressure_jump_Pa": (RAW_STEPS, RAW_NPAN),
    "n1.panel_force_body_N": (RAW_STEPS, RAW_NPAN, 3),
    "n1.bernoulli_booked_solver_accumulator_N": (RAW_STEPS, 3),
    "n1.non_bernoulli_unallocated_solver_accumulator_N": (
        RAW_STEPS,
        3,
    ),
    "n1.booked_solver_accumulator_total_N": (RAW_STEPS, 3),
    "n2.alpha_eff_lb_rad": (RAW_STEPS, RAW_NS),
    "n2.f_qs": (RAW_STEPS, RAW_NS),
    "n2.f2": (RAW_STEPS, RAW_NS),
    "n2.K": (RAW_STEPS, RAW_NS),
    "n2.CNc": (RAW_STEPS, RAW_NS),
    "n2.CNf": (RAW_STEPS, RAW_NS),
    "n2.CNv": (RAW_STEPS, RAW_NS),
    "n2.CV": (RAW_STEPS, RAW_NS),
    "n2.loss_frac": (RAW_STEPS, RAW_NS),
    "n2.separation_panel_candidate_force_body_N": (
        RAW_STEPS,
        RAW_NPAN,
        3,
    ),
    "n2.separation_booked_solver_accumulator_N": (RAW_STEPS, 3),
    "n2.profile_drag_booked_solver_accumulator_N": (RAW_STEPS, 3),
    "n2.booked_solver_accumulator_total_N": (RAW_STEPS, 3),
    "n3.A0_signed": (RAW_STEPS, RAW_NS),
    "n3.lb_lesp_crit": (RAW_STEPS,),
    "n3.A0_excess_pre_cds": (RAW_STEPS, RAW_NS),
    "n3.dCN_drive_after_cds": (RAW_STEPS, RAW_NS),
    "n3.dCN_drive_after_sign": (RAW_STEPS, RAW_NS),
    "n3.dCN_drive_after_f2gate": (RAW_STEPS, RAW_NS),
    "n3.dCN_state_after_memory": (RAW_STEPS, RAW_NS),
    "n3.u_le_normal_signed_m_s": (RAW_STEPS, RAW_NS),
    "n3.Urel_le_m_s": (RAW_STEPS, RAW_NS),
    "n3.q_dyn_Pa": (RAW_STEPS, RAW_NS),
    "n3.alpha_kin_rad": (RAW_STEPS, RAW_NS),
    "n3.event_active": (RAW_STEPS, RAW_NS),
    "n3.event_onset": (RAW_STEPS, RAW_NS),
    "n3.event_sign": (RAW_STEPS, RAW_NS),
    "n3.formation_T_hat": (RAW_STEPS, RAW_NS),
    "n3.tau_v_pre": (RAW_STEPS, RAW_NS),
    "n3.tau_v_post": (RAW_STEPS, RAW_NS),
    "n3.tau_v_reset": (RAW_STEPS, RAW_NS),
    "n3.chord_m": (RAW_STEPS, RAW_NS),
    "n3.dy_single_reference_m": (RAW_STEPS, RAW_NS),
    "n3.dy_single_current_m": (RAW_STEPS, RAW_NS),
    "n3.dy_solver_legacy_m": (RAW_STEPS, RAW_NS),
    "n3.qcdy_single_reference_N": (RAW_STEPS, RAW_NS),
    "n3.qcdy_single_current_N": (RAW_STEPS, RAW_NS),
    "n3.qcdy_physical_mirror_pair_N": (RAW_STEPS, RAW_NS),
    "n3.qcdy_solver_legacy_N": (RAW_STEPS, RAW_NS),
    "n3.ds_panel_force_physical_single_wing_N": (
        RAW_STEPS,
        RAW_NPAN,
        3,
    ),
    "n3.ds_panel_force_solver_legacy_N": (RAW_STEPS, RAW_NPAN, 3),
    "n3.ds_booked_solver_accumulator_N": (RAW_STEPS, 3),
    "n3.vortex_normal_booked_solver_accumulator_N": (RAW_STEPS, 3),
    "n3.booked_solver_accumulator_total_N": (RAW_STEPS, 3),
    "total_solver_accumulator_body_force_N": (RAW_STEPS, 3),
    "rig_drag_body_force_reported_pair_N": (RAW_STEPS, 3),
    "reported_pair_body_force_N": (RAW_STEPS, 3),
    "reported_pair_wind_lift_N": (RAW_STEPS,),
    "reported_pair_wind_thrust_N": (RAW_STEPS,),
    (
        "diagnostic.n1.leading_edge_suction_"
        "solver_accumulator_body_force_N"
    ): (RAW_STEPS, 3),
    (
        "diagnostic.n2.separation_panel_candidate_"
        "resultant_body_force_N"
    ): (RAW_STEPS, 3),
    "diagnostic.n2.candidate_minus_booked_body_force_N": (
        RAW_STEPS,
        3,
    ),
    "diagnostic.wind.n1_leading_edge_suction_L_N": (RAW_STEPS,),
    "diagnostic.wind.n1_leading_edge_suction_T_N": (RAW_STEPS,),
    "diagnostic.wind.n2_separation_booked_L_N": (RAW_STEPS,),
    "diagnostic.wind.n2_separation_booked_T_N": (RAW_STEPS,),
    "diagnostic.wind.n2_separation_panel_candidate_L_N": (RAW_STEPS,),
    "diagnostic.wind.n2_separation_panel_candidate_T_N": (RAW_STEPS,),
    "diagnostic.phase_solver_t_over_T": (RAW_STEPS,),
    "diagnostic.phase_paper_t_over_T": (RAW_STEPS,),
    "diagnostic.alignment_source_code": (RAW_STEPS,),
}

# Ordering is always (T, L).
MEAN_EXPERIMENT_ENDPOINTS = {
    "W1": ("18|a|6.0", "18|b|6.0", 2.6),
    "W2": ("18|a|10.0", "18|b|10.0", 2.6),
    "W3": ("18|a|10.0", "18|b|10.0", 1.4),
    "W4": ("18|a|8.0", "18|b|8.0", 2.6),
    "W5": ("19|a|0", "19|b|0", 2.6),
    "W6": ("19|a|15", "19|b|15", 2.6),
}

CONTRASTS = (
    ("C_U", "U", "W2", "W1"),
    ("C_f", "f", "W2", "W3"),
    ("C_A", "AoA", "W6", "W5"),
    ("C_tw1", "twist", "F16_22p5", "F16_0"),
    ("C_tw2", "twist", "F16_45", "F16_22p5"),
)


class ScoreContractError(RuntimeError):
    """The frozen witness bundle cannot support a scientific decision."""


@dataclass(frozen=True)
class ArtifactBlob:
    """One content-addressed file read exactly once after confinement checks."""

    path: Path
    data: bytes
    sha256: str


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_once(path: Path, *, confined_to: Path | None = None) -> ArtifactBlob:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ScoreContractError(f"artifact is not a regular file: {path}")
        if confined_to is not None:
            proc_fd = Path(f"/proc/self/fd/{descriptor}")
            try:
                opened_path = proc_fd.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise ScoreContractError(
                    f"cannot attest opened artifact path: {path}"
                ) from exc
            if not _within(opened_path, confined_to):
                raise ScoreContractError(
                    f"opened artifact escaped run directory: {path}"
                )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    return ArtifactBlob(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _confined_path(run_dir: Path, relative: Any) -> tuple[Path, Path]:
    if not isinstance(relative, str) or not relative:
        raise ScoreContractError("malformed artifact path")
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise ScoreContractError(f"unsafe artifact path: {relative}")
    try:
        root = run_dir.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ScoreContractError("run directory is missing") from exc
    if not root.is_dir():
        raise ScoreContractError("run path is not a directory")
    try:
        resolved = (root / requested).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ScoreContractError(f"artifact is missing: {relative}") from exc
    if not _within(resolved, root):
        raise ScoreContractError(
            f"artifact escaped run directory: {relative}"
        )
    return root, resolved


def _read_confined(
    run_dir: Path,
    relative: Any,
    *,
    expected_sha256: Any | None = None,
) -> ArtifactBlob:
    root, path = _confined_path(run_dir, relative)
    blob = _read_once(path, confined_to=root)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ScoreContractError("malformed artifact SHA256")
        if blob.sha256 != expected_sha256:
            raise ScoreContractError(f"artifact hash mismatch: {relative}")
    return blob


def _json_from_blob(blob: ArtifactBlob) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ScoreContractError(
            f"{blob.path}: non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            blob.data.decode("utf-8"),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreContractError(f"{blob.path}: malformed JSON") from exc
    if not isinstance(value, dict):
        raise ScoreContractError(f"{blob.path}: expected a JSON object")
    return value


def _npz_from_blob(blob: ArtifactBlob) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(blob.data), allow_pickle=False) as archive:
            names = list(archive.files)
            if len(names) != len(set(names)):
                raise ScoreContractError(
                    f"{blob.path}: duplicate NPZ member"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in names}
    except ScoreContractError:
        raise
    except Exception as exc:
        raise ScoreContractError(f"{blob.path}: malformed NPZ") from exc
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ScoreContractError(f"{blob.path}: object array is forbidden")
    return arrays


def _endpoint_value(
    curve: benchmark.MeasurementCurve,
    nominal_x: float,
) -> tuple[float, dict[str, Any]]:
    raw_x = np.asarray(curve.x, dtype=np.float64)
    values = np.asarray(curve.values_N, dtype=np.float64)
    if raw_x.ndim != 1 or values.shape != raw_x.shape:
        raise ScoreContractError(f"{curve.key}: malformed endpoint curve")
    index = int(np.argmin(np.abs(raw_x - nominal_x)))
    delta = float(raw_x[index] - nominal_x)
    tolerance = benchmark.DIGITIZATION_ENDPOINT_TOLERANCE[
        curve.abscissa
    ]
    if index not in (0, raw_x.size - 1) or abs(delta) > tolerance:
        raise ScoreContractError(
            f"{curve.key}: nominal {nominal_x:g} is not an authorized "
            "raw endpoint projection"
        )
    return float(values[index]), {
        "curve_key": curve.key,
        "source_curve_key": curve.source_key,
        "nominal_x": float(nominal_x),
        "raw_x": float(raw_x[index]),
        "endpoint_delta": delta,
        "endpoint_tolerance": float(tolerance),
        "force_interpolated": False,
        "identity_correction": curve.identity_correction,
    }


def _experimental_mean_vectors() -> tuple[
    dict[str, np.ndarray], dict[str, Any]
]:
    measurements = benchmark.load_measurements()
    validation = benchmark.validate_measurement_contract(measurements)
    if validation.get("passed") is not True:
        raise ScoreContractError(
            f"invalid Fig17/18/19 measurement contract: {validation}"
        )
    values: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for physical_id, (t_key, l_key, nominal_x) in (
        MEAN_EXPERIMENT_ENDPOINTS.items()
    ):
        thrust, thrust_meta = _endpoint_value(
            measurements[t_key], nominal_x
        )
        lift, lift_meta = _endpoint_value(
            measurements[l_key], nominal_x
        )
        values[physical_id] = np.asarray(
            [thrust, lift], dtype=np.float64
        )
        provenance[physical_id] = {
            "T": thrust_meta,
            "L": lift_meta,
            "canonical_panel_rule": (
                "W1-W4 Fig18(a,b); W5-W6 Fig19(a,b)"
            ),
        }
    return values, {
        "measurement_validation": validation,
        "endpoints": provenance,
    }


def _verify_artifact(
    run_dir: Path,
    identity: Mapping[str, Any],
) -> ArtifactBlob:
    if not isinstance(identity, Mapping):
        raise ScoreContractError("malformed artifact identity")
    relative = identity.get("path")
    expected_sha = identity.get("sha256")
    if not isinstance(relative, str) or not _is_sha256(expected_sha):
        raise ScoreContractError("malformed artifact identity")
    return _read_confined(
        run_dir,
        relative,
        expected_sha256=expected_sha,
    )


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _array_schema_fields(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, value in sorted(arrays.items())
    }


def _wind_vectors(
    body_force: np.ndarray,
    aoa_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(aoa_deg)
    force = np.asarray(body_force, dtype=np.float64)
    lift = force[..., 2] * math.cos(angle) - force[..., 0] * math.sin(
        angle
    )
    thrust = -(
        force[..., 0] * math.cos(angle)
        + force[..., 2] * math.sin(angle)
    )
    return thrust, lift


def _robust_mean(values: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median))) + 1.0e-12
    lower = median - 8.0 * 1.4826 * mad
    upper = median + 8.0 * 1.4826 * mad
    return float(np.mean(np.clip(value, lower, upper)))


def _validate_raw_schema(
    *,
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, Any],
    case: witness.CaseContract,
) -> None:
    actual_names = set(arrays)
    expected_names = set(RAW_EXPECTED_SHAPES)
    if actual_names != expected_names:
        raise ScoreContractError(
            f"{case.case_id}: raw field identity mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    malformed_shapes = {
        name: {
            "expected": expected,
            "actual": arrays[name].shape,
        }
        for name, expected in RAW_EXPECTED_SHAPES.items()
        if arrays[name].shape != expected
    }
    if malformed_shapes:
        raise ScoreContractError(
            f"{case.case_id}: raw shape mismatch: {malformed_shapes}"
        )
    non_finite = [
        name
        for name, value in arrays.items()
        if value.dtype.kind in "biufc" and not np.isfinite(value).all()
    ]
    if non_finite:
        raise ScoreContractError(
            f"{case.case_id}: non-finite raw fields: {non_finite}"
        )
    if schema.get("fields") != _array_schema_fields(arrays):
        raise ScoreContractError(f"{case.case_id}: raw schema fields mismatch")
    expected_schema_identity = {
        "schema": witness.RAW_SCHEMA_VERSION,
        "case_id": case.case_id,
        "stage": "G0_exploratory_quick_identity",
        "snapshot_phase": "post_force_pre_shed",
        "time_window": "last_cycle",
        "processing": "none",
        "figure16_alignment_status": "unresolved_external_kinematics",
    }
    mismatches = {
        key: {"expected": expected, "actual": schema.get(key)}
        for key, expected in expected_schema_identity.items()
        if schema.get(key) != expected
    }
    if mismatches:
        raise ScoreContractError(
            f"{case.case_id}: raw schema identity mismatch: {mismatches}"
        )


def _validate_raw_identities(
    arrays: Mapping[str, np.ndarray],
    case: witness.CaseContract,
) -> None:
    if not np.array_equal(
        arrays["last_cycle_step"],
        np.arange(RAW_STEPS),
    ):
        raise ScoreContractError(f"{case.case_id}: last-cycle index mismatch")
    expected_step = np.arange(RAW_STEPS, 2 * RAW_STEPS)
    if not np.array_equal(arrays["step"], expected_step):
        raise ScoreContractError(f"{case.case_id}: solver step mismatch")
    if not np.array_equal(
        arrays["cycle_index"],
        np.full(RAW_STEPS, witness.G0_N_CYCLE - 1),
    ):
        raise ScoreContractError(f"{case.case_id}: cycle index mismatch")
    if not np.array_equal(
        arrays["nc"], np.full(RAW_STEPS, witness.G0_NC)
    ) or not np.array_equal(
        arrays["ns"], np.full(RAW_STEPS, witness.G0_NS)
    ):
        raise ScoreContractError(f"{case.case_id}: raw grid mismatch")
    if not np.all(arrays["snapshot_phase"] == "post_force_pre_shed"):
        raise ScoreContractError(f"{case.case_id}: snapshot phase mismatch")
    if not np.all(
        arrays["diagnostic.alignment_source_code"]
        == "unresolved_external_kinematics"
    ):
        raise ScoreContractError(f"{case.case_id}: alignment code mismatch")

    dt = np.asarray(arrays["dt_s"], dtype=np.float64)
    time = np.asarray(arrays["time_s"], dtype=np.float64)
    if (
        np.any(dt <= 0.0)
        or not np.all(dt == dt[0])
        or np.any(np.diff(time) <= 0.0)
        or _maximum_abs(np.diff(time) - dt[:-1]) > 1.0e-12
    ):
        raise ScoreContractError(f"{case.case_id}: raw time identity failed")

    n1 = np.asarray(
        arrays["n1.booked_solver_accumulator_total_N"],
        dtype=np.float64,
    )
    n2 = np.asarray(
        arrays["n2.booked_solver_accumulator_total_N"],
        dtype=np.float64,
    )
    n3 = np.asarray(
        arrays["n3.booked_solver_accumulator_total_N"],
        dtype=np.float64,
    )
    raw_total = np.asarray(
        arrays["total_solver_accumulator_body_force_N"],
        dtype=np.float64,
    )
    reported_body = np.asarray(
        arrays["reported_pair_body_force_N"],
        dtype=np.float64,
    )
    rig = np.asarray(
        arrays["rig_drag_body_force_reported_pair_N"],
        dtype=np.float64,
    )
    if _maximum_abs(rig - rig[0]) > witness.LEDGER_TOLERANCE_N:
        raise ScoreContractError(f"{case.case_id}: rig force is not constant")
    errors = {
        "node_ledger": _maximum_abs(n1 + n2 + n3 - raw_total),
        "reported_pair": _maximum_abs(
            reported_body - (2.0 * raw_total + rig)
        ),
        "n1_panel": _maximum_abs(
            np.sum(arrays["n1.panel_force_body_N"], axis=1)[..., (0, 2)]
            - arrays["n1.bernoulli_booked_solver_accumulator_N"][
                ..., (0, 2)
            ]
        ),
        "n3_panel": _maximum_abs(
            np.sum(
                arrays["n3.ds_panel_force_solver_legacy_N"],
                axis=1,
            )[..., (0, 2)]
            - arrays["n3.ds_booked_solver_accumulator_N"][..., (0, 2)]
        ),
    }
    if max(errors.values()) > witness.LEDGER_TOLERANCE_N:
        raise ScoreContractError(
            f"{case.case_id}: independent raw ledger failed: {errors}"
        )

    thrust, lift = _wind_vectors(reported_body, case.aoa_deg)
    trace_error = max(
        _maximum_abs(
            thrust - arrays["reported_pair_wind_thrust_N"]
        ),
        _maximum_abs(lift - arrays["reported_pair_wind_lift_N"]),
    )
    if trace_error > witness.LEDGER_TOLERANCE_N:
        raise ScoreContractError(
            f"{case.case_id}: body/wind trace identity failed"
        )


def _raw_case_metrics(
    arrays: Mapping[str, np.ndarray],
    case: witness.CaseContract,
) -> dict[str, Any]:
    raw_half_body = np.asarray(
        arrays["total_solver_accumulator_body_force_N"],
        dtype=np.float64,
    )
    rig = np.asarray(
        arrays["rig_drag_body_force_reported_pair_N"],
        dtype=np.float64,
    )
    robust_body = np.asarray(
        [
            2.0 * _robust_mean(raw_half_body[:, 0]) + rig[0, 0],
            2.0 * _robust_mean(raw_half_body[:, 1]) + rig[0, 1],
            2.0 * _robust_mean(raw_half_body[:, 2]) + rig[0, 2],
        ],
        dtype=np.float64,
    )
    robust_t, robust_l = _wind_vectors(robust_body, case.aoa_deg)

    raw_t = np.asarray(
        arrays["reported_pair_wind_thrust_N"], dtype=np.float64
    )
    raw_l = np.asarray(
        arrays["reported_pair_wind_lift_N"], dtype=np.float64
    )

    n1_half = np.asarray(
        arrays[
            "diagnostic.n1.leading_edge_suction_"
            "solver_accumulator_body_force_N"
        ],
        dtype=np.float64,
    )
    q1_t, q1_l = _wind_vectors(2.0 * n1_half, case.aoa_deg)
    if max(
        _maximum_abs(
            q1_t
            - arrays["diagnostic.wind.n1_leading_edge_suction_T_N"]
        ),
        _maximum_abs(
            q1_l
            - arrays["diagnostic.wind.n1_leading_edge_suction_L_N"]
        ),
    ) > witness.LEDGER_TOLERANCE_N:
        raise ScoreContractError(
            f"{case.case_id}: N1 diagnostic wind identity failed"
        )

    q2_half = np.sum(
        np.asarray(
            arrays["n2.separation_panel_candidate_force_body_N"],
            dtype=np.float64,
        ),
        axis=1,
    )
    q2_t, q2_l = _wind_vectors(2.0 * q2_half, case.aoa_deg)
    if max(
        _maximum_abs(
            q2_t
            - arrays[
                "diagnostic.wind."
                "n2_separation_panel_candidate_T_N"
            ]
        ),
        _maximum_abs(
            q2_l
            - arrays[
                "diagnostic.wind."
                "n2_separation_panel_candidate_L_N"
            ]
        ),
    ) > witness.LEDGER_TOLERANCE_N:
        raise ScoreContractError(
            f"{case.case_id}: N2 diagnostic wind identity failed"
        )
    q2_vectors = np.column_stack((q2_t, q2_l))
    norms = np.linalg.norm(q2_vectors, axis=1)
    unit = np.zeros_like(q2_vectors)
    active = norms > NUMERIC_ZERO
    unit[active] = q2_vectors[active] / norms[active, None]
    if not np.isfinite(unit).all():
        raise ScoreContractError(f"{case.case_id}: non-finite Q2")

    return {
        "model_robust": np.asarray(
            [float(robust_t), float(robust_l)], dtype=np.float64
        ),
        "model_raw": np.asarray(
            [float(np.mean(raw_t)), float(np.mean(raw_l))],
            dtype=np.float64,
        ),
        "q1": -np.asarray(
            [float(np.mean(q1_t)), float(np.mean(q1_l))],
            dtype=np.float64,
        ),
        "q2": np.mean(unit, axis=0),
        "q2_active_step_fraction": float(np.mean(active)),
    }


def _load_case(
    run_dir: Path,
    campaign: Mapping[str, Any],
    case: witness.CaseContract,
    claim_gates: Mapping[str, Any],
) -> dict[str, Any]:
    record = campaign["cases"].get(case.case_id)
    if not isinstance(record, Mapping):
        raise ScoreContractError(f"missing case {case.case_id}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScoreContractError(f"{case.case_id}: malformed artifacts")
    if set(artifacts) != {"raw_npz", "schema_json", "evidence_json"}:
        raise ScoreContractError(
            f"{case.case_id}: artifact identity set mismatch"
        )
    raw_blob = _verify_artifact(run_dir, artifacts["raw_npz"])
    schema_blob = _verify_artifact(run_dir, artifacts["schema_json"])
    evidence_blob = _verify_artifact(run_dir, artifacts["evidence_json"])
    schema = _json_from_blob(schema_blob)
    evidence = _json_from_blob(evidence_blob)
    arrays = _npz_from_blob(raw_blob)
    _validate_raw_schema(arrays=arrays, schema=schema, case=case)
    _validate_raw_identities(arrays, case)

    if evidence.get("schema") != witness.SCHEMA_VERSION:
        raise ScoreContractError(f"{case.case_id}: evidence schema mismatch")
    if evidence.get("case_contract") != witness._jsonable(
        witness.asdict(case)
    ):
        raise ScoreContractError(f"{case.case_id}: case contract mismatch")
    if (
        evidence.get("stage") != "G0_exploratory_quick_identity"
        or evidence.get("production_grid_claim_allowed") is not False
        or evidence.get("observer_role") != "read_only"
        or evidence.get("aerodynamic_formula_modified") is not False
        or evidence.get("force_added_by_runner") is not False
    ):
        raise ScoreContractError(f"{case.case_id}: evidence role mismatch")
    if evidence.get("source_closure_sha256") != campaign.get(
        "source_closure_sha256"
    ):
        raise ScoreContractError(f"{case.case_id}: source identity mismatch")
    if evidence.get("raw_guard", {}).get("passed") is not True:
        raise ScoreContractError(f"{case.case_id}: raw guard failed")
    guards = evidence.get("claim_guards")
    witness._validate_claim_guards(guards)

    array_hash = witness._array_bundle_hash(arrays)
    expected_array_hash = evidence.get("raw_array_bundle_sha256")
    if (
        not _is_sha256(expected_array_hash)
        or array_hash != expected_array_hash
        or schema.get("array_bundle_sha256") != expected_array_hash
        or record.get("raw_array_bundle_sha256") != expected_array_hash
    ):
        raise ScoreContractError(f"{case.case_id}: raw array hash mismatch")

    manifest = evidence.get("claim_manifest")
    if not isinstance(manifest, Mapping):
        raise ScoreContractError(f"{case.case_id}: claim manifest missing")
    if manifest.get("guards") != guards:
        raise ScoreContractError(
            f"{case.case_id}: manifest/guard identity mismatch"
        )
    manifest_sha = evidence.get("claim_manifest_sha256")
    if (
        not _is_sha256(manifest_sha)
        or witness._canonical_hash(manifest) != manifest_sha
    ):
        raise ScoreContractError(
            f"{case.case_id}: claim manifest hash mismatch"
        )
    graph_identity = evidence.get("claim_graph_identity_sha256")
    if (
        not _is_sha256(graph_identity)
        or witness._claim_graph_identity_sha256(manifest)
        != graph_identity
        or record.get("claim_graph_identity_sha256") != graph_identity
        or campaign.get("common_claim_graph_identity_sha256")
        != graph_identity
    ):
        raise ScoreContractError(
            f"{case.case_id}: claim graph identity mismatch"
        )

    manifest_nodes = {
        item.get("id"): item
        for item in manifest.get("nodes", [])
        if isinstance(item, Mapping)
    }
    for claim_id in ("N1", "N2"):
        expected = claim_gates[claim_id]
        node = manifest_nodes.get(claim_id)
        if (
            not isinstance(node, Mapping)
            or node.get("state") != expected["state"]
            or node.get("freeze") is not expected["freeze"]
            or node.get("implementation") != expected["implementation"]
            or node.get("runtime_role") != expected["runtime_role"]
        ):
            raise ScoreContractError(
                f"{case.case_id}: {claim_id} runtime semantic mismatch"
            )

    raw_config = evidence.get("claim_raw_config")
    expected_config = {
        "closure": "v41",
        "nc": witness.G0_NC,
        "ns": witness.G0_NS,
        "n_cycle": witness.G0_N_CYCLE,
        "steps_per_cycle": witness.G0_STEPS_PER_CYCLE,
        "wake_rows": witness.G0_STEPS_PER_CYCLE,
        "U_m_s": case.U_m_s,
        "aoa_deg": case.aoa_deg,
        "freq_hz": case.frequency_Hz,
        "twist_amp_deg": case.solver_twist_amplitude_deg,
        "twist_phase_deg": case.twist_phase_deg,
    }
    if not isinstance(raw_config, Mapping) or any(
        raw_config.get(name) != expected
        for name, expected in expected_config.items()
    ):
        raise ScoreContractError(f"{case.case_id}: raw config mismatch")

    return _raw_case_metrics(arrays, case)


def _parse_frozen_fig16_source(data: bytes) -> dict[str, np.ndarray]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScoreContractError("Figure16 frozen source is not UTF-8") from exc
    try:
        thrust_start = text.index("Figure 16. (a)")
        lift_start = text.index("Figure 16. (b)", thrust_start + 10)
    except ValueError as exc:
        raise ScoreContractError(
            "Figure16 frozen source sections are missing"
        ) from exc

    output: dict[str, np.ndarray] = {}
    for kind, segment in (
        ("T", text[thrust_start:lift_start]),
        ("L", text[lift_start:]),
    ):
        matches = list(
            re.finditer(
                r"Twist amplitude\((\d+(?:\.\d+)?)°?\)",
                segment,
            )
        )
        for index, match in enumerate(matches):
            twist = float(match.group(1))
            block_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(segment)
            )
            pairs = re.findall(
                r"(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)",
                segment[match.end():block_end],
                flags=re.IGNORECASE,
            )
            if not pairs:
                raise ScoreContractError(
                    f"Figure16 frozen source has no {kind}/tw{twist:g}"
                )
            values = np.asarray(
                [
                    (float(time), float(force) * witness.G2N)
                    for time, force in pairs
                ],
                dtype=np.float64,
            )
            values = values[np.argsort(values[:, 0])]
            tag = f"{kind}_tw{twist:g}".replace(".", "p")
            output[f"{tag}_t_over_T"] = values[:, 0]
            output[f"{tag}_force_N"] = values[:, 1]
    return dict(sorted(output.items()))


def _validate_fig16_arrays(
    *,
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, Any],
    identity: Mapping[str, Any],
    expected: Mapping[str, np.ndarray],
) -> None:
    expected_names = {
        f"{kind}_tw{tag}_{suffix}"
        for kind in ("T", "L")
        for tag in ("0", "22p5", "45")
        for suffix in ("t_over_T", "force_N")
    }
    if set(arrays) != expected_names or set(expected) != expected_names:
        raise ScoreContractError("Figure16 field identity mismatch")
    if schema.get("fields") != _array_schema_fields(arrays):
        raise ScoreContractError("Figure16 schema fields mismatch")
    for kind in ("T", "L"):
        for tag in ("0", "22p5", "45"):
            time_name = f"{kind}_tw{tag}_t_over_T"
            force_name = f"{kind}_tw{tag}_force_N"
            time = np.asarray(arrays[time_name], dtype=np.float64)
            force = np.asarray(arrays[force_name], dtype=np.float64)
            if (
                time.ndim != 1
                or force.shape != time.shape
                or time.size == 0
                or not np.isfinite(time).all()
                or not np.isfinite(force).all()
                or np.any(np.diff(time) <= 0.0)
                or time[0] < -0.01
                or time[-1] > 1.01
            ):
                raise ScoreContractError(
                    f"Figure16 malformed trace: {kind}/tw{tag}"
                )
            if not np.array_equal(time, expected[time_name]) or not np.array_equal(
                force, expected[force_name]
            ):
                raise ScoreContractError(
                    f"Figure16 trace differs from frozen source: "
                    f"{kind}/tw{tag}"
                )

    array_hash = witness._array_bundle_hash(arrays)
    expected_hash = witness._array_bundle_hash(expected)
    if (
        array_hash != expected_hash
        or identity.get("array_bundle_sha256") != array_hash
        or schema.get("array_bundle_sha256") != array_hash
    ):
        raise ScoreContractError("Figure16 array bundle hash mismatch")
    schema_identity = {
        "schema": witness.RAW_SCHEMA_VERSION,
        "data_role": "published_filtered_gt",
        "source": str(witness.FIG16_SOURCE.relative_to(ROOT)),
        "source_sha256": FIG16_SOURCE_SHA256,
        "force_conversion": (
            "published grams-force * 9.81 / 1000 -> N"
        ),
        "published_processing": (
            "5th-order Butterworth, 8 Hz; instrument raw unavailable"
        ),
        "runner_processing": "none",
        "refilter_digitization": False,
        "alignment_status": "unresolved_external_kinematics",
        "model_force_cross_correlation_allowed": False,
    }
    if any(
        schema.get(name) != value
        for name, value in schema_identity.items()
    ):
        raise ScoreContractError("Figure16 schema identity mismatch")
    if (
        identity.get("data_role") != "published_filtered_gt"
        or identity.get("alignment_status")
        != "unresolved_external_kinematics"
    ):
        raise ScoreContractError("Figure16 campaign identity mismatch")


def _fig16_experiment(
    run_dir: Path,
    campaign: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    identity = campaign.get("figure16_experiment")
    if not isinstance(identity, Mapping):
        raise ScoreContractError("Figure16 experiment identity missing")
    if set(identity) != {
        "npz",
        "schema",
        "array_bundle_sha256",
        "data_role",
        "alignment_status",
    }:
        raise ScoreContractError("Figure16 experiment identity malformed")
    npz_blob = _verify_artifact(run_dir, identity["npz"])
    schema_blob = _verify_artifact(run_dir, identity["schema"])
    schema = _json_from_blob(schema_blob)
    arrays = _npz_from_blob(npz_blob)

    source_blob = _read_once(witness.FIG16_SOURCE.resolve(strict=True))
    if source_blob.sha256 != FIG16_SOURCE_SHA256:
        raise ScoreContractError("Figure16 frozen source hash mismatch")
    source_relative = str(witness.FIG16_SOURCE.relative_to(ROOT))
    closure_members = campaign.get("source_closure", {}).get("members", {})
    if (
        not isinstance(closure_members, Mapping)
        or closure_members.get(source_relative) != FIG16_SOURCE_SHA256
    ):
        raise ScoreContractError("Figure16 source closure identity mismatch")
    expected = _parse_frozen_fig16_source(source_blob.data)
    _validate_fig16_arrays(
        arrays=arrays,
        schema=schema,
        identity=identity,
        expected=expected,
    )

    values = {}
    for tag in ("0", "22p5", "45"):
        values[f"F16_{tag}"] = np.asarray(
            [
                float(np.mean(arrays[f"T_tw{tag}_force_N"])),
                float(np.mean(arrays[f"L_tw{tag}_force_N"])),
            ],
            dtype=np.float64,
        )
    return values, {
        "alignment_status": "unresolved_external_kinematics",
        "force_cross_correlation_used": False,
        "processing": "published-filtered trace arithmetic mean",
        "npz_sha256": npz_blob.sha256,
        "schema_sha256": schema_blob.sha256,
        "source_sha256": source_blob.sha256,
        "array_bundle_sha256": witness._array_bundle_hash(arrays),
    }


def _template_relation(
    residual: np.ndarray,
    template: np.ndarray,
) -> str:
    material = np.abs(residual) > TAU_CONTRAST_N
    if not np.any(material):
        return "not_material"
    signs_r = np.sign(residual[material])
    values_t = template[material]
    nonzero = np.abs(values_t) > NUMERIC_ZERO
    if not np.any(nonzero):
        return "inactive"
    products = signs_r[nonzero] * np.sign(values_t[nonzero])
    if np.any(products < 0.0):
        return "opposes"
    if np.any(products > 0.0):
        return "supports"
    return "inactive"


def _collinearity(contrast_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q1 = np.concatenate(
        [
            np.asarray(row["q1_TL_N"], dtype=float)
            for row in contrast_rows
        ]
    )
    q2 = np.concatenate(
        [
            np.asarray(row["q2_direction_support_TL"], dtype=float)
            for row in contrast_rows
        ]
    )
    norms = (float(np.linalg.norm(q1)), float(np.linalg.norm(q2)))
    if min(norms) <= NUMERIC_ZERO:
        return {
            "condition_number": math.inf,
            "limit": COLLINEAR_CONDITION_LIMIT,
            "collinear": True,
            "reason": "inactive_template_column",
        }
    matrix = np.column_stack((q1 / norms[0], q2 / norms[1]))
    condition = float(np.linalg.cond(matrix))
    return {
        "condition_number": condition,
        "limit": COLLINEAR_CONDITION_LIMIT,
        "collinear": bool(
            not np.isfinite(condition)
            or condition > COLLINEAR_CONDITION_LIMIT
        ),
        "reason": "normalized_template_matrix",
    }


def _classify(
    *,
    experiment: Mapping[str, np.ndarray],
    cases: Mapping[str, Mapping[str, np.ndarray]],
    model_key: str,
) -> dict[str, Any]:
    rows = []
    for name, family, high, low in CONTRASTS:
        residual_high = experiment[high] - cases[high][model_key]
        residual_low = experiment[low] - cases[low][model_key]
        residual = residual_high - residual_low
        q1 = cases[high]["q1"] - cases[low]["q1"]
        q2 = cases[high]["q2"] - cases[low]["q2"]
        relation_q1 = _template_relation(residual, q1)
        relation_q2 = _template_relation(residual, q2)
        rows.append(
            {
                "id": name,
                "family": family,
                "residual_TL_N": residual.tolist(),
                "material_axes": [
                    axis
                    for axis, value in zip(("T", "L"), residual)
                    if abs(value) > TAU_CONTRAST_N
                ],
                "q1_TL_N": q1.tolist(),
                "q2_direction_support_TL": q2.tolist(),
                "q1_relation": relation_q1,
                "q2_relation": relation_q2,
            }
        )

    material_rows = [row for row in rows if row["material_axes"]]
    if not material_rows:
        return {
            "status": "NO_DECISION_OFFSET_ONLY",
            "contrasts": rows,
            "collinearity": None,
        }

    collinearity = _collinearity(material_rows)
    if collinearity["collinear"]:
        return {
            "status": "NO_DECISION_COLLINEAR_OR_PROCESSING_SENSITIVE",
            "contrasts": rows,
            "collinearity": collinearity,
        }

    both = [
        row
        for row in material_rows
        if row["q1_relation"] == row["q2_relation"] == "supports"
    ]
    if both:
        return {
            "status": "NO_DECISION_MULTIPLE_EXPLANATIONS",
            "contrasts": rows,
            "collinearity": collinearity,
        }

    q1_families = {
        row["family"]
        for row in material_rows
        if row["q1_relation"] == "supports"
        and row["q2_relation"] != "supports"
    }
    q2_families = {
        row["family"]
        for row in material_rows
        if row["q2_relation"] == "supports"
        and row["q1_relation"] != "supports"
    }
    q1_opposed = any(
        row["q1_relation"] == "opposes" for row in material_rows
    )
    q2_opposed = any(
        row["q2_relation"] == "opposes" for row in material_rows
    )
    if len(q2_families) >= 2 and q1_opposed:
        status = "ACTIVE_N2_MISSING_PRESSURE_HYPOTHESIS"
    elif len(q1_families) >= 2 and q2_opposed:
        status = "N1_LEDGER_AUDIT_REQUIRED"
    elif q1_families and q2_families:
        status = "NO_DECISION_MIXED_SOURCE"
    elif not q1_families and not q2_families:
        status = "NO_DECISION_MISSING_OR_STATE_MEDIATED"
    else:
        status = "NO_DECISION_MIXED_SOURCE"
    return {
        "status": status,
        "contrasts": rows,
        "collinearity": collinearity,
        "unique_q1_families": sorted(q1_families),
        "unique_q2_families": sorted(q2_families),
        "q1_opposed_anywhere": q1_opposed,
        "q2_opposed_anywhere": q2_opposed,
    }


def _claim_yaml(
    path: Path,
    campaign: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ScoreContractError("PyYAML is required for claim gates") from exc
    blob = _read_once(path.resolve(strict=True))
    relative = str(path.relative_to(ROOT))
    members = campaign.get("source_closure", {}).get("members", {})
    if (
        not isinstance(members, Mapping)
        or members.get(relative) != blob.sha256
    ):
        raise ScoreContractError(f"{relative}: source closure mismatch")
    try:
        value = yaml.safe_load(blob.data.decode("utf-8"))
    except Exception as exc:
        raise ScoreContractError(f"{relative}: malformed claim YAML") from exc
    if not isinstance(value, dict):
        raise ScoreContractError(f"{relative}: malformed claim YAML")
    return value, blob.sha256


def _require_claim(
    root: Mapping[str, Any],
    claim_id: str,
    *,
    state: str | tuple[str, ...],
    freeze: bool,
) -> Mapping[str, Any]:
    claim = witness._find_claim_child([root], claim_id)
    states = (state,) if isinstance(state, str) else state
    if (
        not isinstance(claim, Mapping)
        or claim.get("state") not in states
        or claim.get("freeze") is not freeze
    ):
        raise ScoreContractError(
            f"{claim_id}: semantic claim gate failed"
        )
    return claim


def _semantic_claim_gates(
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    n1_root, n1_sha = _claim_yaml(N1_YAML, campaign)
    n2_root, n2_sha = _claim_yaml(N2_YAML, campaign)
    n1 = _require_claim(
        n1_root, "N1", state="validated", freeze=True
    )
    n2 = _require_claim(n2_root, "N2", state="partial", freeze=False)
    n22 = _require_claim(
        n2_root, "N2.2", state="falsified", freeze=False
    )
    n25 = _require_claim(n2_root, "N2.5", state="open", freeze=False)
    n26 = _require_claim(
        n2_root, "N2.6", state=("partial", "open"), freeze=False
    )

    if (
        n1.get("implementation")
        != "claim_runtime.components:UVLMComponent"
        or n1.get("runtime_role") != "physics"
        or "v41" not in n1.get("enabled_in", [])
    ):
        raise ScoreContractError("N1 runtime semantic gate failed")
    if (
        n2.get("implementation")
        != "claim_runtime.components:KirchhoffLBComponent"
        or n2.get("runtime_role") != "physics"
        or "v41" not in n2.get("enabled_in", [])
    ):
        raise ScoreContractError("N2 runtime semantic gate failed")
    if "implementation" in n25:
        raise ScoreContractError(
            "N2.5 unexpectedly acquired an executable implementation"
        )

    prereg_blob = _read_once(witness.PREREG.resolve(strict=True))
    prereg_relative = str(witness.PREREG.relative_to(ROOT))
    prereg_identity = campaign.get("preregistration")
    closure_members = campaign.get("source_closure", {}).get("members", {})
    if (
        prereg_blob.sha256 != PREREG_SHA256
        or not isinstance(prereg_identity, Mapping)
        or prereg_identity.get("path") != prereg_relative
        or prereg_identity.get("sha256") != prereg_blob.sha256
        or not isinstance(closure_members, Mapping)
        or closure_members.get(prereg_relative) != prereg_blob.sha256
    ):
        raise ScoreContractError("frozen preregistration identity mismatch")

    def runtime_gate(claim: Mapping[str, Any], yaml_sha: str) -> dict[str, Any]:
        return {
            "id": claim["id"],
            "state": claim["state"],
            "freeze": claim["freeze"],
            "implementation": claim.get("implementation"),
            "runtime_role": claim.get("runtime_role"),
            "yaml_sha256": yaml_sha,
            "passed": True,
        }

    return {
        "N1": runtime_gate(n1, n1_sha),
        "N2": runtime_gate(n2, n2_sha),
        "N2.2": {
            "id": n22["id"],
            "state": n22["state"],
            "freeze": n22["freeze"],
            "passed": True,
        },
        "N2.5": {
            "id": n25["id"],
            "state": n25["state"],
            "freeze": n25["freeze"],
            "candidate_implementation_authorized": False,
            "no_go_basis": "frozen_preregistration",
            "passed": True,
        },
        "N2.6": {
            "id": n26["id"],
            "state": n26["state"],
            "freeze": n26["freeze"],
            "movable": True,
            "passed": True,
        },
        "preregistration_sha256": prereg_blob.sha256,
    }


def _finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return np.isfinite(numeric) and (not nonnegative or numeric >= 0.0)


def _validate_numeric_runtime(runtime: Any) -> None:
    if not isinstance(runtime, Mapping):
        raise ScoreContractError("numeric runtime is missing")
    required = {
        "python",
        "platform",
        "numpy",
        "environment",
        "warp_runtime",
        "solver_config",
        "fluxvortex_module",
        "authoritative_import_roots",
        "fingerprint_sha256",
    }
    if set(runtime) != required:
        raise ScoreContractError("numeric runtime field identity mismatch")
    python = runtime["python"]
    if not isinstance(python, Mapping) or set(python) != {
        "version",
        "executable",
        "implementation",
    }:
        raise ScoreContractError("Python runtime identity mismatch")
    if any(
        not isinstance(python.get(name), str) or not python[name].strip()
        for name in ("version", "executable", "implementation")
    ) or not Path(python["executable"]).is_absolute():
        raise ScoreContractError("Python runtime identity is invalid")
    if (
        not isinstance(runtime["platform"], str)
        or not runtime["platform"].strip()
    ):
        raise ScoreContractError("platform runtime identity is invalid")

    environment = runtime["environment"]
    expected_environment = {
        "FLUXV_DTYPE",
        "FLUXV_DEVICE",
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    if not isinstance(environment, Mapping) or set(environment) != (
        expected_environment
    ):
        raise ScoreContractError("numeric thread environment mismatch")
    for name, value in environment.items():
        if value is None:
            continue
        if not isinstance(value, str):
            raise ScoreContractError(f"invalid runtime environment: {name}")
        if name.endswith("_NUM_THREADS") and (
            not value.isdigit() or int(value) <= 0
        ):
            raise ScoreContractError(
                f"invalid numeric thread setting: {name}"
            )

    def module_identity(
        value: Any,
        *,
        module_name: str | None = None,
        require_root: bool = False,
    ) -> None:
        required_fields = {"module", "path", "sha256"}
        if require_root:
            required_fields |= {"required_root", "relative_path"}
        if not isinstance(value, Mapping) or set(value) != required_fields:
            raise ScoreContractError("runtime module identity mismatch")
        if module_name is not None and value.get("module") != module_name:
            raise ScoreContractError("runtime module name mismatch")
        if (
            not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or not _is_sha256(value.get("sha256"))
        ):
            raise ScoreContractError("runtime module file identity invalid")
        if require_root:
            root = Path(value["required_root"])
            path = Path(value["path"])
            if (
                not root.is_absolute()
                or not isinstance(value.get("relative_path"), str)
                or not _within(path, root)
                or path != root / value["relative_path"]
            ):
                raise ScoreContractError(
                    "runtime module required-root identity invalid"
                )

    numpy_identity = runtime["numpy"]
    if not isinstance(numpy_identity, Mapping) or set(numpy_identity) != {
        "version",
        "module",
        "core_extension",
        "build_config",
        "build_sha256",
    }:
        raise ScoreContractError("NumPy runtime identity mismatch")
    if (
        not isinstance(numpy_identity["version"], str)
        or not numpy_identity["version"]
        or not isinstance(numpy_identity["build_config"], Mapping)
        or not _is_sha256(numpy_identity["build_sha256"])
    ):
        raise ScoreContractError("NumPy runtime identity invalid")
    module_identity(numpy_identity["module"], module_name="numpy")
    module_identity(numpy_identity["core_extension"])
    numpy_payload = dict(numpy_identity)
    numpy_sha = numpy_payload.pop("build_sha256")
    if witness._canonical_hash(numpy_payload) != numpy_sha:
        raise ScoreContractError("NumPy build fingerprint mismatch")

    warp = runtime["warp_runtime"]
    warp_fields = {
        "version",
        "module",
        "native_version",
        "clang_version",
        "llvm_version",
        "host_compiler_version",
        "cuda_available",
        "cuda_driver_version",
        "cuda_toolkit_version",
        "nvrtc_version",
        "cuda_supported_archs",
        "config",
        "device",
    }
    if not isinstance(warp, Mapping) or set(warp) != warp_fields:
        raise ScoreContractError("Warp runtime identity mismatch")
    if not isinstance(warp["version"], str) or not warp["version"]:
        raise ScoreContractError("Warp version is unresolved")
    module_identity(warp["module"], module_name="warp")
    if warp["cuda_available"] is not True:
        raise ScoreContractError("Warp CUDA runtime is unavailable")
    config = warp["config"]
    if not isinstance(config, Mapping) or set(config) != {
        "version",
        "_git_commit_hash",
        "cuda_arch_suffix",
        "llvm_cuda",
        "verify_cuda",
        "fast_math",
        "mode",
    }:
        raise ScoreContractError("Warp config identity mismatch")
    device = warp["device"]
    if not isinstance(device, Mapping) or set(device) != {
        "text",
        "alias",
        "name",
        "ordinal",
        "is_cuda",
        "arch",
        "compute_arch",
        "uuid",
        "pci_bus_id",
    }:
        raise ScoreContractError("Warp device identity mismatch")
    if (
        device.get("is_cuda") is not True
        or not isinstance(device.get("text"), str)
        or "cuda" not in device["text"].lower()
        or not isinstance(device.get("name"), str)
        or not device["name"]
        or isinstance(device.get("arch"), bool)
        or not isinstance(device.get("arch"), int)
        or device["arch"] <= 0
    ):
        raise ScoreContractError("Warp CUDA device identity invalid")

    solver_config = runtime["solver_config"]
    if not isinstance(solver_config, Mapping) or set(solver_config) != {
        "dtype_name",
        "dtype",
        "numpy_dtype",
        "device",
    }:
        raise ScoreContractError("solver config identity mismatch")
    if (
        solver_config.get("numpy_dtype") != "float64"
        or solver_config.get("dtype_name") != "float64"
        or not isinstance(solver_config.get("device"), str)
        or "cuda" not in solver_config["device"].lower()
    ):
        raise ScoreContractError("solver numeric config is not FP64 CUDA")

    module_identity(
        runtime["fluxvortex_module"],
        module_name="fluxvortex",
        require_root=True,
    )
    expected_roots = [str((ROOT / "src").resolve()), str(PLATFORM)]
    if runtime["authoritative_import_roots"] != expected_roots:
        raise ScoreContractError("authoritative import roots mismatch")
    fingerprint = runtime["fingerprint_sha256"]
    payload = dict(runtime)
    payload.pop("fingerprint_sha256")
    if (
        not _is_sha256(fingerprint)
        or witness._canonical_hash(payload) != fingerprint
    ):
        raise ScoreContractError("numeric runtime fingerprint mismatch")


def _preconditioner_case_contract() -> dict[str, Any]:
    return witness._jsonable(
        witness.asdict(
            witness.CaseContract(
                case_id="excluded_current_source_preconditioner",
                family="excluded_preconditioner",
                physical_id="P0",
                U_m_s=8.0,
                frequency_Hz=2.6,
                nominal_twist_deg=0.0,
                aoa_deg=5.0,
                twist_phase_deg=witness.PRODUCTION_PHASE_DEG,
                roles=("excluded_numeric_runtime_preconditioner",),
                coverage="none",
            )
        )
    )


def _validate_preconditioner(
    preconditioner: Any,
    *,
    graph_identity: str,
) -> None:
    if not isinstance(preconditioner, Mapping):
        raise ScoreContractError("session preconditioner is missing")
    if (
        preconditioner.get("excluded_from_scientific_metrics") is not True
        or preconditioner.get("case_contract")
        != _preconditioner_case_contract()
        or preconditioner.get("claim_graph_identity_sha256")
        != graph_identity
    ):
        raise ScoreContractError("preconditioner identity mismatch")
    if not isinstance(preconditioner.get("purpose"), str):
        raise ScoreContractError("preconditioner purpose is missing")
    for name in ("L_wind_N", "T_wind_N", "wall_s"):
        if not _finite_number(
            preconditioner.get(name),
            nonnegative=(name == "wall_s"),
        ):
            raise ScoreContractError(
                f"invalid preconditioner numeric field: {name}"
            )
    witness._validate_claim_guards(preconditioner.get("claim_guards"))
    resolved = preconditioner.get("resolved_call")
    expected = {
        "closure": "v41",
        "nc": witness.G0_NC,
        "ns": witness.G0_NS,
        "n_cycle": witness.G0_N_CYCLE,
        "steps_per_cycle": witness.G0_STEPS_PER_CYCLE,
        "wake_rows": witness.G0_STEPS_PER_CYCLE,
        "U": 8.0,
        "freq": 2.6,
        "twist_amp_deg": 0.0,
        "aoa_deg": 5.0,
        "twist_phase_deg": witness.PRODUCTION_PHASE_DEG,
    }
    if not isinstance(resolved, Mapping) or any(
        resolved.get(name) != value for name, value in expected.items()
    ):
        raise ScoreContractError("preconditioner solver call mismatch")


def _validate_campaign_structure(
    campaign: Mapping[str, Any],
    cases: Sequence[witness.CaseContract],
) -> None:
    expected_case_ids = {case.case_id for case in cases}
    records = campaign.get("cases")
    if not isinstance(records, Mapping) or set(records) != expected_case_ids:
        raise ScoreContractError("campaign case identity set mismatch")
    if campaign.get("completed_case_count") != len(expected_case_ids):
        raise ScoreContractError("campaign completed-case count mismatch")
    graph_identity = campaign.get("common_claim_graph_identity_sha256")
    if not _is_sha256(graph_identity):
        raise ScoreContractError("common claim graph identity is malformed")
    source_identity = campaign.get("source_closure_sha256")
    source_closure = campaign.get("source_closure")
    if (
        not _is_sha256(source_identity)
        or not isinstance(source_closure, Mapping)
        or source_closure.get("members_sha256") != source_identity
    ):
        raise ScoreContractError("source closure identity is malformed")

    runtime = campaign.get("numeric_runtime")
    _validate_numeric_runtime(runtime)
    sessions = campaign.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ScoreContractError("campaign sessions are missing")
    completed: list[str] = []
    reference_preconditioner: Mapping[str, Any] | None = None
    for session in sessions:
        if (
            not isinstance(session, Mapping)
            or session.get("numeric_runtime") != runtime
            or session.get("source_closure_sha256") != source_identity
        ):
            raise ScoreContractError("session runtime identity mismatch")
        _validate_preconditioner(
            session.get("preconditioner"),
            graph_identity=graph_identity,
        )
        preconditioner = session["preconditioner"]
        if reference_preconditioner is None:
            reference_preconditioner = preconditioner
        else:
            deltas = {
                name: abs(
                    float(preconditioner[name])
                    - float(reference_preconditioner[name])
                )
                for name in ("L_wind_N", "T_wind_N")
            }
            if any(value > TAU_F_N for value in deltas.values()):
                raise ScoreContractError(
                    "cross-session preconditioner force drift: "
                    f"{deltas}, tolerance={TAU_F_N}"
                )
        case_ids = session.get("completed_case_ids")
        if (
            not isinstance(case_ids, list)
            or any(not isinstance(item, str) for item in case_ids)
        ):
            raise ScoreContractError("session completed cases malformed")
        completed.extend(case_ids)
    if (
        len(completed) != len(set(completed))
        or set(completed) != expected_case_ids
    ):
        raise ScoreContractError("session completed-case identity mismatch")


def _validate_kinematic_identity_gate(
    campaign: Mapping[str, Any],
    cases: Sequence[witness.CaseContract],
) -> None:
    source_closure, n5_yaml_bytes = witness._source_closure_snapshot()
    if campaign.get("source_closure") != source_closure:
        raise ScoreContractError(
            "current scorer/source closure drift during N5.1c validation"
        )
    n5_relative = str(
        witness.N5_YAML.resolve().relative_to(witness.ROOT.resolve())
    )
    n5_sha256 = source_closure["members"].get(n5_relative)
    if not isinstance(n5_sha256, str):
        raise ScoreContractError(
            "N5 claim hash is missing from the source closure"
        )
    expected = witness._kinematic_identity_gate(
        cases,
        n5_yaml_bytes=n5_yaml_bytes,
        expected_n5_yaml_sha256=n5_sha256,
        source_closure_sha256=source_closure["members_sha256"],
    )
    if campaign.get("kinematic_identity_gate") != expected:
        raise ScoreContractError("N5.1c kinematic identity gate mismatch")


def _case_lookup(
    run_dir: Path,
    campaign: Mapping[str, Any],
    claim_gates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for case in witness._case_contracts():
        loaded = _load_case(run_dir, campaign, case, claim_gates)
        if case.family == "mean":
            suffix = "plus90" if case.twist_phase_deg > 0.0 else "minus90"
            output[f"{case.physical_id}_{suffix}"] = loaded
        elif case.family == "figure16_phase":
            output[case.physical_id] = loaded
    return output


def score(run_dir: Path) -> dict[str, Any]:
    cases_contract = witness._case_contracts()
    manifest_blob = _read_confined(run_dir, "run_manifest.json")
    campaign = _json_from_blob(manifest_blob)
    expected_contract = witness._campaign_contract(cases_contract)
    if (
        campaign.get("schema") != witness.SCHEMA_VERSION
        or campaign.get("status") != "complete"
        or campaign.get("contract") != expected_contract
    ):
        raise ScoreContractError("campaign is not a complete frozen G0 run")
    if campaign.get("source_closure") != witness._source_closure():
        raise ScoreContractError("current scorer/source closure drift")
    _validate_campaign_structure(campaign, cases_contract)
    _validate_kinematic_identity_gate(campaign, cases_contract)
    claim_gates = _semantic_claim_gates(campaign)
    cases = _case_lookup(run_dir, campaign, claim_gates)
    expected_case_keys = {
        *(f"{physical_id}_plus90" for physical_id in MEAN_EXPERIMENT_ENDPOINTS),
        *(f"{physical_id}_minus90" for physical_id in MEAN_EXPERIMENT_ENDPOINTS),
        "F16_0",
        "F16_22p5",
        "F16_45",
    }
    if set(cases) != expected_case_keys:
        raise ScoreContractError("case identity set mismatch")

    mean_experiment, mean_provenance = _experimental_mean_vectors()
    fig16_experiment, fig16_provenance = _fig16_experiment(run_dir, campaign)
    experiment = {**mean_experiment, **fig16_experiment}
    minus_cases = {
        physical_id: cases[f"{physical_id}_minus90"]
        for physical_id in MEAN_EXPERIMENT_ENDPOINTS
    }
    minus_cases.update(
        {
            key: cases[key]
            for key in ("F16_0", "F16_22p5", "F16_45")
        }
    )
    robust = _classify(
        experiment=experiment,
        cases=minus_cases,
        model_key="model_robust",
    )
    raw = _classify(
        experiment=experiment,
        cases=minus_cases,
        model_key="model_raw",
    )
    status = robust["status"]
    if raw["status"] != status:
        status = "NO_DECISION_COLLINEAR_OR_PROCESSING_SENSITIVE"

    plus_cases = {
        physical_id: cases[f"{physical_id}_plus90"]
        for physical_id in MEAN_EXPERIMENT_ENDPOINTS
    }
    plus_experiment = {
        physical_id: mean_experiment[physical_id]
        for physical_id in MEAN_EXPERIMENT_ENDPOINTS
    }
    plus_rows = []
    for name, family, high, low in CONTRASTS[:3]:
        residual = (
            plus_experiment[high] - plus_cases[high]["model_robust"]
            - plus_experiment[low] + plus_cases[low]["model_robust"]
        )
        plus_rows.append(
            {
                "id": name,
                "family": family,
                "residual_TL_N": residual.tolist(),
                "material": bool(
                    np.any(np.abs(residual) > TAU_CONTRAST_N)
                ),
            }
        )
    if (
        status == "NO_DECISION_OFFSET_ONLY"
        and any(row["material"] for row in plus_rows)
    ):
        status = "NO_DECISION_KINEMATIC_CONTAMINATION"

    return {
        "schema": SCHEMA,
        "status": status,
        "claim_state_modified": False,
        # This campaign can authorize only a new shadow preregistration.  It
        # never authorizes an implementation, including under ACTIVE_N2.
        "candidate_implementation_authorized": False,
        "n2p5_candidate_implementation_authorized": False,
        "next_n2p6_shadow_preregistration_authorized": (
            status == "ACTIVE_N2_MISSING_PRESSURE_HYPOTHESIS"
        ),
        "n1_remains_validated_frozen": (
            claim_gates["N1"]["state"] == "validated"
            and claim_gates["N1"]["freeze"] is True
        ),
        "n2p5_claim_state": claim_gates["N2.5"]["state"],
        "n2p5_no_go_basis": claim_gates["N2.5"]["no_go_basis"],
        "claim_semantic_gates": claim_gates,
        "thresholds": {
            "tau_F_N": TAU_F_N,
            "tau_contrast_N": TAU_CONTRAST_N,
            "numeric_zero": NUMERIC_ZERO,
            "collinear_condition_limit": COLLINEAR_CONDITION_LIMIT,
        },
        "minus90_robust": robust,
        "minus90_raw": raw,
        "plus90_kinematic_contamination_only": plus_rows,
        "experiment": {
            "mean": mean_provenance,
            "figure16": fig16_provenance,
        },
        "provenance": {
            "run_manifest": {
                "path": str(manifest_blob.path),
                "sha256": manifest_blob.sha256,
            },
            "preregistration": {
                "path": str(witness.PREREG.relative_to(ROOT)),
                "sha256": claim_gates["preregistration_sha256"],
            },
            "source_closure_sha256": campaign[
                "source_closure_sha256"
            ],
            "phase_identity_used_for_claim_attribution_deg": -90.0,
            "plus90_allowed_to_vote": False,
            "force_interpolation_used": False,
            "figure16_force_cross_correlation_used": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = score(args.run)
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "INVALID_EVIDENCE",
            "error": f"{type(exc).__name__}: {exc}",
            "claim_state_modified": False,
            "candidate_implementation_authorized": False,
            "n2p5_candidate_implementation_authorized": False,
            "next_n2p6_shadow_preregistration_authorized": False,
        }
        witness._write_json_atomic(args.output.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    witness._write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
