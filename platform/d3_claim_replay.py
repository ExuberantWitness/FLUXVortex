"""Research-grade D3/N3.1 replay without changing the V4.1 force path.

The runner executes observer-off, observer-on, and a deterministic observer
repeat for each preregistered twist.  It saves the last-cycle raw snapshots
published through ``claim_raw_out`` and derives every reported metric from
those snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform as py_platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _v2_repro_nc12 import spc_of  # noqa: E402
from lb_sweep118 import BASE  # noqa: E402


NUMERIC_RESULT_FIELDS = (
    "L_wind",
    "T_wind",
    "Fx_body",
    "Fz_body",
    "L_inst",
    "T_inst",
    "Lh_bern",
    "Xh_bern",
    "Lh_stall",
    "Xh_stall",
    "Lh_ds",
    "Xh_ds",
)

IDENTITY_FILES = (
    "platform/_v2_robo.py",
    "platform/lb_dyn.py",
    "platform/lb_static.py",
    "platform/lb_sweep118.py",
    "platform/_v2_repro_nc12.py",
    "platform/_v2_robogeom.py",
    "platform/d3_claim_replay.py",
    "platform/claim_nodes/n1_uvlm.yaml",
    "platform/claim_nodes/n2_kirchhoff.yaml",
    "platform/claim_nodes/n3_ds_vortex.yaml",
    "platform/claim_nodes/n4_ct_consistency.yaml",
    "platform/claim_nodes/n5_twist_coupling.yaml",
    "platform/claim_nodes/n6_d_para.yaml",
    "platform/docs/diag/d4_d8_replay_contract.md",
    "platform/docs/repro_data.json",
    "platform/docs/s6_sweep_v41.json",
)

NUMERIC_THREAD_ENV_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return _sha256_bytes(payload)


def _write_json_atomic(path: Path, value: Any) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True)
    )
    os.replace(partial, path)


def _write_npz_atomic(
    path: Path, bundle: dict[str, np.ndarray]
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **bundle)
    os.replace(partial, path)


def _flatten_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    leaves: dict[str, list[Any]] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else key, item)
        else:
            leaves.setdefault(prefix, []).append(value)

    for record in records:
        visit("", record)
    return {key: np.asarray(values) for key, values in leaves.items()}


def _array_bundle_hash(bundle: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(bundle.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _maximum_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_maximum_delta(left[key], right[key]) for key in left),
            default=0.0,
        )
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.dtype.kind in "SUO" or right_array.dtype.kind in "SUO":
        return 0.0 if np.array_equal(left_array, right_array) else float("inf")
    return float(
        np.max(
            np.abs(left_array.astype(float) - right_array.astype(float)),
            initial=0.0,
        )
    )


def _wind_lift(force: np.ndarray, aoa_rad: float) -> np.ndarray:
    return force[..., 2] * np.cos(aoa_rad) - force[..., 0] * np.sin(aoa_rad)


def _wind_thrust(force: np.ndarray, aoa_rad: float) -> np.ndarray:
    return -(force[..., 0] * np.cos(aoa_rad)
             + force[..., 2] * np.sin(aoa_rad))


def _legacy_clip8_reported_pair(
    flat: dict[str, np.ndarray],
) -> np.ndarray:
    """Replay the production robust mean from the saved one-mesh trace."""

    solver = flat["total_solver_accumulator_body_force_N"]
    rig = flat["rig_drag_body_force_reported_pair_N"]

    def robust_pair_mean(values: np.ndarray) -> float:
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1.0e-12
        half_band = 8.0 * 1.4826 * mad
        return 2.0 * float(np.mean(np.clip(
            values, median - half_band, median + half_band
        )))

    return np.array(
        [
            robust_pair_mean(solver[:, 0]) + float(np.mean(rig[:, 0])),
            0.0,
            robust_pair_mean(solver[:, 2]) + float(np.mean(rig[:, 2])),
        ],
        dtype=float,
    )


def _validate_raw(
    records: list[dict[str, Any]],
    repeated: list[dict[str, Any]],
    result: dict[str, Any],
    steps_per_cycle: int,
    expected_dt_s: float,
) -> dict[str, Any]:
    flat = _flatten_records(records)
    flat_repeat = _flatten_records(repeated)
    finite = all(
        np.isfinite(value).all()
        for value in flat.values()
        if value.dtype.kind in "biufc"
    )
    raw_repeat_delta = _maximum_delta(flat, flat_repeat)
    expected_steps = np.arange(steps_per_cycle)
    step_identity = np.array_equal(
        flat["last_cycle_step"], expected_steps
    )
    cycle_identity = np.array_equal(
        flat["cycle_index"], flat["step"] // steps_per_cycle
    ) and np.unique(flat["cycle_index"]).size == 1
    dt = flat["dt_s"]
    time_step_error = float(
        np.max(
            np.abs(np.diff(flat["time_s"]) - dt[:-1]),
            initial=0.0,
        )
    )
    dt_contract_error = float(np.max(np.abs(dt - expected_dt_s)))
    y_edges = flat["y_ref_edge_m"]
    y_centers = flat["y_ref_center_m"]
    reference_geometry_error = max(
        float(np.max(np.abs(y_edges - y_edges[0]))),
        float(np.max(np.abs(
            y_centers
            - 0.5 * (y_edges[:, :-1] + y_edges[:, 1:])
        ))),
        float(np.max(np.abs(
            flat["n3.dy_single_reference_m"] - np.diff(y_edges, axis=1)
        ))),
    )

    n1_panel = flat["n1.panel_force_body_N"]
    n1_bernoulli = flat["n1.bernoulli_booked_solver_accumulator_N"]
    n3_panel = flat["n3.ds_panel_force_solver_legacy_N"]
    n3_ds_booked = flat["n3.ds_booked_solver_accumulator_N"]
    n1_xz_error = float(np.max(np.abs(
        np.sum(n1_panel, axis=1)[..., (0, 2)]
        - n1_bernoulli[..., (0, 2)]
    )))
    n3_xz_error = float(np.max(np.abs(
        np.sum(n3_panel, axis=1)[..., (0, 2)]
        - n3_ds_booked[..., (0, 2)]
    )))
    dy_scope_ratio = (
        flat["n3.dy_solver_legacy_m"]
        / flat["n3.dy_single_reference_m"]
    )
    qcdy_scope_error = float(np.max(np.abs(
        flat["n3.qcdy_solver_legacy_N"]
        - flat["n3.qcdy_single_reference_N"] * dy_scope_ratio
    )))
    nc = int(flat["nc"][0])
    ns = int(flat["ns"][0])
    n3_solver_scoped = n3_panel.reshape(
        len(records), nc, ns, 3
    )
    n3_physical_scoped = flat[
        "n3.ds_panel_force_physical_single_wing_N"
    ].reshape(len(records), nc, ns, 3)
    n3_force_scope_error = float(np.max(np.abs(
        n3_solver_scoped
        - n3_physical_scoped * dy_scope_ratio[:, None, :, None]
    )))

    reconstructed = (
        flat["n1.booked_solver_accumulator_total_N"]
        + flat["n2.booked_solver_accumulator_total_N"]
        + flat["n3.booked_solver_accumulator_total_N"]
    )
    total_error = float(np.max(np.abs(
        reconstructed - flat["total_solver_accumulator_body_force_N"]
    )))
    lift_error = float(np.max(np.abs(
        flat["reported_pair_wind_lift_N"]
        - np.asarray(result["L_inst"])[-steps_per_cycle:]
    )))
    thrust_error = float(np.max(np.abs(
        flat["reported_pair_wind_thrust_N"]
        - np.asarray(result["T_inst"])[-steps_per_cycle:]
    )))
    aoa_rad = np.radians(float(result["claim_raw_config"]["aoa_deg"]))
    clipped_body = _legacy_clip8_reported_pair(flat)
    clipped_lift = float(_wind_lift(clipped_body, aoa_rad))
    clipped_thrust = float(_wind_thrust(clipped_body, aoa_rad))
    clipped_result_error = max(
        abs(clipped_lift - float(result["L_wind"])),
        abs(clipped_thrust - float(result["T_wind"])),
    )
    passed = (
        len(records) == steps_per_cycle
        and finite
        and step_identity
        and cycle_identity
        and time_step_error <= 1.0e-14
        and dt_contract_error <= 1.0e-16
        and reference_geometry_error <= 1.0e-14
        and raw_repeat_delta == 0.0
        and max(n1_xz_error, n3_xz_error, total_error,
                lift_error, thrust_error, qcdy_scope_error,
                n3_force_scope_error, clipped_result_error) <= 2.0e-12
    )
    return {
        "passed": bool(passed),
        "n_steps": len(records),
        "finite": bool(finite),
        "step_identity": bool(step_identity),
        "cycle_identity": bool(cycle_identity),
        "time_step_error_s": time_step_error,
        "dt_contract_error_s": dt_contract_error,
        "expected_dt_s": expected_dt_s,
        "reference_geometry_error_m": reference_geometry_error,
        "deterministic_raw_repeat_max_delta": raw_repeat_delta,
        "raw_bundle_sha256": _array_bundle_hash(flat),
        "repeat_bundle_sha256": _array_bundle_hash(flat_repeat),
        "n1_panel_to_booked_xz_error_N": n1_xz_error,
        "n3_panel_to_booked_xz_error_N": n3_xz_error,
        "qcdy_scope_relation_error_N": qcdy_scope_error,
        "n3_force_scope_relation_error_N": n3_force_scope_error,
        "n1_plus_n2_plus_n3_total_error_N": total_error,
        "reported_lift_trace_error_N": lift_error,
        "reported_thrust_trace_error_N": thrust_error,
        "legacy_clip8_replay_to_result_error_N": clipped_result_error,
    }


def _span_metrics(
    signed_force_like: np.ndarray,
    dt_s: float,
    eta: np.ndarray,
) -> dict[str, Any]:
    branches = {
        "positive": np.maximum(signed_force_like, 0.0),
        "negative": np.maximum(-signed_force_like, 0.0),
        "absolute": np.abs(signed_force_like),
    }
    result: dict[str, Any] = {
        "signed_Ns": float(np.sum(signed_force_like) * dt_s),
    }
    for name, branch in branches.items():
        per_strip = dt_s * np.sum(branch, axis=0)
        total = float(np.sum(per_strip))
        result[f"{name}_Ns"] = total
        result[f"eta_centroid_{name}"] = (
            float(np.sum(eta * per_strip)) / total
            if total > 0.0 else None
        )
        result[f"outboard_share_{name}"] = (
            float(np.sum(per_strip[eta >= 0.5])) / total
            if total > 0.0 else None
        )
    return result


def _case_metrics(
    records: list[dict[str, Any]],
    twist_nominal_deg: float,
) -> dict[str, Any]:
    flat = _flatten_records(records)
    dt = float(flat["dt_s"][0])
    aoa_rad = np.radians(5.0)
    # These are the exact forces entering the legacy solver accumulator on
    # one geometric mesh.  The production report mirrors x/z by multiplying
    # them by two; this is deliberately not labelled a physical single wing.
    n1_reported = 2.0 * flat["n1.booked_solver_accumulator_total_N"]
    n2_reported = 2.0 * flat["n2.booked_solver_accumulator_total_N"]
    n3_reported = 2.0 * flat["n3.booked_solver_accumulator_total_N"]

    nc = int(flat["nc"][0])
    ns = int(flat["ns"][0])
    n3_solver_panel = flat["n3.ds_panel_force_solver_legacy_N"]
    n3_physical_panel = flat[
        "n3.ds_panel_force_physical_single_wing_N"
    ]
    n3_solver_strip = n3_solver_panel.reshape(
        len(records), nc, ns, 3
    ).sum(axis=1)
    n3_physical_strip = n3_physical_panel.reshape(
        len(records), nc, ns, 3
    ).sum(axis=1)
    n3_solver_reported_lift = _wind_lift(
        2.0 * n3_solver_strip, aoa_rad
    )
    n3_physical_pair_lift = _wind_lift(
        2.0 * n3_physical_strip, aoa_rad
    )
    tau_mask = flat["n3.tau_v_post"] > 4.24
    tau_denominator = float(np.sum(np.abs(n3_solver_reported_lift)))
    tau_fraction = (
        float(np.sum(np.abs(n3_solver_reported_lift) * tau_mask))
        / tau_denominator
        if tau_denominator > 0.0 else 0.0
    )

    eta = flat["eta_ref"][0]
    qcdy_physical_pair = flat["n3.qcdy_physical_mirror_pair_N"]
    qcdy_solver = flat["n3.qcdy_solver_legacy_N"]
    cv = flat["n2.CV"]
    cnv = flat["n2.CNv"]
    a0_excess = flat["n3.A0_excess_pre_cds"]
    n2_panel_resultant = np.sum(
        flat["n2.separation_panel_candidate_force_body_N"], axis=1
    )
    n2_allocation_gap = float(np.max(np.linalg.norm(
        n2_panel_resultant[..., (0, 2)]
        - flat["n2.separation_booked_solver_accumulator_N"][..., (0, 2)],
        axis=1,
    )))
    clipped_body = _legacy_clip8_reported_pair(flat)

    return {
        "twist_nominal_deg": float(twist_nominal_deg),
        "filter_branches": {
            "no_clip": {
                "L_wind_N": float(np.mean(
                    flat["reported_pair_wind_lift_N"]
                )),
                "T_wind_N": float(np.mean(
                    flat["reported_pair_wind_thrust_N"]
                )),
            },
            "legacy_clip8": {
                "L_wind_N": float(_wind_lift(clipped_body, aoa_rad)),
                "T_wind_N": float(_wind_thrust(clipped_body, aoa_rad)),
                "definition": "median +/- 8*1.4826*MAD per body axis",
                "source": "replayed from saved raw body-force trace",
            },
        },
        "reported_cycle_mean_claim_nodes_no_clip_L_wind_N": {
            "N1": float(np.mean(_wind_lift(n1_reported, aoa_rad))),
            "N2": float(np.mean(_wind_lift(n2_reported, aoa_rad))),
            "N3": float(np.mean(_wind_lift(n3_reported, aoa_rad))),
        },
        "LESP_exposure": {
            "physical_mirror_pair_Ns": float(np.sum(
                qcdy_physical_pair * a0_excess
            ) * dt),
            "solver_legacy_one_mesh_Ns": float(np.sum(
                qcdy_solver * a0_excess
            ) * dt),
        },
        "LB_CV_physical_mirror_pair": _span_metrics(
            qcdy_physical_pair * cv, dt, eta
        ),
        "LB_CNv_physical_mirror_pair": _span_metrics(
            qcdy_physical_pair * cnv, dt, eta
        ),
        "N3_DS_physical_mirror_pair_wind_lift": _span_metrics(
            n3_physical_pair_lift, dt, eta
        ),
        "N3_DS_solver_reported_pair_wind_lift": _span_metrics(
            n3_solver_reported_lift, dt, eta
        ),
        "production_N3_abs_lift_after_tau4p24_fraction": tau_fraction,
        "dy_solver_to_single_reference_ratio_mean": float(np.mean(
            flat["n3.dy_solver_legacy_m"]
            / flat["n3.dy_single_reference_m"]
        )),
        "N2_separation_candidate_to_booked_xz_gap_max_N":
            n2_allocation_gap,
    }


def _result_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(_result_field_deltas(left, right).values())


def _numeric_result_bundle(
    result: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(result[name])
        for name in NUMERIC_RESULT_FIELDS
    }


def _result_field_deltas(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    return {
        name: _maximum_delta(left[name], right[name])
        for name in NUMERIC_RESULT_FIELDS
    }


def _resolved_call(gpu_run_twist, call: dict[str, Any]) -> dict[str, Any]:
    bound = inspect.signature(gpu_run_twist).bind_partial(**call)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    resolved.pop("claim_raw_out", None)
    resolved.pop("frames_out", None)
    return _jsonable(resolved)


def _runtime_file_hashes() -> dict[str, str]:
    hashes = {}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}:
            hashes[str(path.relative_to(ROOT))] = _sha256_file(path)
    return dict(sorted(hashes.items()))


def _run_identity() -> dict[str, Any]:
    file_hashes = {}
    for relative in IDENTITY_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"identity input is missing: {relative}")
        file_hashes[relative] = _sha256_file(path)
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *IDENTITY_FILES]
    )
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_identity_diff_sha256": _sha256_bytes(tracked_diff),
        "file_sha256": file_hashes,
        "base_profile_sha256": _canonical_hash(BASE),
        "numeric_thread_environment": {
            key: os.environ.get(key)
            for key in NUMERIC_THREAD_ENV_KEYS
        },
    }


def _raw_schema(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "schema": "d3-claim-raw-v1",
        "snapshot_phase": "post_force_pre_shed",
        "force_scopes": {
            "physical_single_wing": "reference half-wing geometry",
            "solver_legacy": "current dy_lb=2*half_span/ns path",
            "reported_pair": "2*solver accumulator plus one rig drag",
        },
        "fields": {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for key, value in sorted(flat.items())
        },
    }


def _experiment_reference(twist: float) -> dict[str, float]:
    data_path = HERE / "docs" / "repro_data.json"
    data = json.loads(data_path.read_text())
    references = {}
    for key in ("17|b|2.6", "18|d|(8.0, 2.6)"):
        curve = data[key]
        references[key] = float(np.interp(
            twist, np.asarray(curve["x"]), np.asarray(curve["exp"])
        ))
    return references


def _frozen_v41_reference(twist: float) -> dict[str, float]:
    cache_path = HERE / "docs" / "s6_sweep_v41.json"
    cache = json.loads(cache_path.read_text())
    return cache[f"8_2.6_{twist:g}_5"]


def _production_call(twist: float, spc: int) -> dict[str, Any]:
    call = dict(BASE)
    call.update(
        U=8.0,
        aoa_deg=5.0,
        freq=2.6,
        twist_amp_deg=twist / 2.0,
        nc=12,
        ns=16,
        n_cycle=4,
        steps_per_cycle=spc,
        wake_rows=spc,
        closure="v41",
    )
    return call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(HERE / "docs" / "diag" / "d3_claim_replay_20260729"),
    )
    parser.add_argument(
        "--twists", nargs="+", type=float, default=(0.0, 15.0, 22.5, 45.0)
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    formal_twists = (0.0, 15.0, 22.5, 45.0)
    if tuple(args.twists) != formal_twists:
        parser.error(
            f"formal D3 replay requires --twists {' '.join(map(str, formal_twists))}"
        )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    spc = spc_of(8.0, 2.6)
    expected_dt = 1.0 / (2.6 * spc)
    run_identity = _run_identity()
    run_identity["numeric_runtime"] = {
        "python": sys.version,
        "platform": py_platform.platform(),
        "numpy": np.__version__,
        "warp": getattr(wp, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "device_name": getattr(wp.get_device(), "name", None),
    }
    run_identity_sha256 = _canonical_hash(run_identity)
    manifest_path = output / "run_manifest.json"
    summary_path = output / "summary.json"
    if (
        (manifest_path.exists() or summary_path.exists())
        and not args.force
    ):
        raise SystemExit(
            "REFUSE RESUME: D3 numerical provenance is process-local; "
            "use a new output directory or --force for a complete rerun"
        )
    summary: dict[str, Any] = {}
    cases = summary.setdefault("cases", {})
    manifest = {
        "schema": "d3-claim-replay-v1",
        "status": "running",
        "scientific_scope": (
            "D3/N3.1 diagnosis only; no aerodynamic formula or claim promotion"
        ),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "started_unix_s": started,
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "platform": py_platform.platform(),
        "numpy": np.__version__,
        "warp": getattr(wp, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "requested_twists_deg": list(args.twists),
        "spc": spc,
        "dt_s": expected_dt,
        "dtype": {
            "numpy": str(np.dtype(float)),
            "warp_configured": "from diff_uvlm_unsteady_gpu.DTYPE",
        },
        "processing_provenance": {
            "raw": "no clipping, filtering, or phase alignment",
            "legacy_clip8": "median +/- 8*1.4826*MAD per body axis",
            "shape_decision": (
                "raw and legacy_clip8 must agree; disagreement=INCONCLUSIVE"
            ),
        },
        "numeric_thread_environment": {
            key: os.environ.get(key)
            for key in NUMERIC_THREAD_ENV_KEYS
        },
    }
    _write_json_atomic(manifest_path, manifest)

    # Fresh processes have a reproducible first-call numerical transient.
    # An on->off->on order reversal proved that the delta follows call order,
    # not claim_raw_out.  This fixed, excluded cold run preconditions the
    # numerical runtime.  Its delta to the first formal tw0 run is archived;
    # stability is enforced only across the three subsequent formal runs.
    warmup_call = _production_call(0.0, spc)
    warmup_start = time.time()
    print(
        "[D3] numerical runtime preconditioner: fixed tw0, excluded",
        flush=True,
    )
    warmup_result = gpu_run_twist(**warmup_call)
    warmup_bundle = _numeric_result_bundle(warmup_result)
    manifest["runtime_preconditioner"] = {
        "purpose": (
            "remove fresh-process first-call numerical transient; "
            "not a physical sample"
        ),
        "excluded_from_scientific_metrics": True,
        "call": _resolved_call(gpu_run_twist, warmup_call),
        "numeric_result_bundle_sha256": _array_bundle_hash(warmup_bundle),
        "L_wind_N": float(warmup_result["L_wind"]),
        "T_wind_N": float(warmup_result["T_wind"]),
        "wall_s": time.time() - warmup_start,
    }
    _write_json_atomic(manifest_path, manifest)

    for twist in args.twists:
        case_id = f"U8_A5_f2p6_tw{twist:g}".replace(".", "p")
        raw_path = output / f"{case_id}.npz"
        call = _production_call(twist, spc)
        case_start = time.time()
        print(f"[D3] {case_id}: observer off", flush=True)
        observer_off = gpu_run_twist(**call)
        raw: list[dict[str, Any]] = []
        print(f"[D3] {case_id}: observer on", flush=True)
        observer_on = gpu_run_twist(**call, claim_raw_out=raw)
        repeated: list[dict[str, Any]] = []
        print(f"[D3] {case_id}: deterministic repeat", flush=True)
        repeat_result = gpu_run_twist(**call, claim_raw_out=repeated)

        observer_field_deltas = _result_field_deltas(
            observer_off, observer_on
        )
        repeat_field_deltas = _result_field_deltas(
            observer_on, repeat_result
        )
        formal_bundle_hashes = {
            "observer_off": _array_bundle_hash(
                _numeric_result_bundle(observer_off)
            ),
            "observer_on": _array_bundle_hash(
                _numeric_result_bundle(observer_on)
            ),
            "observer_on_repeat": _array_bundle_hash(
                _numeric_result_bundle(repeat_result)
            ),
        }
        formal_bitwise_passed = (
            len(set(formal_bundle_hashes.values())) == 1
        )
        observer_delta = max(observer_field_deltas.values())
        repeat_delta = max(repeat_field_deltas.values())
        validity = _validate_raw(
            raw, repeated, observer_on, spc, expected_dt
        )
        validity["observer_off_on_max_delta"] = observer_delta
        validity["observer_on_repeat_result_max_delta"] = repeat_delta
        validity["observer_off_on_field_max_delta"] = (
            observer_field_deltas
        )
        validity["observer_on_repeat_field_max_delta"] = (
            repeat_field_deltas
        )
        validity["formal_numeric_result_bundle_sha256"] = (
            formal_bundle_hashes
        )
        validity["formal_numeric_result_bitwise_passed"] = bool(
            formal_bitwise_passed
        )
        if twist == 0.0:
            warmup_field_deltas = _result_field_deltas(
                warmup_result, observer_off
            )
            warmup_delta = max(warmup_field_deltas.values())
            validity[
                "excluded_cold_to_first_formal_field_max_delta"
            ] = warmup_field_deltas
            validity[
                "excluded_cold_to_first_formal_max_delta"
            ] = warmup_delta
            validity["excluded_cold_numeric_result_bundle_sha256"] = (
                manifest["runtime_preconditioner"][
                    "numeric_result_bundle_sha256"
                ]
            )
        frozen = _frozen_v41_reference(twist)
        frozen_delta_l = float(observer_on["L_wind"]) - float(frozen["L"])
        frozen_delta_t = float(observer_on["T_wind"]) - float(frozen["T"])
        validity["frozen_v41_delta_L_N"] = frozen_delta_l
        validity["frozen_v41_delta_T_N"] = frozen_delta_t
        validity["frozen_v41_identity_tolerance_N"] = 0.15
        validity["frozen_v41_identity_passed"] = bool(
            max(abs(frozen_delta_l), abs(frozen_delta_t)) <= 0.15
        )
        validity["passed"] = bool(
            validity["passed"]
            and observer_delta == 0.0
            and repeat_delta == 0.0
            and formal_bitwise_passed
            and validity["frozen_v41_identity_passed"]
        )
        flat = _flatten_records(raw)
        _write_npz_atomic(raw_path, flat)
        schema_path = output / f"{case_id}.schema.json"
        _write_json_atomic(schema_path, _raw_schema(flat))
        call_identity = {
            "signature_bound": _resolved_call(gpu_run_twist, call),
            "closure_resolved": observer_on["claim_raw_config"],
        }
        case = {
            "case_id": case_id,
            "call": call_identity,
            "call_sha256": _canonical_hash(call_identity),
            "raw_path": str(raw_path.relative_to(output)),
            "raw_sha256": _sha256_file(raw_path),
            "raw_size_bytes": raw_path.stat().st_size,
            "schema_path": str(schema_path.relative_to(output)),
            "schema_sha256": _sha256_file(schema_path),
            "validity": validity,
            "metrics": _case_metrics(raw, twist),
            "frozen_v41_reference": frozen,
            "experiment_reference_L_N": _experiment_reference(twist),
            "wall_s": time.time() - case_start,
        }
        cases[case_id] = case
        summary["all_valid"] = bool(
            all(item["validity"]["passed"] for item in cases.values())
        )
        _write_json_atomic(summary_path, summary)
        manifest.update({
            "status": "checkpoint",
            "checkpoint_case_ids": sorted(cases),
            "summary_sha256": _sha256_file(summary_path),
        })
        _write_json_atomic(manifest_path, manifest)
        print(
            f"[D3] {case_id}: valid={validity['passed']} "
            f"L={case['metrics']['filter_branches']['no_clip']['L_wind_N']:+.3f} "
            f"T={case['metrics']['filter_branches']['no_clip']['T_wind_N']:+.3f} "
            f"wall={case['wall_s']:.1f}s",
            flush=True,
        )

    ordered = sorted(
        cases.values(), key=lambda item: item["metrics"]["twist_nominal_deg"]
    )
    if {item["metrics"]["twist_nominal_deg"] for item in ordered}.issuperset(
        {0.0, 15.0, 22.5, 45.0}
    ):
        by_twist = {
            item["metrics"]["twist_nominal_deg"]: item["metrics"]
            for item in ordered
        }
        def shape_pass(branch: str) -> bool:
            lift = {
                twist: by_twist[twist]["filter_branches"][branch]["L_wind_N"]
                for twist in (0.0, 15.0, 22.5, 45.0)
            }
            return bool(
                lift[15.0] > lift[22.5] > lift[45.0]
                and lift[15.0] > lift[0.0]
            )

        raw_shape_pass = shape_pass("no_clip")
        clipped_shape_pass = shape_pass("legacy_clip8")
        if raw_shape_pass != clipped_shape_pass:
            shape_status = "INCONCLUSIVE"
        elif raw_shape_pass:
            shape_status = "GO"
        else:
            shape_status = "NO_GO"
        summary["d3_shape_gate"] = {
            "status": shape_status,
            "no_clip_passed": raw_shape_pass,
            "legacy_clip8_passed": clipped_shape_pass,
            "criterion": "L15 > L22.5 > L45 and L15 > L0",
        }
        _write_json_atomic(summary_path, summary)

    summary["all_valid"] = bool(
        cases
        and all(item["validity"]["passed"] for item in cases.values())
    )
    _write_json_atomic(summary_path, summary)
    manifest.update({
        "status": "completed" if summary.get("all_valid") else "invalid",
        "finished_unix_s": time.time(),
        "wall_s": time.time() - started,
        "runtime_file_sha256": _runtime_file_hashes(),
        "summary_sha256": _sha256_file(summary_path),
        "ground_truth_sha256": {
            "platform/docs/repro_data.json": _sha256_file(
                HERE / "docs" / "repro_data.json"
            ),
            "platform/docs/s6_sweep_v41.json": _sha256_file(
                HERE / "docs" / "s6_sweep_v41.json"
            ),
        },
    })
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[D3] complete status={manifest['status']} "
        f"wall={manifest['wall_s']:.1f}s -> {output}",
        flush=True,
    )
    if manifest["status"] != "completed":
        raise SystemExit("INVALID: one or more replay validity gates failed")


if __name__ == "__main__":
    main()
