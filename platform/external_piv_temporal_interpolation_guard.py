#!/usr/bin/env python3
"""Evaluate the pre-registered real-PIV temporal interpolation pre-gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


PLATFORM = Path(__file__).resolve().parent
CASES = (
    PLATFORM
    / "docs"
    / "diag"
    / "external_piv_temporal_interpolation_cases.yaml"
)
DATA = (
    PLATFORM / "data_external" / "otomo_pitching_piv_32sym_samples"
)
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "external_piv_temporal_interpolation_results.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _load(frame: int) -> tuple[Path, np.ndarray]:
    path = DATA / f"LER_{frame:05d}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.loadtxt(path, delimiter=",")
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{path}: expected [n,4], got {array.shape}")
    return path, array


def main() -> int:
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    source = cases["source"]
    center_frame = int(source["center_frame"])
    strides = tuple(int(value) for value in source["symmetric_strides"])
    frames = sorted(
        {center_frame}
        | {center_frame - stride for stride in strides}
        | {center_frame + stride for stride in strides}
    )
    loaded = {}
    file_records = {}
    for frame in frames:
        path, array = _load(frame)
        loaded[frame] = array
        file_records[str(frame)] = {
            "relative_path": str(path.relative_to(PLATFORM.parent)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "shape": tuple(array.shape),
            "all_finite": bool(np.all(np.isfinite(array))),
        }

    center = loaded[center_frame]
    coordinate_residuals = {
        str(frame): float(
            np.max(np.abs(loaded[frame][:, :2] - center[:, :2]))
        )
        for frame in frames
    }
    coordinates_pass = bool(
        max(coordinate_residuals.values()) <= 1.0e-12
    )
    finite_pass = all(record["all_finite"] for record in file_records.values())

    components = {}
    component_passes = {}
    for column, name in ((2, "u"), (3, "v")):
        stride_records = {}
        errors = []
        for stride in strides:
            left = loaded[center_frame - stride][:, column]
            right = loaded[center_frame + stride][:, column]
            truth = center[:, column]
            midpoint_error = _rms(truth - 0.5 * (left + right))
            persistence_error = 0.5 * (
                _rms(truth - left) + _rms(truth - right)
            )
            ratio = (
                midpoint_error / persistence_error
                if persistence_error > 0.0
                else float("inf")
            )
            stride_records[str(stride)] = {
                "half_window_seconds": (
                    stride * float(source["declared_time_step_seconds"])
                ),
                "midpoint_error": midpoint_error,
                "persistence_error": persistence_error,
                "midpoint_to_persistence_ratio": ratio,
            }
            errors.append(midpoint_error)
        fine_improves = bool(
            stride_records[str(strides[0])][
                "midpoint_to_persistence_ratio"
            ]
            < 1.0
        )
        errors_nondecreasing = bool(
            all(
                later >= earlier
                for earlier, later in zip(errors[:-1], errors[1:])
            )
        )
        components[name] = {
            "strides": stride_records,
            "fine_midpoint_improves_on_persistence": fine_improves,
            "midpoint_error_nondecreasing_with_stride": (
                errors_nondecreasing
            ),
        }
        component_passes[name] = bool(
            fine_improves and errors_nondecreasing
        )

    passed = bool(
        cases["frozen_before_adjacent_frame_extraction"]
        and len(loaded) == 7
        and finite_pass
        and coordinates_pass
        and all(component_passes.values())
    )
    result = {
        "version": 1,
        "source": source,
        "files": file_records,
        "coordinate_max_abs_residuals": coordinate_residuals,
        "coordinate_identity_passed": coordinates_pass,
        "finite_values_passed": finite_pass,
        "components": components,
        "decision": "GO" if passed else "NO-GO",
        "claim_effect": {
            "eligible_role": (
                cases["interpretation"]["go_role"] if passed else None
            ),
            "physical_promotion": False,
            "n2_6c1b2b_target_field": "remains_open",
        },
        "passed": passed,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
