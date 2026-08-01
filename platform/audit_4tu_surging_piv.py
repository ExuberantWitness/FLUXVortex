#!/usr/bin/env python3
"""Audit the independently acquired 4TU NACA0021 surging PIV archive.

The archive is useful only if its released fields satisfy the already frozen
N2.6b4f3 near-wall data contract.  This script records archive facts and does
not infer missing coordinates, wall kinematics, or edge conventions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


PLATFORM = Path(__file__).resolve().parent
ARCHIVE = (
    PLATFORM/"data_external"/"4tu_surging_naca0021"/"data.zip"
)
EXPECTED_SHA256 = (
    "10ff83472157e28ea05bb168caa15713aac6e104211cc2275a4ae538e0be9156"
)
DATASET_DOI = "10.4121/0B01A240-559F-4F4B-802D-7B1D36AA0024"
PATTERN = re.compile(
    r"PIVVelocity_fre(?P<frequency>25|5)_Phase"
    r"(?P<phase>0|45|80|90|100|135|180|225|260|270|280|315)deg[.]mat$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_variables(payload: bytes) -> dict[str, np.ndarray]:
    loaded = loadmat(io.BytesIO(payload))
    return {
        key: np.asarray(value)
        for key, value in loaded.items()
        if not key.startswith("__")
    }


def main() -> int:
    if not ARCHIVE.is_file():
        raise SystemExit(
            f"external archive not found: {ARCHIVE}; "
            "download it into the ignored data cache first"
        )
    checksum = _sha256(ARCHIVE)

    dynamic_records = []
    static_record = {}
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            match = PATTERN.search(name)
            if match:
                variables = _public_variables(archive.read(name))
                velocity_fields = {
                    key: variables[key]
                    for key in ("U_final", "V_final", "W_final")
                    if key in variables
                }
                reference_shape = (
                    tuple(next(iter(velocity_fields.values())).shape)
                    if velocity_fields
                    else ()
                )
                finite_fraction = (
                    min(
                        float(np.count_nonzero(np.isfinite(value))/value.size)
                        for value in velocity_fields.values()
                    )
                    if velocity_fields
                    else 0.0
                )
                dynamic_records.append({
                    "name": name,
                    "frequency_token": match.group("frequency"),
                    "phase_deg": int(match.group("phase")),
                    "shape": reference_shape,
                    "variables": tuple(sorted(variables)),
                    "has_coordinates": any(
                        key.lower().startswith(("x_", "y_"))
                        for key in variables
                    ),
                    "velocity_finite_fraction": finite_fraction,
                })
            elif name.endswith("staticVelocity.mat"):
                variables = _public_variables(archive.read(name))
                static_record = {
                    "name": name,
                    "variables": tuple(sorted(variables)),
                    "shape": tuple(variables["U_static"].shape),
                    "has_coordinates": all(
                        key in variables
                        for key in ("x_mtx_3fov", "y_mtx_3fov")
                    ),
                }

    shape_counts = Counter(
        str(tuple(record["shape"]))
        for record in dynamic_records
    )
    phase_coverage = {
        frequency: sorted(
            record["phase_deg"]
            for record in dynamic_records
            if record["frequency_token"] == frequency
        )
        for frequency in ("25", "5")
    }
    all_dynamic_have_coordinates = all(
        record["has_coordinates"]
        for record in dynamic_records
    )
    all_dynamic_shapes_match = len(shape_counts) == 1
    minimum_finite_fraction = min(
        (
            record["velocity_finite_fraction"]
            for record in dynamic_records
        ),
        default=0.0,
    )
    outliers = [
        record["name"]
        for record in dynamic_records
        if tuple(record["shape"]) != (165, 326)
    ]
    missing_contract_fields = [
        "dynamic_spatial_coordinates",
        "time_resolved_snapshots_not_phase_average",
        "wall_position",
        "wall_velocity",
        "surface_normal_and_tangent_basis",
        "normal_rays",
        "edge_position_velocity_and_convention",
        "double_side_surface_topology",
        "target_reynolds_110000_to_190000",
        "three_dimensional_surface_crossflow",
        "transitional_near_wall_profiles",
    ]
    contract_eligible = bool(
        checksum == EXPECTED_SHA256
        and len(dynamic_records) == 24
        and all_dynamic_have_coordinates
        and all_dynamic_shapes_match
        and minimum_finite_fraction == 1.0
        and not missing_contract_fields
    )
    result = {
        "version": 1,
        "dataset": {
            "doi": DATASET_DOI,
            "title": (
                "PIV measurement flow field of the airfoil NACA0021 "
                "under surging motion"
            ),
            "source_kind": "experiment",
            "license": "CC BY 4.0",
            "reported_reynolds": 15000.0,
            "motion": "surging at 90 degree incidence",
            "sampling": "phase averaged",
        },
        "archive": {
            "relative_path": (
                "platform/data_external/4tu_surging_naca0021/data.zip"
            ),
            "git_ignored": True,
            "size_bytes": ARCHIVE.stat().st_size,
            "sha256": checksum,
            "expected_sha256": EXPECTED_SHA256,
            "checksum_valid": checksum == EXPECTED_SHA256,
        },
        "released_field_audit": {
            "dynamic_file_count": len(dynamic_records),
            "phase_coverage": phase_coverage,
            "dynamic_shape_counts": dict(sorted(shape_counts.items())),
            "dynamic_shape_outliers": outliers,
            "all_dynamic_shapes_match": all_dynamic_shapes_match,
            "all_dynamic_files_have_coordinates": (
                all_dynamic_have_coordinates
            ),
            "minimum_velocity_finite_fraction": minimum_finite_fraction,
            "static_file": static_record,
            "static_coordinates_transferable_to_dynamic": False,
            "reason_static_coordinates_not_transferable": (
                "static grid shape differs from dynamic fields and the "
                "release does not declare a coordinate mapping"
            ),
        },
        "n2_6b4f3_contract": {
            "eligible": contract_eligible,
            "missing": missing_contract_fields,
        },
        "allowed_use": [
            "independent archive integrity audit",
            "MAT variable/schema reader failure test",
            "visual phase-field inspection with released metadata",
        ],
        "forbidden_use": [
            "N2.6b4f four-plus-two sufficiency test",
            "near-wall material flow-map integration",
            "N2.6c1 separation-backbone validation",
            "V4.1 load calibration or promotion",
            "inventing dynamic coordinates from images or static grids",
        ],
        "passed": bool(
            checksum == EXPECTED_SHA256
            and len(dynamic_records) == 24
            and not contract_eligible
            and not all_dynamic_have_coordinates
            and not all_dynamic_shapes_match
        ),
    }
    output = PLATFORM/"docs"/"diag"/"field_asset_4tu_surging_audit.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
