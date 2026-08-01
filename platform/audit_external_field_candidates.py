#!/usr/bin/env python3
"""Audit public field candidates against the frozen N2.6b4f3 contract.

This is an acquisition audit, not a decoder or a load-model calibration.
Coverage must be joint within one independent asset.  The script also checks
the locally sampled Zenodo and Edinburgh payloads when they are available.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import yaml


PLATFORM = Path(__file__).resolve().parent
INVENTORY = (
    PLATFORM / "docs" / "diag" / "external_field_candidate_inventory.yaml"
)
OUTPUT = (
    PLATFORM / "docs" / "diag" / "external_field_candidate_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_zenodo_samples() -> dict:
    root = PLATFORM / "data_external" / "zenodo_feather_piv"
    expected = {
        "Avg.mat": {
            "sha256": (
                "58003c0bb287176e50f93dac3e35ed33728513a4c95fb669"
                "228951cc62c55d09"
            ),
            "field_shape": (147, 288),
            "variables": {"P", "omega", "u", "v", "x", "y"},
        },
        "Tau_0.50.mat": {
            "sha256": (
                "bcf78560527c1b102ef2b2629b39eed6dad08cb505cf358"
                "2f81c153de553fba7"
            ),
            "field_shape": (147, 288),
            "variables": {
                "I_image",
                "omega",
                "u",
                "v",
                "x",
                "x_image",
                "y",
                "y_image",
            },
        },
    }
    records = {}
    for name, requirement in expected.items():
        path = root / name
        if not path.is_file():
            records[name] = {"available": False}
            continue
        payload = {
            key: np.asarray(value)
            for key, value in loadmat(path).items()
            if not key.startswith("__")
        }
        field_shapes = {
            key: tuple(payload[key].shape)
            for key in ("x", "y", "u", "v", "omega")
        }
        records[name] = {
            "available": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "variables": sorted(payload),
            "field_shapes": field_shapes,
            "all_finite": all(
                bool(np.all(np.isfinite(value)))
                for value in payload.values()
            ),
            "passed": bool(
                _sha256(path) == requirement["sha256"]
                and set(payload) == requirement["variables"]
                and all(
                    shape == requirement["field_shape"]
                    for shape in field_shapes.values()
                )
            ),
        }
    return records


def _audit_otomo_samples() -> dict:
    root = (
        PLATFORM / "data_external" / "otomo_pitching_piv_32sym_samples"
    )
    expected_hashes = {
        "LER_00001.csv": (
            "adb3f86bdb48db705215b9831f025652376a9dc5cc00a66"
            "e8c561f8f5d4be3b2"
        ),
        "LER_00833.csv": (
            "a69c1ecce724c5e62333e9c3f121a7cb4763ef807ded08ea"
            "93f2fd8319f56bbb"
        ),
        "LER_01666.csv": (
            "1159456f1cd4edbb85decee910e84a3c441f76be7eb6a99b"
            "8d3a82b08e6730fe"
        ),
    }
    records = {}
    for name, expected_hash in expected_hashes.items():
        path = root / name
        if not path.is_file():
            records[name] = {"available": False}
            continue
        array = np.loadtxt(path, delimiter=",")
        nx = int(np.unique(array[:, 0]).size)
        ny = int(np.unique(array[:, 1]).size)
        records[name] = {
            "available": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "shape": tuple(array.shape),
            "grid_shape": (ny, nx),
            "all_finite": bool(np.all(np.isfinite(array))),
            "coordinate_range": {
                "x": [float(np.min(array[:, 0])), float(np.max(array[:, 0]))],
                "y": [float(np.min(array[:, 1])), float(np.max(array[:, 1]))],
            },
            "passed": bool(
                _sha256(path) == expected_hash
                and array.shape == (15876, 4)
                and (ny, nx) == (126, 126)
                and np.all(np.isfinite(array))
            ),
        }
    return records


def main() -> int:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    required = tuple(inventory["required_joint_fields"])
    assets = {}
    for asset_id, asset in inventory["assets"].items():
        coverage = asset["coverage"]
        missing_schema_keys = sorted(set(required) - set(coverage))
        unknown_schema_keys = sorted(set(coverage) - set(required))
        missing_physical_coverage = [
            name for name in required if coverage.get(name) is not True
        ]
        eligible = bool(
            not missing_schema_keys
            and not unknown_schema_keys
            and not missing_physical_coverage
        )
        assets[asset_id] = {
            "persistent_id": asset["persistent_id"],
            "missing_schema_keys": missing_schema_keys,
            "unknown_schema_keys": unknown_schema_keys,
            "missing_physical_coverage": missing_physical_coverage,
            "production_eligible": eligible,
            "allowed_roles": asset["allowed_roles"],
        }

    locally_sampled_payloads = {
        "zenodo_feather_naca2414_piv": _audit_zenodo_samples(),
        "edinburgh_otomo_pitching_piv": _audit_otomo_samples(),
    }
    local_checks_passed = all(
        record.get("passed", False)
        for group in locally_sampled_payloads.values()
        for record in group.values()
    )
    eligible_assets = sorted(
        asset_id
        for asset_id, record in assets.items()
        if record["production_eligible"]
    )
    result = {
        "version": 1,
        "contract": inventory["contract"],
        "evaluated_against_preexisting_frozen_contract": inventory[
            "evaluated_against_preexisting_frozen_contract"
        ],
        "coverage_is_joint_not_union": True,
        "assets": assets,
        "locally_sampled_payloads": locally_sampled_payloads,
        "production_eligible_assets": eligible_assets,
        "request_only_candidates": inventory.get(
            "request_only_candidates", {}
        ),
        "decision": "NO-GO" if not eligible_assets else "GO",
        "claim_effect": {
            "n2_6b4f3_contract": "unchanged_validated",
            "n2_6c1b2b_target_field": "remains_open",
            "n2_6c1b3b_real_backbone": "remains_open",
            "physical_promotion": False,
        },
        "passed": bool(
            inventory["evaluated_against_preexisting_frozen_contract"]
            and inventory["joint_decision"][
                "can_union_assets_for_eligibility"
            ]
            is False
            and not eligible_assets
            and local_checks_passed
            and all(
                not record["missing_schema_keys"]
                and not record["unknown_schema_keys"]
                for record in assets.values()
            )
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
