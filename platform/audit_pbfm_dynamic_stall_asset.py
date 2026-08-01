#!/usr/bin/env python3
"""Audit the public PBFM dynamic-stall asset against N2.6b4f3.

The audit is deliberately metadata-level: the official paper, repository
loader and public dataset manifest already identify the released arrays as a
surface-location-by-phase representation without a volume velocity field.
Downloading 2.4 GB cannot manufacture fields that the release does not
contain.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


PLATFORM = Path(__file__).resolve().parent
CASES = PLATFORM / "docs" / "diag" / "pbfm_dynamic_stall_asset_cases.yaml"
INVENTORY = (
    PLATFORM / "docs" / "diag" / "external_field_candidate_inventory.yaml"
)
OUTPUT = PLATFORM / "docs" / "diag" / "pbfm_dynamic_stall_asset_results.json"


def main() -> int:
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))

    coverage = cases["coverage"]
    required = tuple(inventory["required_joint_fields"])
    missing_schema = sorted(set(required) - set(coverage))
    unknown_schema = sorted(set(coverage) - set(required))
    missing_physical = [
        field for field in required if coverage.get(field) is not True
    ]
    released = cases["asset"]["released_representation"]
    expected = cases["expected_decision"]

    identity_checks = {
        "two_dimensional_rans": (
            cases["asset"]["generation"]["equations"]
            == "unsteady compressible two-dimensional RANS"
        ),
        "surface_time_not_volume_topology": (
            "surface-location-by-cycle-phase"
            in released["topology"]
        ),
        "no_volume_velocity_components": (
            released["explicit_volume_velocity_components"] == []
        ),
        "no_volume_coordinates": (
            released["explicit_volume_coordinates"] == []
        ),
        "loader_channel_partition": (
            released["raw_channels"] == 8
            and len(released["physical_channels"]) == 6
            and len(released["geometry_channels"]) == 2
        ),
        "public_split_manifested": (
            cases["asset"]["official_dataset"]["files"][
                "dynamic_stall_test.h5"
            ]["size_bytes"]
            == 268453888
            and cases["asset"]["official_dataset"]["files"][
                "dynamic_stall_train.h5"
            ]["size_bytes"]
            == 2147616768
        ),
    }
    eligible = bool(
        not missing_schema and not unknown_schema and not missing_physical
    )
    decision_checks = {
        "spatial_lev_no_go": expected["spatial_lev_evidence"] == "NO-GO",
        "flow_map_no_go": expected["material_flow_map"] == "NO-GO",
        "backbone_no_go": expected["material_backbone"] == "NO-GO",
        "surface_time_role_only": (
            expected["unified_panel_pressure_surface_time_test"]
            == "ELIGIBLE_WITH_DOMAIN_LABEL"
        ),
        "no_physical_promotion": expected["physical_promotion"] is False,
    }
    required_missing_for_spatial_state = {
        "time_coordinates",
        "wall_position",
        "wall_velocity",
        "normal_coordinate",
        "edge_position",
        "edge_velocity",
        "edge_convention",
        "velocity",
        "target_reynolds_110000_to_190000",
        "three_dimensional",
        "crossflow",
    }
    passed = bool(
        inventory["contract"] == cases["frozen_contract"]
        and inventory["evaluated_against_preexisting_frozen_contract"]
        and not missing_schema
        and not unknown_schema
        and not eligible
        and required_missing_for_spatial_state.issubset(missing_physical)
        and all(identity_checks.values())
        and all(decision_checks.values())
    )
    result = {
        "version": cases["version"],
        "asset_id": cases["asset"]["id"],
        "evaluated_against": cases["frozen_contract"],
        "missing_schema_keys": missing_schema,
        "unknown_schema_keys": unknown_schema,
        "missing_physical_coverage": missing_physical,
        "production_eligible": eligible,
        "identity_checks": identity_checks,
        "decision_checks": decision_checks,
        "allowed_roles": cases["allowed_roles"],
        "forbidden_roles": cases["forbidden_roles"],
        "claim_effect": {
            "N2.6b4f3a": "unchanged_frozen_six_asset_audit",
            "N2.6b4f3b": "remains_open",
            "N2.6c1b2b": "remains_open",
            "N2.6c1b3b": "remains_open",
            "N3.1i": (
                "surface-time test asset retained; no spatial-LEV promotion"
            ),
        },
        "decision": "NO-GO_SPATIAL_FIELD__KEEP_SURFACE_TIME_ROLE",
        "physical_promotion": False,
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
