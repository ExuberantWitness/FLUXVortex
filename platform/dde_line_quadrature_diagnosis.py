"""No-force order spectrum for the nonzero-dihedral DDE edge integral."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from claim_runtime.material_sheet_advection import (
    self_induced_vertex_star_normal_velocity,
)
from dde_cylindrical_advection_guard import cylindrical_sheet


HERE = Path(__file__).resolve().parent
SPEC_PATH = (
    HERE / "docs" / "diag" / "dde_nonzero_dihedral_advection_cases.yaml"
)
RESULT_PATH = (
    HERE / "docs" / "diag" / "dde_line_quadrature_diagnosis.json"
)
ORDERS = (32, 48, 64, 96, 128, 192)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    surface = cylindrical_sheet(spec["cylinder"])
    values = []
    reports = []
    for order in ORDERS:
        projection = self_induced_vertex_star_normal_velocity(
            surface,
            quadrature_order=order,
        )
        values.append(projection.vertex_velocity)
        reports.append(
            {
                "order": order,
                "condition_number": projection.report.condition_number,
                "vertex_speed_max": float(
                    np.max(np.abs(projection.vertex_normal_speed))
                ),
            }
        )
    reference_scale = max(
        float(np.max(np.linalg.norm(values[-1], axis=1))),
        np.finfo(float).eps,
    )
    changes = []
    for index in range(len(values) - 1):
        change = float(
            np.max(
                np.linalg.norm(
                    values[index + 1] - values[index],
                    axis=1,
                )
            )
        )
        changes.append(
            {
                "orders": [ORDERS[index], ORDERS[index + 1]],
                "max_abs_change": change,
                "change_over_order192_speed": change / reference_scale,
            }
        )
    payload = {
        "role": "diagnosis only; no promotion or force",
        "source_spec": str(SPEC_PATH.relative_to(HERE.parent)),
        "orders": list(ORDERS),
        "reports": reports,
        "successive_changes": changes,
        "order192_vertex_velocity": values[-1].tolist(),
    }
    if args.write:
        RESULT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
