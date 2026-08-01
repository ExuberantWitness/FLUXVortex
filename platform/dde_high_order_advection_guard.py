"""Run the preregistered 128/192-order N3.1j4a2 dynamic gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dde_vertex_star_advection_guard import run as run_vertex_star


HERE = Path(__file__).resolve().parent
SPEC_PATH = (
    HERE / "docs" / "diag" / "dde_high_order_advection_cases.yaml"
)
RESULT_PATH = (
    HERE / "docs" / "diag" / "dde_high_order_advection_results.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    payload = run_vertex_star(spec)
    payload["spec"] = str(SPEC_PATH.relative_to(HERE.parent))
    if args.write:
        RESULT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
