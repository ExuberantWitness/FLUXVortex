"""Executable go/no-go for the preregistered R0 cycle-reduction claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import warp as wp  # noqa: E402

wp.init()

from _v2_robo import gpu_run_twist  # noqa: E402
from lb_sweep184 import ANCHOR, _resolved_call  # noqa: E402


PROBE = (6.0, 2.6, 27.5, 5.0)
FROZEN_ANCHOR = {"L": 6.752083501727169, "T": 0.8217739436030945}
ANCHOR_TOLERANCE_N = 0.15
EXPECTED_PROBE_Q_NUM = np.array(
    [-0.016532795048874012, 0.0, 0.1890927018091382],
    dtype=float,
)
EXPECTED_N1_HASH = (
    "sha256:f4c5d11c28ba5f4d71132c9c601544ab6b9b2728e404df27bac781fd8304dc2c"
)
EXPECTED_N4_HASH = (
    "sha256:1b35c6edc52bf23b04c8e8fc3b9bb5cbac39baa958c4e071b9d5abca61a4cc4f"
)
EXPECTED_R0_HASH = (
    "sha256:a0b0f13578b08db42db094919b0e38ce9420480fae0e0c4bcf39d6f956d52f1b"
)


def _valid_guard(guard: Any) -> bool:
    if not isinstance(guard, dict):
        return False
    error = guard.get("max_abs_error_N")
    tolerance = guard.get("tolerance_N")
    return (
        guard.get("passed") is True
        and isinstance(error, (int, float))
        and not isinstance(error, bool)
        and isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and np.isfinite(error)
        and np.isfinite(tolerance)
        and 0.0 <= float(error) <= float(tolerance)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: str) -> None:
    target = path.resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must remain inside repository: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {target.parent}")
    if target.exists():
        raise FileExistsError(f"refusing to replace versioned evidence: {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="write complete versioned JSON evidence without replacing a file",
    )
    args = parser.parse_args(argv)

    runs = []
    for condition in (ANCHOR, ANCHOR, PROBE):
        result = gpu_run_twist(**_resolved_call(gpu_run_twist, condition))
        actual = {
            "L": float(result["L_wind"]),
            "T": float(result["T_wind"]),
        }
        guards = result["claim_guards"]
        runs.append(
            {
                "condition": list(condition),
                "actual": actual,
                "guards": guards,
                "manifest": result["claim_manifest"],
                "contributions": result["claim_contributions"],
            }
        )

    probe = runs[-1]
    nodes = {
        node["id"]: node for node in probe["manifest"]["nodes"]
    }
    r0_force = np.asarray(
        probe["contributions"]["R0"][0]["body_force"], dtype=float
    )
    q_error = float(np.max(np.abs(r0_force - EXPECTED_PROBE_Q_NUM)))
    required_guards = (
        "force_ledger",
        "unclassified_force",
        "unclassified_physical_force",
        "cycle_reduction",
        "aero_output_invariance",
    )
    anchor_delta = [
        {
            name: abs(run["actual"][name] - FROZEN_ANCHOR[name])
            for name in ("L", "T")
        }
        for run in runs[:2]
    ]
    checks = {
        "all_incall_aero_outputs_bitwise_invariant": all(
            run["guards"]["aero_output_invariance"]["passed"] is True
            and not run["guards"]["aero_output_invariance"]["changed_fields"]
            for run in runs
        ),
        "both_anchors_within_frozen_tolerance": all(
            max(delta.values()) <= ANCHOR_TOLERANCE_N
            for delta in anchor_delta
        ),
        "all_required_guards_pass": all(
            _valid_guard(run["guards"].get(name))
            for run in runs
            for name in required_guards
        ),
        "probe_q_num_matches_preregistered": q_error <= 1.0e-12,
        "probe_physical_remainder_le_1e_9": (
            probe["guards"]["unclassified_physical_force"][
                "max_abs_error_N"
            ]
            <= 1.0e-9
        ),
        "n1_hash_unchanged": (
            nodes["N1"]["implementation_hash"] == EXPECTED_N1_HASH
        ),
        "n4_hash_unchanged": (
            nodes["N4"]["implementation_hash"] == EXPECTED_N4_HASH
        ),
        "r0_hash_independent": (
            nodes["R0"]["implementation_hash"] == EXPECTED_R0_HASH
        ),
        "r0_is_last": probe["manifest"]["topology"][-1] == "R0",
        "r0_channel_identity": (
            probe["contributions"]["R0"][0]["channel"]
            == "numerical_cycle_reduction"
        ),
    }
    report = {
        "claim": "R0 graph-level cycle reduction",
        "sequence": ["anchor", "anchor", "probe"],
        "frozen_anchor": FROZEN_ANCHOR,
        "anchor_tolerance_N": ANCHOR_TOLERANCE_N,
        "anchor_absolute_delta_N": anchor_delta,
        "probe_q_num_body_N": r0_force.tolist(),
        "expected_probe_q_num_body_N": EXPECTED_PROBE_Q_NUM.tolist(),
        "probe_q_num_max_abs_error_N": q_error,
        "verifier_source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PLATFORM / "_v2_robo.py",
                PLATFORM / "claim_runtime" / "components.py",
                PLATFORM / "claim_nodes" / "r0_cycle_reduction.yaml",
                PLATFORM / "lb_sweep184.py",
                Path(__file__).resolve(),
            )
        },
        "checks": checks,
        "go": all(checks.values()),
        "runs": runs,
    }
    payload = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    if args.output is not None:
        _write_json_once(args.output, payload)
    print(payload)
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
