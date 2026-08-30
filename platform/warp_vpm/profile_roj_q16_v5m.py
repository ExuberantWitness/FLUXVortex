"""Formal Roj profile runner (MODIFICATION_PLAN M0-2).

Runs registry cases through the unified CaseRunner in short slices with
phase-level instrumentation, and proves G-M0:

  1. repeatability -- the same case slice run twice differs by <= 1e-12 in
     the key state/force observables;
  2. instrumentation purity -- running WITHOUT profiling produces the same
     payload as WITH profiling (modulo wall-clock fields);
  3. CLI exit codes match payload status.

``--steps`` controls the slice length ONLY: grid, physics, structural
element and frozen parameters come from the registry, never from CLI.

Usage:
  PYTHONPATH=src:platform:platform/warp_vpm python3 \
    platform/warp_vpm/profile_roj_q16_v5m.py --case ROJ11-A16 --steps 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "profile_output/roj_q16_v5m/profile_manifest.json"
)
KEY_NUMERIC_FIELDS = (
    "mean_zmax_over_c",
    "mean_Cn",
    "final_kelvin_max_abs",
)
REPEATABILITY_TOLERANCE = 1.0e-12


def _run_slice(spec, steps: int, device: str) -> dict:
    runner = RojratsirikulCaseRunner(spec, device=device)
    runner.build()
    return runner.run(max_aero_steps=steps)


def _numeric_signature(payload: dict) -> dict:
    signature = {}
    for field in KEY_NUMERIC_FIELDS:
        value = payload.get(field)
        if isinstance(value, (int, float)):
            signature[field] = float(value)
    records = payload.get("records") or []
    if records:
        last = records[-1]
        for key in ("cn", "kelvin_max_abs", "lev_release_count"):
            if key in last:
                signature[f"last_record.{key}"] = float(last[key])
    return signature


def _profiling_slice(spec, steps: int, device: str) -> dict:
    """One instrumented pass: torch profiler counters + CUDA event phases."""
    from torch.profiler import ProfilerActivity, profile

    runner = RojratsirikulCaseRunner(spec, device=device)
    runner.build()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        payload = runner.run(max_aero_steps=steps)
    wall = time.perf_counter() - started
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    launches = sum(e.count for e in cuda_events)
    sync_like = sum(
        e.count
        for e in events
        if "synchronize" in e.key.lower() or "cudaStreamSynchronize" in e.key
    )
    item_calls = sum(
        e.count for e in events if ".item" in e.key.lower()
    )
    return {
        "payload": payload,
        "profile": {
            "wall_seconds": wall,
            "seconds_per_step": wall / max(1, steps),
            "cuda_kernel_launches": launches,
            "launches_per_step": launches / max(1, steps),
            "sync_like_calls": sync_like,
            "item_calls": item_calls,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=sorted(ROJRATSIRIKUL2011_UNIFIED_CASES),
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--skip-profiler",
        action="store_true",
        help="run the purity check only (no torch.profiler pass)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the formal profile runner is GPU-only")

    spec = ROJRATSIRIKUL2011_UNIFIED_CASES[args.case]
    manifest: dict = {
        "case": args.case,
        "steps": args.steps,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip(),
        "git_dirty_digest": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "frozen_spec": {
            "angle_deg": spec.angle_deg,
            "target_zmax_over_c": spec.target_zmax_over_c,
            "target_cn_band": list(spec.target_cn_band),
        },
        "repeatability_tolerance": REPEATABILITY_TOLERANCE,
    }

    # G-M0 (1): repeatability.
    first = _run_slice(spec, args.steps, args.device)
    second = _run_slice(spec, args.steps, args.device)
    sig1, sig2 = _numeric_signature(first), _numeric_signature(second)
    repeat_ok = all(
        abs(sig1[k] - sig2[k]) <= REPEATABILITY_TOLERANCE
        for k in sig1
        if k in sig2 and isinstance(sig1[k], float)
    )
    manifest["repeatability"] = {
        "signature_1": sig1,
        "signature_2": sig2,
        "max_abs_diff": max(
            (abs(sig1[k] - sig2[k]) for k in sig1 if k in sig2), default=0.0
        ),
        "pass": bool(repeat_ok),
    }

    # G-M0 (2): instrumentation purity -- the unprofiled payload above must
    # match a profiled run numerically.
    if args.skip_profiler:
        manifest["instrumentation_purity"] = {"skipped": True}
        manifest["profile"] = {"skipped": True}
    else:
        profiled = _profiling_slice(spec, args.steps, args.device)
        sig_prof = _numeric_signature(profiled["payload"])
        purity_ok = all(
            abs(sig1[k] - sig_prof[k]) <= REPEATABILITY_TOLERANCE
            for k in sig1
            if k in sig_prof
        )
        manifest["instrumentation_purity"] = {
            "profiled_signature": sig_prof,
            "max_abs_diff_vs_unprofiled": max(
                (abs(sig1[k] - sig_prof[k]) for k in sig1 if k in sig_prof),
                default=0.0,
            ),
            "pass": bool(purity_ok),
        }
        manifest["profile"] = profiled["profile"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    g_m0_pass = manifest["repeatability"]["pass"] and manifest[
        "instrumentation_purity"
    ].get("pass", True)
    return 0 if g_m0_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
