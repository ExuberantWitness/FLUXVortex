"""Production-only FLUX-V5M CUDA iteration benchmark/profile harness."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "platform"),
    str(ROOT / "platform/warp_vpm"),
]

from bing_joint_ptera import JointConfig  # noqa: E402
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver  # noqa: E402
from test_ptera_gpu_active_lev import _problem  # noqa: E402
from test_ptera_gpu_only_backend import _small_problem  # noqa: E402


def _run(label: str, solver: CudaJointLEVTEVSolver, *, wake: bool) -> dict[str, object]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    solver.run(
        prescribed_wake=wake,
        calculate_streamlines=False,
        show_progress=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    force = np.asarray(
        [problem.airplanes[0].forces_W for problem in solver.steady_problems]
    )
    return {
        "label": label,
        "steps": solver.num_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": solver.num_steps / elapsed,
        "force_linf": float(np.max(np.abs(force))),
        "cuda_counters": dict(solver.cuda_counters),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX-V5M iteration benchmark requires CUDA")
    active_cfg = JointConfig(
        enable_lev=True,
        joint_tev=False,
        lev_start_step=1,
        particle_capacity=512,
        load_mode="bing",
    )
    joint_cfg = JointConfig(
        enable_lev=True,
        joint_tev=True,
        lev_start_step=1,
        particle_capacity=512,
        load_mode="bing",
    )
    records = [
        _run(
            "active_lev_compile_cold",
            CudaJointLEVTEVSolver(_problem(20), active_cfg, device="cuda:0"),
            wake=True,
        )
    ]
    profile_warm_only = os.environ.get("FLUXV_V5M_PROFILE_WARM_ONLY") == "1"
    if profile_warm_only:
        torch.cuda.cudart().cudaProfilerStart()
    records.extend(
        [
            _run(
                "active_lev_warm",
                CudaJointLEVTEVSolver(_problem(20), active_cfg, device="cuda:0"),
                wake=True,
            ),
            _run(
                "joint_lev_tev",
                CudaJointLEVTEVSolver(_problem(10), joint_cfg, device="cuda:0"),
                wake=True,
            ),
            _run(
                "free_wake",
                CudaJointLEVTEVSolver(
                    _small_problem(), JointConfig(enable_lev=False), device="cuda:0"
                ),
                wake=False,
            ),
        ]
    )
    if profile_warm_only:
        torch.cuda.cudart().cudaProfilerStop()
    payload = {
        "schema": "flux-v5m-production-gpu-iteration-benchmark-v1",
        "device": torch.cuda.get_device_name(0),
        "profile_warm_only": profile_warm_only,
        "records": records,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
