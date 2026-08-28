"""Recover the final A16 payload from the completed partial after the
JSON-serialization crash (slice object in window_selection).

The physics run itself completed all 2100 steps; the partial JSON carries
every record and the z_history NPZ is complete.  This rebuilds the scoring
through the (fixed) _finalize path without any recomputation of physics and
writes the final JSON next to the partial.  The NPZ is NOT rewritten (the
partial's NPZ already covers all steps).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fluxvortex.cases.rojratsirikul2011 import ROJ_A16_PRIMARY
from fluxvortex.runtime.case_runner import (
    PARTIAL_EVERY,
    RojratsirikulCaseRunner,
    _write_json,
)

OUT_DIR = Path(
    "artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current"
)


def main() -> None:
    partial = json.loads((OUT_DIR / "ROJ11_A16_FULL.partial.json").read_text())
    records = partial["records"]
    steps = len(records)
    assert steps == partial["completed_aero_steps"]
    archive = np.load(OUT_DIR / "ROJ11_A16_FULL.z_history.npz")
    z_history = torch.tensor(
        archive["z_history_over_c"] * 0.0688,
        device="cuda:0",
        dtype=torch.float64,
    )
    elapsed = float(partial["elapsed_seconds"])
    wall = [elapsed / steps] * steps

    runner = RojratsirikulCaseRunner(ROJ_A16_PRIMARY)
    runner.build()
    pressure_sum = torch.zeros(
        runner.surface.nc * runner.surface.ns,
        device=runner.device,
        dtype=torch.float64,
    )
    payload = runner._finalize(
        records=records,
        z_history=z_history,
        pressure_sum=pressure_sum,
        step_wall_times=wall,
        execution_gate_only=False,
        execution_status="completed",
        failure=None,
        elapsed_seconds=elapsed,
        output=None,
        substeps=10,
    )
    payload["recovery_note"] = (
        "rescored from the completed partial after a JSON-serialization "
        "crash in the original finalization; physics records and z_history "
        "are the original run's, wall times are uniform placeholders"
    )
    output = OUT_DIR / "ROJ11_A16_FULL.json"
    _write_json(output, payload)
    print(f"written {output}")
    print(json.dumps(payload["result_status"], indent=1))
    print("mean zmax/c = %.5f  mean Cn = %.5f" % (
        payload["mean_zmax_over_c"], payload["mean_Cn"]))


if __name__ == "__main__":
    main()
