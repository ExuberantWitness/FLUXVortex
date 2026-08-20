"""Exercise G0/G0b/G0c failure exits without modifying frozen source files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
CASES = (
    (
        "G0",
        REPO / "platform/warp_vpm/test_g0_steady.py",
        "ok = cl2 <= cl_final <= cl_upper",
        "ok = False  # audit-forced terminal gate failure",
    ),
    (
        "G0b",
        REPO / "platform/warp_vpm/test_g0b_ptera.py",
        "parity_ok = abs(c2[-1] - c1[-1]) < 1e-14",
        "parity_ok = False  # audit-forced parity failure",
    ),
    (
        "G0c",
        REPO / "platform/warp_vpm/test_g0c_lev_active.py",
        'ok = finite and reduced and d["lev_strips"] > 0',
        "ok = False  # audit-forced active-LEV failure",
    ),
)


child = (
    "import sys; p=sys.argv[1]; "
    "g={'__file__':p,'__name__':'__main__'}; "
    "exec(compile(sys.stdin.read(),p,'exec'),g)"
)
environment = dict(os.environ)
environment.update(
    {
        "PYTHONPATH": "src:platform:platform/warp_vpm",
        "MPLCONFIGDIR": "/tmp/mpl-v5m-negative-gates",
        "NUMBA_CACHE_DIR": "/tmp/numba-v5m-negative-gates",
        "PFIELD_DEVICE": "cuda",
    }
)

records = []
for gate, path, original, replacement in CASES:
    source = path.read_text(encoding="utf-8")
    if source.count(original) != 1:
        raise RuntimeError(f"{gate}: exact source target count is not one")
    transformed = source.replace(original, replacement)
    completed = subprocess.run(
        [sys.executable, "-c", child, str(path)],
        cwd=REPO,
        env=environment,
        input=transformed,
        text=True,
        capture_output=True,
        check=False,
    )
    records.append(
        {
            "gate": gate,
            "returncode": completed.returncode,
            "expected_returncode": 1,
            "passed": completed.returncode == 1,
            "stdout_tail": completed.stdout.splitlines()[-4:],
            "stderr_tail": completed.stderr.splitlines()[-4:],
        }
    )

print(json.dumps(records, sort_keys=True, separators=(",", ":")))
if not all(record["passed"] for record in records):
    raise SystemExit("one or more gates did not exit 1 on a forced failure")
