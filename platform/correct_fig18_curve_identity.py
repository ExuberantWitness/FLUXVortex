"""Correct the Fig. 18 thrust-curve wind-speed identities.

The digitized data asset assigned the upper thrust curve to U=10 m/s and the
lower curve to U=6 m/s.  The source PDF shows the opposite in both Fig. 18(a)
and Fig. 18(c): U=6 is the upper (less negative) curve and U=10 is the lower
(more negative) curve.  Lift identities are already correct.

Only the measured ``x``/``exp`` pairs are exchanged.  Model prediction fields
remain attached to their original condition keys.  The operation is
idempotent and refuses an ambiguous input instead of guessing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
TARGETS = (DOCS / "repro_data.json", DOCS / "repro_compare.json")
PAIRS = [
    ("18|a|6.0", "18|a|10.0"),
    ("18|c|(6.0, 2.0)", "18|c|(10.0, 2.0)"),
    ("18|c|(6.0, 2.3)", "18|c|(10.0, 2.3)"),
    ("18|c|(6.0, 2.6)", "18|c|(10.0, 2.6)"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_exp(record: dict) -> float:
    values = [float(value) for value in record["exp"]]
    return sum(values) / len(values)


def correct(path: Path) -> bool:
    data = json.loads(path.read_text())
    states = []
    for key6, key10 in PAIRS:
        mean6 = _mean_exp(data[key6])
        mean10 = _mean_exp(data[key10])
        if mean6 == mean10:
            raise RuntimeError(f"{path.name}: ambiguous equal means for {key6}/{key10}")
        states.append(mean6 > mean10)

    if all(states):
        print(f"{path.name}: already corrected ({_sha256(path)[:12]})")
        return False
    if any(states):
        raise RuntimeError(f"{path.name}: mixed Fig. 18 identity state; refusing partial rewrite")

    before = _sha256(path)
    for key6, key10 in PAIRS:
        for field in ("x", "exp"):
            data[key6][field], data[key10][field] = data[key10][field], data[key6][field]
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{path.name}: corrected {before[:12]} -> {_sha256(path)[:12]}")
    return True


def main() -> None:
    changed = [correct(path) for path in TARGETS]
    if not any(changed):
        print("Fig. 18 thrust identities require no changes")


if __name__ == "__main__":
    main()
