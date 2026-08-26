"""Extract the reduced Yamano MATLAB wake-motion pressure oracle.

This is an offline provenance utility.  It reads the authors' full ``-v7``
checkpoint fixtures and writes only the arrays needed to verify, component by
component, the CUDA port of ``dt_generate_q1234_mat`` and
``Mf2_vec1=A_mat\\(-Gamma_wake_dt_q1234_n)``.  The resulting archive is never
used by the production FSI solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


_FIXTURES = (
    (
        "fixture_step1_t0.0680.mat",
        "1f9bf07638620b701a85c7a0ad816e74cd781ba8dc6c69f929e6b982d4f6b047",
    ),
    (
        "fixture_step2_t0.1360.mat",
        "ccc12dd23774f300f6928edfafc1ccf1f08267532b3bf3721831a2ee687e45c3",
    ),
    (
        "fixture_step3_t0.1995.mat",
        "14ed489d1e08e62d1cc130ac3ca6c2097d9b08dcdf97879d21eddd4dee7e51df",
    ),
    (
        "fixture_step4_t0.2640.mat",
        "981b8884f9531655cdc9b4e5dd612a88b52c0081801e2af426da2c70043fccaf",
    ),
)
_VARIABLES = (
    "time",
    "time_fluid",
    "i_wake_time",
    "rc_vec",
    "dt_rc_vec",
    "r_wake_1",
    "r_wake_2",
    "r_wake_3",
    "r_wake_4",
    "dt_r_wake_1",
    "dt_r_wake_2",
    "dt_r_wake_3",
    "dt_r_wake_4",
    "Gamma_wake",
    "n_vec_i",
    "A_mat",
    "Gamma_wake_dt_q1234_n",
    "Mf2_vec1",
    "dp_lift1",
    "Qf_p_global",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source_dir: Path, output: Path) -> None:
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("yamano2020-matlab-mf2-history-oracle-v1"),
        "source_manifest_json": np.asarray(
            json.dumps(
                [
                    {"name": name, "sha256": digest}
                    for name, digest in _FIXTURES
                ],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
        "step_count": np.asarray(len(_FIXTURES), dtype=np.int64),
    }
    for step, (name, digest) in enumerate(_FIXTURES, start=1):
        fixture = source_dir / name
        observed = _sha256(fixture)
        if observed != digest:
            raise RuntimeError(f"Yamano fixture drift for {fixture}: {observed}")
        values = loadmat(
            fixture,
            variable_names=list(_VARIABLES),
            squeeze_me=True,
        )
        missing = sorted(set(_VARIABLES) - values.keys())
        if missing:
            raise RuntimeError(f"missing variables in {fixture}: {missing}")
        for variable in _VARIABLES:
            value = np.asarray(values[variable])
            if not bool(np.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite {variable} in Yamano fixture {fixture}"
                )
            payload[f"step_{step}_{variable}"] = np.ascontiguousarray(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
