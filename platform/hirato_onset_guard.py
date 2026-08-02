"""Run the preregistered Hirato Case 1/2 Eq.6 onset guards.

This is a no-LEV-force probe.  It observes the LEV-free UVLM/TEV state through
the paper's Eq.6 operator and never invokes the incomplete historical Hirato
constraint.  Use ``--quick`` for the smallest frozen grid or ``--full`` for
the complete grid/time sensitivity matrix in ``hirato_canonical_cases.yaml``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import warp as wp

from _v2_robo import gpu_run_twist


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "hirato_canonical_cases.yaml"


def _alpha_deg(tstar: float, pitch_max_deg: float, rate: float) -> float:
    pmax = np.radians(pitch_max_deg)
    t1 = 1.0
    t2 = t1 + pmax / (2.0 * rate)
    smooth = 6.0
    lncosh = lambda z: np.logaddexp(z, -z) - np.log(2.0)
    profile = lncosh(smooth * (tstar - t1)) - lncosh(
        smooth * (tstar - t2)
    )
    scale = smooth * (t2 - t1)
    fraction = np.clip((profile + scale) / (2.0 * scale), 0.0, 1.0)
    return float(np.degrees(pmax * fraction))


def run_case(
    *,
    case_name: str,
    tip_twist_deg: float,
    nc: int,
    ns: int,
    steps: int,
    spec: dict,
) -> dict:
    shared = spec["shared"]
    chord = 0.1
    half_span = 0.5 * shared["aspect_ratio"] * chord
    nu = 1.5e-5
    speed = shared["chord_reynolds"] * nu / chord
    frequency = speed / (3.0 * chord)
    result = gpu_run_twist(
        U=speed,
        aoa_deg=0.0,
        freq=frequency,
        n_cycle=1,
        steps_per_cycle=steps,
        wake_rows=steps,
        nc=nc,
        ns=ns,
        chord=chord,
        half_span=half_span,
        flap_amp_deg=0.0,
        twist_amp_deg=tip_twist_deg,
        real_geom=False,
        section_geometry="sd7003",
        sym=True,
        pitch_ramp=True,
        pitch_max=shared["alpha_end_deg"],
        pitch_K=shared["reduced_pitch_rate_K"],
        pitch_t0star=1.0,
        lev_shed_mode="hirato_probe",
        a0_crit=shared["lesp_crit"],
        d_para=0.0,
    )
    audit = result["n3_hirato_audit"]
    first = next((row for row in audit if np.any(row["shed_eq6_paper"])), None)
    if first is None:
        return {
            "case": case_name,
            "nc": nc,
            "ns": ns,
            "steps": steps,
            "gate": "fail-no-onset",
        }
    indices = np.flatnonzero(first["shed_eq6_paper"])
    tstar = 3.0 * first["step"] / steps
    strip_centers = (indices + 0.5) / ns
    expected = spec["cases"][case_name]["expected_onset"]
    guards = spec["preregistered_numerical_guards"]
    time_error = abs(tstar - expected["tstar"])
    span_error = float(
        np.min(np.abs(strip_centers - expected["abs_y_over_halfspan"]))
    )
    passed = (
        time_error <= guards["onset_tstar_abs_tolerance"]
        and span_error <= guards["onset_span_abs_y_over_halfspan_tolerance"]
    )
    return {
        "case": case_name,
        "nc": nc,
        "ns": ns,
        "steps": steps,
        "step": int(first["step"]),
        "tstar": float(tstar),
        "alpha_deg": _alpha_deg(
            tstar,
            shared["alpha_end_deg"],
            shared["reduced_pitch_rate_K"],
        ),
        "time_abs_error": float(time_error),
        "first_strip_centers_y_over_halfspan": strip_centers.tolist(),
        "span_abs_error": span_error,
        "max_abs_A0": float(np.max(np.abs(first["A0_eq6_paper"]))),
        "gate": "pass" if passed else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(SPEC_PATH.read_text())
    guards = spec["preregistered_numerical_guards"]
    grids = guards["sensitivity"]["grids"]
    steps = guards["sensitivity"]["time_steps_per_tstar_3"]
    if args.quick:
        grids = grids[:1]
        steps = steps[:1]

    wp.init()
    records = []
    for grid in grids:
        for step_count in steps:
            for case_name in ("case1", "case2"):
                case = spec["cases"][case_name]
                record = run_case(
                    case_name=case_name,
                    tip_twist_deg=case["tip_twist_deg"],
                    nc=grid["nc"],
                    ns=grid["ns"],
                    steps=step_count,
                    spec=spec,
                )
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    payload = {
        "source": str(SPEC_PATH.relative_to(HERE.parent)),
        "role": "read-only Eq.6 onset sensitivity; no LEV force",
        "records": records,
        "all_pass": all(record["gate"] == "pass" for record in records),
    }
    print(f"ALL_PASS={payload['all_pass']}", flush=True)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
