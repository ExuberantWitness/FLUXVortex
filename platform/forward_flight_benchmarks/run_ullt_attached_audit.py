"""Run a no-fit audit of the source-constrained one-state ULLT prototype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cases import IZRAELEVITZ_2017_FIG11, YANG_2025
from .ptera_adapter import (
    build_izraelevitz_fig11_movement,
    build_yang2025_movement,
)
from .ullt_attached import (
    blend_attached_with_uvlm_polar,
    movement_one_state_ullt,
    smooth_separation_fraction,
)
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    augment_uvlm_history,
    movement_polar_residual,
)


FLUXV_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IZRAELEVITZ_SOURCE = (
    FLUXV_ROOT
    / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/source_data/"
    "izraelevitz2017_fig11_digitized.csv"
)
DEFAULT_YANG_RUN = (
    FLUXV_ROOT
    / "docs/forward_flight_large_pitch/reproductions/plev2025/runs/"
    "20260808_multimodel_full"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def run_audit(
    izraelevitz_source: Path,
    yang_run: Path,
) -> dict[str, object]:
    iz_case = IZRAELEVITZ_2017_FIG11
    iz_movement, _ = build_izraelevitz_fig11_movement(
        settings=(4, 12, 256, 4)
    )
    iz_result = movement_one_state_ullt(
        iz_movement,
        source_cycle_step_range=(768, 1023),
        period_s=iz_case.period_s,
        freestream_m_s=iz_case.freestream_m_s,
        rho_kg_m3=iz_case.rho_kg_m3,
        aspect_ratio=iz_case.aspect_ratio,
        area_m2=iz_case.area_m2,
    )
    iz_rows = _rows(izraelevitz_source)
    paper_uvlm_cl = np.asarray(
        [float(row["paper_uvlm_CLalpha"]) for row in iz_rows]
    )
    paper_uvlm_cd = np.asarray(
        [float(row["paper_uvlm_CDalpha"]) for row in iz_rows]
    )
    paper_one_cl = np.asarray([float(row["1_state_CLalpha"]) for row in iz_rows])
    paper_one_cd = np.asarray([float(row["1_state_CDalpha"]) for row in iz_rows])
    normalization = np.sin(np.deg2rad(iz_case.downstroke_midpoint_alpha_deg))
    iz_cl = np.asarray(iz_result["CL"]) / normalization
    iz_cd = np.asarray(iz_result["CD"]) / normalization
    izraelevitz_metrics = {
        "paper_uvlm_CLalpha_RMSE": _rmse(iz_cl, paper_uvlm_cl),
        "paper_uvlm_CDalpha_RMSE": _rmse(iz_cd, paper_uvlm_cd),
        "paper_uvlm_CLalpha_range_NRMSE": (
            _rmse(iz_cl, paper_uvlm_cl) / float(np.ptp(paper_uvlm_cl))
        ),
        "paper_uvlm_CDalpha_range_NRMSE": (
            _rmse(iz_cd, paper_uvlm_cd) / float(np.ptp(paper_uvlm_cd))
        ),
        "digitized_paper_1_state_CLalpha_RMSE": _rmse(iz_cl, paper_one_cl),
        "digitized_paper_1_state_CDalpha_RMSE": _rmse(iz_cd, paper_one_cd),
        "CLalpha_min": float(np.min(iz_cl)),
        "CLalpha_max": float(np.max(iz_cl)),
        "CDalpha_min": float(np.min(iz_cd)),
        "CDalpha_max": float(np.max(iz_cd)),
        "CDalpha_mean": float(np.mean(iz_cd)),
    }

    phase_rows = _rows(yang_run / "phase_histories.csv")
    mean_rows = _rows(yang_run / "mean_characteristics.csv")
    yang_records: list[dict[str, float | str]] = []
    for aoa in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        movement, _ = build_yang2025_movement(aoa)
        ullt = movement_one_state_ullt(
            movement,
            source_cycle_step_range=(200, 299),
            period_s=YANG_2025.period_s,
            freestream_m_s=YANG_2025.freestream_m_s,
            rho_kg_m3=YANG_2025.rho_kg_m3,
            aspect_ratio=YANG_2025.aspect_ratio,
            area_m2=YANG_2025.area_m2,
        )
        polar = movement_polar_residual(
            movement,
            source_cycle_step_range=(200, 299),
            period_s=YANG_2025.period_s,
            freestream_m_s=YANG_2025.freestream_m_s,
            rho_kg_m3=YANG_2025.rho_kg_m3,
            aspect_ratio=YANG_2025.aspect_ratio,
        )
        old_rows = [
            row
            for row in phase_rows
            if row["model"] == "fluxv_uvpm"
            and np.isclose(float(row["aoa_deg"]), aoa)
        ]
        old_rows.sort(key=lambda row: float(row["phase"]))
        phase = np.asarray([float(row["phase"]) for row in old_rows])
        lift = np.asarray([float(row["lift_n"]) for row in old_rows])
        thrust = np.asarray([float(row["thrust_n"]) for row in old_rows])
        baseline = {
            "phase": phase,
            "lift_n": lift,
            "thrust_n": thrust,
            "mean_lift_n": float(np.mean(lift)),
            "mean_thrust_n": float(np.mean(thrust)),
        }
        uvlm_polar = augment_uvlm_history(
            baseline,
            polar,
            rho_kg_m3=YANG_2025.rho_kg_m3,
            freestream_m_s=YANG_2025.freestream_m_s,
            area_m2=YANG_2025.area_m2,
        )
        local_separation = smooth_separation_fraction(
            np.asarray(polar["alpha_rad"]),
            attached_limit_deg=DEFAULT_POLAR_PARAMETERS.attached_limit_deg,
            fully_separated_deg=DEFAULT_POLAR_PARAMETERS.fully_separated_deg,
        )
        area_weights = np.asarray(ullt["strip_area_m2"], dtype=float)
        global_separation = np.average(
            local_separation, axis=1, weights=area_weights
        )
        hybrid = blend_attached_with_uvlm_polar(
            ullt, uvlm_polar, global_separation
        )
        reference = next(
            row
            for row in mean_rows
            if row["model"] == "wind_tunnel_test"
            and np.isclose(float(row["aoa_deg"]), aoa)
        )
        reference_lift_gf = float(reference["mean_lift_gf"])
        reference_drag_gf = float(reference["mean_drag_gf"])
        for model, history in (
            ("one_state_ullt", ullt),
            ("uvlm_plus_polar", uvlm_polar),
            ("ullt_uvlm_polar_load_blend", hybrid),
        ):
            lift_gf = float(np.mean(history["lift_n"]) * 1000.0 / 9.80665)
            if "drag_n" in history:
                drag_n = np.asarray(history["drag_n"], dtype=float)
            else:
                drag_n = -np.asarray(history["thrust_n"], dtype=float)
            drag_gf = float(np.mean(drag_n) * 1000.0 / 9.80665)
            yang_records.append(
                {
                    "aoa_deg": aoa,
                    "model": model,
                    "mean_lift_gf": lift_gf,
                    "mean_drag_gf": drag_gf,
                    "lift_error_gf": lift_gf - reference_lift_gf,
                    "drag_error_gf": drag_gf - reference_drag_gf,
                    "mean_global_separation_fraction": float(
                        np.mean(global_separation)
                    ),
                }
            )
    yang_summary: dict[str, dict[str, float]] = {}
    for model in sorted({str(row["model"]) for row in yang_records}):
        model_rows = [row for row in yang_records if row["model"] == model]
        yang_summary[model] = {
            "lift_MAE_gf": float(
                np.mean([abs(float(row["lift_error_gf"])) for row in model_rows])
            ),
            "drag_MAE_gf": float(
                np.mean([abs(float(row["drag_error_gf"])) for row in model_rows])
            ),
        }
    return {
        "model_input_policy": "no digitized load is read by ullt_attached.py",
        "izraelevitz2017": izraelevitz_metrics,
        "yang2025_by_aoa": yang_records,
        "yang2025_summary": yang_summary,
        "hybrid_caveat": (
            "global load-level blend using the area-weighted local separation "
            "gate; not a UVLM pressure-channel decomposition"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--izraelevitz-source", type=Path, default=DEFAULT_IZRAELEVITZ_SOURCE
    )
    parser.add_argument("--yang-run", type=Path, default=DEFAULT_YANG_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(args.izraelevitz_source, args.yang_run)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
