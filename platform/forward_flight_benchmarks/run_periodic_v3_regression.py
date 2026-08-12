"""Run the isolated persistent-owner v3 regression on all three gates.

This runner reuses the frozen v1/v2 UVLM histories and recomputes only the
geometry/kinematics-only persistence diagnostic and one-state ULLT owner.  It
does not overwrite the v1/v2 artifacts.  The model was introduced after the
Izraelevitz Figure-14 failure was observed and is therefore explicitly an
exploratory repair, not independent confirmation or an LEV-suction model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cases import IZRAELEVITZ_2017_FIG11 as IZ11
from .cases import IZRAELEVITZ_2017_FIG14_SCHERER as IZ14
from .cases import YANG_2025 as YANG
from .periodic_load_ownership import (
    blend_periodic_persistent_owner,
    periodic_incidence_persistence,
)
from .ptera_adapter import (
    build_izraelevitz_fig11_movement,
    build_izraelevitz_scherer_movement,
    build_yang2025_movement,
)
from .run_unified_fluxv_upgrade import GF_TO_N, _curve_metrics, _phase_metrics
from .ullt_attached import movement_one_state_ullt, smooth_separation_fraction
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    FullAnglePolarParameters,
    movement_polar_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = (
    REPO_ROOT / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812"
)
DEFAULT_BASE = DOC_ROOT / "runs/20260812_periodic_v2_ullt_full"
DEFAULT_FIG14_BASE = DOC_ROOT / "runs/20260812_scherer_fig14_experiment_full"
DEFAULT_OUTPUT = DOC_ROOT / "runs/20260812_periodic_v3_persistent_smoke"
MODEL = "fluxv_periodic_v3_persistent"
ABLATION = "fluxv_periodic_v3_mean_passthrough"
MODEL_LABEL = "FluxV v3 persistent ULLT/UVLM owner"
SCHERER_REFERENCE_PARAMETERS = FullAnglePolarParameters(
    section_velocity_reference_fraction_chord=IZ14.pivot_fraction_chord
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return _relative(value)
    raise TypeError(type(value).__name__)


def _cycle_range(movement: Any, period_s: float) -> tuple[int, int]:
    count = len(movement.airplanes[0])
    steps = int(round(period_s / movement.delta_time))
    last = count - 1
    cycles_at_last = last * movement.delta_time / period_s
    end = last if np.isclose(cycles_at_last, round(cycles_at_last)) else count
    return end - steps, end - 1


def _history_from_rows(
    rows: list[dict[str, str]], model: str, *, aoa: float | None = None
) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    if aoa is not None:
        selected = [row for row in selected if np.isclose(float(row["aoa_deg"]), aoa)]
    selected.sort(key=lambda row: float(row["phase"]))
    if not selected:
        raise ValueError(f"no history for {model}, aoa={aoa}")
    if "lift_n" in selected[0]:
        lift = np.asarray([float(row["lift_n"]) for row in selected])
        drag = np.asarray([float(row["drag_n"]) for row in selected])
    else:
        q_area = 0.5 * IZ11.rho_kg_m3 * IZ11.freestream_m_s**2 * IZ11.area_m2
        lift = q_area * np.asarray([float(row["CL"]) for row in selected])
        drag = q_area * np.asarray([float(row["CD"]) for row in selected])
    return {
        "phase": np.asarray([float(row["phase"]) for row in selected]),
        "lift_n": lift,
        "drag_n": drag,
        "mean_lift_n": float(np.mean(lift)),
        "mean_drag_n": float(np.mean(drag)),
    }


def _persistence_and_separation(
    polar: dict[str, Any], strip_area_m2: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    alpha = np.asarray(polar["alpha_rad"], dtype=float)
    persistence, strip_persistence = periodic_incidence_persistence(
        alpha, strip_weights=np.asarray(strip_area_m2, dtype=float)
    )
    local_separation = smooth_separation_fraction(
        alpha,
        attached_limit_deg=DEFAULT_POLAR_PARAMETERS.attached_limit_deg,
        fully_separated_deg=DEFAULT_POLAR_PARAMETERS.fully_separated_deg,
    )
    separation = np.average(
        local_separation,
        axis=1,
        weights=np.asarray(strip_area_m2, dtype=float),
    )
    return persistence, strip_persistence, separation


def _run_yang(
    base: Path, quality: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_means = _read_csv(base / "yang2025_mean_characteristics.csv")
    source_phase = _read_csv(base / "yang2025_phase_histories.csv")
    output_means: list[dict[str, Any]] = []
    output_phase: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for aoa in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        movement, _ = build_yang2025_movement(aoa, quality)
        cycle = _cycle_range(movement, YANG.period_s)
        ullt = movement_one_state_ullt(
            movement,
            source_cycle_step_range=cycle,
            period_s=YANG.period_s,
            freestream_m_s=YANG.freestream_m_s,
            rho_kg_m3=YANG.rho_kg_m3,
            aspect_ratio=YANG.aspect_ratio,
            area_m2=YANG.area_m2,
            output_samples=128,
        )
        polar = movement_polar_residual(
            movement,
            source_cycle_step_range=cycle,
            period_s=YANG.period_s,
            freestream_m_s=YANG.freestream_m_s,
            rho_kg_m3=YANG.rho_kg_m3,
            aspect_ratio=YANG.aspect_ratio,
            output_samples=128,
        )
        persistence, strip_persistence, separation = _persistence_and_separation(
            polar, ullt["strip_area_m2"]
        )
        v1 = _history_from_rows(source_phase, "fluxv_periodic_v1", aoa=aoa)
        v1_mean = next(
            row
            for row in source_means
            if row["model"] == "fluxv_periodic_v1"
            and np.isclose(float(row["aoa_deg"]), aoa)
        )
        v1["mean_lift_n"] = float(v1_mean["mean_lift_n"])
        v1["mean_drag_n"] = float(v1_mean["mean_drag_n"])
        v3 = blend_periodic_persistent_owner(
            ullt,
            v1,
            separation,
            persistence_fraction=persistence,
            rho_kg_m3=YANG.rho_kg_m3,
            freestream_m_s=YANG.freestream_m_s,
            area_m2=YANG.area_m2,
        )
        output_means.append(
            {
                "aoa_deg": aoa,
                "model": MODEL,
                "model_label": MODEL_LABEL,
                "mean_lift_n": v3["mean_lift_n"],
                "mean_drag_n": v3["mean_drag_n"],
                "mean_lift_gf": v3["mean_lift_n"] / GF_TO_N,
                "mean_drag_gf": v3["mean_drag_n"] / GF_TO_N,
                "persistence_fraction": persistence,
            }
        )
        for index, phase in enumerate(v3["phase"]):
            output_phase.append(
                {
                    "aoa_deg": aoa,
                    "model": MODEL,
                    "phase": phase,
                    "lift_n": v3["lift_n"][index],
                    "drag_n": v3["drag_n"][index],
                    "CL": v3["CL"][index],
                    "CD": v3["CD"][index],
                    "CT": v3["CT"][index],
                }
            )
        audit[str(int(aoa))] = {
            "cycle_step_range": cycle,
            "persistence_fraction": persistence,
            "strip_persistence_fraction": strip_persistence,
            "mean_separation_fraction": float(np.mean(separation)),
            "mean_effective_separation_fraction": float(
                np.mean(v3["effective_separation_fraction"])
            ),
        }

    test = sorted(
        (row for row in source_means if row["model"] == "wind_tunnel_test"),
        key=lambda row: float(row["aoa_deg"]),
    )
    prediction = sorted(output_means, key=lambda row: float(row["aoa_deg"]))
    metrics = []
    for channel, key in (("lift", "mean_lift_gf"), ("drag", "mean_drag_gf")):
        metric = _curve_metrics(
            np.asarray([float(row[key]) for row in prediction]),
            np.asarray([float(row[key]) for row in test]),
        )
        metrics.append(
            {
                "gate": "yang2025_experiment",
                "model": MODEL,
                "channel": channel,
                "units": "gf",
                **metric,
            }
        )
    return output_means, output_phase, {"conditions": audit, "metrics": metrics}


def _run_fig11(base: Path, quality: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _read_csv(base / "izraelevitz2017_fig11_phase_histories.csv")
    # The Figure-11 reference curves contain 128 samples, but the established
    # v2 full gate uses a 256-step source cycle before resampling.  The builder
    # default remains a 24-step exploratory discretization even for
    # ``quality=full``, so freeze the audited full settings explicitly here.
    fig11_settings = (4, 12, 256, 4) if quality == "full" else None
    movement, movement_metadata = build_izraelevitz_fig11_movement(
        quality, settings=fig11_settings
    )
    cycle = _cycle_range(movement, IZ11.period_s)
    ullt = movement_one_state_ullt(
        movement,
        source_cycle_step_range=cycle,
        period_s=IZ11.period_s,
        freestream_m_s=IZ11.freestream_m_s,
        rho_kg_m3=IZ11.rho_kg_m3,
        aspect_ratio=IZ11.aspect_ratio,
        area_m2=IZ11.area_m2,
        output_samples=128,
    )
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=cycle,
        period_s=IZ11.period_s,
        freestream_m_s=IZ11.freestream_m_s,
        rho_kg_m3=IZ11.rho_kg_m3,
        aspect_ratio=IZ11.aspect_ratio,
        output_samples=128,
    )
    persistence, strip_persistence, separation = _persistence_and_separation(
        polar, ullt["strip_area_m2"]
    )
    v1 = _history_from_rows(source, "fluxv_periodic_v1")
    v3 = blend_periodic_persistent_owner(
        ullt,
        v1,
        separation,
        persistence_fraction=persistence,
        rho_kg_m3=IZ11.rho_kg_m3,
        freestream_m_s=IZ11.freestream_m_s,
        area_m2=IZ11.area_m2,
    )
    scale = np.sin(np.deg2rad(IZ11.downstroke_midpoint_alpha_deg))
    rows = [
        {
            "model": MODEL,
            "phase": phase,
            "CL_alpha": v3["CL"][index] / scale,
            "CD_alpha": v3["CD"][index] / scale,
            "CL": v3["CL"][index],
            "CD": v3["CD"][index],
            "CT": v3["CT"][index],
        }
        for index, phase in enumerate(v3["phase"])
    ]
    reference = [row for row in source if row["model"] == "paper_uvlm"]
    metrics = []
    for channel, key in (("lift", "CL_alpha"), ("drag", "CD_alpha")):
        metric = _phase_metrics(
            np.asarray([float(row[key]) for row in rows]),
            np.asarray([float(row[key]) for row in reference]),
        )
        metrics.append(
            {
                "gate": "izraelevitz2017_fig11_numerical",
                "model": MODEL,
                "channel": channel,
                "units": "paper_scaled_coefficient",
                **metric,
            }
        )
    return rows, {
        "movement": movement_metadata,
        "cycle_step_range": cycle,
        "persistence_fraction": persistence,
        "strip_persistence_fraction": strip_persistence,
        "mean_separation_fraction": float(np.mean(separation)),
        "metrics": metrics,
    }


def _run_fig14(base: Path, quality: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _read_csv(base / "mean_thrust_vs_phase.csv")
    sensitivity = _read_csv(base / "profile_drag_sensitivity.csv")
    observations = [
        row for row in source if row["data_role"] == "experimental_observation"
    ]
    conditions = sorted(
        {
            (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            for row in observations
        }
    )
    zero_cd = {
        (
            float(row["theta_max_deg"]),
            float(row["phase_offset_deg"]),
            row["model"],
        ): float(row["CT"])
        for row in sensitivity
        if np.isclose(float(row["profile_drag_coefficient"]), 0.0)
    }
    predictions: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    q_area = 0.5 * IZ14.rho_kg_m3 * IZ14.freestream_m_s**2 * IZ14.area_m2
    for theta, psi in conditions:
        movement, _ = build_izraelevitz_scherer_movement(theta, psi, quality)
        cycle = _cycle_range(movement, IZ14.period_s)
        ullt = movement_one_state_ullt(
            movement,
            source_cycle_step_range=cycle,
            period_s=IZ14.period_s,
            freestream_m_s=IZ14.freestream_m_s,
            rho_kg_m3=IZ14.rho_kg_m3,
            aspect_ratio=IZ14.aspect_ratio,
            area_m2=IZ14.area_m2,
            output_samples=128,
        )
        # Separate calls intentionally preserve separate polar and source-Cd0
        # ledgers.  Both use the paper's actual 3/4-chord pitch-axis velocity.
        polar = movement_polar_residual(
            movement,
            source_cycle_step_range=cycle,
            period_s=IZ14.period_s,
            freestream_m_s=IZ14.freestream_m_s,
            rho_kg_m3=IZ14.rho_kg_m3,
            aspect_ratio=IZ14.aspect_ratio,
            output_samples=128,
            parameters=SCHERER_REFERENCE_PARAMETERS,
        )
        profile_kinematics = movement_polar_residual(
            movement,
            source_cycle_step_range=cycle,
            period_s=IZ14.period_s,
            freestream_m_s=IZ14.freestream_m_s,
            rho_kg_m3=IZ14.rho_kg_m3,
            aspect_ratio=IZ14.aspect_ratio,
            output_samples=128,
            parameters=SCHERER_REFERENCE_PARAMETERS,
        )
        persistence, strip_persistence = periodic_incidence_persistence(
            np.asarray(polar["alpha_rad"]),
            strip_weights=np.asarray(polar["mean_strip_area_m2"]),
        )
        profile_ct = (
            -IZ14.profile_drag_coefficient
            * float(profile_kinematics["mean_unit_profile_drag_drag_n"])
            / q_area
        )
        ullt_inviscid_ct = float(ullt["mean_CT"])
        v1_inviscid_ct = zero_cd[(theta, psi, "fluxv_periodic_v1")]
        old_inviscid_ct = zero_cd[(theta, psi, "fluxv_uvpm")]
        v3_ct = (
            (1.0 - persistence) * ullt_inviscid_ct
            + persistence * v1_inviscid_ct
            + profile_ct
        )
        pass_through_ct = (
            old_inviscid_ct
            + persistence * (v1_inviscid_ct - old_inviscid_ct)
            + profile_ct
        )
        for model, ct in ((MODEL, v3_ct), (ABLATION, pass_through_ct)):
            predictions.append(
                {
                    "model": model,
                    "theta_max_deg": theta,
                    "phase_offset_deg": psi,
                    "CT": ct,
                    "persistence_fraction": persistence,
                    "profile_drag_reference_fraction_chord": 0.75,
                }
            )
        audit[f"theta_{theta:g}_psi_{psi:g}"] = {
            "cycle_step_range": cycle,
            "persistence_fraction": persistence,
            "strip_persistence_fraction": strip_persistence,
            "ullt_inviscid_CT": ullt_inviscid_ct,
            "v1_inviscid_CT": v1_inviscid_ct,
            "old_inviscid_CT": old_inviscid_ct,
            "source_Cd0_CT": profile_ct,
        }

    lookup = {
        (row["model"], row["theta_max_deg"], row["phase_offset_deg"]): row["CT"]
        for row in predictions
    }
    metrics = []
    for model in (MODEL, ABLATION):
        error = np.asarray(
            [
                lookup[
                    (
                        model,
                        float(row["theta_max_deg"]),
                        float(row["phase_offset_deg"]),
                    )
                ]
                - float(row["CT"])
                for row in observations
            ]
        )
        metrics.append(
            {
                "gate": "izraelevitz2017_fig14_scherer_experiment",
                "model": model,
                "channel": "mean_CT",
                "units": "coefficient",
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "max_abs_error": float(np.max(np.abs(error))),
                "observation_count": int(error.size),
            }
        )
    return predictions, {"conditions": audit, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fig14-base-run", type=Path, default=DEFAULT_FIG14_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = args.base_run.resolve()
    fig14_base = args.fig14_base_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    yang_means, yang_phase, yang = _run_yang(base, args.quality)
    fig11_rows, fig11 = _run_fig11(base, args.quality)
    fig14_rows, fig14 = _run_fig14(fig14_base, args.quality)
    metrics = yang["metrics"] + fig11["metrics"] + fig14["metrics"]
    result_paths = {
        "yang_means": output / "yang2025_v3_mean_characteristics.csv",
        "yang_phase": output / "yang2025_v3_phase_histories.csv",
        "fig11_phase": output / "izraelevitz2017_fig11_v3_phase_histories.csv",
        "fig14_means": output / "izraelevitz2017_fig14_v3_mean_thrust.csv",
        "metrics": output / "v3_accuracy_metrics.csv",
    }
    _write_csv(result_paths["yang_means"], yang_means)
    _write_csv(result_paths["yang_phase"], yang_phase)
    _write_csv(result_paths["fig11_phase"], fig11_rows)
    _write_csv(result_paths["fig14_means"], fig14_rows)
    _write_csv(result_paths["metrics"], metrics)
    summary = {
        "status": "complete",
        "quality": args.quality,
        "model": MODEL,
        "metrics": metrics,
        "gates": {
            "yang_better_than_frozen_old": all(
                next(
                    row["mae"]
                    for row in metrics
                    if row["gate"] == "yang2025_experiment"
                    and row["channel"] == channel
                )
                < old
                for channel, old in (("lift", 6.8548706331), ("drag", 12.9216637993))
            ),
            "fig11_both_rmse_below_0_5": all(
                row["rmse"] < 0.5
                for row in metrics
                if row["gate"] == "izraelevitz2017_fig11_numerical"
            ),
            "fig14_rmse_below_frozen_old": next(
                row["rmse"]
                for row in metrics
                if row["gate"] == "izraelevitz2017_fig14_scherer_experiment"
                and row["model"] == MODEL
            )
            < 0.0511466231,
        },
        "numerical_sensitivity": {
            "comparison": "smoke-to-full changes grid, time step, and cycles together",
            "interpretation": "development sensitivity only; not formal convergence",
            "fig14_max_abs_delta_CT": 0.005548485905849027,
            "predeclared_target_CT": 0.005,
            "target_passed": False,
        },
    }
    manifest = {
        "run_id": output.name,
        "status": "complete",
        "quality": args.quality,
        "model": MODEL,
        "introduced_after_fig14_failure": True,
        "observation_fit": "none, but model selection is post-Figure-14 and exploratory",
        "model_semantics": (
            "persistence-weighted periodic mean owner and AC gate: p=0 selects "
            "one-state ULLT, p=1 selects UVLM/full-angle-polar; no case id"
        ),
        "not_claimed": [
            "causal transient model",
            "leading-edge-vortex suction-loss closure",
            "independent Figure-14 confirmation",
        ],
        "fig14_profile_drag": {
            "coefficient": IZ14.profile_drag_coefficient,
            "velocity_reference_fraction_chord": 0.75,
            "polar_and_profile_constructed_as_separate_ledgers": True,
            "added_exactly_once": True,
        },
        "base_runs": [_relative(base), _relative(fig14_base)],
        "base_hashes": {
            _relative(base / "run_manifest.json"): _sha256(base / "run_manifest.json"),
            _relative(fig14_base / "run_manifest.json"): _sha256(
                fig14_base / "run_manifest.json"
            ),
        },
        "source_hashes": {
            _relative(Path(__file__)): _sha256(Path(__file__)),
            _relative(Path(__file__).with_name("periodic_load_ownership.py")): _sha256(
                Path(__file__).with_name("periodic_load_ownership.py")
            ),
            _relative(Path(__file__).with_name("uvlm_polar_correction.py")): _sha256(
                Path(__file__).with_name("uvlm_polar_correction.py")
            ),
            _relative(Path(__file__).with_name("ullt_attached.py")): _sha256(
                Path(__file__).with_name("ullt_attached.py")
            ),
            _relative(
                Path(__file__).with_name("plot_periodic_v3_regression.py")
            ): _sha256(Path(__file__).with_name("plot_periodic_v3_regression.py")),
            _relative(
                REPO_ROOT / "platform/tests/test_periodic_load_ownership.py"
            ): _sha256(REPO_ROOT / "platform/tests/test_periodic_load_ownership.py"),
            _relative(
                DOC_ROOT / "source_data/izraelevitz2017_fig11_digitized.csv"
            ): _sha256(DOC_ROOT / "source_data/izraelevitz2017_fig11_digitized.csv"),
            _relative(
                DOC_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"
            ): _sha256(DOC_ROOT / "source_data/izraelevitz2017_fig14_digitized.csv"),
        },
        "audit": {
            "yang": yang["conditions"],
            "fig11": fig11,
            "fig14": fig14["conditions"],
        },
        "limitations": [
            "v3 is a periodic two-pass/post-hoc owner, not an online solver",
            "Figure 11 is numerical UVLM reference, not experiment",
            "Figure 14 publishes cycle means only",
            "Passing Figure 14 by ULLT ownership does not resolve LEV suction physics",
            "Figure-14 smoke-to-full max |delta CT|=0.00555 slightly exceeds the 0.005 development target",
        ],
    }
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    manifest["result_hashes"] = {
        _relative(path): _sha256(path) for path in result_paths.values()
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
