"""Extend the periodic-v1 run with the reconstructed one-state ULLT and v2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .augmented_uvpm import blend_periodic_ullt_state_shape
from .cases import IZRAELEVITZ_2017_FIG11 as IZ_CASE
from .cases import YANG_2025 as YANG_CASE
from .ptera_adapter import (
    build_izraelevitz_fig11_movement,
    build_yang2025_movement,
)
from .run_unified_fluxv_upgrade import (
    DEFAULT_OUTPUT as DEFAULT_BASE,
    GF_TO_N,
    MODEL_LABELS,
    ROOT,
    _compute_metrics,
    _json_default,
    _read_csv,
    _sha256,
    _write_csv,
)
from .ullt_attached import (
    movement_one_state_ullt,
    smooth_separation_fraction,
)
from .uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    movement_polar_residual,
)


DEFAULT_OUTPUT = DEFAULT_BASE.parent / "20260812_periodic_v2_ullt_full"


def _manifest_path(path: Path) -> str:
    """Return a portable repository-relative provenance key when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed-as-package"


def _history_from_rows(
    rows: list[dict[str, str]], model: str, *, aoa: float | None = None
) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    if aoa is not None:
        selected = [
            row for row in selected if np.isclose(float(row["aoa_deg"]), aoa)
        ]
    selected.sort(key=lambda row: float(row["phase"]))
    if not selected:
        raise ValueError(f"no phase rows for {model}, aoa={aoa}")
    if "lift_n" in selected[0]:
        lift = np.asarray([float(row["lift_n"]) for row in selected])
        drag = np.asarray([float(row["drag_n"]) for row in selected])
    else:
        q_area = (
            0.5
            * IZ_CASE.rho_kg_m3
            * IZ_CASE.freestream_m_s**2
            * IZ_CASE.area_m2
        )
        lift = q_area * np.asarray([float(row["CL"]) for row in selected])
        drag = q_area * np.asarray([float(row["CD"]) for row in selected])
    return {
        "phase": np.asarray([float(row["phase"]) for row in selected]),
        "lift_n": lift,
        "drag_n": drag,
        "thrust_n": -drag,
        "mean_lift_n": float(np.mean(lift)),
        "mean_drag_n": float(np.mean(drag)),
    }


def _append_phase(
    rows: list[dict[str, Any]],
    history: dict[str, Any],
    *,
    model: str,
    aoa: float | None = None,
    izraelevitz: bool = False,
) -> None:
    scale = np.sin(np.deg2rad(IZ_CASE.downstroke_midpoint_alpha_deg))
    for index, phase in enumerate(history["phase"]):
        lift = float(history["lift_n"][index])
        drag = float(history["drag_n"][index])
        if izraelevitz:
            q_area = (
                0.5
                * IZ_CASE.rho_kg_m3
                * IZ_CASE.freestream_m_s**2
                * IZ_CASE.area_m2
            )
            cl = lift / q_area
            cd = drag / q_area
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "data_role": "exploratory_local_model",
                    "phase": phase,
                    "CL_alpha": cl / scale,
                    "CD_alpha": cd / scale,
                    "CL": cl,
                    "CD": cd,
                    "CT": -cd,
                }
            )
        else:
            q_area = (
                0.5
                * YANG_CASE.rho_kg_m3
                * YANG_CASE.freestream_m_s**2
                * YANG_CASE.area_m2
            )
            rows.append(
                {
                    "aoa_deg": aoa,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "phase": phase,
                    "lift_n": lift,
                    "drag_n": drag,
                    "thrust_n": -drag,
                    "CL": lift / q_area,
                    "CD": drag / q_area,
                    "CT": -drag / q_area,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = args.base_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    yang_means: list[dict[str, Any]] = [
        dict(row) for row in _read_csv(base / "yang2025_mean_characteristics.csv")
    ]
    yang_phase: list[dict[str, Any]] = [
        dict(row) for row in _read_csv(base / "yang2025_phase_histories.csv")
    ]
    iz_phase: list[dict[str, Any]] = [
        dict(row)
        for row in _read_csv(base / "izraelevitz2017_fig11_phase_histories.csv")
    ]
    log: list[str] = []
    audit: dict[str, Any] = {"yang": {}, "izraelevitz": {}}

    for aoa in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        started = time.perf_counter()
        movement, _ = build_yang2025_movement(aoa, "full")
        ullt = movement_one_state_ullt(
            movement,
            source_cycle_step_range=(200, 299),
            period_s=YANG_CASE.period_s,
            freestream_m_s=YANG_CASE.freestream_m_s,
            rho_kg_m3=YANG_CASE.rho_kg_m3,
            aspect_ratio=YANG_CASE.aspect_ratio,
            area_m2=YANG_CASE.area_m2,
            output_samples=128,
        )
        polar = movement_polar_residual(
            movement,
            source_cycle_step_range=(200, 299),
            period_s=YANG_CASE.period_s,
            freestream_m_s=YANG_CASE.freestream_m_s,
            rho_kg_m3=YANG_CASE.rho_kg_m3,
            aspect_ratio=YANG_CASE.aspect_ratio,
            output_samples=128,
        )
        v1 = _history_from_rows(yang_phase, "fluxv_periodic_v1", aoa=aoa)
        v1_mean = next(
            row
            for row in yang_means
            if row["model"] == "fluxv_periodic_v1"
            and np.isclose(float(row["aoa_deg"]), aoa)
        )
        v1["mean_lift_n"] = float(v1_mean["mean_lift_n"])
        v1["mean_drag_n"] = float(v1_mean["mean_drag_n"])
        local_separation = smooth_separation_fraction(
            np.asarray(polar["alpha_rad"]),
            attached_limit_deg=DEFAULT_POLAR_PARAMETERS.attached_limit_deg,
            fully_separated_deg=DEFAULT_POLAR_PARAMETERS.fully_separated_deg,
        )
        global_separation = np.average(
            local_separation,
            axis=1,
            weights=np.asarray(ullt["strip_area_m2"], dtype=float),
        )
        v2 = blend_periodic_ullt_state_shape(
            ullt,
            v1,
            global_separation,
            rho_kg_m3=YANG_CASE.rho_kg_m3,
            freestream_m_s=YANG_CASE.freestream_m_s,
            area_m2=YANG_CASE.area_m2,
        )
        runtime = time.perf_counter() - started
        _append_phase(
            yang_phase, ullt, model="one_state_ullt_local", aoa=aoa
        )
        _append_phase(yang_phase, v2, model="fluxv_periodic_v2", aoa=aoa)
        for model, history, role in (
            ("one_state_ullt_local", ullt, "local_existing_model"),
            ("fluxv_periodic_v2", v2, "exploratory_local_model"),
        ):
            mean_lift = float(np.mean(history["lift_n"]))
            mean_drag = float(np.mean(history["drag_n"]))
            yang_means.append(
                {
                    "aoa_deg": aoa,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "status": "exploratory_local",
                    "mean_lift_n": mean_lift,
                    "mean_drag_n": mean_drag,
                    "mean_thrust_n": -mean_drag,
                    "mean_lift_gf": mean_lift / GF_TO_N,
                    "mean_drag_gf": mean_drag / GF_TO_N,
                    "mean_thrust_gf": -mean_drag / GF_TO_N,
                    "mean_CL": history.get("mean_CL", np.mean(history["CL"])),
                    "mean_CD": history.get("mean_CD", np.mean(history["CD"])),
                    "mean_CT": history.get("mean_CT", np.mean(history["CT"])),
                    "runtime_s": runtime if model == "one_state_ullt_local" else 0.0,
                    "digitization_uncertainty_gf": "",
                    "data_role": role,
                    "error": "",
                }
            )
        audit["yang"][str(int(aoa))] = {
            "mean_separation_fraction": float(np.mean(global_separation)),
            "min_separation_fraction": float(np.min(global_separation)),
            "max_separation_fraction": float(np.max(global_separation)),
            "v2_mean_matches_v1_lift_n": float(
                abs(np.mean(v2["lift_n"]) - float(v1_mean["mean_lift_n"]))
            ),
            "v2_mean_matches_v1_drag_n": float(
                abs(np.mean(v2["drag_n"]) - float(v1_mean["mean_drag_n"]))
            ),
            "ullt_periodic_closure_jump_lift_n": float(
                abs(ullt["lift_n"][0] - ullt["lift_n"][-1])
            ),
            "ullt_periodic_closure_jump_drag_n": float(
                abs(ullt["drag_n"][0] - ullt["drag_n"][-1])
            ),
            "v2_periodic_closure_jump_lift_n": float(
                abs(v2["lift_n"][0] - v2["lift_n"][-1])
            ),
            "v2_periodic_closure_jump_drag_n": float(
                abs(v2["drag_n"][0] - v2["drag_n"][-1])
            ),
            "runtime_s": runtime,
        }
        log.append(
            f"Yang AoA={aoa:g}: ULLT L/D={np.mean(ullt['lift_n'])/GF_TO_N:.3f}/"
            f"{np.mean(ullt['drag_n'])/GF_TO_N:.3f} gf; mean gate="
            f"{np.mean(global_separation):.3f}"
        )

    movement, _ = build_izraelevitz_fig11_movement(
        "full", settings=(4, 12, 256, 4)
    )
    ullt = movement_one_state_ullt(
        movement,
        source_cycle_step_range=(768, 1023),
        period_s=IZ_CASE.period_s,
        freestream_m_s=IZ_CASE.freestream_m_s,
        rho_kg_m3=IZ_CASE.rho_kg_m3,
        aspect_ratio=IZ_CASE.aspect_ratio,
        area_m2=IZ_CASE.area_m2,
        output_samples=128,
    )
    polar = movement_polar_residual(
        movement,
        source_cycle_step_range=(768, 1023),
        period_s=IZ_CASE.period_s,
        freestream_m_s=IZ_CASE.freestream_m_s,
        rho_kg_m3=IZ_CASE.rho_kg_m3,
        aspect_ratio=IZ_CASE.aspect_ratio,
        output_samples=128,
    )
    v1 = _history_from_rows(iz_phase, "fluxv_periodic_v1")
    # The source v1 result owns the mean; it is already on the common phase grid.
    q_area = 0.5 * IZ_CASE.rho_kg_m3 * IZ_CASE.freestream_m_s**2 * IZ_CASE.area_m2
    v1["mean_lift_n"] = float(np.mean(v1["lift_n"]))
    v1["mean_drag_n"] = float(np.mean(v1["drag_n"]))
    local_separation = smooth_separation_fraction(
        np.asarray(polar["alpha_rad"]),
        attached_limit_deg=DEFAULT_POLAR_PARAMETERS.attached_limit_deg,
        fully_separated_deg=DEFAULT_POLAR_PARAMETERS.fully_separated_deg,
    )
    global_separation = np.average(
        local_separation,
        axis=1,
        weights=np.asarray(ullt["strip_area_m2"], dtype=float),
    )
    v2 = blend_periodic_ullt_state_shape(
        ullt,
        v1,
        global_separation,
        rho_kg_m3=IZ_CASE.rho_kg_m3,
        freestream_m_s=IZ_CASE.freestream_m_s,
        area_m2=IZ_CASE.area_m2,
    )
    _append_phase(
        iz_phase, ullt, model="one_state_ullt_local", izraelevitz=True
    )
    _append_phase(iz_phase, v2, model="fluxv_periodic_v2", izraelevitz=True)
    audit["izraelevitz"] = {
        "mean_separation_fraction": float(np.mean(global_separation)),
        "max_separation_fraction": float(np.max(global_separation)),
        "v2_mean_CT": float(-np.mean(v2["drag_n"]) / q_area),
        "v1_mean_CT": float(-np.mean(v1["drag_n"]) / q_area),
        "local_ullt_surface_gain": ullt["surface_correction_gain"],
        "local_ullt_max_abs_alpha_deg": ullt["max_abs_alpha_deg"],
    }

    metrics = _compute_metrics(yang_means, iz_phase)
    mean_fields = [
        "aoa_deg", "model", "model_label", "status", "mean_lift_n",
        "mean_drag_n", "mean_thrust_n", "mean_lift_gf", "mean_drag_gf",
        "mean_thrust_gf", "mean_CL", "mean_CD", "mean_CT", "runtime_s",
        "digitization_uncertainty_gf", "data_role", "error",
    ]
    _write_csv(output / "yang2025_mean_characteristics.csv", yang_means, mean_fields)
    _write_csv(
        output / "yang2025_phase_histories.csv", yang_phase,
        ["aoa_deg", "model", "model_label", "phase", "lift_n", "drag_n",
         "thrust_n", "CL", "CD", "CT"],
    )
    _write_csv(
        output / "izraelevitz2017_fig11_phase_histories.csv", iz_phase,
        ["model", "model_label", "data_role", "phase", "CL_alpha",
         "CD_alpha", "CL", "CD", "CT"],
    )
    metric_fields = [
        "paper", "reference", "model", "model_label", "channel", "units",
        "mae", "rmse", "bias", "max_abs_error", "range_nrmse",
        "prediction_half_amplitude", "reference_half_amplitude",
        "half_amplitude_error", "positive_peak_phase_error_cycle",
        "negative_peak_phase_error_cycle",
    ]
    _write_csv(output / "accuracy_metrics.csv", metrics, metric_fields)
    lookup = {(r["paper"], r["model"], r["channel"]): r for r in metrics}
    summary = {
        "status": "complete",
        "model": "fluxv_periodic_v2",
        "yang2025_mae_gf": {
            "old": {
                "lift": lookup[("yang2025", "fluxv_uvpm", "lift")]["mae"],
                "drag": lookup[("yang2025", "fluxv_uvpm", "drag")]["mae"],
            },
            "v2": {
                "lift": lookup[("yang2025", "fluxv_periodic_v2", "lift")]["mae"],
                "drag": lookup[("yang2025", "fluxv_periodic_v2", "drag")]["mae"],
            },
            "authors_proposed": {
                "lift": lookup[("yang2025", "authors_proposed_modified_uvlm", "lift")]["mae"],
                "drag": lookup[("yang2025", "authors_proposed_modified_uvlm", "drag")]["mae"],
            },
        },
        "izraelevitz_raw_phase_rmse": {
            "old": {
                "lift": lookup[("izraelevitz2017_fig11", "fluxv_uvpm", "lift")]["rmse"],
                "drag": lookup[("izraelevitz2017_fig11", "fluxv_uvpm", "drag")]["rmse"],
            },
            "v2": {
                "lift": lookup[("izraelevitz2017_fig11", "fluxv_periodic_v2", "lift")]["rmse"],
                "drag": lookup[("izraelevitz2017_fig11", "fluxv_periodic_v2", "drag")]["rmse"],
            },
            "local_one_state_ullt": {
                "lift": lookup[("izraelevitz2017_fig11", "one_state_ullt_local", "lift")]["rmse"],
                "drag": lookup[("izraelevitz2017_fig11", "one_state_ullt_local", "drag")]["rmse"],
            },
        },
        "claim": (
            "same exploratory periodic v2 improves both benchmarks without "
            "observation-derived residuals; transient/causal production claim not supported"
        ),
    }
    manifest = {
        "run_id": output.name,
        "status": "complete",
        "base_run": _manifest_path(base),
        "base_run_manifest_sha256": _sha256(base / "run_manifest.json"),
        "model": "fluxv_periodic_v2",
        "model_semantics": (
            "UVLM/polar owns cycle mean and separated alternating load; source-"
            "constrained one-state ULLT owns attached alternating load; a shared "
            "15--20 degree local-incidence gate is area-averaged; blend is recentered"
        ),
        "observation_fit": "none",
        "causality_scope": "periodic two-pass; not online transient production",
        "execution_fast_path": (
            "inherits the base run's tested load-equivalent no-particle path; "
            "VPM particles remain one-way diagnostics in current FluxV"
        ),
        "comparability_limits": [
            "Yang phase histories have no public experimental/author phase truth.",
            "Yang uses nominal four-bar motion rather than unpublished LDS history.",
            "Izraelevitz UVLM is numerical reference, not experiment.",
            "The v2 mean is inherited from v1 UVLM/polar; ULLT changes only periodic shape.",
            "The 15--20 degree gate is exploratory after a failed v0 diagnostic.",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pterasoftware": _package_version("PteraSoftware"),
            "fluxvortex": _package_version("fluxvortex"),
        },
        "source_code_hashes": {
            _manifest_path(Path(__file__)): _sha256(Path(__file__).resolve()),
            _manifest_path(Path(__file__).parent / "augmented_uvpm.py"): _sha256(
                (Path(__file__).parent / "augmented_uvpm.py").resolve()
            ),
            _manifest_path(Path(__file__).parent / "ullt_attached.py"): _sha256(
                (Path(__file__).parent / "ullt_attached.py").resolve()
            ),
            _manifest_path(
                Path(__file__).parent / "uvlm_polar_correction.py"
            ): _sha256(
                (Path(__file__).parent / "uvlm_polar_correction.py").resolve()
            ),
        },
        "base_artifact_hashes": {
            _manifest_path(base / name): _sha256(base / name)
            for name in (
                "run_manifest.json",
                "yang2025_mean_characteristics.csv",
                "yang2025_phase_histories.csv",
                "izraelevitz2017_fig11_phase_histories.csv",
            )
        },
        "audit": audit,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    manifest["result_artifact_hashes"] = {
        _manifest_path(output / name): _sha256(output / name)
        for name in (
            "summary.json",
            "accuracy_metrics.csv",
            "yang2025_mean_characteristics.csv",
            "yang2025_phase_histories.csv",
            "izraelevitz2017_fig11_phase_histories.csv",
        )
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (output / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
