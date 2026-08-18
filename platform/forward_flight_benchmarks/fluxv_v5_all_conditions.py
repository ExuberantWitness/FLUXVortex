"""Build an auditable long-form data set for every frozen benchmark condition.

The module performs no phase, offset, amplitude, or mean fitting.  It only
normalizes already-frozen outputs onto explicit plotting/scoring contracts.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .baik2012 import BAIK_2012_CASES, sharp_fourier_lowpass


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs/forward_flight_large_pitch/reproductions"
OUTPUT_ROOT = DOCS_ROOT / "fluxv_v5_nextgen_20260814/all_conditions_comparison"

INPUTS = {
    "yang_multimodel": DOCS_ROOT
    / "unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/yang2025_mean_characteristics.csv",
    "yang_v3": DOCS_ROOT
    / "unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full/yang2025_v3_mean_characteristics.csv",
    "yang_v4": DOCS_ROOT
    / "unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_fluxv_v4b_crosspaper_full/yang2025_v4_mean_characteristics.csv",
    "fig14_all": DOCS_ROOT
    / "unified_fluxv_upgrade_20260812/runs/20260812_scherer_fig14_experiment_full/mean_thrust_vs_phase.csv",
    "fig14_v3": DOCS_ROOT
    / "unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full/izraelevitz2017_fig14_v3_mean_thrust.csv",
    "fig14_v4": DOCS_ROOT
    / "unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_fluxv_v4b_crosspaper_full/izraelevitz2017_fig14_v4_mean_thrust.csv",
    "baik_experiment": DOCS_ROOT
    / "baik2012_w1_w4/source_data/baik2012_w1_w4_corrected_total_cl_cd.csv",
    "baik_scored": DOCS_ROOT
    / "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv",
    "v5a_conditions": DOCS_ROOT
    / "fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5a_cache_smoke_frozen/condition_predictions.csv",
    "v5a_phase": DOCS_ROOT
    / "fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5a_cache_smoke_frozen/phase_histories.csv",
    "v5b_gate": DOCS_ROOT
    / "fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5b_force_gate_reproducible/no_lev_current_fluxv_comparison.csv",
}

CURVE_FIELDS = (
    "paper",
    "case_id",
    "view",
    "x_name",
    "x_value",
    "replicate",
    "observable",
    "units",
    "model_id",
    "model_label",
    "data_role",
    "value",
    "uncertainty_minus",
    "uncertainty_plus",
    "uncertainty_kind",
    "canonical_eligible",
    "source_path",
)

METRIC_FIELDS = (
    "paper",
    "scope",
    "case_id",
    "view",
    "model_id",
    "observable",
    "n",
    "aggregation",
    "mae",
    "rmse",
    "bias",
    "max_abs_error",
    "reference_model",
    "rmse_improvement_pct",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _output_key(path: Path, output_root: Path) -> str:
    """Return a portable manifest key for a generated output."""

    return f"output/{path.resolve().relative_to(output_root.resolve()).as_posix()}"


def _curve_row(
    *,
    paper: str,
    case_id: str,
    view: str,
    x_name: str,
    x_value: float,
    observable: str,
    units: str,
    model_id: str,
    model_label: str,
    data_role: str,
    value: float,
    source_path: Path,
    replicate: str = "",
    uncertainty_minus: str | float = "",
    uncertainty_plus: str | float = "",
    uncertainty_kind: str = "",
    canonical_eligible: bool = True,
) -> dict[str, Any]:
    return {
        "paper": paper,
        "case_id": case_id,
        "view": view,
        "x_name": x_name,
        "x_value": float(x_value),
        "replicate": replicate,
        "observable": observable,
        "units": units,
        "model_id": model_id,
        "model_label": model_label,
        "data_role": data_role,
        "value": float(value),
        "uncertainty_minus": uncertainty_minus,
        "uncertainty_plus": uncertainty_plus,
        "uncertainty_kind": uncertainty_kind,
        "canonical_eligible": str(bool(canonical_eligible)).lower(),
        "source_path": _relative(source_path),
    }


def _build_yang(curves: list[dict[str, Any]]) -> None:
    path = INPUTS["yang_multimodel"]
    selected = {
        "wind_tunnel_test": ("experiment", "Wind-tunnel experiment", "experiment"),
        "authors_proposed_modified_uvlm": (
            "authors_proposed_modified_uvlm",
            "Authors' modified UVLM",
            "author_model",
        ),
        "ptera_free_wake_uvlm": (
            "ptera_free_wake_uvlm",
            "Ptera free-wake UVLM",
            "existing_model",
        ),
        "one_state_ullt_local": (
            "one_state_ullt_local",
            "Local one-state ULLT",
            "existing_model",
        ),
    }
    for row in _read_csv(path):
        if row["model"] not in selected:
            continue
        model_id, label, role = selected[row["model"]]
        aoa = float(row["aoa_deg"])
        for observable, field in (("lift", "mean_lift_gf"), ("drag", "mean_drag_gf")):
            uncertainty = 0.4 if model_id == "experiment" else ""
            curves.append(
                _curve_row(
                    paper="yang2025",
                    case_id=f"aoa_{aoa:g}",
                    view="cycle_mean",
                    x_name="installation_aoa_deg",
                    x_value=aoa,
                    observable=observable,
                    units="gf",
                    model_id=model_id,
                    model_label=label,
                    data_role=role,
                    value=float(row[field]),
                    uncertainty_minus=uncertainty,
                    uncertainty_plus=uncertainty,
                    uncertainty_kind=(
                        "digitization_only_not_experimental_ci"
                        if model_id == "experiment"
                        else ""
                    ),
                    source_path=path,
                )
            )

    path = INPUTS["yang_v3"]
    for row in _read_csv(path):
        aoa = float(row["aoa_deg"])
        for observable, field in (("lift", "mean_lift_gf"), ("drag", "mean_drag_gf")):
            curves.append(
                _curve_row(
                    paper="yang2025",
                    case_id=f"aoa_{aoa:g}",
                    view="cycle_mean",
                    x_name="installation_aoa_deg",
                    x_value=aoa,
                    observable=observable,
                    units="gf",
                    model_id="fluxv_v3",
                    model_label="FluxV v3 persistent owner",
                    data_role="local_model",
                    value=float(row[field]),
                    source_path=path,
                )
            )

    path = INPUTS["yang_v4"]
    for row in _read_csv(path):
        aoa = float(row["aoa_deg"])
        for model_id, label, lift_field, drag_field in (
            ("fluxv_old", "FluxV old", "old_fluxv_lift_gf", "old_fluxv_drag_gf"),
            ("fluxv_v1_v2", "FluxV v1/v2", "v1_polar_lift_gf", "v1_polar_drag_gf"),
            ("fluxv_v4b", "FluxV v4b", "v4_lift_gf", "v4_drag_gf"),
        ):
            for observable, field in (("lift", lift_field), ("drag", drag_field)):
                curves.append(
                    _curve_row(
                        paper="yang2025",
                        case_id=f"aoa_{aoa:g}",
                        view="cycle_mean",
                        x_name="installation_aoa_deg",
                        x_value=aoa,
                        observable=observable,
                        units="gf",
                        model_id=model_id,
                        model_label=label,
                        data_role="local_model",
                        value=float(row[field]),
                        source_path=path,
                    )
                )

    path = INPUTS["v5a_conditions"]
    for row in _read_csv(path):
        if row["benchmark"] != "yang2025":
            continue
        aoa = float(row["aoa_deg"])
        for observable, field in (("lift", "v5a_lift_gf"), ("drag", "v5a_drag_gf")):
            curves.append(
                _curve_row(
                    paper="yang2025",
                    case_id=row["case_id"],
                    view="cycle_mean",
                    x_name="installation_aoa_deg",
                    x_value=aoa,
                    observable=observable,
                    units="gf",
                    model_id="fluxv_v5a",
                    model_label="FluxV v5a development proxy",
                    data_role="rejected_development_proxy",
                    value=float(row[field]),
                    canonical_eligible=False,
                    source_path=path,
                )
            )


def _build_fig14(curves: list[dict[str, Any]]) -> None:
    path = INPUTS["fig14_all"]
    mapping = {
        "scherer_1968_experiment": ("experiment", "Scherer experiment", "experiment"),
        "authors_6state_ullt": (
            "authors_6state_ullt",
            "Authors' six-state ULLT",
            "author_model",
        ),
        "authors_1state_ullt": (
            "authors_1state_ullt",
            "Authors' one-state ULLT",
            "author_model",
        ),
        "authors_qs_added_mass": (
            "authors_qs_added_mass",
            "QS + added mass",
            "author_model",
        ),
        "fluxv_uvpm": ("fluxv_old", "FluxV old", "local_model"),
        "fluxv_periodic_v1": ("fluxv_v1_v2", "FluxV v1/v2", "local_model"),
        "one_state_ullt_local": (
            "one_state_ullt_local",
            "Local one-state ULLT",
            "existing_model",
        ),
    }
    for row in _read_csv(path):
        if row["series"] not in mapping:
            continue
        model_id, label, role = mapping[row["series"]]
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        curves.append(
            _curve_row(
                paper="izraelevitz2017_fig14",
                case_id=f"theta_{theta:g}_psi_{psi:g}",
                view="cycle_mean",
                x_name="phase_offset_deg",
                x_value=psi,
                observable="CT",
                units="1",
                model_id=model_id,
                model_label=label,
                data_role=role,
                value=float(row["CT"]),
                replicate=row.get("replicate", "") if model_id == "experiment" else "",
                uncertainty_minus=(
                    row.get("CT_error_minus", "") if model_id == "experiment" else ""
                ),
                uncertainty_plus=(
                    row.get("CT_error_plus", "") if model_id == "experiment" else ""
                ),
                uncertainty_kind=(
                    "digitized_reported_bar_unspecified"
                    if model_id == "experiment"
                    else ""
                ),
                source_path=path,
            )
        )

    path = INPUTS["fig14_v3"]
    for row in _read_csv(path):
        if row["model"] != "fluxv_periodic_v3_persistent":
            continue
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        curves.append(
            _curve_row(
                paper="izraelevitz2017_fig14",
                case_id=f"theta_{theta:g}_psi_{psi:g}",
                view="cycle_mean",
                x_name="phase_offset_deg",
                x_value=psi,
                observable="CT",
                units="1",
                model_id="fluxv_v3",
                model_label="FluxV v3 persistent owner",
                data_role="local_model",
                value=float(row["CT"]),
                source_path=path,
            )
        )

    path = INPUTS["fig14_v4"]
    for row in _read_csv(path):
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        curves.append(
            _curve_row(
                paper="izraelevitz2017_fig14",
                case_id=f"theta_{theta:g}_psi_{psi:g}",
                view="cycle_mean",
                x_name="phase_offset_deg",
                x_value=psi,
                observable="CT",
                units="1",
                model_id="fluxv_v4b",
                model_label="FluxV v4b",
                data_role="local_model",
                value=float(row["v4_CT"]),
                source_path=path,
            )
        )

    path = INPUTS["v5a_conditions"]
    for row in _read_csv(path):
        if row["benchmark"] != "izraelevitz2017_fig14":
            continue
        theta = float(row["theta_max_deg"])
        psi = float(row["phase_offset_deg"])
        curves.append(
            _curve_row(
                paper="izraelevitz2017_fig14",
                case_id=row["case_id"],
                view="cycle_mean",
                x_name="phase_offset_deg",
                x_value=psi,
                observable="CT",
                units="1",
                model_id="fluxv_v5a",
                model_label="FluxV v5a development proxy",
                data_role="rejected_development_proxy",
                value=float(row["v5a_CT"]),
                canonical_eligible=False,
                source_path=path,
            )
        )


def _periodic_interp(
    source_phase: np.ndarray, value: np.ndarray, target: np.ndarray
) -> np.ndarray:
    order = np.argsort(source_phase)
    return np.interp(target, source_phase[order], value[order], period=1.0)


def _build_baik(curves: list[dict[str, Any]]) -> None:
    experiment_path = INPUTS["baik_experiment"]
    experiment_rows = _read_csv(experiment_path)
    experiment: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for case_id in BAIK_2012_CASES:
        selected = [row for row in experiment_rows if row["case"] == case_id]
        selected.sort(key=lambda row: float(row["phase_t_over_T"]))
        selected = [row for row in selected if float(row["phase_t_over_T"]) < 1.0]
        if len(selected) != 400:
            raise ValueError(f"expected 400 unique Baik phases for {case_id}")
        phase = np.asarray([float(row["phase_t_over_T"]) for row in selected])
        cl = np.asarray([float(row["cl"]) for row in selected])
        cd = np.asarray([float(row["cd"]) for row in selected])
        experiment[case_id] = phase, cl, cd
        for view in ("filtered_1hz", "raw_numeric_diagnostic"):
            for observable, values in (("CL", cl), ("CD", cd)):
                for tau, value in zip(phase, values):
                    curves.append(
                        _curve_row(
                            paper="baik2012",
                            case_id=case_id,
                            view=view,
                            x_name="phase_t_over_T",
                            x_value=tau,
                            observable=observable,
                            units="1",
                            model_id="experiment",
                            model_label="Corrected-total experiment",
                            data_role="experiment_already_source_filtered",
                            value=value,
                            source_path=experiment_path,
                        )
                    )

    scored_path = INPUTS["baik_scored"]
    scored = _read_csv(scored_path)
    model_map = {
        "fluxv_old": ("fluxv_old", "FluxV old", "raw_numeric_diagnostic"),
        "fluxv_old_filtered": ("fluxv_old", "FluxV old", "filtered_1hz"),
        "fluxv_v4b": ("fluxv_v4b", "FluxV v4b", "raw_numeric_diagnostic"),
        "fluxv_v4b_filtered": ("fluxv_v4b", "FluxV v4b", "filtered_1hz"),
        "published_standard_theodorsen_digitized": (
            "theodorsen",
            "Published standard Theodorsen",
            "filtered_1hz",
        ),
    }
    for row in scored:
        if row["model"] not in model_map:
            continue
        model_id, label, view = model_map[row["model"]]
        curves.append(
            _curve_row(
                paper="baik2012",
                case_id=row["case_id"],
                view=view,
                x_name="phase_t_over_T",
                x_value=float(row["phase"]),
                observable=row["quantity"],
                units="1",
                model_id=model_id,
                model_label=label,
                data_role=(
                    "published_reference_model"
                    if model_id == "theodorsen"
                    else "local_model"
                ),
                value=float(row["prediction"]),
                source_path=scored_path,
            )
        )

    v5_path = INPUTS["v5a_phase"]
    v5rows = [row for row in _read_csv(v5_path) if row["benchmark"] == "baik2012"]
    for case_id, case in BAIK_2012_CASES.items():
        selected = [row for row in v5rows if row["case_id"] == case_id]
        selected.sort(key=lambda row: float(row["phase"]))
        phase128 = np.asarray([float(row["phase"]) for row in selected])
        if len(phase128) != 128:
            raise ValueError(f"expected 128 v5a phases for {case_id}")
        target = experiment[case_id][0]
        for observable, field in (("CL", "prediction_CL"), ("CD", "prediction_CD")):
            raw = np.asarray([float(row[field]) for row in selected])
            filtered = sharp_fourier_lowpass(
                raw, maximum_harmonic=case.experimental_filter_harmonic
            )
            for view, values in (
                ("raw_numeric_diagnostic", raw),
                ("filtered_1hz", filtered),
            ):
                prediction = _periodic_interp(phase128, values, target)
                for tau, value in zip(target, prediction):
                    curves.append(
                        _curve_row(
                            paper="baik2012",
                            case_id=case_id,
                            view=view,
                            x_name="phase_t_over_T",
                            x_value=tau,
                            observable=observable,
                            units="1",
                            model_id="fluxv_v5a",
                            model_label="FluxV v5a development proxy",
                            data_role="rejected_development_proxy",
                            value=value,
                            canonical_eligible=False,
                            source_path=v5_path,
                        )
                    )


def build_curves() -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    _build_yang(curves)
    _build_fig14(curves)
    _build_baik(curves)
    return curves


def _metric(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    obs = np.asarray(list(observed), dtype=float)
    pred = np.asarray(list(predicted), dtype=float)
    if obs.shape != pred.shape or obs.size == 0 or not np.all(np.isfinite(obs + pred)):
        raise ValueError("metric arrays must be finite, nonempty, and aligned")
    error = pred - obs
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _append_metric(
    rows: list[dict[str, Any]],
    *,
    paper: str,
    scope: str,
    case_id: str,
    view: str,
    model_id: str,
    observable: str,
    observed: Iterable[float],
    predicted: Iterable[float],
    aggregation: str,
    reference_model: str = "",
    reference_rmse: float | None = None,
) -> None:
    values = _metric(observed, predicted)
    rows.append(
        {
            "paper": paper,
            "scope": scope,
            "case_id": case_id,
            "view": view,
            "model_id": model_id,
            "observable": observable,
            "n": len(list(observed))
            if not isinstance(observed, np.ndarray)
            else observed.size,
            "aggregation": aggregation,
            **values,
            "reference_model": reference_model,
            "rmse_improvement_pct": (
                ""
                if reference_rmse is None
                else 100.0 * (reference_rmse - values["rmse"]) / reference_rmse
            ),
        }
    )


def build_metrics(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Yang: exact six-condition cycle means.
    for observable in ("lift", "drag"):
        selected = [
            row
            for row in curves
            if row["paper"] == "yang2025" and row["observable"] == observable
        ]
        truth = {
            float(row["x_value"]): float(row["value"])
            for row in selected
            if row["model_id"] == "experiment"
        }
        by_model: dict[str, dict[float, float]] = defaultdict(dict)
        for row in selected:
            if row["model_id"] != "experiment":
                by_model[row["model_id"]][float(row["x_value"])] = float(row["value"])
        old_rmse = _metric(truth.values(), [by_model["fluxv_old"][x] for x in truth])[
            "rmse"
        ]
        for model_id, values in sorted(by_model.items()):
            common = sorted(set(truth) & set(values))
            _append_metric(
                rows,
                paper="yang2025",
                scope="all_6_aoa",
                case_id="all",
                view="cycle_mean",
                model_id=model_id,
                observable=observable,
                observed=np.asarray([truth[x] for x in common]),
                predicted=np.asarray([values[x] for x in common]),
                aggregation="condition_equal",
                reference_model="fluxv_old",
                reference_rmse=old_rmse,
            )
            for x in common:
                _append_metric(
                    rows,
                    paper="yang2025",
                    scope="condition",
                    case_id=f"aoa_{x:g}",
                    view="cycle_mean",
                    model_id=model_id,
                    observable=observable,
                    observed=np.asarray([truth[x]]),
                    predicted=np.asarray([values[x]]),
                    aggregation="single_condition",
                )

    # Figure 14: primary 14-marker and 12-unique-condition contracts.
    selected = [row for row in curves if row["paper"] == "izraelevitz2017_fig14"]
    experiment = [row for row in selected if row["model_id"] == "experiment"]
    prediction: dict[str, dict[str, float]] = defaultdict(dict)
    for row in selected:
        if row["model_id"] != "experiment":
            prediction[row["model_id"]][row["case_id"]] = float(row["value"])
    for model_id, values in sorted(prediction.items()):
        observations = [row for row in experiment if row["case_id"] in values]
        if not observations:
            continue
        _append_metric(
            rows,
            paper="izraelevitz2017_fig14",
            scope="all_14_markers",
            case_id="all",
            view="cycle_mean",
            model_id=model_id,
            observable="CT",
            observed=np.asarray([float(row["value"]) for row in observations]),
            predicted=np.asarray([values[row["case_id"]] for row in observations]),
            aggregation="observation_equal_replicates_retained",
        )
        for theta in (15.0, 25.0):
            theta_obs = [
                row
                for row in observations
                if row["case_id"].startswith(f"theta_{theta:g}_")
            ]
            if theta_obs:
                _append_metric(
                    rows,
                    paper="izraelevitz2017_fig14",
                    scope=f"theta_{theta:g}_markers",
                    case_id=f"theta_{theta:g}",
                    view="cycle_mean",
                    model_id=model_id,
                    observable="CT",
                    observed=np.asarray([float(row["value"]) for row in theta_obs]),
                    predicted=np.asarray([values[row["case_id"]] for row in theta_obs]),
                    aggregation="observation_equal_replicates_retained",
                )
        grouped_obs: dict[str, list[float]] = defaultdict(list)
        for row in observations:
            grouped_obs[row["case_id"]].append(float(row["value"]))
        common = sorted(grouped_obs)
        _append_metric(
            rows,
            paper="izraelevitz2017_fig14",
            scope="unique_12_conditions",
            case_id="all",
            view="cycle_mean",
            model_id=model_id,
            observable="CT",
            observed=np.asarray([np.mean(grouped_obs[key]) for key in common]),
            predicted=np.asarray([values[key] for key in common]),
            aggregation="unique_condition_equal_replicates_averaged",
        )
        for key in common:
            _append_metric(
                rows,
                paper="izraelevitz2017_fig14",
                scope="condition",
                case_id=key,
                view="cycle_mean",
                model_id=model_id,
                observable="CT",
                observed=np.asarray([np.mean(grouped_obs[key])]),
                predicted=np.asarray([values[key]]),
                aggregation="single_unique_condition",
            )

    # Baik: 400 unique phases, then macro-average the four case metrics.
    for view in ("filtered_1hz", "raw_numeric_diagnostic"):
        for observable in ("CL", "CD"):
            selected = [
                row
                for row in curves
                if row["paper"] == "baik2012"
                and row["view"] == view
                and row["observable"] == observable
            ]
            truth: dict[str, dict[float, float]] = defaultdict(dict)
            predictions: dict[str, dict[str, dict[float, float]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            for row in selected:
                phase = float(row["x_value"])
                if row["model_id"] == "experiment":
                    truth[row["case_id"]][phase] = float(row["value"])
                else:
                    predictions[row["model_id"]][row["case_id"]][phase] = float(
                        row["value"]
                    )
            per_model: dict[str, list[dict[str, float]]] = defaultdict(list)
            for model_id, cases in sorted(predictions.items()):
                for case_id, values in sorted(cases.items()):
                    common = sorted(set(truth[case_id]) & set(values))
                    metric = _metric(
                        [truth[case_id][x] for x in common],
                        [values[x] for x in common],
                    )
                    per_model[model_id].append(metric)
                    rows.append(
                        {
                            "paper": "baik2012",
                            "scope": "case_phase_history",
                            "case_id": case_id,
                            "view": view,
                            "model_id": model_id,
                            "observable": observable,
                            "n": len(common),
                            "aggregation": "phase_equal",
                            **metric,
                            "reference_model": "",
                            "rmse_improvement_pct": "",
                        }
                    )
            old_macro_rmse = np.mean([x["rmse"] for x in per_model["fluxv_old"]])
            for model_id, case_metrics in sorted(per_model.items()):
                rows.append(
                    {
                        "paper": "baik2012",
                        "scope": "macro_4_cases",
                        "case_id": "all",
                        "view": view,
                        "model_id": model_id,
                        "observable": observable,
                        "n": 4 * 400,
                        "aggregation": "mean_of_four_case_metrics",
                        "mae": float(np.mean([x["mae"] for x in case_metrics])),
                        "rmse": float(np.mean([x["rmse"] for x in case_metrics])),
                        "bias": float(np.mean([x["bias"] for x in case_metrics])),
                        "max_abs_error": float(
                            np.max([x["max_abs_error"] for x in case_metrics])
                        ),
                        "reference_model": "fluxv_old",
                        "rmse_improvement_pct": 100.0
                        * (old_macro_rmse - np.mean([x["rmse"] for x in case_metrics]))
                        / old_macro_rmse,
                    }
                )
    return rows


def build_coverage() -> list[dict[str, str]]:
    return [
        {
            "model_id": "fluxv_old",
            "yang2025": "6/6",
            "izraelevitz2017_fig14": "12/12",
            "baik2012": "4/4",
            "status": "scored_baseline",
            "note": "Current FluxV/Ptera prescribed-load channel.",
        },
        {
            "model_id": "fluxv_v4b",
            "yang2025": "6/6",
            "izraelevitz2017_fig14": "12/12",
            "baik2012": "4/4",
            "status": "qualified_current_candidate",
            "note": "Best jointly retained candidate before v5 development.",
        },
        {
            "model_id": "fluxv_v5a",
            "yang2025": "6/6",
            "izraelevitz2017_fig14": "12/12",
            "baik2012": "4/4",
            "status": "cache_proxy_rejected",
            "note": "Kinematic/integrated compatibility proxy; 0/18 promotion gates.",
        },
        {
            "model_id": "fluxv_v5b",
            "yang2025": "not_scored",
            "izraelevitz2017_fig14": "not_scored",
            "baik2012": "not_scored",
            "status": "blocked_before_crosspaper_scoring",
            "note": "No-LEV exact-reduction gate G1 failed; no paper curves exist.",
        },
        {
            "model_id": "ptera_prescribed_wake_uvlm",
            "yang2025": "6/6 duplicate",
            "izraelevitz2017_fig14": "n/a",
            "baik2012": "n/a",
            "status": "omitted_duplicate",
            "note": "Exactly duplicates the FluxV-old load channel in Yang.",
        },
        {
            "model_id": "robofalcon2_coefficient_transfer",
            "yang2025": "6/6 diagnostic",
            "izraelevitz2017_fig14": "n/a",
            "baik2012": "n/a",
            "status": "omitted_cross_domain",
            "note": "Cross-geometry/Re coefficient transfer; omitted from main axes.",
        },
    ]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _find_metric(
    metrics: list[dict[str, Any]],
    *,
    paper: str,
    scope: str,
    view: str,
    model_id: str,
    observable: str,
    field: str,
) -> float:
    selected = [
        row
        for row in metrics
        if row["paper"] == paper
        and row["scope"] == scope
        and row["view"] == view
        and row["model_id"] == model_id
        and row["observable"] == observable
    ]
    if len(selected) != 1:
        raise ValueError(f"metric lookup expected one row, found {len(selected)}")
    return float(selected[0][field])


def build_aggregate_table(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts = (
        ("Yang lift", "yang2025", "all_6_aoa", "cycle_mean", "lift", "mae", "gf", 6),
        ("Yang drag", "yang2025", "all_6_aoa", "cycle_mean", "drag", "mae", "gf", 6),
        (
            "Figure 14 CT (14 markers)",
            "izraelevitz2017_fig14",
            "all_14_markers",
            "cycle_mean",
            "CT",
            "rmse",
            "1",
            14,
        ),
        (
            "Figure 14 CT (12 conditions)",
            "izraelevitz2017_fig14",
            "unique_12_conditions",
            "cycle_mean",
            "CT",
            "rmse",
            "1",
            12,
        ),
        (
            "Baik CL (filtered macro)",
            "baik2012",
            "macro_4_cases",
            "filtered_1hz",
            "CL",
            "rmse",
            "1",
            1600,
        ),
        (
            "Baik CD (filtered macro)",
            "baik2012",
            "macro_4_cases",
            "filtered_1hz",
            "CD",
            "rmse",
            "1",
            1600,
        ),
    )
    output: list[dict[str, Any]] = []
    for label, paper, scope, view, observable, field, units, n in contracts:
        old = _find_metric(
            metrics,
            paper=paper,
            scope=scope,
            view=view,
            model_id="fluxv_old",
            observable=observable,
            field=field,
        )
        v4b = _find_metric(
            metrics,
            paper=paper,
            scope=scope,
            view=view,
            model_id="fluxv_v4b",
            observable=observable,
            field=field,
        )
        v5a = _find_metric(
            metrics,
            paper=paper,
            scope=scope,
            view=view,
            model_id="fluxv_v5a",
            observable=observable,
            field=field,
        )
        output.append(
            {
                "metric": label,
                "error_statistic": field.upper(),
                "units": units,
                "n": n,
                "fluxv_old": old,
                "fluxv_v4b": v4b,
                "fluxv_v5a_proxy": v5a,
                "v4b_over_old": v4b / old,
                "v5a_over_v4b": v5a / v4b,
                "conclusion": "v5a_improves_v4b"
                if v5a < v4b
                else "v5a_regresses_vs_v4b",
            }
        )
    return output


def _write_latex_tables(
    output_root: Path,
    aggregate: list[dict[str, Any]],
    coverage: list[dict[str, str]],
) -> tuple[Path, Path]:
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_tex = table_dir / "table01_aggregate_metrics.tex"
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Metric & Old & v4b & v5a proxy & v5a/v4b & Outcome \\",
        r"\midrule",
    ]
    for row in aggregate:
        lines.append(
            f"{row['metric']} & {float(row['fluxv_old']):.4f} & "
            f"{float(row['fluxv_v4b']):.4f} & {float(row['fluxv_v5a_proxy']):.4f} & "
            f"{float(row['v5a_over_v4b']):.3f} & {row['conclusion'].replace('_', ' ')} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    metric_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coverage_tex = table_dir / "table02_evidence_availability.tex"
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Model & Yang & Figure 14 & Baik & Status \\",
        r"\midrule",
    ]
    for row in coverage[:4]:
        model_label = row["model_id"].replace("_", r"\_")
        lines.append(
            f"{model_label} & {row['yang2025']} & "
            f"{row['izraelevitz2017_fig14']} & {row['baik2012']} & "
            f"{row['status'].replace('_', ' ')} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    coverage_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metric_tex, coverage_tex


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_all(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    curves = build_curves()
    metrics = build_metrics(curves)
    coverage = build_coverage()
    aggregate = build_aggregate_table(metrics)
    data_dir = output_root / "data"
    curve_path = data_dir / "all_conditions_curves.csv"
    metric_path = data_dir / "all_conditions_metrics.csv"
    coverage_path = data_dir / "model_coverage.csv"
    aggregate_path = output_root / "tables/table01_aggregate_metrics.csv"
    _write_csv(curve_path, CURVE_FIELDS, curves)
    _write_csv(metric_path, METRIC_FIELDS, metrics)
    _write_csv(
        coverage_path,
        ("model_id", "yang2025", "izraelevitz2017_fig14", "baik2012", "status", "note"),
        coverage,
    )
    _write_csv(
        aggregate_path,
        (
            "metric",
            "error_statistic",
            "units",
            "n",
            "fluxv_old",
            "fluxv_v4b",
            "fluxv_v5a_proxy",
            "v4b_over_old",
            "v5a_over_v4b",
            "conclusion",
        ),
        aggregate,
    )
    metric_tex, coverage_tex = _write_latex_tables(output_root, aggregate, coverage)
    manifest = {
        "schema_version": 1,
        "condition_contract": {
            "yang2025": "6 cycle-mean installation angles",
            "izraelevitz2017_fig14": "14 experimental markers / 12 unique conditions",
            "baik2012": "W1-W4, 400 unique phases per case",
            "total_frozen_conditions": 22,
        },
        "no_fit_contract": "no phase, amplitude, offset, or mean fitting",
        "v5a_status": "rejected development cache proxy",
        "v5b_status": "blocked_not_scored_after_G1_failure",
        "input_hashes": {_relative(path): sha256(path) for path in INPUTS.values()},
        "source_hashes": {
            _relative(path): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("build_fluxv_v5_all_conditions.py"),
                Path(__file__).with_name("baik2012.py"),
            )
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "pterasoftware": importlib.metadata.version("pterasoftware"),
        },
        "output_hashes": {
            _output_key(path, output_root): sha256(path)
            for path in (
                curve_path,
                metric_path,
                coverage_path,
                aggregate_path,
                metric_tex,
                coverage_tex,
            )
        },
        "row_counts": {
            "curves": len(curves),
            "metrics": len(metrics),
            "coverage": len(coverage),
            "aggregate_table": len(aggregate),
        },
    }
    manifest_path = data_dir / "build_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest
