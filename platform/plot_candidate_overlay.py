"""Plot non-destructive Fig. 17/18/19 measurement/candidate overlays.

The input is a direct condition-key mapping produced by
``lb_sweep_candidate.py``.  Raw measurements are parsed from ``docs/data.md``
through the canonical benchmark module.  Every output is created beside the
candidate JSON with a no-overwrite filename.  For closures that emit matched
V4.1 forces, ``--same-call-counterfactual`` adds that baseline directly from
the same records without a second solver run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

import fig171819_benchmark as benchmark  # noqa: E402


PANEL_LAYOUT = {
    "17": (1, 2, ("a", "b")),
    "18": (2, 2, ("a", "b", "c", "d")),
    "19": (2, 2, ("a", "b", "c", "d")),
}
CHANNEL_LABEL = {"T": "Thrust (N)", "L": "Lift (N)"}
ABSCISSA_LABEL = {
    "twist_deg": "Nominal twist amplitude (deg)",
    "frequency_Hz": "Flapping frequency (Hz)",
}


def _read_results(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), Mapping):
        raw = raw["results"]
    if not isinstance(raw, Mapping):
        raise ValueError("candidate JSON must contain a condition-key mapping")
    return dict(raw)


def _same_call_counterfactual_results(
    results: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Extract an atomic V4.1 counterfactual sweep embedded in candidate rows."""

    output: dict[str, dict[str, float]] = {}
    lift_key = "L_wind_v41_counterfactual"
    thrust_key = "T_wind_v41_counterfactual"
    for condition_key, value in results.items():
        if not _valid_result(value):
            continue
        assert isinstance(value, Mapping)
        present = (lift_key in value, thrust_key in value)
        if not all(present):
            raise ValueError(
                f"{condition_key}: complete same-call V4.1 "
                "counterfactual L/T is required"
            )
        try:
            lift = float(value[lift_key])
            thrust = float(value[thrust_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{condition_key}: invalid same-call counterfactual L/T"
            ) from exc
        if not math.isfinite(lift) or not math.isfinite(thrust):
            raise ValueError(
                f"{condition_key}: non-finite same-call counterfactual L/T"
            )
        output[str(condition_key)] = {"L": lift, "T": thrust}
    if not output:
        raise ValueError("candidate JSON contains no same-call V4.1 counterfactual")
    return output


def _inferred_candidate_label(source: Path) -> str:
    config_path = source.parent / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            candidate_id = config["run_identity"]["candidate_id"]
            if isinstance(candidate_id, str) and candidate_id:
                return candidate_id
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if source.parent.parent.name == "runs":
        return source.parent.parent.parent.name
    return source.parent.parent.name


def _valid_result(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return math.isfinite(float(value["L"])) and math.isfinite(float(value["T"]))
    except (KeyError, TypeError, ValueError):
        return False


def _non_overwriting_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return an unused path without deleting or replacing any existing file."""

    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index:02d}{suffix}"
        index += 1
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _series_label(curve: benchmark.CurveSpec) -> str:
    parameter = curve.key.split("|", 2)[-1]
    if curve.figure == "17":
        return f"f={parameter} Hz"
    if curve.figure == "18" and curve.panel in {"a", "b"}:
        return f"U={parameter} m/s"
    if curve.figure == "18":
        U, freq = parameter.strip("()").split(",")
        return f"U={float(U):g}, f={float(freq):g}"
    return f"AoA={float(parameter):g}°"


def _curve_model(
    curve: benchmark.CurveSpec, results: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for abscissa, condition in zip(curve.x, curve.conditions):
        record = results.get(benchmark.condition_key(condition))
        if _valid_result(record):
            x.append(float(abscissa))
            y.append(float(record[curve.channel]))
    return x, y


def _plot_figure(
    figure_id: str,
    *,
    results: Mapping[str, Any],
    measurements: Mapping[str, benchmark.MeasurementCurve],
    output_dir: Path,
    candidate_label: str,
    baseline_results: Mapping[str, Any] | None,
    baseline_label: str,
    dpi: int,
) -> Path:
    rows, cols, panels = PANEL_LAYOUT[figure_id]
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(8.2 * cols, 5.5 * rows),
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    plotted_model_points = 0
    plotted_curves = 0

    for panel_index, panel in enumerate(panels):
        ax = axes.flat[panel_index]
        curves = [
            curve
            for curve in benchmark.CURVES
            if curve.figure == figure_id and curve.panel == panel
        ]
        for series_index, curve in enumerate(curves):
            color = colors[series_index % len(colors)]
            measurement = measurements[curve.key]
            label = _series_label(curve)
            ax.plot(
                measurement.x,
                measurement.values_N,
                color=color,
                linewidth=1.8,
                marker="x",
                markersize=5,
                alpha=0.75,
                label=label,
            )
            model_x, model_y = _curve_model(curve, results)
            if baseline_results is not None:
                baseline_x, baseline_y = _curve_model(curve, baseline_results)
                if baseline_x:
                    ax.plot(
                        baseline_x,
                        baseline_y,
                        color=color,
                        linestyle=":",
                        linewidth=1.6,
                        alpha=0.85,
                    )
            if model_x:
                ax.plot(
                    model_x,
                    model_y,
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                )
                plotted_model_points += len(model_x)
                plotted_curves += 1
        example = curves[0]
        ax.set(
            title=f"({panel}) {CHANNEL_LABEL[example.channel]}",
            xlabel=ABSCISSA_LABEL[example.abscissa],
            ylabel=CHANNEL_LABEL[example.channel],
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="best")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            marker="x",
            label="Measured (raw digitization)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markerfacecolor="white",
            label=candidate_label,
        ),
    ]
    if baseline_results is not None:
        legend_handles.insert(
            1,
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                linewidth=1.6,
                label=baseline_label,
            ),
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(legend_handles),
        frameon=False,
    )
    fig.suptitle(
        f"Fig. {figure_id}: measured vs {candidate_label} "
        f"({plotted_curves} partial/complete model curves, "
        f"{plotted_model_points} plotted model points)",
        fontsize=13,
        y=0.995,
    )
    # Keep title, figure-level line-style legend, and top-row panel titles in
    # separate vertical bands.  Long candidate labels otherwise overlap all
    # three in the 2x2 Fig. 18/19 layout.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.885))
    output = _non_overwriting_path(
        output_dir, f"fig{figure_id}_candidate_overlay", ".png"
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def generate_overlays(
    candidate_json: Path | str,
    *,
    data_md: Path | str = benchmark.DEFAULT_DATA_MD,
    candidate_label: str | None = None,
    baseline_json: Path | str | None = None,
    baseline_label: str | None = None,
    same_call_counterfactual: bool = False,
    dpi: int = 160,
) -> tuple[Path, ...]:
    source = Path(candidate_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    measurement_path = Path(data_md).expanduser().resolve()
    results = _read_results(source)
    if baseline_json is not None and same_call_counterfactual:
        raise ValueError(
            "baseline_json and same_call_counterfactual are mutually exclusive"
        )
    baseline_source = (
        Path(baseline_json).expanduser().resolve()
        if baseline_json is not None
        else None
    )
    if baseline_source is not None and not baseline_source.is_file():
        raise FileNotFoundError(baseline_source)
    baseline_results = (
        _read_results(baseline_source)
        if baseline_source is not None
        else (
            _same_call_counterfactual_results(results)
            if same_call_counterfactual
            else None
        )
    )
    resolved_baseline_label = baseline_label or (
        "V4.1 same-call counterfactual"
        if same_call_counterfactual
        else "V4.1 frozen"
    )
    measurements = benchmark.load_measurements(measurement_path)
    validation = benchmark.validate_measurement_contract(
        measurements, source_path=measurement_path
    )
    if not validation["passed"]:
        raise ValueError(f"measurement contract failed: {validation}")
    label = candidate_label or _inferred_candidate_label(source)
    outputs = tuple(
        _plot_figure(
            figure,
            results=results,
            measurements=measurements,
            output_dir=source.parent,
            candidate_label=label,
            baseline_results=baseline_results,
            baseline_label=resolved_baseline_label,
            dpi=dpi,
        )
        for figure in ("17", "18", "19")
    )
    manifest_path = _non_overwriting_path(
        source.parent, "candidate_overlay_manifest", ".json"
    )
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_json": str(source),
        "candidate_json_sha256": _sha256(source),
        "candidate_label": label,
        "baseline_json": (
            str(baseline_source) if baseline_source is not None else None
        ),
        "baseline_json_sha256": (
            _sha256(baseline_source) if baseline_source is not None else None
        ),
        "baseline_label": (
            resolved_baseline_label if baseline_results is not None else None
        ),
        "baseline_source_role": (
            "same_call_v41_counterfactual"
            if same_call_counterfactual
            else ("external_json" if baseline_source is not None else None)
        ),
        "same_call_counterfactual_source_sha256": (
            _sha256(source) if same_call_counterfactual else None
        ),
        "measurement_contract": validation,
        "valid_candidate_condition_count": sum(
            _valid_result(value) for value in results.values()
        ),
        "outputs": [str(path) for path in outputs],
        "no_overwrite_policy": "allocate_numeric_suffix_if_target_exists",
    }
    # The selected path is unused; exclusive creation closes the race without
    # replacing any file owned by a previous visualization run.
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_json")
    parser.add_argument("--data-md", default=str(benchmark.DEFAULT_DATA_MD))
    parser.add_argument("--candidate-label")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-json")
    baseline.add_argument(
        "--same-call-counterfactual",
        action="store_true",
        help=(
            "plot the V4.1 counterfactual embedded atomically in each "
            "candidate result record"
        ),
    )
    parser.add_argument("--baseline-label")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        outputs = generate_overlays(
            args.candidate_json,
            data_md=args.data_md,
            candidate_label=args.candidate_label,
            baseline_json=args.baseline_json,
            baseline_label=args.baseline_label,
            same_call_counterfactual=args.same_call_counterfactual,
            dpi=args.dpi,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
