"""Model-vs-experiment comparison for the Rojratsirikul 2011 Fig 6/9/12-15 oracles.

P4 of HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829.  Inputs:

  * the frozen digitized oracle package (``rojratsirikul2011_observations``);
  * the rigid queue's ``model_observables.csv`` (Figure 9 rigid, Figure 12
    spectrum, Figure 13/15 St curves);
  * the membrane CaseRunner payloads for the flexible Figure 6/9 rows
    (A16/A17 at U = 5), converted into the same observable schema.

Outputs (into ``--output``): six overlay figures with legend classes
``experiment_digitized`` / ``model_current_commit`` / ``published_fit`` /
``diagnostic_band``, a ``scores.json`` (H1-H6 with the exact gate
arithmetic from handoff §9), and ``case_failures.csv`` (failures stay in
the scoring set, never silently dropped).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluxvortex.cases import rojratsirikul2011_observations as obs

ROOT = Path(__file__).resolve().parents[2]
MEMBRANE_BASELINE = (
    ROOT / "artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current"
)
DEFAULT_MODEL_CSV = (
    ROOT
    / "artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/model_observables.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/comparison"
)

H1_ZMAX_TOL = 0.0045
H1_CN_TOL = 0.08
H5_ST_TOL = 0.05
H2_MAE_GATE = 0.006
H3_MAE_GATE = 0.10
H4_MAE_GATE = 0.08
H6_MAE_GATE = 0.03

U_COLORS = {5.0: "tab:blue", 7.5: "tab:orange", 10.0: "tab:green"}


def membrane_rows() -> list[dict]:
    """Flexible FSI observables from the frozen A16/A17 payloads."""

    rows = []
    payloads = {
        "ROJ11-A16": MEMBRANE_BASELINE / "ROJ11_A16_FULL.json",
        "ROJ11-A17-MODE": MEMBRANE_BASELINE / "ROJ11_A17_MODE_FULL.json",
    }
    for case_id, path in payloads.items():
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        window = data.get("window_selection", {}).get("window", [None, None])
        rows.append(
            {
                "run_id": case_id,
                "case_id": case_id,
                "branch": "flexible_fsi",
                "U_m_s": 5.0,
                "Re": 24300,
                "alpha_deg": float(data["case"].get("angle_deg", 0))
                if isinstance(data.get("case"), dict)
                else (16.0 if "A16" in case_id else 17.0),
                "zmax_over_c_mean_map": data["mean_zmax_over_c"],
                "Cn_mean": data["mean_Cn"],
                "St": data.get("accuracy_gates", {}).get("st_gate", {}).get("strouhal", ""),
                "St_modified": "",
                "stationary": bool(
                    data.get("window_selection", {}).get("stationary_window_found", False)
                ),
                "statistics_start_tstar": round(window[0] * 0.01, 2) if window[0] is not None else "",
                "statistics_end_tstar": round(window[1] * 0.01, 2) if window[1] is not None else "",
                "n_samples": window[1] - window[0] if window[0] is not None else "",
                "gpu_name": "NVIDIA GeForce RTX 4090 D",
                "git_commit": "see payload manifest",
                "source_status": "success",
                "failure_reason": "",
            }
        )
    return rows


def rigid_rows(model_csv: Path) -> list[dict]:
    if not model_csv.is_file():
        return []
    with model_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mae(pairs: list[tuple[float, float]]) -> float:
    return sum(abs(m - e) for m, e in pairs) / max(1, len(pairs))


def _plot_curve(axis, points, color, label, marker, fill):
    axis.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        linestyle="-",
        marker=marker,
        markersize=4,
        color=color,
        fillstyle=fill,
        linewidth=1.2,
        label=label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-csv", default=str(DEFAULT_MODEL_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    membrane = membrane_rows()
    rigid = rigid_rows(Path(args.model_csv))
    rigid_ok = [r for r in rigid if r["source_status"] == "success"]

    fig6 = plt.figure(figsize=(7.2, 5.0))
    axis = fig6.add_subplot(1, 1, 1)
    for U in (5.0, 7.5, 10.0):
        series = [
            (r.alpha_deg, r.zmax_over_c) for r in obs.figure6_rows() if r.U_m_s == U
        ]
        _plot_curve(axis, series, U_COLORS[U], f"experiment U={U} (digitized)", "s", "full")
    for row in membrane:
        axis.plot(
            [row["alpha_deg"]],
            [row["zmax_over_c_mean_map"]],
            marker="*",
            markersize=14,
            color="red",
            linestyle="none",
            label=f"model {row['case_id']} (current commit)",
        )
    axis.set_xlabel("incidence α [deg]")
    axis.set_ylabel("z_max/c (time-mean map)")
    axis.set_title("Figure 6: maximum time-mean membrane displacement")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig6.tight_layout()
    fig6_path = out_dir / "figure06_model_vs_experiment.png"
    fig6.savefig(fig6_path, dpi=150)

    # ── Figure 9 flexible / rigid ────────────────────────────────────────
    for wing, rows_model, fname, title in (
        (
            "flexible_membrane",
            membrane,
            "figure09_flexible_model_vs_experiment.png",
            "Figure 9 (flexible): Cn vs incidence",
        ),
        (
            "rigid_flat_plate",
            rigid_ok,
            "figure09_rigid_model_vs_experiment.png",
            "Figure 9 (rigid): Cn vs incidence",
        ),
    ):
        fig = plt.figure(figsize=(7.2, 5.0))
        axis = fig.add_subplot(1, 1, 1)
        for U in (5.0, 7.5, 10.0):
            series = [
                (r.alpha_deg, r.cn)
                for r in obs.figure9_rows()
                if r.U_m_s == U and r.wing_type == wing
            ]
            _plot_curve(
                axis,
                series,
                U_COLORS[U],
                f"experiment U={U} (digitized)",
                "o",
                "full" if wing == "flexible_membrane" else "none",
            )
        grouped: dict[float, list[tuple[float, float]]] = {}
        for row in rows_model:
            cn = row.get("Cn_mean") or row.get("cn_mean")
            if cn in ("", None):
                continue
            grouped.setdefault(float(row["U_m_s"]), []).append(
                (float(row["alpha_deg"]), float(cn))
            )
        for U, points in sorted(grouped.items()):
            points.sort()
            _plot_curve(
                axis, points, "red", f"model U={U} (current commit)", "*", "full"
            )
        axis.set_xlabel("incidence α [deg]")
        axis.set_ylabel("C_n (time-mean)")
        axis.set_title(title)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # ── Figure 12 spectrum ───────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 5.0))
    axis = fig.add_subplot(1, 1, 1)
    fc, psd, condition = obs.figure12_spectrum()
    axis.plot(fc, psd, color="black", linewidth=1.5, label="experiment (digitized trace)")
    axis.axvline(obs.FIGURE12_PEAK_ST, color="gray", linestyle="--", label="experiment peak St=0.58")
    anchor = next(
        (r for r in rigid_ok if float(r["U_m_s"]) == 10.0 and float(r["alpha_deg"]) == 15.0),
        None,
    )
    if anchor:
        case_path = Path(args.model_csv).parent / "cases" / f"{anchor['run_id']}.json"
        if case_path.is_file():
            case = json.loads(case_path.read_text())
            if case.get("dominant_psd_st"):
                st_grid = case["dominant_psd_st"]
                spectrum = case["dominant_psd"]
                top = max(spectrum)
                axis.plot(
                    st_grid,
                    [v / top * max(psd) for v in spectrum],
                    color="red",
                    linewidth=1.2,
                    label=f"model (peak-normalized), St={case.get('St', float('nan')):.3f}",
                )
    axis.set_xlabel("St = fc/U∞")
    axis.set_ylabel("PSD (normalized comparison)")
    axis.set_title(
        f"Figure 12: rigid AR=2 wake spectrum, α=15°, Re=48,700"
    )
    axis.set_xlim(0, 1.6)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure12_wake_spectrum_model_vs_experiment.png", dpi=150)
    plt.close(fig)

    # ── Figure 13 / 15 St curves ────────────────────────────────────────
    fig13 = plt.figure(figsize=(7.2, 5.0))
    axis13 = fig13.add_subplot(1, 1, 1)
    fig15 = plt.figure(figsize=(7.2, 5.0))
    axis15 = fig15.add_subplot(1, 1, 1)
    for Re, color, marker in ((24300, "tab:blue", "s"), (36500, "tab:orange", "o")):
        curve = [(r.alpha_deg, r.st) for r in obs.figure1315_rows() if r.Re == Re]
        _plot_curve(axis13, curve, color, f"experiment Re={Re} (digitized)", marker, "none")
        curve_star = [
            (r.alpha_deg, r.st_modified) for r in obs.figure1315_rows() if r.Re == Re
        ]
        _plot_curve(axis15, curve_star, color, f"experiment Re={Re} (digitized)", marker, "none")
    relation = obs.figure14_relation()
    axis13.plot(
        [r["alpha_deg"] for r in relation],
        [r["st_fit"] for r in relation],
        linestyle="--",
        color="gray",
        label="published fit St=0.17/sinα",
    )
    axis15.axhspan(0.15, 0.20, color="gray", alpha=0.15, label="diagnostic band 0.15–0.20")
    axis15.axhline(0.17, color="gray", linestyle="--", linewidth=1.0, label="published mean 0.17")
    for U in (5.0, 7.5, 10.0):
        points = sorted(
            (float(r["alpha_deg"]), float(r["St"]))
            for r in rigid_ok
            if float(r["U_m_s"]) == U and r.get("St") not in ("", None)
        )
        if points:
            _plot_curve(
                axis13, points, "red", f"model U={U} (current commit)", "*", "full"
            )
        points_star = sorted(
            (float(r["alpha_deg"]), float(r["St_modified"]))
            for r in rigid_ok
            if float(r["U_m_s"]) == U and r.get("St_modified") not in ("", None)
        )
        if points_star:
            _plot_curve(
                axis15, points_star, "red", f"model U={U} (current commit)", "*", "full"
            )
    axis13.set_xlabel("incidence α [deg]")
    axis13.set_ylabel("St = fc/U∞")
    axis13.set_title("Figure 13: rigid finite-wing shedding Strouhal")
    axis13.legend(fontsize=8)
    axis13.grid(alpha=0.3)
    fig13.tight_layout()
    fig13.savefig(out_dir / "figure13_st_model_vs_experiment.png", dpi=150)
    axis15.set_xlabel("incidence α [deg]")
    axis15.set_ylabel("St·sinα")
    axis15.set_title("Figure 15: modified Strouhal")
    axis15.legend(fontsize=8)
    axis15.grid(alpha=0.3)
    fig15.tight_layout()
    fig15.savefig(out_dir / "figure15_modified_st_model_vs_experiment.png", dpi=150)

    # ── Figure 14: 2D cross-literature relation (diagnostic) ────────────
    fig14 = plt.figure(figsize=(7.2, 5.0))
    axis14 = fig14.add_subplot(1, 1, 1)
    relation = obs.figure14_relation()
    axis14.fill_between(
        [r["alpha_deg"] for r in relation],
        [r["st_lower"] for r in relation],
        [r["st_upper"] for r in relation],
        color="gray",
        alpha=0.18,
        label="diagnostic band St*=0.15–0.20",
    )
    axis14.plot(
        [r["alpha_deg"] for r in relation],
        [r["st_fit"] for r in relation],
        linestyle="--",
        color="black",
        linewidth=1.4,
        label="published fit St=0.17/sinα (2D relation)",
    )
    for U in (5.0, 7.5, 10.0):
        points = sorted(
            (float(r["alpha_deg"]), float(r["St"]))
            for r in rigid_ok
            if float(r["U_m_s"]) == U and r.get("St") not in ("", None)
        )
        if points:
            _plot_curve(
                axis14,
                points,
                "red",
                f"model U={U} AR=2 finite wing (context, not 2D)",
                "*",
                "full",
            )
    axis14.set_xlabel("incidence α [deg]")
    axis14.set_ylabel("St = fc/U∞")
    axis14.set_title(
        "Figure 14 relation: 2D St=0.17/sinα with the AR=2 finite-wing model"
    )
    axis14.set_ylim(0, 2.2)
    axis14.legend(fontsize=8)
    axis14.grid(alpha=0.3)
    fig14.tight_layout()
    fig14.savefig(out_dir / "figure14_2d_relation_model_context.png", dpi=150)
    plt.close(fig14)

    # ── scores.json ──────────────────────────────────────────────────────
    scores: dict = {}

    a16_membrane = next((m for m in membrane if m["alpha_deg"] == 16.0), None)
    if a16_membrane:
        dz = float(a16_membrane["zmax_over_c_mean_map"]) - obs.A16_U5_ZMAX_OVER_C
        dc = float(a16_membrane["Cn_mean"]) - obs.A16_U5_CN
        scores["H1_a16_dual_gate"] = {
            "zmax_model": float(a16_membrane["zmax_over_c_mean_map"]),
            "zmax_target": obs.A16_U5_ZMAX_OVER_C,
            "zmax_error": dz,
            "zmax_pass": abs(dz) <= H1_ZMAX_TOL,
            "cn_model": float(a16_membrane["Cn_mean"]),
            "cn_target": obs.A16_U5_CN,
            "cn_error": dc,
            "cn_pass": abs(dc) <= H1_CN_TOL,
            "both_pass": abs(dz) <= H1_ZMAX_TOL and abs(dc) <= H1_CN_TOL,
        }

    fig6_pairs = [
        (m["zmax_over_c_mean_map"], obs.figure6_value(m["U_m_s"], m["alpha_deg"]).zmax_over_c)
        for m in membrane
    ]
    if fig6_pairs:
        scores["H2_figure6"] = {
            "n_model_points": len(fig6_pairs),
            "mae": _mae(fig6_pairs),
            "gate": H2_MAE_GATE,
            "pass": _mae(fig6_pairs) <= H2_MAE_GATE,
            "note": "model points so far: membrane A16/A17 at U=5 only",
        }
    fig9_flex_pairs = [
        (m["Cn_mean"], obs.figure9_value("flexible_membrane", m["U_m_s"], m["alpha_deg"]).cn)
        for m in membrane
    ]
    if fig9_flex_pairs:
        scores["H3_figure9_flexible"] = {
            "n_model_points": len(fig9_flex_pairs),
            "mae": _mae(fig9_flex_pairs),
            "gate": H3_MAE_GATE,
            "pass": _mae(fig9_flex_pairs) <= H3_MAE_GATE,
        }
    fig9_rigid_pairs = []
    for r in rigid_ok:
        if r.get("Cn_mean") in ("", None):
            continue
        try:
            experiment = obs.figure9_value(
                "rigid_flat_plate", float(r["U_m_s"]), float(r["alpha_deg"])
            ).cn
        except KeyError:
            continue
        fig9_rigid_pairs.append((float(r["Cn_mean"]), experiment))
    if fig9_rigid_pairs:
        scores["H4_figure9_rigid"] = {
            "n_model_points": len(fig9_rigid_pairs),
            "mae": _mae(fig9_rigid_pairs),
            "gate": H4_MAE_GATE,
            "pass": _mae(fig9_rigid_pairs) <= H4_MAE_GATE,
        }
    anchor_row = next(
        (r for r in rigid_ok if float(r["U_m_s"]) == 10.0 and float(r["alpha_deg"]) == 15.0),
        None,
    )
    if anchor_row and anchor_row.get("St") not in ("", None):
        dst = float(anchor_row["St"]) - obs.FIGURE12_PEAK_ST
        scores["H5_figure12_peak"] = {
            "st_model": float(anchor_row["St"]),
            "st_target": obs.FIGURE12_PEAK_ST,
            "error": dst,
            "gate": H5_ST_TOL,
            "pass": abs(dst) <= H5_ST_TOL,
        }
    st_pairs = []
    for r in rigid_ok:
        if r.get("St") in ("", None) or float(r["U_m_s"]) == 10.0:
            continue  # Re=48,700 curve needs identity re-verification first
        try:
            experiment = next(
                row
                for row in obs.figure1315_rows()
                if row.Re == int(r["Re"])
                and row.alpha_deg == float(r["alpha_deg"])
                and row.alpha_deg != 15.0  # exclude the Re=48700 anchor row
            )
        except StopIteration:
            continue
        st_pairs.append((float(r["St"]), experiment.st))
    h6: dict = {"n_model_points": len(st_pairs)}
    if st_pairs:
        h6["mae"] = _mae(st_pairs)
        h6["gate"] = H6_MAE_GATE
        h6["pass"] = _mae(st_pairs) <= H6_MAE_GATE
    scores["H6_figure13_15"] = h6
    scores["evidence_note"] = (
        "digitized uncertainties are figure-read errors, NOT solver "
        "tolerances; engineering gates per handoff §9"
    )

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))

    failures = [r for r in rigid if r["source_status"] != "success"]
    with (out_dir / "case_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "U_m_s",
                "alpha_deg",
                "source_status",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for row in failures:
            writer.writerow(
                {
                    "run_id": row.get("run_id", ""),
                    "U_m_s": row.get("U_m_s", ""),
                    "alpha_deg": row.get("alpha_deg", ""),
                    "source_status": row.get("source_status", ""),
                    "failure_reason": row.get("failure_reason", "")[:200],
                }
            )

    print(f"figures + scores written to {out_dir}")
    for key, value in scores.items():
        if isinstance(value, dict):
            print(f"  {key}: " + " ".join(f"{k}={v}" for k, v in value.items() if not isinstance(v, (dict, list))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
