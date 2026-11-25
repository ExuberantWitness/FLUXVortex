"""Sweep the ring kernel's regularisation constant and plot the error against the experiment.

The question
------------
The ring kernel's denominator constant (kernels_ring_velocity.py, `|a x b|^2 + eps`,
frozen at 1e-9 m^4) is not inert on the Rojratsirikul et al. (2011) wing and inflates the
solved circulation (note A19).  Every stored error against the experiment was produced
with it in place.  This script varies ONLY that constant, through `FLUXV_RING_EPS_V`,
on the production solver, and draws the error against the digitised experiment for
each case as a function of the constant.

What "error" means here
----------------------
  rigid plate : Cn along the wing normal (the paper's convention), mean over the last
                1200 of 1500 steps (t* 3 .. 15), against Figure 9.
  membrane    : Cn = (Fx sin a + Fz cos a)/(q S) and the mean zmax/c over a FIXED window of
                the per-step records, against Figures 9 and 6.

For the membrane the runner's own gates are not used: on a run shorter than the
statistics window they fall back to every record, startup transient included.  The
window is taken from the records here instead, so every point of the curve is measured
the same way.

One constant is shared by every case.  Nothing else varies: the child processes get the
production defaults for every other switch, and the sweep neutralises any such variable
it finds in its own environment so that a stray export cannot confound a point.

How it runs
-----------
The parent never touches CUDA.  Each (constant, case, length) is one child process with
`FLUXV_RING_EPS_V` set before anything is imported, strictly one at a time, in the
order 3 steps -> 150 steps -> the full length.  A result that already exists is skipped,
so an interrupted sweep resumes.  A child that fails, or a run whose |Cn| exceeds 5,
stops the whole sweep: nothing is carried forward from a bad point.  The plot and the
table are rewritten after every constant completes.

Usage
-----
    python3 platform/warp_vpm/sweep_ring_eps_vs_experiment.py --stage rigid \\
        --eps 1e-12,1e-10,3e-11,3e-10,1e-11 --alphas 5,10,16 --steps 1500
    python3 platform/warp_vpm/sweep_ring_eps_vs_experiment.py --stage membrane \\
        --eps 1e-9,1e-10,3e-11,1e-12 --cases ROJ11-A10,ROJ11-A16,ROJ11-A23 --steps 650 \\
        --window 4.0,6.5
    python3 platform/warp_vpm/sweep_ring_eps_vs_experiment.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))

OUT_ROOT = REPO_ROOT / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/eps_sweep"
# Baselines produced with the frozen constant before this script existed
# (visualize_roj_midspan_flow_rigid.py, 1500 steps, window t* 3..15).
EXISTING_RIGID = (
    REPO_ROOT
    / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/observations/fig7_rigid"
)
EXISTING_MEMBRANE = REPO_ROOT / "artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current"

CHILD_ENV = {
    "PYTHONPATH": "src:platform:platform/warp_vpm",
    "PFIELD_DEVICE": "cuda:0",
    "FLUXV_GPU_ONLY": "1",
    "FLUXV_V5M_FUSE": "1",
    # The packet's closure check has no absolute floor and tripped on floating-point noise
    # at a near-zero load (first steps of the startup ramp, rigid plate alpha = 10 deg,
    # eps = 1e-10: "moment closure 1.212e-19 exceeds 1e-12 x |M|").  With "l1" the
    # tolerance is the forward-error bound of the reduction (load_packet.py).  Recorded
    # with every point; it changes verdicts only, never a number.
    "FLUXV_PACKET_CLOSURE_SCALE": "l1",
}
# Removed from the child environment so that every point runs the production defaults.
NEUTRALISED = (
    "FLUXV_LESP_MASK_OFF", "FLUXV_LE_SHEET_EXTENT", "FLUXV_LE_SHEET_EXTENT_PANELS",
    "FLUXV_LE_PIN", "FLUXV_JOINT_SEPARATION", "FLUXV_ZERO_INDUCED",
    "FLUXV_REUSE_CONVECTED_FIELD", "FLUXV_THREE_SLOT_ENDPOINT_ACCELERATION",
)
CURRENT_CONSTANT = 1.0e-9
DIVERGENCE_LIMIT = 5.0


class SweepStop(RuntimeError):
    """A point failed: the sweep stops and says why."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu_python_processes() -> list[str]:
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout
    return [line.strip() for line in listing.splitlines() if "python" in line.lower()]


def run_child(script: str, arguments: list[str], eps_text: str, log_path: Path) -> int:
    environment = dict(os.environ)
    environment.update(CHILD_ENV)
    for name in NEUTRALISED:
        environment.pop(name, None)
    environment["FLUXV_RING_EPS_V"] = eps_text
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  eps={eps_text}  {script} {' '.join(arguments)}\n"
        )
        handle.flush()
        return subprocess.call(
            [sys.executable, script, *arguments],
            cwd=str(REPO_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT,
        )


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ── rigid plate ────────────────────────────────────────────────────────────
def do_rigid(eps_text: str, alpha: float, stages: list[int]) -> None:
    folder = OUT_ROOT / f"rigid_a{alpha:g}_eps{eps_text}"
    for index, steps in enumerate(stages):
        final = index == len(stages) - 1
        out = folder / ("result.json" if final else f"result_s{steps}.json")
        if out.exists():
            log(f"skip rigid a={alpha:g} eps={eps_text} steps={steps} (exists)")
            continue
        log(f"run  rigid a={alpha:g} eps={eps_text} steps={steps}")
        started = time.time()
        code = run_child(
            "platform/warp_vpm/diagnose_rigid_plate_coefficients.py",
            ["--alpha", str(alpha), "--steps", str(steps),
             "--average-last", str(min(1200, max(steps - 300, 1))), "--out", str(out)],
            eps_text, folder / "log.txt",
        )
        if code != 0 or not out.exists():
            raise SweepStop(f"rigid a={alpha:g} eps={eps_text} steps={steps}: exit {code}, see {folder / 'log.txt'}")
        result = json.loads(out.read_text())
        check = result["kernel_check"]
        if abs(check["self_fused_over_exact"] - check["self_torch_over_exact"]) > 1e-6:
            raise SweepStop(f"rigid a={alpha:g} eps={eps_text}: fused kernel and torch path differ")
        peak = result["max_abs_cn_whole_run"]
        if not finite(peak) or peak >= DIVERGENCE_LIMIT:
            raise SweepStop(f"rigid a={alpha:g} eps={eps_text} steps={steps}: max|Cn| = {peak}")
        if not finite(result["cn_normal"]["mean"]):
            raise SweepStop(f"rigid a={alpha:g} eps={eps_text} steps={steps}: non-finite window mean")
        log(f"done rigid a={alpha:g} eps={eps_text} steps={steps}: Cn {result['cn_normal']['mean']:+.4f}, "
            f"max|Cn| {peak:.4f}, particles {result['particles_last']}, {time.time() - started:.0f} s")


# ── membrane ───────────────────────────────────────────────────────────────
def membrane_window(payload: dict, low: float, high: float) -> list[dict]:
    return [
        record for record in payload["records"]
        if low <= float(record["time_star"]) <= high
    ]


def membrane_metrics(
    payload: dict, low: float, high: float, payload_path: Path | None = None,
    zmax_slice: tuple[int, int] | None = None,
) -> dict | None:
    records = membrane_window(payload, low, high)
    if not records:
        return None
    alpha = math.radians(float(payload["alpha_deg"]))
    q_s = payload["force_coefficient_conventions"]["q_times_s"]
    normal = [
        (float(r["total_aerodynamic_force_n"][0]) * math.sin(alpha)
         + float(r["total_aerodynamic_force_n"][2]) * math.cos(alpha)) / q_s
        for r in records
    ]
    peak = max(abs(float(r["cn"])) for r in payload["records"])
    metrics = {
        "cn_normal": float(np.mean(normal)), "cn_normal_min": float(np.min(normal)),
        "cn_normal_max": float(np.max(normal)),
        # the mean of the per-step maximum: NOT what the paper plots, kept for reference
        "zmax_instant_mean": float(np.mean([float(r["instantaneous_zmax_over_c"]) for r in records])),
        "max_abs_cn": peak, "window_samples": len(records), "alpha_deg": payload["alpha_deg"],
    }
    # The paper's zmax is the maximum of the TIME-AVERAGED membrane shape, which is also
    # what the runner's own gate reports (`zmax_over_c_mean_map`).  It is formed here on
    # the sweep's window from the stored per-step displacement history, so that a 650-step
    # point and a full-length point are measured the same way.  On the stored A16
    # baselines (t* >= 20) this reproduces the gate's 0.044806 / 0.04420.
    history_path = payload_path.with_name(payload_path.stem + ".z_history.npz") if payload_path else None
    if history_path is not None and history_path.exists():
        with np.load(history_path, allow_pickle=False) as data:
            history, time_star = np.asarray(data["z_history_over_c"], dtype=float), np.asarray(data["time_star"], dtype=float)
        if zmax_slice is not None:
            # an explicit index window: what the runner's own gate used on a full-length run
            metrics["zmax_over_c"] = float(history[zmax_slice[0] : zmax_slice[1]].mean(axis=0).max())
            metrics["zmax_window"] = f"records {zmax_slice[0]}..{zmax_slice[1]}"
        else:
            mask = (time_star >= low) & (time_star <= high)
            if mask.any():
                metrics["zmax_over_c"] = float(history[mask].mean(axis=0).max())
                metrics["zmax_window"] = f"t* {low:g}..{high:g}"
    return metrics


def do_membrane(eps_text: str, case: str, stages: list[int], low: float, high: float) -> None:
    folder = OUT_ROOT / f"membrane_{case}_eps{eps_text}"
    for index, steps in enumerate(stages):
        final = index == len(stages) - 1
        out = folder / ("payload.json" if final else f"payload_s{steps}.json")
        if out.exists():
            log(f"skip membrane {case} eps={eps_text} steps={steps} (exists)")
            continue
        log(f"run  membrane {case} eps={eps_text} steps={steps}")
        started = time.time()
        code = run_child(
            "platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py",
            ["--case", case, "--max-aero-steps", str(steps), "--output", str(out)],
            eps_text, folder / "log.txt",
        )
        # The production entry exits 2 when the run COMPLETED and only its accuracy gate
        # failed (result_status.exit_code), which is the expected outcome of every point of
        # this sweep.  A missing payload, or an execution that did not complete, is the
        # failure; the exit code alone is not.
        payload = json.loads(out.read_text()) if out.exists() else None
        status = (payload or {}).get("result_status") or {}
        if (
            payload is None
            or status.get("execution") != "completed"
            or int(status.get("exit_code", 1)) not in (0, 2)
        ):
            raise SweepStop(
                f"membrane {case} eps={eps_text} steps={steps}: exit {code}, "
                f"result_status {status}, see {folder / 'log.txt'}"
            )
        peak = max(abs(float(r["cn"])) for r in payload["records"])
        if not finite(peak) or peak >= DIVERGENCE_LIMIT:
            raise SweepStop(f"membrane {case} eps={eps_text} steps={steps}: max|Cn| = {peak}")
        log(f"done membrane {case} eps={eps_text} steps={steps}: max|Cn| {peak:.4f}, "
            f"{len(payload['records'])} steps, exit {code} "
            f"(accuracy {status.get('accuracy')}, physics {status.get('physics')}, "
            f"numerical {status.get('numerical')}), {time.time() - started:.0f} s")
        if final:
            # the statistics window is a property of the point, not of the payload;
            # store it next to the payload so the collector never has to guess it
            (folder / "window.json").write_text(
                json.dumps({"low": low, "high": high, "eps_text": eps_text}) + "\n", encoding="utf-8"
            )


# ── targets and collection ─────────────────────────────────────────────────
def targets(wing: str, alpha: float) -> dict:
    from fluxvortex.cases.rojratsirikul2011_observations import figure6_value, figure9_value

    row = {"cn": figure9_value(wing, 5.0, float(alpha)).cn}
    if wing == "flexible_membrane":
        row["zmax_over_c"] = figure6_value(5.0, float(alpha)).zmax_over_c
    return row


def collect() -> list[dict]:
    rows: list[dict] = []

    # seeds: rigid baselines at the frozen constant, from the stored field files
    for alpha in (5.0, 10.0, 15.0, 16.0):
        source = EXISTING_RIGID / f"RIGID_A{alpha:g}_t15_off.field.npz"
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as data:
            series = np.asarray(data["cn_series"], dtype=float)
            start = int(data["start_average"])
        window = series[start - 1:]
        target = targets("rigid_flat_plate", alpha)
        rows.append({
            "kind": "rigid", "case": f"rigid {alpha:g} deg", "alpha_deg": alpha,
            "eps": CURRENT_CONSTANT, "steps": int(series.size), "window": f"t* {(start - 1) * 0.01:.2f}..{series.size * 0.01:.2f}",
            "cn_model": float(window.mean()), "cn_target": target["cn"],
            "zmax_model": None, "zmax_target": None,
            "particles_last": None, "max_abs_cn": float(np.abs(series).max()),
            "elapsed_s": None, "status": "existing", "source": str(source.relative_to(REPO_ROOT)),
            "closure_scale": "net",
        })

    for folder in sorted(OUT_ROOT.glob("rigid_a*_eps*")):
        out = folder / "result.json"
        if not out.exists():
            continue
        result = json.loads(out.read_text())
        alpha = float(result["alpha_deg"])
        rows.append({
            "kind": "rigid", "case": f"rigid {alpha:g} deg", "alpha_deg": alpha,
            "eps": float(result["kernel_check"]["eps_torch_path"]), "steps": int(result["steps"]),
            "window": f"steps {result['window'][0]}..{result['window'][1]}",
            "cn_model": result["cn_normal"]["mean"], "cn_target": result["paper_cn_rigid"],
            "zmax_model": None, "zmax_target": None,
            "particles_last": result["particles_last"], "max_abs_cn": result["max_abs_cn_whole_run"],
            "elapsed_s": result["elapsed_seconds"], "status": "sweep",
            "source": str(out.relative_to(REPO_ROOT)),
            "closure_scale": result.get("environment", {}).get("FLUXV_PACKET_CLOSURE_SCALE", "net"),
        })

    for folder in sorted(OUT_ROOT.glob("membrane_*_eps*")):
        out = folder / "payload.json"
        if not out.exists():
            continue
        payload = json.loads(out.read_text())
        steps = len(payload["records"])
        window_file = folder / "window.json"
        if window_file.exists():
            saved = json.loads(window_file.read_text())
            low, high = float(saved["low"]), float(saved["high"])
        else:
            # A scheduler stopped between the end of a run and its window record leaves this.
            # The point is unfinished: it is left out, never given a guessed window, and it
            # must not stop the other stages of the sweep from running.
            log(f"{folder.name}: payload without window.json; point not collected")
            continue
        metrics = membrane_metrics(payload, low, high, out)
        if metrics is None:
            continue
        target = targets("flexible_membrane", metrics["alpha_deg"])
        if "zmax_over_c" not in metrics:
            raise RuntimeError(f"{out}: no displacement history next to the payload; zmax is not measurable")
        rows.append({
            "kind": "membrane", "case": payload["case_id"], "alpha_deg": metrics["alpha_deg"],
            "eps": float(folder.name.split("_eps")[1]), "steps": steps, "window": f"t* {low}..{high}",
            "cn_model": metrics["cn_normal"], "cn_target": target["cn"],
            "zmax_model": metrics["zmax_over_c"], "zmax_target": target["zmax_over_c"],
            "particles_last": None, "max_abs_cn": metrics["max_abs_cn"],
            "elapsed_s": payload.get("elapsed_seconds"), "status": "sweep",
            "source": str(out.relative_to(REPO_ROOT)),
            "closure_scale": "l1",
            "gates": {
                key: (payload.get("result_status") or {}).get(key)
                for key in ("accuracy", "physics", "numerical", "execution")
            },
        })

    # seeds: full-length membrane baselines at the frozen constant (force components stored)
    # Only the legacy-pin baseline: that is the arm the sweep runs.
    for name, case in (
        ("ROJ11_A16_FULL_20261123_CNCONV.json", "ROJ11-A16"),
    ):
        source = EXISTING_MEMBRANE / name
        if not source.exists():
            continue
        payload = json.loads(source.read_text())
        # The runner's zmax gate does not use t* >= 20: on these runs its window selection
        # picked records ~299..2100 (t* ~3..21), and the reported "+3.3%" is that number.
        # The gate's own window is reproduced here for the seed, and labelled.
        gate_window = (payload.get("window_selection") or {}).get("window")
        metrics = membrane_metrics(
            payload, 20.0, 21.0, source, tuple(int(v) for v in gate_window) if gate_window else None
        )
        target = targets("flexible_membrane", metrics["alpha_deg"])
        if "zmax_over_c" not in metrics:
            continue
        pin = "legacy" if "CNCONV" in name else "signed"
        rows.append({
            "kind": "membrane", "case": f"{case} ({pin}, full length)", "alpha_deg": metrics["alpha_deg"],
            "eps": CURRENT_CONSTANT, "steps": len(payload["records"]),
            "window": f"Cn t* 20..21; zmax {metrics.get('zmax_window', '?')}",
            "cn_model": metrics["cn_normal"], "cn_target": target["cn"],
            "zmax_model": metrics["zmax_over_c"], "zmax_target": target["zmax_over_c"],
            "particles_last": None, "max_abs_cn": metrics["max_abs_cn"],
            "elapsed_s": payload.get("elapsed_seconds"), "status": "existing",
            "source": str(source.relative_to(REPO_ROOT)),
        })

    for row in rows:
        row["cn_err_pct"] = 100.0 * (row["cn_model"] / row["cn_target"] - 1.0)
        row["zmax_err_pct"] = (
            100.0 * (row["zmax_model"] / row["zmax_target"] - 1.0)
            if row["zmax_model"] is not None else None
        )
    return rows


def write_table(rows: list[dict]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.setdefault("closure_scale", "net")
    path = OUT_ROOT / "eps_sweep_table.csv"
    columns = ["kind", "case", "alpha_deg", "eps", "steps", "window", "cn_model", "cn_target", "cn_err_pct",
               "zmax_model", "zmax_target", "zmax_err_pct", "particles_last", "max_abs_cn", "elapsed_s",
               "status", "closure_scale", "gates", "source"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["kind"], r["case"], r["eps"], r["steps"])):
            writer.writerow(row)
    return path


def rank(rows: list[dict]) -> dict[str, list[tuple[float, list[dict]]]]:
    """Mean |Cn error| over the sweep cases available at each constant, by family.

    Only points of this sweep count (stored baselines are a different run length for the
    membrane), and only constants at which every family member has exactly one point.
    """

    ranking: dict[str, list[tuple[float, list[dict]]]] = {"rigid": [], "membrane": [], "all": []}
    for family in ranking:
        table: dict[float, list[dict]] = {}
        for row in rows:
            if row["status"] != "sweep":
                continue
            if family != "all" and row["kind"] != family:
                continue
            table.setdefault(row["eps"], []).append(row)
        for eps, group in table.items():
            cases = [r["case"] for r in group]
            if len(cases) != len(set(cases)):
                continue
            ranking[family].append((float(np.mean([abs(r["cn_err_pct"]) for r in group])), group))
    return {family: sorted(value, key=lambda item: item[0]) for family, value in ranking.items()}


def plot(rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 9.0), sharex=True)
    cases = sorted({(r["kind"], r["case"]) for r in rows})
    colours = plt.get_cmap("tab20")
    for index, (kind, case) in enumerate(cases):
        points = sorted(
            (r for r in rows if r["kind"] == kind and r["case"] == case and r["eps"] > 0.0),
            key=lambda r: r["eps"],
        )
        if not points:
            continue
        style = dict(
            color=colours(index % 20), marker="o" if kind == "rigid" else "s",
            linestyle="-" if any(p["status"] == "sweep" for p in points) else "--",
            linewidth=1.6, markersize=5, label=case,
        )
        x = [p["eps"] for p in points]
        axes[0].plot(x, [p["cn_err_pct"] for p in points], **style)
        zpoints = [p for p in points if p["zmax_err_pct"] is not None]
        if zpoints:
            axes[1].plot([p["eps"] for p in zpoints], [p["zmax_err_pct"] for p in zpoints], **style)

    for axis, title in zip(axes, ("Cn along the chord normal", "zmax/c")):
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.axvline(CURRENT_CONSTANT, color="grey", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xlim(3.0e-13, 4.0e-9)   # fixed, so the axes do not jump as points arrive
        axis.set_ylabel("model / experiment - 1  (%)")
        axis.grid(True, alpha=0.3)
        axis.set_title(title, fontsize=10)
    axes[0].annotate("current constant 1e-9", xy=(CURRENT_CONSTANT, axes[0].get_ylim()[1]), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color="grey", va="top")
    axes[1].set_xlabel("ring kernel denominator constant  eps  (m^4);  smaller = closer to the Biot-Savart law;  1e-9 is the frozen value")
    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    if unique:
        axes[0].legend(unique.values(), list(unique), fontsize=8, ncol=2, loc="upper left")
    figure.suptitle(
        "Error against the digitised experiment (Rojratsirikul et al. 2011, U = 5 m/s) "
        "vs the ring kernel constant\nsolid = this sweep (production solver), dashed = earlier full-length baseline",
        fontsize=10,
    )
    figure.tight_layout()
    path = OUT_ROOT / "eps_sweep_error_vs_constant.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def report(rows: list[dict]) -> None:
    print(f"\n{'case':30s} {'eps':>7} {'steps':>5} {'Cn':>8} {'target':>7} {'err':>8} {'zmax':>8} {'target':>7} {'err':>8} {'src':>9}")
    for row in sorted(rows, key=lambda r: (r["kind"], r["case"], r["eps"], r["steps"])):
        z = f"{row['zmax_err_pct']:+7.1f}%" if row["zmax_err_pct"] is not None else "        -"
        zt = f"{row['zmax_target']:7.4f}" if row["zmax_target"] is not None else "      -"
        zm = f"{row['zmax_model']:8.5f}" if row["zmax_model"] is not None else "        -"
        print(f"{row['case'][:30]:30s} {row['eps']:7.0e} {row['steps']:>5} {row['cn_model']:8.4f} "
              f"{row['cn_target']:7.4f} {row['cn_err_pct']:+7.1f}% {zm} {zt} {z} {row['status']:>9}")
    ranking = rank(rows)
    for family in ("rigid", "membrane", "all"):
        if ranking[family]:
            print(f"\none constant shared by the {family} cases of this sweep, mean |Cn error|:")
            for mean_error, group in ranking[family]:
                print(f"  eps {group[0]['eps']:.0e}: {mean_error:5.1f}%   over {sorted({g['case'] for g in group})}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["rigid", "membrane"], default=None)
    parser.add_argument("--eps", default="1e-12,1e-10,3e-11,3e-10,1e-11",
                        help="constants to run, in this order (the frozen 1e-9 is taken from stored baselines)")
    parser.add_argument("--alphas", default="5,10,16")
    parser.add_argument("--cases", default="ROJ11-A10,ROJ11-A16,ROJ11-A23")
    parser.add_argument("--steps", type=int, default=1500, help="full length of each run")
    parser.add_argument("--stages", default="3,150",
                        help="shorter lengths run before the full one, in this order")
    parser.add_argument("--window", default="4.0,6.5",
                        help="statistics window in t* for membrane runs (t* >= 20 runs use 20..21)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    rows = collect()
    if arguments.plot_only or not arguments.stage:
        write_table(rows)
        plot(rows)
        report(rows)
        return 0

    eps_list = [text.strip() for text in arguments.eps.split(",") if text.strip()]
    stages = [int(text) for text in arguments.stages.split(",") if text.strip()] + [arguments.steps]
    busy = gpu_python_processes()
    if busy:
        raise SystemExit(f"another python process is using the GPU: {busy}")
    if arguments.dry_run:
        for eps_text in eps_list:
            if arguments.stage == "rigid":
                for alpha in (float(v) for v in arguments.alphas.split(",")):
                    print(f"would run rigid a={alpha:g} eps={eps_text} stages={stages}")
            else:
                for case in arguments.cases.split(","):
                    print(f"would run membrane {case} eps={eps_text} stages={stages}")
        return 0

    low, high = (float(v) for v in arguments.window.split(","))
    try:
        for eps_text in eps_list:
            if arguments.stage == "rigid":
                for alpha in (float(v) for v in arguments.alphas.split(",")):
                    do_rigid(eps_text, alpha, stages)
            else:
                for case in arguments.cases.split(","):
                    do_membrane(eps_text, case, stages, low, high)
            rows = collect()
            write_table(rows)
            plot(rows)
            report(rows)
            log(f"constant {eps_text} finished; plot and table rewritten in {OUT_ROOT}")
    except SweepStop as stop:
        rows = collect()
        write_table(rows)
        plot(rows)
        report(rows)
        log(f"SWEEP STOPPED: {stop}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
