#!/usr/bin/env python3
"""Benchmark steady-level cruise flight & 1st-order motor power across 8 published avian aircraft.

Runs FLUXVortex's unified v4 GPU UVLM solver (Hirato v2 LEV + connected TE wake v4
+ Squire viscous core + wing slip wall + mixed-precision coalesced Biot-Savart) on:
  1. Cloud Owl (MDPI Appl. Sci. 2023)
  2. RoboFalcon 2.0 (Science Advances 2025) + 7 Hz takeoff peak power check
  3. NPU-Sparrow (Nature Communications 2026) + full 7-speed Source Data curve
  4. Dove (Int. J. Micro Air Vehicles 2018)
  5. Improved High-Speed Dove (MDPI Drones 2024)
  6. RAVEN (Nature 2024 / arXiv:2412.02389)
  7. LisEagle - Adaptive Morphing (npj Robotics 2024) at 8, 10, 12 m/s (neutral vs morphed)
  8. LisEagle - Agile Perching (Nature Communications 2024) + 3-sweep Neutral Point shift
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Set unified v4 solver environment switches BEFORE importing solver modules
_UNIFIED_V4_ENV = {
    "FLUXV_GPU_ONLY": "1",
    "FLUXV_LEV_MODEL": "hirato_v2",
    "FLUXV_TE_WAKE": "v4",
    "FLUXV_CONVECTION_V4": "1",
    "FLUXV_RING_KERNEL": "v3",
    "FLUXV_WAKE_CORE_PANELS": "0.25",
    "FLUXV_WAKE_CORE_AT_COLLOCATION": "1",
    "FLUXV_WING_SLIP_WALL": "0.50",
    "FLUXV_SQUIRE_CORE": "0.02",
    "FLUXV_LESP_UREL": "1",
    "FLUXV_LEV_SEP_SUCTION": "1",
    "FLUXV_RING_MIXED_PRECISION": "1",
    "FLUXV_FAST_RIGID_LOADS": "1",
    "FLUXV_LEV_MAX_ROWS": "512",
    "FLUXV_PACKET_CLOSURE_SCALE": "l1",
}
for _k, _v in _UNIFIED_V4_ENV.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import torch

from fluxvortex.cases.avian_aircraft import AVIAN_AIRCRAFT_REGISTRY, AvianAircraftSpec
from fluxvortex.propulsion.cruise_trim import (
    compute_neutral_point_sweep,
    evaluate_flapping_operating_point,
    solve_fixed_wing_cruise_trim,
    solve_flapping_cruise_trim,
)

LOG = logging.getLogger("avian_cruise")


def _rel_err_pct(sim: float, ref: float | None) -> float | None:
    if ref is None or abs(ref) < 1.0e-9:
        return None
    return 100.0 * (sim - ref) / ref


def run_benchmark(selected_ids: list[str], output_dir: Path, device: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(torch.device(device)),
        "solver_env": {k: os.environ.get(k) for k in sorted(_UNIFIED_V4_ENV)},
        "aircraft": {},
        "extra_benchmarks": {},
    }
    summary_rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for aid in selected_ids:
        spec: AvianAircraftSpec = AVIAN_AIRCRAFT_REGISTRY[aid]
        LOG.info("=== [%s] %s (%s) U=%.2f m/s m=%.3f kg b=%.2f m S=%.4f m^2 ===",
                 aid, spec.name, spec.propulsion_kind, spec.cruise_speed_m_s,
                 spec.mass_kg, spec.span_m, spec.area_m2)

        if spec.propulsion_kind == "flapping":
            trim = solve_flapping_cruise_trim(spec, device=device)
        else:
            trim = solve_fixed_wing_cruise_trim(spec, device=device)

        op = trim.operating_point
        p = op.power
        pub = spec.published_cruise
        err_pct = _rel_err_pct(p.total_elec_power_w, pub.power_w)
        end_err_pct = (
            _rel_err_pct(p.endurance_min, pub.endurance_min)
            if p.endurance_min is not None and pub.endurance_min is not None
            else None
        )

        LOG.info(
            "  TRIMMED: conv=%s it=%d alpha=%.2f deg f=%.2f Hz | L=%.3f/%.3f N (err=%+.2f%%) "
            "T_net=%+.3f N | P_aero=%.2f W -> P_elec=%.2f W (ref=%s, err=%s) | CoT=%.3f | end=%.1f min",
            trim.converged, trim.iterations, op.alpha_deg, op.frequency_hz,
            op.mean_lift_n, op.weight_n, 100.0 * trim.lift_error_rel,
            op.net_thrust_n, p.aero_mech_power_w, p.total_elec_power_w,
            f"{pub.power_w:.2f} W" if pub.power_w is not None else str(pub.power_range_w),
            f"{err_pct:+.1f}%" if err_pct is not None else "n/a",
            p.cost_of_transport,
            p.endurance_min if p.endurance_min is not None else -1.0,
        )

        entry = {
            "aircraft_id": aid,
            "name": spec.name,
            "citation": spec.citation,
            "propulsion_kind": spec.propulsion_kind,
            "mass_kg": spec.mass_kg,
            "span_m": spec.span_m,
            "area_m2": spec.area_m2,
            "cruise_speed_m_s": op.airspeed_m_s,
            "trim": asdict(trim),
            "published_cruise": asdict(pub),
            "power_rel_error_pct": err_pct,
            "endurance_rel_error_pct": end_err_pct,
        }
        results["aircraft"][aid] = entry
        summary_rows.append(
            {
                "aircraft_id": aid,
                "name": spec.name,
                "kind": spec.propulsion_kind,
                "mass_kg": spec.mass_kg,
                "span_m": spec.span_m,
                "U_m_s": op.airspeed_m_s,
                "alpha_deg": round(op.alpha_deg, 2),
                "freq_hz": round(op.frequency_hz, 2),
                "lift_err_pct": round(100.0 * trim.lift_error_rel, 2),
                "thrust_err_pct": round(100.0 * trim.thrust_error_rel, 2),
                "P_aero_W": round(p.aero_mech_power_w, 2),
                "P_elec_W": round(p.total_elec_power_w, 2),
                "P_ref_W": pub.power_w,
                "P_ref_range_W": pub.power_range_w,
                "P_err_pct": round(err_pct, 2) if err_pct is not None else None,
                "CoT": round(p.cost_of_transport, 4),
                "endurance_min": round(p.endurance_min, 1) if p.endurance_min is not None else None,
                "ref_endurance_min": pub.endurance_min,
                "provenance": pub.provenance,
            }
        )

        # Extra published validation benchmarks for specific aircraft
        if aid == "npu_sparrow" and "speed_sweep_source_data" in spec.extra_benchmarks:
            sweep_out = []
            LOG.info("  [NPU-Sparrow] Running 7-point Nature Source Data airspeed curve (4.5 -> 13.2 m/s)...")
            a_guess = op.alpha_deg
            f_guess = op.frequency_hz
            for pt in spec.extra_benchmarks["speed_sweep_source_data"]:
                u_pt = float(pt["airspeed_m_s"])
                if abs(u_pt - spec.cruise_speed_m_s) < 1.0e-6:
                    tr_pt = trim
                else:
                    tr_pt = solve_flapping_cruise_trim(
                        spec,
                        airspeed_m_s=u_pt,
                        cd0_override=float(pt["cd0"]),
                        area_override=float(pt["area_eff"]) if "area_eff" in pt else None,
                        device=device,
                    )
                op_pt = tr_pt.operating_point
                p_sim = op_pt.power.total_elec_power_w
                p_ref = float(pt["power_w"])
                p_err = _rel_err_pct(p_sim, p_ref)
                LOG.info(
                    "    U=%4.1f m/s | alpha=%5.2f deg f=%5.2f Hz | P_sim=%6.2f W vs EXP=%6.3f +/- %.3f W (err=%+5.1f%%)",
                    u_pt, op_pt.alpha_deg, op_pt.frequency_hz, p_sim, p_ref, pt["std_w"], p_err,
                )
                sweep_out.append(
                    {
                        "airspeed_m_s": u_pt,
                        "alpha_deg": op_pt.alpha_deg,
                        "frequency_hz": op_pt.frequency_hz,
                        "lift_err_pct": 100.0 * tr_pt.lift_error_rel,
                        "thrust_err_pct": 100.0 * tr_pt.thrust_error_rel,
                        "P_aero_W": op_pt.power.aero_mech_power_w,
                        "P_elec_W": p_sim,
                        "P_exp_W": p_ref,
                        "P_exp_std_W": float(pt["std_w"]),
                        "rel_error_pct": p_err,
                        "abs_error_W": p_sim - p_ref,
                    }
                )
            mae_w = float(np.mean([abs(r["abs_error_W"]) for r in sweep_out]))
            mape_pct = float(np.mean([abs(r["rel_error_pct"]) for r in sweep_out]))
            results["extra_benchmarks"]["npu_sparrow_speed_sweep"] = {
                "points": sweep_out,
                "mae_w": mae_w,
                "mape_pct": mape_pct,
            }
            LOG.info("  [NPU-Sparrow] 7-point curve MAE = %.2f W, MAPE = %.1f%%", mae_w, mape_pct)

        if aid == "robofalcon_2" and "takeoff_check" in spec.extra_benchmarks:
            tk = spec.extra_benchmarks["takeoff_check"]
            LOG.info("  [RoboFalcon 2.0] Running 7 Hz self-takeoff peak power check (U=%.1f m/s)...", tk["airspeed_m_s"])
            op_tk = evaluate_flapping_operating_point(
                spec,
                alpha_deg=float(tk["alpha_deg"]),
                frequency_hz=float(tk["frequency_hz"]),
                airspeed_m_s=float(tk["airspeed_m_s"]),
                cycles=3,
                wing_profile_cd0=0.045,
                device=device,
            )
            results["extra_benchmarks"]["robofalcon_2_takeoff"] = {
                "airspeed_m_s": op_tk.airspeed_m_s,
                "frequency_hz": op_tk.frequency_hz,
                "alpha_deg": op_tk.alpha_deg,
                "mean_lift_n": op_tk.mean_lift_n,
                "mean_wing_thrust_n": op_tk.mean_wing_thrust_n,
                "mean_elec_power_w": op_tk.power.total_elec_power_w,
                "peak_elec_power_w": op_tk.power.peak_elec_power_w,
                "published_peak_power_w": float(tk["published_peak_power_w"]),
                "peak_rel_error_pct": _rel_err_pct(
                    op_tk.power.peak_elec_power_w, float(tk["published_peak_power_w"])
                ),
            }
            LOG.info(
                "    Takeoff: mean_L=%.2f N, mean_T=%.2f N, P_mean=%.1f W, P_peak=%.1f W (paper: up to %.0f W)",
                op_tk.mean_lift_n, op_tk.mean_wing_thrust_n,
                op_tk.power.total_elec_power_w, op_tk.power.peak_elec_power_w,
                tk["published_peak_power_w"],
            )

        if aid == "liseagle_morphing" and "morphing_speed_sweep" in spec.extra_benchmarks:
            LOG.info("  [LisEagle Morphing] Running 8, 10, 12 m/s neutral vs morphed CoT & power savings...")
            morph_rows = []
            for row in spec.extra_benchmarks["morphing_speed_sweep"]:
                u_m = float(row["airspeed_m_s"])
                n_cfg = row["neutral"]
                m_cfg = row["morphed"]
                tr_n = solve_fixed_wing_cruise_trim(
                    spec,
                    airspeed_m_s=u_m,
                    span_override=n_cfg["span_m"],
                    area_override=n_cfg["area_m2"],
                    sweep_override_deg=n_cfg["sweep_deg"],
                    camber_override_deg=n_cfg["camber_deg"],
                    cd0_area_override_m2=n_cfg["cd0_area_m2"],
                    device=device,
                )
                tr_m = solve_fixed_wing_cruise_trim(
                    spec,
                    airspeed_m_s=u_m,
                    span_override=m_cfg["span_m"],
                    area_override=m_cfg["area_m2"],
                    sweep_override_deg=m_cfg["sweep_deg"],
                    camber_override_deg=m_cfg["camber_deg"],
                    cd0_area_override_m2=m_cfg["cd0_area_m2"],
                    device=device,
                )
                p_n = tr_n.operating_point.power.total_elec_power_w
                p_m = tr_m.operating_point.power.total_elec_power_w
                saving_pct = 100.0 * (p_n - p_m) / max(p_n, 1.0e-9)
                LOG.info(
                    "    U=%4.1f m/s | Neutral: P=%.2f W (CoT=%.4f) | Morphed: P=%.2f W (CoT=%.4f) | Saving=%.1f%% (paper: %.1f%%)",
                    u_m, p_n, tr_n.operating_point.power.cost_of_transport,
                    p_m, tr_m.operating_point.power.cost_of_transport,
                    saving_pct, row["published_saving_pct"],
                )
                morph_rows.append(
                    {
                        "airspeed_m_s": u_m,
                        "neutral_power_w": p_n,
                        "neutral_cot": tr_n.operating_point.power.cost_of_transport,
                        "morphed_power_w": p_m,
                        "morphed_cot": tr_m.operating_point.power.cost_of_transport,
                        "sim_saving_pct": saving_pct,
                        "published_saving_pct": float(row["published_saving_pct"]),
                    }
                )
            results["extra_benchmarks"]["liseagle_morphing_sweep"] = morph_rows

        if aid == "liseagle_perching" and "neutral_point_sweep" in spec.extra_benchmarks:
            LOG.info("  [LisEagle Perching] Running 3D Neutral Point shift across 45 / 87.5 / 130 deg wing sweep...")
            np_res = compute_neutral_point_sweep(spec, device=device)
            results["extra_benchmarks"]["liseagle_neutral_point_sweep"] = np_res
            LOG.info(
                "    Wing AC shift = %.2f cm, Whole-aircraft NP shift = %.2f cm (paper Supp. Table 1: %.1f cm, err=%+.1f%%)",
                np_res["sim_wing_ac_shift_cm"], np_res["sim_np_shift_cm"],
                np_res["published_np_shift_cm"], np_res["np_shift_rel_error_pct"],
            )

    results["total_wall_seconds"] = time.perf_counter() - t_start
    results["summary_table"] = summary_rows

    (output_dir / "avian_cruise_summary.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    if summary_rows:
        with (output_dir / "avian_cruise_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    LOG.info("All %d aircraft finished in %.1f s. Saved to %s",
             len(selected_ids), results["total_wall_seconds"], output_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        default="all",
        help="comma-separated aircraft IDs or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment/20261127_stage2_cruise"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    all_ids = list(AVIAN_AIRCRAFT_REGISTRY.keys())
    selected = all_ids if args.aircraft == "all" else [x.strip() for x in args.aircraft.split(",") if x.strip()]
    run_benchmark(selected, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
