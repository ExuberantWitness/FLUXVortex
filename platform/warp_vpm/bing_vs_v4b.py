"""Per-condition comparison: new chassis vs frozen FluxV v4b (crosspaper run)."""
import csv
import json
from pathlib import Path

import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")

# ---------- Yang ----------
mine = {r["aoa"]: r for r in json.loads(
    Path("/tmp/v5h15-paper/bing_yang_results.json").read_text())["results"]}
rows = list(csv.DictReader(open(v4dir / "yang2025_v4_mean_characteristics.csv")))
print("=== Yang 2025 (lift / drag, gf) ===")
print(f"{'AoA':>4} {'GT lift':>8} {'mine':>7} {'v4b':>7} | "
      f"{'GT drag':>8} {'mine':>7} {'v4b':>7}")
lm, lv, dm, dv = [], [], [], []
for r in rows:
    aoa = float(r["aoa_deg"])
    m = mine[aoa]
    my_lift, my_drag = m["lift"], -m["thrust"]
    v4_lift, v4_drag = float(r["v4_lift_gf"]), float(r["v4_drag_gf"])
    gtl, gtd = float(r["test_lift_gf"]), float(r["test_drag_gf"])
    lm.append(abs(my_lift - gtl)); lv.append(abs(v4_lift - gtl))
    dm.append(abs(my_drag - gtd)); dv.append(abs(v4_drag - gtd))
    print(f"{aoa:4.0f} {gtl:8.1f} {my_lift:7.1f} {v4_lift:7.1f} | "
          f"{gtd:8.1f} {my_drag:7.1f} {v4_drag:7.1f}")
print(f"lift MAE: mine {np.mean(lm):.2f} vs v4b {np.mean(lv):.2f} | "
      f"drag MAE: mine {np.mean(dm):.2f} vs v4b {np.mean(dv):.2f}")
wins_l = sum(1 for a, b in zip(lm, lv) if a < b)
print(f"per-AoA lift wins: mine {wins_l}/6, v4b {6-wins_l}/6")

# ---------- Izra ----------
mine_i = json.loads(
    Path("/tmp/v5h15-paper/bing_izra_results.json").read_text())["results"]
mi = {(r["theta_max"], r["psi"]): r for r in mine_i}
rows = list(csv.DictReader(open(v4dir / "izraelevitz2017_fig14_v4_mean_thrust.csv")))
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
       "unified_fluxv_upgrade_20260812/source_data/"
       "izraelevitz2017_fig14_digitized.csv") as fh:
    for r in csv.DictReader(fh):
        if r["data_role"] == "experimental_observation":
            gt_rows.append(r)
print("\n=== Izraelevitz Fig14 (cycle-mean CT) ===")
print(f"{'cond':>10} {'GT':>8} {'mine':>8} {'v4b':>8}")
em, ev = [], []
for r in rows:
    th, ps = float(r["theta_max_deg"]), float(r["phase_offset_deg"])
    v4 = float(r["v4_CT"])
    for gr in gt_rows:
        if float(gr["theta_max_deg"]) == th and float(gr["phase_offset_deg"]) == ps:
            gct = float(gr["ct"])
            m_ct = mi[(th, ps)]["ct_corr"]
            em.append(abs(m_ct - gct)); ev.append(abs(v4 - gct))
            print(f"{th:4.0f}/{ps:3.0f} {gct:8.4f} {m_ct:8.4f} {v4:8.4f}")
print(f"CT MAE : mine {np.mean(em):.4f} vs v4b {np.mean(ev):.4f}")
print(f"CT RMSE: mine {np.sqrt(np.mean(np.array(em)**2)):.4f} vs "
      f"v4b {np.sqrt(np.mean(np.array(ev)**2)):.4f}")

# ---------- Baik ----------
print("\n=== Baik W1-W4 (CL macro / CD macro RMSE) ===")
print(f"mine (8x8cos/128/3cyc): CL 0.755 CD 0.379")
print(f"v4b  (frozen crosspaper): CL 0.6575 CD 0.3452")
