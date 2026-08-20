"""Four-curve figure per panel: previous version, latest version, V4B,
experiment."""
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path("/tmp/fluxv-v5-nextgen")
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")

# ---------------- Yang ----------------
# latest (polar+T3) parsed from P2 log; previous (ledger T1+T3) from the
# ledger run output (lift unchanged from bare, drag corrected).
yang = dict(aoa=[], lift_prev=[], lift_new=[], lift_gt=[], lift_v4b=[],
            drag_prev=[], drag_new=[], drag_gt=[], drag_v4b=[])
pat = re.compile(
    r"AoA\s+(\d+):\s+lift\s+([-\d.]+)->\s+([-\d.]+) \(GT\s+([-\d.]+)\) \| "
    r"drag\s+([-\d.]+)->\s+([-\d.]+)\+T3\s+([-\d.]+) \(GT\s+([-\d.]+)\)")
for line in open("/tmp/v5h15-paper/p2_polar.log"):
    m = pat.search(line)
    if m:
        yang["aoa"].append(float(m.group(1)))
        yang["lift_new"].append(float(m.group(3)))
        yang["lift_gt"].append(float(m.group(4)))
        yang["drag_new"].append(float(m.group(6)) + float(m.group(7)))
        yang["drag_gt"].append(float(m.group(8)))
with open(v4dir / "yang2025_v4_mean_characteristics.csv") as f:
    for row in csv.DictReader(f):
        yang["lift_v4b"].append(float(row["v4_lift_gf"]))
        yang["drag_v4b"].append(float(row["v4_drag_gf"]))
# previous version (ledger run, recorded output)
yang["lift_prev"] = [0.2, 13.7, 27.1, 40.1, 52.8, 64.9]
yang["drag_prev"] = [-4.7, -4.0, -1.8, 1.7, 6.3, 12.1]

# ---------------- Izra ----------------
iz = json.loads(Path("/tmp/v5h15-paper/izra_v2.json").read_text())
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
v4i = {}
with open(v4dir / "izraelevitz2017_fig14_v4_mean_thrust.csv") as f:
    for row in csv.DictReader(f):
        v4i[(float(row["theta_max_deg"]),
             float(row["phase_offset_deg"]))] = float(row["v4_CT"])
izra_prev = {  # ledger T1@0.239 + T3 (recorded from the P3 run)
    (15, 15): 0.2146, (15, 30): 0.2190, (15, 45): 0.2198, (15, 60): 0.2145,
    (15, 75): 0.2028, (15, 90): 0.1939, (15, 105): 0.1889,
    (25, 45): 0.1165, (25, 60): 0.1037, (25, 75): 0.0812, (25, 90): 0.0650,
    (25, 105): 0.0562}
izra = dict(theta=[], psi=[], prev=[], new=[], gt=[], err=[], v4b=[])
for k in sorted(iz.keys(), key=lambda s: tuple(map(float, s.split("/")))):
    th, ps = map(float, k.split("/"))
    m = [r for r in gt_rows if float(r["theta_max_deg"]) == th
         and float(r["phase_offset_deg"]) == ps]
    izra["theta"].append(th)
    izra["psi"].append(ps)
    izra["new"].append(iz[k]["final"])
    izra["prev"].append(izra_prev[(int(th), int(ps))])
    izra["gt"].append(np.mean([float(r["ct"]) for r in m]))
    izra["err"].append(np.mean([float(r["ct_error_minus"]) for r in m]))
    izra["v4b"].append(v4i[(th, ps)])

# ---------------- figure ----------------
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.36,
                      wspace=0.24)
PREV, PREV_C = "v--", "dimgray"
NEW, NEW_C = "o-", "crimson"


def quad(ax, x, prev, new, v4, gt, xerr=None):
    ax.plot(x, prev, PREV, color=PREV_C, lw=1.5, ms=5,
            label="ours v1 (previous)")
    ax.plot(x, new, NEW, color=NEW_C, lw=2, ms=6, label="ours v2 (latest)")
    ax.plot(x, v4, "s:", color="tab:blue", lw=1.8, ms=6, label="V4B")
    if xerr is not None:
        ax.errorbar(x, gt, yerr=xerr, fmt="k*", ms=13, capsize=3,
                    label="experiment")
    else:
        ax.plot(x, gt, "k*", ms=13, label="experiment")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


ax = fig.add_subplot(gs[0, 0])
quad(ax, yang["aoa"], yang["lift_prev"], yang["lift_new"],
     yang["lift_v4b"], yang["lift_gt"])
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 — lift")

ax = fig.add_subplot(gs[0, 1])
quad(ax, yang["aoa"], yang["drag_prev"], yang["drag_new"],
     yang["drag_v4b"], yang["drag_gt"])
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 — drag")

th_a = np.array(izra["theta"])
ps_a = np.array(izra["psi"])
for j, tv in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = th_a == tv
    quad(ax, ps_a[m], np.array(izra["prev"])[m], np.array(izra["new"])[m],
         np.array(izra["v4b"])[m], np.array(izra["gt"])[m],
         xerr=np.array(izra["err"])[m])
    ax.set_xlabel("phase offset psi [deg]")
    ax.set_ylabel("cycle-mean CT")
    ax.set_title(f"Izraelevitz Fig.14 — CT (theta = {tv:.0f} deg)")

v4b_baik = {}
for r in csv.DictReader(open(
        repo / "docs/forward_flight_large_pitch/reproductions/"
        "baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/"
        "model_phase_histories.csv")):
    if r["model"] == "fluxv_v4b":
        v4b_baik.setdefault(r["case_id"], []).append(
            (float(r["phase"]), float(r["CL"])))
for j, cid in enumerate(("W1", "W2", "W3", "W4")):
    ax = fig.add_subplot(gs[2, :].subgridspec(1, 4)[0, j])
    fz_new = np.load(f"/tmp/v5h15-paper/baik_2dldvm_{cid}.npz")
    fz_prev = np.load(f"/tmp/v5h15-paper/baik_final_{cid}.npz")
    pairs = sorted(v4b_baik[cid])
    vx = np.array([p for p, _ in pairs])
    vy = np.array([v for _, v in pairs])
    ax.plot(fz_prev["phase"], fz_prev["cl"], PREV, color=PREV_C, lw=1.4,
            ms=3, label="ours v1 (3D+transfer)")
    ax.plot(fz_new["phase"], fz_new["cl"], NEW, color=NEW_C, lw=1.8, ms=3,
            label="ours v2 (2D LDVM)")
    ax.plot(vx, vy, ":", color="tab:blue", lw=1.5, label="V4B")
    ax.plot(fz_prev["gt_phase"], fz_prev["gt_cl"],
            "k-", lw=1.6, label="experiment")
    ax.set_xlabel("phase")
    ax.set_title(f"Baik 2012 {cid} — CL", fontsize=10)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("ours v1 (previous) vs ours v2 (latest) vs V4B vs experiment",
             fontsize=13, y=0.99)
fig.savefig("/tmp/v5h15-paper/four_curves.png", dpi=150, bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/four_curves.png")
