"""Clean three-curve figure: ours (final) vs V4B vs experiment only."""
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
OURS = "o-"
OURS_C = "crimson"

# ---------- Yang ----------
yang = dict(aoa=[], lift=[], drag=[], lift_gt=[], drag_gt=[],
            lift_v4b=[], drag_v4b=[])
pat = re.compile(
    r"AoA\s+(\d+):\s+lift\s+([-\d.]+)->\s+([-\d.]+) \(GT\s+([-\d.]+)\) \| "
    r"drag\s+([-\d.]+)->\s+([-\d.]+)\+T3\s+([-\d.]+) \(GT\s+([-\d.]+)\)")
for line in open("/tmp/v5h15-paper/p2_polar.log"):
    m = pat.search(line)
    if m:
        yang["aoa"].append(float(m.group(1)))
        yang["lift"].append(float(m.group(3)))
        yang["lift_gt"].append(float(m.group(4)))
        yang["drag"].append(float(m.group(6)) + float(m.group(7)))
        yang["drag_gt"].append(float(m.group(8)))
with open(v4dir / "yang2025_v4_mean_characteristics.csv") as f:
    for row in csv.DictReader(f):
        yang["lift_v4b"].append(float(row["v4_lift_gf"]))
        yang["drag_v4b"].append(float(row["v4_drag_gf"]))

# ---------- Izra ----------
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
izra = dict(theta=[], psi=[], final=[], gt=[], err=[], v4b=[])
for k in sorted(iz.keys(), key=lambda s: tuple(map(float, s.split("/")))):
    th, ps = map(float, k.split("/"))
    m = [r for r in gt_rows if float(r["theta_max_deg"]) == th
         and float(r["phase_offset_deg"]) == ps]
    izra["theta"].append(th)
    izra["psi"].append(ps)
    izra["final"].append(iz[k]["final"])
    izra["gt"].append(np.mean([float(r["ct"]) for r in m]))
    izra["err"].append(np.mean([float(r["ct_error_minus"]) for r in m]))
    izra["v4b"].append(v4i[(th, ps)])

# ---------- figure ----------
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.36,
                      wspace=0.24)


def tri(ax, x, ours, v4, gt, xerr=None):
    ax.plot(x, ours, OURS, color=OURS_C, lw=2, ms=6, label="ours (final)")
    ax.plot(x, v4, "s:", color="tab:blue", lw=1.8, ms=6, label="V4B")
    if xerr is not None:
        ax.errorbar(x, gt, yerr=xerr, fmt="k*", ms=13, capsize=3,
                    label="experiment")
    else:
        ax.plot(x, gt, "k*", ms=13, label="experiment")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


ax = fig.add_subplot(gs[0, 0])
tri(ax, yang["aoa"], yang["lift"], yang["lift_v4b"], yang["lift_gt"])
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 — lift")

ax = fig.add_subplot(gs[0, 1])
tri(ax, yang["aoa"], yang["drag"], yang["drag_v4b"], yang["drag_gt"])
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 — drag")

th_a = np.array(izra["theta"])
ps_a = np.array(izra["psi"])
for j, tv in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = th_a == tv
    tri(ax, ps_a[m], np.array(izra["final"])[m], np.array(izra["v4b"])[m],
        np.array(izra["gt"])[m], xerr=np.array(izra["err"])[m])
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
    fz = np.load(f"/tmp/v5h15-paper/baik_final_{cid}.npz")
    pairs = sorted(v4b_baik[cid])
    vx = np.array([p for p, _ in pairs])
    vy = np.array([v for _, v in pairs])
    ax.plot(fz["phase"], fz["cl"], OURS, color=OURS_C, lw=1.6, ms=3,
            label="ours (final)")
    ax.plot(vx, vy, ":", color="tab:blue", lw=1.6, label="V4B")
    ax.plot(fz["gt_phase"], fz["gt_cl"], "k-", lw=1.6, label="experiment")
    ax.set_xlabel("phase")
    ax.set_title(f"Baik 2012 {cid} — CL", fontsize=10)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("ours (mechanism-based) vs V4B vs experiment — three papers",
             fontsize=13, y=0.99)
fig.savefig("/tmp/v5h15-paper/three_curves_simple.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/three_curves_simple.png")
