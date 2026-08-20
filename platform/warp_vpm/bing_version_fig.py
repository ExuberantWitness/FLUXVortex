"""Version-comparison characteristics: latest vs previous vs experiment.

Yang:    latest = polar+T3        | previous = bare chassis
Izra:    latest = chassis+LDVMdelta | previous = ledger T1+T3
Baik:    latest = 2D LDVM         | previous = 3D chassis + transfer
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path("/tmp/fluxv-v5-nextgen")
NEW, NEW_C = "o-", "crimson"
OLD, OLD_C = "s--", "tab:blue"

# ---------- Yang ----------
d = np.load("/tmp/v5h15-paper/figure_data.npz", allow_pickle=True)
y = {k: np.asarray(v) for k, v in d["yang"].item().items()}

# ---------- Izra ----------
iz = json.loads(Path("/tmp/v5h15-paper/izra_v2.json").read_text())
prev_ledger = {  # from the P3 ledger run (bing_p3_izra_baik output)
    (15.0, 15.0): 0.2146, (15.0, 30.0): 0.2190, (15.0, 45.0): 0.2198,
    (15.0, 60.0): 0.2145, (15.0, 75.0): 0.2028, (15.0, 90.0): 0.1939,
    (15.0, 105.0): 0.1889, (25.0, 45.0): 0.1165, (25.0, 60.0): 0.1037,
    (25.0, 75.0): 0.0812, (25.0, 90.0): 0.0650, (25.0, 105.0): 0.0562}
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)
izra = dict(theta=[], psi=[], new=[], old=[], gt=[], err=[])
for k in sorted(iz.keys(), key=lambda s: tuple(map(float, s.split("/")))):
    th, ps = map(float, k.split("/"))
    m = [r for r in gt_rows if float(r["theta_max_deg"]) == th
         and float(r["phase_offset_deg"]) == ps]
    izra["theta"].append(th); izra["psi"].append(ps)
    izra["new"].append(iz[k]["final"])
    izra["old"].append(prev_ledger[(th, ps)])
    izra["gt"].append(np.mean([float(r["ct"]) for r in m]))
    izra["err"].append(np.mean([float(r["ct_error_minus"]) for r in m]))

# ---------- figure ----------
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.36,
                      wspace=0.24)


def tri(ax, x, new, old, gt, xerr=None, newlab="latest", oldlab="previous"):
    ax.plot(x, new, NEW, color=NEW_C, lw=2, ms=6, label=newlab)
    ax.plot(x, old, OLD, color=OLD_C, lw=1.6, ms=6, label=oldlab)
    if xerr is not None:
        ax.errorbar(x, gt, yerr=xerr, fmt="k*", ms=13, capsize=3,
                    label="experiment")
    else:
        ax.plot(x, gt, "k*", ms=13, label="experiment")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


ax = fig.add_subplot(gs[0, 0])
tri(ax, y["aoa"], y["lift_model"], y["lift_bare"], y["lift_gt"],
    newlab="latest (polar+T3)", oldlab="previous (bare chassis)")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 — lift")

ax = fig.add_subplot(gs[0, 1])
tri(ax, y["aoa"], y["drag_model"], y["drag_bare"], y["drag_gt"],
    newlab="latest (polar+T3)", oldlab="previous (bare chassis)")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 — drag")

th_a = np.array(izra["theta"]); ps_a = np.array(izra["psi"])
for j, tv in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = th_a == tv
    tri(ax, ps_a[m], np.array(izra["new"])[m], np.array(izra["old"])[m],
        np.array(izra["gt"])[m], xerr=np.array(izra["err"])[m],
        newlab="latest (chassis+LDVM delta)",
        oldlab="previous (ledger T1+T3)")
    ax.set_xlabel("phase offset psi [deg]")
    ax.set_ylabel("cycle-mean CT")
    ax.set_title(f"Izraelevitz Fig.14 — CT (theta = {tv:.0f} deg)")

for j, cid in enumerate(("W1", "W2", "W3", "W4")):
    ax = fig.add_subplot(gs[2, :].subgridspec(1, 4)[0, j])
    new = np.load(f"/tmp/v5h15-paper/baik_2dldvm_{cid}.npz")
    old = np.load(f"/tmp/v5h15-paper/baik_final_{cid}.npz")
    ax.plot(new["phase"], new["cl"], NEW, color=NEW_C, lw=1.6, ms=3,
            label="latest (2D LDVM)")
    ax.plot(old["phase"], old["cl"], OLD, color=OLD_C, lw=1.4, ms=3,
            label="previous (3D+transfer)")
    ax.plot(old["gt_phase"], old["gt_cl"], "k-", lw=1.6, label="experiment")
    ax.set_xlabel("phase")
    ax.set_title(f"Baik 2012 {cid} — CL", fontsize=9)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Latest vs previous model vs experiment — characteristics "
             "across operating conditions", fontsize=13, y=0.99)
fig.savefig("/tmp/v5h15-paper/version_comparison.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/version_comparison.png")
