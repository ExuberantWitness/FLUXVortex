"""Replot three-paper figure with the new Izra v2 results (no solver reruns)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("/tmp/v5h15-paper/figure_data.npz", allow_pickle=True)
yang = {k: np.asarray(v) for k, v in d["yang"].item().items()}
izra_v2 = json.loads(Path("/tmp/v5h15-paper/izra_v2.json").read_text())
baik = {c: json.loads(s) for c, s in d["baik"].item().items()}

fig = plt.figure(figsize=(15, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.34,
                      wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
ax.plot(yang["aoa"], yang["lift_model"], "o-", color="crimson", lw=2,
        label="ours (polar+T3)")
ax.plot(yang["aoa"], yang["lift_bare"], "--", color="gray", lw=1.2,
        label="bare chassis")
ax.plot(yang["aoa"], yang["lift_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["lift_gt"], "k*", ms=13, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 — lift vs AoA   (MAE: ours 4.10 | V4B 4.55 gf)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(yang["aoa"], yang["drag_model"], "o-", color="crimson", lw=2,
        label="ours (polar+T3)")
ax.plot(yang["aoa"], yang["drag_bare"], "--", color="gray", lw=1.2,
        label="bare chassis")
ax.plot(yang["aoa"], yang["drag_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["drag_gt"], "k*", ms=13, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 — drag vs AoA   (MAE: ours 1.52 | V4B 2.64 gf)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Izra v2: chassis + frozen LDVM delta
keys = sorted(izra_v2.keys(), key=lambda s: (float(s.split("/")[0]),
                                             float(s.split("/")[1])))
th = np.array([float(k.split("/")[0]) for k in keys])
ps = np.array([float(k.split("/")[1]) for k in keys])
raw = np.array([izra_v2[k]["raw"] for k in keys])
fin = np.array([izra_v2[k]["final"] for k in keys])
gt_rows = []
import csv
repo = Path("/tmp/fluxv-v5-nextgen")
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")
gtd, v4m = {}, {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/"
          "unified_fluxv_upgrade_20260812/source_data/"
          "izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            k = (float(row["theta_max_deg"]), float(row["phase_offset_deg"]))
            gtd.setdefault(k, []).append(
                (float(row["ct"]), float(row["ct_error_minus"])))
with open(v4dir / "izraelevitz2017_fig14_v4_mean_thrust.csv") as f:
    for row in csv.DictReader(f):
        v4m[(float(row["theta_max_deg"]),
             float(row["phase_offset_deg"]))] = float(row["v4_CT"])

for j, tv in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = th == tv
    gct = np.array([np.mean([x[0] for x in gtd[(t, p)]])
                    for t, p in zip(th[m], ps[m])])
    gerr = np.array([np.mean([x[1] for x in gtd[(t, p)]])
                     for t, p in zip(th[m], ps[m])])
    ax.errorbar(ps[m], gct, yerr=gerr, fmt="k*", ms=12, capsize=3,
                label="experiment")
    ax.plot(ps[m], fin[m], "o-", color="crimson", lw=2,
            label="ours (chassis+frozen LDVM delta)")
    ax.plot(ps[m], raw[m], "--", color="gray", lw=1.2, label="bare chassis")
    ax.plot(ps[m], [v4m[(t, p)] for t, p in zip(th[m], ps[m])], "s:",
            color="tab:blue", lw=1.5, label="V4B (frozen)")
    ax.set_xlabel("phase offset psi [deg]")
    ax.set_ylabel("cycle-mean CT")
    ax.set_title(f"Izraelevitz Fig.14 — theta={tv:.0f} deg family "
                 f"(all-cond MAE: ours 0.018 | V4B 0.020)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

rmses = {"W1": 0.516, "W2": 1.033, "W3": 0.374, "W4": 0.708}
for j, cid in enumerate(("W1", "W2", "W3", "W4")):
    ax = fig.add_subplot(gs[2, :].subgridspec(1, 4)[0, j])
    fz = np.load(f"/tmp/v5h15-paper/baik_final_{cid}.npz")
    ax.plot(fz["gt_phase"], fz["gt_cl"], "k-", lw=1.5, label="experiment")
    ax.plot(fz["phase"], fz["cl"], "-", color="crimson", lw=1.5,
            label="ours (canonical+transfer)")
    ax.plot(fz["phase"], fz["cl_raw"], "--", color="gray", lw=1.0,
            label="raw")
    ax.set_xlabel("phase")
    ax.set_title(f"{cid}  CL RMSE {rmses[cid]:.3f}", fontsize=10)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Mechanism-based chassis vs V4B — 4 wins + 1 exact tie "
             "(cache-clean, canonical filters, zero fitting)", fontsize=13,
             y=0.995)
fig.savefig("/tmp/v5h15-paper/three_paper_curves_v3.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/three_paper_curves_v3.png")
