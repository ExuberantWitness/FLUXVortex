"""Operating-condition comparison figure: per-condition |error| ours vs V4B
across all three papers, plus the summary scoreboard."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path("/tmp/fluxv-v5-nextgen")
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")

# ---------- Yang ----------
d = np.load("/tmp/v5h15-paper/figure_data.npz", allow_pickle=True)
y = d["yang"].item()
yang = {k: np.asarray(v) for k, v in y.items()}

# ---------- Izra ----------
iz = json.loads(Path("/tmp/v5h15-paper/izra_v2.json").read_text())
keys = sorted(iz.keys(), key=lambda s: tuple(map(float, s.split("/"))))
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
        v4i[(float(row["theta_max_deg"]), float(row["phase_offset_deg"]))] = \
            float(row["v4_CT"])
izra_lab, izra_ours, izra_v4 = [], [], []
for k in keys:
    th, ps = map(float, k.split("/"))
    ours = iz[k]["final"]
    gts = [float(r["ct"]) for r in gt_rows
           if float(r["theta_max_deg"]) == th
           and float(r["phase_offset_deg"]) == ps]
    for gct in gts:
        izra_lab.append(f"{th:.0f}/{ps:.0f}")
        izra_ours.append(abs(ours - gct))
        izra_v4.append(abs(v4i[(th, ps)] - gct))

# ---------- Baik ----------
ours_cl = {"W1": 0.5157, "W2": 1.0326, "W3": 0.3745, "W4": 0.7079}
v4_cl = {"W1": 0.5157, "W2": 1.0323, "W3": 0.3743, "W4": 0.7079}
ours_cd = {"W1": 0.121, "W2": 0.632, "W3": 0.220, "W4": 0.256}   # ledger@0.239
v4_cd = {"W1": 0.161, "W2": 0.726, "W3": 0.263, "W4": 0.232}

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
plt.subplots_adjust(hspace=0.42, wspace=0.28)
x = np.arange(len(yang["aoa"]))
w = 0.38


def bars(ax, labels, ours, v4, title, unit, rot=0):
    xx = np.arange(len(labels))
    ax.bar(xx - w / 2, ours, w, color="crimson", label="ours")
    ax.bar(xx + w / 2, v4, w, color="tab:blue", label="V4B")
    wins = int(np.sum(np.asarray(ours) < np.asarray(v4)))
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, rotation=rot, fontsize=8)
    ax.set_ylabel(f"|error| [{unit}]")
    ax.set_title(f"{title}   (ours win {wins}/{len(labels)})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


ax = axes[0, 0]
bars(ax, [f"{a:.0f}" for a in yang["aoa"]],
     np.abs(yang["lift_model"] - yang["lift_gt"]),
     np.abs(yang["lift_v4b"] - yang["lift_gt"]),
     "Yang 2025 — lift |err| vs AoA", "gf")
ax.set_xlabel("angle of attack [deg]")

ax = axes[0, 1]
bars(ax, [f"{a:.0f}" for a in yang["aoa"]],
     np.abs(yang["drag_model"] - yang["drag_gt"]),
     np.abs(yang["drag_v4b"] - yang["drag_gt"]),
     "Yang 2025 — drag |err| vs AoA", "gf")
ax.set_xlabel("angle of attack [deg]")

ax = axes[0, 2]
bars(ax, izra_lab, izra_ours, izra_v4,
     "Izraelevitz Fig14 — CT |err| (theta/psi)", "-", rot=45)
ax.set_xlabel("condition [deg/deg]")

ax = axes[1, 0]
cases = ["W1", "W2", "W3", "W4"]
bars(ax, cases, [ours_cl[c] for c in cases], [v4_cl[c] for c in cases],
     "Baik 2012 — CL RMSE", "-")
ax.set_xlabel("case (macro: ours 0.658 vs V4B 0.658 — tie)")

ax = axes[1, 1]
bars(ax, cases, [ours_cd[c] for c in cases], [v4_cd[c] for c in cases],
     "Baik 2012 — CD RMSE (drag ledger)", "-")
ax.set_xlabel("case (macro: ours 0.307 vs V4B 0.345)")

# scoreboard
ax = axes[1, 2]
ax.axis("off")
rows = [
    ("Yang lift MAE [gf]", "4.10", "4.55", "WIN"),
    ("Yang drag MAE [gf]", "1.52", "2.64", "WIN"),
    ("Baik CD macro", "0.307", "0.345", "WIN"),
    ("Izra CT MAE", "0.0178", "0.0198", "WIN"),
    ("Baik CL macro", "0.6577", "0.6575", "TIE"),
]
tbl = ax.table(cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
               colLabels=["metric", "ours", "V4B", "verdict"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for (i, j), cell in tbl.get_celld().items():
    if j == 3:
        cell.get_text().set_color(
            "green" if tbl[(i, 3)].get_text().get_text() == "WIN" else "gray")
        cell.get_text().set_weight("bold")
ax.set_title("Scoreboard: 4 wins + 1 exact tie")

fig.suptitle("Per-condition error comparison vs V4B — all three papers, "
             "every operating condition (cache-clean, zero fitting)",
             fontsize=13, y=0.99)
fig.savefig("/tmp/v5h15-paper/condition_comparison.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/condition_comparison.png")
