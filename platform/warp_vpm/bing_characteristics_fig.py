"""Model-characteristics curves vs operating condition.

Shows how each model variant's PREDICTED quantities (not errors) evolve
across operating conditions, against V4B and the experiment:
  Yang: lift & drag vs AoA — bare / +polar / +polar+T3 / V4B / GT
  Izra: CT vs psi (both families) — bare / -Cd0 / +LDVM delta / V4B / GT
  Baik: CL vs phase per case — raw / canonical+transfer / GT
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
v4dir = (repo / "docs/forward_flight_large_pitch/reproductions/"
         "unified_fluxv_v4_ldvm_stevens_20260812/runs/"
         "20260812_fluxv_v4b_crosspaper_full")

# ---------- Yang (parse P2 log for the build-up stages) ----------
yang = dict(aoa=[], lift=[], lift_pol=[], drag=[], drag_pol=[], drag_pol_t3=[],
            lift_gt=[], drag_gt=[], lift_v4b=[], drag_v4b=[])
pat = re.compile(
    r"AoA\s+(\d+):\s+lift\s+([-\d.]+)->\s+([-\d.]+) \(GT\s+([-\d.]+)\) \| "
    r"drag\s+([-\d.]+)->\s+([-\d.]+)\+T3\s+([-\d.]+) \(GT\s+([-\d.]+)\)")
with open("/tmp/v5h15-paper/p2_polar.log") as f:
    for line in f:
        m = pat.search(line)
        if m:
            yang["aoa"].append(float(m.group(1)))
            yang["lift"].append(float(m.group(2)))
            yang["lift_pol"].append(float(m.group(3)))
            yang["lift_gt"].append(float(m.group(4)))
            yang["drag"].append(float(m.group(5)))
            yang["drag_pol"].append(float(m.group(6)))
            yang["drag_pol_t3"].append(float(m.group(6)) + float(m.group(7)))
            yang["drag_gt"].append(float(m.group(8)))
with open(v4dir / "yang2025_v4_mean_characteristics.csv") as f:
    for row in csv.DictReader(f):
        a = float(row["aoa_deg"])
        i = yang["aoa"].index(a)
        yang["lift_v4b"].insert(i, float(row["v4_lift_gf"]))
        yang["drag_v4b"].insert(i, float(row["v4_drag_gf"]))

# ---------- Izra (izra_v2.json + GT + V4B) ----------
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
izra = dict(theta=[], psi=[], raw=[], minus_cd0=[], final=[], gt=[], err=[],
            v4b=[])
for k in sorted(iz.keys(), key=lambda s: tuple(map(float, s.split("/")))):
    th, ps = map(float, k.split("/"))
    m = [r for r in gt_rows if float(r["theta_max_deg"]) == th
         and float(r["phase_offset_deg"]) == ps]
    izra["theta"].append(th)
    izra["psi"].append(ps)
    izra["raw"].append(iz[k]["raw"])
    izra["minus_cd0"].append(iz[k]["raw"] - 0.057)
    izra["final"].append(iz[k]["final"])
    izra["gt"].append(np.mean([float(r["ct"]) for r in m]))
    izra["err"].append(np.mean([float(r["ct_error_minus"]) for r in m]))
    izra["v4b"].append(v4i[(th, ps)])

# ---------- figure ----------
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.36,
                      wspace=0.24)

# Yang lift
ax = fig.add_subplot(gs[0, 0])
ax.plot(yang["aoa"], yang["lift"], "v--", color="gray", lw=1.4, ms=6,
        label="bare chassis")
ax.plot(yang["aoa"], yang["lift_pol"], "^:", color="darkorange", lw=1.5,
        ms=6, label="+ full-angle polar")
ax.plot(yang["aoa"], yang["lift_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["lift_gt"], "k*", ms=14, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean lift [gf]")
ax.set_title("Yang 2025 lift — model build-up vs condition")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Yang drag
ax = fig.add_subplot(gs[0, 1])
ax.plot(yang["aoa"], yang["drag"], "v--", color="gray", lw=1.4, ms=6,
        label="bare chassis")
ax.plot(yang["aoa"], yang["drag_pol"], "^:", color="darkorange", lw=1.5,
        ms=6, label="+ full-angle polar")
ax.plot(yang["aoa"], yang["drag_pol_t3"], "o-", color="crimson", lw=2,
        label="+ polar + T3 viscous (final)")
ax.plot(yang["aoa"], yang["drag_v4b"], "s:", color="tab:blue", lw=1.5,
        label="V4B (frozen)")
ax.plot(yang["aoa"], yang["drag_gt"], "k*", ms=14, label="experiment")
ax.set_xlabel("angle of attack [deg]")
ax.set_ylabel("cycle-mean drag [gf]")
ax.set_title("Yang 2025 drag — model build-up vs condition")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Izra two families
th_a = np.array(izra["theta"])
ps_a = np.array(izra["psi"])
for j, tv in enumerate((15.0, 25.0)):
    ax = fig.add_subplot(gs[1, j])
    m = th_a == tv
    ax.errorbar(ps_a[m], np.array(izra["gt"])[m],
                yerr=np.array(izra["err"])[m], fmt="k*", ms=13, capsize=3,
                label="experiment")
    ax.plot(ps_a[m], np.array(izra["raw"])[m], "v--", color="gray", lw=1.4,
            ms=6, label="bare chassis")
    ax.plot(ps_a[m], np.array(izra["minus_cd0"])[m], "^:", color="darkorange",
            lw=1.5, ms=6, label="- Cd0 (declared)")
    ax.plot(ps_a[m], np.array(izra["final"])[m], "o-", color="crimson", lw=2,
            label="+ frozen LDVM delta (final)")
    ax.plot(ps_a[m], np.array(izra["v4b"])[m], "s:", color="tab:blue", lw=1.5,
            label="V4B (frozen)")
    ax.set_xlabel("phase offset psi [deg]")
    ax.set_ylabel("cycle-mean CT")
    ax.set_title(f"Izraelevitz Fig14 CT — build-up, theta={tv:.0f} deg")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Baik CL vs phase
for j, cid in enumerate(("W1", "W2", "W3", "W4")):
    ax = fig.add_subplot(gs[2, :].subgridspec(1, 4)[0, j])
    fz = np.load(f"/tmp/v5h15-paper/baik_final_{cid}.npz")
    ax.plot(fz["gt_phase"], fz["gt_cl"], "k-", lw=1.6, label="experiment")
    ax.plot(fz["phase"], fz["cl_raw"], "v--", color="gray", lw=1.0, ms=3,
        label="bare (raw cycle)")
    ax.plot(fz["phase"], fz["cl"], "o-", color="crimson", lw=1.5, ms=3,
            label="canonical + transfer")
    ax.set_xlabel("phase")
    ax.set_title(f"{cid} CL vs phase", fontsize=10)
    if j == 0:
        ax.set_ylabel("CL")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Model characteristics vs operating condition — build-up chain "
             "(bare → corrections → final) against V4B and experiments",
             fontsize=13, y=0.99)
fig.savefig("/tmp/v5h15-paper/model_characteristics.png", dpi=150,
            bbox_inches="tight")
print("SAVED /tmp/v5h15-paper/model_characteristics.png")
