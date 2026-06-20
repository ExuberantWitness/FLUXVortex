"""Hero figure for the unsteady-FSI MAP-Elites co-design archive (Plan Phase F / 门面).

Reads qd_unsteady_archive.npz (written by codesign_qd_unsteady.py) and renders:
  (left)  the illuminated (翼面 stiffness-taper × 动力系统 control-gain) archive, each cell coloured
          by gust-rejection quality — the MAP-Elites illumination = the killer visual.
  (right) representative co-designs sampled across the archive, showing each high-quality niche
          carries a DISTINCT spanwise 刚柔 distribution AND a distinct control gain.
No re-evaluation: pure post-processing of the saved archive.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from codesign_qd_unsteady import _basis, NX, NY, NB1, NB2        # noqa: E402

HERE = os.path.dirname(__file__)


def main(npz=os.path.join(HERE, "qd_unsteady_archive.npz"), out=os.path.join(HERE, "qd_unsteady_hero.png")):
    d = np.load(npz)
    b1, b2, qual = d["b1"], d["b2"], d["qual"]
    thetas = d["thetas"]; b1e, b2e = d["b1edges"], d["b2edges"]
    B = _basis(NX, NY); ny = NY
    span = np.array([j / max(ny - 1, 1) for j in range(ny)])

    # archive grid: log10 of the gust-deflection energy J=−quality (spans orders of magnitude;
    # lower log10(J) = better gust rejection)
    grid = np.full((NB1, NB2), np.nan)
    for k in range(len(qual)):
        i = int(np.clip(np.searchsorted(b1e, b1[k]) - 1, 0, NB1 - 1))
        j = int(np.clip(np.searchsorted(b2e, b2[k]) - 1, 0, NB2 - 1))
        v = np.log10(max(-qual[k], 1e-6))
        if np.isnan(grid[i, j]) or v < grid[i, j]:     # keep the BEST (lowest J) per cell
            grid[i, j] = v

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
    axH = fig.add_subplot(gs[0, 0])
    im = axH.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis_r",
                    extent=[b1e[0], b1e[-1], b2e[0], b2e[-1]])
    axH.set_xlabel("翼面 axis — spanwise stiffness taper  $E_{tip}/E_{root}$")
    axH.set_ylabel("动力系统 axis — spanwise mass taper  $\\rho_{tip}/\\rho_{root}$")
    axH.set_title(f"MAP-Elites 刚柔×质量 co-design on the differentiable UNSTEADY coupled FSI\n"
                  f"{int(np.isfinite(grid).sum())}/{NB1*NB2} niches illuminated  ·  "
                  f"quality = gust-deflection energy ‖q_N−q_ref‖²  ·  ~7 min on one RTX 4090 (fp64)", fontsize=10.5)
    plt.colorbar(im, ax=axH, label="$\\log_{10}$ gust-deflection energy  (lower = better)")

    # representative niches: top quality cells spread across the (taper × gain) space
    order = np.argsort(qual)[::-1]
    picks, seen = [], set()
    for idx in order:
        ci = (int(np.clip(np.searchsorted(b1e, b1[idx]) - 1, 0, NB1 - 1)) // 3,
              int(np.clip(np.searchsorted(b2e, b2[idx]) - 1, 0, NB2 - 1)) // 3)
        if ci not in seen:
            seen.add(ci); picks.append(idx)
        if len(picks) >= 5:
            break
    axN = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(picks)))
    for c, idx in enumerate(picks):
        th = thetas[idx]
        E = np.exp(B @ th[0:3])[::NX]; R = np.exp(B @ th[3:6])[::NX]   # spanwise profiles
        axH.plot(b1[idx], b2[idx], "o", ms=9, mfc="none", mec=cmap[c], mew=2.2)
        axN.plot(span, E, "-o", color=cmap[c], ms=4,
                 label=f"$E_t/E_r$={b1[idx]:.2f}, $\\rho_t/\\rho_r$={b2[idx]:.2f}  (q={qual[idx]:.1e})")
        axN.plot(span, R, "--", color=cmap[c], lw=1, alpha=0.6)
    axN.set_xlabel("span fraction (root → tip)")
    axN.set_ylabel("scale along span — $E$ (solid), $\\rho$ (dashed)")
    axN.set_title("Representative co-designs — distinct 刚柔 AND 质量 per niche\n"
                  "(each cell = a full spanwise stiffness + mass distribution)", fontsize=10.5)
    axN.grid(alpha=0.3); axN.legend(fontsize=7, loc="best")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved hero figure -> {out}")

    # honest quantitative phenomenon summary
    tb1 = b1[picks]; tb2 = b2[picks]
    print(f"  representative niches: taper {tb1.min():.2f}..{tb1.max():.2f}, k {tb2.min():.1f}..{tb2.max():.1f}")
    return out


if __name__ == "__main__":
    main()
