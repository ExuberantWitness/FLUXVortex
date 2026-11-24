"""Visual cross-check of the MEMBRANE VIBRATION PATTERN against Rojratsirikul et al. (2011).

The experiment is the only truth, and a scalar (zmax/c, Cn) cannot tell whether the model
has the right FLOW PHENOMENON or only the right number.  The paper publishes two images that
can be matched directly:

  Figure 7(b)  time-averaged velocity magnitude + streamlines in the mid-span plane,
               membrane wing, U_inf = 5 m/s.  At alpha = 16 deg the streamlines follow the
               cambered surface, there is NO closed recirculation region (the rigid plate in
               column (a) at the same angle has a large one), and a high-speed band
               (V/U_inf ~ 1.2-1.5) sits over the whole upper surface.
  Figure 10(a) standard deviation of the membrane z-displacement, U_inf = 5 m/s.
               At alpha = 17 deg (the nearest case to A16): NO spanwise peaks, a CHORDWISE
               MODE OF TWO (two bands along the chord), peak zsd/c about 0.004-0.005, and
               nothing at the wing tips.  (At alpha = 10 deg there ARE two peaks near the
               tips, which the paper attributes to the tip vortices; they are gone by 17 deg
               because the tip vortices move away from the surface.)

This script draws the model's counterparts from the stored displacement history of a full
run, in the paper's layout (x/c horizontal, y/c vertical, flow from the leading edge), and
reports the mode counts the way the paper defines them: by counting peaks of the standard
deviation along the chord and along the span.

It needs no simulation: `z_history_over_c` (steps, 15, 31) is written next to every payload.

Usage:
    python3 platform/warp_vpm/visualize_roj_vibration_pattern.py \
        --npz artifacts/baselines/.../ROJ11_A16_FULL_20261123_CNCONV.z_history.npz \
        --out artifacts/experiment/.../fig_vibration_pattern_legacy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

PAPER_ZSD_A17 = (0.004, 0.005)      # peak zsd/c read off Figure 10(a), alpha = 17 deg, U = 5


def count_peaks(profile: np.ndarray, prominence_fraction: float = 0.25) -> int:
    """Interior peaks of a 1-D profile, ignoring the clamped ends.

    The paper counts peaks of the standard deviation; a peak is only a mode if it stands
    clearly above its neighbours, hence the prominence requirement.
    """

    interior = profile[1:-1]
    if interior.max() <= 0.0:
        return 0
    peaks, _ = find_peaks(interior, prominence=prominence_fraction * interior.max())
    return int(len(peaks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-time-star", type=float, default=3.0,
                        help="window start.  The default is NOT the payload's statistics "
                             "window (t*>=20): with dt*=0.01 that holds only 101 samples, "
                             "and the shedding period is t*~2.3-2.6 (St 0.39-0.44), so the "
                             "standard deviation is badly under-resolved there (measured "
                             "zsd/c 0.0004 vs 0.0066 on t*>=3).  The paper averages 20 s.")
    parser.add_argument("--title", default="")
    arguments = parser.parse_args(argv)

    data = np.load(arguments.npz)
    history = data["z_history_over_c"]            # (T, nx, ny), z/c on the membrane grid
    time_star = data["time_star"]
    window = time_star >= arguments.min_time_star
    if window.sum() < 8:
        raise SystemExit(f"only {int(window.sum())} samples after t*={arguments.min_time_star}")
    stationary = history[window]
    mean_shape = stationary.mean(axis=0)
    fluctuation = stationary - mean_shape
    zsd = fluctuation.std(axis=0)                 # the paper's "zsd"
    steps_count, nx, ny = history.shape

    x = np.linspace(0.0, 1.0, nx)                 # chord, LE -> TE
    y = np.linspace(-1.0, 1.0, ny)                # span, -b/2 -> +b/2 (paper's y/c axis)
    chordwise_profile = zsd.mean(axis=1)          # span-averaged -> chordwise modes
    spanwise_profile = zsd.mean(axis=0)           # chord-averaged  -> spanwise modes
    chord_peaks = count_peaks(chordwise_profile)
    span_peaks = count_peaks(spanwise_profile)
    peak_zsd = float(zsd.max())

    # spanwise symmetry: the paper says the modes are symmetric about the mid-span
    left, right = zsd[:, : ny // 2], zsd[:, ny // 2 + 1 :][:, ::-1]
    asymmetry = float(np.abs(left - right).max() / max(peak_zsd, 1e-30))

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for axis in axes[0]:
        axis.set_aspect("equal")
        axis.set_xlabel("x/c")
    for axis in axes[1]:
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.locator_params(axis="y", nbins=5)
        axis.set_xlabel("x/c")

    scale_zsd = max(peak_zsd * 1.05, 1e-6)
    image = axes[0, 0].pcolormesh(x, y, zsd.T, cmap="jet", vmin=0.0, vmax=scale_zsd,
                                  shading="nearest")
    axes[0, 0].set_title(f"zsd/c  (model, t*>= {arguments.min_time_star:g})\n"
                         f"peak {peak_zsd:.5f}", fontsize=10)
    axes[0, 0].set_ylabel("y/c")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)

    image = axes[0, 1].pcolormesh(x, y, mean_shape.T, cmap="jet", shading="nearest")
    axes[0, 1].set_title(f"time-mean z/c  (max {float(mean_shape.max()):.5f})", fontsize=10)
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].plot(x, chordwise_profile, marker="o", ms=3)
    axes[1, 0].set_title(f"chordwise profile (span-averaged zsd): {chord_peaks} mode(s)",
                         fontsize=10)
    axes[1, 0].set_ylabel("zsd/c")
    axes[1, 0].set_ylim(bottom=0.0)

    axes[1, 1].plot(y, spanwise_profile, marker="o", ms=3)
    axes[1, 1].set_title(f"spanwise profile (chord-averaged zsd): {span_peaks} mode(s)",
                         fontsize=10)
    axes[1, 1].set_xlabel("y/c")
    axes[1, 1].set_ylim(bottom=0.0)

    figure.suptitle(arguments.title or Path(arguments.npz).name, fontsize=10)
    figure.tight_layout()
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out, dpi=150)
    plt.close(figure)

    print(f"file                 : {Path(arguments.npz).name}")
    print(f"window               : t* >= {arguments.min_time_star:g}  ({int(window.sum())} samples)")
    print(f"peak zsd/c           : {peak_zsd:.5f}     paper Fig.10(a), alpha=17, U=5: "
          f"{PAPER_ZSD_A17[0]:.3f}-{PAPER_ZSD_A17[1]:.3f}")
    print(f"chordwise modes      : {chord_peaks}              paper, alpha=17: 2")
    print(f"spanwise modes       : {span_peaks}              paper, alpha=17: none visible")
    print(f"spanwise asymmetry   : {asymmetry:.3f}          (0 = symmetric; paper: symmetric in most cases)")
    tip = np.maximum(zsd[:, 0], zsd[:, -1]).max()
    print(f"zsd at the wing tips : {float(tip):.5f}  ({float(tip) / max(peak_zsd, 1e-30):.2f} of peak)"
          f"   paper: no tip peaks at alpha=17")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
