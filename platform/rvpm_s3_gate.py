"""S3 qualitative gate (PROJECT_rvpm / A3-C8 Garmann-Visbal): pitch-up ramp on the
production planform with the rVPM LEV particle path. Gate: inboard-STABLE /
outboard-SHEDDING two-zone LEV + tip-vortex spanwise modulation (qualitative —
no published quantitative 3D force target at this Re, known canon gap).

run:     python platform/rvpm_s3_gate.py run      (GPU, ~30-60 min with substeps)
analyze: python platform/rvpm_s3_gate.py analyze  (spanwise-band structure from snapshots)
"""
import glob
import os
import sys

sys.path[:0] = ["platform", "src"]
import numpy as np

SNAP = "platform/docs/diag/rvpm_s3_snap"
RV = dict(a0_mode='downwash', a0_crit=0.14, les_sep='plateau', attached_drag='uiuc',
          d_para=0.5, cosine_chord='le', part_lev=True, lev_cons=True,
          part_mode='rvpm', part_transport='rvpm')


def run():
    os.environ["RVPM_SNAP"] = SNAP
    from _v2_robo import gpu_run_twist
    import time
    t0 = time.time()
    # Eldredge pitch-up 0->45deg K=0.2 (in-repo Hirato-validated ramp mode), no flap/twist;
    # freq=4/spc=400/1cyc -> 0.25s = t*~7 (ramp t1*=1 -> t2*~2.96, hold after).
    r = gpu_run_twist(U=8.0, aoa_deg=0.0, freq=4.0, flap_amp_deg=0.0, twist_amp_deg=0.0,
                      nc=12, ns=16, n_cycle=1, steps_per_cycle=400, wake_rows=400,
                      pitch_ramp=True, pitch_max=45.0, pitch_K=0.2, pitch_t0star=1.0, **RV)
    print(f"ramp done: L={float(r['L_wind']):+.2f} T={float(r['T_wind']):+.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)


def analyze():
    fs = sorted(glob.glob(SNAP + "_t*.npz"))
    print(f"{len(fs)} snapshots")
    for f in fs[::2]:
        d = np.load(f)
        pp, pa, rings = d["pp"], d["pa"], d["rings"]
        ns = 16
        le0 = 0.5 * (rings[:ns, 0] + rings[:ns, 1])          # LE midpoints per strip (world)
        ymax = np.max(np.abs(le0[:, 1])) + 1e-9
        yfrac = np.abs(pp[:, 1]) / ymax                       # spanwise fraction of particles
        # distance of each particle from the nearest strip LE point
        dle = np.min(np.linalg.norm(pp[:, None, :] - le0[None, :, :], axis=2), axis=1)
        gmag = np.linalg.norm(pa, axis=1)
        line = os.path.basename(f) + " | "
        for b0, b1, tag in ((0.0, 0.4, "in"), (0.4, 0.75, "mid"), (0.75, 1.01, "tip")):
            m = (yfrac >= b0) & (yfrac < b1)
            if m.sum() == 0:
                line += f"{tag}: -  | "
                continue
            line += (f"{tag}: n={m.sum():4d} G={gmag[m].sum():.3f} "
                     f"dLE={np.median(dle[m]):.3f}/{np.percentile(dle[m], 90):.3f} | ")
        print(line, flush=True)
    print("two-zone signature: inboard median dLE stays SMALL (attached feeding) while "
          "mid/outboard dLE grows over the hold (convecting/shedding); tip band carries "
          "distinct circulation modulation.", flush=True)


if __name__ == "__main__":
    (run if (len(sys.argv) > 1 and sys.argv[1] == "run") else analyze)()
