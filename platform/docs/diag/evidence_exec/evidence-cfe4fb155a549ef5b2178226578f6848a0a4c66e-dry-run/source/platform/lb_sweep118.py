"""118-condition rigid sweep on the PRODUCTION closure presets (2026-07-27).

v4.2 campaign Phase -1.3: regenerate the 118-pt baseline self-consistently from
the closure= presets in _v2_robo.gpu_run_twist (the 07-25 candF/v4 jsons were
built by an ad-hoc script whose full call dict was never archived — the
promotion audit could not reproduce them to <0.4N). From this sweep on,
baseline == closure preset == code, by construction.

Conventions (this file IS the recovered 2026-07-25 archived call dict):
  kinematics v2 (kinematics_audit.md): nominal 45 deg flap = +-22.5,
    nominal tw = +-tw/2  -> flap_amp_deg=22.5, twist_amp_deg=tw/2, phase +90.
  grid: nc=12, ns=16, n_cycle=4, spc=spc_of(U,f), wake_rows=spc.
  physics: CFG_PRESETS["H16"] with the explicit overrides below. This exact
    identity was recovered from the original Claude session transcript and
    reproduces the frozen cache at the three E1/E2 probes to <=0.10 N.
  closures: 'v41' (production: L-B closure, wind-lift chop, ds panel-normal
    cds=2.5 Tv-mem, CT off) and 'v4_legacy' (static geo_stall_vec Kirchhoff
    + uiuc polar + kelvin LEV ansari sheet + lev_impulse, a0_crit=0.27).

Outputs (resumable, per-condition checkpoint after every run):
  platform/docs/s6_sweep_v41.json        key "U_f_tw_aoa" -> {L, T}
  platform/docs/s6_sweep_v4_legacy.json

Run:  cd FLUXV && <py> platform/lb_sweep118.py [v41|v4_legacy|both] [--quick]
"""
import json
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
DOCS = os.path.join(HERE, "docs")

TWS = [0., 5., 10., 15., 20., 22.5, 25., 27.5, 30., 35., 40., 45.]
FS = [1.4, 1.7, 2.0, 2.3, 2.6]

# the 118-condition grid (verified against s6_sweep_v41.json 07-25 cache)
CONDS = []
for f in FS:                                   # 60: Fig17/18 U8 aoa5 twist x freq
    for tw in TWS:
        CONDS.append((8.0, f, tw, 5.0))
for f in FS:                                   # 10: Fig18 U6/U10 columns
    CONDS.append((6.0, f, 22.5, 5.0))
    CONDS.append((10.0, f, 22.5, 5.0))
for aoa in (0.0, 10.0, 15.0):                  # 48: Fig19 aoa planes
    for f in FS:
        CONDS.append((8.0, f, 22.5, aoa))
    for tw in TWS:
        CONDS.append((8.0, 2.6, tw, aoa))
CONDS = sorted(set(CONDS))                     # (8,f2.6,tw22.5,aoa) sits in both sub-planes
assert len(CONDS) == 118

# quick-loop fingerprint subset (D1/D2/D3 病灶角区 + 验证区, 12 pts)
QUICK = [(8.0, 2.3, 22.5, 5.0), (8.0, 2.3, 0.0, 5.0), (8.0, 2.3, 45.0, 5.0),
         (6.0, 2.3, 22.5, 5.0), (10.0, 2.3, 22.5, 5.0), (6.0, 2.6, 22.5, 5.0),
         (8.0, 1.4, 22.5, 0.0), (8.0, 2.6, 22.5, 0.0), (8.0, 1.4, 22.5, 15.0),
         (8.0, 2.6, 22.5, 15.0), (8.0, 2.6, 15.0, 5.0), (8.0, 2.3, 15.0, 5.0)]

from _v2_repro_nc12 import CFG_PRESETS, spc_of

BASE = dict(CFG_PRESETS["H16"], fsep_lag=False, cosine_chord="le",
            les_sep="plateau_fn", d_para=0.5, attached_drag="uiuc",
            geo_stall=False, flap_amp_deg=22.5, twist_phase_deg=90.)
V4LEG_EXTRA = dict(a0_crit=0.27, lev_shed_mode='kelvin', lev_sheet=True,
                   lev_place='ansari', lev_sign=1.0, lev_impulse=True)


def key_of(U, f, tw, aoa):
    return f"{U:g}_{f:g}_{tw:g}_{aoa:g}"


def run(which="both", quick=False, force=False, output_suffix=""):
    import warp as wp
    wp.init()
    from _v2_robo import gpu_run_twist
    conds = QUICK if quick else CONDS
    for closure in (("v41", "v4_legacy") if which == "both" else (which,)):
        stem = "s6_sweep_v41" if closure == "v41" else "s6_sweep_v4_legacy"
        out_path = os.path.join(DOCS, f"{stem}{output_suffix}.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        extra = V4LEG_EXTRA if closure == "v4_legacy" else {}
        call_config = dict(BASE)
        call_config.update(extra)
        t_all = time.time()
        for i, (U, f, tw, aoa) in enumerate(conds):
            k = key_of(U, f, tw, aoa)
            if not force and k in res and "L" in res[k]:
                continue
            spc = spc_of(U, f)
            t0 = time.time()
            try:
                r = gpu_run_twist(U=U, aoa_deg=aoa, freq=f,
                                  twist_amp_deg=tw / 2.0,
                                  nc=12, ns=16, n_cycle=4,
                                  steps_per_cycle=spc, wake_rows=spc,
                                  closure=closure, **call_config)
                res[k] = dict(L=float(r["L_wind"]), T=float(r["T_wind"]))
                print(f"[{closure}] {i+1}/{len(conds)} {k}: "
                      f"L={res[k]['L']:+.3f} T={res[k]['T']:+.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                res[k] = dict(fail=f"{type(e).__name__}: {e}")
                print(f"[{closure}] {i+1}/{len(conds)} {k}: FAIL {e}", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
        n_ok = sum(1 for v in res.values() if "L" in v)
        print(f"[{closure}] DONE {n_ok}/{len(conds)} ok, "
              f"wall {(time.time()-t_all)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", choices=("v41", "v4_legacy", "both"), default="both")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute even when an output key already exists")
    parser.add_argument("--output-suffix", default="",
                        help="suffix inserted before .json; use for non-destructive verification")
    args = parser.parse_args()
    if args.force and not args.output_suffix:
        parser.error("--force requires --output-suffix to protect frozen baselines")
    run(args.which, quick=args.quick, force=args.force, output_suffix=args.output_suffix)
