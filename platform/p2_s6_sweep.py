"""P2-S6 sweep driver: flexible Fig17/18/19 points via record+replay,
checkpointed (safe to kill/relaunch; finished conditions are skipped).

Priority subset (coupled recording costs ~3-5 h/condition on this GPU;
the full data.md grid is 122 conditions = weeks — the subset covers the
Fig17 f=2.3 row, the Fig18ab tw=0 columns, and the twist study; extend
PRIORITY as budget allows):

Results accumulate in docs/s6_results.json:
  key "U_f_tw" -> {flex: [L,T], rigid: [L,T], cfg, n_windows, iters_mean}

Run: cd FLUXV && python platform/p2_s6_sweep.py [CFG]      (default K0)
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_HERE, "..")), _HERE,
          os.path.abspath(os.path.join(_HERE, "..", "src"))):
    if p not in sys.path:
        sys.path.insert(0, p)

import p2_s6_replay as rp                                    # noqa: E402

DOCS = os.path.join(_HERE, "docs")
RES = os.path.join(DOCS, "s6_results.json")

# (U, freq, tw) priority order — Fig17 f=2.3 row first (incl. the tw=22.5
# optimum), then Fig18ab tw=0 across U/freq
PRIORITY = [
    (8.0, 2.3, 0.0),
    (8.0, 2.3, 22.5),
    (8.0, 2.3, 45.0),
    (8.0, 2.0, 0.0),
    (8.0, 2.6, 0.0),
    (6.0, 2.3, 0.0),
    (10.0, 2.3, 0.0),
    (8.0, 1.7, 0.0),
    (8.0, 1.4, 0.0),
    (8.0, 2.3, 15.0),
    (8.0, 2.3, 30.0),
    (6.0, 1.4, 0.0),
    (10.0, 2.0, 0.0),
]


def main(cfg="K0"):
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for (U, f, tw) in PRIORITY:
        key = f"{U:g}_{f:g}_{tw:g}"
        npz = rp._npz_path(U, f, tw)
        try:
            if key not in res or not res[key].get("flex"):
                ck = npz + ".ckpt"
                if not os.path.exists(npz) or os.path.exists(ck):
                    print(f"=== RECORD {key} ===", flush=True)
                    t0 = time.time()
                    # rc0_scale=2: wake-core floor 1.0 x row spacing — the
                    # validated reversal/wake-recross regularization (0.5x
                    # died at w55 post-reversal; 1.0x crossed)
                    rp.record(U, f, tw, n_cycles=1.6, rc0_scale=2.0,
                              resume=ck if os.path.exists(ck) else None)
                    print(f"  record wall {time.time()-t0:.0f}s", flush=True)
                    d_ = np.load(npz)
                    if str(d_["fail"]) == "None" and os.path.exists(ck):
                        os.remove(ck)              # complete: drop checkpoint
                d = np.load(npz)
                if str(d["fail"]) != "None":
                    res[key] = dict(fail=str(d["fail"]))
                    json.dump(res, open(RES, "w"), indent=1)
                    continue
                print(f"=== REPLAY {key} [{cfg}] ===", flush=True)
                out = rp.replay(U, f, tw, cfg)
                res[key] = dict(flex=list(out["flex"]), rigid=list(out["rigid"]),
                                cfg=cfg, iters_mean=float(np.mean(d["iters"])),
                                coupled_2L=float(2 * np.mean(
                                    d["lifts"][-int(len(d["lifts"]) / 1.6):])))
                json.dump(res, open(RES, "w"), indent=1)
        except Exception as e:
            print(f"  ERROR {key}: {type(e).__name__}: {e}", flush=True)
            res[key] = dict(error=f"{type(e).__name__}: {e}")
            json.dump(res, open(RES, "w"), indent=1)
    print("SWEEP DONE:", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "K0")
