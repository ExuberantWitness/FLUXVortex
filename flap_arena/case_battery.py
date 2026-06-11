"""Case battery: ONE platform configuration (particle wake + two-pass PC),
many cases — the Fluent-style generality demonstration.

Runs diverse rigid cases fast; prints per-case summaries.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from newton_pc.case import CaseConfig, run_case

CASES = {
    "flap5_AR4":   CaseConfig(kin_amp_deg=5.0),
    "flap15_AR4":  CaseConfig(kin_amp_deg=15.0),
    "flap8_AR2_05Hz": CaseConfig(span=3.0, ns=6, kin_amp_deg=8.0,
                                 kin_period=2.0, cycles=2),
    "static_alpha3": CaseConfig(kin_type="none", alpha_deg=3.0, cycles=1.5),
    "fastflap_smallwing": CaseConfig(chord=0.5, span=1.5, nc=5, ns=6,
                                     kin_amp_deg=10.0, kin_period=0.25,
                                     v_inf=8.0, cycles=4),
}

for name, cfg in CASES.items():
    try:
        res = run_case(cfg)
        L = res.lift
        per = max(1, int(round((cfg.kin_period if cfg.kin_type != "none"
                                else 1.0) / (res.times[1] - res.times[0]))))
        last = L[-per:]
        print(f"[{name:>20}] OK  windows={len(L):4d}  "
              f"L_mean={last.mean():+9.2f}N  L_amp={(last.max()-last.min())/2:9.2f}N  "
              f"solves={res.n_solves}  wall={res.wall_s:.0f}s", flush=True)
    except Exception as e:
        print(f"[{name:>20}] FAIL: {type(e).__name__}: {str(e)[:90]}", flush=True)
