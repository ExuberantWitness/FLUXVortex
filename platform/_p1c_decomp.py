"""P1c: single-variable decomposition of the P3 gate failure (U6 regressed 2.42->3.36, tw22.5 -3.68 left).

P2 changed TWO things at once (faure gate A0pre-strict< + vnf Polhamus saturation) — this isolates them,
plus the plan's fallback hypotheses (a0_crit 0.27, geo_stall double-count, faure off), on the two worst
conditions. All variants are cfg switches (zero fitting; a0_crit values are literature airfoil properties).

  python _p1c_decomp.py            # ~12 runs x 85-165s, prints component table (thrust contributions)
"""
import time, numpy as np
import warp as wp
wp.init()
from _v2_robo import gpu_run_twist
from _v2_repro_nc12 import HIRATO_COMMON

NC, NCYC, SPC = 4, 3, 180
CONDS = [  # (U, aoa, freq, tw, expT)  — the two P3 gate failures
    (6.0, 5.0, 2.0, 0.0, -2.69),
    (8.0, 5.0, 2.6, 22.5, -0.66),
]
VARIANTS = {  # name -> overrides on H2
    'H2':        dict(),                                          # both fixes ON (P3 result, re-run for components)
    'gate_only': dict(vnf_sat=False),                             # isolate: saturation removed
    'sat_only':  dict(faure_gate_pre=False),                      # isolate: gate fix removed
    'old':       dict(vnf_sat=False, faure_gate_pre=False),       # pre-P2 anchor (expect -0.12 / -7.24)
    'crit27':    dict(a0_crit=0.27),                              # LESP crit sensitivity (SD7003@Re2e4, literature)
    'H3':        dict(geo_stall=True, geo_stall_deg=12.0, geo_stall_width=16.0),   # double-count probe
    'nofaure':   dict(attached_drag='none'),                      # how much faure drag is there at all
}


def comps(r, spc):
    last = slice(-spc, None)
    f = lambda a: -2.0 * float(np.mean(np.asarray(a)[last]))     # body-X -> thrust contribution (+ = thrust)
    return f(r['Xh_bern']), f(r['Xh_les']), f(r['Xh_vtx']), -float(r['D_prof'])   # D_prof = 2*mean(Xh_pd[last])


for (U, aoa, freq, tw, expT) in CONDS:
    print(f"\n=== U={U} f={freq} tw={tw} aoa={aoa} | expT={expT:+.2f} ===", flush=True)
    print(f"{'variant':>10} {'L':>7} {'T':>7} {'dT':>7} | {'T_bern':>7} {'T_les':>7} {'T_vnf':>7} {'T_faure':>7}")
    for name, ov in VARIANTS.items():
        kw = dict(HIRATO_COMMON, **ov); t0 = time.time()
        try:
            r = gpu_run_twist(U=U, aoa_deg=aoa, freq=freq, twist_amp_deg=tw, twist_phase_deg=90.0,
                              nc=NC, ns=16, n_cycle=NCYC, steps_per_cycle=SPC, wake_rows=SPC, **kw)
            tb, tl, tv, tp = comps(r, SPC)
            print(f"{name:>10} {r['L_wind']:>+7.2f} {r['T_wind']:>+7.2f} {r['T_wind']-expT:>+7.2f} "
                  f"| {tb:>+7.2f} {tl:>+7.2f} {tv:>+7.2f} {tp:>+7.2f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"{name:>10} ERR {ex}", flush=True)
print("\nP1c DONE", flush=True)
