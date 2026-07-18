"""回移植快环(裁定② 2026-07-18):A0 估计器 xref(Hirato Eq.6)→ downwash(S2 精确求积)。
单变量;6 红线工况;arm A = xref 基线一致性(对 s6_sweep_v4.json 缓存,应逐位近同);
arm B = downwash + a0_crit=0.27(隔离估计器效应);arm C = downwash + 文献 crit 扫
{0.21, 0.18, 0.14}(精确泛函下文献值首次合法适用;预登记假设:某文献值≈或优于基线)。"""
import sys, os, json, time
import numpy as np
HERE = "/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV"
os.chdir(HERE)
sys.path[:0] = [os.path.join(HERE, "platform"), os.path.join(HERE, "src"), HERE]
from _v2_robo import gpu_run_twist
from _v2_repro_nc12 import CFG_PRESETS, spc_of

MEAS = json.load(open("platform/docs/repro_data.json"))
CACHE = json.load(open("platform/docs/s6_sweep_v4.json"))

def at(key, x):
    d = MEAS.get(key)
    return None if d is None else float(np.interp(x, d["x"], d["exp"]))

BASE = dict(CFG_PRESETS["H16"], fsep_lag=True, geo_stall_vec=True, cosine_chord='le',
            les_sep='plateau_fn', d_para=0.5, attached_drag="uiuc")
CONDS = [
    (8.0, 2.3, 0.0, 5.0, "17|a|2.3", "17|b|2.3", 0.0),
    (8.0, 1.4, 22.5, 5.0, "18|a|8.0", "18|b|8.0", 1.4),
    (8.0, 2.6, 22.5, 5.0, "18|a|8.0", "18|b|8.0", 2.6),
    (8.0, 2.3, 45.0, 5.0, "17|a|2.3", "17|b|2.3", 45.0),
    (8.0, 2.6, 22.5, 15.0, "19|a|15", "19|b|15", 2.6),
    (6.0, 2.6, 22.5, 5.0, "18|a|6.0", "18|b|6.0", 2.6),
]

def run(tag, **over):
    gT_l, gL_l = [], []
    for U, f, tw, aoa, kT, kL, x in CONDS:
        spc = spc_of(U, f); t0 = time.time()
        r = gpu_run_twist(U=U, aoa_deg=aoa, freq=f, flap_amp_deg=22.5,
                          twist_amp_deg=0.5 * tw, twist_phase_deg=90.0, nc=12,
                          ns=16, n_cycle=4, steps_per_cycle=spc, wake_rows=spc,
                          **dict(BASE, **over))
        L, T = float(r["L_wind"]), float(r["T_wind"])
        mT, mL = at(kT, x), at(kL, x)
        ck = f"{U:g}_{f:g}_{tw:g}_{aoa:g}"
        cc = CACHE.get(ck)
        cch = f" cacheΔ L{L-cc['L']:+.2f}/T{T-cc['T']:+.2f}" if cc else ""
        print(f"[{tag}] U{U:g}/f{f}/tw{tw:g}/aoa{aoa:g}: gapT={T-mT:+.2f} gapL={L-mL:+.2f}"
              f"{cch}  ({time.time()-t0:.0f}s)", flush=True)
        gT_l.append(abs(T - mT)); gL_l.append(abs(L - mL))
    print(f"[{tag}] MAE: T {np.mean(gT_l):.2f} | L {np.mean(gL_l):.2f}", flush=True)

run("A xref@0.27 基线校验")
run("B dw@0.27", a0_mode='downwash')
for a0 in (0.21, 0.18, 0.14):
    run(f"C dw@{a0}", a0_mode='downwash', a0_crit=a0)
print("GATE DONE", flush=True)
