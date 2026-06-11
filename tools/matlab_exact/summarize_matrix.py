"""Assemble the M*-sweep matrix: err-vs-converged + fluid count per scheme."""
import numpy as np, glob, os, re

def load(pat):
    fs = glob.glob(pat)
    return np.load(fs[0]) if fs else None

REF = {}
for m in ['1.0', '0.5', '0.4', '0.3']:
    r = load(f'/tmp/pilot_linear_bs34_ss1_m{m}_p3.npy')
    if r is None and m == '1.0':
        r = load('/tmp/explore_picard3_bs34.npy')   # (b,t,tip,prederr,nf)
    REF[m] = r

print(f"{'M*':>5} {'scheme':>9} | {'tip_end':>13} {'err_vs_conv':>12} {'fluids':>7}")
for m in ['1.0', '0.5', '0.4', '0.3']:
    ref = REF[m]
    if ref is None:
        print(f"{m:>5}  [no picard3 reference yet]"); continue
    tr = ref[-1, 2]
    for sch, pat in [('std', f'/tmp/pilot_linear_bs34_ss1_m{m}_p1.npy'),
                     ('picard2', f'/tmp/pilot_linear_bs34_ss1_m{m}_p2.npy'),
                     ('hermite', f'/tmp/pilot_hermite_bs34_ss1_m{m}_p1.npy'),
                     ('picard3', f'/tmp/pilot_linear_bs34_ss1_m{m}_p3.npy')]:
        a = load(pat)
        if a is None and sch == 'std' and m == '1.0':
            a = load('/tmp/explore_std_bs34.npy')
        if a is None and sch == 'hermite' and m == '1.0':
            a = load('/tmp/pilot_hermite_bs34_ss1_m1.0_p1.npy')
        if a is None: continue
        tip = a[-1, 2]; nf = int(a[-1, -1])
        err = abs(tip - tr) / abs(tr)
        print(f"{m:>5} {sch:>9} | {tip:+.6e} {err:12.3e} {nf:7d}")
