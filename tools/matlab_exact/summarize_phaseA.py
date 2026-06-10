"""Assemble the Phase A before/after matrix from explore_grad_pc.py outputs.

Each /tmp/explore_{mode}_bs{bs}.npy holds rows (boundary, t_ml, tip, pred_err,
n_fluid_cumulative). Reference = picard3 @ bs=34 (converged fixed point).
Reports, at matched physical times: tip error vs reference, vs MATLAB(=std34),
predictor inconsistency, fluid solves, and wall-clock if logs present.
"""
import numpy as np
import os, re

CASES = [
    ('std',     34), ('tangent', 34), ('picard2', 34), ('picard3', 34),
    ('std',     68), ('tangent', 68), ('picard2', 68),
]

data = {}
for mode, bs in CASES:
    p = f'/tmp/explore_{mode}_bs{bs}.npy'
    if os.path.exists(p):
        data[(mode, bs)] = np.load(p)

def walltime(mode, bs):
    log = {'std': '/tmp/expl_std.log', 'tangent': '/tmp/expl_tangent.log',
           'picard2': '/tmp/expl_picard2.log', 'picard3': '/tmp/expl_picard3.log'}
    if bs == 68:
        log = {'std': '/tmp/expl_std68.log', 'tangent': '/tmp/expl_tan68.log',
               'picard2': '/tmp/expl_pic68.log'}
    f = log.get(mode)
    if not f or not os.path.exists(f):
        return None
    t = None
    for line in open(f):
        m = re.search(r'\((\d+)s\)', line)
        if m:
            t = int(m.group(1))
    return t

ref = data.get(('picard3', 34))
if ref is None:
    ref = data.get(('picard2', 34))
    print('[!] picard3 missing - using picard2 as converged reference')
ml = data.get(('std', 34))   # std34 == MATLAB scheme

print(f"{'mode':>9} {'bs':>3} | {'t*_end':>6} {'tip_end':>13} | "
      f"{'err_vs_conv':>11} {'err_vs_ML':>11} | {'predIncons':>10} {'fluids':>6} {'wall_s':>6}")
for (mode, bs), arr in data.items():
    b, t, tip, perr, nf = arr[-1]
    # match reference at same physical time
    def tip_at(refarr, tt):
        i = np.argmin(np.abs(refarr[:, 1] - tt))
        return refarr[i, 2] if abs(refarr[i, 1] - tt) < 1e-9 else None
    tr = tip_at(ref, t) if ref is not None else None
    tm = tip_at(ml, t) if ml is not None else None
    e_conv = abs(tip - tr) / abs(tr) if tr else float('nan')
    e_ml = abs(tip - tm) / abs(tm) if tm else float('nan')
    w = walltime(mode, bs)
    print(f"{mode:>9} {bs:>3} | {t:6.3f} {tip:+.6e} | "
          f"{e_conv:11.3e} {e_ml:11.3e} | {perr:10.2e} {int(nf):6d} {w if w else '-':>6}")

print("\nNotes: err_vs_conv = |tip - picard3@34| / |.| (distance to converged fixed point)")
print("       err_vs_ML   = |tip - std@34|     (distance to MATLAB scheme)")
print("       predIncons  = max|X_fluidinput - X_corrected| at last boundary")
