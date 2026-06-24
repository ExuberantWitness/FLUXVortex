import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
import flex_aircraft as F
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import NodalForceSet


def _resnorm(s_prev, s_new):
    """Window-end residual on wing vertex positions (state() dict)."""
    a = s_prev['verts'] if isinstance(s_prev, dict) else s_prev
    b = s_new['verts'] if isinstance(s_new, dict) else s_new
    return float(np.linalg.norm((b - a).ravel()))


def build_ac(substeps=16, iterations=1, adaptive=False, added_mass=False,
             interp="linear"):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                        V0=6.0, trim_aoa_deg=6.0, substeps=substeps)
    ac.provider.added_mass_operator = added_mass
    # rebuild the coupler with stronger settings (entry/provider reused)
    kw = dict(entry=ac.entry, provider=ac.provider, substeps=substeps,
              dt=ac.dtw / substeps, mode="two-pass", interp=interp,
              iterations=iterations)
    if adaptive:
        kw['adaptive_tol'] = 1e-4
        kw['residual_norm'] = lambda a, b: _resnorm(a, b)
    ac.pc = WindowPredictorCorrector(**kw)
    ac.pc.initialize(NodalForceSet(np.zeros(ac.entry.shell.ndof)))
    ac.pc.advance(n_substeps=1)
    return ac


def run(ac, N=30):
    lifts, defls = [], []
    div = None
    for i in range(N):
        try:
            o = ac.step_window()
            d = float(np.abs(ac.entry.state()['verts'][..., 2]).max())
        except Exception as e:
            div = i; lifts.append(float('nan')); break
        lifts.append(o['F_lift']); defls.append(d)
        if (not np.isfinite(o['F_lift'])) or d > 1e3:
            div = i; break
    lifts = np.array(lifts)
    seg = lifts[5:][np.isfinite(lifts[5:])] if len(lifts) > 5 else lifts
    lift_std = float(np.std(seg)) if len(seg) else float('nan')
    maxd = float(np.nanmax(defls)) if defls else float('nan')
    survived = div is None and len(lifts) >= N
    return survived, div, lift_std, maxd, len(lifts)


print("=" * 78)
print("(c) COUPLING-SCHEME fixes (the oracle's cure). substeps=16 unless noted.")
print("=" * 78)
configs = [
    ("baseline (iter=1, two-pass)",         dict(iterations=1)),
    ("Picard iter=4",                        dict(iterations=4)),
    ("Picard iter=8",                        dict(iterations=8)),
    ("adaptive Picard (iter<=12,tol=1e-4)",  dict(iterations=12, adaptive=True)),
    ("added_mass ON only",                   dict(added_mass=True)),
    ("added_mass + Picard iter=8",           dict(added_mass=True, iterations=8)),
    ("added_mass + adaptive Picard",         dict(added_mass=True, iterations=12, adaptive=True)),
    ("added_mass + adaptive + substeps=32",  dict(added_mass=True, iterations=12, adaptive=True, substeps=32)),
]
for label, kw in configs:
    try:
        ac = build_ac(**kw)
        surv, div, std, maxd, n = run(ac, 30)
        print(f"  {label:42s}: surv={surv!s:5} div@{(div if div is not None else '-')!s:>3} "
              f"lift_std={std:8.1f}N max_defl={maxd:.3g}m ({n}w)")
    except Exception as e:
        print(f"  {label:42s}: SETUP-EXC {type(e).__name__}: {str(e)[:50]}")
