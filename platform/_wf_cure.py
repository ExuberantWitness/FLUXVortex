"""Decisive test: does the ORACLE's cure (strong fixed-point + Aitken relaxation)
stabilize flex_aircraft, vs the loose two-pass baseline? Lean config set."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
import flex_aircraft as F
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import NodalForceSet
from newton_pc.coupler import WindowStats


def _resn(a, b):
    va = a['verts'] if isinstance(a, dict) else a
    vb = b['verts'] if isinstance(b, dict) else b
    return float(np.linalg.norm((vb - va).ravel()))


# ---- oracle-faithful Aitken window coupler ----
class AitkenPC(WindowPredictorCorrector):
    pc_it = 25
    omega0 = 0.3
    pc_tol = 1e-5

    def advance(self, n_substeps=None):
        t0 = self._t; tf = self._t
        n = self.substeps if n_substeps is None else n_substeps
        snap = self.entry.snapshot()
        F_prev, F_cur = self._F_prev, self._F_cur
        if n > 1:
            self._march(t0, tf, n - 1, lambda b: F_prev.affine(F_cur, 1.0 + b))
        F_new = self.provider.solve(self.entry.state())
        self.entry.restore(snap)
        self._march(t0, tf, n, lambda b: F_cur.affine(F_new, b))
        s_it = self.entry.state()['verts'].copy()
        vels = self.entry.state()['vels'].copy()
        omega = self.omega0; r_prev = None; it = 0; rn = 0.0
        for it in range(self.pc_it):
            F_q = self.provider.solve(dict(verts=s_it, vels=vels))
            self.entry.restore(snap)
            self._march(t0, tf, n, lambda b: F_cur.affine(F_q, b))
            st = self.entry.state()
            s_solve = st['verts'].copy(); vels = st['vels'].copy()
            r = (s_solve - s_it).ravel(); rn = np.linalg.norm(r)
            sn = np.linalg.norm(s_solve.ravel()) + 1e-30
            F_new = F_q
            if rn < self.pc_tol * sn:
                s_it = s_solve; break
            if r_prev is not None:
                dr = r - r_prev
                omega = -omega * float(r_prev @ dr) / (float(dr @ dr) + 1e-30)
                omega = float(np.clip(omega, 0.05, 1.0))
            s_it = (s_it.ravel() + omega * r).reshape(s_it.shape); r_prev = r
        self.entry.restore(snap)
        self._march(t0, tf, n, lambda b: F_cur.affine(F_new, b))
        self._last_iters = it + 1
        self.provider.commit(F_new)
        self._F_prev = self._F_cur; self._F_cur = F_new
        self._t = t0 + n * self.dt; self._window_index += 1
        return WindowStats(self._window_index, self._t, 1, 1, it + 1, rn)


def make(kind, substeps=16, added_mass=False, iterations=1, adaptive=False,
         pc_it=25, omega0=0.3):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                        V0=6.0, trim_aoa_deg=6.0, substeps=substeps)
    ac.provider.added_mass_operator = added_mass
    if kind == 'aitken':
        pc = AitkenPC(entry=ac.entry, provider=ac.provider, substeps=substeps,
                      dt=ac.dtw / substeps, mode="two-pass")
        pc.pc_it = pc_it; pc.omega0 = omega0
    else:
        kw = dict(entry=ac.entry, provider=ac.provider, substeps=substeps,
                  dt=ac.dtw / substeps, mode="two-pass", iterations=iterations)
        if adaptive:
            kw['adaptive_tol'] = 1e-4; kw['residual_norm'] = _resn
        pc = WindowPredictorCorrector(**kw)
    ac.pc = pc
    ac.pc.initialize(NodalForceSet(np.zeros(ac.entry.shell.ndof)))
    ac.pc.advance(n_substeps=1)
    return ac


def run(ac, N=30):
    lifts, defls, its = [], [], []
    div = None
    for i in range(N):
        try:
            o = ac.step_window()
            d = float(np.abs(ac.entry.state()['verts'][..., 2]).max())
        except Exception:
            div = i; lifts.append(float('nan')); break
        lifts.append(o['F_lift']); defls.append(d)
        its.append(getattr(ac.pc, '_last_iters', 1))
        if (not np.isfinite(o['F_lift'])) or d > 1e3:
            div = i; break
    lifts = np.array(lifts)
    seg = lifts[5:][np.isfinite(lifts[5:])] if len(lifts) > 5 else lifts
    std = float(np.std(seg)) if len(seg) else float('nan')
    mean = float(np.mean(seg)) if len(seg) else float('nan')
    maxd = float(np.nanmax(defls)) if defls else float('nan')
    return (div is None and len(lifts) >= N), div, std, mean, maxd, len(lifts), \
        (float(np.mean(its)) if its else 1)


CFGS = [
    ("baseline two-pass iter=1",        dict(kind='native', iterations=1)),
    ("native Picard iter=6",            dict(kind='native', iterations=6)),
    ("native adaptive (iter<=8,1e-4)",  dict(kind='native', iterations=8, adaptive=True)),
    ("AITKEN strong (no madd)",         dict(kind='aitken', pc_it=25)),
    ("AITKEN strong + added_mass",      dict(kind='aitken', pc_it=25, added_mass=True)),
]
print("CONFIG                               surv  div  lift_std(N)   mean(N)  max_defl(m)  it  w")
print("-" * 92)
for label, kw in CFGS:
    try:
        ac = make(**kw)
        surv, div, std, mean, maxd, n, mi = run(ac, 30)
        print(f"{label:36s} {surv!s:5} {(div if div is not None else '-')!s:>3}  "
              f"{std:11.2f}  {mean:7.2f}  {maxd:10.4g}  {mi:4.1f} {n:2d}", flush=True)
    except Exception as e:
        print(f"{label:36s} EXC {type(e).__name__}: {str(e)[:45]}", flush=True)
