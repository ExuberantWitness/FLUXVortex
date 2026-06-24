"""Oracle-faithful strong coupling for flex_aircraft: window-level fixed point on
the structural window-end state with Aitken Δ² dynamic relaxation, re-evaluating
the aero (FlapUVLMProvider.solve) on the RELAXED structural iterate. Monkeypatched
into a private subclass of WindowPredictorCorrector -- no source files edited."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
import flex_aircraft as F
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import NodalForceSet


class AitkenPC(WindowPredictorCorrector):
    """Strong-coupled window: iterate solve(aero on iterate) -> march -> relax."""
    pc_it: int = 20
    pc_tol: float = 1e-5
    omega0: float = 0.3

    def advance(self, n_substeps=None):
        if self._F_cur is None:
            raise RuntimeError("initialize first")
        t0 = self._t; tf = self._t
        n = self.substeps if n_substeps is None else n_substeps
        snap = self.entry.snapshot()
        F_prev, F_cur = self._F_prev, self._F_cur
        # predictor march with extrapolated force to get a starting window-end state
        pred = lambda beta: F_prev.affine(F_cur, 1.0 + beta)
        if n > 1:
            self._march(t0, tf, n - 1, pred)
        F_new = self.provider.solve(self.entry.state())
        # initial corrector march
        self.entry.restore(snap)
        self._march(t0, tf, n, lambda b: F_cur.affine(F_new, b))
        s_it = self.entry.state()['verts'].copy()
        omega = self.omega0; r_prev = None
        it = 0
        for it in range(self.pc_it):
            # re-solve aero on current structural window-end iterate
            self.entry.restore(snap)
            # march to window end so .state() == iterate, then solve there
            # (use the relaxed verts as the aero query state)
            F_q = self.provider.solve(dict(verts=s_it, vels=self.entry.state()['vels']))
            # march the window under interpolated F_cur->F_q
            self.entry.restore(snap)
            self._march(t0, tf, n, lambda b: F_cur.affine(F_q, b))
            s_solve = self.entry.state()['verts'].copy()
            r = (s_solve - s_it).ravel()
            rn = np.linalg.norm(r)
            sn = np.linalg.norm(s_solve.ravel()) + 1e-30
            if rn < self.pc_tol * sn:
                s_it = s_solve; F_new = F_q; break
            if r_prev is not None:
                dr = r - r_prev
                omega = -omega * float(np.dot(r_prev, dr)) / (float(np.dot(dr, dr)) + 1e-30)
                omega = float(np.clip(omega, 0.05, 1.0))
            s_new = s_it.ravel() + omega * r
            s_it = s_new.reshape(s_it.shape); r_prev = r
            F_new = F_q
        # final accepted march to the relaxed iterate's force
        self.entry.restore(snap)
        self._march(t0, tf, n, lambda b: F_cur.affine(F_new, b))
        self._last_iters = it + 1
        self.provider.commit(F_new)
        self._F_prev = self._F_cur; self._F_cur = F_new
        self._t = t0 + n * self.dt; self._window_index += 1
        from newton_pc.coupler import WindowStats
        return WindowStats(self._window_index, self._t, 1, 1, it + 1, float(rn) if 'rn' in dir() else None)


def build(substeps=16, added_mass=False, pc_it=20, omega0=0.3):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                        V0=6.0, trim_aoa_deg=6.0, substeps=substeps)
    ac.provider.added_mass_operator = added_mass
    pc = AitkenPC(entry=ac.entry, provider=ac.provider, substeps=substeps,
                  dt=ac.dtw / substeps, mode="two-pass")
    pc.pc_it = pc_it; pc.omega0 = omega0
    ac.pc = pc
    ac.pc.initialize(NodalForceSet(np.zeros(ac.entry.shell.ndof)))
    ac.pc.advance(n_substeps=1)
    return ac


def run(ac, N=30):
    lifts, defls, iters = [], [], []
    div = None
    for i in range(N):
        try:
            o = ac.step_window()
            d = float(np.abs(ac.entry.state()['verts'][..., 2]).max())
        except Exception:
            div = i; lifts.append(float('nan')); break
        lifts.append(o['F_lift']); defls.append(d)
        iters.append(getattr(ac.pc, '_last_iters', 1))
        if (not np.isfinite(o['F_lift'])) or d > 1e3:
            div = i; break
    lifts = np.array(lifts)
    seg = lifts[5:][np.isfinite(lifts[5:])] if len(lifts) > 5 else lifts
    std = float(np.std(seg)) if len(seg) else float('nan')
    mean = float(np.mean(seg)) if len(seg) else float('nan')
    maxd = float(np.nanmax(defls)) if defls else float('nan')
    return (div is None and len(lifts) >= N), div, std, mean, maxd, len(lifts), \
        (float(np.mean(iters)) if iters else 0)


print("=" * 78)
print("ORACLE-FAITHFUL strong coupling (window fixed-point + Aitken relaxation)")
print("=" * 78)
for label, kw in [
    ("Aitken pc_it=20 (no madd)",      dict(pc_it=20)),
    ("Aitken pc_it=20 + added_mass",   dict(pc_it=20, added_mass=True)),
    ("Aitken pc_it=30 omega0=0.5",     dict(pc_it=30, omega0=0.5)),
]:
    ac = build(**kw)
    surv, div, std, mean, maxd, n, mi = run(ac, 30)
    print(f"  {label:34s}: surv={surv!s:5} div@{(div if div is not None else '-')!s:>3} "
          f"lift_std={std:7.1f}N mean={mean:6.1f}N max_defl={maxd:.3g}m "
          f"avg_it={mi:.1f} ({n}w)")
