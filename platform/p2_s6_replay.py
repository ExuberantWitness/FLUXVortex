"""P2-S6 v1: flexible-wing Fig17/18/19 via DEFORMATION REPLAY.

Pipeline (one-way replay, recorded approximation):
  1. RECORD: run the S5 strong-coupled sim (light provider, madd+IQN, stroke-
     top start) at a data.md condition; store per-window the AERO-grid
     deformation offsets vs the rigid kinematic frame:
         du(t) = verts_flex(t) - verts_rigid(t),  dv(t) likewise
     over the LAST full cycle (first cycle = transient settle).
  2. REPLAY: feed du/dv into gpu_run_twist(deform_hook=...) — the VALIDATED
     production closure stack (K0/H16) evaluates forces on the deformed
     motion. Phase-aligned periodic interpolation in time; bilinear (xf, y)
     interpolation from the coupled (nc=8, ns=16) grid to the replay grid.

Honest scope: the deformation field comes from the light-closure coupled
solve while the forces come from the full closures — first-order consistent
when the flexible correction is small; recorded as the S6 v1 approximation.
Conditions with twist actuation need the twist-prescribed root (S6 v2).

Usage:
  python platform/p2_s6_replay.py record U FREQ           # -> docs/s6_defl_*.npz
  python platform/p2_s6_replay.py replay U FREQ [CFG]     # -> L,T vs rigid
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

DOCS = os.path.join(_HERE, "docs")
D_STAR = 3.78e-3
CHORD, AOA_DEG = 0.287, 5.0
AMP_DEG = 45.0


def _npz_path(U, freq, tw=0.0):
    suf = f"_tw{tw:g}" if tw else ""
    return os.path.join(DOCS, f"s6_defl_U{U:g}_f{freq:g}{suf}.npz")


X_MAIN = 0.08858                     # main-spar line (m aft of LE), assembly v3


class TwistTopKin:
    """Stroke-top release kinematics with the RoboEagle double-crank twist:
    flap theta about body-x COMPOSED with the mechanism pitch psi about the
    MAIN-SPAR line (paper 2.2: the slider twist reaches the wing at the outer
    rod connection -> the root assembly pitches about the spar; the spanwise
    twist DISTRIBUTION is left to the structure). Twist amplitude 2*beta =
    data.md tw (mechanism definition); psi leads flap by 90 deg."""

    def __init__(self, amp_f_rad, period, tw_rad):
        from newton_pc.adapters.flap import FlapKinematics
        self._kin0 = FlapKinematics(amp_f_rad, period)
        self.period = period
        self.beta = 0.5 * tw_rad
        self.om = 2.0 * np.pi / period

    def angles(self, t):                       # flap component (ramp, C_add)
        return self._kin0.angles(t + self.period / 4.0)

    def _psi(self, t):
        # production convention: twist angle psi_p = A_t cos(Om t_rig) applied
        # by [x' = xe+(x-xe)c - z s, z' = (x-xe)s + z c] = rotation about -y.
        # In STANDARD +y rotation terms that is -psi_p; with record time
        # t = t_rig - T/4 (cos -> -sin):  psi_std(t) = +beta sin(Om t).
        return self.beta * np.sin(self.om * t)

    def root_rot(self, t):
        th = self.angles(t)[0]
        ps = self._psi(t)
        cx, sx = np.cos(th), np.sin(th)
        cy, sy = np.cos(ps), np.sin(ps)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Rx @ Ry

    def root_map(self, t, X):
        th = self.angles(t)[0]
        ps = self._psi(t)
        cx, sx = np.cos(th), np.sin(th)
        cy, sy = np.cos(ps), np.sin(ps)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pm = np.array([X_MAIN, 0.0, 0.0])
        return (pm + (np.asarray(X) - pm) @ Ry.T) @ Rx.T


def record(U, freq, tw_deg=0.0, n_cycles=2.0, rc0_scale=1.0, resume=None,
           ckpt_every=15):
    """S5 coupled run; record aero-grid (du, dv) vs rigid frame per window.
    tw_deg > 0: mechanism twist (root pitch about the main spar, TwistTopKin);
    the du/dv reference is then the production LINEAR-twist rigid field so
    that replay total = production rigid + du = our flexible surface.
    rc0_scale: near-field vortex-core scale (reversal regularization).
    resume: path to a .ckpt.npz saved by a previous run — restarts the
    coupled state (structure + wake + coupler anchors) at that window."""
    import pickle
    import warp as wp                                       # noqa: F401
    from scipy.spatial.transform import Rotation as _Rot
    from wing_system import WingModel, WingEntry
    from wing_iqn import add_iqnils
    from newton_pc import WindowPredictorCorrector
    from newton_pc.adapters.flap import FlapUVLMProvider, NodalForceSet

    period = 1.0 / freq
    alpha = np.deg2rad(AOA_DEG)
    dtw = (CHORD / 8) / U
    substeps = 40
    n_windows = int(round(n_cycles * period / dtw))
    model = WingModel(rib_depth=D_STAR)
    kin = TwistTopKin(np.deg2rad(AMP_DEG), period, np.deg2rad(tw_deg))
    entry = WingEntry(model, kin, ramp_T=0.0, load_ramp_T=4 * dtw)
    V = U * np.array([np.cos(alpha), 0, np.sin(alpha)])
    # wake settings (user-directed 2026-07-07): K=18 rows (~2.3 chords) so the
    # wake actually DEVELOPS before the measurement cycle (K=6 never had a
    # wake "in place" — start/reversal transients dominated); wake core FLOOR
    # at 0.5 x row spacing (= chord/16, resolution-adaptive) regularizes the
    # stroke-reversal near field (bound AIC untouched).
    wcm = 0.5 * U * dtw * rc0_scale
    p0 = FlapUVLMProvider(V, 1.225, dtw, K=18, nu=15.06e-6, chord=CHORD,
                          particles=False, max_particles=1, wake_core_min=wcm,
                          kirch_stall=True)
    F0 = p0.solve(entry.state())
    prov = FlapUVLMProvider(V, 1.225, dtw, K=18, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1,
                            added_mass_operator=True, wake_core_min=wcm,
                            kirch_stall=True)
    add_iqnils(prov, mann_after=20, final_at=44)   # sync w/ iterations=45
    # kinematically consistent IC from the FULL rigid root motion at t=0
    # (flap theta_dot(0)=0 at the stroke top, but the twist RATE is nonzero)
    dlt = 1e-6
    u_pre = entry.q[model.trans_map].copy()
    u_pre[:model.nc + 1] = 0.0        # root row: __init__ already wrote the
    # PRESCRIBED (rotated) values there — rotating them again would double-
    # rotate the root (measured 45 deg root-band distortion at t=0)
    pos_pre = model.nodes + u_pre
    pos0 = kin.root_map(0.0, pos_pre)
    posp = kin.root_map(dlt, pos_pre)
    posm = kin.root_map(-dlt, pos_pre)
    entry.q[model.trans_map.ravel()] = (pos0 - model.nodes).ravel()
    entry.dq[:] = 0.0
    entry.dq[model.trans_map.ravel()] = ((posp - posm) / (2 * dlt)).ravel()
    entry.a[model.trans_map.ravel()] = ((posp - 2 * pos0 + posm) / dlt ** 2).ravel()
    rv0 = _Rot.from_matrix(kin.root_rot(0.0)).as_rotvec()
    drv = (_Rot.from_matrix(kin.root_rot(dlt)).as_rotvec()
           - _Rot.from_matrix(kin.root_rot(-dlt)).as_rotvec()) / (2 * dlt)
    for n_ in model.beam_nodes:
        entry.q[model.dof6[n_, 3:]] = rv0
        entry.dq[model.dof6[n_, 3:]] = drv
        entry.a[model.dof6[n_, 3:]] = 0.0
    pc = WindowPredictorCorrector(
        entry=entry, provider=prov, substeps=substeps, dt=dtw / substeps,
        # 45 iterations: the stroke-REVERSAL windows (wing re-crossing its own
        # wake) converge slowly (probe: res 6e-5..9e-4 at the 30 cap; a capped
        # commit there occasionally seeds a divergence within ~6 windows)
        mode="two-pass", iterations=45, min_iterations=3,
        adaptive_tol=3e-5, adaptive_tol_rel=1e-3,
        residual_norm=lambda a, b: float(np.linalg.norm(
            np.asarray(b["verts"]) - np.asarray(a["verts"]))))
    ns_, nc_ = model.ns, model.nc
    g0 = np.zeros((ns_ + 1, nc_ + 1, 3))                     # rest aero surface
    g0[..., 0:2] = model.aero_rest2d
    g0[..., 2] = model.aero_off
    # rigid REFERENCE field for du/dv: flap + the production LINEAR-twist
    # pitch about the swept axis (twisted_corners math), so that the replay
    # total (production rigid + du) equals our flexible surface exactly
    import _v2_robogeom as _rg
    ys_st = np.linspace(0.0, 0.8, ns_ + 1)
    xe_st = _rg.axis_x(ys_st)[:, None]
    yfrac = (ys_st / 0.8)[:, None]
    om_ = 2.0 * np.pi / period
    tw_rad = np.deg2rad(tw_deg)

    def rigid_ref(t):
        th = kin.angles(t)[0]
        psl = tw_rad * yfrac * (-np.sin(om_ * t))            # production phase
        g = g0.copy()
        dx = g0[..., 0] - xe_st
        cz, sz = np.cos(psl), np.sin(psl)
        g[..., 0] = xe_st + dx * cz - g0[..., 2] * sz
        g[..., 2] = dx * sz + g0[..., 2] * cz
        c_, s_ = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])
        return (g.reshape(-1, 3) @ R.T).reshape(ns_ + 1, nc_ + 1, 3)
    ckpt_path = _npz_path(U, freq, tw_deg) + ".ckpt"
    ts, dus, dvs, lifts, thrusts, iters = [], [], [], [], [], []
    w_start = 0
    if resume and os.path.exists(resume):
        ck = pickle.load(open(resume, "rb"))
        entry.q[:] = ck["q"]; entry.dq[:] = ck["dq"]; entry.a[:] = ck["a"]
        entry.t = ck["et"]
        prov.pts = ck["pts"]; prov.gam = list(ck["gam"])
        prov.ages = list(ck["ages"])
        prov.gamma_prev = ck["gamma_prev"]; prov._gb_prev = ck["gb_prev"]
        pc._t = ck["t"]; pc._window_index = ck["wi"]
        pc._F_prev = NodalForceSet(ck["Fp_f"], madd=ck["Fp_m"], a_lag=ck["Fp_al"])
        pc._F_cur = NodalForceSet(ck["Fc_f"], madd=ck["Fc_m"], a_lag=ck["Fc_al"])
        ts, dus, dvs = list(ck["ts"]), list(ck["dus"]), list(ck["dvs"])
        lifts, thrusts, iters = (list(ck["lifts"]), list(ck["thrusts"]),
                                 list(ck["iters"]))
        w_start = ck["w"] + 1
        print(f"  resumed from {resume} at window {w_start} (t={pc._t:.4f})",
              flush=True)
    else:
        pc.initialize(F0)
        pc.advance(n_substeps=1)

    def _save_ckpt(w):
        ck = dict(w=w, t=pc._t, wi=pc._window_index, et=entry.t,
                  q=entry.q.copy(), dq=entry.dq.copy(), a=entry.a.copy(),
                  pts=prov.pts, gam=prov.gam, ages=prov.ages,
                  gamma_prev=prov.gamma_prev, gb_prev=prov._gb_prev,
                  Fp_f=pc._F_prev.f, Fp_m=pc._F_prev.madd,
                  Fp_al=getattr(pc._F_prev, "a_lag", None),
                  Fc_f=pc._F_cur.f, Fc_m=pc._F_cur.madd,
                  Fc_al=getattr(pc._F_cur, "a_lag", None),
                  ts=ts, dus=dus, dvs=dvs, lifts=lifts, thrusts=thrusts,
                  iters=iters)
        with open(ckpt_path, "wb") as fh:
            pickle.dump(ck, fh)

    import time as _time
    t0_ = _time.time()
    fail = None
    for w in range(w_start, n_windows):
        try:
            s = pc.advance()
        except RuntimeError as e:
            fail = f"w{w}: {e}"
            print(f"  FAIL {fail} — saving what was recorded", flush=True)
            break
        st = entry.state()
        t = pc._t
        dt2 = 1e-6
        rig = rigid_ref(t)
        rigv = (rigid_ref(t + dt2) - rigid_ref(t - dt2)) / (2 * dt2)
        du = st["verts"].transpose(1, 0, 2) - rig            # (ns+1, nc+1, 3)
        dv = st["vels"].transpose(1, 0, 2) - rigv
        ts.append(t); dus.append(du); dvs.append(dv)
        Fp = (pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
              if pc._F_cur.payload else np.zeros(3))
        lifts.append(float(-Fp[0] * np.sin(alpha) + Fp[2] * np.cos(alpha)))
        thrusts.append(float(-(Fp[0] * np.cos(alpha) + Fp[2] * np.sin(alpha))))
        iters.append(s.iterations)
        if ckpt_every and (w + 1) % ckpt_every == 0:
            _save_ckpt(w)
        if w % 10 == 0:
            print(f"  w={w:3d}/{n_windows} t={t:.3f}s it={s.iterations:2d} "
                  f"L={lifts[-1]:+7.2f} |du|max={np.abs(du).max()*1e3:5.1f}mm "
                  f"[{_time.time()-t0_:.0f}s]", flush=True)
    np.savez(_npz_path(U, freq, tw_deg), ts=np.array(ts), dus=np.array(dus),
             dvs=np.array(dvs), lifts=np.array(lifts),
             thrusts=np.array(thrusts), iters=np.array(iters),
             period=period, phase0=period / 4.0, fail=str(fail),
             xf=0.5 * (1 - np.cos(np.linspace(0, np.pi, nc_ + 1))),
             ys=np.linspace(0, 0.8, ns_ + 1))
    wpc = int(round(period / dtw))
    if len(lifts) >= wpc:
        print(f"  cycle-mean (last cycle, x2 half-wing): "
              f"2L={2*np.mean(lifts[-wpc:]):+.2f} N  "
              f"2T={2*np.mean(thrusts[-wpc:]):+.2f} N")
    print(f"recorded {len(ts)} windows -> {_npz_path(U, freq, tw_deg)}  "
          f"(iters mean {np.mean(iters):.1f}, fail={fail})")


class DeformInterp:
    """Phase-periodic time interpolation + bilinear (xf, y) grid resampling of
    the recorded deformation, aligned to gpu_run_twist's theta = A sin(Om t)."""

    def __init__(self, npz, nc_to, ns_to, half_span=0.8):
        d = np.load(npz)
        self.period = float(d["period"])
        self.phase0 = float(d["phase0"])                     # record started at top
        ts, dus, dvs = d["ts"], d["dus"], d["dvs"]
        # keep the LAST full cycle, map to phase in [0, T)
        keep = ts >= (ts[-1] - self.period)
        self.tp = (ts[keep] + self.phase0) % self.period     # replay phase
        order = np.argsort(self.tp)
        self.tp = self.tp[order]
        self.du = dus[keep][order]
        self.dv = dvs[keep][order]
        xf_s, ys_s = d["xf"], d["ys"]
        xf_t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc_to + 1)))
        ys_t = np.linspace(0, half_span, ns_to + 1)
        # bilinear weights source->target (separable)
        self.wx = self._lin_w(xf_s, xf_t)                    # (nc_to+1, nc_s+1)
        self.wy = self._lin_w(ys_s, ys_t)                    # (ns_to+1, ns_s+1)

    @staticmethod
    def _lin_w(src, tgt):
        W = np.zeros((len(tgt), len(src)))
        for i, x in enumerate(tgt):
            j = np.clip(np.searchsorted(src, x) - 1, 0, len(src) - 2)
            t = np.clip((x - src[j]) / max(src[j + 1] - src[j], 1e-12), 0, 1)
            W[i, j] = 1 - t
            W[i, j + 1] = t
        return W

    def _resample(self, f):                                  # (ns_s+1, nc_s+1, 3)
        return np.einsum("ab,bck,dc->adk", self.wy, f, self.wx)

    def __call__(self, t):
        ph = t % self.period
        k = np.clip(np.searchsorted(self.tp, ph) - 1, 0, len(self.tp) - 2)
        w = np.clip((ph - self.tp[k]) / max(self.tp[k + 1] - self.tp[k], 1e-12),
                    0, 1)
        du = (1 - w) * self.du[k] + w * self.du[k + 1]
        dv = (1 - w) * self.dv[k] + w * self.dv[k + 1]
        # gpu_run_twist corners are (nc+1, ns+1, 3): transpose after resample
        return (self._resample(du).transpose(1, 0, 2),
                self._resample(dv).transpose(1, 0, 2))


def replay(U, freq, tw=0.0, cfg_name="K0", nc=12, ns=16, n_cycle=4):
    from _v2_robo import gpu_run_twist
    sys.path.insert(0, _HERE)
    from _v2_repro_nc12 import CFG_PRESETS, spc_of
    kw = dict(CFG_PRESETS[cfg_name])
    spc = spc_of(U, freq)
    hook = DeformInterp(_npz_path(U, freq, tw), nc, ns)
    out_flex = gpu_run_twist(U=U, aoa_deg=AOA_DEG, freq=freq, twist_amp_deg=tw,
                             twist_phase_deg=90.0, nc=nc, ns=ns, n_cycle=n_cycle,
                             steps_per_cycle=spc, wake_rows=spc,
                             deform_hook=hook, **kw)
    out_rig = gpu_run_twist(U=U, aoa_deg=AOA_DEG, freq=freq, twist_amp_deg=tw,
                            twist_phase_deg=90.0, nc=nc, ns=ns, n_cycle=n_cycle,
                            steps_per_cycle=spc, wake_rows=spc, **kw)
    Lf, Tf = float(out_flex["L_wind"]), float(out_flex["T_wind"])
    Lr, Tr = float(out_rig["L_wind"]), float(out_rig["T_wind"])
    print(f"S6 replay U={U} f={freq} tw={tw} [{cfg_name}]: "
          f"flex L={Lf:+.3f} T={Tf:+.3f} | rigid L={Lr:+.3f} T={Tr:+.3f} | "
          f"dL={Lf-Lr:+.3f} dT={Tf-Tr:+.3f}")
    return dict(flex=(Lf, Tf), rigid=(Lr, Tr))


if __name__ == "__main__":
    mode = sys.argv[1]
    U_, f_ = float(sys.argv[2]), float(sys.argv[3])
    tw_ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    if mode == "record":
        record(U_, f_, tw_)
    else:
        replay(U_, f_, tw_, sys.argv[5] if len(sys.argv) > 5 else "K0")
