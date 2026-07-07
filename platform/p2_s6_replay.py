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


def _npz_path(U, freq):
    return os.path.join(DOCS, f"s6_defl_U{U:g}_f{freq:g}.npz")


def record(U, freq, n_cycles=2.0):
    """S5 coupled run; record aero-grid (du, dv) vs rigid frame per window."""
    import warp as wp                                       # noqa: F401
    from wing_system import WingModel, WingEntry
    from wing_iqn import add_iqnils
    from newton_pc import WindowPredictorCorrector
    from newton_pc.adapters.flap import FlapKinematics, FlapUVLMProvider

    period = 1.0 / freq
    alpha = np.deg2rad(AOA_DEG)
    dtw = (CHORD / 8) / U
    substeps = 40
    n_windows = int(round(n_cycles * period / dtw))
    model = WingModel(rib_depth=D_STAR)
    kin0 = FlapKinematics(np.deg2rad(AMP_DEG), period)

    class TopKin:                                            # stroke-top release
        def angles(self, t):
            return kin0.angles(t + period / 4.0)
    kin = TopKin()
    entry = WingEntry(model, kin, ramp_T=0.0, load_ramp_T=4 * dtw)
    V = U * np.array([np.cos(alpha), 0, np.sin(alpha)])
    p0 = FlapUVLMProvider(V, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                          particles=False, max_particles=1)
    F0 = p0.solve(entry.state())
    prov = FlapUVLMProvider(V, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1,
                            added_mass_operator=True)
    add_iqnils(prov)
    th0, thd0, thdd0 = kin.angles(0.0)
    c0_, s0_ = np.cos(th0), np.sin(th0)
    R0 = np.array([[1, 0, 0], [0, c0_, -s0_], [0, s0_, c0_]])
    pos_rot = (model.nodes + entry.q[model.trans_map]) @ R0.T
    entry.q[model.trans_map.ravel()] = (pos_rot - model.nodes).ravel()
    al = np.array([thdd0, 0, 0])
    entry.dq[:] = 0.0
    entry.a[model.trans_map.ravel()] = np.cross(al, pos_rot).ravel()
    for n_ in model.beam_nodes:
        entry.q[model.dof6[n_, 3:]] = [th0, 0, 0]
        entry.a[model.dof6[n_, 3:]] = al
    pc = WindowPredictorCorrector(
        entry=entry, provider=prov, substeps=substeps, dt=dtw / substeps,
        mode="two-pass", iterations=30, min_iterations=3,
        adaptive_tol=3e-5, adaptive_tol_rel=1e-3,
        residual_norm=lambda a, b: float(np.linalg.norm(
            np.asarray(b["verts"]) - np.asarray(a["verts"]))))
    pc.initialize(F0)
    pc.advance(n_substeps=1)

    ns_, nc_ = model.ns, model.nc
    g0 = np.zeros((ns_ + 1, nc_ + 1, 3))                     # rest aero surface
    g0[..., 0:2] = model.aero_rest2d
    g0[..., 2] = model.aero_off
    ts, dus, dvs, lifts, iters = [], [], [], [], []
    for w in range(n_windows):
        s = pc.advance()
        st = entry.state()
        t = pc._t
        th, thd, _ = kin.angles(t)
        c_, s2 = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c_, -s2], [0, s2, c_]])
        Rd = thd * np.array([[0, 0, 0], [0, -s2, -c_], [0, c_, -s2]])
        rig = (g0.reshape(-1, 3) @ R.T).reshape(ns_ + 1, nc_ + 1, 3)
        rigv = (g0.reshape(-1, 3) @ Rd.T).reshape(ns_ + 1, nc_ + 1, 3)
        du = st["verts"].transpose(1, 0, 2) - rig            # (ns+1, nc+1, 3)
        dv = st["vels"].transpose(1, 0, 2) - rigv
        ts.append(t); dus.append(du); dvs.append(dv)
        lifts.append(float(pc._F_cur.f.reshape(-1, 9)[:, 2].sum()))
        iters.append(s.iterations)
        if w % 10 == 0:
            print(f"  w={w:3d}/{n_windows} t={t:.3f}s it={s.iterations:2d} "
                  f"L={lifts[-1]:+7.2f} |du|max={np.abs(du).max()*1e3:5.1f}mm",
                  flush=True)
    xf = model.aero_rest2d[0, :, 0] / max(float(CHORD), 1e-9)  # root row = xf*c
    np.savez(_npz_path(U, freq), ts=np.array(ts), dus=np.array(dus),
             dvs=np.array(dvs), lifts=np.array(lifts), iters=np.array(iters),
             period=period, phase0=period / 4.0,
             xf=0.5 * (1 - np.cos(np.linspace(0, np.pi, nc_ + 1))),
             ys=np.linspace(0, 0.8, ns_ + 1))
    print(f"recorded {len(ts)} windows -> {_npz_path(U, freq)}  "
          f"(iters mean {np.mean(iters):.1f})")


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


def replay(U, freq, cfg_name="K0", nc=12, ns=16, n_cycle=4):
    from _v2_robo import gpu_run_twist
    sys.path.insert(0, _HERE)
    from _v2_repro_nc12 import CFG_PRESETS, spc_of
    kw = dict(CFG_PRESETS[cfg_name])
    spc = spc_of(U, freq)
    hook = DeformInterp(_npz_path(U, freq), nc, ns)
    out_flex = gpu_run_twist(U=U, aoa_deg=AOA_DEG, freq=freq, twist_amp_deg=0.0,
                             twist_phase_deg=90.0, nc=nc, ns=ns, n_cycle=n_cycle,
                             steps_per_cycle=spc, wake_rows=spc,
                             deform_hook=hook, **kw)
    out_rig = gpu_run_twist(U=U, aoa_deg=AOA_DEG, freq=freq, twist_amp_deg=0.0,
                            twist_phase_deg=90.0, nc=nc, ns=ns, n_cycle=n_cycle,
                            steps_per_cycle=spc, wake_rows=spc, **kw)
    Lf, Tf = float(out_flex["L_wind"]), float(out_flex["T_wind"])
    Lr, Tr = float(out_rig["L_wind"]), float(out_rig["T_wind"])
    print(f"S6 replay U={U} f={freq} [{cfg_name}]: "
          f"flex L={Lf:+.3f} T={Tf:+.3f} | rigid L={Lr:+.3f} T={Tr:+.3f} | "
          f"dL={Lf-Lr:+.3f} dT={Tf-Tr:+.3f}")
    return dict(flex=(Lf, Tf), rigid=(Lr, Tr))


if __name__ == "__main__":
    mode = sys.argv[1]
    U_, f_ = float(sys.argv[2]), float(sys.argv[3])
    if mode == "record":
        record(U_, f_)
    else:
        replay(U_, f_, sys.argv[4] if len(sys.argv) > 4 else "K0")
