"""Elastic flapping GIF: bird-scale carbon-fiber wing, full coupled FSI.

Bird-scale quasi-isotropic CFRP wing (0.3 x 1.2 m, 3.5 mm, E=60 GPa,
rho=1600), root-driven flapping at 2 Hz +/- 10 deg in 6 m/s flow. Structural
damping zeta_modal ~ 0.22 saturates the near-resonant bending to a bounded
~0.42 m tip deflection (34% span) -- visibly elastic, not rigid. Particle
wake with moment-conserving merge population control.

Each frame shows: wing surface colored by ELASTIC BENDING (elastic z minus
rigid-kinematic z, coolwarm), near-field VLM ring lattice, particle cloud.

Usage: python flap_arena/render_bird_cfrp.py [--cycles 4] [--amp 10] [--every 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from newton_pc import WindowPredictorCorrector  # noqa: E402
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics,  # noqa: E402
                                     FlapUVLMProvider, NodalForceSet)

# bird-scale CFRP arena
CHORD, SPAN, NC, NS = 0.3, 1.2, 5, 6
V_INF, ALPHA = 6.0, np.deg2rad(2.0)
RHO, NU = 1.225, 15.06e-6
PERIOD = 0.5
THICK, RHO_S, E0 = 3.5e-3, 1600.0, 60e9
DAMPING = 0.03
BEND_MAX = 0.42


def render_frame(ax, verts, wake_pts, p_pos, p_alpha, t, n_p, lift, bend):
    ax.cla()
    X, Y, Z = verts[..., 0], verts[..., 1], verts[..., 2]
    face = plt.cm.coolwarm(0.5 + 0.5 * np.clip(bend / BEND_MAX, -1, 1))
    fc = 0.25 * (face[:-1, :-1] + face[1:, :-1]
                 + face[:-1, 1:] + face[1:, 1:])
    ax.plot_surface(X, Y, Z, facecolors=fc, alpha=0.97, edgecolor="#222222",
                    linewidth=0.3, shade=False)
    if wake_pts is not None:
        W = wake_pts
        for r in range(W.shape[0]):
            ax.plot(W[r, :, 0], W[r, :, 1], W[r, :, 2], color="#888888",
                    lw=0.5, alpha=0.6)
        for j in range(W.shape[1]):
            ax.plot(W[:, j, 0], W[:, j, 1], W[:, j, 2], color="#888888",
                    lw=0.35, alpha=0.45)
    if len(p_pos):
        s = np.linalg.norm(p_alpha, axis=1)
        order = np.argsort(p_pos[:, 0])
        ax.scatter(p_pos[order, 0], p_pos[order, 1], p_pos[order, 2],
                   c=s[order], cmap="viridis", s=5.0, alpha=0.8,
                   vmin=0.0, vmax=np.percentile(s, 95), linewidths=0)
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.1, 1.4)
    ax.set_zlim(-0.62, 0.62)
    ax.set_box_aspect((3.0, 1.5, 1.24), zoom=1.4)
    ax.view_init(elev=20, azim=-122)
    ax.set_axis_off()
    ax.set_title(f"CFRP wing  t={t:5.2f}s   bend(tip)={bend.max() - bend.min():.3f} m"
                 f"   particles={n_p:4d}   L={lift:+6.1f} N",
                 fontsize=10, family="monospace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=float, default=4.0)
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--substeps", type=int, default=128)
    ap.add_argument("--out", default="flap_arena/out/flap_cfrp_elastic.gif")
    args = ap.parse_args()

    dtw = (CHORD / NC) / V_INF
    n_windows = int(round(args.cycles * PERIOD / dtw))
    kin = FlapKinematics(np.deg2rad(args.amp), PERIOD)
    entry = FlapEntry(CHORD, SPAN, NC, NS, kin, mode="elastic", kscale=1.0,
                      thickness=THICK, rho_s=RHO_S, E0=E0, damping=DAMPING)
    V_vec = V_INF * np.array([np.cos(ALPHA), 0.0, np.sin(ALPHA)])
    provider = FlapUVLMProvider(V_vec, RHO, dtw, K=8, nu=NU, chord=CHORD,
                                particles=True, max_particles=10**6,
                                pop_scheme="merge", merge_eps=1e-4)
    provider.merge_protect_dist = 1.5
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=args.substeps,
                                  dt=dtw / args.substeps, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))

    fig = plt.figure(figsize=(9.6, 5.0), dpi=115)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    frames = []
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        pc.advance()
        if w % args.every:
            continue
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
        L = -F[0] * np.sin(ALPHA) + F[2] * np.cos(ALPHA)
        st = entry.state()
        th = kin.angles(pc._t)[0]
        zr = (entry.nodes0[:, 1] * np.sin(th)).reshape(NS + 1, NC + 1).T
        bend = st["verts"][..., 2] - zr
        render_frame(ax, st["verts"], provider.pts, provider.p_pos,
                     provider.p_alpha, pc._t, len(provider.p_pos), L, bend)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if w % 20 == 0:
            print(f"w={w:4d} t={pc._t:5.2f}s bend={bend.max()-bend.min():.3f}m "
                  f"n_p={len(provider.p_pos):4d} frames={len(frames)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    plt.close(fig)

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=80, loop=0, optimize=True)
    print(f"GIF: {args.out}  frames={len(imgs)}  "
          f"{os.path.getsize(args.out)/1e6:.1f} MB  "
          f"wall={(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
