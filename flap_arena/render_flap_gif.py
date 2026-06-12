"""Stress test + visualization: PteraSoftware flapping case on the platform.

Runs the canonical-congruent rectangular flapping case (15 deg @ 1 Hz, rigid
kinematic mode) with the particle wake (merge population control), rendering
each Nth window: wing surface, near-field VLM ring lattice (wing + K wake
rows), and the particle cloud. Frames -> animated GIF.

Usage: python flap_arena/render_flap_gif.py [--cycles 15] [--every 3]
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


def render_frame(ax, verts, wake_pts, p_pos, p_alpha, t, n_p, ms, lift):
    ax.cla()
    # wing surface
    X, Y, Z = verts[..., 0], verts[..., 1], verts[..., 2]
    ax.plot_surface(X, Y, Z, color="#4a86c8", alpha=0.85, edgecolor="k",
                    linewidth=0.4, shade=True)
    # near-field wake ring lattice
    if wake_pts is not None:
        W = wake_pts
        for r in range(W.shape[0]):
            ax.plot(W[r, :, 0], W[r, :, 1], W[r, :, 2], color="#777777",
                    lw=0.6, alpha=0.7)
        for j in range(W.shape[1]):
            ax.plot(W[:, j, 0], W[:, j, 1], W[:, j, 2], color="#777777",
                    lw=0.4, alpha=0.5)
    # particles colored by strength
    if len(p_pos):
        s = np.linalg.norm(p_alpha, axis=1)
        order = np.argsort(p_pos[:, 0])          # draw far ones first
        ax.scatter(p_pos[order, 0], p_pos[order, 1], p_pos[order, 2],
                   c=s[order], cmap="plasma", s=4.0, alpha=0.8,
                   vmin=0.0, vmax=np.percentile(s, 95), linewidths=0)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1.0, 7.0)
    ax.set_zlim(-2.8, 2.8)
    ax.set_box_aspect((12.5, 8, 5.6), zoom=1.35)
    ax.view_init(elev=16, azim=-128)
    ax.set_axis_off()
    ax.set_title(f"t={t:6.2f}s   particles={n_p:5d}   {ms:4.0f} ms/window"
                 f"   L={lift:+7.0f} N", fontsize=10, family="monospace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--amp", type=float, default=15.0)
    ap.add_argument("--out", default="flap_arena/out/flap_demo.gif")
    args = ap.parse_args()

    base = np.load("flap_arena/out/ptera_baseline.npz")
    chord, span = float(base["chord"]), float(base["span"])
    nc, ns = int(base["nc"]), int(base["ns"])
    period = float(base["period"])
    V, alpha = float(base["v_inf"]), np.deg2rad(float(base["alpha"]))
    rho = float(base["rho"])
    dtw = float(base["dt_free"])
    n_windows = int(round(args.cycles * period / dtw))

    kin = FlapKinematics(np.deg2rad(args.amp), period)
    entry = FlapEntry(chord, span, nc, ns, kin, mode="kinematic")
    V_vec = V * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    provider = FlapUVLMProvider(V_vec, rho, dtw, K=8, chord=chord,
                                particles=True, max_particles=10**6,
                                pop_scheme="merge", merge_eps=1e-4)
    provider.merge_protect_dist = 7.5
    pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                  substeps=8, dt=dtw / 8, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))

    fig = plt.figure(figsize=(9.6, 5.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    frames = []
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        tw = time.time()
        pc.advance()
        ms = (time.time() - tw) * 1000
        if w % args.every:
            continue
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
        L = -F[0] * np.sin(alpha) + F[2] * np.cos(alpha)
        st = entry.state()
        render_frame(ax, st["verts"], provider.pts, provider.p_pos,
                     provider.p_alpha, pc._t, len(provider.p_pos), ms, L)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
        if w % 60 == 0:
            print(f"w={w:4d} t={pc._t:6.2f}s n_p={len(provider.p_pos):5d} "
                  f"{ms:5.0f}ms frames={len(frames)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    plt.close(fig)

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=70, loop=0, optimize=True)
    sz = os.path.getsize(args.out) / 1e6
    print(f"GIF: {args.out}  frames={len(imgs)}  {sz:.1f} MB  "
          f"total wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
