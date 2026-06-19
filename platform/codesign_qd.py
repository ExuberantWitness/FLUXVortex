"""MAP-Elites + DQD (OMG-MEGA) over the spanwise 刚柔 FIELD — the plan's §6 differentiable
quality-diversity core, driven by the VALIDATED design gradients.

Illuminates the 抗风×效率 behavior space with diverse manufacturable designs:
  · behavior descriptors (BD, 2-D) = (cruise efficiency L/D, gust transmissibility) — the
    performance plane; both are differentiable design aggregates (validated ∂/∂ctrl).
  · objective (quality, maximize) = field smoothness q = -Σ(Δs)² — prefer smooth /
    manufacturable spanwise stiffness profiles. So each cell keeps the SMOOTHEST design that
    achieves that (L/D, gust) performance.

DQD emitter = OMG-MEGA gradient arborescence: at each elite compute ∇q, ∇L/D, ∇gust and
branch along random linear combinations σ·(c0·∇̂q + c1·∇̂(L/D) + c2·∇̂gust). The ∇(L/D) and
∇gust are the Warp-tape SHAC/DQD gradients (validated bit-exact vs FD).

The DQD advantage over blind mutation appears as the FIELD becomes higher-dimensional: a
random K-dim step lands mostly orthogonal to the 2-D behavior manifold (expected BD motion
~σ√(2/K)), while the gradient points straight along it. So a high-resolution spline 刚柔 field
(K≈12) is where differentiable QD pays off — the regime that matters for expensive FSI.

Pure warp+numpy — runs in the fluxvortex env.
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

from gpu_flight_env import (design_agg_sums, design_agg_eff_final, design_agg_gust_final,
                            _NG, _TAPER, _NORM_W, _NORM_MG)

LO, HI = 0.3, 2.5
K_FIELD = 12                    # spline 刚柔 field resolution (high-dim -> DQD pays off)
EFF_RANGE = (21.0, 27.0)
GUST_RANGE = (0.55, 0.92)
GRID = (30, 24)


def agg_and_grad(ctrl, which, K=K_FIELD, dev="cuda"):
    """value[B] and ∂(Σvalue)/∂ctrl via [design_agg_sums(K) -> finalizer]. which: 'eff'|'gust'."""
    B = ctrl.shape[0]
    ctrl_wp = wp.array(ctrl, dtype=wp.float64, device=dev, requires_grad=True)
    sum_wc = wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True)
    sum_mgs = wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True)
    wp.launch(design_agg_sums, dim=B, inputs=[ctrl_wp, K, _NG, np.float64(_TAPER)],
              outputs=[sum_wc, sum_mgs], device=dev)
    pen_mask = wp.array((sum_wc.numpy() / _NORM_W > 2.0).astype(np.float64),
                        dtype=wp.float64, device=dev)
    out = [wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True) for _ in range(3)]
    tape = wp.Tape()
    with tape:
        wp.launch(design_agg_sums, dim=B, inputs=[ctrl_wp, K, _NG, np.float64(_TAPER)],
                  outputs=[sum_wc, sum_mgs], device=dev)
        if which == "eff":
            wp.launch(design_agg_eff_final, dim=B,
                      inputs=[sum_wc, sum_mgs, np.float64(_NORM_W), np.float64(_NORM_MG), pen_mask],
                      outputs=out, device=dev)
        else:
            wp.launch(design_agg_gust_final, dim=B, inputs=[sum_wc, np.float64(_NORM_W)],
                      outputs=out, device=dev)
    out[0].grad = wp.array(np.ones(B, np.float64), dtype=wp.float64, device=dev)
    tape.backward()
    return out[0].numpy(), ctrl_wp.grad.numpy()


def smoothness(ctrl):
    d = np.diff(ctrl, axis=1)
    q = -np.sum(d * d, axis=1)
    g = np.zeros_like(ctrl)
    g[:, :-1] += 2.0 * d
    g[:, 1:] -= 2.0 * d
    return q, g


def evaluate(ctrl):
    eff, _ = agg_and_grad(ctrl, "eff")
    gust, _ = agg_and_grad(ctrl, "gust")
    return eff, gust


class Archive:
    def __init__(self, K=K_FIELD, grid=GRID, eff_range=EFF_RANGE, gust_range=GUST_RANGE):
        self.ni, self.nj = grid
        self.er, self.gr = eff_range, gust_range
        self.q = np.full(grid, -np.inf)
        self.ctrl = np.zeros((self.ni, self.nj, K))
        self.eff = np.full(grid, np.nan); self.gust = np.full(grid, np.nan)

    def _cell(self, eff, gust):
        i = np.clip(((eff - self.er[0]) / (self.er[1] - self.er[0]) * self.ni).astype(int),
                    0, self.ni - 1)
        j = np.clip(((gust - self.gr[0]) / (self.gr[1] - self.gr[0]) * self.nj).astype(int),
                    0, self.nj - 1)
        return i, j

    def add(self, ctrl, q, eff, gust):
        i, j = self._cell(eff, gust)
        for b in range(len(ctrl)):
            if q[b] > self.q[i[b], j[b]]:
                self.q[i[b], j[b]] = q[b]; self.ctrl[i[b], j[b]] = ctrl[b]
                self.eff[i[b], j[b]] = eff[b]; self.gust[i[b], j[b]] = gust[b]

    def coverage(self):
        return np.isfinite(self.q).mean()

    def qd_score(self):
        f = self.q[np.isfinite(self.q)]
        return float(np.sum(f + 10.0)) if len(f) else 0.0

    def elites(self):
        m = np.isfinite(self.q)
        return self.ctrl[m], self.eff[m], self.gust[m], self.q[m]

    def sample(self, n, rng):
        c, _, _, _ = self.elites()
        return c[rng.integers(0, len(c), size=n)] if len(c) else None


def dqd_emit(parents, sigma, n_branch, rng):
    """OMG-MEGA: branch each parent along random combos of ∇q, ∇(L/D), ∇gust (normalized)."""
    _, g_q = smoothness(parents)
    _, g_e = agg_and_grad(parents, "eff")
    _, g_g = agg_and_grad(parents, "gust")
    nrm = lambda g: g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    gq, ge, gg = nrm(g_q), nrm(g_e), nrm(g_g)
    out = []
    for b in range(len(parents)):
        for _ in range(n_branch):
            c = rng.normal(size=3)
            out.append(np.clip(parents[b] + sigma * (c[0]*gq[b] + c[1]*ge[b] + c[2]*gg[b]), LO, HI))
    return np.array(out)


def random_emit(parents, sigma, n_branch, rng):
    """Iso Gaussian mutation (the non-gradient baseline emitter)."""
    K = parents.shape[1]
    out = []
    for b in range(len(parents)):
        for _ in range(n_branch):
            out.append(np.clip(parents[b] + sigma * rng.normal(size=K), LO, HI))
    return np.array(out)


def run(emit, K=K_FIELD, iters=120, batch=24, n_branch=4, sigma=0.15, n_init=24, seed=0, log=None):
    rng = np.random.default_rng(seed)
    arch = Archive(K=K)
    init = rng.uniform(LO, HI, size=(n_init, K))
    q, _ = smoothness(init); eff, gust = evaluate(init)
    arch.add(init, q, eff, gust)
    hist = []
    for it in range(iters):
        parents = arch.sample(batch, rng)
        cand = emit(parents, sigma, n_branch, rng)
        q, _ = smoothness(cand); eff, gust = evaluate(cand)
        arch.add(cand, q, eff, gust)
        hist.append((arch.coverage(), arch.qd_score()))
        if log and (it % 20 == 0 or it == iters - 1):
            log(f"  it {it:3d}: coverage={arch.coverage()*100:5.1f}%  QD-score={arch.qd_score():8.1f}")
    return arch, np.array(hist)


def main():
    wp.init()
    print(f"MAP-Elites + DQD (OMG-MEGA) over the 刚柔 FIELD (K={K_FIELD} spline) — "
          "illuminating 抗风×效率\n")
    print("DQD emitter (gradient arborescence: ∇q, ∇L/D, ∇gust):")
    arch_dqd, h_dqd = run(dqd_emit, iters=120, sigma=0.12, log=print)
    print("\nrandom-mutation emitter (no gradients, same budget):")
    arch_rnd, h_rnd = run(random_emit, iters=120, sigma=0.12, log=print)

    print(f"\nfinal — DQD:    coverage={arch_dqd.coverage()*100:.1f}%  QD-score={arch_dqd.qd_score():.0f}")
    print(f"        random: coverage={arch_rnd.coverage()*100:.1f}%  QD-score={arch_rnd.qd_score():.0f}")
    gain = arch_dqd.qd_score() / max(arch_rnd.qd_score(), 1e-9)
    print(f"  -> DQD fills the 抗风×效率 archive {gain:.2f}× the QD-score of blind mutation "
          f"(K={K_FIELD}: gradients steer along the 2-D behavior manifold; random mostly misses it)")

    np.savez(os.path.join(os.path.dirname(__file__), "..", "docs", "codesign_qd.npz"),
             dqd_q=arch_dqd.q, dqd_eff=arch_dqd.eff, dqd_gust=arch_dqd.gust,
             dqd_ctrl=arch_dqd.ctrl, dqd_hist=h_dqd, rnd_hist=h_rnd, rnd_q=arch_rnd.q,
             eff_range=np.array(EFF_RANGE), gust_range=np.array(GUST_RANGE),
             grid=np.array(GRID), K=K_FIELD)
    print("  saved archive -> docs/codesign_qd.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
