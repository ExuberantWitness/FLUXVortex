"""Gradient-driven co-design: ascend the spanwise 刚柔 FIELD with the validated analytic
gradients — closing the differentiable co-design loop on GPU.

Objective (maximize):   J(ctrl) = -α · gust_factor(ctrl)  +  λ · efficiency(ctrl)
on the two POLICY-INDEPENDENT design aggregates:
  · gust_factor : passive gust transmissibility (plan F1) — the gust input the controller
    must reject. Lower = the design itself sheds more gust (flexible-tip washout). Its
    gradient is the validated SHAC gust design gradient (validate_gpu_env test 5).
  · efficiency  : cruise L/D from the 刚柔 aggregates — the validated DQD design gradient.

Why the design AGGREGATES, not the closed-loop rollout reward: the RL² meta-policy ADAPTS
to each design, so closed-loop reward is largely design-invariant (it masks the design — the
flat-frontier finding) and its frozen-policy design gradient misleads. The passive
transmissibility + efficiency are the clean design-layer objectives co-design optimizes;
both gradients are the Warp-tape ones validated bit-exact vs FD.

Minimizing gust_factor pushes the TIP flexible (tip-biased s_gust); maximizing efficiency
pushes the ROOT stiff (root-biased s_root) -> the field converges to stiff-root/flex-tip.
Sweeping λ/α traces the 抗风×效率 Pareto front directly from gradients (no grid search).

Pure warp+numpy (no torch) — runs in the fluxvortex env.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from gpu_flight_env import (design_agg_sums, design_agg_eff_final, design_agg_gust_final,
                            K_CTRL, _NG, _TAPER, _NORM_W, _NORM_MG)

LO, HI = 0.3, 2.5


def _agg_and_grad(ctrl, which, dev="cuda"):
    """Return (value[B], ∂(Σvalue)/∂ctrl) for which in {'eff','gust'} via the tape over
    [design_agg_sums -> finalizer]. value = efficiency (DQD) or gust_factor (SHAC)."""
    B = ctrl.shape[0]
    ctrl_wp = wp.array(ctrl, dtype=wp.float64, device=dev, requires_grad=True)
    sum_wc = wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True)
    sum_mgs = wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True)
    wp.launch(design_agg_sums, dim=B, inputs=[ctrl_wp, K_CTRL, _NG, np.float64(_TAPER)],
              outputs=[sum_wc, sum_mgs], device=dev)
    pen_mask = wp.array((sum_wc.numpy() / _NORM_W > 2.0).astype(np.float64),
                        dtype=wp.float64, device=dev)
    out = [wp.zeros(B, dtype=wp.float64, device=dev, requires_grad=True) for _ in range(3)]
    tape = wp.Tape()
    with tape:
        wp.launch(design_agg_sums, dim=B, inputs=[ctrl_wp, K_CTRL, _NG, np.float64(_TAPER)],
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


class Adam:
    def __init__(self, shape, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t); vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def optimize(ctrl0, alpha=20.0, lam=1.0, iters=200, lr=0.02, log=None):
    """Ascend J = -α·gust_factor + λ·efficiency with Adam (gradients = validated tapes)."""
    ctrl = ctrl0.copy()
    opt = Adam(ctrl.shape, lr=lr)
    hist = []
    for it in range(iters):
        gf, g_gf = _agg_and_grad(ctrl, "gust")
        eff, g_e = _agg_and_grad(ctrl, "eff")
        J = -alpha * gf + lam * eff
        g = -alpha * g_gf + lam * g_e
        ctrl = np.clip(ctrl + opt.step(g), LO, HI)
        hist.append((float(J.mean()), float(gf.mean()), float(eff.mean())))
        if log and (it % 40 == 0 or it == iters - 1):
            log(f"  it {it:3d}: J={J.mean():7.2f}  gust_factor={gf.mean():.3f}  "
                f"L/D={eff.mean():5.2f}  root={ctrl[:,0].mean():.2f} tip={ctrl[:,-1].mean():.2f}")
    return ctrl, np.array(hist)


def main():
    wp.init()
    print("Gradient-driven co-design: Adam ascent on the 刚柔 field\n"
          "  J = -α·gust_factor (抗风, SHAC grad) + λ·L/D (效率, DQD grad)\n")
    B = 4
    ctrl0 = np.full((B, K_CTRL), 1.2)
    print(f"start: uniform field s=1.2 (B={B} parallel inits)")
    ctrl, hist = optimize(ctrl0, alpha=20.0, lam=1.0, iters=200, log=print)
    print(f"\noptimized field (mean): {np.array2string(ctrl.mean(0), precision=2)}")
    split = ctrl[:, 0].mean() - ctrl[:, -1].mean()
    print(f"  root={ctrl[:,0].mean():.2f}  tip={ctrl[:,-1].mean():.2f}  (root-tip={split:+.2f})  "
          f"-> {'stiff-root / flex-tip ✓' if split > 0.2 else 'no clear split'}")
    print(f"  gust_factor (↓ better): {hist[0,1]:.3f} -> {hist[-1,1]:.3f}   "
          f"L/D (↑ better): {hist[0,2]:.2f} -> {hist[-1,2]:.2f}")

    print("\nλ-sweep — gradient-traced 抗风×效率 Pareto (each weight -> one optimized design):")
    print("  λ/α  | root | tip  | gust_factor↓ | L/D↑")
    for lam in [0.2, 0.6, 1.0, 2.0, 4.0]:
        c, h = optimize(np.full((4, K_CTRL), 1.2), alpha=20.0, lam=lam, iters=160)
        print(f"  {lam/20.0:.3f}| {c[:,0].mean():.2f} | {c[:,-1].mean():.2f} | "
              f"{h[-1,1]:9.3f}    | {h[-1,2]:.2f}")
    print("  -> larger λ weights efficiency (stiffer root); smaller weights 抗风 (more "
          "flexible tip). The analytic gradient walks the 抗风×效率 trade-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
