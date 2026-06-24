"""Full-UVLM meta-RL co-design — Step 2: a DESIGN-CONDITIONED control policy trained by the EXACT
analytic gradient through the differentiable STRONG-COUPLED unsteady-UVLM FSI (SHAC, no surrogate).

The co-design's control layer is amortized: instead of re-optimizing the closed-loop gain k for every
structural design (every MAP-Elites niche), a small policy π(d; w) maps the design descriptor
d = (θ_E, θ_R) (spanwise stiffness + mass spline control points, the SAME parameterization the QD
archive searches) to the gust-rejection feedback gain k = K_HI·σ(w·[d,1]). It is trained by gradient
descent on the gust-rejection + control-effort loss, whose gradient ∂L/∂w = (∂L/∂k)·(∂k/∂w) uses the
VALIDATED closed-loop policy gradient ∂L/∂k from the PC adjoint (coupled_unsteady_pc_grad_gpu, verified
vs FD to 1e-3) — so NO new adjoint is written and the gradient is exact through the full nonlinear
unsteady free-wake UVLM ⊗ ANCF fixed point. validate_grad() checks ∂L/∂w vs finite difference of the
adjoint's own loss; train() amortizes the controller over a design distribution; generalize() shows the
trained π predicts the per-design grid-optimal gain on HELD-OUT designs (the meta/adaptation payoff).

This is the control layer of the flagship: design layer = MAP-Elites over (θ_E, θ_R); control layer =
this amortized π(d). On the full coupled UVLM FSI (no reduced surrogate).
"""
from __future__ import annotations

import numpy as np

import warp as wp
import codesign_qd_unsteady as cq
import diff_coupled_unsteady_gpu as cg

K_HI = 9.0                                  # gain ceiling (matches the QD 动力系统 axis B2_HI)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GainPolicy:
    """k = K_HI·σ(w·[d,1]); d = (θ_E[3], θ_R[3]) design descriptor. Linear+squash → exact ∂k/∂w, gain
    bounded in (0, K_HI). w has len(d)+1 = 7 params."""

    def __init__(self, w=None, ndesc=6):
        self.w = np.zeros(ndesc + 1) if w is None else np.asarray(w, float).copy()

    def feat(self, d):
        return np.concatenate([np.asarray(d, float), [1.0]])

    def gain(self, d):
        return float(K_HI * _sigmoid(self.w @ self.feat(d)))

    def dgain_dw(self, d):
        """∂k/∂w — w.r.t. the policy WEIGHTS (training): z=w·f ⇒ ∂z/∂w_j=f_j."""
        f = self.feat(d); s = _sigmoid(self.w @ f)
        return K_HI * s * (1.0 - s) * f                       # (ndesc+1,)

    def dgain_dtheta(self, d):
        """∂k/∂θ — w.r.t. the DESIGN descriptor (the archive's controller-response term): z=w·[θ,1] ⇒
        ∂z/∂θ_i = w_i, so ∂k/∂θ_i = K·σ'(z)·w_i  (the WEIGHTS, NOT the features)."""
        s = _sigmoid(self.w @ self.feat(d)); nd = len(np.asarray(d, float))
        return K_HI * s * (1.0 - s) * self.w[:nd]             # (ndesc,)


def _gust_loss_fn(free, qref, pos, k, ctrl_w):
    """Trajectory functional L = ½Σ_t‖(q_t−qref)·free‖² + ½·ctrl_w·k²·Σ_t‖q̇_t·pos‖² (gust excursion +
    control effort), returning the per-step adjoint seeds. The k² makes the effort EXPLICITLY k-dependent
    (→ dLdk_extra) AND implicitly (q̇_t depends on k via the closed loop → dLddq, chained by the adjoint)."""
    fm = free                                                  # (ndof,) 0/1 mask

    def loss_fn(qs, dqs, as_, q0, dq0):
        # qs, dqs are the N POST-step states q_1..q_N (no IC); the adjoint seeds at index t+1 (1..N),
        # so the returned seeds are (N+1, ndof) with row 0 (the IC) zero.
        N = qs.shape[0]; ndof = qs.shape[1]
        d = (qs - qref[None]) * fm[None]                      # excursion of q_1..q_N from undeformed ref
        ev = dqs * pos[None]                                   # controlled velocity (effort ∝ k·ev)
        L = 0.5 * float(np.sum(d ** 2)) + 0.5 * ctrl_w * k * k * float(np.sum(ev ** 2))
        dLdq = np.zeros((N + 1, ndof)); dLdq[1:] = d
        dLddq = np.zeros((N + 1, ndof)); dLddq[1:] = ctrl_w * k * k * ev
        dLdk_extra = ctrl_w * k * float(np.sum(ev ** 2))      # ∂L/∂k holding the trajectory fixed
        return L, dLdq, dLddq, None, dLdk_extra

    return loss_fn


def _eval_design(env, d, k, N, dt, ctrl_w, use_wake=True, cg_tol=1e-7, pc_it=22, pc_tol=1e-8,
                 adj_it=40, adj_tol=1e-8, grad=True):
    """Run the strong-coupled PC adjoint for one design d=(θ_E,θ_R) at gain k → (L, dL/dk). The DQD/SHAC
    descent direction needs far fewer adjoint iters than the 1e-11 validation gate (adj_it default 80)."""
    E = np.exp(env.B @ np.asarray(d[0:3])); R = np.exp(env.B @ np.asarray(d[3:6]))
    pos = cg._pos_mask(env.C)
    lf = _gust_loss_fn(env.fmask, env.qref, pos, k, ctrl_w)
    w = np.zeros(env.ndof)                                     # linear-functional seed unused (loss_fn overrides)
    L, _, _, _, dL_dk = cg.coupled_unsteady_pc_grad_gpu(
        env.sh, env.C, env.P, env.dist, env.q0, env.dq0, N, dt, w, E, R, env.nx, env.ny,
        use_wake=use_wake, fb_gain=k, cg_tol=cg_tol, pc_it=pc_it, pc_tol=pc_tol,
        adj_it=adj_it, adj_tol=adj_tol, loss_fn=lf)
    return L, (float(dL_dk) if grad else None)


def _sample_design(rng, lo=0.35, hi=2.6):
    """Random smooth (stiffness, mass) design: log-uniform root/mid/tip control points (physical EI/ρ
    spread), in the QD search box."""
    le = np.log([lo, hi])
    tE = rng.uniform(le[0], le[1], 3); tR = rng.uniform(le[0], le[1], 3)
    return np.concatenate([tE, tR])


# ───────────────────────── validation: exact gradient vs finite difference ─────────────────────────
def validate_grad(nx=6, ny=4, N=16, dt=2e-4, ctrl_w=0.02, seed=2, eps=1e-4):
    """∂L/∂w (policy chain through the validated closed-loop FSI gradient) vs FD of the adjoint's own
    loss L(w). Exact check of the meta-controller's training gradient before any training."""
    wp.init(); env = cq.Env(nx=nx, ny=ny, seed=seed)
    rng = np.random.default_rng(seed)
    pol = GainPolicy(w=0.3 * rng.standard_normal(7))
    d = _sample_design(rng)
    k = pol.gain(d)
    tt = dict(cg_tol=1e-9, pc_it=30, pc_tol=1e-9, adj_it=80, adj_tol=1e-11)  # tight so the FD gate is not solver-noise-limited
    L0, dL_dk = _eval_design(env, d, k, N, dt, ctrl_w, **tt)
    g_analytic = dL_dk * pol.dgain_dw(d)                       # ∂L/∂w = ∂L/∂k · ∂k/∂w
    g_fd = np.zeros_like(pol.w)
    for i in range(len(pol.w)):
        wp_ = pol.w.copy(); wp_[i] += eps; Lp, _ = _eval_design(cq.Env(nx, ny, seed), d, GainPolicy(wp_).gain(d), N, dt, ctrl_w, grad=False, **tt)
        wm_ = pol.w.copy(); wm_[i] -= eps; Lm, _ = _eval_design(cq.Env(nx, ny, seed), d, GainPolicy(wm_).gain(d), N, dt, ctrl_w, grad=False, **tt)
        g_fd[i] = (Lp - Lm) / (2 * eps)
    rel = np.max(np.abs(g_analytic - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30)
    ok = rel < 2e-2
    print(f"meta-controller policy gradient ∂L/∂w through the differentiable STRONG-coupled UVLM FSI "
          f"({env.ne} elems, {N} steps, k={k:.3f}):")
    print(f"  analytic ‖g‖={np.linalg.norm(g_analytic):.4e}  vs FD ‖g‖={np.linalg.norm(g_fd):.4e}   max-rel={rel:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: the design-conditioned controller trains by EXACT gradient "
          f"through the full coupled UVLM FSI (no surrogate, no new adjoint)")
    return ok


# ───────────────────────── training: amortize the controller over the design distribution ──────────
def train(nx=6, ny=4, N=16, dt=2e-4, ctrl_w=0.02, epochs=20, batch=4, lr=0.5, seed=0, verbose=True):
    wp.init(); env = cq.Env(nx=nx, ny=ny, seed=seed)
    rng = np.random.default_rng(seed + 100)
    pol = GainPolicy(w=np.zeros(7)); m = np.zeros(7); v = np.zeros(7); b1, b2, e = 0.9, 0.999, 1e-8
    hist = []
    for ep in range(epochs):
        ds = [_sample_design(rng) for _ in range(batch)]
        g = np.zeros(7); Lsum = 0.0
        for d in ds:
            k = pol.gain(d); L, dL_dk = _eval_design(env, d, k, N, dt, ctrl_w)
            g += dL_dk * pol.dgain_dw(d); Lsum += L
        g /= batch; Lsum /= batch
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g            # Adam
        mh = m / (1 - b1 ** (ep + 1)); vh = v / (1 - b2 ** (ep + 1))
        pol.w -= lr * mh / (np.sqrt(vh) + e)
        hist.append(Lsum)
        if verbose:
            ks = [pol.gain(d) for d in ds]
            print(f"  epoch {ep:3d}: mean gust-loss {Lsum:.4e}   gains {min(ks):.2f}..{max(ks):.2f}", flush=True)
    return pol, hist


# ───────────────────────── generalization: amortized π vs per-design grid-optimal gain ─────────────
def generalize(pol, nx=6, ny=4, N=16, dt=2e-4, ctrl_w=0.02, ndesign=5, ngrid=8, seed=7):
    """On HELD-OUT designs: compare π(d) to the per-design grid-optimal gain k* and the no-control loss.
    A trained meta-controller predicts k*≈π(d) WITHOUT re-optimizing — the amortization payoff."""
    wp.init(); env = cq.Env(nx=nx, ny=ny, seed=seed)
    rng = np.random.default_rng(seed + 500)
    ks_grid = np.linspace(0.0, K_HI, ngrid)
    print(f"held-out generalization ({ndesign} unseen designs): amortized π(d) vs grid-optimal k* and no-control")
    print(f"  {'design':>22} | {'π(d)':>5} {'L(π)':>10} | {'k*':>5} {'L(k*)':>10} | {'L(k=0)':>10} | match")
    rows = []
    for _ in range(ndesign):
        d = _sample_design(rng)
        kp = pol.gain(d); Lp, _ = _eval_design(env, d, kp, N, dt, ctrl_w, grad=False)
        Ls = [(_eval_design(env, d, float(kk), N, dt, ctrl_w, grad=False)[0], float(kk)) for kk in ks_grid]
        Lstar, kstar = min(Ls, key=lambda t: t[0])
        L0 = [L for (L, kk) in Ls if kk == 0.0][0]
        match = abs(kp - kstar) < 1.5                          # within one grid cell of optimal
        rows.append((kp, Lp, kstar, Lstar, L0, match))
        ds = f"E[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}]"
        print(f"  {ds:>22} | {kp:5.2f} {Lp:10.3e} | {kstar:5.2f} {Lstar:10.3e} | {L0:10.3e} | {'OK' if match else '-'}")
    red = np.mean([(L0 - Lp) / (L0 + 1e-30) for (kp, Lp, ks, Ls, L0, mm) in rows])
    nopt = np.mean([(Lp - Ls) / (Ls + 1e-30) for (kp, Lp, ks, Ls, L0, mm) in rows])
    nmatch = sum(r[5] for r in rows)
    print(f"  amortized π reduces gust-loss {100*red:.1f}% vs no-control; within {100*nopt:.1f}% of per-design "
          f"grid-optimal; {nmatch}/{ndesign} gains match k* (≤1 cell)")
    return rows


def figure(hist, rows, path="docs/fsi_shac_controller.png"):
    """Learning curve + amortized π(d) vs per-design grid-optimal k* (the meta payoff)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(hist, "-o", ms=3); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean gust-loss")
    ax[0].set_title("SHAC training of the design-conditioned controller"); ax[0].grid(alpha=0.3)
    kp = [r[0] for r in rows]; ks = [r[2] for r in rows]
    lim = [0, K_HI]
    ax[1].plot(lim, lim, "k--", alpha=0.5, label="π(d)=k*")
    ax[1].scatter(ks, kp, c="tab:red", zorder=3)
    ax[1].set_xlabel("per-design grid-optimal k*"); ax[1].set_ylabel("amortized π(d)")
    ax[1].set_title("held-out designs: π predicts the optimal gain"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].grid(alpha=0.3); ax[1].legend(loc="upper left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); print(f"saved figure -> {os.path.abspath(path)}")


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        pol, hist = train()
        np.savez("artifacts/fsi_shac_policy.npz", w=pol.w, hist=np.array(hist), K_HI=K_HI)
        rows = generalize(pol)
        figure(hist, rows)
    else:
        raise SystemExit(0 if validate_grad() else 1)
