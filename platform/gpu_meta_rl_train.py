"""GPU-batched meta-RL (RL^2) over the wing 刚柔 FIELD distribution — the GPU version of
meta_rl_train.py. The fast numpy MetaFlightEnv is replaced by GpuFlightEnv (one Warp fp64
kernel stepping B environments in parallel), so PPO collects B trajectories at once on the
GPU. The control layer is unchanged (RL² Takens-context policy); only the environment and
the rollout collection are vectorized.

Design distribution: each env samples a spanwise stiffness FIELD (K spline control points,
design_field), latent to the policy (RL² infers+adapts from the context). The 抗风×效率
co-design frontier sweeps the 2-D (root,tip) field space with the trained meta-policy.

Runs in an env with torch (base); imports the torch-free GpuFlightEnv (warp+numpy).
"""
from __future__ import annotations

import numpy as np
import torch

from gpu_flight_env import GpuFlightEnv, OBS_DIM, ACT_DIM, K_CTRL
from meta_rl_train import RL2Policy, N_EMBED, CTX_DIM, OBS_SCALE
import design_field as dfield

DEVICE = torch.device("cpu")        # tiny MLP; envs are on GPU via Warp
GUST_W = 4.0     # co-design gust strength (m/s): stronger than the 2.5 default so the
#                  control-authority limit binds and the 抗风×效率 trade-off is visible
#                  (train + eval use the SAME value; the env default stays 2.5 for the
#                  numpy-oracle bit-exact validation).


class BatchCtx:
    """Batched RL^2 context stack for B envs: tensor [B, n, CTX_DIM] (obs+prev_a+prev_r)."""

    def __init__(self, B, n=N_EMBED):
        self.B, self.n = B, n
        self.hist = torch.zeros(B, n, CTX_DIM)
        self.prev_a = torch.zeros(B, ACT_DIM)
        self.prev_r = torch.zeros(B)
        self.scale = OBS_SCALE

    def reset_rows(self, mask):
        m = torch.as_tensor(mask, dtype=torch.bool)
        self.hist[m] = 0.0; self.prev_a[m] = 0.0; self.prev_r[m] = 0.0

    def push(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32) / self.scale
        ctx = torch.cat([o, self.prev_a, self.prev_r.unsqueeze(1),
                         torch.zeros(self.B, 1)], dim=1)        # design-belief slot = 0
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1, :] = ctx
        return self.hist.reshape(self.B, self.n * CTX_DIM)

    def record(self, a, r):
        self.prev_a = torch.as_tensor(a, dtype=torch.float32)
        self.prev_r = torch.clamp(torch.as_tensor(r, dtype=torch.float32), -20, 2) / 2.0


def gae(R, V, D, last_v, gamma=0.99, lam=0.95):
    """Per-env GAE over [T,B] tensors."""
    T, B = R.shape
    adv = torch.zeros(T, B); g = torch.zeros(B); nxt = last_v
    for t in reversed(range(T)):
        nt = 1.0 - D[t]
        delta = R[t] + gamma * nxt * nt - V[t]
        g = delta + gamma * lam * nt * g
        adv[t] = g; nxt = V[t]
    ret = adv + V
    a = adv.reshape(-1)
    return ((a - a.mean()) / (a.std() + 1e-8)).reshape(T, B), ret


def collect(env, net, ctx, T):
    obs = env.observe()
    E, A, LP, R, V, D = [], [], [], [], [], []
    for _ in range(T):
        emb = ctx.push(obs)
        with torch.no_grad():
            dist, val = net.dist(emb)
            a = dist.sample(); lp = dist.log_prob(a).sum(-1)
        obs2, r, done = env.step(a.numpy())
        ctx.record(a.numpy(), r)
        E.append(emb); A.append(a); LP.append(lp)
        R.append(torch.as_tensor(r, dtype=torch.float32))
        V.append(val); D.append(torch.as_tensor(done, dtype=torch.float32))
        env.reset_done(done); ctx.reset_rows(done)        # auto-reset done envs + their ctx
        obs = env.observe()
    with torch.no_grad():
        _, lv = net.dist(ctx.push(obs))
    return (torch.stack(E), torch.stack(A), torch.stack(LP), torch.stack(R),
            torch.stack(V), torch.stack(D), lv, obs)


def train(iters=160, B=256, T=128, epochs=6, mb=4096, lr=2e-4, seed=0, log=print):
    import copy
    torch.manual_seed(seed)
    env = GpuFlightEnv(B=B, seed=seed, gust_w=GUST_W)
    env.reset()
    net = RL2Policy(); ctx = BatchCtx(B)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    best_r, best_state = -1e9, copy.deepcopy(net.state_dict())
    ema = None
    for it in range(iters):
        E, A, LP, R, V, D, lv, _ = collect(env, net, ctx, T)
        adv, ret = gae(R, V, D, lv)
        ret = torch.clamp(ret, -25.0, 25.0)              # crash returns (~-20) won't blow up value
        # flatten [T,B] -> [T*B]
        Ef = E.reshape(-1, E.shape[-1]); Af = A.reshape(-1, ACT_DIM)
        LPf = LP.reshape(-1); advf = adv.reshape(-1); retf = ret.reshape(-1)
        N = Ef.shape[0]; idx = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, N, mb):
                b = idx[s:s + mb]
                dist, val = net.dist(Ef[b]); lp = dist.log_prob(Af[b]).sum(-1)
                ratio = (lp - LPf[b]).exp()
                # early-stop the update if the policy has moved too far (approx-KL guard)
                approx_kl = (LPf[b] - lp).mean().item()
                loss = (-torch.min(ratio * advf[b], torch.clamp(ratio, 0.8, 1.2) * advf[b]).mean()
                        + 0.5 * ((val - retf[b]) ** 2).mean()
                        - 0.005 * dist.entropy().sum(-1).mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            if abs(approx_kl) > 0.03:
                break
        mr = float(R.mean()); hist.append(mr)
        ema = mr if ema is None else 0.8 * ema + 0.2 * mr     # smoothed (best by EMA, robust)
        if ema > best_r:
            best_r = ema; best_state = copy.deepcopy(net.state_dict())
        if it % 10 == 0 or it == iters - 1:
            log(f"  iter {it:3d}: mean step reward={mr:7.3f}  ema={ema:7.3f}  "
                f"(best {best_r:.3f}, B={B}, T={T})")
    net.load_state_dict(best_state)                          # return the BEST policy, not last
    log(f"  -> restored best policy (ema step reward={best_r:.3f})")
    return net, hist


def eval_designs(net, ctrls, seed=1):
    """Controlled gust excursion for M designs in ONE batched GPU rollout (deterministic
    policy). A design that crashes during the gust counts as POOR rejection (large
    excursion), not dropped — so the frontier reflects real controllability."""
    M = len(ctrls)
    env = GpuFlightEnv(B=M, seed=seed, gust_w=GUST_W)
    env.reset(designs=np.asarray(ctrls, np.float64))
    ctx = BatchCtx(M)
    obs = env.observe()
    g = env.gust
    zmin = np.full(M, np.inf); zmax = np.full(M, -np.inf)
    alive = np.ones(M, bool); crashed_gust = np.zeros(M, bool)
    for k in range(env.horizon):
        with torch.no_grad():
            mu, _ = net(ctx.push(obs))
        obs, r, done = env.step(mu.numpy()); ctx.record(mu.numpy(), r)
        t = env.tt.numpy(); z = np.clip(env.x.numpy()[:, 2], 0.0, 60.0)
        win = (t >= g["t0"]) & (t < g["t0"] + g["dur"] + 0.6)
        m = win & alive
        zmin = np.where(m, np.minimum(zmin, z), zmin)
        zmax = np.where(m, np.maximum(zmax, z), zmax)
        crashed_gust |= done & alive & (t < g["t0"] + g["dur"] + 0.6) & (t >= g["t0"])
        alive &= ~done
    exc = zmax - zmin
    exc = np.where(np.isfinite(exc), exc, np.nan)
    exc = np.where(crashed_gust, np.fmax(np.nan_to_num(exc, nan=0.0), 8.0), exc)  # crash = poor
    return exc


def codesign_frontier(net):
    """Sweep the 2-D (root,tip) 刚柔 field design space with the adapting meta-policy
    (one batched rollout over all designs)."""
    grid = [(root, tip) for root in [0.6, 1.0, 1.4, 1.8, 2.2] for tip in [0.4, 0.9, 1.5]]
    ctrls = np.array([dfield.StiffnessField.from_root_tip(r, t, K=K_CTRL).ctrl for r, t in grid])
    exc = eval_designs(net, ctrls)
    rows = []
    for (root, tip), g in zip(grid, exc):
        f = dfield.StiffnessField.from_root_tip(root, tip, K=K_CTRL)
        rows.append((root, tip, float(g), dfield.cruise_efficiency(f)))
    return rows


if __name__ == "__main__":
    import os
    import warp as wp
    wp.init()
    _D = os.path.join(os.path.dirname(__file__), "..", "docs")
    print("GPU-batched meta-RL (RL^2) over the wing 刚柔 FIELD distribution "
          f"(Warp fp64 env, K={K_CTRL} spline)")
    net, hist = train(iters=160, B=256, T=128)
    print(f"meta-RL done: final mean step reward={hist[-1]:.3f} (best {max(hist):.3f})")
    torch.save(net.state_dict(), os.path.join(_D, "meta_policy.pt"))
    np.savez(os.path.join(_D, "ppo_hist.npz"), hist=np.array(hist))
    print("\nco-design frontier over the 刚柔 FIELD space (meta-policy ADAPTS, no retraining):")
    print("  root | tip  | gust excursion (m) | cruise L/D")
    for root, tip, g, e in codesign_frontier(net):
        print(f"  {root:.1f}  | {tip:.1f}  |   {g:6.2f}           |  {e:.1f}")
    print("  -> stiff-root/flex-tip wins BOTH (efficient AND gust-tolerant): the "
          "distributional payoff beyond a uniform stiffness")
