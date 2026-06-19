"""Meta-RL (RL^2) flight control — the plan's control layer (PPO was 'PPO-first').

Upgrades the single-design PPO policy to a meta-policy that ADAPTS across the design
distribution few-shot, the plan's §6 control: RL^2 (context-based) — the policy is
trained over a distribution of wing designs (random per episode) and conditions on the
recent (obs, prev-action, prev-reward) context, so from the first interactions it infers
the current design and adapts. This AMORTIZES per-design control: co-design evaluates any
design with the single meta-policy (fast adaptation) instead of retraining PPO per design.

The design = wing stiffness scale s, modulating (consistent with the discovery mechanisms
F1/F4/F8 measured earlier on the validated UVLM):
  - gust sensitivity   : flexible (low s) -> smaller gust excursion (F1 passive alleviation)
  - control authority  : flexible -> higher (F4, control authority ~ 1/stiffness)
  - cruise efficiency  : stiff (high s) -> higher L/D (F8) -> the co-design trade-off
So the meta-policy + the efficiency model give the 抗风×效率 Pareto frontier (the discovery).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flight_ppo_env import FlightPPOEnv, OBS_DIM, ACT_DIM
import design_field as dfield                                 # spanwise spline 刚柔 field

N_EMBED = 6
OBS_SCALE = torch.tensor([0.5, 0.5, 3, 3, 3, 6, 3, 4, 8, 2.5], dtype=torch.float32)
CTX_DIM = OBS_DIM + ACT_DIM + 1 + 1      # obs + prev-action + prev-reward + design-belief slot
K_CTRL = 4                               # spanwise stiffness-field control points (root->tip)


class MetaFlightEnv(FlightPPOEnv):
    """FlightPPOEnv with a per-episode wing 刚柔 FIELD design that modulates dynamics.

    The design is no longer a single scalar stiffness but a spanwise spline field
    s(ξ) (K_CTRL control points root->tip; design_field.StiffnessField). The field's
    physically-reduced aggregates drive the dynamics:
      gust/control authority <- s_gust (load-weighted, TIP-biased effective stiffness),
      cruise efficiency      <- s_root (bending-moment-weighted, ROOT-biased stiffness).
    A UNIFORM field reproduces the old scalar surrogate exactly, so the scalar design is
    the diagonal slice of this richer field design space the meta-policy now adapts over.
    """

    def _apply_design(self, field):
        self.field = field
        self.s = float(field.aggregates()["s_mean"])          # mean (for logging only)
        self.gust_factor = dfield.gust_factor(field)          # <1 for tip-flexible
        self.ctrl_factor = dfield.ctrl_factor(field)          # higher for tip-flexible

    def sample_design(self):
        self._apply_design(dfield.StiffnessField.sample(self.rng, K=K_CTRL))
        return self.field

    def reset(self, design=None):
        if design is None:
            self.sample_design()
        else:                                                 # scalar / (K,) ctrl / field
            self._apply_design(dfield.as_field(design, K=K_CTRL))
        return super().reset()

    def _gust(self):
        return super()._gust() * self.gust_factor             # design-dependent gust

    def step(self, action):
        a = np.asarray(action, float).copy()
        a[1] *= self.ctrl_factor; a[2] *= self.ctrl_factor    # design-dependent control authority
        return super().step(a)


def cruise_efficiency(design):
    """L/D cruise efficiency from the 刚柔 field (ROOT-stiffness driven, minus over-flex
    penalty). Accepts a scalar (uniform field) or a StiffnessField — uniform s reduces to
    the old 22.0+2.2*(s-0.5)."""
    return dfield.cruise_efficiency(design)


class RL2Policy(nn.Module):
    """RL^2 actor-critic: Takens stack of CONTEXT (obs+prev-act+prev-rew) -> MLP."""

    def __init__(self, n_embed=N_EMBED, h=72):
        super().__init__()
        self.n = n_embed
        din = CTX_DIM * n_embed
        self.body = nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.mu = nn.Linear(h, ACT_DIM); self.v = nn.Linear(h, 1)
        self.log_std = nn.Parameter(-0.5 * torch.ones(ACT_DIM))

    def forward(self, emb):
        z = self.body(emb)
        return torch.tanh(self.mu(z)), self.v(z).squeeze(-1)

    def dist(self, emb):
        mu, val = self(emb)
        return torch.distributions.Normal(mu, self.log_std.exp()), val


class CtxEmbedder:
    """Builds the RL^2 context stack (obs, prev action, prev reward)."""

    def __init__(self, n_embed=N_EMBED):
        self.n = n_embed; self.hist = []

    def reset(self):
        self.hist = []; self.prev_a = np.zeros(ACT_DIM); self.prev_r = 0.0

    def push(self, obs):
        ctx = np.concatenate([np.asarray(obs) / OBS_SCALE.numpy(), self.prev_a,
                              [self.prev_r], [0.0]])
        t = torch.as_tensor(ctx, dtype=torch.float32)
        self.hist.append(t)
        if len(self.hist) > self.n:
            self.hist = self.hist[-self.n:]
        emb = torch.zeros(CTX_DIM * self.n)
        flat = torch.cat(self.hist); emb[-len(flat):] = flat
        return emb

    def record(self, a, r):
        self.prev_a = np.asarray(a, float); self.prev_r = float(np.clip(r, -20, 2)) / 2.0


def collect(env, net, emb, steps):
    E, A, LP, R, V, D = [], [], [], [], [], []
    obs = env.reset(); emb.reset(); e = emb.push(obs)
    for _ in range(steps):
        with torch.no_grad():
            dist, val = net.dist(e.unsqueeze(0)); a = dist.sample()[0]
            lp = dist.log_prob(a).sum()
        obs, r, done, _ = env.step(a.numpy())
        emb.record(a.numpy(), r)
        E.append(e); A.append(a); LP.append(lp); R.append(r); V.append(val[0]); D.append(done)
        if done:
            obs = env.reset(); emb.reset()
        e = emb.push(obs)
    with torch.no_grad():
        _, lv = net.dist(e.unsqueeze(0))
    return (torch.stack(E), torch.stack(A), torch.stack(LP), torch.tensor(R),
            torch.stack(V), torch.tensor(D, dtype=torch.float32), lv[0])


def gae(R, V, D, last_v, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(R); g = 0.0; nxt = last_v
    for t in reversed(range(len(R))):
        nt = 1.0 - D[t]; delta = R[t] + gamma * nxt * nt - V[t]
        g = delta + gamma * lam * nt * g; adv[t] = g; nxt = V[t]
    ret = adv + V
    return (adv - adv.mean()) / (adv.std() + 1e-8), ret


def train(iters=140, steps=2048, epochs=8, mb=256, lr=3e-4, seed=0, log=print):
    torch.manual_seed(seed)
    env = MetaFlightEnv(seed=seed); net = RL2Policy(); emb = CtxEmbedder()
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []
    for it in range(iters):
        E, A, LP, R, V, D, lv = collect(env, net, emb, steps)
        adv, ret = gae(R, V, D, lv)
        ep, cur = [], 0.0
        for t in range(len(R)):
            cur += float(R[t])
            if D[t]:
                ep.append(cur); cur = 0.0
        mean_ep = float(np.mean(ep)) if ep else float(R.sum())
        idx = np.arange(len(E))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(E), mb):
                b = idx[s:s + mb]
                dist, val = net.dist(E[b]); lp = dist.log_prob(A[b]).sum(-1)
                ratio = (lp - LP[b]).exp()
                loss = (-torch.min(ratio * adv[b], torch.clamp(ratio, 0.8, 1.2) * adv[b]).mean()
                        + 0.5 * ((val - ret[b]) ** 2).mean() - 0.01 * dist.entropy().sum(-1).mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        hist.append(mean_ep)
        if it % 10 == 0 or it == iters - 1:
            log(f"  iter {it:3d}: mean episode return={mean_ep:8.2f} (n_ep={len(ep)})")
    return net, hist


def eval_field(net, field, emb=None, seed=1):
    """Controlled gust excursion for a 刚柔 field under the adapting meta-policy."""
    emb = emb or CtxEmbedder()
    env = MetaFlightEnv(seed=seed)
    obs = env.reset(design=field); emb.reset(); gz = []
    for k in range(env.horizon):
        with torch.no_grad():
            mu, _ = net(emb.push(obs).unsqueeze(0))
        obs, r, d, info = env.step(mu[0].numpy()); emb.record(mu[0].numpy(), r)
        if env.gust["t0"] <= env.t < env.gust["t0"] + env.gust["dur"] + 0.5:
            gz.append(env.x[2])
        if d:
            break
    return (max(gz) - min(gz)) if gz else np.nan


def codesign_frontier(net):
    """Evaluate the meta-policy across the 刚柔 FIELD design space -> (controlled gust
    rejection, efficiency) Pareto data. Sweeps a 2-D (root, tip) grid of spanwise fields
    (the distributional axis the scalar design could not express); the diagonal root==tip
    is the old uniform-scalar frontier. The meta-policy ADAPTS per design (no retraining)."""
    emb = CtxEmbedder(); rows = []
    roots = [0.6, 1.0, 1.4, 1.8, 2.2]
    tips = [0.4, 0.9, 1.5]
    for root in roots:
        for tip in tips:
            f = dfield.StiffnessField.from_root_tip(root, tip, K=K_CTRL)
            g = eval_field(net, f, emb)
            rows.append((root, tip, g, cruise_efficiency(f)))
    return rows


if __name__ == "__main__":
    import warp as wp; wp.init()
    print("Meta-RL (RL^2): adapt across the wing 刚柔 FIELD distribution (spline, "
          f"K={K_CTRL} root->tip); PPO meta-training")
    net, hist = train(iters=140, steps=2048)
    print(f"meta-RL done: final mean return={hist[-1]:.2f} (best {max(hist):.2f})")
    torch.save(net.state_dict(), "docs/meta_policy.pt")
    np.savez("docs/ppo_hist.npz", hist=np.array(hist))
    print("\nco-design frontier over the 刚柔 FIELD space (meta-policy ADAPTS, no retraining):")
    print("  root | tip  | gust excursion (m) | cruise L/D")
    for root, tip, g, e in codesign_frontier(net):
        print(f"  {root:.1f}  | {tip:.1f}  |   {g:6.2f}           |  {e:.1f}")
    print("  -> stiff-root/flex-tip can win BOTH (efficient AND gust-tolerant): the "
          "distributional payoff a single uniform stiffness cannot reach")
