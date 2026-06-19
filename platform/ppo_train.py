"""PPO training of a Takens-embedding NN policy on the fast UVLM-grounded flight env
(plan §6, PPO-first). Minimal clipped-PPO in torch (stable-baselines3 unavailable).

The policy = Takens delay embedding (stack the last n observations, plan §6) -> MLP
actor-critic, Gaussian continuous actions over [wing_aoa, roll, yaw, thrust]. Trains the
flapping MAV to hold a stable ~45deg-AoA cruise and reject a vertical gust. This is the
real NN controller (replacing the PD/attitude scaffold); SHAC is the later upgrade.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flight_ppo_env import FlightPPOEnv, OBS_DIM, ACT_DIM

DEVICE = "cpu"        # tiny net + Python env loop -> CPU avoids per-step GPU sync
N_EMBED = 6
OBS_SCALE = torch.tensor([0.5, 0.5, 3, 3, 3, 6, 3, 4, 8, 2.5], dtype=torch.float32)


class TakensAC(nn.Module):
    """Takens-embedding actor-critic: stack last n obs -> MLP -> (action mean, value)."""

    def __init__(self, n_embed=N_EMBED, h=64):
        super().__init__()
        self.n = n_embed
        din = OBS_DIM * n_embed
        self.body = nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.mu = nn.Linear(h, ACT_DIM)
        self.v = nn.Linear(h, 1)
        self.log_std = nn.Parameter(-0.5 * torch.ones(ACT_DIM))

    def forward(self, emb):
        z = self.body(emb)
        return torch.tanh(self.mu(z)), self.v(z).squeeze(-1)

    def dist(self, emb):
        mu, val = self(emb)
        return torch.distributions.Normal(mu, self.log_std.exp()), val


class Embedder:
    def __init__(self, n_embed=N_EMBED):
        self.n = n_embed; self.hist = []

    def reset(self):
        self.hist = []

    def push(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32) / OBS_SCALE
        self.hist.append(o)
        if len(self.hist) > self.n:
            self.hist = self.hist[-self.n:]
        emb = torch.zeros(OBS_DIM * self.n)
        flat = torch.cat(self.hist)
        emb[-len(flat):] = flat
        return emb


def collect(env, net, emb, steps):
    """Roll out `steps` transitions (reset on done). Returns tensors."""
    E, A, LP, R, V, D = [], [], [], [], [], []
    obs = env.reset(); emb.reset()
    e = emb.push(obs)
    for _ in range(steps):
        with torch.no_grad():
            dist, val = net.dist(e.unsqueeze(0))
            a = dist.sample()[0]
            lp = dist.log_prob(a).sum()
        obs, r, done, _ = env.step(a.numpy())
        E.append(e); A.append(a); LP.append(lp); R.append(r); V.append(val[0]); D.append(done)
        if done:
            obs = env.reset(); emb.reset()
        e = emb.push(obs)
    with torch.no_grad():
        _, last_v = net.dist(e.unsqueeze(0))
    return (torch.stack(E), torch.stack(A), torch.stack(LP), torch.tensor(R),
            torch.stack(V), torch.tensor(D, dtype=torch.float32), last_v[0])


def gae(R, V, D, last_v, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(R); gae_ = 0.0; nxt = last_v
    for t in reversed(range(len(R))):
        nonterm = 1.0 - D[t]
        delta = R[t] + gamma * nxt * nonterm - V[t]
        gae_ = delta + gamma * lam * nonterm * gae_
        adv[t] = gae_; nxt = V[t]
    ret = adv + V
    return (adv - adv.mean()) / (adv.std() + 1e-8), ret


def train(iters=120, steps=2048, epochs=8, mb=256, lr=3e-4, seed=0, log=print):
    torch.manual_seed(seed)
    env = FlightPPOEnv(seed=seed)
    net = TakensAC(); emb = Embedder()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        E, A, LP, R, V, D, last_v = collect(env, net, emb, steps)
        adv, ret = gae(R, V, D, last_v)
        # episode-return diagnostic: sum rewards between done flags
        ep_r, cur = [], 0.0
        for t in range(len(R)):
            cur += float(R[t])
            if D[t]:
                ep_r.append(cur); cur = 0.0
        mean_ep = float(np.mean(ep_r)) if ep_r else float(R.sum())
        n = len(E); idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                b = idx[s:s + mb]
                dist, val = net.dist(E[b])
                lp = dist.log_prob(A[b]).sum(-1)
                ratio = (lp - LP[b]).exp()
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv[b]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = ((val - ret[b]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pol_loss + 0.5 * v_loss - 0.01 * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        hist.append(mean_ep)
        if it % 10 == 0 or it == iters - 1:
            log(f"  iter {it:3d}: mean episode return={mean_ep:8.2f}  "
                f"(n_ep={len(ep_r)}, log_std={net.log_std.mean().item():+.2f})")
    return net, hist


if __name__ == "__main__":
    import warp as wp; wp.init()
    print("PPO: Takens NN policy on the fast UVLM-grounded flight env")
    env = FlightPPOEnv(); emb = Embedder(); net0 = TakensAC()
    # baseline (untrained)
    from flight_ppo_env import rollout
    class P:
        def reset(self): emb.reset()
        def act(self, o):
            with torch.no_grad():
                mu, _ = net0(emb.push(o).unsqueeze(0))
            return mu[0].numpy()
    r0, _ = rollout(env, P())
    print(f"  untrained baseline episode return = {r0:.2f}")
    net, hist = train(iters=120, steps=2048)
    print(f"PPO done: final mean return = {hist[-1]:.2f} (baseline {r0:.2f}, "
          f"best {max(hist):.2f})")
    np.savez("docs/ppo_hist.npz", hist=np.array(hist))
