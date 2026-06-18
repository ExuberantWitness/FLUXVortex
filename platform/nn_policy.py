"""Neural-network Takens-embedding policy + evolution-strategies trainer (plan §6).

Replaces the hand-set PD (control_eval.TakensPolicy) with a real trained neural net:
a Takens delay-embedding (stack the last n observations) feeding a 2-hidden-layer MLP
that commands all 16 actuators of the full 6-DOF aircraft. Trained by OpenAI-ES
(antithetic, rank-normalized) — gradient-free, robust to the non-smooth/contact-free
flight dynamics, and parallelizable; PPO and differentiable SHAC are the plan's noted
successors (§6). The policy net is tiny; the compute is the flight rollouts.
"""
from __future__ import annotations

import numpy as np

from flight_env import OBS_DIM, ACT_DIM

# fixed observation scaling (angles, rates, velocities, errors, flap) -> ~unit
OBS_SCALE = np.array([0.5, 0.5, 0.5, 3.0, 3.0, 3.0, 10.0, 3.0, 3.0, 5.0, 5.0, 0.5, 0.5])


class TakensNN:
    """Delay-embedding MLP policy. Weights are a single flat vector (for ES)."""

    def __init__(self, n_embed=12, h1=64, h2=32, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        self.n = n_embed
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.IN = obs_dim * n_embed
        self.shapes = [(h1, self.IN), (h1,), (h2, h1), (h2,), (act_dim, h2), (act_dim,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = int(sum(self.sizes))
        self.hist = []
        self._set(np.zeros(self.dim))

    def _set(self, theta):
        self.theta = np.asarray(theta, float)
        ws, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            ws.append(self.theta[i:i + n].reshape(s)); i += n
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = ws

    def reset(self):
        self.hist = []

    def act(self, obs):
        o = np.asarray(obs, float) / OBS_SCALE
        self.hist.append(o)
        if len(self.hist) > self.n:
            self.hist = self.hist[-self.n:]
        emb = np.zeros(self.IN)
        flat = np.concatenate(self.hist)
        emb[-len(flat):] = flat                  # right-align, zero-pad early steps
        h = np.tanh(self.W1 @ emb + self.b1)
        h = np.tanh(self.W2 @ h + self.b2)
        return np.tanh(self.W3 @ h + self.b3)


def es_train(env_fn, policy, *, generations=60, pop=40, sigma=0.1, lr=0.03,
             seed=0, log=print):
    """OpenAI-ES on the flat policy weights. env_fn() -> a (reusable) FlightEnv."""
    from flight_env import rollout
    rng = np.random.default_rng(seed)
    env = env_fn()
    theta = 0.1 * rng.standard_normal(policy.dim)
    half = pop // 2
    best = -1e18; best_theta = theta.copy()
    hist = []
    for g in range(generations):
        eps = rng.standard_normal((half, policy.dim))
        eps = np.concatenate([eps, -eps], 0)     # antithetic
        R = np.zeros(pop)
        for i in range(pop):
            policy._set(theta + sigma * eps[i])
            R[i], _ = rollout(env, policy)
        # rank-normalize rewards to [-0.5, 0.5]
        ranks = np.argsort(np.argsort(R)) / (pop - 1) - 0.5
        grad = (eps.T @ ranks) / (pop * sigma)
        theta = theta + lr * grad
        policy._set(theta)
        r_eval, _ = rollout(env, policy)
        if r_eval > best:
            best = r_eval; best_theta = theta.copy()
        hist.append((float(R.mean()), float(R.max()), float(r_eval)))
        log(f"  gen {g:3d}: pop mean={R.mean():8.2f} max={R.max():8.2f}  "
            f"eval={r_eval:8.2f}  best={best:8.2f}")
    policy._set(best_theta)
    return best_theta, best, hist


if __name__ == "__main__":
    import warp as wp
    from flight_env import FlightEnv, rollout
    wp.init()
    pol = TakensNN(n_embed=12)
    print(f"TakensNN policy: {pol.dim} weights (Takens n={pol.n}, IN={pol.IN} -> 64 -> 32 -> 16)")
    env_fn = lambda: FlightEnv(horizon=300, substep=2)
    # baseline: untrained / zero policy crashes fast
    env = env_fn()
    pol._set(np.zeros(pol.dim))
    r0, _ = rollout(env, pol)
    print(f"zero-policy baseline reward = {r0:.2f}")
    theta, best, hist = es_train(env_fn, pol, generations=40, pop=40)
    print(f"ES done: best eval reward = {best:.2f} (baseline {r0:.2f})")
    np.savez("docs/nn_policy_trained.npz", theta=theta, hist=np.array(hist))
