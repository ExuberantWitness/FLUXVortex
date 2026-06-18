"""Closed-loop control on the co-design platform (control layer / Layer-0 control).

The plan's control (plan §6) is a Takens-time-delay-embedding policy trained
PPO-first that actuates the aircraft to reject gusts. This module shows the
control loop CLOSES on the real coupled platform: a policy reads the wing's recent
state history (Takens embedding) and injects a corrective control load inside the
predictor-corrector FSI loop, reducing the gust-induced excursion vs the passive
(uncontrolled) wing.

`TakensPolicy` is the plan's decided architecture (stack the last n observation
steps -> feedforward map -> action). Here it is instantiated as a PD feedback (a
degenerate policy with hand-set weights) so we can demonstrate closed-loop gust
rejection without an RL training run; PPO/SHAC learn the weights in production
(the coupled-eval RL needs the A100 cluster). The policy net is a tiny map (the
heavy compute stays the Warp FSI sim).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.join(_FLUXV, "tests"),
          os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from newton_pc import WindowPredictorCorrector                  # noqa: E402
from ancf_solver import WarpANCFEntry, NodalForceSet            # noqa: E402
from coupled_fsi import UVLMForceProvider                       # noqa: E402
from design_map import DesignMap                                # noqa: E402


class TakensPolicy:
    """Takens delay-embedding policy: action = f(stack of last n observations).

    obs here = tip (z, z_rate); embedding = the last n obs stacked; f = a linear
    map W (the production net; PD weights for this closed-loop demo). Action =
    scalar corrective tip load.
    """

    def __init__(self, n_embed=20, kp=50.0, kd=0.5):
        self.n = n_embed
        # PD weights over the embedding: act on the most-recent (z, z_rate)
        self.W = np.zeros((1, 2 * n_embed))
        self.W[0, -2] = -kp        # -kp * z_now
        self.W[0, -1] = -kd        # -kd * zrate_now
        self.hist = []

    def reset(self):
        self.hist = []

    def act(self, obs):
        self.hist.append(np.asarray(obs, float))
        if len(self.hist) > self.n:
            self.hist = self.hist[-self.n:]
        emb = np.zeros(2 * self.n)
        h = np.concatenate(self.hist) if self.hist else np.zeros(0)
        if len(h):
            emb[-len(h):] = h      # right-align (zero-pad early steps)
        return float((self.W @ emb).ravel()[0])


class ControlledProvider(UVLMForceProvider):
    """UVLM provider that adds the policy's corrective tip load to the aero force."""

    def bind_control(self, policy, tip, z0):
        self.policy = policy; self.tip = tip; self.z0 = z0; self._prev_z = z0
        return self

    def solve(self, state):
        fs = super().solve(state)
        if self.policy is None:
            return fs
        z = float(self.entry.q.numpy()[0, 9 * self.tip + 2])
        zrate = (z - self._prev_z); self._prev_z = z
        u = self.policy.act([z - self.z0, zrate])
        fs.gen = fs.gen.copy()
        fs.gen[0, 9 * self.tip + 2] += u        # corrective z-load at the tip
        return fs


def _run(dmap, design, policy, *, n_base=2, n_gust=3, n_recover=4, substeps=34,
         dt=2e-4, gust_w=2.5, device=None):
    dev = device or cfg.DEVICE
    shell = dmap.to_shell(design)
    from run_standalone_yamano import yamano_params
    from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
    params = yamano_params(); V0 = float(params["V_inf"])
    solver = StandaloneHybridSolver(
        shell, np.array([V0, 0.0, 0.0]), rho_fluid=params["rho_fluid"],
        structural_dt=dt, uvlm_dt_ratio=substeps, integrator="implicit",
        relaxation=1.0, newton_tol=1e-4, max_newton=20, max_particles=5000,
        wake_truncation=5.5, core_radius=1e-6, coupling="strong")
    entry = WarpANCFEntry(shell, B=1, alpha_v=0.5, c_damp=2.0, device=dev)
    tip = shell.nn - 1
    z0 = float(entry.q.numpy()[0, 9 * tip + 2])
    provider = ControlledProvider(solver, entry, wake=False)
    provider.policy = None
    if policy is not None:
        policy.reset(); provider.bind_control(policy, tip, z0)
    pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                                  dt=dt, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros((1, entry.ndof), dtype=cfg.NP_DTYPE)))
    pc.advance(n_substeps=1)
    peak = 0.0
    total = n_base + n_gust + n_recover
    for w in range(total):
        gz = 0.0
        if n_base <= w < n_base + n_gust:
            frac = (w - n_base + 0.5) / n_gust
            gz = 0.5 * gust_w * (1.0 - np.cos(2.0 * np.pi * frac))
        provider.fluid.V_inf = np.array([V0, 0.0, gz])
        pc.advance()
        peak = max(peak, abs(float(entry.q.numpy()[0, 9 * tip + 2]) - z0))
    return peak


def verify() -> bool:
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    design = [1.0, 1.0]
    passive = _run(dmap, design, None)
    controlled = _run(dmap, design, TakensPolicy(n_embed=20))
    reduced = controlled < passive
    print(f"Closed-loop gust rejection (Takens-embedding policy in the FSI loop):")
    print(f"  peak tip excursion: passive={passive:.4e}  controlled={controlled:.4e}  "
          f"reduction={100*(1-controlled/passive):.1f}%")
    print(f"  -> {'PASS' if reduced else 'FAIL'}: control loop closes on the coupled "
          f"platform (PPO/SHAC learn the policy weights in production)")
    return reduced


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
