"""Red-line battery for the Warp GPU flight env vs the numpy oracle.

The numpy MetaFlightEnv + design_field are the VALIDATION ORACLE. This proves the Warp
batched GPU env reproduces them, so moving co-design onto the GPU does not change the
physics — plus the env is differentiable (Warp tape vs finite differences).

  1. design aggregates : GPU design_aggregates == design_field (s_gust/s_root/factors/eff)
  2. per-step dynamics  : one step from random states/actions/designs == numpy MetaFlightEnv
  3. short rollout      : 120-step fixed-action rollout reward-sum/trajectory match
  4. tape vs FD         : ∂(Σ efficiency)/∂ctrl from the Warp tape == finite differences
"""
from __future__ import annotations

import numpy as np
import warp as wp

import design_field as dfield
from gpu_flight_env import (GpuFlightEnv, design_agg_eff, K_CTRL, _NG, _TAPER,
                            _NORM_W, _NORM_MG, OBS_DIM, ACT_DIM)
from meta_rl_train import MetaFlightEnv


def _rand_fields(n, rng, K=K_CTRL):
    return np.exp(rng.uniform(np.log(0.3), np.log(2.5), size=(n, K))).astype(np.float64)


def _rand_states(n, rng, alt0=30.0, Vc=8.0):
    """Random but physical flight states (near the launch manifold)."""
    a = np.deg2rad(45.0)
    x = np.c_[rng.normal(0, 1, n), rng.normal(0, 1, n), alt0 + rng.normal(0, 3, n)]
    qbase = np.array([0.0, -np.sin(a / 2), 0.0, np.cos(a / 2)])
    q = qbase + 0.1 * rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = np.c_[Vc + rng.normal(0, 1.5, n), rng.normal(0, 1, n), rng.normal(0, 1, n)]
    om = 0.3 * rng.standard_normal((n, 3))
    t = rng.uniform(0.0, 2.0, n)
    return (x.astype(np.float64), q.astype(np.float64), v.astype(np.float64),
            om.astype(np.float64), t.astype(np.float64))


def test_aggregates(rng):
    n = 256
    ctrl = _rand_fields(n, rng)
    gpu = GpuFlightEnv(B=n)
    gpu.set_designs(ctrl)
    g_gf, g_cf, g_eff = gpu.gust_factor.numpy(), gpu.ctrl_factor.numpy(), gpu.efficiency.numpy()
    g_sg, g_sr = gpu.s_gust.numpy(), gpu.s_root.numpy()
    # numpy oracle
    o_sg = np.array([dfield.StiffnessField(c).s_gust() for c in ctrl])
    o_sr = np.array([dfield.StiffnessField(c).s_root() for c in ctrl])
    o_gf = np.array([dfield.gust_factor(dfield.StiffnessField(c)) for c in ctrl])
    o_eff = np.array([dfield.cruise_efficiency(dfield.StiffnessField(c)) for c in ctrl])
    def rel(a, b):
        return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))
    rs = dict(s_gust=rel(g_sg, o_sg), s_root=rel(g_sr, o_sr),
              gust_factor=rel(g_gf, o_gf), efficiency=rel(g_eff, o_eff))
    ok = all(v < 1e-9 for v in rs.values())
    print(f"[1 aggregates] rel: " + "  ".join(f"{k}={v:.2e}" for k, v in rs.items())
          + f"  -> {'PASS' if ok else 'FAIL'} (fp64 GPU vs fp64 numpy)")
    return ok


def test_step(rng):
    n = 512
    ctrl = _rand_fields(n, rng)
    x, q, v, om, t = _rand_states(n, rng)
    act = rng.uniform(-1.2, 1.2, size=(n, ACT_DIM)).astype(np.float32)
    # GPU: set state directly, step once
    gpu = GpuFlightEnv(B=n)
    gpu.set_designs(ctrl)
    gpu.x.assign(x); gpu.q.assign(q); gpu.v.assign(v); gpu.om.assign(om)
    gpu.tt.assign(t); gpu.stepi.assign(np.zeros(n, np.int32))
    g_obs, g_rew, _ = gpu.step(act)
    # numpy oracle, per env
    o_obs = np.zeros((n, OBS_DIM)); o_rew = np.zeros(n)
    env = MetaFlightEnv()
    for i in range(n):
        env._apply_design(dfield.StiffnessField(ctrl[i]))
        env.x = x[i].astype(np.float64).copy(); env.q = q[i].astype(np.float64).copy()
        env.v = v[i].astype(np.float64).copy(); env.om = om[i].astype(np.float64).copy()
        env.t = float(t[i]); env.step_i = 0
        o_obs[i], o_rew[i], _, _ = env.step(act[i].astype(np.float64))
    obs_err = float(np.max(np.abs(g_obs - o_obs)))
    rew_err = float(np.max(np.abs(g_rew - o_rew)))
    ok = obs_err < 1e-9 and rew_err < 1e-9
    print(f"[2 per-step ]  max|Δobs|={obs_err:.2e}  max|Δreward|={rew_err:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'} ({n} random states/actions/designs)")
    return ok


def test_rollout(rng):
    n = 64
    ctrl = _rand_fields(n, rng)
    T = 120
    acts = rng.uniform(-0.6, 0.6, size=(T, n, ACT_DIM)).astype(np.float32)
    gpu = GpuFlightEnv(B=n)
    gpu.reset(designs=ctrl)
    g_sum = np.zeros(n); g_alive = np.ones(n, bool)
    for k in range(T):
        _, r, d = gpu.step(acts[k])
        g_sum += r * g_alive
        g_alive &= ~d
    # numpy oracle
    o_sum = np.zeros(n)
    env = MetaFlightEnv()
    for i in range(n):
        env._apply_design(dfield.StiffnessField(ctrl[i]))
        env.reset(design=dfield.StiffnessField(ctrl[i]))
        for k in range(T):
            _, r, d, _ = env.step(acts[k, i].astype(np.float64))
            o_sum[i] += r
            if d:
                break
    err = float(np.max(np.abs(g_sum - o_sum)))
    rel = float(np.max(np.abs(g_sum - o_sum) / (np.abs(o_sum) + 1.0)))
    ok = rel < 1e-7
    print(f"[3 rollout  ]  {T} steps, max|Δreturn|={err:.3e}  rel={rel:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'} (fp64 over rollout)")
    return ok


def test_grad(rng):
    n = 32
    dev = "cuda"
    ctrl = _rand_fields(n, rng)
    ctrl_wp = wp.array(ctrl, dtype=wp.float64, device=dev, requires_grad=True)
    # detached relu mask (pen active iff compliance C>2), from the numpy oracle
    Cv = np.array([dfield.StiffnessField(c).feather_compliance() for c in ctrl])
    pen_mask = wp.array((Cv > 2.0).astype(np.float64), dtype=wp.float64, device=dev)
    # fresh output arrays (zero-grad) so no stale .grad from non-tape launches leaks in
    out = [wp.zeros(n, dtype=wp.float64, device=dev, requires_grad=True) for _ in range(3)]
    tape = wp.Tape()
    with tape:
        wp.launch(design_agg_eff, dim=n,
                  inputs=[ctrl_wp, K_CTRL, _NG, np.float64(_TAPER),
                          np.float64(_NORM_W), np.float64(_NORM_MG), pen_mask],
                  outputs=out, device=dev)
    out[0].grad = wp.array(np.ones(n, np.float64), dtype=wp.float64, device=dev)  # eff seed
    tape.backward()
    g_ad = ctrl_wp.grad.numpy()                      # ∂(Σ eff)/∂ctrl
    # finite differences
    eps = 1e-5
    g_fd = np.zeros_like(ctrl)
    def eff_of(c):
        return np.array([dfield.cruise_efficiency(dfield.StiffnessField(cc)) for cc in c])
    for k in range(K_CTRL):
        cp = ctrl.copy(); cp[:, k] += eps
        cm = ctrl.copy(); cm[:, k] -= eps
        g_fd[:, k] = (eff_of(cp) - eff_of(cm)) / (2 * eps)
    rel = float(np.max(np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-2)))
    ok = rel < 1e-4
    print(f"[4 tape vs FD]  ∂(Σeff)/∂ctrl  max rel err={rel:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'} (Warp autodiff through aggregates)")
    return ok


if __name__ == "__main__":
    wp.init()
    rng = np.random.default_rng(0)
    print("Warp GPU flight env  vs  numpy oracle (MetaFlightEnv + design_field)")
    r1 = test_aggregates(rng)
    r2 = test_step(rng)
    r3 = test_rollout(rng)
    r4 = test_grad(rng)
    allok = r1 and r2 and r3 and r4
    print(f"\nGPU env red lines: aggregates={r1} per-step={r2} rollout={r3} grad={r4}  "
          f"-> {'ALL PASS' if allok else 'FAIL'}")
    raise SystemExit(0 if allok else 1)
