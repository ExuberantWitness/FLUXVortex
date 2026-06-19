"""Differentiable rollout through the flight dynamics (SHAC design gradient) — the full
∂(Σ reward)/∂ctrl, chaining the design 刚柔 field aggregates into a T-step Warp rollout.

This is the SHAC piece beyond the gust-aggregate gradient (validate_gpu_env test 5): the
design gradient now flows through the WHOLE rollout — ctrl -> design_agg_sums ->
gust_factor/ctrl_factor -> T steps of dynamics -> reward.

Two Warp requirements force a separate `step_diff` kernel (the in-place `step_kernel` used
for fast PPO forward is NOT tape-able across launches):
  1. NO in-place state mutation across launches — each step reads x[t] and writes x[t+1]
     into a distinct buffer, so the tape can reverse the whole rollout.
  2. NO gradient-severing gates on differentiable quantities — the action clamp, reward
     clips, the crash branch, and the `if ln>1e-6` lift-direction guard all sever the
     adjoint of ctrl_factor/gust_factor in Warp. The differentiable rollout drops them
     (actions kept in the linear region, no early termination, always-normalize lift dir).
     The reward is the smooth quadratic (crash cutoff is the non-differentiable boundary
     handled by gradient checkpointing / subgradient in a full SHAC run).

validate(): ∂(Σreward)/∂ctrl from the Warp tape vs central finite differences of the SAME
GPU forward (so AD and FD use an identical reward function — a tight self-consistency red
line for the through-rollout design gradient).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from uvlm_db import AeroDB
import design_field as dfield
from gpu_flight_env import (interp1, interp_ctrl, design_agg_sums, design_agg_gust_final,
                            K_CTRL, _NG, _TAPER, _NORM_W, G, _TWO_PI, ACT_DIM)

wp.set_module_options({"enable_backward": True})


@wp.kernel
def step_diff(x_in: wp.array(dtype=wp.vec3d), v_in: wp.array(dtype=wp.vec3d),
              om_in: wp.array(dtype=wp.vec3d), q_in: wp.array(dtype=wp.quatd),
              act: wp.array2d(dtype=wp.float64),
              gust_factor: wp.array(dtype=wp.float64), ctrl_factor: wp.array(dtype=wp.float64),
              a_tab: wp.array(dtype=wp.float64), cl_tab: wp.array(dtype=wp.float64),
              cd_tab: wp.array(dtype=wp.float64), ntab: int,
              S: wp.float64, att: wp.float64, dt: wp.float64, m: wp.float64,
              Vc: wp.float64, alt0: wp.float64, Tmax: wp.float64, Mmax: wp.float64,
              gw: wp.float64, gt0: wp.float64, gdur: wp.float64,
              Ix: wp.float64, Iy: wp.float64, Iz: wp.float64, tnow: wp.float64,
              x_out: wp.array(dtype=wp.vec3d), v_out: wp.array(dtype=wp.vec3d),
              om_out: wp.array(dtype=wp.vec3d), q_out: wp.array(dtype=wp.quatd),
              reward: wp.array(dtype=wp.float64)):
    i = wp.tid()
    # actions: NO clamp (kept in the linear region) so ctrl_factor's adjoint flows
    a0 = act[i, 0]
    a1 = act[i, 1] * ctrl_factor[i]
    a2 = act[i, 2] * ctrl_factor[i]
    a3 = act[i, 3]
    wing_aoa = a0 * att
    roll_m = a1 * Mmax; yaw_m = a2 * Mmax
    thr = wp.float64(0.5) * (a3 + wp.float64(1.0))

    gust = wp.float64(0.0)
    if tnow >= gt0 and tnow < gt0 + gdur:
        fr = (tnow - gt0) / gdur
        gust = wp.float64(0.5) * gw * (wp.float64(1.0) - wp.cos(wp.float64(_TWO_PI) * fr))
    gust = gust * gust_factor[i]

    qq = q_in[i]
    qx = qq[0]; qy = qq[1]; qz = qq[2]; qw = qq[3]
    r00 = wp.float64(1.0) - wp.float64(2.0) * (qy * qy + qz * qz)
    r01 = wp.float64(2.0) * (qx * qy - qz * qw)
    r02 = wp.float64(2.0) * (qx * qz + qy * qw)
    r10 = wp.float64(2.0) * (qx * qy + qz * qw)
    r11 = wp.float64(1.0) - wp.float64(2.0) * (qx * qx + qz * qz)
    r12 = wp.float64(2.0) * (qy * qz - qx * qw)
    r20 = wp.float64(2.0) * (qx * qz - qy * qw)
    r21 = wp.float64(2.0) * (qy * qz + qx * qw)
    r22 = wp.float64(1.0) - wp.float64(2.0) * (qx * qx + qy * qy)

    vv = v_in[i]
    dwx = wp.float64(0.0) - vv[0]; dwy = wp.float64(0.0) - vv[1]; dwz = gust - vv[2]
    vrx = r00 * dwx + r10 * dwy + r20 * dwz
    vry = r01 * dwx + r11 * dwy + r21 * dwz
    vrz = r02 * dwx + r12 * dwy + r22 * dwz
    sp = wp.sqrt(vrx * vrx + vry * vry + vrz * vrz) + wp.float64(1e-9)
    q_dyn = wp.float64(0.5) * wp.float64(1.225) * sp * sp

    aa = wp.abs(wing_aoa)             # aero AoA is action-driven (no ctrl-gradient through it)
    if aa > att:
        aa = att
    sgn = wp.float64(0.0)
    if wing_aoa > wp.float64(0.0):
        sgn = wp.float64(1.0)
    if wing_aoa < wp.float64(0.0):
        sgn = wp.float64(-1.0)
    cl = interp1(aa, a_tab, cl_tab, ntab) * sgn
    cd = interp1(aa, a_tab, cd_tab, ntab)

    ux = vrx / sp; uy = vry / sp; uz = vrz / sp
    lx = wp.float64(0.0) - uz * ux
    ly = wp.float64(0.0) - uz * uy
    lz = wp.float64(1.0) - uz * uz
    ln = wp.sqrt(lx * lx + ly * ly + lz * lz) + wp.float64(1e-9)   # always normalize (no gate)
    lx = lx / ln; ly = ly / ln; lz = lz / ln

    qs = q_dyn * S
    fbx = qs * (cl * lx + cd * ux) + thr * Tmax
    fby = qs * (cl * ly + cd * uy)
    fbz = qs * (cl * lz + cd * uz)
    fwx = r00 * fbx + r01 * fby + r02 * fbz
    fwy = r10 * fbx + r11 * fby + r12 * fbz
    fwz = r20 * fbx + r21 * fby + r22 * fbz - m * wp.float64(G)
    omv = om_in[i]
    mwx = r00 * roll_m + r02 * yaw_m - wp.float64(0.05) * omv[0]
    mwy = r10 * roll_m + r12 * yaw_m - wp.float64(0.05) * omv[1]
    mwz = r20 * roll_m + r22 * yaw_m - wp.float64(0.05) * omv[2]

    nvx = vv[0] + fwx / m * dt; nvy = vv[1] + fwy / m * dt; nvz = vv[2] + fwz / m * dt
    nx = x_in[i][0] + nvx * dt; ny = x_in[i][1] + nvy * dt; nz = x_in[i][2] + nvz * dt
    Iox = Ix * omv[0]; Ioy = Iy * omv[1]; Ioz = Iz * omv[2]
    cx = omv[1] * Ioz - omv[2] * Ioy; cy = omv[2] * Iox - omv[0] * Ioz; cz = omv[0] * Ioy - omv[1] * Iox
    nomx = omv[0] + (mwx - cx) / Ix * dt
    nomy = omv[1] + (mwy - cy) / Iy * dt
    nomz = omv[2] + (mwz - cz) / Iz * dt
    dqx = wp.float64(0.5) * (nomx * qw + nomy * qz - nomz * qy)
    dqy = wp.float64(0.5) * (nomy * qw + nomz * qx - nomx * qz)
    dqz = wp.float64(0.5) * (nomz * qw + nomx * qy - nomy * qx)
    dqw = wp.float64(0.5) * (-(nomx * qx + nomy * qy + nomz * qz))
    nqx = qx + dqx * dt; nqy = qy + dqy * dt; nqz = qz + dqz * dt; nqw = qw + dqw * dt
    qn = wp.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw) + wp.float64(1e-12)
    nqx = nqx / qn; nqy = nqy / qn; nqz = nqz / qn; nqw = nqw / qn

    x_out[i] = wp.vec3d(nx, ny, nz)
    v_out[i] = wp.vec3d(nvx, nvy, nvz)
    om_out[i] = wp.vec3d(nomx, nomy, nomz)
    q_out[i] = wp.quatd(nqx, nqy, nqz, nqw)

    roll = wp.atan2(wp.float64(2.0) * (nqw * nqx + nqy * nqz),
                    wp.float64(1.0) - wp.float64(2.0) * (nqx * nqx + nqy * nqy))
    # smooth reward (no clip / no crash override -> AD-FD consistent)
    zc = nz - alt0; vc = nvx - Vc
    r = wp.float64(1.0) - wp.float64(0.04) * zc * zc - wp.float64(0.05) * vc * vc
    r = r - wp.float64(0.3) * nvz * nvz - wp.float64(0.2) * roll * roll - wp.float64(0.1) * nvy * nvy
    r = r - wp.float64(0.02) * (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    reward[i] = r


class _Params:
    def __init__(self):
        a = AeroDB()
        self.S, self.att = a.S, a.att
        self.a_tab = a._a.astype(np.float64); self.cl = a._cl.astype(np.float64)
        self.cd = a._cd.astype(np.float64); self.ntab = len(a._a)
        self.dt, self.m, self.Vc, self.alt0 = 0.01, 0.45, 8.0, 30.0
        self.Tmax, self.Mmax = 6.0, 0.4
        self.gw, self.gt0, self.gdur = 4.0, 1.0, 0.5
        self.Ix, self.Iy, self.Iz = 3e-3, 5e-3, 5e-3
        self.body_aoa = np.deg2rad(45.0)


def rollout(ctrl, actions, T, P, want_grad=False, dev="cuda"):
    """ctrl (B,K), actions (T,B,ACT_DIM). Returns (per-env total reward [B], grad [B,K] or None).
    want_grad=True records the tape ctrl -> aggregates -> T steps -> reward and backprops."""
    B = ctrl.shape[0]; rg = want_grad
    f64 = wp.float64
    a_tab = wp.array(P.a_tab, dtype=f64, device=dev)
    cl_tab = wp.array(P.cl, dtype=f64, device=dev); cd_tab = wp.array(P.cd, dtype=f64, device=dev)
    ctrl_wp = wp.array(ctrl, dtype=f64, device=dev, requires_grad=rg)
    sum_wc = wp.zeros(B, dtype=f64, device=dev, requires_grad=rg)
    sum_mgs = wp.zeros(B, dtype=f64, device=dev, requires_grad=rg)
    gf = wp.zeros(B, dtype=f64, device=dev, requires_grad=rg)
    cf = wp.zeros(B, dtype=f64, device=dev, requires_grad=rg)
    sg = wp.zeros(B, dtype=f64, device=dev, requires_grad=rg)
    # per-step state buffers (distinct -> tape-able)
    xs = [wp.zeros(B, dtype=wp.vec3d, device=dev, requires_grad=rg) for _ in range(T + 1)]
    vs = [wp.zeros(B, dtype=wp.vec3d, device=dev, requires_grad=rg) for _ in range(T + 1)]
    oms = [wp.zeros(B, dtype=wp.vec3d, device=dev, requires_grad=rg) for _ in range(T + 1)]
    qs = [wp.zeros(B, dtype=wp.quatd, device=dev, requires_grad=rg) for _ in range(T + 1)]
    rew = [wp.zeros(B, dtype=f64, device=dev, requires_grad=rg) for _ in range(T)]
    acts = [wp.array(np.ascontiguousarray(actions[t], np.float64), dtype=f64, device=dev)
            for t in range(T)]
    a = P.body_aoa
    xs[0].assign(np.tile([0., 0., P.alt0], (B, 1)).astype(np.float64))
    vs[0].assign(np.tile([P.Vc, 0., 0.], (B, 1)).astype(np.float64))
    qs[0].assign(np.tile([0., -np.sin(a / 2), 0., np.cos(a / 2)], (B, 1)).astype(np.float64))

    def body():
        wp.launch(design_agg_sums, dim=B, inputs=[ctrl_wp, K_CTRL, _NG, np.float64(_TAPER)],
                  outputs=[sum_wc, sum_mgs], device=dev)
        wp.launch(design_agg_gust_final, dim=B, inputs=[sum_wc, np.float64(_NORM_W)],
                  outputs=[gf, cf, sg], device=dev)
        for t in range(T):
            wp.launch(step_diff, dim=B,
                      inputs=[xs[t], vs[t], oms[t], qs[t], acts[t], gf, cf,
                              a_tab, cl_tab, cd_tab, P.ntab,
                              np.float64(P.S), np.float64(P.att), np.float64(P.dt),
                              np.float64(P.m), np.float64(P.Vc), np.float64(P.alt0),
                              np.float64(P.Tmax), np.float64(P.Mmax), np.float64(P.gw),
                              np.float64(P.gt0), np.float64(P.gdur), np.float64(P.Ix),
                              np.float64(P.Iy), np.float64(P.Iz), np.float64(t * P.dt)],
                      outputs=[xs[t + 1], vs[t + 1], oms[t + 1], qs[t + 1], rew[t]],
                      device=dev)

    if want_grad:
        tape = wp.Tape()
        with tape:
            body()
        for t in range(T):
            rew[t].grad = wp.array(np.ones(B, np.float64), dtype=f64, device=dev)
        tape.backward()
        grad = ctrl_wp.grad.numpy()
    else:
        body(); grad = None
    total = np.sum([rew[t].numpy() for t in range(T)], axis=0)    # per-env Σ_t reward
    return total, grad


def validate(B=8, T=170, seed=0, act_scale=0.2):
    """T=170 steps (1.7s) spans the gust window (t0=1.0,dur=0.5) so BOTH design paths are
    exercised: ctrl_factor (every step) AND gust_factor (during the gust). Small actions
    keep the un-terminated rollout bounded for a clean AD-vs-FD comparison."""
    wp.init()
    P = _Params()
    rng = np.random.default_rng(seed)
    ctrl = np.exp(rng.uniform(np.log(0.4), np.log(2.4), size=(B, K_CTRL)))
    actions = (act_scale * rng.uniform(-1.0, 1.0, size=(T, B, ACT_DIM))).astype(np.float64)
    print(f"SHAC through-rollout design gradient  ∂(Σreward)/∂ctrl  (B={B}, T={T} steps, "
          f"spans the gust window)")
    total, g_ad = rollout(ctrl, actions, T, P, want_grad=True)
    finite = np.all(np.isfinite(g_ad))
    # central finite differences of the SAME GPU forward
    eps = 1e-6; g_fd = np.zeros_like(ctrl)
    for k in range(K_CTRL):
        cp = ctrl.copy(); cp[:, k] += eps
        cm = ctrl.copy(); cm[:, k] -= eps
        tp, _ = rollout(cp, actions, T, P, want_grad=False)
        tm, _ = rollout(cm, actions, T, P, want_grad=False)
        g_fd[:, k] = (tp - tm) / (2 * eps)
    rel = float(np.max(np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-3)))
    ok = finite and rel < 1e-5
    print(f"  finite={finite}  max rel err vs FD={rel:.2e}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  (design gradient flows ctrl -> 刚柔 aggregates -> {T}-step dynamics -> reward;"
          f" the full SHAC design gradient)")
    print(f"  sample ∂(Σreward)/∂ctrl[0] = {np.array2string(g_ad[0], precision=3)}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
