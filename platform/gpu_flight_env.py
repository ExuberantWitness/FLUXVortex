"""GPU-native batched flight environment (Warp, fp64) — replaces the numpy FlightPPOEnv/
MetaFlightEnv surrogate with a single Warp kernel stepping B environments in parallel,
end-to-end differentiable (Warp tape). The plan's Layer-0 "GPU 批量向量化环境" +
"单大 kernel 跨 B 环境" + "可微 = Warp autodiff" + "精度 = 全程 fp64".

Two Warp kernels:
  · design_aggregates : ctrl[B,K] spanwise 刚柔 field control points -> per-env
    (gust_factor, ctrl_factor, efficiency) via the SAME load-weighted (tip-biased) /
    bending-moment-weighted (root-biased) reduction as design_field.py. Differentiable
    w.r.t. ctrl -> DQD design gradients flow on the GPU (no numpy). The ctrl-independent
    normalizers ∫w, ∫m are host-precomputed constants (so the kernel never divides by a
    loop-accumulated variable, which would break Warp's adjoint).
  · step : batched 6-DOF rigid-body flight dynamics with the UVLM-tabulated aero
    (a_tab/cl_tab/cd_tab), gust, control authority, reward — one thread per environment.

The numpy FlightPPOEnv/MetaFlightEnv + design_field stay as the VALIDATION ORACLE
(validate_gpu_env.py: GPU==numpy per-step/rollout/aggregates + tape vs FD). fp64 matches
the coupled-FSI golden precision; throughput drops on consumer GPUs but stays batched.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from uvlm_db import AeroDB
import design_field as dfield

OBS_DIM = 10
ACT_DIM = 4
K_CTRL = 4                      # spanwise stiffness-field control points (root->tip)
_NG = 96                        # spanwise quadrature (matches design_field._NG)
_TAPER = dfield._TAPER
G = 9.81
_TWO_PI = 6.283185307179586

wp.set_module_options({"enable_backward": True})

# host-precomputed normalization constants (independent of ctrl) — passed as scalar params
# so the kernel never divides by a loop-accumulated variable (that breaks Warp's adjoint).
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
_XG = np.linspace(0.0, 1.0, _NG)
_DX = _XG[1] - _XG[0]
_WG = _XG * (1.0 - (1.0 - _TAPER) * _XG)
_W1 = 0.5 - (1.0 - _TAPER) / 3.0
_MG = _W1 - (0.5 * _XG ** 2 - (1.0 - _TAPER) * _XG ** 3 / 3.0)
# BARE trapezoid weighted sums (NO dx) — the kernel accumulates sum_wc/sum_mgs the same
# way (endpoint weight 0.5, no dx), so the dx cancels in C=sum_wc/norm_w as in design_field.
_NORM_W = float(_trapz(_WG, _XG) / _DX)      # Σ wt·w   (= ∫w dξ / dξ)
_NORM_MG = float(_trapz(_MG, _XG) / _DX)     # Σ wt·m


# ───────────────────────── Warp device functions ─────────────────────────────
@wp.func
def interp1(a: wp.float64, xs: wp.array(dtype=wp.float64),
            ys: wp.array(dtype=wp.float64), n: int) -> wp.float64:
    """np.interp clone on sorted xs (clamps to endpoints)."""
    res = ys[n - 1]
    if a <= xs[0]:
        res = ys[0]
    else:
        found = int(0)
        for k in range(n - 1):
            if found == 0 and a <= xs[k + 1]:
                t = (a - xs[k]) / (xs[k + 1] - xs[k])
                res = ys[k] * (wp.float64(1.0) - t) + ys[k + 1] * t
                found = 1
    return res


@wp.func
def interp_ctrl(ctrl: wp.array2d(dtype=wp.float64), e: int, K: int,
                xi: wp.float64) -> wp.float64:
    """Linear interp of the K control points at uniform xi_k=k/(K-1) (np.interp clone).
    Single-return (no early return) so the Warp adjoint stays well-defined."""
    pos = xi * wp.float64(K - 1)
    k = int(pos)
    if k > K - 2:
        k = K - 2                       # clamp so k, k+1 are valid (xi=1 -> t=1 -> ctrl[K-1])
    t = pos - wp.float64(k)
    return ctrl[e, k] * (wp.float64(1.0) - t) + ctrl[e, k + 1] * t


# ───────────────────────── design-field aggregates (differentiable) ──────────
# Split into TWO kernels by which accumulator sits in a DENOMINATOR. The efficiency path
# uses the shared accumulators only in NUMERATORS (s_root=sum_mgs/norm, C=sum_wc/norm) ->
# fully adjoint-clean. The gust path needs s_gust=norm_w/sum_wc (accumulator in the
# denominator). Keeping them separate means the DQD design gradient ∂efficiency/∂ctrl
# never shares a variable with the denominator-division (which trips a Warp adjoint NaN).
@wp.kernel
def design_agg_eff(ctrl: wp.array2d(dtype=wp.float64), K: int, NG: int,
                   taper: wp.float64, norm_w: wp.float64, norm_mg: wp.float64,
                   pen_mask: wp.array(dtype=wp.float64),
                   efficiency: wp.array(dtype=wp.float64),
                   s_root_o: wp.array(dtype=wp.float64),
                   c_o: wp.array(dtype=wp.float64)):
    e = wp.tid()
    dx = wp.float64(1.0) / wp.float64(NG - 1)
    one_mt = wp.float64(1.0) - taper
    W1 = wp.float64(0.5) - one_mt / wp.float64(3.0)        # W(1)
    sum_wc = wp.float64(0.0); sum_mgs = wp.float64(0.0)
    for g in range(NG):
        xi = wp.float64(g) * dx
        sg = interp_ctrl(ctrl, e, K, xi)
        ch = wp.float64(1.0) - one_mt * xi
        w = xi * ch                                        # aero load weight
        Wxi = wp.float64(0.5) * xi * xi - one_mt * xi * xi * xi / wp.float64(3.0)
        mg = W1 - Wxi                                       # root-biased bending weight
        wt = wp.float64(1.0)
        if g == 0 or g == NG - 1:
            wt = wp.float64(0.5)                            # trapezoid endpoints
        sum_wc += wt * (w / sg)
        sum_mgs += wt * mg * sg
    C = sum_wc / norm_w                                     # tip-biased compliance (numerator)
    s_root = sum_mgs / norm_mg                              # root-biased stiffness (numerator)
    # over-flex penalty 3·max(0,C-2): the relu mask (pen_mask[e]=1 iff C>2) is a DETACHED
    # input (precomputed host-side from C). Warp severs C's adjoint through any data-dependent
    # gate (if/wp.max/wp.clamp/wp.step on C), so we keep pen LINEAR in C with a constant mask
    # -> ∂pen/∂C = 3·mask is exactly the relu subgradient (PyTorch-style saved mask).
    pen = wp.float64(3.0) * pen_mask[e] * (C - wp.float64(2.0))
    efficiency[e] = wp.float64(22.0) + wp.float64(2.2) * (s_root - wp.float64(0.5)) - pen
    s_root_o[e] = s_root
    c_o[e] = C


@wp.kernel
def design_agg_gust(ctrl: wp.array2d(dtype=wp.float64), K: int, NG: int,
                    taper: wp.float64, norm_w: wp.float64,
                    gust_factor: wp.array(dtype=wp.float64),
                    ctrl_factor: wp.array(dtype=wp.float64),
                    s_gust_o: wp.array(dtype=wp.float64),
                    c_o: wp.array(dtype=wp.float64)):
    e = wp.tid()
    dx = wp.float64(1.0) / wp.float64(NG - 1)
    one_mt = wp.float64(1.0) - taper
    sum_wc = wp.float64(0.0)
    for g in range(NG):
        xi = wp.float64(g) * dx
        sg = interp_ctrl(ctrl, e, K, xi)
        w = xi * (wp.float64(1.0) - one_mt * xi)
        wt = wp.float64(1.0)
        if g == 0 or g == NG - 1:
            wt = wp.float64(0.5)
        sum_wc += wt * (w / sg)
    s_gust = norm_w / sum_wc                                # = 1/C (tip-biased stiffness)
    gust_factor[e] = wp.float64(1.0) / (wp.float64(0.6) + wp.float64(0.4) * s_gust)
    ctrl_factor[e] = wp.float64(1.6) - wp.float64(0.5) * (s_gust - wp.float64(0.5))
    s_gust_o[e] = s_gust
    c_o[e] = sum_wc / norm_w                                # compliance C (for the relu mask)


# ───────────────────────── batched 6-DOF flight step ─────────────────────────
@wp.kernel
def step_kernel(act: wp.array2d(dtype=wp.float64),
                x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                om: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.quatd),
                tt: wp.array(dtype=wp.float64), stepi: wp.array(dtype=wp.int32),
                gust_factor: wp.array(dtype=wp.float64),
                ctrl_factor: wp.array(dtype=wp.float64),
                a_tab: wp.array(dtype=wp.float64), cl_tab: wp.array(dtype=wp.float64),
                cd_tab: wp.array(dtype=wp.float64), ntab: int,
                S: wp.float64, att: wp.float64, dt: wp.float64, m: wp.float64,
                Vc: wp.float64, alt0: wp.float64, Tmax: wp.float64, Mmax: wp.float64,
                gw: wp.float64, gt0: wp.float64, gdur: wp.float64,
                Ix: wp.float64, Iy: wp.float64, Iz: wp.float64, horizon: int,
                obs: wp.array2d(dtype=wp.float64), reward: wp.array(dtype=wp.float64),
                done: wp.array(dtype=wp.int32)):
    i = wp.tid()
    # --- actions (clip to [-1,1] AFTER design control-authority scaling) ---
    a0 = wp.clamp(act[i, 0], wp.float64(-1.0), wp.float64(1.0))
    a1 = wp.clamp(act[i, 1] * ctrl_factor[i], wp.float64(-1.0), wp.float64(1.0))
    a2 = wp.clamp(act[i, 2] * ctrl_factor[i], wp.float64(-1.0), wp.float64(1.0))
    a3 = wp.clamp(act[i, 3], wp.float64(-1.0), wp.float64(1.0))
    wing_aoa = a0 * att
    roll_m = a1 * Mmax
    yaw_m = a2 * Mmax
    thr = wp.float64(0.5) * (a3 + wp.float64(1.0))

    # --- gust (design-scaled 1-cosine) ---
    t = tt[i]
    gust = wp.float64(0.0)
    if t >= gt0 and t < gt0 + gdur:
        fr = (t - gt0) / gdur
        gust = wp.float64(0.5) * gw * (wp.float64(1.0) - wp.cos(wp.float64(_TWO_PI) * fr))
    gust = gust * gust_factor[i]

    # --- rotation matrix from quaternion (x,y,z,w) ---
    qq = q[i]
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

    vv = v[i]
    # relative wind (body) = R^T (Vg - v), Vg = (0,0,gust)
    dwx = wp.float64(0.0) - vv[0]
    dwy = wp.float64(0.0) - vv[1]
    dwz = gust - vv[2]
    vrx = r00 * dwx + r10 * dwy + r20 * dwz      # R^T row = R column
    vry = r01 * dwx + r11 * dwy + r21 * dwz
    vrz = r02 * dwx + r12 * dwy + r22 * dwz
    sp = wp.sqrt(vrx * vrx + vry * vry + vrz * vrz) + wp.float64(1e-9)
    q_dyn = wp.float64(0.5) * wp.float64(1.225) * sp * sp

    # aero coeffs (feather |aoa| to attached limit; sign by aoa)
    aa = wp.abs(wing_aoa)
    if aa > att:
        aa = att
    sgn = wp.float64(0.0)
    if wing_aoa > wp.float64(0.0):
        sgn = wp.float64(1.0)
    if wing_aoa < wp.float64(0.0):
        sgn = wp.float64(-1.0)
    cl = interp1(aa, a_tab, cl_tab, ntab) * sgn
    cd = interp1(aa, a_tab, cd_tab, ntab)

    # lift direction = z_world projected orthogonal to relative-wind unit
    ux = vrx / sp; uy = vry / sp; uz = vrz / sp
    d = uz                                        # dot([0,0,1], uhat)
    lx = wp.float64(0.0) - d * ux
    ly = wp.float64(0.0) - d * uy
    lz = wp.float64(1.0) - d * uz
    ln = wp.sqrt(lx * lx + ly * ly + lz * lz)
    if ln > wp.float64(1e-6):
        lx = lx / ln; ly = ly / ln; lz = lz / ln
    else:
        lx = wp.float64(0.0); ly = wp.float64(0.0); lz = wp.float64(1.0)

    qs = q_dyn * S
    fbx = qs * (cl * lx + cd * ux) + thr * Tmax
    fby = qs * (cl * ly + cd * uy)
    fbz = qs * (cl * lz + cd * uz)
    # F_world = R F_body + gravity
    fwx = r00 * fbx + r01 * fby + r02 * fbz
    fwy = r10 * fbx + r11 * fby + r12 * fbz
    fwz = r20 * fbx + r21 * fby + r22 * fbz - m * wp.float64(G)
    # M_world = R [roll_m,0,yaw_m] - 0.05 om
    omv = om[i]
    mwx = r00 * roll_m + r02 * yaw_m - wp.float64(0.05) * omv[0]
    mwy = r10 * roll_m + r12 * yaw_m - wp.float64(0.05) * omv[1]
    mwz = r20 * roll_m + r22 * yaw_m - wp.float64(0.05) * omv[2]

    # integrate (semi-implicit, matches numpy order)
    nvx = vv[0] + fwx / m * dt
    nvy = vv[1] + fwy / m * dt
    nvz = vv[2] + fwz / m * dt
    nx = x[i][0] + nvx * dt
    ny = x[i][1] + nvy * dt
    nz = x[i][2] + nvz * dt
    # om_dot = Iinv (M - om x (I om)), I diagonal
    Iox = Ix * omv[0]; Ioy = Iy * omv[1]; Ioz = Iz * omv[2]
    cx = omv[1] * Ioz - omv[2] * Ioy
    cy = omv[2] * Iox - omv[0] * Ioz
    cz = omv[0] * Ioy - omv[1] * Iox
    nomx = omv[0] + (mwx - cx) / Ix * dt
    nomy = omv[1] + (mwy - cy) / Iy * dt
    nomz = omv[2] + (mwz - cz) / Iz * dt
    # quaternion integrate with world omega (matches numpy _quat_integrate)
    dqx = wp.float64(0.5) * (nomx * qw + nomy * qz - nomz * qy)
    dqy = wp.float64(0.5) * (nomy * qw + nomz * qx - nomx * qz)
    dqz = wp.float64(0.5) * (nomz * qw + nomx * qy - nomy * qx)
    dqw = wp.float64(0.5) * (-(nomx * qx + nomy * qy + nomz * qz))
    nqx = qx + dqx * dt; nqy = qy + dqy * dt
    nqz = qz + dqz * dt; nqw = qw + dqw * dt
    qn = wp.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw) + wp.float64(1e-12)
    nqx = nqx / qn; nqy = nqy / qn; nqz = nqz / qn; nqw = nqw / qn

    # write back state
    v[i] = wp.vec3d(nvx, nvy, nvz)
    x[i] = wp.vec3d(nx, ny, nz)
    om[i] = wp.vec3d(nomx, nomy, nomz)
    q[i] = wp.quatd(nqx, nqy, nqz, nqw)
    tt[i] = t + dt
    stepi[i] = stepi[i] + 1

    # observation (euler pitch/roll from new quaternion)
    roll = wp.atan2(wp.float64(2.0) * (nqw * nqx + nqy * nqz),
                    wp.float64(1.0) - wp.float64(2.0) * (nqx * nqx + nqy * nqy))
    sarg = wp.float64(2.0) * (nqw * nqy - nqz * nqx)
    sarg = wp.clamp(sarg, wp.float64(-1.0), wp.float64(1.0))
    pitch = wp.asin(sarg)
    obs[i, 0] = pitch; obs[i, 1] = roll
    obs[i, 2] = nomx; obs[i, 3] = nomy; obs[i, 4] = nomz
    obs[i, 5] = nvx - Vc; obs[i, 6] = nvy; obs[i, 7] = nvz
    obs[i, 8] = nz - alt0
    # obs gust = next-step gust value (matches numpy _obs() called after t+=dt)
    tn = t + dt
    gn = wp.float64(0.0)
    if tn >= gt0 and tn < gt0 + gdur:
        fr2 = (tn - gt0) / gdur
        gn = wp.float64(0.5) * gw * (wp.float64(1.0) - wp.cos(wp.float64(_TWO_PI) * fr2))
    obs[i, 9] = gn * gust_factor[i]

    # reward + termination
    zc = wp.clamp(nz - alt0, wp.float64(-15.0), wp.float64(15.0))
    vc = wp.clamp(nvx - Vc, wp.float64(-10.0), wp.float64(10.0))
    wc = wp.clamp(nvz, wp.float64(-10.0), wp.float64(10.0))
    r = wp.float64(1.0) - wp.float64(0.04) * zc * zc - wp.float64(0.05) * vc * vc
    r = r - wp.float64(0.3) * wc * wc - wp.float64(0.2) * roll * roll - wp.float64(0.1) * nvy * nvy
    r = r - wp.float64(0.02) * (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    finite = (nz == nz) and (roll == roll)        # NaN check
    crashed = (not finite) or nz < wp.float64(5.0) or nz > wp.float64(60.0) or wp.abs(roll) > wp.float64(1.4)
    if crashed:
        reward[i] = wp.float64(-20.0)
        done[i] = 1
    else:
        reward[i] = wp.clamp(r, wp.float64(-20.0), wp.float64(2.0))
        if stepi[i] >= horizon:
            done[i] = 1
        else:
            done[i] = 0


class GpuFlightEnv:
    """Batched B-environment Warp flight env (drop-in dynamics for the numpy MetaFlightEnv,
    parallel on GPU + differentiable, fp64)."""

    def __init__(self, B=1024, seed=0, device="cuda", horizon=400, dt=0.01, m=0.45,
                 V_cruise=8.0, alt=30.0, body_aoa_deg=45.0, gust_w=2.5, gust_t0=1.0,
                 gust_dur=0.5, K=K_CTRL):
        self.B, self.K, self.device = B, K, device
        self.horizon, self.dt, self.m = horizon, dt, m
        self.V_cruise, self.alt0, self.body_aoa = V_cruise, alt, np.deg2rad(body_aoa_deg)
        self.gust = dict(w=gust_w, t0=gust_t0, dur=gust_dur)
        self.T_max, self.M_max = 6.0, 0.4
        self.rng = np.random.default_rng(seed)
        aero = AeroDB()
        self.S, self.att = aero.S, aero.att
        self.Ix, self.Iy, self.Iz = 3e-3, 5e-3, 5e-3
        # device arrays (fp64)
        f64 = wp.float64
        self.a_tab = wp.array(aero._a.astype(np.float64), dtype=f64, device=device)
        self.cl_tab = wp.array(aero._cl.astype(np.float64), dtype=f64, device=device)
        self.cd_tab = wp.array(aero._cd.astype(np.float64), dtype=f64, device=device)
        self.ntab = len(aero._a)
        self.x = wp.zeros(B, dtype=wp.vec3d, device=device)
        self.v = wp.zeros(B, dtype=wp.vec3d, device=device)
        self.om = wp.zeros(B, dtype=wp.vec3d, device=device)
        self.q = wp.zeros(B, dtype=wp.quatd, device=device)
        self.tt = wp.zeros(B, dtype=f64, device=device)
        self.stepi = wp.zeros(B, dtype=wp.int32, device=device)
        self.ctrl = wp.zeros((B, K), dtype=f64, device=device, requires_grad=True)
        self.gust_factor = wp.zeros(B, dtype=f64, device=device, requires_grad=True)
        self.ctrl_factor = wp.zeros(B, dtype=f64, device=device, requires_grad=True)
        self.efficiency = wp.zeros(B, dtype=f64, device=device, requires_grad=True)
        self.s_gust = wp.zeros(B, dtype=f64, device=device)
        self.s_root = wp.zeros(B, dtype=f64, device=device)
        self.C = wp.zeros(B, dtype=f64, device=device)
        self.pen_mask = wp.zeros(B, dtype=f64, device=device)
        self.obs = wp.zeros((B, OBS_DIM), dtype=f64, device=device)
        self.reward = wp.zeros(B, dtype=f64, device=device)
        self.done = wp.zeros(B, dtype=wp.int32, device=device)

    def set_designs(self, ctrl):
        """ctrl: (B,K) stiffness control points (root->tip). Launches the differentiable
        aggregate kernel -> gust_factor/ctrl_factor/efficiency on GPU."""
        self.ctrl.assign(np.ascontiguousarray(ctrl, dtype=np.float64))
        # gust pass first — it yields the compliance C used to build the detached relu mask
        wp.launch(design_agg_gust, dim=self.B,
                  inputs=[self.ctrl, self.K, _NG, np.float64(_TAPER), np.float64(_NORM_W)],
                  outputs=[self.gust_factor, self.ctrl_factor, self.s_gust, self.C],
                  device=self.device)
        self.pen_mask.assign((self.C.numpy() > 2.0).astype(np.float64))   # relu mask (detached)
        wp.launch(design_agg_eff, dim=self.B,
                  inputs=[self.ctrl, self.K, _NG, np.float64(_TAPER),
                          np.float64(_NORM_W), np.float64(_NORM_MG), self.pen_mask],
                  outputs=[self.efficiency, self.s_root, self.C], device=self.device)

    def sample_designs(self, lo=0.3, hi=2.5):
        c = np.exp(self.rng.uniform(np.log(lo), np.log(hi), size=(self.B, self.K)))
        self.set_designs(c)
        return c

    def reset(self, designs=None):
        if designs is None:
            self.sample_designs()
        else:
            d = np.asarray(designs, np.float64)
            if d.ndim == 1:                       # scalar-per-env or single field broadcast
                d = np.broadcast_to(d, (self.B, self.K))
            self.set_designs(d)
        a = self.body_aoa
        x0 = np.tile([0.0, 0.0, self.alt0], (self.B, 1)).astype(np.float64)
        q0 = np.tile([0.0, -np.sin(a / 2), 0.0, np.cos(a / 2)], (self.B, 1)).astype(np.float64)
        v0 = np.tile([self.V_cruise, 0.0, 0.0], (self.B, 1)).astype(np.float64)
        self.x.assign(x0); self.q.assign(q0); self.v.assign(v0)
        self.om.zero_(); self.tt.zero_(); self.stepi.zero_(); self.done.zero_()
        return self.observe()

    def observe(self):
        return self.obs.numpy()

    def step(self, actions):
        act = wp.array(np.ascontiguousarray(actions, np.float64), dtype=wp.float64,
                       device=self.device)
        wp.launch(step_kernel, dim=self.B,
                  inputs=[act, self.x, self.v, self.om, self.q, self.tt, self.stepi,
                          self.gust_factor, self.ctrl_factor,
                          self.a_tab, self.cl_tab, self.cd_tab, self.ntab,
                          np.float64(self.S), np.float64(self.att), np.float64(self.dt),
                          np.float64(self.m), np.float64(self.V_cruise),
                          np.float64(self.alt0), np.float64(self.T_max),
                          np.float64(self.M_max), np.float64(self.gust["w"]),
                          np.float64(self.gust["t0"]), np.float64(self.gust["dur"]),
                          np.float64(self.Ix), np.float64(self.Iy), np.float64(self.Iz),
                          self.horizon],
                  outputs=[self.obs, self.reward, self.done], device=self.device)
        return self.obs.numpy(), self.reward.numpy(), self.done.numpy().astype(bool)


if __name__ == "__main__":
    import time
    wp.init()
    B = 4096
    env = GpuFlightEnv(B=B)
    env.reset()
    a = np.zeros((B, ACT_DIM), np.float64)
    wp.synchronize()
    t0 = time.time(); n = 400
    for _ in range(n):
        env.step(a)
    wp.synchronize()
    dt = time.time() - t0
    print(f"GPU batched flight env (fp64): B={B}, {n} steps in {dt:.3f}s -> "
          f"{B * n / dt / 1e6:.1f}M env-steps/s ({1e9 * dt / (B * n):.1f} ns/env-step)")
    print(f"  aggregates: s_gust[0]={env.s_gust.numpy()[0]:.3f} "
          f"efficiency[0]={env.efficiency.numpy()[0]:.2f}")
