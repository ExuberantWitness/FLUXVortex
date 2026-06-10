"""Batched UVLM Warp kernels (env dimension leading).

Phase 1 start: AIC assembly. Each thread = one (env, target_panel, source_panel)
triple. Reuses `device.ring_vel` (CPU-bit-exact Biot-Savart incl. -V sign).

Array layout (all device-resident, dtype from config):
  colloc  : (B, P)      VEC3   collocation points
  normals : (B, P)      VEC3   unit panel normals
  corners : (B, P, 4)   VEC3   ring corners [Fr, Fl, Bl, Br]
  AIC     : (B, P, P)   DTYPE  influence matrix  AIC[e,i,j] = ring_vel(rc_i, ring_j, 1)·n_i
"""
from __future__ import annotations
import warp as wp
from . import config
from .device import ring_vel, dt_ring_vel

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def aic_kernel(colloc: wp.array(dtype=VEC3, ndim=2),
               normals: wp.array(dtype=VEC3, ndim=2),
               corners: wp.array(dtype=VEC3, ndim=3),
               eps2: DTYPE,
               AIC: wp.array(dtype=DTYPE, ndim=3)):
    e, i, j = wp.tid()
    p = colloc[e, i]
    n = normals[e, i]
    v = ring_vel(p, corners[e, j, 0], corners[e, j, 1],
                 corners[e, j, 2], corners[e, j, 3],
                 DTYPE(1.0), eps2)
    AIC[e, i, j] = wp.dot(v, n)


def build_aic_batched(colloc_wp, normals_wp, corners_wp, core_radius,
                      out_aic=None, device=None):
    """Launch the batched AIC kernel.

    colloc_wp/normals_wp : wp.array (B, P) VEC3
    corners_wp           : wp.array (B, P, 4) VEC3
    Returns wp.array (B, P, P) DTYPE.
    """
    device = device or config.DEVICE
    B = colloc_wp.shape[0]
    P = colloc_wp.shape[1]
    if out_aic is None:
        out_aic = wp.zeros((B, P, P), dtype=DTYPE, device=device)
    eps2 = config.NP_DTYPE(core_radius * core_radius)
    wp.launch(aic_kernel, dim=(B, P, P),
              inputs=[colloc_wp, normals_wp, corners_wp, DTYPE(eps2)],
              outputs=[out_aic], device=device)
    return out_aic


@wp.kernel
def induction_kernel(targets: wp.array(dtype=VEC3, ndim=2),    # (B, M) field pts
                     corners: wp.array(dtype=VEC3, ndim=3),    # (B, S, 4) source rings
                     gamma: wp.array(dtype=DTYPE, ndim=2),     # (B, S) source circ
                     eps2: DTYPE,
                     V_out: wp.array(dtype=VEC3, ndim=2)):     # (B, M) accumulated
    """Σ_source ring_vel(target, ring_s, gamma_s) — matches CPU
    compute_bound_induction_at_colloc / compute_wake_velocity_at_colloc
    (ring_vel already carries the -V sign; result == Python V_bound/V_wake)."""
    e, i, j = wp.tid()      # env, target, source
    g = gamma[e, j]
    if wp.abs(g) < DTYPE(1.0e-15):
        return
    p = targets[e, i]
    v = ring_vel(p, corners[e, j, 0], corners[e, j, 1],
                 corners[e, j, 2], corners[e, j, 3], g, eps2)
    wp.atomic_add(V_out, e, i, v)


@wp.kernel
def dp_lift1_kernel(gamma: wp.array(dtype=DTYPE, ndim=3),     # (B, nc, ns) bound circ
                    corners: wp.array(dtype=VEC3, ndim=4),    # (B, nc, ns, 4)
                    V_ext: wp.array(dtype=VEC3, ndim=3),      # (B, nc, ns) V_wake+V_gamma
                    V_inf: VEC3,
                    rho: DTYPE,
                    ns: int,
                    dp: wp.array(dtype=DTYPE, ndim=3)):       # (B, nc, ns) out
    """Steady Bernoulli pressure dp_lift1 = ρ·V_surf1·(τx·dΓ/dx + τy·dΓ/dy).
    Mirrors standalone_uvlm.compute_forces (dp_no_vstruct, no Mf2/unsteady):
    chordwise backward diff, spanwise zero-padded central diff (MATLAB convention).
    V_surf1 = V_inf + V_ext."""
    e, i, j = wp.tid()
    c0 = corners[e, i, j, 0]; c1 = corners[e, i, j, 1]
    c2 = corners[e, i, j, 2]; c3 = corners[e, i, j, 3]
    tx = (c1 - c0 + c2 - c3) * DTYPE(0.5)   # (r21+r34)/2
    ty = (c0 - c3 + c1 - c2) * DTYPE(0.5)   # (r14+r23)/2
    dxn = wp.length(tx) + DTYPE(1.0e-15)
    dyn = wp.length(ty) + DTYPE(1.0e-15)
    txh = tx / dxn
    tyh = ty / dyn

    g = gamma[e, i, j]
    # chordwise dΓ/dx
    if i == 0:
        dgx = g / dxn
    else:
        dgx = (g - gamma[e, i - 1, j]) / dxn
    # spanwise dΓ/dy (zero-padded central)
    dgy = DTYPE(0.0)
    if ns > 1:
        if j == 0:
            dgy = g / dyn
        elif j == ns - 1:
            dgy = -g / dyn
        else:
            dgy = (gamma[e, i, j + 1] - gamma[e, i, j - 1]) / (DTYPE(2.0) * dyn)

    Vs = V_inf + V_ext[e, i, j]
    dp[e, i, j] = rho * (wp.dot(Vs, txh) * dgx + wp.dot(Vs, tyh) * dgy)


def compute_dp_lift1_batched(gamma_wp, corners_wp, V_ext_wp, V_inf_vec, rho,
                             out_dp=None, device=None):
    """dp_lift1 per panel. gamma_wp (B,nc,ns); corners_wp (B,nc,ns,4) VEC3;
    V_ext_wp (B,nc,ns) VEC3. Returns (B,nc,ns) DTYPE."""
    device = device or config.DEVICE
    B, nc, ns = gamma_wp.shape
    if out_dp is None:
        out_dp = wp.zeros((B, nc, ns), dtype=DTYPE, device=device)
    vinf = VEC3(config.NP_DTYPE(V_inf_vec[0]), config.NP_DTYPE(V_inf_vec[1]),
                config.NP_DTYPE(V_inf_vec[2]))
    wp.launch(dp_lift1_kernel, dim=(B, nc, ns),
              inputs=[gamma_wp, corners_wp, V_ext_wp, vinf, DTYPE(config.NP_DTYPE(rho)), ns],
              outputs=[out_dp], device=device)
    return out_dp


@wp.kernel
def mf2_scalar_kernel(colloc: wp.array(dtype=VEC3, ndim=2),       # (B, P)
                      normals: wp.array(dtype=VEC3, ndim=2),      # (B, P)
                      dt_colloc: wp.array(dtype=VEC3, ndim=2),    # (B, P) dt_rc
                      wcorn: wp.array(dtype=VEC3, ndim=3),        # (B, Sw, 4)
                      wdt: wp.array(dtype=VEC3, ndim=3),          # (B, Sw, 4)
                      wgamma: wp.array(dtype=DTYPE, ndim=2),      # (B, Sw)
                      eps2: DTYPE,
                      scal: wp.array(dtype=DTYPE, ndim=2)):       # (B, P) out
    """Gamma_wake_dt_q1234_n[e,i] = Σ_wake (Γ_w dt_ring·n_i). Mirrors CPU
    compute_mf2_vec1 inner accumulation (then Mf2_vec1 = AIC⁻¹·(-scal))."""
    e, i, j = wp.tid()
    g = wgamma[e, j]
    if wp.abs(g) < DTYPE(1.0e-15):
        return
    p = colloc[e, i]
    dtp = dt_colloc[e, i]
    dtq = dt_ring_vel(p, wcorn[e, j, 0], wcorn[e, j, 1], wcorn[e, j, 2], wcorn[e, j, 3],
                      g, dtp, wdt[e, j, 0], wdt[e, j, 1], wdt[e, j, 2], wdt[e, j, 3], eps2)
    wp.atomic_add(scal, e, i, wp.dot(dtq, normals[e, i]))


def compute_mf2_scalar_batched(colloc_wp, normals_wp, dt_colloc_wp,
                               wcorn_wp, wdt_wp, wgamma_wp, core_radius,
                               out_scal=None, device=None):
    """Gamma_wake_dt_q1234_n per colloc point. Returns (B, P) DTYPE.
    Final Mf2_vec1 = AIC⁻¹ · (-scalar) (solve done by caller / Phase 1c kernel)."""
    device = device or config.DEVICE
    B, P = colloc_wp.shape
    Sw = wgamma_wp.shape[1]
    if out_scal is None:
        out_scal = wp.zeros((B, P), dtype=DTYPE, device=device)
    else:
        out_scal.zero_()
    eps2 = config.NP_DTYPE(core_radius * core_radius)
    wp.launch(mf2_scalar_kernel, dim=(B, P, Sw),
              inputs=[colloc_wp, normals_wp, dt_colloc_wp, wcorn_wp, wdt_wp,
                      wgamma_wp, DTYPE(eps2)],
              outputs=[out_scal], device=device)
    return out_scal


def induce_velocity_batched(targets_wp, src_corners_wp, src_gamma_wp,
                            core_radius, out_V=None, device=None):
    """Σ over source rings of ring-vortex velocity at each target point.

    targets_wp     : (B, M) VEC3
    src_corners_wp : (B, S, 4) VEC3
    src_gamma_wp   : (B, S) DTYPE
    Returns (B, M) VEC3 (Python sign convention = -V_matlab).
    """
    device = device or config.DEVICE
    B = targets_wp.shape[0]
    M = targets_wp.shape[1]
    S = src_gamma_wp.shape[1]
    if out_V is None:
        out_V = wp.zeros((B, M), dtype=VEC3, device=device)
    else:
        out_V.zero_()
    eps2 = config.NP_DTYPE(core_radius * core_radius)
    wp.launch(induction_kernel, dim=(B, M, S),
              inputs=[targets_wp, src_corners_wp, src_gamma_wp, DTYPE(eps2)],
              outputs=[out_V], device=device)
    return out_V
