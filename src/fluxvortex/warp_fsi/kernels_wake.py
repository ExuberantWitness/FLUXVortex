"""Batched GPU wake: fixed-capacity ring-vortex wake with shed/advect/truncate.

The wake is the O(N²) N-body part of UVLM — the highest-leverage GPU target
(wake-corner advection: every wake corner is induced by every bound + wake
ring). Reuses `device.ring_vel` for all induction.

State (per env, fixed capacity MAXW rows × ns rings, row-major oldest-first):
  wcorners : (B, MAXW*ns, 4) VEC3   ring corners [Fr, Fl, Bl, Br]
  wgamma   : (B, MAXW*ns)    DTYPE  ring circulation (0 = inactive/truncated)
  n_rows   : host int (lockstep across envs)

Advection is FROZEN-SNAPSHOT forward Euler: all corner velocities are evaluated
from the pre-step snapshot, then applied together. (MATLAB generate_wake.m is
frozen-snapshot RK4; the CPU standalone's in-place sequential Euler is its own
deviation — CPU grows a matching `advect_frozen` flag for validation.)
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config
from .device import ring_vel

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def shed_kernel(corners: wp.array(dtype=VEC3, ndim=3),    # (B, P, 4) bound rings
                gamma_shed: wp.array(dtype=DTYPE, ndim=2),  # (B, P) source circ (prev solve)
                vinf_dt: VEC3,                            # V_inf * d_t_wake
                nc: int, ns: int, slot: int,              # slot = row index to write
                wcorners: wp.array(dtype=VEC3, ndim=3),   # (B, MAXW*ns, 4) out
                wgamma: wp.array(dtype=DTYPE, ndim=2)):   # (B, MAXW*ns) out
    """New TE-attached wake row: front edge pinned at TE, back edge one full
    convection step downstream (CPU shed_wake / MATLAB generate_wake)."""
    e, j = wp.tid()
    p_te = (nc - 1) * ns + j          # TE panel index
    Fr = corners[e, p_te, 0]
    Fl = corners[e, p_te, 1]
    k = slot * ns + j
    wcorners[e, k, 0] = Fr
    wcorners[e, k, 1] = Fl
    wcorners[e, k, 2] = Fl + vinf_dt
    wcorners[e, k, 3] = Fr + vinf_dt
    wgamma[e, k] = gamma_shed[e, p_te]


@wp.kernel
def wake_corner_vel_bound_kernel(
        wcorners: wp.array(dtype=VEC3, ndim=3),   # (B, NW, 4) snapshot
        wgamma: wp.array(dtype=DTYPE, ndim=2),    # (B, NW) (skip inactive targets)
        bcorners: wp.array(dtype=VEC3, ndim=3),   # (B, P, 4) bound rings
        bgamma: wp.array(dtype=DTYPE, ndim=2),    # (B, P)
        eps2: DTYPE,
        V: wp.array(dtype=VEC3, ndim=3)):         # (B, NW, 4) accumulated
    """Bound-ring induction at every wake corner (no exclusion)."""
    e, t, j = wp.tid()        # t = target ring*; j = bound source
    r = t / 4
    kv = t - r * 4
    if wp.abs(wgamma[e, r]) < DTYPE(1.0e-15):
        return
    g = bgamma[e, j]
    if wp.abs(g) < DTYPE(1.0e-15):
        return
    p = wcorners[e, r, kv]
    v = ring_vel(p, bcorners[e, j, 0], bcorners[e, j, 1],
                 bcorners[e, j, 2], bcorners[e, j, 3], g, eps2)
    wp.atomic_add(V, e, r, kv, v)


@wp.kernel
def wake_corner_vel_wake_kernel(
        wcorners: wp.array(dtype=VEC3, ndim=3),   # (B, NW, 4) snapshot
        wgamma: wp.array(dtype=DTYPE, ndim=2),    # (B, NW)
        eps2: DTYPE,
        V: wp.array(dtype=VEC3, ndim=3)):         # (B, NW, 4) accumulated
    """Wake self-induction at wake corners, EXCLUDING own ring (CPU advect_wake
    skips w2==w, js2==js)."""
    e, t, j = wp.tid()        # t = target corner flat; j = wake source ring
    r = t / 4
    kv = t - r * 4
    if r == j:
        return
    if wp.abs(wgamma[e, r]) < DTYPE(1.0e-15):
        return
    g = wgamma[e, j]
    if wp.abs(g) < DTYPE(1.0e-15):
        return
    p = wcorners[e, r, kv]
    v = ring_vel(p, wcorners[e, j, 0], wcorners[e, j, 1],
                 wcorners[e, j, 2], wcorners[e, j, 3], g, eps2)
    wp.atomic_add(V, e, r, kv, v)


@wp.kernel
def wake_apply_euler_kernel(
        V: wp.array(dtype=VEC3, ndim=3),          # (B, NW, 4) induced
        wgamma: wp.array(dtype=DTYPE, ndim=2),    # (B, NW)
        vinf: VEC3, dt: DTYPE,
        wcorners: wp.array(dtype=VEC3, ndim=3)):  # (B, NW, 4) in/out
    e, t = wp.tid()
    r = t / 4
    kv = t - r * 4
    if wp.abs(wgamma[e, r]) < DTYPE(1.0e-15):
        return
    wcorners[e, r, kv] = wcorners[e, r, kv] + (vinf + V[e, r, kv]) * dt


@wp.kernel
def truncate_kernel(wcorners: wp.array(dtype=VEC3, ndim=3),  # (B, NW, 4)
                    max_x: DTYPE,
                    wgamma: wp.array(dtype=DTYPE, ndim=2)):  # (B, NW) in/out
    """Deactivate rings whose corner-mean x exceeds max_x (CPU truncate_wake
    removes by row centroid; per-ring mean x is equivalent for row-uniform
    convection and keeps it a pure per-ring operation)."""
    e, r = wp.tid()
    if wp.abs(wgamma[e, r]) < DTYPE(1.0e-15):
        return
    cx = (wcorners[e, r, 0][0] + wcorners[e, r, 1][0]
          + wcorners[e, r, 2][0] + wcorners[e, r, 3][0]) * DTYPE(0.25)
    if cx > max_x:
        wgamma[e, r] = DTYPE(0.0)


@wp.kernel
def update_newest_gamma_kernel(gamma_prev: wp.array(dtype=DTYPE, ndim=2),  # (B, P)
                               nc: int, ns: int, slot: int,
                               wgamma: wp.array(dtype=DTYPE, ndim=2)):     # (B, MAXW*ns)
    """After the bound solve: newest wake row takes the PREVIOUS solve's bound
    TE circulation (CPU _uvlm_step: wake_gamma[-1] = gamma_prev[nc-1])."""
    e, j = wp.tid()
    wgamma[e, slot * ns + j] = gamma_prev[e, (nc - 1) * ns + j]


class GpuWake:
    """Fixed-capacity batched wake state + shed/advect/truncate operations."""

    def __init__(self, B, ns, max_rows, core_radius, V_inf, dt_wake,
                 truncation_x, device=None):
        device = device or config.DEVICE
        NP = config.NP_DTYPE
        self.device = device
        self.B, self.ns, self.MAXW = B, ns, max_rows
        self.NW = max_rows * ns
        self.eps2 = NP(core_radius * core_radius)
        self.vinf = VEC3(*np.asarray(V_inf, dtype=NP).tolist())
        self.vinf_dt = VEC3(*(np.asarray(V_inf, dtype=NP) * NP(dt_wake)).tolist())
        self.dt = NP(dt_wake)
        self.max_x = NP(truncation_x)
        self.wcorners = wp.zeros((B, self.NW, 4), dtype=VEC3, device=device)
        self.wgamma = wp.zeros((B, self.NW), dtype=DTYPE, device=device)
        self._V = wp.zeros((B, self.NW, 4), dtype=VEC3, device=device)
        self.n_rows = 0          # rows shed so far (host, lockstep)
        self.newest_slot = -1

    def advect(self, bound_corners, bound_gamma):
        """Frozen-snapshot Euler advection of all active wake corners."""
        if self.n_rows == 0:
            return
        B, NW = self.B, self.NW
        self._V.zero_()
        wp.launch(wake_corner_vel_bound_kernel, dim=(B, NW * 4, bound_gamma.shape[1]),
                  inputs=[self.wcorners, self.wgamma, bound_corners, bound_gamma,
                          DTYPE(self.eps2)],
                  outputs=[self._V], device=self.device)
        wp.launch(wake_corner_vel_wake_kernel, dim=(B, NW * 4, NW),
                  inputs=[self.wcorners, self.wgamma, DTYPE(self.eps2)],
                  outputs=[self._V], device=self.device)
        wp.launch(wake_apply_euler_kernel, dim=(B, NW * 4),
                  inputs=[self._V, self.wgamma, self.vinf, DTYPE(self.dt)],
                  outputs=[self.wcorners], device=self.device)

    def shed(self, bound_corners, gamma_shed, nc):
        """Shed a new TE row (delayed-Kutta source = previous solve's gamma)."""
        if self.n_rows >= self.MAXW:
            raise RuntimeError(f"wake capacity exceeded: MAXW={self.MAXW}")
        slot = self.n_rows
        wp.launch(shed_kernel, dim=(self.B, self.ns),
                  inputs=[bound_corners, gamma_shed, self.vinf_dt, nc, self.ns, slot],
                  outputs=[self.wcorners, self.wgamma], device=self.device)
        self.n_rows += 1
        self.newest_slot = slot

    def update_newest_gamma(self, gamma_prev, nc):
        if self.newest_slot < 0:
            return
        wp.launch(update_newest_gamma_kernel, dim=(self.B, self.ns),
                  inputs=[gamma_prev, nc, self.ns, self.newest_slot],
                  outputs=[self.wgamma], device=self.device)

    def truncate(self):
        if self.n_rows == 0:
            return
        wp.launch(truncate_kernel, dim=(self.B, self.NW),
                  inputs=[self.wcorners, DTYPE(self.max_x)],
                  outputs=[self.wgamma], device=self.device)
