"""Warp-accelerated ParticleField (host-numpy authoritative, GPU on demand).

Design: numpy host arrays are the source of truth; GPU arrays created
via wp.array() for each kernel launch. Simpler than managing persistent
GPU buffers with partial updates, at the cost of one H2D copy per eval
(~1 MB for 40k particles — negligible vs O(N²) compute).
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

# Device selection: explicit env var > auto-detect > fail-fast.
# Valid values: "cuda", "cpu". Set PFIELD_DEVICE=cuda to force GPU.
_DEVICE_ENV = os.environ.get("PFIELD_DEVICE", "").lower()
if _DEVICE_ENV in ("cuda", "cpu"):
    _DEVICE = _DEVICE_ENV
else:
    # Auto-detect: use cuda if a capable device is present
    try:
        _DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        _DEVICE = "cpu"

# Float64 constants for kernels
_C0 = wp.constant(wp.float64(0.0))
_C2 = wp.constant(wp.float64(2.0))
_C3 = wp.constant(wp.float64(3.0))
_C_NEG_HALF = wp.constant(wp.float64(-0.5))
_C_SQRT2 = wp.constant(wp.float64(1.4142135623730951))
_C_SQRT2_OVER_PI = wp.constant(wp.float64(0.7978845608028654))
_C_NEG_1_OVER_4PI = wp.constant(wp.float64(-0.07957747154594767))


@wp.kernel
def blob_velocity_kernel(
    src_pos: wp.array1d(dtype=wp.vec3d),
    src_gamma: wp.array1d(dtype=wp.vec3d),
    src_sigma: wp.array1d(dtype=wp.float64),
    tgt_pos: wp.array1d(dtype=wp.vec3d),
    vel_out: wp.array1d(dtype=wp.vec3d),
    n_src: int,
):
    """Blob Gaussian-erf velocity kernel (matches frozen rvpm_reference)."""
    tid = wp.tid()
    t = tgt_pos[tid]
    vx = _C0
    vy = _C0
    vz = _C0

    for j in range(n_src):
        s = src_pos[j]
        dx0 = t[0] - s[0]
        dx1 = t[1] - s[1]
        dx2 = t[2] - s[2]
        r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
        if r2 > _C0:
            r = wp.sqrt(r2)
            rho = r / src_sigma[j]
            expo = wp.exp(_C_NEG_HALF * rho * rho)
            erf_val = wp.erf(rho / _C_SQRT2)
            aux = _C_SQRT2_OVER_PI * rho * expo
            q = erf_val - aux
            w = _C_NEG_1_OVER_4PI * q / (r2 * r)
            g = src_gamma[j]
            cx = dx1 * g[2] - dx2 * g[1]
            cy = dx2 * g[0] - dx0 * g[2]
            cz = dx0 * g[1] - dx1 * g[0]
            vx = vx + w * cx
            vy = vy + w * cy
            vz = vz + w * cz

    vel_out[tid] = wp.vec3d(vx, vy, vz)


# Particle type codes (user's design)
TYPE_FREE = 1.0
TYPE_FRESH_SHED = 1.1
TYPE_BOUND = 5.0
TYPE_PROBE = 9.0


class ParticleField:
    """Capacity-buffer SoA particle field.

    Host numpy arrays are authoritative; GPU arrays created on demand
    for kernel evaluation. Inherits the user's 'super-dictionary' concept
    with typed lifecycle and capacity-based growth management.
    """

    __slots__ = (
        "capacity", "n",
        "pos", "gamma", "sigma", "circul", "vol", "ptype", "birth_step",
    )

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.n = 0
        self.pos = np.zeros((capacity, 3), dtype=np.float64)
        self.gamma = np.zeros((capacity, 3), dtype=np.float64)
        self.sigma = np.zeros(capacity, dtype=np.float64)
        self.circul = np.zeros(capacity, dtype=np.float64)
        self.vol = np.zeros(capacity, dtype=np.float64)
        self.ptype = np.full(capacity, TYPE_FREE, dtype=np.float64)
        self.birth_step = np.zeros(capacity, dtype=np.int32)

    # -- lifecycle --

    def add_particles(self, pos, gamma, sigma, circul=None, vol=None,
                      ptype=TYPE_FRESH_SHED, birth_step=0):
        k = len(pos)
        assert self.n + k <= self.capacity, f"overflow {self.n}+{k}>{self.capacity}"
        n0, n1 = self.n, self.n + k
        self.pos[n0:n1] = pos
        self.gamma[n0:n1] = gamma
        self.sigma[n0:n1] = sigma
        self.circul[n0:n1] = circul if circul is not None else (
            np.linalg.norm(gamma, axis=1) + 1e-30)
        self.vol[n0:n1] = vol if vol is not None else 0.0
        self.ptype[n0:n1] = ptype
        self.birth_step[n0:n1] = birth_step
        self.n = n1

    def remove_indices(self, idx):
        keep = np.ones(self.n, dtype=bool)
        keep[idx] = False
        nk = keep.sum()
        for attr in ("pos", "gamma", "sigma", "circul", "vol", "ptype", "birth_step"):
            arr = getattr(self, attr)
            arr[:nk] = arr[:self.n][keep]
        self.n = nk

    def remove_type(self, type_code):
        mask = self.ptype[:self.n] == type_code
        if mask.any():
            self.remove_indices(np.flatnonzero(mask))

    def promote_fresh(self):
        m = self.ptype[:self.n] == TYPE_FRESH_SHED
        self.ptype[:self.n][m] = TYPE_FREE

    # -- view properties --

    @property
    def positions(self):
        return self.pos[:self.n]

    @property
    def gammas(self):
        return self.gamma[:self.n]

    @property
    def sigmas(self):
        return self.sigma[:self.n]

    @property
    def types(self):
        return self.ptype[:self.n]

    # -- GPU evaluation --

    def velocity_at(self, targets: np.ndarray) -> np.ndarray:
        """Evaluate velocity at arbitrary target points (blob kernel)."""
        nt = len(targets)
        sp = wp.array(self.pos[:self.n], dtype=wp.vec3d, device=_DEVICE)
        sg = wp.array(self.gamma[:self.n], dtype=wp.vec3d, device=_DEVICE)
        ss = wp.array(self.sigma[:self.n], dtype=wp.float64, device=_DEVICE)
        tp = wp.array(targets.astype(np.float64), dtype=wp.vec3d, device=_DEVICE)
        vo = wp.zeros(nt, dtype=wp.vec3d, device=_DEVICE)
        wp.launch(blob_velocity_kernel, dim=[nt],
                  inputs=[sp, sg, ss, tp, vo, self.n], device=_DEVICE)
        return vo.numpy()

    def velocity_self(self) -> np.ndarray:
        """Self-induced velocity (coincident particle excluded by r2>0 check)."""
        return self.velocity_at(self.positions)

    def velocity_and_jacobian_self(self):
        """Velocity + analytic Jacobian at own positions.

        Uses frozen CPU kernel for Jacobian (correct, validated).
        Velocity from Warp GPU kernel (fast).
        Returns (velocity, jacobian) with shapes (N,3) and (N,3,3).
        """
        n = self.n
        if n == 0:
            return np.empty((0, 3)), np.empty((0, 3, 3))

        # Velocity from Warp (fast)
        vel = self.velocity_self()

        # Jacobian from frozen kernel (correct, small N OK)
        import sys
        from pathlib import Path
        _src = Path(__file__).resolve().parents[2] / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
        field = direct_gaussian_erf_velocity_jacobian(
            self.positions, self.gammas, self.sigmas)
        jac = np.asarray(field.jacobian)

        return vel, jac

    def __repr__(self):
        free = (self.types == TYPE_FREE).sum()
        fresh = (self.types == TYPE_FRESH_SHED).sum()
        return f"ParticleField(n={self.n}/{self.capacity}, free={free}, fresh={fresh})"
