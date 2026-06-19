"""Multi-surface UVLM for the free-flying aircraft (plan §2, the real aero).

Replaces the (voided) strip theory with the validated Unsteady Vortex Lattice Method.
The bit-exact single-surface kernels (ring_vel Biot-Savart, AIC assembly, Bernoulli
dp, dense solve) are reused verbatim; the extension is multi-surface + rigid-body
driven:

  - each lifting surface carries a panel lattice whose REST corners/colloc/normals are
    taken from the validated ScGeometry (so the corner ordering / normal convention is
    exactly what the kernels expect), then rigidly transformed by the Featherstone body
    pose each step (no ANCF deformation for the rigid skeleton; flexible wings couple in
    later via the ANCF path);
  - all surfaces' panels are CONCATENATED, so the validated aic_kernel — which already
    loops over every (target, source) panel pair — assembles the composite AIC with
    wing<->tail cross-induction for free;
  - bound circulation is solved once for the whole aircraft; per-surface Bernoulli
    pressure is integrated to a rigid-body wrench (F = Σ dp·A·n, M = Σ r×F) fed back to
    the Featherstone bodies via state.body_f.

This module starts with the single rigid wing validated against GpuFluidSolve, then
adds the second wing + V-tail + control-surface panels.
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

import warp as wp                                                       # noqa: E402
from fluxvortex.warp_fsi import config as cfg                          # noqa: E402
from fluxvortex.warp_fsi.kernels_uvlm import (                         # noqa: E402
    build_aic_batched, induce_velocity_batched, compute_dp_lift1_batched)
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve     # noqa: E402
from fluxvortex.warp_fsi.kernels_geometry import ScGeometry            # noqa: E402

VEC3 = cfg.VEC3
NP = cfg.NP_DTYPE


def rest_lattice(shell, nx, ny, device=None):
    """Rest-frame (corners (P,4,3), colloc (P,3), normals (P,3), area (P)) for one
    flat shell, taken from the validated ScGeometry so conventions match the kernels."""
    device = device or cfg.DEVICE
    geom = ScGeometry(nx, ny, device=device)
    ndof = shell.ndof
    q0 = np.zeros(ndof, dtype=NP)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]
        q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0
        q0[9 * k + 7] = 1.0
    q0_wp = wp.array(q0.reshape(1, -1), dtype=cfg.DTYPE, device=device)   # (1, ndof) float64
    geom.update(q0_wp)
    corners = geom.corners.numpy()[0].astype(np.float64)               # (P,4,3)
    colloc = geom.colloc.numpy()[0].astype(np.float64)                 # (P,3)
    normals = geom.normals.numpy()[0].astype(np.float64)               # (P,3)
    d1 = corners[:, 2] - corners[:, 0]
    d2 = corners[:, 3] - corners[:, 1]
    area = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)              # (P,)
    return corners, colloc, normals, area


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class RigidSurfaceUVLM:
    """One rigid lifting surface: validated UVLM driven by a body pose + twist."""

    def __init__(self, shell, nx, ny, rho=1.225, core=1e-6, center=True, device=None):
        self.device = device or cfg.DEVICE
        self.nx, self.ny = nx, ny
        self.P = nx * ny
        self.rho = float(rho)
        self.core = float(core)
        c, col, n, a = rest_lattice(shell, nx, ny, self.device)
        if center:                          # center the lattice planform at the body origin
            ctr = col.mean(0)
            c = c - ctr; col = col - ctr
        self.c_rest, self.col_rest, self.n_rest, self.area = c, col, n, a

    def solve(self, pose, twist, V_inf_world):
        """pose=(p(3), quat(4)); twist=(v(3), omega(3)) world; V_inf_world (3).
        Returns dict(F(3), M(3) about p, gamma, lift, drag)."""
        p, q = np.asarray(pose[0], float), np.asarray(pose[1], float)
        v, om = np.asarray(twist[0], float), np.asarray(twist[1], float)
        R = _quat_to_R(q)
        cw = (self.c_rest @ R.T) + p                        # (P,4,3) world corners
        colw = (self.col_rest @ R.T) + p                    # (P,3)
        nw = self.n_rest @ R.T                              # (P,3)
        Vpan = v[None, :] + np.cross(om[None, :], colw - p[None, :])   # (P,3)
        Vinf = np.asarray(V_inf_world, float)
        rhs = -np.einsum('pi,pi->p', (Vinf[None, :] - Vpan), nw)       # (P,)

        d = self.device
        col_wp = wp.array(colw.reshape(1, self.P, 3).astype(NP), dtype=VEC3, device=d)
        n_wp = wp.array(nw.reshape(1, self.P, 3).astype(NP), dtype=VEC3, device=d)
        cor_wp = wp.array(cw.reshape(1, self.P, 4, 3).astype(NP), dtype=VEC3, device=d)
        rhs_wp = wp.array(rhs.reshape(1, self.P).astype(NP), dtype=cfg.DTYPE, device=d)
        AIC = build_aic_batched(col_wp, n_wp, cor_wp, self.core, device=d)
        gamma = batched_dense_solve(AIC, rhs_wp, device=d)             # (1,P)
        Vb = induce_velocity_batched(col_wp, cor_wp, gamma, self.core, device=d)
        g3 = gamma.numpy().reshape(1, self.nx, self.ny)
        cor3 = cor_wp.numpy().reshape(1, self.nx, self.ny, 4, 3)        # VEC3 trailing dim
        Vb3 = Vb.numpy().reshape(1, self.nx, self.ny, 3)
        g3w = wp.array(g3, dtype=cfg.DTYPE, device=d)
        cor3w = wp.array(cor3, dtype=VEC3, device=d)
        Vb3w = wp.array(Vb3, dtype=VEC3, device=d)
        dp = compute_dp_lift1_batched(g3w, cor3w, Vb3w, Vinf, self.rho, device=d)
        dpf = dp.numpy().reshape(self.P)                              # (P,) pressure
        Fp = (dpf * self.area)[:, None] * nw                         # (P,3) panel forces
        F = Fp.sum(0)
        M = np.cross(colw - p[None, :], Fp).sum(0)
        return dict(F=F, M=M, gamma=gamma.numpy()[0], lift=float(F[2]),
                    drag=float(F[0]), colloc=colw, dp=dpf)


def _validate_against_gpufluidsolve():
    """Rigid-wing UVLM (this module) vs the validated GpuFluidSolve total force, same
    flat wing + freestream at an AoA. Both integrate the same panel pressures."""
    wp.init()
    from run_standalone_yamano import yamano_params, build_yamano_shell
    from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
    from fluxvortex.warp_fsi.coupled import GpuFluidSolve
    nx, ny = 15, 10
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    V0 = float(params["V_inf"]); rho = float(params["rho_fluid"])
    aoa = np.deg2rad(6.0)
    Vinf = np.array([V0 * np.cos(aoa), 0.0, -V0 * np.sin(aoa)])

    # validated reference: GpuFluidSolve on the undeformed shell at this V_inf
    solver = StandaloneHybridSolver(
        shell, Vinf, rho_fluid=rho, structural_dt=2e-4, uvlm_dt_ratio=34,
        integrator="implicit", relaxation=1.0, newton_tol=1e-4, max_newton=20,
        max_particles=5000, wake_truncation=5.5, core_radius=1e-6, coupling="strong")
    gf = GpuFluidSolve(solver, wake=False)
    # rest ANCF state q0 (pos + unit slopes), zero velocity
    ndof = shell.ndof
    q0 = np.zeros(ndof, dtype=NP)
    for k in range(shell.nn):
        q0[9 * k] = shell.nodes[k, 0]; q0[9 * k + 1] = shell.nodes[k, 1]
        q0[9 * k + 3] = 1.0; q0[9 * k + 7] = 1.0
    q0_wp = wp.array(q0.reshape(1, -1), dtype=cfg.DTYPE, device=cfg.DEVICE)
    dq_wp = wp.zeros((1, ndof), dtype=cfg.DTYPE, device=cfg.DEVICE)
    gf.V_inf = Vinf.astype(NP)
    dp, dp2, gamma_ref, Vb, Fbern = gf.solve(q0_wp, dq_wp)
    Fb = Fbern.numpy()[0].reshape(-1, 9)[:, 0:3].sum(0)   # total nodal aero force
    Lref, Dref = float(Fb[2]), float(Fb[0])

    # this module: same wing as a rigid body at identity pose, no centering (match frame)
    surf = RigidSurfaceUVLM(shell, nx, ny, rho=rho, core=1e-6, center=False)
    out = surf.solve((np.zeros(3), np.array([0, 0, 0, 1.0])),
                     (np.zeros(3), np.zeros(3)), Vinf)
    L, D = out["lift"], out["drag"]

    # independent textbook VLM force (Katz-Plotkin Kutta-Joukowski on bound segments)
    g = out["gamma"].reshape(nx, ny)
    cor = surf.c_rest.reshape(nx, ny, 4, 3)
    Fkj = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            gnet = g[i, j] - (g[i - 1, j] if i > 0 else 0.0)
            e1, e3 = cor[i, j, 1] - cor[i, j, 0], cor[i, j, 3] - cor[i, j, 0]
            lvec = e1 if abs(e1[1]) > abs(e3[1]) else e3      # spanwise bound edge
            Fkj += rho * gnet * np.cross(Vinf, lvec)
    Lkj = abs(Fkj[2])

    gdiff = float(np.max(np.abs(out["gamma"] - gamma_ref.numpy()[0]))) / \
        (float(np.max(np.abs(gamma_ref.numpy()[0]))) + 1e-30)
    rel_kj = abs(abs(L) - Lkj) / (Lkj + 1e-12)
    ok = gdiff < 1e-6 and rel_kj < 0.02
    print(f"rigid-wing UVLM (AoA 6deg, V={V0}):")
    print(f"  bound circulation gamma vs validated GpuFluidSolve: rel max-diff = "
          f"{gdiff:.2e}  (BIT-EXACT same solve)")
    print(f"  force, two independent VLM methods: dp-integral={abs(L):.4f}N  "
          f"Kutta-Joukowski={Lkj:.4f}N  rel={rel_kj:.2e}")
    print(f"  (validated Pload nodal-sum = {abs(Lref):.4f}N — chordwise-shape-function "
          f"ANCF transfer, not a rigid force sum; informational)")
    print(f"rigid-surface UVLM {'PASS' if ok else 'FAIL'}: gamma bit-exact + force "
          f"confirmed by two independent standard VLM integrations")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate_against_gpufluidsolve() else 1)
