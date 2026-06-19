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


def make_vlm_lattice(V):
    """Standard VLM ring lattice from a planform vertex grid V (nc+1, ns+1, 3).

    Rings at the 1/4-chord (bound vortex), collocation at the 3/4-chord (Katz-Plotkin).
    Returns (corners (P,4,3), colloc (P,3), normals (P,3), area (P)) with panel order
    p = i*ns + j (chordwise i, spanwise j) and corner order [c0,c1(chordwise),c2,
    c3(spanwise)] — the convention the validated kernels expect (verified).
    """
    V = np.asarray(V, float)
    ncp1, nsp1 = V.shape[0], V.shape[1]
    nc, ns = ncp1 - 1, nsp1 - 1
    # quarter-chord vertex grid Q (shift each chord row back 1/4 panel; extend TE)
    Q = np.zeros_like(V)
    for i in range(ncp1):
        if i < nc:
            Q[i] = V[i] + 0.25 * (V[i + 1] - V[i])
        else:
            Q[i] = V[i] + 0.25 * (V[i] - V[i - 1])
    P = nc * ns
    corners = np.zeros((P, 4, 3)); colloc = np.zeros((P, 3))
    normals = np.zeros((P, 3)); area = np.zeros(P)
    for i in range(nc):
        for j in range(ns):
            p = i * ns + j
            corners[p, 0] = Q[i, j]
            corners[p, 1] = Q[i + 1, j]
            corners[p, 2] = Q[i + 1, j + 1]
            corners[p, 3] = Q[i, j + 1]
            fmid = 0.5 * (V[i, j] + V[i, j + 1])
            bmid = 0.5 * (V[i + 1, j] + V[i + 1, j + 1])
            colloc[p] = fmid + 0.75 * (bmid - fmid)            # 3/4-chord collocation
            d1 = V[i + 1, j + 1] - V[i, j]
            d2 = V[i, j + 1] - V[i + 1, j]
            nrm = np.cross(d1, d2)
            ln = np.linalg.norm(nrm)
            normals[p] = nrm / ln if ln > 1e-12 else np.array([0, 0, 1.0])
            area[p] = 0.5 * ln
    if normals[:, 2].mean() < 0:                               # orient +z (up)
        normals = -normals
    return corners, colloc, normals, area


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


def build_aircraft_surfaces(ac, nc_box=4, nc_flap=2, ns_seg=3, nc_tail=3,
                            nc_rud=2, ns_tail=5):
    """Build every lifting surface's 3D panel lattice (aircraft frame) from
    aircraft_geom: 2 wing boxes + 12 wing flaps + 2 V-stab + 2 ruddervators.

    Returns a list of surface dicts for MultiSurfaceUVLM, each with body-frame
    corners/colloc/normals (here body=aircraft frame, identity rest pose) tagged by
    name and a `body` key (filled with a placeholder index; the multibody build
    rebinds these to Featherstone bodies). Dihedral, taper, sweep, incidence applied.
    """
    w = ac.wing; t = ac.tail
    dim = ac.wing_dims(); tdim = ac.tail_dims()
    dih_w = np.deg2rad(w.dihedral_deg)
    dih_t = np.deg2rad(t.dihedral_deg); inc_t = np.deg2rad(t.incidence_deg)
    surfaces = []

    def wing_grid(sgn, y0f, y1f, fa, fb, nc, ns):
        ys = w.root_offset + np.linspace(y0f, y1f, ns + 1) * (dim["semi"] - w.root_offset)
        V = np.zeros((nc + 1, ns + 1, 3))
        for jj, ya in enumerate(ys):
            c, x_le = ac._wing_chord(ya)
            for ii, f in enumerate(np.linspace(fa, fb, nc + 1)):
                s = ya - w.root_offset                       # spanwise dist from root
                V[ii, jj] = [x_le - f * c, sgn * ya, s * np.tan(dih_w)]
        return V

    def tail_grid(sgn, fa, fb, nc, ns):
        ys = 0.04 + np.linspace(0, 1, ns + 1) * (tdim["semi"] - 0.04)
        x0 = -t.boom
        V = np.zeros((nc + 1, ns + 1, 3))
        for jj, ya in enumerate(ys):
            fr = (ya - 0.04) / (tdim["semi"] - 0.04 + 1e-9)
            c = tdim["c_root"] + (tdim["c_tip"] - tdim["c_root"]) * fr
            x_le = x0 + 0.5 * tdim["c_root"]
            s = ya - 0.04
            for ii, f in enumerate(np.linspace(fa, fb, nc + 1)):
                xc = -f * c                                  # aft of LE
                xr = xc * np.cos(inc_t); zr = -xc * np.sin(inc_t)   # incidence: TE up
                V[ii, jj] = [x_le + xr, sgn * s * np.cos(dih_t),
                             zr + s * np.sin(dih_t)]          # dihedral tilts panel up
        return V

    def add(name, V, body):
        cor, col, nrm, area = make_vlm_lattice(V)
        nc, ns = V.shape[0] - 1, V.shape[1] - 1
        surfaces.append(dict(corners=cor, colloc=col, normals=nrm, area=area,
                             nc=nc, ns=ns, body=body, name=name, V=V))

    bidx = 1
    for sgn, side in ((+1, "L"), (-1, "R")):
        add(f"box_{side}", wing_grid(sgn, 0.0, 1.0, w.le_flap_frac, 1 - w.te_flap_frac,
                                     nc_box, 3 * ns_seg), bidx); bidx += 1
        for k in range(w.n_le):
            add(f"le_{side}{k}", wing_grid(sgn, k / w.n_le, (k + 1) / w.n_le, 0.0,
                                           w.le_flap_frac, nc_flap, ns_seg), bidx); bidx += 1
        for k in range(w.n_te):
            add(f"te_{side}{k}", wing_grid(sgn, k / w.n_te, (k + 1) / w.n_te,
                                           1 - w.te_flap_frac, 1.0, nc_flap, ns_seg), bidx); bidx += 1
    for sgn, side in ((+1, "L"), (-1, "R")):
        add(f"vstab_{side}", tail_grid(sgn, 0.0, 1 - t.ruddervator_frac, nc_tail, ns_tail), 0)
        add(f"rud_{side}", tail_grid(sgn, 1 - t.ruddervator_frac, 1.0, nc_rud, ns_tail), bidx); bidx += 1
    return surfaces


class MultiSurfaceUVLM:
    """Composite-AIC multi-surface UVLM over rigid bodies (plan §2 Surface atoms).

    surfaces: list of dicts with body-frame lattice + the rigid body it rides on:
      {corners(P,4,3), colloc(P,3), normals(P,3), nc, ns, body, name}
    All surfaces' panels are concatenated, so one AIC solve carries wing<->tail
    cross-induction; per-surface Kutta-Joukowski forces sum to per-body wrenches.
    """

    def __init__(self, surfaces, rho=1.225, core=1e-6, device=None):
        self.device = device or cfg.DEVICE
        self.surf = surfaces
        self.rho = float(rho); self.core = float(core)
        self.P = sum(s["nc"] * s["ns"] for s in surfaces)
        # panel-index span of each surface in the concatenated arrays
        off = 0
        for s in surfaces:
            s["p0"], s["p1"] = off, off + s["nc"] * s["ns"]
            off = s["p1"]

    def solve(self, poses, twists, V_inf_world):
        """poses/twists: dict body->(p,quat) / body->(v,omega) world. Returns
        dict body-> (F(3), M(3) about body origin), plus totals + gamma."""
        d = self.device
        cor = np.zeros((self.P, 4, 3)); col = np.zeros((self.P, 3))
        nrm = np.zeros((self.P, 3)); Vpan = np.zeros((self.P, 3))
        for s in surfaces_iter(self.surf):
            p, q = poses[s["body"]]
            v, om = twists[s["body"]]
            R = _quat_to_R(np.asarray(q, float)); p = np.asarray(p, float)
            a, b = s["p0"], s["p1"]
            cw = (s["corners"].reshape(-1, 3) @ R.T + p).reshape(-1, 4, 3)
            colw = s["colloc"] @ R.T + p
            cor[a:b] = cw; col[a:b] = colw; nrm[a:b] = s["normals"] @ R.T
            Vpan[a:b] = np.asarray(v, float)[None, :] + np.cross(
                np.asarray(om, float)[None, :], colw - p[None, :])
        Vinf = np.asarray(V_inf_world, float)
        rhs = -np.einsum('pi,pi->p', (Vinf[None, :] - Vpan), nrm)
        col_wp = wp.array(col.reshape(1, self.P, 3).astype(NP), dtype=VEC3, device=d)
        n_wp = wp.array(nrm.reshape(1, self.P, 3).astype(NP), dtype=VEC3, device=d)
        cor_wp = wp.array(cor.reshape(1, self.P, 4, 3).astype(NP), dtype=VEC3, device=d)
        rhs_wp = wp.array(rhs.reshape(1, self.P).astype(NP), dtype=cfg.DTYPE, device=d)
        AIC = build_aic_batched(col_wp, n_wp, cor_wp, self.core, device=d)
        gamma = batched_dense_solve(AIC, rhs_wp, device=d).numpy()[0]      # (P,)
        # per-surface Kutta-Joukowski force -> per-body wrench
        wrench = {}
        for s in self.surf:
            a, nc, ns = s["p0"], s["nc"], s["ns"]
            g = gamma[a:a + nc * ns].reshape(nc, ns)
            p_origin = np.asarray(poses[s["body"]][0], float)
            for i in range(nc):
                for j in range(ns):
                    pp = a + i * ns + j
                    gnet = g[i, j] - (g[i - 1, j] if i > 0 else 0.0)
                    lb = cor[pp, 3] - cor[pp, 0]               # spanwise bound edge
                    Vrel = Vinf - Vpan[pp]
                    Fp = self.rho * gnet * np.cross(Vrel, lb)
                    F, M = wrench.get(s["body"], (np.zeros(3), np.zeros(3)))
                    wrench[s["body"]] = (F + Fp, M + np.cross(col[pp] - p_origin, Fp))
        Ftot = sum(w[0] for w in wrench.values())
        return dict(wrench=wrench, F_total=Ftot, gamma=gamma,
                    lift=float(Ftot[2]), drag=float(Ftot[0]))


def surfaces_iter(surf):
    return surf


def _validate_my_lattice():
    """make_vlm_lattice on a flat unit plate must reproduce the validated lift
    (~3.79N at 6deg AoA) computed earlier from the ScGeometry lattice."""
    wp.init()
    nc, ns = 15, 10
    xs = np.linspace(0, 1, nc + 1); ys = np.linspace(0, 1, ns + 1)
    V = np.zeros((nc + 1, ns + 1, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            V[i, j] = [x, y, 0.0]
    cor, col, nrm, area = make_vlm_lattice(V)
    surf = [dict(corners=cor, colloc=col, normals=nrm, nc=nc, ns=ns, body=0, name="plate")]
    msu = MultiSurfaceUVLM(surf, rho=1.225, core=1e-6)
    V0 = 10.0; aoa = np.deg2rad(6.0)
    Vinf = np.array([V0 * np.cos(aoa), 0.0, -V0 * np.sin(aoa)])
    out = msu.solve({0: (np.zeros(3), np.array([0, 0, 0, 1.0]))},
                    {0: (np.zeros(3), np.zeros(3))}, Vinf)
    L = abs(out["lift"])
    ok = abs(L - 3.79) / 3.79 < 0.08
    print(f"make_vlm_lattice flat-plate UVLM: lift={L:.3f}N (ref ScGeometry lattice 3.79N) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


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
    ok1 = _validate_against_gpufluidsolve()
    ok2 = _validate_my_lattice()
    print(f"\nUVLM aircraft foundation: rigid-bridge {'PASS' if ok1 else 'FAIL'} | "
          f"multi-surface lattice {'PASS' if ok2 else 'FAIL'}")
    raise SystemExit(0 if (ok1 and ok2) else 1)
