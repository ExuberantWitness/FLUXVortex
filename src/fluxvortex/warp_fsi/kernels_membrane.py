"""Batched tension-field membrane element (CST, relaxed strain energy) — P2-S2.

3-node constant-strain triangle, total-Lagrangian, 3 translational DOF/node
(no bending, no rotations). Isotropic pretension N0 enters as a 2nd-PK initial
stress (linear energy term); out-of-plane stiffness is PURELY the prestress
geometric stiffness K_G(N0) — carried automatically by the FD tangent.

Constitutive: three-branch relaxed strain energy in principal-strain space
(Pipkin 1986 / Steigmann 1990; modern variant Zhang & Kiendl CMAME 2024).
Per unit reference area, thickness h, StVK plane stress (beta = E/(1-nu^2)),
principal Green strains e1 >= e2, kappa = N0/(h*beta):

  W_full = N0 (e1+e2) + h*beta/2 (e1^2 + 2 nu e1 e2 + e2^2)     [taut]
  W_wr   = N0 (1-nu) e1 + h*E/2 e1^2 - N0^2/(2 h beta)          [wrinkled]
           (= min_{e2} W_full, envelope => C1 across the boundary)
  W_sl   = W_wr(e1_sl),  e1_sl = -N0 (1-nu)/(h E)               [slack]

Mixed (Roddeman) criterion is intrinsic to the branch boundaries expressed in
TOTAL membrane forces: taut<->wrinkled at n2_full = 0, wrinkled<->slack at
n1_uni = 0. Regularization: W_eta = (1-eta) W_relaxed + eta W_full — keeps the
tangent >= eta*K_taut > 0 (dense LU never singular; Zhang & Kiendl eta = 1e-4)
at the price of O(eta) residual compressive force (documented, zero-fit safe:
eta is a *numerical* parameter with a published calibration protocol).

Differentiability (P4 hooks, forward path unchanged): branch Heaviside ->
sigmoid(k * n2/N0) with forward k->inf, backward finite k (LESP precedent);
principal sqrt(D) -> sqrt(D + eps^2). Not enabled in S2.

Literature basis: platform/docs/p2_s2_membrane_research.md.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3
MAT22 = wp.types.matrix(shape=(2, 2), dtype=DTYPE)
VEC9 = wp.types.vector(length=9, dtype=DTYPE)


# ─── element core (@wp.func) ──────────────────────────────────────────────────

@wp.func
def _green_strain(g1: VEC3, g2: VEC3, Bm: MAT22) -> MAT22:
    """E = 1/2 (Bm^T G Bm - I), G = current metric of the edge vectors."""
    G = MAT22(wp.dot(g1, g1), wp.dot(g1, g2),
              wp.dot(g1, g2), wp.dot(g2, g2))
    C = wp.transpose(Bm) * (G * Bm)
    return MAT22(DTYPE(0.5) * (C[0, 0] - DTYPE(1.0)), DTYPE(0.5) * C[0, 1],
                 DTYPE(0.5) * C[1, 0], DTYPE(0.5) * (C[1, 1] - DTYPE(1.0)))


@wp.func
def _principal(E: MAT22):
    """Principal strains of a symmetric 2x2 (e1 >= e2) and sqrt-discriminant."""
    tr = E[0, 0] + E[1, 1]
    dd = DTYPE(0.5) * (E[0, 0] - E[1, 1])
    sd = wp.sqrt(dd * dd + E[0, 1] * E[0, 1])
    return DTYPE(0.5) * tr + sd, DTYPE(0.5) * tr - sd, sd


@wp.func
def _branch_id(e1: DTYPE, e2: DTYPE,
               N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, forced: int) -> int:
    """0=taut, 1=wrinkled, 2=slack. forced>=0 overrides the criterion (IMP-style
    state latching for chatter-free static solves; -1 = evaluate from strain)."""
    if forced >= 0:
        return forced
    nf2 = N0 + hb * (e2 + nu * e1)
    if nf2 > DTYPE(0.0):
        return 0
    if e1 > -N0 * (DTYPE(1.0) - nu) / hE:
        return 1
    return 2


@wp.func
def _branch_forces(e1: DTYPE, e2: DTYPE,
                   N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, eta: DTYPE,
                   forced: int, w0: DTYPE, w1: DTYPE, w2: DTYPE):
    """Principal TOTAL membrane forces (n1, n2) [N/m], eta-blended branches.
    forced==-2: CONTINUOUS latching — blend the three branch forces with the
    frozen per-step weights (w0,w1,w2) (variational within the step, smooth
    across steps: kills the taut<->wrinkled chatter that pumps the two-stage
    integrator). forced>=0: hard latch. forced==-1: sharp criterion."""
    nf1 = N0 + hb * (e1 + nu * e2)              # full (taut) branch
    nf2 = N0 + hb * (e2 + nu * e1)
    nw1 = N0 * (DTYPE(1.0) - nu) + hE * e1      # wrinkled uniaxial
    n1r = DTYPE(0.0); n2r = DTYPE(0.0)
    if forced == -2:
        n1r = w0 * nf1 + w1 * nw1
        n2r = w0 * nf2
    else:
        br = _branch_id(e1, e2, N0, hb, hE, nu, forced)
        if br == 0:                              # taut
            n1r = nf1; n2r = nf2
        elif br == 1:                            # wrinkled (uniaxial)
            n1r = nw1
            n2r = DTYPE(0.0)
        # else slack: both zero
    one_m = DTYPE(1.0) - eta
    return one_m * n1r + eta * nf1, one_m * n2r + eta * nf2


@wp.func
def _membrane_W(E: MAT22, N0: DTYPE, hb: DTYPE, hE: DTYPE,
                nu: DTYPE, eta: DTYPE, forced: int,
                w0: DTYPE, w1: DTYPE, w2: DTYPE) -> DTYPE:
    """Relaxed energy per unit reference area, eta-blended (see _branch_forces
    for the forced/weights convention)."""
    e1, e2, _ = _principal(E)
    Wf = N0 * (e1 + e2) + DTYPE(0.5) * hb * (e1 * e1 + DTYPE(2.0) * nu * e1 * e2 + e2 * e2)
    e1sl = -N0 * (DTYPE(1.0) - nu) / hE
    Wwr_const = -DTYPE(0.5) * N0 * N0 / hb
    Wwr = N0 * (DTYPE(1.0) - nu) * e1 + DTYPE(0.5) * hE * e1 * e1 + Wwr_const
    Wsl = N0 * (DTYPE(1.0) - nu) * e1sl + DTYPE(0.5) * hE * e1sl * e1sl + Wwr_const
    Wr = Wf
    if forced == -2:
        Wr = w0 * Wf + w1 * Wwr + w2 * Wsl
    else:
        br = _branch_id(e1, e2, N0, hb, hE, nu, forced)
        if br == 1:
            Wr = Wwr
        elif br == 2:
            Wr = Wsl
    return (DTYPE(1.0) - eta) * Wr + eta * Wf


@wp.func
def _membrane_N(E: MAT22, N0: DTYPE, hb: DTYPE, hE: DTYPE,
                nu: DTYPE, eta: DTYPE, forced: int,
                w0: DTYPE, w1: DTYPE, w2: DTYPE) -> MAT22:
    """Total membrane force tensor N = dW/dE (2x2, principal reconstruction)."""
    e1, e2, sd = _principal(E)
    n1, n2 = _branch_forces(e1, e2, N0, hb, hE, nu, eta, forced, w0, w1, w2)
    a = DTYPE(0.5) * (n1 + n2)
    if sd > DTYPE(1e-14):
        b = DTYPE(0.5) * (n1 - n2) / sd
        tr2 = DTYPE(0.5) * (E[0, 0] + E[1, 1])
        return MAT22(a + b * (E[0, 0] - tr2), b * E[0, 1],
                     b * E[1, 0], a + b * (E[1, 1] - tr2))
    return MAT22(a, DTYPE(0.0), DTYPE(0.0), a)


@wp.func
def _membrane_elem_force(p: VEC9, dX1: VEC3, dX2: VEC3, Bm: MAT22, A0: DTYPE,
                         N0: DTYPE, hb: DTYPE, hE: DTYPE,
                         nu: DTYPE, eta: DTYPE, forced: int,
                         w0: DTYPE, w1: DTYPE, w2: DTYPE) -> VEC9:
    """Q_e = d(A0 W)/dp. p = [u1,u2,u3]; edges g_i = dX_i + (u_{i+1}-u_1)."""
    u1 = wp.vector(p[0], p[1], p[2])
    u2 = wp.vector(p[3], p[4], p[5])
    u3 = wp.vector(p[6], p[7], p[8])
    g1 = dX1 + u2 - u1
    g2 = dX2 + u3 - u1
    E = _green_strain(g1, g2, Bm)
    N = _membrane_N(E, N0, hb, hE, nu, eta, forced, w0, w1, w2)
    # N : dE = Ht : dG / 2, Ht = Bm N Bm^T  =>  f_g1 = A0 (Ht00 g1 + Ht01 g2) ...
    Ht = Bm * (N * wp.transpose(Bm))
    fg1 = A0 * (Ht[0, 0] * g1 + Ht[0, 1] * g2)
    fg2 = A0 * (Ht[1, 0] * g1 + Ht[1, 1] * g2)
    f = VEC9()
    f[0] = -fg1[0] - fg2[0]; f[1] = -fg1[1] - fg2[1]; f[2] = -fg1[2] - fg2[2]
    f[3] = fg1[0]; f[4] = fg1[1]; f[5] = fg1[2]
    f[6] = fg2[0]; f[7] = fg2[1]; f[8] = fg2[2]
    return f


# ─── kernels ──────────────────────────────────────────────────────────────────

@wp.kernel
def membrane_force_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                          edofs: wp.array(dtype=wp.int32, ndim=2),   # (ne, 9)
                          dX1: wp.array(dtype=VEC3, ndim=1),
                          dX2: wp.array(dtype=VEC3, ndim=1),
                          Bm: wp.array(dtype=MAT22, ndim=1),
                          A0: wp.array(dtype=DTYPE, ndim=1),
                          N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, eta: DTYPE,
                          br: wp.array(dtype=wp.int32, ndim=1),
                          wts: wp.array(dtype=DTYPE, ndim=2),
                          Q: wp.array(dtype=DTYPE, ndim=2)):
    e, el = wp.tid()
    p = VEC9()
    for a in range(9):
        p[a] = q[e, edofs[el, a]]
    f = _membrane_elem_force(p, dX1[el], dX2[el], Bm[el], A0[el], N0, hb, hE, nu, eta,
                             br[el], wts[el, 0], wts[el, 1], wts[el, 2])
    for a in range(9):
        wp.atomic_add(Q, e, edofs[el, a], f[a])


@wp.kernel
def membrane_energy_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                           edofs: wp.array(dtype=wp.int32, ndim=2),
                           dX1: wp.array(dtype=VEC3, ndim=1),
                           dX2: wp.array(dtype=VEC3, ndim=1),
                           Bm: wp.array(dtype=MAT22, ndim=1),
                           A0: wp.array(dtype=DTYPE, ndim=1),
                           N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, eta: DTYPE,
                           br: wp.array(dtype=wp.int32, ndim=1),
                           wts: wp.array(dtype=DTYPE, ndim=2),
                           W: wp.array(dtype=DTYPE, ndim=1)):
    e, el = wp.tid()
    u1 = wp.vector(q[e, edofs[el, 0]], q[e, edofs[el, 1]], q[e, edofs[el, 2]])
    u2 = wp.vector(q[e, edofs[el, 3]], q[e, edofs[el, 4]], q[e, edofs[el, 5]])
    u3 = wp.vector(q[e, edofs[el, 6]], q[e, edofs[el, 7]], q[e, edofs[el, 8]])
    g1 = dX1[el] + u2 - u1
    g2 = dX2[el] + u3 - u1
    E = _green_strain(g1, g2, Bm[el])
    wp.atomic_add(W, e, A0[el] * _membrane_W(E, N0, hb, hE, nu, eta, br[el], wts[el, 0], wts[el, 1], wts[el, 2]))


@wp.kernel
def membrane_state_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                          edofs: wp.array(dtype=wp.int32, ndim=2),
                          dX1: wp.array(dtype=VEC3, ndim=1),
                          dX2: wp.array(dtype=VEC3, ndim=1),
                          Bm: wp.array(dtype=MAT22, ndim=1),
                          N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, eta: DTYPE,
                          br: wp.array(dtype=wp.int32, ndim=1),
                          wts: wp.array(dtype=DTYPE, ndim=2),
                          out: wp.array(dtype=DTYPE, ndim=3)):     # (B, ne, 4)
    """Diagnostics: (e1, e2, n1, n2) per element (gates + S4 monitoring)."""
    e, el = wp.tid()
    u1 = wp.vector(q[e, edofs[el, 0]], q[e, edofs[el, 1]], q[e, edofs[el, 2]])
    u2 = wp.vector(q[e, edofs[el, 3]], q[e, edofs[el, 4]], q[e, edofs[el, 5]])
    u3 = wp.vector(q[e, edofs[el, 6]], q[e, edofs[el, 7]], q[e, edofs[el, 8]])
    g1 = dX1[el] + u2 - u1
    g2 = dX2[el] + u3 - u1
    E = _green_strain(g1, g2, Bm[el])
    e1, e2, _ = _principal(E)
    n1, n2 = _branch_forces(e1, e2, N0, hb, hE, nu, eta, br[el],
                            wts[el, 0], wts[el, 1], wts[el, 2])
    out[e, el, 0] = e1; out[e, el, 1] = e2
    out[e, el, 2] = n1; out[e, el, 3] = n2


@wp.kernel
def membrane_ktan_fd_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                            edofs: wp.array(dtype=wp.int32, ndim=2),
                            dX1: wp.array(dtype=VEC3, ndim=1),
                            dX2: wp.array(dtype=VEC3, ndim=1),
                            Bm: wp.array(dtype=MAT22, ndim=1),
                            A0: wp.array(dtype=DTYPE, ndim=1),
                            N0: DTYPE, hb: DTYPE, hE: DTYPE, nu: DTYPE, eta: DTYPE,
                            br: wp.array(dtype=wp.int32, ndim=1),
                            wts: wp.array(dtype=DTYPE, ndim=2),
                            rel_h: DTYPE,
                            min_edge: wp.array(dtype=DTYPE, ndim=1),
                            Kraw: wp.array(dtype=DTYPE, ndim=4)):  # (B, ne, 9, 9)
    # FD step PER ELEMENT: h = rel_h * min_edge -> strain excursion == rel_h,
    # independent of element size. A fixed absolute h on sliver elements (wing
    # tip: edges ~2mm) gives strain excursions ~ kappa = N0/(h beta) that CROSS
    # the wrinkling boundary: columns then mix taut/wrinkled branches and the
    # symmetrized result is no tangent of any energy (measured -3e4 spurious
    # negative eigenvalue at the tip corner, N0=30 + camber).
    e, el, j = wp.tid()
    h_fd = rel_h * min_edge[el]
    p = VEC9()
    for a in range(9):
        p[a] = q[e, edofs[el, a]]
    pp = p
    pp[j] = p[j] + h_fd
    fp = _membrane_elem_force(pp, dX1[el], dX2[el], Bm[el], A0[el], N0, hb, hE, nu, eta,
                              br[el], wts[el, 0], wts[el, 1], wts[el, 2])
    pm = p
    pm[j] = p[j] - h_fd
    fm = _membrane_elem_force(pm, dX1[el], dX2[el], Bm[el], A0[el], N0, hb, hE, nu, eta,
                              br[el], wts[el, 0], wts[el, 1], wts[el, 2])
    inv2h = DTYPE(1.0) / (DTYPE(2.0) * h_fd)
    for i in range(9):
        Kraw[e, el, i, j] = (fp[i] - fm[i]) * inv2h


@wp.kernel
def _symmetrize9_kernel(Kraw: wp.array(dtype=DTYPE, ndim=4),
                        Kblk: wp.array(dtype=DTYPE, ndim=4)):
    e, el, i, j = wp.tid()
    Kblk[e, el, i, j] = DTYPE(0.5) * (Kraw[e, el, i, j] + Kraw[e, el, j, i])


# ─── host constants + assembly ────────────────────────────────────────────────

class MembraneConstants:
    """Element constants for the tension-field CST membrane.

    nodes (nn,3); tris (ne,3) int; material: h, E, nu, rho; pretension N0 [N/m];
    eta residual-stiffness blend. dof_map (nn,3) optional (S3 hook: alias
    membrane translations onto beam-node translational DOFs); ndof override.
    """

    def __init__(self, nodes, tris, h, E, nu, rho, N0, eta=1e-4,
                 dof_map=None, ndof=None, device=None):
        device = device or config.DEVICE
        NP = config.NP_DTYPE
        nodes = np.asarray(nodes, float)
        tris = np.asarray(tris, dtype=np.int64)
        nn = len(nodes); ne = len(tris)
        if dof_map is None:
            dof_map = np.arange(nn * 3, dtype=np.int64).reshape(nn, 3)
        dof_map = np.asarray(dof_map, dtype=np.int64)
        self.ndof = int(ndof if ndof is not None else dof_map.max() + 1)

        dX1 = np.zeros((ne, 3)); dX2 = np.zeros((ne, 3))
        Bm = np.zeros((ne, 2, 2)); A0 = np.zeros(ne)
        Me = np.zeros((ne, 9, 9))
        edofs = np.zeros((ne, 9), dtype=np.int32)
        min_edge = np.zeros(ne)
        for e, (i, j, k) in enumerate(tris):
            G1 = nodes[j] - nodes[i]; G2 = nodes[k] - nodes[i]
            nrm = np.cross(G1, G2)
            A2 = np.linalg.norm(nrm)
            assert A2 > 1e-14, f"degenerate triangle {e}"
            A0[e] = 0.5 * A2
            min_edge[e] = min(np.linalg.norm(G1), np.linalg.norm(G2),
                              np.linalg.norm(nodes[k] - nodes[j]))
            t1 = G1 / np.linalg.norm(G1)
            nv = nrm / A2
            t2 = np.cross(nv, t1)
            Dm = np.array([[np.linalg.norm(G1), G2 @ t1],
                           [0.0, G2 @ t2]])
            Bm[e] = np.linalg.inv(Dm)
            dX1[e] = G1; dX2[e] = G2
            m = rho * h * A0[e] / 12.0
            for (a, b, w) in ((0, 0, 2.), (1, 1, 2.), (2, 2, 2.),
                              (0, 1, 1.), (1, 0, 1.), (0, 2, 1.),
                              (2, 0, 1.), (1, 2, 1.), (2, 1, 1.)):
                Me[e, 3*a:3*a+3, 3*b:3*b+3] = w * m * np.eye(3)
            edofs[e] = np.concatenate([dof_map[i], dof_map[j], dof_map[k]]).astype(np.int32)

        self.nn = nn; self.ne = ne
        self.nodes_np = nodes; self.tris_np = tris
        self.dof_map_np = dof_map
        self.edofs_np = edofs
        self.Me_np = Me; self.A0_np = A0
        beta = E / (1.0 - nu * nu)
        self.h, self.E, self.nu, self.rho = float(h), float(E), float(nu), float(rho)
        self.N0, self.eta = float(N0), float(eta)
        self.hb, self.hE = float(h * beta), float(h * E)
        self.device = device
        self.dX1 = wp.array(dX1.astype(NP), dtype=VEC3, device=device)
        self.dX2 = wp.array(dX2.astype(NP), dtype=VEC3, device=device)
        self.Bm = wp.array(Bm.astype(NP), dtype=MAT22, device=device)
        self.A0 = wp.array(A0.astype(NP), dtype=DTYPE, device=device)
        self.edofs = wp.array(edofs, dtype=wp.int32, device=device)
        self.Me = wp.array(Me.astype(NP), dtype=DTYPE, device=device)
        self.min_edge = wp.array(min_edge.astype(NP), dtype=DTYPE, device=device)
        self.branch_np = -np.ones(ne, dtype=np.int32)      # -1 = auto (criterion)
        self.branch = wp.array(self.branch_np, dtype=wp.int32, device=device)
        self.wts_np = np.zeros((ne, 3), dtype=NP)          # continuous-latch weights
        self.wts = wp.array(self.wts_np, dtype=DTYPE, device=device)
        self.bc_dofs = set()
        self.free_np = np.ones(self.ndof, dtype=NP)
        self.free = wp.array(self.free_np, dtype=DTYPE, device=device)

    def _params(self):
        NP = config.NP_DTYPE
        return [DTYPE(NP(self.N0)), DTYPE(NP(self.hb)), DTYPE(NP(self.hE)),
                DTYPE(NP(self.nu)), DTYPE(NP(self.eta))]

    def freeze_branches(self, ids):
        """IMP-style state latching: force per-element branch (0/1/2), -1=auto."""
        self.branch_np = np.asarray(ids, dtype=np.int32).copy()
        self.branch = wp.array(self.branch_np, dtype=wp.int32, device=self.device)
        return self

    def unfreeze_branches(self):
        return self.freeze_branches(-np.ones(self.ne, dtype=np.int32))

    def set_soft_weights(self, w):
        """Continuous latching: per-element branch weights (ne,3), frozen for
        the step; branch id -2 selects the weighted blend."""
        NP = config.NP_DTYPE
        self.wts_np = np.asarray(w, dtype=NP).reshape(self.ne, 3)
        self.wts = wp.array(self.wts_np, dtype=DTYPE, device=self.device)
        return self.freeze_branches(-2 * np.ones(self.ne, dtype=np.int32))

    def set_bc(self, node_ids, dirs=(0, 1, 2)):
        for n in node_ids:
            for d in dirs:
                self.bc_dofs.add(int(self.dof_map_np[n][d]))
        self.free_np = np.ones(self.ndof, dtype=config.NP_DTYPE)
        for d in self.bc_dofs:
            self.free_np[d] = 0.0
        self.free = wp.array(self.free_np, dtype=DTYPE, device=self.device)
        return self


def membrane_internal_force(q_wp, C: MembraneConstants, device=None):
    device = device or C.device
    B = q_wp.shape[0]
    Q = wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    wp.launch(membrane_force_kernel, dim=(B, C.ne),
              inputs=[q_wp, C.edofs, C.dX1, C.dX2, C.Bm, C.A0] + C._params() + [C.branch, C.wts],
              outputs=[Q], device=device)
    return Q


def membrane_energy_total(q_wp, C: MembraneConstants, device=None):
    device = device or C.device
    B = q_wp.shape[0]
    W = wp.zeros(B, dtype=DTYPE, device=device)
    wp.launch(membrane_energy_kernel, dim=(B, C.ne),
              inputs=[q_wp, C.edofs, C.dX1, C.dX2, C.Bm, C.A0] + C._params() + [C.branch, C.wts],
              outputs=[W], device=device)
    return W


def membrane_state(q_wp, C: MembraneConstants, device=None):
    device = device or C.device
    B = q_wp.shape[0]
    out = wp.zeros((B, C.ne, 4), dtype=DTYPE, device=device)
    wp.launch(membrane_state_kernel, dim=(B, C.ne),
              inputs=[q_wp, C.edofs, C.dX1, C.dX2, C.Bm] + C._params() + [C.branch, C.wts],
              outputs=[out], device=device)
    return out


def assemble_membrane_kblocks(q_wp, C: MembraneConstants, rel_h=1e-6,
                              symmetrize=True, device=None):
    """rel_h = FD strain excursion (per-element h = rel_h * min_edge); keep
    rel_h << kappa = N0/(h beta) so columns never cross the wrinkle boundary."""
    device = device or C.device
    NP = config.NP_DTYPE
    B = q_wp.shape[0]
    Kraw = wp.zeros((B, C.ne, 9, 9), dtype=DTYPE, device=device)
    wp.launch(membrane_ktan_fd_kernel, dim=(B, C.ne, 9),
              inputs=[q_wp, C.edofs, C.dX1, C.dX2, C.Bm, C.A0] + C._params()
                     + [C.branch, C.wts, DTYPE(NP(rel_h)), C.min_edge],
              outputs=[Kraw], device=device)
    if not symmetrize:
        return Kraw
    Kblk = wp.zeros((B, C.ne, 9, 9), dtype=DTYPE, device=device)
    wp.launch(_symmetrize9_kernel, dim=(B, C.ne, 9, 9),
              inputs=[Kraw], outputs=[Kblk], device=device)
    return Kblk
