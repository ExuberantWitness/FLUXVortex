"""Batched geometrically-exact 3D beam (Simo-Reissner) element kernels — P2-S1.

2-node element, 1 Gauss point (reduced integration, shear-lock free), 6 DOF/node
[u(3), psi(3)] with psi the GLOBAL total rotation vector of the node
(nodal triad Lambda_i = exp(hat(psi_i)) @ Lambda0_elem; rigid joints = shared psi).

Kinematics (material strain measures, objective by construction —
Crisfield-Jelenic relative-rotation interpolation, constant-strain element):
    A_rel = Lambda0^T R1^T R2 Lambda0          (material relative rotation)
    phi   = log(A_rel),   kappa = phi / L0     (material curvature: twist+bend)
    Lm    = R1 Lambda0 exp(hat(phi)/2)         (mid triad)
    Gamma = Lm^T x' - e1,  x' = (X2+u2-X1-u1)/L0   (axial + shear strain)
Constitutive (small strain, linear elastic):
    N = diag(EA, ks GA, ks GA) Gamma ;  M = diag(GJ, EI2, EI3) kappa
Internal force = exact analytic variation of E = L0/2 (Gamma^T CF Gamma + kappa^T CM kappa)
in psi coordinates (right-Jacobian chain), verified force==dE/dp (energy gate).

Consistent tangent: built by column-wise CENTRAL DIFFERENCING of the closed-form
force (fp64, h=1e-7 -> ~1e-9 relative accuracy). This includes material +
geometric (stress-stiffening) + parameterization terms all at once (decision D-e);
symmetrized (K+K^T)/2 for the PCG operator. The tangent only enters the Newmark
damping operator (internal force enters the RHS exactly), so 1e-9-consistent is
numerically indistinguishable from closed-form for the solver's purposes.

Validity chart: |psi|, |phi| < pi (log/exp charts). Production use is flapping
+-45 deg + elastic — far from the chart edge; entries enforce max|psi| checks.

Documented approximations (see docs/p2_s1_beam3d.md):
 - additive Newmark update of psi (psi_dot != omega is 2nd order in increments)
 - constant reference-frame consistent mass (linear interp + fixed rotary inertia)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import warp as wp
from . import config

DTYPE = config.DTYPE
VEC3 = config.VEC3
MAT33 = config.MAT33
VEC12 = wp.types.vector(length=12, dtype=DTYPE)


# ─── Section constants (zero-fit: physics only) ──────────────────────────────

@dataclass(frozen=True)
class TubeSection:
    """Thin-wall circular tube; all stiffness/inertia derived from geometry+material.

    G note (D-f, 2026-07-05): no torsional measurement exists in the paper/eiDATA
    (checked — Fig13/eiDATA is chordwise bending k only; the paper defines wing
    twist resistance via chordwise EI, anchored by K_MEAS). Pultruded carbon tube
    torsion is matrix-dominated: G12 ~ 4-6 GPa (NOT isotropic E/2(1+nu) ~ 58 GPa,
    which overestimates ~10x). Default G = 5 GPa; S6 runs a GJ sensitivity sweep
    (4/5/6 GPa). This is a material-property choice, not a fit.
    """
    D_out: float          # outer diameter (m)
    wall: float           # wall thickness (m)
    E: float              # axial modulus (Pa)
    rho: float            # density (kg/m^3)
    G: float              # shear modulus (Pa) — see note above
    ks: float = 0.5       # shear correction (thin-wall circular tube)

    @property
    def D_in(self): return self.D_out - 2.0 * self.wall
    @property
    def A(self): return math.pi * (self.D_out**2 - self.D_in**2) / 4.0
    @property
    def I(self): return math.pi * (self.D_out**4 - self.D_in**4) / 64.0
    @property
    def J(self): return 2.0 * self.I
    @property
    def EA(self): return self.E * self.A
    @property
    def EI(self): return self.E * self.I
    @property
    def GJ(self): return self.G * self.J
    @property
    def GAks(self): return self.G * self.A * self.ks
    @property
    def m_lin(self): return self.rho * self.A          # kg/m
    @property
    def rhoJ(self): return self.rho * self.J           # torsional inertia / length
    @property
    def rhoI(self): return self.rho * self.I           # bending rotary inertia / length


# RoboEagle spars (user-supplied geometry; E/rho literature-anchored, zero-fit).
# EI_main = 43.5 N*m^2, 0.045 kg/m ; EI_aux = 7.66 N*m^2, 0.025 kg/m.
MAIN_SPAR = TubeSection(D_out=10e-3, wall=1e-3, E=150e9, rho=1592.0, G=5e9)
AUX_SPAR = TubeSection(D_out=6e-3, wall=1e-3, E=150e9, rho=1592.0, G=5e9)


# ─── SO(3) utilities (@wp.func, series-guarded) ──────────────────────────────

@wp.func
def _eye3() -> MAT33:
    return MAT33(DTYPE(1.0), DTYPE(0.0), DTYPE(0.0),
                 DTYPE(0.0), DTYPE(1.0), DTYPE(0.0),
                 DTYPE(0.0), DTYPE(0.0), DTYPE(1.0))


@wp.func
def so3_exp(v: VEC3) -> MAT33:
    """Rodrigues exp map; series for small angle."""
    th2 = wp.dot(v, v)
    th = wp.sqrt(th2)
    a = DTYPE(1.0) - th2 / DTYPE(6.0)          # sin(th)/th
    b = DTYPE(0.5) - th2 / DTYPE(24.0)         # (1-cos th)/th^2
    if th > DTYPE(1e-4):
        a = wp.sin(th) / th
        b = (DTYPE(1.0) - wp.cos(th)) / th2
    K = wp.skew(v)
    return _eye3() + a * K + b * (K * K)


@wp.func
def so3_log(R: MAT33) -> VEC3:
    """Rotation vector of R; series for small angle. Chart |phi| < pi."""
    ct = DTYPE(0.5) * (wp.trace(R) - DTYPE(1.0))
    ct = wp.clamp(ct, DTYPE(-1.0), DTYPE(1.0))
    th = wp.acos(ct)
    w = wp.vector(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])  # 2 sin(th) axis
    f = DTYPE(0.5) + th * th / DTYPE(12.0)     # th/(2 sin th)
    if th > DTYPE(1e-4):
        f = th / (DTYPE(2.0) * wp.sin(th))
    return f * w


@wp.func
def so3_jr(v: VEC3) -> MAT33:
    """Right Jacobian: exp(hat(v+dv)) = exp(hat(v)) exp(hat(Jr(v) dv))."""
    th2 = wp.dot(v, v)
    th = wp.sqrt(th2)
    b = DTYPE(0.5) - th2 / DTYPE(24.0)         # (1-cos th)/th^2
    c = DTYPE(1.0) / DTYPE(6.0) - th2 / DTYPE(120.0)   # (th-sin th)/th^3
    if th > DTYPE(1e-4):
        b = (DTYPE(1.0) - wp.cos(th)) / th2
        c = (th - wp.sin(th)) / (th2 * th)
    K = wp.skew(v)
    return _eye3() - b * K + c * (K * K)


@wp.func
def so3_jr_inv(v: VEC3) -> MAT33:
    """Inverse right Jacobian; series for small angle."""
    th2 = wp.dot(v, v)
    th = wp.sqrt(th2)
    d = DTYPE(1.0) / DTYPE(12.0) + th2 / DTYPE(720.0)
    if th > DTYPE(1e-4):
        d = DTYPE(1.0) / th2 - (DTYPE(1.0) + wp.cos(th)) / (DTYPE(2.0) * th * wp.sin(th))
    K = wp.skew(v)
    return _eye3() + DTYPE(0.5) * K + d * (K * K)


# ─── Element energy / internal force (closed form) ───────────────────────────

@wp.func
def _beam_elem_energy(p: VEC12, dX: VEC3, Lam0: MAT33, L0: DTYPE,
                      CF: VEC3, CM: VEC3) -> DTYPE:
    u1 = wp.vector(p[0], p[1], p[2]);  ps1 = wp.vector(p[3], p[4], p[5])
    u2 = wp.vector(p[6], p[7], p[8]);  ps2 = wp.vector(p[9], p[10], p[11])
    R1 = so3_exp(ps1); R2 = so3_exp(ps2)
    Arel = wp.transpose(Lam0) * (wp.transpose(R1) * R2) * Lam0
    phi = so3_log(Arel)
    kap = phi / L0
    Lm = R1 * (Lam0 * so3_exp(DTYPE(0.5) * phi))
    xp = (dX + u2 - u1) / L0
    vm = wp.transpose(Lm) * xp
    Gam = wp.vector(vm[0] - DTYPE(1.0), vm[1], vm[2])
    eF = CF[0] * Gam[0] * Gam[0] + CF[1] * Gam[1] * Gam[1] + CF[2] * Gam[2] * Gam[2]
    eM = CM[0] * kap[0] * kap[0] + CM[1] * kap[1] * kap[1] + CM[2] * kap[2] * kap[2]
    return DTYPE(0.5) * L0 * (eF + eM)


@wp.func
def _beam_elem_force(p: VEC12, dX: VEC3, Lam0: MAT33, L0: DTYPE,
                     CF: VEC3, CM: VEC3) -> VEC12:
    """Q_e = dE/dp (exact analytic variation in psi coordinates)."""
    u1 = wp.vector(p[0], p[1], p[2]);  ps1 = wp.vector(p[3], p[4], p[5])
    u2 = wp.vector(p[6], p[7], p[8]);  ps2 = wp.vector(p[9], p[10], p[11])
    R1 = so3_exp(ps1); R2 = so3_exp(ps2)
    Lam0T = wp.transpose(Lam0)
    Arel = Lam0T * (wp.transpose(R1) * R2) * Lam0
    phi = so3_log(Arel)
    kap = phi / L0
    Ph = Lam0 * so3_exp(DTYPE(0.5) * phi)          # Lambda0 exp(phi/2)
    Lm = R1 * Ph
    xp = (dX + u2 - u1) / L0
    vm = wp.transpose(Lm) * xp
    Gam = wp.vector(vm[0] - DTYPE(1.0), vm[1], vm[2])
    N = wp.vector(CF[0] * Gam[0], CF[1] * Gam[1], CF[2] * Gam[2])
    M = wp.vector(CM[0] * kap[0], CM[1] * kap[1], CM[2] * kap[2])
    # curvature variation blocks: dkappa = Gk1 dpsi1 + Gk2 dpsi2
    invL = DTYPE(1.0) / L0
    Jinv = so3_jr_inv(phi)
    Jr1 = so3_jr(ps1); Jr2 = so3_jr(ps2)
    Gk2 = (Jinv * (Lam0T * Jr2)) * invL
    Gk1 = -((Jinv * (wp.transpose(Arel) * (Lam0T * Jr1))) * invL)
    # Gamma variation: dGam = skew(vm) c + Lm^T (du2-du1)/L0,
    #   c = Ph^T Jr1 dpsi1 + (1/2) Jr(phi/2) dphi,  dphi = L0 (Gk1 dpsi1 + Gk2 dpsi2)
    Jh = so3_jr(DTYPE(0.5) * phi)
    Vh = wp.skew(vm)
    GG1 = Vh * (wp.transpose(Ph) * Jr1 + (DTYPE(0.5) * L0) * (Jh * Gk1))
    GG2 = Vh * ((DTYPE(0.5) * L0) * (Jh * Gk2))
    # nodal forces (delta_W = L0 (dGam^T N + dkap^T M))
    LmN = Lm * N
    Qp1 = L0 * (wp.transpose(GG1) * N + wp.transpose(Gk1) * M)
    Qp2 = L0 * (wp.transpose(GG2) * N + wp.transpose(Gk2) * M)
    f = VEC12()
    f[0] = -LmN[0]; f[1] = -LmN[1]; f[2] = -LmN[2]
    f[3] = Qp1[0];  f[4] = Qp1[1];  f[5] = Qp1[2]
    f[6] = LmN[0];  f[7] = LmN[1];  f[8] = LmN[2]
    f[9] = Qp2[0];  f[10] = Qp2[1]; f[11] = Qp2[2]
    return f


# ─── Kernels ──────────────────────────────────────────────────────────────────

@wp.kernel
def beam_force_kernel(q: wp.array(dtype=DTYPE, ndim=2),        # (B, ndof)
                      edofs: wp.array(dtype=wp.int32, ndim=2), # (ne, 12)
                      dX: wp.array(dtype=VEC3, ndim=1),        # (ne,) X2-X1
                      Lam0: wp.array(dtype=MAT33, ndim=1),     # (ne,)
                      L0: wp.array(dtype=DTYPE, ndim=1),       # (ne,)
                      CF: wp.array(dtype=VEC3, ndim=1),        # (ne,) [EA, ksGA, ksGA]
                      CM: wp.array(dtype=VEC3, ndim=1),        # (ne,) [GJ, EI2, EI3]
                      Q: wp.array(dtype=DTYPE, ndim=2)):       # (B, ndof) accumulate
    e, el = wp.tid()
    p = VEC12()
    for a in range(12):
        p[a] = q[e, edofs[el, a]]
    f = _beam_elem_force(p, dX[el], Lam0[el], L0[el], CF[el], CM[el])
    for a in range(12):
        wp.atomic_add(Q, e, edofs[el, a], f[a])


@wp.kernel
def beam_energy_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                       edofs: wp.array(dtype=wp.int32, ndim=2),
                       dX: wp.array(dtype=VEC3, ndim=1),
                       Lam0: wp.array(dtype=MAT33, ndim=1),
                       L0: wp.array(dtype=DTYPE, ndim=1),
                       CF: wp.array(dtype=VEC3, ndim=1),
                       CM: wp.array(dtype=VEC3, ndim=1),
                       E: wp.array(dtype=DTYPE, ndim=1)):      # (B,) accumulate
    e, el = wp.tid()
    p = VEC12()
    for a in range(12):
        p[a] = q[e, edofs[el, a]]
    en = _beam_elem_energy(p, dX[el], Lam0[el], L0[el], CF[el], CM[el])
    wp.atomic_add(E, e, en)


@wp.kernel
def beam_ktan_fd_kernel(q: wp.array(dtype=DTYPE, ndim=2),
                        edofs: wp.array(dtype=wp.int32, ndim=2),
                        dX: wp.array(dtype=VEC3, ndim=1),
                        Lam0: wp.array(dtype=MAT33, ndim=1),
                        L0: wp.array(dtype=DTYPE, ndim=1),
                        CF: wp.array(dtype=VEC3, ndim=1),
                        CM: wp.array(dtype=VEC3, ndim=1),
                        h: DTYPE,
                        Kraw: wp.array(dtype=DTYPE, ndim=4)):  # (B, ne, 12, 12) out
    """Column j of the consistent tangent by central difference of the closed-form
    force (exact to ~1e-9 in fp64; includes material+geometric+parameterization)."""
    e, el, j = wp.tid()
    p = VEC12()
    for a in range(12):
        p[a] = q[e, edofs[el, a]]
    pp = p
    pp[j] = p[j] + h
    fp = _beam_elem_force(pp, dX[el], Lam0[el], L0[el], CF[el], CM[el])
    pm = p
    pm[j] = p[j] - h
    fm = _beam_elem_force(pm, dX[el], Lam0[el], L0[el], CF[el], CM[el])
    inv2h = DTYPE(1.0) / (DTYPE(2.0) * h)
    for i in range(12):
        Kraw[e, el, i, j] = (fp[i] - fm[i]) * inv2h


@wp.kernel
def _symmetrize_kernel(Kraw: wp.array(dtype=DTYPE, ndim=4),
                       Kblk: wp.array(dtype=DTYPE, ndim=4)):
    e, el, i, j = wp.tid()
    Kblk[e, el, i, j] = DTYPE(0.5) * (Kraw[e, el, i, j] + Kraw[e, el, j, i])


# ─── Host constants + assembly ────────────────────────────────────────────────

def _frame_from_axis(t):
    """Reference triad columns [t, n, b] with a deterministic normal choice."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = ref - t * (ref @ t)
    n /= np.linalg.norm(n)
    return np.column_stack([t, n, np.cross(t, n)])


class Beam3DConstants:
    """Uploaded element constants + DOF map for the geometrically-exact beam.

    nodes (nn,3); elems (ne,2) int; sections: TubeSection or list per element.
    dof_map: optional (nn,6) int global DOF indices (default 6n+d) — the S3
    beam-membrane coupling hook (alias translational DOFs onto membrane nodes,
    keep psi independently numbered). ndof: total global DOF count (default 6nn).
    """

    def __init__(self, nodes, elems, sections, dof_map=None, ndof=None,
                 Lam0=None, device=None):
        device = device or config.DEVICE
        NP = config.NP_DTYPE
        nodes = np.asarray(nodes, float)
        elems = np.asarray(elems, dtype=np.int64)
        nn = len(nodes); ne = len(elems)
        if not isinstance(sections, (list, tuple)):
            sections = [sections] * ne
        assert len(sections) == ne
        if dof_map is None:
            dof_map = np.arange(nn * 6, dtype=np.int64).reshape(nn, 6)
        dof_map = np.asarray(dof_map, dtype=np.int64)
        self.ndof = int(ndof if ndof is not None else dof_map.max() + 1)

        dX = np.zeros((ne, 3)); L0 = np.zeros(ne)
        Lam = np.zeros((ne, 3, 3))
        CF = np.zeros((ne, 3)); CM = np.zeros((ne, 3))
        Me = np.zeros((ne, 12, 12))
        edofs = np.zeros((ne, 12), dtype=np.int32)
        for e, (i, j) in enumerate(elems):
            d = nodes[j] - nodes[i]
            L = np.linalg.norm(d)
            t = d / L
            dX[e] = d; L0[e] = L
            Lam[e] = _frame_from_axis(t) if Lam0 is None else np.asarray(Lam0)[e]
            s = sections[e]
            CF[e] = [s.EA, s.GAks, s.GAks]
            CM[e] = [s.GJ, s.EI, s.EI]
            # constant consistent mass: linear interp (translation isotropic;
            # rotary inertia in the reference frame — documented approximation)
            mt = s.m_lin * L / 6.0
            Ir = Lam[e] @ np.diag([s.rhoJ, s.rhoI, s.rhoI]) @ Lam[e].T * L / 6.0
            for (a, b, w) in ((0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)):
                Me[e, 6*a:6*a+3, 6*b:6*b+3] = w * mt * np.eye(3)
                Me[e, 6*a+3:6*a+6, 6*b+3:6*b+6] = w * Ir
            edofs[e] = np.concatenate([dof_map[i], dof_map[j]]).astype(np.int32)

        self.nn = nn; self.ne = ne
        self.nodes_np = nodes; self.elems_np = elems
        self.dof_map_np = dof_map
        self.edofs_np = edofs
        self.Me_np = Me
        self.L0_np = L0; self.Lam0_np = Lam
        self.CF_np = CF; self.CM_np = CM
        self.device = device
        self.dX = wp.array(dX.astype(NP), dtype=VEC3, device=device)
        self.Lam0 = wp.array(Lam.astype(NP), dtype=MAT33, device=device)
        self.L0 = wp.array(L0.astype(NP), dtype=DTYPE, device=device)
        self.CF = wp.array(CF.astype(NP), dtype=VEC3, device=device)
        self.CM = wp.array(CM.astype(NP), dtype=VEC3, device=device)
        self.edofs = wp.array(edofs, dtype=wp.int32, device=device)
        self.Me = wp.array(Me.astype(NP), dtype=DTYPE, device=device)
        # free-DOF mask (1 free / 0 BC or prescribed)
        self.bc_dofs = set()
        self.free_np = np.ones(self.ndof, dtype=NP)
        self.free = wp.array(self.free_np, dtype=DTYPE, device=device)

    def set_bc(self, node_ids, fix_rot=True):
        """Clamp nodes (all 6 DOF, or translations only). Re-uploads the mask."""
        for n in node_ids:
            dofs = self.dof_map_np[n][:6 if fix_rot else 3]
            self.bc_dofs.update(int(d) for d in dofs)
        self.free_np = np.ones(self.ndof, dtype=config.NP_DTYPE)
        for d in self.bc_dofs:
            self.free_np[d] = 0.0
        self.free = wp.array(self.free_np, dtype=DTYPE, device=self.device)
        return self


def beam_internal_force(q_wp, C: Beam3DConstants, out=None, device=None):
    """Q_int (B, ndof) = assembled dE/dq."""
    device = device or C.device
    B = q_wp.shape[0]
    Q = out if out is not None else wp.zeros((B, C.ndof), dtype=DTYPE, device=device)
    if out is not None:
        Q.zero_()
    wp.launch(beam_force_kernel, dim=(B, C.ne),
              inputs=[q_wp, C.edofs, C.dX, C.Lam0, C.L0, C.CF, C.CM],
              outputs=[Q], device=device)
    return Q


def beam_energy_total(q_wp, C: Beam3DConstants, device=None):
    """Total elastic energy per env, (B,)."""
    device = device or C.device
    B = q_wp.shape[0]
    E = wp.zeros(B, dtype=DTYPE, device=device)
    wp.launch(beam_energy_kernel, dim=(B, C.ne),
              inputs=[q_wp, C.edofs, C.dX, C.Lam0, C.L0, C.CF, C.CM],
              outputs=[E], device=device)
    return E


def assemble_beam_kblocks(q_wp, C: Beam3DConstants, h_fd=1e-7, symmetrize=True,
                          device=None):
    """Consistent tangent blocks (B, ne, 12, 12) at q (central-difference of the
    closed-form force; symmetrized for the PCG operator)."""
    device = device or C.device
    NP = config.NP_DTYPE
    B = q_wp.shape[0]
    Kraw = wp.zeros((B, C.ne, 12, 12), dtype=DTYPE, device=device)
    wp.launch(beam_ktan_fd_kernel, dim=(B, C.ne, 12),
              inputs=[q_wp, C.edofs, C.dX, C.Lam0, C.L0, C.CF, C.CM, DTYPE(NP(h_fd))],
              outputs=[Kraw], device=device)
    if not symmetrize:
        return Kraw
    Kblk = wp.zeros((B, C.ne, 12, 12), dtype=DTYPE, device=device)
    wp.launch(_symmetrize_kernel, dim=(B, C.ne, 12, 12),
              inputs=[Kraw], outputs=[Kblk], device=device)
    return Kblk


def scatter_beam_global(Kblk_np, edofs_np, ndof):
    """Host scatter (ne,12,12) element blocks -> dense (ndof,ndof) (tests only)."""
    K = np.zeros((ndof, ndof))
    for el in range(Kblk_np.shape[0]):
        d = edofs_np[el]
        K[np.ix_(d, d)] += Kblk_np[el]
    return K


# ─── Beam Newmark step (dense batched solve) ─────────────────────────────────
# Same two-stage block-reduced scheme as batched_solver.gpu_newmark_step /
# modules/numerical_solver.step, but the inner S-solve uses the (already
# verified) batched dense LU instead of Jacobi-PCG: the beam tangent spans
# EA/L ~ 1e8 .. GJ/L ~ 1e2 (kappa ~ 1e6), which defeats Jacobi-PCG (measured
# 876 ms/step at ndof=102, CG at max_iter); dense LU: ndof ~ O(100) per env.

@wp.kernel
def _scatter_S_kernel(Me: wp.array(dtype=DTYPE, ndim=3),     # (ne,12,12)
                      Kblk: wp.array(dtype=DTYPE, ndim=4),   # (B,ne,12,12)
                      edofs: wp.array(dtype=wp.int32, ndim=2),
                      coef: DTYPE,
                      S: wp.array(dtype=DTYPE, ndim=3)):     # (B,ndof,ndof) accumulate
    e, el, a, b = wp.tid()
    wp.atomic_add(S, e, edofs[el, a], edofs[el, b],
                  Me[el, a, b] + coef * Kblk[e, el, a, b])


@wp.kernel
def _bc_rows_cols_kernel(free: wp.array(dtype=DTYPE, ndim=1),
                         S: wp.array(dtype=DTYPE, ndim=3)):
    """Zero BC rows/cols, unit diagonal: solve leaves BC entries of rhs (=0) fixed."""
    e, i, j = wp.tid()
    v = S[e, i, j] * free[i] * free[j]
    if i == j and free[i] < DTYPE(0.5):
        v = DTYPE(1.0)
    S[e, i, j] = v


def beam_newmark_step(q_n, dq_n, Kblk, C,
                      F_const, Q_int_n, recompute_Q,
                      alpha_v=0.5, c_damp=2.0, dt=1e-4, device=None):
    """One block-reduced Newmark step (matches numerical_solver.step semantics
    with Q_mem==0, Q_bend==full internal force -> trapezoidal averaging of Q).

    q_n, dq_n, F_const, Q_int_n: (B, ndof) wp arrays. recompute_Q(q_p1)->(B,ndof).
    Returns (q_new, dq_new). BC DOFs are held at q_n (prescribed handled by caller).
    C is duck-typed (Beam3DConstants, MembraneConstants, ...): needs
    ne/ndof/Me/edofs/free/device; block size taken from edofs.shape[1].
    """
    from .batched_solver import apply_MK, batched_dense_solve, _saxpy_kernel, \
        _lincomb_mask, _copy_free_else
    device = device or C.device
    NP = config.NP_DTYPE
    B = q_n.shape[0]
    ndof = C.ndof
    nblk = C.edofs.shape[1]
    coef = alpha_v * c_damp * dt * dt / 2.0
    Dbl = c_damp * dt / 2.0
    dtN = DTYPE(NP(dt)); adt = DTYPE(NP(alpha_v * dt))
    Me, edofs, free = C.Me, C.edofs, C.free

    S = wp.zeros((B, ndof, ndof), dtype=DTYPE, device=device)
    wp.launch(_scatter_S_kernel, dim=(B, C.ne, nblk, nblk),
              inputs=[Me, Kblk, edofs, DTYPE(NP(coef))], outputs=[S], device=device)
    wp.launch(_bc_rows_cols_kernel, dim=(B, ndof, ndof), inputs=[free], outputs=[S],
              device=device)
    tmp = wp.zeros((B, ndof), dtype=DTYPE, device=device)

    def solveA1(b1, b2):
        apply_MK(b1, tmp, Me, Kblk, edofs, free, 0.0, Dbl, device)
        rhs = wp.clone(b2)
        wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[rhs, DTYPE(-1.0), tmp], device=device)
        x2 = batched_dense_solve(S, rhs, device=device)
        x1 = wp.clone(b1)
        wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[x1, adt, x2], device=device)
        return x1, x2

    # homogeneous A2·X_n
    b1 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(_lincomb_mask, dim=(B, ndof),
              inputs=[b1, DTYPE(1.0), q_n, DTYPE(NP((1.0 - alpha_v) * dt)), dq_n, free],
              device=device)
    qf = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    dqf = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(_lincomb_mask, dim=(B, ndof), inputs=[qf, DTYPE(1.0), q_n, DTYPE(0.0), q_n, free], device=device)
    wp.launch(_lincomb_mask, dim=(B, ndof), inputs=[dqf, DTYPE(1.0), dq_n, DTYPE(0.0), dq_n, free], device=device)
    tmp2 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    apply_MK(qf, tmp, Me, Kblk, edofs, free, 0.0, Dbl, device)
    apply_MK(dqf, tmp2, Me, Kblk, edofs, free, 1.0, 0.0, device)
    b2 = wp.clone(tmp)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[b2, DTYPE(1.0), tmp2], device=device)
    a1, a2 = solveA1(b1, b2)

    zero = wp.zeros((B, ndof), dtype=DTYPE, device=device)

    # stage 0
    Qg = wp.clone(F_const)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[Qg, DTYPE(-1.0), Q_int_n], device=device)
    wp.launch(_lincomb_mask, dim=(B, ndof), inputs=[Qg, DTYPE(1.0), Qg, DTYPE(0.0), Qg, free], device=device)
    s01, s02 = solveA1(zero, Qg)
    qp1f = wp.clone(a1); wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[qp1f, dtN, s01], device=device)
    dqp1f = wp.clone(a2); wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[dqp1f, dtN, s02], device=device)
    q_p1 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(_copy_free_else, dim=(B, ndof), inputs=[q_p1, qp1f, q_n, free], device=device)

    # stage 1: trapezoidal average of the full internal force
    Q_p1 = recompute_Q(q_p1)
    Qg2 = wp.clone(F_const)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(-0.5), Q_int_n], device=device)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(-0.5), Q_p1], device=device)
    wp.launch(_lincomb_mask, dim=(B, ndof), inputs=[Qg2, DTYPE(1.0), Qg2, DTYPE(0.0), Qg2, free], device=device)
    s11, s12 = solveA1(zero, Qg2)
    qnf = wp.clone(a1); wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[qnf, dtN, s11], device=device)
    dqnf = wp.clone(a2); wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[dqnf, dtN, s12], device=device)
    q_new = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    dq_new = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(_copy_free_else, dim=(B, ndof), inputs=[q_new, qnf, q_n, free], device=device)
    wp.launch(_copy_free_else, dim=(B, ndof), inputs=[dq_new, dqnf, dq_n, free], device=device)
    return q_new, dq_new
