"""RoboEagle mixed beam-membrane wing (P2-S3): assembly + host Newmark + FSI entry.

Structure on the shared aero grid (node id = j*(nc+1)+i, robowing_real geometry):
  - main carbon spar  : spanwise chain at the grid line nearest 30% chord
  - aux  carbon spar  : spanwise chain at the grid line nearest 70% chord
  - 7 plywood ribs    : chordwise chains at ~(k+1)/8 span stations
  - membrane          : tension-field CST over every cell (2 tris/quad)
  - root (j=0)        : aluminum rib = prescribed rigid flapping frame
Coupling = conforming shared TRANSLATIONS (membrane edge nodes are beam nodes on
the spar/rib lines), psi numbered separately (membrane passes no moments) —
the parachute/sail/kite mainstream (docs/p2_s2_membrane_research.md §4).

DOF layout: translations 3n..3n+2 for every node n; rotations appended after
3*nn for beam nodes only. ndof = 3*nn + 3*n_beam_nodes.

Solver: same two-stage block-reduced Newmark as batched_solver.gpu_newmark_step /
kernels_beam3d.beam_newmark_step, but the inner S-solve uses HOST scipy splu
(mixed system ndof ~ 5e2: single-thread GPU dense LU measured 100x slower;
elements/forces/tangents stay on Warp). Equivalence vs the GPU reference is a
gate in p2_s3_wing.py.

Rib section: plywood strip 3mm wide; DEPTH is calibrated to the measured
chordwise stiffness K_MEAS (single measured quantity -> single constant, the
same sanctioned anchor calibrate_Ex used — but the compliance now lives in rib
BENDING instead of being smeared into the membrane modulus).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                            # noqa: E402
from fluxvortex.warp_fsi import config as cfg                # noqa: E402
from fluxvortex.warp_fsi import kernels_beam3d as kb         # noqa: E402
from fluxvortex.warp_fsi import kernels_membrane as km       # noqa: E402
import _v2_robogeom as rg                                    # noqa: E402

# membrane skin (Mylar, research-anchored: docs/p2_s2_membrane_research.md)
MEM_H, MEM_E, MEM_NU, MEM_RHO = 5e-5, 4e9, 0.3, 1390.0
N0_DEFAULT = 30.0
# plywood ribs (literature-anchored constants, as in _v2_flex_robo 3mat)
E_PLY, RHO_PLY, G_PLY = 8e9, 700.0, 0.6e9
RIB_WIDTH = 3e-3
K_MEAS_11B = 166.9          # measured chordwise stiffness (N/m) at (y=588.6mm, TE)
MEAS_Y, MEAS_X = 0.5886, 0.2735


@dataclass(frozen=True)
class RectSection:
    """Rectangular strip: width w (in-plane), depth d (out-of-plane, z)."""
    w: float; d: float; E: float; rho: float; G: float; ks: float = 5.0 / 6.0

    @property
    def A(self): return self.w * self.d
    @property
    def I2(self): return self.d * self.w**3 / 12.0        # bend about n(~z): in-plane
    @property
    def I3(self): return self.w * self.d**3 / 12.0        # bend about b(~y): out-of-plane
    @property
    def J(self):                                          # thin rectangle torsion
        a, b = max(self.w, self.d), min(self.w, self.d)
        return a * b**3 * (1.0 / 3.0 - 0.21 * b / a * (1 - b**4 / (12 * a**4)))
    @property
    def EA(self): return self.E * self.A
    @property
    def EI(self): return self.E * self.I3
    @property
    def EI2(self): return self.E * self.I2
    @property
    def EI3(self): return self.E * self.I3
    @property
    def GJ(self): return self.G * self.J
    @property
    def GAks(self): return self.G * self.A * self.ks
    @property
    def m_lin(self): return self.rho * self.A
    @property
    def rhoJ(self): return self.rho * (self.I2 + self.I3)
    @property
    def rhoI(self): return self.rho * self.I3
    @property
    def rhoI2(self): return self.rho * self.I2
    @property
    def rhoI3(self): return self.rho * self.I3


class WingModel:
    def __init__(self, nc=8, ns=16, N0=N0_DEFAULT, rib_depth=6e-3, device=None):
        self.nc, self.ns = nc, ns
        self.N0, self.rib_depth = float(N0), float(rib_depth)
        device = device or cfg.DEVICE
        C0 = rg.robowing_real(nc, ns)                     # (nc+1, ns+1, 3)
        nn = (nc + 1) * (ns + 1)
        nodes = np.zeros((nn, 3))
        for j in range(ns + 1):
            for i in range(nc + 1):
                nodes[j * (nc + 1) + i] = C0[i, j]
        self.nodes = nodes; self.nn = nn

        xf = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc + 1)))
        self.i_spar = int(np.argmin(np.abs(xf - 0.30)))
        self.i_aux = int(np.argmin(np.abs(xf - 0.70)))
        self.rib_js = sorted({int(round((k + 1) / 8 * ns)) for k in range(7)} - {0})

        nid = lambda i, j: j * (nc + 1) + i
        beam_elems, sections = [], []
        rib_sec = RectSection(RIB_WIDTH, self.rib_depth, E_PLY, RHO_PLY, G_PLY)
        for j in range(ns):                                # spanwise spars
            beam_elems.append([nid(self.i_spar, j), nid(self.i_spar, j + 1)])
            sections.append(kb.MAIN_SPAR)
            beam_elems.append([nid(self.i_aux, j), nid(self.i_aux, j + 1)])
            sections.append(kb.AUX_SPAR)
        for j in self.rib_js:                              # chordwise ribs
            for i in range(nc):
                beam_elems.append([nid(i, j), nid(i + 1, j)])
                sections.append(rib_sec)
        beam_elems = np.array(beam_elems)
        self.beam_nodes = sorted(set(beam_elems.ravel().tolist()))
        rot_rank = {n: r for r, n in enumerate(self.beam_nodes)}
        self.ndof = 3 * nn + 3 * len(self.beam_nodes)

        trans_map = np.arange(3 * nn, dtype=np.int64).reshape(nn, 3)
        dof6 = np.zeros((nn, 6), dtype=np.int64)
        dof6[:, :3] = trans_map
        for n, r in rot_rank.items():
            dof6[n, 3:] = 3 * nn + 3 * r + np.arange(3)
        self.trans_map, self.dof6 = trans_map, dof6

        # assembly assertions (research §4)
        for j in self.rib_js:                              # rib-spar crossings shared
            assert nid(self.i_spar, j) in rot_rank and nid(self.i_aux, j) in rot_rank
        tris = []
        for j in range(ns):
            for i in range(nc):
                p = nid(i, j)
                tris.append([p, p + 1, p + nc + 2])
                tris.append([p, p + nc + 2, p + nc + 1])
        tris = np.array(tris)
        assert tris.max() < nn and beam_elems.max() < nn  # conforming: shared node ids

        self.BeamC = kb.Beam3DConstants(nodes, beam_elems, sections,
                                        dof_map=dof6, ndof=self.ndof, device=device)
        self.MemC = km.MembraneConstants(nodes, tris, MEM_H, MEM_E, MEM_NU, MEM_RHO,
                                         N0, dof_map=trans_map, ndof=self.ndof,
                                         device=device)
        self.device = device
        # rotary-inertia mass scaling (x1e4): physical rotary inertia ~1e-11
        # kg m^2/node makes M^-1 K ~ 1e14 on psi DOFs — the linearly-implicit
        # two-stage scheme (implicitness ONLY via S = M + coef*K) is violently
        # unstable there at c_damp=2 (measured: one step -> 1e13). Scaling to
        # ~1e-7 kills the contrast; physical-mode pollution < 1% (rho*I*k^2
        # = 4e-4 << rho*A = 0.045 at the highest structural wavenumber).
        ridx = np.array([3, 4, 5, 9, 10, 11])
        self.BeamC.Me_np[:, np.ix_(ridx, ridx)[0], np.ix_(ridx, ridx)[1]] *= 1e4
        # constant mass (host sparse)
        self.M = (self._scatter(self.BeamC.Me_np, self.BeamC.edofs_np)
                  + self._scatter(self.MemC.Me_np, self.MemC.edofs_np)).tocsc()
        self.bc_dofs: set[int] = set()
        self.mass_report = dict(
            membrane=MEM_RHO * MEM_H * self.MemC.A0_np.sum(),
            spars=kb.MAIN_SPAR.m_lin * 0.8 + kb.AUX_SPAR.m_lin * 0.8,
            ribs=sum(rib_sec.m_lin * np.linalg.norm(nodes[b] - nodes[a])
                     for (a, b), s in zip(beam_elems, sections) if s is rib_sec))

    # ── assembly helpers ─────────────────────────────────────────────────────
    def _scatter(self, blocks, edofs):
        ne, nb, _ = blocks.shape
        rows = np.repeat(edofs, nb, axis=1).ravel()
        cols = np.tile(edofs, (1, nb)).ravel()
        return coo_matrix((blocks.ravel(), (rows, cols)),
                          shape=(self.ndof, self.ndof))

    def _wpq(self, q_np):
        return wp.array(np.ascontiguousarray(q_np[None, :], dtype=cfg.NP_DTYPE),
                        dtype=cfg.DTYPE, device=self.device)

    def Q_int(self, q_np):
        qw = self._wpq(q_np)
        Qb = kb.beam_internal_force(qw, self.BeamC).numpy()[0]
        Qm = km.membrane_internal_force(qw, self.MemC).numpy()[0]
        return Qb + Qm

    def K_csc(self, q_np, symmetrize=True, latch=True):
        """Assembled tangent. latch=True (default) freezes each membrane
        element's branch from the CURRENT state before the FD tangent: at
        equilibrium the free-edge elements sit exactly ON the wrinkle boundary
        (n2 -> 0+ is the physical answer), so un-latched FD columns straddle
        branches and the symmetrized mixture is no tangent of any energy
        (measured: fake -375 Hz modes; corrupted damping operator -> dynamic
        chatter pumping). The latched tangent is the one-sided consistent
        Hessian — PSD by the relaxed-energy quasiconvexity. Forces stay
        branch-free (exact, C0)."""
        qw = self._wpq(q_np)
        Kb = kb.assemble_beam_kblocks(qw, self.BeamC, symmetrize=symmetrize).numpy()[0]
        if latch and bool((self.MemC.branch_np == -1).all()):  # respect existing latch/weights
            st = km.membrane_state(qw, self.MemC).numpy()[0]
            e1, e2 = st[:, 0], st[:, 1]
            nf2 = self.MemC.N0 + self.MemC.hb * (e2 + self.MemC.nu * e1)
            e1sl = -self.MemC.N0 * (1 - self.MemC.nu) / self.MemC.hE
            ids = np.where(nf2 > 0, 0, np.where(e1 > e1sl, 1, 2)).astype(np.int32)
            saved = self.MemC.branch_np.copy()
            self.MemC.freeze_branches(ids)
            try:
                Km_ = km.assemble_membrane_kblocks(qw, self.MemC,
                                                   symmetrize=symmetrize).numpy()[0]
            finally:
                self.MemC.freeze_branches(saved)
        else:
            Km_ = km.assemble_membrane_kblocks(qw, self.MemC,
                                               symmetrize=symmetrize).numpy()[0]
        return (self._scatter(Kb, self.BeamC.edofs_np)
                + self._scatter(Km_, self.MemC.edofs_np)).tocsc()

    def set_bc(self, dofs):
        self.bc_dofs.update(int(d) for d in dofs)
        self.free = np.array(sorted(set(range(self.ndof)) - self.bc_dofs))
        return self

    def clamp_root(self):
        """Static clamp of the aluminum root rib: j=0 translations + spar-root psi."""
        dofs = []
        for n in range(self.nc + 1):                       # j=0 row: node id == i
            dofs += list(self.trans_map[n])
        for n in (self.i_spar, self.i_aux):                # spar root psi
            dofs += list(self.dof6[n, 3:])
        return self.set_bc(dofs)

    # ── analysis ─────────────────────────────────────────────────────────────
    def _latch_soft(self, q_np, k_soft=25.0):
        """Freeze CONTINUOUS branch weights from the state at q (sigmoid over
        the wrinkle criteria, band ~4%% N0). Shared by statics and dynamics."""
        from fluxvortex.warp_fsi import kernels_membrane as _km
        st = _km.membrane_state(self._wpq(q_np), self.MemC).numpy()[0]
        e1, e2 = st[:, 0], st[:, 1]
        a = k_soft * (self.MemC.N0 + self.MemC.hb * (e2 + self.MemC.nu * e1)) / self.MemC.N0
        b = k_soft * (self.MemC.N0 * (1 - self.MemC.nu) + self.MemC.hE * e1) \
            / (self.MemC.N0 * (1 - self.MemC.nu))
        s_t = 1.0 / (1.0 + np.exp(-np.clip(a, -40, 40)))
        s_w = 1.0 / (1.0 + np.exp(-np.clip(b, -40, 40)))
        self.MemC.set_soft_weights(np.stack([s_t, (1 - s_t) * s_w,
                                             (1 - s_t) * (1 - s_w)], axis=1))
        return self

    def static_newton(self, f_const, q0=None, load_steps=2, tol=1e-6, max_outer=10):
        """Latched-branch static solve under dead load f_const (same IMP state-
        latching as pre_equilibrate: plain Newton chatters on the flipping
        wrinkle tangent). Load-stepped for robustness."""
        from fluxvortex.warp_fsi import kernels_membrane as _km
        q = np.zeros(self.ndof) if q0 is None else q0.copy()
        free = self.free
        hb, hE, nu, N0 = self.MemC.hb, self.MemC.hE, self.MemC.nu, self.MemC.N0
        e1sl = -N0 * (1 - nu) / hE
        scale = max(1.0, np.linalg.norm(f_const))
        rn = np.nan
        for lam in np.linspace(1.0 / load_steps, 1.0, load_steps):
            for outer in range(max_outer):
                self._latch_soft(q)
                try:
                    for it in range(12):
                        r = lam * f_const - self.Q_int(q)
                        rn = np.linalg.norm(r[free])
                        if rn < 0.1 * tol * scale:
                            break
                        if not np.isfinite(rn) or rn > 1e8 * scale:
                            raise RuntimeError(f"static Newton diverged at lam={lam}")
                        # mu-regularized Newton + step cap: the eta residual-
                        # compression leaves a tiny NEGATIVE curvature direction
                        # (~ -0.1 N/m) in the latched K — plain Newton oscillates
                        # along it; mu=2 N/m is invisible to physical directions.
                        from scipy.sparse import identity as _eye
                        K = self.K_csc(q, symmetrize=False, latch=False)
                        Kff = K[free][:, free].tocsc() + 2.0 * _eye(len(free), format="csc")
                        dq = np.zeros(self.ndof)
                        dq[free] = splu(Kff).solve(r[free])
                        step = min(1.0, 5e-3 / max(np.abs(dq).max(), 1e-30))
                        q += step * dq
                finally:
                    self.MemC.unfreeze_branches()
                if rn < tol * scale:
                    break
        rn = np.linalg.norm((f_const - self.Q_int(q))[free])
        return q, rn

    def pre_equilibrate(self, tol=1e-6, max_outer=15, verbose=False):
        """Relax the uniform-N0 free-edge imbalance to static equilibrium
        (tension-field 'form-finding lite': TE/LE strips redistribute/wrinkle).
        REQUIRED before any dynamics: at q=0 the free-edge residual (~10 N) on
        mg-scale edge nodes is an impulsive load no integrator survives (the
        cold-start lesion of docs/p2_step0_3mat_probe.md, pretension-driven).

        Method: IMP-style STATE-LATCHED Newton (research 决策点1: freeze each
        element's taut/wrinkled/slack branch -> smooth C2 problem -> Newton ->
        re-assign branches -> repeat to a fixed point). Plain Newton and damped/
        kinetic pseudo-dynamics both fail here: free-edge elements sit exactly
        ON the wrinkle boundary and the flipping tangent chatters/pumps.
        """
        from fluxvortex.warp_fsi import kernels_membrane as _km
        q = np.zeros(self.ndof)
        free = self.free
        r0 = np.linalg.norm(self.Q_int(q)[free])
        rn = r0
        prev_ids = None
        n_outer = 0
        hb, hE, nu, N0 = self.MemC.hb, self.MemC.hE, self.MemC.nu, self.MemC.N0
        e1sl = -N0 * (1 - nu) / hE
        try:
            from scipy.sparse import identity as _eye
            for outer in range(max_outer):
                n_outer = outer + 1
                if rn < tol * r0 and outer > 1:
                    break
                self._latch_soft(q)
                for it in range(15):                     # Newton on the smooth problem
                    r = -self.Q_int(q)
                    rn = np.linalg.norm(r[free])
                    if rn < 0.1 * tol * r0:
                        break
                    if not np.isfinite(rn) or rn > 1e8 * r0:
                        raise RuntimeError("state-latched Newton diverged")
                    K = self.K_csc(q, symmetrize=False, latch=False)
                    Kff = K[free][:, free].tocsc() + 2.0 * _eye(len(free), format="csc")
                    dq = np.zeros(self.ndof)
                    dq[free] = splu(Kff).solve(r[free])
                    step = min(1.0, 5e-3 / max(np.abs(dq).max(), 1e-30))
                    q += (0.5 if outer == 0 and it == 0 else step) * dq
                if verbose:
                    print(f"  pre-eq outer{outer}: |r|={rn:.3e} "
                          f"branches taut/wr/sl = {(ids==0).sum()}/{(ids==1).sum()}/{(ids==2).sum()}")
        finally:
            self.MemC.unfreeze_branches()

        # phase 2: saddle escape. At low N0 the CAMBERED free LE strip's
        # symmetric equilibrium is a saddle (tension on an arch = negative
        # geometric stiffness for the flattening mode; measured -375/-110/-18 Hz
        # at N0=30, all supported on the LE strip). Kick along the unstable
        # mode, settle with damped dynamics, re-polish with latched Newton.
        from scipy.linalg import eigh as _eigh
        n_escape = 0
        for _try in range(3):
            K = self.K_csc(q, symmetrize=True).toarray()
            Md = self.M.toarray()
            w2, V = _eigh(K[np.ix_(free, free)], Md[np.ix_(free, free)],
                          subset_by_index=[0, 0])
            # trigger only BEYOND the eta-artifact band (|f| <~ 25 Hz small
            # negatives at wrinkled free-edge slivers are a documented artifact,
            # dynamically benign); GPU-atomic nondeterminism makes marginal
            # triggers flip run-to-run.
            if w2[0] >= -(2 * np.pi * 25.0) ** 2:
                break
            n_escape += 1
            q_before = q.copy()
            rn_before = np.linalg.norm(self.Q_int(q)[free])
            v = np.zeros(self.ndof); v[free] = V[:, 0]
            v /= max(np.abs(v).max(), 1e-30)
            # 1-D energy scan along the unstable mode (both signs): the saddle
            # sits between post-snap wells; dynamics is NOT used here (branch-
            # flipping states poison the two-stage stepper — measured).
            def Wtot(qv):
                qw = self._wpq(qv)
                return float(km.membrane_energy_total(qw, self.MemC).numpy()[0]
                             + kb.beam_energy_total(qw, self.BeamC).numpy()[0])
            best_s, best_W = 0.0, Wtot(q)
            for s in np.concatenate([np.linspace(-8e-3, -2e-4, 24),
                                     np.linspace(2e-4, 8e-3, 24)]):
                Ws = Wtot(q + s * v)
                if Ws < best_W:
                    best_s, best_W = s, Ws
            if best_s == 0.0:
                # no energy descent along the "negative" mode: a small negative
                # tangent eigenvalue from the eta residual-compression geometric
                # stiffness at strongly compressed free-edge slivers (documented
                # artifact, dynamically benign) — the state is a true minimum.
                break
            q = q + best_s * v
            # latched-Newton re-polish about the settled state
            from scipy.sparse import identity as _eye2
            for outer in range(8):
                self._latch_soft(q)
                try:
                    for it in range(12):
                        r = -self.Q_int(q)
                        rn = np.linalg.norm(r[free])
                        if rn < 0.1 * tol * r0:
                            break
                        K2 = self.K_csc(q, symmetrize=False, latch=False)
                        Kff2 = K2[free][:, free].tocsc() + 2.0 * _eye2(len(free), format="csc")
                        dqn = np.zeros(self.ndof)
                        dqn[free] = splu(Kff2).solve(r[free])
                        q += dqn
                finally:
                    self.MemC.unfreeze_branches()
                if rn < tol * r0:
                    break
            if np.linalg.norm(self.Q_int(q)[free]) > rn_before:   # escape made it worse
                q = q_before
                break

        # residual bookkeeping: rn_soft = the solved (continuous-latched)
        # problem's own residual; rn_sharp = sharp-criterion evaluation at the
        # soft solution — its floor is the sigmoid band mismatch (~4% N0 forces
        # on boundary elements), NOT a convergence failure. Report both.
        self._latch_soft(q)
        try:
            rn_soft = np.linalg.norm(self.Q_int(q)[free])
        finally:
            self.MemC.unfreeze_branches()
        rn_sharp = np.linalg.norm(self.Q_int(q)[free])
        st = _km.membrane_state(self._wpq(q), self.MemC).numpy()[0]
        n_wr = int(np.sum(st[:, 3] <= self.N0 * 1e-3))
        return q, dict(resid=rn_soft, resid_sharp=rn_sharp, resid0=r0,
                       umax=float(np.abs(q).max()),
                       n_wrinkled=n_wr, ne=self.MemC.ne, iters_outer=n_outer,
                       n_escape=n_escape)

    def modal(self, q0=None, k=8, soft=True):
        """Modes about q0. soft=True (default): consistent tangent of the
        continuous-latched (working) formulation — hard-latching at a soft
        equilibrium mis-assigns borderline sliver elements and manufactures
        spurious large negatives (measured -495 Hz)."""
        from scipy.linalg import eigh
        q = np.zeros(self.ndof) if q0 is None else q0
        if soft:
            self._latch_soft(q)
            try:
                K = self.K_csc(q, symmetrize=True, latch=False).toarray()
            finally:
                self.MemC.unfreeze_branches()
        else:
            K = self.K_csc(q, symmetrize=True).toarray()
        M = self.M.toarray()
        free = self.free
        w2, V = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
                     subset_by_index=[0, k - 1])
        phi = np.zeros((self.ndof, k)); phi[free] = V
        # SIGNED frequencies: small negatives (|f| <~ 20 Hz) are the eta
        # residual-compression geometric-stiffness artifact on wrinkled
        # free-edge slivers (dynamically benign; energy-scan verified minimum).
        return np.sign(w2) * np.sqrt(np.abs(w2)) / (2 * np.pi), phi

    def chordwise_k(self, F=0.5):
        """Simulated measured-protocol stiffness: clamp root, z-load the node
        nearest the paper's measurement point (y=588.6mm, x=273.5mm ~ TE)."""
        d2 = (self.nodes[:, 1] - MEAS_Y)**2 + (self.nodes[:, 0] - MEAS_X)**2
        n_load = int(np.argmin(d2))
        zdof = self.trans_map[n_load][2]
        f = np.zeros(self.ndof); f[zdof] = F
        K = self.K_csc(np.zeros(self.ndof), symmetrize=False)
        free = self.free
        d = splu(K[free][:, free].tocsc()).solve(f[free])
        delta = d[np.where(free == zdof)[0][0]]
        return F / delta, n_load


def host_newmark_step(model, q, dq, dt, F_const,
                      presc=None, alpha_v=0.5, c_damp=2.0):
    """Two-stage block-reduced Newmark (verbatim port of beam_newmark_step math,
    scipy splu inner solve). presc = (pd, qb, dqb, ddqb) with values at t_end.
    Q_mem==0 mapping -> trapezoidal averaging of the full internal force.

    CONTINUOUS latching: smooth branch weights (sigmoid over the wrinkle
    criteria, band ~4% N0) are frozen at q_n for the WHOLE step — variational
    within the step, continuous across steps; the step-to-step taut<->wrinkled
    tangent jump that pumps this integrator (measured blow-ups) cannot occur.
    """
    ndof = model.ndof
    free = model.free
    coef = alpha_v * c_damp * dt * dt / 2.0
    Dbl = c_damp * dt / 2.0
    MemC = getattr(model, "MemC", None)
    if MemC is not None:
        from fluxvortex.warp_fsi import kernels_membrane as _km
        st = _km.membrane_state(model._wpq(q), MemC).numpy()[0]
        e1, e2 = st[:, 0], st[:, 1]
        k_soft = 25.0
        a = k_soft * (MemC.N0 + MemC.hb * (e2 + MemC.nu * e1)) / MemC.N0
        b = k_soft * (MemC.N0 * (1 - MemC.nu) + MemC.hE * e1) / (MemC.N0 * (1 - MemC.nu))
        s_t = 1.0 / (1.0 + np.exp(-np.clip(a, -40, 40)))
        s_w = 1.0 / (1.0 + np.exp(-np.clip(b, -40, 40)))
        MemC.set_soft_weights(np.stack([s_t, (1 - s_t) * s_w,
                                        (1 - s_t) * (1 - s_w)], axis=1))
    try:
        K = model.K_csc(q, symmetrize=True, latch=False)   # weights already set
        M = model.M
        S = (M + coef * K)[free][:, free].tocsc()
        # symmetric Jacobi equilibration: the wing S spans ~10 decades (rib
        # free-end torsion DOFs) — pure scaling, but superlu's pivot threshold
        # misreads it as 'exactly singular'.
        dsc = 1.0 / np.sqrt(np.maximum(S.diagonal(), 1e-300))
        from scipy.sparse import diags
        Dsc = diags(dsc)
        lu = splu((Dsc @ S @ Dsc).tocsc())
        solve_S = lambda bb: dsc * lu.solve(dsc * bb)
        mask = np.zeros(ndof); mask[free] = 1.0

        if presc is not None:
            pd, qb, dqb, ddqb = presc
            F_const = F_const - (M[:, pd] @ ddqb)          # hook 1: inertial coupling

        def solveA1(b1, b2):
            rhs = (b2 - Dbl * (K @ b1))
            x2 = np.zeros(ndof); x2[free] = solve_S(rhs[free])
            return b1 + alpha_v * dt * x2, x2

        qf, dqf = mask * q, mask * dq
        b1 = mask * (q + (1 - alpha_v) * dt * dq)
        b2 = Dbl * (K @ qf) + M @ dqf
        a1, a2 = solveA1(b1, b2)

        Qn = model.Q_int(q)
        Qg = mask * (F_const - Qn)
        s01, s02 = solveA1(np.zeros(ndof), Qg)
        q_p1 = q.copy(); q_p1[free] = (a1 + dt * s01)[free]
        if presc is not None:
            q_p1[pd] = qb                                  # hook 2: scatter q_b(t_end)
        Q_p1 = model.Q_int(q_p1)

        Qg2 = mask * (F_const - 0.5 * Qn - 0.5 * Q_p1)
        s11, s12 = solveA1(np.zeros(ndof), Qg2)
        q_new = q.copy(); dq_new = dq.copy()
        q_new[free] = (a1 + dt * s11)[free]
        dq_new[free] = (a2 + dt * s12)[free]
        if presc is not None:                              # hook 3: writeback
            q_new[pd] = qb; dq_new[pd] = dqb
    finally:
        if MemC is not None:
            MemC.unfreeze_branches()
    return q_new, dq_new


def host_implicit_step(model, q, dq, a, dt, F_const, presc=None,
                       gamma=0.6, newton_tol=1e-8, max_newton=8, beta_R=1e-3,
                       M_add=None):
    """Fully implicit Newmark-beta (average-acceleration family, gamma=0.6 for
    algorithmic high-frequency damping — the generalized-alpha-class dissipation
    the membrane-FSI literature recommends) with Newton iterations per substep.

    Adopted for the WING driver after the linearly-implicit two-stage scheme hit
    its validity edge on this system: under continuous root excitation the
    frozen-K damping operator mismatches the rotating membrane geometric
    stiffness and pumps a dt-independent instability (measured: c_damp=2/4/8 all
    diverge, only delayed). Beam/membrane elements, gates and the aero
    production path are untouched.

    Soft-latched branch weights are frozen at q_n for the whole substep
    (variational within the step, smooth across steps). Prescribed DOFs carry
    their exact (qb, dqb, ddqb) — inertial coupling rides M @ a naturally.
    Returns (q1, dq1, a1).
    """
    from scipy.sparse import diags, identity as _eye
    ndof = model.ndof
    free = model.free
    beta = 0.25 * (gamma + 0.5) ** 2
    MemC = getattr(model, "MemC", None)
    if MemC is not None:
        model._latch_soft(q)
    try:
        M = model.M
        if M_add is not None:
            # added-mass operator: M_eff = M - dF_aero/da (the MATLAB Qf_p_mat
            # treatment; mandatory here — membrane added-mass ratio ~5 makes the
            # loose two-pass PC otherwise unstable, exactly as the research
            # (决策点3) and the measured w~10 aero blow-up say).
            M = M - M_add
        q_pred = q + dt * dq + dt * dt * (0.5 - beta) * a
        dq_pred = dq + dt * (1 - gamma) * a
        q1 = q_pred.copy()
        if presc is not None:
            pd, qb, dqb, ddqb = presc
            q1[pd] = qb
        a1 = (q1 - q_pred) / (beta * dt * dt)
        if presc is not None:
            a1[pd] = ddqb
        scale = max(1.0, float(np.linalg.norm(F_const)))
        # beta_R: stiffness-proportional structural damping (membrane material
        # loss + joints; zeta ~ 1% at 16 Hz). The undamped wing rings up to
        # repeated deep-wrinkle excursions and eventually an unconvergeable
        # substep. Rayleigh C = beta_R * K(q_n) FROZEN for the step (a C that
        # tracks K(q1) makes Newton chase a moving residual — measured).
        Cd = beta_R * model.K_csc(q, symmetrize=True, latch=False)
        for it in range(max_newton):
            v1 = dq_pred + gamma * dt * a1
            r = F_const - model.Q_int(q1) - M @ a1 - Cd @ v1
            rn = np.linalg.norm(r[free])
            if not np.isfinite(rn):
                rn = np.inf                                  # reject cleanly
                break
            if rn < newton_tol * scale:
                break
            K = model.K_csc(q1, symmetrize=True, latch=False)
            S = (K + M / (beta * dt * dt)
                 + (gamma / (beta * dt)) * Cd)[free][:, free].tocsc()
            dsc = 1.0 / np.sqrt(np.maximum(S.diagonal(), 1e-300))
            Dsc = diags(dsc)
            try:
                lu = splu((Dsc @ S @ Dsc).tocsc())
                dq1 = dsc * lu.solve(dsc * r[free])
            except RuntimeError:                             # singular from a bad state
                rn = np.inf
                break
            q1[free] += dq1
            a1 = (q1 - q_pred) / (beta * dt * dt)
            if presc is not None:
                a1[pd] = ddqb
        converged = bool(np.isfinite(rn) and rn < 1e-4 * scale)
        dq1_vec = dq_pred + gamma * dt * a1
        if presc is not None:
            q1[pd] = qb
            dq1_vec[pd] = dqb
    finally:
        if MemC is not None:
            MemC.unfreeze_branches()
    return q1, dq1_vec, a1, converged


class WingEntry:
    """StructuralEntry for the mixed wing: prescribed root flapping frame with
    smooth amplitude ramp (S4 cold start), host Newmark inner stepper."""

    def __init__(self, model: WingModel, kin, ramp_T=0.0, load_ramp_T=0.0):
        self.m = model
        self.kin = kin
        self.ramp_T = float(ramp_T)
        self.load_ramp_T = float(load_ramp_T)   # C1 aero-load ramp-in (cold start)
        nc = model.nc
        root_nodes = list(range(nc + 1))                    # j=0 row
        pd = []
        for n in root_nodes:
            pd += list(model.trans_map[n])
        for n in (model.i_spar, model.i_aux):               # spar roots carry psi
            pd += list(model.dof6[n, 3:])
        self.pd = np.array(sorted(pd))
        model.set_bc(self.pd)                               # held by the solver mask
        self._X_root = model.nodes[root_nodes]              # (nc+1, 3)
        self._spar_psi_slots = [np.where(self.pd == model.dof6[n, 3])[0][0]
                                for n in (model.i_spar, model.i_aux)]
        # start from the pretension equilibrium (free-edge relaxation), never
        # from the raw uniform-N0 state — see WingModel.pre_equilibrate.
        self.q, self.preeq_info = model.pre_equilibrate()
        self.dq = np.zeros(model.ndof)
        self.a = np.zeros(model.ndof)
        self.t = 0.0
        qb, dqb, _ = self._cb(0.0)
        self.q[self.pd] = qb; self.dq[self.pd] = dqb

    def _angles(self, t):
        th, thd, thdd = self.kin.angles(t)
        if self.ramp_T > 0 and t < self.ramp_T:             # C1 cosine envelope
            w = np.pi / self.ramp_T
            r = 0.5 * (1 - np.cos(w * t))
            rd = 0.5 * w * np.sin(w * t)
            rdd = 0.5 * w * w * np.cos(w * t)
            return (r * th, r * thd + rd * th,
                    r * thdd + 2 * rd * thd + rdd * th)
        return th, thd, thdd

    def _cb(self, t):
        th, thd, thdd = self._angles(t)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        Rp = np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]])
        Rpp = np.array([[0, 0, 0], [0, -c, s], [0, -s, -c]])
        X = self._X_root
        u = (X @ (R - np.eye(3)).T).ravel()
        du = thd * (X @ Rp.T).ravel()
        ddu = (thdd * (X @ Rp.T) + thd**2 * (X @ Rpp.T)).ravel()
        npd = len(self.pd)
        qb = np.zeros(npd); dqb = np.zeros(npd); ddqb = np.zeros(npd)
        ntr = 3 * len(X)
        qb[:ntr], dqb[:ntr], ddqb[:ntr] = u, du, ddu
        for slot in self._spar_psi_slots:                   # single-axis: psi=(th,0,0) exact
            qb[slot] = th; dqb[slot] = thd; ddqb[slot] = thdd
        return qb, dqb, ddqb

    # protocol ---------------------------------------------------------------
    def snapshot(self):
        return (self.t, self.q.copy(), self.dq.copy(), self.a.copy())

    def restore(self, snap):
        self.t, q, dq, a = snap
        self.q = q.copy(); self.dq = dq.copy(); self.a = a.copy()

    def substep(self, t, dt, forces=None):
        f = np.zeros(self.m.ndof)
        M_add = None
        if forces is not None:
            f9 = np.asarray(forces.f).reshape(-1, 9)        # provider 9-dof layout
            f[self.m.trans_map.ravel()] = f9[:, 0:3].ravel()
            ramp = 1.0
            if self.load_ramp_T > 0 and t < self.load_ramp_T:   # impulsive-start
                ramp = 0.5 * (1 - np.cos(np.pi * t / self.load_ramp_T))  # absorber
                f *= ramp
            if getattr(forces, "madd", None) is not None:
                from scipy.sparse import coo_matrix as _coo
                nn = self.m.nn
                zidx9 = 9 * np.arange(nn) + 2
                Mzz = ramp * np.asarray(forces.madd)[np.ix_(zidx9, zidx9)]
                zd = self.m.trans_map[:, 2]
                rows = np.repeat(zd, nn); cols = np.tile(zd, nn)
                M_add = _coo((Mzz.ravel(), (rows, cols)),
                             shape=(self.m.ndof, self.m.ndof)).tocsc()
        self._advance(t - dt, t, dt, f, M_add, depth=0)
        self.t = t
        return self

    def _advance(self, t0, t1, dt, f, M_add, depth):
        """Implicit substep with adaptive halving: an unconverged Newton substep
        (acute wrinkle/sliver events at flapping speed) must never be accepted."""
        qb, dqb, ddqb = self._cb(t1)
        try:
            q1, dq1, a1, ok = host_implicit_step(
                self.m, self.q, self.dq, self.a, t1 - t0, f,
                presc=(self.pd, qb, dqb, ddqb), M_add=M_add)
        except RuntimeError:
            ok, q1 = False, self.q                           # treat as unconverged
        if ok and np.all(np.isfinite(q1)):
            self.q, self.dq, self.a = q1, dq1, a1
            return
        if depth >= 4:
            raise RuntimeError(f"substep failed to converge at t={t1:.4f} (depth {depth})")
        tm_ = 0.5 * (t0 + t1)
        self._advance(t0, tm_, dt, f, M_add, depth + 1)
        self._advance(tm_, t1, dt, f, M_add, depth + 1)

    def state(self):
        nc, ns = self.m.nc, self.m.ns
        u = self.q[self.m.trans_map.ravel()].reshape(-1, 3)
        du = self.dq[self.m.trans_map.ravel()].reshape(-1, 3)
        verts = (self.m.nodes + u).reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        vels = du.reshape(ns + 1, nc + 1, 3).transpose(1, 0, 2)
        return dict(verts=verts.copy(), vels=vels.copy())
