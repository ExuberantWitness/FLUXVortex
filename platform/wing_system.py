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
import wing_mesh as wmesh                                    # noqa: E402
import wing_mass as wmass                                    # noqa: E402

# membrane skin (Mylar, research-anchored: docs/p2_s2_membrane_research.md)
MEM_H, MEM_E, MEM_NU, MEM_RHO = 5e-5, 4e9, 0.3, 1390.0
N0_DEFAULT = 30.0
# plywood ribs (literature-anchored constants, as in _v2_flex_robo 3mat)
E_PLY, RHO_PLY, G_PLY = 8e9, 700.0, 0.6e9
RIB_WIDTH = 3e-3
K_MEAS_11B = 166.9          # measured chordwise stiffness (N/m) at (y=588.6mm, TE)
MEAS_Y, MEAS_X = 0.5886, 0.2735
# LE rod: paper-recorded member ("8 mm carbon rod spar"); SOLID assumed (user
# decision 2026-07-06; if the real one is a tube, supply the wall here)
LE_SPAR = kb.TubeSection(D_out=8e-3, wall=4e-3, E=150e9, rho=1592.0, G=5e9)
# TE hem: kite-fabric construction ALWAYS hems the raw trailing edge (double
# fold + stitch; tension-stiff, bending-soft). Missing it leaves bare-edge
# membrane slivers whose free-edge flutter aliases through the window
# coupling (measured 2-window sawtooth at the max-speed stroke phases).
# 15 mm double fold of the polyester fabric: EA ~ 3.6 kN, out-of-plane EI
# ~ 8e-6 N m^2 (floppy), 1.3 g/m — zero-fit fabric constants (hem_sec below).


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
        # flat structural mesh (assembly v3): z=0, three STRAIGHT rods each
        # extended to the planform edge arc, tip pockets re-triangulated so the
        # rod tails are membrane edges (all asserted inside flat_wing_mesh)
        mesh = wmesh.flat_wing_mesh(nc, ns)
        self.mesh = mesh
        nodes, tris = mesh["nodes"], mesh["tris"]
        nn = len(nodes)
        self.nodes = nodes; self.nn = nn
        self.nid_grid = mesh["nid_grid"]                   # (ns+1, nc+1) grid ids
        self.i_le, self.i_spar, self.i_aux = 0, 3, 5       # rod grid columns
        self.rib_js = mesh["rib_js"]

        chains = mesh["chains"]
        beam_elems, sections = [], []
        rib_sec = RectSection(RIB_WIDTH, self.rib_depth, E_PLY, RHO_PLY, G_PLY)
        hem_sec = RectSection(15e-3, 1.6e-4, 1.5e9, 525.0, 5e8)  # TE hem (fabric x2)
        for name, sec in (("le", LE_SPAR), ("main", kb.MAIN_SPAR),
                          ("aux", kb.AUX_SPAR), ("te", hem_sec)):
            ch = chains[name]
            for a, b in zip(ch[:-1], ch[1:]):
                beam_elems.append([a, b]); sections.append(sec)
        for r in chains["ribs"]:                           # chordwise ribs
            for a, b in zip(r[:-1], r[1:]):
                beam_elems.append([a, b]); sections.append(rib_sec)
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

        nid = lambda i, j: j * (nc + 1) + i
        for j in self.rib_js:                              # rib-rod crossings shared
            assert all(nid(i_, j) in rot_rank
                       for i_ in (self.i_le, self.i_spar, self.i_aux))
        assert tris.max() < nn and beam_elems.max() < nn  # conforming: shared node ids

        # ── aero surface: DECOUPLED from the structural pinned mesh ─────────
        # The structural grid lines are pinned to the straight rods; in the
        # tip taper that produces strongly skewed/twisted aero panels whose
        # AIC rows are ill-conditioned (measured: gamma at the taper TE panel
        # doubling per window, 4 -> 1246, local force blow-up). The AERO
        # lattice therefore keeps the healthy COSINE-FRACTION grid (the
        # S1-exit-validated panel geometry); structure <-> aero exchange goes
        # through work-consistent barycentric interpolation W precomputed on
        # the rest planform (both meshes are material-fixed):
        #   pos_aero = W @ pos_struct  |  f_struct = W^T @ f_aero.
        from scipy.sparse import csr_matrix, kron as _spkron, identity as _spI
        xf = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc + 1)))
        ya = np.linspace(0.0, rg.HALF_SPAN, ns + 1)
        ca = rg.chord_at(ya)
        ar = np.zeros((ns + 1, nc + 1, 2))
        ar[..., 0] = xf[None, :] * ca[:, None]
        ar[..., 1] = ya[:, None]
        self.aero_rest2d = ar                              # (ns+1, nc+1, 2)
        self.aero_off = rg.naca_camber(np.tile(xf, (ns + 1, 1))) * ca[:, None]
        nn_a = (nc + 1) * (ns + 1)
        p2d = nodes[:, :2]
        Wr, Wc, Wv = [], [], []
        t0_, t1_, t2_ = tris[:, 0], tris[:, 1], tris[:, 2]
        v0 = p2d[t1_] - p2d[t0_]
        v1 = p2d[t2_] - p2d[t0_]
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        for pa in range(nn_a):
            pt = ar.reshape(-1, 2)[pa]
            d = pt[None, :] - p2d[t0_]
            l1 = (d[:, 0] * v1[:, 1] - d[:, 1] * v1[:, 0]) / den
            l2 = (v0[:, 0] * d[:, 1] - v0[:, 1] * d[:, 0]) / den
            l0 = 1.0 - l1 - l2
            ok = np.where((l0 > -1e-9) & (l1 > -1e-9) & (l2 > -1e-9))[0]
            assert len(ok), f"aero node {pa} outside structural mesh"
            e = ok[np.argmax(np.minimum(np.minimum(l0[ok], l1[ok]), l2[ok]))]
            lam = np.clip([l0[e], l1[e], l2[e]], 0.0, None)
            lam = lam / lam.sum()
            for nd_, lv in zip(tris[e], lam):
                if lv > 1e-12:
                    Wr.append(pa); Wc.append(int(nd_)); Wv.append(float(lv))
        self.W_a2s = csr_matrix((Wv, (Wr, Wc)), shape=(nn_a, nn))
        self.W3 = _spkron(self.W_a2s, _spI(3, format="csr"), format="csr")

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
        # strip-theory added mass (flat-plate potential flow, m_a = rho pi c^2/4
        # per unit span — analytic zero-fit constant). Node share = tributary
        # membrane area / local chord. The provider's exact AIC-based madd is
        # measured INDEFINITE (docs/p2_s4_fsi.md) — this PSD physical operator
        # is the implicit added-mass stabilizer instead.
        rho_air = 1.225
        A_trib = np.zeros(nn)
        for e, tri in enumerate(tris):
            for nd_ in tri:
                A_trib[nd_] += self.MemC.A0_np[e] / 3.0
        c_node = np.maximum(rg.chord_at(nodes[:, 1]), 1e-3)
        self.m_added_node = rho_air * np.pi * c_node ** 2 / 4.0 * (A_trib / c_node)
        # mass budget via the parametric mass program (co-design channel)
        members = {"le_spar": (chains["le"], LE_SPAR.m_lin),
                   "main_spar": (chains["main"], kb.MAIN_SPAR.m_lin),
                   "aux_spar": (chains["aux"], kb.AUX_SPAR.m_lin),
                   "te_hem": (chains["te"], hem_sec.m_lin)}
        for k_, r in enumerate(chains["ribs"]):
            members[f"rib{k_}"] = (r, rib_sec.m_lin)
        self.mass_detail = wmass.budget(nodes, tris, members, MEM_H, MEM_RHO)
        tt = self.mass_detail["totals"]
        self.mass_report = dict(
            membrane=tt["membrane"],
            spars=tt["le_spar"] + tt["main_spar"] + tt["aux_spar"],
            ribs=sum(v for n_, v in tt.items() if n_.startswith("rib")))

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
        """Static clamp of the aluminum root rib: j=0 translations + rod-root psi."""
        dofs = []
        for n in range(self.nc + 1):                       # j=0 row: node id == i
            dofs += list(self.trans_map[n])
        for n in (self.i_le, self.i_spar, self.i_aux):     # rod root psi (x3)
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
                # fixed-point convergence must be judged with FRESH weights at
                # the current q (the solved stale-weight residual is ~1e-7 by
                # construction and says nothing about the latch fixed point —
                # measured: breaking on it left a 0.21 N re-latch residual on
                # the flat assembly while the true fixed point converges ~4x
                # per outer to 1e-6 N).
                self._latch_soft(q)
                rn = np.linalg.norm(self.Q_int(q)[free])
                if verbose:
                    print(f"  pre-eq outer{outer}: fixed-point |r|={rn:.3e}")
                if rn < tol * r0 and outer > 1:
                    break
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
                       M_add=None, C_add=None):
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
        if C_add is not None:
            # implicit quasi-steady aero damping (strip theory dF/dv = pi rho U
            # A_node along the flap normal, analytic zero-fit, PSD). Air IS the
            # membrane's dominant physical damper (zeta >> 1); with the coupling
            # under-relaxed the lagged provider cannot supply it — implicit
            # treatment stops the ring the aero would physically kill.
            Cd = Cd + C_add
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
            # trust-region-ish cap: dynamic substep displacements are ~v*dt
            # (mm scale); a wild Newton direction on a near-degenerate sliver
            # must not leave the physical neighborhood.
            mx = np.abs(dq1).max()
            if mx > 1e-2:
                dq1 *= 1e-2 / mx
            # backtracking line search on the residual (up to 3 halvings; keep
            # the best step — residuals here are M/(beta dt^2)-dominated, far
            # better conditioned than statics, so mild backtracking suffices)
            best = None
            for step_ in (1.0, 0.5, 0.25):
                q_try = q1.copy(); q_try[free] += step_ * dq1
                a_try = (q_try - q_pred) / (beta * dt * dt)
                if presc is not None:
                    a_try[pd] = ddqb
                v_try = dq_pred + gamma * dt * a_try
                r_try = F_const - model.Q_int(q_try) - M @ a_try - Cd @ v_try
                rn_try = np.linalg.norm(r_try[free])
                if not np.isfinite(rn_try):
                    continue
                if best is None or rn_try < best[0]:
                    best = (rn_try, q_try, a_try)
                if rn_try < rn:
                    break
            if best is None:
                rn = np.inf
                break
            _, q1, a1 = best
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
        for n in (model.i_le, model.i_spar, model.i_aux):   # rod roots carry psi
            pd += list(model.dof6[n, 3:])
        self.pd = np.array(sorted(pd))
        model.set_bc(self.pd)                               # held by the solver mask
        self._X_root = model.nodes[root_nodes]              # (nc+1, 3)
        self._spar_psi_slots = [np.where(self.pd == model.dof6[n, 3])[0][0]
                                for n in (model.i_le, model.i_spar, model.i_aux)]
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
            # work-consistent transfer aero -> structure: f_s = W^T f_a
            f[self.m.trans_map.ravel()] = (self.m.W_a2s.T @ f9[:, 0:3]).ravel()
            ramp = 1.0
            if self.load_ramp_T > 0 and t < self.load_ramp_T:   # impulsive-start
                ramp = 0.5 * (1 - np.cos(np.pi * t / self.load_ramp_T))  # absorber
                f *= ramp
            from scipy.sparse import coo_matrix as _coo
            th_n = self._angles(t)[0]
            nvec = np.array([0.0, -np.sin(th_n), np.cos(th_n)])
            tm3 = self.m.trans_map
            rows = np.repeat(tm3, 3, axis=1).ravel()
            cols = np.tile(tm3, (1, 3)).ravel()
            madd = getattr(forces, "madd", None)
            if madd is not None:
                # S5 mode (BNV generalized-Robin form): the provider force
                # keeps its FULL dGamma/dt; the UVLM-consistent added-mass
                # operator (full n(x)n, symmetrized + sign-projected, <= 0)
                # goes onto the LHS (M_eff = M - madd) PAIRED with the RHS
                # compensation -madd*a_lag (a_lag = accel at the provider
                # solve, riding the force set) — identical fixed point, no
                # double count, and the added-mass channel gain is cancelled.
                # a_lag and strip M_a of the fallback path are OFF here.
                # NOT scaled by the load ramp: it is an operator (physical
                # apparent mass), not a load — ramping it re-opens the
                # added-mass gap exactly in the most violent start-up windows.
                nn_a = self.m.W_a2s.shape[0]
                t9 = (9 * np.arange(nn_a)[:, None] + np.arange(3)[None, :]).ravel()
                Ms = np.asarray(madd)[np.ix_(t9, t9)]       # aero-space (3na)^2
                W3 = self.m.W3
                a_lag9 = getattr(forces, "a_lag", None)
                if a_lag9 is not None:                      # -madd*a_lag on RHS
                    f[self.m.trans_map.ravel()] -= W3.T @ (Ms @ np.asarray(a_lag9))
                Ms_s = np.asarray(W3.T @ (W3.T @ Ms.T).T)   # W3^T Ms W3
                gd = self.m.trans_map.ravel()
                rws = np.repeat(gd, len(gd)); cls = np.tile(gd, len(gd))
                M_add = _coo((Ms_s.ravel(), (rws, cls)),
                             shape=(self.m.ndof, self.m.ndof)).tocsc()
            else:
                # fallback: implicit strip added mass along the instantaneous
                # flap normal + Jacobian-lagged compensation (BNV-style scalar
                # surrogate; insufficient alone at mu~5 — S5 research)
                blk = np.einsum("n,i,j->nij", self.m.m_added_node, nvec, nvec)
                M_a = _coo((blk.ravel(), (rows, cols)),
                           shape=(self.m.ndof, self.m.ndof)).tocsc()
                a_lag = (forces.payload or {}).get("a_lag") if hasattr(forces, "payload") else None
                if a_lag is not None:
                    f = f + M_a @ a_lag
                M_add = -M_a
            rho_air, U_inf = 1.225, 8.0
            # dcoef = pi rho U A_trib  (m_added_node = rho pi c^2/4 * A/c)
            dcoef = np.pi * rho_air * U_inf * (self.m.m_added_node /
                    (rho_air * np.pi * np.maximum(rg.chord_at(self.m.nodes[:, 1]), 1e-3) / 4.0))
            blkc = np.einsum("n,i,j->nij", dcoef, nvec, nvec)
            C_add = _coo((blkc.ravel(), (rows, cols)),
                         shape=(self.m.ndof, self.m.ndof)).tocsc()
        self._advance(t - dt, t, dt, f, M_add, C_add if forces is not None else None, depth=0)
        self.t = t
        return self

    def _advance(self, t0, t1, dt, f, M_add, C_add, depth):
        """Implicit substep with adaptive halving: an unconverged Newton substep
        (acute wrinkle/sliver events at flapping speed) must never be accepted."""
        qb, dqb, ddqb = self._cb(t1)
        try:
            q1, dq1, a1, ok = host_implicit_step(
                self.m, self.q, self.dq, self.a, t1 - t0, f,
                presc=(self.pd, qb, dqb, ddqb), M_add=M_add, C_add=C_add)
        except RuntimeError:
            ok, q1 = False, self.q                           # treat as unconverged
        if ok and np.all(np.isfinite(q1)):
            self.q, self.dq, self.a = q1, dq1, a1
            return
        if depth >= 4:
            # last resort: transient eta boost (1e-4 -> 1e-2) — deep-wrinkle/
            # sliver states gain 100x residual stiffness and become integrable;
            # the regularization is local in time (restored immediately) and
            # its force error is O(eta_boost) on the affected elements only.
            MemC = self.m.MemC
            eta0 = MemC.eta
            try:
                MemC.eta = 1e-2
                self._eta_boosts = getattr(self, "_eta_boosts", 0) + 1
                q1, dq1, a1, ok = host_implicit_step(
                    self.m, self.q, self.dq, self.a, t1 - t0, f,
                    presc=(self.pd,) + self._cb(t1), M_add=M_add, C_add=C_add)
                if ok and np.all(np.isfinite(q1)):
                    self.q, self.dq, self.a = q1, dq1, a1
                    return
            except RuntimeError:
                pass
            finally:
                MemC.eta = eta0
            raise RuntimeError(f"substep failed to converge at t={t1:.4f} (depth {depth})")
        tm_ = 0.5 * (t0 + t1)
        self._advance(t0, tm_, dt, f, M_add, C_add, depth + 1)
        self._advance(tm_, t1, dt, f, M_add, C_add, depth + 1)

    def state(self):
        """Aero interface state: the AERO surface = deformed structural grid +
        NACA2406 camber offset along the local (deformed) normal. The structure
        itself is flat (assembly v3); the offset keeps the UVLM geometry equal
        to the rigid Fig17-19 baseline. Offset rotation-rate contribution to
        vels is neglected (|off|<=5.7mm x thd<=14.5rad/s ~ 0.08 m/s << U)."""
        m = self.m
        u = self.q[m.trans_map.ravel()].reshape(-1, 3)
        du = self.dq[m.trans_map.ravel()].reshape(-1, 3)
        da = self.a[m.trans_map.ravel()].reshape(-1, 3)
        g = (m.W_a2s @ (m.nodes + u)).reshape(m.ns + 1, m.nc + 1, 3)
        vg = (m.W_a2s @ du).reshape(m.ns + 1, m.nc + 1, 3)
        ag = (m.W_a2s @ da).reshape(m.ns + 1, m.nc + 1, 3)
        ti = np.gradient(g, axis=1)                         # chordwise tangent
        tj = np.gradient(g, axis=0)                         # spanwise tangent
        nrm = np.cross(ti, tj)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-30)
        verts = g + nrm * m.aero_off[..., None]
        return dict(verts=verts.transpose(1, 0, 2).copy(),
                    vels=vg.transpose(1, 0, 2).copy(),
                    accel=ag.transpose(1, 0, 2).copy())
