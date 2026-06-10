"""Phase 4d — full GPU aero-coupled time loop (strong coupling, geom+AIC).

Assembles the validated per-kernel pieces into a device-resident coupled
trajectory that reproduces the CPU `standalone_hybrid_solver._run_strong` with
`enable_sc_geometry()` (single-pass block coupling, nowake — the landed
t*=1.0=0.776 config). Per block of `uvlm_ratio` structural steps:

  fluid solve (block boundary):
    geom = ScGeometry(q)                         # corners/colloc/normals (exact Sc)
    AIC  = build_aic_batched(geom)               # deformed-AIC rebuild
    rhs  = -(V_inf - V_struct)·n                 # gamma_rhs_kernel
    gamma= AIC^{-1} rhs                          # batched_dense_solve
    Vb   = Σ ring_vel(colloc, bound_rings,gamma) # induce_velocity_batched
    dp1  = dp_lift1(gamma, corners, V_inf+Vb)    # dp_lift1_flat_kernel
    Fbern= P_load · (dp1·n)                       # dp_times_n_kernel + CSR matvec

  structural march (uvlm_ratio steps):
    F_const = pulse(t) + Fbern (held over block)
    + velocity coupling (lift2, mf2_1) per step  [added incrementally]
    Newmark block-reduced step                   # gpu_newmark_step
"""
from __future__ import annotations
import numpy as np
import warp as wp
from . import config
from .kernels_geometry import ScGeometry
from .kernels_uvlm import build_aic_batched, induce_velocity_batched
from .kernels_coupling import CSR
from .batched_solver import batched_dense_solve

DTYPE = config.DTYPE
VEC3 = config.VEC3


@wp.kernel
def gamma_rhs_kernel(vinf: VEC3,
                     vstruct: wp.array(dtype=VEC3, ndim=2),   # (B, P)
                     normals: wp.array(dtype=VEC3, ndim=2),   # (B, P)
                     rhs: wp.array(dtype=DTYPE, ndim=2)):     # (B, P) out
    e, i = wp.tid()
    veff = vinf - vstruct[e, i]
    rhs[e, i] = -wp.dot(veff, normals[e, i])


@wp.kernel
def dp_lift1_flat_kernel(gamma: wp.array(dtype=DTYPE, ndim=2),    # (B, P) flat p=i*ns+j
                         corners: wp.array(dtype=VEC3, ndim=3),   # (B, P, 4)
                         V_ext: wp.array(dtype=VEC3, ndim=2),     # (B, P) bound induction
                         vinf: VEC3, rho: DTYPE, nc: int, ns: int,
                         dp: wp.array(dtype=DTYPE, ndim=2),       # (B, P) dp_lift1
                         dp2: wp.array(dtype=VEC3, ndim=2)):      # (B, P) dp_lift2 vec
    e, p = wp.tid()
    i = p / ns
    j = p - i * ns
    c0 = corners[e, p, 0]; c1 = corners[e, p, 1]
    c2 = corners[e, p, 2]; c3 = corners[e, p, 3]
    tx = (c1 - c0 + c2 - c3) * DTYPE(0.5)
    ty = (c0 - c3 + c1 - c2) * DTYPE(0.5)
    dxn = wp.length(tx) + DTYPE(1.0e-15)
    dyn = wp.length(ty) + DTYPE(1.0e-15)
    txh = tx / dxn
    tyh = ty / dyn
    g = gamma[e, p]
    if i == 0:
        dgx = g / dxn
    else:
        dgx = (g - gamma[e, p - ns]) / dxn
    dgy = DTYPE(0.0)
    if ns > 1:
        if j == 0:
            dgy = g / dyn
        elif j == ns - 1:
            dgy = -g / dyn
        else:
            dgy = (gamma[e, p + 1] - gamma[e, p - 1]) / (DTYPE(2.0) * dyn)
    Vs = vinf + V_ext[e, p]
    dp[e, p] = rho * (wp.dot(Vs, txh) * dgx + wp.dot(Vs, tyh) * dgy)
    # dp_lift2 = ρ·(τx·dΓ/dx + τy·dΓ/dy)  (VEC3, consumed per-step with V_struct)
    dp2[e, p] = rho * (txh * dgx + tyh * dgy)


@wp.kernel
def dp_times_n_kernel(dp: wp.array(dtype=DTYPE, ndim=2),       # (B, P)
                      normals: wp.array(dtype=VEC3, ndim=2),   # (B, P)
                      dpn: wp.array(dtype=DTYPE, ndim=2)):     # (B, 3P) out
    e, p = wp.tid()
    n = normals[e, p]
    v = dp[e, p]
    dpn[e, 3 * p] = v * n[0]
    dpn[e, 3 * p + 1] = v * n[1]
    dpn[e, 3 * p + 2] = v * n[2]


@wp.kernel
def vertex_vel_kernel(dq: wp.array(dtype=DTYPE, ndim=2),        # (B, ndof)
                      closest: wp.array(dtype=wp.int32, ndim=1),  # (NV,)
                      dtv: wp.array(dtype=VEC3, ndim=2)):        # (B, NV) out
    """Vertex velocity by closest-ANCF-node lookup (=_compute_uvlm_vertex_velocities)."""
    e, v = wp.tid()
    nd = closest[v]
    dtv[e, v] = wp.vector(dq[e, nd * 9], dq[e, nd * 9 + 1], dq[e, nd * 9 + 2])


@wp.kernel
def dt_normals_kernel(verts: wp.array(dtype=VEC3, ndim=1),      # (NV,) frozen ref grid
                      dtv: wp.array(dtype=VEC3, ndim=2),        # (B, NV) vertex vel
                      ns: int,
                      dtn: wp.array(dtype=VEC3, ndim=2)):       # (B, P) out
    """dt of unit normal (=compute_dt_normals): n=(d1×d2)/|.|,
    dt_n = dc/|.| − n·(n·dc/|.|),  dc = dd1×d2 + d1×dd2."""
    e, p = wp.tid()
    i = p / ns
    j = p - i * ns
    w = ns + 1
    LEin = i * w + j
    LEout = i * w + (j + 1)
    TEin = (i + 1) * w + j
    TEout = (i + 1) * w + (j + 1)
    d1 = verts[TEout] - verts[LEin]
    d2 = verts[LEout] - verts[TEin]
    dd1 = dtv[e, TEout] - dtv[e, LEin]
    dd2 = dtv[e, LEout] - dtv[e, TEin]
    cr = wp.cross(d1, d2)
    dcr = wp.cross(dd1, d2) + wp.cross(d1, dd2)
    L = wp.length(cr) + DTYPE(1.0e-30)
    nu = cr / L
    dcn = dcr / L
    dtn[e, p] = dcn - nu * wp.dot(nu, dcn)


@wp.kernel
def mf2_1_scalar_kernel(vstruct: wp.array(dtype=VEC3, ndim=2),  # (B, P)
                        vinf: VEC3,
                        dtn: wp.array(dtype=VEC3, ndim=2),      # (B, P)
                        scal: wp.array(dtype=DTYPE, ndim=2)):   # (B, P) out
    """slip·dt_n, slip = V_struct − V_inf (V_wake=0 nowake)."""
    e, p = wp.tid()
    slip = vstruct[e, p] - vinf
    scal[e, p] = wp.dot(slip, dtn[e, p])


@wp.kernel
def mf2_1_dpn_kernel(x: wp.array(dtype=DTYPE, ndim=2),          # (B, P) = AIC^{-1}·scal
                     normals: wp.array(dtype=VEC3, ndim=2),     # (B, P) held normals(q_n)
                     rho: DTYPE,
                     dpn: wp.array(dtype=DTYPE, ndim=2)):       # (B, 3P) out
    """pressure×n = ρ·pressure·n,  pressure = -x  (Python AIC=-AIC_ml sign)."""
    e, p = wp.tid()
    s = -rho * x[e, p]
    n = normals[e, p]
    dpn[e, 3 * p] = s * n[0]
    dpn[e, 3 * p + 1] = s * n[1]
    dpn[e, 3 * p + 2] = s * n[2]


class GpuFluidSolve:
    """Device fluid solve (nowake): q,dq -> forces_no_vstruct (Bernoulli) and
    the load-transferred nodal F_bernoulli. Mirrors CPU _uvlm_step + _load_transfer
    of forces_no_vstruct (the held aero of the single-pass block coupling)."""

    def __init__(self, solver, scgeom=None, device=None, wake=False,
                 wake_max_rows=64):
        device = device or config.DEVICE
        self.device = device
        self.nc, self.ns = solver._nx, solver._ny
        self.P = self.nc * self.ns
        self.ndof = solver.shell.ndof
        self.geom = scgeom or ScGeometry(self.nc, self.ns, device=device)
        self.Pload = CSR(solver._P_load, device)
        self.V_inf = np.asarray(solver.uvlm._V_inf, dtype=config.NP_DTYPE)
        self.rho = config.NP_DTYPE(solver.uvlm._rho)
        self.core = solver.uvlm._core_radius
        self.use_wake = wake
        self._wake_max_rows = wake_max_rows
        self._dt_wake_conv = float(solver._dt_uvlm)          # wake convection step
        self._trunc_x = float(getattr(solver, '_wake_truncation', 5.5))
        self.wake = None                                     # GpuWake, built on first solve
        # ── mf2_1: flat vertex grid (frozen ref, since enable_sc_geometry never
        #    updates _verts) + closest-ANCF-node map (constant) for dt_normals ──
        verts = np.asarray(solver.uvlm._verts, dtype=config.NP_DTYPE)   # (nc+1,ns+1,3)
        self.NV = (self.nc + 1) * (self.ns + 1)
        nodes = solver.shell.positions()
        closest = np.zeros(self.NV, dtype=np.int32)
        for vi in range(self.nc + 1):
            for vj in range(self.ns + 1):
                x, y = verts[vi, vj, 0], verts[vi, vj, 1]
                closest[vi * (self.ns + 1) + vj] = int(
                    np.argmin(np.abs(nodes[:, 0] - x) + np.abs(nodes[:, 1] - y)))
        vflat = verts.reshape(self.NV, 3)
        self.verts_wp = wp.array(vflat, dtype=VEC3, device=device)
        self.closest_wp = wp.array(closest, dtype=wp.int32, device=device)
        self.AIC = None     # held block AIC for mf2_1 solve
        self._B = None

    def _alloc(self, B):
        if self._B == B:
            return
        d = self.device
        self.rhs = wp.zeros((B, self.P), dtype=DTYPE, device=d)
        self.dp = wp.zeros((B, self.P), dtype=DTYPE, device=d)
        self.dp2 = wp.zeros((B, self.P), dtype=VEC3, device=d)
        self.dpn = wp.zeros((B, 3 * self.P), dtype=DTYPE, device=d)
        self.dpn_l2 = wp.zeros((B, 3 * self.P), dtype=DTYPE, device=d)
        self.dtv = wp.zeros((B, self.NV), dtype=VEC3, device=d)
        self.dtn = wp.zeros((B, self.P), dtype=VEC3, device=d)
        self.mscal = wp.zeros((B, self.P), dtype=DTYPE, device=d)
        self.dpn_mf = wp.zeros((B, 3 * self.P), dtype=DTYPE, device=d)
        if self.use_wake:
            from .kernels_wake import GpuWake
            self.wake = GpuWake(B, self.ns, self._wake_max_rows, self.core,
                                self.V_inf, self._dt_wake_conv, self._trunc_x,
                                device=d)
            self.Vw = wp.zeros((B, self.P), dtype=VEC3, device=d)
            self.Vext = wp.zeros((B, self.P), dtype=VEC3, device=d)
            self.wscal = wp.zeros((B, self.P), dtype=DTYPE, device=d)
            self.nwscal = wp.zeros((B, self.P), dtype=DTYPE, device=d)
            # wake corner velocities for Mf2_vec1: V_inf broadcast (CPU default)
            NP = config.NP_DTYPE
            wdt_np = np.ascontiguousarray(
                np.broadcast_to(self.V_inf, (B, self.wake.NW, 4, 3)), dtype=NP)
            self.wdt = wp.array(wdt_np, dtype=VEC3, device=d)   # (B, NW, 4)
            # gamma double-buffer: cur = last solve, prev2 = two solves ago
            self.gamma_cur = wp.zeros((B, self.P), dtype=DTYPE, device=d)
            self.gamma_prev2 = wp.zeros((B, self.P), dtype=DTYPE, device=d)
            # host flags mirroring CPU shed-skip (shed only if source nonzero)
            self._g1_nonzero = False    # gamma_{k-1} nonzero?
            self._g2_nonzero = False    # gamma_{k-2} nonzero?
        self._B = B

    def mf2_1_force(self, dq_wp, normals_held):
        """F_mf2_1 = P_load·(ρ·pressure·n),  pressure = -AIC^{-1}·(slip·dt_n),
        slip = V_struct - V_inf (+ V_wake=0 nowake),  dt_n from compute_dt_normals
        (flat ref verts + closest-node vertex velocities). AIC held from block solve;
        normals held at q_n. Mirrors CPU _compute_mf2_1_force exactly."""
        B = dq_wp.shape[0]
        self._alloc(B)
        d = self.device
        wp.launch(vertex_vel_kernel, dim=(B, self.NV),
                  inputs=[dq_wp, self.closest_wp], outputs=[self.dtv], device=d)
        wp.launch(dt_normals_kernel, dim=(B, self.P),
                  inputs=[self.verts_wp, self.dtv, self.ns], outputs=[self.dtn], device=d)
        vstruct = self.geom.struct_velocity(dq_wp)
        vinf = VEC3(*(self.V_inf.tolist()))
        wp.launch(mf2_1_scalar_kernel, dim=(B, self.P),
                  inputs=[vstruct, vinf, self.dtn], outputs=[self.mscal], device=d)
        x = batched_dense_solve(self.AIC, self.mscal, device=d)   # AIC·x = scalar
        wp.launch(mf2_1_dpn_kernel, dim=(B, self.P),
                  inputs=[x, normals_held, DTYPE(self.rho)], outputs=[self.dpn_mf], device=d)
        return self.Pload.matvec(self.dpn_mf)

    def velocity_force(self, dq_wp, dp2_held, normals_held, with_mf2_1=True):
        """Full velocity-coupling nodal force F_lift2 (+ F_mf2_1). Returns (B,ndof)."""
        from .batched_solver import _saxpy_kernel
        F = self.lift2_force(dq_wp, dp2_held, normals_held)
        if with_mf2_1:
            Fmf = self.mf2_1_force(dq_wp, normals_held)
            wp.launch(_saxpy_kernel, dim=F.shape, inputs=[F, DTYPE(1.0), Fmf],
                      device=self.device)
        return F

    def lift2_force(self, dq_wp, dp2_held, normals_held):
        """F_lift2 = P_load·(-(V_struct·dp_lift2)·n).  V_struct=Sc_col·dq;
        dp_lift2 and normals are HELD (block solve / q_n geometry)."""
        B = dq_wp.shape[0]
        self._alloc(B)
        vstruct = self.geom.struct_velocity(dq_wp)
        wp.launch(lift2_dpn_kernel, dim=(B, self.P),
                  inputs=[vstruct, dp2_held, normals_held],
                  outputs=[self.dpn_l2], device=self.device)
        return self.Pload.matvec(self.dpn_l2)

    def solve(self, q_wp, dq_wp):
        """Returns (forces_no_vstruct (B,P) VEC3, F_bernoulli (B,ndof)).
        With use_wake: CPU _uvlm_step order — advect → shed → solve(+wake RHS) →
        newest-row gamma update → V_wake into forces (+Mf2_vec1) → truncate."""
        B = q_wp.shape[0]
        self._alloc(B)
        d = self.device
        g = self.geom
        g.update(q_wp)
        vstruct = g.struct_velocity(dq_wp)
        vinf = VEC3(*(self.V_inf.tolist()))
        wk = self.wake if self.use_wake else None
        if wk is not None:
            # 1) advect existing wake (frozen snapshot; bound source = gamma_cur
            #    = previous solve, matching CPU's use of self.gamma)
            wk.advect(g.corners, self.gamma_cur)
            # 2) shed TE row, delayed-Kutta source = gamma two solves ago.
            #    CPU shed_wake skips while the source is all-zero (start-up).
            if self._g2_nonzero:
                wk.shed(g.corners, self.gamma_prev2, self.nc)
        # gamma RHS = -(V_inf - V_struct)·n  (+ V_wake·n with wake)
        wp.launch(gamma_rhs_kernel, dim=(B, self.P),
                  inputs=[vinf, vstruct, g.normals], outputs=[self.rhs], device=d)
        if wk is not None:
            induce_velocity_batched(g.colloc, wk.wcorners, wk.wgamma, self.core,
                                    out_V=self.Vw, device=d)
            wp.launch(add_dot_n_kernel, dim=(B, self.P),
                      inputs=[self.Vw, g.normals], outputs=[self.rhs], device=d)
        # AIC (deformed) and gamma solve
        AIC = build_aic_batched(g.colloc, g.normals, g.corners, self.core, device=d)
        self.AIC = AIC     # hold for mf2_1 AIC^{-1} solve (batched_dense_solve clones)
        gamma = batched_dense_solve(AIC, self.rhs, device=d)   # (B, P)
        if wk is not None:
            # 4) newest wake row takes the PREVIOUS solve's bound TE circulation
            #    (CPU: only when rows exist)
            if wk.n_rows > 0:
                wk.update_newest_gamma(self.gamma_cur, self.nc)
            # advance gamma double-buffer (prev2 <- cur <- new) + host flags
            wp.copy(self.gamma_prev2, self.gamma_cur)
            wp.copy(self.gamma_cur, gamma)
            self._g2_nonzero = self._g1_nonzero
            self._g1_nonzero = bool(np.abs(gamma.numpy()).max() > 1e-15)
        # bound induction at colloc
        Vb = induce_velocity_batched(g.colloc, g.corners, gamma, self.core, device=d)
        if wk is not None:
            # V_ext = V_bound + V_wake (wake state already updated this step)
            induce_velocity_batched(g.colloc, wk.wcorners, wk.wgamma, self.core,
                                    out_V=self.Vw, device=d)
            wp.launch(vec3_add_kernel, dim=(B, self.P),
                      inputs=[Vb, self.Vw], outputs=[self.Vext], device=d)
            V_ext = self.Vext
        else:
            V_ext = Vb
        # dp_lift1 (+ dp_lift2 vector for velocity coupling)
        wp.launch(dp_lift1_flat_kernel, dim=(B, self.P),
                  inputs=[gamma, g.corners, V_ext, vinf, DTYPE(self.rho),
                          self.nc, self.ns],
                  outputs=[self.dp, self.dp2], device=d)
        if wk is not None and wk.n_rows > 0:
            # Mf2_vec1 = AIC^{-1}·(-Σ_wake Γw·dt_ring·n);  dp += ρ·Mf2_vec1
            from .kernels_uvlm import compute_mf2_scalar_batched
            compute_mf2_scalar_batched(g.colloc, g.normals, vstruct,
                                       wk.wcorners, self.wdt, wk.wgamma,
                                       self.core, out_scal=self.wscal, device=d)
            wp.launch(negate_kernel, dim=(B, self.P),
                      inputs=[self.wscal], outputs=[self.nwscal], device=d)
            mf2v = batched_dense_solve(AIC, self.nwscal, device=d)
            wp.launch(dp_add_scaled_kernel, dim=(B, self.P),
                      inputs=[mf2v, DTYPE(self.rho)], outputs=[self.dp], device=d)
        # dp·n -> load transfer
        wp.launch(dp_times_n_kernel, dim=(B, self.P),
                  inputs=[self.dp, g.normals], outputs=[self.dpn], device=d)
        Fbern = self.Pload.matvec(self.dpn)                    # (B, ndof)
        if wk is not None:
            wk.truncate()
        return self.dp, self.dp2, gamma, Vb, Fbern


@wp.kernel
def add_dot_n_kernel(V: wp.array(dtype=VEC3, ndim=2),        # (B, P)
                     normals: wp.array(dtype=VEC3, ndim=2),  # (B, P)
                     rhs: wp.array(dtype=DTYPE, ndim=2)):    # (B, P) in/out
    """rhs += V·n (wake influence on the no-penetration RHS, Python sign)."""
    e, i = wp.tid()
    rhs[e, i] = rhs[e, i] + wp.dot(V[e, i], normals[e, i])


@wp.kernel
def vec3_add_kernel(a: wp.array(dtype=VEC3, ndim=2),
                    b: wp.array(dtype=VEC3, ndim=2),
                    out: wp.array(dtype=VEC3, ndim=2)):
    e, i = wp.tid()
    out[e, i] = a[e, i] + b[e, i]


@wp.kernel
def dp_add_scaled_kernel(x: wp.array(dtype=DTYPE, ndim=2),   # (B, P) Mf2_vec1
                         c: DTYPE,                           # rho
                         dp: wp.array(dtype=DTYPE, ndim=2)):  # (B, P) in/out
    e, i = wp.tid()
    dp[e, i] = dp[e, i] + c * x[e, i]


@wp.kernel
def negate_kernel(x: wp.array(dtype=DTYPE, ndim=2),
                  out: wp.array(dtype=DTYPE, ndim=2)):
    e, i = wp.tid()
    out[e, i] = -x[e, i]


@wp.kernel
def lift2_dpn_kernel(vstruct: wp.array(dtype=VEC3, ndim=2),    # (B, P) Sc_col·dq
                     dp2: wp.array(dtype=VEC3, ndim=2),        # (B, P) held dp_lift2
                     normals: wp.array(dtype=VEC3, ndim=2),    # (B, P) held normals(q_n)
                     dpn: wp.array(dtype=DTYPE, ndim=2)):      # (B, 3P) out (pressure×n)
    """lift2 panel pressure×n = -(V_struct·dp_lift2)·n  (=_compute_lift2_force/area)."""
    e, p = wp.tid()
    s = -wp.dot(vstruct[e, p], dp2[e, p])
    n = normals[e, p]
    dpn[e, 3 * p] = s * n[0]
    dpn[e, 3 * p + 1] = s * n[1]
    dpn[e, 3 * p + 2] = s * n[2]


@wp.kernel
def fc_combine_kernel(s: DTYPE,
                      pulse: wp.array(dtype=DTYPE, ndim=2),    # (B, ndof)
                      Fbern: wp.array(dtype=DTYPE, ndim=2),    # (B, ndof)
                      out: wp.array(dtype=DTYPE, ndim=2)):     # (B, ndof)
    e, i = wp.tid()
    out[e, i] = s * pulse[e, i] + Fbern[e, i]


def gpu_coupled_trajectory(C, gfs, q0, dq0, pulse_shape, profile, dt, n_steps,
                           uvlm_ratio=34, alpha_v=0.5, c_damp=2.0, cg_tol=1e-12,
                           tip_dof=None, madd=None, madd_diag=None,
                           velocity_coupling=False, device=None):
    """GPU single-pass strong-coupled trajectory (nowake) matching CPU _run_strong
    + enable_sc_geometry with _constant_aero. Per block: fluid solve at block start
    (carried from previous block end), march uvlm_ratio Newmark steps with held
    F_bernoulli (+ optional lift2 velocity coupling, Newmark-averaged), solve fluid
    at block end for next block.

    velocity_coupling=True adds the lift2 term (Qf_p_lift2): F_vel(dq) =
    P_load·(-(Sc_col·dq · dp_lift2)·n) with dp_lift2 held from the block solve and
    normals refreshed at each step's q_n. (mf2_1 / Qf_p_mat0 not yet ported — needs
    dt_normals.) Returns (q, dq, tip_history)."""
    from .kernels_ancf import assemble_kmem_blocks, assemble_internal_force_sep
    from .batched_solver import gpu_newmark_step
    device = device or config.DEVICE
    NP = config.NP_DTYPE
    B, ndof = q0.shape
    q = wp.clone(q0); dq = wp.clone(dq0)
    pulse_dev = wp.array(np.broadcast_to(pulse_shape, (B, ndof)).astype(NP).copy(),
                         dtype=DTYPE, device=device)
    Fvel0 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    Fc = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    tip = np.zeros((n_steps, B)) if tip_dof is not None else None

    def recompute_bend(q_p1):
        return assemble_internal_force_sep(q_p1, C, device)[1]

    # initial fluid solve at q0 (flat plate) -> F_bernoulli, dp_lift2 for block 0
    _, dp2, _, _, Fbern = gfs.solve(q, dq)
    dp2_held = wp.clone(dp2)

    step = 0
    while step < n_steps:
        for _k in range(uvlm_ratio):
            if step >= n_steps:
                break
            t = (step + 1) * dt
            s = float(profile(t))
            wp.launch(fc_combine_kernel, dim=(B, ndof),
                      inputs=[DTYPE(NP(s)), pulse_dev, Fbern], outputs=[Fc], device=device)
            if velocity_coupling:
                gfs.geom.update(q)                 # normals at q_n (held over step)
                normals_qn = gfs.geom.normals
                mf2 = (velocity_coupling != 'lift2')   # True=full(lift2+mf2_1)
                Fvel_n = gfs.velocity_force(dq, dp2_held, normals_qn, with_mf2_1=mf2)
                def rfvel(qp1, dqp1, _d=dp2_held, _n=normals_qn, _m=mf2):
                    return gfs.velocity_force(dqp1, _d, _n, with_mf2_1=_m)
            else:
                Fvel_n = Fvel0
                rfvel = None
            Kblk = assemble_kmem_blocks(q, C, device)
            Qmem, Qbend = assemble_internal_force_sep(q, C, device)
            q, dq = gpu_newmark_step(q, dq, Kblk, C.Me, C.edofs, C.free, ndof,
                                     Fc, Qmem, Qbend, Fvel_n, recompute_bend, rfvel,
                                     alpha_v=alpha_v, c_damp=c_damp, dt=dt,
                                     cg_tol=cg_tol, device=device,
                                     madd=madd, madd_diag=madd_diag)
            if tip is not None:
                wp.synchronize()
                tip[step] = q.numpy()[:, tip_dof]
            step += 1
        if step < n_steps:
            _, dp2, _, _, Fbern = gfs.solve(q, dq)   # block-end fluid solve
            dp2_held = wp.clone(dp2)
    return q, dq, tip

