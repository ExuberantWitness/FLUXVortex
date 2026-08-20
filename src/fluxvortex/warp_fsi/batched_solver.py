"""Batched linear solvers (one independent system per environment).

- batched_dense_solve: dense LU with partial pivoting, one thread per env.
  Used for the aero AIC solve (non-symmetric, N≈150, once per fluid step).
  Correct + fully parallel across envs; not peak-perf for a single env (one
  thread does the whole O(N³) factorization) — fine since it is NOT the hot loop
  and parallelizes across the many environments that are the actual workload.

- (Phase 3) the SPD structural solve will use a vendored kamino matrix-free CR.
"""
from __future__ import annotations
import math
import warp as wp
from . import config

DTYPE = config.DTYPE


@wp.kernel
def _nonfinite_2d_kernel(
    values: wp.array(dtype=DTYPE, ndim=2),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    i, j = wp.tid()
    if not wp.isfinite(values[i, j]):
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _nonfinite_3d_kernel(
    values: wp.array(dtype=DTYPE, ndim=3),
    flag: wp.array(dtype=wp.int32, ndim=1),
):
    i, j, k = wp.tid()
    if not wp.isfinite(values[i, j, k]):
        wp.atomic_max(flag, 0, 1)


def _require_finite_gpu_array(value, device, name):
    if not value.device.is_cuda or value.device.alias != wp.get_device(device).alias:
        raise ValueError(f"{name} must reside on the selected CUDA device")
    flag = wp.zeros(1, dtype=wp.int32, device=device)
    kernel = _nonfinite_2d_kernel if value.ndim == 2 else _nonfinite_3d_kernel
    wp.launch(kernel, dim=value.shape, inputs=[value], outputs=[flag], device=device)
    wp.synchronize()
    if bool(flag.numpy()[0]):
        raise FloatingPointError(f"{name} contains non-finite values")


@wp.kernel
def _lu_solve_kernel(
    A: wp.array(dtype=DTYPE, ndim=3),  # (B,N,N) work copy (destroyed)
    b: wp.array(dtype=DTYPE, ndim=2),  # (B,N) rhs -> solution
    N: int,
):
    e = wp.tid()
    # LU factorization with partial pivoting, in place on A[e], applied to b[e].
    for k in range(N):
        piv = k
        amax = wp.abs(A[e, k, k])
        for r in range(k + 1, N):
            v = wp.abs(A[e, r, k])
            if v > amax:
                amax = v
                piv = r
        if piv != k:
            for c in range(N):
                t = A[e, k, c]
                A[e, k, c] = A[e, piv, c]
                A[e, piv, c] = t
            tb = b[e, k]
            b[e, k] = b[e, piv]
            b[e, piv] = tb
        akk = A[e, k, k]
        for r in range(k + 1, N):
            f = A[e, r, k] / akk
            for c in range(k + 1, N):
                A[e, r, c] = A[e, r, c] - f * A[e, k, c]
            b[e, r] = b[e, r] - f * b[e, k]
    # back substitution
    for kk in range(N):
        k = N - 1 - kk
        s = b[e, k]
        for c in range(k + 1, N):
            s = s - A[e, k, c] * b[e, c]
        b[e, k] = s / A[e, k, k]


def batched_dense_solve(A_wp, b_wp, device=None, in_place_b=False):
    """Solve A[e] x[e] = b[e] for each env e. A_wp (B,N,N), b_wp (B,N).

    A_wp is copied (kernel destroys its work copy). Returns solution (B,N).
    """
    device = device or config.DEVICE
    B, N, _ = A_wp.shape
    if B < 1 or N < 1 or A_wp.shape != (B, N, N) or b_wp.shape != (B, N):
        raise ValueError("CUDA dense solve requires non-empty aligned batched systems")
    if not wp.get_device(device).is_cuda:
        raise RuntimeError("FLUX-V5M dense solve has no host implementation")
    _require_finite_gpu_array(A_wp, device, "dense matrix")
    _require_finite_gpu_array(b_wp, device, "dense right-hand side")
    A_work = wp.clone(A_wp)
    x = b_wp if in_place_b else wp.clone(b_wp)
    wp.launch(_lu_solve_kernel, dim=B, inputs=[A_work, x, N], device=device)
    _require_finite_gpu_array(x, device, "dense solution")
    return x


# ─── Matrix-free batched structural CG (S = M + coef·K, SPD) ────────────────
# S·v computed from per-element blocks (M_e shared, K_e per-env), with a free-DOF
# mask (Dirichlet BC). All batched over environments; no global matrix stored.


@wp.kernel
def smatvec_kernel(
    v: wp.array(dtype=DTYPE, ndim=2),  # (B, ndof) input
    Me: wp.array(dtype=DTYPE, ndim=3),  # (ne, nblk, nblk) shared mass
    Kblk: wp.array(dtype=DTYPE, ndim=4),  # (B, ne, nblk, nblk) per-env tangent
    edofs: wp.array(dtype=wp.int32, ndim=2),  # (ne, nblk)
    free: wp.array(dtype=DTYPE, ndim=1),  # (ndof,) 1 free / 0 BC
    cM: DTYPE,
    cK: DTYPE,
    nblk: int,
    w: wp.array(dtype=DTYPE, ndim=2),
):  # (B, ndof) out (accumulate)
    """w += free⊙((cM·M + cK·K)·(free⊙v)).  cM=1,cK=coef -> S; cM=0,cK=.. -> K; cM=1,cK=0 -> M."""
    e, el, a = wp.tid()
    da = edofs[el, a]
    acc = DTYPE(0.0)
    for b in range(nblk):
        db = edofs[el, b]
        vb = v[e, db] * free[db]
        acc = acc + (cM * Me[el, a, b] + cK * Kblk[e, el, a, b]) * vb
    wp.atomic_add(w, e, da, free[da] * acc)


@wp.kernel
def _masked_axpy_kernel(
    y: wp.array(dtype=DTYPE, ndim=2),
    c: DTYPE,
    x: wp.array(dtype=DTYPE, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
):
    e, d = wp.tid()
    y[e, d] = y[e, d] + c * free[d] * x[e, d]


def apply_MK(
    vin, wout, Me, Kblk, edofs, free, cM, cK, device, madd=None, madd_tmp=None
):
    """wout = free⊙((cM·M + cK·K)·(free⊙vin)) − cM·free⊙(M_added·vin).

    Adds the −M_added term (M_eff = M − M_added) when `madd` (a CSR) is given and
    cM≠0. Assumes vin has bc=0 (true for all CG/Newmark vectors)."""
    NP = config.NP_DTYPE
    nblk = edofs.shape[1]
    wout.zero_()
    wp.launch(
        smatvec_kernel,
        dim=(vin.shape[0], edofs.shape[0], nblk),
        inputs=[vin, Me, Kblk, edofs, free, DTYPE(NP(cM)), DTYPE(NP(cK)), nblk],
        outputs=[wout],
        device=device,
    )
    if madd is not None and cM != 0.0:
        mv = madd.matvec(vin, out=madd_tmp)
        wp.launch(
            _masked_axpy_kernel,
            dim=wout.shape,
            inputs=[wout, DTYPE(NP(-cM)), mv, free],
            device=device,
        )


@wp.kernel
def sdiag_kernel(
    Me: wp.array(dtype=DTYPE, ndim=3),
    Kblk: wp.array(dtype=DTYPE, ndim=4),
    edofs: wp.array(dtype=wp.int32, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
    coef: DTYPE,
    diag: wp.array(dtype=DTYPE, ndim=2),
):  # (B, ndof) out
    """diag(S)[e,da] = Σ_el (Me[el,a,a] + coef·Kblk[e,el,a,a]) for da=edofs[el,a]."""
    e, el, a = wp.tid()
    da = edofs[el, a]
    wp.atomic_add(diag, e, da, free[da] * (Me[el, a, a] + coef * Kblk[e, el, a, a]))


@wp.kernel
def _sub_bcast_kernel(
    diag: wp.array(dtype=DTYPE, ndim=2),
    md: wp.array(dtype=DTYPE, ndim=1),
    free: wp.array(dtype=DTYPE, ndim=1),
):
    """diag[e,d] -= free[d]·md[d]  (subtract shared M_added diagonal)."""
    e, d = wp.tid()
    diag[e, d] = diag[e, d] - free[d] * md[d]


@wp.kernel
def _minv_apply(
    diag: wp.array(dtype=DTYPE, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
    r: wp.array(dtype=DTYPE, ndim=2),
    z: wp.array(dtype=DTYPE, ndim=2),
):
    e, d = wp.tid()
    dd = diag[e, d]
    if free[d] > DTYPE(0.5) and dd != DTYPE(0.0):
        z[e, d] = r[e, d] / dd
    else:
        z[e, d] = DTYPE(0.0)


@wp.kernel
def _dot_kernel(
    a: wp.array(dtype=DTYPE, ndim=2),
    b: wp.array(dtype=DTYPE, ndim=2),
    ndof: int,
    out: wp.array(dtype=DTYPE, ndim=1),
):
    e = wp.tid()
    s = DTYPE(0.0)
    for d in range(ndof):
        s = s + a[e, d] * b[e, d]
    out[e] = s


@wp.kernel
def _dot_parallel_kernel(
    a: wp.array(dtype=DTYPE, ndim=2),
    b: wp.array(dtype=DTYPE, ndim=2),
    out: wp.array(dtype=DTYPE, ndim=1),
):
    """Parallel batch dot product used by the structural PCG solver."""
    e, d = wp.tid()
    wp.atomic_add(out, e, a[e, d] * b[e, d])


@wp.kernel
def _axpy_kernel(
    y: wp.array(dtype=DTYPE, ndim=2),
    coef: wp.array(dtype=DTYPE, ndim=1),
    x: wp.array(dtype=DTYPE, ndim=2),
    sign: DTYPE,
):
    e, d = wp.tid()
    y[e, d] = y[e, d] + sign * coef[e] * x[e, d]


@wp.kernel
def _xpby_kernel(
    p: wp.array(dtype=DTYPE, ndim=2),
    r: wp.array(dtype=DTYPE, ndim=2),
    beta: wp.array(dtype=DTYPE, ndim=1),
):
    e, d = wp.tid()
    p[e, d] = r[e, d] + beta[e] * p[e, d]


@wp.kernel
def _div_kernel(
    num: wp.array(dtype=DTYPE, ndim=1),
    den: wp.array(dtype=DTYPE, ndim=1),
    out: wp.array(dtype=DTYPE, ndim=1),
):
    e = wp.tid()
    out[e] = num[e] / den[e]


@wp.kernel
def _pcg_update_solution_residual(
    x: wp.array(dtype=DTYPE, ndim=2),
    r: wp.array(dtype=DTYPE, ndim=2),
    p: wp.array(dtype=DTYPE, ndim=2),
    sp: wp.array(dtype=DTYPE, ndim=2),
    rz: wp.array(dtype=DTYPE, ndim=1),
    psp: wp.array(dtype=DTYPE, ndim=1),
):
    """Fuse alpha, x and residual updates into one DOF-parallel launch."""
    e, d = wp.tid()
    alpha = rz[e] / psp[e]
    x[e, d] = x[e, d] + alpha * p[e, d]
    r[e, d] = r[e, d] - alpha * sp[e, d]


@wp.kernel
def _pcg_precondition_dot(
    diag: wp.array(dtype=DTYPE, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
    r: wp.array(dtype=DTYPE, ndim=2),
    z: wp.array(dtype=DTYPE, ndim=2),
    rz_new: wp.array(dtype=DTYPE, ndim=1),
):
    """Fuse Jacobi preconditioning with the r·z reduction."""
    e, d = wp.tid()
    dd = diag[e, d]
    value = DTYPE(0.0)
    if free[d] > DTYPE(0.5) and dd != DTYPE(0.0):
        value = r[e, d] / dd
    z[e, d] = value
    wp.atomic_add(rz_new, e, r[e, d] * value)


@wp.kernel
def _pcg_update_direction(
    p: wp.array(dtype=DTYPE, ndim=2),
    z: wp.array(dtype=DTYPE, ndim=2),
    rz_new: wp.array(dtype=DTYPE, ndim=1),
    rz: wp.array(dtype=DTYPE, ndim=1),
):
    """Fuse beta evaluation and search-direction update."""
    e, d = wp.tid()
    beta = rz_new[e] / rz[e]
    p[e, d] = z[e, d] + beta * p[e, d]


def _dot(a, b, ndof, out, device):
    if a.shape[1] != ndof or b.shape != a.shape:
        raise ValueError("CUDA dot operands must be aligned (B, ndof) arrays")
    out.zero_()
    wp.launch(
        _dot_parallel_kernel,
        dim=(a.shape[0], ndof),
        inputs=[a, b],
        outputs=[out],
        device=device,
    )


@wp.kernel
def _max_scalar_kernel(
    values: wp.array(dtype=DTYPE, ndim=1), out: wp.array(dtype=DTYPE, ndim=1)
):
    """GPU reduction used by the host convergence-control decision."""
    index = wp.tid()
    wp.atomic_max(out, 0, values[index])


def _gpu_max_scalar(values, device):
    out = wp.zeros(1, dtype=DTYPE, device=device)
    wp.launch(
        _max_scalar_kernel,
        dim=values.shape[0],
        inputs=[values],
        outputs=[out],
        device=device,
    )
    wp.synchronize()
    return float(out.numpy()[0])


def structural_cg(
    b_wp,
    Me,
    Kblk,
    edofs,
    free,
    coef,
    ndof,
    max_iter=2000,
    tol=None,
    device=None,
    check_every=8,
    madd=None,
    madd_diag=None,
):
    """Solve (M − M_added + coef·K)·x = b per env via matrix-free batched Jacobi-PCG.
    b_wp (B, ndof) must satisfy b[bc]=0. Returns (x (B,ndof), n_iters).

    `madd` (CSR M_added) + `madd_diag` (its diagonal, ndof) add the −M_added term
    (added mass). The operator is ill-conditioned so Jacobi PCG is essential.
    """
    device = device or config.DEVICE
    tol = tol if tol is not None else config.CR_TOL
    B = b_wp.shape[0]
    if B < 1 or ndof < 1 or b_wp.shape != (B, ndof):
        raise ValueError("CUDA structural solve requires a non-empty aligned batch")
    if max_iter < 1 or check_every < 1:
        raise ValueError("CUDA structural solve iteration controls must be positive")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("CUDA structural tolerance must be finite and positive")
    _require_finite_gpu_array(b_wp, device, "structural right-hand side")
    ne = edofs.shape[0]
    NP = config.NP_DTYPE
    coefD = DTYPE(NP(coef))
    madd_tmp = (
        wp.zeros((B, ndof), dtype=DTYPE, device=device) if madd is not None else None
    )

    def mv(vin, wout):  # S·v = (M − M_added + coef·K)·v
        apply_MK(
            vin,
            wout,
            Me,
            Kblk,
            edofs,
            free,
            1.0,
            coef,
            device,
            madd=madd,
            madd_tmp=madd_tmp,
        )

    # Jacobi preconditioner: diag(S)
    diag = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(
        sdiag_kernel,
        dim=(B, ne, edofs.shape[1]),
        inputs=[Me, Kblk, edofs, free, coefD],
        outputs=[diag],
        device=device,
    )
    if madd_diag is not None:
        wp.launch(
            _sub_bcast_kernel,
            dim=(B, ndof),
            inputs=[diag, madd_diag, free],
            device=device,
        )

    x = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    r = wp.clone(b_wp)
    z = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    p = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    Sp = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    rz = wp.zeros(B, dtype=DTYPE, device=device)
    rz_new = wp.zeros(B, dtype=DTYPE, device=device)
    pSp = wp.zeros(B, dtype=DTYPE, device=device)
    rr = wp.zeros(B, dtype=DTYPE, device=device)
    wp.launch(
        _minv_apply, dim=(B, ndof), inputs=[diag, free, r], outputs=[z], device=device
    )
    wp.copy(p, z)
    _dot(r, z, ndof, rz, device)
    _dot(b_wp, b_wp, ndof, rr, device)  # |b|² for relative residual
    bnorm2 = _gpu_max_scalar(rr, device) + 1e-300

    iters = 0
    converged = False
    for it in range(max_iter):
        mv(p, Sp)
        _dot(p, Sp, ndof, pSp, device)
        wp.launch(
            _pcg_update_solution_residual,
            dim=(B, ndof),
            inputs=[x, r, p, Sp, rz, pSp],
            device=device,
        )
        iters = it + 1
        if it % check_every == 0:
            _dot(r, r, ndof, rr, device)
            if math.sqrt(_gpu_max_scalar(rr, device) / bnorm2) < tol:
                converged = True
                break
        rz_new.zero_()
        wp.launch(
            _pcg_precondition_dot,
            dim=(B, ndof),
            inputs=[diag, free, r, z],
            outputs=[rz_new],
            device=device,
        )
        wp.launch(
            _pcg_update_direction,
            dim=(B, ndof),
            inputs=[p, z, rz_new, rz],
            device=device,
        )
        wp.copy(rz, rz_new)
    _require_finite_gpu_array(x, device, "structural solution")
    if not converged:
        _require_finite_gpu_array(r, device, "structural residual")
        _dot(r, r, ndof, rr, device)
        relative_residual = math.sqrt(_gpu_max_scalar(rr, device) / bnorm2)
        if not math.isfinite(relative_residual) or relative_residual >= tol:
            raise RuntimeError(
                "CUDA structural PCG did not converge after "
                f"{iters} iterations: relative residual={relative_residual:.17g}, "
                f"tolerance={tol:.17g}"
            )
    return x, iters


# ─── GPU Newmark step (block-reduced, matches modules/numerical_solver.step) ──


@wp.kernel
def _saxpy_kernel(
    y: wp.array(dtype=DTYPE, ndim=2), c: DTYPE, x: wp.array(dtype=DTYPE, ndim=2)
):
    e, d = wp.tid()
    y[e, d] = y[e, d] + c * x[e, d]


@wp.kernel
def _lincomb_mask(
    out: wp.array(dtype=DTYPE, ndim=2),
    c1: DTYPE,
    a: wp.array(dtype=DTYPE, ndim=2),
    c2: DTYPE,
    b: wp.array(dtype=DTYPE, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
):
    e, d = wp.tid()
    out[e, d] = free[d] * (c1 * a[e, d] + c2 * b[e, d])


@wp.kernel
def _copy_free_else(
    dst: wp.array(dtype=DTYPE, ndim=2),
    xfree: wp.array(dtype=DTYPE, ndim=2),
    base: wp.array(dtype=DTYPE, ndim=2),
    free: wp.array(dtype=DTYPE, ndim=1),
):
    """dst = free⊙xfree + (1-free)⊙base  (free DOFs from solve, BC DOFs held)."""
    e, d = wp.tid()
    if free[d] > DTYPE(0.5):
        dst[e, d] = xfree[e, d]
    else:
        dst[e, d] = base[e, d]


def gpu_newmark_step(
    q_n,
    dq_n,
    Kblk,
    Me,
    edofs,
    free,
    ndof,
    F_const,
    Qmem_n,
    Qbend_n,
    F_vel_n,
    recompute_bend,
    recompute_fvel,
    alpha_v,
    c_damp,
    dt,
    cg_tol=None,
    device=None,
    madd=None,
    madd_diag=None,
):
    """One block-reduced Newmark step on GPU (matches numerical_solver.step).

    q_n, dq_n, F_const, Qmem_n, Qbend_n, F_vel_n: (B, ndof).
    recompute_bend(q_p1) -> Qbend_p1 (B,ndof);  recompute_fvel(q_p1,dq_p1) -> F_vel_p1.
    Returns (q_new, dq_new).
    """
    device = device or config.DEVICE
    NP = config.NP_DTYPE
    B = q_n.shape[0]
    coef = alpha_v * c_damp * dt * dt / 2.0  # S = M + coef·K
    Dbl = c_damp * dt / 2.0  # D_bl = Dbl·K
    dtN = DTYPE(NP(dt))
    adt = DTYPE(NP(alpha_v * dt))

    tmp = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    tmp2 = wp.zeros((B, ndof), dtype=DTYPE, device=device)

    def solveA1(b1, b2):
        # rhs = b2 - D_bl·b1 ; x2 = S^{-1} rhs (PCG) ; x1 = b1 + alpha·dt·x2
        apply_MK(
            b1, tmp, Me, Kblk, edofs, free, 0.0, Dbl, device
        )  # tmp = D_bl·b1 (K only)
        rhs = wp.clone(b2)
        wp.launch(
            _saxpy_kernel, dim=(B, ndof), inputs=[rhs, DTYPE(-1.0), tmp], device=device
        )
        x2, _ = structural_cg(
            rhs,
            Me,
            Kblk,
            edofs,
            free,
            coef,
            ndof,
            tol=cg_tol,
            device=device,
            madd=madd,
            madd_diag=madd_diag,
        )
        x1 = wp.clone(b1)
        wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[x1, adt, x2], device=device)
        return x1, x2

    # homogeneous A2·X_n: b1 = free⊙(q + (1-α)dt·dq); b2 = D_bl·(free⊙q) + M·(free⊙dq)
    b1 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(
        _lincomb_mask,
        dim=(B, ndof),
        inputs=[b1, DTYPE(1.0), q_n, DTYPE(NP((1.0 - alpha_v) * dt)), dq_n, free],
        device=device,
    )
    qf = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    dqf = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(
        _lincomb_mask,
        dim=(B, ndof),
        inputs=[qf, DTYPE(1.0), q_n, DTYPE(0.0), q_n, free],
        device=device,
    )
    wp.launch(
        _lincomb_mask,
        dim=(B, ndof),
        inputs=[dqf, DTYPE(1.0), dq_n, DTYPE(0.0), dq_n, free],
        device=device,
    )
    apply_MK(qf, tmp, Me, Kblk, edofs, free, 0.0, Dbl, device)  # D_bl·qf (K only)
    apply_MK(
        dqf,
        tmp2,
        Me,
        Kblk,
        edofs,
        free,
        1.0,
        0.0,
        device,  # M_eff·dqf = (M−M_added)·dqf
        madd=madd,
        madd_tmp=wp.zeros((B, ndof), dtype=DTYPE, device=device)
        if madd is not None
        else None,
    )
    b2 = wp.clone(tmp)
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[b2, DTYPE(1.0), tmp2], device=device
    )
    a1, a2 = solveA1(b1, b2)

    zero = wp.zeros((B, ndof), dtype=DTYPE, device=device)

    # stage 0: Q_global = F_const + F_vel_n - (Qmem_n + Qbend_n)
    Qg = wp.clone(F_const)
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg, DTYPE(1.0), F_vel_n], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg, DTYPE(-1.0), Qmem_n], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg, DTYPE(-1.0), Qbend_n], device=device
    )
    # mask BC in rhs
    wp.launch(
        _lincomb_mask,
        dim=(B, ndof),
        inputs=[Qg, DTYPE(1.0), Qg, DTYPE(0.0), Qg, free],
        device=device,
    )
    s01, s02 = solveA1(zero, Qg)
    qp1f = wp.clone(a1)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[qp1f, dtN, s01], device=device)
    dqp1f = wp.clone(a2)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[dqp1f, dtN, s02], device=device)
    # full q_p1, dq_p1 (free from solve, BC held)
    q_p1 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    dq_p1 = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(
        _copy_free_else, dim=(B, ndof), inputs=[q_p1, qp1f, q_n, free], device=device
    )
    wp.launch(
        _copy_free_else, dim=(B, ndof), inputs=[dq_p1, dqp1f, dq_n, free], device=device
    )

    # stage 1: averaged bending + velocity
    Qbend_p1 = recompute_bend(q_p1)
    F_vel_p1 = recompute_fvel(q_p1, dq_p1) if recompute_fvel is not None else zero
    Qg2 = wp.clone(F_const)
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(0.5), F_vel_n], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(0.5), F_vel_p1], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(-1.0), Qmem_n], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(-0.5), Qbend_n], device=device
    )
    wp.launch(
        _saxpy_kernel, dim=(B, ndof), inputs=[Qg2, DTYPE(-0.5), Qbend_p1], device=device
    )
    wp.launch(
        _lincomb_mask,
        dim=(B, ndof),
        inputs=[Qg2, DTYPE(1.0), Qg2, DTYPE(0.0), Qg2, free],
        device=device,
    )
    s11, s12 = solveA1(zero, Qg2)
    qnf = wp.clone(a1)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[qnf, dtN, s11], device=device)
    dqnf = wp.clone(a2)
    wp.launch(_saxpy_kernel, dim=(B, ndof), inputs=[dqnf, dtN, s12], device=device)
    q_new = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    dq_new = wp.zeros((B, ndof), dtype=DTYPE, device=device)
    wp.launch(
        _copy_free_else, dim=(B, ndof), inputs=[q_new, qnf, q_n, free], device=device
    )
    wp.launch(
        _copy_free_else, dim=(B, ndof), inputs=[dq_new, dqnf, dq_n, free], device=device
    )
    return q_new, dq_new
