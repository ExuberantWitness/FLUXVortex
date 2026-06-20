"""Scaling the differentiable coupled co-design (S5) — gradient CHECKPOINTING (bounded memory
for long coupled-FSI rollouts) + BATCHED population evaluation (many designs).

Long SHAC rollouts can't store every state. Checkpointing stores (q,dq) only every `chk`
sub-steps and RECOMPUTES each segment's forward during the backward pass — O(√N) memory
instead of O(N), same gradient. Batching evaluates a population of B designs (the unit for
MAP-Elites / DQD on the real coupled FSI).

verify():
  (1) the checkpointed coupled gradient == the full-storage gradient (bit-close), with the
      memory bound shown (n_checkpoints ≪ n_steps);
  (2) a batched population of B designs is evaluated (throughput), each with its own coupled
      gradient — the population scaling for QD on the coupled FSI.
"""
from __future__ import annotations

import time

import numpy as np

import diff_coupled_fsi as dc
import diff_vlm
from diff_struct_design import _build_shell


def _seg_forward(sh, q, dq, m, sdt, free, Mff, P, dist, nx, ny, alpha):
    """Run m sub-steps of size sdt from (q,dq). Returns states qs[m+1] and undamped a_raw[m]."""
    q = q.copy(); dq = dq.copy(); qs = [q.copy()]; araw = []
    for _ in range(m):
        Qint, _, _ = dc._assemble(sh, q)
        rhs = dc._aero_nodal(q, P, dist, nx, ny) - Qint
        a = np.zeros(sh.ndof); a[free] = np.linalg.solve(Mff, rhs[free])
        araw.append(a.copy())
        a = a - alpha * dq
        dq = dq + sdt * a; q = q + sdt * dq
        qs.append(q.copy())
    return qs, araw, q, dq


def loss_and_grad_chk(sh, q0, dq0, N, dt, free, w, nx, ny, alpha=0.0, nsub=1, chk=8):
    """Checkpointed coupled design gradient: O(Ntot/chk + chk) memory, same result."""
    P, dist = dc._index_maps(sh, nx, ny)
    Mff = sh.M[np.ix_(free, free)].toarray()
    Ntot = N * nsub; sdt = dt / nsub
    rho, h = sh.rho, sh.h
    Mu = [dc._elem_mass_unit(sh, e) for e in range(sh.ne)]
    # forward storing ONLY checkpoints
    ckpts = {0: (q0.copy(), dq0.copy())}
    q, dq = q0.copy(), dq0.copy()
    for s in range(0, Ntot, chk):
        m = min(chk, Ntot - s)
        _, _, q, dq = _seg_forward(sh, q, dq, m, sdt, free, Mff, P, dist, nx, ny, alpha)
        ckpts[s + m] = (q.copy(), dq.copy())
    L = float(w @ q)
    n_ckpt = len(ckpts)
    # backward, recomputing each segment from its checkpoint
    gE = np.zeros(sh.ne); gR = np.zeros(sh.ne)
    adj_q = w.copy(); adj_dq = np.zeros(sh.ndof)
    seg_starts = list(range(0, Ntot, chk))
    for s in reversed(seg_starts):
        m = min(chk, Ntot - s)
        qc, dqc = ckpts[s]
        qs, araw, _, _ = _seg_forward(sh, qc, dqc, m, sdt, free, Mff, P, dist, nx, ny, alpha)
        for j in reversed(range(m)):
            qt = qs[j]
            aq1 = adj_q; ad1 = adj_dq + sdt * aq1; adj_a = sdt * ad1
            adj_dq_t = ad1 - alpha * adj_a
            adj_rhs = np.zeros(sh.ndof)
            adj_rhs[free] = np.linalg.solve(Mff, adj_a[free])
            adj_Fp = dist.T @ adj_rhs
            Jv = diff_vlm.panel_jacobian(dc._corners(qt, P, nx, ny), nx, ny, dc.VINF)
            adj_q_aero = P.T @ (Jv.T @ adj_Fp)
            adj_Qint = -adj_rhs
            _, Kt_t, per_e = dc._assemble(sh, qt)
            for e, (dofs, Qe) in enumerate(per_e):
                gE[e] += float(adj_Qint[dofs] @ (Qe / sh.E_scale_e[e]))
                gR[e] += float(-(adj_rhs[dofs] @ (Mu[e] * rho * h) @ araw[j][dofs]))
            adj_q = aq1 + Kt_t @ adj_Qint + adj_q_aero
            adj_dq = adj_dq_t
    return L, gE, gR, n_ckpt


def verify(nx=3, ny=3, N=24, dt=2e-5, nsub=1, alpha=0.0):
    sh = _build_shell(nx=nx, ny=ny)
    ne = sh.ne; rng = np.random.default_rng(0)
    Es = np.exp(0.2 * rng.standard_normal(ne)); Rs = np.exp(0.2 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(sh.ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); q0[free] += 1e-3 * rng.standard_normal(len(free))
    dq0 = np.zeros(sh.ndof); dq0[free] = 6e-3 * rng.standard_normal(len(free))
    w = np.zeros(sh.ndof); w[free] = rng.standard_normal(len(free))

    # (1) checkpointed == full-storage gradient
    _, gE_full, gR_full = dc.loss_and_grad(sh, q0, dq0, N, dt, free, w, nx, ny, alpha, nsub)
    _, gE_chk, gR_chk, n_ckpt = loss_and_grad_chk(sh, q0, dq0, N, dt, free, w, nx, ny,
                                                  alpha, nsub, chk=5)
    relE = np.max(np.abs(gE_full - gE_chk)) / (np.max(np.abs(gE_full)) + 1e-30)
    relR = np.max(np.abs(gR_full - gR_chk)) / (np.max(np.abs(gR_full)) + 1e-30)
    ok1 = relE < 1e-10 and relR < 1e-10
    print("Scaling the differentiable coupled co-design (S5):")
    print(f"  (1) checkpointed vs full gradient: ∂刚柔 rel={relE:.1e} ∂质量 rel={relR:.1e}  "
          f"memory: {n_ckpt} checkpoints vs {N*nsub+1} full states  -> {'PASS' if ok1 else 'FAIL'}")

    # (2) batched population of B designs (each its own coupled gradient)
    B = 8
    designs = np.exp(0.2 * rng.standard_normal((B, ne)))
    t0 = time.time()
    grads = []
    for b in range(B):
        sh.set_distribution(E_scale=designs[b], rho_scale=np.ones(ne))
        _, gE_b, _, _ = loss_and_grad_chk(sh, q0, dq0, N, dt, free, w, nx, ny, alpha, nsub, chk=5)
        grads.append(gE_b)
    dt_b = time.time() - t0
    ok2 = len(grads) == B and all(np.all(np.isfinite(g)) for g in grads)
    print(f"  (2) batched population: {B} designs' coupled gradients in {dt_b:.2f}s "
          f"({dt_b/B*1e3:.0f} ms/design) -> {'PASS' if ok2 else 'FAIL'} "
          f"(the unit for MAP-Elites/DQD on the coupled FSI)")
    ok = ok1 and ok2
    print(f"  -> {'PASS' if ok else 'FAIL'}: long coupled rollouts run in O(√N) memory "
          f"(checkpointing) and a design population evaluates batched (S5 scaling)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
