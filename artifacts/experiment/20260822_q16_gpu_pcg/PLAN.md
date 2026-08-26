# Q16 GPU PCG and Frozen-Linearization Plan

## 1. Run contract

- run id: `20260822_q16_gpu_pcg`
- tier: `auxiliary/dev` solver-performance checkpoint
- hypothesis: a CUDA-resident material/mass Jacobi preconditioner, evaluated
  from one frozen Q16 Newton linearization, reduces total Krylov work and wall
  time for the existing nonzero strong-FSI structural solve without changing
  the governing residual, accepted solution, mandatory separated-flow path or
  transactional publication semantics.
- baseline: the unpreconditioned matrix-free Newmark--Newton--CG solver and the
  bounded real strong-FSI step frozen on 2026-08-22.

## 2. Frozen algorithm

1. At each Newton correction, evaluate the current MITC16/ANS/EAS strain,
   projected B operator and condensed EAS parameter exactly once on CUDA.
2. Reuse that immutable linearization for every Krylov matrix-vector product.
3. Form a CUDA float64 Jacobi diagonal from the exact consistent-mass diagonal
   plus the positive condensed material-tangent diagonal at that same state.
   The nonlinear geometric tangent remains in the matrix-vector product; it is
   not silently removed from the governing equation.
4. Run left-preconditioned conjugate gradients with the original, unscaled
   residual norm as the convergence oracle.
5. Reject non-finite/non-positive diagonal entries, non-positive curvature,
   nonconvergence or any host/device/dtype drift before publishing a result.

## 3. Non-negotiable constraints

- Q16 only; no Q9, reduced structural model or CPU numerical fallback.
- separated LEV, joint TEV and free wake remain mandatory in the integration
  test; this work may not weaken those gates.
- all linearization, diagonal construction, PCG vector operations and dot
  products remain CUDA float64.
- the preconditioner changes convergence only, never the nonlinear residual or
  accepted Newton state.
- failed solve leaves the structural and aerodynamic parents unchanged and a
  clean retry must remain equivalent to a fresh owner.

## 4. Red/green tests

- frozen baseline records unpreconditioned CG count and synchronized wall time
  for one high-modulus, nonzero Q16 step.
- PCG returns the same converged state/velocity/acceleration within the existing
  nonlinear tolerance and uses fewer Krylov iterations.
- malformed diagonal/non-positive curvature/nonconvergence fail closed.
- the existing real strong-FSI success, injected failure, rollback and clean
  retry gates remain green with mandatory separated flow active.

## 5. Acceptance

- focused structural comparison shows a strict CG iteration reduction and no
  material change in the converged nonlinear solution.
- the bounded nonzero real FSI step still converges and commits each owner once.
- measured synchronized wall time improves; otherwise the implementation is
  retained only as numerical groundwork and is not claimed as a speedup.
- selected Q16/LEV/TEV/FSI suites and static/whitespace gates pass.
- no paper case, GT, scorer or formal matrix is accessed.

## 6. Tooling

The managed experiment shell/artifact/memory tools are unavailable in this
session. Local non-interactive commands and this versioned artifact directory
are the disclosed fallback.

## 7. Claim boundary

Passing supports this bounded Q16 solver and one-step FSI workload. It does not
yet establish large-mesh scaling, multi-step stability, constitutive-range
generality, experimental accuracy or paper-level validation.

## 8. Revision log

| Date | Revision | Reason |
|---|---|---|
| 2026-08-22 | Initial freeze | Remove repeated linearization and add GPU PCG |
