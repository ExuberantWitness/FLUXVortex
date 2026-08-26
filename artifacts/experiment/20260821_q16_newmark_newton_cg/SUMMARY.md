# Q16 CUDA Newmark--Newton--CG result

Result: **PASS for one GPU-only structural trial**.

The projected shared-node Q16 model now advances with Newmark average
acceleration and a matrix-free Newton--CG correction. Internal force, mass
action, tangent action, vector updates and batched CG are evaluated on CUDA;
the host reads only bounded convergence/failure scalars and selects active
batches. No dense global tangent or CPU numerical fallback is used.

The reference state is bitwise stationary with a roundoff free residual of
`9.13e-12`. The registered batch-2 load converges in one Newton correction and
32 CG iterations to relative residual `3.00e-11`, preserving all 24 clamped
single-element DOFs exactly. Forced nonconvergence raises without mutating any
input; clean retry is bitwise equal to a fresh solve. Host/float32/wrong shape,
nonfinite, boundary drift and underflow/overflow time-step attacks reject.

A warm development diagnostic with equal maximum load reduced time per sample
from `49.4 ms` at batch 1 to `5.06 ms` at batch 32. This is evidence that the
expensive path is batch-parallel, not a production-scale speedup claim. The
current correctness slice uses unpreconditioned CG and no Newton line search;
both require profiling/robustness work before long-time FSI.
