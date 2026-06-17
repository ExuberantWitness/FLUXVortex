# P1 differentiability finding — warp_fsi ANCF backward/adjoint

**Status:** forward bit-exact (12/12 layers on the 4090); **backward/adjoint of the
ANCF bending operator returns NaN**, precisely located. Differentiability (Warp
tape through the structural step) is **not yet working** for the ANCF bending path.

## What works (Warp autodiff confirmed)
Verified `tape == FD` (rel ≤ 1e-9) for, in increasing complexity:
- trivial `y = x²`;
- the **Featherstone rigid solver** (P0: free-flight + revolute + servo + spring);
- `normalize(cross(a,b))·c` (manual `/√(·)` and `wp.normalize` builtin);
- loop-accumulate via `@wp.func` → `normalize` → `dot`;
- **gather** `q[e, edofs[el,a]]` (adjoint = atomic scatter) + multi-element launch
  + overlapping `edofs` + `normalize` → all differentiable.

## What fails
- `assemble_internal_force_sep` (the ANCF internal force) → `tape.backward` gives
  `q.grad = NaN`, while central-difference FD gives finite, correct values
  (e.g. 3.59e6). Holds even with **all intermediates and constant arrays
  `requires_grad=True`**, so it is **not** a missing-grad-buffer issue.
- Isolated to `ancf_force_gauss_kernel`, the **bending curvature-gradient** block:
  ```
  nvec = cross(dxr, dyr); nn = length(nvec); nhat = nvec/nn
  dn   = cross(dxr, sya) - cross(dyr, sxa)
  dnh  = (dn - nhat*dot(nhat, dn)) / nn        # <-- adjoint NaNs here
  dk[...] = dot(d2x, dnh) + dot(nhat, dxx)
  ```
  The membrane-only assembly (no normalize) is fully differentiable.

  **The trigger resists simplified reproduction.** A standalone kernel reproducing
  the *exact* bending structure — loop-accumulated `dxr`/`dyr` from distinct
  shape-derivative coefficients (so `nn≈1`), `nn = wp.length(cross(dxr,dyr))` (and
  manual `sqrt(dot)`), the `dnh = (dn - n̂(n̂·dn))/nn` projection, summed over a
  per-`a` loop — is **differentiable** (`tape==FD`, rel 2.2e-9). Ruled out as the
  cause: `wp.length` builtin (manual `sqrt` behaves identically), the projection
  structure itself, the dynamic gather + atomic scatter, multi-element launch,
  loop-accumulation via `@wp.func`, and near-degenerate normals (the forward is
  validated bit-exact and well-behaved, so no Gauss point has small `nn`).
  What remains unique to the real kernel: 36-DOF elements over a 9-DOF-per-node
  `q` with the real `edofs` gather and the 4-D `_col3`, at full mesh scale. The
  NaN therefore needs **instrumented debugging of the actual kernel** (locate the
  first non-finite adjoint by Gauss point / DOF), not more black-box isolation.

## Fix (next focused work — the plan's flagged largest-effort item)
1. Provide a **custom adjoint** for the bending-normal-gradient contribution:
   wrap the `(nvec, dxr, dyr, sxa, sya, d2x, dxx) → dk_a` map in a `@wp.func` with
   an explicit `@wp.func_grad`, hand-deriving the Jacobian of the tangent-plane
   projection (avoids Warp's auto-adjoint of the nested normalize/cross).
2. **Or** restructure: kernel-1 writes per-Gauss `(dxr, dyr, d2x, d2y, d2xy, nhat,
   nn)` to arrays; kernel-2 reads them as clean inputs and assembles `dk` — so the
   normalize and the cross don't share a loop-carried nonlinear input.
3. Re-validate **per-kernel `tape == FD`** + keep the forward bit-exact golden
   (12-layer suite must stay 12/12).

## Why this does NOT block the near-term critical path
The plan chose **PPO-first (model-free) control** *precisely because* differentiable
long-rollout gradients are hard (P0 also showed Featherstone backward degrades over
long horizons). **PPO needs no structural adjoint.** The DQD design search can use
the gradients that already work + finite differences at coarse scale. The structural
adjoint enablement above is required for the **SHAC upgrade (later phase)**, not for
iteration-1. The forward coupled FSI (committed, GPU-stable) is the near-term gate,
and it is met.

## Repro
`platform/verify_redlines.py` (forward red lines, PASS) and the isolation scripts
under `/tmp/tape_*.py`, `/tmp/trig*.py` (kept as scratch). The backward NaN repro:
build `ANCFConstants`, run `assemble_internal_force_sep` inside a `wp.Tape` with a
`requires_grad` `q`, `tape.backward(loss=…)` → `q.grad` is NaN.
