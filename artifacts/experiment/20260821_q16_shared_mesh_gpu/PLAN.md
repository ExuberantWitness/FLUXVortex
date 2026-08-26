# Q16 shared-node mesh and deterministic CUDA assembly

## 1. Objective

- run id: `20260821_q16_shared_mesh_gpu`
- selected idea: connect fixed Q16 macro-elements through one shared global node
  state and retain deterministic matrix-free CUDA gather/assembly.
- user requirements: Q16 only; no Q9 intermediate; GPU-parallel structural
  path; later compatible with mandatory separated LEV/free-wake FSI.
- research question: can the verified element-local Q16 residual/mass/Jv be
  assembled on a shared high-order mesh without duplicate interface ownership,
  atomics or CPU production calculation?
- null hypothesis: local 96-DOF operators cannot be composed without changing
  their verified equations or losing deterministic accumulation.
- alternative hypothesis: fixed connectivity plus node-to-element CSR provides
  exact CPU/CUDA parity and shared-edge continuity.

## 2. Baseline and comparability

- baseline: the green element-local Q16 continuum/transfer checkpoint on branch
  `run/q16-lev-tev-pc-fsi-20260821`.
- primary metrics: node/connectivity exactness, energy-force derivative, mass,
  CUDA force/mass/Jv parity, repeated bitwise output.
- comparability risk: this remains unprojected StVK and is not a thin-shell or
  FSI validation.

## 3. Code translation

| Path | Change | Risk |
|---|---|---|
| `src/fluxvortex/q16_ancf_mesh.py` | exact global node/connectivity schema, rectangular Q16 mesh, CPU oracle assembly | edge ordering drift |
| `src/fluxvortex/warp_fsi/kernels_q16_mesh.py` | CUDA global-to-local gather and stable CSR local-to-global assembly | duplicate/missing contributions |
| `tests/test_q16_ancf_shared_mesh_gpu.py` | shared edge, energy/mass/Jv, CUDA parity and attacks | toy-only coverage |

## 4. Execution design

- minimal experiment: two chordwise Q16 elements, then a 2x2 map.
- smoke: focused shared-mesh tests.
- full run: joint all current Q16 tests plus V5M FSI/active-LEV baselines.
- stop: duplicate global ownership, non-continuous shared edge, force/Jv mismatch,
  nondeterministic repeated CUDA assembly or host-array acceptance.
- abandonment: if deterministic CSR requires modifying the verified element
  equations, stop and redesign connectivity rather than use atomics.

## 5. Runtime strategy

- GPU: existing RTX 4090 D, CUDA float64.
- expected runtime: under 10 minutes for tests.
- outputs: this directory plus exact source/test hashes and summary.
- tool fallback: required `bash_exec/artifact/memory` tools are unavailable;
  commands use the available terminal and durable local summaries.

## 6. Recovery

- no fallback to Q9, disconnected elements, CPU production assembly or atomic
  floating-point scatter.
- a failed shared-node gate blocks ANS/EAS, nonlinear stepping and FSI.

## 7. Checklist

- `CHECKLIST.md`

## 8. Revision log

| Date | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-21 | initial shared-mesh contract | element-local kernel is green | none |
