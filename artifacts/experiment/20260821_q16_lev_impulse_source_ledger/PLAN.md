# Q16 LEV Impulse Source Ledger Plan

## 1. Objective

- run id: `20260821_q16_lev_impulse_source_ledger`
- selected idea: retain the exact spanwise source-strip identity of every CUDA
  LEV ring particle and decompose the already-authorized global
  free-vortex/bound-sheet impulse into source-strip contributions without
  changing the frozen global force formula.
- user's core requirement: continue the real Q16 + predictor/corrector + FSI
  path with separated LEV, joint TEV and free wake always active.
- non-negotiable constraints: CUDA float64 scientific arrays; no Q9/toy
  substitute; no attached-flow fallback; no global-force area/node smearing;
  no change to the global aerodynamic force result; no paper/GT/scorer run.
- research question: can each non-zero multi-step LEV impulse force be given a
  durable causal span-strip owner while exactly closing back to the frozen
  global impulse force?
- null hypothesis: particle history has lost the shedding strip identity, so
  the impulse cannot be localized without inventing a distribution.
- alternative hypothesis: storing the ring index on every shed leg and
  carrying it through convection/removal permits an auditable per-strip
  impulse ledger whose sum closes the existing global result.
- preflight clarification: an initial unnumbered screen read appeared to show
  a duplicate roll assignment; the subsequent line-numbered source check
  showed one roll of the original `rings`. The independent four-edge oracle
  confirms that current forward/reverse orientation is already correct.

## 2. Baseline And Comparability

- baseline commit: `dc43d4cc0c2290ee34df942e51cbb05e13afbb0d` plus the
  active Q16 branch state recorded by
  `20260821_q16_incremental_endpoint_motion`.
- last-known-good integration: endpoint-motion focused `4/4`, incremental
  owner/geometry/motion `15/15`, bounded joint surface `131/131`.
- baseline code hashes:
  - `pfield_torch_gpu.py`: `0a45e6717dd0703fef358ad46b39a79f627337466e57df7c3923cb0fe15cd7ff`
  - `bing_joint_ptera_gpu.py`: `52222cc16916395ad7b1ebfd2dff327ed9a3e047e4c961f40ae038a8fb3bc983`
  - `q16_incremental_ptera_owner.py`: `1066fa6aaa5ff7bfa6d4bdfedf675ce9492de2744bbf8c4714ea68594dcff258`
- dataset/split: bounded production-path CUDA 2x3-panel separated-flow pilot;
  no paper data.
- primary metrics: ring-leg source identity, source identity persistence,
  per-strip/global impulse closure, parent transaction invariance.
- comparability rule: the existing global impulse and total-force reduction
  remain authoritative and bitwise unchanged; the new strip ledger is an
  additional decomposition, not a replacement force model.
- pre-change two-step oracle (`LESP=0.001`, 2x3 panels): particle count `24`,
  packet SHA `e7c051e91c0e02f5b5d3bc8c9eb28f8a2a4439837baf213059b56908dc8e7659`,
  impulse components
  `0x1.2c175830af8fdp+7, 0x1.ea00000000001p-50,
  -0x1.f1c223732f08cp+4`; these must remain bitwise identical.

## 3. Code Translation Plan

| Path | Change | Scientific reason |
|---|---|---|
| `platform/warp_vpm/pfield_torch_gpu.py` | add CUDA `int64` source-strip storage, exact input gate, ring-derived identities, removal/snapshot persistence | particles must retain which strip shed each correctly oriented ring leg |
| `platform/warp_vpm/bing_joint_ptera_gpu.py` | compute/store per-strip impulse and force alongside the unchanged global reduction; retain current LE endpoints | establish causal strip ownership and a later surface-work anchor |
| `platform/warp_vpm/q16_incremental_ptera_owner.py` | bind new tensors into the scientific state hash | branch/retry semantics must detect ledger drift |
| `platform/warp_vpm/test_q16_lev_impulse_source_ledger.py` | tests-first source, persistence, closure, tamper and retry gates | distinguish a real source ledger from a post-hoc force split |

## 4. Execution Design

- minimal test: independently enumerate the four adjacent directed edges of
  two CUDA rings in forward and reverse senses, confirm exact position/gamma/
  sigma/circulation and strip IDs, then remove/reorder particles and confirm
  IDs follow their exact particles.
- real pilot: advance at least two production separated-flow steps so the
  impulse force is non-zero; verify each strip force is finite and the strip
  sum closes the unchanged global impulse force.
- hostile tests: CPU/wrong-dtype/source-shape inputs; invalid strip IDs;
  source tensor mutation in an incremental branch; clean retry.
- stop condition: any unknown particle owner in the production LEV impulse,
  non-finite strip result, closure failure, reduced aerodynamic mode, or parent
  mutation.
- explicit non-claim: this checkpoint does not yet apply impulse force to Q16
  generalized coordinates.  The next gate must define a source-linked surface
  work operator and verify force, moment and virtual work.

## 5. Runtime And Evidence

- CUDA device: configured production CUDA device, float64.
- budget: bounded focused and joint regression only; under three minutes.
- artifact directory: this directory.
- managed `bash_exec`, artifact and memory services are unavailable in this
  environment; local non-interactive commands and repository artifacts are the
  disclosed fallback.

## 6. Next Gate

After this ledger passes, construct a Q16 work operator from each strip's
actual leading-edge endpoint pair.  The operator must use the same interpolation
rows that generated the Ptera endpoints, preserve resultant and midpoint
moment, pass an independent virtual-work oracle, and replace—not hide—the
current unresolved-impulse stop.

## 7. Revision Log

| Date | Change | Impact |
|---|---|---|
| 2026-08-21 | Initial source-ledger contract | Auxiliary integration only; no paper claim |
