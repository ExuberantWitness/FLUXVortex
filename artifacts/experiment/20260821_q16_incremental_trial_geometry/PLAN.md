# Q16 Incremental Trial Geometry Plan

## 1. Objective

- run id: `20260821_q16_incremental_trial_geometry`
- selected idea: bind only a trusted incremental aerodynamic branch's next
  Panel grid to CUDA-interpolated Q16 `q_trial`, then let that candidate
  geometry create its own pending wake and aerodynamic load.
- baseline: bitwise-valid incremental owner checkpoint
  `20260821_q16_incremental_aero_owner`.
- mandatory mode: separated LEV + joint TEV + free wake; no fallback.
- research question: from one committed aerodynamic parent, do different Q16
  trial shapes produce isolated, deterministic differences in real wake and
  load state without mutating the parent?

## 2. Scientific Contract

- geometry is the Q16 interpolated midsurface in world coordinates, transformed
  on CUDA float64 into the exact next operating-point frame.
- the complete structured Panel owner is rebuilt, validated, detached and
  atomically installed only on the branch's exact `next_step`.
- previous committed panels and all parent wake/LEV/TEV state remain immutable.
- Ptera movement velocity remains its original causal finite difference between
  committed previous geometry and candidate current geometry.
- `dq_trial` is deliberately not claimed as consumed in this unit.  Newmark
  endpoint velocity and Ptera interval velocity require a separate temporal
  interpolation contract; silently treating them as equal is forbidden.
- active-LEV impulse remains unresolved and therefore cannot complete the Q16
  generalized-force transfer.

## 3. Code And Tests

| Path | Change |
|---|---|
| `platform/warp_vpm/q16_incremental_ptera_owner.py` | add session-authorized, detached next-Panel owner replacement and reseal |
| `platform/warp_vpm/q16_ptera_trial_kinematics.py` | add incremental single-state CUDA Q16 geometry binder |
| `platform/warp_vpm/test_q16_incremental_trial_geometry.py` | actual branch/wake/load isolation, determinism and hostile lifecycle tests |

## 4. Gates

- same parent + same q => identical scientific receipt and packet SHA.
- same parent + discriminative q => different next geometry, real wake and load
  SHA, with nonzero separated LEV and joint TEV.
- parent pickle SHA, counters, particles and step index remain exact.
- input mutation, host/mixed dtype, wrong topology, panel discontinuity,
  non-next-step binding and post-advance rebinding fail closed.

## 5. Scope And Runtime

- bounded synthetic geometry mechanism test on RTX 4090 D, CUDA float64.
- no paper matrix, GT or scorer.
- no full structural Newton loop until temporal velocity and impulse-work gates
  are closed.
- managed experiment services remain unavailable; local commands plus durable
  repository artifacts are the disclosed fallback.

## 6. Revision Log

| Time | Change | Reason |
|---|---|---|
| 2026-08-21 | Initial Q16 incremental branch geometry contract | next causal FSI data-path unit after incremental owner parity |
