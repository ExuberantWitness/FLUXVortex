# Q16 LEV Impulse Work Transfer Plan

## 1. Objective

- run id: `20260821_q16_lev_impulse_work_transfer`
- selected idea: define each source-owned span-strip impulse force as acting at
  the midpoint velocity of the same two leading-edge Q16 interpolation rows
  that generated the strip's real Ptera source endpoints. Algebraically this
  is one half-force on each endpoint followed by the exact Q16 transpose.
- non-negotiable constraints: mandatory separated LEV, joint TEV and free
  wake; CUDA float64; no Q9/toy structure; no area/node smear of a global
  force; no host numerical fallback; no paper/GT/scorer run.
- research question: does this explicit source-line work contract produce a
  Q16 generalized force that exactly preserves strip resultant, midpoint
  moment and virtual work on the real two-step branch?
- null hypothesis: the span ledger still lacks enough kinematic ownership for
  a work-conjugate Q16 map.
- alternative hypothesis: using the exact two endpoint interpolation rows
  makes the declared midpoint contract algebraically closed and auditable.

## 2. Scientific Scope

This is a **model closure**, not a derivation of a unique local pressure field
from global vortex impulse. The contract is scientifically narrower:

`delta W_strip = F_strip dot 0.5*(delta x_left + delta x_right)`.

It is causal because the two rows are the actual Q16 rows that formed the LEV
source ring. It preserves the moment of a force at the source-line midpoint.
It does not claim a chordwise traction distribution or an aerodynamic impulse
couple not present in the current solver.

## 3. Baseline

- source-ledger artifact: `20260821_q16_lev_impulse_source_ledger`.
- source-ledger code hashes:
  - particle field `81e189f7...f229`
  - CUDA solver `58e3fc8e...fac5`
  - incremental owner `d08e6013...6ebe`
- last joint gate: `146/146 PASS`.
- real two-step load: 3 strips, 24 particles, nonzero global impulse, packet
  SHA `e7c051e91c0e02f5b5d3bc8c9eb28f8a2a4439837baf213059b56908dc8e7659`.

## 4. Implementation Map

| Path | Change |
|---|---|
| `src/fluxvortex/warp_fsi/q16_lev_impulse_transfer.py` | exact frozen CUDA strip-load object and Q16 endpoint transpose |
| `tests/test_q16_lev_impulse_transfer_gpu.py` | synthetic algebraic oracle, real incremental Q16 branch, tamper/geometry/device gates |

No aerodynamic or structural equation is changed in this checkpoint.

## 5. Acceptance Gates

- real load is obtained only after two incremental separated-flow steps with
  Q16 position and velocity binding;
- load seal binds strip forces, source endpoints and particle source IDs;
- current Q16 interpolated leading-edge points equal solver endpoints within a
  scaled float64 geometry gate;
- generalized resultant equals strip resultant;
- generalized moment equals midpoint force moment;
- structural virtual work equals source-line midpoint virtual work;
- altered load, geometry, device, dtype or source identity rejects;
- existing unresolved-impulse transfer remains fail-closed until the combined
  resolved+impulse operator is implemented.

## 6. Execution

- tier: auxiliary/dev production-path integration.
- budget: bounded GPU tests and joint regression, under three minutes.
- managed `bash_exec`, artifact and memory services are unavailable; local
  non-interactive commands and hashed repository artifacts are the disclosed
  fallback.
- no paper matrix, GT, scorer or observation access.

## 7. Next Gate

After this operator passes, compose it with the real resolved Ptera point-load
kinematic operator and the Q16 Newton/predictor transaction. A structural step
may commit only when their combined force, moment and virtual work close and
the aerodynamic branch commits atomically with it.
