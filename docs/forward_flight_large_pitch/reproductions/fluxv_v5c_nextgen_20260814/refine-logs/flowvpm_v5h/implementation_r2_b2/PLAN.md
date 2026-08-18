# v5h R2/B2 conservative TE bridge plan

## 1. Objective

- run id: `20260814_fluxv_v5h_r2_b2_te_shadow`
- tier: `auxiliary/mechanical prerequisite`
- objective: construct a topology-aware, conservative ring-edge to vector-particle shadow mapping after R0–R1 direct transport parity passed.
- hypothesis: a global directed-edge graph can remove duplicate shared edges and map each physical edge to vector-particle strength `Gamma_p = gamma_edge * delta_l` while preserving incidence, vector moment, lineage, and a single physical owner.
- hard scope boundary:
  - shadow particles do not feed Ptera AIC, wake, loads, or rVPM time marching;
  - no LEV/LESP birth law;
  - no target-paper observations or scores;
  - no modification of legacy particle/solver or v4b files;
  - no FMM, SFS, viscosity, clips, coordinate-rounding topology, or target tuning.

## 2. Baseline and contracts

- topology baseline: Ptera's shared `wing.gridWrvp_GP1_CgP1[row, span_node]` vertices with explicit stable node IDs. Independent `RingVortex` corner objects are materialized views, not topology identity.
- field baseline: source ring/finite-segment induced velocity at probes outside the source core.
- particle backend: R0–R1 Gaussian-erf direct backend.
- edge assembly:
  1. each Ptera ring supplies `BR -> FR -> FL -> BL -> BR`, its source traversal order;
  2. a canonical edge key uses node identity, not rounded coordinates;
  3. edge circulation is the signed sum of incident ring circulations;
  4. identical node IDs with inconsistent coordinates fail closed;
  5. zero-net internal edges may be removed only after the signed ledger is recorded;
  6. edge merging is allowed only within an identical kernel channel (`kernel_family`, core law, age law). Different Ptera/material kernels may share a topological edge ID but remain distinct source channels.
- deposition:
  - split each retained edge into an explicit number of equal subsegments;
  - particle position is the subsegment midpoint;
  - particle vector strength is `gamma_edge * delta_l_subsegment`;
  - `sigma = lambda * h` with `h = |delta_l_subsegment|` and frozen `lambda=2.125` for this shadow experiment;
  - stable particle ID and lineage contain source edge, subdivision index, ring incidences, step, and owner state.

## 3. Implementation touchpoints

Only new files are authorized:

- `src/fluxvortex/rvpm_edge_bridge.py`
- `tests/test_rvpm_edge_bridge.py`
- `tools/v5h_flowvpm_oracle/evaluate_edge_bridge.py` (deterministic evidence writer only)
- `tests/test_rvpm_edge_bridge_artifact.py` (artifact/hash/coverage regression only)
- optional isolated Ptera read-only adapter/test after its source semantics are audited:
  - `platform/forward_flight_benchmarks/v5h_ptera_te_shadow.py`
  - `platform/tests/test_v5h_ptera_te_shadow.py`

R0–R1 files may be imported but not modified unless an independently reproduced bug blocks B2.

## 4. Mechanical gates

### G0 — exact-off and validation

- disabled dispatcher does not evaluate bridge inputs and returns no feedback field;
- malformed topology, repeated IDs with inconsistent coordinates, nonfinite values, nonpositive sigma/length, and unsupported ownership transitions fail closed;
- production topology never uses floating-coordinate rounding as identity.

### G1 — edge incidence

- one ring produces four retained edges with the original orientation ledger;
- two adjacent equal-strength rings cancel their shared edge exactly;
- unequal adjacent strengths leave exactly the signed circulation difference;
- orientation reversal changes only the signed incidence representation and not the physical edge result;
- incidence residual and edge reconstruction residual `<=1e-12`.

### G2 — conservative deposition

- for every edge, `sum(Gamma_p) = gamma_edge * (x_end-x_start)` to `<=1e-14` absolute and `<=1e-12` relative;
- global vector moment equals the edge graph ledger to the same tolerance;
- all IDs/lineage are unique and stable under deterministic re-execution;
- clip/nonfinite/owner-conflict counts are zero.

### G3 — field convergence in shadow mode

- use analytic single segment and rectangular ring cases with probes at least four finest sigmas from all source segments;
- subdivisions `4/8/16/32` with the same frozen `lambda=2.125`;
- finest velocity relative L2 `<=1%` against the finite-segment/ring reference;
- last two error reduction ratios `>=1.5` unless the finest error is already `<=1e-10`;
- no force or target metric is evaluated.

### G4 — Ptera TE extraction, after source audit

- extracted Ptera node/edge orientation reconstructs the native ring traversal and wake strengths exactly;
- same physical shared edge appears once in the global graph;
- current Ptera ring remains the physical owner and particles are marked `diagnostic_shadow`, so feedback call count is zero;
- particles-off/native state and load histories remain bitwise identical.
- next-step wake `RingVortex` objects, not preallocated flattened arrays, are the fact source after Ptera populates the next wake; arrays become current only after the next `_collapse_geometry()`.
- the current FluxV solver forces the Ptera TE wake to prescribed-wake motion; this shadow run records that fact and does not claim free-wake TE transport.

Any failed gate stops R2. Exclusive replacement is a separate later run and is not authorized here.

## 5. Execution and evidence

1. finish the Ptera source/time-layer audit;
2. implement pure topology/deposition helpers and analytic tests;
3. run deterministic field refinement without Ptera;
4. only if G0–G3 pass, add a minimal non-target Ptera TE shadow extraction;
5. write a fresh run directory with manifest, CSV refinement data, metrics, hashes, and claim validation.

No target GT may be read in prediction or scoring code. The strongest allowed result is “conservative TE shadow bridge mechanics pass.”

## 6. Ptera source-audit findings

- The shared wake grid supplies exact node identity and has `(rows+1)*(span_cells+1)` physical vertices.
- Ptera reconstructs an independent ring object per cell; object strengths collapse into current flat arrays at the next solve step.
- Edge strength is the full signed incidence result `gamma_edge = B.T @ Gamma_cell`; the `/2` convention used by some panel-load ledgers is attribution only and must not enter the source edge.
- Ptera's age/strength-dependent ring core means legacy ring sources with unequal kernel signatures are not algebraically mergeable. G0–G3 therefore use one frozen Gaussian-erf particle channel, while G4 must preserve per-kernel-channel provenance.
- Global edge topology removes duplicated span seams and exposes newborn/pseudovortex cancellation, but it does not by itself solve an `O(dt)` influence column or `q~1/dt` birth law.

## 7. Stop/go

- `GO` to a future exclusive-owner experiment only if G0–G4 all pass.
- `STOP` if conservation requires clipping, coordinate tolerance identity, duplicate ring+particle feedback, or source-dependent fitting.
- R3 manufactured LE transport and all paper scoring remain blocked.
