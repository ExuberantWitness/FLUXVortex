# Experiment plan: v5h10 global-row Baik W2 slice

## Claims

| Claim | Minimum evidence | Scope |
|---|---|---|
| C1. A complete spanwise release row can update one transported live boundary without duplicating shared edges. | 2-cell and 8-cell, three releases; direct full-graph U/J/impulse parity; exact rollback. | Mechanical only. |
| C2. The committed net row can enter native Ptera collocation/load evaluation and exactly one rVPM transport in the required time order. | W2 first-active three-layer raw run with call counters and parent-state hashes. | Development paper-case diagnostic. |

## Frozen W2 inputs

- case `W2`; smoke geometry `2 x 8`; 32 Ptera steps/cycle; two-cycle
  movement definition; `dt=T/32=0.11125 s` and convective `dt*=0.0981`;
- DVM source `Lcrit=0.11`; no Table-4.1 sensitivity, no tuning;
- source core `0.02c` is dimensionalized with the frozen W2 chord
  `c=0.076 m`, giving rVPM birth `sigma=0.00152 m`; the deposition target
  spacing is fixed before execution at `sigma/2.125 =
  0.0007152941176470589 m`;
- this Vatistas-source-core to Gaussian-rVPM-radius identification is a
  gate-defined bridge, not a paper input or a demonstrated kernel equivalence;
  core/spacing sensitivity remains mandatory before any promotion;
- each row circulation is the persisted cumulative LEV circulation through
  that source step; the artifact also records the newborn increment and the
  Kelvin ledger separately;
- first raw slice starts at the first active LESP step and stops after three
  release layers;
- DVM source indices are one-based while Ptera state indices are zero-based:
  frozen source steps `4,5,6` map to Ptera states `3,4,5` and phases
  `3/32,4/32,5/32`; this mapping is frozen before the first raw candidate;
- Ptera is the sole Kutta--Joukowski and unsteady-load owner;
- raw run does not open the corrected-total observation CSV.

### Pre-execution clock amendment (2026-08-16)

The initial interface draft paired source steps `4,5,6` with Ptera indices
`4,5,6`. A read-only protocol audit, before dependency hashes were bound and
before any candidate run, showed that Ptera index `i` is at `i*dt`, whereas
source step `n` consumes kinematics at phase `(n-1)/32`. The corrected pairing
above gives DVM/Ptera chord-vector discrepancies of `1.39e-17`, `1.39e-17`,
and `3.47e-18 m`; the rejected first pairing differed by
`6.600649e-3 m`. No target observation was read and the frozen DVM source
values were not changed.

## Required order

`DVM source -> global-row proposal -> atomic net commit -> Ptera collocation
and native loads -> one atomic macro rVPM transport made from the frozen number
of independent three-stage LSRK3 substeps, with the physical cloud and exact
live-material/frontier tracer pack sharing every stage-pre field -> transport
attestation`.

### No-GT integrator amendment after the first mechanical STOP (2026-08-16)

The original preregistration required finite, positive `sigma` and one
three-stage LSRK3 transport per release layer.  Before dependency hashes were
bound, before any candidate artifact, and without opening the observation CSV,
the canonical three-layer replay failed in layer 3 at LSRK3 stage 2 with two
negative radii.  This amendment freezes the only permitted retry:

The choice to study fixed dyadic `N=(32,64,128)` and to preselect N64 was made
after observing the no-GT inherited-third-layer substep diagnostic, including
the lack of 32-to-64 convergence.  It is therefore a diagnostic-informed,
post-STOP solver amendment, not an untouched original preregistration.  None of
the complete fresh 32/64/128 trajectories, paper observations, or candidate
scores may be used to alter the frozen range, candidate, reference, or gates.

- the pinned FLOWVPM reformulated rVPM RHS, direct Gaussian-erf kernel,
  transposed stretching convention, `f=0`, `g=0.2`, inviscid/no-SFS setting,
  no-relaxation setting, birth core, spacing, Ptera epsilon, and frozen-parent
  Ptera field remain unchanged;
- run three complete trajectories from the same initial state with fixed
  dyadic `N=(32,64,128)` independent LSRK3 substeps per release layer and
  `dt_sub=0.11125/N s`; every substep resets its low-storage RK storage;
- every trajectory uses the same N in all three release layers; a replay of
  only the inherited third-layer checkpoint is diagnostic: it informed the
  study range and preassigned candidate/reference roles, but cannot prove
  convergence or authorize a formal run;
- execution order is fixed as `N=32 -> 64 -> 128`; within each N, layers run
  `1 -> 2 -> 3`, and within a layer substeps/stages are strictly ascending;
- do not stop the study early when one level remains positive.  All three
  levels are required so that the `32->64` and `64->128` differences can be
  compared.  The formal candidate is fixed now at `N=64`; `N=128` is its
  verification reference, not a fallback candidate.  If the full `64->128`
  trajectory fails any gate, this experiment stops rather than switching to
  128 or trying another N;
- no runtime-adaptive N, clipping, sigma floor, state repair, core/spacing
  change, stretching removal, relaxation change, or observation-dependent
  selection is permitted.

The stability thresholds `dt_sub*||J||_F <= 1.5`, the dimensionless velocity
limit `<=0.5`, particle `|Gamma|*sigma^2` relative drift
`<=1e-6`, fine state relative-L2 `<=1e-6`, and refinement ratio `>=1.5`
are inherited from the earlier no-target v5h frontier mechanical gate.  Their
use in this W2 slice is a post-failure, pre-retry amendment, not part of the
original W2 preregistration.  The fixed-probe `U/J <=1e-4` and integrated-load
`<=0.2%` limits were previously frozen for the no-target direct/FMM parity
gate; applying them to adjacent temporal trajectories is a new, explicitly
declared use made before the complete-trajectory results are observed.

For each release layer and each `(substep, stage)`, require finite arrays,
strictly positive sigma, `dt_sub*max(||J_total||_F) <=1.5`, and
`dt_sub*max(|u_total-u_gal|/sigma) <=0.5`.  Here `u_gal` is frozen from the
analytically prescribed uniform Ptera freestream `_currentVInf_GP1__E`; Ptera's
parent-field evaluator explicitly does not include movement velocity, so no
rigid-body term is subtracted.  `u_gal` may not be fitted or estimated from
particle results.  This total-deformation observable is a new W2 extension of
the earlier self-field gate.  For every complete layer require maximum
particle `|Gamma|*sigma^2` relative drift `<=1e-6`.  Particle states are aligned
by stable particle ID and material/frontier tracers by their frozen ledger
order.  For each channel define
`d(a,b)=||q_a-q_b||_2` and
`e(a,b)=d(a,b)/max(1e-15,||q_b||_2)`. At every layer the `64->128`
relative-L2 differences of position, Gamma, sigma, and material/frontier tracer
position must each be `<=1e-6`.
At the three frozen no-target probes, both velocity and Jacobian relative-L2
differences must be `<=1e-4`.  These are the post-layer direct rVPM self-field
observables at GP1 coordinates `[(0.05,0.30,0.30),
(0.10,0.15,0.40),(0.20,0.45,0.50)] m`; Ptera's parent field is not silently
substituted for them.  The finite-difference Ptera Jacobian keeps the frozen
rule `epsilon=2^-10*min(layer_initial_sigma,c_ref)` at every layer and N; the
  rule is fixed, while the resulting dimensional epsilon is allowed to follow
  the converging layer-initial state. `layer_initial_sigma` means exactly
  `min(state.sigma)` at post-commit/pre-transport. The primary aerodynamic
  stability observables are the Ptera-owned three-layer stacked total-force
  vector and the separately stacked total-moment vector; their `64->128`
  relative-L2 differences must each be `<=0.002`. Force and moment are never
  concatenated into one dimensional norm. Raw CL/CD are reported for both
  levels but are not an alternative selector or an additional tuning channel.

For every state/tracer/probe channel, the ratio
`d(32,64)/d(64,128)` must be `>=1.5`; a channel is exempt only when both
absolute differences `d(32,64)` and `d(64,128)` are `<=1e-14`. Relative
differences may not be substituted into this ratio. All comparisons, raw
arrays, norms, scales, and per-stage maxima must be persisted before a decision
is emitted.  Each layer's ledger must contain exactly `N` independent LSRK3
calls, `3N` self-field and Ptera-center stage evaluations, and `18N` Ptera
offset evaluations.  Source, row commit, collocation feedback, four load
batches, and native load ownership remain once per layer.

Failure of the fixed `N=64` candidate against its `N=128` reference, failure of
any intermediate stage, failure of any
adjacent-resolution or ratio gate, reuse of a checkpoint produced with a
different earlier-layer N is a strict STOP. The inherited-checkpoint replay is
reported as exploratory diagnosis and is not required to equal a fresh full
trajectory whose earlier layers used another N. A formal failure routes to a
separately
preregistered spatial-overlap/remesh investigation, not to promoting `N=128`,
trying `N=256`, or changing the integrator in this experiment.

## Raw outputs

The authoritative convergence bundle contains exactly 12 files:
`raw_steps.csv`, `source_events.csv`, `owner_events.jsonl`,
`particle_counts.csv`, `raw_loads.csv`, `transport_stages.csv`,
`trajectory_arrays.json`, `convergence.json`, `summary.json`,
`run_manifest.json`, `SHA256SUMS`, and `run.log`.

For a completed PASS, `raw_steps.csv` has exactly nine `(N,layer)` rows.
`transport_stages.csv` is a normalized table with exactly
`3 layers * 3 stages * (32+64+128) = 2016` completed rows; stage numbers may not
be pivoted into three columns that overwrite repeated substeps. `raw_loads.csv`
has exactly `9*(16 panels+1 total)=153` rows.
`trajectory_arrays.json` preserves nine layer records, each containing both
the post-commit/pre-transport macro-start and layer-end particle states, stable
particle IDs, material/frontier tracer arrays, the frozen-probe self-field U/J,
per-particle `|Gamma|*sigma^2` start/end arrays, and total force/moment arrays
with exact dtype/shape/content hashes. Particles absent from the macro-start
state are invalid; zero-invariant rows require exactly zero end invariant and
use no relative division.
`convergence.json` is recomputed only after reopening those durable arrays; it
may not copy producer-reported pass booleans or norms.

All PASS collections have exact composite keys and counts:
`source_events.csv` has nine `(N,source_step)` rows;
`owner_events.jsonl`, `particle_counts.csv`, and the trajectory collection each
have nine `(N,layer)` rows/entries; `raw_loads.csv` is keyed by
`(N,layer,scope,panel_id)` with 16 panel rows plus one total row per layer-run;
and the stage table is keyed by `(N,layer,substep,stage)`. Duplicate, missing,
foreign, or out-of-order keys are a STOP even when aggregate counts match.

The bundle identifies `N=64` as its only formal candidate and `N=128` as the
verification reference. A convergence or runtime failure still publishes the
same 12-file checksummed bundle with `STOP` status, but its tables contain only
the exact completed prefix. `summary.json` and `run.log` identify the failed
`(N,layer,substep,stage)` coordinate and whether the failed stage began; if it
began, `transport_stages.csv` contains exactly one terminal `failed` row after
the completed prefix. Any arrays or hashes already produced by that failed
stage, including a nonpositive post-sigma, are retained; only fields never
produced may be null. A failure before transport has zero stage rows for that
layer, and zero for the whole run only if no earlier layer completed.
Empty/header-only tables and an empty trajectory collection are valid only for
an early STOP. The writer must
not invent rows to reach PASS counts, discard partial evidence, or leave it
beside a success summary. The subsequent paper
diagnostic may consume only the N64 rows of a frozen bundle whose convergence
gate passed; it does not rerun an unbound standalone N64 slice.

`SHA256SUMS` contains exactly 11 rows, one for every other file in the bundle,
and deliberately excludes itself. `run_manifest.json` binds code, inputs,
configuration, command, environment allowlist, and execution identity but does
not attempt a cyclic result hash. `summary.json` binds the semantic digests of
the raw tables, arrays, and convergence result; the external checksum file is
the final byte-level closure.

Every source/load row records phase/time, A0 pre/post,
birth mode, LEV/TEV strengths, Kelvin residual, owner digests, changed IDs,
particle counts, feedback/load counters, no-penetration residual, panel and
total forces, raw CL/CD, transport stages, and parent hashes.

### Compact stage evidence

The coupling may not retain all full `3N` stage arrays in memory. It calls the
unchanged v5h4 LSRK3 primitive once per substep (thereby resetting RK storage),
advances the material/frontier tracer pack with matching freshly reset storage,
and streams each stage through a read-only validator. Only the layer final
arrays and a compact chained stage ledger remain in the result.

Each normalized stage row binds `N`, layer, source/Ptera step, substep/stage
indices, `dt_sub`, RK coefficients, physical pre/post state hashes, self/Ptera/
total U/J and gamma/sigma-rate hashes, all RK storage pre/post hashes, tracer
pre/post/self/Ptera/total/storage hashes, Ptera parent hashes, sigma minima,
the two stability maxima and their stable particle IDs, exact field-call
counts, and previous/current evidence-chain hashes. Stage 1 storage for both
physical and tracer states must be exactly zero in every substep.

The row-owner transport attestation binds `transport_substep_count`,
`transport_stage_count`, and the same `common_stage_trace_sha256`; final support
equality alone is insufficient because an intermediate tracer may not be
altered and repaired before attestation. Coupling particle hashes and the
row-owner's domain-tagged transport digest remain separately named and may not
be compared as if they shared a digest domain.

Artifact-level independent recomputation is deliberately limited to layer-end
state/tracer/probe convergence, particle invariants, panel-to-total and stacked
total-load gates, counts, hashes, and all summary decisions using
`trajectory_arrays.json` and the raw tables. The compact stage ledger is a
runtime attestation tied to frozen source hashes, targeted stage/RK/storage
tests, and independent fresh A/B execution; the bundle does not contain enough
full Ptera parent and per-stage arrays to reconstruct every U/J and RK update
from bytes alone. No claim of artifact-only stage replay or hostile-process
proof is permitted.

## Hard STOP

- per-cell owners or duplicated shared-edge incidence;
- any Ptera/RHS evaluation before atomic net commit;
- support, sign, circulation, ID, time, inactive/restart mismatch;
- source/Ptera phase or zero-based/one-based clock mismatch;
- any live material support not transported in the same `3N` stage-pre fields
  as its physical particle cloud;
- any clone/counter/fresh-upstream particle in the physical cloud;
- nonfinite arrays, nonpositive sigma, particle-cap breach;
- any drift in the prescribed newborn birth sigma/spacing (transported sigma
  is required to evolve under the pinned rVPM RHS) or use of newborn-only
  rather than persisted cumulative row circulation;
- Kelvin residual > 1e-10 or no-penetration residual > 1e-12;
- feedback/load calls not exactly one collocation plus four load batches;
- any layer not using exactly one atomic macro transport containing the frozen
  N independent three-stage LSRK3 substeps, or any parent mutation;
- raw candidate reading GT, fitting parameters, or overwriting an artifact.

After raw freeze, the separately executed diagnostic score uses 400 unique W2
phase samples and the source-equivalent 1 Hz Fourier filter, with no phase,
amplitude, or offset fit. Frozen candidate gates are CL RMSE <=
1.1869728057492306, CD RMSE <= 0.7314313811433125, and strict Q1 improvement
over CL/CD 1.5656896456956615/0.5598942841503666. Failure is preserved.
