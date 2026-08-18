# v5h11 B3 execution amendment — 2026-08-16

Status: effective and immutable before any formal B3 candidate run.

This document is an execution-only amendment to the frozen v5h11 IR-WRK3
preregistration.  It does not change the method, W2 case, time step, source
threshold, transport levels, candidate selection, convergence thresholds, or
STOP rules.  It makes the evidence and dependency contracts executable after
the v5h10 third-layer additive-sigma failure exposed gaps in the first artifact
protocol.

## Observation boundary

- Observation access for this amendment and B3 remains `none`.
- Paper target arrays, the corrected-total CL/CD CSV, and the scorer remain
  unopened and unimported.
- Exploratory v5h10 failure diagnostics may motivate this execution branch but
  may not select a new threshold, transport level, probe, or W2 parameter.
- A disposable real layer-1 N32 smoke is allowed only after all code and
  dependency leaves below are frozen.  It is a protocol/mechanics preflight,
  is not part of A or B, and cannot select or tune a parameter.

## Fixed formal execution

- Formal transport levels are exactly `32, 64, 128`, in that order.
- Each level starts with a fresh process-level solver, DVM sources, row owner,
  row committer, and Ptera problem.  No particle, source, owner, solver, or
  checkpoint state crosses levels.
- Active mapping is exactly source steps `4, 5, 6` to zero-based Ptera steps
  `3, 4, 5`; the layer order is `1, 2, 3`.
- The W2 birth core is `0.00152 m`; target spacing is
  `0.0007152941176470589 m`.
- A and B are separate fresh processes.  Their first nine semantic artifact
  files must be byte-identical while UUID, UTC, output path, and replicate
  identity must differ.
- The formal command executes the audited runner file directly.  It must not
  use `python -m forward_flight_benchmarks...`, because package initialization
  would import Ptera and unrelated benchmark modules before dependency
  preflight.

## Stage evidence

Every completed stage persists and hashes the exact compact stream record plus
these three scalar gates:

1. `invariant_residual_over_slog_max`, computed on active particles as
   `max(residual / Slog)` with the same operation order as the stream gate; it
   must be at most `512 * eps(Float64)`;
2. `h_jacobian_frobenius`, at most `1.5`;
3. `h_convective_over_sigma`, using the frozen Galilean velocity, at most
   `0.5`.

The old absolute `invariant_residual_max <= 1e-10` artifact check is not an
equivalent substitute and is forbidden.  Raw operands that the compact stream
does not retain must not be reverse-engineered or invented.

A failed stage is different from a completed stage.  The STOP row binds its
six-dimensional coordinate, phase, failure type/message, and the completed
chain prefix.  Evidence not exposed by `IRWRK3StreamStopped` is serialized as
null/empty, never fabricated as a nominal `2/2/6` call ledger or a finite
stability value.  Previously completed layers and compact stages remain in the
12-file STOP bundle.

## Stable IDs, probes, and macro-start capture

- Physical particles, material-support tracers, and the nine ordered frontier
  nodes have separate stable-ID arrays.  ID tuples must match exactly across
  N before any ID-aligned norm is computed.
- Frontier IDs are exactly `frontier-node:0` through `frontier-node:8` in that
  order.
- The fixed GP1 self-field probes are exactly
  `(0.05, 0.30, 0.30) m`, `(0.10, 0.15, 0.40) m`, and
  `(0.20, 0.45, 0.50) m`.  Their provenance remains bound to the frozen v5h7
  and v5h8 probe-source files; no W2 output selected them.
- Because `V5H11LayerResult` contains endpoint arrays but not every macro-start
  array, the executor-owned row committer freezes start X/Gamma/sigma, all
  stable IDs, the eight-cell source-event/Kelvin ledger, changed/appended IDs,
  and commit/owner digests at the commit boundary.  A layer result is accepted
  only if its owner/state/source digests close against that private capture.

## Runner/executor ABI and dependency DAG

- The runner verifies an external acyclic audit token, then its dependency
  manifest, then every leaf byte hash before spec-loading the executor.
- The manifest and token are never leaves and no leaf may contain their digest.
  The runner, runner test, executor, executor test, this amendment, prereg
  freeze, numerical modules, W2/source modules, and declared installed-package
  metadata are leaves.
- The runner injects an exact `FormalExecutorAPI` containing sink/STOP class
  identity, schemas, levels, IDs, probes, and read-only verified leaf
  path/hash maps.  The executor must not import the runner.
- Before constructing a source or solver, the executor rehashes the injected
  leaves and verifies each loaded scientific module's resolved `__file__`
  against the corresponding leaf.  A good manifest file paired with a
  different runtime import is a STOP.
- The dependency claim is a bounded audited application-module set, not a
  cryptographic proof over arbitrary standard-library/native code or hostile
  mutation of private process memory.  Observed relevant repository,
  FluxVortex, and Ptera module origins are recorded; any application module
  outside the admitted inventory is a STOP.  Python, NumPy, SciPy,
  FluxVortex, and Ptera versions/metadata are recorded in the run manifest.
- Arbitrary coherent rewriting of trusted private closure memory is outside
  this in-process threat model.  Formal A and B process isolation is mandatory
  so one run cannot reuse or mutate another run's live registries.

## Artifact and STOP semantics

- The artifact remains exactly the preregistered 12-file bundle.
- A completed layer is committed atomically only after all `3N` completed
  stages, one raw/source/owner/count/trajectory row, and exactly 16 panel plus
  one total-load row close.
- A STOP without a failed-stage row binds the terminal coordinate to the exact
  next coordinate implied by the durable prefix.  A generic callback failure
  may not erase an already completed prefix or replace its coordinate by an
  all-null terminal.
- Publication is staging-directory plus `RENAME_NOREPLACE`; an existing or
  concurrently won destination is never overwritten.
- A synthetic protocol fixture uses a different schema/execution mode and can
  never be consumed as a formal PASS.
- Compact stage hashes permit independent replay of schema, ordering, call
  counts, hash chains, and persisted scalar gates.  They do not reconstruct
  discarded intermediate arrays; reports must state this evidence boundary.

## Advancement rule

The paper-data/scoring boundary remains sealed until all of the following are
true: disposable layer-1 smoke PASS; formal A and B each complete the exact
32/64/128 three-layer matrix; semantic bytes match; convergence/stability,
owner, source, Ptera-load, and artifact gates pass; and a fresh read-only audit
issues the next unlock token.  Any failure archives a STOP bundle and ends this
branch without changing N, thresholds, method, or W2 inputs.
