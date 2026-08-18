# V5H12 execution-only repair plan

Status: amended and preregistered before implementation or execution.  The
2026-08-16T21:14:12+08:00 amendment makes V5H11 history immutable, moves all
implementation into a V5H12 namespace, and freezes source-parent durability.

## Run contract

- Run ID: `v5h12-execution-repair`.
- Research type: parser/STOP tests and disposable smoke are `auxiliary/dev`;
  fresh formal A/B are the inherited `main/test` inner-convergence gate.
- Research question: can the already-produced V5H11 scientific record be
  converted into the frozen artifact schema without duplicating ownership or
  losing the exact STOP coordinate?
- Null hypothesis: repairing the parser cannot close the formal protocol
  without changing scientific bytes or weakening a gate.
- Alternative hypothesis: a seven-field observer payload plus the record-owned
  normalized invariant, source-before-stage single commit, and exact conversion
  coordinate close the protocol while all scientific controls and inherited
  gates remain unchanged.
- Baseline: immutable V5H11 formal-A STOP at
  `/tmp/fluxv-v5h11-b3-formal-A-20260816`.
- Dataset/split: frozen Baik W2 execution inputs only; paper observation arrays
  are not a dataset available to this branch.
- Primary acceptance signal: cross-module ownership regression PASS followed by
  exact full-matrix formal A/B completion and semantic equality.
- Required metric keys: the inherited V5H11 normalized invariant, stage
  stability, state/tracer, probe U/J, force/moment, Kelvin, no-penetration,
  ownership, call-ledger, and artifact-closure gates.  No metric definition may
  change.
- Comparability risk: accidentally editing coupling/stream bytes, creating an
  eight-field replacement payload, or treating smoke as claim-carrying evidence.

## Selected change

V5H12 is a narrow execution-protocol successor to the stopped V5H11 formal-A
attempt.  It does not introduce a new aerodynamic or rVPM method.  Its only
permitted changes are to make ownership of already-existing compact-stage
evidence executable, append each source parent durably before any dependent
stage row, and preserve the exact next coordinate when conversion stops.

The motivating V5H11 formal-A result is a protocol STOP, not a numerical gate
failure: the coupling observer emitted seven FD/stability fields, while the
executor parser required an eighth normalized-invariant field that is already
owned by `IRWRK3StreamStageRecord`.  The attempted artifact contains zero
durable scientific rows and has semantic result SHA-256
`6ddc0714f9f3496501d7c9353af46073fe35ecd1ad25112dee87d4ecef43c420`.

## Non-negotiable boundary

- Paper observation access remains `none`; W2 ground truth and the scorer stay
  unopened and unimported.
- IR-WRK3 equations, coefficients, reconstruction, stream computation,
  coupling fields, Ptera adapter, row owner, source mechanics, W2 case, birth
  core, spacing, `N=32/64/128`, candidate/reference roles, probes, thresholds,
  artifact count, A/B rule, and advancement rule remain exactly V5H11.
- All V5H11 history is immutable.  This includes the V5H11 runner, executor,
  coupling, stream, and all four focused V5H11 tests; their frozen byte hashes
  are in `FREEZE_INPUTS.json`.
- No result-driven promotion, extra N, threshold relaxation, clipping, floor,
  projection, rebase, stretching change, per-substep relaxation, core/spacing
  change, or GT access is allowed.
- The V5H11 STOP artifact is immutable evidence.  It may be read and hashed but
  never overwritten or relabelled as a V5H12 result.

## Exact parser/ownership contract

The observer-owned payload schema remains
`v5h11-stage-fd-stability-v1` with exactly seven fields:

1. `fd_physical_evaluation_sha256`
2. `fd_tracer_evaluation_sha256`
3. `h_convective_over_sigma`
4. `h_jacobian_frobenius`
5. `source_state_sha256`
6. `stage`
7. `substep`

The executor must decode and validate those seven fields without adding,
guessing, or reverse-engineering an eighth payload field.  The normalized
invariant gate is independently owned by the stream compact record as
`IRWRK3StreamStageRecord.invariant_residual_over_slog_max`; the executor reads
that exact Float64 value from the record, verifies it is finite, non-negative,
and at most `512*eps(Float64)`, and writes the same value to the artifact row.
It must not mutate the observer payload or compute a substitute from discarded
arrays.  Record coordinates and `source_state_sha256` must still bind exactly
to the seven-field payload.

This is an ownership repair only.  The scientific bytes produced by the
coupling and stream are expected to remain byte-identical for identical inputs.

## Source-parent durability contract

Before appending the first transport-stage row for a layer, the V5H12 executor
must durably append that layer's exact source-parent event.  Every dependent
stage row therefore has an already-durable parent.  The source event is
appended exactly once: the later completed-layer transaction may verify the
existing source key/hash but must not append it again.  A duplicate append,
even with identical bytes, or a second append with an inconsistent key/hash is
a protocol STOP.  No source value, ordering, parent hash, or scientific byte is
changed by this repair.

If stage conversion stops after the source event is durable but before the
first stage row is appended, the durable prefix legitimately contains that
source event.  The terminal coordinate is nevertheless the exact unbegun next
stage, not an all-partial source-only coordinate.

## Exact STOP-coordinate contract

An exception during compact-record parsing or row conversion must not collapse
to an all-null `coupling_callback` coordinate.  The V5H12 executor owns the
active formal coordinate and must report the exact record being converted:

`(transport_substeps, layer, source_step_index, ptera_step_index, substep, stage)`.

If no stage row was durably appended, `stage_began=false` and the coordinate is
the exact next durable stage.  For the V5H11 failure surface this is
`(32, 1, 4, 3, 1, 1)`.  Previously durable source/stage/layer prefixes must be
retained.  A parser failure must serialize unavailable scientific evidence as
null/empty and must never invent a failed numerical stage, call ledger, or
stability value.

Only the V5H12 runner may add a conversion-phase validator for this stronger
unbegun-next-stage coordinate after a durable source parent.  Existing generic
source-only failure semantics remain unchanged; they must not be globally
reinterpreted as conversion failures.

## Allowed code-change map

Implementation is a mechanical fork into exactly four new paths:

| New V5H12 path | Mechanical source | Only admitted delta |
|---|---|---|
| `platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py` | frozen V5H11 executor | seven-plus-one evidence ownership; source-before-stage single commit; exact conversion STOP coordinate |
| `platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py` | frozen V5H11 runner | load V5H12 executor; validate conversion-phase unbegun-next-stage coordinate; preserve generic source-only semantics |
| `platform/tests/test_fluxv_v5h12_baik_w2_executor.py` | frozen V5H11 executor test | focused V5H12 positive/negative ownership and source-order tests |
| `platform/tests/test_run_fluxv_v5h12_baik_w2.py` | frozen V5H11 runner test | V5H12 namespace, STOP-coordinate, single-source-commit, and artifact tests |

The four corresponding V5H11 runner/executor/test files are immutable controls
and must not be modified.  Coupling and stream source/test files are likewise
immutable controls and may only be re-run.  Any other scientific/application
file change invalidates this branch and requires a separately named
preregistration.

## Validation order

1. Rehash every frozen input before editing; any unexplained drift is STOP.
2. Add a real cross-module regression using an actual coupling-generated
   compact record, not only an executor-local synthetic eight-field fixture.
3. Add negative tests for missing/extra observer fields, record-owned invariant
   non-finite/negative/over-gate values, payload/record coordinate drift, source
   after-stage ordering, duplicate/inconsistent source append, and exact STOP
   coordinates before/after durable prefixes.
4. Run the two new V5H12 suites plus the frozen V5H11 executor, runner,
   coupling, and stream suites.  Every V5H11 production/test hash must remain
   equal to the frozen input hash.
5. Create a fresh dependency manifest and external audit token that bind the
   repaired leaves; an old V5H11 token must fail closed.
6. Run one disposable real `N32/layer1` smoke with no durable formal artifact.
   It is mechanics/protocol evidence only and cannot select parameters.
7. Only after steps 1--6 pass, run fresh formal A and B as separate processes
   over the complete `32/64/128 x layers 1/2/3` matrix.  Existing destinations
   must never be reused or overwritten.
8. Require the first nine A/B semantic files to be byte-identical, provenance
   to differ, all numerical/artifact gates to pass, and a fresh read-only audit
   before any paper-data unlock.

## Runtime and monitoring

- Focused tests are expected to be minutes; disposable smoke is expected to be
  minutes; inherited formal A/B budget remains at most two CPU hours total.
- Exact commands, repaired-leaf hashes, fresh token, and fresh output paths must
  be recorded before execution.  No command may target the immutable V5H11
  formal-A directory.
- Formal runs use durable logs and new staging destinations with
  `RENAME_NOREPLACE`.  Monitor at approximately 60 s, 120 s, 300 s, 600 s, then
  1800 s intervals while a run remains healthy.
- Continue only while logs advance, dependency hashes stay closed, row counts
  form a valid prefix, values remain finite, and no gate STOP is reported.
- Kill/relaunch only for a documented environment or invocation invalidity;
  numerical/protocol gate failure is archived as STOP, not retried with a
  changed contract.
- Safe efficiency levers are limited to reusing read-only source/prehistory
  inputs and focused tests before Ptera.  Parallel levels, checkpoint reuse,
  reduced matrices, and altered arithmetic are forbidden because they change
  the execution/comparability contract.

## Recovery rules

- If the cross-module fixture fails, repair only the new V5H12 executor/runner
  ownership, source ordering, or exact coordinate propagation and rerun focused
  tests; do not launch smoke.
- If a new dependency token cannot be issued, record `dependencies_unbound`
  before constructing Ptera.
- If smoke fails a scientific gate, end V5H12; do not reinterpret it as a
  parser-only success.
- If formal A becomes non-comparable, publish its exact-prefix STOP and do not
  run B as though A had passed.
- Strongest alternative explanation to test first: the observed failure is not
  the field-count mismatch but a deeper payload/record provenance drift.  The
  real cross-module fixture must distinguish this before compute is spent.

## Success and abandonment

V5H12 succeeds as an execution repair only if the cross-module fixture proves
the seven-plus-one ownership contract, the exact STOP coordinate is durable,
the disposable smoke passes, and formal A/B complete under the unchanged V5H11
scientific contract.  Success does not itself imply paper accuracy.

Any scientific-file drift, changed physical byte stream, missing exact
coordinate, failed smoke, failed formal numerical gate, A/B semantic mismatch,
dependency drift, or GT read is a strict STOP.  Such a failure may not be
rescued inside V5H12 by changing the method, N, W2 inputs, or thresholds.

## Control surface and revision log

- Living checklist: `CHECKLIST.md`.
- Tracker: `EXPERIMENT_TRACKER.md`.
- Next unchecked item: rehash frozen leaves immediately before implementation.

| Time | Change | Reason | Comparability impact |
|---|---|---|---|
| 2026-08-16T21:11:40+08:00 | initial V5H12 execution-only preregistration | V5H11 formal A exposed a seven-versus-eight ownership mismatch and all-null STOP coordinate | none permitted; all scientific inputs remain frozen |
| 2026-08-16T21:14:12+08:00 | pre-implementation amendment: mechanical V5H12 namespace plus source-parent durability | prevent mutation of historical V5H11 and close the newly identified stage-before-source blocker | no scientific-byte change; change surface narrowed to four new files |

## Expected outputs

- this governance package;
- new V5H12 executor/runner leaves and two V5H12 focused regressions, with all
  V5H11 leaves unchanged;
- fresh dependency manifest/token;
- disposable smoke summary with `target_read_count=0`;
- fresh immutable 12-file formal A and B bundles, or an exact-prefix STOP bundle;
- fresh audit decision.  No paper score is produced in this branch before the
  inherited unlock gate is satisfied.
