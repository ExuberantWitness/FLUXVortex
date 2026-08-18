# v5h R0–R1 Main Experiment Plan

## 1. Objective

- canonical run id: `20260814_fluxv_v5h_r0_r1_oracle_direct_v2`
- superseded run id: `20260814_fluxv_v5h_r0_r1_oracle_direct` (schema-v1; retained only as historical evidence)
- tier: `auxiliary/dev`, but it is a hard prerequisite for every later v5h claim
- selected idea: freeze FLOWVPM.jl as an offline Float64 numerical oracle, then implement an isolated direct Python rVPM backend that reproduces its Gaussian-erf `U/J`, reformulated state RHS, low-storage RK3, and corrected Pedrizzetti relaxation. Do not modify the legacy FluxV particle path during this run.
- user's core requirement: continue toward a three-dimensional FluxV method after reading FLOWVPM code.
- non-negotiable constraints:
  - no Yang/Figure14/Baik ground-truth reads or performance scoring;
  - no Ptera coupling in R0–R1;
  - no hard clipping, NaN clearing, ridge/cap, FMM, SFS, viscosity, or target-data parameter fitting;
  - preserve the dirty shared worktree and touch only v5h-owned new files and artifacts.
- research question: can a clean direct Python implementation reproduce the pinned Julia FLOWVPM numerical equations closely enough to serve as the trustworthy transport layer for later 3D coupling?
- null hypothesis: the local implementation cannot meet the frozen Julia parity tolerances because the kernel, tensor convention, RK stage order, or relaxation semantics remain inconsistent.
- alternative hypothesis: the isolated implementation meets every parity tolerance without limiters.

## 2. Baseline And Comparability

- baseline id: `flowvpm_jl_4f433fb_direct_float64`
- baseline variant: reformulated VPM `f=0,g=1/5`, Gaussian-erf kernel, direct interactions, inviscid, no-SFS, corrected Pedrizzetti, upstream LSRK3
- dataset/split: deterministic analytic and synthetic particle snapshots only; no experimental dataset
- primary metrics:
  - `velocity_relative_l2`
  - `jacobian_relative_l2`
  - `stage_state_relative_l2_{x,gamma,sigma}`
  - `relaxation_gamma_norm_relative_error`
- required metric targets:
  - `velocity_relative_l2 <= 1e-12`
  - `jacobian_relative_l2 <= 1e-11`
  - every RK stage state relative error `<= 1e-11`
  - corrected relaxation parity `<= 1e-12`
  - nonfinite/clip count exactly zero
- comparability risks:
  - Julia package/API drift despite pinned source;
  - vector/matrix storage order and `J` transpose convention;
  - self-interaction exclusion and kernel normalization;
  - upstream tests may use FMM or different defaults unless every option is explicit;
  - local runtime lacks the skill-preferred `bash_exec`; all fallback commands must be copied into durable logs.
  - the FLOWVPM repository does not ship a `Manifest.toml`; registry `FastMultipole v2.0.4` is API-incompatible with the pinned FLOWVPM commit, so the oracle environment must also pin FastMultipole commit `adc4f26` (resolved as v2.2.0), matching the upstream port commit rather than silently accepting registry drift.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `tools/v5h_flowvpm_oracle/Project.toml` | isolated environment | pin FLOWVPM `4f433f...` and FastMultipole `adc4f26` | package/API isolation | upstream has no Manifest |
| `tools/v5h_flowvpm_oracle/export_oracle.jl` | absent | export deterministic snapshots, U/J, every RK stage, relaxation | numerical truth surface | private upstream API details |
| `src/fluxvortex/rvpm_reference.py` | absent | direct Gaussian-erf U/J and strict state validation | source-faithful oracle-compatible backend | kernel signs/storage |
| `src/fluxvortex/rvpm_transport.py` | absent | reformulated RHS, LSRK3, corrected relaxation | preserve upstream stage semantics | in-place update ordering |
| `tests/test_rvpm_reference.py` | absent | analytic/direct unit tests and fail-closed tests | catch local errors before Julia | self-transcription risk |
| `tests/test_rvpm_flowvpm_parity.py` | absent | compare frozen Julia artifacts to Python | independent parity gate | schema/version mismatch |

Legacy `src/fluxvortex/particles.py`, `solver.py`, v5f files, and all target-paper runners remain read-only in this run.

## 4. Execution Design

- minimal experiment:
  1. provision isolated Julia 1.10 and instantiate pinned FLOWVPM;
  2. run upstream small tests and export one 3-particle deterministic snapshot;
  3. implement Python U/J only and pass parity;
  4. add one reformulated RHS/RK stage and corrected relaxation parity;
  5. only then expand to multi-stage/multi-step smoke.
- smoke/pilot:
  - 3–8 particles with fixed literal states;
  - one evaluation point set including near-core and far-field points;
  - one LSRK stage and one relaxation step;
  - run in Float64/direct mode.
- full R1 run:
  - at least three deterministic snapshots and two time steps;
  - isolated ring and leapfrog upstream test path when practical;
  - fresh output directory, manifest, hashes, metrics, summary, and raw logs.
- expected outputs:
  - `runs/20260814_fluxv_v5h_r0_r1_oracle_direct_v2/{oracle,metrics}`;
  - `run_manifest.json`, `artifact_manifest.json`, `metrics.json`, `metrics.md`, `summary.md`, `runlog.summary.md`, `claim_validation.md`;
  - frozen Julia environment and Python source hashes.
- stop condition: any frozen parity metric exceeds threshold after one concrete implementation correction, or Julia upstream cannot be instantiated/reproduced.
- abandonment condition: parity requires hard clipping, hidden default changes, target constants, or a materially different formulation.
- strongest alternative hypothesis: use Julia FLOWVPM itself as an offline/production batch transport backend. It remains a fallback only if Python parity is infeasible; per-stage Python–Julia co-simulation is not the preferred architecture.

## 5. Runtime Strategy

- initial commands: environment checks, Julia download/checksum, `Pkg.instantiate`, upstream bounded tests, oracle export, Python pytest.
- expected budget: 2–4 CPU·h for R0 and 4–12 CPU·h for R1; <5 GB durable output.
- logs: managed `bash_exec` was unavailable. Exact commands, summarized terminal outcomes, hashes, and replay checks are preserved in the run manifest and Markdown summaries; no raw byte-addressed terminal log is claimed.
- safe efficiency levers: tiny deterministic snapshots, direct vectorized NumPy, no FMM/GPU, one bounded upstream test subset before the official small suite.
- monitoring: package setup or tests longer than 60 s are launched in a persistent terminal session and checked at 60/120/300 s; no blind retries.
- kill/relaunch triggers: no output for 10 minutes during package resolution, repeated identical dependency failure, nonfinite oracle state, or output schema mismatch.

## 6. Fallbacks And Recovery

- download failure: use the approved proxy and checksum the official Julia tarball; do not install system-wide.
- package failure: record exact resolver error; the single justified recovery already used is pinning FastMultipole `adc4f26` after registry v2.0.4 failed on missing `metadata_per_body`; any further dependency failure stops R0 rather than triggering open-ended resolver churn.
- API mismatch: adapt only the oracle wrapper, never edit upstream FLOWVPM source.
- Python parity failure: isolate kernel normalization → tensor ordering → stage timing in that order; change one layer per retry.
- resource pressure: reduce snapshot size, not precision or tolerances.
- non-comparable full run: retain the last-known-good pilot and mark R1 partial rather than relaxing gates.

## 7. Checklist Link

- checklist: `implementation_r0_r1/CHECKLIST.md`
- next action after closeout: start a separate B2 plan for the conservative TE surface-to-particle shadow bridge

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-14 19:00 CST | initial R0–R1 contract | user authorized continuation after plan review | target-paper scoring remains blocked |
| 2026-08-14 19:35 CST | pin FastMultipole commit `adc4f26` alongside FLOWVPM | registry v2.0.4 failed FLOWVPM precompile with `metadata_per_body` API mismatch; upstream FLOWVPM commit explicitly targets `adc4f26` | restores source-compatible R0 environment without changing numerical formulation; exact Project/Manifest hashes recorded in checklist |
| 2026-08-14 19:31 CST | close schema-v1 R0–R1 as provisional direct transport parity PASS | pinned upstream 14/14 and the initial cross-language gates passed | schema audit required before treating the artifact as canonical |
| 2026-08-14 20:10 CST | freeze schema-v2 as the canonical R0–R1 evidence | separated state snapshots from valid RHS U/J, added configuration gates, independent probes, near-field sweep, affine-in-time freestream fixture, and artifact regression; Python 29/29 | R2/B2 may start; Ptera, force, LEV birth, and target-paper scoring remain blocked |
