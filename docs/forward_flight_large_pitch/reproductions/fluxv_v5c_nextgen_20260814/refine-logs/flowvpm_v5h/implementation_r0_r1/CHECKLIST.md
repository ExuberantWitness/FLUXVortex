# v5h R0–R1 Main Experiment Checklist

## Identity

- canonical run id: `20260814_fluxv_v5h_r0_r1_oracle_direct_v2`
- superseded schema-v1 run: `20260814_fluxv_v5h_r0_r1_oracle_direct`
- idea id: `fluxv_v5h_flowvpm_transport`
- stage: `experiment / auxiliary-dev prerequisite`
- workspace: `/tmp/fluxv-v5-nextgen`
- branch: `agent/fluxv-v5c`
- baseline: FLOWVPM.jl `4f433fb09f6baad25db65c9905e0d9cbb09663ce`

## Planning

- [x] selected idea summarized
- [x] baseline and parity contract frozen
- [x] code touchpoints listed
- [x] smoke plan written
- [x] full R1 plan written
- [x] fallback and hard-stop rules written
- [x] target-paper data explicitly excluded
- [x] isolated Julia identity/checksum recorded: Julia 1.10.11 official Linux x86_64 tarball SHA256 `fb49c6b174600cd2051e37ba3f7330f8acf06dd00bce609bab6611387fdb37bf`

## Implementation

- [x] Julia Project/Manifest frozen: Project SHA256 `3dc90d083f3d19acaace3b0a0bc7b129c691c68b28eaa866ec48de456d54c175`; Manifest SHA256 `dd7a7baa49fea120c41e1485611f4b65aedb5cceeb30fdbf1831db8da1d6ab6f`
- [x] upstream oracle exporter implemented
- [x] Python direct U/J backend implemented
- [x] reformulated RHS/LSRK3 implemented
- [x] corrected Pedrizzetti implemented with old-Gamma norm semantics
- [x] stable HDF5 plus deterministic JSON mirror schema documented
- [x] schema-v2 separates state-only `pre/post` snapshots from valid RHS U/J and gates the complete numerical configuration
- [x] fail-fast finite/shape/positive-sigma guards implemented
- [x] legacy FluxV particle and target-paper files remain untouched

### Environment incident

- [x] registry FastMultipole v2.0.4 incompatibility preserved as evidence: FLOWVPM precompile failed on missing `metadata_per_body`
- [x] exactly one source-justified recovery applied: FastMultipole commit `adc4f26` (v2.2.0), matching the pinned FLOWVPM port target
- [x] FLOWVPM v4.0.4 at commit `4f433fb09f6baad25db65c9905e0d9cbb09663ce` and FastMultipole `adc4f26` both precompile in the isolated depot

## Pilot / Smoke

- [x] upstream bounded and official tests executed
- [x] five-source plus three-probe oracle artifact exported
- [x] Python analytic tests pass
- [x] U/J parity gates pass
- [x] two-step, six-stage state parity passes
- [x] relaxation parity passes
- [x] exact commands, summaries, source/result hashes and replay evidence are durable; raw terminal-log gap documented
- [x] canonical JSON and metrics replay byte-identically; HDF5 semantic replay satisfies `h5diff=0`

## Main R1 Run

- [x] fresh run directory created
- [x] full deterministic parity matrix launched
- [x] monitoring cadence used for the official Julia suite
- [x] no runtime route drift after the single source-justified FastMultipole pin
- [x] no clip/nonfinite event

## Validation

- [x] declared source/input/result hashes match
- [x] all required metrics finite and present
- [x] U relative L2 `<=1e-12`: `1.3149583542588946e-16`
- [x] J relative L2 `<=1e-11`: `8.135639459129776e-17`
- [x] all RK stage state errors `<=1e-11`: worst `2.455653689598918e-16`
- [x] relaxation parity `<=1e-12`: `0`
- [x] upstream ring/leapfrog evidence recorded: 10 single-ring plus 4 leapfrog testsets passed
- [x] claim classified: supported for direct transport parity only
- [x] Python schema-v2 suite passes 29/29; official pinned FLOWVPM suite passes 14/14 testsets

## Closeout

- [x] run manifest, metrics, summary, runlog summary and claim validation completed
- [x] reusable success/failure lesson recorded in project docs
- [x] exact next action is B2 TE bridge; target-paper scoring remains blocked
