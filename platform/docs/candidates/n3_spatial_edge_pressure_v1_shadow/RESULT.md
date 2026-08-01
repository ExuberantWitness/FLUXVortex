# N3.1j5 result — terminal G1 NO-GO

Date: 2026-07-29  
Claim node: `N3.1j5`  
Executable package: `n3_spatial_edge_pressure_v1_shadow`  
Decision: `FALSIFIED / FROZEN` for this executable package

## Scope of the decision

This result closes the single preregistered N3-only experiment in `PLAN.md`.
It does **not** falsify the parent claim that an LEV load model needs spatial
circulation, position/motion and bound-reaction state.  It does falsify this
specific q16 thin-panel P2 shadow as an eligible Fig. 17/18/19 promotion
candidate, because its first numerical-family gate failed.

No measurement values were read during either solver run.  This is therefore
a numerical-eligibility and mechanism-isolation result, not a claim that the
candidate improves Fig. 17/18/19 accuracy.

## Frozen formal evidence

Both runs used the same candidate ID, closure, quick grid, local source
closure and numerical environment.  They differed only in the preregistered
P2 quadrature coordinate.

- q16: `runs/20260729_212029/`
- q24: `runs/20260729_212146/`
- terminal score:
  `runs/20260729_212029/G1_quadrature_score.json`
- score SHA-256:
  `33b0cfd6d97ac75ba097b24079e2b099c0eb8a3f91675986e0e6e6f53e931a9e`

The run identity records Python 3.12.13, NumPy 2.4.6, Warp 1.14.0,
float64, CUDA device UUID
`GPU-3023ccab-f90d-7404-2ad0-c8b0cf28babf`, every loaded local source,
the claim YAML, and the preregistration/control documents.

## G0 implementation result

G0 passed:

- V4.1 closure-profile identity: true;
- all six required claim guards passed in all three sentinel conditions;
- maximum force-ledger residual:
  `8.881784197001252e-16 N`;
- maximum unclassified-force residual:
  `8.881784197001252e-16 N`;
- maximum old-N3 substitution residual: `0 N`;
- maximum pressure/force decomposition residual:
  `4.440892098500626e-16 N`;
- attached/no-release P2 N3: bitwise zero;
- `N3.1j5` appeared as one `internal_stage` owned by `N3` and did not
  enter the top-level topology or book a second force.

The implementation is therefore sufficiently isolated to interpret G1 as a
numerical result rather than a duplicate-force or graph-wiring failure.

## G1 result

The preregistered pass band was
`max(|F_q16-F_q24|/|F_q24|) <= 0.5%` over reported lift/thrust components.

| Sentinel | Channel | q16 (N) | q24 (N) | Relative change |
|---|---:|---:|---:|---:|
| twist 0 deg | L | 6.689644 | 6.689644 | 0.000000% |
| twist 0 deg | T | -1.157742 | -1.157742 | 0.000000% |
| twist 22.5 deg | L | 6.021980 | 6.021830 | 0.002485% |
| twist 22.5 deg | T | -2.202329 | -2.202304 | 0.001124% |
| twist 45 deg | L | 5.762290 | 5.799418 | **0.640208%** |
| twist 45 deg | T | -3.185710 | -3.202994 | **0.539617%** |

At the high-twist witness, the wind-axis force-vector norm changed by
`0.618164%`.  Thus both the frozen component-wise definition and the
post-exposure vector-norm diagnostic exceed `0.5%`.

Decision: `TERMINAL_NO_GO`.

Per the serial preregistration, the half-time-step test, representative32,
confirmed151, and complete Fig. 17/18/19 overlays were not run.

## Mechanism fingerprint retained for the parent claim

The q16 same-call counterfactual shows that replacing only old N3 changes
lift increasingly with twist but barely changes thrust:

| Nominal twist | Candidate - V4.1 lift (N) | Candidate - V4.1 thrust (N) |
|---:|---:|---:|
| 0 deg | -0.019988 | +0.001255 |
| 22.5 deg | -0.562403 | +0.060140 |
| 45 deg | -1.210515 | +0.077456 |

This is a useful causal result despite the NO-GO:

1. removing the v0 non-N3 confounds preserves the expected spatial-N3
   lift-rolloff signal;
2. the present P2 pressure state is not yet quadrature-eligible at the
   high-twist boundary;
3. the dominant thrust disease is not repaired by an N3-only normal-pressure
   rewrite and remains associated with the N2/chordwise separated-pressure
   closure or an eventual actual-thickness unified-pressure route.

That third item is a diagnosis for the next research cycle, not authorization
to patch N2 in this candidate.

## Claim writeback

- `N3.1j5 -> falsified, freeze=true`;
- do not rerun this executable package with q32, a relaxed threshold,
  changed `A0crit`, clipping, decay or force scaling;
- retain `N3.1j -> partial`: the spatial-vortex mechanism is not falsified;
- any successor must have a new claim ID, new primary-source mechanism
  decision and a new preregistered numerical family.
