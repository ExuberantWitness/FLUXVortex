# Q16 Real FSI Long-Horizon Checklist

## Identity

- run id: `20260822_q16_real_fsi_long_horizon`
- idea id: Q16 mandatory-separated-flow long-horizon FSI
- stage: auxiliary/dev, evidence level `minimum -> solid`

## Planning

- [x] selected idea summarized in 1--2 sentences
- [x] baseline and comparability contract confirmed
- [x] code touchpoints listed
- [x] smoke plan written
- [x] full run plan written
- [x] fallback options written

## Implementation

- [x] CUDA endpoint work-balance audit
- [x] one-step result binds work evidence before publication
- [x] trajectory records bind state/work history
- [x] exact prefix resume without resetting the chain
- [x] unrelated changes avoided or justified

## Pilot / Smoke

- [x] four-step CUDA pilot executed
- [x] initial failed pilot metrics and residual histories interpretable
- [x] Newmark predictor-alias root cause isolated
- [x] corrected 30-iteration pilot classified as monotone slow convergence
- [x] fixed-relaxation 48-iteration retry classified as a near-gate cycle
- [x] bounded CUDA Aitken relaxation implemented with fixed-mode audit baseline
- [x] matrix-free preconditioned CUDA GMRES fallback implemented for indefinite tangents
- [x] Aitken/30 retry: step 2 passed in 11 iterations; step 3 remained above gate
- [x] corrected four-step pilot executed
- [x] four-step work closure and particle-capacity gates pass
- [x] four-step comparability still holds

## Main Run

- [x] eight-step segmented trajectory completed
- [x] resumed prefix remains exact
- [x] exhausted next coordinate leaves eight-step owner unchanged
- [x] monitoring and runtime deviations recorded

## Validation

- [x] focused affected tests pass
- [x] selected broad regressions pass
- [x] metrics and claims recorded durably
- [x] artifact manifest independently verified

## Closeout

- [x] result summarized with an explicit long-time/multi-cycle boundary
- [x] next action selected
