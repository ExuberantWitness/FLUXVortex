# Unified FluxV UVLM-preserving upgrade checklist

## Identity

- run id: `20260812_unified_uvlm_polar`
- idea id: `finite-wing-full-angle-polar-residual-v1`
- stage: complete (exploratory periodic v3; post-Figure-14 repair)

## Planning

- [x] selected idea summarized in `1-2` sentences
- [x] baseline and comparability contract confirmed
- [x] code touchpoints listed
- [x] smoke plan written
- [x] full run plan written
- [x] fallback options written

## Implementation

- [x] intended files modified
- [x] unrelated changes avoided or justified
- [x] risky logic guarded or sanity-checked
- [x] plan updated if the implementation route changed

## Pilot / Smoke

- [x] smoke command executed
- [x] outputs look valid
- [x] metrics / logs are interpretable
- [x] comparability still holds

## Main Run

- [x] real run launched
- [x] monitoring cadence started
- [x] health signals confirmed
- [x] major runtime deviations reflected in `PLAN.md`

## Validation

- [x] outputs exist
- [x] metrics are complete
- [x] baseline delta is comparable
- [x] main claim is classified as supported / refuted / inconclusive
- [x] result recorded durably

## Closeout

- [x] main experiment summarized in `1-2` sentences
- [x] next action is explicit

Final classification: the frozen exploratory periodic v2 improves both
development benchmarks, but fails the subsequently added Izraelevitz Figure-
14 / Scherer experimental gate.  The post-hoc v3 persistent-owner repair
passes Yang, Figure 11, and Figure 14 non-degradation gates, but this is not an
independent generalization result and remains non-causal periodic processing.

Next action: freeze v3 without further Figure-14 adjustment and evaluate it on
a genuinely unseen forward-flight large-pitch dataset before any production
merge; a causal implementation and LEV/dynamic-stall validation remain open.

## Figure-14 load-ownership repair (`20260812_fig14_load_ownership_fix`)

- [x] failure mechanism and immutable baselines recorded in `PLAN.md`
- [x] no-fit, no-case-switch acceptance contract frozen
- [x] reusable persistent mean/AC owner implemented without case id
- [x] exact-reduction, phase/strip-ledger, and symmetric zero-mean tests added
- [x] bounded three-benchmark smoke passed
- [x] full Figure-14, Yang-2025, and Figure-11 matrices completed
- [x] metrics independently recomputed and PNG/PDF figures visually inspected
- [x] summary report, Figure-14 audit, and closeout checklist updated

v3 full gate record:

- Yang experiment MAE, lift/drag: `3.5791/3.8911 gf` versus old
  `6.8549/12.9217 gf`;
- Figure-11 numerical-reference RMSE, lift/drag: `0.15462/0.31420` versus old
  `2.43099/0.95313`;
- Figure-14 experimental mean-CT RMSE: `0.04719` versus old `0.05115` and
  pass-through ablation `0.05204`.

Mandatory qualification: v3 was introduced after the Figure-14 failure was
observed.  Its three-gate pass is `posthoc_exploratory`, not blind confirmation;
`p=0` on all Figure-14 conditions makes the one-state ULLT the complete
periodic owner, so no LEV-suction or instantaneous-load claim follows.
