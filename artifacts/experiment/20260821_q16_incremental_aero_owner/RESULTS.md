# Q16 Incremental Aerodynamic Owner Results

## Outcome

`PASS / GO` for the execution-lifecycle refactor.  This is not yet a completed
FSI load transfer.

The real CUDA aerodynamic solver can now be advanced one state at a time while
retaining separated LEV particles, joint TEV circulation, free-wake history,
load state and evidence.  The incremental route reuses the existing numerical
methods and is bitwise equal to the monolithic six-state `run()` oracle.

The key causal result is that wake birth from solved state `n-1` is deferred
until candidate geometry `n` arrives.  A committed parent can therefore fork
multiple predictor/corrector candidates; each branch constructs and advances
its own real wake without changing the parent.

## Metrics

| Metric | Observed | Gate |
|---|---:|---:|
| real states | `6` | exact |
| receipts | `6` | one per state |
| force max absolute delta vs monolithic | `0.0` | bitwise |
| wake grids | bitwise equal | bitwise |
| LEV particle positions | bitwise equal | bitwise |
| final LEV particle count | `60` | equal and nonzero |
| wake convection count | `5` | `states - 1` |
| CUDA counters | exactly equal | exact tree |
| final load packet SHA | equal | exact |
| focused tests | `7 passed` | all pass |
| joint tests | `123 passed` | all pass |

## Scope

- Supported: exact incremental execution of the current mandatory real
  aerodynamic model.
- Supported: trusted branch creation from a completed state without parent
  mutation.
- Supported: deterministic scientific receipt across two branches with the
  same numerical state.
- Not yet supported: authorized replacement of the candidate branch's Panel
  geometry from Q16 at each incremental state.
- Not yet supported: a temporal contract connecting Newmark endpoint velocity
  to the movement velocity used by Ptera.
- Still blocked: active-LEV global impulse has no independently justified Q16
  flexible-work distribution.
