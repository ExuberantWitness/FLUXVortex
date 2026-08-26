# Q16 Incremental Trial Geometry Results

## Outcome

`PASS / GO` for causal Q16 displacement-to-aerodynamic-branch coupling;
`PARTIAL / STOP` for claiming a completed structural FSI step.

One committed aerodynamic parent can now issue multiple trusted branches.  A
branch accepts exactly one next Q16 `q_trial`, interpolates its panel vertices
on CUDA float64, installs a detached continuous Panel owner, completes the
parent's pending wake birth using that geometry, and solves the real
separated-LEV/joint-TEV/free-wake state.

Same-parent/same-q branches produce identical geometry evidence, scientific
receipt, wake and load-packet SHA.  Different q trials produce different real
wake and loads while the parent stays unchanged.

## Metrics

| Metric | Observed | Gate |
|---|---:|---:|
| branch-A geometry SHA | `f897468b...e4b1b` | sealed |
| branch-B geometry SHA | `18f7bf48...e9c0` | different |
| wake max absolute delta | `0.02204999999999999` | `> 0` |
| total-force delta norm | `22.982585070259752` | `> 0` |
| load packet SHA equal | `false` | discriminative |
| scientific receipt equal | `false` | discriminative |
| LEV particles A/B | `12 / 12` | both nonzero |
| parent completed states | `1` | unchanged |
| parent wake convection | `0` | pending and unchanged |
| focused tests | `4 passed` | all pass |
| joint tests | `127 passed` | all pass |

The same-q control separately requires exact equality of geometry evidence,
receipt, wake grid and packet SHA and passes.

## Scope

- Supported: actual Q16 candidate position changes causal real aerodynamic wake
  and loads in an isolated predictor branch.
- Supported: branch geometry is detached; mutating the caller's CUDA q after
  binding cannot alter installed panels.
- Supported: reduced aerodynamic modes, host q, wrong topology, rebinding and
  exhausted lifecycle fail closed.
- Not supported: direct use of structural `dq_trial`.  Ptera currently derives
  boundary velocity from the interval displacement difference.  Equating that
  quantity to a Newmark endpoint velocity would be a new temporal assumption.
- Still blocked: nonzero active-LEV impulse has no Q16 local-work contract, so
  no complete generalized force or joint structural commit is published.
