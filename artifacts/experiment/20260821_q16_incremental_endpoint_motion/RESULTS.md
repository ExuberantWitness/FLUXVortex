# Q16 Incremental Endpoint Motion Results

## Outcome

`PASS / GO` for the Q16 Newmark endpoint-motion data path.

`PARTIAL / STOP` for a complete Q16 FSI commit because the active-LEV global
impulse still has no declared Q16 work-conjugate application contract.

The same committed aerodynamic parent was forked into real separated-flow
branches with identical `q_trial` and different `dq_trial`.  Both branches had
identical installed current geometry, while their sealed endpoint-velocity
fields, LE/TE relative velocities, wake, scientific receipts and aerodynamic
load packets differed.  The parent remained byte-identical.

## Main Metrics

| Metric | Observed | Gate |
|---|---:|---:|
| current geometry SHA equal | `true` | required |
| endpoint velocity SHA equal | `false` | discriminative |
| total-force delta norm | `3.7709407499921745` | `> 0` |
| wake max absolute delta | `0.001575` | `> 0` |
| packet SHA equal | `false` | discriminative |
| receipt SHA equal | `false` | discriminative |
| motion shadow error, slow | `1.1370570283941728e-15` | `<=128 eps` |
| motion shadow error, fast | `1.3036446921965705e-15` | `<=128 eps` |
| LEV particles, slow/fast | `12 / 12` | both nonzero |
| parent unchanged | `true` | required |
| focused tests | `4 passed` | all pass |
| joint tests | `131 passed` | all pass |

## Mechanism

Q16 interpolation produces both current world vertices and endpoint world
velocities on CUDA float64.  Both are transformed into the current Ptera
geometry frame.  The incremental session owns a detached velocity grid and
uses it to create `x_shadow = x_current - dt*v_endpoint` only for Ptera's
current-step finite-difference operands:

- collocation no-penetration velocity;
- leading/trailing-edge relative velocity used by joint LEV/TEV formation;
- trailing bound-ring initialization;
- all four bound-vortex line-center movement velocities used in loads.

The actual previous Panel geometry, circulation history, LEV/TEV state and
wake are not rewritten.  The first state rejects nonzero structural velocity
because no preceding aerodynamic state exists.

## Claim Boundary

- Supported: `dq_trial` now changes real flow and load state rather than only
  evidence metadata.
- Supported: the endpoint velocity comes from the same Q16 transfer map as the
  endpoint position and is detached, sealed and lifecycle-bound.
- Supported: separated LEV, joint TEV and free wake remain mandatory.
- Not claimed: a substep Hermite/Newmark time quadrature inside one Ptera step;
  this is a current-endpoint boundary-velocity contract.
- Blocked: the nonzero global LEV impulse lacks a local point/moment
  decomposition, so complete virtual work and a structural commit are not yet
  available.
