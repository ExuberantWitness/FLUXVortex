# Q16 Incremental Endpoint Motion Checklist

## Identity

- run id: `20260821_q16_incremental_endpoint_motion`
- idea id: Q16 predictor endpoint motion
- stage: implementation/pilot

## Planning

- [x] selected idea and non-claims frozen
- [x] geometry baseline and comparability confirmed
- [x] code touchpoints and fallback listed
- [x] bounded CUDA smoke and joint run specified

## Implementation

- [x] tests-first RED captured
- [x] sealed current vertex-velocity owner implemented
- [x] collocation, LE/TE, vortex-init and load-center shadows share one field
- [x] Q16 CUDA `(q_trial,dq_trial)` binding implemented
- [x] first state rejects nonzero structural velocity

## Pilot / Smoke

- [x] same `(q,dq)` exact determinism
- [x] same `q`, different `dq` changes real wake/load
- [x] parent state and geometry remain exact
- [x] caller mutation and lifecycle attacks fail closed

## Validation

- [x] focused tests pass: 4/4
- [x] Q16 joint regression passes: 131/131
- [x] Black/Ruff/py_compile/diff-check pass
- [x] artifact hashes verify

## Closeout

- [x] result and claim boundary recorded
- [x] next action (LEV impulse virtual work) explicit
