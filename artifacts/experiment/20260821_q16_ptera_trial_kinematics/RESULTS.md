# Q16-to-Ptera Trial Kinematics Results

## Outcome

`PARTIAL / GO` for the real trial-kinematics path; `STOP` for claiming a
completed multi-step Q16 FSI step.

The structural trial variables now enter the real aerodynamic solver.  CUDA
Q16 interpolation constructs a two-state Ptera branch as
`q_previous = q_trial - dt*dq_trial` and `q_current = q_trial`.  With identical
final `q_trial`, changing only `dq_trial` changes the previous panel geometry
and the sealed real load packet.  The bounded pilot keeps separated LEV, joint
TEV and free wake active.

This is not yet a reusable time-marching FSI owner.  The current Ptera API runs
an entire trajectory and has no incremental remesh/advance operation.  In
addition, the active-LEV two-state result contains a non-zero impulse force but
no declared flexible-structure application point.  The production transfer
therefore rejects the step, and the aerodynamic parent remains pristine.

## Metrics

| Metric | Observed | Gate |
|---|---:|---:|
| Q16/Ptera vertex count | `12` | exact `3 x 4` grid |
| inferred vertex-velocity max error | `1.1015494072452725e-15` | `<= 3e-14` |
| final vertex SHA, stationary vs moving | equal | equal `q_trial` |
| previous vertex SHA, stationary vs moving | different | different `dq_trial` |
| total-force delta norm | `3.228208771159716` | `> 0` |
| load-packet SHA | different | discriminative |
| separated-LEV particles | `24 / 24` | both `> 0` |
| joint-TEV strip values | `3 / 3` | both present |
| unresolved impulse norm | `4.73463070251006 / 6.378994656173283` | non-zero must STOP completed transfer |
| focused regression | `7 passed` | all pass |
| joint regression | `116 passed` | all pass |

## What Changed

1. Added an exact Q16 surface-map/Ptera panel-topology owner.
2. Interpolated Q16 positions and velocities on CUDA float64 and reconstructed
   previous/current world-frame vertices.
3. Converted vertices to each operating point's Ptera frame on CUDA.
4. Rebuilt complete immutable Panel objects and atomically replaced the panel
   array only on a pristine isolated solver branch.
5. Added topology, lifecycle, CUDA/dtype, finite/degenerate geometry and
   mandatory separated-LEV/joint-TEV/free-wake gates.
6. Added an end-to-end transaction negative gate proving that the real active
   LEV impulse cannot be silently converted into Q16 generalized work and that
   failure does not advance the live parent.

## Scope

- Supported: one real two-state predictor interval consumes the same Q16
  `q_trial/dq_trial` whose load response is observed.
- Supported: trial velocity changes real free-wake/LEV/TEV aerodynamic loads.
- Supported: unresolved LEV work fails closed transactionally.
- Not supported: incremental multi-step wake/LEV/TEV ownership.
- Not supported: completed active-LEV Q16 generalized force until an
  independently justified impulse/local-work model exists.
- Not run: paper matrix, GT, scorer, or CPU numerical fallback.
