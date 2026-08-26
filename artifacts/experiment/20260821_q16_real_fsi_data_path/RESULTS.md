# Q16 Real FSI Data Path Results

## Outcome

`PARTIAL / GO` for the resolved aerodynamic-load path; `STOP` for claiming a
complete LEV-coupled structural load.

The real CUDA free-wake + joint-TEV + separated-LEV solver now retains the
already-computed four vortex-leg forces and unsteady panel force together with
their world-frame application points.  These 30 point loads in the bounded
pilot enter the exact CUDA transpose of the Q16 shared-node surface map.  The
one-step real pilot has zero impulse derivative and reaches Q16 generalized
coordinates.  The two-step real pilot exposes a non-zero unresolved impulse
norm of `138.13173693641363`; completed transfer correctly stops instead of
inventing a surface distribution.

## Scientific fixes

1. Corrected the rigid wrench transformation from the dimensionally invalid
   `R*M + t` to `R*M + t cross (R*F)`.
2. Preserved five actual CUDA point-force mechanisms per aerodynamic panel:
   four effective bound-vortex legs and the unsteady pressure force.
3. Added a sealed CUDA float64 packet whose force and moment are independently
   recomputed from its point rows before transfer.
4. Added Torch-to-Warp CUDA sharing and exact Q16 transpose transfer with a
   pre-transfer geometry gate and pre/post packet-drift gates.
5. Allowed coincident physical load rows only when they share one Q16 element
   owner; the existing cross-element duplicate registration remains rejected.

## Metrics

| Metric | Result | Gate |
|---|---:|---:|
| real packet point count | 30 | `5 * 6 panels` |
| resolved + impulse force max error | `2.842170943040401e-14` | scaled `4096 eps` |
| resolved point moment max error | `2.842170943040401e-14` | scaled `4096 eps` |
| synthetic virtual-work relative error | `1.3877787807814457e-17` | `< 1e-12` |
| two-step unresolved impulse norm | `138.13173693641363` | must be exactly zero for completed transfer |
| joint regression | `109 passed` | all pass |

## Verification

- new packet + real branch + active LEV focused: `17 passed`
- full Q16/transaction/real-LEV joint surface: `109 passed in 11.83s`
- Black check: PASS
- Ruff: PASS
- py_compile: PASS
- `git diff --check`: PASS
- hardware observed by Warp: NVIDIA GeForce RTX 4090 D
- no paper matrix, GT, scorer, or CPU numerical fallback was run

## Claim validation

- **Supported:** real spatially resolved aerodynamic point loads now reach Q16
  generalized coordinates with force, moment, geometry, dtype/device, mutation,
  and virtual-work gates.
- **Supported:** mandatory separated LEV, joint TEV and free-wake solver output is
  the source of the real pilot packet.
- **Blocked:** a complete multi-step Q16 FSI force cannot yet be claimed because
  the existing impulse model supplies only a net force, not a surface
  application-point or flexible-mode work contract.
- **Blocked:** structural trial geometry/velocity is not yet pushed back into a
  per-trial Ptera geometry/wake step.  The one-step pilot registers the returned
  real points to a Q16 surface after the aerodynamic step; this is not the final
  predictor/corrector kinematics path.

## Next action

Implement a preregistered impulse-work model before any full coupled commit.
The preferred scientific route is to retain per-particle and bound-sheet
impulse changes with an independently defined reference wrench and prove what
part of flexible virtual work is determined.  In parallel, add a trial-time
Q16-to-Ptera geometry/velocity adapter so every aerodynamic branch consumes the
same `q_trial/dq_trial` that receives its generalized force.  Only after both
gates pass should the Newmark predictor/corrector commit a real LEV/TEV wake
branch and Q16 state together.
