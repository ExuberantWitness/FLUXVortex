# Q16 LEV Impulse Work Transfer Results

## Outcome

`PASS / GO` for the declared source-line midpoint work contract.

`PARTIAL / STOP` for a complete Q16 FSI step. The LEV impulse component now
reaches Q16 generalized coordinates, but it has not yet been composed with a
general real-Ptera resolved-load kinematic map and the joint structural/aero
transaction.

## Metrics

| Metric | Observed |
|---|---:|
| synthetic resultant max error | `8.881784197001252e-16` |
| synthetic midpoint-moment max error | `1.7763568394002505e-15` |
| synthetic virtual-work absolute error | `6.938893903907228e-18` |
| real source strips / particles | `3 / 24` |
| real impulse-force norm | `7.240457759620909` |
| real Q16 generalized-force norm | `4.512299102912722` |
| real strip-load SHA | `a2a4e0d9...b31f7` |
| focused / combined / joint tests | `4 / 9 / 150` passed |

## Implemented Contract

For strip force `F_s` and its actual Q16-generated leading-edge endpoints
`x_s^L, x_s^R`, the structural work is defined as

```text
delta W_s = F_s · (delta x_s^L + delta x_s^R) / 2.
```

The production CUDA operator assembles one half-force on each source endpoint
and applies the exact transpose of the same Q16 surface interpolation map. It
does not create an area distribution. The frozen load binds strip forces,
endpoint coordinates and every current particle's source-strip ID.

## Claim Boundary

- Supported: the explicit source-line midpoint model is algebraically
  work-conjugate and preserves its own resultant and moment.
- Supported: a real multi-step Q16-driven separated-LEV/joint-TEV/free-wake
  branch produces a nonzero LEV generalized force.
- Supported: changed force, source ID, geometry, device or dtype rejects.
- Not claimed: the source-line midpoint contract is the unique local traction
  recoverable from vortex impulse; it is an explicit reduced-order closure.
- Blocked: the real resolved vortex-leg/unsteady-pressure points still need a
  general algebraic Q16 kinematic owner before their generalized force can be
  added to this impulse component in the Newton transaction.
- No paper/GT/scorer data was accessed.

## Next Action

Implement a real Ptera load-point kinematic map as algebraic combinations of
the Q16 panel-vertex interpolation rows (including trailing-ring motion terms),
then compose resolved and impulse generalized forces inside one predictor
branch and commit aero + Q16 only after Newton convergence.
