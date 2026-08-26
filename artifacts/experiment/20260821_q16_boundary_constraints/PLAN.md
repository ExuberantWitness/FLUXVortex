# Q16 boundary-constraint ownership plan

- run id: `20260821_q16_boundary_constraints`
- scope: immutable essential-boundary owner for shared-node Q16 states
- prerequisite: projected shared-node MITC16+ANS/EAS CPU/CUDA operator PASS

## Contract

1. One exact owner freezes sorted unique constrained global DOF indices and
   their prescribed float64 values before structural stepping.
2. A clamped root selects all six Q16 nodal coordinates on the minimum
   spanwise boundary; no element-local duplicate ownership is permitted.
3. State enforcement, admissible-direction projection, residual projection
   and reaction extraction obey the exact complementary decomposition
   `v = P_free(v) + R_constrained(v)`.
4. CPU and CUDA implement the same mask. Production CUDA accepts only
   device-resident float64 arrays and has no host numerical fallback.
5. Wrong dtype/shape, duplicate/out-of-range DOFs, nonfinite values and a
   mismatched mesh fail closed before a structural solve.

## Tests-first gates

- exact root-node and DOF ownership on a 2x1 Q16 mesh;
- prescribed state and projector/reaction complement identities;
- virtual work uses only admissible directions;
- projected MITC16+ANS/EAS force/Jv remain finite after root enforcement;
- CUDA state/projector/reaction parity and bitwise repeated projection;
- host, float32, wrong-shape and nonfinite attacks reject.

## Claim boundary

Passing this slice establishes boundary-condition ownership only. It does not
establish nonlinear equilibrium, time integration, structural stability, or
complete FSI.
