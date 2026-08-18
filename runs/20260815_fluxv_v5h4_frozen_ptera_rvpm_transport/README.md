# FluxV v5h4 frozen-Ptera-to-rVPM transport slice

Status: `go_frozen_parent_transport_mechanics_only`.

This non-target auxiliary run advances one live v5h2 dyadic cumulative rVPM
cloud through one shared LSRK3 step using the sum of its analytic Gaussian-erf
self field and a frozen, parent-only FluxV/Ptera field.  The Ptera field is its
native bound rings, current prescribed ring wake, and freestream.  Calling the
unbound `UVPMHybridSolver.calculate_solution_velocity` deliberately bypasses
the v5h3 feedback override, so the cloud is not induced on itself twice.

The Ptera spatial Jacobian is reconstructed by centered differences at the
preregistered relative perturbations `2^-8`, `2^-10`, and `2^-12`, scaled by
`min(sigma_min,c_ref)`.  Across that family, the complete transported
position, circulation-vector, and smoothing-radius states have adjacent
difference ratios `15.9794`, `15.7699`, and `15.9421`; the independent
12-point Ptera-Jacobian ratio is `15.2037`.  These are close to the factor 16
expected when epsilon is divided by four for a second-order centered
difference.  The formal-run test uses a conservative operational floor of 12;
the earlier preregistration required a common second-order limit but did not
assign a numeric ratio floor.

Headline evidence:

- disabled transport is bitwise equal to the existing rVPM self+freestream
  step and does not inspect the external field;
- an enabled zero external velocity/Jacobian is bitwise equal to the rVPM
  self-field step;
- the live nominal row advances 208 particles in exactly three LSRK3 stages,
  with three parent center calls and 18 finite-difference calls;
- independent storage/RHS replay reproduces every stage bitwise;
- the complete Ptera field-input/load/legacy-VPM state hash is identical
  before and after transport;
- two fresh three-epsilon executions have the same combined state SHA-256
  `dfeeb8967c44577ea1d25f6a7b68f93bb32f78e197e503cf39e1a9ae0035279f`;
- 14 focused and 319 related tests pass; Black and Ruff pass.

This is a one-step frozen-field mechanics result.  It does not yet synchronize
release, Ptera solve/load, and rVPM transport across a cumulative time march;
it reads no Yang, Izraelevitz, or Baik observation and makes no aerodynamic
accuracy claim.  Paper scoring remains blocked.
