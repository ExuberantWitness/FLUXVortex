# v5h two-release frontier vertical gate — STOP

This is an observation-free, noncanonical mechanical artifact. It does not
read Yang, Izraelevitz/Scherer, or Baik measurements and contains no force or
load score.

The frozen run evaluated 36 configurations: three non-target geometries,
four fixed-sigma deposition levels, and three transport-substep levels.
Source/Kelvin, node-ribbon topology, fixed-sigma deposition, passive rVPM
transport, temporal-layer overlap, and time refinement all passed. The run
stopped because the preregistered frontier-position spatial family did not
converge when particle spacing and smoothing radius were reduced together.
Far-field probe velocities did converge over the original four levels.

Subsequent observation-free counterfactuals (not part of `summary.json`)
separated quadrature spacing from smoothing radius: fixed-radius quadrature
refinement converged, while shrinking the core at fixed quadrature reproduced
the growing on-sheet frontier difference. Therefore this artifact is retained
as an honest STOP for the combined refinement contract; it is not evidence of
general rVPM instability and cannot support target-paper scoring.

`summary.json` contains hashes of the directly executed source chain. Its
external SHA-256 is recorded in `SHA256SUMS`.
