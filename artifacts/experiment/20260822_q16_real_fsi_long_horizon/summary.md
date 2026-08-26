# Summary

The real CUDA Q16 + mandatory separated-LEV + joint-TEV + free-wake FSI path
now passes an eight-step, 4+4 resumed long-horizon development gate.  Exact
per-step evidence is in `metrics.json`; scientific scope and exclusions are in
`claim_validation.md`.

Next: replace the static-inflow horizon with a frequency-defined periodic FSI
case, calibrate structural damping from that case, and require at least three
resolved cycles plus cycle-to-cycle force/deformation convergence before using
the term “multi-cycle validation.”

