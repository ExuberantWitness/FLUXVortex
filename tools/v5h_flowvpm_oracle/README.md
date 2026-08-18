# FluxV v5h FLOWVPM oracle

This directory is an isolated numerical-oracle environment for FLOWVPM.jl commit
`4f433fb09f6baad25db65c9905e0d9cbb09663ce`.

The oracle is intentionally separate from the production Python implementation.
It exports deterministic Float64 direct-interaction snapshots used to verify the
Gaussian-erf velocity/Jacobian, reformulated VPM state update, low-storage RK3,
and corrected Pedrizzetti relaxation.

It is not a leading-edge separation, LESP, or aerodynamic-force model. Target
paper observations are prohibited from this environment.
