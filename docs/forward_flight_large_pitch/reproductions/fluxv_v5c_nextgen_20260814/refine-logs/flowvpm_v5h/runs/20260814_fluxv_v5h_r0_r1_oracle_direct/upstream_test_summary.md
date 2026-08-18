# Pinned FLOWVPM upstream test summary

Environment identity:

- Julia `1.10.11`, one Julia thread and one BLAS thread.
- FLOWVPM `4.0.4`, commit `4f433fb09f6baad25db65c9905e0d9cbb09663ce`, tree `ecb0fc0b7f7cda244cef695ff06ce23719ad1920`.
- FastMultipole `2.2.0`, commit `adc4f264732de3dbbd492758e729af0b35db54b2`, manifest rev `adc4f26`, tree `313cf60bed67629b1da6fb94b3b25394bd4f51ec`.
- `allow_reresolve=false`; all four commit/tree strings were asserted before the test call.

Result: 14/14 testsets passed.

- 10 single-vortex-ring testsets: Euler/direct, RK/direct, FMM, full inviscid, reformulation, viscous, constant-SFS Euler/RK, and dynamic-SFS Euler/RK.
- 4 thin-leapfrog testsets: classic VPM, reformulated VPM, dynamic SFS, and viscosity.

This suite validates the pinned upstream environment. The Python implementation in R1 intentionally covers only the direct, inviscid, no-SFS Gaussian-erf/rVPM subset; it does not claim FMM, SFS, or viscosity parity.
