# Pinned FLOWVPM upstream test summary

- Julia `1.10.11`, one Julia thread and one BLAS thread.
- FLOWVPM `4.0.4`, commit `4f433fb09f6baad25db65c9905e0d9cbb09663ce`, tree `ecb0fc0b7f7cda244cef695ff06ce23719ad1920`.
- FastMultipole `2.2.0`, commit `adc4f264732de3dbbd492758e729af0b35db54b2`, tree `313cf60bed67629b1da6fb94b3b25394bd4f51ec`.
- `allow_reresolve=false`; commit/tree strings were asserted before `Pkg.test`.

Result: 14/14 testsets passed—10 single-vortex-ring and 4 thin-leapfrog variants. The v5h Python subset remains direct/inviscid/no-SFS; upstream FMM/SFS/viscous passes validate the pinned package environment but are not claimed as Python parity.
