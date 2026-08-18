# FluxV v5h D0–D1 checklist

## Planning and ownership

- [x] FluxV/Ptera retained as bound-AIC and final-load owner
- [x] DVM restricted to LEV/TEV source generation
- [x] FLOWVPM restricted to post-birth particle transport
- [x] force-additive DVM path prohibited from this candidate
- [x] target-paper observations and scoring prohibited
- [x] GPL source remains external; clean-room behavior only

## D0 author parity

- [x] archive/source/reference hashes frozen
- [x] known Fortran bounds bug documented and isolated in audit harness
- [x] exact bundled motion replayed only in the external forensic harness
      (the in-repository analytic golden is not row-identical)
- [x] first Python/Fortran divergence located
- [x] 499 rows and first onset row 116 reproduced
- [x] final LEV/TEV counts equal 174/499
- [ ] row-wise source/Kelvin tolerances passed
- [x] no clip/ridge/NaN clearing in the frozen D0 replay

## D1 source interface

- [x] source-only API implemented and independently audited PASS
- [x] explicit section/Re/Lcrit provenance required
- [x] A0 pre/post and signed cap residual exported
- [x] new LEV and coupled TEV circulation exported
- [x] birth positions and restart/continuous state exported
- [x] Kelvin/deleted-circulation ledger exported
- [x] disabled/no-trigger exact reduction passed
- [x] serialized output contains no force/load fields
- [x] canonical flag remains false until D0 full parity

The D1 source audit closed its signed-LESP, full Kelvin/deletion, unique strip
lineage, source-role, camber-identity, unit/sign, derived-scale, and strict
configuration gates.  It remains noncanonical and is eligible only for the
next no-feedback mechanical handoff.

## Three-dimensional handoff

- [ ] stable strip/source IDs mapped to global shared edges
- [ ] circulation/vector-moment conservation passed
- [ ] intermittent active-boundary topology retained
- [ ] no coordinate rounding or force projection used

## Closeout

- [ ] fresh artifacts and manifests written
- [ ] independent current-code replay completed
- [ ] decision: manufactured 3-D source/transport GO or STOP
