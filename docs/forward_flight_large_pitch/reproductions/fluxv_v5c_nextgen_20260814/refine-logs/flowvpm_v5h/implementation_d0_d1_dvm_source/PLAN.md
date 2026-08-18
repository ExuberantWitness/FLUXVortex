# FluxV v5h D0–D1 Ramesh LDVM source plan

## 1. Objective

- run id: `20260814_fluxv_v5h_d0_d1_ldvm_source`
- tier: `auxiliary/mechanical prerequisite`
- objective: qualify the author-distributed two-dimensional LDVM v2.5 as an
  offline source oracle, then expose a clean-room FluxV interface that emits
  only LEV/TEV circulation source data.  Ptera/FluxV remains the finite-wing
  AIC and surface-load owner; FLOWVPM-derived rVPM remains the post-birth
  transport candidate.
- null hypothesis: the current Python primitive cannot reproduce the author's
  499-step source history and therefore is not eligible to seed a three-
  dimensional material wake.
- alternative hypothesis: a source-faithful Python path reproduces the frozen
  onset, circulation, birth, and Kelvin landmarks without copying GPL Fortran
  code or reading target-paper observations.

## 2. Architecture and ownership

```text
Ptera/FluxV UVLM: bound circulation, AIC, final surface load
                 ^                         |
                 | induced velocity        | local section motion/flow
                 |                         v
FLOWVPM rVPM transport <- global edge map <- Ramesh LDVM source law
```

- DVM owns only `when/where/how-much` LEV and coupled newest-TEV circulation.
- The global edge graph owns the unique three-dimensional source topology.
- rVPM owns post-birth position, vector strength, core, stretching, and
  relaxation.
- Ptera owns bound circulation and the final KJ/unsteady surface load.
- DVM `CL/CD/CN/CS`, impulse, polar residual, and any second force provider are
  prohibited from the production source interface.

## 3. Frozen oracle

- author archive SHA256:
  `fd574792bfcceed61d1d7d890edf95081ff13d6745e4f89c1cbc0261c5643d03`
- `ldvm.f95` SHA256:
  `eada8df2df605cc8bd929bbf9edf3672a85004d81c621a5637663d2a1b286c09`
- author force output SHA256:
  `8ce564d9358dcaf1c43fc433c969d2d3375eec9022eb4b7bf98ba759f0c620bb`
- reference case: SD7003, `Re=30000`, `Lcrit=0.18`, LE pivot,
  45-degree Eldredge ramp-hold-return, `K=0.2`, 500 motion rows and 499
  advanced/output rows.
- frozen landmarks:
  - first capped output row 116, `t*=1.638354`, `A0_post=0.18`;
  - first `Gamma_LEV=0.0052488387605094` and coupled newest
    `Gamma_TEV=-0.01357988564529845`;
  - 174 capped rows / 174 material LEVs and 499 TEVs;
  - per-step Kelvin residual `<1e-5` at printed-output precision.

The Fortran audit harness may correct only the known out-of-bounds
`aterm_prev(0:3)` initialization range.  The correction must leave the author
output unchanged to its printed precision.  The Fortran source and outputs
remain external because the distributed code is GPLv3-or-later.

## 4. D0 parity gates

1. The external full-history audit replays the exact archive member.  The
   in-repository GPL-free guard uses an independently reconstructed,
   source-arithmetic/source-format motion and gates event topology rather than
   claiming row-identical motion input.
2. Freeze the Fortran time layer: provisional newest TEV, pre-cap Fourier
   rates, joint LEV/TEV Kelvin+LESP solve, bound rebuild, Euler convection, and
   load/output.
3. Compare row-wise `A0`, bound circulation, new and accumulated LEV/TEV
   circulation, positions, active/restart state, and Kelvin residual.
4. Locate the first Python/Fortran state divergence before changing code.
5. Required canonical gates:
   - 499 finite advanced rows;
   - first onset agrees within one printed time sample and ultimately exactly;
   - final LEV count exactly 174, not the current partial-parity allowance 172;
   - `max|A0_post| <= Lcrit + 1e-10` while active;
   - Kelvin residual `<=1e-10` in the unrounded clean-room state;
   - no hard cap, clipping, ridge, NaN clearing, or target-data parameter.

Any unresolved 172/174 discrepancy marks D0 `PARTIAL_NO_GO` and keeps D1
noncanonical.

The frozen first-divergence investigation found three source-consistency
defects, none of which is an accuracy-fit parameter:

1. the Python primitive used the SD7003 camber slope in its boundary condition
   but omitted the camber ordinate from the bound-station world geometry;
2. it retained the first solved TEV whereas the author source explicitly
   clears that initialization vortex before wake roll-up;
3. first/restart LEV placement used only the old wake, whereas the author time
   layer includes the same-step provisional newest TEV in the local LE velocity.

The author shedding mask is Fortran `i_step=117..290`, corresponding to output
rows `116..289`, for exactly 174 consecutive LEVs.  Source-parity code must
also consume the per-step rounded motion `dt_i`; merely matching the final
count is insufficient.

## 5. D1 source-only API

Authorized new files:

- `platform/forward_flight_benchmarks/v5h_dvm_source.py`
- `platform/tests/test_v5h_dvm_source.py`

Each step must expose, with explicit units and source lineage:

- `A0_pre`, `A0_post`, signed `LESP_critical` and residual;
- active, first/restart, continuous-shedding, and inactive state;
- `Gamma_LEV_new`, coupled `Gamma_TEV_new`, and deleted-circulation ledger;
- LEV/TEV birth position and the exact birth-law provenance;
- total Kelvin residual and immutable material-source IDs;
- section family, Reynolds number, threshold source, time/core convention, and
  `canonical_eligible`.

Unknown section/Re threshold provenance fails closed.  Disabled and
never-triggered paths must not evaluate source-only inputs and must emit no
source.  Serialized source records must contain no lift, drag, normal force,
axial suction, moment, or force correction.

## 6. D1-to-3D bridge contract

- A strip DVM record is not independently projected to finite-wing force.
- Neighboring strip source circulation is assembled on explicit shared span
  nodes before particle deposition.
- The bridge must preserve signed circulation and vector moment and must not
  use coordinate rounding to identify topology.
- Spanwise-free boundaries and intermittent active-region boundaries remain
  explicit edges; they may not be smoothed away for stability.
- Source strength, particle spacing, physical core, release interval, and
  transport substep are independent inputs.
- A strip-centre DVM birth coordinate may not be interpolated to span-node
  endpoints.  The three-dimensional fact source must instead provide one
  node-owned LE/TE anchor, birth point, local velocity ledger, lineage, and
  first/continuous/restart state at every shared span node.
- The circulation conversion is explicit:
  `Gamma[m^2/s] = Gamma_star * U_reference[m/s] * chord[m]`; strip width is
  never multiplied into circulation.
- Each active source cell traverses
  `birth_left -> birth_right -> anchor_right -> anchor_left`, and shared-edge
  strength is the full signed incidence sum, never a half-average.

## 7. Stop/go and target isolation

- `GO` to a manufactured three-dimensional source/transport run only after D0
  full parity and D1 source-ledger gates pass.
- `STOP` if parity requires copying Fortran, using DVM loads, fitting Lcrit to
  Yang/Figure14/Baik, or clipping source circulation.
- Yang, Izraelevitz Figure 14, and Baik W1–W4 observations remain unread and
  unscored throughout D0–D1.
- Strongest allowed interim claim: “the isolated DVM source contract is
  mechanically qualified against the frozen author oracle.”

## 8. Evidence

- fresh run directory, source/result hashes, exact compiler flags, and external
  archive identity;
- row-level first-divergence table and source history;
- `run_manifest.json`, `artifact_manifest.json`, `metrics.json`, and claim
  validation;
- independent current-code replay before promotion.
