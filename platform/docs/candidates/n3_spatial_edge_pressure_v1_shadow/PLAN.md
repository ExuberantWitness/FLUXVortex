# N3 spatial edge-pressure v1 shadow preregistration

Date: 2026-07-29  
Claim node: `N3.1j5`  
Closure: `n3_spatial_edge_pressure_v1_shadow`  
Status: PRE-REGISTERED BEFORE IMPLEMENTATION

## 1. Research question

Can the spatial P2 LEV mechanism improve the Fig. 17/18/19 force fingerprint
when it replaces **only** V4.1's old N3 scalar dynamic-stall force, while the
complete V4.1 trajectory and N1/N2/N4/N6 force channels remain unchanged?

This experiment tests the confound identified in `n3_spatial_pressure_v0`.
That v0 simultaneously disabled leading-edge suction and legacy LEV channels,
changed the N2 force direction, coupled P2 back into the production bound/wake
trajectory, and replaced N3.  Its accuracy result therefore was not an
N3-only causal test.

## 2. Mechanism decision

Primary-source evidence fixes the following roles:

- Hirato et al. (2019), *Journal of Aircraft*,
  DOI `10.2514/1.C035124`, pp. 4 and 6, Eq. 13--22: LESP governs LEV
  release; LEV velocity and potential-jump history enter panel pressure,
  while leading-edge suction remains a separate tangential edge term.
- Ramesh et al. (2014), *JFM* 751,
  DOI `10.1017/jfm.2014.297`: `A0/LESP` is a shedding criterion, not a
  sustained positive LEV-force amplitude.
- Martínez-Carmena et al. (2022), AIAA 2022-2416, pp. 5--6:
  `A0` controls leading-edge shear-layer circulation supply; fixed LESP
  during LEV formation can overpredict lift and underpredict drag.
- Li et al. (2023), *JFM* 972 A30,
  DOI `10.1017/jfm.2023.569`, pp. 14--20: vortex loading depends on
  circulation, position/motion, and the induced bound reaction; a
  vortex-only impulse partition is not the complete LEV lift.
- Garrick (1936/1937), NACA TR 567: thin-airfoil axial force contains a
  leading-edge suction boundary term in addition to the projection of
  distributed normal loading.

Therefore this candidate has exactly one physical rewrite:

```text
old N3: A0 excess -> Tv scalar -> additive panel-normal force
new N3: LESP release -> spatial P2 state (Gamma, position, motion)
        -> with/without-P2 unified thin-panel pressure difference
```

The V4.1 leading-edge suction channel is retained as the thin-surface edge
term.  The P2 contribution is a distributed panel-normal pressure increment;
it may not add an impulse, vortex-normal, or fitted total-force term.

## 3. Claim boundary and exclusions

Allowed:

- one new open child `N3.1j5`;
- read-only access to the actual V4.1 AIC, right-hand side, geometry, wake,
  local velocity, and bound solution;
- a private P2 state and a private coupled bound counterfactual;
- one per-panel N3 pressure difference containing the P2-induced velocity,
  P2 potential-jump rate, and bound reaction;
- replacement of old N3 only after the V4.1 time step is complete.

Frozen and unchanged:

- N1 production gamma/wake trajectory, Bernoulli force, vortex impulse and
  leading-edge suction;
- N2 separation/profile-drag states and forces;
- N4 bookkeeping;
- N6 rig drag;
- grid, kinematics, constants, V4.1 closure, and public V4.1 results.

Explicitly excluded:

- actual-thickness pressure.  `N3.1j3b6c` is frozen falsified because the
  live N1 representation has 17 body-filament shell crossings;
- P2 self-advection, near-wall viscous inventory, and a new separation-flux
  model.  Their absence is a promotion blocker, not a tunable correction;
- changing `A0crit`, pressure clipping, decay, force rescaling, or any
  Fig-target-selected constant;
- reusing or unfreezing the falsified `N3.1j0` v0 closure.

This is a diagnostic shadow.  It cannot become production merely by improving
total force.

## 4. Frozen implementation identity

The production V4.1 step runs first and is not given any P2 input.  The P2
shadow then advances from read-only copies.  For every time step:

```text
F_candidate(t) =
    F_v41_counterfactual(t)
  - F_N3_v41(t)
  + F_N3_P2_shadow(t)
```

`F_N3_P2_shadow` is the exact per-panel difference between:

1. the same pressure operator using the private P2 coupled bound state,
   P2-induced velocity and `Gamma_bound + q_P2`; and
2. the same operator using the untouched V4.1 bound state and no P2.

Only after the full V4.1 loop, and before cycle reduction, may the old N3
history be replaced.  The original V4.1 total history is retained in the same
call as a matched-grid counterfactual.  No two-run mean-force splice is
allowed.

## 5. Hypothesis and strongest alternative

Alternative hypothesis:

> The large v0 thrust degradation was primarily caused by deleting the
> V4.1 edge-suction/N1 path and changing N2.  Once these channels are frozen,
> a spatial P2 N3 replacement reduces representative Fig. 17/18/19 error
> relative to the same-call V4.1 counterfactual.

Null hypothesis:

> Even under strict N3-only isolation, the present P2 state does not improve
> the matched representative error/trend fingerprint or fails time
> convergence.

Strongest competing mechanism:

> The remaining accuracy disease is N2.5/N2.6 chordwise separated-pressure
> closure rather than the N3 spatial state.

No competing mechanism will be implemented in this run.

## 6. Go/no-go sequence

### G0: implementation and identity

- closure profile equals V4.1 for every production physics flag;
- P2 cannot write N1 AIC/RHS/gamma/wake or force accumulators;
- N1/N2/N4/N6 histories and mean force channels remain unchanged;
- substitution identity residual `<=1e-12 N`;
- ForceLedger residual `<=1e-9 N`;
- attached/no-release limit gives bitwise-zero P2 N3.

Any failure is an implementation NO-GO.

### G1: numerical family

- P2 quadrature 16->24 changes sentinel total force by `<=0.5%`;
- `dt -> dt/2` changes the high-twist sentinel total L and T by `<=5%`;
- all states finite; no pressure clipping or target-data access.

Any failure blocks full-scope promotion.

### G2: representative Fig. 17/18/19 screen

Use the frozen 32-condition nested representative scope and the same-call
matched-grid V4.1 counterfactual.  Candidate GO requires all of:

- confirmed-scope point-weighted overall MAE strictly lower than the
  counterfactual;
- lift MAE and thrust MAE each no worse than `+5%`;
- total trend-capture count no lower;
- no new slope-sign reversal in the pre-registered Fig. 18 frequency or
  Fig. 17 twist witnesses.

If G2 fails, stop and mark `N3.1j5` falsified for this executable package.
Do not run confirmed151.

### G3: complete figures

Only after G0--G2 pass:

- run all 151 confirmed solver conditions for the 42 confirmed curves;
- generate Fig. 17, Fig. 18 and Fig. 19(a,b);
- keep Fig. 19(c,d) conditional and outside the primary score unless their
  frequency identity is independently resolved;
- compare candidate, same-call counterfactual, frozen production V4.1 and
  measurements with scope labels visible.

Promotion still remains NO-GO until P2 material transport and the
actual-thickness/viscous pressure route have independent evidence.

## 7. Result writeback

- G0 implementation failure: keep `N3.1j5=open`, record implementation
  failure.
- G1/G2 scientific failure: `N3.1j5=falsified, freeze=true`; preserve the
  parent `N3.1j=partial`.
- G3 accuracy improvement: `N3.1j5=partial`; it supports the N3-only spatial
  mechanism but does not validate production sufficiency.

