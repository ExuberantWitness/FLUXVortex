# FluxV v5b shared-wake implementation contract

## Scope

v5b retains a UVLM bound-circulation solve and replaces v4b's additive LDVM
force discrepancy with a chronological TE/LE material wake.  LDVM/polar,
impulse, Polhamus and VNF total-force providers are disabled in this branch.
The only admissible force path is one UVLM surface-pressure ledger.

The first implementation reuses the equation-checked
`claim_runtime.hirato_live_shadow` state machine.  It is not allowed to claim
three-paper force accuracy until the pressure adapter and exact-reduction gates
pass.

## Time layer

At entry to step `n`, only material TE/LE history from earlier steps may enter
the fixed RHS.  The provisional bound solve defines the pre-constraint LESP.
Active strips are then solved in the augmented bound/newborn-LEV system.  A
nascent LEV is present once in the current solve/load and becomes immutable
history at step `n+1`.

The next TE row obeys Hirato Eq. 9:

```text
Gamma_TE[n+1] = Gamma_bound,rear[n] + Gamma_LEV[n]
```

Old material circulation is immutable; only vortex vertices convect.

## Force ownership

The pressure ledger contains:

1. surface advection of the current UVLM potential jump;
2. the bound-potential rate;
3. Hirato Eq. 17's active LE-sheet potential rate.

These channels are summed before multiplying panel area and normal once.
Source-specified profile drag, if a benchmark publishes one, is a separately
named viscous ledger and may be added once.  No LDVM force history is eligible.

## Promotion gates

- G0: existing Ramesh reference and Hirato equation/topology tests pass.
- G1: pristine/no-LEV execution is an exact UVLM reduction.
- G2: augmented no-through and LESP residuals close; Eq. 9/Kelvin closes.
- G3: material strengths remain bitwise immutable after birth.
- G4: pressure and force channels close to `1e-12`; there is one load owner.
- G5: halving `dt` gives bounded newborn velocity and `Gamma_birth = O(dt)`.
- G6: a one-strip/high-AR adapter approaches the 2-D Ramesh onset before any
  cross-paper performance run.

Failure of G1--G5 blocks all Yang/Figure-14/Baik accuracy claims.  A no-force
or partially coupled shadow may still be reported as an implementation result,
but it is not called FluxV v5b performance.

## Explicit exclusions

- no case ID, measured load or residual table enters shedding;
- no tuning of LESP threshold, core radius or time step to the three targets;
- no reuse of the failed v5a high-pass force residual;
- no P2 pressure/root selection or force path;
- no current newborn is counted both as an AIC unknown and fixed history.
