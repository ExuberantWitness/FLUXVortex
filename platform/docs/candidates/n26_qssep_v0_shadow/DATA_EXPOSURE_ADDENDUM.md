# N2.6-QSSEP data exposure addendum

Date: 2026-08-01

## Measurement exposure

This candidate runner reads NO measured values during the solver loop
(`measurement_values_read_by_runner: False`).  The only measurement contact
is the post-run scorecard comparison against the frozen fixed-name V4.1
baseline (read-only).

## What the candidate reads

- `solver_channels` from `gpu_run_twist` (uvlm/separation/ds_vortex/...);
- per-strip `aeff_sep` (L-B A0 inversion, N2.1 validated) and `loss_frac`
  (N2.1 separated fraction) for the separation gate;
- free-stream `U` for q=0.5*rho*U^2.

## What the candidate never reads

- Fig. 17/18/19 measured forces (no retuning by construction);
- any fitted constant (C_D,sep=1.8 is the literature flat-plate value;
  DeLaurier 1993 / Pomerenk & Ristroph 2025).
