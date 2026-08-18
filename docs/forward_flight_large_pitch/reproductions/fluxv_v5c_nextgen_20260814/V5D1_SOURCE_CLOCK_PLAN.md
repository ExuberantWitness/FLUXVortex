# FluxV v5d1 source-clock correction plan

## Purpose

The existing v4b causal-incidence owner advances the published Jones states
with `U_inf*dt/c`.  Izraelevitz et al. define the state time instead as

`d(t_tilde) = 2 |V_perp,local| dt / c_local`.

The published pole magnitudes `0.30` and `0.045` are therefore per unit
`t_tilde`, not per unit `U_inf*dt/c`.  v5d1 is a correctness-only replacement
of that clock.  It does not change the UVLM, polar, LDVM, profile-drag, LESP or
force-projection branches.

## Frozen implementation contract

1. Evaluate local kinematic incidence and speed at `0.75c` on every strip.
2. Remove the spanwise component before forming `V_perp,local`.
3. Advance the same two-stage signed/absolute owner with a positive
   time-and-strip field
   `delta_t_tilde = 2 |V_perp,local| dt / c_local`.
4. Retain source pole magnitudes `0.30` and `0.045` without fitting.
5. Use actual strip areas for the wing-integrated persistence fraction.
6. Warm up only by repeating past kinematics; no future sample may enter a
   current state.
7. Keep the frozen load equation
   `F=(1-p)(F_UVLM+Delta_F_LDVM)+p(F_UVLM+Delta_F_polar)`.
8. A disabled module is an exact identity; non-finite or non-positive local
   time increments fail closed.

The first run remains a kinematic/cache adapter: it omits same-time-layer UVLM
induced velocity.  It is therefore non-canonical even though its clock and
reference point are source-correct.

## Prediction and scoring order

The physical inputs, poles, warm-up rule and all 22 predictions are written
before loading experimental observations.  Scoring then reuses the frozen
Yang, Figure-14 and Baik metric code.  No phase, amplitude, offset, threshold
or case-specific correction is allowed.

## Frozen gates

- mechanical: finite state, bounds `[0,1]`, prefix causality, exact disabled
  reduction, constant-state analytic decay, area-weighted ledger closure;
- Yang: lift and drag MAE must each be no worse than the corrected reference;
- Figure 14: all-14, unique-12, 15-degree and 25-degree RMSE must each be no
  worse than v5c0;
- Baik: filtered macro CL/CD and every W1--W4 CL/CD RMSE must each be no worse
  than v4b;
- canonical promotion is forbidden because the adapter lacks the current UVLM
  induced strip velocity.

Any paper gate failure archives v5d1 as a source-correct negative result.  No
parameter revision is authorized from the target residuals.

