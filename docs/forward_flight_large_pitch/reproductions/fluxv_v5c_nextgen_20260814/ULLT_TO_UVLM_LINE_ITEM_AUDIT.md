# ULLT-to-UVLM line-item audit

## Decision

Total-force ULLT blending, adding ULLT to UVLM, replacing only the UVLM Kutta--
Joukowski term, and labelling Ptera's `dGamma/dt` term as pure added mass are
all rejected.

A mechanical shadow is conditionally authorized only for replacing the entire
Ptera unsteady-Bernoulli closure while retaining the UVLM circulation, wake and
Kutta--Joukowski force.

## Existing Ptera ledger

Ptera computes, panel by panel,

`F_total = F_KJ + F_dGamma`,

`F_dGamma = -rho (Gamma_n-Gamma_previous) A n / dt`.

`F_dGamma` is a code-level unsteady-pressure channel.  It is not independently
identified as pure kinematic added mass, so replacing only part of it would be
an unsupported decomposition.

## Minimal source-derived replacement

For each strip define

`y_Gamma = 2 Gamma_UVLM / (c Cl_alpha)`,

`d(t_tilde)=2 v_perp dt/c`, and advance the one-state closure

`dx/d(t_tilde) = -1.25 x + 0.625 y_Gamma`,

`y_phi = 2.5 y_Gamma - 3 x`.

For a constant step the exact update is

`x_next = exp(-1.25 d(t_tilde)) x + 0.5[1-exp(-1.25 d(t_tilde))] y_Gamma`.

The replacement normal-load discrepancy is

`Delta L'_phiGamma = 0.5 rho c Cl_alpha v_perp (y_phi-y_Gamma)`.

The candidate force is

`F_new = F_KJ,UVLM + Delta F_phiGamma + F_AM,kin + Delta F_LEV`.

When enabled, the original `F_dGamma` must be removed completely.  The
kinematic added-mass matrix follows Izraelevitz Eqs. (35)--(39), with the
existing aspect-ratio-dependent `K_am`; it cannot remain hard-coded to 0.85.

## Paired LDVM compatibility

`Delta CNnc = CNnc_separated-CNnc_attached` is a separation-induced
unsteady-pressure residual, not a second full added-mass model.  It may coexist
once while LDVM LEV circulation is not part of the UVLM circulation field.  If
material LEV circulation is later solved inside UVLM, the external `Delta
CNnc` must be removed.

Before any force evaluation, the no-LEV limit must show all four paired
components `CNc`, `CNnc`, `CNnonl`, and `CS` individually equal to zero.

## Required exporter and gates

The current airplane-level aggregate is insufficient.  A legal shadow needs
panel/strip `F_KJ`, `F_dGamma`, current/previous circulation, area, chord,
strip width, local axes, 0.75c velocity, acceleration and frame transform.
The strip circulation-collapse rule must be frozen before scoring.

Required exact limits include module-off bitwise replay; `F_total-F_KJ-
F_dGamma=0`; constant circulation and zero acceleration reducing to KJ; strip
sum replaying the airplane ledger; analytic constant-input state; time-step
refinement; and one unique pressure owner.

This route is a conditional mechanical GO only.  It is not expected to solve
the Figure-14 25-degree LEV loss by itself and cannot yet be called a new FluxV
generation.

