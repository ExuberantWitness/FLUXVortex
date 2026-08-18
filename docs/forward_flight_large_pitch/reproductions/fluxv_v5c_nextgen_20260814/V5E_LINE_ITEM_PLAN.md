# FluxV v5e line-item replacement plan

## Problem anchor

The v5c rate-sensitive suction proxy and both v5d owner variants failed the
frozen three-paper no-regression gate.  No threshold, pole, source clock, or
region boundary will be selected from the Yang, Scherer, or Baik force data.

The next candidate preserves the UVLM circulation solve, prescribed material
TE wake, induced velocities, and Kutta--Joukowski panel forces.  It changes one
explicit code-level line item only: Ptera's unsteady-Bernoulli force

`F_dGamma = -rho (Gamma_n - Gamma_previous) A n / dt`.

When the candidate is enabled this term is removed in full.  It is replaced by
the one-state lift--circulation mismatch from Izraelevitz et al. (2017) plus
their independent kinematic added-mass model.  An external paired LDVM term may
be added once as a separated-minus-attached residual; no complete ULLT, LDVM,
polar, impulse, or pressure load is added.

## Frozen strip definition

The ULLT state is driven by a lift-equivalent UVLM strip circulation defined
from the retained Kutta--Joukowski force, not selected from target accuracy.
The local lift direction is frozen as

`e_L = unit(V_perp_vector cross e_span)`,

and the scalar is

`Gamma_eq = (F_KJ_strip dot e_L) / (rho V_perp ds)`.

This is a declared lift-component equivalent, not a claim that the four-leg
ring-lattice force has one unique lifting-line circulation.  The retained KJ
force component orthogonal to `e_L` is not removed.  Near-zero `V_perp ds`
must fail closed or use an explicitly derived zero-load limit; it must never
be clipped to improve a benchmark.

The source time is

`Delta t_tilde = 2 V_perp dt / c`.

With `Cl_alpha=2 pi`, the source-frozen one-state closure is

`y_Gamma = 2 Gamma_eq / (c Cl_alpha)`,

`x_next = exp(-1.25 Delta t_tilde) x
          + 0.5 [1-exp(-1.25 Delta t_tilde)] y_Gamma`,

`y_phi = 2.5 y_Gamma - 3 x`.

The normal strip-force replacement is

`Delta F_phiGamma = 0.5 rho c Cl_alpha V_perp
                    (y_phi-y_Gamma) ds e_L`.

The added-mass landmarks `0.85` at AR=3 and `0.95` at AR=6 are published; the
linear interpolation with endpoint clamping is an explicit project transfer
rule, not a paper formula.  The added-mass adapter must provide a provenance
tag containing Eqs. (35)--(39), AR, factor, frame, derivative convention and
source hash.  Equation (42)'s Hoerner gain is not applied because the UVLM KJ
force already owns the finite-wing lifting-surface response.

## Staged execution

1. Export, without changing solver loads, every panel's total force, KJ force,
   unsteady-Bernoulli force, current/previous circulation, area, normal,
   collocation point, strip topology, and frame transform.
2. Prove panel and airplane closure and module-off exact replay.
3. Construct a shadow strip history and prove the analytic constant-input,
   periodic-state, coordinate-sign, and time-step refinement limits.
4. Run one representative condition from each paper.  Stop if any mechanical
   gate fails or if the replacement produces non-finite loads.
5. Only after those gates pass, freeze the implementation hashes and run all
   22 development conditions with predictions written before loading truth.

## Mechanical gates

- Disabled path is bitwise identical to the parent FluxV solver.
- Per panel and per airplane,
  `F_total - F_KJ - F_dGamma = 0` to `1e-12` relative scale.
- Strip sums reproduce panel and airplane ledgers to `1e-12`.
- Constant circulation with zero kinematic acceleration and equilibrium state
  gives `F_new = F_KJ`.
- The exact exponential state update matches its analytic constant-input
  solution and closes over repeated periodic cycles.
- Samples use a right-endpoint, backward-ZOH contract: `y_Gamma[n]` is held on
  `(t[n-1],t[n]]`, the state is advanced, and then `y_phi[n]` is evaluated.
- Halving `dt` at a fixed physical phase produces a convergent state and force.
- Exactly one force owner is used; the old `F_dGamma` is absent when the new
  closure is enabled.
- With no LEV, every paired LDVM component (`CNc`, `CNnc`, `CNnonl`, `CS`)
  vanishes individually before projection.

## Promotion gate

The candidate is promoted only if every frozen Yang lift/drag, Figure 14
all-14/subgroup/unique-12, and Baik W1--W4 filtered CL/CD metric is no worse
than the corresponding audited baseline.  Aggregate improvement cannot hide a
single failed paper or channel.  These are development data, so passing would
justify only a stronger multi-case candidate, not held-out generalization.

## Closeout

The mechanical implementation and three representative movements passed all
ledger gates.  The subsequent accuracy screen failed before all-22 execution:
Yang drag regressed, Figure-14 representative CT error increased by more than
an order of magnitude, and Baik W2 lift RMSE increased by about 49.5 percent.
The candidate is therefore archived without parameter adjustment.  See
`V5E_MECHANICAL_AND_EARLY_STOP_RESULT.md`.
