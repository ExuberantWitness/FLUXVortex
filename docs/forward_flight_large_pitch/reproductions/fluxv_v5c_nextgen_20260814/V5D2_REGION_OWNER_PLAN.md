# FluxV v5d2 source-region owner plan

## Hypothesis

Yang et al. (2025) Eqs. (11)--(12) distinguish three local leading-edge
incidence regions:

- attached `A`: `|alpha_LE| <= alpha_sep`;
- coherent-LEV `L`: `alpha_sep < |alpha_LE| <= C_alpha alpha_sep`;
- collapsed/full-angle `P`: `|alpha_LE| > C_alpha alpha_sep`.

The published Yang coefficient `C_alpha=5` is frozen before scoring and is
transferred as an explicit hypothesis.  The three already-existing load
vertices remain

`A=F_UVLM`, `L=F_UVLM+Delta F_LDVM`, and
`P=F_UVLM+Delta F_polar`.

This experiment asks whether source-defined region ownership is a better
observation-free selector than the heuristic persistence state.  It does not
claim that the external LDVM or polar branch is Yang's native PLEV model.

## Frozen section inputs

- Yang: `alpha_sep=5 deg`, directly published for the reconstructed plate;
- Figure 14: `alpha_sep=CLmax/CLa=0.90/0.065=13.846... deg`, the already
  declared static-polar mapping hypothesis;
- Baik: `alpha_sep=asin(0.11)=6.315... deg`, the already declared Ramesh
  flat-plate cross-Re/thickness transfer hypothesis.

No threshold is selected from a target force curve.  The input is the same
kinematic local incidence at `0.75c` used by the v5d1 source-clock audit.

## Non-canonical adapter

The available caches contain integrated branch loads, not strip-resolved
branch loads or same-layer UVLM induced incidence.  The first runner therefore
area-averages strip region indicators into `wA,wL,wP`, then blends the three
integrated branch histories.  It is a diagnostic proxy and is never eligible
for canonical promotion.

## Gates

1. `wA,wL,wP` are finite, nonnegative and sum to one within `1e-12`.
2. Disabled mode returns `A` exactly; boundary assignments follow the equations
   exactly; future samples cannot alter a common prefix.
3. All 22 predictions are generated before observations are loaded.
4. Yang lift/drag, all four Figure-14 score groups, Baik macro CL/CD and every
   W1--W4 CL/CD row must be no worse than the corrected v4b/v5c0 reference.
5. Any failure archives the result without tuning `C_alpha`, `alpha_sep`, phase,
   amplitude, offset, or branch coefficients.

Even a full numerical pass remains development evidence.  A production model
would still require strip-resolved branch forces, current induced incidence,
and either native material LEV coupling or a legal ULLT-to-UVLM line-item
replacement.

