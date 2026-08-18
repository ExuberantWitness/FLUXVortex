# FluxV v5d source and load-owner audit

## Decision

Two tempting shortcuts are rejected before implementation:

1. a second fitted axial-suction state for subcritical fast rotation; and
2. target-driven reweighting of the existing UVLM, polar and paired-LDVM total
   force branches.

The authorized next step is only a parameter-free `v5d0` shadow ledger that
separates the three existing owners exactly. It is infrastructure, not a new
accuracy result. A performance candidate requires either an explicit ULLT to
UVLM line-item replacement or a material SVDVM/LEV path that first passes
source-parity and geometry-closure gates.

## Why the subcritical axial-state shortcut is rejected

Ramesh defines

`Gamma_B = pi U c (A0 + A1/2)` and `CS = 2 pi A0^2`,

with a section- and Reynolds-dependent `Lcrit`. Martínez-Carmena supplies the
material leading-edge circulation rate

`Gamma_dot_LE = U^2 A0^2 / rLE_bar`,

but the 2022 model still relies on a separately determined onset. The 2023
thesis derivative condition is primarily an offline onset diagnostic; it is
not a validated, parameter-free causal shedding gate, and its printed
pressure-side inequality is not symmetric after rewriting it in `|A0|`.

Consequently there is no source-closed value for a new rotation threshold,
recovery pole or filter window. Izraelevitz's `b=-0.25` pole belongs to an
attached Wagner/ULLT state, not a published LEV suction-recovery law. Reusing it
again would create a new exploratory closure rather than a literature
implementation.

The material SVDVM rate remains scientifically useful, but cross-paper scoring
is blocked because Yang and the NACA 63A015 Figure-14 wing lack an independently
frozen equivalent leading-edge radius under the selected two-dimensional
section mapping. Baik's rounded plate is geometrically closed and can later be
used after non-target source-parity.

## What the current branches actually own

Let

- `A` be attached UVLM;
- `L=A+Delta_LDVM` be UVLM plus paired separated-minus-attached LDVM;
- `P=A+Delta_polar` be UVLM plus the full-angle polar residual.

The current v4b ledger is

`F_v4b = (1-p)L + pP`.

It is a two-branch blend; attached flow is implicit in both residual branches.
The proposed shadow rewrites it without changing any value:

`wP=p`, `wL=(1-p)m`, `wA=(1-p)(1-m)`,

`F=A + wP Delta_polar + wL Delta_LDVM`,

where `m` is only a material/paired-discrepancy witness. The weights are
non-negative, sum to one and cannot read an observation, paper name or case ID.

## Capacity audit: why weights alone are insufficient

The following are diagnostic upper bounds, not models. Any phase-wise convex
hull uses observations and is therefore noncausal target leakage.

| Benchmark | Frozen v4b/v5c0 | Best existing branch or oracle |
|---|---:|---:|
| Yang lift/drag MAE, gf | 4.5545 / 2.6440 | polar-only 3.9514 / 2.0626 |
| Figure-14 all-marker CT RMSE | 0.024251 | LDVM-only 0.023877 |
| Figure-14 25-degree CT RMSE | gate 0.02284 | UVLM A/L/P hull lower bound 0.023866 |
| Baik filtered macro CL/CD RMSE | 0.65754 / 0.34515 | LDVM-only 0.58951 / 0.25977 |

Even an observation-leaking A/L/P convex hull cannot pass the frozen
Figure-14 25-degree gate. A local ULLT attached diagnostic lowers that bound,
but choosing its total force would replace rather than preserve the UVLM
backbone. Therefore a legal candidate must identify the exact UVLM line item
that the ULLT state replaces; adding or blending two total-force solvers is
forbidden.

## v5d0 shadow gates

- `wA,wL,wP >= 0`, `wA+wL+wP=1` within `1e-12`;
- exactly one attached baseline plus two residuals;
- module-off, no-separation and attached limits reduce exactly;
- v4b/v5c0 phase replay within `1e-12`;
- prefix causality within `1e-12`;
- no observation, paper or case identity in the owner;
- no ULLT, v5c suction, LDVM full force or profile-drag double addition;
- `shadow_only`, `canonical_eligible=false`, and no paper accuracy claim.

## Performance-candidate prerequisites

1. Decompose Ptera/UVLM and ULLT into compatible circulatory, potential-rate
   and added-mass line items.
2. Define one replacement owner in a shared coordinate/sign/time convention.
3. Export same-time-layer strip velocity, `A0`, bound circulation and axial
   suction.
4. Freeze all coefficients from primary sources or a non-target source-parity
   case before examining the 22 target scores.
5. Preserve the existing strict no-regression gates and require an unseen
   transfer case before any generalization claim.
