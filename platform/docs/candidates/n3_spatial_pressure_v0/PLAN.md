# N3 spatial-pressure v0 experiment plan

## 1. Objective

- run id: `n3_spatial_pressure_v0_20260729_183255`
- selected idea: replace V4.1's scalar sustained `dCN_ds` force with an
  explicit near-field LEV spatial state. LESP is only a release/feed
  condition; the LEV changes the bound solution and one panel-pressure ledger.
- user's core requirement: implement the potential direction now and show
  Fig. 17/18/19 comparisons.
- non-negotiable constraints:
  - no retuning to Fig. 17/18/19;
  - no independent impulse/vortex-force ledger;
  - no total-force redistribution for co-design;
  - do not revive the frozen/falsified constant-ring Hirato production claim;
  - preserve existing user assets and frozen V4.1 files.
- research question: can a minimally executable spatial-LEV/unique-pressure
  replacement produce finite, conservative, spatially resolved loads and
  improve the existing Fig. 17/18/19 accuracy fingerprint?
- null hypothesis: the candidate is unstable, non-conservative, or does not
  improve the frozen comparison.
- alternative hypothesis: the candidate remains finite and conservative and
  reduces confirmed-scope Fig. 17/18/19 error without scalar force tuning.

## 2. Baseline And Comparability

- baseline id: existing frozen `platform/docs/s6_sweep_v41.json`, supplemented
  only where an already-existing V4.1 result is available.
- baseline variant: `closure=v41`.
- dataset: raw digitized samples parsed from `platform/docs/data.md`.
- primary metrics: confirmed-scope point-weighted MAE/RMSE and trend capture.
- required metric keys: Fig. 17/18/19 per-figure and L/T aggregates, force
  ledger residual, panel-pressure ledger residual, finite-state guard.
- comparability risks:
  - the user explicitly cancelled the fresh V4.1 validation; therefore this is
    exploratory evidence and cannot support a formal superiority claim;
  - Fig. 19(c,d) frequency identity remains unresolved and is shown only as a
    conditional diagnostic;
  - v0 may use a smaller pilot grid before the contract grid.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/claim_runtime/` | no-force spatial and pressure primitives | add one candidate state/pressure adapter | make N3 spatial state executable | numerical cost |
| `platform/_v2_robo.py` | production time loop | add isolated candidate closure and one unique panel-force path | expose candidate through compatible API | dirty, large solver |
| `platform/claim_nodes/*.yaml` | runtime closure eligibility | enable existing runtime nodes for the candidate; add explicit experimental role | keep graph/ledger valid | governance drift |
| `platform/lb_sweep_candidate.py` | absent | add isolated smoke/representative/full runner | do not mutate frozen runners/results | runtime |
| `platform/plot_fig171819_candidate.py` | absent | add raw-data overlay/score output | visual inspection without overwriting docs figures | plot identity |
| `platform/tests/` | component/solver guards | add candidate pressure and uniqueness tests | prevent double booking and silent fallback | test cost |

## 4. Execution Design

- minimal experiment: three sentinel cases covering AoA 0/5/15 and
  twist 0/22.5/45 behavior.
- smoke: low-cycle/low-resolution finite-state, pressure-ledger, force-ledger,
  and repeatability checks.
- representative run: 32 unique conditions covering one complete curve in
  every confirmed Fig. panel plus conditional Fig. 19(c,d) diagnostics.
- full run: 151 confirmed conditions; optionally add 33 conditions for a
  conditional 184-condition visual package.
- expected outputs: isolated sweep JSON, manifest, scorecard, per-figure PNG
  and JSON sidecars, metrics and summary under this candidate directory.
- stop condition: NaN/Inf, duplicated force ledger, non-closing circulation or
  pressure ledger, runaway state growth, or smoke runtime incompatible with a
  bounded campaign.
- abandonment condition: the mechanism requires target-selected constants,
  pressure clipping as the only stability mechanism, or reactivation of a
  falsified claim.
- strongest alternative hypothesis: the dominant error is N2 separated
  pressure/drag rather than N3 spatial LEV state.

## 5. Runtime Strategy

- smoke command: to be frozen after the adapter API passes unit tests.
- main command: `lb_sweep_candidate.py --scope representative32`, followed by
  `confirmed151` if the pilot gates pass.
- output root:
  `platform/docs/candidates/n3_spatial_pressure_v0/runs/`.
- no frames or raw per-step histories in full sweeps; compressed panel fields
  only for sentinels.
- monitor at approximately 60 s, 120 s, 300 s, 600 s, then 1800 s.
- kill/relaunch only for a concrete implementation or infrastructure failure,
  never because the measured curve is unfavorable.

## 6. Fallbacks And Recovery

- If the full continuous-sheet operator is too slow, retain the same physical
  candidate on the representative grid and report the scale blocker; do not
  substitute the falsified constant-ring model.
- If the candidate API breaks the legacy closure, revert only the candidate
  delta and keep V4.1 behavior protected by compatibility tests.
- If the candidate fails smoke physics guards, publish a NO-GO diagnostic
  rather than tuning it to the target figures.

## 7. Checklist Link

- checklist: `CHECKLIST.md`
- next unchecked item: finish executable-component/interface audit.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-07-29 18:32 +08:00 | User cancelled fresh V4.1 validation and authorized direct candidate implementation | explicit user direction | comparison downgraded to exploratory |
| 2026-07-29 19:20 +08:00 | Existing constant-ring Hirato, pure-rVPM, and actual-body repeated-insertion paths rejected by implementation audit | their claim nodes are frozen falsified/dead-end or have an existing time-refinement NO-GO | none may be renamed as the new candidate |
| 2026-07-29 19:28 +08:00 | Candidate discretization fixed as the causal P2 near-LEV band below | smallest executable route that preserves a spatial state and a unique panel-pressure force | permits an exploratory Fig run without scalar-force tuning |

## 9. Frozen v0 Candidate Definition

The executable v0 is a **thin-camber-surface research candidate**, not the
final actual-thickness pressure model.

At step \(n\), the existing N1 bound lattice and TEV wake are retained.  N3
owns a chronological set of continuous quadratic material doublet bands.  A
new band joins the convected release edge from \(n-1\) to the current
leading-edge release location.  Its three explicit temporal traces are

```text
q(0)   = q_(n-1)
q(1/2) = 0.5 * (q_(n-1) + q_n)
q(1)   = q_n
```

This is the registered first-order-in-time causal discretization; its
adequacy is decided only by the `dt`, `dt/2` family.  It is not selected from
Fig. 17/18/19.

For every spanwise source basis, the current-row influence is evaluated from
the same continuous P2 geometry.  The bound and source strengths are solved
together so that active Eq. 6 LESP observations satisfy the inherited
critical value.  LESP therefore decides release and circulation supply only;
it never appears in a force-amplitude expression.  Inactive strips receive
zero current-row source.  The NACA-2406 value of the inherited critical LESP
is explicitly `inherited-uncertain`; this v0 may test it but may not choose it
from the target figures.

The current release location uses the already documented Ramesh first-LEV
placement and explicit chord-normal finite-wing embedding.  Once born, P2
strengths are material invariants.  Geometry is advanced with Heun using the
N1 bound/TEV local velocity.  The omission of P2 self-advection in v0 is an
explicit model approximation and a promotion blocker, not a hidden fitted
roll-up curve.

The wing potential jump during active release is

```text
psi_panel = Gamma_bound + q_n(strip)
```

and the only pressure force is

```text
delta_p = rho * (V_total,t · grad_s(psi)
                 + dGamma_bound/dt
                 + dq_release/dt)
f_panel = delta_p * area * normal
```

assembled by `unified_panel_pressure`.  The N3 ledger entry is the exact
per-panel difference between this coupled pressure and the same-step N1
counterfactual pressure.  Hence `N1 + N3 == coupled` by construction, while
old `dCN_ds`, LEV impulse, VNF, particle force, and independent LE suction are
all zero in this closure.

Numerical quadrature is not a physical parameter.  The registered family is
edge order `8/16/24`; order 16 is allowed for the campaign only if order
16→24 changes sentinel forces by at most 0.5% and the vectorized operator
matches the independent order-24 reference.

## 10. Frozen v0 GO/NO-GO Order

1. algebra: P2 continuity, current-row source/bound residual, active LESP
   residual, pressure ledger, and `N1 + N3 == coupled`;
2. attached limit: no release implies bitwise-zero N3 pressure contribution;
3. time family: `dt`, `dt/2` must not show source strength or induced velocity
   proportional to `1/dt`; force difference must contract;
4. three sentinel conditions: finite, repeatable, no pressure clipping, no
   target-data access;
5. representative 32 curves;
6. confirmed 151 only if steps 1--5 pass and projected runtime is bounded.

Failure at steps 1--4 is a candidate NO-GO.  It does not authorize a core,
clamp, decay constant, source smoothing, or Fig-target parameter scan.
