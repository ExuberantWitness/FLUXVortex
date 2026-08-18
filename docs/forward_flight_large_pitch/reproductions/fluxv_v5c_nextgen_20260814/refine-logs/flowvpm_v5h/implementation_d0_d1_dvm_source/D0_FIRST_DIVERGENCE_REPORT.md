# D0 author-source first-divergence report

Status: event-history parity passed; full row-wise strength parity remains
qualified because the clean-room backend uses a direct linear Kelvin solve
while the author program uses finite-tolerance Newton iterations.

No target-paper observation was read and no threshold, core, or coefficient
was selected from Yang, Izraelevitz Figure 14, or Baik W1–W4.

## Frozen reference

- case: author LDVM v2.5 SD7003, `Re=30000`, `Lcrit=0.18`, LE pivot,
  45-degree Eldredge ramp-hold-return, `K=0.2`;
- 500 formatted motion rows, 499 advanced/output rows;
- author shedding mask: Fortran `i_step=117..290`, equivalent to output rows
  `116..289`, 174 consecutive LEVs;
- final wake count: 174 LEVs and 499 TEVs.

## First divergences in the pre-fix Python path

| divergence | author source | previous Python |
|---|---:|---:|
| first numerical, `i_step=2`, `A0_pre` | `-0.0664283610692` | `-0.0664293073119` |
| first persistent state | first solved TEV cleared | retained `|Gamma|=0.0139647` |
| `i_step=289`, `A0_pre` | `0.181447235778`, LEV 173 | `0.179988363296`, inactive |
| `i_step=290`, `A0_pre` | `0.180191476356`, LEV 174 | `0.178686408747`, inactive |

The previous Python mask ended at Fortran step 288 and contained 172 LEVs.

## Causal factor audit

| isolated Python change | final LEVs | last shedding step |
|---|---:|---:|
| previous implementation | 172 | 288 |
| first-TEV initialization clear only | 173 | 289 |
| provisional newest TEV in first/restart LE velocity only | 173 | 289 |
| camber ordinate in world geometry only | 174 | 290 |
| camber ordinate + first-TEV clear | 174 | 290 |

Reverse ablations in an external author-source audit produced the same causal
ordering.  Omitting all three details reduces the author topology to 172.
This excludes LESP threshold and Vatistas core radius as the explanation.

## Implemented clean-room source mode

`LDVM2D(source_parity=True)` is isolated from the historical default path and
implements:

1. paired camber ordinate and slope arrays at the cosine stations;
2. the author world transform
   `X=sx+(x-xp)cos(alpha)+zc sin(alpha)`,
   `Z=sy-(x-xp)sin(alpha)+zc cos(alpha)`;
3. persistence of the first solved TEV as zero while retaining solved/stored
   diagnostics;
4. same-step provisional TEV induction in first/restart LEV birth velocity;
5. per-step `dt_i` from a source-arithmetic, seven-decimal formatted analytic
   reconstruction.  Exact author-file replay exists only in the external
   forensic harness; the in-repository clean-room guard does not claim
   row-identical motion input.

The source-parity test now requires the exact output-row mask `116..289`, 174
LEVs, 499 TEVs, the first `t*=1.638354` cap, signed `A0_post=0.18`, and the
published first load landmarks within the already measured clean-room
tolerances.  It no longer accepts a final-count error of two.

## Remaining qualification

- The Python solver closes its linear Kelvin system to roundoff; the author
  source stops Newton iteration at `1e-5`.  Event/onset topology can therefore
  be exact without claiming bitwise circulation history equality.
- The author initialization deliberately clears the first solved TEV after
  the first Kelvin solve, so the persisted-wake Kelvin ledger has an explicit
  initialization-discard term on that row.
- The reference has constant freestream.  A future variable-speed adapter must
  separately record previous-step translational speed and current-step
  boundary/birth speed before it is called source-faithful.
