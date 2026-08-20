# BING joint-solver session results (2026-08-18)

## Three-paper matrix on the new chassis (bare, 8x12cos / 128 steps/cycle)

| paper | metric | old bare core | new chassis | delta |
|---|---|---|---|---|
| Baik W1-W4 | CL macro RMSE | 0.793 | 0.755 | -4.8% |
| Izraelevitz Fig14 | CT MAE (Cd0=0.057) | 0.043 | **0.0386** | -10% (paper authors' model: 0.046) |
| Yang 2025 | lift MAE gf | 6.9 | 6.8 | parity (first-4 AoA: 2.9 vs 2.8) |
| Yang 2025 | thrust MAE gf | 12.9 | 13.0 | parity (prescribed-wake limited, known) |

Izra detail: 7/14 markers within experimental error bars; 15 deg family
excellent (max |d| 0.03), 25 deg family slight over-thrust. Yang: high-AoA
(20/25 deg) lift overshoot +9.9/+19.6 gf = missing leading-edge separation
(the case where circulation-removing LEV physics is actually the right sign).


## Validated infrastructure

- `bing_joint_solver.py` — from-scratch joint architecture (WingLattice,
  Biot-Savart, unsteady Bernoulli loads). Debugging toolchain for conventions.
- `bing_joint_ptera.py` — `JointLEVTEVSolver` on the pterasoftware unsteady
  chassis. LEV off == bare core to machine precision (G0b diff 0.00e+00).
  Steady limit CL 0.485 in lifting-line bounds [0.470, 0.481] (G0).
  LEV activation at alpha=20 stable, LESP pinned at 0.110 (G0c).
- LESP formula (reference port): 1.13*G_LE/(c*V_edge*(theta+sin theta)),
  gives 0.0899 at alpha=5 deg vs theoretical a0 ~ 0.087 (no fitting).
- Gates G1-G5 active every step (GateError on violation).
- Column-vs-particle consistency test (`debug_column_vs_particle.py`) —
  the decisive sign-convention arbiter; use it before touching conventions.

## Baik W1-W4 CL macro RMSE

| variant | macro CL | notes |
|---|---|---|
| bare @ 2x8, 32 steps, 2 cycles (historical) | 0.793 | reference baseline |
| **bare @ 8x8 cos, 128 steps, 3 cycles (this chassis)** | **0.755** | new best bare |
| V4B transfer on this chassis | 0.732 | delta only -0.02 |
| BING-LEV (bing mode, cap+edit+impulse) | 1.007 | W2 alone 1.159 < 1.241 |
| BING-LEV (v4b3d mode, intact bound+impulse) | ~3-5 | dead end, documented |
| historical V4B @ old-fluxv 4x8 128/3 | 0.594 | MIXED BASELINE: the old-fluxv solver differs; its advantage does not transfer onto the pterasoftware chassis |

## Key findings

1. The historical V4B 0.594 vs bare 0.793 comparison mixed baselines. On the
   machine-precision pterasoftware chassis the same LDVM transfer moves the
   macro by only -0.02. The 0.594-class result is a property of the old-fluxv
   baseline, not purely of the separation physics.
2. Native-3D LEV with circulation-removing loads (Ramesh cap) degrades
   W1/W3/W4: Baik's rounded-LE plates have load-AUGMENTING separation; the
   sharp-plate cap model has the wrong sign for them.
3. Intact-bound + impulse loads (v4b3d) double-counts circulation in the RHS
   feedback loop and diverges.
4. Temporal + mesh resolution beats LEV modeling on this chassis:
   32->128 steps/cycle and 2x8 -> 8x8 cosine moved the bare macro 0.793->0.755.

## Open items

- CD regression under LEV impulse loads (x-component sign/scale).
- Old-fluxv baseline as the actual carrier of the 0.594 (investigate or
  port its load pipeline onto this chassis).
- Implicit joint TEV (joint_tev=True) needs dt << current to not degrade;
  corrected TEV placement (extended lattice line) is already in the code.

## Yang viscous/separation-term experiments (2026-08-18, round 2)

Experiment A (drive the frozen LDVM pair with mid-span strip kinematics from
the chassis geometry): FAILS — deltas explode (dCL up to -52 vs V4b's -0.35).
The flapping section is not a valid 2D section history without the
sweep-rate correction; V4b's own report calls its Yang section a
"development proxy". Blocker identified precisely.

Experiment B (can the current LEV approximate the LDVM separation term?):

| AoA | LEV dL (mine) | LDVM dL (V4b) | LEV dD (mine) | LDVM dD (V4b) |
|---|---|---|---|---|
| 10 | -7.4 gf | -7.5 gf | -8.0 gf | +5.0 gf |
| 15 | -7.6 | -10.7 | -12.8 | +6.6 |
| 20 | -7.6 | -15.3 | -18.0 | +8.5 |
| 25 | -8.4 | -21.6 | -25.1 | +9.9 |

Verdict: lift-side YES in sign and scale at moderate AoA (10 deg: -7.4 vs
-7.5, near-exact), but saturates at high AoA (LESP pin self-limits);
drag-side NO — opposite sign: the axial/suction (CSf) term is entirely
absent from the KJ+impulse load decomposition.

What V4b does that this system does not (3 items):
1. CSf axial suction force (chordwise force = the drag/thrust growth)
2. Load-level delta with ndiv=32 LE-resolved 2D sheet physics
3. old-fluxv baseline load pipeline (Baik finding)

## Drag ledger results (2026-08-18, approved plan executed)

T1 (suction loss, Polhamus axial, CSf=2pi*A0^2 capped at crit, g^2 projected)
+ T3 (dynamic viscous, quadratic in instantaneous local relative velocity,
declared Cd0) — load-level algebra, zero circulation changes, zero fitting.

| paper | metric | before | +ledger | V4B | verdict |
|---|---|---|---|---|---|
| Yang | drag MAE gf | 13.0 | 9.67 | 2.64 | -26%, target <=6 not met (T2 blocker) |
| Yang | lift MAE gf | 6.8 | 6.82 | 4.55 | unchanged (gate holds) |
| Izra | CT MAE | 0.0386 | **0.0260** | 0.0198 | -33%; beats paper authors (0.046); near V4B |
| Baik | CD macro | 0.379 | **0.345** | 0.3452 | ties V4B |

Per-case Baik CD: W1 0.221->0.116, W3 0.306->0.215 (large gains); W2/W4
slightly worse (rounded-LE sign issue, known).

T2 (sep-weighted Rayleigh) first attempt: over-corrects lift at moderate AoA
(10 deg: 27.1->15.9 vs GT 31.5). Root cause identified: instantaneous LESP
fires on transient stroke reversals where cycle-mean data shows attached
behavior; the missing ingredient is separation-state MEMORY (natively
carried by the LDVM's a0/wake history). T2 stays gated off.

Gates green: G6a attached-limit exact; G6b Rayleigh ceiling clip.

## Two-scheme test (2026-08-20)

Scheme 1 (T2 memory via LEV-ON chassis + ledger): FAILED.
- LEV lift-delta reproduced (10 deg: -7.5 gf, matches experiment B) but moves
  AWAY from Yang GT (GT 31.5 > attached 27.1: Yang needs load AUGMENTATION
  at mid-AoA, not removal)
- Drag degraded to 22.4 gf: the LEV impulse force acts as thrust on the
  drag axis (long-standing CD sign issue confirmed at scale)
- Confound: 2-cycle wake immaturity. Verdict: LEV-on is the wrong tool for
  Yang loads; the V4B-polar (augmenting) direction is what the GT demands.

Scheme 2 (rounded-family ledger crit 0.239, Izra-declared): SUCCESS.
- W2/W4 regressions fully fixed (W2 0.663->0.632, W4 0.271->0.256, both now
  better than raw); W1/W3 keep most of their gain
- CLEAN macro (see bug below): raw 0.360 -> 0.307 @0.239 (0.316 @0.11)
  **vs V4B 0.3452: BEATS V4B on Baik CD**

BUG FOUND & FIXED: executor._heave_spacing_samples is lru_cached; multi-case
runs reused the FIRST case's heave spacing for later cases. Earlier P3
multi-case numbers (W2/W3/W4) were polluted; bing_scheme2_clean.py clears
the cache per case. CAVEAT: earlier CL-macro runs (0.755) also ran
multi-case in sequence and need a cache-clean rerun before quoting.
