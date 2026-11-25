# A23 — New defaults, the free-wake core, and the leading-edge vortex after Hirato: first results

Date 2026-11-25 (evening). Plan of record: the approved plan of this session (GitHub sync, stage A
defaults, stage B = leading-edge vortex after Hirato et al., flapping comparison first). Worktree
`FLUXV_RUNS/v5m-fa8eaca`, branch `run/q16-lev-tev-pc-fsi-20260821`. The three commits of the sync
are on GitHub (`97599d7`, `b82816e`, `e78b8b1`); everything in this note is later and uncommitted.

Run directory: `artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/stageB/`.

## 1. What the owner decided

1. The constant stays in the denominator and is never zero; its default becomes 1e-12.
2. The trailing-edge wake fix (implicit nascent row, connected chain) is the default with it.
3. The leading-edge vortex is rebuilt after Hirato et al. with the strength solved in the 3-D
   system; the 2-D source bank is not used for it.
4. The explicit leading-edge suction of the paper (Eq. 20) is added.
5. The new model is the default as soon as it is implemented; the flapping wing comes first.

Decision 4 is reversed by the data in section 6: the suction is in the code behind
`FLUXV_LE_SUCTION` and is off by default.  Section 6 says why.

## 2. Stage A: the defaults

* `FLUXV_RING_EPS_V` defaults to 1e-12 (m^4); the kernel's form is unchanged.
* `FLUXV_TE_WAKE` defaults to v3: the nascent trailing-edge row is solved implicitly and the
  wake is one connected sheet.
* Plan V3 Bug-4: the reference length of the kernel's core is the wing's own chordwise panel
  length (`mean_chordwise_panel`), no longer `1/chordwise_panels` metres.  Inert: the old-default
  150-step run is unchanged to the printed digits (0.3186 both times).
* The parity test `tests/test_roj_unified_case_runner_parity_gpu.py` passes 6/6 under the new
  defaults.

Rigid plate, alpha = 5 deg, 150 steps, window the last 50:

| configuration | Cn |
|---|---|
| new default (1e-12, implicit row) | 0.2138 |
| 1e-12, legacy wake order | 0.1545 |
| 1e-9, legacy wake order (the old default) | 0.3186 |

## 3. The flapping wing diverges at 1e-12 without a core, and what fixes it

With the two defaults above and the old leading-edge vortex, the flapping wing
(Izraelevitz-Scherer Fig. 14) blew up: `IZRA-15-015` gave thrust coefficients of +3e6 and
`IZRA-15-030` +3e7.  A per-step probe of the first condition (`/tmp/zcode_flap_probe.py`, the
kinematics, grid and settings of the production runner) located it:

| step | t* | max abs suction parameter | wake corner to the nearest collocation point (panels) |
|---|---|---|---|
| 321 | 15.05 | 0.165 | 0.500 |
| 329 | 15.42 | 22.7 | 0.242 |
| 337 | 15.80 | 458.8 | 0.265 |

The first three cycles are periodic and sane (parameter up to 0.36).  The wing passes through its
own wake; a filament sweeping across the surface passes through every distance, and at 1e-12 the
constant only tames those below 0.03 mm (0.2 mm at 1e-9).  This is Bug-4 of the plan author: the
free wake had no core and the old constant was doing that job.

The fix keeps the constant where the owner put it and gives the free wake the core the author's
own code already had: `rough=True` sets `core_fraction = 0.1`, which nothing ever passed, and the
core radius is `core_fraction * max(longest edge of the ring, chordwise panel)`.  It is now
applied wherever free material points are convected (every filament they see, the bound sheet
included, so the trailing-edge legs still cancel) and wherever the wake is evaluated at the
collocation points and at the load midpoints; never in the influence matrix and never on the
nascent trailing-edge row, whose back leg must cancel the bound trailing-edge leg exactly.
`FLUXV_FREE_WAKE_CORE_FRACTION`, default 0.1, 0 switches it off.  For the Izraelevitz wing the
core is 1.27 mm, 0.27 U dt.

With the core the same condition runs its 512 steps periodic: parameter up to 0.36, streamwise
force between -148 N and +49 N.

With that core and the old leading-edge vortex the whole of Fig. 14 runs (section 5, "stage A").

## 4. The leading-edge vortex after Hirato et al. (FLUXV_LEV_MODEL=hirato, the default)

New module `src/fluxvortex/warp_fsi/lev_hirato.py` (pure functions), wired into
`Q16NativeV5MSolver`.

* **Sheet.** One chain of at most 3 ring rows per strip (`NativeV5MState.lev_chain`, a
  `LevChain`: rings, strengths, rows in use, last step's strength, which rows are joined).  The
  nascent ring has its aft corners on the geometric leading edge; the pseudovortex ring spans the
  chord with the same strength, so the leading-edge filament cancels (the paper's Fig. 5).
* **Placement.** A strip that shed at the previous step too splits its last ring at one third of
  the way to the chain's end (Eq. 7).  A strip that starts again after a pause gets a first ring
  by Ramesh's rule, U A0 dt (sin a, cos a) / sqrt 2, and nothing is joined to the old rows: the
  paper presumes a sheet that keeps shedding, and a flapping wing stops and restarts every cycle.
  The rule "joined only if it shed at the previous step" has no constant of its own.  A first
  version always re-linked, so a strip that started again put its new ring a third of the way to
  a chain that had convected away (the ring lengths were not measured).  Measured on
  `IZRA-15-015`: 23 000 particles against 19 000 with the rule, and a suction parameter of 0.69
  against 0.57.
* **Strength.** Solved in the 3-D system with the bound circulation (`lev_hirato.solve_coupled`):
  no penetration on every row but the leading-edge row of a shedding strip, that row pinned so
  the suction parameter equals the critical value (`lesp_crit` and the factor 1.13 untouched),
  and one closure row per shedding strip: no penetration at that leading-edge row.  The nascent
  trailing-edge row of a shedding strip carries the trailing-edge circulation plus the shed one.
  The active set is iterated: a strip that did not shed may cross the critical value through its
  neighbours' induction (the second decision of the paper's flowchart).
* **Convection.** The chains take the same Runge-Kutta step as the wake; particles see them.
* **Leaving rows.** The row that leaves the 3-row buffer goes to the particle field as a closed
  ring of full strength, with the smoothing radius and spacing of the old deposit.
* **Loads.** The collocation points and the filament midpoints see the chains and the
  pseudovortex rings; the unsteady term carries d/dt of the bound circulation together with that
  of the strip's leading-edge vortex (the paper's Eq. 17).
* **Tests.** `tests/test_lev_hirato_cpu.py`, 13, CPU: topology, the split, the joined rule, the
  release, the closed loop, the sign of the shed circulation, the negative feedback, the coupled
  solve (residual 2.5e-14 on the Rojratsirikul lattice, 20 strips shedding at 16 deg), suction.

On the Rojratsirikul lattice at 16 deg the coupled solve raises the trailing-edge circulation of a
shedding strip from 0.0458 to 0.0657 in one step: the lift of the leading-edge vortex, which the
old deposit (a closed ribbon carrying only the increment, 5.8 % of an open filament's induction)
never gave.

The rigid plate at 5 deg, 150 steps, is unchanged by the model: Cn 0.2137 with and without it (no
strip sheds there).

## 5. Flapping wing, Fig. 14: three configurations

All on the new defaults (1e-12, implicit trailing-edge row, free-wake core 0.1), 12 conditions,
thrust coefficient = the last cycle's mean, +0.057 of profile drag as the runner adds it.

| run | MAE | bias | RMSE | largest error | in the error bar |
|---|---|---|---|---|---|
| old path, 1e-9 (the old default) | 0.0409 | +0.0310 | 0.0470 | 0.0844 | 1 / 14 |
| old path, 1e-12 | 0.2422 | -0.2422 | 0.3044 | 0.8225 | 0 / 14 |
| stage A: new defaults, old leading-edge vortex | 0.0974 | -0.0593 | 0.2567 | 0.9496 | 4 / 14 |
| new leading-edge vortex, no explicit suction (the default now) | **0.0206** | +0.0009 | 0.0268 | 0.0599 | 5 / 14 |

The runner's own gate on the mean absolute error is 0.0175; the default is at 0.0206.

Per point: experiment, then stage A (new defaults, old leading-edge vortex) and the new
leading-edge vortex, each with its error (run minus experiment):

| point | experiment | stage A | error | new leading-edge vortex | error |
|---|---|---|---|---|---|
| 15-015 | 0.1231 | 0.1998 | +0.0767 | 0.1374 | +0.0143 |
| 15-015 (repeat) | 0.1449 | 0.1998 | +0.0550 | 0.1374 | -0.0074 |
| 15-030 | 0.1797 | -0.7700 | -0.9496 | 0.1632 | -0.0164 |
| 15-045 | 0.2344 | 0.1744 | -0.0600 | 0.1879 | -0.0464 |
| 15-060 | 0.2306 | 0.1977 | -0.0330 | 0.1993 | -0.0314 |
| 15-075 | 0.2126 | 0.2045 | -0.0081 | 0.2058 | -0.0068 |
| 15-075 (repeat) | 0.2052 | 0.2045 | -0.0006 | 0.2058 | +0.0007 |
| 15-090 | 0.1915 | 0.2001 | +0.0087 | 0.2001 | +0.0087 |
| 15-105 | 0.1635 | 0.2011 | +0.0376 | 0.2011 | +0.0376 |
| 25-045 | 0.0960 | 0.0756 | -0.0204 | 0.0811 | -0.0149 |
| 25-060 | 0.1062 | 0.0823 | -0.0239 | 0.0925 | -0.0138 |
| 25-075 | 0.0844 | 0.0833 | -0.0010 | 0.0833 | -0.0010 |
| 25-090 | 0.0433 | 0.0725 | +0.0292 | 0.0725 | +0.0292 |
| 25-105 | 0.0121 | 0.0720 | +0.0599 | 0.0720 | +0.0599 |

Reading:

* Stage A: thirteen points within 0.08 of the experiment, four inside its error bar, mean
  absolute error 0.032 over those thirteen; the fourteenth, `IZRA-15-030`, collapses (-0.95) and is
  what the 0.0974 almost entirely is.  Without the core this condition diverged (section 3); with
  the core it collapses instead.
* New leading-edge vortex: `IZRA-15-030` is -0.016.  Every point is within 0.06 of the experiment,
  the mean error is zero to three digits, and the mean absolute error is half the old default's.
* Five points are the same in both runs (15-090, 15-105, 25-075, 25-090, 25-105): no strip
  reaches the critical suction parameter there, nothing is shed, and the two models are the same
  solver.  Their errors, +0.009 to +0.060, have nothing to do with the leading-edge vortex; the
  two largest are where the measured thrust is smallest (0.043 and 0.012).
* The errors with shedding are now both signs and small: +0.014 and -0.007 on 15-015, -0.016 on
  15-030, -0.046 on 15-045.

Folders: `stageB/izraelevitz_fig14_stageA_baseline`, `stageB/izraelevitz_fig14_hirato`;
`platform/warp_vpm/compare_flapping_runs.py` prints the table from any set of run folders.

## 6. The explicit suction counts the force twice on a shedding strip

`IZRA-15-015`, experiment 0.1231 and 0.1449:

| configuration | thrust coefficient |
|---|---|
| new leading-edge vortex, no explicit suction | 0.1374 (errors +0.014, -0.008) |
| new leading-edge vortex, explicit suction on | 0.3848 (+0.262, +0.240) |
| old leading-edge vortex | 0.1998 |

The loads here are Kutta-Joukowski vector forces on the bound filaments, evaluated in a velocity
field that contains the leading-edge rings.  On a shedding strip the suction is in that vector
force; the explicit term of the paper, pi rho c U^2 A0^2 with A0 at its critical value, adds up
to 52 N on this wing (a thrust coefficient of 0.36) on top of it.  So it is off by default.

On an attached strip the evidence points the other way: the rigid plate at 5 deg, 150 steps, has
a streamwise coefficient of +0.0085 without the term and -0.0013 with it, which is the inviscid
flat plate with its leading-edge suction.  Whether the term belongs on attached strips is not
settled; the flag is there for it.

## 6b. The rigid plate under the new default path: 16 deg is a regression

Rigid plate, 1500 steps, window t* 3-15, the default path (1e-12, implicit trailing-edge row,
free-wake core, leading-edge vortex after Hirato, no explicit suction):

| alpha | Cn | window min / max | Kutta-Joukowski | against experiment | against Helmbold | particles | seconds |
|---|---|---|---|---|---|---|---|
| 5 | 0.2197 | 0.2190 / 0.2198 | 0.2201 | +10.7 % (0.1985) | -3.1 % | 0 | 589 |
| 10 | 0.4350 | 0.4336 / 0.4353 | 0.4392 | -3.5 % (0.4508) | -3.7 % | 0 | 589 |
| 16 | **0.1651** | -0.3215 / +0.6227 | 0.1536 | **-77.8 %** (0.7431) | -77.0 % | 26955 | 2804 |

5 and 10 deg do not shed and are the attached plate: 0.2189 and 0.4358 under the exact kernel
(note A22), 0.2197 and 0.4350 here.  The wake spacing is one U dt.

16 deg sheds, and there the new leading-edge vortex is far worse than the old one:

| every 100 steps | Cn |
|---|---|
| old deposit (note A22, exact kernel) | 0.62 0.59 0.65 0.58 0.65 0.62 0.64 0.64 0.63 0.63 0.59 0.65 0.61 0.64 |
| new (this run) | 0.38 0.21 0.22 0.15 0.03 0.01 0.25 0.24 0.08 0.17 0.18 0.17 -0.02 |

The old one settles into a periodic shedding; the new one never settles.  At the last step the
trailing-edge row is at -0.005 m^2/s at mid-span while the newest wake row carries 1.91 against
the trailing-edge row's 0.82: almost all of what is shed is the leading-edge part, and the near
wake is disordered (row spacing between -0.085 and +0.107 chord).

Two defects of the implementation, both found by reading the code after this run:

1. **Rows that leave the buffer are smoothed away.**  The chain keeps 3 rows as exact rings and
   hands the rest to the particle field.  In continuous shedding a row is 1-3 U dt long, and the
   particles it becomes have the smoothing radius of the old deposit, 0.04 chord = 4 U dt of this
   wing.  The front and back edges of such a ring cancel within that radius, and the streamwise
   legs of the sheet, once particles, act on the collocation points of the next strip (2.3 mm
   from the node line) at 13 % of the filament law.  So the sheet is 3 rows long and nothing more:
   the defect of the old deposit (note A14) in a new place.
2. **The steady lift of the leading-edge vortex is not in the loads.**  The pseudovortex ring is
   bound vorticity: the trailing edge sheds the bound circulation plus lambda, and the unsteady
   term carries d/dt of the sum.  But the Kutta-Joukowski forces are integrated on the bound
   filaments only; the pseudovortex ring's own filaments get no force.  On the flapping wing the
   unsteady term dominates and the gap was invisible there.

Until both are repaired the flapping table of section 5 is preliminary as well: it was obtained
with the loads of item 2.

## 7. Open

* Gate G-Flap of the plan (Izra 14 points: no blow-up, no collapse, mean absolute error at
  least back to the old 0.0409 and towards 0.0175): no blow-up, no collapse, 0.0206.  The four
  attached points (+0.009 to +0.060) are what separates it from the 0.0175.
* Time step halved on the flapping wing with the new model: not run.
* The new model costs about 1.8 times the old one on the flapping wing (0.70 s against 0.40 s a
  step on `IZRA-15-015`): 19 000 particles against 9 000, and the active-set iteration.
* `particle_max_age_steps = 64` on the flapping wing is half a cycle: the rows handed to the
  particle field live that long and are then culled.  Frozen setting; recorded.
* On the membrane wing the structural load is the author's pressure path; the suction, if it were
  on, would enter the packet and the panel forces but not that path.
* The load-derivative operator handed to the structural solver is the unconstrained matrix.
* The paper states its method for wings of AR 4 and higher; these wings are AR 2 and 3.
* Test inventory under the new defaults; rigid 5, 10, 16 deg and the membrane at 1500 / 650
  steps; the time-step halving; all after the flapping table.

## 8. Mistakes of the evening, for the record

* Two runs were launched on code that was not yet right: the first baseline diverged (no core),
  and the first full run with the new model always re-linked the chains.  Both were stopped and
  their partial outputs removed; neither number is used above.
* The plan said no core would be needed at 1e-12.  That was wrong for the flapping wing.
* The recommendation to add the explicit suction was wrong for this force formulation.
* The Edit tool twice reported that an existing file did not exist; nothing was lost.  Later
  edits went through a script that requires each replaced string to be unique.
