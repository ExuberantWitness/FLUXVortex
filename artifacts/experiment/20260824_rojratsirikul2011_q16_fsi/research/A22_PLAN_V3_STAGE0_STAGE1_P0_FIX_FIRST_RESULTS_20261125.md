# A22 — Plan V3, stage 0 and stage 1 (P0-Fix): what was done, measured, and corrected

Date 2026-11-25. Plan: `/home/exuber/grok/FLUX_V5M_PLAN_AUTHOR_REPLY_AND_V3_20261125.md`.
Worktree `FLUXV_RUNS/v5m-fa8eaca`, branch `run/q16-lev-tev-pc-fsi-20260821`. Everything here is
uncommitted. The production default is bit-identical to before (parity test 6/6 after the
core edits; a final run follows the chain).

Run directory: `artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/v3_stage1/`.
Legacy state dumps: `.../eps_sweep/state_rigid_a5_eps{1e-12,1e-10}.npz`.

## 1. Stage 0 — the state the production solve actually saw

`diagnose_rigid_plate_coefficients.py --dump-state` now captures the wake at the influence-matrix
call (inside `propose()`, after convection and before the new row is inserted). Rigid plate,
alpha = 5 deg, 1500 steps. `diagnose_production_state_resolve_cpu.py` rebuilds the linear system
from that state and re-solves it on the CPU.

| system (last step, Kutta-Joukowski on the trailing-edge row) | legacy eps 1e-12 | legacy eps 1e-10 | v3 |
|---|---|---|---|
| production (the circulation the solver returned) | +0.15294 | +0.26258 | +0.21953 |
| convected wake, NO nascent row — residual of production Gamma | 7.8e-15 | 5.4e-15 | 2.7e-01 |
| same, re-solved | +0.15294 | +0.26258 | +0.18660 |
| convected wake + nascent row tied to the TE unknowns — residual | 3.4e-01 | 4.2e-02 | **7.2e-15** |
| same, re-solved | +0.19179 | +0.27288 | +0.21953 |
| straight connected wake along the freestream, every row tied | +0.22044 | +0.25420 | +0.21951 |

Reading:

* The legacy production circulation is, to rounding, the solution of the system the plan author
  describes: the circulation is solved against a convected wake that has no row at the trailing
  edge (plan V3, section 1 question 3).
* The v3 production circulation is, to rounding, the solution of the system with the nascent row
  tied to the trailing-edge unknowns and the convected rows on the right-hand side. The
  implementation is the design.
* On the v3 wake, leaving the trailing-edge gap open costs 0.2195 -> 0.1866 (-15 %), the same
  size as in the steady lattice (`diagnose_ring_lattice_steady_cpu.py`, "gap1").
* On the LEGACY wake shapes, closing the gap with an implicit nascent row raises Cn (1e-12:
  0.1529 -> 0.1918; 1e-10: 0.2626 -> 0.2729). These wakes are not ones v3 would produce; the
  number is what the one change is worth on that wake, nothing more.

**Withdrawn numbers.** An earlier version of this table, reported in conversation, gave
"0.07872 / 0.08997" for the nascent-row row of the two legacy states, with an explanation
involving the overlapping rows. Both were wrong: the script solved that system against the
freestream alone, i.e. without the 300 stored wake rows. The explanation was invented after the
fact for a wrong number. The script is corrected (the stored rows stay on the right-hand side,
as in the solver).

## 2. The legacy wake at solve time (eps 1e-12, step 1500)

Positions in U*dt = 0.0100 chord behind the ring trailing edge (the ring lattice ends a quarter
panel = 0.0167 chord aft of the physical trailing edge; the diagnostic script measures from the
physical one).

| row | upstream end (x, z) | downstream end (x, z) | circulation sum |
|---|---|---|---|
| 0 | (+1.013, -1.399) | (+2.477, -0.075) | +0.78916 |
| 1 | (+1.013, -1.399) | (+3.476, +0.088) | +0.78916 |
| 2 | (+3.476, +0.088) | (+4.468, +0.152) | +0.78916 |
| 3 | (+4.468, +0.152) | (+5.463, +0.175) | +0.78916 |

Rows 0 and 1 share their upstream corners, which sit 1.4 U*dt below the trailing-edge line;
from row 2 on the chain is connected with a spacing of 0.995 U*dt. After the legacy insertion the
new row and the re-riveted row both start at the trailing edge; one step later their shared
front corners have been thrown down by the induction of the narrow closed rows behind them. At
eps 1e-10 the throw is 0.28 U*dt, at 1e-9 it is hidden. This is the "singular undershoot" of the
plan, on the production state.

## 3. What the flags do

* `FLUXV_RING_KERNEL=v3` (`kernels_ring_velocity.py`): the exact law; a dimensionless collinear
  guard (`FLUXV_RING_COLLINEAR_REL`, 1e-12) replaces the m^4 constant; optional fixed-radius
  fourth-order core. Legacy code untouched.
* `FLUXV_TE_WAKE=v3` (`q16_flux_v5m_native.py`): the row that leaves the trailing edge this step
  is built before the solve, spans [trailing edge, upstream corners of the newest convected row],
  and its unit influence joins the trailing-edge columns (`system_aic`); the deposited leading-edge
  part is a known strength on the right-hand side (`system_rhs`); every load evaluation
  (collocation `external_flow`, load-packet velocities, the rigid KJ refinement, `mf2_history`,
  the load-derivative matrix) sees the same system; the insertion stores the row that was solved.
  Not available with `joint_separation_solve` (rejected at settings construction).
* `FLUXV_WAKE_CORE_PANELS` (default 0.25 chordwise panel): the core radius. Applied where free
  material points are convected (wake nodes, particles) to EVERY filament they see, and to the
  wake filaments of the load evaluation. Never in the influence matrix.

## 4. Finding: the trailing-edge junction under the exact law

With the core on the wake filaments only, the first v3 run (3 steps) threw the upstream corners of
the stored wake rows 0.29 U*dt UP in one step in which the freestream was 0.022 m/s: a normal
velocity of about 1.4 m/s at a corner sitting on the trailing edge.

Mechanism. The upstream corners of the newest row lie ON the lines of the bound trailing-edge
leg and of the two streamwise legs that meet there. At the first Runge-Kutta stage the collinear
guard zeroes every leg through the point. At the second stage the point is a distance h off the
line, and the exact law returns (Gamma_j - Gamma_j+1) / (4 pi h) from the two anti-parallel
streamwise legs of the last ring (their strengths differ by the spanwise gradient of the
circulation). The constant 1e-9 damped exactly this neighbourhood in the legacy kernel, which is
why it never showed. In a developed flow h is smaller and the term larger; it is not a start-up
effect.

Control, rigid plate 5 deg, 3 steps, normal offset of the three rows' upstream corners (chord):

| core | offsets |
|---|---|
| wake filaments only | +0.0000, +0.0029, +0.0016 |
| every filament, 0.25 panel | +0.0000, +0.0000, +0.0000 |
| no core | +0.0000, -0.0000, +0.0010 |

This widens the plan's "core where free vorticity is evaluated close to filaments" to the bound
sheet in the convection evaluation. It is reported to the plan author as a deviation with these
numbers.

## 5. v3 on the rigid plate, alpha = 5 deg

| run | Cn (window mean) | Kutta-Joukowski | wake behind the edge |
|---|---|---|---|
| legacy, eps 1e-9, 1500 steps | +0.3622 (earlier sweep) | — | rows 0, 1 overlap |
| 3 steps | +0.0083 (ramp) | +0.0001 | 3 rows at the edge (ramp) |
| 150 steps, last 50 | +0.2129 | +0.1829 | 0.0167 0.0267 0.0366 0.0466 0.0566 0.0666 |
| 1500 steps, last 1200 | **+0.2189** (min 0.2182, max 0.2190) | +0.2193 | same, spacing 0.0100 (min 0.0100, max 0.0101), first six normal offsets <= 0.0007 |

* Gate G-Scale: velocity x length invariant under x1 / x10 / x100 to 1e-10 on the fused kernel and
  on the torch path (`tests/test_ring_kernel_v3_gpu.py::test_v3_scale_gate_g_scale`). Pass.
* Gate G-VLM: Cn 0.2189 against 0.2194 +- 2 % (0.2150-0.2238): pass, -0.2 %. Kutta-Joukowski
  against Cn 0.18 % (< 1 %): pass.
* Against the experiment (0.1985): +10.3 %; against Helmbold (0.2267): -3.4 %.
* The 150-step run has Kutta-Joukowski 14 % below the force: a transient (circulation still
  changing); at 1500 steps the two agree to 0.2 %.
* Time 526 s against 507-513 s for the legacy run of the same length (+3 %).

The time-marching result is 0.2-0.7 % below the straight-wake value (0.2195-0.2204): the cost of
letting the wake roll up at AR = 2.

## 6. Chain of 2026-11-25 11:11-12:50 (v3 kernel and v3 trailing-edge wake, core 0.25 panel)

**Gate G-dt**, rigid plate 5 deg, wake held at 3 chords (same physical end time in every run):

| dt* | steps | Cn | Kutta-Joukowski | drift against the next coarser step | max abs Cn | seconds |
|---|---|---|---|---|---|---|
| 0.02 | 750 | 0.218962 | 0.219358 | - | 0.2924 | 122 |
| 0.01 | 1500 | 0.218943 | 0.219331 | -0.009 % | 0.2901 | 526 |
| 0.005 | 3000 | 0.218935 | 0.219319 | -0.004 % | 0.2888 | 3516 |

Gate: drift below 0.5 % per halving (the legacy path had +8 %). Pass. The wake spacing is one
U dt at every step size (0.0200 / 0.0100 / 0.0050, minimum equal to nominal).

**Rigid plate at 10 deg and 16 deg**, 1500 steps, against the experiment (digitised Fig. 9) and
Helmbold (2.598/rad at AR = 2):

| alpha | v3 Cn | v3 Kutta-Joukowski | against experiment | against Helmbold | particles at the last step | seconds | legacy at 1e-9 | legacy at 1e-12 |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.2189 | 0.2193 | +10.3 % (0.1985) | -3.4 % | 0 | 526 | +82.5 % | -23.2 % |
| 10 | 0.4358 | 0.4391 | -3.3 % (0.4508) | -3.5 % | 0 | 516 | +56.3 % | -39.8 % |
| 16 | 0.6329 | 0.6447 | -14.8 % (0.7431) | -11.7 % | 19792 | 1751 | +47.8 % | -30.6 % |

At 16 deg the window Cn moves between 0.574 and 0.697 and the leading-edge row carries both
signs: the old LEV deposit is still in place there. The remaining -14.8 % is the LEV model's.

**Legacy 150-step control** (eps 1e-9): Cn 0.3186, Kutta-Joukowski 0.2689. The 14-16 % gap between
the force and the steady Kutta-Joukowski value at 150 steps is a transient of the start-up, the
same on both paths; at 1500 steps the two agree to 0.2 % at 5 deg, 0.8 % at 10 deg and 1.9 % at
16 deg (unsteady there, with particles).

**S0-bench**, rigid plate 5 deg, full wake (300 rows, 100 free), v3, 60 timed steps after 400:

| phase | s/step | share |
|---|---|---|
| convection (`_advance_material_state`, RK3, 12000 free nodes x 9450 rings) | 0.3375 | 83 % |
| rigid Kutta-Joukowski refinement | 0.0216 | 5 % |
| author load assembly | 0.0090 | 2 % |
| load packet | 0.0015 | 0.4 % |
| influence matrix | 0.0007 | 0.2 % |
| rest (wake at the collocation points, solves, ledger) | about 0.037 | 9 % |

1.36e9 leg evaluations per step in the convection at 4.0e9 per second (float64). The speed line
is the wake self-convection and nothing else.

**Kernel tests**: 7 CPU and 4 GPU pass (`tests/test_ring_kernel_v3_cpu.py`,
`tests/test_ring_kernel_v3_gpu.py`, the fourth is the G-Scale kernel test). The default path
re-ran bit-identically over 1500 steps (Cn 0.26166047247843266 at eps 1e-10, 17 digits against
the sweep run of the day before).

## 6b. The constant test the user asked for (legacy path, constant in the denominator)

| constant | rigid 5 deg | rigid 10 deg | rigid 16 deg | flapping Fig. 14, MAE of CT | flapping bias |
|---|---|---|---|---|---|
| 1e-9 (old default) | +82.5 % | +56.3 % | +47.8 % | 0.0409 | +0.031 |
| 1e-10 | +31.8 % | +6.2 % | -2.6 % | 0.1099 | -0.097 |
| 1e-12 | -23.2 % | -39.8 % | -30.6 % | 0.2422 | -0.242 |
| 1e-16 | -25.4 % (0.1480) | -40.8 % (0.2667) | interrupted | not run | - |
| 1e-20 | not run | not run | not run | 0.3031 | -0.303 |

Membrane at 1e-12, 650 steps: A10 Cn 0.3335 against 0.5569 (-40.1 %), zmax/c +2.4 %; A16 0.5774
against 0.9200 (-37.2 %), zmax/c -13.3 %.

Flapping, per point (model CT): at 1e-12 ten of the fourteen points have the wrong sign; the
model/experiment ratio has median -0.16. Between 1e-12 and 1e-20 single points jump without order
(15-030 deg: +0.173 at 1e-10, -0.643 at 1e-12, +0.160 at 1e-20, experiment 0.180). The largest
LESP reading logged every 64 steps is 0.26 at 1e-10, 0.30 at 1e-12 and 3.17 at 1e-20.

The user's decision after this table: the constant stays in the denominator and its default
becomes 1e-12 (never zero); the trailing-edge wake fix goes into the default with it; the LEV is
rebuilt after Hirato et al. with the strength solved in the 3-D system and an explicit
leading-edge suction force; the new model becomes the default when implemented. The plan of
record is the approved plan of 2026-11-25 (session log); A23 will carry its results.

Corrections made while writing this: the membrane row of an earlier message ("-23 % to -40 %")
was wrong (it is -40.1 % and -37.2 %); the `membrane_ROJ11-A16_eps1e-12` point had run to
completion (650 records) but had no `window.json`, because the scheduler was stopped before it
wrote one; the record was restored from the scheduler's single window (t* 4.0-6.5, the window
of the A10 point of the same invocation) and the collector now skips such points with a log
line instead of raising.

**Scale invariance under a dimensional constant** (information, not a gate): at 1e-12 the
cut-off length sqrt(eps)/L is 0.22 mm on the wing panels (L = 4.59 mm) and 1.45 mm on the wake
streamwise legs (L = U dt = 0.69 mm). Scaling the geometry by ten changes the relative cut-off
tenfold; the solver-level x10 / x100 comparison was not run.

## 7. Mistakes in this segment, for the record

* Several Edit calls to ONE file were issued in a single parallel batch; the file came out with a
  duplicated `def native_aic` line and a duplicated block. Caught by the next import; removed.
  Rule: edits to one file are sequential.
* The eps 1e-10 state dump was lost once: it imported the solver at the instant the file was
  being edited. Rerun after the edits. Rule: no edit to a module during the first minute of a
  background run that imports it.
* `nascent_unit` summed the kernel output over the component axis (the kernel already sums the
  legs); same mistake in the re-solve script. Both fixed before any result depended on them.
* The re-solve script's right-hand side omitted the stored wake rows (section 1).
