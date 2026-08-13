# Meng 2025 preliminary frozen-transfer run audit

Date: 2026-08-13

Conditions: Figure 16, `U=8 m/s`, installation `alpha=5 deg`, `f=2 Hz`,
published twist amplitudes `0, 22.5, 45 deg peak-to-peak`.

Discretization: diagnostic smoke grid, 2 chord x 8 semispan panels per half,
24 steps/cycle, 2 cycles, 1 wake cycle retained.

Policy: no Meng force datum used for geometry correction, phase shift, scale,
offset, profile drag, LESP threshold, or any other fit.

## Executed baseline

The source-audited nominal geometry and movement run successfully through the
actual `UVPMHybridSolver` prescribed-wake load channel. The complete saved
three-condition matrix is:

| Twist p-p | Figure-16 L/Tnet | FluxV old L/T | old+polar L/T |
|---:|---:|---:|---:|
| 0 deg | 824/-139 gf | 664/+105 gf | 639/+55 gf |
| 22.5 deg | 794/-80 gf | 647/-4 gf | 645/-7 gf |
| 45 deg | 620/-230 gf | 617/-237 gf | 591/-260 gf |

For the central 22.5-deg condition:

| Quantity | FluxV old smoke | Figure-16 digitized balance mean |
|---|---:|---:|
| Mean lift | 646.684 gf | 794 +/-10 gf |
| Mean net thrust | -4.117 gf | -80 +/-3 gf |
| Mean CL | 0.401903 | not published |
| Mean CT | -0.002560 | not published |

The final-cycle source range was steps 24--47. Runtime is environment/JIT
dependent and is not used as an aerodynamic metric. The lift difference is
-147 gf (-18.5% relative to the digitized mean), but this
is still conditional on the drawing-derived outer outline/spar axis and the
unreported test density.  The +75.9 gf streamwise residual cannot be labelled
pure wing-model drag error because the paper does not publish a support/
mechanism tare.

The already frozen full-angle-polar residual was also evaluated on the same
movement. Its cycle-mean increments were `delta L=-0.0121 N` and
`delta D=+0.0314 N`, so the corresponding UVLM+polar diagnostic is about
645.45 gf lift and -7.32 gf net thrust. It does not manufacture the missing
whole-rig drag offset.

## Why a scored full v4b result is rejected

FluxV v4b requires a section/Re-specific LESP threshold with provenance.
Meng et al. publish no airfoil, thickness, static polar, stall angle or LESP
threshold.  As a stress test only, the Yang thin-plate mapping
`Lcrit=sin(5 deg)` was transferred unchanged. The paired LDVM discrepancy
then failed both strip and time-step robustness checks. With eight strips,
all three tested resolutions terminate with a singular linear system:

| LDVM steps/cycle | Result |
|---:|---|
| 48 | `LinAlgError: Singular matrix` |
| 96 | `LinAlgError: Singular matrix` |
| 128 | `LinAlgError: Singular matrix` |

An earlier six-strip stress test returned finite but plainly divergent values:

| LDVM steps/cycle | Mean delta lift | Mean delta drag | Max absolute delta lift |
|---:|---:|---:|---:|
| 48 | +279.34 N | +22.02 N | 11,043.63 N |
| 96 | -87.60 N | -38.82 N | 11,461.06 N |
| 128 | +1,736.77 N | -141.22 N | 225,785.79 N |

The lift increment changes sign between 48 and 96 steps and grows by orders of
magnitude at 128 steps; changing only the strip discretization makes the
linear solve singular. Roughly one third of the area-weighted time steps
trigger a new LEV event under this unsupported threshold.  A single 48-step
raw blend produced 4549.8 gf lift and -316.4 gf net thrust, but that number is
explicitly **invalid**: it is a non-converged stress-test artifact and must not
be plotted or scored as a v4b prediction.

No clipping, cap, tuned threshold, profile-drag offset or observation-derived
fallback was introduced.  Therefore the honest current result is:

- old FluxV and the frozen UVLM+polar branch are executable;
- full v4b is **blocked for Meng** by an unidentifiable section threshold and
  a numerically non-convergent forced threshold transfer;
- this failure is evidence about the present transfer model, not evidence of
  fabrication or incorrect force measurements in Meng et al.;
- the next legitimate model step is a section-independent bounded shedding
  closure (or a documented thin-membrane static-polar/LESP calibration), then
  a preregistered rerun.  It is not legitimate to choose `Lcrit` from Figure
  16--19 force agreement.

Executable artifacts are saved under
`runs/20260813_fig16_threecase_smoke/`, including phase histories, means,
the rejected stress matrix, PNG/PDF comparison, source/result hashes and a
machine-readable summary.
