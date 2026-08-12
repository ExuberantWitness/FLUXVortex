# Three-dimensional computation classification of the two LDVM dissertations

## Question and classification rule

The audit asks whether either dissertation provides a genuinely
three-dimensional computational case suitable for validating a finite-wing
FluxV implementation.  The following categories are kept separate:

- **2D low-order:** one airfoil section, with no spanwise degrees of freedom;
- **quasi-2D experiment:** a physical finite-span apparatus or a measurement
  plane treated as a representative section;
- **3D CFD:** a computational domain and mesh resolving a finite wing and its
  spanwise flow;
- **3D low-order:** a finite-wing circulation/wake computation with spanwise
  states and loads.

A solver whose name contains `3D`, a finite-span laboratory model, or a review
citation to a finite-wing paper is not by itself a 3D computational case.

## Audited dissertations

| Dissertation | SHA-256 |
|---|---|
| K. Ramesh, 2013, LDVM foundation | `735f0cb9af7636bf3fa3e21845f4a722b40a55004c409891af0687fa87f740c4` |
| M. Martínez-Carmena, 2023, separated-vortex DVM development | `aad40937d9b9f92506291d0840e984bf5c9f5df5d1683884bb11dd0141575cfd` |

The PDFs were audited from the separately acquired
`docs/forward_flight_large_pitch/literature/candidates_20260812/` directory and
are intentionally not duplicated in this result package.

## Evidence table

| Work and evidence | What is actually present | Strict classification | Suitable as a 3D FluxV validation? |
|---|---|---|---|
| Ramesh 2013, PDF p. 25 / printed p. 3 | The dissertation objective explicitly describes a low-order model for “two-dimensional flows” past airfoils. | 2D low-order | No |
| Ramesh 2013, Chapter 4 and Table 4.1, PDF p. 127 / printed p. 105 | SD7003, NACA 0015 and flat-plate pitch/plunge cases are parameterized by section, Reynolds number and critical LESP.  There is no span grid, planform or spanwise load. | 2D low-order cases | No |
| Ramesh 2013, PDF p. 136 / printed p. 114 | Chapter 5 CFD is stated to use a “2-D body-fitted mesh” with 92,400 cells. | 2D CFD | No |
| Ramesh 2013, PDF p. 173 / printed p. 150 | The immersed-boundary domain is stated to use a “2-D computational mesh” with 684,992 cells. | 2D CFD | No |
| Ramesh 2013, PDF p. 174 / printed p. 152 | The appendix again identifies the body-fitted CFD mesh as 2D with 92,400 cells. | 2D CFD | No |
| Ramesh 2013, Appendix B, PDF pp. 179-182 / printed pp. 157-160, Figures B.1-B.2 | A physical flat plate is tested in the AFRL water tunnel.  Dye is injected near the three-quarter-semispan location and forces are measured on the apparatus. | Finite physical apparatus used for quasi-2D/sectional validation; not a 3D computation | No |
| Martínez-Carmena 2023, PDF p. 34 / thesis p. 3 | The author notes that 3D effects can be important but explicitly says the dissertation restricts itself to two-dimensional flows. | Explicit 2D scope | No |
| Martínez-Carmena 2023, PDF p. 60 / thesis p. 29 | A cross-section at an arbitrary span station is introduced as a two-dimensional simplification. | Quasi-2D sectional model | No |
| Martínez-Carmena 2023, PDF pp. 76-77 / thesis pp. 45-46, Figure 3.4 | The OpenFOAM grid and reported cell counts describe an airfoil-section mesh. | 2D CFD | No |
| Martínez-Carmena 2023, PDF p. 82 / thesis p. 51 | The text explicitly says the two-dimensionality discards 3D lift-induced drag. | Explicit 2D force model | No |
| Martínez-Carmena 2023, Chapters 5-6 | CFD/DVM comparisons concern NACA-section ramp motions, separated shear layers and reattachment, without a finite-wing spanwise state. | 2D CFD plus 2D DVM | No |

## Verdict

Neither dissertation contains a genuine 3D low-order calculation or a genuine
3D CFD finite-wing validation case.  Ramesh includes finite physical
water-tunnel hardware and sectional flow visualization, and both theses discuss
three-dimensional phenomena or cite 3D literature.  Those facts do not convert
their computational cases into 3D benchmarks.

`CFL3D` in Ramesh is the name of a CFD solver used by cited/comparison work; it
is not evidence that the dissertation's validation meshes resolve a finite
wing.  The dissertation's own mesh descriptions quoted above are explicitly
two-dimensional.

## What can safely transfer to FluxV

The dissertations remain valuable for testing two-dimensional section-level
mechanisms:

- Fourier extraction of leading-edge suction;
- critical-LESP onset, intermittent shedding and material-vortex placement;
- Kelvin/LESP simultaneous closure;
- two-dimensional load identities and LEV timing;
- dependence of a calibrated threshold on airfoil family and Reynolds number.

They cannot validate:

- spanwise LESP coupling;
- tip-vortex interaction;
- finite-wing induced drag;
- three-dimensional LEV topology or spanwise convection;
- the mapping from per-strip Kelvin constraints into one finite-wing UVLM wake.

Those claims require an independent finite-wing experimental or computational
benchmark.

## Martínez-Carmena mechanism relevant to a later ablation

Chapter 5 introduces a separated-vortex DVM in which leading-edge circulation
feeding is not held at a constant suction threshold.  Its key relations are

\[
L=A_0,
\]

\[
\dot\Gamma_{LE}=\frac{U^2A_0^2}{r_{LE}},
\qquad
\Gamma_i(t)=\frac{U^2A_0^2(t)}{r_{LE}}\Delta t,
\]

from Eqs. 5.1, 5.5 and 5.6.  For symmetric NACA sections the dissertation uses

\[
r_{LE}=1.1019t^2,
\]

with thickness $t$ expressed as a fraction of chord.  Equation 5.12 supplies
an onset derivative criterion.  The author shows that the suction parameter
does not generally remain fixed after separation, which is a direct limitation
of constant-\(L\) LDVM variants.

This supports a later **alternative SVDVM ablation** after constant-critical-
LESP source parity is established.  It does not support stacking the two
leading-edge circulation owners, and it does not supply the missing 3D
finite-wing closure.
