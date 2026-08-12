# Izraelevitz et al. (2017) Figure 14 digitization

## Source identity

- Paper: J. S. Izraelevitz, Q. Zhu, and M. S. Triantafyllou, *State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span*, AIAA Journal 55(4), 2017.
- DOI: `10.2514/1.J055144`.
- Source figure: Figure 14 on PDF page 13.
- Experimental series: open black squares labeled `Scherer 1968`.
- Numerical reference series: authors' six-state ULLT, one-state ULLT, and quasi-steady plus added-mass curves.

The paper contains Figures 1--15; there is no Figure 17.  Figure 11 is a numerical UVLM comparison.  Figure 14 is the paper's experimental comparison and is therefore the correct experimental benchmark.

## Geometry and operating conditions printed in the paper

- rectangular finite wing, aspect ratio `AR=3`;
- heave-pitch motion about the three-quarter-chord line;
- `z(t)=h cos(omega t)` and `theta(t)=theta_max cos(omega t+psi)`;
- `h/c=0.6`, `St=0.2`, and `k_mid=0.52`;
- phase offset `psi=15--105 deg`;
- pitch amplitudes `theta_max=15 deg` and `25 deg`;
- the authors add `Cd0=0.057`, reported from Scherer's static foil tests, to all inviscid predictions.

Scherer's original report (DTIC `AD0673776`) closes the dimensional geometry:

- NACA 63A015, rectangular planform with slightly rounded tips;
- chord `4 in = 0.1016 m`, span `12 in = 0.3048 m`, `AR=3`;
- three-quarter-chord pitch axis;
- Figure 23 supplies the `theta_max=15 deg, J'=6` experiment used in the top
  panel; Figure 29 supplies `theta_max=25 deg, J'=6` for the lower panel.

There is a source conflict that must not be hidden: Scherer's original static
test table reports `CD0=0.027`, whereas Izraelevitz et al. apply `Cd0=0.057`.
The main reproduction follows the 2017 Figure-14 implementation value and
reports `0.027` as a fixed source sensitivity; neither value is selected using
the Figure-14 observations.

## Extraction method

The PDF page is vector artwork.  Curve vertices, square-marker centres, and error-bar endpoints were read directly from the PDF drawing paths with PyMuPDF; no raster cursor placement was used.  Axis maps were obtained from the vector grid:

- top panel: `y=134.1401825 -> CT=0`, `y=56.2197571 -> CT=0.35`;
- bottom panel: `y=241.9563904 -> CT=0`, `y=164.4504700 -> CT=0.25`;
- horizontal axis: `x=67.7190552 -> psi=0 deg`, `x=276.8464050 -> psi=120 deg`.

The two experimental markers at `theta_max=15 deg, psi=15 deg` and at `psi=75 deg` are retained as separate plotted observations rather than silently averaged.  Figure 14 does not define the statistical meaning of the error bars; the CSV therefore records their digitized upper/lower half spans without calling them standard deviations or confidence intervals.

The `theta_max=25 deg` QS+added-mass curve is clipped above the top axis at `psi=15 deg`; no numerical value is invented for that point.
