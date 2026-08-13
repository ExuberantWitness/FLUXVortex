# Razak–Dimitriadis / Lambert phase-load digitization

The openly accessible source used for the numerical reconstruction is
Lambert et al., *Aerodynamic Optimization of a Flapping Wing in Hovering and
Forward Flight*, Aerospace 4(2), 22 (2017), DOI 10.3390/aerospace4020022.
The downloaded PDF has SHA-256
`114369840882385831758322d986514c0e8c114cfc5739a677eea25ff0bdc84b`.
It reproduces the experimental curves of Razak and Dimitriadis (2014).

Figures 9, 10, 11, 13, 14 and 15 were read from the PDF's vector drawing
commands rather than raster-traced. Figure 12 is intentionally excluded: the
paper subtracts each drag curve's own maximum, so it has no absolute-CD zero.

The published `Experiment` curve is a long-dashed path. In Figures 13--15 a
Katz-model path uses the same dash pattern, so the experimental path was
identified by requiring every published error-bar centre to lie on it. Across
all 12 panels the largest centre-to-path mismatch is 0.0002442 PDF point.

Files:

- `razak_lambert_experiment_errorbar_centres.csv`: 740 visible experimental
  centres with phase, coefficient, published one-cycle standard deviation and
  upper/lower limits. These centres are the scored observations.
- `razak_lambert_experiment_curves_raw.csv`: 752 raw path vertices, including
  clipped right-boundary intersections. These are retained for provenance but
  are not the primary scoring samples.
- `razak_lambert_vector_audit.json`: panel bounds, axis calibrations, vector
  path indices, dash patterns, case definitions and geometric residuals.
- `razak_lambert_extracted_vectors.png`: visual extraction audit.
- `SHA256SUMS.json`: hashes of all four generated artifacts.

No phase shift, amplitude scale or curve fit was applied. Phase is the
normalized displayed cycle. The nominal harmonic model cannot reproduce the
unpublished measured 64-sample motion history used by Lambert, so every model
comparison is labelled a nominal-motion diagnostic.
