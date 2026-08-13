# Baik W1--W4 experimental force-history freeze

## Authoritative source

- Yeon Sik Baik, *Unsteady Force Generation and Vortex Dynamics of Pitching
  and Plunging Airfoils at Low Reynolds Number*, University of Michigan PhD
  dissertation, 2011.
- Deep Blue item: <https://deepblue.lib.umich.edu/items/e1cec17c-27e7-46c6-8956-704134cb257a>
- Deep Blue bitstream UUID: `1ef3aed8-ca97-48ec-b992-85acc3be814a`.
- Downloaded source PDF (not committed because it is 29.99 MB):
  `baik_2011_dissertation.pdf`, 29,993,055 bytes, SHA-256
  `2efbf3becd339df61cc9275e2e933700ef75216504580d5f4e5cca1e80eadc0a`.

To rerun the extractor, download that exact Deep Blue bitstream into this
directory under the stated filename and verify its SHA-256 first.  The four
embedded source JPEGs used by the frozen extraction are committed under
`extracted/`, so visual/source-pixel auditing does not require committing the
large dissertation PDF.

The experimental ground truth is the corrected total hydrodynamic force in
thesis Figures 5.24--5.27: physical PDF pages 204--207, printed pages 179--182.
The figure-to-case mapping is W1/Fig. 5.24, W2/Fig. 5.25, W3/Fig. 5.26 and
W4/Fig. 5.27.  The PDF contains each figure as a 1318 x 1602, 220 dpi CMYK
JPEG, not as vector paths.  The original embedded JPEGs are retained in
`extracted/`.

Do **not** substitute the AIAA precursor curves.  The precursor used an older
relative baseline, reports no W4 curve, and its means are materially different.

## Force datum and processing in the source

The thesis states that the direct ATI Mini40 force record was processed as
follows:

1. an in-air tare test measured inertial loading and was subtracted from the
   submerged force test;
2. sensor pre-trigger bias was removed by a relative measurement;
3. the separately measured steady hydrodynamic force was added back, producing
   the corrected total load used here;
4. data were sampled at 2000 Hz, Fourier low-pass filtered at 1 Hz, and the
   first and last five cycles were removed;
5. the plotted force coefficients used 500 phase-averaged samples.

The source reports approximately +/-0.02 as a maximum error / 95% confidence
interval in force coefficient.  The dotted confidence bounds are visibly
plotted only for the normalized Fx and Fy panels; the CL/CD panels contain the
measurement curve and a dotted horizontal cycle-mean guide.

## Digitization method

`extract_baik2012_fig524_527.py` applies a fixed pixel-axis calibration to the
CL and CD panels.  The measured centerline is traced with one common dynamic
programming rule across all eight panels.  The explicitly labelled horizontal
mean guide is masked so it cannot be mistaken for a measurement curve.

There is no phase shift, phase optimization, amplitude fit, offset fit, scale
fit, model-output completion, or copying between cases.  The printed means are
used only to locate the mean guide and as an independent QA check.  They are
not imposed on the extracted waveform.

The source-pixel traces are written to
`baik2012_w1_w4_corrected_total_cl_cd_source_pixels.csv`.  Because the left and
right panels have slightly different raster widths, the paired comparison file
`baik2012_w1_w4_corrected_total_cl_cd.csv` independently interpolates each
source trace onto a declared 401-point phase grid.  That interpolation does not
increase the information content.

The overlay `baik2012_fig524_527_digitization_overlay.png` is the primary visual
audit.  Red is the extracted centerline.  Integrated trace means versus printed
panel means are:

| case | variable | digitized | printed | absolute difference |
|---|---|---:|---:|---:|
| W1 | CD | 0.0311 | 0.0315 | 0.0004 |
| W1 | CL | 1.0436 | 1.0400 | 0.0036 |
| W2 | CD | -0.1264 | -0.1270 | 0.0006 |
| W2 | CL | 2.1357 | 2.1100 | 0.0257 |
| W3 | CD | 0.1263 | 0.1270 | 0.0007 |
| W3 | CL | 1.1433 | 1.1400 | 0.0033 |
| W4 | CD | -0.3072 | -0.3080 | 0.0008 |
| W4 | CL | 1.3856 | 1.3700 | 0.0156 |

W2 CL spans six coefficient units in only 408 vertical pixels; its 0.026 mean
difference is compatible with a roughly two-pixel raster/line-centre ambiguity.
It is retained without offset correction.

## Digitization uncertainty

The phase reading uncertainty is approximately +/-0.004 cycle (about two
pixels).  Conservative, panel-specific curve-reading uncertainties are:

| case | CL reading | CD reading |
|---|---:|---:|
| W1 | +/-0.02 | +/-0.004 |
| W2 | +/-0.03 | +/-0.023 |
| W3 | +/-0.02 | +/-0.004 |
| W4 | +/-0.03 | +/-0.020 |

These are raster-reading uncertainties and are distinct from the source's
approximately +/-0.02 experimental uncertainty.  Neither should be interpreted
as permission to phase-align or rescale a prediction.

## Published analytical baselines

Figures 5.28--5.31 are on physical PDF pages 208--209 (printed pages 183--184).
Their legend unambiguously identifies:

- thick line: measured W1--W4 lift;
- open circles: standard Theodorsen;
- crosses: Theodorsen with `C(k)=1`.

The thick measured line in these figures is not used to overwrite or complete
the experimental GT.  `extract_baik2012_fig528_531_theodorsen.py` extracts the
unique open-circle centres as a secondary author-model reference: 29 for W1
and 28 for W2--W4.  The raw Hough detector returned one exact duplicate centre
in each of W2--W4; these duplicates are audited and removed rather than
double-counted as published markers.  The
cross-marker raster extractor was rejected after overlay review, so only its
identity is frozen; an exact C(k)=1 baseline should instead be generated from
the declared analytical formula.  This rejection is deliberate provenance,
not missing experimental data.
