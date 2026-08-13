# Yang 2023 and Baik 2012 reproducibility audit

## Yang et al. IMAV 2023

Source: H.-H. Yang et al., *Numerical Simulation Framework of Bird-Inspired
Ornithopter in Forward Flight*, IMAV 2023, open PDF
<https://www.imavs.org/papers/2023/3.pdf>. The audited local copy has SHA-256
`e158713c4520d757be347e36a7038d60b0c080a12a27db0c2ed91a51eb1b59ba`.

The rigid validation wing is straightforward to reconstruct aerodynamically:
a 130 mm by 250 mm rectangular, 1 mm-thick balsa wing (`t/c=0.77%`) driven by
a planar crank-rocker. The published mechanism parameters are `phi0=12 deg`,
`L0=37.6 mm`, `L1=10.0 mm`, `L2=33.5 mm`, and `L3=20.0 mm`; the nominal
motion reconstructs to about -15 to +45 deg. Conditions are `U=5.5 m/s`,
`f=2.5 Hz`, and installation angles 0, 5, 10, 15, 20 and 25 deg.

Figure 7 publishes only cycle-mean lift and thrust at the six angles. It does
not publish phase loads, the laser-measured angle history, phase zero, root
offset, tare/inertial-load procedure, sampling/filtering, repeat count, error
bars, or a complete thrust-axis convention. Consequently:

- geometry and nominal four-bar motion: high reproducibility;
- six mean-load points: usable after digitization (about +/-0.8 gf reading
  uncertainty);
- actual experimental kinematics and force processing: not closed;
- appropriate use: secondary 3-D mean-load scan, not phase validation.

The 2023 and 2025 experiments must not be mixed. The 2025 mechanism uses
different link lengths, approximately -30 to +40 deg motion and an 80 mm
joint-to-wing-root offset, and its simulation used unpublished laser histories.

## Baik et al. JFM 2012

The relevant paper is *Unsteady force generation and vortex dynamics of
pitching and plunging aerofoils*, JFM 709 (2012), DOI
10.1017/jfm.2012.318. It is distinct from the authors' same-year
*Experiments in Fluids* SD7003 study.

The most complete accessible specification is Baik's University of Michigan
dissertation, available from
<https://deepblue.lib.umich.edu/items/e1cec17c-27e7-46c6-8956-704134cb257a>.
The related AIAA precursor is
<https://deepblue.lib.umich.edu/items/48a8a6ca-efa7-4ea8-a20c-b16ffb571af8>.
The wide-Strouhal experiments use a 76 mm-chord, 600 mm-span
flat plate with rounded edges, `t/c=6.25%`, quarter-chord pitching, endplate,
`Re about 5000`, and four cases:

| Case | St | k | h0/c | pitch amplitude | Period |
|---|---:|---:|---:|---:|---:|
| W1 | 0.16 | 0.5 | 0.50 | 13.16 deg | 7.13 s |
| W2 | 0.32 | 1.0 | 0.50 | 33.73 deg | 3.56 s |
| W3 | 0.16 | 1.0 | 0.25 | 13.16 deg | 3.56 s |
| W4 | 0.32 | 0.5 | 1.00 | 33.73 deg | 7.13 s |

High-St motion is not an ordinary sinusoidal plunge. The apparatus enforces a
sinusoidal plunge-induced angle,
`atan(-h_dot/U)=alpha_pl,max sin(2 pi f t)`, and integrates that velocity to
obtain displacement. Replacing it with harmonic displacement is a material
reconstruction error. An AIAA precursor misprints W3 `k=0.5`; definitions,
period and the dissertation identify `k=1.0`.

The dissertation Figures 5.24--5.27 (printed pages 179--182) provide
complete-cycle `Fx`, `Fy`, `Cd`,
`Cl`, `Cm` and `Cp`, including corrected total hydrodynamic loads. The AIAA
precursor subtracts the initial 8-degree steady load and must not be mixed with
these corrected totals. The experiment reports 2000 Hz sampling, about 110
cycles (discarding five at each end), a 1 Hz Fourier low-pass, 500 phase
averages in the wide-St study, and coefficient confidence intervals about
+/-0.02. This makes Baik W1--W4 a high-quality future two-dimensional
LEV/pitch-plunge validation set, although it does not establish large-wing 3-D
generalization.

Neither Yang nor Baik uses NACA 0012/0013. Thielicke 2011 and Heathcote 2008
remain excluded from the present campaign as requested.
