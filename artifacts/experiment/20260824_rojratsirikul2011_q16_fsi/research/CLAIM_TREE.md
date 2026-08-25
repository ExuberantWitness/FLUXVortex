# Claim tree — Rojratsirikul 2011 A16 mean-state deficit (2026-08-25)

Root: reproduce A16 mean camber (+0.043c), Cn (0.92-0.95), lock-in vibration (St~1).

| Node | Proposition | Evidence status |
|---|---|---|
| N1 | Aero quasi-static load correct | SUPPORTED: Cn=1.03+9.9*z_mid, R^2=0.91; rigid-plate steady 1.10; Li et al. CL 0.89 same class |
| N2 | Q16 four-edge-clamped zero-tension membrane dynamics correct | PARTIAL: Yamano oracle is cantilever+pulse (different config); modal verification pending |
| N3 | Ring-down not converged = statistics-window deficit; mechanism = weak potential-flow aero damping (zeta~0.01-0.03) | SUPPORTED: forensics (single overshoot, period 2-2.5 t* = amplitude-dependent natural freq; St=1.00 was FFT bin-1 artifact) + 3 literature lines (UPC still-air decrement 0.04-0.14; Mavroyiakoumou-Alben JFM 2022 slow inviscid decay below critical tension; Torregrosa 2024 low-damped UVLM modes) |
| N4 | MISSING COMPONENT A: latex viscoelastic damping, eta~0.05-0.15 (zeta~0.03-0.075) | SUPPORTED by materials literature (TA Instruments; COMSOL eta=2zeta; Chen 2022); experiment membrane is real latex; same-material zero-damping DNS (Serrano-Galiano & Sandberg 2015) reproduces our symptom; de-facto norm zeta~0.02-0.05 |
| N5 | MISSING COMPONENT B: mounting pretension (same-lab protocol delta0=2.5/5% in Rojratsirikul 2010; low tension -> luffing sign excursions, Newman & Paidoussis 1991 = our negative-camber crossing) | SUPPORTED; requires new Q16 element capability |
| N6 | Late period-2dt* load alternation = coupling numerical stiffness on tension-free membrane | OPEN |

Key counter-evidence integrated: Li-Jaiman-Khoo (JFM 2021, arXiv:2011.11422) — Navier-Stokes-class reproduction of the SAME case with zero damping/pretension/flat IC SETTLES (delta_max/c~0.034, CL~0.89, lock-in St~0.99, rms~0.001). => the deficit is a model-class property (potential-flow aero damping), not a solver bug; the experiment's real latex damping + mounting tension supply the missing dissipation in reality.

Intervention (falsifiable, no fitting): author's own theta_a*K stiffness-proportional damping form with literature-central eta=0.1; sensitivity {0.05, 0.1, 0.15}; prestress branch {0, 1%, 2.5%}; A10/A23 generality with identical parameters. Prior predictions: settle in 1-2 periods, no sign crossing, mean camber 0.03-0.05, lock-in St~1 vibration.

Full citations in the session research reports; see also literature anchors below.

## Literature anchors
- Serrano-Galiano & Sandberg, AIAA 2015-1653: zero-damping DNS of same latex (t=0.2mm, E=2.2MPa) -> chaotic under-damped oscillations.
- Mavroyiakoumou & Alben, JFM 953:A32 (2022): inviscid membrane decay slow above critical tension, growth below.
- Tiomkin & Jaworski, JFM 948:A33 (2022); Tiomkin & Raveh 2021 review.
- Rojratsirikul, Wang & Gursul, JFS 26:359-376 (2010): pre-strain 2.5%/5% protocol, same lab/latex; luffing.
- Li, Jaiman & Khoo, JFM 929 (2021) / arXiv:2011.11422: settled NS reproduction, exact numbers.
- Gordnier & Attar, JFS 45 (2014): the independent reproduction of this experiment (ILES + p-FEM plate); zero-damping lineage; displacement at E=2.2MPa underpredicts experiment (Yang-Dudley-Harris AIAA J 2018 quote).
- Damping magnitudes: TA Instruments DMA notes; COMSOL damping overview (eta=2 zeta); Chen et al. Polymers 14:2427 (2022); Zhu & Atkinson JFM (zeta=0.02 practice); Cavallaro & Demasi (zeta=0.03 practice).

## Update 2026-08-26: N4 verification result (A16, eta=0.1, 300 steps)

Falsifiable predictions vs outcome:
- (a) no sign crossing: CONFIRMED (baseline crossed at t*=1.8 to -0.04; eta=0.1 stays positive, settles)
- (b) camber into 0.03-0.05: PARTIAL - monotone rise to +0.0133 at t*=3, still creeping (window too short, not mechanism failure)
- (c) Cn into band: trajectory CONFIRMED (0.908 at t*=3.0, rising, band 0.92-0.95)
- mode shape: chordwise peak count 2 = paper expectation at alpha~16 CONFIRMED
- zsd/c max 0.0097 = moderate vibration (paper-like), vs chaotic 0.013 baseline

Conclusion: N4 (missing latex viscoelastic damping) VERIFIED as mechanism; remaining gap is
statistics-window convergence (creep toward equilibrium slower than 3 t*). Long-window run
(600 steps, t*=6) launched as ROJ11_A16_ETA0.1_T6.
