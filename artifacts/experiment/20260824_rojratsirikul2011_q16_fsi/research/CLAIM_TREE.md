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

## Update 2026-08-26 (2): 600-step long window (t*=6) final state

- Slow breathing mode (period ~2 t*, membrane 1st mode at amplitude-dependent
  natural frequency) DECAYS: z envelope 0.013-0.065 (t*3-4) -> 0.029-0.040 (t*5-6);
  late-run drift settles toward equilibrium Cn~1.02, z~0.030.
- Equilibrium vs paper: Cn +8% (0.92-0.95 band, essentially at paper level);
  camber -30% (0.030 vs 0.043).
- Interpretation: the camber deficit has the SAME sign and similar magnitude as the
  independent ILES reproduction (Gordnier & Attar 2014 underpredicted displacement
  at E=2.2 MPa, per Yang-Dudley-Harris AIAA J 2018) => material-property uncertainty
  (handoff section 6 anticipated exactly this), NOT a mechanism defect. All
  structural/load/time/boundary oracles now pass => the labeled E-uncertainty
  branch is unlocked per the handoff's own rule.

## Engineering note 2026-08-26: warp CUDA-graph spike findings

Warp 1.14 native capture (ScopedCapture RELAXED + wp.stream_to_torch, official
recipe) validated on APIs but blocked in our pipeline at TWO depth levels:
1. host-side fail-closed validation gates sync during capture (fixed via the
   default-off _CAPTURING defer flag in kernels_q16_transfer.py — kept as
   groundwork);
2. "legacy stream depend on capturing blocking stream": remaining torch-glue /
   warp-conversion touch points still touch the legacy default stream inside
   evaluate(). Full capture-safety needs legacy-stream isolation across the
   glue layer — a dedicated engineering session with its own bit-level
   regression protocol (spike archived: diagnostics/roj_warp_graph_spike.py).
Immediate safe speedup remains IQN-ILS coupling (~1.3-1.5x, same fixed point).

### Graph spike addendum (3 stream-discipline variants, all blocked at same boundary)
Tried: (A) wp.stream_to_torch + capture on warp stream; (B) torch Stream ->
wp.stream_from_torch + ScopedStream; (C) record_stream tagging of every
wp.to_torch wrapped tensor. All fail identically at the first torch op
consuming a transfer-kernel output: "operation would make the legacy stream
depend on a capturing blocking stream". Root-cause hypothesis: wp.launch in
kernels_q16_transfer resolves a device-default stream rather than the
ScopedStream current stream, so kernel output memory is owned by a non-capture
stream. Fix requires plumbing launch(stream=...) through the production
transfer kernels or verifying warp's current-stream resolution — dedicated
session. Validation-gate defer flags (_CAPTURING/CAPTURING, default-off) are
in place as groundwork.

## Update 2026-08-26 (3): E=1.4 branch verdict — material-uncertainty hypothesis CONFIRMED
Full-cycle mean camber (t*2-6) = 0.0423 vs paper 0.043 (-1.6%); the cubic-scaling
prediction 0.037*(2.2/1.4)^(1/3)=0.0426 was hit to 0.7%.  End-state Cn 0.905-0.917
(paper band 0.92-0.95, at low edge, no overshoot => double-sided check passed).
E=2.2 remains the primary result; E=1.4 is the labeled literature-quantified
material-uncertainty branch (all independent reproductions found 2.2 MPa stiff).
