I have thorough, specific coverage. One key confirmation: the UVLM + simplified Leishman-Beddoes load-estimation approach for flapping rectangular wings (the Murua/Palacios lineage and the Aerospace MDPI 2017 "Vortex Lattice Simulations of Attached and Separated Flows around Flapping Wings"). I now have all the pieces with named authors, years, venues, equations, and direct assessments. Let me write the synthesis.

---

# RECENT (2020-2025) LEV / DYNAMIC-STALL MODELS FOR FLAPPING FLIGHT — assessed against your 4 gaps

Bottom line up front: **no single published model solves all four of your gaps on a bird-scale flapping wing at once** — that gap is real and is your contribution opportunity. But each of your four failures has a clean, named, equation-level fix in the recent literature, and three of them point to the *same* root cause: your LEV is a lift-only, monotonically-growing, never-detaching core, whereas the physics (and every good recent model) makes the LEV (a) **detach/saturate by a state variable, not by duration**, and (b) **carry a chordwise (thrust/drag) projection, not just lift**. I give the specific papers, equations, and what each gets right/wrong below.

---

## TIER 1 — The directly-transferable fixes (read these first)

### [A] Ayancik & Mulleners 2022, *J. Fluid Mech.* 944, A17 — "All you need is time to generalise the Goman–Khrabrov dynamic stall model" → **fixes gap (2) cleanly, and disarms your cruise/AoA conflict**
EPFL group. This is the single most useful paper for you. The Goman–Khrabrov (G-K) model carries ONE internal state `X` (fraction of attached flow, X=1 attached, X=0 separated):

  τ₁ Ẋ(t) + X(t) = X₀( α(t) − τ₂ α̇(t) )

with lift recovered by Kirchhoff's law  C_L = C_Lα · sinα · ((1+√X)/2)².

Why this is exactly your gap (2): the static separation curve **X₀(α) is where the AoA-dependence lives** — it gives CL_max-then-drop because X₀ collapses past static stall, so lift rises with α then *falls*. Your model "keeps growing with AoA" precisely because you have no X₀(α). The argument `α − τ₂α̇` is the key: **stall is triggered by the delayed effective angle, which automatically separates the slow base-AoA component from the fast oscillatory flapping α̇** — this is the formal answer to your complaint "cannot distinguish base-AoA stall from the oscillatory flapping alpha." Their advance over plain G-K: they replace the two empirical constants with physics, via the dynamic-stall-delay power law (fit R²=0.978, valid Re 7.5e4–1e6, your Re~1e5 is in range):

  Δt_ds·U∞/c = 0.0815·(α̇_ss·c / 2U∞)^(−7/9) + 4.24,  τ₂ = Δt_ds (ramp) or the sinusoidal form Eq. 3.4; τ₁ from post-stall vortex-shedding Strouhal (St≈0.25).

Gets right: AoA-dependent CL_max-then-drop with no per-condition tuning; delay scales with instantaneous pitch-rate so the same constant set spans your 1.4–2.6 Hz. Limitation: it is a 2D trailing-edge-separation model — it models the bound-circulation collapse (your "full stall: bound collapses"), not the LEV convective overshoot itself. So you would **keep your LDVM-LEV for the attached/overshoot regime and gate it / blend it with X** for the collapse. This is the cleanest published object to graft onto your sectional strip.

### [B] AbuNawas & Qawasmeh 2026, *J. Aircraft* (in press) — "Coupling Dynamic Stall with Lifting-Line Theory for Gust and Maneuver Analysis" → **the architectural template for embedding [A] into a 3D method**
Augments classical lifting-line with the G-K model in **state-space form**: structural DOF + Wagner & Küssner aerodynamic-lag states + the G-K internal separation state, all integrated together; circulatory C_L reconstructed from motion + memory + stall. This is the published precedent for exactly what you'd do: put the G-K `X` state on each UVLM strip, driven by the strip's local α_eff, with Wagner/Küssner handling the unsteady lag you currently fake with your "2nd-order convection-delay." Caveat: their target is flutter/gust, not flapping propulsion — they do not report thrust. You'd be extending it.

### [C] Hernandez Gelado & Ramesh 2022, AIAA AVIATION Forum, paper 2022-4105 — "A Reduced-Order Discrete-Vortex Method for Flows with Leading-Edge Vortex Shedding" (N-LEV LDVM) → **diagnoses gaps (1)+(2) as the same bug and gives two detachment criteria**
This paper is almost a mirror of your method and its failure. They limit the LEV to N elements merged into one core (≈ your "N-LEV merging into ONE coherent core"). They state two findings you are living:
- "the merged vortex will keep ingesting vorticity generated at the leading edge **indefinitely, not allowing the formation of new LEVs**" → this is your **gap (1)/(2)**: a core that never detaches keeps adding lift, breaks up/down symmetry at α=0, and grows monotonically with AoA.
- "**excessive computation of lift … in the early stages of LEV formation … a non-physical impulse produced by the merging**" → a known artifact of lumping a sheet into a core (they cite Darakananda & Eldredge's impulse-matching correction). If your +37% at α=0 partly comes from the *merge step* (not only the convection lag), this is the named culprit and named fix.

Their proposed detachment triggers (use these to make your LEV detach by *state*, killing the duration-driven growth):
1. **Trailing-edge flow reversal / rear-stagnation-point reaching the TE** (Kissing et al. 2020; Widmann & Tropea) → when met, stop feeding the core and shed it.
2. **Maximum accumulated LEV circulation** threshold.

Honest status: in the 2022 paper they implement N-LEV *without* the detachment condition yet ("performs well until the instant of LEV detachment"), i.e. they have the same open problem; the criteria are proposed and partly characterized, not yet a finished closed model. So you are at the frontier here, not behind it.

### [D] Kamimizu, Liu & Nakata 2025, arXiv:2508.18703 — "Data-Driven Discovery and Formulation Refines the Quasi-Steady Model of Flapping-Wing Aerodynamics" → **directly targets gap (3), the advance-ratio error**
Chiba group. They show the standard QS lumping `‖v_body+v_flap‖² C_F(α)` is *wrong across advance ratio* and split it into three sparse-regression-discovered terms:

  dF ∝ ‖v_body‖²·C_F,bd(α) + ‖v_flap‖²·C_F,fl(α) + 2(v_body·v_flap)·C_F,cp(α)

i.e. body, flapping, and a **cross-coupling** term each get their own α-dependent coefficient. The effective coefficient Ĉ_F then *varies with flight speed/advance ratio* instead of being constant. Validated on hawkmoth forward flight over **J = 0–1.1** (your range J~0.58–1 sits inside) — cuts hawkmoth error 29% → 11–16%. This is the precise mechanism for your gap (3): you "calibrated at J~1 cruise" with effectively constant coefficients, so the flapping-dominated J~0.58 point drifts +18–22%. The cross term `2(v_body·v_flap)` is exactly the down/up-stroke asymmetry that a single fixed C set misrepresents. They also discover a **rotational Wagner term** ∝ (Δω)·sign(a)·√|a| and a **spanwise-velocity** term — both relevant if your residual has stroke-reversal structure. Limitation: it's a blade-element QS model (no explicit shed wake), insect-scale CFD-trained; you'd refit the three coefficient sets to your bird-scale data, but the *functional decomposition* transfers directly.

---

## TIER 2 — The thrust gap (4): the hard one, but the physics is named

Your gap (4) is the deepest and the literature is blunt that **a Bernoulli/LESP-capped potential model structurally cannot produce flapping thrust** — you correctly diagnosed that your x-force ~0 because you cap LE-suction and take LEV as lift-only. Three results pin down what's missing:

### [E] Polhamus leading-edge-suction analogy (NASA TN D-3767, 1966) — the conceptual key you're half-using
When the LEV forms, the LE suction is *lost as a chordwise force* and **rotated to act normal to the chord as extra "vortex lift."** You apply the *cap* (lose the suction → correct, kills overshoot) but then you take the vortex contribution as **lift only**. The analogy says the same missing suction has a defined geometric projection. The multiple sources above state it explicitly: "the LEV on a thin airfoil gives an **extra suction force parallel to the normal force, giving rise to extra lift AND drag**." Your model is internally inconsistent: it keeps the normal projection, discards the tangential one. That discarded tangential projection, integrated over the flap cycle with the stroke-plane tilt, is a chunk of your missing thrust.

### [F] Gao, Lu et al. 2019, *AIP Advances* 9, 035314 — "Passing-over leading-edge vortex: the thrust booster in heaving airfoil" → **the sign rule for LEV thrust vs drag**
The LEV produces **thrust only while it sits upstream of the airfoil's max-thickness point** (surface tilted upstream → low-pressure core pulls forward); once it convects past, the same core becomes **drag**. At high transverse (flapping) velocity the LEV can be pushed back over the LE and boost transient thrust "several times." Force quantified via near-field pressure-Poisson + boundary-vorticity-flux theory. Direct implication: a *time-resolved* LEV-position-dependent chordwise force is what makes flapping thrust **grow with frequency** (higher flap velocity → LEV held forward longer → more forward suction). Your model's thrust doesn't grow with freq because it has no such term. This is the mechanistic origin of your "OPPOSITE frequency trend."

### [G] Fernandez-Feria & Alaminos-Quesada 2018, *J. Fluid Mech.* — "Unsteady thrust, lift and moment of a 2D flapping thin airfoil in the presence of leading-edge vortices" (+ Fernandez-Feria 2016 *PRF* 1, 084502; Gordillo 2025 *JFM* "A note on the thrust of airfoils") → **closed-form LEV-thrust corrections**
The linear-potential vortical-impulse line generalizes Garrick's LE-suction thrust with an extra term: mean thrust gains a contribution depending on Theodorsen C(k) **and a second complex function C₁(k)** of reduced frequency, capturing the full bound+wake vorticity (Garrick only had LE-suction + pressure projection). Their LEV correction (Brown–Michael shed LEV) gives a time-averaged thrust change **∝ a₀²k³** at high k for pitching (negligible for *pure* heave — note this, it matters for your α=0 case). Gordillo 2025 shows the impulse-theory thrust equals Garrick's pressure integration *only if wake vertical velocities are computed self-consistently* — a caution if you try to bolt thrust on naively. These give you the analytic frequency-scaling (T_mean rising with k) that your LESP-capped Bernoulli x-force is missing.

### [H] Nakata, Liu & Bomphrey 2015, *J. Fluid Mech.* 783 — "A CFD-informed quasi-steady model of flapping-wing aerodynamics" → **confirms your form-drag term AND the uncomfortable truth about chordwise force**
Gives exactly the drag law you already adopted:  **C_D = C_Dmin·cos²α + C_Dmax·sin²α**  (your "Cd~sin²α separated-flow pressure drag" — you're on solid published ground for the *half* of gap 4 you already found). Lift uses C_L = C_L1(α³−πα²/2)+C_L2(α²−πα/2). BUT the honest caveat for you: in their wing-fixed frame they report **the chordwise (and spanwise) force is negligible — 1.3% and 0.3% of normal force**. So the *blade-element/normal-force* school essentially says "there is almost no chordwise force, the thrust comes from tilting the big normal force through the stroke." That is a genuinely *different stance* from the LEV-suction-thrust school [E,F,G]. This is the real unresolved fork in the literature and you should state it: your missing ~4 N net thrust is either (i) a mis-resolved **projection of the normal force** through the instantaneous stroke-plane geometry (Nakata/QS view — check your kinematic tilt bookkeeping first, it's the cheaper bug), or (ii) a genuinely **missing LEV chordwise-suction term** (Polhamus/Gao/Fernandez-Feria view). My read: do (i) first — a constant +4N offset that flips thrust-vs-drag and has the wrong frequency slope smells partly like a frame/projection bookkeeping error plus a missing freq-growing vortex-thrust term; you likely need BOTH the sin²α form drag (have it) and a Gao-style LEV-position chordwise term (don't have it).

---

## TIER 3 — Validation landscape on bird-scale flapping wings (what's actually been checked against measured lift AND thrust)

Honest assessment: **bird-scale, cross-frequency, cross-windspeed validation of a low-order LEV model against *both* cycle-mean lift and thrust is essentially absent** — this is exactly the white space your AST paper occupies.

- **UVLM + simplified Leishman–Beddoes for flapping rectangular wings** — Murua/Palacios lineage; "Vortex Lattice Simulations of Attached and Separated Flows around Flapping Wings," *Aerospace* 2017, 4(2), 22; UVLM with Katz / Joukowski / simplified-L-B load estimation, validated on the barnacle-goose / Robofly. Closest existing "UVLM + dynamic-stall" object; thrust treatment is weak and it isn't validated cross-J on a bird.
- **Robird** — ICAS 2022 paper 0653 "Investigation Flapping-Flight Aerodynamics of a Robotic Bird" + the unsteady-lifting-line + 2D3C-PIV validation: U=6 m/s, f=3 Hz, pitch 0–20°. Key empirical law for you: **cycle-mean lift depends on Strouhal number, not pitch amplitude; cycle-mean thrust depends on St AND pitch amplitude** — a clean cross-check target for your model's lift/thrust scaling.
- **Hirato, Shen, Gopalarathnam, Edwards 2019**, *J. Aircraft* 56(4):1626 "Vortex-Sheet Representation of LEV Shedding from Finite Wings" + **"Flow criticality governs LEV initiation on finite wings"** *JFM* — the proper 3D-UVLM-with-LESP precedent: max LESP at LEV initiation is ~independent of wing geometry and pivot. This is the rigorous basis for your per-strip LESP cap, and the right thing to cite/extend for a *finite-wing* (not strip-independent) LEV criterion.
- **Jin, Ji, Ravi, Young, Pereira & Tian 2025**, *J. Fluid Mech.* 1022 — "Lift increment scaling and its failure due to the LEV detachment transition for a flapping wing under perturbations" (IB-LBM, Re 2000 & 20000): lift increment scales linearly with perturbation α_eff **until** the LEV detachment mechanism switches from **bluff-body shedding → vorticity-layer eruption**, after which the scaling *fails*. This is the physical statement of your gap (2) ceiling and tells you the cap is not a single LESP value but a *regime transition*; relevant but at lower Re than yours.
- **Baldan & Guardone 2024**, arXiv:2401.14728 — CNN + physics-informed-loss ROM for NACA0012 deep dynamic stall (Re=1.35e5, your Re). Predicts full surface pressure/skin-friction over the cycle incl. CL_max-then-drop. Could serve as a sectional surrogate, but trained on pure pitching at k≤0.2 — your flapping α_eff swings to ~50° are far outside its training envelope, so not plug-and-play.

---

## What gets the AoA-dependence right vs the thrust trend right (your explicit question)

- **Right AoA-dependence (stall: CL_max-then-drop):** [A] Ayancik–Mulleners G-K (via X₀(α)), [H] Nakata CD/CL polynomial forms, [E] Polhamus (high-α vortex lift), [J] Jin et al. (the detachment-transition ceiling). G-K is the one you can *implement as an ODE per strip*.
- **Right thrust trend (grows with frequency):** [F] Gao passing-over LEV (position-dependent chordwise suction), [G] Fernandez-Feria/Gordillo (analytic T_mean(k) with C₁(k), ∝k³ LEV correction). None of these is packaged for a 3D bird wing — you'd port the *term*, not a code.
- **Right advance-ratio behavior:** [D] Kamimizu–Liu–Nakata (the v_body/v_flap/cross decomposition) — the only one purpose-built for your gap (3).
- **Nobody** gets all three simultaneously + validates on bird-scale lift AND thrust across f, U, AoA. That is the unoccupied cell.

---

## Concrete recommendation mapping to your 4 gaps

1. **α=0, +37%:** First test whether the offset is the **merge-impulse artifact** [C] (Darakananda–Eldredge impulse-matching correction) vs your convection-lag breaking symmetry. The principled cure for the lag-asymmetry: make the LEV **detach by state** (TE-reversal / max-Γ, [C]) so up- and down-stroke cores are created-and-shed symmetrically and cancel, instead of a persistent lagged core that nets +lift. Cross-check against the thin-airfoil symmetry theorem (symmetric strokes at 0° geometric AoA must give zero cycle-mean lift).
2. **α=15°, +12%:** Replace duration-driven growth with **G-K X-state gated by X₀(α)** [A] + the **detachment-transition ceiling** [J]. Drive stall by the delayed angle `α−τ₂α̇` so cruise's high *oscillatory* α_eff (fast α̇, large τ₂α̇ shift) does NOT trigger the static-stall cap that hurts your cruise — this is the formal fix to your "cannot distinguish base-AoA from flapping α" complaint.
3. **U=6, J~0.58, +18–22%:** Adopt the **v_body/v_flap/cross-coupling coefficient split** [D]; refit C_F,bd / C_F,fl / C_F,cp to your data so the effective coefficients vary with J instead of being frozen at your J~1 calibration.
4. **Thrust +4N constant, wrong freq slope:** Keep your **C_D=C_Dmin cos²α+C_Dmax sin²α** form drag [H]; ADD a **frequency-growing LEV chordwise-suction term** — the rotated/un-capped tangential projection of the LEV suction [E], made position-dependent per Gao [F], with analytic scaling cross-checked against Fernandez-Feria's C₁(k) [G]. AND audit the **normal-force→global-frame projection** through your stroke-plane tilt first [H], because a *constant* +4N offset is the signature of a steady bookkeeping/zero-offset error, not only a missing unsteady term.

---

Sources:
- [Ayancik & Mulleners 2022, JFM — generalised Goman–Khrabrov](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/all-you-need-is-time-to-generalise-the-gomankhrabrov-dynamic-stall-model/5DAFFF508F0811B159C53499531D1BDB) ([arXiv](https://arxiv.org/pdf/2110.08516))
- [AbuNawas & Qawasmeh 2026, J. Aircraft — Coupling Dynamic Stall with Lifting-Line Theory](https://arc.aiaa.org/doi/10.2514/1.C038542)
- [Hernandez Gelado & Ramesh 2022, AIAA — N-LEV reduced-order LDVM](https://arc.aiaa.org/doi/10.2514/6.2022-4105) ([arXiv](https://arxiv.org/abs/2206.11597))
- [Kamimizu, Liu & Nakata 2025, arXiv — Data-Driven QS model (advance-ratio)](https://arxiv.org/html/2508.18703v1)
- [Polhamus 1966, NASA TN D-3767 — leading-edge-suction analogy](https://ntrs.nasa.gov/api/citations/19680022518/downloads/19680022518.pdf)
- [Gao et al. 2019, AIP Advances — Passing-over LEV thrust booster](https://pubs.aip.org/aip/adv/article/9/3/035314/1077216/Passing-over-leading-edge-vortex-The-thrust)
- [Fernandez-Feria & Alaminos-Quesada 2018, JFM — unsteady thrust/lift with LEVs](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/unsteady-thrust-lift-and-moment-of-a-twodimensional-flapping-thin-airfoil-in-the-presence-of-leadingedge-vortices-a-first-approximation-from-linear-potential-theory/C912DC9D8543748F1ABA132CCDD87302)
- [Gordillo 2025, JFM — A note on the thrust of airfoils](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/note-on-the-thrust-of-airfoils/7FAFDD4D11149DED20D700A0D9475736)
- [Nakata, Liu & Bomphrey 2015, JFM — CFD-informed quasi-steady model](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/cfdinformed-quasisteady-model-of-flappingwing-aerodynamics/4A8293671D820CA68C9859423FFC68DF)
- [Jin, Ji, Ravi, Young, Pereira & Tian 2025, JFM — lift increment scaling & LEV detachment transition](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/lift-increment-scaling-and-its-failure-due-to-the-leadingedge-vortex-detachment-transition-for-a-flapping-wing-under-perturbations/CDB002EF442EC87768F6A3DEA7A2E62C)
- [Baldan & Guardone 2024, arXiv — DNN physics-based ROM for dynamic stall](https://arxiv.org/pdf/2401.14728)
- [Hirato, Shen, Gopalarathnam, Edwards 2019, J. Aircraft — vortex-sheet LEV shedding from finite wings](https://arc.aiaa.org/doi/10.2514/1.C035124)
- [Flow criticality governs LEV initiation on finite wings, JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/flow-criticality-governs-leadingedgevortex-initiation-on-finite-wings-in-unsteady-flow/F972838273C64F8DC9B9591B3404AC2D)
- [Vortex Lattice Simulations of Attached and Separated Flows around Flapping Wings, Aerospace 2017](https://mdpi.com/2226-4310/4/2/22/htm)
- [Investigation Flapping-Flight Aerodynamics of a Robotic Bird (Robird), ICAS 2022](https://www.icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0653_paper.pdf)
- [Ramesh et al. 2014, JFM — LESP discrete-vortex method (foundational)](https://eprints.gla.ac.uk/206131/7/206131.pdf)