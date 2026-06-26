I have everything needed. The Garrick leading-edge-suction decomposition (potential lift with zero LE-suction + vortex lift = "missing" suction rotated normal) directly addresses the user's thrust gap. The Fernandez-Feria / JFM linear-potential LEV thrust paper and the Leishman-Beddoes vortex-decay-after-peak both directly map onto gaps (4) and (2). I have the complete bookkeeping picture. Let me write the final synthesis.

---

# Sectional dynamic-stall ↔ 3D potential-flow hybrids: force-bookkeeping, and how it maps to your RoboEagle gaps

## 0. The one-sentence answer to your headline question

There are **three** force-bookkeeping conventions in the literature, and they are **mutually exclusive** — mixing them is exactly the double-counting trap:

| Convention | Where the LEV/stall lift lives | Bound circulation Γ | Used by |
|---|---|---|---|
| **(A) α-correction / decambering** | *Inside* a corrected Γ — the bound vorticity is bent so the panel reproduces the 2D **viscous** polar CL(α_eff). No separate force. | **Modified** (reduced at stall) | CAMRAD II (table mode), DUST / Proulx-Cabana NL-UVLM, Gallay-Laurendeau decambering, Kirchhoff-VLM |
| **(B) Indicial superposition** | A **separate additive increment** ΔC_N^v on top of the attached-flow normal force, with its own birth/convection/decay ODE. | **Unchanged** by the vortex; the increment is bolted on | Leishman-Beddoes, ONERA, CAMRAD II (L-B mode) |
| **(C) Impulse / total-vortex-system** | **Emergent** — force = rate of change of impulse of *all* vortices (bound + shed LEV + TEV). The LEV's lift appears automatically through its induced velocity on the bound sheet; shedding it *reduces* bound Γ. | **Reduced** when an LEV sheds (circulation is physically carried away) | Ramesh LDVM, Ansari-Zbikowski-Knowles, Katz-Plotkin DVM |

**Your current RoboEagle model is a fourth, non-standard hybrid** and that is the root of every gap: you compute the **attached** potential lift from a no-penetration solve (convention A boundary condition), then **add the LEV as a separate Bernoulli surface-pressure force that does NOT feed back into Γ** (a hand-built convention B), but you size that increment with a Garrick/LESP **leading-edge-suction** argument (convention C's physics). You took the *lift* projection of the LE-suction (C) but dropped its *thrust* projection, applied it additively (B) without the decay ODE that B requires, and bypassed the Γ-reduction that both A and C enforce. Each gap below is one specific seam of this Frankenstein.

---

## 1. Convention A — the α-iterative / decambering correction (the modern UVLM-strip standard)

This is the dominant convention for *exactly your problem class* (3D UVLM/lifting-line + 2D sectional viscous data on rotor/flapping blades), and it is the cleanest answer to "how do you add the sectional increment without double-counting."

**Canonical, open-access, equation-complete sources:**
- **Proulx-Cabana, V., Nguyen, M.T., Prothin, S., Michon, G., Laurendeau, E. (2022).** "A Hybrid Non-Linear Unsteady Vortex Lattice-Vortex Particle Method for Rotor Blades Aerodynamic Simulations." *Fluids* **7(2):81.* (open access) — this is the method the Politecnico/DUST and the later arXiv papers all defer to for the coupling.
- **arXiv:2511.11430** (2025, Politecnico di Milano group — Cocco/Savino lineage), "Nonlinear Unsteady Vortex-Lattice Vortex-Particle Method ... for Rotorcraft" — open HTML, gives the loop verbatim.
- **Cocco, A., Savino, A., Colli, A., Masarati, P., Zanotti, A. (2024).** "A non-linear unsteady vortex-lattice method for rotorcraft applications." *The Aeronautical Journal* **128(1328):2308–2330.**
- Foundational: **Parenteau & Laurendeau, "Unsteady Coupling Algorithm for Lifting-Line Methods"** (AIAA SciTech 2017); **Gallay & Laurendeau (2015/2016)** nonlinear VLM by decambering.

**The exact bookkeeping (verbatim from arXiv:2511.11430, their Section 2.1):**

The loop runs per chordwise strip:
1. Solve UVLM → bound Γ → inviscid section lift `C_L^inv`.
2. Effective angle of attack: `α_e = C_L^inv/(2π) − α_local + α_3D`
3. Look up the **viscous** `C_L^vis` from the 2D database (`.c81`) at `α_e`.
4. **Correct the boundary condition (the RHS angle), not the force:**
   `α_local ← α_local + ε (C_L^vis − C_L^inv/(2π))`, with relaxation `ε = 0.2`.
5. Iterate to `‖C_L^inv − C_L^vis‖ < 1e-3`.

**Why this cannot double-count:** the *only* lift in the model is Kutta-Joukowski on the bound rings, `F = ρ(Γ_i − Γ_{i−1}) u×r + ρ (dΓ/dt) A n` (their Eq. 8). Step 4 injects a Δα that *bends the bound circulation* until the panel's own KJ lift equals the viscous-table lift. Stall is represented as a **reduction of Γ** (the table CL rolls over, so the loop drives Γ down). There is never a "second" lift added. Drag and pitching moment, which the potential model *cannot* produce, are the *only* quantities added as separate sectional vectors (from table Cd, Cm). Dynamic stall enters by making the database an **unsteady** polar (or feeding `α_e` into a Leishman-Beddoes state model that returns the unsteady CL used in step 3).

**Direct implication for you:** in convention A there is **no question of "the LEV killing the overshoot via bound-reduction"** — the overshoot *is* the temporarily elevated unsteady-table CL, delivered *through* Γ. Your fear ("LEV in the no-penetration solve reduces bound and kills the overshoot") is an artifact of having split the model; in A the bound-reduction and the overshoot are the *same* mechanism handled consistently in time by the unsteady polar's hysteresis.

---

## 2. Convention B — Leishman-Beddoes indicial superposition (the rotor/wind-turbine workhorse)

**Sources:** Leishman & Beddoes (1989, *J. AHS*); the implementation/calibration review **Ayton/Pirrung et al. or rather** **"The Beddoes-Leishman dynamic stall model: critical aspects in implementation and calibration," *Renewable & Sustainable Energy Reviews* (2024)**; **Wayne Johnson, "Rotorcraft Aerodynamics Models for a Comprehensive Analysis" (CAMRAD II)** which implements **five** dynamic-stall models (Johnson, Boeing, Leishman-Beddoes, ONERA EDLIN, ONERA BH) selectable on top of a 2nd-order lifting-line + free wake.

**The bookkeeping (this is the part you half-reinvented):** total normal force is a **sum of three subsystems**:
`C_N = C_N^circ(attached, unsteady) + C_N^impulsive(apparent mass) + C_N^v(vortex)`.
The vortex increment is **explicitly separate and additive**, and critically it is **fed and bled by an ODE**:
- *Feed:* `dC_N^v/dt`'s source is the difference between attached circulatory lift and the Kirchhoff trailing-edge-separated lift: `C_v = C_N^circ (1 − ((1+√f)/2)²)`, where `f` is the Kirchhoff separation point.
- *Convection + decay:* `dC_N^v/dt = dC_v/dt − C_N^v/T_v`, with the vortex time constant `T_v` (and Beddoes' refinement: **`T_v` switches to accelerate the decay once the vortex passes the trailing edge / `τ_v > T_vl`**).

**This is the single most important fix for your Gap (2).** Your model "keeps GROWING with AoA because growth is stall-DURATION driven." That is precisely the symptom of having a vortex feed term **with no `−C_N^v/T_v` sink and no `τ_v > T_vl` cutoff.** L-B's `T_v` decay is *exactly* the "CL_max-then-drop" / detach-and-shed physics you're missing at 15°: the vortex lift rises while the LEV traverses the chord, then is **forcibly killed** when it convects off, so deeper/longer stall does **not** monotonically add lift — it triggers the collapse. Port the two L-B constants (`T_v ≈ 6`, `T_vl ≈ 5` in chord-time units, then recalibrate) onto your existing increment and the +12% at 15° should bend over.

---

## 3. Convention C — impulse / total-vortex-system (Ramesh LDVM, your actual physics ancestor)

**Sources:**
- **Ramesh, K., Gopalarathnam, A., Granlund, K., Ol, M.V., Edwards, J.R. (2014).** "Discrete-vortex method with novel shedding criterion for unsteady airfoil flows with intermittent leading-edge vortex shedding." *J. Fluid Mech.* **751:500–538.** — the LDVM/LESP paper your model descends from.
- **Ramesh et al. (2013/2018), "Theoretical modeling of leading edge vortices using the leading-edge suction parameter"** and **"Model Reduction in Discrete-Vortex Methods," *AIAA J.* (2018).**
- **Ansari, Żbikowski & Knowles (2006),** *Proc. IMechE Part G* **220:61–83 (Part 1)** and **220:169–186 (Part 2)** — nonlinear unsteady, LEV + TEV vortex sheets, force from impulse.

**The bookkeeping:** LESP `= A_0`, the **zeroth Fourier coefficient** of the bound vortex-sheet/camberline downwash distribution (thin-airfoil theory; Katz-Plotkin §). The criterion: **shed an LEV whenever `|A_0| > LESP_crit`**, releasing exactly enough LEV circulation to **drive `A_0` back to `LESP_crit`** each step (this *is* your "core circulation capped at U·c·sin α_crit"). Force is computed **two equivalent ways**, both over the *entire* vortex system:
- **Joukowski/impulse:** `F = ρ d/dt ∮ (x × γ) dΓ` over bound + all shed vortices — the LEV's lift is **emergent**, never added by hand.
- **Katz-Plotkin (Bernoulli/pressure):** unsteady Bernoulli over the bound sheet, where the shed LEV enters **through its induced velocity** in the `∂φ/∂t` and `½(...)` terms.

**The decisive point for your model's design choice:** in convention C, when an LEV sheds, the **bound circulation is reduced** (circulation is conserved; what goes into the LEV leaves the bound sheet). The LEV's lift then shows up *as* the induced effect of that shed vorticity sitting above the airfoil — **it is not double-counted because it was subtracted from the bound sheet first.** Your choice to put the LEV "in the Bernoulli force ONLY, not the no-penetration solve, to avoid bound-reduction killing the overshoot" is therefore a **departure from C's conservation law.** It works at cruise (you tuned it there) but it is *non-conservative*, and that non-conservation is what leaks +37% at α=0 and breaks frequency trends — you are adding a vortex force whose reaction was never taken out of the bound sheet.

---

## 4. Now, gap-by-gap — what specific literature mechanism fixes each

### Gap (1): α=0°, +37%, the up/down LEV should cancel but your convection-delay lag breaks symmetry

The lag itself is legitimate (it is L-B's `T_v` convection, §2). The problem is that you implemented the **feed** as a lag but not the **conservation**. Two literature-grounded fixes:

- **Make it convention-C-consistent (preferred):** when the LEV sheds on the downstroke and again (opposite sign) on the upstroke, each must **debit the bound sheet** with the same time-lag. If both the +LEV force *and* its bound-circulation debit carry the *same* `T_v` lag, the net over a symmetric cycle returns to ~0 by construction (the impulse of a symmetric shed-vortex pair integrates to zero net vertical force). Your asymmetry is purely because the **credit (force) is lagged but the debit (bound reduction) is absent/instant.** Symmetrize the bookkeeping and α=0 self-cancels — this is the LDVM/impulse guarantee.
- **L-B alternative:** L-B's vortex increment is driven by `dC_v/dt` and is **sign-following** with a *symmetric* `T_v`; on a symmetric oscillation the positive and negative vortex contributions cancel in the mean *because both the feed and the decay are odd in the half-cycle*. Adopt L-B's `dC_N^v/dt = dC_v/dt − C_N^v/T_v` form verbatim and the spurious mean vanishes; your current "2nd-order convection-delay" likely only delays the rise, not the fall.

There is no clean closed-form "α=0 correction" — the honest fix is structural: the lag must apply to a *conserved* quantity, not to a free-floating force.

### Gap (2): α=15°, +12%, increment peaks at ~10° then drops; yours keeps growing

This is the **missing `T_v` decay + `τ_v > T_vl` chord-passage cutoff** (§2), and the **deep-stall vortex-detachment** literature:
- L-B critical-aspects review (*Ren. Sust. Energy Rev.* 2024) and **Mohamed et al., *Wind Energy* (2021), "Modeling dynamic loads on oscillating airfoils with emphasis on dynamic stall vortices"** — both document that the lift *collapse* after CL_max is governed entirely by the vortex **convection/decay time constant**, not by stall strength. A sudden CL drop at ~18° is the canonical signature.
- The key conceptual move you need: **separate the LEV "strength" (LESP-capped, fine for cruise) from the LEV "fate" (born → traverse → detach → collapse).** Your growth is stall-duration driven because you have birth+traverse but no detach. Add the detachment trigger (vortex centroid passes TE, or `τ_v > T_vl`) → force the increment to decay → you recover CL_max-then-drop. This *also* resolves your stated dilemma ("static-stall caps hurt cruise"): you do **not** cap the instantaneous force; you cap the **vortex lifetime**. Cruise's high instantaneous α_eff≈50° is *transient* (the vortex never completes a chord traverse before kinematics reverse), so the lifetime-cutoff barely fires at cruise but fires hard at sustained 15° base AoA. **This is the single cleanest distinguisher between base-AoA stall and oscillatory-α you asked for** — it is a *time-domain* discriminator (dwell time at high α), not an instantaneous-α discriminator.

### Gap (3): U=6 m/s high-freq, +18-22%, J≈0.58 flapping-dominated vs J≈1 cruise calibration

This is an **advance-ratio / reduced-frequency consistency** problem, and it is the textbook failure mode of holding `T_v`, `LESP_crit`, and the lag constant fixed across `k`. In all three conventions the vortex constants are **functions of reduced frequency / Mach**, and at low J the reduced frequency `k = πfc/U` roughly **doubles**:
- L-B `T_v` and the attached-flow indicial constants are calibrated vs `k`; CAMRAD II's tables carry `k`- and Mach-dependent stall-delay factors. The Politecnico/Proulx-Cabana coupling (§1) handles this *automatically* because the unsteady part is the genuine `dΓ/dt` Bernoulli term, which scales correctly with `k` **only if Γ is the conserved bound circulation** — which yours is not for the LEV part.
- Honest assessment: there is **no single fixed constant set valid from J≈1 to J≈0.58** in a lift-only-LEV model. Either (a) make `T_v` and the lag `k`-dependent (interpolate the two operating points — defensible, this is standard L-B practice), or (b) fix the conservation (§3) so the unsteady scaling becomes physical and the `k`-dependence falls out of `dΓ/dt` naturally. Option (b) is the principled one; (a) is what a referee will accept as "engineering calibration."

### Gap (4): THRUST, constant +4N offset AND opposite frequency trend — the big one

Two **distinct** physics gaps, both named in the literature:

**(4a) The constant offset ≈ separated-flow form/pressure drag.** You already located half as `Cd ~ sin²α` pressure drag. This is exactly **DeLaurier's post-stall / friction bookkeeping** and the **Viterna-Corrigan** flat-plate post-stall extrapolation used in every BEM+dynamic-stall code:
- **DeLaurier (1993), "An aerodynamic model for flapping-wing flight," *Aeronautical Journal*** — sectional friction drag `dD_f = ½ρ V_x² C_df c dy`, plus a post-stall normal-force regime (he sets `C_D,post-stall ≈ 1.98 ≈ 2`, i.e. full flat-plate `Cd=2 sin²α` once `α'` exceeds `α_stall≈20°`). Your `Cd~sin²α` is the linearization of this. **Add the flat-plate post-stall `Cd≈1.9–2.0·sin²α` ramp (Viterna-Corrigan) above the sectional stall angle** and the remaining half of the +4N pressure drag appears — this is the standard, non-fabricated closure.

**(4b) The opposite frequency trend = missing flapping-propulsion / LEV vortex thrust.** This is the **Garrick leading-edge-suction *thrust* projection you deliberately dropped**, and it is the crux:
- **Garrick, I.E. (1936), NACA Report 567, "Propulsion of a flapping and oscillating airfoil"** — the foundational result: a heaving/pitching airfoil's **leading-edge suction force points FORWARD (thrust)**, scaling as `~ f²` (∝ reduced-frequency², i.e. grows with flapping speed). This is *literally* the `freph²` propulsion growth your measured net-drag-decrease-with-freq implies.
- **The decomposition you need (from the LEV-suction literature, e.g. Polhamus analogy as used by DeLaurier/Ansari):** lift = (potential lift with **zero** LE-suction) + (vortex lift = the **"missing" LE-suction rotated to act normal to the chord**). **You took only the normal/lift rotation of the missing suction and threw away its chordwise component.** But the *attached* LE-suction (before it is lost to the LEV) is **chordwise-forward = thrust.** Your statement "Bernoulli x-force ≈ 0; LE-suction is LESP-capped; LEV thrust not taken" is the precise bug: capping `A_0` at `LESP_crit` caps the suction *magnitude* but you must still **apply the capped suction as a forward force**, not zero it.
- **Modern closed-form to lift directly:** **Fernandez-Feria, R. (and Alaminos-Quesada) — "Unsteady thrust, lift and moment of a two-dimensional flapping thin airfoil in the presence of leading-edge vortices: a first approximation from linear potential theory," *J. Fluid Mech.* (2021).** This gives explicit linear-potential expressions for the **thrust contribution of LEVs** on a flapping foil — i.e. a ready-made `ΔT_LEV(t)` term you can graft on, derived consistently so it does not double-count the Garrick suction. Also **"Passing-over leading-edge vortex: the thrust booster in heaving airfoil," *AIP Advances* 9:035314 (2019)** quantifies how the convecting LEV *adds* thrust as it passes over the surface — directly the `f²` trend.

**Bottom line for (4):** add (i) the **forward LE-suction force** `dT_s = 2πη_s (α_eff-terms) ρ U V c dy` (DeLaurier Eq. 31 form — note it is **+thrust**, and η_s<1 is partial-suction efficiency), capped in *magnitude* by LESP but **not zeroed**; plus (ii) the **LEV vortex-thrust** chordwise projection (Fernandez-Feria term); plus (iii) the **post-stall flat-plate pressure drag** (Viterna/DeLaurier). Term (i)+(ii) grow ∝f² (fixes the trend); term (iii) supplies the constant offset. Your model currently has only a tiny piece of (iii) and **none** of (i)/(ii) — hence net thrust where there should be net drag at low f, and no growth with f.

---

## 5. Honest "no-clean-answer" caveats

- **Distinguishing base-AoA stall from oscillatory-α stall has no instantaneous-α solution** (you found this empirically). The *only* clean discriminator in the literature is **time-domain**: the dynamic-stall vortex lifetime / dwell-time-above-critical (`τ_v` vs `T_vl`). That is a genuine state, not a static cap. Accept the state variable or accept the error — there is no static-table fix.
- **A single fixed constant set across J≈1→0.58 is not supported by any of these models.** L-B, ONERA, and CAMRAD all carry `k`/Mach-dependent constants. Your "one fixed constant set" success at cruise is real but does not extrapolate; the field's honest answer is `k`-scheduled constants *or* a conservative (impulse-consistent) formulation that makes the scaling physical.
- **Lift-only LEV models are thermodynamically inconsistent for thrust.** Every code that gets flapping *thrust* right (DeLaurier, Garrick, Fernandez-Feria, Ansari-Z-K) computes the **chordwise** force from the **same** circulation/suction that produces lift. A model that takes the lift projection of the LEV but not the thrust projection *cannot* close the force balance — this is structural, not a tuning issue. Gap (4)'s trend is unfixable without a chordwise LEV/suction term.
- **The cleanest single refactor** that dissolves Gaps 1, 3, and half of 4 at once is to **move your LEV from "added Bernoulli force" (non-conservative B-hybrid) to the conventional convention A or C**: let the LEV **reduce the bound circulation** (LESP→A_0 cap drives the shed strength), compute *all* forces (lift **and** chordwise thrust) from the resulting total vortex system via unsteady Bernoulli/impulse. Then α=0 cancels by symmetry, thrust grows ∝f² automatically, and `k`-scaling is physical. The cost: you lose the convenient "LEV doesn't touch the solve" decoupling — but that decoupling is the source of the leaks. Gap 2 still additionally needs the **`T_v` decay + chord-passage cutoff** regardless of which convention you pick.

---

## Key references (authors, year, venue) — ranked by directness to your fix

**Tier 1 — port these mechanisms directly:**
1. **Ramesh, Gopalarathnam, Granlund, Ol, Edwards (2014),** *J. Fluid Mech.* 751:500–538 — LDVM/LESP, `A_0` cap, impulse force over total vortex system. *Your ancestor; gives the conservative bookkeeping you departed from.*
2. **Leishman & Beddoes (1989),** *J. Am. Helicopter Soc.* 34(3) + review *Ren. Sust. Energy Rev.* (2024) — the `C_N^v` feed/decay ODE with `T_v`, `T_vl`. *Fixes Gap 2 (the drop at 15°) and Gap 1 (symmetric cancellation).*
3. **DeLaurier (1993),** *Aeronautical Journal* 97:125–130 — strip-theory flapping model: partial LE-suction `dT_s` (forward thrust, Eq. 31), post-stall flat-plate `C_D≈1.98`, friction `dD_f`. *Fixes Gap 4a + supplies the +thrust term for 4b.*
4. **Garrick (1936),** NACA Report 567 — flapping LE-suction thrust ∝f². *The missing frequency trend in Gap 4b.*
5. **Fernandez-Feria & Alaminos-Quesada (2021),** *J. Fluid Mech.* — closed-form **LEV thrust/lift/moment** for a flapping thin airfoil, linear potential. *Ready-made ΔT_LEV(t) for Gap 4b without double-counting.*

**Tier 2 — the clean hybrid bookkeeping templates (convention A):**
6. **Proulx-Cabana, Nguyen, Prothin, Michon, Laurendeau (2022),** *Fluids* 7(2):81 (open access) — α-iterative UVLM viscous coupling, full equations.
7. **Cocco, Savino, Colli, Masarati, Zanotti (2024),** *Aeronautical Journal* 128(1328):2308–2330; and **arXiv:2511.11430 (2025)** — the same coupling, rotorcraft, with the verbatim `α_e` and Δα-relaxation loop.
8. **Wayne Johnson, "Rotorcraft Aerodynamics Models for a Comprehensive Analysis" (CAMRAD II)** — 2nd-order lifting-line + free wake + **5 selectable dynamic-stall models** as separable increments; the reference for convention-B-on-top-of-wake bookkeeping.

**Tier 3 — supporting / nonlinear-VLM stall:**
9. **Ansari, Żbikowski, Knowles (2006),** *Proc. IMechE Part G* 220:61–83 & 169–186 — nonlinear unsteady LEV+TEV, impulse force, hover MAV.
10. **Gallay & Laurendeau (2015),** *AIAA J.* — nonlinear VLM by decambering; **"Lift Prediction Including Stall, Using VLM with Kirchhoff-Based Correction," *J. Aircraft* (2017)** — stall via Kirchhoff `f` modifying the influence coefficients (convention A, boundary-condition form).
11. **Nakata, Liu, Bomphrey (2015),** CFD-informed quasi-steady flapping model — bird/insect-scale LEV lift+thrust closure.

**Sources:**
- [Ramesh et al. 2014, discrete-vortex with shedding criterion (ResearchGate)](https://www.researchgate.net/publication/264119736_Discrete-vortex_method_with_novel_shedding_criterion_for_unsteady_airfoil_flows_with_intermittent_leading-edge_vortex_shedding)
- [Ramesh et al., LESP theoretical modeling (Glasgow Enlighten)](https://eprints.gla.ac.uk/99367/)
- [Proulx-Cabana et al. 2022, Hybrid NL-UVLM-VPM, Fluids 7(2):81](https://www.mdpi.com/2311-5521/7/2/81) · [HAL mirror](https://hal.science/hal-04067885/)
- [arXiv:2511.11430, Nonlinear UVLM-VPM for rotorcraft (HTML)](https://arxiv.org/html/2511.11430v1)
- [Cocco et al. 2024, non-linear unsteady VLM for rotorcraft, Aeronautical Journal](https://www.cambridge.org/core/product/724D43C209EC3CE4F7E3692A303940B7)
- [Wayne Johnson, CAMRAD II aerodynamics models (PDF)](http://johnson-aeronautics.com/documents/CIIaerodynamics.pdf) · [CAMRAD II papers index](http://johnson-aeronautics.com/CAMRADIIpapers.html)
- [Leishman-Beddoes critical-aspects review, Ren. Sust. Energy Rev. 2024](https://www.sciencedirect.com/science/article/pii/S1364032124004039)
- [DeLaurier aerodynamic model for flapping-wing flight, Aeronautical Journal](https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/an-aerodynamic-model-for-flappingwing-flight/D748A705C93C0DBECF0C94600E85F644)
- [DeLaurier model equations reproduced (IJAAE)](https://vibgyorpublishers.org/content/ijaae/fulltext.php?aid=ijaae-3-017)
- [Garrick LE-suction / LEV thrust decomposition overview (ScienceDirect topic)](https://www.sciencedirect.com/topics/engineering/leading-edge-vortex)
- [Fernandez-Feria, unsteady thrust/lift of flapping airfoil with LEVs, JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/unsteady-thrust-lift-and-moment-of-a-twodimensional-flapping-thin-airfoil-in-the-presence-of-leadingedge-vortices-a-first-approximation-from-linear-potential-theory/C912DC9D8543748F1ABA132CCDD87302)
- [Passing-over LEV: thrust booster in heaving airfoil, AIP Advances 9:035314](https://pubs.aip.org/aip/adv/article/9/3/035314/1077216/Passing-over-leading-edge-vortex-The-thrust)
- [Ansari-Zbikowski-Knowles LEV & MAV aerodynamics (Aeronautical Journal)](https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/leadingedge-vortex-and-aerodynamics-of-insectbased-flappingwing-micro-air-vehicles/D206274C80C2815EAE5DA944984BEE38)
- [Kirchhoff-based correction VLM stall prediction, J. Aircraft](https://arc.aiaa.org/doi/10.2514/1.C034451)
- [Nonlinear VLM for wind-turbine stall (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0960148118310371)
- [Mohamed et al. 2021, dynamic-stall vortex decay modeling, Wind Energy](https://onlinelibrary.wiley.com/doi/10.1002/we.2627)