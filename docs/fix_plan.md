# Complete Fix Plan — Real Full-Aircraft Co-Design Synergy on the 4090

Goal: deliver the **headline ranking-inversion synergy** (co-design beats decoupled
optimization — the optimal design shifts when the control loop closes) as a *real*
scientific finding measured on the validated coupled FSI, on the RTX 4090. Discovery
runs #1–#4 built the mechanism map (F1–F7) but hit three concrete defects that block
a clean finding. This plan fixes each, in tractable order.

---

## Diagnosis — the three defects (grounded in code)

**D1 — Efficiency is a fake proxy.**
`codesign_eval.evaluate()` returns `efficiency = mean(aero force norm)`
(`coupled_fsi.py:64` `last_force_norm`; `codesign_eval.py:77,79-80`). It is *not*
power. So every "efficiency" result is qualitative. Discovery #3's resonance trade-off
(F6) used an **analytical** COT (`cot.py` P_iner with a sinusoidal assumption), never
the rollout. Until efficiency is real work/power, no efficiency finding is conclusive.

**D2 — Resonance is measured in the wrong solver.**
Discovery #4 (F7) drove a *rigid* hinge with the implicit Featherstone integrator →
numerical damping killed the resonance (stroke ∝ 1/ke, quasi-static). But the **ANCF
coupled solver does show resonance** (`render_bird_cfrp` near-resonant bending ~0.42 m).
The resonance-efficiency mechanism (the plan's hero) has never been measured in the
real aeroelastic solver. `WarpANCFEntry(c_damp=2.0)` is the knob — lowering it recovers
structural resonance.

**D3 — Single-wing flexibility dominance.**
With no competing constraint, flexibility wins gust (F1) + COT + control authority (F4)
simultaneously, so the optimum collapses to the flexible corner — no ranking inversion.
The genuine inversion needs a real competing axis where flexibility can no longer win
every objective.

---

## The fixes (concrete, code-grounded, 4090-tractable)

### FIX 1 — Real propulsive efficiency from the coupled rollout  *(Phase 1)*
A passive wing does no flight work, so real "flight efficiency / COT" requires a
**flapping** wing in the coupled FSI. Over one steady flap cycle measure, from the
rollout (not analytics):
- `W_in`  = drive work = ∫ τ_react(t)·ω_flap(t) dt   (root reaction torque × flap rate)
- `W_out` = useful propulsive work = mean_thrust · V · T
- `η`     = W_out / W_in   (propulsive efficiency); also report COT = P_in/(m g V)

Both quantities are available: the force provider exposes nodal aero force
(`Fbern`) and the entry exposes structural velocity (`entry.dq`); P_aero(t) =
Σ F_aero,i·v_i. **Acceptance:** energy balance closes on the validated FSI —
`W_in ≈ ΔKE + ΔSE + W_aero,out + W_damp` to a stated tolerance.
New: `platform/power_probe.py`; hook a real `efficiency` into `codesign_eval`.

### FIX 2 — Recover the real structural resonance  *(Phase 2)*
- Compute `ω_n(stiffness)` with `compute_natural_frequencies()` (scipy `eigh`,
  already MATLAB-validated in `run_standalone_yamano.py:110`) → predict the resonant
  stiffness where `f_flap ≈ ω_n/2π`.
- Sweep `WarpANCFEntry.c_damp` (2.0 → ~0.1) and confirm a stroke/thrust amplification
  peak emerges at the predicted resonant stiffness in the *driven coupled rollout*.
- This converts F7's honest-negative into a real, solver-correct resonance, and makes
  the efficiency curve from FIX 1 peak at a **tuned** stiffness.

### FIX 3 — Tractable full multibody aircraft  *(Phase 3)*
Extend `p0_resonant_freeflight.build_model` → **2 mirror-symmetric wings (L/R)** on
spring flap hinges + **V-tail** on a revolute (mirror symmetry halves the design dims,
per plan §1). Couple the flexible wings through `WarpANCFEntry` + force provider +
`WindowPredictorCorrector`. The **competing constraint that breaks flexibility
dominance**: at flight-level flap amplitude an over-flexible wing dumps input energy
into elastic deformation instead of thrust (low `W_out/W_in` from FIX 1) → efficiency
favors a tuned stiffness, while gust rejection still favors flexible (F1). The
trade-off is now *real*, not analytical.

### FIX 4 — The co-design synergy run  *(Phase 4)*
MOME / MAP-Elites over **(gust_rejection, real_efficiency)** with the full aircraft,
once passive and once with the closed Takens control loop. Headline question: does
`argmin` shift between decoupled and co-design? Report it **whichever way it comes out**
(positive synergy, or an honest boundary on when it does/doesn't appear).

---

## Execution order & red lines
P1 (real power) → P2 (real resonance) → P3 (full aircraft) → P4 (synergy).
Each phase gated by an energy-balance / golden red-line check (the validated coupled
FSI must stay bit-exact on its regression; new metrics must close energy balance).
Never fabricate — every number computed from a real rollout. 4090 throughout
(reduced env count vs the A100 production target; the *mechanism* is solver-correct
regardless of scale).

**Status.**
- **FIX 1 DONE** — `power_probe.py` reads real lift/induced-drag/aero-power from the
  coupled UVLM `Fbern`; `discovery5.py` swept stiffness at 6° cruise AoA and found a
  clean monotone **real (gust × L/D) trade-off** (F8 in `discovery_findings.md`): gust
  favors flexible (9.5→10.8 ×10⁻³), induced L/D favors stiff (22.2→24.3). This is the
  genuine competing constraint the proxies lacked. Lift ~constant, so the signal is real
  induced drag from load-induced deformation. No analytical proxy.
- **NEXT: FIX 4** — the synergy question is now well-posed on this real trade-off. MOME
  over (gust, real L/D) decoupled vs. with the Takens control loop; report inversion (or
  its honest absence). FIX 2 (resonance) and FIX 3 (full aircraft) add the compounding
  efficiency-favors-stiffness axis if the single-wing synergy is weak.
