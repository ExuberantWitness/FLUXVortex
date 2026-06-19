# Aerodynamic modeling literature assessment — flapping MAV at high body AoA

Research question (user 调研 via research-pipeline): does the literature support the
hybrid modeling **main wing on attached UVLM (flapping reduces effective AoA) + LEV
(LESP-LDVM) only on the leading-edge control surfaces**, at the MAV's ~45° body angle
of attack? Honest boundary required.

## Verdict: SUPPORTED, with one explicit boundary condition

The three modeling choices are each backed by established literature; the only caveat is
that the "main wing stays attached" assumption is **conditional on sufficient flapping**
(plunge velocity must keep the local effective AoA below the shallow-stall limit ~14°).

### Q1 — Flapping reduces the wing's *effective* AoA (main wing attached)  ✅ (conditional)
- Effective AoA of a plunging/flapping section: `α_eff = α_geom − atan(ḣ/U)` — set by the
  static pitch **plus** the apparent inflow from the vertical (plunge) velocity. The body
  geometric AoA is **not** the section AoA.
- Pitched/plunging airfoils at **shallow-stall kinematics (α_eff ≈ 2.4°–13.6°) remain
  attached**; beyond a critical effective incidence the LEV lifts off the surface and lift
  breaks down (separation). Stall onset reported near reduced-amplitude `kh ≈ 0.35`.
- **Boundary (honest):** the main wing stays attached **only if the flapping keeps α_eff
  below ~14°**. At 45° body AoA this requires enough plunge velocity (amplitude × freq).
  If flapping is weak — e.g. the *unpowered* high-AoA perching climb, where large birds
  are observed **not to flap** — the wing **will** separate. → The platform must verify the
  effective-AoA distribution stays < ~14° at the operating flapping.

### Q2 — LEV only at the leading-edge control surfaces  ✅ (strongly supported)
- **LESP-LDVM** (Ramesh, Gopalarathnam, Granlund, Ol & Edwards, *JFM* 751, 2014, 500–538):
  the leading-edge suction parameter (nondimensional LE suction) crosses a single critical
  value → intermittent LEV shedding initiates/terminates. This is exactly the ported method
  (`lev_dvm.py`).
- **Biological analog = the alula:** a leading-edge feather device covering ~5–20% span that
  generates a tip vortex, suppresses separation and **delays stall / enhances lift at high
  AoA**, used in slow flight and landing — "comparable to traditional slats / vortex
  generators." Engineering version: **LEAD (Leading-Edge Alula-inspired Device)**, Ito & Duan
  et al. (Illinois, SMASIS 2018). → The aircraft's deflectable **LE control surfaces are the
  alula analog**: the device that controls the LEV at high AoA. Putting the LDVM/LEV there is
  directly biomimetically and aerodynamically justified.

### Q3 — Hybrid UVLM (attached) + sectional separation/LEV  ✅ (strongly supported)
- This is the established **non-linear UVLM (NL-VLM)** paradigm. Hybrid NL-UVLM–VPM for rotor
  blades (Proietti et al., *Fluids* 7(2):81, 2022): *"viscous effects such as separation are
  entirely contained within the higher-fidelity [stripwise 2D] database, and the role of the
  3D VLM is simply to find the local effective angle of attack."* — precisely our architecture:
  3D UVLM gives the local effective AoA / induced field, a **sectional** model carries the
  separation/LEV.
- **Extended UVLM for insect flapping wings** (Nguyen et al., *J. Aircraft*, 10.2514/1.C033456):
  leading-edge-suction analogy + vortex-core-growth incorporated into conventional UVLM for the
  LE effects. UVLM is explicitly a *"medium-fidelity tool for non-stationary loads in low-speed,
  high-Re, **attached-flow** conditions"* — valid for the main wing, not for separated flow
  (hence the sectional LDVM where separation occurs).

### Q4 — High-AoA flapping flight, biological/engineering reference  ✅
- Birds reach high AoA in perching (rapid pitch-up <0.2 s); the **alula** is deployed for the
  high-AoA/slow-flight regime. Powered dive uses flapping for thrust; the agile unpowered climb
  often does **not** flap (→ the separation boundary above).
- **HIT-Hawk** (Zhong & Xu, *Appl. Sci.* 12(6):3176, 2022 — the sizing reference): 1.6/1.8 m
  span FWAVs, wind-tunnel power model correlating power with **AoA**, flapping frequency and
  inflow velocity; the lift varies with flapping angle through the cycle (not a fixed
  lift = weight) — consistent with the effective-AoA-through-the-stroke picture.

## Implication for the platform (honest)
1. The modeling (UVLM main wing + LDVM LE surfaces) is **literature-grounded and standard**
   (NL-VLM viscous-inviscid stripwise hybrid; alula-analog LE LEV control).
2. **Required check:** compute the main-wing **effective AoA** under the operating flapping and
   confirm it stays below the shallow-stall limit (~14°). The earlier blow-up (L/W=18 at 45°)
   was a **static** test (no flapping) — not the operating condition; with the flapping plunge
   velocity the section AoA is much lower. This is the concrete validation to add.

## Key references
- Ramesh, Gopalarathnam, Granlund, Ol, Edwards. *Discrete-vortex method with novel shedding
  criterion for unsteady aerofoil flows with intermittent leading-edge vortex shedding.* JFM
  751 (2014) 500–538.
- Proietti et al. *A Hybrid Non-Linear Unsteady Vortex Lattice–Vortex Particle Method for Rotor
  Blades.* Fluids 7(2):81 (2022).
- Nguyen et al. *Extended Unsteady Vortex-Lattice Method for Insect Flapping Wings.* J. Aircraft
  (10.2514/1.C033456).
- Ito, Duan et al. *A Leading-Edge Alula-Inspired Device (LEAD) for Stall Mitigation and Lift
  Enhancement.* SMASIS 2018; alula function studies (Sci. Rep. 5:9914).
- Zhong & Xu. *Power Modeling and Experiment Study of Large Flapping-Wing Flying Robot during
  Forward Flight.* Appl. Sci. 12(6):3176 (2022).
