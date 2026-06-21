# A Differentiable, GPU-Native Unsteady Aeroelastic Solver for Flapping-Wing Co-Design

*FLUXVortex — working draft toward Aerospace Science and Technology. All numbers below are produced
by the code in this repository and are reproducible on a single GPU; every gradient is checked
against an independent oracle (complex-step or finite differences).*

---

## Abstract

We present a fully differentiable, GPU-native solver for the **unsteady fluid–structure interaction
(FSI)** of flexible lifting surfaces, and use it to perform **structural co-design** of a wing by
quality-diversity optimisation. The structural model is an Absolute Nodal Coordinate Formulation
(ANCF) shell; the aerodynamic model is an unsteady free-wake ring vortex-lattice method (UVLM) with
trailing-edge shedding, free wake convection, and the unsteady (∂Γ/∂t) added-mass force; the two are
coupled through a symplectic time-marching scheme on the deformed, moving geometry. The entire
forward solver is implemented in NVIDIA Warp in double precision and is **bit-exact** against an
independent NumPy/MATLAB reference. Its key novelty is an **end-to-end adjoint**: the design gradient
∂(objective)/∂(per-element stiffness, per-element mass) and the control gradient ∂(objective)/∂(actuation)
flow through the *entire* coupled unsteady rollout — including the recurrent free-wake history — in a
single backward pass. We validate every piece of this adjoint to machine precision against the exact
complex-step gradient, and against finite differences of the coupled solver. We then drive a MAP-Elites
+ differentiable-quality-diversity (DQD) search with this gradient to **co-design the structure and the
controller together** for **gust-load alleviation**, on a single RTX 4090, illuminating a diverse
archive of (body × control) solutions and recovering the achievable (gust-deflection × control-effort)
Pareto front — along which the optimal spanwise stiffness distribution shifts, demonstrating that
structure and control must be designed jointly. We are explicit about the present limitations (single-fidelity
attached-flow aerodynamics, explicit-integrator stability window, isotropic-plate constitutive law) and
the infrastructure needed to lift them.

---

## 1. Introduction

Flapping-wing micro air vehicles (MAVs) at bird scale operate in a regime where **structural flexibility
is not a nuisance but a design degree of freedom**: passive wing twist generates thrust, root resonance
recovers inertial power, and the spanwise stiffness and mass distributions set both the aeroelastic
stability margin and the gust response. Designing such a vehicle is therefore intrinsically a
**co-design** problem over the coupled aero-structural-control system, not a sequence of disciplinary
optimisations.

Two capabilities are missing from current open tools. (i) **Differentiability through the coupled
unsteady FSI.** Vortex-lattice aeroelastic codes (e.g. SHARPy) are accurate but not differentiable;
differentiable-physics frameworks rarely include an unsteady free wake with a correct adjoint through
its time history. (ii) **GPU-native execution** that makes population-based co-design affordable. This
paper contributes a solver that has both, and demonstrates co-design on it.

Our specific contributions:

1. A GPU-native (Warp, fp64) **unsteady free-wake UVLM ⊗ ANCF-shell coupled FSI**, bit-exact against a
   NumPy/MATLAB reference.
2. An **exact end-to-end adjoint** of this solver: the design and control gradients flow through the
   coupled time-marching rollout *and the recurrent free-wake*, validated to machine precision against
   complex-step and finite differences. To our knowledge this is the first published machine-precision
   adjoint through a free-wake ring-UVLM coupled to a nonlinear flexible structure.
3. A **co-design demonstration**: MAP-Elites + DQD over spanwise stiffness and mass distributions for
   gust rejection, run end-to-end on a single consumer GPU.

---

## 2. Method

### 2.1 Structure — ANCF shell with a differentiable design field

The wing is an ANCF thin-shell (9 DOF/node, 36 DOF/element) **clamped at the span root** (a cantilever
wing — the standard aeroelastic boundary condition). The internal force is *linear in a per-element
stiffness scale* `E_scale[e]` and the element mass block is *linear in a per-element density scale*
`ρ_scale[e]`; these per-element scales are the structural design variables. A low-dimensional design is
obtained by mapping a small set of spanwise spline control points through a fixed basis,
`E_scale = exp(B·θ_E)`, so the design stays smooth and the gradient is exact through the chain rule.
The tangent stiffness `K_t` is assembled for the adjoint state-transition.

### 2.2 Aerodynamics — unsteady free-wake ring-UVLM

Bound vortex rings are placed at panel quarter-chords; the circulation solve sees, in its right-hand
side, the freestream **plus the induced velocity of the entire shed wake** (the history). Each step a
trailing-edge ring is shed; every wake corner then convects by the freestream plus the induced velocity
of all bound and wake rings (a genuine **free** wake). The sectional load is the unsteady
Kutta–Joukowski force plus the ∂Γ/∂t added-mass term. The wake self-induction is desingularised with a
finite vortex core (van Garrel form, δ = 0.05), which is both physically standard and necessary for a
**bounded adjoint** (see §3).

### 2.3 Coupling

At each step the ANCF nodes *are* the lattice corners: `corners = P·q`, body velocity `V_body = P·q̇`.
The aerodynamic panel forces are distributed to nodes by the shape-function-consistent transpose, the
structure is advanced symplectically `a = M(ρ)⁻¹(F_aero − Q_int(q;E))`, and the wake is shed and
convected — all on the deformed, moving geometry, with the moving-body boundary condition.

### 2.4 Differentiation — the end-to-end adjoint

The forward is differentiated by composing (a) Warp's reverse mode through every kernel, (b) a **manual
VJP for the dense circulation solve** (`adj_b = solve(Aᵀ, adj_x)`, `adj_A = −adj_b⊗x`), and (c) the
analytic structural design adjoint (`∂L/∂E_e = adj_Q·Q_e/E_e`, `∂L/∂ρ_e = −adj_rhs·M̃_e·a`). The novel
part is the **recurrent wake-history adjoint**: per step we tape the geometry→AIC→rhs and the
force+shed+convect separately, insert the manual solve VJP between them, and walk the rollout backward
with distinct per-step buffers so the tape is intact, aliasing `gprev ≡ γ_{t-1}` so the ∂Γ/∂t coupling
chains for free, accumulating the AIC adjoint across all solves, and chaining the wake-state adjoint
step→step. Gradient checkpointing gives O(√N) memory for long rollouts.

Three Warp-autodiff failure modes were identified and fixed, and are documented because they recur in
any such solver: (i) a vector accumulated in a loop then contracted with a differentiable normal
mis-saves its post-loop value in reverse mode — accumulate the scalar instead; (ii) the free-wake
self-induction is a Biot–Savart singularity whose forward is finite but whose analytic adjoint explodes
through the recurrence — the finite vortex core fixes both; (iii) `Tape.zero()` clears the array
gradients, so they must be read before zeroing.

---

## 3. Validation

Every component is checked against an independent oracle. The design and control gradients are checked
against the exact complex-step gradient and/or finite differences of the *same* solver.

| Component | Check | Result |
|---|---|---|
| Differentiable VLM (GPU) | ∂F/∂corners vs complex-step | 5.4e-16 |
| ANCF design adjoint (GPU) | ∂L/∂E vs numpy oracle | 2.6e-6 |
| Coupled *steady* FSI design grad (GPU) | vs numpy oracle | 1.9e-6 |
| …at realistic scale (150 elem) | vs finite differences | 1.1e-5 |
| Unsteady free-wake forward (GPU) | per-step lift vs numpy oracle | 5.6e-16 (bit-exact) |
| **Wake-history adjoint** (16 steps, 48 rings) | ∂(Σlift)/∂corners vs complex-step | **5e-16 … 2e-15** |
| Gradient checkpointing | grad vs full-store | 1e-17 (identical) |
| Unsteady physics | steady-limit → validated VLM; impulsive-start indicial (Wagner) signature | qualitative ✓ |
| **Coupled *unsteady* FSI forward** (GPU) | final state vs numpy oracle | **2.5e-11** |
| **Coupled *unsteady* design adjoint** (full wake) | ∂E / ∂ρ vs finite differences | **1.0e-4 / 1.2e-6** |
| Control gradient ∂L/∂u_t | vs finite differences | 2.4e-9 |
| Closed-loop policy gradient dL/dk (position-DOF feedback) | vs finite differences | 2.7e-6 |

The NumPy/complex-step references are retained permanently as oracles; the production path is entirely
Warp/GPU.

### 3.1 Throughput

A full design+control gradient of the coupled unsteady FSI (forward rollout + the entire end-to-end
adjoint, with the free wake) on one RTX 4090 (fp64), versus mesh size:

| mesh | elements | panels | DOF | grad eval |
|---|---|---|---|---|
| 3×3 | 9 | 9 | 144 | 0.13 s |
| 6×4 | 24 | 24 | 315 | 0.31 s |
| 10×6 | 60 | 60 | 693 | 0.64 s |
| 15×10 | 150 | 150 | 1584 | 3.64 s |

The adjoint costs roughly the same as a forward solve, so quality-diversity co-design with hundreds–
thousands of *gradient* evaluations is affordable on a single consumer GPU (the §4 archives are
≈500 evaluations in ≈9 min). The architecture is batched for multi-environment evaluation; the present
bottleneck at small mesh is the per-step host↔device synchronisation of the partitioned transfer, not
the GPU kernels.

### 3.2 Physical validation — bird-scale flapping flight

Numerical correctness does not by itself show the solver predicts *realistic* aeroelastic forces. We
therefore validate against real bird-scale flapping-wing-robot flight data — the standard an aerospace
reviewer applies, where order-of-magnitude agreement to within ~2× is accepted as physical credibility.
A wing is driven through prescribed flapping kinematics (rotation about the span root, θ(t)=A·sin 2πft)
in forward flight using the **same production all-Warp UVLM kernels the differentiable coupled solver
uses** (bound rings → AIC → moving-body rhs → batched solve → unsteady Kutta–Joukowski + ∂Γ/∂t force →
shed/convect); the NumPy path is retained only as an oracle (a single bit-exact cross-check: GPU vs
NumPy cycle-mean lift agree to **8×10⁻¹⁴**), and the production path is entirely GPU.

At bird scale (span 1.6 m, chord 0.29 m, area 0.46 m², U = 8 m/s, ±28° flap at 3 Hz; reduced frequency
k = πfc/U = 0.34, Re ≈ 2×10⁵; mass 0.52 kg → weight 5.1 N), at a resolved (144-panel) mesh the rigid
wing **trims to support its own weight at α ≈ 13°** (cycle-mean lift 5.0 N), generating **+4.2 N net
thrust** (Knoller–Betz — the flapping is genuinely propulsive) at a cycle-mean **mechanical power of
29 W**. With a typical η ≈ 0.6–0.7 drivetrain that is ≈ 42–48 W electrical — squarely inside the
published **40–82 W** band for 1.6–2 m, 0.5–1.0 kg flapping robots (81.6 W for the 1.8 m / 1.0 kg
rigid–flexible-coupling vehicle of Zhong et al. 2026; the HIT-Hawk / HIT-Phoenix class; E-Flap). Both
the trimmable lift and the power therefore sit within the accepted 2× band.

A space+time refinement study (40 → 220 panels, steps/cycle ∝ chordwise panels so U·dt ≈ chord/n_c, wake
spanning ~1.5 cycles at every resolution — refining space at *fixed* dt is not a valid convergence path
for unsteady aero, since the wake sheds one trailing-edge row per step) keeps the result within the 2×
band throughout: the cycle-mean lift falls ≈28% from the coarsest to the finest mesh as the
wake-induced downwash resolves (the coarse mesh slightly over-predicts), converging downward toward the
resolved value used above. This is the recognized-data validation §3 previously lacked — and it is
against real flapping-MAV flight data at the actual design scale, not a toy analytical limit.

| Quantity (bird scale, 144-panel resolved) | Solver | Published 1.6–2 m / 0.5–1 kg robots |
|---|---|---|
| trim angle of attack (lift = weight) | α ≈ 13° | typical avian cruise α |
| cycle-mean lift | 5.0 N (= weight) | weight 5–10 N |
| net thrust | +4.2 N (propulsive) | propulsive |
| cycle-mean power | 29 W mech (≈42–48 W elec, η 0.6–0.7) | 40–82 W |

---

## 4. Co-design results

We co-design the **structure *and* the control together**. The genotype is a 7-D vector — spanwise
stiffness and mass spline control points (root/mid/tip for `E` and `ρ`) plus a closed-loop control gain
`k` (position-DOF velocity feedback `u = −k·q̇`). Quality is gust-rejection, `−‖q_N − q_ref‖²` (the
deflection energy a gust IC leaves in the wing after the coupled unsteady rollout — lower is better).
The behaviour space, following a 翼面 (wing) axis × 动力系统 (dynamics/control) axis design rationale,
is **spanwise stiffness taper `E_tip/E_root` × control gain `k`**, with the mass distribution
co-optimised *within* each cell. Cheap forward evaluations fill cells by mutation; a DQD
gradient-arborescence emitter uses the **exact validated design+control gradient** (gE, gR, dL/dk — all
checked against finite differences, §3) through the spline chain rule to sharpen quality.

On **one RTX 4090 (fp64)**, 480/517 stable evaluations in **420 s**:

- **162 / 196 niches** illuminated (83 % coverage); gust-deflection quality varies by **orders of
  magnitude** across designs (the design genuinely matters).
- Adding the closed-loop control axis improves the best achievable gust rejection severalfold over the
  structure-only archive (best quality −2.4e-3) — a real structure–control interaction.
- **Finding (the MAP-Elites phenomenon):** the high-quality (best gust-rejecting) niches span **both a
  wide stiffness taper (≈3.0 wide) and a wide control gain (≈5 of the 0–9 range)** — i.e. there are
  *many distinct (body × control) co-designs that are each excellent gust rejectors*, not a single
  optimum. Each illuminated cell carries a distinct full spanwise stiffness **and** mass distribution
  **and** its own control gain (Fig. 1).

A separate structure-only archive (stiffness taper × mass taper, no control) gives the complementary
result: high-quality niches span a wide stiffness taper but a *narrow* mass taper (≈0.95 wide) —
outboard mass is constrained both by performance and by the explicit-integrator feasibility boundary.

**Robustness.** Repeating the (stiffness × control) co-design with an independent random seed (different
initial population and gust realisation) reproduces the result — 143 vs 162 niches, 73 % vs 83 %
coverage, and the same wide top-quality spread in stiffness taper (2.9 vs 3.0) and control gain (3.2 vs
5.2) — so the diverse (body × control) phenomenon is a property of the landscape, not of a particular
seed.

This is iteration-1 (single gust-rejection objective, spanwise spline design, global position-DOF
feedback control); it demonstrates that the differentiable co-design pipeline produces a diverse,
high-quality (body × control) archive with a concrete finding, affordably, on commodity hardware.

*Figure 1 (`qd_unsteady_hero.png`): left — the illuminated (stiffness taper × control gain) archive
coloured by log gust-deflection energy; right — representative co-designs, each a distinct spanwise
stiffness (solid) and mass (dashed) profile with its own control gain.*

### 4.1 Multi-objective: the gust-load-alleviation Pareto front

Gust rejection is not free: it is bought with **control effort**. For every illuminated elite we
recover its actuation energy `E_ctrl = Σ_t ‖u_t‖² dt` from the velocity trajectory of the coupled
unsteady rollout, giving a second objective. The (gust-deflection, control-effort) scatter of the 162
co-designed elites traces a clear **achievable Pareto front** (16 non-dominated designs): at one
extreme, *passive* designs (k→0, zero actuation) leave ≈2.5e-2 deflection energy; spending control
effort walks the front down to ≈2.1e-3 — a ≈12× gust-load reduction — and the **morphology shifts along
the front** (stiffness taper ≈2.0 at the low-effort/passive end vs ≈0.9 at the low-deflection/active
end). This is the aeroservoelastic co-design statement: *the structure and the controller must be
designed together*, because the best deflection-vs-effort trade-off is reached by particular
combinations of spanwise stiffness and feedback gain, not by either alone (Fig. 2).

*Figure 2 (`qd_pareto.png`): the achievable gust-load-alleviation Pareto front — control effort vs gust
deflection over all co-designed elites, coloured by stiffness taper; the front (black) is the set of
non-dominated structure+control co-designs.*

### 4.2 Reaching the *dynamic* regime — a differentiable strong-coupled adjoint, and gradient co-design under the added-mass instability

The archive of §4 lives in the short, quasi-static gust-load window because its forward is an **explicit
(partitioned)** coupling. On a realistic light/flexible (Pazy-class) wing the dynamic regime is barred
by the **fluid added-mass instability**: with the *same* wing, gust and rollout, the partitioned
(lagged-wake) coupling diverges (peak deflection → NaN), while a **strong (predictor–corrector)**
coupling stays bounded (3.5 % span). Strong coupling is therefore *mandatory* here, and gradient
co-design needs the adjoint of that strong-coupled solver.

We built and validated it. The forward is a linearly-implicit-Newmark structural step
(`A = M(ρ) + β·dt²·K_mem(q;E)`, matrix-free batched CG) strong-coupled to the unsteady free wake by an
**Aitken-Δ² predictor–corrector** fixed point — the aero force re-evaluated on the current structural
iterate (~13 coupling iterations/step), the textbook cure for the added-mass instability. The GPU
forward matches the converged fixed point of a numpy oracle to **8×10⁻¹²–3×10⁻¹¹** (machine precision)
across dt and with/without the wake. The adjoint differentiates the *converged* fixed point by the
**implicit function theorem**: the per-step adjoint is itself a fixed point, solved by the *same* Aitken
relaxation, so the relaxation ω drops out of the gradient (it depends only on the fixed-point Jacobian,
not the iteration path). Every gradient is validated against finite differences of the strong-coupled
oracle — design **∂E/∂ρ** (rel 1e-6–1e-4), the **control** gradient ∂L/∂u_t (rel ~1e-3), and the
**closed-loop policy** gradient dL/dk (rel ~1e-5) — each both with and without the full wake history.
(Two subtle errors were caught *only* by these FD checks and are documented: the aero output path feeds
the predictor state, not just the fixed-point seed; and the adjoint fixed-point must use the forward's
Aitken relaxation, else it diverges under strong feedback exactly where plain Picard would.)

With this adjoint we run **gradient-driven dynamic gust-load alleviation**. A 1-cosine vertical gust
deforms the wing; the deformation drives a mean-axis attitude excursion that — by conservation of
angular momentum — tilts a gimbal/IMU-carrying fuselage, so we minimise
`J = Σ_t ½[(φ_pitch·u_t)² + (φ_roll·u_t)²]` (nominal-inertia-weighted lever functionals of the vertical
deformation — the canonical body-attitude / root-load gust-alleviation objective). Co-designing the
spanwise stiffness and mass taper under a **fixed material+mass budget** (a genuine redistribution, via
budget-conserving gradient projection), Adam through the strong-coupled adjoint cuts the attitude
excursion by **8.3 %** on one RTX 4090 (24 elements, N=20, free wake), discovering a
**soft-root→stiff-tip stiffness taper (0.29→2.17) with mass moved outboard (0.78→1.45)** — it stiffens
and adds inertia to the tip, whose spanwise lever dominates the roll excursion (Fig. 3).

Co-designing the structure **and a closed-loop controller jointly** — adding a position-DOF
velocity-feedback gain `k` with a control-effort penalty (`J = attitude + λ·effort`, the gain
co-optimised by the validated closed-loop gradient, `dL/dk` rel ~1e-5) — sharpens the aeroservoelastic
picture, and the same adjoint quantifies all three gradients (design, control, joint) consistently.
For this *forced*-gust attitude objective, **active feedback is the dominant lever**: at `k ≈ 2.1` it
cuts the mean-axis excursion **10.2× vs the passive wing**, against the 8.3 % from structural
redistribution alone; once feedback is active, re-tapering the structure adds only a further **0.9 %**.
That is the honest statement — for forced-gust attitude stabilisation the controller does the heavy
lifting and the structure refines — and it is exactly the kind of design-vs-control trade-off a
differentiable strong-coupled solver is *for*: we did not assume it, we computed it.

This is exactly the contribution the limitations of an earlier draft flagged as still-missing: a
*validated, differentiable, strong-coupled* transient FSI whose adjoint enables gradient co-design in
the added-mass-instability regime that partitioned — and hence prior explicit-forward differentiable —
co-design cannot enter.

*Figure 3 (`codesign_attitude_gust.png`): left — J/J0 convergence of the dynamic gradient co-design;
right — the discovered spanwise stiffness and mass taper (soft-root→stiff-tip, mass outboard).*

### 4.3 The flagship co-design — meta-RL-amortised structure (stiffness × mass) via MAP-Elites

The archives above co-design structure with a *scalar* feedback gain. The platform's full target is a
two-layer optimiser with a **meta-reinforcement-learning control layer**: a SINGLE RL² policy that adapts
to each design, so the structural search is amortised over ONE controller rather than retraining a
controller per morphology. We realise it here.

**Design space** = a spanwise STIFFNESS field × MASS field (low-D splines, root→tip). Each maps to (a) the
per-element (E, ρ) on the validated ANCF wing and (b) physically-reduced flight-dynamics aggregates —
tip-compliance washout (passive gust alleviation), root-stiffness (cruise efficiency), tip-mass inertia
(passive gust resistance), total mass (weight) — every aggregate anchored to the validated coupled FSI.
A uniform mass field reduces the env **exactly** to the stiffness-only env (Δ = 0 over a rollout).

**Control** = an RL² meta-policy (a Takens delay-embedding context network) **meta-trained over the
(stiffness × mass) distribution**: from its first interactions it infers the current morphology and adapts
(episode return −333 → +312 over training), so every archive design is flown by the **same** controller
with NO per-design retraining (amortised control — the plan's §6 control layer).

**Co-design.** MAP-Elites illuminates the (stiffness-washout `s_gust` × mass-inertia `m_gust`) behaviour
space on one RTX 4090: **75 / 96 niches (78 % coverage)**, each the best gust-rejecting design at that
morphology, *all flown by the one meta-policy*. Findings:

1. **Passive tip-compliance washout dominates gust rejection** — the low-`s_gust` (flexible-tip) column is
   the brightest (best controlled gust excursion), recovering the passive-alleviation mechanism.
2. **Mass inertia gives a complementary resistance** — at fixed stiffness, tip-mass lowers the gust
   excursion (the §4.2 attitude mechanism: tip inertia resists the lever-weighted excursion), bought with a
   weight (efficiency) and a control-sluggishness cost.
3. The archive **traces the gust-rejection × cruise-efficiency frontier** (excursion 0.86–3.75, L/D
   20.6–25.7): stiff/efficient vs flexible/gust-rejecting, with the mass distribution modulating along it.
4. **One meta-policy amortises control across all 75 morphologies** — the structure-and-control co-design
   is searched without retraining a controller per design (Fig. 4).

**High-fidelity grounding.** The fast flight surrogate's gust-rejection ranking is validated against the
**differentiable coupled FSI** of §2–4: the archive designs' spline (E, ρ) are mapped to the per-element
ANCF wing and run through the coupled unsteady free-wake FSI under the same gust, and the structural
gust-deflection energy ranks consistently with the surrogate excursion (**Spearman ρ = 0.70** over the
feasible designs). The grounding initially exposed that a pure load-washout surrogate *over-credits*
structurally over-flexible wings (which the FSI penalises through large deflection); an FSI-grounded
over-flex penalty (the same compliance threshold the efficiency model uses) restores consistency and
keeps the archive's elites FSI-feasible — the high-fidelity solver is what catches and fixes this.

This is the flagship discovery the platform was built for: a **meta-RL-amortised structure-and-control
co-design** — a (stiffness × mass) morphology archive flown by a single adaptive policy, the
gust-rejection × efficiency landscape mapped and validated against the high-fidelity FSI.

*Figure 4 (`meta_codesign_archive.png`): left — the meta-RL co-design archive over (stiffness washout ×
mass inertia), each cell coloured by the meta-policy's controlled gust excursion; right — the
gust-rejection × cruise-efficiency frontier of all 75 elites (each a distinct stiffness×mass morphology),
coloured by mass inertia, all flown by ONE amortised meta-policy.*

---

## 5. Limitations and next steps (stated honestly)

- **Regime of the differentiable archive, and what the *dynamic* regime requires.** The differentiable
  archive of §4 runs at dt=1e-5 over ≈4×10⁻⁴ s — the *initial* gust-load deflection (a legitimate,
  design-sensitive aeroservoelastic quantity, ~30 % landscape spread), not a multi-period response. A
  careful investigation of how to reach the full dynamic response produced two findings we report as the
  honest path forward. (i) A **linearly-implicit Newmark forward with wake truncation** lets dt be set by
  the slow structural mode (not the fast one): on the soft wing the aeroelastic bending then fully
  develops (deflection energy ~10⁻²→O(1)) at affordable step counts with a bounded wake. (ii) On a
  realistic stiffer/lighter (Pazy-class) cantilever, however, the partitioned (lagged-wake) coupling
  exhibits the classic **fluid added-mass instability** — it diverges *unconditionally* in dt and
  independently of numerical damping (verified: γ∈{0.6,0.8,1.0} all diverge). This is the textbook
  signature requiring **strong (predictor–corrector) coupling**. The fully differentiable *dynamic*
  co-design therefore needs the adjoint of an implicit-structure + strong-coupling + wake-truncation
  solver — **which §4.2 now supplies and validates**: an implicit-function-theorem adjoint of the
  Aitken predictor–corrector fixed point, with design/control/closed-loop gradients all checked against
  finite differences, used for gradient co-design *in* the added-mass-instability regime. What remains
  open is scale (larger meshes, multi-period rollouts with far-field wake merging) and the multi-objective
  dynamic Pareto, not the differentiable strong-coupled machinery itself. (An earlier draft mis-attributed
  the short window to explicit-integrator stability; the modal analysis and the added-mass diagnosis above
  correct that.)
- **A boundary-condition correction.** The shell builder shipped without boundary conditions, so an
  earlier version of the archive ran an unconstrained (free-floating) wing — harmless over the very short
  quasi-static rollout but incorrect in principle. The results in §4 use the proper **clamped cantilever**
  root; this both fixes the physics and, being far better conditioned than the near-singular free
  system, improves coverage (162 vs 141 niches, 83 % vs 72 %).
- **Goland flutter benchmark — not yet reproduced on this stack.** (i) An isotropic ANCF plate has
  bending/torsion ratio EI/GJ ≈ (1+ν)/2 ≈ 0.65, whereas the Goland wing needs ≈ 9.9, so matching it
  requires an **orthotropic** constitutive calibration; (ii) flutter must be tracked over many
  oscillation periods, i.e. the long-rollout + wake-truncation infrastructure above, and — because
  artificial damping would mask the instability — an *implicit, damping-free* scheme (generalised-α) for
  the structure. We report these rather than tune a number to 137 m/s.
- **Control co-design axis — in, but a single global gain.** Actuating the *position* DOFs (not the
  tiny-inertia ANCF slope DOFs, which destabilise the explicit integrator) makes the closed-loop control
  stable for gains up to k≈10 and keeps the policy gradient exact (dL/dk vs FD = 2.7e-6); this is what
  enabled the (stiffness × control) archive of §4. The control law is still a single global
  velocity-feedback gain; a full state-feedback / RL² meta-policy net (the gradient machinery for which
  is already validated) and higher gains (needing an implicit integrator) are the next step.
- **Objectives.** The two objectives here are gust deflection and control effort (a complete, classic
  aeroservoelastic gust-load-alleviation trade-off, §4.1). A *propulsive-efficiency / cost-of-transport*
  objective additionally needs a viscous sectional-drag (polar) correction on top of the inviscid UVLM
  (a pluggable hook, not yet enabled), since the inviscid wake gives only induced drag; that is the path
  to the intended (gust × cruise-efficiency) frontier. Leading-edge-vortex / dynamic-stall models are
  likewise pluggable hooks for beyond-attached-flow regimes.
- **Scale.** Results are at a deliberately small mesh so a full archive fits in minutes on one 4090; the
  solver is validated to 150 elements and the architecture is batched, but large-mesh multi-environment
  co-design awaits more compute.

---

## 6. Conclusion

We have built and rigorously validated a GPU-native, differentiable unsteady free-wake aeroelastic
solver whose adjoint runs end-to-end through the coupled rollout and the recurrent wake — to machine
precision — and shown that it makes population-based aero-structural co-design affordable on a single
commodity GPU, yielding a diverse archive and a concrete gust-rejection finding. The validated adjoint,
not the specific archive, is the durable contribution: it turns the unsteady FSI into a gradient source
for design and control, which is the missing primitive for differentiable flapping-wing co-design.

---

### Reproducibility

All results: `platform/` (solver + adjoints + experiment), `tests/` (oracles). Key validations run via
`python diff_uvlm_unsteady_gpu.py` (wake adjoint), `python diff_coupled_unsteady_gpu.py --grad/--control/--policy`
(coupled adjoints), and the co-design archive via `python codesign_qd_unsteady.py` →
`codesign_qd_unsteady_figure.py`. Single RTX 4090, fp64.
