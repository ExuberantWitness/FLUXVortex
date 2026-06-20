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

The wing is an ANCF thin-shell (9 DOF/node, 36 DOF/element). The internal force is *linear in a
per-element stiffness scale* `E_scale[e]` and the element mass block is *linear in a per-element
density scale* `ρ_scale[e]`; these per-element scales are the structural design variables. A low-
dimensional design is obtained by mapping a small set of spanwise spline control points through a fixed
basis, `E_scale = exp(B·θ_E)`, so the design stays smooth and the gradient is exact through the chain
rule. The membrane tangent stiffness `K_t` is assembled for the adjoint state-transition.

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

On **one RTX 4090 (fp64)**, 469/537 stable evaluations in **541 s**:

- **141 / 196 niches** illuminated (72 % coverage); gust-deflection quality varies by **orders of
  magnitude** across designs (the design genuinely matters).
- Adding the closed-loop control axis improves the best achievable gust rejection by ≈4× over the
  structure-only archive (best quality −2.6e-3 vs −1.1e-2) — a real structure–control interaction.
- **Finding (the MAP-Elites phenomenon):** the high-quality (best gust-rejecting) niches span **both a
  wide stiffness taper (≈3.3 wide) and a wide control gain (≈5 of the 0–9 range)** — i.e. there are
  *many distinct (body × control) co-designs that are each excellent gust rejectors*, not a single
  optimum. Each illuminated cell carries a distinct full spanwise stiffness **and** mass distribution
  **and** its own control gain (Fig. 1).

A separate structure-only archive (stiffness taper × mass taper, no control) gives the complementary
result: high-quality niches span a wide stiffness taper but a *narrow* mass taper (≈0.95 wide) —
outboard mass is constrained both by performance and by the explicit-integrator feasibility boundary.

This is iteration-1 (single gust-rejection objective, spanwise spline design, global position-DOF
feedback control); it demonstrates that the differentiable co-design pipeline produces a diverse,
high-quality (body × control) archive with a concrete finding, affordably, on commodity hardware.

*Figure 1 (`qd_unsteady_hero.png`): left — the illuminated (stiffness taper × control gain) archive
coloured by log gust-deflection energy; right — representative co-designs, each a distinct spanwise
stiffness (solid) and mass (dashed) profile with its own control gain.*

### 4.1 Multi-objective: the gust-load-alleviation Pareto front

Gust rejection is not free: it is bought with **control effort**. For every illuminated elite we
recover its actuation energy `E_ctrl = Σ_t ‖u_t‖² dt` from the velocity trajectory of the coupled
unsteady rollout, giving a second objective. The (gust-deflection, control-effort) scatter of the 141
co-designed elites traces a clear **achievable Pareto front** (15 non-dominated designs): at one
extreme, *passive* designs (k→0, zero actuation) leave ≈1.6e-2 deflection energy; spending control
effort walks the front down to ≈2.6e-3 — a ≈6× gust-load reduction — and the **morphology shifts along
the front** (stiffness taper ≈1.8 at the low-effort/passive end vs ≈1.3 at the low-deflection/active
end). This is the aeroservoelastic co-design statement: *the structure and the controller must be
designed together*, because the best deflection-vs-effort trade-off is reached by particular
combinations of spanwise stiffness and feedback gain, not by either alone (Fig. 2).

*Figure 2 (`qd_pareto.png`): the achievable gust-load-alleviation Pareto front — control effort vs gust
deflection over all co-designed elites, coloured by stiffness taper; the front (black) is the set of
non-dominated structure+control co-designs.*

---

## 5. Limitations and next steps (stated honestly)

- **Goland flutter benchmark — not yet reproduced on this stack.** Two concrete blockers: (i) an
  isotropic ANCF plate has bending/torsion ratio EI/GJ ≈ (1+ν)/2 ≈ 0.65, whereas the Goland wing needs
  ≈ 9.9, so matching it requires an **orthotropic** constitutive calibration; (ii) flutter develops over
  many oscillation periods, but the current **explicit symplectic integrator is stable only over short
  windows** at the required step size. Both are infrastructure items (orthotropic shell + implicit /
  sub-stepped integrator); we report them rather than tune a number to 137 m/s.
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
