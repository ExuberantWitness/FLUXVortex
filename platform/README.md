# FLUXVortex · Differentiable GPU Aeroelastic Co-Design Platform (iteration 1)

A **differentiable, GPU-native, Newton-style** fluid–structure platform for
**control + structural-distribution co-design** of flexible flapping-wing aircraft.
Built on a predictor-corrector FSI coupler validated **bit-exact** against a
MATLAB/NumPy golden, extended into an end-to-end co-design loop: *design → coupled
FSI under gust → (gust-rejection, efficiency) → quality-diversity archive*, plus a
closed-loop control policy and a fully differentiable structural operator.

Everything below is **verified on an RTX 4090** (pure Warp / `fp64`).

---

## Run the co-design (one command)

```bash
cd FLUXV/src
FLUXV_DEVICE=cuda:0 python ../platform/codesign.py                       # fast, budget 40
FLUXV_DEVICE=cuda:0 python ../platform/codesign.py --dqd --budget 30     # DQD gradient emitter
FLUXV_DEVICE=cuda:0 python ../platform/codesign.py --full --budget 8 --out frontier.npz
```
or config-as-code:
```python
from codesign import CoDesign
res = CoDesign(mode="fast").run(budget=40, emitter="dqd")   # mode "full" = real coupled FSI
res.report();  res.save("frontier.npz")                      # gust x efficiency Pareto frontier
```
`mode`: **fast** = structural proxy (~1 s/design) · **full** = real coupled FSI + 1-cosine
gust (~3 min/design, A100 for scale). `emitter`: **random** mutation · **dqd** gradient
(plan §6). Output = the non-dominated **gust-rejection × efficiency** frontier + saved
archive — the discovery hero artifact.

---

## What runs today (every line is a passing check on GPU)

| Layer | Module | What it does | Verified |
|---|---|---|---|
| **Atom** | `ancf_solver.py` | ANCF flexible-shell solver (orthotropic), Newton `StructuralEntry` | bit-exact vs golden Newmark, `q/dq` rel **1e-13** |
| **Atom** | `verify_redlines.py` | orthotropic constitutive red lines | reduction **rel 0.0**, true-orthotropic ✓ |
| **Atom** | `coupled_fsi.py` | predictor-corrector coupled FSI (ANCF + UVLM) | GPU-stable, aero-engaged, PC-corrects-vs-lagged |
| **Diff** | `diff_ancf.py` | differentiable internal force (custom **K_t** adjoint) | nonlinear-loss grad vs FD **4e-8** |
| **Diff** | `diff_step.py` / `diff_rollout.py` | differentiable structural step / multi-step rollout | d/ddq **exact**, rollout vs FD **2e-4** |
| **Diff** | `diff_solve.py` | differentiable AIC dense-solve adjoint | d/db **3e-9**, d/dA **7e-10** |
| **Diff** | `verify_tangent_jacobian.py` | proof: tangent stiffness *is* the exact ANCF Jacobian | K_t·δq vs FD **7e-6** |
| **Layer 2** | `design_map.py` | design vector → orthotropic material → response + sensitivity | stiffer→smaller deflection; FD design grad |
| **Eval** | `codesign_eval.py` | design → coupled FSI + 1-cosine gust → (gust, efficiency) | distinct, finite metrics per design |
| **Layer 0** | `map_elites.py` | MAP-Elites quality-diversity archive over the design space | 53% coverage, 34 diverse elites |
| **Layer 0** | `dqd.py` | differentiable-QD gradient emitter | beats random mutation at equal budget |
| **Control** | `control_eval.py` | Takens-embedding policy, closed-loop in the FSI loop | **76% gust reduction** vs passive |
| **Free-flight** | `p0_resonant_freeflight.py` | differentiable rigid + revolute + servo + resonant spring | tape == FD (Featherstone) |

Run any check (from `FLUXV/src`):
```bash
FLUXV_DEVICE=cuda:0 python ../platform/verify_redlines.py        # ANCF red lines
FLUXV_DEVICE=cuda:0 python ../platform/coupled_fsi.py            # coupled FSI loop
FLUXV_DEVICE=cuda:0 python ../platform/diff_step.py             # differentiable step
FLUXV_DEVICE=cuda:0 python ../platform/codesign_eval.py         # design -> (gust, efficiency)
FLUXV_DEVICE=cuda:0 python ../platform/map_elites.py            # QD archive
FLUXV_DEVICE=cuda:0 python ../platform/control_eval.py          # closed-loop gust rejection
```

---

## The idea

**Co-design** searches the joint space of *(structural design × control policy)*,
evaluating each candidate with a high-fidelity coupled FSI rollout under a gust,
to illuminate the **gust-rejection × flight-efficiency** frontier — and surface
**non-intuitive structure-control synergies** a decoupled optimizer cannot reach.
(Already visible here: under a *transient* gust, a stiffer wing can deflect *more*
— a modal/damping effect the quality-diversity archive is built to reveal.)

- **Design search** = MAP-Elites + a differentiable-QD gradient emitter.
- **Control** = a Takens-delay-embedding policy (PPO-first, SHAC upgrade) actuating
  in the loop.
- **Evaluation** = the validated predictor-corrector coupled FSI on GPU.

## Differentiability — solved exactly, pure Warp

Warp's auto-adjoint of the ANCF bending kernel NaNs. We bypass it: the internal
force's Jacobian **is** the tangent stiffness `K_t`, which every Newmark step
already builds (`assemble_kmem_blocks`). So the structural adjoint is
`adj_q = K_t · adj_Qint` — exact, free, and forward-bit-exact. Composed with the
mass-solve adjoint and the AIC dense-solve adjoint, the structural rollout is
differentiable end to end (chained-adjoint vs FD ≈ 2e-4). See
[`../docs/p1_differentiability_finding.md`](../docs/p1_differentiability_finding.md).

## Status

Iteration 1 (no propeller) is demonstrated end to end on GPU. Production scale-out
(full MOME + PPO/SHAC over the coupled evaluator) targets an A100 cluster; the
thrust-vectoring propeller (rVPM-in-Warp) is iteration 2.
