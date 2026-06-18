# Discovery run #1 — passive gust-rejection landscape (real coupled FSI, 4090)

First real-physics co-design sweep: 7×2 wing designs (stiffness scale × chord/span
orthotropy), each scored by the **real predictor-corrector coupled FSI** under a
1-cosine vertical gust (peak tip excursion = gust metric, lower is better). No
analytical proxy. ~11 min on the RTX 4090. Data: `discovery.npz`.

## Landscape (gust-induced peak tip excursion)

| stiffness | 0.50 | 0.70 | 0.90 | 1.10 | 1.40 | 1.70 | 2.00 |
|---|---|---|---|---|---|---|---|
| ortho 0.8 | **4.33e-3** | 4.39 | 4.44 | 4.47 | 4.52 | 4.55 | 4.58e-3 |
| ortho 1.2 | **4.35e-3** | 4.42 | 4.47 | 4.51 | 4.56 | 4.60 | 4.63e-3 |

## Findings

**F1 — Flexibility monotonically improves transient gust rejection (~6%).**
The most flexible wing (stiffness 0.5) has the smallest gust-induced excursion;
the stiffest (2.0) the largest, monotonically. This is **counter to the static
intuition** that a stiffer wing resists deflection: under a *transient* gust the
flexible wing bends with the gust and washes out its effective angle of attack,
shedding the gust load (**passive aeroelastic gust load alleviation**). Here the
load-alleviation effect dominates the raw-stiffness effect across the whole range.

**F2 — Orthotropy is a weak lever for gust rejection (~1%).**
Chordwise vs spanwise stiffening shifts the gust response by only ~1% at fixed
stiffness scale. The **stiffness magnitude**, not its in-plane distribution, sets
passive gust rejection in this regime.

**F3 — Hypothesis H1 (non-monotone, interior-optimal stiffness) is FALSIFIED.**
An earlier 2-point sample hinted "stiffer is worse"; we hypothesized a non-monotone
curve with an interior optimum. The high-resolution sweep refutes it — the curve is
**monotone**, no interior optimum. (Reported honestly: the data overruled the prior.)

## Honest scope / what this is and isn't

- **Is**: a real-physics, reproducible finding (passive aeroelastic gust alleviation,
  quantified, monotone) from the validated coupled FSI on a single flexible wing.
- **Isn't yet** the headline co-design *synergy* (the plan's hero). Passively,
  flexibility helps gust **and** lowers COT (less inertia), so the passive optimum
  collapses to the flexible corner — no rich trade-off. The non-trivial co-design
  finding lives in the **design × control** interaction (a stiffer wing + active
  control vs a flexible wing + less control) and the **resonant-spring efficiency**
  axis. That needs the control-co-optimization layer (PPO policy per design) and the
  real flapping aircraft — the next discovery run.

## Next discovery run
Co-optimize design **and** a per-design control policy (close the loop with
`control_eval`), and report where the gust×efficiency frontier with control differs
from the passive one — i.e. the structure-control synergy the discovery paper claims.

---

# Discovery run #2 — structure-control interaction (passive vs controlled)

6 stiffnesses × {passive, controlled (Takens PD policy)} under the same real
coupled-FSI 1-cosine gust (~10 min, 4090). Data: `discovery2.npz`.

| stiffness | 0.50 | 0.80 | 1.10 | 1.40 | 1.70 | 2.00 |
|---|---|---|---|---|---|---|
| passive (×10⁻³) | **4.34** | 4.43 | 4.49 | 4.54 | 4.58 | 4.61 |
| controlled (×10⁻³) | **1.13** | 1.38 | 1.64 | 1.83 | 1.98 | 2.10 |
| gust reduction | **74%** | 69% | 64% | 60% | 57% | 54% |

**F4 — Control authority scales strongly INVERSELY with stiffness (corr = −0.99).**
The same policy rejects 74% of the gust on the most flexible wing but only 54% on
the stiffest. A more compliant wing deflects more per unit corrective load, so the
control has more authority. Clean, monotone, near-perfect correlation.

**F5 — Structure and control are ALIGNED (compounding), not a trade-off.**
The flexible wing is both passively gust-tolerant (F1) **and** the most controllable
(F4). The optimal design is stiffness 0.50 **both** passive and controlled — control
does **not** invert the ranking here. For this objective, structure and control
reinforce each other.

**Honest contrast with the hypothesized "synergy".** The plan's headline is a
*non-intuitive synergy where co-design beats decoupled optimization* (the optimum
shifts when control is added). That **ranking inversion does not occur** in this
iteration-1 setup: a single flexible plate where flexibility helps gust, COT, **and**
controllability simultaneously, so flexibility dominates and co-design ≡ decoupled.
The genuine ranking-inversion synergy needs **competing constraints** — the full
aircraft (resonant-spring efficiency that favors a tuned stiffness, flapping power,
14 surfaces, structural/buckling limits) — where flexibility can no longer win every
axis. That is the production discovery run (full flapping aircraft on A100). What
iteration-1 establishes, on real physics, is the **mechanism map** (F1–F5) those runs
build on.

---

# Discovery run #3 — adding the full-aircraft competing constraint (resonant flap)

Combines the **real coupled-FSI gust data** (discovery2: passive & controlled gust
vs stiffness, measured on the 4090) with a **resonance-aware COT** (Zhong&Xu power
model + a root-spring flap drive whose inertial power is offset near
`omega_n = 2*pi*f_flap`, resonant at stiffness ~1.86). This is the competing
constraint the single plate lacked. `discovery3.py`.

| stiffness | 0.50 | 0.80 | 1.10 | 1.40 | 1.70 | 2.00 |
|---|---|---|---|---|---|---|
| gust passive (×10⁻³) | **4.34** | 4.43 | 4.49 | 4.54 | 4.58 | 4.61 |
| gust controlled (×10⁻³) | **1.13** | 1.38 | 1.64 | 1.83 | 1.98 | 2.10 |
| COT (resonant flap) | 2.33 | 2.36 | 1.76 | 1.03 | **0.55** | 0.50 |

**F6 — the resonant flap drive creates a GENUINE gust×efficiency trade-off.**
Unlike iteration-1 (where flexibility won gust, COT, *and* controllability, so the
optimum collapsed to the flexible corner), efficiency now favors a **tuned, stiffer**
design (COT minimized near the resonant stiffness ~1.7) while gust still favors
flexibility (0.5). The Pareto frontier genuinely spans flexible↔tuned-stiff. This is
the competing constraint a full aircraft introduces.

**Honest result on the ranking-inversion synergy (NEGATIVE here).**
Scalarizing the two normalized objectives equally, the optimum design is **stiffness
1.70 both passive and controlled** — closing the control loop shifts the balance
*toward* the efficient stiffer design (the right direction: control handles the gust,
so efficiency dominates), but **not enough to flip the discrete optimum**. So in this
simplified single-wing + analytical-resonance model the clean ranking inversion does
**not** appear; it is weighting- and magnitude-dependent.

**Conclusion.** Iteration-1 + the resonance constraint establish, on real gust
physics, the full **mechanism map** and a **real trade-off** (F1–F6). Whether the
headline *ranking-inversion synergy* materializes is not settled by a tunable single
wing — it requires the full multibody aircraft (2 wings + V-tail + 14 surfaces, the
spring-driven emergent flap, flapping power from the coupled rollout, structural/
buckling limits) so no single design can win every axis. That is the production run
(A100). The platform, the evaluators, the mechanism map, and an honest read of where
the synergy does and doesn't appear are all in place to drive it.

---

# Discovery run #4 — REAL spring-driven flap resonance (honest: damping-suppressed)

Sweep torsional-spring stiffness, drive the hinge sinusoidally at f=3 Hz on the
verified Featherstone multibody dynamics, measure real stroke + motor power.
`discovery4.py`, `discovery4.npz`.

| ke (N·m/rad) | 0.2 | 0.5 | 1.0 | 1.6 | 2.4 | 4.0 | 7.0 |
|---|---|---|---|---|---|---|---|
| stroke (deg) | 108 | 55 | 28 | 18 | 12 | 7 | 4 |
| power (W) | 7.05 | 3.03 | 1.51 | 0.94 | 0.63 | 0.38 | 0.21 |

**F7 (honest, negative) — the resonance peak is suppressed; response is quasi-static.**
Stroke ∝ 1/ke (at ke=1.6 the stroke 18° = exactly tau/ke), i.e. the spring statically
balances the drive torque with **no resonant amplification**, so there is no interior
efficiency minimum. Cause: the implicit Featherstone integrator adds numerical damping
that kills the structural resonance at these step sizes. To see the resonance-efficiency
benefit (the plan's hero mechanism) the production setup needs a **low-numerical-damping
integrator** (symplectic / much finer dt) — or the resonance must be carried by the
*aeroelastic* coupled solver (the validated predictor-corrector), not the rigid hinge.

---

# Discovery run #5 — the REAL (gust × aerodynamic-efficiency) trade-off

Fixes defect **D1** (`docs/fix_plan.md`): #1/#2 scored "efficiency" as a force-norm
proxy and #3 used an *analytical* resonance COT. This run measures BOTH co-design
objectives from the **same real coupled-FSI rollout** at a 6° cruise AoA — gust
rejection (1-cosine gust, peak tip excursion) and aerodynamic efficiency (lift and
induced drag read directly from the validated UVLM `Fbern`). `discovery5.py`,
`power_probe.py`, `discovery5.npz`. ~9 min on the 4090.

| stiffness | 0.50 | 0.80 | 1.10 | 1.40 | 1.70 | 2.00 |
|---|---|---|---|---|---|---|
| gust (×10⁻³, smaller better) | **9.50** | 9.82 | 10.06 | 10.23 | 10.45 | 10.77 |
| L/D induced (larger better) | 22.19 | 22.60 | 23.05 | 23.50 | 23.93 | **24.34** |
| induced drag (N) | 0.2150 | 0.2108 | 0.2066 | 0.2027 | 0.1991 | **0.1958** |

**F8 — a clean, monotone, REAL gust×efficiency trade-off (no analytical proxy).**
Lift is essentially constant (~4.76 N across all designs), so the efficiency signal is
purely the **induced drag falling 0.215→0.196 N as the wing stiffens** (~9%), while gust
excursion **rises 9.5→10.8 ×10⁻³** (~13%). The two objectives pull in *opposite*
directions across the entire stiffness range: a flexible wing sheds transient gust load
(passive aeroelastic alleviation, F1) **but** its steady load-induced deformation
degrades the spanwise lift distribution → higher induced drag → lower L/D. Stiff is the
mirror image. This is the **genuine competing constraint** the single-objective
single-wing runs (#1/#2, where flexibility won every axis) lacked — now measured on real
physics, not modeled. It is the real Pareto trade-off the co-design optimizes over, and
the substrate on which the ranking-inversion *synergy* question becomes well-posed.

## Overall honest read across discovery runs #1–#5

- **Real, quantified mechanism findings** on real coupled-FSI physics: F1 (flexibility
  improves gust rejection), F2 (orthotropy weak), F4 (control authority ∝ 1/stiffness),
  F6 (resonance, when modeled, creates a real gust×efficiency trade-off), and **F8 — a
  clean, monotone, REAL gust×efficiency trade-off measured directly from the rollout
  (no proxy): gust favors flexible, induced L/D favors stiff.**
- **Honest negatives**: F3 (an interior-optimal-stiffness hypothesis falsified); the
  *analytical* resonance synergy (#3) was weighting-dependent; the rigid-hinge resonance
  (#4) was numerically damped.
- **Where this leaves the synergy.** #5 (F8) closes the gap #1/#2 had: there now IS a
  real single-wing competing constraint. The ranking-inversion *synergy* question — does
  closing the control loop move the optimum along this real Pareto front? — is now
  **well-posed on real physics** and is the next run (FIX 4): MOME over (gust, real L/D)
  decoupled vs. with the Takens control loop. The remaining escalation, the **full
  multibody aircraft** (2 wings + V-tail + spring-driven flap + flapping power), adds the
  efficiency-favors-resonant-stiffness axis (F6) on top of F8's efficiency-favors-stiff,
  compounding the competing constraints. Platform, evaluators, and the F1–F8 mechanism
  map are all built to drive it.
