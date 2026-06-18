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
