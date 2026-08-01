# N2.6e1b attached-outer numerical refinement result

- Run: `n26e1b_attached_outer_refinement_20260730`
- Verdict: **NO-GO**
- Generated (UTC): `2026-07-29T19:01:56.978529+00:00`
- Scope: attached outer-flow numerical admissibility only; no target-response or force validation.

## Gate summary

| gate | pass |
|---|---:|
| seven frozen unique cases present | True |
| all cases completed | True |
| all normalized algebraic residuals <= 1e-9 | True |
| all final-two-level observable changes <= 2% | False |

## Cases

| case | status | Gamma_B | Gamma_W | x_Gamma/c | y_Gamma/c | max Kelvin | max BC | max Eq7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `p16_n32_core0.02` | completed | -2.20979275 | 2.20979275 | 2.61496704 | -0.063502067 | 4.934e-17 | 8.203e-16 | 7.542e-13 |
| `p32_n16_core0.02` | completed | -2.16041559 | 2.16041559 | 2.60072036 | -0.0595697495 | 4.934e-17 | 1.160e-15 | 2.486e-12 |
| `p32_n32_core0.01` | completed | -2.18172873 | 2.18172873 | 2.60602945 | -0.0635202415 | 4.934e-17 | 9.940e-16 | 1.602e-12 |
| `p32_n32_core0.02` | completed | -2.18298662 | 2.18298662 | 2.60643131 | -0.0634975966 | 4.934e-17 | 1.035e-15 | 1.608e-12 |
| `p32_n32_core0.04` | completed | -2.18783007 | 2.18783007 | 2.60797509 | -0.0634235543 | 4.934e-17 | 9.870e-16 | 1.630e-12 |
| `p32_n64_core0.02` | completed | -2.19909838 | 2.19909838 | 2.61319822 | -0.065408785 | 4.934e-17 | 1.309e-15 | 6.608e-13 |
| `p64_n32_core0.02` | completed | -2.15768202 | 2.15768202 | 2.59556767 | -0.0633924598 | 4.934e-17 | 2.467e-15 | 1.153e-12 |

## Final-two-level comparisons

| family | middle -> fine | worst score | pass |
|---|---|---:|---:|
| panel | `p32_n32_core0.02` -> `p64_n32_core0.02` | 0.0915834 | False |
| time | `p32_n32_core0.02` -> `p32_n64_core0.02` | 0.0292191 | False |
| core | `p32_n32_core0.02` -> `p32_n32_core0.01` | 0.000730851 | True |

## Decision

NO-GO for the current N2.6e1b attached-outer numerical gate. The frozen levels and thresholds were not changed; the failed case or observable must be diagnosed before downstream use.

Machine-readable details, including every observable score and traceback, are in the companion JSON.
