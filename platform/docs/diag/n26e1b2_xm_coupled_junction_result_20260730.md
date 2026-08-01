# N2.6e1b2 Xia--Mohseni coupled junction gate result

- Run: `n26e1b2_xm_coupled_junction_gate_20260730`
- Verdict: **NO-GO**
- Generated (UTC): `2026-07-29T19:58:27.690100+00:00`
- Scope: fixed-wing canonical spatial operator only; no pressure, force, target response, or production coupling.

## Gate summary

| gate | pass |
|---|---:|
| exact 27-case matrix | True |
| all cases completed | True |
| algebra, formation and condition gates | True |
| all nine symmetric cases explicit no-birth | True |
| all nine mirror pairs | False |
| every Cauchy score finite | True |
| every final Cauchy score <= 2% | False |
| every final Cauchy score nonincreasing | False |

## Cases

| case | status | stage | cond2 | normal/U | Kelvin/(Uc) | Kutta/U |
|---|---|---|---:|---:|---:|---:|
| `side1_dominant__p32__eps1of4` | completed | forming | 2857.77 | 8.431e-16 | 3.851e-16 | 0.000e+00 |
| `mirror_side2_dominant__p32__eps1of4` | completed | forming | 2857.77 | 1.388e-15 | 1.422e-16 | 1.110e-16 |
| `symmetric_no_birth__p32__eps1of4` | completed | no_birth | 1521.35 | 7.234e-16 | 1.426e-17 | — |
| `side1_dominant__p32__eps1of8` | completed | forming | 4207.61 | 9.554e-16 | 1.527e-16 | 0.000e+00 |
| `mirror_side2_dominant__p32__eps1of8` | completed | forming | 4207.61 | 1.053e-15 | 1.041e-16 | 0.000e+00 |
| `symmetric_no_birth__p32__eps1of8` | completed | no_birth | 2237.4 | 7.268e-16 | 4.808e-16 | — |
| `side1_dominant__p32__eps1of16` | completed | forming | 5731.1 | 1.013e-15 | 3.123e-16 | 5.551e-17 |
| `mirror_side2_dominant__p32__eps1of16` | completed | forming | 5731.1 | 1.094e-15 | 2.359e-16 | 0.000e+00 |
| `symmetric_no_birth__p32__eps1of16` | completed | no_birth | 3046.29 | 4.441e-16 | 5.307e-17 | — |
| `side1_dominant__p64__eps1of4` | completed | forming | 8474.75 | 2.005e-15 | 1.874e-16 | 0.000e+00 |
| `mirror_side2_dominant__p64__eps1of4` | completed | forming | 8474.75 | 2.609e-15 | 2.220e-16 | 0.000e+00 |
| `symmetric_no_birth__p64__eps1of4` | completed | no_birth | 4250.88 | 2.248e-15 | 1.246e-15 | — |
| `side1_dominant__p64__eps1of8` | completed | forming | 12524.6 | 2.272e-15 | 3.469e-17 | 0.000e+00 |
| `mirror_side2_dominant__p64__eps1of8` | completed | forming | 12524.6 | 2.377e-15 | 2.776e-17 | 0.000e+00 |
| `symmetric_no_birth__p64__eps1of8` | completed | no_birth | 6251.37 | 1.874e-15 | 1.962e-16 | — |
| `side1_dominant__p64__eps1of16` | completed | forming | 17105.3 | 1.991e-15 | 1.214e-16 | 5.551e-17 |
| `mirror_side2_dominant__p64__eps1of16` | completed | forming | 17105.3 | 2.790e-15 | 0.000e+00 | 0.000e+00 |
| `symmetric_no_birth__p64__eps1of16` | completed | no_birth | 8511.68 | 1.776e-15 | 1.459e-15 | — |
| `side1_dominant__p128__eps1of4` | completed | forming | 28709.9 | 2.831e-15 | 4.094e-16 | 0.000e+00 |
| `mirror_side2_dominant__p128__eps1of4` | completed | forming | 28709.9 | 5.295e-15 | 9.714e-17 | 0.000e+00 |
| `symmetric_no_birth__p128__eps1of4` | completed | no_birth | 11982.1 | 2.026e-15 | 3.505e-16 | — |
| `side1_dominant__p128__eps1of8` | completed | forming | 42694.8 | 2.841e-15 | 1.422e-16 | 0.000e+00 |
| `mirror_side2_dominant__p128__eps1of8` | completed | forming | 42694.8 | 4.559e-15 | 6.592e-17 | 0.000e+00 |
| `symmetric_no_birth__p128__eps1of8` | completed | no_birth | 17618.1 | 2.026e-15 | 9.334e-16 | — |
| `side1_dominant__p128__eps1of16` | completed | forming | 58514.6 | 3.194e-15 | 1.804e-16 | 0.000e+00 |
| `mirror_side2_dominant__p128__eps1of16` | completed | forming | 58514.6 | 3.803e-15 | 7.980e-17 | 0.000e+00 |
| `symmetric_no_birth__p128__eps1of16` | completed | no_birth | 23987.6 | 2.914e-15 | 2.787e-16 | — |

## Largest final Cauchy scores

| axis.case.metric | final score | pass |
|---|---:|---:|
| `epsilon.side1_dominant.p128.gamma2` | 1.08972989 | False |
| `epsilon.mirror_side2_dominant.p128.gamma1` | 1.08972989 | False |
| `epsilon.mirror_side2_dominant.p64.gamma1` | 0.786593676 | False |
| `epsilon.side1_dominant.p64.gamma2` | 0.786593676 | False |
| `epsilon.mirror_side2_dominant.p32.gamma1` | 0.557169732 | False |
| `epsilon.side1_dominant.p32.gamma2` | 0.557169732 | False |
| `panel.side1_dominant.gamma2` | 0.388755083 | False |
| `panel.mirror_side2_dominant.gamma1` | 0.388755083 | False |
| `epsilon.mirror_side2_dominant.p32.gamma2` | 0.278792407 | False |
| `epsilon.side1_dominant.p32.gamma1` | 0.278792407 | False |
| `epsilon.mirror_side2_dominant.p64.gamma2` | 0.268156004 | False |
| `epsilon.side1_dominant.p64.gamma1` | 0.268156004 | False |
| `epsilon.side1_dominant.p128.gamma1` | 0.257003386 | False |
| `epsilon.mirror_side2_dominant.p128.gamma2` | 0.257003386 | False |
| `panel.side1_dominant.gamma1` | 0.085717956 | False |
| `panel.mirror_side2_dominant.gamma2` | 0.085717956 | False |
| `epsilon.mirror_side2_dominant.p32.gamma_g` | 0.0374730971 | False |
| `epsilon.side1_dominant.p32.gamma_g` | 0.0374730971 | False |
| `epsilon.side1_dominant.p32.Gamma_bound` | 0.0343313872 | False |
| `epsilon.mirror_side2_dominant.p32.Gamma_g` | 0.0343313872 | False |

## Decision

NO-GO for this fixed-budget N2.6e1b2 implementation: not converged within the preregistered budget. This result does not falsify the continuous Xia--Mohseni mechanism.

All signed values, wrapped angular differences, both Cauchy intervals, exact matrix records, and source hashes are in the companion JSON.
