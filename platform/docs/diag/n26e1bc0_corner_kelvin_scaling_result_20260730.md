# N2.6e1bc0 regular-corner / Kelvin scaling result

- Run: `n26e1bc0_corner_kelvin_scaling_20260730`
- Verdict: **PHYSICS-NO-GO**
- Protocol: **PROTOCOL-PASS**
- Physics state: **PHYSICS-NO-GO**
- Generated (UTC): `2026-07-29T20:52:36.601852+00:00`
- Scope: frozen shadow scaling diagnostic only; no parent-claim promotion.

## Protocol health

| check | pass |
|---|---:|
| `preregistration_hash_matches` | True |
| `exact_frozen_case_set` | True |
| `all_cases_completed` | True |
| `all_values_finite` | True |
| `all_times_aligned` | True |
| `all_branches_internally_consistent` | True |
| `all_branch_identities_equal` | True |
| `all_births_generic` | True |
| `all_five_algebraic_residuals_pass` | True |
| `birth_identity_residuals_pass` | True |

## Frozen cases at t*=0.2 s

| case | status | branch | r/c | A_lower | A_upper | A_mean | actual Gamma_birth/(Uc) | max residual |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `p64_dt0p00625` | completed | lower | 0.0003060620701 | 11.17349208 | 9.13997988 | 10.15673598 | 0.005372221965 | 6.465e-13 |
| `p128_dt0p00625` | completed | lower | 7.652775899e-05 | 11.43928326 | 9.080342683 | 10.25981297 | 0.005320488117 | 6.373e-13 |
| `p256_dt0p00625` | completed | lower | 1.913270494e-05 | 11.75214464 | 9.020089665 | 10.38611715 | 0.005271975561 | 5.747e-13 |
| `p256_dt0p025` | completed | lower | 1.913270494e-05 | 11.69029483 | 9.131597482 | 10.41094615 | 0.01979702121 | 2.364e-12 |
| `p256_dt0p0125` | completed | lower | 1.913270494e-05 | 11.73401178 | 9.07323968 | 10.40362573 | 0.01028615658 | 1.165e-12 |
| `p256_dt0p003125` | completed | lower | 1.913270494e-05 | 11.75777812 | 8.965043178 | 10.36141065 | 0.002688124309 | 3.210e-13 |

## Physics scaling decision

- regular-only predicted exponent `p* = 1.1292006035`
- four-point log OLS `p_K = 0.9606123259`
- local orders (coarse to fine): `0.9445793485, 0.9642884329, 0.9717438273`
- modal changes 128->256: `lower=2.662164%`, `upper=0.667987%`, `mean=1.216087%`

The necessary regular-corner-only scaling gate failed. N2.6e1bc0 regular-corner-only is falsified/frozen; broader moving-interface circulation theories are not adjudicated.

The companion JSON contains every stage branch identity, all five residual maxima, actual-versus-solver birth ledgers, the fixed four-point fit, and complete input/source hashes.
