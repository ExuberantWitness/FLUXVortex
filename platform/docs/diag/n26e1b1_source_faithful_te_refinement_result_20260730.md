# N2.6e1b1 source-faithful TE refinement result

- Run: `n26e1b1_source_faithful_te_refinement_20260730`
- Verdict: **NO-GO**
- Generated (UTC): `2026-07-29T19:21:01.222191+00:00`
- Scope: nearest-control-point TE spatial asymptotics only; no IBL, pressure, or force validation.

## Gate summary

| gate | pass |
|---|---:|
| three frozen spatial levels present | True |
| all cases completed | True |
| no branch ambiguity | True |
| all normalized algebraic residuals <= 1e-9 | True |
| every score finite | True |
| every 128->256 score <= 2% | False |
| every final score nonincreasing | False |

## Cases

| case | status | side | lower/U | upper/U | mean/U | jump/U | length/c | Gamma_new/(Uc) | max residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `p128_n32_core0.02` | completed | lower | 0.653736417 | 0.599287021 | 0.626511719 | 0.0544493964 | 0.0704825684 | 0.00383773331 | 9.538e-13 |
| `p256_n32_core0.02` | completed | lower | 0.609062588 | 0.548992013 | 0.5790273 | 0.0600705742 | 0.0651405713 | 0.00391303152 | 1.165e-12 |
| `p64_n32_core0.02` | completed | lower | 0.703393008 | 0.654093258 | 0.678743133 | 0.0492997507 | 0.0763586024 | 0.00376446006 | 1.153e-12 |

## Preregistered metric scores

| group.metric | 64->128 | 128->256 | final <=2% | nonincreasing | pass |
|---|---:|---:|---:|---:|---:|
| `local_birth.te_lower_downstream_trace` | 0.0759581222 | 0.0733485045 | False | True | False |
| `local_birth.te_upper_downstream_trace` | 0.0914524002 | 0.0916133686 | False | False | False |
| `local_birth.te_mean_downstream_trace` | 0.0833686138 | 0.0820072193 | False | True | False |
| `local_birth.te_emission_jump_ccw` | 0.0945767282 | 0.0935762281 | False | True | False |
| `local_birth.newborn_segment_length` | 0.0833686139 | 0.0820072193 | False | True | False |
| `local_birth.newborn_sheet_strength_ccw` | 0.0945767282 | 0.0935762281 | False | True | False |
| `local_birth.newborn_circulation_ccw` | 0.00366366238 | 0.00376491057 | True | False | False |
| `local_birth.newborn_endpoint_body_x` | 0.00540713681 | 0.00493953613 | True | True | True |
| `local_birth.newborn_endpoint_body_y` | 0.0523214913 | 0.0477011497 | False | True | False |
| `global_wake.bound_circulation_ccw` | 0.0109769035 | 0.0104526183 | True | True | True |
| `global_wake.wake_circulation_ccw` | 0.0109769035 | 0.0104526183 | True | True | True |
| `global_wake.wake_signed_centroid_x` | 0.00440971652 | 0.00436991431 | True | True | True |
| `global_wake.wake_signed_centroid_y` | 0.00266846938 | 0.00345461646 | True | False | False |
| `global_wake.wake_first_moment_x` | 0.0154350251 | 0.0148682097 | True | True | True |
| `global_wake.wake_first_moment_y` | 0.0102511037 | 0.0103087788 | True | False | False |
| `algebraic.maximum_kelvin_residual_over_Uc` | 0 | 0 | True | True | True |
| `algebraic.maximum_normal_bc_residual_over_U` | 3.70074342e-15 | 2.34997207e-13 | True | True | True |
| `algebraic.maximum_eq7_residual_over_c` | 9.95592497e-12 | 1.05700171e-11 | True | True | True |
| `algebraic.maximum_eq8_residual_over_U` | 9.86864911e-15 | 4.4408921e-14 | True | True | True |
| `algebraic.maximum_linear_system_residual_over_U` | 2.46716228e-15 | 2.36847579e-13 | True | True | True |

## Decision

NO-GO for N2.6e1b1. The source-specified fixed-grid spatial refinement path failed at least one frozen completion, algebra, 2%, or monotonicity condition; its thresholds remain unchanged.

Machine-readable values, both interval scores, source hashes, and any failure tracebacks are in the companion JSON.
