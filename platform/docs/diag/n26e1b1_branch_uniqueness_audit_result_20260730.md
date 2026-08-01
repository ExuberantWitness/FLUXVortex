# N2.6e1b1 independent branch-uniqueness audit

- Audit verdict: **FAIL**
- Formal N2.6e1b1 verdict remains: **NO-GO**
- Generated (UTC): `2026-07-29T19:39:11.361006+00:00`

## Audit gates

| gate | pass |
|---|---:|
| exact 64/128/256 matrix | True |
| all cases completed | True |
| all 99 calls contain successful lower and upper evidence | True |
| formal case metrics reproduced bitwise | False |
| formal reference/source hashes valid | True |

## Cases

| case | calls | two branches every call | selection every call | formal metrics bitwise | pass |
|---|---:|---:|---:|---:|---:|
| `p128_n32_core0.02` | 33 | True | True | False | False |
| `p256_n32_core0.02` | 33 | True | True | False | False |
| `p64_n32_core0.02` | 33 | True | True | True | True |

## Failed selector calls

None.

This audit is diagnostic only. It cannot reverse the formal N2.6e1b1 NO-GO.
