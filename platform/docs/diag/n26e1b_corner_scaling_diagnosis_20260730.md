# N2.6e1b finite-corner scaling diagnosis

Status: **POST-RESULT DIAGNOSIS ONLY**; no claim promotion.

## Geometry prediction

- solid trailing angle: `20.595196772 deg`
- post-Kutta regular velocity exponent: `0.060680333855`
- predicted 128->256 raw-trace change: `8.775765%`

## Frozen trace fingerprint

| trace | exponent 64->128 | exponent 128->256 | raw change 128->256 | geometry-normalized change 128->256 |
|---|---:|---:|---:|---:|
| lower | 0.052817059 | 0.051060763 | 7.334850% | 1.324665% |
| upper | 0.063131892 | 0.063232808 | 9.161337% | 0.354465% |
| mean | 0.057768766 | 0.056856703 | 8.200722% | 0.528650% |
| jump | -0.071676125 | -0.070873243 | 9.357623% | 16.670430% |

- newborn integrated-circulation change 128->256: `1.924294%`

## Decision boundary

The lower/upper/mean point traces follow the geometry-owned finite-corner power law, while the jump follows a different Kelvin-coupled scaling and the integrated newborn circulation is already much more stable. This is descriptive evidence for replacing point gamma_TE by a weak, coupled junction state; it does not validate that replacement.

This artifact does not authorize endpoint extrapolation, a selected epsilon, a new force term, or production use.
