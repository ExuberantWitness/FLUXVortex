# FluxV v5d1 source-clock result

## Decision

The source-clock correction is retained as a correctness finding but rejected
as a cross-paper performance candidate.  All mechanical gates passed; the
strict paper gates did not.

Canonical artifact:

`runs/20260814_fluxv_v5d1_source_clock_all22_reproducible`

## Mechanical result

- disabled-path residual: `0`;
- source-clock formula residual: `0`;
- maximum repeated-cycle state difference after the frozen warm-up:
  `3.68e-12`;
- Yang frozen-v4b replay residual: `7.11e-15 gf`;
- persistence range over all saved phase rows: `4.77e-5` to `0.99807`;
- 22 conditions and 2816 integrated phase rows completed;
- the adapter remains non-canonical because it omits same-time-layer UVLM
  induced strip velocity.

## Accuracy result

| Benchmark / primary metric | corrected reference | v5d1 source clock | outcome |
|---|---:|---:|---|
| Yang lift MAE, gf | 4.55451 | 4.77281 | worse, FAIL |
| Yang drag MAE, gf | 2.64400 | 2.61014 | better |
| Figure-14 all-14 CT RMSE | 0.024251 | 0.025466 | worse, FAIL |
| Figure-14 15-degree CT RMSE | 0.020759 | 0.022720 | worse, FAIL |
| Figure-14 25-degree CT RMSE | 0.029512 | 0.029778 | worse, FAIL |
| Figure-14 unique-12 CT RMSE | 0.025700 | 0.026923 | worse, FAIL |
| Baik filtered macro CL RMSE | 0.657542 | 0.622979 | better |
| Baik filtered macro CD RMSE | 0.345152 | 0.304709 | better |

All eight Baik W1--W4 CL/CD rows improved:

| Case | CL RMSE | CD RMSE |
|---|---:|---:|
| W1 | 0.515665 -> 0.498209 | 0.160907 -> 0.156978 |
| W2 | 1.032306 -> 0.991838 | 0.725678 -> 0.597793 |
| W3 | 0.374319 -> 0.358613 | 0.262591 -> 0.246054 |
| W4 | 0.707878 -> 0.643258 | 0.231434 -> 0.218009 |

The mean persistence change explains the mixed direction: approximately
`0.677 -> 0.693` for Yang, `0.0118 -> 0.0496` for Figure 14, and
`0.771 -> 0.567` for Baik.  The corrected clock fixes a source-unit error but
does not supply the missing physical branch selector.

## Allowed claim

The published half-chord time scale materially improves the frozen Baik
transfer and slightly improves Yang drag, but it does not simultaneously
improve the three development benchmarks.  The result is a non-canonical,
previously inspected development comparison and cannot support a generalization
or production claim.

