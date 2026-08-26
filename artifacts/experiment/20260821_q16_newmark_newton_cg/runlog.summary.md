# Run log summary

The registered RED failed at import because the production structural solver
did not exist. The first GREEN run exposed only an over-strong assertion that
the floating reference residual must equal exact zero; the state itself was
bitwise stationary and the measured free residual was `9.13e-12`. The test was
corrected to use the pre-registered Newton gate.

After implementation and formatting:

- focused suite: 4/4 PASS;
- expanded Q16/V5M joint suite: 88/88 PASS;
- Black, Ruff, py_compile and whitespace checks: PASS;
- forced one-Newton-step failure raised, preserved all four input arrays and
  produced a clean retry bitwise equal to a fresh solve;
- subnormal and overflowing time steps rejected before any CUDA operator call.

No full aerodynamic run, experimental ground truth, separated-LEV integration
or co-design optimization was executed in this slice. A persistent managed
`bash_exec` log could not be exported because that tool was unavailable in the
session; this file records the exact limitation rather than claiming one.
