# DRAFT comment for newton-physics PR #2848 (NOT posted — for user approval)

---

Following your suggestion, we prototyped aeroelastic coupling as an external
package on this PR's pattern: a window **predictor-corrector** coupler
(predict with extrapolated forces → solve at the predicted state → rewind →
re-march with per-substep force interpolation), driving Newton via
`State.particle_f`, with rewind built on the same idea as `iteration_restart`.

Two results worth sharing:

- On an aeroelastic benchmark validated against a MATLAB reference to 1e-6:
  zero-order-holding forces across substeps (the current substep semantics)
  accumulates **46.8%** error by t*=3; interpolating the same solves between
  window endpoints stays at **5.9e-5** — same arena, same solve count.
- On a Newton-native SolverVBD cloth-in-wind demo: **10.5×** accuracy gain at
  equal solve budget in the strongly-coupled regime.

Code, adapters, and benchmarks: <REPO LINK>. The substep force-interpolation
hook is tiny and independently useful — happy to contribute it upstream if
there's interest.

---

**发帖前待定**:`<REPO LINK>`(FLUXVortex/newton_pc 或独立仓库)+ 你的最终批准。
