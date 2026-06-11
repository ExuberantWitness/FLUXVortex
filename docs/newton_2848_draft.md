# DRAFT comment for newton-physics PR #2848 (NOT posted — for user approval)

---

Following your suggestion, we prototyped aeroelastic coupling as an external
package on this PR's pattern: a window **predictor-corrector** coupler
(predict with extrapolated forces → solve at the predicted state → rewind →
re-march with per-substep force interpolation), driving Newton via
`State.particle_f`, with rewind built on the same idea as `iteration_restart`.

Results:

- On an aeroelastic benchmark validated against a MATLAB reference to 1e-6:
  zero-order-holding forces across substeps (the current substep semantics)
  accumulates **46.8%** error by t*=3; the predictor-corrector with the same
  solve count stays at **5.9e-5**, and holds **5.5e-4** out to t*=6 through
  multiple large-amplitude flapping reversals.
- On a Newton-native SolverVBD cloth-in-wind demo: **10.5×** accuracy gain at
  equal solve budget in the strongly-coupled regime.

Why we think this matters for Newton: it shows **partitioned, black-box
coupling can reach reference-grade accuracy over long horizons** — the
fidelity normally associated with monolithic solvers — for ~2× structural
march cost and zero changes to solver internals. That would open the coupled
framework to validation-grade scientific/engineering simulation
(aeroelasticity, added-mass-dominated FSI, multi-rate coupling with stiff
feedback), a regime beyond what lagged proxy exchange can sustain.

Code, adapters, and benchmarks: <REPO LINK>. The substep force-interpolation
hook is tiny and independently useful — happy to contribute it upstream if
there's interest.

---

**发帖前待定**:`<REPO LINK>`(FLUXVortex/newton_pc 或独立仓库)+ 你的最终批准。
