# DRAFT comment for newton-physics PR #2848 (NOT posted — for user approval)

> Suggested venue: a comment on PR #2848, following up the earlier aeroelastic
> proposal the team replied to ("prototype outside the core package ... track
> this PR and use the coupling pattern as the basis").

---

Following your suggestion to prototype aeroelastic coupling outside the core
package using this PR's pattern, we built a small external coupler and ran a
controlled study that may be useful input for the framework design. Two
findings stand out, both reproducible:

**1. Sub-step force interpolation matters enormously for tightly-coupled
multi-rate problems.** In the current framework, coupler-written forces are
zero-order-held across an entry's substeps (`solver_coupled.py` substep loop).
On an aeroelastic benchmark we can validate against an independent MATLAB
reference to 1e-6 (cantilevered plate, UVLM lifting surface + shell, 34
structural substeps per fluid solve), holding the force constant across the
window accumulates to **46.8%** worst amplitude error by t*=3, while linearly
interpolating the same per-window force solves between window endpoints stays
at **5.9e-5** — same arena, same number of force solves, only the substep
force schedule differs. On a Newton-native arena (SolverVBD cloth in wind,
ring-vortex aero injected via `State.particle_f`), the same change is worth
10.5× against a per-step-coupled reference at equal solve budget in the
strongly-coupled regime.

**2. A predictor pass makes the window self-consistent at negligible cost.**
Solving the force provider at a *predicted* window-end state (march once with
extrapolated forces, solve, rewind via the same state-restart machinery this
PR already has in `iteration_restart`, then re-march with interpolated
forces) removes the one-window force lag. Combined with (1) this is the
difference between the scheme diverging from and matching the monolithic
reference in our benchmark.

The external package (window coupler + a Newton VBD adapter + the benchmark
evidence) lives at <REPO LINK>. The coupler maps cleanly onto this PR's
concepts: window rewind generalizes `iteration_restart` from one top-level
step to a multi-substep window; force interpolation would slot into the
substep loop as an optional per-substep force schedule; added-mass-style
operators ride the same interpolation and pair naturally with the
effective-mass hooks. Happy to adapt any of it if there's interest — and to
contribute the substep force-interpolation hook upstream, which is tiny and
independently useful.

---

**附:发帖前需要替换/确认的项**
- `<REPO LINK>` → FLUXVortex 仓库 newton_pc 路径(或独立仓库,用户决定)
- 是否附 lagged-vs-two-pass 对比图(可从 npc_*_b*.npy 数据出图)
- 语气/长度按用户偏好调整
