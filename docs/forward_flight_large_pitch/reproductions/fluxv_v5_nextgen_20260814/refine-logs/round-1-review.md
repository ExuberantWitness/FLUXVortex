# Round 1 Review

<details open>
<summary>Raw same-family senior review</summary>

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 5.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 9.0 |
| Feasibility | 4.0 |
| Validation Focus | 7.0 |
| Venue Readiness | 5.5 |
| **Weighted overall** | **6.48/10** |

**CALIBRATION: none**  
**Verdict: REVISE**  
**Drift warning:** The proposal still addresses the anchored problem, but risks drifting from a three-paper Pareto repair into a full 3-D LEV solver, universal section model, and apparatus-boundary solver.

## GAP

The proposal correctly identifies global persistence, duplicate load ownership and one-way LEV wake as the main defects. However, it tries to solve three maturity levels at once: local ownership, a shared 3-D LE/TE wake, and the Baik apparatus boundary. The mathematical interfaces are not yet closed and `equilibrium polar + LDVM dynamic discrepancy + shared-wake feedback` can count the same separation load three times. The smallest falsifiable next step is a v5a local-owner model; shared wake belongs to a later v5b only after a residual-based go/no-go decision.

## Critical revisions

1. Split development into mutually exclusive stages:
   - v5a uses an equilibrium section residual and a paired-LDVM transient force discrepancy, without shared-wake feedback.
   - v5b uses a shared TE/LE material wake; LDVM then supplies only onset/shedding constraints, not another force discrepancy.
2. Remove production ULLT load ownership from v5a; use ULLT only as an attached-flow oracle and parameter/pole source.
3. Define the equilibrium residual and its induced/profile-drag ownership exactly.
4. Define a causal equilibrium-removal operator for the paired LDVM discrepancy; an undefined `LDVM equilibrium solution` is not executable.
5. Do not count the Baik apparatus adapter as a common-kernel contribution. Compare v4b and v5 with the same boundary adapter.
6. List every new state/parameter and its provenance. Do not hide a multi-threshold four-state model behind the term “one local state”.

## Important revisions

- Reduce the state to an LESP trigger plus one convective relaxation state.
- Remove first-round variable-rate shedding, upstream-edge switching, shared wake and wall/endplate operator.
- Treat three-paper leave-one-paper-out as retrospective robustness, not held-out validation.
- Pre-register a fourth untouched experiment and freeze code, parameter table, scoring and failure gates before observing its loads.
- If v5b proceeds, choose either explicit birth or same-time coupled correction and state the unknowns/equations and exact Ptera time layer.
- “Deleted-circulation ledger” alone is not conservative truncation; retain a long wake or preserve circulation and impulse in a far-wake representation.

## Simplification opportunities

1. Implement v5a first: retained UVLM + explicit equilibrium section residual + local LESP owner + paired-LDVM transient discrepancy.
2. Keep ULLT outside the production load path.
3. Gate shared wake, variable-rate shedding, edge switching and apparatus boundary work behind residual evidence.

## Modernization opportunities

NONE. This is a deterministic conservation and load-ownership problem; ML/LLM/VLM/Diffusion/RL would add unidentifiable parameters rather than solve the bottleneck.

</details>
