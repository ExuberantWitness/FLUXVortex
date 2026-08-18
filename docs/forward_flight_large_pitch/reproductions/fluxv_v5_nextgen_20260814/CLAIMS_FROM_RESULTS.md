# Result-to-Claim Verdict

- `claim_supported`: **no**
- `confidence`: **high** for rejecting the intended positive claim
- `review_independence`: `same-family`
- `acceptance_status`: `provisional`
- `integrity_status`: `warn`

Intended claim: v5a or v5b yields a model that improves over FluxV v4b across
Yang 2025, Izraelevitz/Scherer Figure 14 and Baik W1–W4.

The evidence does not support that claim. v5a improves only Yang and regresses
Figure 14 and Baik. v5b has no paper scores because its LEV-disabled path differs
from current FluxV by up to `0.556435`, far above the `1e-12` exact-reduction
gate, and G6 was not run.

Supported revised claim:

> The v5a cache proxy improves Yang aggregate errors but degrades Figure 14 and
> Baik and is rejected. Standalone v5b passes limited internal diagnostics but
> fails current-FluxV no-LEV reduction, so its cross-paper accuracy remains
> unscored. FluxV v4b remains the recommended version within the current
> three-paper evidence.

Missing evidence: a Ptera-native v5b implementation that passes step/panel/load
no-LEV identity; an implemented high-AR Ramesh force gate; a frozen full
three-paper matrix; LEV off/on ablations; and an independent force-balance oracle.

## Feasibility addendum after the complete condition matrix

- `claim_supported`: **partial**, only for one more tightly gated v5c study.
- `development_decision`: **pivot within FluxV**.
- `confidence`: **medium**.

Supported planning claim:

> The evidence justifies testing a v5c that preserves corrected v4b exactly and
> adds a local causal rate-sensitive suction-loss discrepancy. It does not yet
> support a claim that such a model improves all three papers.

Promotion requires non-inferiority on every primary Yang, Figure 14 and Baik
channel, at least one improvement larger than numerical/digitization
uncertainty, and a separately frozen unseen transfer case. The three current
datasets cannot serve as held-out generalization evidence.
