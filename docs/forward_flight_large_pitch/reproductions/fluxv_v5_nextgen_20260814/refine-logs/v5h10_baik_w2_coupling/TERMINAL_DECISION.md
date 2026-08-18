# Terminal decision: v5h10 Baik-W2 fixed-substep route

- Decision time: `2026-08-16T13:25:28+08:00`
- Decision: `V5H10_N64_STATUS=STOP`
- Paper comparison: `BLOCKED_BY_V5H10_STOP`
- Historical amendment freeze SHA-256:
  `81389d2beb7c39cf2ca4a2b930846109ba0374748f0f5ee7447ada8aaafe503c`
- No-GT diagnostic SHA-256:
  `b2690d79e9edae5c25ae5307ef738aa9e6d4d04c0449afd87edc43a1dbf471b1`
- Historical PLAN snapshot:
  `PLAN_20260816_130055.md`, SHA-256
  `83c44866f38abd3e67d6b5213eec631457b8c11363b4f15c4d24ebae652e31af`
- Historical CHECKLIST snapshot:
  `CHECKLIST_20260816_130055.md`, SHA-256
  `7f4998b0347faabc2e27bee017f23302baaabf04da48d5fa31736cd64d1e690c`

## Decision

The preassigned `N=64` candidate is stopped.  No fresh formal N64 matrix will
be executed, because the already completed no-GT exploratory matrix gives a
conservative falsification of the route:

- layer-2/3 `N=64 -> 128` state differences and the layer-3 probe differences
  exceed their frozen tolerances;
- the N128 layer-2/3 per-particle `|Gamma| sigma^2` drift exceeds `1e-6`;
- positivity, stability maxima, empirical error-reduction ratios, and the
  reported layer-3 load channels pass, but none is an alternate selector.

The diagnostic is summary evidence rather than a replayable positive
artifact.  That limitation cannot weaken a conservative STOP.

## Prohibited continuations

- Do not promote N128 to candidate or try N256.
- Do not relax thresholds, clip/floor sigma, change the core/spacing, disable
  stretching, or change the integrator inside v5h10.
- Do not open or score the W2 paper observation for this candidate.
- Do not publish a detached N64 raw slice or describe this as paper accuracy.

## Only authorized branch

The only continuation is a separately named, separately preregistered
structure-preserving-integrator experiment.  It must first pass no-GT
manufactured order, invariant, zero/near-zero circulation, physical/tracer
same-stage, transaction, and W2 inner-convergence gates.  Paper observations
remain sealed until that new branch produces and passes a fresh audited raw
artifact.

The historical amendment freeze is intentionally unchanged.  This terminal
decision is the discoverable closure that supersedes its
`preregistered_pending_execution` operational status without rewriting the
historical record.
