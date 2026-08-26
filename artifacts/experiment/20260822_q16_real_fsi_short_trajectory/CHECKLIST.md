# Q16 Real FSI Short-Trajectory Checklist

## Identity

- run id: `20260822_q16_real_fsi_short_trajectory`
- idea id: Q16 mandatory separated-flow multi-step FSI continuity
- stage: auxiliary/dev integration

## Planning

- [x] selected idea summarized
- [x] baseline and comparability contract confirmed
- [x] code touchpoints listed
- [x] smoke and full-run plans written
- [x] fallback options written

## Implementation

- [x] immutable per-step trajectory record and hash chain
- [x] exact owner/generation/time-coordinate continuity gates
- [x] stopped result with completed prefix and failed coordinate
- [x] unrelated physics and tolerance changes avoided

## Pilot / Smoke

- [x] direct two-step real owner advance executed
- [x] second step consumes first committed LEV/TEV/free-wake state
- [x] all metrics and hashes finite/interpretable

## Main Run

- [x] bounded trajectory success test
- [x] failed next step preserves completed prefix/live owner
- [x] fresh deterministic replay or explicit semantic tolerance decision

## Validation

- [x] focused tests pass
- [x] selected broad regressions pass
- [x] static and whitespace gates pass
- [x] metrics and claim map recorded durably

## Closeout

- [x] result summarized with explicit claim boundary
- [x] next action selected: extend the real Ptera horizon before testing a
  4--8-step history and energy/work-ledger stability gate
