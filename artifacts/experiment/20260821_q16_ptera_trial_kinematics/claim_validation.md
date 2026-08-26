# Claim Validation

| Claim | Evidence | Expected | Observed | Verdict |
|---|---|---|---|---|
| Q16 trial position drives Ptera geometry | current vertex comparison | exact within floating transform bound | max error below `2e-14` | supported |
| Q16 trial velocity drives Ptera motion | `(current-previous)/dt` | matches interpolated `dq` | `1.1015e-15` max error | supported |
| Trial velocity changes real loads | same q, different dq | distinct sealed result | force delta `3.2282`, distinct packet SHA | supported |
| Mandatory separated flow remains integrated | real solver state | LEV particles, TEV and free wake | `24` LEV particles, 3 TEV strips, non-prescribed wake | supported |
| Failed active-LEV transfer is transactional | real owner callback | no parent advance | parent pristine, generation `0` | supported |
| Full active-LEV Q16 force is complete | impulse work boundary | zero unresolved load or justified local work | impulse norms `4.7346/6.3790` without application point | refuted for current model |
| Multi-step predictor/corrector is complete | reusable solver owner | incremental remesh/advance and joint commit | only pristine two-state trajectory supported | blocked |
