# Claim Validation

| Claim | Evidence | Expected | Observed | Verdict |
|---|---|---|---|---|
| Resolved real aerodynamic loads reach Q16 | one-step real LEV/TEV/free-wake test | exact generalized load | force/moment and geometry gates pass | supported |
| Transfer is work conjugate | synthetic independent direction | relative error `<1e-12` | `1.3878e-17` | supported |
| Real load packet is internally closed | two-step production packet | scaled floating closure | force/moment `2.8422e-14` | supported |
| Full LEV load is structurally complete | two-step impulse | zero unresolved component | norm `138.1317` | refuted for current implementation |
| Full predictor/corrector is coupled | trial geometry consumption | same q/dq drives aero and receives load | geometry adapter absent | inconclusive / blocked |
