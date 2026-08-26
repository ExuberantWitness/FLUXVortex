# Claim Validation

| Claim | Metric / gate | Verdict |
|---|---|---|
| Source-line endpoint split preserves resultant | `8.88e-16` max error | supported |
| It preserves midpoint force moment | `1.78e-15` max error | supported |
| It is Q16 work-conjugate | `6.94e-18` work error | supported |
| Real separated-LEV impulse reaches Q16 DOFs | generalized norm `4.5123` | supported |
| The load/source/geometry boundary fails closed | hostile tests | supported |
| Complete resolved+impulse Q16 aerodynamic load exists | resolved real-point operator not general | blocked |
| Coupled Newmark/Newton transaction is complete | combined load not yet installed | blocked |
| Paper accuracy changed or improved | no paper run | not evaluated |
