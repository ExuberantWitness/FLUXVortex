# Claim Validation

| Claim | Evidence | Expected | Observed | Verdict |
|---|---|---|---|---|
| Same parent/q is deterministic | two trusted branches | equal geometry/wake/load/receipt | exact equality | supported |
| Q16 q changes real flow state | two deformations | different wake and load | wake delta `0.02205`, force delta `22.9826` | supported |
| Parent remains immutable | parent before/after SHA/state | no mutation | exact | supported |
| Separated flow remains mandatory | branch result | active LEV, joint TEV, free wake | 12 LEV particles each; TEV present | supported |
| Q16 endpoint velocity drives flow | explicit dq contract | mapped dq used consistently | not part of this unit | blocked |
| Completed Q16 FSI force exists | work-conjugate all-load transfer | no unresolved force | LEV impulse remains global-only | blocked |
