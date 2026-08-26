# Claim Validation

| Claim | Evidence | Expected | Observed | Verdict |
|---|---|---|---|---|
| Incremental execution preserves the solver | six-state monolithic differential | bitwise result | zero force delta, equal wake/particles/counters/packet | supported |
| Separated flow remains integrated | final solver state | nonzero LEV and joint TEV | 60 LEV particles and joint TEV present | supported |
| Branch trial cannot mutate parent | two internally issued forks | parent SHA/state fixed | parent unchanged, branch receipts equal | supported |
| Incremental state is lifecycle-safe | hostile order/mode/drift gates | rejection | all targeted gates pass | supported |
| Q16 geometry drives every incremental trial | runtime geometry replacement | candidate q changes branch geometry/wake/load | not implemented in this checkpoint | blocked |
| Completed active-LEV Q16 work exists | impulse localization | work-conjugate force | no local impulse-work model | blocked |
