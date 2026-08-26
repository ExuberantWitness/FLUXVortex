# Summary

The active-LEV impulse now has a causal CUDA spanwise ownership ledger. Every
shed ring leg retains its source strip through convection and compaction, and
the independently accumulated strip forces close the unchanged global impulse
force while exposing the real leading-edge endpoint pair for each strip.

This clears the source-identity gate, not the full FSI gate. The next step is
an exact Q16 transpose work operator tied to those endpoint rows; until its
resultant, moment and virtual work close, non-zero impulse must remain unable
to commit a structural step.
