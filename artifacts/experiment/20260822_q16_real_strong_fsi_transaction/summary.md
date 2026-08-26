# Summary

The actual Q16 FSI data path is now end to end for one bounded nonzero-load
step: Newmark predictor -> real Ptera branch -> separated LEV/joint TEV/free
wake -> complete conservative Q16 load -> structural solve -> fixed-point
replay -> prepared double-owner commit.

Status: `PASS_SINGLE_STEP / PARTIAL_MULTI_STEP`.
