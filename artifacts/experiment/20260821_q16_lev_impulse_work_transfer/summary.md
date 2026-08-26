# Summary

The source-owned active-LEV impulse now reaches real Q16 generalized
coordinates through an explicit CUDA float64 work-conjugate operator. Each
span-strip force acts at its actual leading-edge source-line midpoint via the
same two Q16 interpolation rows that generated the aerodynamic endpoints.

This closes the LEV impulse component, not the entire FSI load. The remaining
critical implementation is a general algebraic Q16 owner for all real Ptera
resolved load points, followed by combined-load Newton evaluation and atomic
aero/structure commit.
