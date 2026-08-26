# Q16 MITC16+ANS/EAS CUDA result

Result: **PASS for the registered CUDA element-operator scope**.

The CUDA implementation evaluates tying samples, projected strains and their
directional derivatives in parallel, performs one bounded 108-point scalar
EAS reduction per batch/element, and gathers force/Jv deterministically per
degree of freedom. It does not assemble a 96x96 tangent matrix, use
floating-point atomic scatter, or fall back to host numerical evaluation.

Against the independent NumPy oracle, maximum enhanced-parameter error is
`2.46e-21`; relative L2 errors are `6.24e-12` for force and `9.17e-16` for Jv.
Repeated force evaluation is bitwise identical. Host arrays, float32, wrong
shape and nonfinite state attacks fail closed.

The `2.45 ms` warm measurement is a diagnostic for batch=2 and one element;
it is not a scale or end-to-end FSI benchmark.
