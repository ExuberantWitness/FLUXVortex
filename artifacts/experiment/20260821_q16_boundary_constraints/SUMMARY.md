# Q16 boundary-constraint result

Result: **PASS for immutable essential-boundary ownership**.

The 2x1 Q16 mesh has seven unique nodes on its minimum-span root, producing
42 prescribed position/director degrees of freedom. The frozen owner restores
the exact reference values, projects admissible directions/residuals and
extracts the complementary reaction without element-local duplicate
ownership. The CUDA operator performs the same operations on resident
float64 arrays and repeats bitwise.

Focused tests are 5/5 and the expanded Q16/transaction/transfer/V5M joint suite
is 88/88. Host arrays, float32, wrong shapes, nonfinite values and malformed
ownership reject. This result does not establish nonlinear equilibrium or a
time integrator.
