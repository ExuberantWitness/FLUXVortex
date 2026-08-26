# Summary

The actual CUDA aerodynamic load path now reaches Q16 generalized coordinates
for spatially resolved vortex-leg and unsteady-pressure forces.  The transfer
is force/moment closed and work conjugate, and a real one-step mandatory
LEV/TEV/free-wake pilot passes.

This does **not** yet complete multi-step FSI.  The two-step pilot contains a
large non-zero LEV impulse force for which the current model provides no
surface application point, so the new boundary stops rather than smearing it.
The next implementation must define and validate impulse work and must feed the
Q16 trial geometry/velocity into the aerodynamic branch before joint commit.
