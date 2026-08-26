# Q16 Real Strong-FSI Transaction Results

## Outcome

`PASS / GO` for one bounded, real, nonzero-load strong-FSI step.

The 5-degree, one-Q16-element, 2x3-panel pilot repeatedly forks one unchanged
Ptera parent, advances separated LEV + joint TEV + free wake, transfers the
complete resolved and impulse load, and solves one Q16 Newmark step. A final
replay is prepared before both owners advance once.

The final focused baseline passed `4/4 in 152.58 s`; the additional injected
second-aero-evaluation failure passed independently. The wider Q16/LEV/TEV/FSI
surface passed `157/157 in 17.59 s`.

## Important fixes exposed by integration

1. Before any particles exist, the mandatory LEV mechanism now publishes an
   explicit zero strip-force record instead of omitting the schema.
2. Empty particle-source ownership is legal only as a zero-length provenance
   vector; strip topology remains explicit and validated.
3. The discrete high-stiffness reference internal-force remainder is frozen
   and subtracted, preserving the StVK stress-free reference state rather than
   treating quadrature roundoff as physical load.
4. Aerodynamic and structural time steps must match exactly.
5. All fallible validation, clone and hash work precedes the two owner pointer
   swaps; failed evaluation, nonconvergence or drift commits neither owner.

## Claim boundary

This is a single-step integration checkpoint, not an FSI validation case. It
does not establish multi-step stability, time/grid convergence, constitutive
generality, experimental accuracy or production throughput. The unpreconditioned
matrix-free CG is currently expensive on this high-order Q16 pilot and is the
next performance/scaling blocker. No GT/scorer/paper data was accessed.
