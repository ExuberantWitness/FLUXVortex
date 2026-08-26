# Run Log Summary

- Tests-first collection initially failed because the packet module and wrench
  transform did not exist.
- The first real Q16 pilot failed because actual vortex mechanisms contain
  coincident application points; the transfer-map contract was narrowed to
  permit repeated rows only under the same element owner.
- A first affine registration assumption also failed: actual vortex application
  points are not represented by one affine chord line.  The bounded test now
  uses two Q16 chordwise macro elements and solves only test-fixture nodal
  geometry; no such CPU fit exists in production.
- Final focused and joint runs passed.  No retry changed the aerodynamic model,
  Q16 method, LEV/TEV mode, wake mode, or numerical thresholds.
