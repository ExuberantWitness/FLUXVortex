# Summary

The Q16 predictor now supplies both candidate displacement and endpoint
velocity to an isolated real separated-flow aerodynamic branch.  Velocity is
no longer decorative: at fixed geometry it changes the LE/TE boundary motion,
free wake and force while the committed parent stays exact.

This clears the motion-side gate.  The next gate is not another geometry toy:
it is the work-conjugate localization of the active-LEV impulse, followed by
the actual Q16 Newton/predictor transaction commit.
