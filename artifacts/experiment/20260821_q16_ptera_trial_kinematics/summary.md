# Summary

Q16 `q_trial/dq_trial` now reaches real Ptera panel geometry and changes the
mandatory separated-LEV/joint-TEV/free-wake load response.  The two-state CUDA
bridge is mechanically and transactionally tested.

The implementation remains intentionally partial.  It cannot yet reuse a
committed wake across structural time steps, and the active-LEV impulse lacks a
work-conjugate flexible application point.  The next safe development unit is
an incremental aerodynamic step owner; a completed Q16 FSI commit remains
blocked until the impulse/local-work model is scientifically defined.
