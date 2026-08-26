# Summary

The real separated-LEV/joint-TEV/free-wake solver now has a causal incremental
owner.  Six single-state advances are bitwise identical to the original
monolithic run, and trusted predictor branches no longer mutate their parent.

The next implementation can safely replace only a fork's next Panel geometry
from Q16 before advancing it.  Full FSI remains blocked until the temporal
velocity contract and LEV impulse work distribution are independently closed.
