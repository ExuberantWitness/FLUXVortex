# Run Log Summary

- Tests-first initially failed because the incremental owner module was absent.
- The first live-state seal cloned and re-pickled CUDA tensors; Torch storage
  identities made that SHA non-repeatable.  The implementation now separates a
  same-object live seal from a canonical scientific-state digest.
- The first step attempt failed because Ptera wake birth reads the next
  trailing-edge ring.  Instead of requiring non-causal look-ahead ownership,
  wake birth is deferred until the candidate next geometry arrives.  For a
  frozen trajectory this reordering is bitwise equal to monolithic `run()`.
- Arbitrary externally pickled solvers cannot self-resume as trusted branches.
  `session.fork()` creates the clone and binds a fresh live seal internally.
- Final focused and joint tests passed after Black formatting.  No paper data,
  GT or scorer was accessed.
