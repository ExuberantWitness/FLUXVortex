# Experiment-audit request

Fresh read-only review of the frozen FLUX-V5M full-GPU parallel optimization candidate.

Required checks:

1. Independently re-run the 40 focused GPU tests and static gates.
2. Reproduce CPU/mixed LDVM input, float32 FSI, and exhausted-PCG hostile probes.
3. Verify the final Baik, three-paper, CUDA metrics, Nsight, and SQLite hashes.
4. Recompute SQLite kernel/synchronization counts and the end-to-end wall/memory deltas.
5. Return A–F, warnings, and claim impact without editing or rerunning the full paper matrix.
