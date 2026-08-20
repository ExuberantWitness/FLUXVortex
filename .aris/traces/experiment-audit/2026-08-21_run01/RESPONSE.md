# Experiment-audit response

Verdict: **PASS with WARNINGS**.
Reviewer: `/root/gpu_v2_fresh_audit`
Independence: same-family; provisional.

- 40 focused tests and all static checks passed.
- Warp-FSI 12/12 passed; STRUCT_CG converged in 145 iterations.
- LDVM host/mixed/float32 inputs, float32 FSI, and exhausted PCG all fail closed.
- Final result and profile hashes match.
- Ptera matrix kernels: 35,575 to 21,725; kernel time 57.158837 to 35.968836 ms.
- Warp-FSI kernels: 78,758 to 40,475; synchronizations 7,201 to 1,203; kernel time 1.760184 to 0.413734 s.
- Three-paper wall time: 582.976084 to 448.384719 s, or 1.30017x.
- Peak GPU memory: 11,104 to 6,556 MiB.

Warnings: Baik should bind the pure GT CSV; measurements are single-run; memory reduction belongs to the combined configuration; the external MATLAB fixture must be explicitly selected; CPU orchestration/I/O/geometry/telemetry remain outside the GPU science-plane claim.
