# v5h node-DVM fixed-core auxiliary gate

Verdict: **GO for bounded v1 mechanics**.

This is an observation-free, noncanonical mechanical artifact. Shared-node DVM instances supply geometry only; cell-centre DVM instances supply ribbon circulation only. The physical smoothing radius was fixed before quadrature refinement, while requested and realized edge spacings were recorded separately.

The reported core scan is diagnostic-only and cannot select the registered smoothing radius. Cumulative-cloud v2 and all paper scoring remain blocked regardless of this bounded v1 verdict.

`raw_refinement.json` contains every state/frontier/probe array needed to reconstruct the refinement metrics. `semantic_manifest.json` excludes run UUID, timestamps, paths, and other invocation provenance so independent runs can be compared by one deterministic digest.
