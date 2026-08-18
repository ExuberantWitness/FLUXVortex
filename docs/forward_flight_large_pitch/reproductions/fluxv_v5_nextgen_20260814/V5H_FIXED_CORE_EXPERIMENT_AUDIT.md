# FluxV v5h fixed-core v1 experiment audit

**Date:** 2026-08-15  
**Run:** `20260815_fluxv_v5h_dvm_node_birth_fixed_core_gate`  
**Evaluation type:** `simulation_only + self_supervised_proxy`  
**Review independence:** same-family, provisional  
**Decision:** `WARN`, no blocking `FAIL`; retain `go_v1_only`

## Supported claim

With a fixed physical birth smoothing radius of `0.085 m`, independent
deposition spacings `0.04/0.02/0.01/0.005 m`, and no target-observation or load
path, the bounded two-release mechanical gate is finite, deterministic, and
passes its preregistered time/quadrature criteria for the generic straight,
taper, and twist geometries.

This result supports only the single-cloud v1 handoff.  It does not validate a
cumulative cloud, a third release, Ptera feedback or loads, target-paper
accuracy, or general three-dimensional stability.

## A--F integrity result

| Check | Verdict | Evidence and limitation |
|---|---|---|
| A. Observation access | PASS | Runtime open-audit and guarded entry points report zero target reads and zero target/Ptera solver calls. Package initialization still imports some Ptera/benchmark definitions, but no builder or solver is executed. |
| B. Metrics and normalization | PASS with scope warning | L2 errors are self-convergence metrics, not observation-normalized scores. Core scan is diagnostic-only and cannot select the reported core. |
| C. Results and hashes | PASS | Formal artifact SHA closure passes; raw arrays are stored and independently reloaded to reproduce all reported refinement gates. |
| D. Executed gates | PASS | First/continuous/restart, disabled, both incidence signs, placement/patch binding, global exact-once, two-release transport, replay, time and quadrature gates all execute. |
| E. Scope | WARN | Three synthetic geometries, four span cells, two releases, one LEV-family cloud; no hinge, multi-wing, target geometry, or physical force feedback. |
| F. Classification | PASS | Mechanical simulation/self-supervised proxy, not real-GT validation. |

## Recomputed numerical gates

- Time-refinement error ratios are approximately `8.0004--8.0020`; all fine
  relative L2 errors are below `3.3e-13` against the `1e-6` limit.
- Frontier quadrature ratios are `1.95--4.05`; finest errors are at most
  `2.32e-7`.
- Fixed-probe quadrature ratios are `1.85--4.03`; finest errors are at most
  `2.78e-5`.
- Required ratio is `>=1.5`; required finest quadrature error is `<=0.01`.
- All 36 primary configurations, deterministic replay, restart audit, smoke,
  patch binding and global exact-once gates pass.

## Frozen implementation evidence

- Runner SHA-256:
  `dc6ac3948dc9a2fe26dc8dd7fa28fe66591f0c45f6fdde8359e776fd00579cf1`
- Runner-test SHA-256:
  `b05fe4d32b88d4cf6c143f5c90c32053d0cfa6e88fb35b8632d8ef6646b230f1`
- Ribbon global-exact-once SHA-256:
  `f9c1d64e3ceaeb1b2b686276eea535436603edfea97f3c3882f2c275668778d9`
- Ribbon-test SHA-256:
  `c444c79b74022fdfb8038322c2fd05b22f6aafe9927eea656459db2842ebe04b`
- Semantic result SHA-256:
  `c18f0f0795d11c9ba166d34900135cbb44bae67c558f261e1ce964ec8f41e3e5`
- Focused runner tests: `22 passed`.
- Extended v5h/rVPM regression: `266 passed`.
- Fresh ribbon attack audit: `77` focused and `224` joint tests passed.

Formal evidence is in
`runs/20260815_fluxv_v5h_dvm_node_birth_fixed_core_gate/`.  Its
`SHA256SUMS` verifies all ten payload files.  `raw_refinement.json` stores the
36 configuration arrays plus the replay arrays; `recomputed_gates.json`
reconstructs the reported gates from that artifact.  `semantic_manifest.json`
separates deterministic result content from run-specific UUID, clock, argv,
output path, Git state, environment and log records.

## Audit fixes closed before release

1. Cross-mapper ribbon reuse is now rejected by a role-aware global live-token
   registry bound to wing, patch/frame, cell, event and placement identities.
2. Every enabled runner path supplies and verifies its live node-placement
   result; patch binding and global exact-once are top-level hard gates.
3. Raw state/frontier/probe arrays and independent gate recomputation are
   durable artifacts.
4. Real argv, output path, UTC, UUID, Git state, environment, packages and an
   observed repository-module snapshot are recorded.
5. Runtime instrumentation guards both module-level and package-alias target
   entry points.
6. A failed minimal smoke writes a complete STOP artifact rather than losing
   evidence.

## Promotion boundary

`go_v1_only` authorizes only the next non-target experiment: cumulative-cloud
v2.  Ptera feedback/load coupling and Yang/Izraelevitz/Baik scoring remain
blocked until cumulative exact-append, three-release lifecycle, time/space
refinement, rollback and integrity gates all pass.
