# 实验跟踪：v5h11 IR-WRK3

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V5H11-M0-001 | M0 | RHS/invariant identity | symbolic + independent NumPy/SciPy oracle | manufactured | chain-rule and log-invariant residual | MUST | TODO | no W2 output or production reducer |
| V5H11-M0-002 | M0 | analytic third order | IR-WRK3, N=4/8/16/32 | stretch/rotation/shear | X/Gamma/sigma error, p>=2.8 | MUST | TODO | exact matrix exponential oracle |
| V5H11-M0-003 | M0 | reject tempting fixes | log-sigma/projection/q-freeze | manufactured negatives | first mismatch, rejection | MUST | TODO | cannot silently alias IR-WRK3 |
| V5H11-M1-001 | M1 | zero/near-zero and caps | IR-WRK3 attacks | boundary matrix | zero bits, 0-call STOP, clean retry | MUST | TODO | threshold frozen before run |
| V5H11-M1-002 | M1 | self+external convergence | IR-WRK3 N=1/2/4/8 | synthetic cloud | state/field/order/invariant | MUST | TODO | fine reference independent |
| V5H11-M1-003 | M1 | same-stage tracer ledger | physical+material+frontier | synthetic cloud | hashes, calls, storage reset | MUST | TODO | direct=6N, center=6N, offset=18N |
| V5H11-M1-004 | M1 | rollback/provenance | callable/tree/replay attacks | synthetic | state unchanged, clean retry | MUST | TODO | no published partial state |
| V5H11-M2-001 | M2 | W2 inner candidate | IR-WRK3 N=32/64/128 | source4-6/Ptera3-5 | state/tracer/probe/load/stability | MUST | BLOCKED_BY_M0_M1 | N64 only candidate |
| V5H11-M2-002A | M2 | deterministic artifact A | fresh process | W2 inner | 12-file closure | MUST | BLOCKED_BY_M2_001 | no GT |
| V5H11-M2-002B | M2 | deterministic artifact B | fresh process | W2 inner | semantic byte parity | MUST | BLOCKED_BY_M2_001 | different run provenance |
| V5H11-M2-003 | M2 | fresh integrity audit | read-only auditor | artifacts A/B | claims, hashes, raw recompute | MUST | BLOCKED_BY_M2_002 | same-family provisional |
| V5H11-M3-001 | M3 | outer source/Ptera time convergence | outer 32/64/128, inner N64 | full W2 raw | common-phase loads/probes/ratio | MUST | BLOCKED_BY_M2 | no observation |
| V5H11-M3-002 | M3 | lifecycle/remesh gate | current row owner | full W2 raw | inactive/restart/support ownership | MUST | BLOCKED_BY_M2 | remesh_required is STOP |
| V5H11-M3-003 | M3 | inner-contamination control | outer P128, inner N64/N128 | full W2 raw | inner difference <=10% outer difference | MUST | BLOCKED_BY_M2 | attribution gate |
| V5H11-M3-004 | M3 | FD-J sensitivity | P64, epsilon 2^-9/2^-10/2^-11 | full W2 raw | force/moment/probe change <=.002 | MUST | BLOCKED_BY_M3_001 | nominal is not selected post hoc |
| V5H11-M3-005A | M3 | final raw artifact A | frozen outer P64 candidate | full W2 raw | all counters/loads/hash closure | MUST | BLOCKED_BY_M3_001 | observation sealed |
| V5H11-M3-005B | M3 | final raw artifact B | frozen outer P64 candidate | full W2 raw | semantic determinism | MUST | BLOCKED_BY_M3_001 | independent process |
| V5H11-M3-006 | M3 | final raw fresh audit | read-only auditor | artifacts A/B | raw recompute and unlock eligibility | MUST | BLOCKED_BY_M3_005 | emits audit token, never GT |
| V5H11-M4-001 | M4 | first paper comparison | read-only frozen scorer | immutable raw vs frozen W2 observation | prereg RMSE/Q1, no fitting | MUST | BLOCKED_BY_RAW_AUDIT | separate output directory |
