# Experiment Tracker: FluxV v5h cumulative-cloud v2

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V5H-C001 | M0 | v1 exact reduction | cumulative v2 vs passive-frontier v1 | straight, 2 cells, 1 release | bitwise state/frontier parity | MUST | PASS | physical arrays are bitwise equal |
| V5H-C002 | M0 | schema and transaction attacks | cumulative report/handoff | synthetic attacks | reject/retry/state immutability | MUST | PASS | 1-ULP, clone, reorder and cross-wing attacks fail closed with clean retry |
| V5H-C003 | M1 | close step2-to-step3 blocker | exact-append cumulative cloud | straight, 2 cells, 3 releases | additive count, slice/ID/lineage | MUST | PASS | one combined LSRK3 field and exact release slices pass |
| V5H-C004 | M1 | prevent hard-coded three-step path | exact-append cumulative cloud | straight, 2 cells, 4 releases | step/time continuity | MUST | PASS | fourth release completes without mapper reset |
| V5H-C005 | M2 | lifecycle ownership | active/inactive/restart/continuous | straight, 2 cells | frontier use, restart, count | MUST | PASS | old cloud advances while inactive; restart/continuous ownership closes |
| V5H-C006 | M2 | partial activity | split/shrink/grow active cells | straight, 4 cells | fact set and node coverage | MUST | PASS | underresolved mixed boundaries reject transactionally; no interpolation |
| V5H-C007 | M3 | fixed-core full gate | 3 geometries x 4 h x 3 time | generic non-target | frontier/probe L2, ratios, invariants | MUST | FAIL | 36/36 mechanics, 108/108 releases and all time families pass; straight/taper first probe ratios `1.450785`/`1.496492` are `<1.5`; twist passes |
| V5H-C008 | M4 | deterministic artifacts | two fresh full runs | full gate | semantic SHA, raw recompute | MUST | PASS | fresh semantic replays agree; formal bundle closes at `cc8666ad...` with distinct run provenance |
| V5H-C009 | M4 | fresh adversarial audit | frozen code and artifacts | full gate | A--F integrity verdict | MUST | PASS-with-WARN | strict STOP is reproducible; zero target reads/calls, but package eager imports mean runtime import closure is not complete |
| V5H-C010 | appendix | TEV-family smoke | TEV-only cumulative cloud | straight | topology/lineage only | NICE | TODO | must not delay LEV gate |

## Frozen outcome

- formal artifact: `runs/20260815_fluxv_v5h_cumulative_cloud_gate`;
- runner/test/core SHA-256: `155e1ea8c704c5f58e077eca59f67b54b908be54e552c1b40bd595a78750a02b`,
  `86b0ae5b6d9d7b8f739878fc9d6b3e46c05dd28589a3d28d1387e94557095256`,
  `8b4c3efb19293952a854308508d5d76d55f95268a0ee03acd916eb13287f3d49`;
- semantic SHA-256:
  `cc8666ad43ea291aae03a2b1b7e549e8dd6711b2e1c3006a07e293d5437a4759`;
- claim decision: C1 **yes**, C2 **no**, strict `STOP`; no target observation
  was read and no guarded Ptera/target solver, load or builder was called.

## v5h2 next-candidate preregistration

Use a separate dyadic-panel deposition API.  For each edge length `L`, freeze
`n0=max(1,ceil(L/0.04 m))` and `n(level)=n0*2**level` for levels `0..3`, while
holding `sigma_birth=0.085 m`.  Retain the same straight/taper/twist by three
time-resolution by three-release matrix and the same `>=1.5` ratio gate.  Level
0 must be bitwise equal to frozen v2, every edge must double its panel count at
every level, and two fresh semantic artifacts must agree with distinct
provenance.  Any parity/doubling/core/matrix/threshold/mechanics/time/space/
artifact/runtime-boundary failure is immediate `STOP`; target scoring and Ptera
feedback remain forbidden.

## v5h2 executed outcome

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V5H2-D001 | M0 | exact reduction | dyadic level 0 vs frozen cumulative v2 | straight plus bridge graph families | cloud/IDs/lineage/release ledger | MUST | PASS | level-0 physical cloud and ownership ledger are bitwise equal |
| V5H2-D002 | M1--M2 | cumulative lifecycle | dyadic exact-append cloud | first/continuous and active/inactive/restart/continuous | counts, slices, sidecars, frontier | MUST | PASS | no phantom release; cross-version live-ribbon exact-once closes |
| V5H2-D003 | M3 | full dyadic gate | 3 geometries x 4 levels x 3 time | generic non-target | frontier/probe L2, time/h ratios, invariants | MUST | PASS | 36/36 configurations and 108/108 releases pass; all spatial ratios are approximately 4 |
| V5H2-D004 | M4 | deterministic artifacts | two independent full processes | complete matrix | raw disk recompute, manifests, semantic SHA | MUST | PASS | A/B semantic files byte-identical at `dece97d1...`; UUID/time/path distinct |

Final v5h2 implementation SHA-256 values are: dyadic bridge `26a101c3...`,
dyadic cumulative core `db9332b7...`, gate runner `b7ae4448...`, with tests
`72410c52...`, `c1b163d4...`, and `9913db77...`.  The decision is
`go_v5h2_mechanics_only`: C1 remains supported and the dyadic version of C2 is
supported on the frozen non-target matrix.  This is not a load-accuracy or
paper-performance result; Ptera feedback and target scoring remain blocked.
The project-local bundle is
`runs/20260815_fluxv_v5h2_dyadic_cumulative_cloud_gate` and independently
recomputes to the same `dece97d1...` semantic digest as the two `/tmp` runs.

| V5H3-P001 | vertical slice | native rVPM-to-Ptera feedback | FluxV `UVPMHybridSolver` + live v5h2 cloud | generic straight, steps 0--2 | exact reduction, call ledger, AIC/load replay | MUST | PASS | `go_one_way_native_feedback_mechanics_only`; 11 focused/117 related tests, max residual `2.78e-17`; Ptera remains unique load owner and target scoring is blocked |
| V5H4-T001 | vertical slice | frozen Ptera-to-rVPM transport | parent-only Ptera field + live v5h2 cloud | generic straight, one transport step | exact reduction, stage replay, Jacobian epsilon convergence, zero parent writes | MUST | PASS | `go_frozen_parent_transport_mechanics_only`; 14 focused/319 related tests; all epsilon ratios `>15.2`; parent hash unchanged; target scoring blocked |
| V5H5-S001 | vertical slice | synchronized partitioned time march | DVM release -> native Ptera -> one combined rVPM transport -> frontier | generic straight, three layers | exact ownership/order, additive slices, lineage, replay, reductions | MUST | PASS | `go_three_layer_synchronized_mechanics_only`; 102/204/306 particles, max Ptera residual `2.78e-17`, exact replay/prefix, both one-way reductions bitwise exact, 10 focused/176 related tests; taper/twist and target scoring blocked |
| V5H6-R001 | generic refinement | generic geometry and refinement | frozen v5h5 owner order with geometry-consistent Ptera/DVM inputs | straight/taper/twist; dyadic 0/1/2; dt .02/.01/.005 | frontier/probe relative difference <=.02, ratio >=1.25, all mechanics | MUST | FAIL | 15/15 mechanics pass; taper/twist spatial probe and all six temporal gates fail; deterministic STOP artifact summary `55ac0c4e...`; no target access |
| V5H7-T001 | topology oracle | temporal connected-sheet ownership | manufactured fixed-horizon straight ribbon | independent rings vs connected temporal edge ledger | impulse, fixed probes, shared-edge ownership | MUST | PASS | independent rings vanish linearly; connected sheet has first-order nonzero limit; deterministic summary `5b3c925f...`; no target/Ptera/load calls |
| V5H8-I001 | incremental oracle | causal connected-sheet append/update | direct full graph vs incremental live boundary | 1--4 releases; zero then bounded affine transport | field/Jacobian/impulse parity, prefix identity, rollback, fresh-redeposit negative control | MUST | PASS | `go_v5h8_bounded_affine_live_basis_mechanics_only`; zero residual maxima `6.94e-18/1.249e-16/0`, affine `6.94e-18/2.776e-17/4.337e-19`; fresh relative differences `.0011468/.0014550/.0013030`; 38 module/9 runner/131 joint tests; artifact `runs/20260815_fluxv_v5h8_incremental_sheet_oracle`, summary `af38f29d...`; fresh audit PASS with `cap-after-allocation` WARN; ordinary fresh append NO-GO, restricted live-basis clone GO, production blocked pending a conservative boundary update/remesh owner |
