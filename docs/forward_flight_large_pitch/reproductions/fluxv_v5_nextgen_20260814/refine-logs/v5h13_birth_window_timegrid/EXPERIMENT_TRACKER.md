# V5H13 tracker

| ID | Gate | Evidence | Verdict | Status | Notes |
|---|---|---|---|---|---|
| V5H13-G0-001 | preregistration | PLAN/FREEZE (r=4,k=5, amendment, predictions) | complete before numbers | PASS_DOC | Gate-1 approved |
| V5H13-G1-001 | fork + schedule mechanism | tests-first | all pass | TODO | |
| V5H13-G1-002 | analytic oracle order>=2 | graded vs uniform grid | pass | TODO | |
| V5H13-G1-003 | focused regression + static + hostile review | suites/diff/review | pass | TODO | |
| V5H13-G2-001 | dependency re-sign | manifest/token; old fail-closed | pass | BLOCKED_BY_G1 | |
| V5H13-G2-002 | disposable smoke + prediction | N32/layer1; window peak 0.1243±20% | pass | BLOCKED_BY_G2_001 | |
| V5H13-G3-001A | formal A | full matrix | PASS or exact-prefix STOP | BLOCKED_BY_G2_002 | |
| V5H13-G3-001B | formal B + parity + audit | 9 files byte-identical | pass | BLOCKED_BY_G3_001A | |

## 终局记录（2026-08-17）

- **G1-002 解析 oracle PASS**：graded 格栅阶数 ≥2（预注册门槛；实测
  2.15→2.71→2.87→3.0 渐近），且 graded 在每个 N 严格优于 uniform。
- **Fresh hostile review PASS（无 must-fix）**；三个 should-note 测试加固
  已全部落地（coupling N=6 细/粗槽断言、stream 格栅值断言、runner dt
  漂移负测试）；b3_amendment 与 prereg_freeze 路径修正为 V5H13 治理目录。
- **G2-001 PASS**：manifest+token `/tmp/fluxv-v5h13-audit-20260817-Qh52lH/`
  （41 leaves + 205 modules）verified；executor 全链路加载；
  V5H12/V5H11 旧 token 均 fail closed。
- **G2-002 graded smoke PASS**：141 stages、282/282/846/1128 账本
  （graded 策略确认生效）、invariant/parent/observation/库存全过；
  layer-1 max conv 0.0719 与 V5H12 一致（layer 1 无出生瞬态，符合预期）。
  summary SHA `af859d08…`。
- **G3-001A formal A = 科学门 STOP（窗口后首个粗步）**：
  `/tmp/fluxv-v5h13-formal-A-20260817`（verify_artifact 只读复验 PASS）。
  - **预注册预测检验 PASS（0.1243 精确命中）**：layer-2 graded 窗口
    （substeps 1-20）峰值 0.1243 vs 预测 0.4972/4=0.1243±20% —— 积分器
    精确兑现子循环精度，Idea B 机理成立。
  - STOP 点：N_eff=47 **layer 3 substep 21**（分级窗口 20 细步后的第一个
    粗步 T/32），`h*max|U-U_gal|/sigma` 越界；layer-3 窗口内 max 0.1368
    （≈粗步等效 0.547，略超 0.5）。layer-2 通过（max 0.4928 贴线）；
    durable prefix 2 层 + 343 stages。
  - **定量诊断**：layer-3 出生瞬态的衰减时间长于 k=5（20 细步 ≈ 5 个
    粗步）的冻结窗口；窗口末端瞬时对流数按粗步计 ≈0.547，需再降 ~9%
    才能回粗步。参照 layer-2 剖面（0.46@1 → ~0.44@6 → 0.41@9 →
    0.21@32 粗步时间），瞬态需 ~10-12 个粗步等效才稳定低于 0.5。
  - **预注册后果**：r=4/k=5 冻结，禁止本分支内调整；formal B 禁止；
    V5H13 以 execution-mechanism-validated + scientific-gate-STOP 结束。
- **下一分支（V5H14，需新预注册）的预注册证据**：本工件的 layer-2/3
  剖面给出窗口长度选择的先验依据（k=10 粗步等效为临界、k=12 留裕度）；
  或转向整层出生 σ 调度（Idea C，带保真守卫）。

Current frontier (2026-08-17, implementation pass 1 COMPLETE): the graded
birth-window time grid is implemented across all four layers and every V5H13
suite is GREEN (stream 26, coupling 12, executor 33, runner 51 = 122 tests,
fresh-process). Frozen V5H11/V5H12 controls drift-free; V5H12 suites still
pass (33/51) and frozen V5H11 suites still pass (11/25). Static gates clean
(py_compile/Black/Ruff). Semantics implemented: stream accepts
`substep_delta_times` (exact float list, fsum==delta_time isclose, count must
equal transport_substeps; result.delta_time := exact integrated fsum; record
schema gains hashed `substep_delta_time`; tree validator binds per-substep
constant dt + exact fsum); coupling builds `graded_substep_delta_times` with
frozen K=5/R=4, config flag `birth_window_refinement` (formal + diagnostic
construct True, synthetic default False preserves frozen test semantics),
observer gates use `view.substep_delta_time` (the preregistered honest-dt
amendment), layer results carry N_eff (47/79/143) with role map; executor
rows carry per-record dt, nominal mapping for config, smoke counters 141
stages/282/282/846/1128; runner FORMAL_LEVELS=(47,79,143), per-substep dt
cross-check, post-matrix final coordinate (143,3,6,5,143,3), full matrix
2421 stages. Final leaf SHA-256 (post-Black): stream
`0b7ef14f6404bb8b…`, coupling `7065096775cd4d57…`, executor
`1e4b6e0160c7cb5a…`, runner `bc8e05794d76a2bb…`, coupling test
`e5afd5e82b7d0ee5…`, executor test `325e9ef0c1fd5769…`, runner test
`01b56233871fd14e…`, stream test `2a4e8438298fcd20…`. Next admissible
actions: analytic-oracle order test (G1-002), fresh hostile review of the
8-file diff, dependency re-sign (G2-001), disposable N32/layer1 graded smoke
with prediction check (layer-2 window peak ~ 0.4972/4 = 0.1243, +/-20%) then
formal A.
