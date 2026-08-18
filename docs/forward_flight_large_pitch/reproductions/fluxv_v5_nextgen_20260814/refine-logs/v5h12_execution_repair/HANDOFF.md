# V5H12 execution-only repair 开发接手说明

状态：**H0-H6 已完成（代码+测试 GREEN，fresh review PASS）；等待 H7 依赖重签与授权**

最后盘点时间：2026-08-16（H0-H6 完成更新）

仓库：`/tmp/fluxv-v5-nextgen`

当前阶段：V5H12-G1 完成。executor/runner 三个根因已修复，全部测试 GREEN；
**只证明执行协议可用，论文复现与 formal A/B 尚未执行（H7 起需要新授权）**

## 1. 接手任务的一句话定义

在全新 V5H12 runner/executor 命名空间中，只修复 V5H11 formal-A 暴露的执行协议问题：

1. coupling observer 的 7 字段 payload 与 executor 的错误 8 字段期待不一致；
2. executor 当前先追加 stage、后追加其 source parent；
3. conversion failure 被 runner 降格为全空 STOP 坐标。

不得改变 IR-WRK3、coupling、source、Ptera、W2 参数、`N=(32,64,128)`、阈值、探针、载荷或论文比较规则。修复后也只能先证明执行协议可用，不能直接宣称论文复现成功。

## 2. 接手前必须完整阅读

按顺序阅读并遵守：

1. `PLAN.md`
2. `CHECKLIST.md`
3. `EXPERIMENT_TRACKER.md`
4. `FREEZE_INPUTS.json`
5. 本文件 `HANDOFF.md`
6. 父分支 `../v5h11_ir_wrk3/B3_EXECUTION_AMENDMENT_20260816.md`

治理文件已经在任何 V5H12 实现之前冻结。若这些文件与本 handoff 冲突，以 `PLAN.md` 和 `FREEZE_INPUTS.json` 的更严格约束为准。

## 3. 当前事实与失败归因

### 3.1 V5H11 formal-A 的真实结果

- 工件：`/tmp/fluxv-v5h11-b3-formal-A-20260816`
- 状态：`STOP`
- `stop_code`：`coupling_callback_error`
- 消息：`ExecutorContractError: compact stage evidence field set drift`
- 所有 durable scientific row 数：0
- 论文目标读取：0
- semantic result SHA-256：
  `6ddc0714f9f3496501d7c9353af46073fe35ecd1ad25112dee87d4ecef43c420`

这个结果证明的是 **artifact integration failure**，不是数值门失败，也不是模型已失败。原工件必须永久保留，禁止覆盖、续写或重标为 V5H12。

### 3.2 根因 1：7 字段 observer 与 8 字段 parser

冻结 coupling observer 在
`platform/forward_flight_benchmarks/fluxv_v5h11_baik_coupling.py`
的 `_StageFieldComposer.observe` 中产生 exact 7 字段：

1. `fd_physical_evaluation_sha256`
2. `fd_tracer_evaluation_sha256`
3. `h_convective_over_sigma`
4. `h_jacobian_frobenius`
5. `source_state_sha256`
6. `stage`
7. `substep`

`invariant_residual_over_slog_max` 在 observer 返回之后由冻结 stream 计算，并作为
`IRWRK3StreamStageRecord` 的字段进入 record SHA 和 stage chain。V5H11 executor 却错误地要求 observer JSON 也包含该第 8 字段，因此首条 compact record 必然失败。

正确所有权是“7 + 1”：7 个 observer 字段来自 payload，normalized invariant 只从已验证的 compact record 读取，禁止复制进 payload、猜测或从已丢弃数组反推。

### 3.3 根因 2：source/stage durable 顺序

V5H11 `emit_completed_layer` 当前顺序是：

`stage rows -> commit_completed_layer(source + layer bundle)`

runner sink 的协议却要求依赖 stage 之前已经有 durable source parent。修完根因 1 后，首条 stage append 会立刻因为缺 parent 而停止。

V5H12 必须采用：

`source exactly once -> stage rows -> completed-layer bundle verifies same source`

相同 source 的第二次 append 也必须拒绝；completed-layer transaction 只能验证预提交 source 的 canonical bytes，不能再次追加。

### 3.4 根因 3：STOP 坐标丢失

conversion 在第一条记录失败时，V5H11 formal-A 得到全空终点。V5H12 冻结的更强合同是：

`(N, layer, source_step, ptera_step, substep, stage) = (32,1,4,3,1,1)`，
且 `stage_began=false`。

若已经成功持久化若干 stage，则 STOP 必须保留其 prefix，并指向下一条未开始的 record。例如第 1 条成功、第 2 条转换失败时，坐标应为 `(32,1,4,3,1,2)`。

这只适用于 conversion phase。generic solver 在只产生 source、尚未进入 conversion 时的 source-only terminal 语义必须保持不变。

V5H12 在 executor、runner 和 tests 中必须共用以下唯一字面量，不得各自选择可自洽的新名称：

- conversion stop code：`stage_evidence_conversion_error`
- conversion phase：`artifact_stage_conversion`

这两个值只覆盖 source 已 durable 后的 compact-record parse、row conversion 或 completed-stage append 失败；source parent 自身尚未 durable 的失败不准冒充该 phase。

## 4. 不可修改的历史控制

以下 8 个 V5H11 文件是 immutable controls：

| 文件 | SHA-256 |
|---|---|
| `platform/forward_flight_benchmarks/run_fluxv_v5h11_baik_w2.py` | `1cb6e0d4616ddd921f0704422756b2a470ea007ccffb3088040c575f24fd580f` |
| `platform/forward_flight_benchmarks/fluxv_v5h11_baik_w2_executor.py` | `469c12f9acabde173eae07036bb8b91e50b190876fae12795dc2ab0a3bd121f5` |
| `platform/forward_flight_benchmarks/fluxv_v5h11_baik_coupling.py` | `37b0d13ec61ecfbaa4d5455c326a0b21fe61d200f13436574be89cf6e16d8962` |
| `src/fluxvortex/rvpm_ir_wrk3_stream.py` | `3565c698f8270bf211af048b84400ce1097d9d85e4ccda225520ce5975aa4018` |
| `platform/tests/test_run_fluxv_v5h11_baik_w2.py` | `c3a92dd45bc90715c7fbb848473fb6a61bed65d5b877c658a43b36456c428eb8` |
| `platform/tests/test_fluxv_v5h11_baik_w2_executor.py` | `7db31cad849cb93d827cfdd9908e29f91adc80c79c6b7e9b02df18432f93d165` |
| `platform/tests/test_fluxv_v5h11_baik_coupling.py` | `f163ab0179c95a87c3a0082b3445a0d3508efc20e38c73243124ecb65ebfbe1b` |
| `tests/test_rvpm_ir_wrk3_stream.py` | `2009ba128827915ac2eda515f3f8823852232ac001a7a0c5f19856353b27e464` |

禁止通过修改这些文件“让测试通过”。工作树还含其他用户/历史改动；不要 reset、checkout、清理或格式化无关文件。

## 5. 已完成内容

### 5.1 治理包

| 文件 | SHA-256 |
|---|---|
| `PLAN.md` | `346063da9029d3a15dcaae99f96725e2bac96a5912671a3471fabf0268b2335f` |
| `CHECKLIST.md` | `53e27c67e1a59c2f44cfe1cabfa0877bcc8105cbdd6b0b7a50e1cfc090d02ca4` |
| `EXPERIMENT_TRACKER.md` | `5446bdcdfb1e4db915001fcba4dcc399e9311cb35566aeb0659b7e32e72ca9d1` |
| `FREEZE_INPUTS.json` | `712f4e504f4c1d3f3ca87b84f0f98012e9066eeda94e3b2aefcf383aceab11da` |

已独立重哈希过治理、历史 controls 和 formal-A 工件，全部匹配。V5H12 四个目标文件在 governance amendment 时均不存在。

### 5.0b P1-P4 证据闭环记录（2026-08-16，同日续）

- **P1（H3 exactly-once）CLOSED**：`commit_completed_layer` 的隐式 source
  append fallback 已删除（硬性 pre-append 要求 + 无条件 canonical bytes 一致
  守卫）；6 个定向测试全 GREEN（pre-commit 拒绝且 sink 不变、identical /
  inconsistent duplicate 拒绝、source→96 stages→commit 后 source_count=1、
  source 失败阻断 + clean retry、同 key 不同 bytes 拒绝）。说明：
  commit-without-source 从空 sink 测试，因为公共 ABI 下"96 stages 无
  source"不可达（每条 stage append 先被拒）；commit 后预 append 下一层
  source 是合法路径并已断言。
- **P2（H4 联合回归）V5H12 范围 CLOSED**：两个 V5H12 测试文件加入
  sys.modules 隔离 fixture（`pristine_audited_modules` /
  `restore_runtime_modules_after` / autouse `_pristine_audited_runtime_modules`），
  未触碰 runner/executor 任何 gate。V5H12 executor+runner 同进程
  **84 passed / 0 failed**；逐文件 fresh-process 33/51/20/42/11/25 全 PASS。
  字面 §9.3 六文件命令仍失败 12 个测试，全部位于冻结 V5H11 文件；仅冻结
  V5H11 三文件（executor+runner+coupling，无任何 V5H12 文件）即可复现同样
  12 个失败——冻结文件之间的 collection-time 结构性干扰（coupling 测试模块
  顶层真实 import vs. executor origin-attest / runner runtime-inventory），
  4 文件授权范围内不可修复。**需要显式 amendment**（选项：按"逐文件独立
  进程"重定义联合命令 / 授权 conftest.py / 解冻 V5H11 测试），amendment
  获批前 smoke 保持 BLOCKED。
- **P3（H6）CLOSED**：可追踪工件 `H6_FRESH_HOSTILE_AUDIT_20260816.md`
  （SHA `4801356299d0270d77a83b05346a128e7b864cc209a20c5f61474f828eea5e9b`）。
  Round-1 reviewer（agent_1c5b46a9）FAIL：发现 must-fix——序列化自校验
  （`verify_artifact` 路径）未转发 `stop_code`，首条 stage 的 conversion
  STOP 发布时崩溃。已修复（两个校验调用点对称转发）+ mutation 验证
  （移除修复 → 定向 publish 测试在 runner:3791 失败；恢复 → 通过）+
  新增 `test_first_stage_conversion_stop_publishes_and_reverifies_byte_identically`
  与 `test_commit_rejects_same_key_source_with_different_canonical_bytes`。
  Round-2 reviewer（agent_65ccac08）对最终叶 **PASS（无 must-fix）**；攻击
  矩阵命令与 stdout 摘录嵌入工件。
- **P4（H7 依赖重签）EVIDENCE COMPLETE**：新 manifest+token 位于
  `/tmp/fluxv-v5h12-audit-20260816-4W8c03/`（41 leaves + 56 runtime modules，
  绑定最终叶）；fresh 进程 `_verified_dependency_audit` PASS；
  `_load_formal_executor` 全链路（加载 executor + attest + observed 捕获）
  PASS；旧 V5H11 token fail closed（`DependencyFreezeError: dependency audit
  token schema_id is invalid`）。
- **最终叶 SHA**：runner `5e0777d82147827a0ebcd9520f3a6cfdade592bc392d3306ac77e7a9085f05fe`、
  executor `5c74a9ff…7053`（round-1 后未变）、runner test `e8b3de12…b2fa`、
  executor test `3990bb32…5b07`（未变）。冻结 V5H11 八控制 + formal-A 工件
  全程无漂移（round-2 reviewer 逐一 MATCH）。
- **下一节点**：等待 §9.3 联合命令治理 amendment；获批后运行 fresh
  disposable `N32/layer1` smoke（G2-002），随后 formal A/B。GT/scorer 保持
  sealed。

### 5.0c 执行链记录（2026-08-17，V5H12 终局）

1. **Amendment**：`AMENDMENT_JOINT_REGRESSION_20260817.md`（选项 1：逐文件
   独立进程）。批准依据：所有者指示推进。G1-006 在 amendment 下 PASS。
2. **Token 重签（最终）**：`/tmp/fluxv-v5h12-audit-20260817-GF3WxC/`
   （41 leaves + 201 runtime modules，目录全量枚举；56 条版本在 smoke 首跑时
   被 `pterasoftware.steady_horseshoe_vortex_lattice_method` 动态导入打出
   缺口后废弃）。新 token verified；`_load_formal_executor` 闭链 PASS。
3. **H8/G2-002 smoke PASS**（`/tmp/fluxv-v5h12-smoke-20260817-8FY4Wi/`）：
   96 stages、192/192/576/768 账本、max invariant 0.0 ≤ 1.14e-13、
   no-penetration 1.04e-17、Kelvin 2.83e-19、parent before==after、
   observation none、observed 52 ⊆ declared 201、canonical SHA 复算一致
   （summary SHA `8a10eb4c…`，独立审计 JSON 落盘）。
4. **H9/G3-001A formal A = 数值门 STOP**（`/tmp/fluxv-v5h12-formal-A-20260817/`，
   12 文件，verify_artifact 只读复验 PASS）：
   - stop_code `ir_wrk3_stream_stopped`，
     `FloatingPointError: h*max|U-U_gal|/sigma exceeded the B3 gate`；
   - 终点 `(32,3,6,5,1,1)` phase `stream:observer` stage_began=true；
   - durable prefix：2 完整层 + 193 stages + layer-3 source；194 行
     transport_stages（193 completed + 1 failed 行）；
   - 语义：执行修复目标（协议层）全部达成——7+1 解析、source 先行、
     conversion/stream STOP 精确坐标、发布自校验全通过；V5H11 的协议根因
     已消除。本次 STOP 是**科学门**（B3 对流稳定性，N32 layer3 首阶段）。
5. **formal B 依治理禁止**（A 非 PASS）；P7 outer gate 无预注册实现，
   且 inner 门未过，paper-data unlock 不可达。GT/scorer 保持 sealed。
6. **V5H12 结论**：execution-repair 成功；inner formal convergence 未达成，
   原因是 layer-3 对流稳定性门。继续走向论文复现需要**新的预注册分支**
   处理该科学问题（sigma 演化/网格/步进策略等），禁止在本分支调参挽救。

### 5.0a P0 状态修正（2026-08-16，依外部审计）

外部审计 `PAPER_REPRODUCTION_STATUS_AND_NEXT_PLAN_20260816.md` 判定 §5.0 的
"完成"结论过早。修正后状态：H4 = FAIL（HANDOFF §9.3 同进程联合命令实测
V5H12 4 failed/73 passed，含 cross-module 测试引入的第 4 个失败）；H6 =
UNVERIFIED（无可追踪审计工件）；H1 = PROVISIONAL（无 RED 时间顺序快照）；
H3 = WARN（commit 隐式 source append fallback 未删、定向闭环缺失）；H7 保持
BLOCKED。后续闭环：P1 删 fallback+5 个定向测试；P2 测试生命周期内隔离/
恢复 sys.modules 使 §9.3 全绿（不得放宽 runner runtime-inventory 门）；
P3 生成 `H6_FRESH_HOSTILE_AUDIT_20260816.md` 可追踪工件；P4 重签。§5.0 及
§5.0 内引用的 CHECKLIST/TRACKER 哈希自此视为 historical。

§5.1 表内治理包哈希为初始冻结值（historical）；当前文件哈希以文件系统
实时计算为准，接手者必须自行重哈希。

### 5.0 H0-H6 完成记录（2026-08-16，接手 agent）

- H0：全部冻结哈希复验 PASS（治理包、8 个 V5H11 controls、formal-A 工件、V5H12 中间态与 5.2 节一致）。
- H1（RED 证据，均为只改测试时采集）：
  1. 7/8：`test_actual_coupling_record_carries_exact_seven_observer_fields_and_parses` 对真实 `transport_v5h11_committed_layer` N=1 记录抛 `ExecutorContractError: compact stage evidence field set drift`；
  2. 顺序：一次性 harness 证明中间态 executor 在 96 次 stage append 中没有任何独立 source append（source 只在最终 commit 内隐式发生）；
  3. 坐标：`test_conversion_stop_terminal_preserves_unbegun_next_stage_coordinate` 抛 `ValueError: unbegun STOP terminal is not the exact durable-prefix coordinate`（期望 (32,1,4,3,1,1)，实际要求 (32,None,4,3,None,None)）；executor 侧转换失败只抛裸 ExecutorContractError。
- H2/H3（GREEN）：7+1 正门 + 负矩阵（None/int/NaN/Inf/负/nextafter-over-gate/等门 bitwise/第 8 字段 payload/重复 JSON key）；source→96 stages→commit 顺序；source append 失败时 stage calls=0；第 1/第 2 条 record STOP 坐标 (32,1,4,3,1,1)/(32,1,4,3,1,2)。
- H4：六 suite 逐文件 fresh-process 全 PASS（33+44+20+42+11+25）；8 个 V5H11 SHA 全程无漂移。
- H5：py_compile/Black/Ruff/git diff --check（含 no-index 逐文件）全干净。
- H6：fresh hostile 只读 review PASS（无 must-fix；两个 should-note 已记录进 tracker）。
- 最终 SHA：runner `063725d1…d282b`、executor `5c74a9ff…7053`、runner test `8612d52b…d4a9`、executor test `b155af9a…fbd6`。
- 已知预存在问题（非本分支缺陷）：同进程先跑 executor suite 再跑 runner suite 会使后者 3 个 synthetic-token 测试报 `dependency_drift`（`_capture_observed_runtime_modules` 拒绝未 manifest 的已加载模块；冻结 V5H11 对同样复现）。联合回归须逐文件独立进程执行。
- 下一节点：H7（G2-001）——冻结四个 V5H12 叶新 SHA、生成新 manifest/token、证明旧 token fail closed；之后 H8 disposable smoke。

### 5.2 此前 V5H12 中间态（历史记录）

四个新文件已经由 V5H11 机械 fork 出来，但 **只完成命名空间切换，未完成行为修复**：

| 文件 | 当前中间态 SHA-256 | 状态 |
|---|---|---|
| `platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py` | `cc046d80417bdfc123fc780295bc7c5a01b15c308e3bff72504d10d47f8995f5` | artifact/API/dependency namespace 已改为 V5H12；STOP validator 未修 |
| `platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py` | `b4feea443408a4663e9307a5a2cd70932a9956739d9019504266432380e22da8` | factory/API/smoke namespace已改；7+1/source顺序/STOP 未修 |
| `platform/tests/test_run_fluxv_v5h12_baik_w2.py` | `1eab687cd59f28169447327b0007e245178126ce2139c255d556b2f1bbdd593a` | 仅机械更新模块路径/工厂/部分 schema；没有新增 V5H12 红测 |
| `platform/tests/test_fluxv_v5h12_baik_w2_executor.py` | `dbaa3d310561e80397eb002b540e89c1de4d11fa433afcfd3d09280d50ba7ac4` | 仅机械更新模块路径/工厂；仍使用错误的 8 字段 synthetic fixture |

新 runner/executor 的 `py_compile` 在这一中间态通过；两份新测试尚未在 fork 后运行，真实跨模块红测尚未落盘。

## 6. 允许的最终代码范围

只能修改以下 4 个新文件：

- `platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py`
- `platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py`
- `platform/tests/test_fluxv_v5h12_baik_w2_executor.py`
- `platform/tests/test_run_fluxv_v5h12_baik_w2.py`

可以更新本目录的 `CHECKLIST.md`、`EXPERIMENT_TRACKER.md` 和本 handoff 以记录证据，但不得回写或改写冻结的 `PLAN.md`/`FREEZE_INPUTS.json`；若发现治理合同本身错误，必须先形成显式 amendment，再实现。

## 7. 后续开发计划

### 阶段 A：恢复现场与冻结复验

1. 读取第 2 节所有文件。
2. `git status --short`，确认只处理允许路径；保留其他脏改动。
3. 使用 `FREEZE_INPUTS.json` 重新计算所有 immutable V5H11 文件和 formal-A 文件 SHA。
4. 比较 V5H12 与其 V5H11 mechanical source 的差异，确认当前只有本 handoff 第 5.2 节所述 namespace 差异。

任何 frozen hash 漂移均立即 STOP，不得继续实现。

### 阶段 B：先补 RED，不改生产语义

先只改两份 V5H12 tests：

1. 把 executor-local `_stage_record` fixture 改为真实 7 字段 payload；normalized 只放在 record attribute。
2. 用公开 `transport_v5h11_committed_layer` 的 synthetic `N=1` 路径生成真实 coupling result，取一条 actual compact record：
   - payload key 必须 exact 7；
   - schema 必须仍为 `v5h11-stage-fd-stability-v1`；
   - coupling/stream validator 必须先通过；
   - 送入 V5H12 parser 时，当前中间态应复现 `field set drift`。
3. 加 fake sink 记录调用顺序，证明当前 executor 的行为是 `stage` 早于 `source`。
4. 加 conversion failure fixture，证明当前 runner/executor 组合不能保留 `(32,1,4,3,1,1)`。

必须先保存这三个预期 RED 的测试名、失败摘要和命令。不要为了让 RED 消失而修改 frozen coupling/stream。

### 阶段 C：最小 executor 修复

在 V5H12 executor 中：

1. 将 `STAGE_EVIDENCE_FIELDS` 改为 exact 7 字段，建议重命名为 `OBSERVER_EVIDENCE_FIELDS`。
2. `parse_stage_evidence` 继续严格验证：
   - schema；
   - payload exact bytes 类型；
   - payload SHA；
   - evidence SHA；
   - strict canonical ASCII JSON；
   - exact field set；
   - source/substep/stage 与 record 一致；
   - FD/source SHA 为 64-hex；
   - `h_jacobian_frobenius`、`h_convective_over_sigma` 的 hex Float64。
3. normalized invariant 只读
   `record.invariant_residual_over_slog_max`，要求：
   - `type(value) is float`；
   - finite；
   - `value >= 0`；
   - `value <= 512 * eps(Float64)`。
4. 返回给 artifact row 的 normalized 值必须与 record value 相同，payload bytes、payload SHA、evidence SHA 不得改变。
5. `emit_completed_layer` 必须：
   - 先冻结 `source_row = capture.source_row()`；
   - 调用 sink 的 source append exactly once；
   - 再逐条转换/追加 stage；
   - 最后 completed-layer transaction 只验证同一 source bytes。
6. 转换或 stage append 异常时，包装成 injected `stop_constructor`：
   - stop code 固定为 `stage_evidence_conversion_error`；
   - phase 固定为 `artifact_stage_conversion`；
   - coordinate 是当前 record 的 `(N,layer,source,ptera,substep,stage)`；
   - `stage_began=false`；
   - 不追加 failed scientific stage row。

不要改变 stream/coupling 的 hash 或字段。

### 阶段 D：最小 runner 修复

只为 conversion phase 增加 validation 分支：

- 已有 durable source、没有 failed stage row、`stage_began=false` 时，可接受由 executor 提供的“下一未开始 stage”六坐标；
- 坐标必须等于 `EXPECTED_STAGE_KEYS[len(completed_stage_rows)]` 及对应 source/Ptera step；
- 仅 `stage_evidence_conversion_error` + `artifact_stage_conversion` 可走此分支；
- generic source-only solver STOP 仍使用现有 `(N,None,source,ptera,None,None)`；
- 其他 phase 不得借此放宽 terminal 校验。

runner 仍须在 invalid executor terminal 时回退到 last-good publishable STOP，且不能删除已 durable prefix。

### 阶段 E：完整 GREEN 与 fresh audit

依次运行第 8 节的检验节点。通过前不得生成新 dependency manifest/token，不得构建正式或 disposable-smoke Ptera case；治理要求重跑的冻结 small-N focused test 不在此禁令内。

### 阶段 F：依赖重签与 smoke

只有 G1 全部通过后：

1. 冻结 4 个 V5H12 叶的新 SHA。
2. 生成新的、无自哈希环的 dependency manifest 和 external audit token。
3. 证明旧 V5H11 token 对新叶 fail closed。
4. fresh process 运行一次 disposable `N32/layer1` smoke；无 sink、无 formal artifact、无 GT/scorer。
5. 对 smoke summary 做独立 canonical SHA、计数、有限性、阈值、parent before/after 与 observed runtime inventory 审计。

V5H11 的旧 smoke 不能替代 V5H12 smoke，因为 runner/executor leaves 已变化。

### 阶段 G：正式 A/B 与论文门

只有 fresh smoke PASS 后才允许：

1. fresh formal A：完整 `32/64/128 x 3 layers`；
2. A 必须完整 PASS，否则保存 12-file exact-prefix STOP，禁止运行 B；
3. fresh formal B：独立进程、独立目标路径；
4. A/B 前 9 个 semantic files 逐字节相同，UUID/UTC/path/replicate 不同；
5. fresh read-only artifact audit。

即使 V5H12 A/B 全部通过，仍只完成 inner execution/convergence。之后还需要父计划的 moving-parent/source-release outer gate、fresh A/B artifact closure 和 scorer unlock token，才能接近论文数据校验。GT/scorer 在这些门之前始终 sealed。

## 8. 接手开发效果检验节点

| 节点 | 进入条件 | 必做检验 | 通过标准 | 失败动作 |
|---|---|---|---|---|
| H0 现场完整 | 开始接手 | frozen SHA、formal-A hash、allowed-path status | 全匹配；历史文件相对 FREEZE SHA 无漂移（不要求整个工作树 clean） | STOP，报告漂移 |
| H1 RED 可复现 | 仅测试发生变化 | actual 7-field coupling record、stage-before-source、conversion terminal 三个红测 | 失败原因分别精准落在 8/7、顺序、坐标；不是 import/fixture 错 | 修测试，不改科学文件 |
| H2 Parser GREEN | H1 有效 | 7+1 正门和 strict negative matrix | 正门过；missing/extra/duplicate/noncanonical/tamper/NaN/Inf/negative/overgate 全拒绝 | V5H12 STOP 或仅修 parser |
| H3 Transaction GREEN | H2 过 | source→stage→layer 顺序；duplicate/inconsistent source；第1/第2 record STOP | source exact once；prefix/next coordinate 精确；generic source-only 不变 | 仅修 executor/runner 协议 |
| H4 Regression GREEN | H3 过 | 新 V5H12 两套 + 冻结 V5H11 runner/executor/coupling/stream tests | 全 PASS；V5H11 8 个 SHA 未变 | 不重签、不 smoke |
| H5 Static/差分 | H4 过 | py_compile、Black、Ruff、diff-check、V5H12↔V5H11 normalized diff 人审 | 无格式/静态错；差异仅 namespace + prereg permitted behavior/tests | 收窄改动 |
| H6 Fresh audit | H5 过 | 新进程只读 hostile audit | 无 must-fix；7+1/source/STOP 攻击均 fail closed | 保持 G2 blocked |
| H7 Dependency closure | H6 过 | manifest/token paths+hashes+runtime inventory、old-token rejection | 新 token verified；旧 token拒绝；无 back-edge/self-cycle | `dependencies_unbound` STOP |
| H8 Disposable smoke | H7 过 | fresh real N32/layer1；独立 summary SHA/计数/门 | 所有 inherited mechanics gates PASS；read count 0；无 formal files | 结束 V5H12，不调参 |
| H9 Formal A | H8 过 | 完整 A 或 exact-prefix STOP | 只有完整 PASS 才可进 B | 保存 STOP；禁止 B |
| H10 Formal B/A-B | H9 过 | 独立 B、9 semantic byte parity、provenance difference | 全部成立且 fresh audit PASS | 不解封论文数据 |
| H11 后续论文门 | H10 过 | inherited outer convergence/B4、unlock audit/token | 全部通过才允许 GT/scorer | 保持 sealed |

## 9. 建议命令

所有命令从 `/tmp/fluxv-v5-nextgen` 执行；使用 fresh `/tmp` cache，避免仓库写缓存。

### 9.1 冻结复验

```bash
FREEZE=docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/refine-logs/v5h12_execution_repair/FREEZE_INPUTS.json
jq empty "$FREEZE"
```

随后遍历 `parent_governance_sha256`、`immutable_v5h11_historical_leaves` 和 `formal_a_stop.semantic_file_sha256`，逐项用 `sha256sum` 比较；另将 `formal_a_stop.run_manifest_file_sha256` 对应 `run_manifest.json`、`formal_a_stop.summary_file_sha256` 对应 `summary.json`、`formal_a_stop.sha256sums_file_sha256` 对应大写文件名 `SHA256SUMS`。不要只相信 JSON 内的自述。

### 9.2 RED/GREEN focused

测试名落定后，先单跑真实跨模块用例，再跑两份新 suite：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_CACHE_DIR=/tmp/fluxv-v5h12-numba \
MPLCONFIGDIR=/tmp/fluxv-v5h12-mpl \
PYTHONPATH=src:platform \
pytest -q \
  platform/tests/test_fluxv_v5h12_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h12_baik_w2.py
```

### 9.3 冻结控制联合回归

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_CACHE_DIR=/tmp/fluxv-v5h12-numba-joint \
MPLCONFIGDIR=/tmp/fluxv-v5h12-mpl-joint \
PYTHONPATH=src:platform \
pytest -q \
  platform/tests/test_fluxv_v5h12_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h12_baik_w2.py \
  platform/tests/test_fluxv_v5h11_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h11_baik_w2.py \
  platform/tests/test_fluxv_v5h11_baik_coupling.py \
  tests/test_rvpm_ir_wrk3_stream.py
```

这一步可能导入 Ptera 定义，但不得启动正式 solver/matrix；若 test selection 意外包含真实 formal N，立即停止并收窄选择。

### 9.4 静态门

```bash
PYTHONPYCACHEPREFIX=/tmp/fluxv-v5h12-pycache python -m py_compile \
  platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py \
  platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h12_baik_w2.py \
  platform/tests/test_fluxv_v5h12_baik_w2_executor.py

black --check \
  platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py \
  platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h12_baik_w2.py \
  platform/tests/test_fluxv_v5h12_baik_w2_executor.py

ruff check \
  platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py \
  platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py \
  platform/tests/test_run_fluxv_v5h12_baik_w2.py \
  platform/tests/test_fluxv_v5h12_baik_w2_executor.py

git diff --check
```

四个 V5H12 目标目前是 untracked，普通 `git diff --check` 不会覆盖它们；还要逐文件运行 `git diff --no-index --check /dev/null <file>`。无 whitespace diagnostics 才通过；该命令仅因“新文件内容不同”返回 exit 1 是正常现象，不能把 exit code 1 本身误判为格式失败。

## 10. 必须补齐的测试矩阵

### Parser/evidence

- actual coupling record 的 payload exact 7 fields；
- normalized 只来自 compact record；
- normalized 写入 artifact row 时 bitwise 相同；
- payload bytes/SHA/evidence SHA 不变；
- missing、extra、重复 JSON key、noncanonical JSON 拒绝；
- payload source/substep/stage drift 拒绝；
- record normalized 为非 exact-float、NaN、Inf、负数、nextafter-over-gate 拒绝；
- equal gate 接受；
- 仅改 record normalized 而不重算 record/hash 时，stream validator 拒绝。

### Source/order/transaction

- success 顺序为 source→3N stages→atomic layer；
- source canonical bytes 与 completed-layer 传入的 source 完全相同；
- duplicate identical source 拒绝；
- inconsistent duplicate source 拒绝；
- source append 失败时 stage calls=0；
- stage conversion 第 1 条失败时 source prefix=1、stage prefix=0；
- 第 2 条失败时 source prefix=1、stage prefix=1；
- clean retry 使用 fresh sink 成功。

### STOP

- first conversion failure：`(32,1,4,3,1,1), stage_began=false`；
- second conversion failure：`(32,1,4,3,1,2), stage_began=false`；
- 两者必须使用 code=`stage_evidence_conversion_error`、phase=`artifact_stage_conversion`；
- 无伪造 failed stage row；
- durable prefix 保留；
- generic source-only stop 仍为历史 partial coordinate；
- hostile wrong executor coordinate 被 runner替换为 last-good publishable STOP。

## 11. 明确禁止

- 不读取论文 observation/GT，不导入或调用 scorer。
- 不运行正式 N32/64/128 matrix，直到 H7 通过。
- 不用旧 V5H11 token 启动 V5H12。
- 不修改任何 V5H11 文件。
- 不把 normalized invariant 加回 observer payload。
- 不用 reverse engineering、floor、clip、projection、rebase 或阈值放宽“修复”结果。
- 不用 smoke 选择参数，也不把 smoke 描述为论文精度证据。
- 不 overwrite 任一已有 artifact 目录。
- 不清理或覆盖用户的其他脏工作树内容。

## 12. 接手者第一条进度报告模板

```text
已读取 V5H12 PLAN/CHECKLIST/TRACKER/FREEZE/HANDOFF；冻结哈希复验为 <PASS/FAIL>。
历史 V5H11 8 个控制文件 <无漂移/列出漂移>；formal-A 工件 <无漂移/列出漂移>。
当前仅接管 4 个 V5H12 允许路径；GT/scorer/formal matrix 保持 sealed。
下一节点：H1 RED，目标是用 actual coupling compact record 复现 7/8 字段、source顺序和conversion坐标三个失败。
```

## 13. 完成 handoff 的判定

下一位 agent 只有在以下证据齐全时才能宣告“接手开发完成”：

1. H0–H6 全部 PASS；
2. 4 个 V5H12 最终源码/测试 SHA 已记录；
3. 8 个 V5H11 immutable SHA 仍逐字匹配；
4. RED 证据与 GREEN 证据均有命令和摘要；
5. fresh reviewer 明确给出无 must-fix 的 scoped PASS；
6. tracker 只更新到真实达到的节点。

H7–H10 属后续执行授权，不能因代码测试通过而自动勾选。论文数据校验仍需 H11，当前不在解封状态。
