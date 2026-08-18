# V5H12 fresh hostile audit（可追踪工件，H6 closure）

日期：2026-08-16（UTC 见文末时间戳节）
仓库：`/tmp/fluxv-v5-nextgen`
审计对象：V5H12 execution-repair 四叶（最终冻结 SHA 见下）
方法：两轮独立 fresh hostile 只读 review（不同 reviewer agent）+ 实弹攻击矩阵命令（stdout 摘录附后）+ mutation 验证。

## 1. 审计输入身份（SHA-256，round-2 复核全 MATCH）

最终四叶（round-2 后冻结）：

| 文件 | SHA-256 |
|---|---|
| `platform/forward_flight_benchmarks/run_fluxv_v5h12_baik_w2.py` | `5e0777d82147827a0ebcd9520f3a6cfdade592bc392d3306ac77e7a9085f05fe` |
| `platform/forward_flight_benchmarks/fluxv_v5h12_baik_w2_executor.py` | `5c74a9ffe245a0212aacf06067c477c6ddcc384e1c71b97a2aa1bd017bfb7053` |
| `platform/tests/test_run_fluxv_v5h12_baik_w2.py` | `e8b3de1271cc8cffa09e6e1252a68595662a2b2328bbdc943a18bdb5b3c3b2fa` |
| `platform/tests/test_fluxv_v5h12_baik_w2_executor.py` | `3990bb32309eb69d858e768a4d476ff42e815524afdf38108aaf29314f875b07` |

冻结 V5H11 八控制（两轮均逐字 MATCH，无漂移）：见 `FREEZE_INPUTS.json`
`immutable_v5h11_historical_leaves`（值不变，此处不重复）。治理包
`FREEZE_INPUTS.json` SHA `712f4e504f4c1d3f3ca87b84f0f98012e9066eeda94e3b2aefcf383aceab11da`。

## 2. Round 1（reviewer agent `agent_1c5b46a9-2a9a-487c-807a-67342f843846`）

对 round-1 中间四叶（runner `d95a7bdb…`、executor `5c74a9ff…`、runner test
`3481b5c6…`、executor test `3990bb32…`）做只读 hostile 差分审计。

**verdict：FAIL（1 must-fix，1 should-note）**

- F1（must-fix）：`_validate_serialized_counts` → `_validate_stage_sequence`
  未转发 `stop_code`。首条 stage 的 conversion STOP（`(32,1,4,3,1,1)`）在内存
  校验通过，但发布自检 `verify_artifact` 的序列化路径只接受
  `(32,None,4,3,None,None)`，发布在 `_publish_directory_noreplace` 之前以未
  处理 ValueError 崩溃（fail-closed，无坏工件，但本修复分支的核心场景无法
  落盘，且遗留 staging 目录）。位置：runner 序列化校验调用处。
- F2（should-note）：commit 时 byte-identity 守卫
  （"precommitted source event differs from layer bundle"）无定向测试。
- 其余 hostile 检查（a)-(g)）全部通过：无 source append 残留路径、conversion
  包装不吞 pre-source 失败、sys.modules fixtures 不触碰任何 gate（仅 spy）、
  无科学字节改动、两个调整过的 P1 测试仍有效（删守卫会失败）、conversion
  分支对其他 (code,phase) 不可达、无空洞测试。

### F1/F2 修复与 mutation 验证

- F1 修复：序列化路径调用补
  `stop_code=(None if status == "PASS" else str(summary["stop_code"]))`。
- F1 定向测试 `test_first_stage_conversion_stop_publishes_and_reverifies_byte_identically`：
  synthetic executor 追加合法 source 行后以精确 `(32,1,4,3,1,1)` +
  `stage_evidence_conversion_error`/`artifact_stage_conversion`/`stage_began=false`
  stop；断言正式发布 summary 的 stop_code/terminal/row_counts
  （source=1、stages=0）并 `verify_artifact` 字节级复验通过。
- **Mutation 验证（非空洞证明）**：临时移除该单行修复后，该测试在
  runner:3791（conversion 分支 raise）失败；恢复修复后通过。stdout 摘录：
  `FAILED ...test_first_stage_conversion_stop_publishes_and_reverifies_byte_identically`（移除时）
  → `1 passed`（恢复时）。
- F2 修复：`test_commit_rejects_same_key_source_with_different_canonical_bytes`
  在合法 source+96-stage 前缀后，以同 key 不同 canonical bytes 的 source
  commit，断言 `"precommitted source event differs"` 且 sink 前缀不变。

## 3. Round 2（reviewer agent `agent_65ccac08-8e73-4590-9ea3-a294924245fd`）

对修复后最终四叶（第 1 节 SHA）重审。

**verdict：PASS (no must-fix)**

- F1 RESOLVED（转发确认在两个调用点对称；对非 conversion (code,phase) 无任何
  放宽：fallback 仍被 code+phase+stage_began=false+durable-source 四重门限定，
  `str()` 强转无法洗白类型）。
- F2 RESOLVED（定向测试钉住精确错误串）。
- Namespace 归一化对冻结 V5H11 runner 的全量 diff（98 行）仅含许可 delta 清单，
  无 `source_already_committed` 残留。
- 12 个 SHA 全 MATCH（最终四叶 + 冻结八控制）。
- informational：byte-identity 测试未断言其余表为空（结构性由守卫先于所有
  append 保证）；v5h11 域 schema 字符串保留为 schema 连续性（非 delta）。

## 4. 实弹攻击矩阵（本机执行，stdout 摘录）

命令环境：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/fluxv-v5h12-audit
MPLCONFIGDIR=/tmp/fluxv-v5h12-audit-mpl PYTHONPATH=src:platform pytest -q -p no:cacheprovider`

```text
### A. legacy 8th observer field / payload binding
1 passed in 0.07s
### B. record-owned normalized invariant hostile values + bitwise equal-gate
7 passed in 0.07s
### G. actual coupling 7-field positive control
1 passed in 1.27s
### C. source exactly-once matrix (5 directed)
5 passed in 1.03s
### D/E. conversion STOP coordinates + generic source-only + hostile coordinate
2 passed in 0.20s
2 passed in 0.07s
### F1. dependency drift: mutable leaf + alias/substitute + fake distribution metadata
6 passed in 0.46s
### F2. formal entry isolation (no Ptera before preflight)
1 passed in 0.11s
### F3. frozen V5H11 leaves + governance rehash
62d10d5d9062a9d87fa3fdd1b5ff236bb65eb93f12f7fe3793bbaa28ee61cf63  -
```

附加（P2 证据，同日）：

- V5H12 同进程 pair（executor+runner 一个 pytest 进程）：`82 passed`（修复
  serialized-path 前的计数；最终状态 `84 passed`，见 tracker）。
- 完整 HANDOFF §9.3 六文件同进程：`12 failed, 168 passed`；其中 12 个失败
  全部位于冻结 V5H11 文件。
- **冻结 V5H11 三文件单独同进程（不含任何 V5H12 文件）**：同样 `12 failed`
  （同名测试集）——证明剩余失败为冻结文件之间的结构性 collection-time 干扰
  （coupling 测试模块顶层真实 import vs. executor origin-attest / runner
  runtime-inventory 测试），在四文件授权范围内不可修复，需治理 amendment。

## 5. 结论

- 两轮独立 hostile review + 实弹攻击矩阵 + mutation 验证后：**无 must-fix**。
- 审计输入 SHA 与重签前最终四叶一致（第 1 节）。
- H6 由 UNVERIFIED 关闭为 PASS（工件即本文件；本文件 SHA 记录于
  `EXPERIMENT_TRACKER.md`）。

## 6. 时间戳

- Round 1 完成：2026-08-16（UTC，本 session）
- Round 2 完成：2026-08-16T15:48:39Z 前后（最终叶冻结时刻）
- 工件落盘：2026-08-16（UTC）
