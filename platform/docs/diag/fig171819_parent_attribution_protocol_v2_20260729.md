# Fig17/18/19 父 Claim 归因协议 v2

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计或 contribution  
**阶段**：G2 父节点归因；只允许输出 active parent hypothesis 或明确
no-decision

## 0. 协议边界与演化

旧协议
`platform/docs/diag/fig171819_parent_attribution_protocol_20260729.md`
（SHA
`28a7bc629ba20c2ae9d729a0f0316663d35c61538af4eb7142f49905f3815f94`）
不能消费 active-disease v3，永久保留但判为 NO-GO。旧协议把所有失败简化为
“方向恢复”、要求峰谷两侧都失败，并以“不同 support”代替 pairwise-disjoint
support。

本协议只消费：

- active-disease protocol v3；
- 由 v3 selector 生成的 `ACTIVE_DISEASE_FROZEN`；
- 同一 complete fresh151 bundle 的贡献账本。

本阶段的 frozen-state leave-one-parent-force-out 只是加性归因诊断，不能证明
状态重算后的因果效应。因此最高结论固定为：

```yaml
causal_status: HYPOTHESIS_ONLY
claim_writeback_allowed: false
```

禁止选择子节点、修改 claim YAML、提出候选公式、调整参数或把 N6 作为方案。

## 1. 外部冻结根与输入硬门

本协议、active-disease v3、selector、Prepare/Evaluate 实现、全部零数据测试、
fresh scorer/benchmark/fingerprint 及两级 authorization 必须共同存在于
`refs/fluxv-evidence/fig171819-active-disease-v3` 所指向的隔离 evidence
commit。运行时接受已在 fresh 完成前向用户公布的 literal commit SHA，不接受
可移动 ref 作为身份。

selector、Prepare 和 Evaluate 必须分别使用：

```text
git show <evidence_commit_sha>:<path>
```

读取权威 blob，并验证当前执行源码和协议逐字节一致。工作树文件、生成器自己的
receipt、`MANIFEST.md` 或活动 goal 都不能自证冻结时间。

Prepare 只接受同时满足以下条件的输入：

- fresh manifest `status=complete` 且 failures 为空；
- result、contributions、case guards、expected keys 属于同一 confirmed151；
- scorecard/residual/fingerprint 精确为 42 curves、434 measurements、
  151 conditions、34 physical families、8 alias groups；
- Fig19(c,d) 的 8 条曲线、96 点、33 conditional-only conditions 零泄漏；
- scorer execution receipt、文件 SHA、graph/call/runtime/session identity、
  guard tolerances、source closure 和 body/wind force ledger 全部通过；
- active-disease artifact 状态严格为 `ACTIVE_DISEASE_FROZEN`，其输入 bundle、
  scorecard、fingerprint、authorization 和 evidence commit 与本次输入一致；
- 节点身份严格为 N1/N2/N3/N4/N5/N6/R0 当前生产角色，其中只有 N2、N3
  是 eligible parent。

任一输入缺失、非有限、维度错误、151 未完成、哈希漂移或 receipt 不闭合，
输出 `INVALID_EVIDENCE`，不得产生数值排名或科学 no-decision。

## 2. Prepare/Evaluate 两阶段隔离

### 2.1 Prepare

Prepare 可以读取：

- complete baseline receipt；
- measurement/benchmark contract；
- scorecard、residual fingerprint；
- `ACTIVE_DISEASE_FROZEN` 及其 selector receipt。

Prepare 必须从冻结源码独立复算 active disease 的：

- disease ID、shape class 和 component signature；
- 全部支持 physical-family IDs 和 official aliases；
- 每条曲线的 measurement indices、系数、实验 `E_j`、V4.1 `B_j` 和状态；
- canonical `alpha_q`、effective support、support-conflict graph 和 exact MIS；
- 所有 PASS 分量与完整 pairwise guard 集；
- 固定阈值和逐曲线 MAE 基线。

Prepare 不得打开、`stat`、hash、接收路径或通过异常信息探测 contribution。
输出必须先写临时文件，再原子 rename；receipt 最后写入。

### 2.2 Evaluate

只有 Prepare artifact 与 receipt 完整落档且绑定 fingerprint SHA 后，Evaluate
才允许读取 contribution。Evaluate 必须逐字段复算并验证 Prepare，且不得：

- 新增、删除或重新选择 PF/curve/component/guard；
- 修改 contrast、threshold、alias、support 或 MIS 定义；
- 根据 contribution 改变 active disease；
- 重新拟合 V4.1 状态或运行新的气动求解。

## 3. 父贡献与冻结态反事实

只评估两个父节点：

```text
N2 = separation + profile_drag
N3 = ds_vortex + vortex_normal
```

N1/N4/R0 为 frozen/validated 或簿记角色，N5 为 falsified observer，N6
为 necessary physical rig drag 但也是 dead-end；它们均不得成为 active parent。
子通道只能作为诊断分解报告，不能分别投票。

在模型求值使用的相同 measurement x 上，对每个预登记分量 `j` 和父节点 `p`：

\[
E_j=c_j^Ty_{\rm exp},\qquad
B_j=c_j^Ty_{\rm V4.1},\qquad
G_{p,j}=c_j^Ty_p,\qquad
D_{p,j}=B_j-G_{p,j}.
\]

实验力不得插值；模型和父贡献使用 fingerprint 已冻结的同一 solver 插值权重。
所有值必须有限，且从面板/条带贡献到账本总力的误差不超过 `1e-9 N`。

固定阈值：

\[
\tau_F=0.15\ {\rm N},\qquad
\tau_c=0.30\ {\rm N}.
\]

所有改善边界均使用严格 `>`；等于门限为未通过。

## 4. 分量级恢复门

每个 baseline 非 PASS 分量必须单独通过，不能以 bundle 平均掩盖失败。
定义：

\[
\Delta m_{p,j}=|B_j-E_j|-|D_{p,j}-E_j|.
\]

### 4.1 REVERSED

若 baseline 状态为 `REVERSED`，父节点仅在同时满足时恢复该分量：

```text
D_pj * E_j > 0
delta_m_pj > tau_c
(B_j - E_j) * G_pj > 0
```

第一项要求真正恢复实验方向；零不算恢复。

### 4.2 UNDER / OVER

若 baseline 状态为 `UNDER` 或 `OVER`，父节点仅在同时满足时恢复该分量：

```text
D_pj * E_j > 0
delta_m_pj > tau_c
(B_j - E_j) * G_pj > 0
```

这里 baseline 方向本来正确，因此不使用“恢复方向”措辞；要求删除该父贡献后
保持实验方向并取得超过传播门的误差改善。允许跨越 UNDER/OVER 边界，但不允许
跨成反向。

### 4.3 PASS 分量

active disease 的完整 PEAK/TROUGH bundle 中，baseline 为 PASS 的一侧必须作为
guard 保留：

```text
|D_pj - E_j| <= tau_c
D_pj * E_j > 0
```

任何非 PASS 分量未恢复，或任何 PASS 分量损坏，该 official curve 对该父节点
均为失败。不得删除不利分量或拆分峰/谷 bundle。

## 5. 完整 pairwise guard 与曲线 MAE

Prepare 从 v3 接收每条 winner-support curve 的全部冻结 guard。Evaluate 对每个
guard `g` 计算：

\[
D_{p,g}=B_g-G_{p,g}.
\]

所有 guard 必须继续满足：

```text
|D_pg - E_g| <= tau_c
D_pg * E_g > 0
```

一个父节点损坏任一支持曲线上的任一 guard，则该父节点整体不合格；不能只撤掉
该 PF 后继续投票。

此外，每条 official support curve 都必须满足固定点等权 MAE 门：

\[
{\rm MAE}_{B,q}-{ \rm MAE}_{D_p,q}>0.15\ {\rm N}.
\]

MAE 只在该 official curve 的原始 measurement points 上计算；不插值实验值，
不以曲线点数之外的权重重排。

## 6. Alias consensus、support 与独立复现

同一 PF 的所有 official aliases 必须对以下内容逐项一致：

- v3 disease 身份、component states 和 canonical alpha maps；
- N2/N3 各自的逐分量恢复、PASS/guard 和 curve-MAE 裁决；
- parent contribution 的 canonical solver support。

任一 alias 分歧使该 PF 对该父节点为 `ALIAS_SENSITIVE` 并撤票；不得选择有利
alias。PF support 是全部 aliases、全部 disease components、全部 guards 的
非零 canonical solver condition keys 并集。

一个 PF 对父节点 `p` 的 `RESTORED` 定义为：所有 aliases 的全部错误分量、
PASS 分量、guards 和 curve-MAE 门均通过。所有未恢复 PF 也必须完整保存，
禁止从 ledger 中删除。

对全部 `RESTORED` PF 建 support-conflict graph：两个 PF support 有交集即连边。
独立恢复数是 exact maximum independent set 的基数，即最大 pairwise-disjoint
PF 子集。必须保存全部最大集合；字典序只用于稳定序列化，不能打破科学并列。

父节点取得复现票必须满足：

```text
max_pairwise_disjoint_restored_pf_count >= 2
```

“不同 PF”或“support 不相等”不能替代 pairwise-disjoint。

## 7. 父节点整体改善与唯一裁决

对每个 `RESTORED` PF，先对其 official aliases 的 curve-MAE improvement
算术平均，得到 `PF_MAE_IMPROVEMENT_p`。父节点的 family-equal 改善为所有
`RESTORED` PF 的等权平均，必须严格：

```text
FAMILY_EQUAL_MAE_IMPROVEMENT_p > 0.15 N
```

父节点 `PARENT_RESTORATION_PASS` 当且仅当：

1. 没有任何 active-disease support curve 的 PASS/pairwise guard 损坏；
2. 至少两个 pairwise-disjoint `RESTORED` PF；
3. family-equal MAE improvement 超过 `0.15 N`。

固定裁决：

| N2 | N3 | 输出 |
|---|---|---|
| PASS | FAIL | `ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS` |
| FAIL | PASS | `ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS` |
| PASS | PASS | `NO_DECISION_MULTIPLE_PARENTS` |
| FAIL | FAIL | `NO_DECISION_NO_REPLICATED_PARENT_RESTORATION` |

两个父节点都失败时，只能说明现有加性父力擦除不能唯一解释病灶。若错误主要经
N2→N3 状态依赖传播，冻结态擦除也会失败。因此附加诊断可使用：

```text
NO_DECISION_MISSING_OR_STATE_MEDIATED
```

但不得直接判定“缺组件”，不得绕过后续机理文献裁决。N6 即使在只读负控制中
出现改善，也只能记录 `NO_DECISION_FORBIDDEN_PATH`，禁止成为候选。

## 8. Reason precedence 与互斥输出

所有触发理由保存为有序 `all_triggered_reasons`；primary reason 按以下优先级：

1. `INVALID_EVIDENCE`
2. `NO_DECISION_ALIAS_SENSITIVE`
3. `NO_DECISION_GUARD_DAMAGE`
4. `NO_DECISION_MULTIPLE_PARENTS`
5. `NO_DECISION_NO_REPLICATED_PARENT_RESTORATION`
6. `NO_DECISION_MISSING_OR_STATE_MEDIATED`
7. `NO_DECISION_FORBIDDEN_PATH`

`INVALID_EVIDENCE` 与科学输出互斥。有效运行只允许以下三类 schema。

### 8.1 Invalid

```yaml
status: INVALID_EVIDENCE
evidence_commit_sha: ...
failed_gates: [...]
claim_decision: NO_DECISION
```

### 8.2 Active parent hypothesis

```yaml
status: ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS | ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS
causal_status: HYPOTHESIS_ONLY
evidence_commit_sha: ...
authorization_sha256: ...
input_bundle_id: ...
active_disease_sha256: ...
prepare_sha256: ...
parent: N2 | N3
parent_component_inventory: [...]
component_results: [...]
guard_results: [...]
curve_results: [...]
physical_family_results: [...]
support_conflict_graph: {...}
all_maximum_disjoint_pf_sets: [...]
family_equal_mae_improvement_N: ...
claim_writeback_allowed: false
reason: LITERATURE_MECHANISM_ADJUDICATION_REQUIRED
```

### 8.3 No decision

```yaml
status: NO_DECISION_*
causal_status: HYPOTHESIS_ONLY
evidence_commit_sha: ...
authorization_sha256: ...
input_bundle_id: ...
active_disease_sha256: ...
prepare_sha256: ...
all_triggered_reasons: [...]
N2_results: {...}
N3_results: {...}
claim_decision: NO_DECISION
reason: <primary status>
```

## 9. 后续研究授权

只有唯一 active parent hypothesis 才授权围绕“冻结 disease + 唯一父 claim”
开展 research-pipeline 一手文献检索。文献搜索必须同时覆盖：

- 支持该父节点为错组件的直接机理；
- 缺组件与状态介导的主要竞争解释；
- 相邻可迁移的空间涡态、统一面板压力和守恒传力理论；
- 能证伪首选解释的反证来源。

文献阶段停止条件不是篇数，而是形成一个可审计的“错组件/缺组件”裁决，并导出
一个具有空间面板载荷、力/力矩/功闭合和明确 go/no-go 预测的单一候选。

任何 no-decision 的下一步只能补充不改变模型的诊断证据或正式报告不可辨识性，
禁止直接跳到 LESP、空间涡、rVPM、结构、常数扫描或候选实现。
