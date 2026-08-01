# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v3

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取任何 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改

## 0. 不可删的协议演化

- v1 SHA
  `fe956d511f5cb608e9e6f35535aff134aac51b059e8fa2cb50ff7c788c0526fd`：
  零数据审计 NO-GO；伪三权重、support 独立性错误、shape/guard 未定义。
- v2 SHA
  `acfba78b98aab21929fdbddd0c034f310dbcdab9200a1dd6c7d3562d746f0a0b`：
  第二轮零数据审计 NO-GO；平台反例、方向合并、两 PF 稳健性退化、guard
  不完备、外部根不独立。

两版均保留。v3 的任何后续修订必须发生在 fresh 完成和精度读取之前，并产生新
版本，禁止原地改门。

## 1. 不可自证的外部冻结根

selector 实现和全部零数据攻击测试完成后、fresh151 完成前，必须用隔离临时
Git index 创建一个不移动当前分支的 evidence commit，并建立：

```text
refs/fluxv-evidence/fig171819-active-disease-v3
```

该 commit 的 tree 至少包含：

- 本 v3；
- parent attribution protocol v2；
- selector、Prepare、Evaluate 实现与测试；
- `fig171819_confirmed_compare.py` 与测试；
- `fig171819_benchmark.py` 与测试；
- `fig171819_residual_fingerprint.py` 与测试；
- `platform/docs/data.md`；
- fresh postprocess authorization 与 preregistration；
- active-disease execution authorization JSON。

evidence commit SHA 必须在 fresh 完成前：

1. 写入用户可见的 Codex commentary；
2. 镜像到 active goal contract 和 append-only `MANIFEST.md`。

运行命令必须显式传入这个已公布的 commit SHA。selector/Prepare/Evaluate
必须用 `git show <commit>:<path>` 逐项取得权威 blob，并验证当前文件字节一致；
只读取工作树 authorization 或让 generator 回写自己的 SHA 均无效。ref 后续即使
被移动，也必须按已公布的 commit SHA 验证。

## 2. 数值输入硬门

只接受 `fig171819_confirmed_compare.py` 从同一 complete fresh151 bundle 原子
生成并由 execution receipt 绑定的 scorecard、residual artifact 和 fingerprint：

- 42 confirmed curves、434 raw measurements、151 solver conditions；
- 34 physical families、8 alias groups；
- Fig19(c,d) 零泄漏；
- measurement force 不插值；
- complete-resume、固定 V4.1 graph/call/runtime/guard/source closure 全过；
- 所有路径、输入和源码 SHA 与 evidence commit/authorization 一致。

任一失败输出第11.1节 `INVALID_EVIDENCE`，且不得生成 active/no-decision
科学产物。旧 mixed/full184、旧118和 partial sweep 不得补齐。

所有输入数组和派生量必须先通过：

```text
isfinite == true
shape/length == frozen contract
x strictly increasing
```

## 3. 固定阈值

\[
\tau_F=0.15\ {\rm N},\qquad
\tau_c=\tau_F\lVert c\rVert_1=0.30\ {\rm N},\qquad
\delta_{\rm rank}=0.15\ {\rm N}.
\]

- `tau_F`：既有单值跨进程数值复现门；
- `tau_c`：本协议所有两点零和 contrast 的传播门；
- `delta_rank`：候选 disease 排名的实践效应门。

它们都不是实验置信界。所有可辨识和排名边界使用严格 `>`；等于门限即不可辨识。

## 4. 纯实验 shape 分类

对一条 official curve，按原始测量横坐标取实验值
`e[0],...,e[n-1]`。本节不读取模型或 contribution。

定义：

```text
max_drawdown(a,b) = max_{a <= i < j <= b}(e[i]-e[j])
max_rebound(a,b)  = max_{a <= i < j <= b}(e[j]-e[i])
```

只有 `>tau_c` 的累计反向变化是 material reversal。

### 4.1 类型特异的候选

`PEAK` 只检查最大值：

1. 全局最大值在 `1e-12 N` 内只有一个索引 `k_max`；
2. `0<k_max<n-1`；
3. `e[k_max]-e[0] > tau_c`；
4. `e[k_max]-e[-1] > tau_c`；
5. `max_drawdown(0,k_max) <= tau_c`；
6. `max_rebound(k_max,n-1) <= tau_c`。

固定分量：

```text
RISE    = e[k_max]-e[0]
ROLLOFF = e[k_max]-e[-1]
```

`TROUGH` 只检查最小值：

1. 全局最小值在 `1e-12 N` 内只有一个索引 `k_min`；
2. `0<k_min<n-1`；
3. `e[0]-e[k_min] > tau_c`；
4. `e[-1]-e[k_min] > tau_c`；
5. `max_rebound(0,k_min) <= tau_c`；
6. `max_drawdown(k_min,n-1) <= tau_c`。

固定分量：

```text
FALL     = e[0]-e[k_min]
RECOVERY = e[-1]-e[k_min]
```

最大值并列只使 PEAK 不合格，最小值并列只使 TROUGH 不合格；两端相同的标准峰
`[0,1,0]`、标准谷和带平头的单调 END 不得被全局 plateau 规则误杀。

### 4.2 唯一分类

1. PEAK 与 TROUGH 同时合格：
   `INELIGIBLE_COMPLEX_SHAPE`。
2. 只有一个合格：采用该类型。
3. 两者均不合格，计算 `END=e[-1]-e[0]`：
   - `END > tau_c` 且 `max_drawdown(0,n-1) <= tau_c`：
     `END_INCREASING`；
   - `END < -tau_c` 且 `max_rebound(0,n-1) <= tau_c`：
     `END_DECREASING`；
   - 其他：
     `INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE`。

全部42条曲线必须进入 classification ledger，保存每个条件值与排除理由。

## 5. Baseline 分量状态与原子 bundle

模型必须在 fingerprint 的相同 measurement x 上求值。每个预定分量 `j`：

\[
E_j=c_j^Ty_{\rm exp},\quad
B_j=c_j^Ty_{\rm V4.1},\quad
m_j=|B_j-E_j|.
\]

分类保证 `E_j` 非零。状态按固定顺序判定：

```text
PASS      : m_j <= tau_c
REVERSED  : m_j > tau_c and B_j*E_j < 0
UNDER     : m_j > tau_c and B_j*E_j >= 0 and |B_j| < |E_j|
OVER      : m_j > tau_c and B_j*E_j >= 0 and |B_j| >= |E_j|
```

所以 `B=0` 属于 UNDER，不是 REVERSED。至少一个分量非 PASS 才支持 disease。
PEAK/TROUGH 永远保持完整 bundle：错误侧全部进入待恢复集合，PASS 侧全部进入
guard；禁止拆侧建 disease。

curve severity：

\[
s_{\rm curve}={1\over m}\sum_{j=1}^{m}\max(0,m_j-\tau_c),
\]

其中 `m=1`（END）或 `m=2`（PEAK/TROUGH），PASS 分量固定计零。

Disease ID：

```text
(channel, abscissa, shape_class,
 ordered[(component_name, sign(E_j), baseline_state_j)])
```

`shape_class` 区分 `END_INCREASING/END_DECREASING/PEAK/TROUGH`。跨 PF 的极值
索引和物理位置允许随工况变化，表示同一种 failure mode 在不同 regime 复现；
但它们必须逐 curve 保存，且同一 PF 的 aliases 必须严格一致。

## 6. Canonical alpha support、alias 与独立性

measurement contrast 系数 `c_i` 经 solver 插值后：

\[
\alpha_q=\sum_i c_iw_{iq}.
\]

canonicalization 固定为：

1. condition keys 按四个数值字段排序；
2. 系数绝对值 `<=1e-12` 的抵消项删除；
3. nominal index 必须精确相同；
4. raw/evaluation x、`c_i`、interpolation weights 和 `alpha_q` 用
   `rtol=0, atol=1e-12` 比较；
5. 不允许先舍入再 hash 决定物理相等。

curve bundle effective support 是所有分量非零 alpha condition keys 的并集。

同一 PF 的所有 aliases 必须一致于：

- shape class、component names、`sign(E)` 和 baseline states；
- 极值 nominal index、raw x 和 evaluation x；
- measurement coefficients；
- 每个分量的 canonical alpha map。

任一不一致则该 PF 为 `ALIAS_SENSITIVE` 并撤票。PF support 不能通过挑选某个
alias 缩小；诊断中另保存所有 alias support 的并集。

每个 disease 建 support-conflict graph：两个 PF 的 effective support 有交集即
连边。独立复现数是该图最大 independent set 的基数，即最大 pairwise-disjoint
PF 子集。必须用确定性 exact/branch-and-bound；保存全部最大集合，PF ID 字典序
仅用于序列化。condition union 大小不参与排名。

Disease 必须满足：

```text
max_pairwise_disjoint_pf_count >= 2
```

## 7. 非退化稳健性视角与排名

一个 PF 的 severity 是其 consensus aliases 的 curve severity 算术平均。

主视角：

1. `PF_EQUAL_MEAN`：全部 consensus PF severity 的算术平均；
2. `PF_REPLICATION_FLOOR`：全部 consensus PF severity 的最小值；
3. `SUPPORT_CLUSTER_EQUAL_MEAN`：support-conflict graph 每个 connected
   component 内先平均，再让 component 等权；
4. `PF_MEDIAN`：全部 consensus PF severity 的标准中位数。

`PF_EQUAL_MEAN` 与 `PF_REPLICATION_FLOOR` 是所有 eligible disease 的两个固定
非等价视角。以下视角发生结构退化时只报告、不重复计票：

- 所有 connected components 大小相同：
  `SUPPORT_CLUSTER_EQUAL_MEAN` 标记 `DEGENERATE_WITH_PF_EQUAL_MEAN`；
- consensus PF 少于3：
  `PF_MEDIAN` 标记 `DEGENERATE_FOR_N_LT_3`。

对所有非退化视角分别排名。若 eligible disease 多于一个：

1. 所有非退化视角必须具有同一严格第一名；
2. 每个视角 winner 比各自 runner-up 高 `>delta_rank`；
3. ID 字典序不得打破数值并列。

若全局只有一个 eligible disease，runner-up margin vacuous，但独立、alias 和
shadow-LOPO 门仍必须通过。

### 7.1 Shadow leave-one-PF-out

对 winner 的每一个 consensus support PF 逐一删除；每次都必须从全局 candidate
ledger 重新计算：

- alias consensus；
- canonical alpha/support graph 与 exact MIS；
- candidate eligibility；
- 全部非退化视角、winner 和 runner-up margins。

shadow 运行仅为稳定性诊断，将独立复现门临时降为1；它不能晋升主证据。删除
任一 winner PF 后，shadow winner 必须仍为同一 disease，且剩余 winner 的
`PF_REPLICATION_FLOOR > 0`。否则
`NO_DECISION_LEAVE_ONE_PF_SENSITIVE`。

必须报告但不参与 winner 的敏感性指标：

- official-curve-equal contrast excess；
- 每条支持曲线先减去自身 residual mean 后，汇集全部 raw points 的 centered
  point-weighted absolute residual；
- PF/curve/point 数、support overlaps 和全部最大 disjoint sets。

## 8. 完整 guard 集

对 winner 每条支持曲线，Prepare 必须枚举所有 measurement index pairs
`0<=i<j<n`：

```text
E_ij = y_exp[j]-y_exp[i]
B_ij = y_v41[j]-y_v41[i]
```

凡 `|E_ij|>tau_c` 且 `|B_ij-E_ij|<=tau_c` 的两点零和 contrast 都冻结为 guard。
这自然包含合格的 global END、显著相邻变化及显著非相邻累计变化。与 disease
bundle 重复的记录按 canonical coefficient map 去重。

每个 guard 保存 indices、coefficients、E/B、canonical alpha 和门限。
Evaluate 不得新增/删除；父力擦除后必须继续满足：

```text
|D_ij-E_ij| <= tau_c
D_ij*E_ij > 0
```

## 9. G2 接口

G2 必须使用 parent attribution protocol v2，并复用本协议的 canonical alpha、
support-conflict graph、exact MIS、strict thresholds、component states 和 guards：

- REVERSED：恢复实验方向且误差改善过门；
- UNDER/OVER：保持实验方向且误差改善过门；
- 每个错误分量分别通过；
- 每个 PASS/完整 guard 继续通过；
- 最后再过整曲线 MAE 与 PF 独立复现门。

旧 parent protocol 不得消费 v3 disease artifact。

## 10. Reason precedence

所有触发理由以以下固定顺序保存到 `all_triggered_reasons`，第一个作为 primary：

1. `INVALID_EVIDENCE`
2. `NO_DECISION_NO_ELIGIBLE_ZERO_SUM_DISEASE`
3. `NO_DECISION_NO_INDEPENDENT_REPLICATION`
4. `NO_DECISION_ALIAS_SENSITIVE`
5. `NO_DECISION_DEGENERATE_ROBUSTNESS_VIEWS`
6. `NO_DECISION_WEIGHTING_SENSITIVE`
7. `NO_DECISION_DISEASE_TIE`
8. `NO_DECISION_LEAVE_ONE_PF_SENSITIVE`

`ALIAS_SENSITIVE` 只有在反事实地恢复被撤 PF 会形成 eligible disease 或改变
winner 时才触发；无关 alias 分歧只记 ledger。没有符号反转或无 eligible disease
不能推出 `OFFSET_ONLY`。

## 11. 三种互斥 schema

### 11.1 Invalid

```yaml
status: INVALID_EVIDENCE
evidence_commit_sha: ...
failed_gates: [...]
claim_decision: NO_DECISION
```

不得包含数值 disease 排名。

### 11.2 Active

```yaml
status: ACTIVE_DISEASE_FROZEN
evidence_commit_sha: ...
authorization_sha256: ...
input_bundle_id: ...
scorecard_sha256: ...
fingerprint_sha256: ...
disease_id: ...
shape_class: ...
component_signature: [...]
support_physical_family_ids: [...]
max_pairwise_disjoint_pf_count: ...
all_maximum_disjoint_pf_sets: [...]
support_official_curve_keys: [...]
contrasts: [...]
guard_contrasts: [...]
rankings: {...}
shadow_leave_one_pf_out: [...]
classification_ledger_42: [...]
candidate_ledger: [...]
exclusion_ledger: [...]
claim_decision: NO_DECISION
reason: NODE_ATTRIBUTION_REQUIRED
```

### 11.3 No decision

```yaml
status: NO_DECISION_*
evidence_commit_sha: ...
authorization_sha256: ...
input_bundle_id: ...
scorecard_sha256: ...
fingerprint_sha256: ...
all_triggered_reasons: [...]
classification_ledger_42: [...]
candidate_ledger: [...]
exclusion_ledger: [...]
rankings: {...}
claim_decision: NO_DECISION
reason: <primary status>
```

No-decision 不得伪填 active-only 字段。

## 12. 阶段边界

`select-disease` 原子写入；Prepare 只接受其回执并从 evidence commit 再次独立
复算。Prepare 可以读取 baseline receipt、measurement contract、scorecard 和
fingerprint，但不得打开、stat 或 hash contribution，也不得输出 N2/N3、候选
公式、参数或文献结论。

只有 `ACTIVE_DISEASE_FROZEN` 才允许进入 G2。任何 `NO_DECISION_*` 的下一步是
补充不改变模型的诊断证据或报告不可辨识性，禁止跳到 LESP、空间涡、rVPM、
结构、参数扫或任意候选。
