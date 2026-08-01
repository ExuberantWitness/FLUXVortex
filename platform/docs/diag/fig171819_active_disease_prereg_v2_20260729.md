# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v2

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取任何 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改  
**取代**：v1 SHA
`fe956d511f5cb608e9e6f35535aff134aac51b059e8fa2cb50ff7c788c0526fd`
已由零数据数学审计判定 NO-GO；v1 保留，不删除、不事后改写

## 1. 外部冻结与输入硬门

执行 `select-disease` 前，必须在 fresh151 仍未完成时生成独立 authorization，
至少固定以下 SHA：

- 本 v2 协议；
- selector 实现与测试；
- `fig171819_confirmed_compare.py`；
- `fig171819_benchmark.py`；
- `fig171819_residual_fingerprint.py`；
- `platform/docs/data.md`；
- fresh postprocess authorization 与 preregistration。

authorization 的路径和 SHA 必须先写入 active goal contract 与 `MANIFEST.md`。
selector、Prepare 和 Evaluate 都必须实读并验证该 authorization；selector
实现或测试漂移即 `INVALID_EVIDENCE`。不得让 selector 用自己刚生成的回执证明
自己。

数值输入只接受 `fig171819_confirmed_compare.py` 从同一 complete fresh151
bundle 原子生成并由 execution receipt 绑定的 scorecard、residual artifact 和
fingerprint。输入必须通过：

- 42 confirmed curves、434 raw measurements、151 solver conditions；
- 34 physical families、8 alias groups；
- Fig19(c,d) 零泄漏；
- measurement force 不插值；
- complete-resume、固定 V4.1 graph/call/runtime/guard/source closure 全过；
- scorer、benchmark、measurement、fingerprint 和 postprocess authorization
  的冻结 SHA 全过。

任一失败输出 `INVALID_EVIDENCE`。不得用旧 mixed/full184、旧118或 partial sweep
补齐。

## 2. 固定阈值及其语义

\[
\tau_F=0.15\ {\rm N},\qquad
\tau_c=\tau_F\lVert c\rVert_1=0.30\ {\rm N},\qquad
\delta_{\rm rank}=0.15\ {\rm N}.
\]

- `tau_F` 是既有单值跨进程数值复现门；
- `tau_c` 是所有本协议两点零和 contrast 的传播门；
- `delta_rank` 是候选 disease 排名的实践效应门。

三者都不是实验置信区间或数字化不确定度。所有边界使用严格 `>`；等于门限时
视为不可辨识。不得把 `0.15 N` 直接施加到两点差分。

## 3. 只由实验数据确定的曲线形状

对每条 official curve，按原始测量横坐标顺序取实验值
`e[0],...,e[n-1]`。分类不读取模型值或节点 contribution。

### 3.1 Material reversal

对序列区间 `[a,b]`：

```text
max_drawdown(a,b) = max_{a <= i < j <= b}(e[i]-e[j])
max_rebound(a,b)  = max_{a <= i < j <= b}(e[j]-e[i])
```

只有超过 `tau_c` 的反向累计变化才算 material reversal；相邻点小噪声不能单独
制造 turn。

### 3.2 唯一分类伪代码

1. 若全局最大值或最小值在 `1e-12 N` 内并列于多个索引，分类为
   `INELIGIBLE_PLATEAU`。
2. 令 `k_max=argmax(e)`。若 `0<k_max<n-1`，且：
   - `e[k_max]-e[0] > tau_c`；
   - `e[k_max]-e[-1] > tau_c`；
   - `max_drawdown(0,k_max) <= tau_c`；
   - `max_rebound(k_max,n-1) <= tau_c`；
   
   则 `PEAK` 合格，固定：
   `RISE=e[k_max]-e[0]`、
   `ROLLOFF=e[k_max]-e[-1]`。
3. 令 `k_min=argmin(e)`。若 `0<k_min<n-1`，且：
   - `e[0]-e[k_min] > tau_c`；
   - `e[-1]-e[k_min] > tau_c`；
   - `max_rebound(0,k_min) <= tau_c`；
   - `max_drawdown(k_min,n-1) <= tau_c`；
   
   则 `TROUGH` 合格，固定：
   `FALL=e[0]-e[k_min]`、
   `RECOVERY=e[-1]-e[k_min]`。
4. 若 PEAK 与 TROUGH 同时合格，分类为 `INELIGIBLE_COMPLEX_SHAPE`。
5. 若两者均不合格，令 `END=e[-1]-e[0]`：
   - `END > tau_c` 且 `max_drawdown(0,n-1) <= tau_c`，分类为
     `END_INCREASING`；
   - `END < -tau_c` 且 `max_rebound(0,n-1) <= tau_c`，分类为
     `END_DECREASING`；
   - 其他情况分类为 `INELIGIBLE_NO_ROBUST_ZERO_SUM_SHAPE`。

分类 ledger 必须保存全部42条曲线及入选/排除理由，不能只保存 winner。

## 4. Baseline 失败状态与原子 bundle

模型值必须在与 fingerprint 相同的 measurement x 上求值。对固定 bundle 的每个
分量 `j`：

\[
E_j=c_j^Ty_{\rm exp},\quad
B_j=c_j^Ty_{\rm V4.1},\quad
m_j=|B_j-E_j|.
\]

状态固定为：

```text
PASS      : m_j <= tau_c
REVERSED  : m_j > tau_c and sign(B_j) != sign(E_j)
UNDER     : m_j > tau_c, same sign, and |B_j| < |E_j|
OVER      : m_j > tau_c, same sign, and |B_j| >= |E_j|
```

分类保证 `E_j` 非零。curve 只有至少一个分量不是 `PASS` 才支持 disease。
PEAK/TROUGH 始终作为完整 bundle：

- 错误侧必须全部进入待恢复集合；
- `PASS` 侧记录为 guard，后续不得损伤；
- 禁止只抽取最有利的一侧另建 disease。

curve-level excess 固定为：

\[
s_{\rm curve}={1\over m}\sum_{j=1}^{m}\max(0,m_j-\tau_c),
\]

其中 `m=1`（END）或 `m=2`（PEAK/TROUGH），`PASS` 分量以零计入。

Disease ID 只由

```text
(channel, abscissa, bundle_type, ordered_component_state_signature)
```

组成。极值索引、物理位置、系数和 condition support 不并入跨 PF 的 disease ID，
但必须逐 curve 保存；同一 PF 的 aliases 必须在这些字段上严格一致。

## 5. 插值后的有效 support 与独立复现

对 measurement contrast 系数 `c_i` 和其 solver 插值权重 `w_iq`，先计算：

\[
\alpha_q=\sum_i c_iw_{iq}.
\]

绝对值 `<=1e-12` 的抵消项删除。一个 curve bundle 的 effective support 是所有
分量非零 `alpha_q` condition keys 的并集。

同一 PF 的所有 official aliases 必须同时一致于：

- shape/bundle 类型；
- 极值索引和原始物理横坐标；
- contrast 系数；
- 每个分量的完整 `alpha_q` map；
- ordered component state signature。

任一不一致则整个 PF 标为 `ALIAS_SENSITIVE` 并撤票。

对每个 disease 建 support-conflict graph：两个 PF 的 effective support 有交集
即连边。独立复现数定义为该图最大 independent set 的基数，也就是最大
pairwise-disjoint PF 子集；必须用确定性 exact/branch-and-bound 求解，不能用
support hash 去重或顺序相关 greedy。若存在多个最大子集，全部保存；PF ID
字典序只用于稳定序列化。

Disease 至少需要 `max_pairwise_disjoint_pf_count >= 2`。condition union 大小只作
覆盖描述，不参与排名。

## 6. 三个非退化稳健性视角

只对通过第5节独立复现门的 disease 排名。每个 PF 的 severity 是其 aliases 的
curve-level excess 算术平均。

三个主视角固定为：

1. `PF_EQUAL_MEAN`：所有 consensus PF severity 等权平均；
2. `SUPPORT_CLUSTER_EQUAL_MEAN`：在 support-conflict graph 的每个 connected
   component 内先对 PF severity 平均，再让各 component 等权；
3. `PF_MEDIAN`：所有 consensus PF severity 的中位数。

此外必须报告但不参与主 winner 的：

- official-curve-equal contrast excess；
- 每条支持曲线先减去自身 residual mean 后，汇集全部 raw measurement points 的
  centered point-weighted absolute residual；
- point/curve/PF 数和所有 support overlaps。

每个主视角独立排序。只有以下条件同时满足才冻结 active disease：

1. 三个视角具有同一个严格第一名；
2. 每个视角中 winner 比各自 runner-up 高 `> delta_rank`；
3. winner 通过至少两个 pairwise-disjoint PF；
4. 若 winner 有至少三个 pairwise-disjoint PF，则逐一 leave-one-PF-out 后三个
   视角的 winner 均不改变；
5. 若全局只有一个 eligible disease，runner-up margin 视为 vacuous，但独立复现
   和 alias 门仍必须通过。

ID 字典序绝不打破科学并列。最大 independent sets 只用于独立复现门和审计，
不从中挑一个集合计算主分数，因此不存在通过选择有利 disjoint set 改变 winner
的自由度。

固定失败语义：

| 情形 | 输出 |
|---|---|
| 无 eligible 零和 disease | `NO_DECISION_NO_ELIGIBLE_ZERO_SUM_DISEASE` |
| 无 disease 达到两个 disjoint PF | `NO_DECISION_NO_INDEPENDENT_REPLICATION` |
| 三个视角第一名不同 | `NO_DECISION_WEIGHTING_SENSITIVE` |
| 任一 runner-up margin 不足 | `NO_DECISION_DISEASE_TIE` |
| alias 撤票改变 winner | `NO_DECISION_ALIAS_SENSITIVE` |
| leave-one-PF-out 改变 winner | `NO_DECISION_LEAVE_ONE_PF_SENSITIVE` |

没有符号反转或没有 eligible disease 不能推出 `OFFSET_ONLY`。

## 7. Prepare 同时冻结 guard contrasts

对 winner 的每条支持曲线，Prepare 除 disease bundle 外，还必须冻结：

- bundle 中所有 `PASS` 分量；
- `|E|>tau_c` 且 baseline 状态为 `PASS` 的 global END；
- `|E|>tau_c` 且 baseline 状态为 `PASS` 的相邻两点零和 contrast。

每个 guard 保存 measurement indices、系数、实验/基线值、有效 `alpha_q` map 和
门限。Evaluate 不得新增/删除 guard；父节点擦除后，所有原 `PASS` guard 必须仍
满足 `|D-E|<=tau_c` 且不得发生实验方向的符号翻转。

## 8. 两种互斥输出 schema

### 8.1 Active

```yaml
status: ACTIVE_DISEASE_FROZEN
authorization_sha256: ...
input_bundle_id: ...
scorecard_sha256: ...
fingerprint_sha256: ...
disease_id: ...
channel: L | T
abscissa: twist_deg | frequency_Hz
bundle_type: END | PEAK | TROUGH
component_state_signature: [...]
support_physical_family_ids: [...]
max_pairwise_disjoint_pf_count: ...
all_maximum_disjoint_pf_sets: [...]
support_official_curve_keys: [...]
alias_consensus: [...]
contrasts: [...]
guard_contrasts: [...]
witness_condition_keys: [...]
rankings:
  PF_EQUAL_MEAN: [...]
  SUPPORT_CLUSTER_EQUAL_MEAN: [...]
  PF_MEDIAN: [...]
leave_one_pf_out: [...]
claim_decision: NO_DECISION
reason: NODE_ATTRIBUTION_REQUIRED
```

### 8.2 No decision

```yaml
status: NO_DECISION_*
authorization_sha256: ...
input_bundle_id: ...
scorecard_sha256: ...
fingerprint_sha256: ...
candidate_ledger: [...]
exclusion_ledger: [...]
rankings: {...}
claim_decision: NO_DECISION
reason: <与status一致>
```

No-decision 产物不得伪填 active-only 字段。

## 9. 后续边界

`select-disease` 必须原子写入；Prepare 只接受该 selector 回执并再次独立复算。
Prepare 不得打开、stat 或 hash contribution 文件，也不得在输出中出现 N2/N3、
候选公式、参数或文献结论。

只有 `ACTIVE_DISEASE_FROZEN` 才允许进入 G2。任何 `NO_DECISION_*` 的下一步是
补充不改变模型的诊断证据或报告不可辨识性，不得跳到 LESP、空间涡、rVPM、
结构、参数扫或任意候选。
