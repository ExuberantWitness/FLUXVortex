# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v4

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改

## 0. 规范组成与版本演化

本 v4 是一个不可分割的规范组合：

1. 基础协议
   `fig171819_active_disease_prereg_v3_20260729.md`，SHA
   `8faec5de90cf563146b9950c1ca2a5200597623087595ae481f563d2c32f7f0f`；
2. 本文件中的替换条款。

本文件明确替换基础协议的冲突部分；没有被替换的 v3 条款继续有效。执行、测试、
authorization 和 evidence commit 必须同时绑定两份文件，不能只引用其中之一。

v3 经第三轮零数据审计判为 NO-GO，永久保留。阻断项为：候选依赖的排名视角、
LOPO margin、guard 去重、alias 反事实及 reason predicate 未完全形式化。v4
不得原地修改；若再次 NO-GO，必须发布新版本。

## 1. 替换 v3 §1：非循环的外部冻结与直接执行

冻结使用两个不可自引用的 Git 对象：

### 1.1 Evidence commit

第一个隔离 commit 的 tree 包含 v3、本 v4、parent attribution 最新协议、
selector/Prepare/Evaluate 及测试、fresh scorer/benchmark/fingerprint 及测试、
数据合同和两级 authorization policy。

commit 内的 authorization 只能包含 policy、nonce、文件路径和预期 blob SHA；
禁止包含、要求或预测尚未生成的 evidence commit SHA。

### 1.2 External attestation commit

evidence commit 生成后，第二个隔离 attestation commit 必须：

- 以 evidence commit 为 parent；
- 保存固定 payload：

```json
{
  "schema_version": 1,
  "evidence_commit_sha": "<40 hex>",
  "authorization_blob_sha256": "<64 hex>",
  "fresh_run_id": "20260729_135128",
  "fresh_status_when_attested": "running",
  "launcher_contract": "DETACHED_COMMIT_DIRECT_EXEC_V1"
}
```

- 不在 payload 中包含或预测自身 attestation commit SHA。

两个 literal commit SHA 都必须在 fresh 完成前写入用户可见 commentary，并镜像
到 goal contract 和 append-only `MANIFEST.md`。refs 只作导航；运行时只接受已
公布的 literal SHA。

### 1.3 独立启动

不得由待验证的工作树 selector 自证其源码。最小 launcher 必须：

1. 由系统 Git 按 literal evidence SHA 创建 repo 内、专用、detached worktree；
2. 验证 detached `HEAD` 精确等于 evidence SHA；
3. 从该 worktree 直接执行 selector/Prepare/Evaluate；
4. 将 external attestation payload 和 literal attestation SHA 作为只读参数传入；
5. 禁止导入原工作树的 `platform/`。

launcher contract 本身作为 blob 纳入 attestation commit；实际执行也必须从
literal attestation commit 提取该 blob。临时执行目录不得放在 `/tmp`，结束后
只能清理该次 launcher 创建且已验证路径的 worktree。

## 2. 补充 v3 §2/§4：有限维度、区间和系数

每条 official curve 必须满足 `n>=2`。PEAK/TROUGH 只在 `n>=3` 时参与候选；
否则只允许 END。

定义空索引对集合上的 `max_drawdown` 和 `max_rebound` 为 `0`。近极值集合为：

\[
I_{\max}=\{i:|e_i-\max(e)|\le10^{-12}{\rm N}\},\qquad
I_{\min}=\{i:|e_i-\min(e)|\le10^{-12}{\rm N}\}.
\]

PEAK 要求 `|I_max|=1`，TROUGH 要求 `|I_min|=1`。所有其他门继续使用 v3
规定的严格 `>` 或 `<=`。

固定 measurement coefficient vectors：

```text
END       : c[0]=-1, c[n-1]=+1
RISE      : c[0]=-1, c[k_max]=+1
ROLLOFF   : c[k_max]=+1, c[n-1]=-1
FALL      : c[0]=+1, c[k_min]=-1
RECOVERY  : c[k_min]=-1, c[n-1]=+1
```

未列索引系数均为零。每个 contrast 的唯一实验身份是：

```text
(official_curve_key,
 ordered[(measurement_index, measurement_coefficient)])
```

## 3. 替换 v3 §6 的 alias 相等与 condition key

canonical solver condition key 是按顺序的四元组：

```text
(U_mps: finite float,
 frequency_Hz: finite float,
 twist_deg: finite float,
 aoa_deg: finite float)
```

字符串序列化必须调用 evidence commit 内冻结的唯一 `condition_key()`；物理比较
使用四元组，不使用字符串近似或舍入 hash。

同一 PF 的 aliases 对每个 unordered pair 做一致性检查，不使用传递聚类。
一致性只要求：

- shape class；
- component names、canonical nominal indices 和 coefficient vectors；
- 每个 component 的 `sign(E)` 与 baseline state。

raw/evaluation x 和各 alias 的 interpolation alpha 不要求 `1e-12` 相等，因为
它们来自分别数字化的横坐标。每条 alias 保留自己的 raw/evaluation x、E/B 和
alpha 并独立求值。PF effective support 是全部 consensus aliases、全部 disease
components 的非零 alpha condition-key 并集。

任一 unordered alias pair 不一致，则整个 PF 确定性撤票并记录
`ALIAS_WITHDRAWN`。不构造“恢复哪个 alias”的反事实世界；alias 撤票本身不作为
独立 primary reason。它对 eligibility 的影响由撤票后的固定 candidate ledger
自然产生。

## 4. 替换 v3 §7：全局固定排名视角

只有两个预登记主排名视角，且对所有 eligible diseases 全局相同：

1. `PF_EQUAL_MEAN`：全部 consensus PF severity 的算术平均；
2. `PF_REPLICATION_FLOOR`：全部 consensus PF severity 的最小值。

`SUPPORT_CLUSTER_EQUAL_MEAN`、`PF_MEDIAN`、official-curve-equal 和 centered
point-weighted residual 永远只作 audit diagnostics，禁止参与 active/no-decision
裁决。因此不存在候选依赖的 view activation，也不输出
`NO_DECISION_DEGENERATE_ROBUSTNESS_VIEWS` 或
`NO_DECISION_WEIGHTING_SENSITIVE`。

对撤除 alias-sensitive PF 后的 candidate ledger，先以
`max_pairwise_disjoint_pf_count>=2` 得到 eligible set `D`。

对每个主视角 `v`：

```text
M_v = max(score_v[d] for d in D)
A_v = {d in D: abs(score_v[d] - M_v) <= 1e-12 N}
```

- 任一 `|A_v| != 1`：`NO_DECISION_DISEASE_TIE`；
- 两个 singleton argmax 不同：`NO_DECISION_VIEW_DISAGREEMENT`；
- 若 `|D|>1`，对共同 winner `w`：

```text
runner_v = max(score_v[d] for d in D if d != w)
score_v[w] - runner_v > delta_rank
```

  必须在两个视角分别成立；否则
  `NO_DECISION_INSUFFICIENT_RANK_MARGIN`；
- 若 `|D|=1`，runner margin 为 vacuous，但所有其他门继续执行。

字典序只允许稳定序列化，不能决定 argmax。

## 5. 替换 v3 §7.1：完整 shadow LOPO

对共同 winner 的每个 consensus PF `q`，从全局 42-curve candidate ledger 删除
该 PF 的全部 official aliases，包括它在所有 competing disease 中的出现，然后
从零重算：

- unordered-pair alias consensus；
- PF alpha-support union；
- support-conflict graph 与 exact MIS；
- candidate eligibility；
- 两个固定主视角；
- argmax sets、共同 winner 和全部 runner-up margins；
- 本 v4 §7 的所有 reason predicates。

shadow 中唯一变化是独立复现门临时从2降为1。每一次重算仍必须：

```text
两个视角都有唯一 argmax
两个视角 winner 相同
若 eligible disease >1，每个视角 margin > delta_rank
winner disease ID 与原 winner 相同
```

任一失败固定输出 `NO_DECISION_LEAVE_ONE_PF_SENSITIVE`。删除
`PF_REPLICATION_FLOOR>0` 这一恒真式，不以它代替实际效应 margin。

## 6. 替换 v3 §8：不得按 alpha 去重 guard

所有满足 v3 pairwise guard 门的 measurement contrast 都是独立科学义务。唯一
允许的去重键是：

```text
(official_curve_key,
 ordered[(measurement_index, measurement_coefficient)])
```

不同 official curve 或不同 measurement pair 即使 canonical alpha 完全相同，
也必须分别保存和分别对各自 `E_ij/B_ij` 求值。实现可以按 alpha 缓存同一个父
贡献 `G`，但不能合并、删除或平均 guard obligations。

## 7. 替换 v3 §10：完整 reason predicate 与优先级

先计算全部 Boolean predicates，再按以下顺序选择 primary status，并保存所有
触发项：

1. `INVALID_EVIDENCE`：任一输入、身份、有限值、形状或 receipt 硬门失败；
2. `NO_DECISION_NO_PRE_REPLICATION_CANDIDATE`：撤票前没有任何至少含一个
   consensus PF、且至少一个 component 非 PASS 的 disease；
3. `NO_DECISION_NO_INDEPENDENT_REPLICATION`：存在上述 candidate，但撤除
   alias-sensitive PF 后没有 disease 的 exact MIS 达到2；
4. `NO_DECISION_DISEASE_TIE`：任一主视角的 `|A_v|!=1`；
5. `NO_DECISION_VIEW_DISAGREEMENT`：两个 singleton argmax 不同；
6. `NO_DECISION_INSUFFICIENT_RANK_MARGIN`：共同 winner 存在，但任一适用
   runner-up margin 不严格大于 `delta_rank`；
7. `NO_DECISION_LEAVE_ONE_PF_SENSITIVE`：任一完整 shadow LOPO 失败。

只有所有 predicate 均未触发时输出 `ACTIVE_DISEASE_FROZEN`。删除 v3 中未定义
或候选依赖的 reason names。多个原因同时触发时仍保存
`all_triggered_reasons`，但不得计算依赖于前置失败输入的下游 predicate。

## 8. Schema 收紧

有效 status 枚举固定为：

```text
ACTIVE_DISEASE_FROZEN
NO_DECISION_NO_PRE_REPLICATION_CANDIDATE
NO_DECISION_NO_INDEPENDENT_REPLICATION
NO_DECISION_DISEASE_TIE
NO_DECISION_VIEW_DISAGREEMENT
NO_DECISION_INSUFFICIENT_RANK_MARGIN
NO_DECISION_LEAVE_ONE_PF_SENSITIVE
INVALID_EVIDENCE
```

禁止 schema 中使用 `NO_DECISION_*` wildcard。Active 和所有 no-decision
产物都必须包含全42条 classification ledger、完整 candidate/exclusion ledger、
alias unordered-pair ledger、两个主视角 score/argmax/margin、全部 exact
maximum independent sets 和全部 shadow LOPO records。Invalid 不得包含数值
排名。

## 9. 阶段边界

selector 和 Prepare 只能接受 v3+v4 组合及其 evidence/attestation literal
commits。只有通过本 v4 全部门的 `ACTIVE_DISEASE_FROZEN` 才能进入 parent
attribution v3。任何其他状态禁止读取 contribution、选择 N2/N3、查方案文献或
实现模型。
