# Fig17/18/19 父 Claim 归因协议 v3

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计或 contribution  
**阶段**：G2 父节点归因；只允许输出唯一 active parent hypothesis 或确定
no-decision

## 0. 规范组成与版本演化

本 v3 是不可分割的规范组合：

1. 基础协议
   `fig171819_parent_attribution_protocol_v2_20260729.md`，SHA
   `ca61e3e58dbc13c5f09a2cc55ce662e4b0c733ddeb9dd9c795e97e9f7619f734`；
2. 本文件中的替换条款；
3. active-disease v3+v4 规范组合。

本文件替换冲突条款；其余 parent-v2 条款继续有效。parent-v2 经第三轮零数据
审计判为 NO-GO，永久保留。阻断项为挑有利 PF、guard support 污染正向独立性、
leaf 归属未冻结及 parent truth table 与 reason precedence 冲突。

## 1. 替换 parent-v2 §1：阶段化输入门

Prepare 的输入硬门只验证 contribution-free bundle：

- complete fresh manifest、result、case evidence/guards；
- scorecard、residual、fingerprint 和 baseline execution receipt；
- v4 `ACTIVE_DISEASE_FROZEN` 及 selector receipt；
- evidence/attestation literal commits、源码和协议身份。

Prepare 的 API、参数对象、环境和异常信息均不得包含 contribution path、hash、
size、mtime 或 existence bit。

以下门只属于 Evaluate，并且只能在 Prepare receipt 已原子落档后执行：

- contributions path/hash/timestamp 与 complete fresh bundle 一致；
- contribution condition keys 精确等于151个 confirmed keys；
- graph/runtime/call/guard/source closure；
- body/wind force ledger 与 leaf/parent ledger。

Evaluate 输入失败输出 `INVALID_EVIDENCE`，不是父节点 FAIL。

## 2. 补充 parent-v2 §3：叶级唯一性与父聚合

冻结唯一物理 leaf inventory：

```text
N2 leaves = {separation, profile_drag}
N3 leaves = {ds_vortex, vortex_normal}
```

对每个 confirmed condition 和每个 wind/body-frame force component，Evaluate
必须验证：

```text
N2 leaves ∩ N3 leaves == empty
每个 expected leaf 恰好一个记录
不存在 expected inventory 之外的 physics leaf
每个 leaf 只有一个 parent owner
N2 aggregate == separation + profile_drag
N3 aggregate == ds_vortex + vortex_normal
```

aggregate 与 leaf sum、body/wind transform 及全局 ForceLedger 的绝对误差均
不得超过 `1e-9 N`。diagnostic/observer/necessary-physics 角色不能混入 N2/N3
aggregate。任一失败为全局 `INVALID_EVIDENCE`。

## 3. 替换 parent-v2 §6：PF 全覆盖与正向 support

父节点正向恢复的 PF 集固定为 v4 active disease 的全部
`support_physical_family_ids`。不得选择子集，不得只保留有利 PF。

每个 PF 的正向 independence support 精确沿用 v4 冻结的 active disease bundle
alpha support：

```text
PF positive support =
union(nonzero canonical alpha condition keys
      over all disease components and all official aliases)
```

它不包含 pairwise guard 的 alpha，不包含 contribution 幅值 support，也不因父
节点改变。guards 是全局 veto obligations；guard condition 不能给正向
support-conflict graph 增加边。

同一 PF 的每条 official alias 都使用自身冻结 interpolation alpha、E/B 和父
贡献求值。PF 只有在全部 aliases 的全部非 PASS components、PASS components、
all-pair guards 和 curve-MAE 门均通过时才为 `RESTORED`。

父节点取得 `PARENT_PASS` 必须满足：

```text
RESTORED_PF_SET == ACTIVE_DISEASE_SUPPORT_PF_SET
max_pairwise_disjoint_pf_count_on_frozen_positive_support >= 2
```

任何 support PF 未恢复即该父节点 `PARENT_FAIL`。不得以 exact MIS 选出的有利
子集掩盖其他 PF；MIS 只证明全覆盖集合中至少存在两个 pairwise-disjoint
复现。

## 4. 替换 parent-v2 §7：MAE 定义

每条 official support curve 仍必须满足：

```text
curve_MAE_improvement_N > 0.15
```

family-equal MAE 在全部 active-disease support PF 上计算，而不是在
`RESTORED` 子集上计算：

1. 每个 PF 先等权平均全部 official aliases 的 curve-MAE improvement；
2. 再等权平均全部 support PF。

由于全覆盖和逐曲线门已经蕴含 family mean `>0.15 N`，该量明确标记为
`DERIVED_INVARIANT_AUDIT`，不冒充第三个独立证据。若数值不满足，说明实现或
账本不一致，输出 `INVALID_EVIDENCE`。

## 5. 替换 parent-v2 §8：先算父真值，再算全局状态

在有效证据下，每个父节点只产生：

```text
PARENT_PASS
PARENT_FAIL
```

所有 alias不一致、分量未恢复、PASS损坏、guard损坏、curve-MAE失败或PF未全
覆盖，都是该父节点的确定 local failure reason；它们不能越过最终真值表抢占
另一个父节点的唯一 PASS。

先完整计算 N2/N3 两个 Boolean，再固定映射：

| N2 | N3 | 全局 status |
|---|---|---|
| PASS | FAIL | `ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS` |
| FAIL | PASS | `ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS` |
| PASS | PASS | `NO_DECISION_MULTIPLE_PARENTS` |
| FAIL | FAIL | `NO_DECISION_NO_PARENT_FULL_COVERAGE` |

所以“N2 PASS、N3 guard damage”必须输出 active N2；反向同理。全局
`all_triggered_reasons` 先保存 status，再按 `N2`、`N3` 和固定 local reason
枚举排序，不能改变 status。

两个父节点都 FAIL 时，固定添加 secondary diagnostic：

```text
NO_DECISION_MISSING_OR_STATE_MEDIATED
```

它不是可选 status，不能直接裁决缺组件。N6 不在本协议计算，固定报告：

```text
N6_negative_control: NOT_EVALUATED_FOR_PARENT_SELECTION
```

删除 `NO_DECISION_FORBIDDEN_PATH` 和所有可选 N6 规则。

## 6. Parent local reason 枚举

每个父节点的 local reasons 只允许以下固定顺序：

1. `PARENT_FAIL_ALIAS_NONUNIFORM`
2. `PARENT_FAIL_COMPONENT_REVERSED_NOT_RESTORED`
3. `PARENT_FAIL_COMPONENT_UNDER_NOT_IMPROVED`
4. `PARENT_FAIL_COMPONENT_OVER_NOT_IMPROVED`
5. `PARENT_FAIL_PASS_COMPONENT_DAMAGED`
6. `PARENT_FAIL_PAIRWISE_GUARD_DAMAGED`
7. `PARENT_FAIL_CURVE_MAE`
8. `PARENT_FAIL_PF_NOT_FULLY_RESTORED`
9. `PARENT_FAIL_NO_PAIRWISE_DISJOINT_REPLICATION`

一个父节点可以保存多个 local reasons。`PARENT_PASS` 必须有空 local reason
列表。

## 7. Canonical support 与 alias 求值补充

parent contribution 不建立基于幅值的新 support。对一个冻结 measurement
contrast：

\[
G_{p,j}=\sum_q\alpha_{j,q}F_{p,q},
\]

其中 `alpha` 完全来自 v4 fingerprint，绝对值 `<=1e-12` 的项按 v4 删除；
`F_{p,q}` 是该 condition 经过叶级唯一性检查的 parent aggregate。不同 aliases
使用各自 alpha；PF positive support 取 aliases 的 condition-key 并集。

alias 一致性不是比较 alpha 数值相等，而是要求每条 alias 分别得到相同的
`RESTORED`/`FAILED` 裁决。只要有 alias 失败，PF 不全恢复，父节点 FAIL。

## 8. 收紧输出 schema

有效 status 枚举固定为：

```text
ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS
ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS
NO_DECISION_MULTIPLE_PARENTS
NO_DECISION_NO_PARENT_FULL_COVERAGE
INVALID_EVIDENCE
```

禁止 wildcard status。每个有效科学产物必须包含：

```yaml
causal_status: HYPOTHESIS_ONLY
claim_writeback_allowed: false
N2:
  status: PARENT_PASS | PARENT_FAIL
  local_reasons: [...]
  restored_pf_ids: [...]
  all_support_pf_ids: [...]
  component_results: [...]
  pass_component_results: [...]
  guard_results: [...]
  curve_mae_results: [...]
  positive_support_conflict_graph: {...}
  all_maximum_disjoint_pf_sets: [...]
N3: <same schema>
family_equal_mae:
  role: DERIVED_INVARIANT_AUDIT
N6_negative_control: NOT_EVALUATED_FOR_PARENT_SELECTION
```

Active parent hypothesis 仍必须输出：

```yaml
reason: LITERATURE_MECHANISM_ADJUDICATION_REQUIRED
spatial_panel_load_required: true
force_and_moment_ledger_required: true
posthoc_total_force_redistribution_forbidden: true
```

Invalid 不得包含数值 parent ranking。

## 9. 冻结与阶段边界

parent v3 与 active-disease v3+v4 使用同一 evidence commit 和 external
attestation commit；commit 内 authorization 不含未来 commit SHA。执行必须走
active-v4 §1 的 detached-commit direct launcher。

只有两个 active status 才授权围绕“已冻结 disease + 唯一 parent”开展一手文献
研究。两个 no-decision status 都禁止选择子节点、候选公式、参数或模型实现。
