# Fig17/18/19 父 Claim 归因协议 v5

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计或 contribution  
**阶段**：G2 父节点归因

## 0. 规范组成

权威 parent 规范按顺序叠加 parent-v2、parent-v3、parent-v4 和本 v5。
前序 SHA：

```text
parent-v2 ca61e3e58dbc13c5f09a2cc55ce662e4b0c733ddeb9dd9c795e97e9f7619f734
parent-v3 f1e1cd98105d21c88da580136adf3ae088de75f53319e883f2c47cb33f1ca1a1
parent-v4 71cbd1c1b129d35656d02e107ca39550c7a022b18dc96f33cd25576e1109526a
```

本文件替换冲突条款，其余继续有效，并继承 active v3+v4+v5+v6。

## 1. 替换 parent-v4 §6：验证态与科学态分离

`parent_evaluation.evaluation_status` 固定枚举：

```text
NOT_EVALUATED_UPSTREAM_FAILURE
EVALUATED_INVALID_INVARIANT
EVALUATED
```

状态机：

1. contribution identity、global leaf inventory、aggregate/ForceLedger closure
   任一失败：
   - global `status=INVALID_EVIDENCE`；
   - `evaluation_status=NOT_EVALUATED_UPSTREAM_FAILURE`；
   - 禁止父恢复数值和科学 truth table。
2. 上述硬门通过后，计算 N2/N3 restoration 和 family-MAE derived invariant。
   若任一 PASS implication 失败：
   - global `status=INVALID_EVIDENCE`；
   - `evaluation_status=EVALUATED_INVALID_INVARIANT`；
   - validation receipt 保存 family-MAE 数值与失败 implication；
   - 禁止发布四项科学 truth-table status。
3. 全部验证通过：
   - `evaluation_status=EVALUATED`；
   - 只允许四项科学 status：

```text
ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS
ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS
NO_DECISION_MULTIPLE_PARENTS
NO_DECISION_NO_PARENT_FULL_COVERAGE
```

parent-v4 中“硬门通过后产生五种 status”的文字作废；第五个
`INVALID_EVIDENCE` 是验证失败状态，不是科学 truth-table 结果。

## 2. Invalid invariant receipt

`EVALUATED_INVALID_INVARIANT` receipt 至少包含：

```yaml
status: INVALID_EVIDENCE
parent_evaluation:
  evaluation_status: EVALUATED_INVALID_INVARIANT
family_equal_mae:
  role: DERIVED_INVARIANT_AUDIT
  units: N
  N2_improvement_N: <finite float>
  N3_improvement_N: <finite float>
  failed_implications: [N2_PASS_IMPLIES_THRESHOLD | N3_PASS_IMPLIES_THRESHOLD]
```

这些值只解释验证器为何 fail-closed，不构成 parent ranking 或科学结果。

## 3. Attestation 与阶段边界

parent 全流程继承 active-v6 的 exact tree-delta 和 exact payload-blob-byte
hash。Prepare/Evaluate/receipt 必须绑定 active v3+v4+v5+v6 与 parent
v2+v3+v4+v5 的完整 SHA 列表。

本 v5 未获独立零数据 GO 前不得创建 authorization/evidence/attestation commit，
不得读取 contribution 或执行 parent attribution。
