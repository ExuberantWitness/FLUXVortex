# Fig17/18/19 父 Claim 归因协议 v4

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计或 contribution  
**阶段**：G2 父节点归因；只允许输出唯一 active parent hypothesis 或确定
no-decision

## 0. 规范组成

权威规范按顺序叠加：

1. parent attribution v2，SHA
   `ca61e3e58dbc13c5f09a2cc55ce662e4b0c733ddeb9dd9c795e97e9f7619f734`；
2. parent attribution v3，SHA
   `f1e1cd98105d21c88da580136adf3ae088de75f53319e883f2c47cb33f1ca1a1`；
3. 本 v4；
4. active-disease v3+v4+v5。

后者替换前者冲突条款，其余继续有效。本 v4 修复第四轮零数据审计发现的全局
leaf inventory、attestation output binding、family-MAE schema 和 alias local
reason 精确性。

## 1. 替换 parent-v3 §2：全局与 eligible-parent leaf inventory

全局冻结 contribution inventory 是 canonical `(owner_node, leaf_id, role)`
三元组集合：

```text
N1:
  (uvlm, physics)
  (vortex_impulse, physics)
  (leading_edge_suction, physics)
  (uvlm_remainder, physics)
N2:
  (separation, physics)
  (profile_drag, physics)
N3:
  (ds_vortex, physics)
  (vortex_normal, physics)
N4:
  (ct_consistency, diagnostic)
N5:
  <no ForceLedger leaf; observer only>
N6:
  (rig_drag, necessary_physics)
R0:
  (numerical_cycle_reduction, diagnostic)
```

只允许这11个 canonical leaf records。每个 confirmed condition 必须：

- 每个 expected leaf 恰好一个 owner、一个 record；
- `(owner_node, leaf_id)` 全局唯一；
- N2/N3 eligible leaves 互斥；
- 不存在未注册 leaf、重复 leaf 或跨 owner 共享；
- role 与冻结 inventory 一致；
- N2/N3 aggregate 分别精确等于自己的两个 eligible leaves 之和；
- N1/N4/N6/R0 不混入 eligible parent aggregate；
- N5 不产生 ForceLedger leaf。

“禁止 inventory 外 physics leaf”只表示禁止上述全局冻结集合之外的 leaf，不能
误杀已注册的 N1 physics 或 N6 necessary-physics。

所有 vector/frame closure 容差继续为 `1e-9 N`。任一 inventory 或 aggregate
失败为全局 `INVALID_EVIDENCE`。

## 2. 补充 parent-v3 §7：alias parent signature

每条 official alias 对父节点生成：

```text
alias_parent_signature =
  (ordered[(component_name, component_restoration_boolean)],
   all_pass_components_hold,
   all_pairwise_guards_hold,
   curve_mae_pass,
   final_curve_restored)
```

同一 PF 的所有 unordered alias pairs 比较完整 signature：

- signature 不同：记录 `PARENT_FAIL_ALIAS_NONUNIFORM`；
- signature 相同但均失败：不记录 nonuniform，只记录实际 component/guard/MAE
  local reasons；
- signature 相同且均通过：alias-consistent。

每条 alias 的每个 guard 仍分别保存和求值；signature 比较不能合并 guard
obligations。

## 3. 补充 parent-v3 §4/§8：family-MAE 数值 schema

全部 support PF 的 family-equal MAE 按 active-v5 的 canonical ID order 和
`math.fsum` 计算。输出固定为：

```yaml
family_equal_mae:
  role: DERIVED_INVARIANT_AUDIT
  units: N
  N2_improvement_N: <finite float>
  N3_improvement_N: <finite float>
  invariant_check: PASS
```

`invariant_check` 验证两个 implication：

```text
N2 == PARENT_PASS  => N2_improvement_N > 0.15
N3 == PARENT_PASS  => N3_improvement_N > 0.15
```

失败父节点的 family mean 仍如实计算，可以小于或等于门；它不触发 invariant
失败。任一 PASS implication 不成立说明实现/账本矛盾，输出
`INVALID_EVIDENCE`。

## 4. 补充 parent-v3 §5/§6：reason 序列化顺序

科学产物的 reason 序列化固定为：

1. global status；
2. N2 local reasons，按 parent-v3 §6 枚举；
3. N3 local reasons，按 parent-v3 §6 枚举；
4. secondary diagnostics。

只有 `N2=FAIL,N3=FAIL` 时 secondary diagnostics 精确为：

```text
[NO_DECISION_MISSING_OR_STATE_MEDIATED]
```

其他 truth-table rows 的 secondary diagnostics 为空。secondary diagnostic
永远不能改变 global status。

## 5. Attestation 与输出绑定

本协议完全继承 active-v5 的固定 payload、launcher、authorization paths 和
system-Git bootstrap。所有 parent Prepare/Evaluate/receipt/scientific outputs
必须包含：

```yaml
evidence_commit_sha: <40 hex>
attestation_commit_sha: <40 hex>
attestation_payload_sha256: <64 hex>
authorization_blob_sha256: <64 hex>
launcher_blob_sha256: <64 hex>
```

Prepare receipt 还必须证明 contribution path/hash/metadata 从未进入其参数、
环境、打开文件集合或异常。Evaluate receipt 必须绑定同一个 attestation 并在
读取 contribution 前验证 Prepare receipt。

## 6. Schema 短路

若 Evaluate 的 contribution identity、全局 inventory 或 ledger closure 失败：

```yaml
status: INVALID_EVIDENCE
parent_evaluation:
  evaluation_status: NOT_EVALUATED_UPSTREAM_FAILURE
```

禁止输出 N2/N3 数值 restoration、family MAE 或 truth-table status。只有全部
硬门通过后，才允许 `parent_evaluation.evaluation_status=EVALUATED` 并产生
parent-v3 的五种有效 status。

## 7. 阶段边界

parent 实现和测试必须绑定 parent-v2+v3+v4 与 active-v3+v4+v5 全组合。未获
独立零数据 GO 前不得创建 authorization/evidence commit，不得读取 contribution
或执行归因。
