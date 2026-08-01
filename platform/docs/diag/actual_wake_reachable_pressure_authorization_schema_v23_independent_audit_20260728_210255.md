# transport-v2.3 authorization schema successor 第五次独立复审

**时间**：2026-07-28T21:02:55+08:00  
**review independence**：two fresh-context same-family read-only audits  
**被审 MD SHA**：`c16d43ecd51058ba5f1e03fc0d36ace99c4855087f4576251657d4fd3dfcd7bc`  
**被审 JSON SHA**：`a421ffd5dc04c236a05037adbb255f11afa8453d6caec0764a2d8d914378c9ad`

## 裁决

```text
FAIL / STRICT_PARSER_DEFINITION_NOT_YET_CAUSAL_OR_NAMESPACE_COMPLETE
remaining blocker groups = 3
```

未实现 parser/consumer，未生成 production credential/ticket/marker/result，未运行
history。timestamped/latest MD、JSON 各自 byte-identical；JSON normative Markdown
SHA 命中且 JSON 合法。

## 已闭环

- final-ticket `P(TKT)` 的逐字段 source map；
- bearer 的 32 raw-byte commitment domain；
- Q/M/A/C/Z/TKT pointer equality/inequality matrix；
- C.scope 从 Q projection 正向构造；
- observed definition/source/runtime/registry/quarantine内容；
- identity、manifest/B-U-R/clearance digest inputs和无环 artifact order。

## Blocker 1：Pre-Q projection 因果闭环

Q 是第一 artifact 且必须携带 projection；唯一 `P` 的 domain 却是 C/Z 之后才存在
的 full final ticket。Q 时刻没有合法 builder；rejection 永远没有 ticket，更不能
引用 P。必须定义 exact `FrozenTicketSemanticPlanV1` / pre-ticket builder，并要求：

```text
Q.ticket_semantic_projection = BuildProjectionV1(plan inputs)
P(final ticket) = Q.ticket_semantic_projection
```

## Blocker 2：standalone seam source contract

- registry 使用的 `guard` 未绑定 exact module/path/hash；
- `S_pre/S_1/S_2` 未定义 snapshot constructor/type；
- `expected_authorization_sha256` 未明确 hard-equal `raw(TKT)`。

## Blocker 3：filesystem namespace/profile

- production fixed path 与 isolated same-seam fixture 无显式 profile carve-out；
- safe repo-relative 未冻结 root/base和完整 POSIX lexical algorithm；
- `frozen_run_directory` 无 exact one-component grammar/source；
- G6 fixed dirfd、canonical basename和 owned temp-name profile未唯一指定。

## 最小闭合要求

- 定义 pre-ticket plan exact inputs/builder，rejection只绑定 Q/plan；
- 绑定已加载 guard source，定义 `_loaded_dependency_state()` snapshots，并绑定
  external raw ticket digest；
- 定义 schema-bound production/isolated namespace profiles，同一 parser、无
  parser bypass；
- 冻结 repo-relative lexical algorithm、review run-dir grammar与 G6 temp naming；
- 修正“所有 child raw/canonical 均被存储绑定”的过宽文字；
- 新 timestamped successor 后做第六次只读复审。

## 最大允许结论

字段级 schema 已大体闭合，但 Q 尚不能因果构造，且 production/fixture path 集合
不唯一。parser/consumer、G3/G5/G6、production credential、ticket 和 history
继续锁定；科学裁决仍为 `UNKNOWN`。
