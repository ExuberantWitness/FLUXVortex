# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v6

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改

## 0. 规范组成

权威 active-disease 规范按顺序叠加 v3、v4、v5 和本 v6。前序 SHA 为：

```text
v3 8faec5de90cf563146b9950c1ca2a5200597623087595ae481f563d2c32f7f0f
v4 65ab9595db6b71d0e2fb6eee085dbf87c0d9c7eca0aaf0a4c6f6fae9d623c6bc
v5 174d0f18d33035278a46d27c04c68244183762c9a65058702fb2685f01b69940
```

本文件替换冲突条款，其余继续有效。v6 修复第五轮零数据审计发现的跨 PF
Disease ID、attestation tree delta、payload byte hash 和 shadow short-circuit。

## 1. 替换 v5 §1/§2：global disease 与 alias geometry 分离

全局聚合和排名只使用：

```text
global_disease_id =
  (channel,
   abscissa,
   shape_class,
   ordered[(component_name, sign(E), baseline_state)])
```

同一 PF 内 unordered alias pair 的一致性使用：

```text
alias_geometry_signature =
  (global_disease_id,
   ordered[(component_name,
            canonical_nominal_indices,
            measurement_coefficient_vector)])
```

所以：

- `L_PRE_ALIAS`、`L_CONSENSUS`、exact MIS 和 ranking 按
  `global_disease_id` 分组；
- `alias_geometry_signature` 只用于同一 PF 内的 alias withdrawal；
- 不同 PF 的极值 nominal index、raw/evaluation x、coefficient vector 和物理
  位置允许不同，仍可属于同一 global disease；
- 同一 PF aliases 的 geometry signature 任一 unordered pair 不同，则整个 PF
  为 `ALIAS_WITHDRAWN`。

v5 中把 nominal indices/coefficient vector 放入全局 Disease ID 的文字作废。

## 2. 替换 v5 §5：attestation 必须是单一路径新增

system-Git bootstrap 除 v5 全部检查外，必须验证 evidence→attestation 的精确
tree delta：

```text
git diff-tree --no-commit-id --name-status -r \
  <evidence_sha> <attestation_sha>
```

规范化输出必须精确只有一条：

```text
A<TAB>platform/docs/diag/fig171819_active_disease_attestation_20260729.json
```

并分别验证：

1. payload path 在 evidence tree 中不存在；
2. payload path 在 attestation tree 中是 regular blob；
3. attestation 恰有一个 parent 且等于 evidence SHA；
4. launcher path 在 evidence 与 attestation tree 的 Git blob OID 精确相同；
5. authorization path 在 evidence 与 attestation tree 的 Git blob OID 精确相同；
6. 除新增 payload 外，不存在 mode、path、blob 或 subtree 变化。

任何 rename/copy/mode change、第二个新增路径、launcher/auth 替换或额外 parent
均 fail-closed。只有通过 tree-delta 门后才允许解析 payload。

## 3. 替换“canonical JSON SHA256”：精确 blob-byte hash

`attestation_payload_sha256` 定义为 attestation commit 固定 payload path 的
Git blob **精确字节序列**的 SHA256，包括原文件的 UTF-8 bytes、空白和 terminal
newline；不解析后重序列化，不使用未命名的 canonical JSON。

bootstrap 顺序固定为：

1. 由 Git object database 读取 exact blob bytes；
2. 对 exact bytes 计算 SHA256 并与 fresh 完成前公布值比较；
3. 再以严格 UTF-8 和 duplicate-key-rejecting JSON parser 解析；
4. 验证 v5 固定 schema，不允许未知字段。

receipt 中的 payload hash 始终指 exact committed blob bytes。

## 4. 替换 v5 §4：shadow short-circuit 精确状态

`shadow_lopo.evaluation_status` 枚举扩展为：

```text
EVALUATED
NOT_RUN_NO_ELIGIBLE_MAIN_DISEASE
NOT_RUN_MAIN_DECISION_GATE_FAILED
NOT_RUN_UPSTREAM_FAILURE
```

固定映射：

- `INVALID_EVIDENCE`：
  `NOT_RUN_UPSTREAM_FAILURE`；
- main `NO_PRE_REPLICATION_CANDIDATE` 或 `NO_INDEPENDENT_REPLICATION`：
  `NOT_RUN_NO_ELIGIBLE_MAIN_DISEASE`；
- main `DISEASE_TIE`、`VIEW_DISAGREEMENT` 或
  `INSUFFICIENT_RANK_MARGIN`：
  `NOT_RUN_MAIN_DECISION_GATE_FAILED`；
- main 已通过所有 predicates 1–6并形成 gate-qualified winner：
  `EVALUATED`，随后可能 active 或 LOPO-sensitive no-decision。

“main winner”从此只表示通过 unique-view 和 margin 全部门的
gate-qualified winner，不表示裸数值 argmax。

## 5. 阶段边界

selector/Prepare/Evaluate 与 authorization 必须绑定 active v3+v4+v5+v6
组合。v6 未获独立零数据 GO 前不得创建 evidence/attestation commit 或运行
selector。
