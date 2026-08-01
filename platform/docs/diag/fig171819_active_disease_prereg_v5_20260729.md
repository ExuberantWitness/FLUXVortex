# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v5

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改

## 0. 规范组成

权威规范是以下三份文件按顺序叠加：

1. active-disease v3，SHA
   `8faec5de90cf563146b9950c1ca2a5200597623087595ae481f563d2c32f7f0f`；
2. active-disease v4，SHA
   `65ab9595db6b71d0e2fb6eee085dbf87c0d9c7eca0aaf0a4c6f6fae9d623c6bc`；
3. 本 v5。

后者替换前者的冲突条款，其余条款继续有效。v5 修复第四轮零数据审计发现的
完整 Disease ID、短路 schema、shadow reason 递归、detached launcher 唯一身份
和确定归约问题。前三个版本均不得原地修改。

## 1. 补充 v4 §3：完整 alias Disease ID

同一 PF 的每个 unordered alias pair 必须比较完整非数值 Disease ID：

```text
(channel,
 abscissa,
 shape_class,
 ordered[(component_name,
          canonical_nominal_indices,
          measurement_coefficient_vector,
          sign(E),
          baseline_state)])
```

`channel` 只允许 `L` 或 `T`；`abscissa` 只允许 evidence-committed benchmark
枚举。任一字段不相等即整个 PF 为 `ALIAS_WITHDRAWN`。不存在跨 channel 或跨
abscissa 的 alias consensus。

## 2. 补充 v4 §4/§7：命名 ledger stages

固定三个 ledger stages：

1. `L_CLASSIFIED`：42条 official curves 完成纯实验 shape 与 baseline component
   state 后的逐曲线记录；
2. `L_PRE_ALIAS`：按完整 Disease ID 和 PF ID 分组，但尚未执行 unordered-pair
   alias consensus；一个 disease 只要至少有一个 PF occurrence 且至少一个
   component 非 PASS，即为 pre-replication candidate；
3. `L_CONSENSUS`：撤除所有 `ALIAS_WITHDRAWN` PF 后的固定 ledger；exact MIS、
   eligibility 和排名只在这里计算。

reason predicates 精确定义为：

- `NO_DECISION_NO_PRE_REPLICATION_CANDIDATE`：
  `L_PRE_ALIAS` 的 disease set 为空；
- `NO_DECISION_NO_INDEPENDENT_REPLICATION`：
  `L_PRE_ALIAS` 非空，但 `L_CONSENSUS` 中没有 exact MIS 达到2的 disease。

不得以自然语言“撤票前/后”替代这三个命名阶段。

## 3. 替换 v4 §5/§7：main-only reasons 与非递归 shadow

v4 reason predicates 1–6 只在 main `L_CONSENSUS` 上依次计算。只有 main
predicates 1–6 全部未触发并得到唯一 active winner 后，才运行 shadow LOPO。

每个 shadow 删除一个 PF 后，重新执行与 main 相同的数值算法，但：

- exact MIS threshold 临时为1；
- 只产生 nested `shadow_local_status`：

```text
SHADOW_PASS
SHADOW_NO_PRE_REPLICATION_CANDIDATE
SHADOW_NO_INDEPENDENT_REPLICATION
SHADOW_DISEASE_TIE
SHADOW_VIEW_DISAGREEMENT
SHADOW_INSUFFICIENT_RANK_MARGIN
SHADOW_WINNER_CHANGED
```

- shadow 不运行另一个 shadow，不计算全局 predicate 7，不递归；
- 任一 `shadow_local_status != SHADOW_PASS` 只设置一个全局 predicate：
  `NO_DECISION_LEAVE_ONE_PF_SENSITIVE`。

shadow 内部的 tie/view/margin 只能作为 nested code，不能提升为 main
`NO_DECISION_DISEASE_TIE`、`VIEW_DISAGREEMENT` 或
`INSUFFICIENT_RANK_MARGIN`。

## 4. 替换 v4 §8：可短路的互斥 schema

所有 active/no-decision 产物都必须包含：

```yaml
ledger_stages:
  classified: [...]
  pre_alias: [...]
  consensus: [...]
alias_pair_ledger: [...]
rankings:
  evaluation_status: EVALUATED | NOT_EVALUATED_EMPTY_ELIGIBLE_SET | NOT_EVALUATED_UPSTREAM_FAILURE
shadow_lopo:
  evaluation_status: EVALUATED | NOT_RUN_NO_MAIN_WINNER | NOT_RUN_UPSTREAM_FAILURE
```

字段规则：

- `rankings.evaluation_status != EVALUATED` 时，禁止出现 numeric scores、
  argmax sets、runner-up 或 margins；
- main 没有唯一 winner 时，
  `shadow_lopo.evaluation_status=NOT_RUN_NO_MAIN_WINNER`，禁止出现 shadow
  records；
- `INVALID_EVIDENCE` 时两个对象都为
  `NOT_EVALUATED_UPSTREAM_FAILURE/NOT_RUN_UPSTREAM_FAILURE`，禁止科学 ledger
  和数值排名；
- 只有 `ACTIVE_DISEASE_FROZEN` 或
  `NO_DECISION_LEAVE_ONE_PF_SENSITIVE` 可以有完整 shadow records；
- exact maximum independent sets 只在相应 support graph 实际被求值时存在；
  否则使用显式 `evaluation_status`，禁止伪造空集合代表“已求值”。

这替换 v4 中“所有 no-decision 都必须含数值排名和全部 shadow”的过强条款。

## 5. 替换 v4 §1：固定 attestation 路径和 bootstrap

固定路径：

```text
ATTESTATION_PAYLOAD_PATH =
  platform/docs/diag/fig171819_active_disease_attestation_20260729.json
LAUNCHER_PATH =
  platform/fig171819_evidence_launcher.py
AUTHORIZATION_PATH =
  platform/docs/diag/fig171819_active_disease_execution_authorization_20260729.json
```

launcher 和 authorization 必须先存在于 evidence commit。attestation commit
必须恰有一个 parent，且该 parent 精确为 payload 中的 evidence commit SHA。
attestation commit 在继承的 tree 上只新增固定 payload。

payload schema 固定为：

```json
{
  "schema_version": 1,
  "artifact_type": "fig171819_active_disease_external_attestation",
  "evidence_commit_sha": "<40 lowercase hex>",
  "authorization_path": "platform/docs/diag/fig171819_active_disease_execution_authorization_20260729.json",
  "authorization_blob_sha256": "<64 lowercase hex>",
  "launcher_path": "platform/fig171819_evidence_launcher.py",
  "launcher_blob_sha256": "<64 lowercase hex>",
  "attestation_payload_path": "platform/docs/diag/fig171819_active_disease_attestation_20260729.json",
  "fresh_run_id": "20260729_135128",
  "fresh_status_when_attested": "running",
  "launcher_contract": "DETACHED_COMMIT_DIRECT_EXEC_V1"
}
```

系统 Git bootstrap 在执行任何 payload/launcher Python 前必须：

1. 按已公布的 literal attestation SHA 读取 commit object；
2. 验证 commit 恰有一个 parent；
3. 读取固定 payload path，验证 JSON schema 和 canonical JSON SHA256；
4. 验证唯一 parent 等于 payload 的 evidence SHA；
5. 从 attestation tree 的固定 authorization/launcher paths 读取 blobs；
6. 验证两个 blob SHA256 精确匹配 payload；
7. 验证 authorization 自身不包含 evidence 或 attestation SHA 字段；
8. 从 attestation tree 的固定 launcher path 直接执行已验证 launcher；
9. launcher 创建 evidence SHA 的 repo-local detached worktree，验证 HEAD，
   并禁止导入原工作树 `platform/`。

不存在候选 launcher path、搜索路径或“任选满足 contract 的 blob”。

在 fresh 完成前必须向用户公布并镜像：

```text
evidence_commit_sha
attestation_commit_sha
attestation_payload_sha256
authorization_blob_sha256
launcher_blob_sha256
```

## 6. 输出必须绑定第二个 commit

每个 active/no-decision 产物和 receipt 必须保存：

```yaml
evidence_commit_sha: <40 hex>
attestation_commit_sha: <40 hex>
attestation_payload_sha256: <64 hex>
authorization_blob_sha256: <64 hex>
launcher_blob_sha256: <64 hex>
```

Invalid receipt 只要成功解析到其中某字段，就必须保存该字段及相应 parse/verify
状态；不得把未解析字段伪填为空 hash。

## 7. 确定性浮点归约

所有 curve MAE、component severity、alias mean、PF mean 和 ranking mean：

1. 按 canonical measurement index、official curve key、PF ID 或 Disease ID
   的相应字典序固定输入顺序；
2. 使用 CPython evidence environment 的 `math.fsum`；
3. 除以精确整数计数；
4. 禁止 NumPy/GPU 并行 reduction 或遍历字典自然顺序。

min/max、argmax-set `1e-12 N` 和 exact MIS 仍按 v3+v4 定义。运行 manifest 必须
绑定 Python implementation/version；环境漂移为 `INVALID_EVIDENCE`。

## 8. 阶段边界

selector/Prepare/Evaluate 实现和测试必须绑定 v3+v4+v5 全组合。未获独立零数据
GO 前不得创建 authorization/evidence commit，不得读取 live 精度，不得运行
selector。
