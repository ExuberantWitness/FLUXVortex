# S3ai-v2.2 / transport-v2.3 authorization schema 预注册 successor

**时间**：2026-07-28T20:51:55+08:00  
**artifact**：`actual_wake_reachable_pressure_authorization_schema_v23_preregistration_20260728_205155`  
**attempt id**：`S3ai-v2.2-transport-v2.3-successor-20260728_185556`  
**状态**：`SCHEMA SUCCESSOR PREREGISTERED / IMPLEMENTATION LOCKED PENDING AUDIT`  
**normative authority**：本 Markdown  
**claim-state change**：`false`

本版保留并 supersede：

- 20:16:39 schema 草案 MD SHA
  `228e00afed4393fa4d677fd18efabb560684e32c2844b85119146c98965fa621`；
- 20:24:40 独立 `FAIL / 7 blockers` audit MD SHA
  `268488467e989e3655da1cc45bd34edff2840e8710c3b8d5911ef7865fd34e60`、
  JSON SHA
  `11ec623e9747d00e65b5fae1950e0b97823e26721bfa35c361ca54b59910118a`。
- 20:26:18 第一版 successor MD SHA
  `768532b369fcbc9374d0e77e247b5b3191573e0ecffc6b2590dcbb02d47217a2`；
- 20:31:07 successor 独立复审 `FAIL / 3 remaining blocker groups` MD SHA
  `6573e93ff5a8ef0657bc7ca7de1ee2d3ac60f96e477c167f5c6c7a1a9301df3a`、
  JSON SHA
  `cbb847579d3126d0dd35396b45623e4738ab95f838332a4914236312983cc317`。
- 20:32:00 第二版 successor MD SHA
  `d15db9d628362358792533a69bb0cb98665b21f77e0e2ec1187f5d23ae5bbf47`、
  JSON SHA
  `fee0fe15b5c3d7c6dc42109a26b9b6d4db9ee90378d999ad96787276c50ae6b8`；
- 20:40:18 第三次独立复审 `FAIL / 1 remaining blocker group` MD SHA
  `f0d7abc2cc000f01b8238d696a65f5ef1d39c8e0d1c49764bcdcdb2cc3ff608d`、
  JSON SHA
  `44c9d6459df88e89820d1e40230e94c91f9de0ad9d799bf7eb1f14df4284e9ad`。
- 20:41:05 第三版 successor MD SHA
  `60330e24ea21dac3db69b04727c630e5cc61bc5abdca9a51a7527844f88aea9a`、
  JSON SHA
  `b5fc3f9d0ff3f0ef028bc8c03829386c099874b122a2fb25f978dd9d14525f1c`；
- 20:51:12 两路只读第四次复审 `FAIL / 2 remaining blocker groups` MD SHA
  `ab16002f737dfdbe454187ca1831afeef7bea4cf71233acb8ec21bc15bbb0ed0`、
  JSON SHA
  `e1dc3b2f07c2c87ba18f3880735e38f10f0d9614d4ef9fb4d9bcf384b5760e45`。

旧草案仍是 FAIL，不以本 successor 回写历史。

## 0. Authority 和许可

本 Markdown 是唯一规范性 schema authority。配套 JSON 只索引本文件 raw SHA；
差异使 gate 失败，不从两者选择较宽规则。

本版通过独立 audit 前，不允许实现 parser。即使通过，授权也仅限：

- strict ticket/request/clearance parser；
-同一 private loader seam 和 no-history fixtures；
- authorization loader 的三时点 dependency-zero-delta guard；
- transport-only consumer/provenance migration；
- strict bearer stdin；
-相应 definition tests。

任何阶段都不因本文件允许创建 production issuance request、token、clearance、
ticket、marker、result 或 history，不允许改科学 collector/aggregator、公式、
常数、case registry 或 claim state。

## 1. 无自引用的 TicketSemanticProjectionV1

projection 不再“从 ticket 删除若干字段”，而是由正向 allowlist 构造一个独立
object。精确字段：

```text
schema_version
ticket_artifact
ticket_version
attempt_id
scientific_protocol_version
transport_protocol_version
assurance_profile
authorization_plan
bound_artifacts
implementation_identity
definition_chain
execution_sources
stable_runtime_identity
bound_manifests
result
scope
old_failure_quarantine
ordered_case_names
case_identity_sha256
ordered_registry_manifest_sha256
```

`authorization_plan` 精确为：

```text
id
external_raw_sha256_required
single_use_token_sha256
formal_execution_allowed
execution_limit
decision
post_marker_retry_allowed
```

`P(ticket)` 只接受已通过第 2 节 exact top-level/nested schema 的 full ticket，
并按下列唯一正向赋值构造全新 strict JSON object；`deepcopy` 表示 JSON value
深复制，不保留可变引用：

```text
P.schema_version                  := ticket.schema_version
P.ticket_artifact                 := ticket.artifact
P.ticket_version                  := ticket.version
P.attempt_id                      := ticket.attempt_id
P.scientific_protocol_version     := ticket.scientific_protocol_version
P.transport_protocol_version      := ticket.transport_protocol_version
P.assurance_profile               := ticket.assurance_profile
P.authorization_plan             := deepcopy({
  id: ticket.authorization.id,
  external_raw_sha256_required:
      ticket.authorization.external_raw_sha256_required,
  single_use_token_sha256:
      ticket.authorization.single_use_token_sha256,
  formal_execution_allowed:
      ticket.authorization.formal_execution_allowed,
  execution_limit: ticket.authorization.execution_limit,
  decision: ticket.authorization.decision,
  post_marker_retry_allowed:
      ticket.authorization.post_marker_retry_allowed
})
P.bound_artifacts                 := deepcopy(ticket.bound_artifacts)
P.implementation_identity         := deepcopy(ticket.implementation_identity)
P.definition_chain                := deepcopy(ticket.definition_chain)
P.execution_sources               := deepcopy(ticket.execution_sources)
P.stable_runtime_identity         := deepcopy(ticket.stable_runtime_identity)
P.bound_manifests                 := deepcopy(ticket.bound_manifests)
P.result                          := deepcopy(ticket.result)
P.scope                           := deepcopy(ticket.scope)
P.old_failure_quarantine          := deepcopy(ticket.old_failure_quarantine)
P.ordered_case_names              := deepcopy(ticket.ordered_case_names)
P.case_identity_sha256            := deepcopy(ticket.case_identity_sha256)
P.ordered_registry_manifest_sha256
                                      := ticket.ordered_registry_manifest_sha256
```

P 不定义 reconstructed-ticket 或其他 domain extension；其 domain 只有 full
ticket，codomain 只有上述 projection object。

projection 明确且完全不包含：

```text
authorization.canonical_sha256
ticket.issuance_request（整个 mapping）
ticket.second_bounded_audit（整个 mapping）
任何 issuance request raw/canonical digest
任何 request/response/trace metadata digest
任何 ticket_semantic_projection_sha256 字段
任何 authorization raw/canonical digest
```

`bound_manifests` 的 exact fields 也不含上述动态 digest，见第 2.5 节。

projection digest 只存在于 projection 外部：

- issuance request 的 `ticket_semantic_projection_sha256`；
- final ticket 的 `issuance_request.ticket_semantic_projection_sha256`；
- final ticket 的 `second_bounded_audit.ticket_semantic_projection_sha256`；
- clearance review binding。

必须测试：

```text
P(ticket) == issuance_request.ticket_semantic_projection
H(P) == all external projection digests
changing only excluded dynamic ticket fields leaves P byte-identical
changing any included source field either changes P or fails exact ticket schema
P recursively contains no key named canonical_sha256 for authorization
P contains no key/path matching issuance_request, second_bounded_audit,
  ticket_semantic_projection_sha256
```

因此不再存在 `P contains H(P)`。

## 2. Final authorization ticket exact schema

reserved path：

```text
platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml
```

后缀虽为 `.yaml`，raw bytes 必须是 duplicate-safe canonical pretty JSON subset。
顶层 exact fields：

```text
artifact
schema_version
version
status
attempt_id
scientific_protocol_version
transport_protocol_version
assurance_profile
authorization
issuance_request
bound_artifacts
implementation_identity
definition_chain
execution_sources
stable_runtime_identity
bound_manifests
second_bounded_audit
result
scope
old_failure_quarantine
ordered_case_names
case_identity_sha256
ordered_registry_manifest_sha256
```

固定身份：

```text
artifact=actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556
schema_version=1.0
version=S3ai-v2.3-one-shot-authorization-v2
status=accepted_after_g5_independent_bounded_audit
attempt_id=S3ai-v2.2-transport-v2.3-successor-20260728_185556
scientific_protocol_version=S3ai-v2.2
transport_protocol_version=S3ai-v2.3
assurance_profile=RESEARCH_ACCIDENTAL_DRIFT
```

### 2.1 `authorization`

exact fields：

```text
id
canonical_sha256
external_raw_sha256_required
single_use_token_sha256
formal_execution_allowed
execution_limit
decision
post_marker_retry_allowed
```

固定语义：

```text
external_raw_sha256_required=true
formal_execution_allowed=true
execution_limit=1
decision=YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE
post_marker_retry_allowed=false
```

删除 `token_presentation_limit` 和“pre-marker cryptographic no-reuse”强 claim。
broker 仍按治理流程只交付一次，但本 profile 明确：

```text
durable_premarker_token_presentation_limit_enforced=false
premarker_token_reuse_protection_claimed=false
```

`execution_limit=1` 由 first-history 前的 durable attempt marker 约束；marker 后任何
失败不可 retry。pre-marker 失败没有产生物理 observation；若 broker/custodian
中断，治理流程销毁 custody 并要求新的 issuance successor，但不声称对恶意内存
复制有技术强制。

### 2.2 `issuance_request`

exact fields：

```text
path
raw_sha256
canonical_sha256
ticket_semantic_projection_sha256
```

### 2.3 `implementation_identity`

所有 nested `implementation_identity` 都是同一个
`ImplementationIdentityBodyV1`，exact fields：

```text
provider
model
model_family
agent_id
trace_id
source_trace_metadata_path
source_trace_metadata_sha256
```

`source_trace_metadata_sha256` 精确表示 source trace metadata canonical-pretty
raw bytes（含唯一结尾 LF）的 raw SHA256；不是 canonical-object SHA。

该 body 由 G5 前冻结的外部 implementation trace anchor 产生。ticket、
projection、Q、M、A 中的 nested `implementation_identity` 必须逐字段 exact
等于 `anchor.identity`，不得与 anchor envelope 本身比较。

### 2.4 `bound_artifacts`

exact keys：

```text
wrapper_source
wrapper_definition
transport_preregistration
dependency_capture_protocol
authorization_schema_preregistration
bootstrap_contract
implementation_identity
g0_quarantine_audit
g2_implementation_diff_audit
g3_capture_instantiation
dependency_manifest
g3_consensus_certificate
g3_independent_audit
g4_definition_test_audit
interpretation_contract_markdown
interpretation_contract_json
```

每值 exact `{path, sha256}`；safe repo-relative、无 `..`、non-symlink、no-follow。

### 2.5 `bound_manifests`

exact fields：

```text
bound_artifact_map_sha256
execution_source_map_sha256
stable_runtime_identity_sha256
ordered_registry_manifest_sha256
frozen_definition_chain_sha256
transport_preregistration_raw_sha256
dependency_capture_protocol_raw_sha256
authorization_schema_preregistration_raw_sha256
dependency_manifest_raw_sha256
dependency_manifest_canonical_sha256
dependency_B_paths_sha256
dependency_U_paths_sha256
dependency_R_paths_sha256
g3_consensus_canonical_sha256
bootstrap_contract_canonical_sha256
old_failure_quarantine_canonical_sha256
```

没有 issuance request、audit response 或 projection digest。

各 digest 的 exact input 唯一定义如下；此表同时禁止同名字段自由取值：

```text
bound_artifact_map_sha256
  = SHA256(canonical JSON of the complete exact bound_artifacts mapping)
execution_source_map_sha256
  = SHA256(canonical JSON of the complete exact execution_sources mapping)
stable_runtime_identity_sha256
  = SHA256(canonical JSON of the complete stable_runtime_identity object)
ordered_registry_manifest_sha256
  = SHA256(canonical JSON of the ordered
      [{name,case_identity_sha256}, ...] array)
frozen_definition_chain_sha256
  = SHA256(canonical JSON of the complete definition_chain mapping)
transport_preregistration_raw_sha256
  = bound_artifacts.transport_preregistration.sha256
dependency_capture_protocol_raw_sha256
  = bound_artifacts.dependency_capture_protocol.sha256
authorization_schema_preregistration_raw_sha256
  = bound_artifacts.authorization_schema_preregistration.sha256
dependency_manifest_raw_sha256
  = bound_artifacts.dependency_manifest.sha256
  = DependencyManifest.raw_sha256
dependency_manifest_canonical_sha256
  = DependencyManifest.canonical_sha256 from the bound audited parser
dependency_B_paths_sha256
  = SHA256(canonical JSON of list(DependencyManifest.B))
dependency_U_paths_sha256
  = SHA256(canonical JSON of list(DependencyManifest.U))
dependency_R_paths_sha256
  = SHA256(canonical JSON of list(DependencyManifest.R))
g3_consensus_canonical_sha256
  = SHA256(canonical JSON of the strict parsed object at
      bound_artifacts.g3_consensus_certificate.path)
bootstrap_contract_canonical_sha256
  = SHA256(canonical JSON of the strict parsed object at
      bound_artifacts.bootstrap_contract.path)
old_failure_quarantine_canonical_sha256
  = SHA256(canonical JSON of the complete exact old_failure_quarantine mapping)
```

这里 B/U/R 使用 dependency manifest 中通过 parser 后保留的 exact list order，
不额外 sort、set 重排或重新发现路径。G3 consensus certificate 与 bootstrap
contract 必须是 duplicate-safe strict JSON object；其 raw SHA 仍分别由
`bound_artifacts` 绑定，canonical SHA 只按第 6 节 object encoding 重算。

### 2.6 `second_bounded_audit`

exact fields：

```text
invocation_metadata_path
invocation_metadata_raw_sha256
invocation_metadata_canonical_sha256
request_path
request_raw_sha256
request_canonical_sha256
response_path
response_raw_sha256
response_canonical_sha256
trace_metadata_path
trace_metadata_raw_sha256
clearance_canonical_sha256
ticket_semantic_projection_sha256
```

由于 response object 就是 clearance object，必须硬等同：

```text
response_canonical_sha256 == clearance_canonical_sha256
```

二者保留为两个语义标签，但不得取不同值。

### 2.7 `result`

exact fields和值：

```text
path=platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json
marker_path=platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json.lock
overwrite_allowed=false
latest_pointer_write_allowed=false
old_canonical_result_write_allowed=false
atomic_no_replace_required=true
```

### 2.8 `scope`

exact fields和值：

```text
one_31_history_successor_execution_only=true
transport_failure_successor_not_retry=true
claim_state_change_allowed=false
production_activation_allowed=false
force_hp_state_ves_118_fig_allowed=false
malicious_local_writer_or_swap_restore_protection=false
premarker_token_reuse_protection_claimed=false
```

### 2.9 `old_failure_quarantine`

exact mapping 是以下七个 safe repo-relative path→raw SHA：

```text
platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_20260728_143034.yaml
  = 39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e
platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_20260728_134229.json.lock
  = 42f9cb852128b24d2e28330ebf2ad9911764c845fba6370b4220aadbd5d6d778
platform/docs/diag/actual_wake_reachable_pressure_formal_run_20260728_1616.log
  = 61ae80785419e6502c0e4544ba8d9757785c36fbb6e24bb7cac8bba988c8c60d
platform/docs/diag/actual_wake_reachable_pressure_formal_failure_forensics_20260728_183314.md
  = b0e05a7114722120da571ad5396ce14370a471512d9bf6daae4a4319f07a632e
platform/docs/diag/actual_wake_reachable_pressure_formal_failure_forensics_20260728_183314.json
  = 5ed17ab7d0bb3f793d188af41f7e7acd9f2cc7855f46bd57c14bce5df2219d9c
platform/docs/diag/actual_wake_reachable_pressure_formal_failure_forensics_correction_20260728_183755.md
  = 657ec4c12a2266e277a8f8e0cc3923f6e8d124942435903ecc7d8780255ef414
platform/docs/diag/actual_wake_reachable_pressure_formal_failure_forensics_correction_20260728_183755.json
  = a4990e4159e8b93c47707329830089e11078f4a2995c32c123dd904c9acc32b5
```

再加 exact：

```text
old_canonical_result_absent=true
old_latest_result_absent=true
```

其 canonical digest 必须等于 `bound_manifests` 中绑定值。

### 2.10 Observed nested inputs

以下 ticket objects 不另开可自由扩展 schema，而是 exact 等于 loader seam 已有
观测输入或 audited pure computation；这本身就是 unknown/missing-field rule：

```text
ticket.definition_chain
  = {
      "S3ai-v2": definition.frozen_definition_chain["S3ai-v2"],
      "S3ai-v2.1": definition.frozen_definition_chain["S3ai-v2.1"],
      "S3ai-v2.2": definition.frozen_definition_chain["S3ai-v2.2"],
      "implementation_audit":
          definition.frozen_definition_chain["implementation_audit"]
    }
```

上述四个 entry 各自 exact `{file,sha256}`；不复制 wrapper-definition 中的
`invariant` prose。

```text
ticket.execution_sources       = dict(source_fingerprints)
ticket.stable_runtime_identity = dict(stable_runtime_identity)
```

因此 source/runtime 的任意多一、少一或值漂移都失败；其唯一生产者分别是已绑定
wrapper 的 `_source_fingerprints(definition)` 与 `_stable_runtime_identity()`。

registry 的唯一 pure computation 是：

```text
cases      := guard.frozen_history_cases()
names      := [case.name for case in cases]
identities := {
  case.name: guard._case_identity_payload(case)["sha256"] for case in cases
}
registry_manifest := [
  {"name": name, "case_identity_sha256": identities[name]} for name in names
]
registry_sha := SHA256(_canonical_json_bytes(registry_manifest))

ticket.ordered_case_names                  = names
ticket.case_identity_sha256                = identities
ticket.ordered_registry_manifest_sha256    = registry_sha
ticket.bound_manifests.ordered_registry_manifest_sha256 = registry_sha
```

该 computation 不调用 contract loader、mesh、march 或 collector；正式 frozen
contract 稍后仍由 `_verify_contract_authorization` 对同一 names/identities/chain
二次核验。

## 3. Issuance request exact schema

所有 G5 child artifacts 采用唯一单向顺序：

```text
Q issuance request
→ M invocation metadata
→ A review request
→ C clearance/rejection response
→ Z post-review trace metadata
→ final authorization ticket
```

后一个 artifact 可以绑定前一个；前一个只能声明后一个的 planned path、artifact
identity 和 call prefix，不能包含未来 raw/canonical digest。

top-level exact fields：

```text
artifact
schema_version
version
status
attempt_id
planned_authorization_path
planned_result_path
planned_marker_path
single_use_token_sha256
ticket_semantic_projection
ticket_semantic_projection_sha256
implementation_identity
planned_invocation
review_requirements
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_issuance_request
schema_version=1.0
version=S3ai-v2.3-issuance-request-v2
status=REQUEST_INDEPENDENT_REVIEW_NO_TICKET_EXISTS
```

`planned_invocation` exact fields：

```text
path
artifact
schema_version
version
call_prefix
```

固定 artifact/version 为：

```text
artifact=actual_wake_reachable_pressure_v23_review_invocation
schema_version=1.0
version=S3ai-v2.3-review-invocation-v1
```

Q 不含 M raw/canonical digest，因此不存在 Q↔M。

`review_requirements` exact：

```text
acceptance_prefilled=false
rejection_allowed=true
required_independence=genuine-cross-family
blocking_findings_must_be_empty_for_accept=true
```

### 3.1 Implementation identity anchor

`bound_artifacts.implementation_identity` 指向 strict JSON anchor envelope，
top-level exact fields：

```text
artifact
schema_version
version
identity
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_implementation_identity
schema_version=1.0
version=S3ai-v2.3-implementation-identity-v1
```

`identity` 必须是第 2.3 节唯一的 `ImplementationIdentityBodyV1`。body 中所有
identity string 均为 nonempty；path 通过 safe repo-relative/no-follow；
`source_trace_metadata_sha256` 是 lowercase raw SHA。ticket、projection、Q、
M、A 中的 nested object 必须 exact 等于 `anchor.identity`；anchor envelope
自身只由 `bound_artifacts.implementation_identity.{path,sha256}` 以 raw bytes
绑定，不能把 10/7 字段对象混为一谈。

`source_trace_metadata_path` 指向的 strict JSON exact fields：

```text
artifact
schema_version
version
provider
model
model_family
agent_id
trace_id
source_kind
source_artifact_map_sha256
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_implementation_source_trace
schema_version=1.0
version=S3ai-v2.3-implementation-source-trace-v1
source_kind=implementation_and_definition_audit_trace
```

五个 identity fields 必须 exact 等于 `anchor.identity` 中同名字段。

`source_artifact_map_sha256` 的唯一输入对象命名为
`ImplementationSourceArtifactMapV1`，exact keys：

```text
wrapper_source
g2_implementation_diff_audit
g4_definition_test_audit
```

每值 exact `{path,sha256}`，并逐对象 exact 等于 final ticket
`bound_artifacts` 中同名值；`path` 是 safe repo-relative，`sha256` 是对应文件
raw SHA。其 digest 唯一定义为：

```text
SHA256(_canonical_json_bytes(ImplementationSourceArtifactMapV1))
```

其中 `_canonical_json_bytes` 精确采用第 6 节规则。map 不增加独立文件或自由
顺序；parser 从 final `bound_artifacts` 正向投影这三个键后重算。

## 4. Invocation、review request 与 post-review trace

### 4.1 Invocation metadata

strict JSON exact fields：

```text
artifact
schema_version
version
trace_id
created_at_utc
allowed_trace_root
call_prefix
implementation_identity
requested_reviewer_provider
requested_reviewer_model
requested_reviewer_model_family
issuance_request_path
issuance_request_raw_sha256
issuance_request_canonical_sha256
ticket_semantic_projection_sha256
planned_review_request
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_review_invocation
schema_version=1.0
version=S3ai-v2.3-review-invocation-v1
allowed_trace_root=.aris/traces/research-review
```

`created_at_utc` 必须是 UTC RFC3339 `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`；
`trace_id/call_prefix` 匹配 `[a-z0-9][a-z0-9_-]{7,127}`。

`planned_review_request` exact：

```text
path
artifact
schema_version
version
call_prefix
```

固定 artifact/version 为：

```text
artifact=actual_wake_reachable_pressure_v23_review_request
schema_version=1.0
version=S3ai-v2.3-review-request-v1
```

M 绑定 Q raw/projection，但不含未来 A raw/canonical、response 或 trace digest。

### 4.2 Review request

A 是 canonical pretty strict JSON，top-level exact fields：

```text
artifact
schema_version
version
status
attempt_id
trace_id
call_prefix
issuance_request
invocation_metadata
implementation_identity
review_target
required_checks
response_contract
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_review_request
schema_version=1.0
version=S3ai-v2.3-review-request-v1
status=REQUEST_NEUTRAL_INDEPENDENT_ACCEPT_OR_REJECT
```

`issuance_request` exact
`{path,raw_sha256,canonical_sha256,ticket_semantic_projection_sha256}`；
`invocation_metadata` exact `{path,raw_sha256,canonical_sha256}`。

`review_target` exact
`{provider,model,model_family,required_independence}`，最后一项固定
`genuine-cross-family`。`required_checks` 必须 exact 等于下列 nonempty、
ordered、unique literal array，不能增删、换序或用同义词替换：

```text
[
  "G0_QUARANTINE_INTACT",
  "G2_SCIENCE_ZERO_DRIFT",
  "G3_DEPENDENCY_CAPTURE_ACCEPTED",
  "G4_DEFINITION_TESTS_ACCEPTED",
  "TICKET_SEMANTIC_PROJECTION_EXACT",
  "TOKEN_COMMITMENT_BOUND",
  "OLD_FAILURE_QUARANTINE_BOUND",
  "RESULT_NAMESPACE_AND_SCOPE_LOCKED",
  "HARD_NONCLAIMS_PRESERVED",
  "GENUINE_CROSS_FAMILY_TRACE_VERIFIED"
]
```

request 不得预填 verdict。

`response_contract` exact：

```text
accepted_artifact=S3ai-v2.3-successor-one-shot-execution-clearance
accepted_version=2
rejected_artifact=S3ai-v2.3-successor-one-shot-execution-rejection
rejected_version=1
response_path=<same allowed root/call-prefix>.response.json
post_review_trace_path=<same allowed root/call-prefix>.trace.json
```

### 4.3 Call-prefix/path derivation

G5 executable instantiation 冻结一个 exact `call_prefix`，必须满足上式 grammar。
路径机械派生：

```text
<allowed_root>/<frozen_run_directory>/<call_prefix>.invocation.json
<allowed_root>/<frozen_run_directory>/<call_prefix>.request.json
<allowed_root>/<frozen_run_directory>/<call_prefix>.response.json
<allowed_root>/<frozen_run_directory>/<call_prefix>.trace.json
```

`frozen_run_directory` 和 call_prefix 在 Q 中作为 planned path bytes 固定；M/A/C/Z
必须反向解析得到完全相同二者。禁止任何启发式 basename 选择或 alternate path。

### 4.4 Post-review trace metadata

由 review orchestrator 在 response 完成后写 strict JSON，exact fields：

```text
artifact
schema_version
version
trace_id
provider
model
model_family
agent_id
request_path
request_raw_sha256
response_path
response_raw_sha256
invocation_metadata_path
invocation_metadata_raw_sha256
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_post_review_trace
schema_version=1.0
version=S3ai-v2.3-post-review-trace-v1
```

所有 path/string nonempty，所有 SHA lowercase 64 hex。Z 在 C 后产生，所以可以
绑定 response raw；C 不含未来 Z hash。

ticket 在事后绑定其 raw SHA；clearance 不包含未来 trace metadata SHA。request、
response、invocation、trace metadata 必须位于 allowed root、共享 exact
`call_prefix`，safe relative、non-symlink、no-follow。

cross-family 必须由 parser 比较：

```text
review trace model_family != implementation_identity.model_family
review trace trace_id != implementation_identity.trace_id
clearance review identity == post-review trace identity
all request/response/invocation paths and hashes exact equal
```

字符串自报而没有 post-review trace 证据不算 independent。

## 5. Clearance exact schema

accepted clearance top-level exact fields：

```text
artifact
schema_version
version
attempt_id
verdict
review
bindings
grant
scope
blocking_findings
```

固定：

```text
artifact=S3ai-v2.3-successor-one-shot-execution-clearance
schema_version=1.0
version=2
attempt_id=S3ai-v2.2-transport-v2.3-successor-20260728_185556
verdict=ACCEPT_EXACT_V23_SUCCESSOR_ONE_SHOT_31_HISTORY_EXECUTION
blocking_findings=[]
```

### 5.1 `review` exact fields

```text
actual_independence
provider
model
model_family
agent_id
trace_id
implementation_model_family
invocation_metadata_path
invocation_metadata_raw_sha256
invocation_metadata_canonical_sha256
request_raw_sha256
request_canonical_sha256
issuance_request_raw_sha256
issuance_request_canonical_sha256
ticket_semantic_projection_sha256
```

`actual_independence=genuine-cross-family`，但 parser 仍以第 4.4 节 trace metadata
外证为准。

### 5.2 `bindings` exact fields

```text
reviewed_wrapper_sha256
reviewed_wrapper_definition_sha256
transport_preregistration_raw_sha256
dependency_capture_protocol_raw_sha256
authorization_schema_preregistration_raw_sha256
bound_artifact_map_sha256
execution_source_map_sha256
stable_runtime_identity_sha256
bootstrap_contract_canonical_sha256
dependency_manifest_raw_sha256
dependency_manifest_canonical_sha256
dependency_B_paths_sha256
dependency_U_paths_sha256
dependency_R_paths_sha256
g3_consensus_canonical_sha256
g3_audit_raw_sha256
g4_audit_raw_sha256
ordered_registry_manifest_sha256
frozen_definition_chain_sha256
old_failure_quarantine_canonical_sha256
```

这些字段不得由 clearance 自报。C 产生时的权威来源是
`Q.ticket_semantic_projection`；下表中的 `bound_artifacts` 和
`bound_manifests` 均指该 projection 内同名 object。final ticket 稍后必须由
第 7.1 节证明 `P(ticket) == Q.ticket_semantic_projection`。parser 按下列唯一
映射重算并 exact 比较：

```text
reviewed_wrapper_sha256
  = bound_artifacts.wrapper_source.sha256
reviewed_wrapper_definition_sha256
  = bound_artifacts.wrapper_definition.sha256
transport_preregistration_raw_sha256
  = bound_manifests.transport_preregistration_raw_sha256
dependency_capture_protocol_raw_sha256
  = bound_manifests.dependency_capture_protocol_raw_sha256
authorization_schema_preregistration_raw_sha256
  = bound_manifests.authorization_schema_preregistration_raw_sha256
bound_artifact_map_sha256
  = bound_manifests.bound_artifact_map_sha256
execution_source_map_sha256
  = bound_manifests.execution_source_map_sha256
stable_runtime_identity_sha256
  = bound_manifests.stable_runtime_identity_sha256
bootstrap_contract_canonical_sha256
  = bound_manifests.bootstrap_contract_canonical_sha256
dependency_manifest_raw_sha256
  = bound_manifests.dependency_manifest_raw_sha256
dependency_manifest_canonical_sha256
  = bound_manifests.dependency_manifest_canonical_sha256
dependency_B_paths_sha256
  = bound_manifests.dependency_B_paths_sha256
dependency_U_paths_sha256
  = bound_manifests.dependency_U_paths_sha256
dependency_R_paths_sha256
  = bound_manifests.dependency_R_paths_sha256
g3_consensus_canonical_sha256
  = bound_manifests.g3_consensus_canonical_sha256
g3_audit_raw_sha256
  = bound_artifacts.g3_independent_audit.sha256
g4_audit_raw_sha256
  = bound_artifacts.g4_definition_test_audit.sha256
ordered_registry_manifest_sha256
  = bound_manifests.ordered_registry_manifest_sha256
frozen_definition_chain_sha256
  = bound_manifests.frozen_definition_chain_sha256
old_failure_quarantine_canonical_sha256
  = bound_manifests.old_failure_quarantine_canonical_sha256
```

### 5.3 `grant` exact fields

```text
authorization_id
decision
single_use_token_sha256
authorization_ticket_creation_allowed
formal_execution_allowed
execution_limit
post_marker_retry_allowed
```

固定：

```text
decision=YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE
authorization_ticket_creation_allowed=true
formal_execution_allowed=true
execution_limit=1
post_marker_retry_allowed=false
```

### 5.4 `scope` exact fields

对 accepted 和 rejected C 都按 Q 正向构造，不引用未来 ticket：

```text
C.scope := deepcopy(Q.ticket_semantic_projection.scope)
           + {
               result_path: Q.planned_result_path,
               marker_path: Q.planned_marker_path
             }
```

G5 rejection 使用独立 artifact
`S3ai-v2.3-successor-one-shot-execution-rejection`、`version=1`，其 grant 全 false、
`blocking_findings` 非空；rejection 不能进入 ticket，也不能生成 token/ticket。
rejection top-level、review、bindings、grant、scope fields 与 accepted clearance
完全相同，仅固定：

```text
artifact=S3ai-v2.3-successor-one-shot-execution-rejection
version=1
verdict=REJECT_V23_SUCCESSOR_ONE_SHOT_EXECUTION
authorization_ticket_creation_allowed=false
formal_execution_allowed=false
execution_limit=0
post_marker_retry_allowed=false
blocking_findings=<nonempty ordered unique strings>
```

其 `authorization_id/single_use_token_sha256/decision` 仍必须分别 exact 等于
`Q.ticket_semantic_projection.authorization_plan.{id,
single_use_token_sha256,decision}`，以证明拒绝的是哪个 exact projection；这不
构成 grant。

## 6. Canonical encoding

所有 JSON：

```text
UTF-8 strict JSON types
duplicate keys rejected
NaN/Inf rejected
string keys
canonical: sort_keys=true,separators=(",",":"),ensure_ascii=false,no LF
pretty raw: indent=2,sort_keys=true,ensure_ascii=false,allow_nan=false,+ one LF
```

除明确写为 integer/bool/array/object 的字段外，所有字段均为 nonempty string；
所有 SHA 为 64 lowercase hex；integer 不接受 bool；所有 path 先通过各节规定的
safe root、lexical normalization、non-symlink、no-follow 规则。unknown/missing
field、duplicate array item 和替代 scalar representation 均拒绝。

ticket raw 必须与 pretty reserialization byte-identical。

ticket self digest：

1. 深复制 ticket；
2. 只将 `authorization.canonical_sha256` 置 64 个 `0`；
3. 对完整 ticket canonical JSON 取 SHA；
4. raw SHA 只由外部参数传入，不进入 ticket。

clearance、issuance、implementation identity/source trace、invocation、review
request 和 post-review trace 都没有 self-digest field；分别计算 raw/canonical
SHA并由其后继消费者绑定。

## 7. 同一 private loader seam 与 G3 等价

实现唯一 private seam：

```text
_load_and_verify_authorization_from_ticket_path(
  ticket_path,
  expected_authorization_sha256,
  second_audit_token,
  definition,
  dependency_manifest,
  source_fingerprints,
  stable_runtime_identity,
) -> _VerifiedAuthorization
```

seam 只信任 `ticket_path`。它必须：

1. strict parse ticket；
2. 从 `ticket.issuance_request.path` 派生并解析 Q；
3. 从 `Q.planned_invocation.path` 派生并解析 M；
4. 从 `M.planned_review_request.path` 派生并解析 A；
5. 从 ticket `second_bounded_audit` 派生 C/Z，并与
   `A.response_contract` paths exact cross-check；
6. 对 Q/M/A/C/Z 的 call-prefix、identity、paths、raw/canonical hashes逐项
   交叉相等。

### 7.1 Normative cross-object equality/inequality matrix

本节替代“同名或相关字段自行推断”。记：

```text
TKT = parsed final ticket
P   = P(TKT)
I   = parsed implementation anchor.identity
T   = 64 lowercase hex stdin decode 后的 32 raw bytes
path(X) = X 的 safe repo-relative observed path
raw(X)  = SHA256(X exact raw bytes)
can(X)  = SHA256(_canonical_json_bytes(parsed X object))
```

除显式 rejection 分支外，任一行不 exact equal 就 fail。

**Campaign、projection 和固定 namespace：**

```text
path(TKT)
  = Q.planned_authorization_path
  = fixed reserved authorization path

TKT.attempt_id
  = P.attempt_id
  = Q.attempt_id
  = A.attempt_id
  = C.attempt_id
  = fixed attempt id

Q.ticket_semantic_projection = P
Q.ticket_semantic_projection_sha256
  = SHA256(_canonical_json_bytes(P))
  = TKT.issuance_request.ticket_semantic_projection_sha256
  = TKT.second_bounded_audit.ticket_semantic_projection_sha256
  = C.review.ticket_semantic_projection_sha256

TKT.issuance_request.path             = path(Q)
TKT.issuance_request.raw_sha256       = raw(Q)
TKT.issuance_request.canonical_sha256 = can(Q)

Q.planned_result_path
  = P.result.path
  = TKT.result.path
  = C.scope.result_path
  = fixed v2.3 result path

Q.planned_marker_path
  = P.result.marker_path
  = TKT.result.marker_path
  = C.scope.marker_path
  = fixed v2.3 marker path

P.result = TKT.result
P.scope  = TKT.scope
C.scope without {result_path,marker_path} = P.scope
```

**Authorization plan 与 bearer commitment：**

```text
P.authorization_plan
  = projection of TKT.authorization defined in §1

Q.single_use_token_sha256
  = P.authorization_plan.single_use_token_sha256
  = TKT.authorization.single_use_token_sha256
  = C.grant.single_use_token_sha256
  = SHA256(T)

P.authorization_plan.id
  = TKT.authorization.id
  = C.grant.authorization_id

P.authorization_plan.decision
  = TKT.authorization.decision
  = C.grant.decision
```

accepted C 还必须：

```text
C.grant.formal_execution_allowed
  = P.authorization_plan.formal_execution_allowed = true
C.grant.execution_limit
  = P.authorization_plan.execution_limit = 1
C.grant.post_marker_retry_allowed
  = P.authorization_plan.post_marker_retry_allowed = false
C.grant.authorization_ticket_creation_allowed = true
```

rejected C 不产生 TKT；但其
`authorization_id/single_use_token_sha256/decision` 仍按上表绑定 Q/P，且：

```text
formal_execution_allowed=false
execution_limit=0
post_marker_retry_allowed=false
authorization_ticket_creation_allowed=false
```

**Implementation identity 与 reviewer identity（两个不同 equivalence class）：**

```text
TKT.implementation_identity
  = P.implementation_identity
  = Q.implementation_identity
  = M.implementation_identity
  = A.implementation_identity
  = I

M.trace_id
  = A.trace_id
  = C.review.trace_id
  = Z.trace_id

M.requested_reviewer_provider
  = A.review_target.provider
  = C.review.provider
  = Z.provider
M.requested_reviewer_model
  = A.review_target.model
  = C.review.model
  = Z.model
M.requested_reviewer_model_family
  = A.review_target.model_family
  = C.review.model_family
  = Z.model_family
C.review.agent_id = Z.agent_id
C.review.implementation_model_family = I.model_family
A.review_target.required_independence
  = C.review.actual_independence
  = genuine-cross-family

Z.model_family != I.model_family
Z.trace_id     != I.trace_id
```

因此 “implementation identity exact equal” 与 “reviewer cross-family unequal”
不再混为同一个 identity group。

**Q→M→A path、identity 和 prior-artifact bindings：**

```text
Q.planned_invocation.path           = path(M)
Q.planned_invocation.artifact       = M.artifact
Q.planned_invocation.schema_version = M.schema_version
Q.planned_invocation.version        = M.version
Q.planned_invocation.call_prefix    = M.call_prefix

M.issuance_request_path             = path(Q)
M.issuance_request_raw_sha256       = raw(Q)
M.issuance_request_canonical_sha256 = can(Q)
M.ticket_semantic_projection_sha256
  = Q.ticket_semantic_projection_sha256

M.planned_review_request.path           = path(A)
M.planned_review_request.artifact       = A.artifact
M.planned_review_request.schema_version = A.schema_version
M.planned_review_request.version        = A.version
M.planned_review_request.call_prefix
  = M.call_prefix = A.call_prefix

A.issuance_request.path
  = path(Q)
A.issuance_request.raw_sha256
  = raw(Q)
A.issuance_request.canonical_sha256
  = can(Q)
A.issuance_request.ticket_semantic_projection_sha256
  = Q.ticket_semantic_projection_sha256
A.invocation_metadata.path
  = path(M)
A.invocation_metadata.raw_sha256
  = raw(M)
A.invocation_metadata.canonical_sha256
  = can(M)
```

**A→C→Z 与 final ticket audit bindings：**

```text
A.response_contract.response_path          = path(C)
A.response_contract.post_review_trace_path = path(Z)

TKT.second_bounded_audit.invocation_metadata_path
  = path(M)
TKT.second_bounded_audit.invocation_metadata_raw_sha256
  = raw(M)
TKT.second_bounded_audit.invocation_metadata_canonical_sha256
  = can(M)
TKT.second_bounded_audit.request_path
  = path(A)
TKT.second_bounded_audit.request_raw_sha256
  = raw(A)
TKT.second_bounded_audit.request_canonical_sha256
  = can(A)
TKT.second_bounded_audit.response_path
  = path(C)
TKT.second_bounded_audit.response_raw_sha256
  = raw(C)
TKT.second_bounded_audit.response_canonical_sha256
  = can(C)
  = TKT.second_bounded_audit.clearance_canonical_sha256
TKT.second_bounded_audit.trace_metadata_path
  = path(Z)
TKT.second_bounded_audit.trace_metadata_raw_sha256
  = raw(Z)

Z.request_path                 = path(A)
Z.request_raw_sha256           = raw(A)
Z.response_path                = path(C)
Z.response_raw_sha256          = raw(C)
Z.invocation_metadata_path     = path(M)
Z.invocation_metadata_raw_sha256 = raw(M)

C.review.invocation_metadata_path
  = path(M)
C.review.invocation_metadata_raw_sha256
  = raw(M)
C.review.invocation_metadata_canonical_sha256
  = can(M)
C.review.request_raw_sha256
  = raw(A)
C.review.request_canonical_sha256
  = can(A)
C.review.issuance_request_raw_sha256
  = raw(Q)
C.review.issuance_request_canonical_sha256
  = can(Q)
```

M/A/C/Z 的四条 path 还必须全部由第 4.3 节同一个 allowed root、
frozen-run-directory 和 call-prefix 机械派生。

**Registry 与 manifest mirrors：**

```text
P.ordered_registry_manifest_sha256
  = TKT.ordered_registry_manifest_sha256
  = P.bound_manifests.ordered_registry_manifest_sha256
  = TKT.bound_manifests.ordered_registry_manifest_sha256
  = C.bindings.ordered_registry_manifest_sha256
  = §2.10 registry_sha

P.ordered_case_names   = TKT.ordered_case_names   = §2.10 names
P.case_identity_sha256 = TKT.case_identity_sha256 = §2.10 identities
```

accepted C 才允许产生 final ticket，且 G6 在写票据前必须再次证明
`P(TKT) == Q.ticket_semantic_projection` 及本矩阵全部成立。rejected C 到此终止，
不存在可供“未来 ticket 相等”掩盖差异的路径。

child paths 不能作为独立可信参数。public `_load_and_verify_authorization()` 只把
正式 `AUTHORIZATION_PATH` 作为 `ticket_path` 调用同一 seam；definition tests 和
G3 的 synthetic ticket 则在 ticket 自身字段内指向隔离 fixture paths，仍调用
完全相同 seam，禁止 monkeypatch 绕过 parser。

seam 前记录 `S_pre`，返回后连续记录 `S_1/S_2`，要求：

```text
S_pre == S_1 == S_2
file-backed dependency delta = 0
formal collector/mesh/march/marker/result calls = 0
```

positive fixture 必须通过
Q→M→A→C→Z→full ticket→projection→loader roundtrip；所有 fail controls 在
seam 内 marker 前终止。

## 8. Transport-only consumer/provenance migration

successor 独立 audit 通过后，明确允许且仅允许修改：

```text
_verify_contract_authorization
_verify_ticket_frozen_files_before_loader
_stable_execution_input
_augment_result
_validate_serialized_result
_run_authorized_once.publication_pre_link_gate
```

以消费 nested clearance，不得生成兼容性 flatten shadow。
`publication_pre_link_gate` 是当前 `_run_authorized_once` 内的 nested closure，
只允许扩展 audit-file/identity 重验，不得移动 marker、collector、validator 或
publication 顺序。

result `one_shot_provenance.second_bounded_audit` exact shape：

```text
verdict
review {actual_independence,provider,model,model_family,agent_id,trace_id}
issuance_request {path,raw_sha256,canonical_sha256,ticket_semantic_projection_sha256}
request {path,raw_sha256,canonical_sha256}
response {path,raw_sha256,canonical_sha256}
invocation_metadata {path,raw_sha256,canonical_sha256}
trace_metadata {path,raw_sha256}
clearance_canonical_sha256
```

contract/registry hashes从 `clearance.bindings` 读取；independence 从
`clearance.review` 读取。serialized validator 与 expected verified object 逐字段
exact 比较。科学 result/case/decision 字段、collector、aggregator 和物理 input
不变，G2 必须证明 science-zero-drift。

`verified.audit_files` exact 等于：

```text
all bound_artifacts paths
+ issuance request
+ review request
+ clearance response
+ invocation metadata
+ post-review trace metadata
+ implementation source trace metadata
```

这些 bytes 在 marker 前、history 后和 pre-link gate 全部 no-follow 复核。

## 9. Bearer 与 G6

stdin 只接受恰好 `64 lowercase hex + one LF + EOF`。大写、空白、额外 bytes、
argv/env token 均拒绝。token preimage 只在 live broker memory，G3/G4 accepted
后且 G5 request 前生成 commitment；本文件不授权现在生成。

byte domain 唯一定义：

```text
stdin_raw must have length 65
stdin_raw[0:64] must match ASCII [0-9a-f]{64}
stdin_raw[64:65] must equal b"\n"
the next read must return EOF
T := hex-decode(stdin_raw[0:64])      # exactly 32 raw bytes
single_use_token_sha256 := SHA256(T)  # lowercase hex digest
```

64-byte ASCII hex文本和结尾 LF 都不进入 commitment hash；public/private loader
接收的 `second_audit_token` 必须 exact 等于 32-byte `T`。

G6 exact protocol：

```text
open one fixed dirfd
→ O_EXCL/O_NOFOLLOW create owned temp
→ canonical pretty ticket bytes + LF
→ fsync(temp)
→ pre-link full revalidation and reserved absence
→ no-replace linkat owned temp to canonical name
→ fsync(dirfd)
→ no-follow reopen canonical; verify raw/canonical/inode/nlink
→ unlink only owned temp through same dirfd
→ fsync(dirfd)
→ verify temp absent and canonical unchanged
→ release token once to custodian
```

canonical link 成功后的任何 cleanup/hash/launch异常都使 namespace
`AUTHORIZATION_NAMESPACE_POISONED`；canonical 不删除、不覆盖，broker 销毁
custody，另开 successor。本 profile 不声称对恶意复制的 pre-marker token 提供
durable reuse protection。

## 10. Tests

除首版全部 fail controls外，必须新增：

- projection positive roundtrip、逐字段source map、excluded-field invariance、正向
  allowlist；
- projection 任意 self/projection/request/audit digest 注入拒绝；
- MD normative SHA binding；
- accepted/rejected clearance exact nested schema；
- post-review trace 与 invocation metadata 顺序/同 prefix/cross-family；
- private seam positive fixture 三时点 zero-delta；
- nested consumer/provenance exact roundtrip和 tamper rejection；
- `audit_files` union 缺一/多一/漂移；
- strict bearer；
- G6 link/canonical verify/temp unlink/fsync，每个阶段 fault injection；
- post-link failure 保留 canonical、标记 poisoned、绝不删除第三方文件；
-所有 tests 的 formal collector/mesh/march/marker/result call count=0。

## 11. Hard nonclaims

本 successor 不声称：

- 首版 schema 通过；
- 本 successor 已通过独立 audit或已允许实现；
- parser/consumer migration 已完成；
- production G3/G5/G6、token/ticket/history/physics result 已存在；
- pre-marker token reuse 有 durable/cryptographic enforcement；
- V4.1、claim tree、三点、118 或 Fig17/18/19 已修改；
- `RESEARCH_ACCIDENTAL_DRIFT` 之外的 adversarial protection。
