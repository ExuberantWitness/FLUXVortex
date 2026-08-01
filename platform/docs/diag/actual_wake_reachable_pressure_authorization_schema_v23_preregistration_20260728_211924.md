# S3ai-v2.2 / transport-v2.3 authorization schema 预注册 successor

**时间**：2026-07-28T21:19:24+08:00  
**artifact**：`actual_wake_reachable_pressure_authorization_schema_v23_preregistration_20260728_211924`  
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
- 20:51:55 第四版 successor MD SHA
  `c16d43ecd51058ba5f1e03fc0d36ace99c4855087f4576251657d4fd3dfcd7bc`、
  JSON SHA
  `a421ffd5dc04c236a05037adbb255f11afa8453d6caec0764a2d8d914378c9ad`；
- 21:02:55 两路只读第五次复审 `FAIL / 3 remaining blocker groups` MD SHA
  `5c93f6c5b37159106b69900a75b9a49aebdf3077bfd271dd2fdae29d669253d3`、
  JSON SHA
  `be043682e80c2d9065bd3f79139e4c0a0574a4a9066a5347f9349a38e363e52b`。
- 21:03:41 第五版 successor MD SHA
  `1e0c8e577a0ad02eded625443b5ba69cc579b4f4cea73a96afa58d600fdd49f6`、
  JSON SHA
  `82af94de8898c543cdaefdaa7ade44f085322b55c57b57bfecb97d83c7cf39c4`；
- 21:18:29 两路只读第六次复审 `FAIL / 3 remaining blocker groups` MD SHA
  `318ad94258d4edac57ab17e71441d5c978a4d20ddc704725342e9bcb37e9fe7f`、
  JSON SHA
  `cbd35dbb40a5e150986552ae571a322d8e5305d477de5336d9e42d43671f9add`。

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

### 0.1 `NamespaceProfileV1` 与唯一 path algorithm

同一 private parser 接受一个显式 typed profile，而不是读取 ticket 自报的
filesystem authority。exact fields：

```text
mode
repo_root
isolation_root
issuance_request_path
ticket_path
result_path
marker_path
allowed_trace_root
production_credential_allowed
```

所有 `*_root/*_path` 是 absolute `Path`，在 seam 入口冻结；profile 本身不是
JSON artifact。两种且仅两种 mode：

```text
PRODUCTION_RESERVED_V23
ISOLATED_DEFINITION_FIXTURE_V1
```

production profile：

```text
repo_root       = current fixed REPO_ROOT
isolation_root  = repo_root
issuance_request_path
  = repo_root/platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml.issuance.json
ticket_path     = fixed formal AUTHORIZATION_PATH
result_path     = fixed formal RESULT_PATH
marker_path     = fixed formal ATTEMPT_MARKER_PATH
allowed_trace_root
  = repo_root/.aris/traces/research-review
production_credential_allowed = true
```

fixture profile 的 owned base 唯一为：

```text
fixture_base
  = repo_root/.aris/definition-fixtures/transport-v23
```

fixture profile：

```text
repo_root       = current fixed REPO_ROOT
isolation_root  = fixture_base/<fixture_id>
fixture_id      = exact one component matching fixture-[0-9a-f]{24}
ticket_path     = isolation_root/authorization-ticket.json
issuance_request_path
  = isolation_root/authorization-ticket.json.issuance.json
result_path     = isolation_root/result.json
marker_path     = isolation_root/result.json.lock
allowed_trace_root
  = isolation_root/.aris/traces/research-review
production_credential_allowed = false
```

`fixture_base` 的 parent chain 必须已存在、逐级 non-symlink；fixture owner 在
seam 前用该 parent dirfd 对 exact `fixture_id` 做 exclusive `mkdirat(0700)`。
预存在、非 directory、symlink或不是本次 fixture owner创建的 root全部拒绝。
fixture结束只允许 owner从叶到根删除该 isolation root 内自己创建的 names；
不能递归删除未知或非 owned entry。

令 component-prefix `a <=p b` 表示 a 的完整 POSIX component tuple 是 b 的
前缀（含相等）。production dynamic namespace set 精确为 production profile 的：

```text
issuance_request_path
ticket_path
result_path
marker_path
allowed_trace_root
```

fixture set 精确为：

```text
isolation_root
issuance_request_path
ticket_path
result_path
marker_path
allowed_trace_root
```

对每个 fixture set成员 `F` 和每个 production set成员 `R`，必须同时
`not(F <=p R)` 且 `not(R <=p F)`。这是一条双向 component-prefix不可比规则，
不是字符串 startswith，也不是只禁 fixture“包含”production。

在 fixture 内：

- 四个 file paths `issuance_request/ticket/result/marker` 必须是
  isolation root 的 strict descendants，pairwise component-prefix不可比；
- `allowed_trace_root` 必须是 isolation root 的 strict descendant directory，
  并与上述四个 file paths双向 component-prefix不可比；
- Q 的 observed path 必须 exact 等于 `issuance_request_path`；
- M/A/C/Z 必须是 `allowed_trace_root` 的 strict descendants并满足第 4.3 节；
- 所有其他 synthetic artifact path必须是 isolation root的 strict descendant。

fixture 只允许 definition tests、G2/G3 no-history controls和 G6 publisher fault
fixtures；formal collector/mesh/march calls 必须为 0。它不能生成 production
credential，也不能让 public `_load_and_verify_authorization()` 返回。public
adapter 必须构造 production profile；tests/G3 直接调用同一 private seam并显式
传 fixture profile。这不是 monkeypatch parser、跳过字段或使用第二个 parser。

JSON 内所有 repo-relative path 的唯一算法：

1. 输入必须是 nonempty UTF-8 string，无 NUL、反斜杠或
   `*`/`?`/`[`/`]`；
2. `PurePosixPath(value)` 必须非 absolute，且 `value == pure.as_posix()`；
3. 每个 component 必须不为空且不属于 `.`/`..`；
4. lexical absolute target 只按 `repo_root.joinpath(*pure.parts)` 构造，不
   `resolve()`、不 `normpath()`、不做 alternate basename选择；
5. seam 只以
   `os.open(repo_root, O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)` 打开 root；
   `lstat(repo_root)` 必须是 directory且其 `(st_dev,st_ino)` exact 等于
   root-fd `fstat`；记录该 identity。所有 parent component 从该 fd逐级
   `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`，final read使用
   `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`；
6. profile/root/path 的 lexical相对关系在相同 component tuples上判断；
7. 任一 symlink、root inode漂移、非 regular final file或 path mismatch失败；
   S_2 后、seam return前必须以相同 flags从 fixed absolute `repo_root` fresh
   reopen并要求 identity等于最初 pinned root。G6 另按第 9 节在 link前后重走
   ticket parent。

待创建而尚不存在的 Q/ticket/result/marker/temp 只验证 parent chain和 lexical
name；存在性规则由相应阶段单独校验。

## 1. 无自引用的 TicketSemanticProjectionV1

projection 不再“从 ticket 删除若干字段”，而是由正向 allowlist 构造一个独立
object。精确字段：

```text
schema_version
namespace_profile
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
P.namespace_profile               := ticket.namespace_profile
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

Q 之前另有且只有一个因果可用 builder：

```text
BuildProjectionV1(FrozenTicketSemanticPlanInputsV1) -> projection
```

`FrozenTicketSemanticPlanInputsV1` 是 typed in-memory record，不是 future ticket，
exact fields：

```text
namespace_profile
authorization_id
single_use_token_sha256
bound_artifacts
implementation_identity
definition
dependency_manifest
source_fingerprints
stable_runtime_identity
old_failure_quarantine
```

其权威源和 builder 输出逐项固定：

```text
namespace_profile
  := explicit §0.1 profile
authorization_id
  := new nonempty G5-planned id, unequal retired id
single_use_token_sha256
  := SHA256(32-byte T), unequal retired commitment
bound_artifacts
  := complete pre-Q exact map from §2.4; every file already exists;
     its implementation anchor/source trace has already passed the
     pre-Q SourceMap_B validation in §3.1
implementation_identity
  := parsed bound implementation anchor.identity after that validation
definition
  := exact already-loaded wrapper definition mapping
dependency_manifest
  := exact already-parsed G3 reserved manifest
source_fingerprints
  := exact `_source_fingerprints(definition)` result
stable_runtime_identity
  := exact `_stable_runtime_identity()` result
old_failure_quarantine
  := exact §2.9 observed mapping and absences
```

`BuildProjectionV1` 输出：

```text
schema_version              := "1.0"
namespace_profile           := namespace_profile.mode
ticket_artifact             :=
  "actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556"
ticket_version              := "S3ai-v2.3-one-shot-authorization-v2"
attempt_id                  := fixed attempt id
scientific_protocol_version := "S3ai-v2.2"
transport_protocol_version  := "S3ai-v2.3"
assurance_profile           := "RESEARCH_ACCIDENTAL_DRIFT"
authorization_plan          := {
  id: authorization_id,
  external_raw_sha256_required: true,
  single_use_token_sha256: single_use_token_sha256,
  formal_execution_allowed: true,
  execution_limit: 1,
  decision:
    "YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE",
  post_marker_retry_allowed: false
}
bound_artifacts             := deep copy validated bound_artifacts
implementation_identity     := deep copy validated implementation_identity
definition_chain            := §2.10 projection from definition
execution_sources           := deep copy source_fingerprints
stable_runtime_identity     := deep copy stable_runtime_identity
bound_manifests             := recompute every field by §2.5
result                      := {
  path: repo-relative(namespace_profile.result_path),
  marker_path: repo-relative(namespace_profile.marker_path),
  overwrite_allowed: false,
  latest_pointer_write_allowed: false,
  old_canonical_result_write_allowed: false,
  atomic_no_replace_required: true
}
scope                       := exact §2.8 mapping
old_failure_quarantine      := deep copy validated old_failure_quarantine
ordered_case_names          := §2.10 names
case_identity_sha256        := §2.10 identities
ordered_registry_manifest_sha256 := §2.10 registry_sha
```

builder 不读取、不接受也不预测 Q/M/A/C/Z path、raw/canonical digest或 ticket
self digest。因果规则：

```text
Q.ticket_semantic_projection
  = BuildProjectionV1(plan inputs)
Q.ticket_semantic_projection_sha256
  = SHA256(canonical JSON of that object)

accepted branch after C/Z:
  P(final TKT)
    = BuildProjectionV1(the same frozen plan inputs)
    = Q.ticket_semantic_projection

rejected branch:
  C binds Q/Build only; P and TKT do not exist
```

final-ticket extractor P 与 pre-ticket builder是两条不同输入路径、同一个唯一
codomain schema；positive tests 必须逐字段证明它们相等，任何第二种手工 Q
projection组装均禁止。

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

这是 production profile 的唯一 reserved path；fixture profile 使用 §0.1
`ticket_path` 且必须与该 path 不同。

production Q reserved path：

```text
platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml.issuance.json
```

它 exact 等于 production profile 的 `issuance_request_path`；fixture Q exact
使用 fixture profile 的同名 path。任何 ticket-declared alternate Q path失败。

后缀虽为 `.yaml`，raw bytes 必须是 duplicate-safe canonical pretty JSON subset。
顶层 exact fields：

```text
artifact
schema_version
version
status
namespace_profile
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
attempt_id=S3ai-v2.2-transport-v2.3-successor-20260728_185556
scientific_protocol_version=S3ai-v2.2
transport_protocol_version=S3ai-v2.3
assurance_profile=RESEARCH_ACCIDENTAL_DRIFT
```

profile-dependent exact：

```text
production:
  namespace_profile=PRODUCTION_RESERVED_V23
  status=accepted_after_g5_independent_bounded_audit
fixture:
  namespace_profile=ISOLATED_DEFINITION_FIXTURE_V1
  status=synthetic_definition_fixture_no_production_authority
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
path=repo-relative(namespace_profile.result_path)
marker_path=repo-relative(namespace_profile.marker_path)
overwrite_allowed=false
latest_pointer_write_allowed=false
old_canonical_result_write_allowed=false
atomic_no_replace_required=true
```

production profile 因而得到固定正式 result/marker literals；fixture profile得到
显式 isolation root内的 pair，二者使用同一字段校验。

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

本节 `guard` 唯一指已经在 wrapper module import阶段加载的 module object
`actual_wake_reachable_pressure_obstruction_v2_guard`，并必须同时满足：

```text
guard.__file__ source path
  = repo_root/platform/actual_wake_reachable_pressure_obstruction_v2_guard.py
source_fingerprints[
  "platform/actual_wake_reachable_pressure_obstruction_v2_guard.py"
]
  = SHA256(no-follow exact guard source bytes)
guard is the same module object already used by the bound wrapper
```

seam 内不得 import、reload、`importlib` 解析或替换 guard。

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

每值 exact `{path,sha256}`；`path` 是 safe repo-relative，`sha256` 是对应文件
raw SHA。其 digest 唯一定义为：

```text
SHA256(_canonical_json_bytes(ImplementationSourceArtifactMapV1))
```

其中 `_canonical_json_bytes` 精确采用第 6 节规则。map 不增加独立文件或自由
顺序。它有两个有序、因果独立的 producer：

```text
pre-Q:
  SourceMap_B := exact positive projection of
    FrozenTicketSemanticPlanInputsV1.bound_artifacts
  keys := the three exact keys above
  source trace source_artifact_map_sha256
    = SHA256(canonical JSON of SourceMap_B)

accepted final-ticket verification:
  SourceMap_T := exact positive projection of TKT.bound_artifacts
  SourceMap_T = SourceMap_B
  SHA256(canonical JSON of SourceMap_T)
    = source trace source_artifact_map_sha256

rejected:
  validate SourceMap_B only; TKT and SourceMap_T do not exist
```

构造 plan/Build/Q 之前，seam 或 G5 pre-Q builder必须已经读取并 raw-bind anchor与
source trace，验证 `SourceMap_B`。它不得等待、预测或反向读取 final ticket。
accepted parser 在 TKT 出现后执行第二次独立投影；两次等值由 `P(TKT)=Build`
间接且由本节 direct equality同时证明。

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
allowed_trace_root=repo-relative(namespace_profile.allowed_trace_root)
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

`frozen_run_directory` 必须恰好一个 component，匹配
`[0-9]{4}-[0-9]{2}-[0-9]{2}_run[0-9]{2}`；它由
`Q.planned_invocation.path` 在 exact allowed root 后的唯一中间 component 给出。
call_prefix 由同一路径 filename 去掉 exact `.invocation.json` suffix 后给出。
Q/M/A/C/Z 必须反向解析得到完全相同二者。禁止多层 run directory、空 component、
启发式 basename、alternate suffix 或 alternate path。

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
request 和 post-review trace 都没有 self-digest field。只按各 exact schema
枚举的字段存储 binding：Q/M/A/C 同时绑定 raw/canonical；implementation anchor、
implementation source trace和 Z 只存 raw binding。所有对象仍 strict parse并可
重算 canonical SHA，但未枚举的 canonical digest不声称被持久字段绑定。

## 7. 同一 private loader seam 与 G3 等价

实现唯一 private seam：

```text
_load_and_verify_authorization_from_ticket_path(
  namespace_profile,
  ticket_path,
  expected_authorization_sha256,
  second_audit_token,
  definition,
  dependency_manifest,
  source_fingerprints,
  stable_runtime_identity,
) -> _VerifiedAuthorization
```

seam 对 filesystem namespace只信任显式 `namespace_profile` 和
`ticket_path`；入口第一条 hard equality 是
`ticket_path == namespace_profile.ticket_path`。seam 先按第 0.1 节 pin root，
再读 ticket；child artifact paths仍只能从 ticket/Q/M/A派生，但 observed Q
必须 exact 等于 `namespace_profile.issuance_request_path`。它必须：

1. strict parse ticket；
2. 要求 `ticket.issuance_request.path` 等于 profile-owned Q path，再派生并解析 Q；
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
B   = BuildProjectionV1(the frozen pre-ticket plan inputs)
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
  = repo-relative(namespace_profile.ticket_path)

path(Q)
  = TKT.issuance_request.path
  = repo-relative(namespace_profile.issuance_request_path)

expected_authorization_sha256
  = raw(TKT)
```

`expected_authorization_sha256` 必须先通过 64 lowercase hex 校验，且不得等于
retired authorization raw SHA；它表示 exact ticket raw bytes（含 pretty JSON
唯一结尾 LF）的 SHA，不是 canonical-object digest。任何 mismatch 在 child
artifact read、marker、collector之前失败。

```text
TKT.namespace_profile
  = P.namespace_profile
  = B.namespace_profile
  = namespace_profile.mode

TKT.attempt_id
  = P.attempt_id
  = B.attempt_id
  = Q.attempt_id
  = A.attempt_id
  = C.attempt_id
  = fixed attempt id

Q.ticket_semantic_projection = B
accepted branch only: P = B
Q.ticket_semantic_projection_sha256
  = SHA256(_canonical_json_bytes(B))
  = TKT.issuance_request.ticket_semantic_projection_sha256
  = TKT.second_bounded_audit.ticket_semantic_projection_sha256
  = C.review.ticket_semantic_projection_sha256

TKT.issuance_request.path             = path(Q)
TKT.issuance_request.raw_sha256       = raw(Q)
TKT.issuance_request.canonical_sha256 = can(Q)

Q.planned_result_path
  = B.result.path
  = P.result.path                  # accepted branch only
  = TKT.result.path
  = C.scope.result_path
  = repo-relative(namespace_profile.result_path)

Q.planned_marker_path
  = B.result.marker_path
  = P.result.marker_path           # accepted branch only
  = TKT.result.marker_path
  = C.scope.marker_path
  = repo-relative(namespace_profile.marker_path)

B.result = P.result = TKT.result   # accepted branch only for P/TKT
B.scope  = P.scope  = TKT.scope    # accepted branch only for P/TKT
C.scope without {result_path,marker_path} = B.scope
```

**Authorization plan 与 bearer commitment：**

```text
B.authorization_plan
  = the authorization plan constructed by BuildProjectionV1
accepted branch only:
  P.authorization_plan
    = projection of TKT.authorization defined in §1
    = B.authorization_plan

Q.single_use_token_sha256
  = B.authorization_plan.single_use_token_sha256
  = P.authorization_plan.single_use_token_sha256  # accepted only
  = TKT.authorization.single_use_token_sha256
  = C.grant.single_use_token_sha256
  = SHA256(T)

B.authorization_plan.id
  = P.authorization_plan.id                         # accepted only
  = TKT.authorization.id
  = C.grant.authorization_id

B.authorization_plan.decision
  = P.authorization_plan.decision                   # accepted only
  = TKT.authorization.decision
  = C.grant.decision
```

本表及后续矩阵中的 `P.*`/`TKT.*` terms 只在 accepted branch存在；rejected
branch保留 B、Q、M、A、C、Z terms 的同一等式子集，不虚构 final ticket。

accepted C 还必须：

```text
C.grant.formal_execution_allowed
  = B.authorization_plan.formal_execution_allowed = true
C.grant.execution_limit
  = B.authorization_plan.execution_limit = 1
C.grant.post_marker_retry_allowed
  = B.authorization_plan.post_marker_retry_allowed = false
C.grant.authorization_ticket_creation_allowed = true
```

rejected C 不产生 TKT；但其
`authorization_id/single_use_token_sha256/decision` 仍按上表绑定 Q/B，且：

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
  = B.implementation_identity
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
M.allowed_trace_root
  = repo-relative(namespace_profile.allowed_trace_root)

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
B.ordered_registry_manifest_sha256
  = P.ordered_registry_manifest_sha256
  = TKT.ordered_registry_manifest_sha256
  = B.bound_manifests.ordered_registry_manifest_sha256
  = P.bound_manifests.ordered_registry_manifest_sha256
  = TKT.bound_manifests.ordered_registry_manifest_sha256
  = C.bindings.ordered_registry_manifest_sha256
  = §2.10 registry_sha

B.ordered_case_names = P.ordered_case_names
  = TKT.ordered_case_names = §2.10 names
B.case_identity_sha256 = P.case_identity_sha256
  = TKT.case_identity_sha256 = §2.10 identities
```

accepted C 才允许产生 final ticket，且 G6 在写票据前必须再次证明
`P(TKT) == Q.ticket_semantic_projection` 及本矩阵全部成立。rejected C 到此终止，
不存在可供“未来 ticket 相等”掩盖差异的路径。

child paths 不能作为独立可信参数。public `_load_and_verify_authorization()` 只把
§0.1 production profile和正式 `AUTHORIZATION_PATH` 传给同一 seam；definition
tests 和 G3 则显式传 fixture profile，synthetic ticket自身字段指向该 profile
的隔离 paths。两者执行完全相同 parser/schema/matrix；禁止 monkeypatch parser、
跳过 field check或让 fixture profile进入 public adapter。

snapshot constructor 唯一为当前 bound wrapper 中已经存在的
`_loaded_dependency_state()`，不得换成 `sys.modules` names-only、import log或
manifest projection。它返回 lexically sorted
`dict[absolute_canonical_path, fingerprint_mapping]`；fingerprint mapping、
native `/proc/self/maps` identity和 sorted module aliases全部按该 bound function
现有输出保留，比较 domain 是完整 strict JSON object。

调用时点：

```text
S_pre := _loaded_dependency_state() immediately before ticket raw read
S_1   := _loaded_dependency_state() after all Q/M/A/C/Z/artifact checks
S_2   := _loaded_dependency_state() immediately after S_1, before return
```

private seam在 S_2 equality通过后才返回；不是“返回后由另一 wrapper猜测”。
要求：

```text
S_pre == S_1 == S_2
canonical JSON bytes(S_pre) == canonical JSON bytes(S_1)
                           == canonical JSON bytes(S_2)
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

G6 namespace 唯一定义：

```text
ticket_raw_sha := SHA256(_pretty_json_bytes(TKT))
ticket_dir     := namespace_profile.ticket_path.parent
canonical_name:= namespace_profile.ticket_path.name
temp_name      := "." + canonical_name
                  + ".g6-" + ticket_raw_sha[0:24] + ".tmp"
```

`canonical_name/temp_name` 都必须是单一 POSIX component，无 `/`、`.`/`..` 或
alternate spelling。G6 通过 §0.1 root dirfd逐级 no-follow只打开一次
`ticket_dir`，记录其 `(st_dev,st_ino)`；后续 temp create、link、canonical
reopen、temp unlink全部只使用该 dirfd和上述两个 exact names。temp 预存在即
失败且绝不删除；只有本次 `O_EXCL` 创建并记录 inode的 temp才属于本次发行。
production issuance 只接受 production profile + accepted C；fixture profile只
允许同算法 fault tests，不释放 production bearer、不启动 history。

记 owned temp 写完并 fsync后的：

```text
owned_identity := (fstat(temp_fd).st_dev, st_ino, st_size)
```

`st_size` 必须等于 exact pretty ticket raw byte length。所有 `nlink` 比较为
exact integer，不接受“至少”。G6 exact state machine：

```text
S0 PINNED
  fresh-open fixed repo_root with §0.1 flags
  walk lexical ticket parent and pin ticket_dir identity D
  require canonical_name ENOENT and temp_name ENOENT

S1 OWNED_TEMP
  O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC create temp_name
  write exact pretty ticket bytes; fsync(temp_fd)
  require temp_fd is regular, st_nlink=1
  no-follow stat(temp_name) has same owned_identity and st_nlink=1

S2 PRELINK_VERIFIED
  perform full Q/M/A/C/Z/projection/credential revalidation
  fresh-open fixed repo_root and re-walk ticket parent
  require re-walk parent (st_dev,st_ino)=D
  require canonical_name ENOENT
  require temp_fd/temp_name still same owned_identity and nlink=1

S3 LINKED
  linkat(temp_name, canonical_name), no replace
  fsync(ticket_dir_fd)
  require temp_fd, no-follow temp_name and no-follow canonical_name are regular
  require all three have exact same (st_dev,st_ino,st_size)=owned_identity
  require all three st_nlink=2
  no-follow open/read canonical bytes; require raw/canonical/self digest exact

S4 CANONICAL_ONLY
  immediately before unlink, repeat temp-name owned_identity and nlink=2 check
  unlink(temp_name) through the pinned ticket_dir_fd
  fsync(ticket_dir_fd)
  require temp_name ENOENT
  require temp_fd and no-follow canonical are same owned_identity, st_nlink=1
  require canonical raw/canonical/self digest unchanged
  fresh-open fixed repo_root and re-walk ticket parent
  require re-walk parent identity=D and canonical name resolves to owned_identity

S5 RELEASED
  only after every S4 equality passes, release T once to custodian
```

从 S1 到 linkat成功前的任一 failure 是 `PRELINK_FAILURE`。cleanup只允许：

```text
no-follow stat(temp_name) == owned_identity and nlink=1
→ unlink exact temp_name through pinned ticket_dir_fd
→ fsync(ticket_dir_fd)
→ require temp_name ENOENT and canonical_name ENOENT
```

若 temp name不再指向 owned inode、cleanup/fsync/absence检查失败，则不删除未知
entry，状态转为 `AUTHORIZATION_NAMESPACE_POISONED`，broker销毁 custody。
若 temp 尚未由本次 O_EXCL成功创建，则没有 owned cleanup authority。

linkat成功是唯一 commit boundary。S3以后任何 hash、nlink、unlink、fsync、
parent re-walk或 launch异常都转为
`AUTHORIZATION_NAMESPACE_POISONED`；不删除、不覆盖 canonical，也不尝试清理
temp link，broker销毁 custody并另开 successor。只有 S5 可以释放 token。
本 profile不声称对恶意复制的 pre-marker token提供 durable reuse protection。

## 10. Tests

除首版全部 fail controls外，必须新增：

- projection positive roundtrip、逐字段source map、excluded-field invariance、正向
  allowlist；
- pre-ticket BuildProjection positive/rejection causality与 final P(TKT)==Build；
- production/fixture NamespaceProfile交叉拒绝、Q profile ownership、双向
  component-prefix隔离、exact repo-root/path/run-dir/temp-name；
- projection 任意 self/projection/request/audit digest 注入拒绝；
- MD normative SHA binding；
- accepted/rejected clearance exact nested schema；
- post-review trace 与 invocation metadata 顺序/同 prefix/cross-family；
- private seam positive fixture 三时点 zero-delta；
- exact guard source binding、S constructor和 external raw ticket SHA mismatch；
- nested consumer/provenance exact roundtrip和 tamper rejection；
- `audit_files` union 缺一/多一/漂移；
- strict bearer；
- G6 `nlink=1→2→1`、三名同 inode、parent re-walk、owned pre-link cleanup以及
  link/canonical/temp unlink/fsync每个阶段 fault injection；
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
