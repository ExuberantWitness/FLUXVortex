# S3ai-v2.2 / transport-v2.3 G5/G6 authorization schema 预注册

**时间**：2026-07-28T20:16:39+08:00  
**artifact**：`actual_wake_reachable_pressure_authorization_schema_v23_preregistration_20260728_201639`  
**attempt id**：`S3ai-v2.2-transport-v2.3-successor-20260728_185556`  
**状态**：`SCHEMA PREREGISTERED / PARSER IMPLEMENTATION ALLOWED / ISSUANCE AND EXECUTION LOCKED`  
**claim-state change**：`false`

## 0. 本次授权边界

当前 v2.3 `_load_and_verify_authorization()` 在完成旧票拒绝和有限读取后必然
fail-closed。为了冻结最终 wrapper 并进入生产 G3，本工件授权的唯一修改是：

- 实现严格、只读的 v2.3 authorization、issuance request 和 clearance parser；
- 实现 authorization path 的 dependency-zero-delta 检查；
- 增加 no-history fail-closed 负控；
- 收紧 bearer stdin 的精确语法；
- 让正式 public entry 在 reserved ticket 不存在时继续 fail-closed。

本工件不允许创建 issuance request、clearance、production token、authorization
ticket、marker 或 result；不允许执行 collector/mesh/time march；不允许改
气动公式、常数、claim state、31-case registry 或 frozen aggregator。

合法顺序：

```text
schema freeze
→ parser + negative controls
→ final wrapper G2/G4
→ executable G3 successor + production closure
→ G5 issuance request and independent clearance
→ G6 atomic ticket issuance and one-shot launch
```

当前 wrapper SHA `c88846a6f1503301ead37c0e03190b756534cbe12f3c2a4a3ef2b619f16eb30a`
是 preimplementation identity。parser 修改后必须重新绑定 wrapper、63-source map、
bootstrap 和所有相关 audit；旧 `209/218` 不是生产闭包。

## 1. 三个循环的破环

### 1.1 Parser—G3—ticket

先实现 parser，但保持正式 `AUTHORIZATION_PATH` 不存在。生产 G3 在最终 parser
进入 wrapper 后执行，并证明 authorization verification path：

- 无函数内 import、`importlib/__import__` 或 `ctypes.CDLL/dlopen`；
- 只调用 bootstrap 已加载的 helper；
- synthetic success/failure parser 前后 file-backed dependency exact equal；
- capture-only pre-marker path 与正式 auth 后 baseline 在 dependency loading 上
  等价。

正式 runner 必须在 auth loader 前后取 fixed-point dependency snapshot 并要求
零 delta。失败发生在 marker 和 collector 之前。

### 1.2 Token—G5

选择 custody-precommit：

1. 只在 G3/G4 accepted 后启动冻结的独立 broker；
2. broker 生成恰好 32 个 CSPRNG bytes `T`，只公开 `H(T)=SHA256(T)`；
3. `T` 只保存在 broker live memory，不进入 repo、`/tmp`、trace、argv、env、
   shell history 或日志；
4. issuance request 和 G5 clearance 绑定 `H(T)`；
5. G6 ticket 原子写入并复核后，broker 才通过专用 pipe 将 `T` 一次性交给
   execution custodian；
6. G5 拒绝、broker 中断、G6 失败或 namespace poisoned 时销毁 custody，
   `T` 永不复用，另开 timestamped issuance successor。

不声称 Python immutable `bytes` 可可信 zeroize；只声明不持久化、broker 退出和
commitment 不复用。

### 1.3 Clearance—ticket

clearance 不绑定 final authorization raw/canonical digest，避免 ticket 内含
clearance hash造成自引用。

G5 前先冻结 `issuance_request.json`。它包含完整
`ticket_semantic_projection`、`H(T)`、paths/digests/scope，但不含最终 response
hash和 authorization self-digest。G5 clearance 绑定 request raw/canonical SHA
及 projection SHA。G6 ticket 的 semantic projection 必须逐字段等于 request，
只额外加入 second-audit references 和 authorization self-digest。

## 2. Authorization ticket

reserved path：

```text
platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml
```

虽为 `.yaml` 后缀，raw bytes 必须是 canonical pretty JSON subset；YAML
anchor、alias、merge、comment、替代标量和 duplicate key 均拒绝。

顶层字段精确为：

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
version=S3ai-v2.3-one-shot-authorization-v1
status=accepted_after_g5_independent_bounded_audit
attempt_id=S3ai-v2.2-transport-v2.3-successor-20260728_185556
scientific_protocol_version=S3ai-v2.2
transport_protocol_version=S3ai-v2.3
assurance_profile=RESEARCH_ACCIDENTAL_DRIFT
```

### 2.1 `authorization`

精确字段：

```text
id
canonical_sha256
external_raw_sha256_required
single_use_token_sha256
formal_execution_allowed
execution_limit
token_presentation_limit
decision
retry_allowed
```

固定语义：

```text
canonical_sha256=<self-zero canonical digest>
external_raw_sha256_required=true
single_use_token_sha256=H(T)
formal_execution_allowed=true
execution_limit=1
token_presentation_limit=1
decision=YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE
retry_allowed=false
```

`id` 必须由 issuance request 预先冻结，且不得等于旧 id。旧 raw SHA、旧 token
commitment 和任何已使用 commitment 均拒绝。

### 2.2 `issuance_request`

精确字段：

```text
path
raw_sha256
canonical_sha256
ticket_semantic_projection_sha256
```

### 2.3 `bound_artifacts`

键集合由 executable G3 successor 冻结，至少包含：

```text
wrapper_source
wrapper_definition
transport_preregistration
authorization_schema_preregistration
bootstrap_contract
g0_quarantine_audit
g2_implementation_diff_audit
g3_capture_preregistration
dependency_manifest
g3_consensus_certificate
g3_independent_audit
g4_definition_test_audit
interpretation_contract_markdown
interpretation_contract_json
```

每项精确为 `{path, sha256}`。path 必须 safe repo-relative、无 `..`、非
symlink，并以 no-follow 读取。映射键不能在 G6 时增删。

### 2.4 `bound_manifests`

精确字段：

```text
bound_artifact_map_sha256
execution_source_map_sha256
stable_runtime_identity_sha256
ordered_registry_manifest_sha256
frozen_definition_chain_sha256
transport_preregistration_raw_sha256
dependency_manifest_raw_sha256
dependency_manifest_canonical_sha256
dependency_B_paths_sha256
dependency_U_paths_sha256
dependency_R_paths_sha256
g3_consensus_canonical_sha256
bootstrap_contract_canonical_sha256
old_failure_quarantine_canonical_sha256
issuance_request_raw_sha256
issuance_request_canonical_sha256
ticket_semantic_projection_sha256
```

`B/U/R` 必须在 G3 manifest 中按 lexical canonical path 排序；各 digest 是相应
路径数组的 canonical JSON SHA256，不能让顺序成为自由度。

### 2.5 `second_bounded_audit`

精确字段：

```text
request_path
request_raw_sha256
request_canonical_sha256
response_path
response_raw_sha256
clearance_canonical_sha256
trace_metadata_path
trace_metadata_sha256
```

三项文件均 no-follow 读取。request/response/metadata 必须 strict UTF-8 JSON；
Markdown/YAML、duplicate keys 和 noncanonical values 拒绝。

### 2.6 `result`

精确字段和值：

```text
path=platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json
marker_path=platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json.lock
overwrite_allowed=false
latest_pointer_write_allowed=false
old_canonical_result_write_allowed=false
atomic_no_replace_required=true
retry_allowed=false
```

### 2.7 `scope`

精确字段和值：

```text
one_31_history_successor_execution_only=true
transport_failure_successor_not_retry=true
claim_state_change_allowed=false
production_activation_allowed=false
force_hp_state_ves_118_fig_allowed=false
malicious_local_writer_or_swap_restore_protection=false
```

不得仅以 `.get()` 校验部分字段；nested schema 必须 exact keys。

## 3. Issuance request

顶层精确字段：

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
review_requirements
```

固定：

```text
artifact=actual_wake_reachable_pressure_v23_issuance_request
schema_version=1.0
version=S3ai-v2.3-issuance-request-v1
status=REQUEST_INDEPENDENT_REVIEW_NO_TICKET_EXISTS
```

projection 必须包含 final ticket 除以下动态字段外的全部语义：

- `authorization.canonical_sha256`；
- `second_bounded_audit` 的最终 response/trace hashes；
- issuance request 自身的 raw/canonical binding。

它必须覆盖 attempt/protocol/profile、authorization id/decision/limits/token
commitment、wrapper/bootstrap/G0–G4、dependency manifest 与 B/U/R、source/
runtime/definition/registry、result/marker、scope 和 old quarantine。

`review_requirements` 精确为：

```text
acceptance_prefilled=false
rejection_allowed=true
required_independence=genuine-cross-family
blocking_findings_must_be_empty_for_accept=true
```

request 不得诱导 reviewer 只返回 ACCEPT。

## 4. Clearance

clearance 是 strict JSON，顶层精确字段：

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

accepted 固定值：

```text
artifact=S3ai-v2.3-successor-one-shot-execution-clearance
schema_version=1.0
version=1
attempt_id=S3ai-v2.2-transport-v2.3-successor-20260728_185556
verdict=ACCEPT_EXACT_V23_SUCCESSOR_ONE_SHOT_31_HISTORY_EXECUTION
```

### 4.1 `review`

精确字段：

```text
actual_independence
provider
model
model_family
agent_id
trace_id
trace_metadata_sha256
request_raw_sha256
issuance_request_raw_sha256
issuance_request_canonical_sha256
ticket_semantic_projection_sha256
```

`actual_independence=genuine-cross-family` 不能靠字符串自证；trace metadata 必须
证明 reviewer family 与 implementation family 不同，否则 REJECT。

### 4.2 `bindings`

精确键集合由 issuance request 冻结，至少镜像：

```text
reviewed_wrapper_sha256
reviewed_wrapper_definition_sha256
transport_preregistration_raw_sha256
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

### 4.3 `grant`

精确字段和值：

```text
authorization_id=<issuance request id>
decision=YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE
single_use_token_sha256=H(T)
authorization_ticket_creation_allowed=true
formal_execution_allowed=true
execution_limit=1
token_presentation_limit=1
retry_allowed=false
```

`scope` 必须逐项镜像 ticket scope 和固定 result/marker path。只有
`blocking_findings == []` 才允许 G6。REJECT 可包含 findings，但不得创建 ticket。

## 5. Canonical encoding

通用 canonical JSON：

```text
strict JSON types
string mapping keys
sort_keys=true
separators=(",", ":")
ensure_ascii=false
allow_nan=false
UTF-8
no trailing newline
duplicate keys rejected
```

authorization canonical digest：

1. 深复制完整 payload；
2. 只把 `authorization.canonical_sha256` 替换为 64 个 `"0"`；
3. 不清零其他字段；
4. 对 canonical JSON bytes 取 SHA256。

ticket raw bytes 必须严格等于：

```text
json.dumps(indent=2, sort_keys=true, ensure_ascii=false, allow_nan=false)
+ "\n"
```

raw SHA 不写入 ticket 自身；由 G6 写后计算并通过
`--expected-authorization-sha256` 外部传入。

clearance 无 self-field，canonical SHA 为整个 object 的 canonical JSON SHA；
另行绑定 raw SHA。issuance request 同时绑定 raw/canonical SHA。必须满足：

```text
ticket projection object == issuance_request projection object
ticket projection SHA == request projection SHA == clearance projection SHA
```

## 6. Parser 职责

实现拆分为：

```text
_parse_v23_authorization_schema(raw)
_parse_v23_issuance_request(raw)
_parse_v23_clearance(raw)
_ticket_semantic_projection(payload)
_load_and_verify_authorization(...)
_require_verified_v23_identity(...)
```

loader 严格按序：

1. 拒绝 retired SHA/id/token；
2. no-follow 读取 ticket，验证外部 raw SHA和 canonical raw bytes；
3. exact schema、self-digest；
4. no-follow 加载 issuance request并核对 projection；
5. 核对 32-byte bearer commitment；
6. 核对所有 bound artifacts、manifest raw/canonical/B/U/R；
7. 核对 source map、stable runtime、bootstrap、definition chain、registry；
8. strict 解析 audit request/response/trace metadata；
9. 由 metadata 验证 genuine cross-family；
10. 核对 clearance 全部镜像字段、result/scope/quarantine；
11. 返回完整 `_VerifiedAuthorization`。

现有 dataclass 字段全部填充，不增加隐式授权。`_require_verified_v23_identity()`
必须 exact schema 复核。`_review_trace_hashes()` 必须使用 duplicate-safe JSON
parser。

authorization loader 前后必须使用连续 fixed-point snapshot 证明零新增
file-backed dependency；任何 delta 在 marker 前失败。

## 7. Bearer 输入

CLI 只接受：

```text
恰好 64 个 lowercase hex
随后恰好一个 LF
随后 EOF
```

大写、空白、额外行、缺 LF、非 32-byte decode 均拒绝。禁止通过 argv 或环境
传 token。第一次呈现后 commitment 不复用，即使 pre-marker 失败。

## 8. No-history 负控

至少覆盖：

- ticket absent，malformed UTF-8，nonmapping，duplicate key；
- YAML alias/anchor/merge/comment/noncanonical raw；
- top/nested unknown/missing fields，bool-as-int，NaN/Inf，nonstring key；
- external raw/self canonical SHA 错误；
- retired raw/id/token、已使用 commitment；
- bearer 长度、大小写、空白、额外 stdin bytes；
- issuance raw/canonical/projection drift；
- ticket projection 与 request 任一字段不同；
- prefilled ACCEPT、禁止拒绝或 independence metadata 不成立；
- request/response/metadata path 越界、symlink、hash 漂移、duplicate keys；
- wrapper/definition/transport prereg/schema prereg/G0–G4 任一 drift；
- dependency path/raw/canonical/B/U/R drift；
- source/runtime/bootstrap/registry/definition chain drift；
- result/marker 错路或 overwrite/latest/old-result 权限变 true；
- production/118/Fig/malicious-writer/claim-state 权限变 true；
- old quarantine drift、old result/latest 出现；
- new result/marker 预存在；
- parser/import observer 产生 dependency delta；
- 所有失败发生于 marker/collector/mesh/march 之前；
- parser tests 不创建正式 request/clearance/ticket/token。

synthetic tests 只能使用完全隔离路径；formal collector、canonical mesh 和 time
march tripwire call count 必须为零。

## 9. G6 原子发行

只有 G5 ACCEPT 且 `blocking_findings=[]` 后允许：

```text
same-directory O_EXCL temp
→ canonical bytes
→ fsync(temp)
→ pre-link revalidate all bindings and target absence
→ no-replace link to reserved authorization path
→ fsync(directory)
→ no-follow reread/raw hash
→ externally retain raw SHA
→ broker releases token once to execution custodian
```

reserved path 预存在即失败。link 后异常标记
`AUTHORIZATION_NAMESPACE_POISONED`，不得删除、覆盖或复用 token，必须新开
successor。

## 10. Hard nonclaims

本工件不声称：

- parser 已实现或通过最终 G2/G4；
- production G3 已运行；
- issuance request、cross-family clearance、token 或 ticket 已存在；
- formal execution 已授权或运行；
- 31-history 物理结果已观察；
- V4.1 模型、claim state、三点、118 或 Fig17/18/19 已改变；
- bearer 可以可信内存 zeroize；
- 当前 wrapper 强制 Python `-I`；
- 恶意本地 writer、swap/restore 或瞬时 C-level load 得到防护。
