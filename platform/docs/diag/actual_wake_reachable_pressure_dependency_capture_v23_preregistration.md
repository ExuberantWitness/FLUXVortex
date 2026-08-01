# S3ai-v2.2 / transport-v2.3 G3 dependency-capture 协议预注册 successor

**时间**：2026-07-28T20:29:30+08:00  
**artifact**：`actual_wake_reachable_pressure_dependency_capture_v23_preregistration_20260728_202930`  
**attempt id**：`S3ai-v2.2-transport-v2.3-successor-20260728_185556`  
**状态**：`PREREGISTERED PROTOCOL SUCCESSOR / EXECUTABLE INSTANTIATION LOCKED`  
**normative authority**：本 Markdown  
**claim-state change**：`false`

本版保留并 supersede：

- 20:08:50 protocol-only 草案（SHA256
  `6b5ab72b9f6b0002e57e730a67585f989162943a240998c108b1c61bf4faf97f`）；
- 20:20:04 独立审计 `FAIL / 8 blockers`（MD SHA256
  `5634fca8ba1072acf047c426406175ab169a2c1157ffc48465fbff8adff8ed2f`，
  JSON SHA256
  `9250ae8874a68987d0f86f1164e835fb4be0f2659c6eb6568dd7a236cbfd6dbd`）。
- 20:20:49 第一版 successor（MD SHA256
  `2b673462d7946f762ef04b43a868c3d1bc0df94dce1fc718e8e656e38bd172a1`）；
- 20:28:33 successor 独立复审 `FAIL / 2 remaining blockers`（MD SHA256
  `0451a51ba07ae59e7c97a967bf984581e514f7c6fdeec17e6c0488d0aca660e2`，
  JSON SHA256
  `4d0532e7f7a6aa78affb09a96623e2f49f9e83968aeb2ac086268324f90ab9f4`）。

旧草案及其 FAIL 审计不覆盖、不删除。

## 0. Authority 和边界

本 Markdown 是本 protocol successor 的唯一规范性 authority。配套 JSON 只是
机器可读索引，必须包含本文件 raw SHA；发生差异时一律以本文件为准并使 gate
失败，不能从二者中选择较宽解释。

本版仍不执行 capture，不形成 B/U/R，不写 reserved manifest，不生成
authorization/token/marker/result，不运行 history，不修改气动模型或 claim。
它只修正可执行方法；最终 production G3 必须由新的 executable instantiation
逐 hash 绑定已接受的 G0、最终 wrapper G2/G4、bootstrap、seed fixture、完整
load-site matrix、capture harness、supervisor 和 owner auditor。

当前 `209→218`、`+9 numpy.polynomial` 仅为非规范性诊断，禁止硬编码为
acceptance。

## 1. 固定集合语义

本 campaign 不允许 optional：

```text
O = ∅
U = B ∪ R
E_final = U
required members = B ∪ R
```

定义：

- `B`：seed-bound pre-marker-equivalent path 达到 fixed point 后的唯一
  file-backed baseline；
- `E`：append-only ever-seen ledger；
- `R = E_final − B`：完整 frozen trigger path 的唯一 mandatory delta；
- `U = B ∪ R`。

多数表决、intersection 掩盖分歧、自动 union、将分歧成员降为 optional、按
package/prefix/glob/version 扩权全部禁止。未来若需要 optional，必须另开协议，
逐条件分支至少三复本并引入多 phase schema。

## 2. 破除 Discovery—candidate 循环

一个 evidence replicate 是 clean twin 与 instrumented twin 两个独立 fresh
process。生产 series 包含：

```text
Discovery pairs: D1-C/D1-I, D2-C/D2-I, D3-C/D3-I
Replay pairs:    V1-C/V1-I, V2-C/V2-I, V3-C/V3-I
total child processes = 12
```

### 2.1 Frozen seed-bound Discovery

在 candidate 不存在时，D1–D3 使用同一冻结 seed fixture。seed 只用于触发
parser/control flow，不提供 production membership authority。每个 D pair：

1. 执行 exact source-only bootstrap；
2. 验证 old quarantine 与全部 reserved absence；
3. 使用 seed manifest 调用 manifest schema parser 和一个 seed member 的
   no-follow fingerprint；
4. 使用 seed authorization/request/clearance 调用最终 parser 的 synthetic
   success path；
5. 调用所有不依赖 full candidate 的正式 pre-marker helper，包括 source、
   stable runtime/input、contract、registry 和 result-path guards；
6. 达到 pre-marker fixed point，形成 `B_D`；
7. 执行 frozen full trigger matrix；
8. 形成 append-only `E_D` 和 `R_D=E_D−B_D`；
9. 只经 stdout/control pipe 输出 canonical raw capture。

Discovery 不声称执行 full candidate fingerprint。full-member loop、真实 ticket
bytes 和 candidate-bound stable-input 中被推迟的部分必须：

- 通过 rooted source/AST/call-graph 证明无 import/dlopen/lazy attribute；
- 在三个 seed synthetic success processes 中 dependency delta 为零；
- 在 V1–V3 的完整 candidate path 中再次证明零 delta。

### 2.2 Mechanical candidate

只有六个 D twins 均通过且：

```text
B_D1-C = B_D1-I = ... = B_D3-I
R_D1-C = R_D1-I = ... = R_D3-I
fingerprint/first-seen phase exact equal
clean raw event chains exact equal within D1-C/D2-C/D3-C
instrumented raw event chains exact equal within D1-I/D2-I/D3-I
clean↔instrumented filtered checkpoint-event projection exact equal
```

才由冻结程序机械构造 candidate；人工增删、排序选择或 identity 修补禁止。
candidate 永不占 reserved path。

### 2.3 Candidate-bound Replay

V1–V3 每个 twin 使用完全相同的 candidate raw bytes，执行：

- full manifest parser 和所有 U member fingerprint；
- final authorization parser 的 candidate-bound synthetic success；
- full pre-marker stable runtime/input、contract、registry；
-真实 `_DependencyLedger` 和 provenance；
-同一 trigger matrix；
-纯合成 real aggregate、两个 validators、augmentation、pretty serialization；
-final fixed point。

通过必须：

```text
all 12 B exact equal
all 12 R exact equal
all 12 U/fingerprint/first-seen-phase exact equal
D clean raw event chains exact equal within clean group
D instrumented raw event chains exact equal within instrumented group
V clean raw event chains exact equal within clean group
V instrumented raw event chains exact equal within instrumented group
all 12 filtered checkpoint-event projections exact equal
removed registered member count = 0
E_final = U
```

任一 Replay 新增 candidate 外 member 或 `B_V != B_D`，整个 series NO-GO；
不得在同一 series 改 candidate 后重放。

## 3. Exact bootstrap 和 pre-wrapper budget

executable instantiation 必须冻结 bootstrap UTF-8 bytes/SHA、cwd、argv、完整环境
映射，以及 wrapper exec 前：

```text
sorted sys.modules names
file-backed module path/fingerprint set
normalized native path/device/inode set
their canonical digests
```

bootstrap 只能使用解释器启动已有模块，或最终 wrapper 无条件顶层必导入模块。
明确禁止在 wrapper exec 前导入：

```text
importlib
importlib.util
importlib.metadata
pkgutil
packaging
csv
email
conda
subprocess
readelf/owner-enrichment helpers
pytest/unittest.mock/test modules
```

若 bootstrap 自身确需某模块，必须在 executable instantiation 中逐项声明，并
证明它也由 final wrapper 无条件顶层加载；否则 NO-GO。

最低 argv：

```text
/home/exuber/anaconda3/envs/fluxvortex/bin/python
-B
-X
pycache_prefix=/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/.one_shot_no_bytecode_cache
-
```

bootstrap 必须 `O_NOFOLLOW open → fstat → read → fstat`，从同一 wrapper bytes
hash/compile/exec，注入 `_BOOTSTRAP_WRAPPER_SHA256`，调用
`_require_source_execution_mode()`，并保证固定 pycache path 始终不存在。

当前 wrapper line 343 的条件式 `from importlib.util import source_from_cache`
必须满足：

- pre-wrapper `importlib.util` 不存在；
- fixed pycache prefix 为空、local/external Python origins 均为 `.py`；
- `.pyc/.pyo` 分支静态不可达；
- 三个 Discovery clean twins 动态证明 branch call count=0。

本 protocol 不声明 Python `-I`；若最终实现要求 `-I`，须在 wrapper/runtime/
bootstrap/G2/G4 全部显式绑定。

## 4. 完整 load-site inventory

不能只搜索 NumPy/PyYAML。executable instantiation 必须对 exact 63-source rooted
closure 生成逐位点 matrix，覆盖：

```text
AST Import / ImportFrom
函数或条件分支内 import
__import__ / importlib / module-spec loading
module __getattr__ / lazy attribute
ctypes.CDLL / PyDLL / dlopen
NumPy/PyYAML extension and lazy namespace
stdlib lazy imports
native loader and linked-library side effects
```

每个 site 记录：

```text
site_id
repo-relative file and source SHA
line/function/rooted call chain
first legal phase
synthetic fixture SHA
dynamic trigger id or static-unreachable/import-free proof id
possible Python/native files
per-file justification template
```

所有 source site 必须恰好落入“动态触发”或“冻结证明”之一，不能遗漏或双重
归类。静态 inventory 还须结合 target ELF `DT_NEEDED`/loader maps；只读工具
在 out-of-process auditor 中运行，工具 path/SHA/version 绑定。

最低动态 families：

- seed-bound/full-candidate pre-marker；
-全部 formal-reachable `leggauss` quadrature orders；
- NumPy `solve/cond/matrix_rank/svd/lstsq/eigvalsh/cholesky` 等独立 family；
-其余 rooted stdlib/third-party/native lazy site；
-纯合成 31-observation GO 与 PROTOCOL-NO-GO；
-真实 frozen validator、augmentation、closure provenance；
-pretty serialization、serialized validator、prepublication read-only helpers。

formal collector、canonical mesh、time march 和 atomic publication 都装 tripwire，
12 个 child 的 call count 必须全为零。

## 5. Measurement non-interference

snapshot primitive 必须是冻结的 import-free source，且只使用 preloaded
`sys.modules/os/hashlib/path/stat` 能力。每个 checkpoint 记录：

```text
S_pre  = primitive entry 前的 module-name/native projection
S_1    = 第一次完整 measurement 后
S_2    = 第二次完整 measurement 后
```

要求 `S_pre == S_1 == S_2`；若 capture machinery 本身增加 member，则它必须在
baseline 前稳定加载并成为 B，不能被忽略。

每个 D/V pair 还要求：

```text
clean twin membership/fingerprint/native maps
==
instrumented twin membership/fingerprint/native maps
```

instrumented twin 的 import observer enter 前、enter 后、exit 前、exit 后均做
零增量证明。observer event ledger 可以不同于 clean twin 的“无 observer”字段，
但物理 membership、fingerprint、phase projection 必须相同。

raw event chain 只在同类 twin 内比较：

```text
D1-C/D2-C/D3-C exact raw chain
D1-I/D2-I/D3-I exact raw chain
V1-C/V2-C/V3-C exact raw chain
V1-I/V2-I/V3-I exact raw chain
```

C↔I 使用冻结的 `checkpoint_event_projection_v1`：

1. 从 raw events 删除 `event_type == observed_import_request`；
2. 其余 event 只允许 `checkpoint_confirmed_successful_load` 和
   `registered_member_removed`；
3. 按原相对顺序重新从 `sequence=0` 编号；
4. 删除原 `previous_event_sha256/event_sha256`，从 64 个 `0` 开始以 canonical
   JSON 重新计算 projected hash chain；
5. 对 snapshots 删除 observer-dependent `event_count`、
   `event_chain_sha256`、`snapshot_sha256`，保留 phase、loaded/ever-seen paths、
   members 及其 hashes，再计算 projected snapshot SHA；
6. projection object 精确为
   `{schema_version:"checkpoint-event-projection-v1", events:[...],
   snapshots:[...]}`，其 SHA 为 canonical JSON SHA256。

所有 12 child 的 projection object 与 SHA 必须 exact equal。任何
`registered_member_removed` 仍立即失败；filter 不能用于隐藏 membership、phase
或 fingerprint 差异。

enrichment 永远 out-of-process；实测在 target 导入 metadata/csv/email 会额外
增加 41 files，因此 target 内出现 enrichment import 直接失败。

## 6. 11-field identity 的 kind-discriminated canonical grammar

保持 transport-v2.3 的 11 个顶层 member fields，不扩字段；但
`module_and_distribution_identity` 的三个 string 和
`origin_or_package` 使用以下冻结语法，并由 parser 与独立 auditor共同验证。

### 6.1 `module_and_distribution_identity`

仍精确为：

```text
{module: string, distribution: string, version: string}
```

`module`：

- 若 raw capture 的 `module_names` 非空：
  `py:<lexically sorted unique comma-separated full aliases>`；
- 若 native `module_names` 为空：
  `elf-soname:<exact SONAME>`，此时 SONAME 不得为空；
- Python source 不允许空 module；
- native 可以没有 SONAME/build-id，只要有至少一个 exact Python module alias；
- native 若既无 module alias 又无 SONAME则失败。

alias 必须匹配 `[A-Za-z_][A-Za-z0-9_.]*`。多 alias 全量排序编码，不能任选一个。

`distribution`：

```text
wheel:<pct(PEP503-normalized-name)>
conda:<pct(name)>:<pct(build)>:<pct(channel)>:<pct(subdir)>
dpkg:<pct(package)>:<pct(arch)>
```

`pct` 的定义唯一：把原 string 编为 UTF-8 bytes；ASCII
`A-Z a-z 0-9 - . _ ~` 原样保留，其余每个 byte 编为 uppercase `%HH`。因此
component 内的 `: / %` 均被编码，parser 只按 literal `:` 分隔固定数量字段，
禁止 `split/rsplit` 启发式。`version` 字段为 `pct(exact owner version)`。

wheel name normalization 冻结为 PEP 503：

```text
re.sub(r"[-_.]+", "-", METADATA["Name"]).lower()
```

输入必须是 METADATA 中唯一、非空的 ASCII distribution name；不从目录名猜测。
conda/dpkg 每个 component 都来自绑定的 owner metadata exact field。

### 6.2 `origin_or_package`

是按 key lexical sort 的 `key=percent-encoded-value;...`。percent encoding 只
允许 RFC3986 unreserved 原样，其余以 uppercase `%HH` UTF-8 编码。最少包含：

```text
manager
owner_metadata_path
owner_metadata_sha256
materialized_record_path
materialized_record_digest
elf_class
elf_data
elf_osabi
elf_machine
elf_soname          (none allowed)
elf_build_id        (none allowed)
loader_roles        (sorted comma list, percent encoded)
owner_evidence_tool_path
owner_evidence_tool_sha256
```

Python source 的 ELF 项固定 `none`。SONAME/build-id 是补充证据，不能代替
distribution owner。

`justification` 使用同一 canonical key/value grammar，绑定 12 child capture
IDs、baseline/trigger/site IDs、event sequence 和 native certificate IDs。

当前 schema parser 若不验证上述 grammar，executable instantiation 不得晋升；
需要在 final wrapper 中增加 transport-only grammar validation 并重做 G2/G4。

## 7. Owner enrichment 的唯一归属算法

raw child 退出后才由独立 auditor 处理；每个 U path 先收集所有 owner candidates，
最后要求 candidate count 恰好 1。禁止“pip 优先/conda 优先”等短路。

### 7.1 Wheel `dist-info/RECORD`

- 使用 RFC4180 CSV parser；
- `.dist-info` 必须含唯一 METADATA `Name/Version` 和同目录 `RECORD`；
- distribution root 唯一定义为该 `.dist-info` 目录的 lexical parent；
- `.dist-info` 目录须位于冻结 environment prefix 内，且 METADATA 的 PEP503
  normalized Name 与 owner identity exact equal；不由 `.dist-info` 文件名反推；
- RECORD row 的 POSIX relative path从 distribution root lexical join 后
  `normpath`；
- `../../../bin/f2py` 等 `..` 先做 lexical `normpath`，允许离开 site-packages，
  但最终必须仍在冻结 environment prefix；
- candidate path 不 `resolve()`，避免 symlink target 假归属；
- RECORD hash/size 非空且与 file bytes匹配；
- 绑定 METADATA/RECORD raw SHA、row bytes、declared digest/size。

### 7.2 Conda

- 枚举 exact `conda-meta/*.json`，绑定每个 metadata SHA；
- materialized candidate 以 `files` 为准；
- `paths_data.paths` 用于 hash/类型证据，但 noarch
  `site-packages/...` 必须通过同 package 的 materialized `files` 映射到
  `lib/python3.12/site-packages/...`；
- 优先验证 `sha256_in_prefix`，否则 exact `sha256`；
- lexical installed path 不 `resolve()`；因此 symlink 名不会把 target 错归给
  symlink owner。

### 7.3 Dpkg/usrmerge

- runtime manifest 保留 `/usr/lib/...` canonical path；
- 只有冻结且已验证的 root-level `/lib → /usr/lib` usrmerge symlink 可产生
  `/lib/...` owner alias；
- alias `resolve()` 必须等于 runtime path，且 stat dev/inode exact equal；
- dpkg `.list` 必须 exact 命中 alias，`.md5sums` 必须匹配；
- 绑定 dpkg-query/tool SHA、status、`.list`、`.md5sums` SHA和 package version。

unknown、0 owner、>1 owner、metadata hash mismatch 或 prefix escape 均失败。

## 8. Native maps certificate

每个 fixed-point checkpoint 由 supervisor 在 child 存活时读取
`/proc/<pid>/maps`，以 `/proc/<pid>/stat` starttime 和 executable dev/inode
防 PID reuse。记录 raw maps SHA，但跨进程只比较 normalized：

```text
canonical path
mapped device/inode
stat device/inode
sha256/size
mapping count
module aliases
loader roles
SONAME nullable
build-id nullable
```

`(deleted)`、无 maps 证据、device/inode 不一致均失败。ASLR address 和 raw maps
SHA 不要求跨进程相同。本 profile 不声称捕获 checkpoint 间瞬时
`dlopen/unload`。

## 9. Quarantine、namespace 和失败语义

12 个 child 在 wrapper 前、baseline 后、matrix 后、serialization 后都验证旧
七项 SHA、old result/latest absence、新 auth/token/result/marker absence。
candidate/capture/audit 使用 timestamped no-replace paths；不在 `/tmp` 保存
持久证据。

任一失败：

- 保留 raw capture、candidate（若已形成）和 failure artifact；
- reserved manifest 保持不存在；
- 不消费 token，不创建 marker/result；
- 同一 series 不重试；
- physics/claim decision 保持 `UNKNOWN`。

## 10. Executable instantiation 和 reserved commit

只有未来 executable instantiation 同时绑定以下 exact path/SHA 才可开始 D1：

- 本 normative Markdown及其 JSON index；
- accepted G0 quarantine audit；
- accepted authorization-schema audit及最终 parser wrapper；
- final-wrapper G2/G4；
- 63-source map；
- bootstrap、seed fixture、load-site matrix、capture harness、supervisor、
  owner auditor；
- old lineage；
- reserved manifest/auth/result/marker absence。

六 pairs 完成后必须有新的独立 audit，裁决只能：

```text
G3_DEPENDENCY_CLOSURE_CAPTURE_ACCEPTED
G3_DEPENDENCY_CLOSURE_CAPTURE_NO_GO
```

只有 ACCEPT 才可按：

```text
same-directory O_EXCL/O_NOFOLLOW owned temp
→ canonical pretty JSON + LF
→ fsync(temp)
→ pre-link full revalidation
→ no-replace hard link
→ fsync(directory)
→ no-follow raw/canonical revalidation
```

一次提交 reserved manifest。link 后异常使 namespace
`G3_RESERVED_NAMESPACE_POISONED`；已链接文件不删除、不覆盖，另开 successor。

## 11. Hard nonclaims

本 protocol successor 不声称：

- 20:08 草案已通过审计；
- executable G3 已实例化或运行；
- B/U/R、owner manifest 或 reserved manifest 已建立；
- authorization/token/formal history/物理结果存在；
- V4.1、claim tree、三点、118、Fig17/18/19 已修改或验证；
- owner metadata 在正式 runner 内被重新解析；
- Python `-I`、恶意 writer、swap/restore、failed import 或瞬时 native load 得到
  超出 `RESEARCH_ACCIDENTAL_DRIFT` 的保证。
