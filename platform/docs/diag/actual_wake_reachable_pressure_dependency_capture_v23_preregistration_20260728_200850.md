# S3ai-v2.2 / transport-v2.3 G3 dependency-capture 协议预注册

**时间**：2026-07-28T20:08:50+08:00  
**artifact**：`actual_wake_reachable_pressure_dependency_capture_v23_preregistration_20260728_200850`  
**attempt id**：`S3ai-v2.2-transport-v2.3-successor-20260728_185556`  
**scientific protocol**：`S3ai-v2.2`（冻结不变）  
**transport protocol**：`S3ai-v2.3`  
**状态**：`PREREGISTERED_PROTOCOL_ONLY / CAPTURE NOT EXECUTED / RESERVED MANIFEST WRITE LOCKED`  
**claim-state change**：`false`

## 0. 裁决边界

本协议只回答一个运输/证据问题：怎样在不运行 31 条正式 histories 的条件下，
形成并独立审计真实、有限、逐文件的 dependency closure `B/U/R`。

它不观察气动物理结论，不修改模型、claim state、冻结 collector/aggregator，
不生成 formal authorization、token、marker、result，也不授权 118 sweep 或
Fig17/18/19。

当前探针的：

```text
wrapper import: 209 = 170 Python source + 39 native
leggauss 后:    218 = 179 Python source + 39 native
observed delta: 9 个 numpy.polynomial Python source
```

仅为非规范性诊断，不是正式 `B/U/R`。任何实现或审计不得硬编码
`170/179/209/218/9`。

## 1. 为什么本工件是 protocol-only

正式 `S0 == B` checkpoint 位于 dependency manifest、authorization、contract
和 stable-input 处理之后，存在：

```text
manifest → authorization → formal B → manifest
```

同时，当前 wrapper 的 `_load_and_verify_authorization()` 仍故意 fail-closed；
后续实现冻结的 G5/G6 parser 会改变 wrapper SHA。若现在把当前 wrapper-import
snapshot 命名为生产 `B`，既没有到达真实 pre-marker 边界，也会绑定一个随后
必然变化的 wrapper。

因此本文件只预注册捕获方法。生产 Discovery/Replay 和 reserved manifest 写入
必须由 successor preregistration 绑定最终 authorization parser、最终 wrapper
SHA、最终测试 SHA、冻结 bootstrap、trigger matrix 和 synthetic fixture。

当前参考身份是：

| 资产 | SHA256 |
|---|---|
| v2.3 wrapper | `c88846a6f1503301ead37c0e03190b756534cbe12f3c2a4a3ef2b619f16eb30a` |
| v2.3 tests | `5e5a258d238d33fd706fe50b07ec5dff4b1a81d70759d845f00c7c51b632831d` |
| transport prereg JSON | `9d8db7f66a03d7450fb2d5f33e66dae553f07fa2b5e40906b9792cac034beac7` |
| wrapper definition | `641e3111a811b97f8da9da8b6cfdecde2db0054fe931e4af053b479ee1253e5c` |
| frozen guard | `d2f05dd9a4951c082ed3949f59d95dec10f9a052885394d24a6621ec1b295b73` |

任何 wrapper/test/bootstrap/fixture/matrix 变化均要求新的 timestamped successor；
不得把本文件静默改造成可执行授权。

## 2. 生产 G3 的前置门

生产 G3 接受前必须同时满足：

1. G0 勘误后的 old-quarantine fresh audit 明确为 `ACCEPTED`；
2. G5/G6 authorization/clearance schema 已预注册，parser 已实现但 ticket/token
   尚未签发；
3. 对最终 wrapper SHA 的 G2 science-zero-drift audit 为 `ACCEPTED`；
4. 对最终 wrapper/tests 的 G4 mandatory no-history controls 全过；
5. 固定环境为：

   ```text
   /home/exuber/anaconda3/envs/fluxvortex/bin/python
   Python 3.12.13
   NumPy 2.4.6
   PyYAML 6.0.3
   cwd=/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV
   ```

6. 63-file local execution-source closure 以完整 path→SHA map 绑定；
7. reserved manifest、new authorization、token、result、marker 均不存在；
8. bootstrap bytes、trigger matrix、synthetic fixture 和 capture supervisor 均已
   冻结并逐 hash 绑定。

G0 或最终 G2/G4 未通过时，只允许 diagnostic capture；其输出不能作为六个生产
replicate 之一。

## 3. 冻结 namespace

successor 必须物化下列确切、不可覆盖路径：

```text
stdin bootstrap:
  actual_wake_reachable_pressure_dependency_capture_v23_bootstrap_<STAMP>.txt
trigger matrix:
  actual_wake_reachable_pressure_dependency_capture_v23_trigger_matrix_<STAMP>.json
pure-synthetic fixture:
  actual_wake_reachable_pressure_dependency_capture_v23_fixture_<STAMP>.json
discovery:
  ..._discovery_r01_<STAMP>.json
  ..._discovery_r02_<STAMP>.json
  ..._discovery_r03_<STAMP>.json
candidate, never reserved:
  ..._candidate_<STAMP>.json
locked replay:
  ..._replay_r01_<STAMP>.json
  ..._replay_r02_<STAMP>.json
  ..._replay_r03_<STAMP>.json
consensus:
  ..._consensus_<STAMP>.json
independent audit:
  ..._audit_<STAMP>.{md,json}
failure:
  ..._failure_<STAMP>.json
single reserved target:
  platform/docs/diag/actual_wake_reachable_pressure_dependency_closure_v23_20260728_185556.json
```

所有 capture/candidate/audit 工件必须 no-replace 写入，不在 `/tmp` 保存持久
证据。失败 series 的 raw evidence 保留，不能在同一 namespace 静默重跑。

## 4. Source-only bootstrap

successor 必须冻结 bootstrap UTF-8 bytes、SHA256、cwd、argv 和环境映射。
最低 argv 为：

```text
<absolute fluxvortex python>
-B
-X
pycache_prefix=/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/.one_shot_no_bytecode_cache
-
```

bootstrap 必须：

1. fresh process 启动且固定 pycache path 不存在；
2. wrapper 是唯一预加载的 repository-local source；
3. 对 wrapper 做 `O_NOFOLLOW open → fstat → read → fstat`；
4. 校验同一 bytes SHA，从同一 bytes `compile/exec`；
5. 注入 `_BOOTSTRAP_WRAPPER_SHA256`；
6. 调用 `_require_source_execution_mode()`；
7. 退出前再次确认固定 pycache path 不存在。

当前实现没有检查 `sys.flags.isolated`，本协议只声明
`fresh_process_clean_source_only=true`，不声明 `python_dash_I_isolation=true`。
若未来加入 `-I` claim，wrapper 必须运行时强制检查并重新通过 G2/G4。

## 5. 六个全新进程

生产 G3 必须包含：

```text
Discovery:     D1, D2, D3
Locked Replay: V1, V2, V3
```

不得在一个已污染进程中模拟多个 replicate。

每个 Discovery 必须按固定顺序：

1. bootstrap/source verification；
2. transport prereg、wrapper definition、63 sources、old quarantine 的只读校验；
3. 执行 pre-marker-equivalent helper chain；
4. 对 baseline 连续做两次 fixed-point dependency/native snapshot；
5. 依固定 trigger matrix 逐项执行，每项前后 checkpoint；
6. 执行 post-collector/pre-publication 的纯合成 helper chain；
7. 构建并序列化 capture payload；
8. 序列化后再做一次 fixed-point snapshot；
9. child 仅向 stdout/control pipe 输出 canonical payload，不写证据文件。

D1–D3 完成后，candidate 只能由预注册的机械规则生成，禁止人工增删。
V1–V3 使用同一 candidate bytes 完整重放 parser、全成员 fingerprint、
pre-marker-equivalent chain、trigger matrix、ledger/provenance、真实
aggregate/validators/augmentation/serialization 和 final fixed point。

## 6. 破除 manifest—authorization—B 循环

capture-only pre-marker path 必须执行所有不需要真实 token 的正式 helper，包括
candidate manifest parser、全成员 fingerprint、stable runtime/input、frozen
contract 和 registry。

被省略的真实 authorization loader 必须同时满足：

- 冻结 AST/call-graph 证明无函数内 import、无 `importlib/__import__`、无新
  `ctypes.CDLL/dlopen`；
- 只调用 bootstrap 已加载的 stdlib/NumPy/PyYAML helper；
- no-history synthetic authorization success/failure tests 的 loader 前后
  file-backed dependency snapshot 完全相同；
- 三个 fresh process 均重复得到零 dependency delta。

任一条件失败即 `G3 NO-GO`。此时 wrapper-import 的 `209` 不能被命名为 B。

## 7. 冻结 trigger matrix

每个 trigger 必须声明：

```text
trigger_id, phase, formal_call_sites, source_sha256,
synthetic_input_sha256, coverage_basis, checkpoint_before/after,
possible_first_loaded_files, per_file_justification_template
```

并固定：

```text
expected_no_history=true
formal_collector_allowed=false
canonical_mesh_builder_allowed=false
time_march_allowed=false
```

至少覆盖：

1. `P0_PREMARKER_EQUIVALENT`：manifest/parser/fingerprint、contract、registry、
   stable runtime/input 和 authorization-loader 零增量证明；
2. `F1_NUMPY_POLYNOMIAL`：formal rooted call graph 可达的全部
   `np.polynomial.legendre.leggauss` 调用点及冻结 quadrature orders；
3. `F2_NUMPY_LINALG_NATIVE`：formal 路径的 `solve/cond/matrix_rank/svd/lstsq`
   等独立 API family，以微型有限非奇异数组触发；
4. `F3_OTHER_EXTERNAL_CALLS`：rooted AST/call graph 的其余第三方调用，每个
   调用点必须有 synthetic trigger 或冻结的 import-free 证明；
5. `A1_SYNTHETIC_31_GO` 与 `A2_SYNTHETIC_31_PROTOCOL_NO_GO`：纯合成 31
   observations 经真实 aggregator，且不导入 pytest/unittest.mock/tests；
6. `V1_FROZEN_RESULT_VALIDATOR`；
7. `V2_AUGMENT_AND_CLOSURE_PROVENANCE`；
8. `V3_PRETTY_SERIALIZE_AND_SERIALIZED_VALIDATOR`；
9. `P1_PREPUBLICATION_HELPERS`：只读 helper，不创建 marker/result，不调用
   atomic publication。

本次 rooted call-graph 只读审计发现：62 个实现源的第三方根只有 NumPy 和
PyYAML；formal collector 唯一新增 lazy 家族是 `numpy.polynomial`。这只是
trigger coverage 的先验，不能替代动态六复本证据。

所有 replicate 必须通过本地 tripwire 证明：

```text
formal_collector_calls=0
canonical_mesh_builder_calls=0
time_march_calls=0
formal_histories=0
```

无法无 history 触发且无法静态证明不加载依赖的 formal 分支使 G3 失败。

## 8. B/U/R/O 与一致性

定义：

- `L[s,r,k]`：checkpoint `k` 当前 file-backed loaded set；
- `E[s,r,k]`：截至 `k` 的 append-only ever-seen set；
- `B[s,r]`：pre-marker-equivalent fixed-point baseline；
- `Δ[s,r] = E[s,r,final] − B[s,r]`。

通过必须满足六个 replicate 的：

```text
B exact equal
Δ exact equal
每个 path 的 sha256/dev/inode/size exact equal
first-seen phase exact equal
registered-member removal = 0
```

然后机械定义：

```text
B = 唯一 consensus baseline
R = 唯一 deterministic mandatory delta
O = 预声明条件分支的逐文件 optional set
U = B ∪ R ∪ O
```

若没有通过独立审计的 optional，则 `O=∅`、`U=B∪R`、`E_final=U`。

禁止多数表决、intersection 掩盖分歧、自动 union、把分歧成员降为 optional、
按目录/glob/版本扩展 U，或依据旧 `+9` 直接推断 R。若同一 optional 在不同
合法路径可能首次出现于不同 phase，当前单值 `allowed_phase` schema 无法表达，
必须失败并修订 schema。

## 9. 逐文件 enrichment

U 中每个文件必须且只能包含 wrapper schema 的 11 个字段：

```text
canonical_path
module_and_distribution_identity
kind
origin_or_package
sha256
st_dev
st_ino
st_size
allowed_phase
required_or_optional
justification
```

约束：

- canonical path 绝对、词法规范、resolve 相同、非 symlink、普通文件；
- Python identity 使用确切 `{module, distribution, version}`；多 alias 依冻结
  排序规范完整编码；
- native identity 使用确切 ELF SONAME/loader identity；
- owner 由 dist-info `RECORD`、`conda-meta`、CPython build、OS package
  metadata 或 ELF SONAME/build-id 唯一证明；未知或多义即失败；
- fingerprint 来自同一 no-follow fd 的 read/fstat；
- B phase 必须为 `baseline_pre_marker`；R/O phase 为六复本唯一首现阶段；
- required 恰好等于 `B∪R`；
- justification 逐文件引用 replicate、trigger/call-site、import event 和
  native maps certificate。

测试占位身份 `observed-runtime-file`、`definition-control`、generic path 和
generic justification 禁止进入正式 manifest。

enrichment 必须由 target 退出后的独立进程完成，不能为了归属解析而污染 target
的 B/U/R。

## 10. Native maps 与观测无扰动

baseline、可能新增 native 的 trigger 后及 final checkpoint 均须记录：

```text
replicate_id, child_pid, proc_starttime, running executable dev/inode,
checkpoint_id, raw /proc/<pid>/maps SHA256,
normalized (path, mapped_device, mapped_inode, stat identity, sha256,
            size, mapping_count),
normalized-members SHA256, deleted_mapping_count
```

supervisor 必须在 child 存活时读取 maps，并以 PID starttime/executable inode
排除 PID reuse。child 与 supervisor 的 normalized native certificate 必须
一致。`(deleted)` mapping、device/inode 不一致、无 maps 证据均失败。

ASLR 使 raw maps hash 可跨进程不同；只要求 normalized
`path/device/inode` set 六复本完全相同。本 profile 不声称捕获两个 checkpoint
之间瞬时 `dlopen/unload`。

每个 checkpoint 连续测量两次并要求：

```text
M(S) == M(M(S))
```

Discovery 与带 import observer 的 Locked Replay 最终集合也必须相同；observer
引入任何 file-backed member 即失败。out-of-process enrichment 的 imports 不
计入 target closure。

## 11. Quarantine 与失败语义

每个 D/V replicate 在 wrapper exec 前、baseline 后、matrix 后和 final
serialization 后验证七项旧资产：

| 资产 | SHA256 |
|---|---|
| old authorization | `39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e` |
| old marker | `42f9cb852128b24d2e28330ebf2ad9911764c845fba6370b4220aadbd5d6d778` |
| old failure log | `61ae80785419e6502c0e4544ba8d9757785c36fbb6e24bb7cac8bba988c8c60d` |
| original forensics MD | `b0e05a7114722120da571ad5396ce14370a471512d9bf6daae4a4319f07a632e` |
| original forensics JSON | `5ed17ab7d0bb3f793d188af41f7e7acd9f2cc7855f46bd57c14bce5df2219d9c` |
| correction MD | `657ec4c12a2266e277a8f8e0cc3923f6e8d124942435903ecc7d8780255ef414` |
| correction JSON | `a4990e4159e8b93c47707329830089e11078f4a2995c32c123dd904c9acc32b5` |

old canonical result/latest、新 auth/token/result/marker 必须始终不存在；reserved
manifest 在最终原子提交前必须不存在。

任一失败必须：

- 保留 raw capture 和 timestamped failure record；
- reserved manifest 保持不存在；
- 不创建 marker/result，不消费或模拟 authorization；
- 科学结论保持 `UNOBSERVED/UNKNOWN`；
- 修订时另开 successor namespace。

## 12. 独立 audit 与 reserved manifest

独立 auditor 必须重新计算 prereg/bootstrap/matrix/fixture/wrapper/definition/
guard/63 sources 的 SHA，六复本 B/Δ/E/fingerprint/phase，candidate 机械生成，
owner、native certificate、measurement non-interference、quarantine、namespace
absence 和 parser/ledger/provenance 验证。

审计只允许：

```text
G3_DEPENDENCY_CLOSURE_CAPTURE_ACCEPTED
G3_DEPENDENCY_CLOSURE_CAPTURE_NO_GO
```

不得以 `PARTIAL PASS` 晋升。

只有 G0、最终 G2/G4、D1–D3、candidate、V1–V3、exact consensus、trigger
coverage、11-field enrichment、native certificate、non-interference 和独立
audit 全部接受，且 reserved/auth/token/result/marker 仍不存在，才允许写
reserved manifest。

原子协议：

1. 同目录 `O_CREAT|O_EXCL|O_NOFOLLOW` 创建 owned temp；
2. 写唯一 canonical pretty JSON bytes + newline，`fsync(temp_fd)`；
3. pre-link gate 重验全部关键 SHA、quarantine 和目标不存在；
4. no-replace hard link 创建 reserved path，`fsync(directory_fd)`；
5. no-follow 重读并验证 raw/canonical SHA；
6. 只清理未链接且本过程拥有的 temp。

reserved path 预存在即失败且不采用。若 link 成功后发生异常，该 v2.3 namespace
标记 `G3_RESERVED_NAMESPACE_POISONED`；已链接文件不删除、不覆盖，必须另开
successor。reserved manifest 的存在仍不生成或授权 ticket/token。

## 13. Hard nonclaims

本工件不声称：

- 已运行任何 history 或获得 reachable-pressure 物理结论；
- 已完成生产 G3、G5/G6、模型改写、三点门、118 或 Fig17/18/19；
- `209/218/+9` 是最终闭包；
- closure 可跨环境、机器、workspace 或 inode 复用；
- 抵抗恶意本地 writer、swap/restore、未授权 alias；
- 捕获 failed/partially initialized import 或瞬时 C-level load/unload；
- 未覆盖 formal 分支不会加载其他依赖；
- 当前 wrapper 已强制 Python `-I`；
- 本 protocol-only 预注册可写 reserved manifest。

后续合法顺序为：

```text
G0 accepted
→ G5/G6 schema preregistered
→ authorization parser implemented, no ticket issued
→ final-wrapper G2/G4 accepted
→ executable G3 successor preregistration
→ D1/D2/D3
→ mechanical candidate
→ V1/V2/V3
→ independent audit
→ atomic reserved manifest
→ G3 accepted
```
