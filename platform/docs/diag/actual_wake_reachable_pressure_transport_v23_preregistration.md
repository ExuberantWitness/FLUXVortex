# S3ai-v2.2 / transport-v2.3 successor one-shot 预注册

**预注册时间**：2026-07-28T18:55:56+08:00  
**attempt id**：`S3ai-v2.2-transport-v2.3-successor-20260728_185556`  
**scientific protocol**：`S3ai-v2.2`（冻结不变）  
**transport protocol**：`S3ai-v2.3`（仅执行运输层）  
**状态**：`PREREGISTERED / IMPLEMENTATION NOT AUTHORIZED / EXECUTION NOT AUTHORIZED`  
**claim-state change**：`false`

本版 supersede 仅限定义层草案：

- `..._20260728_184227`：3 个 MD/JSON 一致性 blocker；
- `..._20260728_185112`：1 个 assurance-profile ledger 分账 blocker。

两版均保留为 `FAIL` 历史，没有被覆盖。

## 0. 裁决

本 successor 采用 **有限、逐文件授权的 dependency allowlist closure**，不采用
“只预热 `leggauss` 后继续要求 start/end exact set”作为正式保证。

旧 wrapper 的组成部分错在把“已加载依赖集合不增长”当作运行身份稳定性。正式
路径允许冻结代码进行预先声明的 lazy import；正确约束是：

```text
实际加载成员始终属于有限授权闭包
+ 每个成员的 bytes/device/inode/size 与授权一致
+ 静态运行身份、科学源码和冻结输入 start=end
```

170→179 是旧 attempt 的确定性故障指纹，不预先冒充新 wrapper 的最终依赖清单。
新实现完成后必须在 clean、no-history bootstrap 下重新产生并审计 `B/U/R`。

## 1. 旧 attempt 永久隔离

下列旧资产保持原路径和原 bytes，不删除、不重命名、不覆盖：

| Asset | Identity |
|---|---|
| old authorization | SHA256 `39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e` |
| old permanent marker | SHA256 `42f9cb852128b24d2e28330ebf2ad9911764c845fba6370b4220aadbd5d6d778`；`retry_allowed=false` |
| old failure log | SHA256 `61ae80785419e6502c0e4544ba8d9757785c36fbb6e24bb7cac8bba988c8c60d` |
| original forensics MD | SHA256 `b0e05a7114722120da571ad5396ce14370a471512d9bf6daae4a4319f07a632e` |
| original forensics JSON | SHA256 `5ed17ab7d0bb3f793d188af41f7e7acd9f2cc7855f46bd57c14bce5df2219d9c` |
| correction MD | SHA256 `657ec4c12a2266e277a8f8e0cc3923f6e8d124942435903ecc7d8780255ef414` |
| correction JSON | SHA256 `a4990e4159e8b93c47707329830089e11078f4a2995c32c123dd904c9acc32b5` |

旧 canonical result 和旧 latest result 继续不存在。v2.3 是一次新的 successor
attempt，**不是 retry**；旧 authorization id、token、marker 都不能授权它。

## 2. 新 artifact namespace

```text
wrapper:
  platform/actual_wake_reachable_pressure_obstruction_v23_one_shot.py
dependency manifest:
  platform/docs/diag/actual_wake_reachable_pressure_dependency_closure_v23_20260728_185556.json
authorization:
  platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_v23_20260728_185556.yaml
result:
  platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json
marker:
  platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_s3ai_v22_transport_v23_20260728_185556.json.lock
interpretation contract:
  platform/docs/diag/actual_wake_reachable_pressure_result_interpretation_contract_v23_20260728_185556.{md,json}
```

新 result 不写旧 canonical path，也不写旧 latest pointer。

## 3. 依赖闭包的类型化定义

- `B`：clean bootstrap 在 marker 前必须精确加载的 baseline set；
- `U`：authorization 绑定的有限 dependency universe；
- `R`：正式路径必须实际加载的 mandatory lazy set；
- `E`：进程内 monotonic ever-seen import/load ledger。

`U` 的每一项必须逐文件记录：

```text
canonical_path, module/distribution identity, kind, origin/package, sha256,
st_dev, st_ino, st_size, allowed_phase,
required_or_optional, justification
```

禁止 glob、目录前缀、仅包版本或“整个 NumPy 包均允许”。未经逐项理由的
optional dependency 不进入 `U`。

## 4. Fail-closed invariants

1. marker 前实际集合必须 `S0 == B`；多、少或任一 fingerprint 不同均失败。
2. 全程必须 `B ⊆ E ⊆ U`；首次出现的 Python/native member 必须入 ledger 并
   校验。
3. 正式路径结束必须 `R ⊆ E`；若没有经审计的 optional，则要求 `E == U`。
4. 已经在 checkpoint 证实成功加载并写入 ledger 的 member，随后从
   `sys.modules` 移除不能抹去证据；end-loaded set 必须与该 ledger 自洽。
5. `U` 中所有文件（包括未加载 optional）在 marker 前和发布前复核
   bytes/device/inode/size。
6. loaded native member 同时与 `/proc/self/maps` 映射 inode 自洽；
   deleted/replaced mapping 失败。
7. 63-file local execution source closure、冻结 v2/v2.1/v2.2 定义、31-case
   registry/identity、collector 和 aggregator 必须 start=end exact。
8. `stable_runtime_identity` 单独包含解释器、NumPy/build、环境变量、CPU
   affinity、rounding、workspace 等，并要求 start=end exact。
9. dependency snapshot 从 `stable_runtime_identity` 和
   `stable_execution_input` equality 中剥离；start/end/delta/ever-seen 分别
   计算 digest，按闭包谓词判定。
10. post-marker 任何 transport failure 都永久消耗新 authorization，result
    absent，scientific decision 为 `UNKNOWN`。
11. transport 合法时，即使冻结 aggregator 输出 `PROTOCOL-NO-GO`，也必须原子
    发布；transport 不得擦除不利科学结果。
12. 发布前必须同时通过 frozen-result validator、dependency closure
    certificate、stable input equality、local-source equality、marker identity
    和 atomic no-replace gate。

### Assurance profile 与加载边界

本 successor 继承 v2.2 已冻结的 research trust boundary：防 accidental
deployment/runtime drift，不声称抵抗一个能在 import 窗口恶意
“替换—加载—恢复”的同 UID writer。选择：

```text
assurance_profile = RESEARCH_ACCIDENTAL_DRIFT
malicious_local_writer_protection = OUT_OF_SCOPE
```

该 profile 的授权硬门是：固定 `fluxvortex` Python 3.12 环境、isolated
source-only bootstrap、逐文件 `B/U/R` manifest、append-only import ledger、
pre/post 全 `U` fingerprint、native baseline `/proc/self/maps`、63 local
sources 与 stable runtime/input exact equality。它足以消除已证实的 170→179
lazy-import 错件，并发现非恶意的部署/运行漂移。

research profile 的 mandatory ledger 只承诺记录：观察到的 import request、
在 checkpoint 证实成功的 loaded member、start/end/delta 和已登记 member 的
后续移除。failed/partially-initialized、在两个 checkpoint 之间完成的瞬时
load/unload，以及未授权 alias 的强制捕获只作 best-effort observation，不是
当前 authorization claim。

若未来要声称抵抗恶意 swap/restore 或瞬时 C-level `dlopen`，必须另选
`ADVERSARIAL_LOCAL_WRITER` profile，并在 authorization 前取得以下之一：

1. audited verified loader：`O_NOFOLLOW` fd 读取、hash，并从同一持有 bytes
   编译/映射；或
2. repo、解释器、packages、native libraries 位于 content-addressed read-only
   execution image。

当前 profile 不作该强声明，也不让该扩展安全目标阻断气动科研。

## 5. 旧 wrapper 的第二层故障必须一起删除

只移除旧 `_runtime_identity()` 内的 exact-set 比较不够。旧
`run_authorized_once()` 还要求：

```text
runtime_end == runtime_start
end_snapshot == start_snapshot
```

而二者都嵌入完整 loaded-dependency set，仍会拒绝合法 growth。v2.3 必须把：

```text
stable_runtime_identity
dependency_closure_snapshot
stable_execution_input
```

三类账分开比较，不能仅把故障推迟到下一道 equality。

## 6. No-history negative controls

所有测试使用 fake collector，并断言正式 collector、mesh 和 time march
从未被调用：

1. clean baseline exact 通过；baseline 多/少一项在 marker 前失败；
2. authorized required lazy import 成功并记录于 delta/ledger；
3. 未授权 pure-Python import 失败，同包但不在 `U` 也失败；
4. 未授权 native library load 失败；
5. authorized path 改 bytes、same bytes/new inode、symlink/alternate origin、
   `.pyc` 替代 source 均失败；
6. import 后从 `sys.modules` 删除仍失败；
7. mandatory `R` 缺一项失败；
8. wildcard/prefix/version-only manifest 在 schema 解析阶段失败；
9. `U` 中未加载文件在运行中改变也失败；
10. member 在错误 execution phase 加载失败；
11. 旧 auth SHA、旧 token 或旧 auth id 传入 v2.3，在 marker 前失败；
12. old auth、marker、failure log、original forensics MD/JSON、correction
    MD/JSON 共 7 项逐 hash 在 fake run 前后保持；old result/latest 始终不存在；
13. new result/marker 预存在或并发 publication 时失败且不覆盖；
14. fake collector 后 closure violation：new marker 保留、result absent、第二次
    调用拒绝；
15. fake `PROTOCOL-NO-GO` 加合法 lazy delta：必须原子发布；
16. stable runtime、local source、`U` digest 或 second-audit binding drift
    均失败；
17. serialized result 缺失或篡改 closure manifest/start/end/delta/ledger 任一
    字段，read-only validator 拒绝；
18. 所有定义测试断言 frozen collector/mesh/march call count 为零。

以下 3 项只在 `ADVERSARIAL_LOCAL_WRITER` profile 下成为授权硬门；当前
research profile 记录为明确 nonclaim：

19. import-window swap-and-restore 必须失败；
20. failed/partially-initialized import 必须由 verified load boundary 捕获；
21. 同一 bytes 经未授权 alias/module-distribution identity 加载必须失败。

## 7. Gate 顺序

| Gate | 必要证据 | 当前 |
|---|---|---|
| G0 failure quarantine | fresh audit 接受 transport failure、result absent、physics unknown，绑定旧 hashes | `PARTIAL`：核心因果通过；勘误后需复核 |
| G1 preregistration | 冻结本文件、artifact IDs、closure 规则、负控和 result contract | `THIS ARTIFACT` |
| G2 implementation-diff audit | 新 wrapper 仅改变 transport/dependency/path；科学 chain 零 diff | `LOCKED` |
| G3 no-history closure audit | 在固定 fluxvortex Python 3.12 isolated bootstrap 下重新形成并审计逐文件 `B/U/R`、ledger、native baseline | `LOCKED` |
| G4 definition tests | 当前 research profile 的 18 类 mandatory controls 全过；3 类 adversarial controls 保持 conditional/nonclaim | `LOCKED` |
| G5 independent bounded audit | 逐 hash 绑定 wrapper、prereg、manifest、runtime、63 sources、registry、definition chain、new paths、old lineage、new token commitment与 assurance profile | `LOCKED` |
| G6 ticket issuance | 仅 G5 accepted 后生成新 auth/token，execution_limit=1 | `LOCKED` |
| G7 launch pre-marker | no-follow read/hash/compile；验证旧 quarantine、新 paths absence、全部 identity | `LOCKED` |
| G8 execute/publish | 31 histories→冻结 aggregate/validate→closure/stable checks→atomic no-replace | `LOCKED` |
| G9 claim gate | canonical result 存在且 fresh result audit 通过 | `LOCKED` |

新 authorization 的 decision 必须是：

```text
YES_NEW_ONE_SHOT_31_HISTORY_SUCCESSOR_AFTER_TRANSPORT_FAILURE
```

并继续明确：

```text
production_activation_allowed = false
force_hp_state_ves_118_fig_allowed = false
malicious_local_writer_or_swap_restore_protection = false
```

## 8. Claim-tree 裁决

- 物理 target claim 保持 `open`；
- 原 15-check result 保持 `UNOBSERVED`；
- old exact-set transport invariant 保持
  `FALSIFIED_AS_IMPLEMENTED_FOR_DECLARED_PATH`；
- v2.3 closure invariant 是新的 `open` evidence/transport claim；
- 本预注册不授权实现、formal run、模型改写、118 sweep 或 Fig17/18/19 验收。

只有 G0–G6 完成后，才允许消费新的 one-shot authorization。
