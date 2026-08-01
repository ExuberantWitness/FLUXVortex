# S3ai-v2.2 正式失败取证勘误：漂移证据边界

**勘误时间**：2026-07-28T18:37:55+08:00  
**对象**：`actual_wake_reachable_pressure_formal_failure_forensics_20260728_183314`  
**原 MD SHA256**：`b0e05a7114722120da571ad5396ce14370a471512d9bf6daae4a4319f07a632e`  
**原 JSON SHA256**：`5ed17ab7d0bb3f793d188af41f7e7acd9f2cc7855f46bd57c14bce5df2219d9c`  
**原件处理**：保留、不覆盖  
**claim-state change**：`false`

## 独立审计裁决

原取证件为 `FAIL_ONE_EVIDENCE_BOUNDARY_BLOCKER`。核心因果判断通过，但原文将
“事后 checkpoint 未观察到漂移”写成了“运行期间不存在真实漂移”，证据强度
过界。

## 收紧后的唯一合法表述

1. 事后检查中，授权绑定的 63 个 repository execution sources 和 170 个
   import-time external Python sources 均与授权相符，没有观察到 checkpoint
   drift。
2. 独立、无 token、无 history、无 marker 的复现证明：
   `leggauss(8)` 必然使 loaded pure-Python source set 从 170 增至 179，原 170
   项在该复现中无变化。
3. 正式 pressure path 必然调用该函数，因此 170→179 已是 exact-set invariant
   失败的充分且不可避免原因。
4. 失败进程没有保存 end-fingerprint diff；一个同时发生、随后恢复的瞬时漂移
   不可恢复，也不能排除。
5. 所以 failure 不是文件漂移的证据，也不能被表述为“已证明绝无文件漂移”。

## 不受勘误影响的裁决

- collector、aggregation、`_validate_frozen_result()` 已返回；
- 物理 `stage_decision` 仍为 `UNKNOWN`；
- 15 项 scientific checks 是否通过仍为 `UNKNOWN`；
- 数值结果不可从现存产物恢复；
- 原 marker `retry_allowed=false`，旧授权不得重用；
- 新运行必须使用新 wrapper、独立审计和新 one-shot authorization。

本勘误不运行任何 history，不修改 formal physics、YAML、31-history registry、
旧 marker、旧 authorization 或旧 timestamped 取证件。
