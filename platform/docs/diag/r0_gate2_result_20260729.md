# R0 第二次执行门结果：GO

时间：2026-07-29 10:42（Asia/Shanghai）  
预登记：`r0_gate2_incall_prereg_20260729.md`  
验证器：`platform/verify_cycle_reduction_r0.py`

完整机器可读证据：
`platform/docs/diag/r0_gate2_result_20260729_104623.json`
（SHA256
`4d1be9da3b4974b440f00175c2bc0feb546c936a31fe0f31abfd00b850935666`）。
该 JSON 保存三次调用的全部 guards、claim manifests、逐节点 contributions
以及最终 promoted/frozen 源码哈希。

## 三调用结果

| 调用 | L / T (N) | 相对冻结 anchor 最大差 | 同调用旧输出逐 bit |
|---|---:|---:|---|
| anchor 1 | 6.7547385337454235 / 0.8200054465778610 | 0.0026550320 N | PASS |
| anchor 2 | 6.7582556239740420 / 0.8209533573827736 | 0.0061721222 N | PASS |
| probe | 5.7494141888987595 / 0.7533914662914185 | 不适用 | PASS |

冻结 anchor 为
`L=6.752083501727169 N, T=0.8217739436030945 N`，正式复现容差
`0.15 N`。两个 anchor 均通过。

## R0 与物理账本

- probe R0：
  `[-0.01653279504887406, 0, 0.18909270180913929] N`
- 与第一次预登记旧 N1 remainder 的最大差：
  `1.0824674490095276e-15 N`
- probe `unclassified_physical_force`：`0 N`
- 三次 `force_ledger`、`unclassified_force`、
  `unclassified_physical_force`、`cycle_reduction`、
  `aero_output_invariance`：全部 PASS
- 每次在 ClaimGraph 前后检查的十个既有字段：
  `L, Fx, T, P, Fx_body, Fz_body, L_body, T_body_f, L_wind, T_wind`
  全部逐 bit 不变
- N1 implementation hash：
  `sha256:f4c5d11c28ba5f4d71132c9c601544ab6b9b2728e404df27bac781fd8304dc2c`
  （未变化）
- N4 implementation hash：
  `sha256:1b35c6edc52bf23b04c8e8fc3b9bb5cbac39baa958c4e071b9d5abca61a4cc4f`
  （未变化）
- R0 implementation hash：
  `sha256:a0b0f13578b08db42db094919b0e38ce9420480fae0e0c4bcf39d6f956d52f1b`
- R0 为 v41 拓扑末节点，且从唯一 canonical
  `numerical_cycle_reduction` 同时发布状态和记入 ForceLedger。

## 裁决与权限边界

Gate 2 **GO**。R0 的命题“robust total 与 arithmetic physical ledger
之差是图级数值归约，不是 N1 未分类气动力”在本次可动空间内验证。

该 GO 只授权：

1. 将 R0 晋升为 validated/frozen；
2. 将 runner 绑定到本次仅账本身份变化后的 `_v2_robo.py` 哈希；
3. 从冻结 118 seed 重新开始全新的 66 点运行。

它不构成任何气动精度提升，不授权修改 N1–N6 物理公式，也不修复既存
`v4_legacy` 兼容账本问题。

第一次实现/协议已作为 `R0.1 falsified/frozen` 留档；本次通过的唯一
实现/协议为 `R0.2 validated/frozen`，不会把 Gate 1 失败改写为成功。
