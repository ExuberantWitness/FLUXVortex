# V4.1 主推力 witness：复现门 NO-GO

**日期**：2026-07-29  
**裁决**：NO-GO；停止剩余 5 个节点 witness，不放宽容差，不复用失败进程

## 1. 预登记门

`v41_confirmed42_primary_thrust_witness_prereg_20260729.md` 要求六点的
`L_wind/T_wind` 均相对冻结 baseline 不超过 `0.15 N`。

## 2. 实际执行

资产：

- runner：`platform/run_claim_witnesses.py`
  (`sha256:26bddd29cd8c05f3eac4998f471d5f34ea54c786b48c47edd41d78410d8e8074`)
- 失败 artifact：
  `v41_confirmed42_primary_thrust_witness_20260729_130215.json`
  (`sha256:3ec88b530230286c6c97f795a513a3b806ea559336b0817ed3dc48de6b999525`)

顺序：

1. 冷 anchor（丢弃）；
2. warm formal anchor；
3. W1 `6_2.6_22.5_5`；
4. W2 `10_2.6_22.5_5`，失败并硬停。

结果：

| 条件 | ΔL | ΔT | 裁决 |
|---|---:|---:|---|
| warm anchor `8_2.6_0_5` | 0.006172 N | 0.000821 N | PASS |
| W1 `6_2.6_22.5_5` | 0 | 0 | PASS |
| W2 `10_2.6_22.5_5` | **0.187948 N** | 0.096359 N | **FAIL** |

所有已保存调用的 graph identity 与 force ledger 均通过；失败发生在冻结数值身份，
不是力账本。

## 3. 新发现

W1/W2 都来自旧 `s6_sweep_v41.json` 的 118 点种子，不属于 2026-07-29 新算的
66 点。W1 精确复现，但 W2 未过门。W2 冻结值为：

```text
L=10.494815563247444 N
T=0.3271429637522524 N
```

相邻两个本轮 fresh 点为：

```text
tw20: L=10.460428988056291, T=0.42582990777887575
tw25: L=10.861347697998415, T=0.3848291362742552
```

失败值的绝对差与相邻 fresh 点共同提示旧中心点可能不是当前执行身份下的平滑
曲线值；但失败 runner 在抛错前未保存观测值符号，因此此处不做缓存陈旧或数值
混沌裁决。

## 4. 影响

- `184/184` 仍证明 solver grid 文件完整；
- confirmed/conditional scope 隔离仍有效；
- 但“85 个旧 confirmed 种子 + 66 个 fresh 点”不能再称为完全 fresh、
  同一执行身份的 authoritative baseline；
- 在 W2 身份闭合前，confirmed42 残差只能保留为 provisional descriptive
  fingerprint，不得用于唯一 claim 归因或候选晋升。

## 5. 下一门

先预登记并执行单点序列复现审计，保存每次 W2 原始结果而不是超阈即丢弃：

- cold/warm anchor；
- W2 连续重复；
- fresh 邻点 tw20/tw25；
- 再次 W2，判断缓存漂移、运行间分叉或调用序列依赖。

审计前禁止继续剩余节点 witness，也禁止将 `0.15 N` 改大以让本次通过。
