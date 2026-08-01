# V4.1 fresh confirmed151 评分与残差指纹预登记

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`、尚未读取 fresh
三图精度统计时  
**阶段**：研究流程 0 → ①；只授权可信基线验收和描述性病因定位

## 1. 固定输入

本轮只接受同一运行身份的三个文件：

- `platform/docs/s6_sweep_v41_confirmed151_fresh_20260729_135128.json`
- `platform/docs/diag/fig171819_v41_confirmed151_fresh_manifest_20260729_135128.json`
- `platform/docs/diag/fig171819_v41_confirmed151_contributions_20260729_135128.json`

运行完成前这三个文件的哈希必然变化，因此不得把 partial 哈希登记成最终输入哈希。
最终哈希只能从完成态 manifest 的 `result_sha256` 和
`contributions_sha256` 读取，并写入后处理执行回执。

冻结的测量与评分实现为：

| 资产 | SHA-256 |
|---|---|
| `platform/docs/data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` |
| `platform/fig171819_benchmark.py` | `ae18ef391d1e68f114911d8d0a7341d3ae43752197a9009d6fb639ea01007179` |
| `platform/fig171819_residual_fingerprint.py` | `127db39b6028f1be676a10f95dc932f35e29402fd590dc979d97e269f4bc14e8` |
| `platform/docs/diag/v41_fresh151_postprocess_authorization_20260729.json` | `22ca928d5240ed6195fefe9da1f48c121102712f1f39681d813f90646b8d3cab` |

评分前不得因 fresh 数值结果修改数据身份、曲线映射、插值方向、主证据范围或
误差定义。

## 2. 完成态硬门

任何评分前必须同时满足：

1. runner 正常退出，manifest `status=complete`；
2. result、contribution cases、case guards 的键集合均精确等于预登记的
   151 个 confirmed condition；
3. `completed_condition_count=151`，confirmed coverage 为
   `42 curves / 434 raw points / 151 conditions`；
4. 当前 result/contributions SHA-256 分别等于 manifest 收据；
5. 所有逐工况 guards、力账闭合、claim graph identity、resolved call
   contract、runtime identity 和 140/11 source closure 通过；
6. 对相同 timestamp 再执行一次 `--resume`，必须在零新增 GPU 工况下完成
   151 条重验；
7. Fig19(c,d) 的 8 条条件曲线和 96 个测点不得进入主评分或病因排序。

Postprocessor 必须以固定 SHA 读取上述 authorization，并据此验证 source mapping、
V4.1 graph、resolved-call、runtime 和 guard 的固定合同；禁止只让 manifest、
graph hash 或 guard 自己证明自己。

任一门失败的状态为 `INVALID_EVIDENCE`，不是模型精度 NO-GO，也不得用旧
mixed baseline 填补。

## 3. 固定后处理顺序

1. 先生成 schema-v3 scorecard；模型曲线插值到原始测量横坐标，实验力值不插值。
2. 从 scorecard 生成 confirmed-only residual artifact，并绑定 result、
   complete manifest、scorecard、`data.md` 和 scorer source 的哈希。
3. 在不读取节点 contribution 的情况下生成描述性残差指纹。
4. 最后才允许打开 contribution，执行另行预登记的父节点归因。

`fig171819_benchmark.py` 因 Fig19(c,d) 身份未解决应返回
`promotion_eligible=false`；只有且恰好存在
`fig19_cd_fixed_frequency_unresolved` blocker 是当前预期状态，不能把它当成
confirmed42 评分失败。

这里的 `184` 也不是独立于曲线身份的先验真值：它成立的前提是 Fig19(c,d)
推力/升力通道共用当前尚未证实的固定频率。若权威来源证明两通道频率不同，必须
按真实身份重建全局 condition union；禁止为了维持 184 这个数字而错绑实验。

## 4. 冻结统计单位

必须同时报告：

- 434 点 point-weighted；
- 42 条官方曲线 equal-weighted；
- 34 个独立 physical family equal-weighted；
- 按 Fig17/Fig18/Fig19 × L/T 分层的 family-equal MAE、bias 和趋势指标；
- 8 个 alias group 的测量一致性，且每组在 claim 归因中最多计一票。

主病因排序以 34-family equal-weighted 和预登记零和趋势 contrast 为准。
point-weighted/42-curve 统计只作敏感性报告；三种排序冲突时输出
`NO_DECISION_WEIGHTING_SENSITIVE`，不得选择 claim。

## 5. 输出与禁止结论

预期新增、不可覆盖的产物：

- fresh151 schema-v3 scorecard；
- fresh151 confirmed-only residual artifact；
- fresh151 confirmed42/34-family descriptive fingerprint；
- postprocess execution receipt（记录全部输入、源码和输出哈希）。

本阶段允许的最高结论是“V4.1 fresh 数据残差病灶 + 待归因 witness”。
在父节点 leave-one-force-out 归因完成前，固定输出：

```text
claim_decision = NO_DECISION
reason = NODE_ATTRIBUTION_REQUIRED
```

禁止从 partial 结果、总 L/T 相关性、最大单点误差或旧 mixed baseline 直接选择
N2/N3；禁止启动 LESP 调参、结构模型、空间涡候选或任何常数补偿。

## 6. 后续候选的通用晋升门

以下阈值写于 fresh baseline 精度统计和候选结果之前，后续只能由独立数值复现
审计使其失效，不能根据候选表现放宽：

- baseline 与 candidate 必须引用同一个 immutable baseline bundle，并在同一
  scorer、measurement parser 和 physical-family contract 下成对评分；
- candidate 的 overall 34-PF equal-weighted MAE 至少比 fresh V4.1 降低
  `0.15 N`；
- overall-L、overall-T 以及每个 Fig×channel 的 PF-equal MAE 均不得比
  fresh V4.1 恶化超过 `0.15 N`；
- PF-equal 趋势捕获不得下降，不得新增预登记 slope-sign 或 turn-topology
  失败；
- 机理效应必须在至少两个独立 PF 复现；同一 alias group 只能计一票；
- 代表点跨独立进程复现满足
  `max(|ΔL|, |ΔT|) <= 0.15 N`；
- candidate 每一步必须提供有作用位置的空间载荷，面板积分后的全翼力与关于
  固定参考点的力矩必须同时闭合；只有合力账、求总力后经验重分配、按面积/质量/
  刚度加权回填均为 co-design NO-GO。

通过 confirmed42 只授权继续解决 Fig19(c,d) 身份并进入最终 50 曲线门，不等于
全局 claim 晋升。
