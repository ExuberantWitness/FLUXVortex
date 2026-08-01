# Fig17/18/19 evidence-scope 评分隔离预登记

**日期**：2026-07-29  
**状态**：PRE-REGISTERED，尚未修改评分器  
**类型**：数据证据合同；不修改 V4.1 气动公式、参数、网格、运动学或 claim state

## 1. 已观测病因

Fig19(c,d) 扭转扫的固定频率在论文 PDF、正文、图例、caption、JATS 与现有
原始数据记录中均未被声明。当前 `2.6 Hz` 只能作为条件性 solver 假设，不能作为
claim 选择证据。

现有 schema-v2 scorecard 把 Fig19(c,d) 的 96 个条件性测点与其余来源已确认
测点混合。冻结 118 点上的直接反例是：

- confirmed-only 总 MAE：`1.2793254479505114 N`；
- 混合总 MAE：`1.4206 N`；
- confirmed-only lift MAE：`0.8976018362015258 N`；
- 混合 lift MAE：`1.0709 N`。

因此，顶层 `promotion_eligible=false` 虽能阻止发布，却不能阻止条件数据污染
残差排序和 claim 病因判断。

## 2. 冻结输入

| 资产 | SHA-256 | 身份 |
|---|---|---|
| `s6_sweep_v41_full184_20260729_105013.json` | `e8e903b8760f760f2a379032d2d2b3e814ba789f8e3e2fafa6de65ea851b8ea6` | 184/184 数值结果 |
| `fig171819_v41_baseline_manifest_20260729_105013.json` | `5f0f02b4346bef03c8bab012088c1a65401fa463040f34caaa7b0b4df859bc82` | runner 身份与 66 点 guards |
| `scorecard_v41_full184_20260729_105013.json` | `8c399a96759853807891b731b501f9592a5fa9d36a463510ca087e62afbf9a5b` | 原 schema-v2 条件性 scorecard；只读保留 |
| `data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` | 50 曲线、530 原始测点 |

旧 result、manifest 和 scorecard 禁止覆盖。新评分器只能生成新文件名的
scope-aware scorecard 与 confirmed residual artifact。

## 3. 唯一改写命题

证据状态属于“实验曲线—solver 工况绑定”，不属于 solver condition 本身。
Fig19(c,d) 与 confirmed 曲线共享 15 个工况，因此禁止建立
`condition_key -> evidence_scope` 单值映射。

预登记范围：

| 范围 | 曲线 | 原始测点 | 支撑 solver 条件 |
|---|---:|---:|---:|
| `confirmed` | 42 | 434 | 151 |
| `conditional_fig19_cd` | 8 | 96 | 48 |
| 交集 | — | — | 15 |
| conditional-only | — | — | 33 |
| 全部 | 50 | 530 | 184 |

Confirmed 由 Fig17 全 10 条、Fig18 全 24 条、Fig19(a,b) 8 条组成。其
151 个条件必须精确分解为冻结种子 85 点加本轮 invariant Fig18 新增 66 点。

## 4. 预登记实现

1. `CurveSpec` 显式保存 `evidence_scope`；每个 score row 继承该字段。
2. `coverage()` 可分别计算 global、confirmed 和 conditional 范围；extra key
   始终相对于全局 184 合同判断。
3. schema 升为 v3：
   - 顶层 `rows/aggregates` 明确定义为 confirmed-only 主证据；
   - conditional rows/aggregates 只位于具名的独立 scope；
   - 禁止保留无标签的 50 曲线混合主统计。
4. 原始 50 曲线、530 点的数据身份校验保持不变，不因 scoped scoring 弱化。
5. 独立 confirmed artifact 保存 42 条完整逐点残差、434 点统计、151 条件
   coverage、排除的 8 条曲线及原因，以及 result、runner manifest、
   原 scorecard、data 和新 scorer 的 SHA-256。
6. Confirmed artifact 只允许用于 V4.1 病因诊断和候选预登记；不得用于宣称
   50 曲线全局晋升。

## 5. GO / NO-GO

### GO

- 精确计数满足 `42/8` 曲线、`434/96` 测点、`151/48` 条件、
  `15` 条件交集与 `33` conditional-only；
- 冻结 118 点 scoped coverage 为 confirmed `85/151`、`30/42` 完整曲线，
  conditional `48/48`、`8/8`；
- 完整结果为 confirmed `151/151`、`42/42`，conditional `48/48`、
  `8/8`，global `184/184`、`50/50`；
- 只改变 33 个 conditional-only 条件时，confirmed rows 与 aggregates
  必须逐字典不变；
- 15 个共享条件同时存在于两个 support set，不能被全局隔离；
- 顶层 confirmed aggregate 为 434 点，artifact 不含 conditional curve row；
- benchmark 与 runner 的全部非 GPU 测试通过；
- 旧三项冻结资产哈希保持不变，fixed-name baseline 仍不存在。

### NO-GO

任一精确计数错误、conditional 扰动改变 confirmed 统计、共享条件被错误移除、
实验 force 值被插值、旧资产被覆盖、无限定混合 aggregate 继续作为主证据、
或现有测试退化，均停止病因归因。

## 6. 后续边界

评分隔离通过只授权生成 confirmed42 残差指纹，不授权直接选择 N2/N3，也不授权
LESP、LEV、结构或常数修改。唯一 claim 挂接必须等待逐通道 witness attribution；
Fig19(c,d) 频率身份获得权威来源前，完整 50 曲线 production promotion 继续关闭。
