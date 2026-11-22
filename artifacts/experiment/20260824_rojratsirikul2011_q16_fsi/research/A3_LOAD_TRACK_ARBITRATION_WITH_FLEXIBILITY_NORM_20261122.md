# A3：载荷双轨用柔度能量范数与 6-DOF 扳手裁决 —— 膜翼上两条轨道**不等价**，且 5P 包过报合力 ~1.4×（2026-11-22）

> 提交：`q16_load_metrics.py`（补 `wrench_components`）、`q16_shadow_resolved_consumer.py`、`runtime/case_runner.py`
> 数据：`/tmp/a3_a16.json`（t\*=0.12）、`/tmp/a3_a16_30.json`（t\*=0.30）
> 验收：默认路径 30 步 digest 链**逐位不变**（shadow 消费者是纯审计，不参与载荷）

## 1. 之前缺什么

- `q16_load_metrics.py` 的 docstring 声称提供 `wrench_components`，但 `__all__` 只有两个函数、实现里没有它 —— 文档承诺了一个不存在的 API。A3 把它做成真的（转出 `q16_rigid_body_wrench.aero_generalized_wrench`）。
- shadow 消费者的比较量全是 **R^dof 欧氏范数**，把 N / N·m / N·m² 混在一起。历史记录的 "0.338 = 35%" 就是这个数。
- 消费者**拿不到刚度**，所以无法回答"这个分歧会不会推动结构"。

## 2. 现在怎么算

- 消费者新增可选 `stiffness_reference`（惰性取 `structural._reference_tangent_matrix_torch`）与 `reference_rows`（`runner._reference_rows_gpu`）；
- 每次采样额外记录：`flexibility_gap_vs_legacy_constant`、`flexibility_gap_vs_legacy_three_term`、两侧的 `flexibility_norm`、以及两侧的 `wrench_{force,moment}_norm`；
- 这些字段进入 `resolved_load_evidence.shadow_5p_to_q16` 的摘要。

## 3. 结果（A16，两个时刻，各 1 个采样）

| 量 | t\*=0.12 | t\*=0.30 |
|---|---|---|
| 原始 L2：5P 包 vs legacy **常量** | 0.3381 | 0.4405 |
| **柔度范数：5P 包 vs legacy 常量** | **0.0713** | **0.0554** |
| 原始 L2：5P 包 vs legacy **三槽和** | 0.3655 | 0.4538 |
| **柔度范数：5P 包 vs legacy 三槽和** | **0.4522** | **0.3896** |
| 柔度范数：5P 包 / legacy | 0.3893 / 0.2687 | 0.9220 / 0.6673 |
| 扳手合力：5P 包 / legacy | 0.03472 / 0.02384 | 0.08285 / 0.05936 |
| 扳手合力比 | **1.456** | **1.395** |
| 扳手合力矩比 | **1.451** | **1.386** |

## 4. 裁决（E-1）

**① 历史上那个 0.338 有一半是单位假象，但只在"对常量"这个比较上。** 对 legacy 常量，柔度范数把 0.338→0.071、0.440→0.055，即缩到 1/5–1/8。所以早先"35% 分歧"里大部分是 DOF 量纲与量级差异，这一点原判断成立。

**② 但对 legacy 三槽和，柔度范数反而更大**（0.452 / 0.390，而原始 L2 是 0.365 / 0.454）。也就是说：5P 包能比较好地复现 legacy 的**常量**分量，却在刚度加权意义上**没有复现非定常两槽**（`velocity_force` + `Mf1`）。这两个时刻的方向一致。

**③ 扳手进一步说明这不是"方向不同"而是"整体偏大"**：5P 包的合力与合力矩都稳定是 legacy 的 **1.39–1.46 倍**，力和力矩的比值几乎相同（1.456/1.451、1.395/1.386）。

⇒ **结论：膜翼上两条载荷轨道不等价。** 迁移不能只换常量槽（⑬ 原先设想的形态），必须三槽一起迁；而且 5P 包当前对刚体合力的过报约 **1.4×**，这与"膜翼 Cn 偏高 +34–36%"的已知偏差方向一致。

## 5. 边界

- 每次运行只落 **1 个 shadow 采样**（t\*=0.12 / 0.30），所以上面每个数都是单点值，不是窗口均值；两个时刻的一致性只是方向性证据。
- `K_ref` 用的是 `structural._reference_tangent_matrix_torch`（准牛顿参考切线），不是精确 Hessian；柔度范数对它的选择敏感，`regularisation=1e-12`（相对对角最大值）用于防止奇异。
- 只有 A16 一个工况；刚翼 α=5 的双轨等价（A/B=1.0001）是上一轮用 `Cn` 投影判的，没有用柔度范数重跑。
- `wrench_*` 的 `origin` 用默认值（未指定参考点），力与力矩的量级只在与 legacy 的**比值**意义下使用。
