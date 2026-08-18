# FluxV 三论文全部工况曲线与模型对比

## 结论摘要

已完成 22 个冻结工况的全部特性曲线，不再只给聚合数字：

- Yang：6 个安装角的平均升力、阻力随迎角变化；
- Izraelevitz Figure 14：14 个实验 marker / 12 个唯一工况的平均推力随相位差变化；
- Baik：W1--W4 的完整周期相位升力和阻力，共 8 条实验载荷通道；
- 另给 Baik 未滤波高频诊断、全部逐工况误差比热图和 v5b 的阻断门控图。

全矩阵结果表明：**v4b 仍是本轮三论文中应保留的联合候选。v5a 只在 Yang 周期均值上改善 v4b，但在 Figure 14 和 Baik 上明确退化；v5b 在进入论文评分前就因 no-LEV 精确退化失败而被阻断。** 因而当前没有证据把 v5a 或 v5b 宣称为比 v4b 更泛化的下一代模型。

## 聚合对比

| 数据集 / 指标 | FluxV old | FluxV v4b | v5a proxy | v5a / v4b | 结论 |
|---|---:|---:|---:|---:|---|
| Yang 升力 MAE [gf], n=6 | 6.8549 | 4.5545 | 3.9514 | 0.868 | v5a 改善 |
| Yang 阻力 MAE [gf], n=6 | 12.9216 | 2.6440 | 2.0626 | 0.780 | v5a 改善 |
| Figure 14 CT RMSE, 14 markers | 0.05115 | 0.02595 | 0.09470 | 3.649 | v5a 显著退化 |
| Figure 14 CT RMSE, 12 unique | 0.04351 | 0.02751 | 0.08930 | 3.246 | v5a 显著退化 |
| Baik filtered CL macro RMSE | 0.69484 | 0.65754 | 0.80165 | 1.219 | v5a 退化 |
| Baik filtered CD macro RMSE | 0.40728 | 0.34515 | 0.40441 | 1.172 | v5a 退化 |

这些行保留各自物理单位和评分定义，没有把 gf、CT、CL、CD 混成一个总分。机器可读表为 `tables/table01_aggregate_metrics.csv`。

## 逐工况发现

### Yang 2025

- v4b 相对 old：升力仅 2/6 个迎角改善，但阻力 6/6 改善；聚合升阻力误差都显著低于 old。
- v5a 相对 v4b：升力 4/6、阻力 5/6 个迎角改善；20 deg 升力和 25 deg 升阻力没有继续改善。
- v5a 的六点平均载荷与既有 v1/v2 数值重合；这说明 Yang 上的提升来自已经存在的 equilibrium/polar 分支，不是新状态方程得到的独立泛化证据。
- 作者 modified UVLM 的升/阻力 MAE 为 3.350/1.933 gf，仍略优于 v5a 的 3.951/2.063 gf。

图：`figures/fig01_yang_all_conditions.pdf`。

### Izraelevitz 2017 Figure 14

- v4b 相对 old 在 12 个唯一工况中的 9 个改善，14-marker RMSE 降低 49.3%。
- v5a 相对 v4b 仅 2/12 个唯一工况改善；在 theta=15 deg、psi=15/30/45/60/75 deg 以及 theta=25 deg、psi=45/60 deg 出现明显过修正。
- v5a 的 14-marker RMSE 是 v4b 的 3.649 倍，直接否决“同一 v5a 机制同时改善三篇论文”的假设。
- theta=25 deg、psi=15/30 deg 只有作者参考模型，没有实验观测；图中保留为 reference-only，不插值或补造 FluxV 预测。

图：`figures/fig02_izraelevitz_fig14_all_conditions.pdf`。

### Baik 2012 W1--W4

- v4b 相对 old 在 W1--W4 的 CL/CD 共 8/8 个 source-matched filtered 通道均改善。
- v5a 相对 v4b 在 8/8 个通道全部退化；最明显的是 W2/W3/W4 的升力和 W1/W4 的阻力。
- filtered 主图严格采用论文 1 Hz 带宽口径；raw 补充图显示 v5a 还带来明显高频振荡，不能靠滤波主图掩盖。
- published standard Theodorsen 只有升力曲线，没有人为补造阻力预测。

主图：`figures/fig03_baik_w1_w4_filtered.pdf`；高频诊断：`figures/figS01_baik_w1_w4_raw_numeric.pdf`。

### v5b 状态

v5b 没有 22 个工况的论文性能曲线。它在共同 Yang 15 deg 运动输入、LEV 关闭的前提下仍不能精确退化为 current FluxV：最大相位差为 abs(dCL)=0.556、abs(dCD)=0.529。因此 G1 失败后按预定合同停止，三论文评分为 `blocked_not_scored`。任何给 v5b 补画论文曲线的做法都会伪造证据。

门控图：`figures/fig05_v5b_no_lev_gate.pdf`。

## 如何阅读误差热图

`figures/fig04_all_condition_error_ratio.pdf` 的每个格子都是 v5a/v4b 的误差比：

- Yang 和 Figure 14：逐工况绝对误差；
- Baik：每个 W 工况的完整相位 RMSE；
- 蓝色、比值小于 1 表示 v5a 改善；红色、比值大于 1 表示退化；
- Figure 14 中极大的 20.11x、10.81x 等比值来自 v4b 在相应点已接近实验，而 v5a 再次大幅修正，并非跨单位归一化造成。

## 证据与复现文件

- 全部 27,435 条长表曲线：`data/all_conditions_curves.csv`；
- 全部逐点、分组和宏指标：`data/all_conditions_metrics.csv`；
- 模型覆盖与未评分原因：`data/model_coverage.csv`；
- 输入/输出哈希和行数：`data/build_manifest.json`；
- 聚合表 CSV/LaTeX：`tables/`；
- 图件哈希、脚本哈希和自包含图注：`figures/figure_manifest.json`、`figures/latex_includes.tex`；
- 完整数据与绘图口径：`ALL_CONDITIONS_FIGURE_PLAN.md`。

验证结果：新增全工况测试 4/4 通过；与既有 v5/LDVM/Hirato 回归合并后为
`101 passed, 6 subtests passed`。在仓库外第二个输出目录端到端重建后，数据、表格、
12 个 PNG/PDF 图件及两份 manifest 均逐字节一致；两个 manifest 声明的 46 项输入、
源码、输出和 LaTeX 哈希全部匹配。误差热图 PDF 不含嵌入式栅格图像。这些检查证明
工件可重算和身份一致，不替代物理模型验证。

## 科学边界

1. Yang 和 Figure 14 只有周期均值真值，不能声称实验相位验证。
2. v5a 是已经被拒绝的 cache compatibility proxy，不是 canonical next-generation solver。
3. v5b 的图只验证失败门，不是实验精度图。
4. Baik 实验是 wall/endplate 准二维工况，而本地 UVLM 是 free-tip surrogate；模型身份限制不因曲线完整而消失。
5. 本报告支持“v4b 在现有候选中最稳健、v5a/v5b 未晋级”，不支持“已经得到统一更优的 v5”。
