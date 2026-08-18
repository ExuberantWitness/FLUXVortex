# 全工况曲线冻结合同

## 范围

本图集中的“所有工况”严格指已经有冻结实验真值、并且进入本轮交叉比较的数据：

- Yang 2025：6 个安装迎角，画周期平均升力和有符号阻力；
- Izraelevitz 2017 Figure 14 / Scherer 1968：14 个实验观测、12 个唯一运动工况，画周期平均推力系数；
- Baik 2012：W1--W4，每个工况 400 个唯一相位点，画完整相位升力和阻力；
- 共 22 个冻结工况。Figure 14 的两个重复观测在 14-marker 主指标中分别保留。

## 公平性合同

1. 不允许任何相位、幅值、均值、偏置或时间平移拟合。
2. Yang 只公开周期均值，不能把本地模型相位历史称为实验相位验证。
3. Figure 14 只公开周期平均推力，不能生成不存在的实验瞬时载荷曲线。
4. Baik 主图采用原文匹配的 1 Hz 数值滤波：W1/W4 保留至第 7 谐波，W2/W3 保留至第 3 谐波；公开实验本身不再重复滤波。
5. Baik 原 CSV 的 phase=1 是 phase=0 的重复端点；评分只使用 400 个唯一相位，绘图数据也不重复该端点。
6. Yang 的 ±0.4 gf 仅是 PDF 数字化不确定度，不是实验统计误差或置信区间。
7. Figure 14 的非对称误差条是原图报告条的数字化结果，论文没有公开其统计定义；指标不按误差条加权。
8. 不跨 gf 与无量纲系数求“总平均分”。

## 模型身份

- `FluxV old`：当前/旧 FluxV 的基线载荷通道；
- `FluxV v4b`：三论文中当前保留的 qualified candidate；
- `FluxV v5a development proxy`：冻结 cache-adapter 诊断，已被 promotion gate 否决；
- `FluxV v5b`：no-LEV 精确退化门 G1 失败，未运行 22 个论文工况，因此只画单独的门控图，不补造论文曲线；
- Yang 的 v5a 六点均值与 v1/v2 在机器精度内重合，图中明确合并身份；
- Ptera prescribed-wake 在 Yang 上与 FluxV-old 载荷通道重复，不重复画线；
- RoboFalcon2 coefficient transfer 是跨几何/雷诺数诊断，避免压缩主图坐标轴，不进入主比较。

## 输出图

1. `fig01_yang_all_conditions`：6 个迎角的升阻力特性；
2. `fig02_izraelevitz_fig14_all_conditions`：两个俯仰幅值下的全部 CT--相位差曲线；
3. `fig03_baik_w1_w4_filtered`：W1--W4 source-matched 1 Hz 主相位图；
4. `figS01_baik_w1_w4_raw_numeric`：未滤波数值高频诊断；
5. `fig04_all_condition_error_ratio`：逐工况 v5a/v4b 误差比；
6. `fig05_v5b_no_lev_gate`：v5b 被阻断的 no-LEV 门控证据。

每幅图都提供 320 dpi PNG 与矢量 PDF；图注集中在 `figures/latex_includes.tex`，哈希在 `figures/figure_manifest.json`。

