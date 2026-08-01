# N2.6e1bc0 regular-corner / Kelvin scaling：result-to-claim

日期：2026-07-30  
Claim：`N2.6e1bc0-FCR-KELVIN-SCALING`  
裁决：`PHYSICS-NO-GO / FALSIFIED / FROZEN`

## 1. 预登记与协议健康

正式运行严格使用
`n26e1bc0_corner_kelvin_scaling_prereg_20260730.md`，其冻结 SHA256 为
`86c972bc49391b3ecae26ab11191b27213cd0911a89519fc6b97294f2e03383e`。
没有读取压力、力、Fig17/18/19 目标值，也没有扫描常数、相位、拟合窗口
或阈值。

六个唯一工况全部完成。时间网格均精确落在 `t*=0.2 s`；所有阶段采用
相同 `lower` 分支；五类代数残差的全局最大值为 `2.364e-12`；实际
Kelvin birth 与 solver newborn 的最大差为 `2.776e-16`。因此这是物理
缩放门失败，不是 schema、分支、矩阵或时间对齐失败。

## 2. 决定性数据指纹

regular-only 有限角模型预言

\[
p_*=\frac{1+\beta}{1-\beta}=1.1292006035 .
\]

冻结四层时间轴
`dt={0.025,0.0125,0.00625,0.003125} s` 得到

\[
|\Gamma_{\rm birth}|=
\{0.17817319,\ 0.09257541,\ 0.04744778,\ 0.02419312\}\ {\rm m^2/s}.
\]

四点 log--log OLS 给出

\[
p_K=0.9606123259,
\]

相邻局部阶为
`0.94457935/0.96428843/0.97174383`。结果稳定趋近一般光滑
Kelvin 交换的 `O(dt)`，而不是 regular-only 的 `O(dt^1.1292)`。

空间轴的 `128 -> 256` modal 变化为：

- lower：`2.662164%`；
- upper：`0.667987%`；
- mean：`1.216087%`。

即使 upper/mean regular 坐标较稳定，时间阶仍不相容；lower 又独立越过
预登记的 `2%` 门。因此不能用加密或重新选择一侧挽救该子命题。

## 3. Claim 裁决

证伪并冻结的精确命题是：

> 消去首奇异 Kutta 模态后的有界 regular outer corner mode，单独就足以
> 同时给出有限形成速度、正则片强和 generic Kelvin 出生环量率。

结果不证伪：

- gauge-invariant Kelvin 有向 edge 账；
- 有限角弱空间作为 outer reference；
- strong viscous--inviscid interaction；
- 具有额外黏性 inner/profile 幅值的尾缘选择；
- 拥有 bulk/sheet/interface inventory 的完整 moving-interface theory。

## 4. 可动空间与禁区

父节点 `N2.6e1bc` 保持 `open`。下一候选必须只选择一个由一手来源支持的
新增组成部分：黏性尾缘 inner/profile 幅值或有限 forming-zone 库存；
它必须在统一 trial state 中与完整 body BIE、IBL、规范不变 Kelvin 和
材料 potential-jump history 联立。

以下路线因本结果或既有 claim 证据继续禁止：

- 最近控制点或 endpoint `gamma_TE`；
- 以 epsilon/core/采样半径制造有限值；
- 把 `Gamma_birth/dt` 回填成独立 weak-UK 通量；
- 全局 `dotGamma_b + Jomega - DeltaP/rho` 三项式；
- 用 Fig17/18/19 目标力选择内区常数；
- 在看到本结果后改变时间层、观察相位或拟合窗口。

正式数据：
`n26e1bc0_corner_kelvin_scaling_result_20260730.json`。人读摘要：
`n26e1bc0_corner_kelvin_scaling_result_20260730.md`。
