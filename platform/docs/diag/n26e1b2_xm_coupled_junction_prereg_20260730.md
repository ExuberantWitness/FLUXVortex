# N2.6e1b2 Xia--Mohseni 联立 junction 空间门预登记

日期：2026-07-30  
Claim：`N2.6e1b2`  
状态：`SUPERSEDED BEFORE EXECUTION`

本版本从未执行。预执行的独立符号与数值轴审计发现：把无 `g`、只有
Kelvin 的初始化端点直接当成 `alpha=6 deg` 的 previous-state provider，
会把有限角 corner 的初始化奇异状态混入空间门；同时本版本尚未冻结
predictor/current 分工、dimensionless row scaling 和 epsilon 几何操作。
执行契约由
`n26e1b2_xm_coupled_junction_prereg_v2_20260730.md` 取代，网格、
epsilon 序列、`2%` 门和 `Delta t U/c` 均未因目标响应而改变。

## 候选和本轮边界

本轮只实现一个候选：

> actual NACA0015 上的线性节点束缚涡片 + 一个常强度 forming panel +
> no-through/Kelvin/finite-angle Kutta 同一线性系统。

它是 `N2.6e1b` 的 shadow 空间算子，不覆盖
`svi_dw_unsteady_outer_2d.py`，不修改 V4.1，不计算压力或力，不读取
Figure 12、Fig17/18/19。冻结的 `finite_angle_sheet_formation()` 只作为
方向/速度代数 oracle 被调用，禁止复制或改式。

## 冻结 canonical

- 几何：closed NACA0015，`c=1`，cosine actual surface；
- 参考系：固定翼体，`U=1`，`alpha=+6 deg` 及其 `-6 deg` 镜像；
- `+6 deg` 位于 Xia--Mohseni 公开附着验证的 `0--10 deg` 范围内；
- 初始化：无历史尾迹、总环量为零，以 `N` 个无穿透方程和一个 Kelvin
  方程求 `N+1` 个独立束缚节点强度；
- forming step：用初始化状态的两个物理尾缘节点强度确定方向和
  `u_g`，取来源验证量级 `Delta t U/c = 0.01`；当前
  `N+1` 个束缚节点强度与 `gamma_g` 再由 `N+2` 方程联立；
- 历史点涡/core/IBL/transpiration/壁面运动均不进入本空间门。

网格层级固定为 `32/64/128 panels-per-side`。尾缘第一和最后面板按
`epsilon` 裁去同一弧长，forming panel 从相同 offset 开始；
`epsilon/h_TE` 固定为 `1/4, 1/8, 1/16`。这三个值只构成
`epsilon/h_TE -> 0` 的 Cauchy 序列，不得从任何响应中选择“最佳值”。

## 实现必须具名输出

- `N+1` 个线性束缚节点强度，首末尾缘节点保持独立；
- `gamma1`、`gamma2` 和映射后的 `u1_plus/u2_minus`；
- `delta_theta1/2`、绝对 forming 方向、`u_g`；
- 当前 `gamma_g`、forming 长度和积分环量；
- bound/forming/total circulation；
- 每个控制点无穿透残差、Kelvin 残差、Kutta 残差、线性系统残差和条件数；
- `+/-6 deg` 的镜像量；
- 每个 panel 与 epsilon 层级的上述冻结结果。

## 单元与解析门

1. 线性涡面板两个节点基函数分别与至少 32 点 Gauss--Legendre 直接积分
   对照，远场速度归一误差 `<=1e-11`；
2. 两个节点强度相等时，线性面板速度逐位退化为已有常强度涡段速度，
   归一误差 `<=1e-12`；
3. 节点基函数线性叠加、端点交换和几何旋转协变误差 `<=1e-12`；
4. `alpha=0` 的初始化必须进入显式 no-birth 状态，不允许用任意方向或
   `0/0` 形成速度伪造尾迹；
5. `+/-6 deg` 的环量、节点尾缘状态、forming 方向和强度满足镜像/反号，
   归一误差 `<=1e-10`。

## 代数与 Cauchy go/no-go

每一冻结工况必须：

- 最大无穿透残差 `/U <=1e-10`；
- Kelvin 残差 `/Uc <=1e-10`；
- finite-angle Kutta 残差 `/U <=1e-10`；
- 线性系统最大残差 `/U <=1e-10`；
- Xia 形成 oracle 的角度、方向、强度和环量率归一残差 `<=1e-12`；
- 条件数有限且 `<=1e12`。

对 `|gamma1|, |gamma2|, |gamma_g|, u_g, delta_theta1,
delta_theta2, forming circulation, bound circulation` 定义

```text
score(coarse,fine) =
    abs(fine-coarse) / max(abs(fine), 0.02*physical_scale)
```

速度/片强度尺度为 `U`，环量尺度为 `Uc`，角度尺度为 `1 rad`。

必须同时满足：

1. 在每个 panel 层级，`epsilon/h: 1/8 -> 1/16` 的所有 score
   `<=2%`，且不大于 `1/4 -> 1/8`（允许 `1e-12` 裕量）；
2. 取每层 `epsilon/h=1/16` 的数值，`64 -> 128 panels-per-side`
   的所有 score `<=2%`，且不大于 `32 -> 64`；
3. 所有层级与镜像工况均完成，不得丢弃失败层级或只报告积分环量。

任一失败即 `N2.6e1b2 = falsified/frozen`。禁止改阈值、追加更细固定网格、
选择某个 epsilon、改 AoA/时间步或加入阻尼后重走。

全部通过只将 `N2.6e1b2` 晋升为 validated/frozen，并授权另行预登记：

- 移动壁面 `gamma_B-gamma_b` 映射；
- forming panel 向物质点涡的 RK4 时间推进和 panel/dt/core 三轴门。

它不授权 `N2.6e1c`、Figure 12 或目标翼载荷。

## 证据

- `research_n26e1b2_xm_coupled_junction_decision_20260730.md`
- Xia & Mohseni (2017), DOI `10.1017/jfm.2017.513`, §6.1.
- Dumoulin, Eldredge & Chatelain (2023),
  DOI `10.1017/jfm.2023.997`, §2.
