# N2.6e1b2 Xia--Mohseni 联立 junction 空间门预登记 v2

日期：2026-07-30  
Claim：`N2.6e1b2`  
状态：`PREREGISTERED / NOT EXECUTED`  
取代：`n26e1b2_xm_coupled_junction_prereg_20260730.md`（未执行）

## 预执行修订理由

一手文献、符号映射和数值轴的独立审计在任何正式候选运行之前完成，发现
v1 的 canonical 有六处定义不足：无 `g` 初始化端点被误用为已形成的
previous state；epsilon 裁切几何、Xia/仓库符号、previous predictor 与
current solve、量纲缩放、signed Cauchy 和失败解释未完全冻结。

本版只修复实验定义。以下内容保持不变：单一候选、实际 NACA0015、
`32/64/128 panels-per-side`、`epsilon/h=1/4,1/8,1/16`、
`Delta t U/c=0.01`、`2%` 门、禁止读取任何压力/力/目标响应。

## 候选和本轮边界

本轮只实现：

> actual NACA0015 上的线性节点束缚涡片 + 一个常强度 forming panel +
> no-through/Kelvin/finite-angle Kutta 同一线性系统。

它是 `N2.6e1b` 的 shadow **空间算子**，不覆盖现有 solver 或 V4.1，
不计算压力或力。冻结的 `finite_angle_sheet_formation()` 仅提供
previous-state 的方向和相对形成速度，不复制、不改式。

## 冻结几何、参考系与符号

- `n_side in {32,64,128}`，总面板数 `N_p=2*n_side`；
- actual surface 的仓库轮廓为 clockwise
  `lower TE -> LE -> upper TE`；算子内部唯一采用
  `nodes_ccw=surface.contour_nodes[::-1]`，即
  `upper TE -> LE -> lower TE`；
- `t_ccw=-t_cw[::-1]`，`n_out,ccw=n_out,cw[::-1]`，
  `L_ccw=L_cw[::-1]`；
- CCW node `0` 是 upper TE 的 `gamma1`，node `N_p` 是 lower TE
  的 `gamma2`；
- 固定壁面时
  `u1_plus=gamma1`、`u2_minus=-gamma2`。一般移动壁面只可在后续门用
  `gamma_physical=gamma_B-u_body dot t_ccw`，本门不得偷加；
- current Kutta 行为
  `gamma_g=gamma1*cos(delta1)+gamma2*cos(delta2)`；
- Kelvin 行全部采用正 CCW 环量：
  `trapezoid(gamma_B,L)+gamma_g*L_g+sum(Gamma_old)=Gamma_ref`。

参考系唯一冻结为**静止翼体 + 均匀来流**：
`u_inf=(cos(alpha),sin(alpha))` 直接进入 no-through RHS；不再叠加等价
的刚体平移，禁止 Galilean 双计。

## epsilon 几何

对裁切前上下 terminal panel 长度取相同的
`h_TE=min(L_first,L_last)`，再定义
`epsilon=r*h_TE`，`r in {1/4,1/8,1/16}`。`h_TE` 不随裁切重算。

只做以下操作：

1. 沿 upper CCW 第一面板把 node 0 从 TE 向内移动 `epsilon`；
2. 沿 lower CCW 最后一面板把 node `N_p` 从 TE 向内移动
   `epsilon`；
3. 其余 actual-surface 节点保持不变；
4. forming direction 为 `d_g` 时，
   `z_g0=z_TE+epsilon*d_g`，
   `z_g1=z_TE+(epsilon+u_g^n*Delta t)*d_g`。

因此 forming 积分长度严格为 `L_g=u_g^n*Delta t`，epsilon 不进入
circulation length。

## previous predictor 与 current solve

空间门不把无 `g` 初始化的 finite-corner endpoint 当成收敛的 previous
provider。它直接复用已 validated/frozen 的无量纲 formation canonical：

| case | alpha | previous `u1_plus/U` | previous `u2_minus/U` |
|---|---:|---:|---:|
| side1 dominant | +6 deg | -2 | -1 |
| mirror side2 dominant | -6 deg | -1 | -2 |
| symmetric no-birth | 0 deg | -1 | -1 |

这些是既有 `actual_boundary_finite_angle_sheet_cases.yaml` 的解析 oracle，
不是模型参数或目标拟合。`+/-6 deg` 在 Xia 公开的附着 `0--10 deg`
验证范围内。

- previous `u1_plus/u2_minus` 只计算 `delta1^n,delta2^n,d_g^n,u_g^n`；
- current `N_p+1` 个 `gamma_B` 和 `gamma_g` 只由
  `N_p` 个无穿透方程、一个 Kelvin 方程和 Eq. (3.5) 联立求得；
- **禁止**要求 current `gamma_g` 等于 previous oracle 的
  `sheet_strength`；
- previous `dotGamma` 只作 predictor diagnostic；
- current forming circulation 唯一定义为
  `Gamma_g=gamma_g^(n+1)*u_g^n*Delta t`；
- symmetric case 必须返回无 segment 的显式 no-birth 状态，禁止为
  `gamma_g=dotGamma=0` 伪造 `u_g`。

`U=c=1`，`Delta t U/c=0.01`；该时间步来自 Dumoulin et al. 的公开
验证离散，只隔离空间门。无历史点涡、core、IBL、transpiration 或压力。

## 冻结解析单测

### 线性涡面板

panel `P0=(0,0), P1=(2,0)`，target `X=(0.6,0.8)`，正 CCW 点涡
强度的两个节点基函数必须为：

```text
start = (-0.1584393245247987, -0.00198777112016646)
end   = (-0.11135238755548409, -0.07404941370833168)
```

`gamma0=1.3,gamma1=-0.4` 时总速度为
`(-0.16143016686004469,+0.027035663027116272)`；与 32 点或更高
Gauss--Legendre 直接积分的绝对误差 `<=1e-11`。

此外必须满足：

- 相等节点强度逐位退化为已有常强度涡段，归一误差 `<=1e-12`；
- endpoint+strength 同时交换不变，旋转协变误差 `<=1e-12`；
- CCW 自面板 midpoint 的 principal-value tangential 为零，outward-normal
  start/end 自项分别为 `-1/(2*pi),+1/(2*pi)`；
- 镜像时涡量按伪标量反号。

### 几何和镜像

- NACA0015 wedge 必须由 upper/lower downstream tangents 推导，不得输入
  wake angle；`32/64/128` 的参考值约为
  `20.5637/20.5873/20.5932 deg`；
- 镜像节点强度必须满足
  `gamma_minus[i]=-gamma_plus[N_p-i]`；
- `gamma1/gamma2/gamma_g/Gamma_g` 按镜像反号并交换侧别，
  `delta1_minus=delta2_plus`，`u_g` 不变，
  `d_g_minus=(d_gx_plus,-d_gy_plus)`；
- symmetric case 只允许 no-birth。

## 量纲缩放与代数门

先以 `gamma/U` 为未知；normal/Kutta 行除以 `U`，Kelvin 行除以 `Uc`。
只在这个固定 scaled system 上计算矩阵 2-norm condition number 和
infinity-norm residual。

每个非退化工况必须：

- 最大无穿透残差 `/U <=1e-10`；
- Kelvin 残差 `/Uc <=1e-10`；
- current finite-angle Kutta 残差 `/U <=1e-10`；
- scaled linear-system infinity residual `<=1e-10`；
- previous formation oracle 的角度、方向、强度和环量率归一残差
  `<=1e-12`；
- scaled 2-norm condition number 有限且 `<=1e12`。

## panel--epsilon 双轴 Cauchy 门

对以下**带符号**量比较：

```text
gamma1, gamma2, gamma_g, Gamma_g, Gamma_bound,
u_g, delta1, delta2, absolute_forming_angle
```

角度差一律 wrap 到 `[-pi,pi]`。定义

```text
score(coarse,fine) =
    abs(wrapped_or_linear_difference) /
    max(abs(fine),0.02*physical_scale)
```

强度/速度尺度 `U`，环量尺度 `Uc`，角度尺度 `1 rad`。必须同时满足：

1. 每个 panel 层级的 `epsilon/h: 1/8 -> 1/16` 所有 score `<=2%`，
   且不大于 `1/4 -> 1/8`（允许 `1e-12` 裕量）；
2. 取每层 `epsilon/h=1/16`，`64 -> 128` 的所有 score `<=2%`，
   且不大于 `32 -> 64`；
3. 三个 panel、三个 epsilon、正负镜像与 no-birth 全部完成；禁止丢弃
   失败层级或只报告积分环量。

## go/no-go 和解释边界

- 全部通过：只将 `N2.6e1b2` 的**固定 canonical 空间离散**晋升为
  validated/frozen，授权另行预登记移动壁面和
  forming-to-material-wake 的 panel/dt/core 时间门。
- 任一失败：`N2.6e1b2` 的当前固定预算实现
  `not converged within preregistered budget`，登记
  falsified/frozen；禁止事后改网格、epsilon、阈值、AoA 或时间步重跑。

无论结果如何，都不能据此证伪 Xia--Mohseni 连续机理，也不能授权
`N2.6e1c`、Figure 12、Fig17/18/19、压力或生产载荷。

## 禁止项

- 最近控制点、端点标量外推、固定 bisector/表面 tangent；
- 当前 `gamma_g` 后处理覆盖、双重 Kutta 或 Kelvin；
- epsilon/core/阻尼/clamp/网格由目标响应选择；
- LESP 作为持续涡力幅值；
- 任何 Figure 12、Fig17/18/19、压力或力读取。

## 一手证据

- Xia & Mohseni (2017), DOI `10.1017/jfm.2017.513`, Eqs. (2.5)--(2.6),
  (3.5), (5.1), (5.4), (6.1), (6.3), (6.6)--(6.8).
- Dumoulin, Eldredge & Chatelain (2023),
  DOI `10.1017/jfm.2023.997`, §2 and validation discretization.
- `actual_boundary_finite_angle_sheet_cases.yaml` and
  `actual_boundary_finite_angle_sheet_results.json`.
- `research_n26e1b2_xm_coupled_junction_decision_20260730.md`.
