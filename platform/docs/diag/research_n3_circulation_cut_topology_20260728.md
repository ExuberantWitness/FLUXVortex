# N3 实际边界非零环量的 potential-cut/wake 拓扑裁决

日期：2026-07-28  
Claim：`N3.1j3b6d`  
前置证据：`N3.1j3b6d3d/e` 已通过成对弱算子与附着球压力门。

## ① 病因定位

附着球 oracle 证明连续 P2 actual-boundary source–doublet 方程可以同时恢复
势、物面速度和 `Cp`。但这个系统的 body trace 是全局单值且闭合的。对任何
这样的标量势，

\[
\oint_{\partial B}\nabla_s\phi\cdot d\boldsymbol l
=\phi(\text{终点})-\phi(\text{起点})=0.
\]

所以它不能仅靠“再加一个环量约束方程”承载 N1 的非零束缚环量。若直接把
非零环量塞进同一组周期 P2 自由度，线性系统只能把它抵消、形成局部尖峰或
变得不相容。这是拓扑缺件，不是矩阵精度或待调常数。

## ② 学科机理

NASA CR-189854 明确指出，从升力面尾缘延伸的 wake cut 使势流能够拥有
非零环量，并在 cut 两侧施加质量和法向动量跳跃条件。

Moored 的三维非定常 BEM 使用 body source/doublet 加可变形 doublet wake，
并指出只有在系统中加入显式或隐式 Kutta 条件后才能支持 bound circulation。

Berci–Righi 的非定常 Kutta–Joukowski 推导把 circulatory potential 定义在
带零厚度 wake 势不连续的 Laplace 域上；wake 势跳随物质点守恒，并与 bound
circulation 按 Kelvin 账配对。

Bernasconi 等的高阶势流方法把连续高阶奇异分布统一用于厚体、薄面和 wake，
说明 body 高阶连续性与 wake 的有意不连续必须同时存在，而不是互相替代。

一手来源：

- NASA CR-189854:
  https://ntrs.nasa.gov/citations/19920015560
- Moored, *Computers & Fluids* (2017):
  https://arxiv.org/abs/1703.08259
- Berci & Righi, *AIAA Journal* (2022):
  https://doi.org/10.2514/1.J061894
- Bernasconi et al., *IJNME* 72 (2007):
  https://doi.org/10.1002/nme.2099

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 全局单值闭合 P2 body trace 可承载非零环量 | 错件 |
| 提高 body P2 阶数或网格可解除零环量恒等式 | 错件 |
| 分类 potential cut / doublet wake jump | 缺件 |
| N1 环量直接写入 source、core、offset 或压力 | 禁止 |

## ④ 机理方案与预登记

在进入三维 wing–wake 联立前，先做二维圆柱拓扑/压力 oracle：

1. 同一圆周 P2 trace 分别采用 periodic 单值拓扑和 cut 双端点拓扑；
2. periodic trace 的离散环量必须在机器精度内恒为零；
3. cut trace 的端点势跳和切向梯度积分必须同时等于规定 `Gamma`；
4. 用
   `phi=2Ua cos(theta)+Gamma theta/(2pi)` 得到切向速度；
5. 只执行一次 Bernoulli 和一次圆周压力积分，恢复
   `D=0, L'=-rho U Gamma`；
6. 加任意常势后速度和力必须不变。

完整网格族、阈值和符号在
`actual_boundary_circulation_cut_cases.yaml` 中于实现前冻结。通过只授权
“3D actual-boundary 方程必须显式带 wake jump”这一拓扑命题，不授权任何
有限翼、尾缘、wake relaxation 或生产载荷。

## 执行结果

预登记后才实现 `claim_runtime/circulation_cut_oracle.py`，使用相同曲线
P2 trace、相同一次求导、一次 Bernoulli 和一次压力积分完成全部算例。

| 门 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| closed trace 最大离散环量 | `7.216e-16` | `1e-13` | PASS |
| cut trace 环量相对误差 | `1.110e-15` | `1e-13` | PASS |
| cut 势跃变相对误差 | `2.776e-16` | `1e-13` | PASS |
| 64 面板切向速度 RMS | `6.447e-4` | `0.01` | PASS |
| 64 面板 Kutta–Joukowski 升力相对误差 | `9.674e-8` | `0.005` | PASS |
| 64 面板阻力绝对值 | `5.613e-16` | `1e-12` | PASS |
| 常势平移后速度/力最大变化 | `8.038e-14 / 1.454e-14` | `1e-13 / 1e-13` | PASS |

速度 RMS 随 `8/16/32/64` 面板严格下降：
`4.088e-2 → 1.029e-2 → 2.578e-3 → 6.447e-4`；升力误差也严格下降：
`3.901e-4 → 2.467e-5 → 1.547e-6 → 9.674e-8`。

因此 S2a 为 **GO**，但 GO 的范围很窄：

- `N3.1j3b6d4`：“全局单值闭合 trace 可直接承担非零环量”被证伪并冻结；
- `N3.1j3b6d5`：“非零环量需要分类势切面/尾迹势跃变”在二维拓扑与附着
  压力 oracle 范围内验证并冻结；
- 三维有限翼 body–wake 联立、Kelvin 新生强度、有限 base Kutta、尾迹
  物质推进、非定常压力和生产载荷仍未验证，不能由本结果外推。
