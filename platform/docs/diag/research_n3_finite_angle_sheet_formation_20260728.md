# N3 S2c：有限角尖尾缘的守恒耦合新生涡片

日期：2026-07-28  
Claim：`N3.1j3b6d`  
前置证据：S2b sharp-cusp Kutta–pressure 门为 GO。

## ① 病因定位

S2b 的 Joukowski 尾缘只有一个共同切向。真实 NACA-2406 则有上、下两个
尾缘角和实体 base。在进入 blunt base 前，必须先隔离更简单的 finite-angle
sharp edge：

> 当上下切向不同，形成片方向不再由“流线沿尾缘切向”唯一给出。

仓库原 `K1_finite_angle_sharp_unsteady` 只登记了研究问题，没有冻结输入族、
方程残差和阈值，因此不能直接作为可执行证据。

## ② 学科机理

Xia–Mohseni（JFM 830, 2017；DOI `10.1017/jfm.2017.513`；
arXiv `1611.05729`）从 Kutta、Kelvin/环量、质量和动量得到：

\[
\Delta\theta_1+\Delta\theta_2=\Delta\theta_0,
\]

\[
u_{1+}\sin\Delta\theta_1
=u_{2-}\sin\Delta\theta_2,
\]

\[
\gamma_g=u_{1+}\cos\Delta\theta_1-u_{2-}\cos\Delta\theta_2
=u_{g+}-u_{g-},
\]

\[
\dot\Gamma_g=u_g\gamma_g
=\tfrac12(u_{2-}^2-u_{1+}^2),
\]

以及 Eq. 4.19 的切向动量残差。形成片方向必须位于两个表面切向之间，并随
两侧来流连续变化。固定 bisector 或固定一侧切向仅分别是对称与单侧停滞极限。

该论文同时明确：这是忽略涡片有限质量/动量厚度的高 Reynolds 数第一近似。
所以此门不能冒充 N2 的黏性库存，也不能冒充实体 base。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| finite-angle unsteady wake 始终沿 bisector | 错件 |
| 始终沿上或下切向 | 错件，仅单侧速度为零时成立 |
| 方向、强度和相对速度由守恒方程组联合决定 | 缺件 |
| 上述单一 junction 已解决 NACA-2406 双角+base | 错件 |

## ④ 方案与预登记

对固定 `40°` 楔角冻结六个无量纲速度算例：

- 对称 bisector；
- 上侧占优及其镜像；
- 两个单侧停滞切向极限；
- 上侧占优的 2 倍速度尺度。

非退化情形直接由解析式求
`Delta theta1/2, gamma_g, u_g, u_g+/u_g-`，再独立回代角度、方向、Kutta
强度、环量率和动量五个残差。对称情形只允许识别 `gamma=dotGamma=0` 与
bisector；禁止给零强度片捏造相对速度。

完整公式、输入、阈值和禁止项已在
`actual_boundary_finite_angle_sheet_cases.yaml` 中于实现前冻结。通过只允许
进入 blunt-base 双 junction/质量动量拓扑门，不允许直接接 RoboEagle 生产力。

## S2c 首次执行：印刷动量式符号矛盾

首次执行只有直接转录的 Eq. 4.19 动量残差失败，最大归一残差
`0.09523`；其余方向、强度、环量率、出流速度、镜像、尺度、bisector 和
切向极限全部通过。

这不是实现中的数值误差。论文对控制面法向速度的约定是
`u1+,u2-<=0`（入流）、`ug+,ug->=0`（出流），且两个夹角非负。按印刷
Eq. 4.19 的两个减号，三项必然同号，不能和为零。

从论文紧邻的原始质量式

\[
u_{1+}S_1+u_{g-}S_{g-}=0,\qquad
u_{2-}S_2+u_{g+}S_{g+}=0
\]

和切向/法向动量式直接消去四个正控制面长度，则严格得到

\[
u_{1+}u_{2-}\sin\Delta\theta_0
+u_{1+}u_{g+}\sin\Delta\theta_1
+u_{2-}u_{g-}\sin\Delta\theta_2=0.
\]

这个带符号通量的加号形式与论文最终方向式相容。Le Provost 等 JFM 977
(2023) 的实际系统也使用 Xia–Mohseni 的最终方向和强度式，而没有复用矛盾的
印刷减号残差。

因此在再次实现前登记 S2c1：仅将 diagnostic momentum residual 改为从原始
控制体方程独立推导的 signed-flux 形式；其余公式、算例、阈值和门全部不变。
若仍失败，不得继续 blunt-base。

## S2c1 执行结果与 claim 改写

重新执行后全部冻结门通过：

| 指标 | 最大残差 |
|---|---:|
| 角度和 | `0` |
| 形成方向 | `1.8504e-17` |
| Kutta 强度 | `0` |
| 环量率 | `0` |
| signed-control-volume 动量 | `4.1119e-18` |
| 镜像 | `5.5511e-17` |
| 速度尺度协变 | `0` |
| 对称 bisector | `0` |
| 单侧停滞切向极限 | `0` |

所有可识别算例的两侧新生片速度均非负，最小值为 `0`。因此：

- `N3.1j3b6d8`：把印刷 Eq. 4.19 的 two-minus 标量式与文中有符号法向
  速度约定直接联用，判为 `falsified/frozen`；
- `N3.1j3b6d9`：从原始控制体式重新消元得到的有限角尖缘形成恒等式，
  仅在二维、高 Reynolds 数、零涡片厚度范围判为 `validated/frozen`。

这次 GO 只回答“单个有限角尖锐 junction 如何无参数地决定新生片方向、
强度和相对运动”。它没有把 NACA-2406 的上下两个尾缘角和实体 base
偷换成一个 junction，也没有建立 material potential/Kelvin 历史、三维展向
junction 或生产压力。因此 `N3.1j3b6d` 父节点继续保持 `partial`，
`production_activation_allowed=false`。
