# N2.6e1bc-SVI-WUK-WPJ 最小 shadow 预登记

日期：2026-07-30  
Claim：`N2.6e1bc`  
状态：`SUPERSEDED / IMPLEMENTATION PAUSED / PRODUCTION FORBIDDEN`

> 执行前独立审计发现 Kelvin 势跃账、weak-UK 控制体推导、可观测状态、
> 有限角弱空间、残差映射和收敛轴尚未闭合。本文件保留为审计前快照，
> 不再授权 A1/A2 或 solver 实现；详见
> `n26e1bc_prereg_independent_audit_20260730.md`。其中
> `R_K=Gamma_b+sum(mu_w,k)-Gamma_0` 已明确判为错误，禁止执行。

## 1. 单一研究问题

二维 actual NACA0015 的 outer、双侧 IBL、单条 material TE wake 和
弱式 unsteady-Kutta 若在同一参考连通 DAE stage 联立，是否能在不消费
任何载荷目标的情况下同时得到：

1. 有界且网格/时间收敛的新生积分环量和形成方向；
2. 收敛的双侧实际翼面统一压力；
3. Kelvin、IBL 亏损、尾缘弱通量和横向动量的独立守恒闭合？

本门不验证分离、第二尾迹、Figure 12 或 RoboEagle。

## 2. 冻结方程角色

每个 stage 至少包含：

\[
R_\perp=(u_e-U_b)\cdot n
-\rho_e^{-1}\partial_s(\rho_e u_e\delta^*)=0,
\]

\[
R_M^\pm=0,\quad R_E^\pm=0,\quad R_\xi^\pm=0,
\]

\[
R_K=\Gamma_b+\sum_k\mu_{w,k}-\Gamma_{0}=0,
\]

\[
R_{UK}=\dot\Gamma_b+
\mathcal J_{\omega,TE}^{CV}
-\frac{p_L-p_U}{\rho}=0.
\]

`p_L-p_U` 必须来自同一收敛状态的非定常 Bernoulli，不是独立输入。
控制体涡量通量为

\[
\mathcal J_{\omega,TE}^{CV}
=\oint_{\partial A_{TE}}\omega
(u-v_{CV})\cdot n\,ds .
\]

旧 wake 满足

\[
\dot X_{w,k}=\bar u_{w,k},\qquad D\mu_{w,k}/Dt=0 .
\]

形成方向不靠 `R_UK=0` 的任意多根选取；它必须从零扰动一致初态沿规定
运动幅值 continuation，并以尾缘控制体横向动量

\[
R_\theta=e_\perp(\theta_{w,0})\cdot
\left[
\frac{d}{dt}\int_{A_{TE}}\rho u\,dA+
\oint_{\partial A_{TE}}
\{\rho u(u-v_{CV})\cdot n+pn-\tau\cdot n\}\,ds
\right]
\]

作不参与求根的独立 kill guard。

## 3. 冻结第一实现范围

- section：closed NACA0015，`c=1`；
- incompressible，actual two-sided surface；
- attached、single TE wake，禁止 separation wake；
- 双侧 IBL 状态为 `delta_star/theta/(n or sqrt_Ctau)`；
- 来源常数固定为 `ncrit=9` 和
  `sqrt(Ctau,tr)=0.7 sqrt(Ctau,eq)`；
- 零扰动一致初态开始，以小幅半余弦俯仰 continuation 到 `6 deg`；
- 不读取 Figure 12 或 RoboEagle；
- 旧 material jump 只由历史提供，新生 jump 只由当步联立系统产生；
- pressure 只在收敛后由总势历史、总速度和壁面速度计算一次；
- 当前实现首先使用二维 private solver；不得把 post-solver
  `ClaimComponent` 冒充强耦合。

实现按同一候选的三个必要子门推进，不允许并行尝试替代闭合：

1. `A0 equation/limit gate`：typed residual、量纲、符号、零亏损、
   `Delta p -> 0`、Kelvin 和材料输运制造解；
2. `A1 reference-branch gate`：单步及短窗 continuation、解析/有限差分
   Jacobian、唯一参考分支；
3. `A2 Cauchy gate`：共同缩小 `h/dt/CV` 后的状态、压力和横向动量。

任一子门失败立即终止该候选；不得在同一 claim 中追加新方程。

## 4. 预登记离散族

正式 A2 前冻结并只运行：

| level | panels/side | ramp steps | TE control-volume radius |
|---|---:|---:|---:|
| L0 | 32 | 16 | `2 h_TE` |
| L1 | 64 | 32 | `2 h_TE` |
| L2 | 128 | 64 | `2 h_TE` |

这里 `h_TE` 是该层最靠近尾缘的实际 panel arc length；系数 `2` 只定义
包含上下各一个完整 terminal panel 的最小离散控制体，不是可调物理长度。
若离散拓扑不能构造该闭合控制体，运行 fail closed。不得在看到结果后换成
固定弦长、最优 radius 或额外 epsilon。

所有层使用同一物理半余弦运动窗、同一来源闭合和同一 continuation 次序。
L0/L1 只作收敛序列，不允许因 L2 失败再追加 L3。

## 5. GO/NO-GO

### A0：方程与极限

- 所有量纲和符号制造解误差 `<=1e-12`；
- `R_perp/R_M/R_E/R_xi/R_K/R_UK` 缩放残差 `<=1e-9`；
- zero-deficit/zero-history 极限回到候选自己的 actual-surface inviscid
  baseline；
- 静态低频、`Delta p_TE -> 0` 极限恢复经典 pressure-Kutta；
- old material `mu_w` mutation 为零；
- 非法双 closure、目标字段、missing history 和非闭合 CV 必须 fail closed。

### A1：参考分支

- analytic/central-FD reduced Jacobian 相对误差 `<=1e-6`；
- mass-scaled rank deficiency 为零；
- 正负镜像误差 `<=1e-8`；
- 二分运动幅值的 tangent-predictor error 至少按二阶收缩；
- 从冻结参考初态只得到一个连续、有界分支；发现多根、换支或奇异
  Jacobian 即 NO-GO。

### A2：共同 Cauchy

L1 -> L2 必须全部满足：

- `Gamma_birth/(Uc)`、forming angle 和 wake first moment 变化 `<=2%`；
- 双侧 `Cp` 的 common-space relative L2 变化 `<=5%`；
- 周期均值 `L/T/M` 变化 `<=2%`，但这些量不参与任何参数选择；
- `R_theta` 的尺度化范数相对 L1 至少下降，且 L2 `<=1e-3`；
- panel-pressure 积分与独立控制体总力的相对差 `<=1%`；
- Kelvin、材料 jump、IBL 质量亏损和一次成力账全部满足各自代数门。

任一非有限量、运行异常、缺历史、未闭合控制体、多根、换支或任一门失败，
整体 verdict 为 `NO-GO`。

## 6. 晋升边界

全部通过只把 `N2.6e1bc` 晋升为二维 attached/single-wake
`validated/frozen`，并授权另行预登记：

1. `N2.6e1d` 的 `Cf=0` 分离、第二尾迹和 Figure 12；
2. 通过 Figure 12 后的 `N2.6e2` 目标条带表示。

它不直接授权 Fig17/18/19、三维横流、全翼生产、VES、结构改写或删除 V4.1。
若失败，`N2.6e1bc` 必须 `falsified/frozen`；后继只能回到新的学科机理，
不能放宽门、调控制体、增加 core 或用目标力选择分支。
