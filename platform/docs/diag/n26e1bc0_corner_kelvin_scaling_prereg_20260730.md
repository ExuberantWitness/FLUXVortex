# N2.6e1bc0 regular-corner / Kelvin 时间阶判别预登记

日期：2026-07-30  
Claim：`N2.6e1bc0-FCR-KELVIN-SCALING`  
状态：`PREREGISTERED / SHADOW DIAGNOSTIC AUTHORIZED / PRODUCTION OFF`

## 1. 单一问题

只检验：

> 消去有限角首奇异模态后的 regular outer corner mode，是否能同时提供
> 材料新生 wake 的有限形成速度、正则片强和 generic Kelvin 出生环量率？

本门不实现 strong IBL、weak-CV、pressure、force、Figure 12 或
Fig17/18/19；不修改 V4.1。若这个必要缩放门失败，禁止先实现更大的
regular-corner-only solver。

## 2. 已冻结的病因与空间证据

closed NACA0015 的实体尾缘角、外流角和 regular 速度幂次为

\[
\tau=20.595196771981^\circ,\qquad
\Omega=2\pi-\tau,
\]

\[
\lambda_1=\frac{\pi}{\Omega},\qquad
\lambda_2=\frac{2\pi}{\Omega}=1+\beta,\qquad
\beta=0.060680333855 .
\]

`\lambda_1` 是必须由 Kutta/正则性消去的首奇异模态；下一模态满足

\[
u^\pm(r)=A^\pm r^\beta+o(r^\beta).
\]

既有 `64/128/256 panels/side` 数据中，`u/r^\beta` 的 mean 系数为
`9.9814/10.0217/10.0750`，末级变化 `0.53%`；upper/lower 为
`0.35%/1.32%`。这些数据只证明 modal coordinate 比 raw point trace
稳定，不证明它能闭合出生。

## 3. 预登记缩放反例

若两侧 regular mode 经 Xia 型无质量形成关系给出

\[
\gamma_g(r)=G_Xr^\beta,\qquad
u_g(r)=V_Xr^\beta ,
\]

并显式选择 `dr/dt=V_X r^\beta, r(0)=0` 的“立即出射”分支，则

\[
\ell(\Delta t)
=[(1-\beta)V_X\Delta t]^{1/(1-\beta)}
\sim\Delta t^{1.0646003},
\]

\[
\Gamma_X
=\int_0^\ell G_Xr^\beta\,dr
\sim\Delta t^{p_*},\qquad
p_*=\frac{1+\beta}{1-\beta}=1.1292006 .
\]

因此 regular-only 预言 `Gamma_X/dt -> 0`。另一方面，在
`\dot Gamma_b(t*) != 0` 的光滑 generic stage，

\[
\Gamma_K=-\Delta\Gamma_b
=-\dot\Gamma_b(t^*)\Delta t+O(\Delta t^2),
\]

应有时间阶 `p_K=1`。若 Kelvin 强制 `Gamma_K=O(dt)` 而
`\ell=O(dt^{1/(1-beta)})`，则正则 wake 系数

\[
G_K=\frac{(1+\beta)\Gamma_K}{\ell^{1+\beta}}
\sim\Delta t^{-0.1292006}
\]

发散，与 `G_X=O(1)` 冲突。

此外，`0<beta<1` 时 `dr/dt=V r^beta, r(0)=0` 不满足唯一性条件；
除立即出射外还存在任意 waiting-time 分支。因此“立即出射”是本门明确
测试的候选分支，不得称为由 regular mode 自动得到的唯一解。

## 4. 冻结工况与独立轴

- closed NACA0015，`c=1 m`；
- `U=9 m/s`；
- quarter-chord pivot；
- 半余弦俯仰
  \[
  \alpha(t)=3^\circ[1-\cos(\pi t/0.4)];
  \]
- generic 观察相位 `t*=0.2 s`，此处 pitch rate 非零；
- zero IBL/transpiration，single TE wake；
- material wake core 保持既有 shadow 身份 `0.02c`，只用于旧 wake
  对流；不得扫描或解释为物理尺度；
- 不读取任何 pressure、force 或目标响应。

只运行：

### 空间轴

`panels/side = {64,128,256}`，固定 `dt=0.00625 s`
（64 ramp steps），观察 `t*=0.2 s` 的：

- upper/lower/mean `u/r^beta`；
- terminal radius；
- `Gamma_K=-Delta Gamma_b`；
- Kelvin、normal BC、Eq.7/Eq.8 和线性系统残差。

### 时间轴

固定 `256 panels/side`，运行

```text
dt = {0.025, 0.0125, 0.00625, 0.003125} s
ramp_steps = {16, 32, 64, 128}
```

所有层在同一 `t*=0.2 s` 取该步实际
`abs(Gamma_birth)`。拟合唯一采用四点普通最小二乘
`log(abs(Gamma_birth))` 对 `log(dt)`；同时逐对报告 local order，
但不得据此改换拟合窗口。

共同锚点为 `N=256, dt=0.003125 s`。空间和时间轴不得耦合加密。

## 5. 健康门

- 所有预登记层级完成，分支身份一致且量有限；
- `t*=0.2 s` 必须逐位落在时间网格上；
- `abs(Gamma_birth)/(Uc) > 1e-10`，排除零率特殊时刻；
- 最大 Kelvin、normal BC、Eq.7、Eq.8 和 scaled linear residual
  均 `<=1e-9`；
- modal coefficient 的定义只使用几何给定 `beta` 和 terminal midpoint
  radius；禁止拟合 beta、epsilon、core 或采样半径。

健康门失败为 `PROTOCOL/IMPLEMENTATION-NO-GO`，物理子命题保持 `open`。

## 6. 物理 GO/NO-GO

regular-corner-only 必要门为：

1. upper/lower/mean modal coefficient 的 `128 -> 256` 变化均 `<=2%`；
2. 四点时间阶满足 `abs(p_K-p_*) <= 0.03`；
3. 最细两个时间层的 local order 不远离 `p_*` 超过 `0.05`。

全部满足只记为 `REPRESENTATION/SCALING GO`，授权另行预登记 enriched
body--wake compatibility；不晋升 `N2.6e1bc`。

若健康门通过但 `p_K≈1`、与 `p_*` 不符：

`PHYSICS-NO-GO / N2.6e1bc0 regular-corner-only falsified/frozen`。

该结果只证伪“有界 regular outer mode 足以闭合 generic birth”。它不证伪
strong VI、材料势跃或完整 moving-interface circulation theory。下一候选
必须显式拥有有来源的 viscous inner/profile state、受控 singular amplitude
或有限 forming-zone inventory；禁止退回 endpoint gamma、epsilon、
最近 control point、`Gamma_birth/dt` weak-UK 回填或目标载荷拟合。

## 7. 结果状态机

- schema/history/matrix/nonfinite 失败：
  `PROTOCOL-NO-GO`，claim 保持 open；
- 数值健康全部通过且缩放反例成立：
  `PHYSICS-NO-GO`，只冻结 `N2.6e1bc0`；
- 缩放必要门通过：
  `REPRESENTATION-GO`，父节点仍 open。

不允许把协议失败写成物理证伪，也不允许在看到结果后改变相位、时间层、
拟合窗口或阈值。
