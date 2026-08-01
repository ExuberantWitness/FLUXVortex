# N3 S3ai：物理可达 material-history 上的 pressure obstruction 门

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
状态：**PRE-REGISTERED / NOT EXECUTED / production off**

## ① 病因定位

S3ah 原计划对任意 previous-trace P2 扰动测 `rank(G_A)`。正式执行前已证明
该 rank 由 Morino Schur 满秩恒等锁定为 7，不能识别物理缺态。

当前真正未测的数据规律是：

> S3e 已能从物理 incidence history、旧 material Kelvin 记忆和 half/full
> actual solve 生成连续的相邻兼容状态，但没有观察这些状态是否同时满足
> unsteady pressure rate。

可动空间只增加只读的 \(R_P\) 与 direct-\(W\) 审计；S3e 的方程、历史、
规定对流、时间 stage 与拓扑全部冻结。

## ② 学科机理

Dumoulin–Eldredge–Chatelain 把无穿透、Kelvin 与 unsteady Kutta 作为同一
时间系统中不同约束，要求它们共享真实历史。Xia–Mohseni 的无质量
sharp-edge rate 由两侧速度、形成方向、片强度和相对速度联合决定，排除任意
previous-trace forcing。DeVoria–Mohseni 2019 则规定：若后续真的需要有限
forming zone，它必须有独立面质量、动量与 entrainment 守恒，不能由本门的
pressure residual 反定义。

## ③ 缺件还是错件

本门只裁决一个窄命题：

\[
D_P=0
\]

是否在 S3e 的 prescribed-convection 可达路径上对 incidence history 一阶
相切。

- 非零、收敛的 obstruction：只证伪这个具名 pressure law 在这条路径上的
  自动相切；下一步先查规定 forming geometry、pressure observation 和
  material transfer。
- 未见 obstruction：只表示这一方向没有一阶反例，不验证其他运动。
- 离散/时间/Kelvin 门不过：表示 canonical 还不能作物理裁决。

任何结果都不能直接推出七维状态、VES 或一个全局标量。

## ④ 冻结方案与 go/no-go

沿

\[
\alpha(t;\varepsilon)=\varepsilon\sin(\pi t/T)
\]

分别运行 \(+\varepsilon\) 与 \(-\varepsilon\) 的 S3e material-wake march。
每一半时刻从 actual compatible state 直接组装 \(W_{\rm direct}\)，并观察

\[
P_m=f_a\!\left[\tfrac12
\left(|w_u|^2-|w_l|^2\right)\right].
\]

每步未强加残差为

\[
R_{P,n}
=M_a(g_{n+1}-g_n)+\Delta t\,P_m,
\]

窗口残差为 \(\sum_nR_{P,n}\)。中心切向见证为

\[
\Omega_\varepsilon=
\frac{R_P(+\varepsilon)-R_P(-\varepsilon)}
{2\varepsilon}.
\]

同时冻结：

1. 所有 stage 的 direct BIE、\(g-C\phi\) 与 zero-tip；
2. old material rows、history seam、midpoint/current attachment；
3. 仓库符号 `Gamma_bound=-mu` 下逐步 telescoping Kelvin 账；
4. \(\varepsilon\) 二阶中心差分和 \(\Delta t\) 二阶 Cauchy；
5. 零攻角 quadrature floor，且不从结果中减去该 floor；
6. 非零见证必须至少是全部冻结数值不确定度的 100 倍。

正式 cases 已在任何执行前冻结于
`actual_wake_reachable_pressure_obstruction_cases_20260728_125228.yaml`。
本门没有 pressure root、force、LESP、状态变量、118 工况或
Fig17/18/19。

## 2026-07-28 13:03 执行前协议审计更正

v1 在任何正式执行前被独立审计否决，未生成结果 JSON。阻断包括：
direct-BIE 重解自证、Kelvin 望远镜恒等、body/direct-W 求积阶混用、
material attachment trace 未独立提取、对称零解“严格下降”误判，以及
pressure uncertainty 混入不同量纲的 BIE/compatibility/Kelvin residual。

因此本文件前述第④节只保留为已否决的 v1 设计历史，不构成可执行物理门。
完整审计与 v2 必改项见
`research_n3_actual_wake_reachable_pressure_obstruction_audit_20260728_130355.md`。
开放 claim 状态不变；未授权 forming state、VES、pressure force、production
或 Fig17/18/19。
