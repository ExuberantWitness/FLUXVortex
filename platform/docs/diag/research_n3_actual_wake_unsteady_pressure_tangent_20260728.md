# N3 S3ah：相容参考态上的非定常压力闭合切向缺陷门

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3e3`  
执行状态：**ABORTED BEFORE FORMAL EXECUTION / CIRCULAR-RANK-NO-GO /
production off**

> 2026-07-28 12:44:37 的独立公式审计证明：在本预登记已经要求
> \(B\)、Morino block、\(J_P\)、\(M_a\) 与 \(Q\) 满秩/正定的前提下，
> `rank(G_A)=7` 是 Schur 补恒等结果，而不是可发现的物理缺态证据。
> 因此本门未正式执行，原冻结预登记保留但不得用于 claim 晋升。完整审计见
> `research_n3_actual_wake_unsteady_pressure_tangent_audit_20260728_124437.md`。

## ① 病因定位

S3ag 已证明“小 closure residual + 完整 body BIE”不蕴含物理 junction：

- prescribed birth 的完整 BIE 与通量残差达到机器精度，但
  \(A=g-C\phi\) 在四个时间尺度保持 `0.0133–0.0160`；
- 两种 steady pressure observation 都收敛，却得到不同的大 \(g\) 与大
  \(A\) 分支。

但 S3ag 只用了一个 \(1-y^2\) 展向 envelope。它只能证明缺陷至少有一维，
不能证明缺失状态必须有七维，也没有排除更窄的零新增态假设：

> 当 closure 由同一 actual body side velocity、同一历史和一致参考态产生时，
> 它的局部分支可能自动保持 \(g-C\phi=0\)。

当前 unknown 为 \((\phi_n,g_r)\)，规范算例 \(n=81,r=7\)。完整 BIE
\(n\) 条和 compatibility \(r\) 条已经组成方阵。把 birth/pressure 动力学再
作为独立 \(r\) 条加入会多出 \(r\) 条方程。显式写出
\(\theta_g,\gamma_g,u_g\) 也不能自动解决计数：若它们各自由 Xia 守恒式决定，
新增未知与新增方程相同，净自由度仍为零。

因此本门不添加状态。它先测量一个明确 closure 在七个展向历史方向上的
compatibility tangent rank。

## ② 学科机理

Xia–Mohseni 对 finite-angle sharp junction 联立 unsteady Kutta、环量、质量
与动量，决定 forming sheet 的方向、强度和相对速度。该结果说明这些量必须
联合求解，但不把可代数消去的形成量变成一个独立材料库存。

Dumoulin–Eldredge–Chatelain 的 panel model 同时保留 no-through-flow、
Kelvin 与 unsteady Kutta，且将 shed-panel strength 作为明确未知。它支持
“不同物理约束必须在同一系统中计数”，不支持用一个 residual 小值代替
compatibility。

Leroy–Devinant 与 Morino–Bernardini 说明 glued body/wake sheet 的 potential
jump 与边缘正则性必须相容；所以 \(A=g-C\phi\) 是观察量，不是可调 penalty。

DeVoria–Mohseni 的 vortex-entrainment sheet（VES）进一步指出，把有限厚度
黏性层压成零厚面时，切向涡量以外还可能需要 entrainment、面质量与动量状态；
非切向 sharp-edge shedding 尤其不能由一个裸 vortex-sheet scalar 自动闭合。
这只为“若零态失败，下一步去哪里找物理状态”提供方向，尚不授权 VES 或任意
P2 slack。

## ③ 缺件还是错件

本门同时审三个命题：

| 命题 | 本门判据 |
|---|---|
| actual-coupled pressure closure 自动保持 compatibility | \(G_A\) rank 0 才保留 |
| 一个全局形成标量足够 | \(G_A\) rank 大于 1 即结构性证伪 |
| 七维 P2 forming inventory 已获物理授权 | 本门不能授权；即使 rank 7 仍需 finite-forming-zone/VES 守恒推导 |

禁止把 \(q\)、\(\theta_g\)、\(\gamma_g\)、\(u_g\) 仅换名为“新状态”。若可由
当前代数关系消去，它们没有新增记忆或库存。也禁止直接令
\(\zeta=g-C\phi\)；那只是把 residual 改名。

## ④ 方案与 go/no-go

选择唯一候选：

\[
B\phi+W_{\rm direct}g=b,
\]

\[
F_P =
M_a(C\phi-c^-)
+\Delta t\,f_a\!\left[
\tfrac12\left(|w_u|^2-|w_l|^2\right)
\right]=0.
\]

这里 \(W_{\rm direct}\) 从 prescribed actual material-wake geometry 直接组装；
\(f_a\) 是 weak active-P2 line observation；没有 collocation、pressure force
或载荷目标。

从 Morino-compatible current state \(g_0=C\phi_0\) 出发，选择

\[
c^-_0=c_0+\Delta t M_a^{-1}P_0
\]

使离散 reference closure 精确为零。该式只定义一致的历史参考态；它不是生产
pressure offset。规范零攻角下 \(P_0\) 必须随 boundary quadrature 收敛到零。

取

\[
Q=L_M^{-\mathsf T},\qquad Q^\mathsf T M_a Q=I,
\]

以七个质量正交 P2 模态扰动 \(c^-\)。令
\(\Phi_g=-B^{-1}W_{\rm direct}\)，则

\[
J_P=M_a C\Phi_g+\Delta t\,P_g ,
\]

\[
Dg=J_P^{-1}M_aQ,
\]

\[
G_A=L_M^\mathsf T(I-C\Phi_g)Dg .
\]

正式执行同时检查：

1. q5/q8/q10/q12 的 reference pressure residual 收敛；
2. direct \(W\) 与冻结消元块一致、rank 7；
3. \(J_P\) 满秩，解析导数与中心差分一致；
4. 七模态正负 continuation 的 tangent predictor 误差为二阶；
5. \(G_A\) 在质量度量下的 rank、最小奇异值与换基协变；
6. 单一全局标量 counterfactual 的最大 rank 为 1。

若 `rank(G_A)=0`，零新增态候选继续；若 rank 大于 1，全局标量 NO-GO；若
rank 为 7，只能裁决“分布式状态在代数维数上最低需要七维”。下一步必须从
finite forming-zone/VES 的质量、环量和动量守恒推导状态、记忆及
\(V_f/H_f/M_f\) coupling，不能先实现自由 \(\zeta\)。

完整阈值已在
`actual_wake_unsteady_pressure_tangent_cases_20260728_123758.yaml`
于正式执行前冻结。

本门无力、无 LESP 幅值、无 N1/N4 改动、无 production、无 118 工况，也不生成
Fig17/18/19。

## 正式执行前审计裁决

令

\[
H=I-C\Phi_g=I+CB^{-1}W.
\]

S3af 已冻结 \(B\) 与 Morino block
\(\left[\begin{smallmatrix}B&W\\-C&I\end{smallmatrix}\right]\) 满秩，
所以 \(H\) 必为满秩 7。若本门再要求 \(J_P\) 满秩、\(M_a\) 正定且
\(Q=L_M^{-\mathsf T}\)，则

\[
\operatorname{rank}
\left[
L_M^\mathsf T HJ_P^{-1}M_aQ
\right]
=\operatorname{rank}(H)=7.
\]

故原 `rank0/rank7` 分支不是开放实验，`rank7` 也不能授权七维物理状态。
此外，制造的 \(c^-_0\) 只在代数上 closure-consistent，尚未证明是
BIE/compatibility/Kelvin/material-transport 可达历史。

下一门改为：先沿物理可达的兼容状态路径生成相邻时刻，再测未强加的
pressure/birth 动力残差及其在可达切空间上的 transversality/cokernel。
在此之前禁止实现 \(\zeta\)、VES 或任何 residual slack。

## 一手来源

- Xia & Mohseni, *Journal of Fluid Mechanics* 830 (2017),
  DOI `10.1017/jfm.2017.513`.
- Dumoulin, Eldredge & Chatelain, *Journal of Fluid Mechanics* 977 (2023),
  DOI `10.1017/jfm.2023.997`.
- DeVoria & Mohseni, *Journal of Fluid Mechanics* 866 (2019),
  DOI `10.1017/jfm.2019.134`.
- Leroy & Devinant, *International Journal for Numerical Methods in Fluids*
  29(1) (1999), DOI
  `10.1002/(SICI)1097-0363(19990115)29:1<75::AID-FLD773>3.0.CO;2-7`.
- Morino & Bernardini, *Finite Elements in Analysis and Design* 38 (2001),
  DOI `10.1016/S0955-7997(01)00063-7`.
