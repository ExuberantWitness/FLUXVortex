# N3 实际边界 lifting-potential 重表示裁决

日期：2026-07-28  
Claim：`N3.1j3b6c → N3.1j3b6d1-d3`  
范围：气动载荷模型；N1 冻结；不做结构模型。

## ① 病因定位

### 原始薄格场不是实际厚体外势

代表 `v41 U8/aoa15/f2.6/tw11.25` 的闭壳通量由 bound 通道主导：

| 通道 | 通量 |
|---|---:|
| freestream | `-3.47e-17` |
| bound direct | `+0.0610303` |
| bound image | `-0.0312306` |
| wake direct | `-7.06e-5` |
| production total | `+0.0297291` |

对共置线段先按有向环量消去后，仍有 17 条 production 活动涡丝穿壳。
其中 direct 对有限 trailing base 的 9 次命中恰为 `ns+1` 条末排弦向涡丝；
这不是半翼 root cap 独有伪影。158→1246 面加密也没有使 bound 通量单调
趋零。因此 `N3.1j3b6c` 的错误是**奇异支撑/边界表示错件**。

### S0 把“势方程”和“物面压力”拆开

预登记后，在同一实际闭边界上实现常元 Morino 方程：

\[
\sigma=(\boldsymbol u_{inc}-\boldsymbol u_{wall})\cdot\boldsymbol n,
\qquad
D_{inside}\mu+S\sigma=0 .
\]

单位球 20/80/320 面结果：

- 内部势残差：最细 `1.13e-15`；
- exterior surface identity 残差：`1.24e-15`；
- prescribed source flux：`5.38e-18`；
- 离体解析势误差：`0.3494→0.1105→0.02932`；
- 条件数：`2.34→2.08`。

所以 Green 符号、source 身份和实际边界势方程通过。但把逐面常 doublet
直接当周界 ring 在物面普通求速度，最细：

- 法向残差 `0.03778`，未过 `0.03`；
- `Cp` RMS `0.56681`，未过 `0.08`。

阈值不放宽。数据指纹说明缺口在**连续势跳及片上有限部/表面梯度**，不是
势方程或待调常数。

## ② 学科机理

Erickson, NASA TP-2995 指出：

1. 常 doublet 与 ring vortex 的等价只对同一面板周界成立；
2. 连续 doublet 分布消除面板边界出现的伪线涡；
3. 高阶 source/doublet 才能稳定支持实际表面模型。

Dusto & Epton, NASA CR-152323 把非定常 source/doublet 放在实际厚体表面，
并明确尾缘邻域需要对基本 Morino 边界处理作专门修正。

Le Provost 等的 JFM 2023 低阶涡模型把物面 bound sheet、无穿透、Kelvin
守恒和 unsteady Kutta/新生片强度放入同一方程组；片上速度是主值边界算子，
不是从另一套穿体涡格直接采样。

一手来源：

- NASA TP-2995:
  https://ntrs.nasa.gov/citations/19910009745
- Dusto & Epton, NASA CR-152323:
  https://ntrs.nasa.gov/citations/19800017771
- Le Provost et al., JFM 2023:
  https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/lightweight-vortex-model-for-unsteady-motion-of-airfoils/4AA09A7BBD05F868544E27D69410B944
- Kemp, NASA TM-80104:
  https://ntrs.nasa.gov/citations/19790016785

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 原始 N1 ring 场 + 单值 source 可成为实际厚体外势 | 错件，`N3.1j3b6c` falsified/frozen |
| 实际边界 interior-Dirichlet source/doublet 势方程 | 正确地基，`N3.1j3b6d1` validated/frozen |
| 常元周界 ring 普通值可直接形成物面 Cp | 错件，`N3.1j3b6d2` falsified/frozen |
| 连续 P2 actual-boundary doublet + Galerkin + 片上有限部/梯度 | 缺件，`N3.1j3b6d3` partial |
| 调 core/offset/source 容差 | 禁止；会掩盖支撑和边界算子错误 |

N1 仍可作为环量约束候选或初值，但不能再作为实际壳入射速度的原始奇异场。
若 actual-boundary 方程与规定 N1 环量冲突，应触发 N1 freeze review。

## ④ 机理方案

下一生产候选不是“更密的常元”，而是：

1. 在实际 NACA-2406 上/下/base/tip 闭边界定义连续 P2 doublet/势跳；
2. 用同阶 trial/test 构造 actual-boundary Galerkin 内部 Dirichlet 方程；
3. source 由相对壁面法向速度规定，不作为调参未知量；
4. 内部边 trace 连续；只有已分类 TE/base/LEV 分离界面可暴露涡片；
5. 把 N1 环量作为独立约束候选，分别报告 Kelvin、边界和 Kutta 冲突；
6. 用 P2 片上有限部速度或势的表面梯度，不用常元 ring 普通值；
7. 与 TE/base wake 联立后保存三时刻 material potential；
8. 所有速度和势率先合并，再执行一次 Bernoulli 和一次压力积分。

S0 的 `equation_oracle_gate=GO` 不授权压力；总 stage 为 NO-GO。连续 P2、
带环量 canonical、full-wing symmetry、finite-base wake 和 material-pressure
history 未通过前，V4.1 生产路径保持不变。
