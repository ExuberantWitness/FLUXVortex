# N3 S3ab：topology-growth space-time birth 决策

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b1`  
状态：**EXECUTED / GO**

## ① 病因定位

S3x 已验证固定两带、固定 P2 身份下的 previous-time midpoint Cauchy；
S3aa 已验证在冻结 old/free state 时，newborn half/full body trace 必须逐
stage 联立求解。但把两者直接拼接会产生一个结构性矛盾：

- 插入前 actual topology 有 45 个 P2 DOFs；
- 插入后有 63 个 P2 DOFs；
- newborn 带在 `t_n` 的面积严格为零，所以不存在可逆的 63×63 初始质量
  矩阵；
- 直接调用固定拓扑 S3x 必须先为新增 18 个 DOFs 编造初值；
- copy、average、clamp 已被 S3u/S3aa 的反事实证据排除。

因此病灶不是 S3x 的时间常数、求积阶次或容差，而是 claim tree 中缺少
**topology birth boundary operator**。可动空间只限于：把 newborn 带看作
一个 characteristic space-time slab，其数据从 body/newborn coupled trace
沿入流边界进入；不得给零面积带补初值或做新旧网格插值。

## ② 学科机理

### 2.1 既有 evolving-surface transport 的适用边界

Dziuk 与 Elliott 的 surface transport theorem 对由既有 `M(0)` 经速度场
映射得到的 `M(t)` 成立，并明确区分物理 normal velocity 与任意 tangential
mesh velocity。它支持 S3x 的固定材料域 ALE 输运，却不能为 `t_n` 时不存在
的 newborn 区域提供初值。

来源：

- Dziuk & Elliott, *L²-estimates for the evolving surface finite element
  method*, Mathematics of Computation 82 (2013), 1–24,
  https://doi.org/10.1090/S0025-5718-2012-02601-9

### 2.2 拓扑变化必须作为 space-time slab

Olshanskii–Reusken 的 space-time weak formulation把整个 evolving surface
视为时空流形；Grande–Olshanskii–Reusken进一步展示该表示可自然跨越
topological change。相关 moving-domain space-time DG 文献指出：space-time
离散直接满足 geometric conservation law，并避免拓扑变化时的 conservative
interpolation 问题。

来源：

- Olshanskii & Reusken, *Error Analysis of a Space-Time Finite Element
  Method for Solving PDEs on Evolving Surfaces*, SIAM J. Numer. Anal. 52
  (2014), https://doi.org/10.1137/130936877
- Grande, Olshanskii & Reusken, *A space-time FEM for PDEs on evolving
  surfaces*, arXiv:1403.0277, https://arxiv.org/abs/1403.0277
- Luo, Absillis & Nourgaliev, *A moving discontinuous Galerkin finite
  element method with interface condition enforcement for compressible
  flows*, J. Comput. Phys. 445 (2021) 110618,
  https://doi.org/10.1016/j.jcp.2021.110618

### 2.3 尾迹出生数据来自释放 trace

Krebs 的 DDE 算法规定 newly-created wake strengths 在创建步由 surface/
newborn 联立问题赋值，之后保持 material identity。它与 S3aa 的 actual
body-wake trace 证据共同说明：newborn 的信息边界是释放时刻
`g(t)`，不是一个人为的二维初始场。

来源：

- Krebs, *A Distributed Doublet-Based Method for Unsteady Aerodynamic
  Analysis with Relaxed Wakes*, Carleton University, 2021, §2.2、§3.1–3.3
- Krebs, Bramesfeld & Cole, *A Distributed-Doublet Method for Unsteady
  Aerodynamic Analysis with Relaxed Wakes*, Aerospace 9 (2022) 28,
  https://doi.org/10.3390/aerospace9010028

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| S3x 固定拓扑 material transport | validated/frozen，适用域正确 |
| S3z weak-normal geometry | validated/frozen |
| S3aa coupled newborn stage trace | validated/frozen |
| 45→63 直接初值扩张 | 未定义，且会制造 18 个初值 |
| 新旧网格 remap | 机理不符，不允许作为修补 |
| characteristic space-time birth boundary | **缺件** |
| actual repeated insertion | 必须等待 birth operator 通过 |

## ④ S3ab 方案与预登记

先验证一个没有气动力、没有可调参数的最小 characteristic birth problem：

\[
Q_b=\{(t,x):0<t<\Delta t,\;0<x<c t\},\qquad
\partial_t\mu+c\,\partial_x\mu=0,
\]

其唯一数据为 inflow trace
\(\mu(t,0)=g(t)\)。取参考坐标
\(r=t/\Delta t,\;q=x/(c\Delta t)\)，则 `Q_b` 是三角形
`(0,0)-(1,0)-(1,1)`，characteristic coordinate 为
\(\tau=r-q\)。

采用一个 P2 space-time triangle：

- inflow edge 的 3 个 P2 DOFs 由 `g(0), g(1/2), g(1)` 显式给出；
- 其余 3 个 DOFs 由
  \(\int_{Q_b}N_i(\partial_r+\partial_q)\mu_h=0\) 求解；
- API 不允许传入 newborn initial state；
- endpoint 行必须恢复
  `[g(t_n), g(t_{n+1/2}), g(t_{n+1})]` 的物质时间身份。

GO 门：

1. 3×3 free block 满秩且 condition ≤ 10；
2. 常数、线性、二次 `g` 恢复误差 ≤ `2e-12`；
3. 全 P2 weak residual ≤ `2e-12`；
4. newborn mass identity
   \(\int_0^{c\Delta t}\mu(x,\Delta t)dx
   =c\int_0^{\Delta t}g(t)dt\) 闭合；
5. 对不同 `c, Δt` 结果保持尺度不变；
6. 至少三次连续 birth 后 chronology/seam 精确；
7. 光滑非多项式 trace 的 L2 时间 Cauchy contraction ≥ 6；
8. 任何 initial newborn scalar、epsilon、filter、remap、pressure、force、
   LESP、structure 均不存在。

GO 只验证“出生边界可替代零面积初值”这个组成命题。它不会直接晋升 actual
repeated-insertion，也不会授权 unified pressure 或 production。

## 执行结果

预登记门原样执行，`10/10 GO`：

| 指标 | 结果 |
|---|---:|
| prescribed / solved / artificial initial DOFs | 3 / 3 / 0 |
| free rank deficiency / condition | 0 / 5.0986 |
| P0/P1/P2 recovery error | 8.88e-16 |
| all-test weak residual | 1.94e-16 |
| endpoint identity / mass balance | 2.66e-15 / 5.55e-17 |
| positive `dt,c` scale invariance | 0 |
| repeated step counts | 3 / 6 / 12 / 24 |
| chronological seam | 2.66e-15 |
| exact L2 errors | 3.56e-5 / 4.48e-6 / 5.60e-7 / 7.01e-8 |
| Cauchy contractions | 7.958 / 7.990 |
| input mutation | 0 |

结果证明：零面积 newborn 区域不需要、也不应拥有二维初始状态。P2 inflow
edge 的三个 release-trace 值构成完整数据，另外三个 space-time P2 DOFs 由
characteristic weak equation 唯一确定；连续 birth 在没有 remap、epsilon、
filter 或载荷目标的情况下保持质量与三阶 L2 时间行为。

## Claim 裁决

- `N3.1j3b6d18c2b3b3b2c2b1`：validated/frozen；
- `N3.1j3b6d18c2b3b3b2c2b`：保持 partial；
- 下一可动空间仅为 `...c2b2`：把该 birth boundary 与 actual 45-DOF
  old-state transport、S3aa body algebraic trace组成同一 repeated-insertion
  slab；
- unified pressure、force、LESP 与 production 继续禁止。
