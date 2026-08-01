# N3 S3j：P2 材料势跳的离散 ALE 输运

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b2a`  
状态：**EXECUTED / GO**

## ① 病因定位

S3i 已证明 continuum ALE identity，并以 `0.0567519` 的解析反例证伪
“法向几何＋冻结 P2 系数”。但当前 runtime 只有：

- P2 face-local 势跳与共边连续性检查；
- 法向几何速度和 Heun 推进；
- 没有相对于 mesh 的材料标量输运方程。

因此不能把 S3i 的连续方程直接当成离散模型已闭合。

## ② 学科机理

Elliott–Venkataraman 的 ALE evolving-surface FEM 和 Dziuk–Elliott 的
surface-PDE 变分框架都表明：当 mesh velocity 与 material velocity 不同时，
basis transport 会产生显式 relative-velocity advection term；在 Lagrangian
极限该项消失。

这里的输运量是 free-wake potential jump `μ`，它是材料标量，不是面积密度。
所以采用非守恒强式的 Galerkin 投影：

```text
M_ij = ∫ N_i N_j dS
C_ij = ∫ N_i c_t·∇_s N_j dS
M μ_dot + C μ = 0
```

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| S3i continuum identity | 已验证，冻结 |
| global shared P2 scalar topology | 缺件 |
| consistent mass/advection operator | 缺件 |
| mass lumping/upwind/artificial diffusion | 无机理授权，本门禁止 |
| actual induced velocity/pressure/force | 后续，本门禁止 |

## ④ 预登记

采用单位方形 P1 三角几何、全局连续 P2 标量、`3/6/12` cell families 和 S3i
同一闭式切向材料流。用半离散矩阵指数消除时间积分误差，只裁决空间组件。

GO 必须同时满足：mass 满秩、常量零率、共边零跳、L2 单调收敛与 Cauchy 门、
刚体客观性、q10/q12 求积一致性。阈值已冻结在
`p2_wake_gauge_transport_cases.yaml`；失败后禁止加稳定化或改工况补救。

## ⑤ 执行结果与 claim 裁决

预登记配置未改：

- `3/6/12` cell families 的 relative L2 error 为
  `3.43359e-4 / 5.11419e-5 / 9.55532e-6`；
- 两级误差收缩为 `6.714 / 5.352`；
- mass rank deficiency 为0，最大条件数 `22.39`；
- constant-rate residual `1.79e-14`，shared trace jump 为0；
- rigid mass/advection matrix 误差最大 `2.26e-17`，终态标量误差
  `2.22e-15`；
- q10/q12 终态差 `4.44e-15`。

故 `N3.1j3b6d18c2b2a` 只在 stationary planar、continuous-P2、
semidiscrete spatial operator 范围 `validated/frozen`。没有使用 mass
lumping、upwind、人工扩散或 limiter。moving/curved geometry、history
seams 和 actual induced velocity 仍未验证。
