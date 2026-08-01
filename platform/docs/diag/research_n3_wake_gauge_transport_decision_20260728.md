# N3 S3i：自由尾迹的切向规范与材料势跳输运

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b`  
状态：**EXECUTED / GO**

## ① 病因定位

S3h 证明给定一个完整速度 provider 时，free wake 顶点、最新 body edge、
chronological seams 和材料强度可以按 Heun 语义一致推进。但接入实际 DDE
速度不能直接复用该结论：

- `N3.1j4c1` 已证伪把全部 sheet-average 三维速度强制投影成连续 P1
  顶点速度；
- `N3.1j4c5/c6` 只验证顶点星**法向几何速度**；
- `MaterialWakeBand.material_update()` 在任何几何更新中原样冻结
  `potential_jump_rows`。

若网格仅按法向速度运动，它不是流体材料网格。对单位方形平面涡片取
`u_t=[0.3 sin(pi s), -0.2 sin(pi r)]`、非均匀 P2 势跳和 `T=0.2`，
边界切向速度为零，几何集合完全不变；但“法向几何＋冻结 μ”相对精确材料
场的最大误差已达 `0.0567519`（场程约5.69%）。因此异常不来自 attachment、
时间步、涡核或强度方程。

## ② 学科机理

Ambrose–Masmoudi 的三维曲面运动式把速度拆为法向几何速度和两个可选择的
切向参数化速度；切向分量只重参数化曲面。Ambrose 的 Birkhoff–Rott 工作
进一步区分连续法向速度与有跳跃的单侧切向速度。

JFM 的自由涡片方程则明确把：

- sheet-average/Birkhoff–Rott 速度作为 Lagrangian sheet 速度；
- 未归一化 circulation 作为材料变量；
- `D_t gamma=0` 作为 Kelvin 材料守恒。

两者合并意味着：若计算网格速度 `w_mesh` 不等于材料平均速度 `u_bar`，
材料标量不能继续按节点冻结，而必须满足

```text
∂μ/∂t|mesh + (u_bar_t - w_mesh_t) · ∇_s μ = 0 .
```

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| N3.1j4c5/c6 法向几何速度 | 已验证，冻结 |
| full-vector continuous-P1 material velocity | 已证伪，禁止复活 |
| normal-only mesh 上冻结 P2 μ | 错件；解析反例 |
| tangential-gauge ALE scalar transport | 缺件 |
| actual body/source/wake velocity求值 | 后续缺件，本门不混入 |
| pressure/force/结构响应 | 禁止进入本门 |

## ④ 方案与预登记

先验证 continuum identity，不先选择离散格式。规范平面涡片使用有闭式逆
映射的二维切向速度、非均匀 P2 `μ0`、Eulerian/0.37/Lagrangian 三个切向
规范和一个通用三维刚体变换。冻结门包括：

- 材料轨迹 `D_t μ=0`；
- 三种规范的 ALE residual 与同一物理标量场；
- 刚体客观性；
- 边界速度为零；
- naive frozen-μ 误差必须至少 `0.05`。

完整配置在 `wake_tangential_gauge_transport_cases.yaml`。GO 只会证伪
`c2b1` 并验证 `c2b2` 的连续方程身份；离散 P2 输运、实际诱导速度、
post-relaxation equilibrium、压力和力仍需独立预登记。

## ⑤ 执行结果与 claim 裁决

冻结配置未改，所有门通过：

| 指标 | 结果 | 冻结门 |
|---|---:|---:|
| material-trajectory μ error | 4.441e-16 | ≤5e-14 |
| max ALE residual（三规范） | 1.665e-16 | ≤5e-14 |
| cross-gauge scalar error | 4.441e-16 | ≤5e-14 |
| rigid-frame residual | 3.036e-16 | ≤5e-14 |
| boundary velocity | 3.674e-17 | ≤5e-14 |
| naive frozen-μ error | 0.0567519 | ≥0.05 |

因此：

- `N3.1j3b6d18c2b1` falsified/frozen：normal-only geometry 不能同时把
  P2 节点系数当材料不变量；
- `N3.1j3b6d18c2b2` validated/frozen：只冻结 continuum ALE identity；
- `N3.1j3b6d18c2b3` 保持 open。

本门没有选择空间离散、时间积分、稳定化或限制器，也没有接实际诱导速度。
下一门只能验证离散 P2 transport 的一致性、Kelvin 账、规范不变性和收敛。
