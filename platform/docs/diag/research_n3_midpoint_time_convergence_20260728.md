# N3 S3e：显式半时刻物质尾迹的时间收敛预登记

日期：2026-07-28  
Claim：`N3.1j3b6d18b`  
当前状态：**EXECUTED / GO（窄门）**  
运行角色：无压力、无力的时间离散与 Kelvin-history oracle。

## ① 病因定位

S3d 已经证明如下 affine 分区成立：

```text
current body + newest-current wake row = source RHS - known old wake
```

但它把 newest band 的 `previous/middle` 两行当成外部输入，因此没有回答
`middle` 从哪里来。若直接用 `(mu_n+mu_n+1)/2`，只能得到一个插值值，不能证明
半时刻 actual-boundary、旧 wake 诱导和当前无穿透同时成立。

病因挂在 `N3.1j3b6d18`，可动空间仅为时间 stage、band 出生顺序和规定的均匀
材料对流。S3a–d 已冻结的 body/cut topology、P2 空间方程、符号、求积、N1
以及所有 pressure/force 路径都不可动。

## ② 学科机理

Krebs 2021 §3.2–3.3 与 Fig.3.8 给出的核心不是“对端点取平均”，而是：

1. 每一时间步生成一排新 wake elements；
2. surface 与 newly-created wake 的强度互相影响，必须在同一无穿透/Kutta
   系统中求解；
3. 已脱落 material strength 后续保持，不应被当前解覆盖；
4. 单次 surface/wake 更新可能留下 flow-tangency 误差并造成 timestep
   dependence，原 DDE 路线在 wake relaxation 前后分别迭代。

Krebs–Bramesfeld–Cole（*Aerospace* 9, 2022, 28）再次明确这一点。
Dumoulin–Eldredge–Chatelain（*JFM* 977, 2023, A22）则从 unsteady
panel/vortex-sheet 角度表明 bound impermeability、尾缘 shedding 和 wake
circulation transfer 必须保留统一时间索引。

这些来源支持“半时刻也必须解耦合方程”，但并没有证明**一次显式 midpoint**
已经足够。因此 S3e 是可证伪候选，而不是文献背书后的既定方案。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| old material rows 作为 known RHS | S3d 已验证，冻结 |
| full-band middle row 由端点平均产生 | 错件，缺少半时刻边界方程 |
| 半时刻用同一 coupled equation 生成 `mu_mid` | 当前待验证的缺件候选 |
| 一次 half-stage 足以消除 timestep dependence | 未知，必须由 Cauchy 门裁决 |
| 在时间门前计算 `dmu/dt` pressure | 禁止 |

## ④ 冻结方案与 go/no-go

固定 `alpha(t)=5° sin(pi t/T)`、`T=1`、`U=1`，运行
`dt={0.5,0.25,0.125}`。每个 full step：

1. 旧 material bands 只对流几何，三行 P2 strength bitwise 不变；
2. 在 `t_n+dt/2` 建一个半宽 active band，用同一 coupled equation 求
   `mu_mid`；
3. 在 `t_n+dt` 建完整 active band，其三行严格为
   `[mu_n, mu_mid, mu_n+1]`；
4. 把完整 band 追加到 chronological history。

判据同时覆盖：

- 半时刻来源、非端点平均、旧状态不变、全部 interface/attachment；
- rank、弱残差、条件数、tip identity 和 band shape regularity；
- 固定终止时间的 body-cut jump Cauchy；
- 同一组离面物理探针的 wake-induced velocity Cauchy。

预登记阈值冻结在
`actual_boundary_midpoint_time_convergence_cases.yaml`。若任一门失败，
`N3.1j3b6d18b` 的“一次显式 midpoint 足够”即 NO-GO，下一方向必须改为
Krebs 型步内 coupled equilibrium/iteration；不得调常数、改空间算子、放宽
阈值或用压力/总力救活。

即使 GO，也只验证固定几何、均匀对流下的 material-history 时间离散；不授权
自由尾迹、LEV、压力、力或生产路径。

## 执行结果：GO，但不宣称统一二阶

预登记后运行了 3 个时间步族，共 28 个 half/full actual-boundary–wake
联立系统。全部离散身份门通过：

- old material strength、规定的 old-geometry convection、history 的
  time/geometry/trace interface、midpoint row identity、current attachment
  和 tip jump 的最大误差均为 `0`；
- old wake 独立未知量为 `0`，有旧历史的最小 known-RHS 范数为
  `4.96e-5`；
- rank deficiency 为 `0`，最大条件数 `110.82`，最大归一弱残差
  `3.84e-16`；
- 最大 band 长宽比为 `1`；
- `mu_mid` 与 `(mu_n+mu_n+1)/2` 的最大差为 `0.01740`，排除“端点平均
  冒充半时刻解”。

固定 `T=1` 的双 Cauchy 结果：

| 观测量 | coarse→medium | medium→fine | 收缩比 | 最细相对变化 |
|---|---:|---:|---:|---:|
| body cut jump | `3.852e-3` | `6.800e-4` | `5.665` | `1.530%` |
| 离面 wake probe velocity | `1.530e-3` | `8.657e-4` | `1.767` | `4.616%` |

因此 `N3.1j3b6d18b` 仅在 **fixed actual boundary＋prescribed uniform
convection＋显式半时刻 material history** 范围 `validated/frozen`。
离面场收缩比没有资格支持“统一二阶”表述。

下一节点不是 pressure，而是 `N3.1j3b6d18c`：moving/deforming body 与
wake relaxation 后，按 Krebs Fig.3.8 在每个 stage 内求
surface/newborn-wake coupled equilibrium，并重新检查 timestep
independence。通过前继续禁止 `dmu/dt` pressure、LEV、force 和 production。
