# N3 S3t：owner-aware actual-wake geometry/P2 stage velocity

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b2b`  
状态：**EXECUTED / GO**

## ① 病因定位

S3s 已把实际 material history 变成 typed open-boundary P1/P2 topology，
但仍不能直接把 S3n ledger 接到两个消费者：

1. 几何消费者需要 frozen c5 的“四内点 → vertex-star 法向 P1 极限”，同时
   newest edge 必须取 body kinematics；
2. scalar 消费者需要 P2 Gauss points 上的相对切向材料速度，而 frozen
   transport callback 只收到裸坐标。

同一世界坐标在 sheet 上可能属于不同 owner limit。因此第二个断点不是“多传
一个数组”的软件问题，而是 on-sheet 算子身份缺失。

## ② 学科机理

Krebs 的 wake vertex local velocity 与高阶 element interior points 分属几何
推进和场表示。Ambrose 的 Birkhoff–Rott 结果进一步说明，sheet-average 是
具名 mean limit；边/面上的裸点不能替代 owner。NASA TP-2995 同样要求
on-panel/edge limit 显式处理。

ALE 侧，Elliott–Venkataraman 的 relative velocity 是：

```text
c = u_material - w_mesh.
```

几何只消费 `u_material·n`，计算网格切向规范在本候选取零；P2 transport
消费 `c_t`。因此不能复活已经 falsified 的 continuous full-vector P1 投影。

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| S3n 四通道物理速度 | 已验证，冻结 |
| S3s open topology / P2 rows | 已验证，冻结 |
| c5 四内点 vertex-star normal | 已验证范围，冻结 |
| full-vector continuous-P1 | falsified，禁止 |
| body-edge fluid velocity投影 | 错件；应取body kinematics |
| owner-aware arbitrary P2 quadrature | 缺件 |
| owner-fed consistent-P2 assembly adapter | 缺件 |
| 时间推进/重解 | 下一阶段 |

## ④ S3t 预登记

候选不改变任何物理通道或 weak form：

- geometry query 继续使用每面四个 Krebs strict-interior points；
- free P1 只作 vertex-star normal projection；
- body P1 edge 精确插入给定 kinematic velocity；
- P2 query 为带 patch/face/barycentric 的 order-4 triangle quadrature；
- `u_material-w_mesh` 只在 P2 assembly 内取 tangential component。

门包括 closed shared-range projector 等价、monolithic transport matrix
等价、actual 四通道、刚体客观性和失败语义。完整阈值已在实现前冻结于
`actual_wake_owned_stage_velocity_cases.yaml`。本门不改变 geometry 或
scalar state，也不选择 fixed point。

## ⑤ 执行结果与 claim 裁决

冻结配置未改，11/11 门通过：

| 指标 | 结果 |
|---|---:|
| geometry / P2 owned query 重建误差 | `2.221e-16` |
| free-P1 切向泄漏 | `1.388e-17` |
| body attachment / chronological seam 误差 | `0 / 0` |
| open ↔ frozen closed projector 共享范围误差 | `0` |
| owner-fed ↔ frozen P2 mass/advection 误差 | `0 / 0` |
| actual 四通道 ledger closure | `0` |
| actual P2 mass rank deficiency / condition | `0 / 22.0782` |
| 常数输运残差 | `8.101e-15` |
| rigid projection / mass / advection 误差 | `2.914e-16 / 1.735e-17 / 1.110e-16` |
| 非法 owner、body shape、velocity/query | `3/3` fail closed |

因此 `N3.1j3b6d18c2b3b2b` 只在以下固定 stage 范围
`validated/frozen`：

> owner-preserving query、free-P1 法向/body-essential 几何速度，以及消费
> 同一实际四通道场的 consistent-P2 相对切向矩阵装配。

两个数值不能被 GO 隐藏：

- vertex-star 对实际面法向样本的最大残差为输入峰值的 `55.20%`；
- P2 ALE 装配前被明确排除的相对法向分量峰值为 `0.3295`。

前者与已冻结 c5 的语义一致：高阶面内内容保持在 residual 中，不被 P1 截距
吸收；后者证明 scalar transport 不能误食 geometry-normal 通道。它们不是可调
常数，也不证明时间推进正确。下一阶段必须把二者作为具名 stage diagnostics，
并以耦合残差、时间 Cauchy 和 timestep independence 裁决显式 stage re-solve
还是 present/averaged-time implicit iteration。

本阶段没有更新几何或 P2 scalar，没有求解 pressure/force，也没有触碰结构。
