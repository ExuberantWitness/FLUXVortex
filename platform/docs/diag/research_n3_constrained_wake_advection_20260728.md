# N3 S3h：body-attached material-wake Heun 语义

日期：2026-07-28  
Claim：`N3.1j3b6d18c2a`  
当前状态：**EXECUTED / GO**  
角色：无压力、无力的制造几何积分器。

## ① 病因定位

S3g 已让任意三维 wake geometry 能以有向 material identity 重新进入
actual-boundary 方程。但现有 `advance_assembly_normal_geometry_heun()` 对每个
顶点执行同一自由几何更新，没有“newest edge 等于当前 body cut”的 essential
boundary。

若直接组合，relaxation 会先把最新 wake edge 从尾缘移走，下一次
actual-boundary solve 只能 attachment failure。病因挂在
`N3.1j3b6d18c2`，属于缺少**气动运动学边界条件**，不是结构响应模型。

## ② 学科机理

Krebs 的时间步把 lifting surface 移到新位置，并在 previous/current trailing
edge 之间创建新 material elements；旧 wake 顶点随后按 local velocity relax，
旧强度保持。标准 unsteady panel marching 同样区分：

- 当前 shedding edge：由当前 trailing-edge 几何给定；
- 已脱落 wake corners：按外流与诱导速度自由推进；
- 新旧 circulation：在 Kutta/Kelvin 账中连接。

因此 body edge 是 essential geometry boundary，不是自由 wake 速度和 body
速度的加权平均。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| N3.1j4c5/c6 无约束 Heun | free sheet 范围已验证，冻结 |
| 把 attached TE vertices 当自由 wake vertices | 错件 |
| 每 stage 的 prescribed body-cut position | 缺件 |
| seam velocity 不一致后再平均 | 禁止；应 fail closed |
| 本门接压力或结构动力学 | 禁止 |

## ④ 预登记

先只验证积分语义。自由顶点使用制造仿射速度 `v=A x+b(t)`；最新 current edge
在每个完成 stage 精确取给定 body-cut trajectory。运行 `2/4/8` 步时间族，
检查：

- input/strength 不变；
- attachment、history seam 和 duplicate seam velocity；
- free-vertex Heun 二阶 Cauchy；
- 整体刚体变换客观性；
- 面积与 seam-mismatch fail-closed。

阈值已冻结于 `constrained_material_wake_advection_cases.yaml`。只有该门通过，
才允许 c2b 接入实际 body+wake velocity 与 post-relaxation equilibrium。

## ⑤ 执行结果与 claim 裁决

预登记配置未改。`2/4/8` 步族和刚体反事实的结果为：

| 指标 | 结果 | 冻结门 | 裁决 |
|---|---:|---:|---|
| free-vertex Cauchy ratio | 3.9953388401 | ≥3.5 | PASS |
| finest relative change | 1.60993e-5 | ≤0.01 | PASS |
| newest-edge attachment | 0 | ≤1e-14 | PASS |
| history seam | 0 | ≤1e-14 | PASS |
| duplicate seam velocity | 0 | ≤1e-14 | PASS |
| material-strength mutation | 0 | =0 | PASS |
| rigid-frame error | 8.88178e-16 | ≤1e-11 | PASS |
| minimum face-area ratio | 0.958423 | ≥0.90 | PASS |
| mismatch fail-closed | 1 | ≥1 | PASS |

因此 `N3.1j3b6d18c2a` 在“制造速度场＋给定 body-cut 运动学＋无压力/无力”
范围内 `validated/frozen`。这不是实际自由尾迹物理的通过证据：
`actual body+wake induced velocity`、relaxation 后 surface/newborn-wake
强度重解、移动边界势率、压力和力仍全部未验证。下一可动空间严格限定为
`N3.1j3b6d18c2b`。
