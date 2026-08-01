# N3 S3c：shape-regular 多带物质 wake 联立门

日期：2026-07-28  
Claim：`N3.1j3b6d17`  
运行角色：无压力、无力的空间离散/方程门。

## ① 病因定位

S3b 的高阶指纹把两个现象分开：

- 固定 `x=8` 时，提高 paired singular order 可使 root jump 趋向
  `−0.1265`；
- 固定 order、持续拉长同一条 band 时，解在 `x>8` 重新漂移，且镜像误差
  增长。

因此病因挂到 `N3.1j3b6d16` 的 **single elongated wake
discretization**，可动空间仅是 wake chordwise topology 与相应积分分区。

## ② 学科机理

NASA panel-method wake network 把远尾迹拆成多个 panels/strips，并单独做
wake-integral convergence。Krebs DDE 则在每个时间步生成新 material
elements：旧元素保持自己的强度和几何，新元素只覆盖相邻 shedding edges
之间的区域。

这两条证据共同排除“把第一条带不断拉长”。合理离散应是：

1. 每条 band 长宽比受控；
2. 旧到新按材料时间排序；
3. 相邻 band 的 P2 trace/geometry/time 显式接合；
4. 只有最新的 TE-adjacent band 与 body 形成共边奇异 pair；
5. 更老的 band 与 body 分离，使用普通光滑积分。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 一个二次面元可同时承担 newborn 与任意远场 wake | 错件 |
| 提高单面元 q 可替代 wake chordwise refinement | 错件 |
| shape-regular chronological material bands | 缺件 |
| 多带通过即可声称 unsteady Kelvin 已完成 | 错件 |

## ④ 预登记

保持 S3a body/cut 不变，以 `0.5c` chordwise band 覆盖到
`x=4/8/12`，使 band chord 与 span cell width 相等。冻结
`q=6/8/10`，检查 history 接口、唯一环量账、rank/残差、symmetry 随 q
下降、q8→q10 Cauchy 和 x8→x12 far-wake Cauchy。

完整阈值与禁止项已在实现前写入
`actual_boundary_material_wake_history_cases.yaml`。通过仍不产生 pressure
或 force，只允许下一步 unsteady Kelvin-history canonical。

## 执行结果：GO

全部冻结门通过：

- cutoff `x=12` 由 22 条 `0.5c` band 构成，最大长宽比严格为 1；
- history 的 time/geometry/P2 trace 接口误差全部为 0；
- body-wake common-edge pair 始终只有最新带的 8 对；
- 独立 wake amplitude 为 0，矩阵 rank deficiency 为 0，最大条件数
  `271.46`，弱残差 `3.83e-16`；
- zero-alpha 归一误差随 `q=6/8/10` 从
  `6.83e-5→2.54e-5→7.86e-6`；
- incidence antisymmetry 从
  `1.36e-4→5.06e-5→1.57e-5`；
- span mirror 从 `1.26e-4→5.02e-5→1.55e-5`；
- `q8→q10` root Cauchy 为 `9.23e-5`；
- `x8→x12` far-wake Cauchy 为 `1.153e-3`；
- tip jump 与最新 wake attachment 均为 0。

相对于 S3b 单长带的 `x4→x8=0.315`，多带 far-wake 误差降低约 273 倍，
且未新增任何拟合量。因此 `N3.1j3b6d17` 只在 **steady、fixed wake
geometry、同一重复 material trace、equation/refinement** 范围
`validated/frozen`。

下一节点必须改变时间状态而不是继续改空间网格：冻结三个或更多时刻的非恒定
body jump，把每个 shedding interval 出生的 P2 trace 固定到相应 material
band，检验 old bands 不被当前解重写、最新 band 与当前 body jump 接合、
全局 Kelvin 账和时间阶收敛。该节点仍不计算 pressure/force。
