# N3 S3n：actual body–wake sheet velocity ledger

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3a`  
状态：**PREREGISTERED / NOT EXECUTED**

## ① 病因定位

S3m 已验证 typed body-inflow 的 P2 材料势跳输运，但实际速度还没有闭合：

- `material_wake_time_march.py` 仍规定匀速 `x` 对流；
- `constrained_material_wake_advection.py` 只有抽象 provider；
- runtime shadow 只计算 wake 自诱导且只取法向几何速度；
- actual-boundary 解没有在 wake owner points 上发布 body source/doublet；
- wake 移动后没有 re-solve/耦合残差 driver。

所以病因是**缺具名物理速度组成与后续 fixed point**，不是诱导速度常数错误。

## ② 学科机理

Krebs 2021 Fig.3.1/3.4/3.8 明确给出：

```text
solve surface/newborn strength
  -> move wake vertices with local velocity
  -> resolve surface/newborn strength on moved geometry
```

且 wake relaxation 前后都要恢复连续 circulation 与 flow tangency。其局部
速度来自相互作用的 aerodynamic surfaces 和整个 wake。

对 doublet sheet，本体上的物理几何速度必须取 Birkhoff–Rott/
sheet-average 极限；Ambrose 的连续理论和仓库已冻结的
`N3.1j4b5a` 数值门共同排除单侧值、offset 或任意 core。

来源：

- Krebs dissertation (2021), Ch.3/Fig.3.1,3.4,3.8；
- Krebs, Bramesfeld & Cole, *Aerospace* 9 (2022) 28：
  https://doi.org/10.3390/aerospace9010028
- Ambrose, *SIAM J. Appl. Math.* 85 (2025)：
  https://doi.org/10.1137/24M164848X
- Erickson, NASA TP-2995 (1990)。

## ③ 缺件还是错件

| 组成部分 | 裁决 |
|---|---|
| external incident | 必须显式且声明已含来源 |
| actual body source | 缺 ledger channel |
| actual body P2 doublet | 缺 ledger channel |
| full-wake sheet-average | 已有算子，缺全历史组合 |
| 仅 wake self velocity | 错件，不是 local total velocity |
| external 中重复带入 body/wake | 错件，必须 fail closed |
| 直接做 relax fixed point | 顺序错误，先验证固定快照速度账 |

因此 `c2b3` 拆为：

- `c2b3a`：fixed-snapshot four-channel velocity ledger；
- `c2b3b`：用已验证 ledger 做 geometry relax，再重解 strength 并监控
  named residual。

## ④ S3n 预登记

从 S3e 的 `t=0→0.5, dt=0.25` 生成 far trace 为零的两条 material bands，
只对 free old/seam geometry 作确定性展向对称弯曲，再以已冻结有向
attachment 重解 actual-boundary state。

每个 wake face 使用四个严格内部 owner points，计算且只计算：

```text
u_sheet =
  u_external
  + u_body_source
  + u_body_P2_doublet
  + u_full_wake_sheet_average
```

预登记门覆盖 query identity、history/attachment、四通道非零、逐项求和、
body/wake 求积、assembly 与逐band等价、刚体客观性、duplicate-source 和非法
query 失败语义。

输入和阈值已冻结于
`actual_body_wake_velocity_ledger_cases.yaml`。本门禁止移动 geometry、
输运 `mu`、fixed-point iteration、pressure、force、LEV、target load 和结构
动力学。

## S3n 执行结果：NO-GO

首轮测量误把 body P2 通道接到二维面积 Gauss oracle，而不是已冻结的
解析径向/边积分算子。该首轮结果未删除，保存在
`actual_body_wake_velocity_ledger_results_initial_wrong_body_operator.json`。
修正测量器身份后，保持预登记工况、阶次和阈值不变重跑。

10 项门中 9 项通过：

- query 重构 `2.22e-16`，history、far trace、tip 和 attachment 均为零；
- 四个通道全部有限且非零，账本逐项和误差为零；
- assembly 与逐 band 求和误差 `1.71e-14`；
- 刚体旋转/平移通道误差 `1.53e-13`；
- duplicate-source 与非法 query/provider 均按预登记 fail closed。

唯一失败为求积：

- body analytic-radial/edge operator：`q52→72 = 9.478e-3`；
- full-wake sheet average：`q64→80 = 3.416e-3`。

最高误差点不是随机点，而是 Krebs 顶点侧内点
`lambda=(0.0333,0.0333,0.9333)`。body 主犯面与该点的归一法向距离仅
`3.24e-3`，且投影在尾缘顶点外侧；wake 误差则主要来自同 patch 的 owner
finite-part 项。诊断性地继续到 `q=192/256` 后，wake 变化降到
`1.19e-9`，证明公式存在稳定极限，但普通 Gauss 没有在冻结阶次内解析近边/
近顶点层。

所以 `c2b3a` 不是“漏了速度来源”，也不是要加 core。裁决是：

> 四通道物理组成得到 9/10 支持，但其近奇异数值组成部分缺失；
> `c2b3a` 保持 partial，`c2b3b` 禁止启动。

## S3o 文献裁决与下一门

Johnson NASA CR-3079 对多项式 source/doublet 面元以闭式 `H/F` 递推计算
影响系数。Montanelli–Aussal–Haddar（2022）及
Montanelli–Collino–Haddar（2024）对二次三角形进一步指出：目标点贴近
面元时，应先在参考三角形定位近奇异原像，再进行奇异项消去/continuation
和 transplanted Gauss；普通 Gauss 只在奇异点远离边界时保持快速收敛。
Johnston–Johnston–Elliott（2013）则用以投影点为中心的
sinh–sigmoidal 变换处理三角形的近弱/强奇异积分。

一手来源：

- Johnson, NASA CR-3079 (1980):
  https://ntrs.nasa.gov/citations/19800015776
- Montanelli, Aussal & Haddar, *SIAM J. Sci. Comput.* 44 (2022):
  https://arxiv.org/abs/2111.13151
- Montanelli, Collino & Haddar, *SIAM J. Sci. Comput.* 46 (2024):
  https://doi.org/10.1137/23M1605594
- Johnston, Johnston & Elliott, *J. Comput. Appl. Math.* 245 (2013):
  https://doi.org/10.1016/j.cam.2012.12.018

S3o 因而只允许在已经精确径向约化/finite-part 后，对每条直边用目标点最近
线段位置构造无量纲 `s=s0+d*sinh(t)` 坐标；该变换不改变积分、没有物理
参数，也不允许 offset/core。先过解析常强度涡环、独立高阶参考、近顶点
Cauchy、刚体客观性和失败语义，再用原 S3n 阶次/阈值复跑。

## S3o→S3q 闭环与 S3n 最终复跑：GO

S3o 证明“只做 sinh 移植”在制造近顶点失败；S3p 把完整 P2 边界涡解析抽出，
验证了 constant-ring，却仍定位到面积涡余项；S3q 再用 `I0/I1/I2/J0/J1`
端点矩解析 owner/coplanar finite-part，全部子门通过。

随后以 `target_sinh_analytic_sheet` 重跑原 S3n 十项门，未修改工况、四通道、
来源身份或物理阈值：

- 10/10 checks GO；
- body/wake 求积均通过；
- ledger closure `0`；
- full wake 与逐 band 和误差 `1.73e-14`；
- rigid channel `1.46e-13`；
- duplicate source 和 invalid query/provider 仍 fail closed。

因此 `N3.1j3b6d18c2b3a` validated/frozen。该结论只覆盖固定快照 local
velocity ledger；`c2b3b` 的 geometry relaxation、强度重解和耦合残差仍为
open，不得把本 GO 解释为自由尾迹或载荷模型已完成。
