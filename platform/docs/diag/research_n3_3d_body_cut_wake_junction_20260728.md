# N3 S3a：三维实际边界势切面到物质 wake 的接口

日期：2026-07-28  
Claim：`N3.1j3b6d ↔ N3.1j4`  
运行角色：无力 topology/trace oracle。

## ① 病因定位

仓库已经分别具备：

- actual-boundary 连续 P2 trace 与成对 Galerkin 算子；
- 非零环量需要 classified potential cut 的二维证明；
- P2 DDE 静态场、材料 Kelvin、短时无核输运和显式多 patch 接口；
- `mu_DDE=-Gamma_N1` 的方向/量纲身份。

但这些证据之间仍缺一个三维接点：

> 闭合物面几何保持 watertight 时，尾缘上、下两侧的**势自由度**如何复制，
> 以及该势跃如何逐节点交给第一条 material wake band？

若这个接口未验证，后续“body P2＋wake P2”联立会在无意中把势切面重新焊死，
或者把同一 circulation 重复记账。

## ② 学科机理

Erickson 的 NASA TP-2995 和 NASA TM-88355 的 lifting-potential
branch-cut 处理都区分两件事：

1. 物体几何仍然闭合；
2. 标量势在分类 wake cut 两侧允许 jump。

Le Provost 等（JFM 977, 2023）把 bound-sheet 无穿透、Kelvin 与 newborn
wake 放进同一系统；Krebs–Bramesfeld–Cole（*Aerospace* 9, 2022, 28）
则提供连续分布 doublet 的物质 wake 表示。两者共同要求：body cut jump 与
wake potential jump 是同一个物理 circulation 账，而不是两项可相加的力。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 闭合几何意味着尾缘势 DOF 也必须焊接 | 错件 |
| 复制物理坐标形成几何裂缝 | 错件 |
| body cut jump 与 wake `mu` 分别自由选择 | 错件，会重复/漏记环量 |
| 共享几何＋分类势 DOF 复制＋逐节点 jump 接口 | 缺件 |

## ④ 方案与预登记

采用全翼三维 diamond finite-angle canonical：

- 几何为单一闭合三角壳；
- 只沿 TE 展向线复制上/下 P2 势 DOF；
- 两翼尖 jump 为零，端点保持单值；
- 制造 jump `mu(y)=1-y²`；
- 当前 TE 边连接一条显式三时刻、内部连续的 P2 material TEV band；
- 对 body jump、wake trace、规范不变性、N1/DDE sign round-trip、
  非 cut 连续性和物面 watertight 分别设硬门。

完整输入、阈值和禁止项已在实现前冻结于
`actual_boundary_3d_cut_wake_junction_cases.yaml`。通过只授权下一步
body–wake 联立方程门，不授权压力、力、有限 base 或生产。

## 执行结果与 claim 裁决

规范算例在预登记不变的条件下得到 `GO`：

- 闭合翼体为 20 顶点、36 三角面，boundary/nonmanifold/orientation
  mismatch 均为 0，物理顶点最大变化严格为 0；
- 连续 P2 body trace 原有 74 个自由度；只复制 3 个内部 TE 顶点和
  4 个 TE 边中点自由度，得到 81 个自由度；
- 所有非 cut 边仍共享同一组三个 P2 trace DOF，mismatch 数为 0；
- 上下 cut 的坐标配对误差、`mu(y)=1-y²` body jump 误差、当前 wake
  edge 接合误差、规范平移变化和 N1/DDE sign round-trip 误差均为 0；
- 两翼尖 jump 为 0；对称中心导数误差 `3.18e-15`；wake 内部 P2
  trace jump 为 0。

因此：

1. `N3.1j3b6d14`：“watertight 几何迫使尾缘标量势也单值”被
   `falsified/frozen`；
2. `N3.1j3b6d15`：“共享几何、分类 P2 势复制及同一 material-wake
   jump 可形成严格三维 trace junction”仅在 topology/trace 范围
   `validated/frozen`。

这不是压力或载荷模型已经完成。下一步只允许建立 **body–wake 联立方程
门**，检验无穿透、规范零空间、Kelvin 约束和 wake jump 是否组成不重复
记账且可辨识的线性系统。有限钝 base 仍沿 `N3.1j3b6d11–13` 保持
`NO-GO`；不得用本规范的 coincident sharp cut 偷换真实 NACA-2406 base。
