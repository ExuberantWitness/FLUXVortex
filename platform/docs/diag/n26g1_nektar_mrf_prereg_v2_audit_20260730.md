# N2.6g1 Nektar++ MRF 来源门预登记 v2 独立审计

日期：2026-07-30  
审计对象：`n26g1_nektar_mrf_source_prereg_v2_20260730.md`  
对象 SHA-256：
`14d29b5b6523ed523efae255b345edab61a1a5b2f31e54d265f986963d3c9b0b`  
裁决：`BLOCKED — NOT PASS TO EXECUTE`  
执行状态：archive 未解包，build/source flow/target 均未启动。

## 1. 已关闭的 v1 P0

v2 已正确关闭：

- \(t<0.2\)、切换点和右支的运动学定义；
- 零参考量的固定尺度误差；
- source session 的方程、SVV、BC/IC、pressure gauge 和线性求解器；
- 惯性轴、法向、traction、CL/CD/CM 符号和尺度；
- closed-source/open-target 的角色转换；
- source/reference/scorer 的 SHA 和支持域。

这些内容不得因本次阻塞而回退。

## 2. 尚未关闭的 P0

1. **Scotch 绑定无效。** 固定 commit 的 `FindScotch.cmake` 从
   `SCOTCH_INCDIR/SCOTCH_DIR/SCOTCH_HOME` 搜 header，再重写
   `SCOTCH_INCLUDE_DIR`。v2 只预置 cache 变量，不能证明用了候选 sysroot；
   还未冻结运行时 library path。
2. **mesh family 自相矛盾且 generator 不唯一。** “全部 triangle”与
   “BL 可 quad”冲突；Spline control/parameterization、curve orientation、
   physical IDs、BoundaryLayer/Box/Min fields 及 Gmsh/NekMesh argv 不完整。
3. **K0--K2 不可机械执行。** K0 无翼却比较 force；K1 没有冻结完整 MRF
   residual 和可调用 case；K2 没有输入/期望数组，且不能验证 native
   `AeroForces`。
4. **checkpoint/primitive 计划与 12 GiB 门冲突。** 每 0.01 保存五个 runs
   会超预算；峰值只有跑完 force curve 才知道，现计划不能保证抓到对应场。
5. **Cauchy 的 range-normalized L2 未写公式和固定 reference range。**
6. **G2 仍不是保守结构传力。** 点值求值不是 cross-mass/consistent load
   transfer；弱导数未定义；upper/lower 未分离；`\xi` 一阶权重不等于物理
   moment，也没有刚体旋转和非刚体结构模式的直接虚功账。

## 3. P1

1. `all other NEKTAR_SOLVER_*=OFF` 不是可执行 CMake token，必须逐项枚举。
2. `NEKTAR_USE_SYSTEM_BLAS_LAPACK=ON` 不能唯一绑定已列 OpenBLAS，必须走
   OpenBLAS 分支并冻结最终 realpath/SHA/`ldd`。
3. MRF 数学式尚未变成唯一 XML；必须冻结 `sign/floor/max` 的切换语义、
   `FRAMEVELOCITY`、`PIVOTPOINT` 和实际 knot oracle。
4. source field retention、restart replay 与 formal force curve 的关系必须
   明确，不能用 replay 拼接或替换正式曲线。

## 4. 裁决边界

这些问题都发生在任何 Nektar 自建输出之前，故允许另立 v3 修订；它们不是
source physics 失败，也不授权换版本、换候选或观察 target。v2 保留原文和
SHA，不覆盖。只有 v3 复合规范经未参与起草者给出 `PASS TO EXECUTE`，
才可解除 archive 的 audit hold。
