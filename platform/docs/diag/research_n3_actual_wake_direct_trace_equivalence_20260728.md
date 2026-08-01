# N3 S3ac0：actual direct-trace execution equivalence

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2a`  
状态：**VALIDATED / FROZEN（仅执行等价性）**

## ① 病因定位

S3ac 单步 composition 的全部物理残差通过，但一次 step 用时约 10 分钟。
逐调用审计发现成本不是新物理，而是 S3aa 的 zero＋9 unit＋verify＋
counterfactual 共 12 次 actual-boundary 重复装配。

在 `solve_actual_boundary_body_wake_p2()` 的 prescribed-history 分支中：

- `fixed_rows` 只取完成的 old bands；
- newest band 的 `active_rows` 只取前两行；
- newest current row 始终由 `active_cut_map @ body_potential` 生成；
- 候选 history 中输入的 current row 不进入 matrix 或 RHS。

因此 S3aa 的 trace residual 严格为
\(R(g)=g-G(\text{geometry, old/free rows})\)，Jacobian 是单位阵。这解释了
已冻结结果的 condition `1.0000000000000002`，也说明 9 个 unit solves 只是在
重复证明源码已经显式给出的仿射身份。

可动空间仅为新增一个 composition-level direct evaluator；不得修改或删除
冻结的 S3aa 实现。

## ② 机理与代数依据

边界元方程对 prescribed wake doublet rows 和 body potential 是线性的；
newest current row作为 body-cut jump 的代数输出，而非独立输入。直接取一次
actual solve 的 `global_p2_state[body_attachment_dofs]`，与 S3aa 求
`R(g)=0` 是同一个方程，不是近似、缓存或降阶。

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| S3aa 物理/代数命题 | validated/frozen，不修改 |
| zero＋unit 证伪构造 | 有效证据门，但不适合作为多步执行内核 |
| direct one-solve evaluator | 缺少已验证的等价执行路径 |
| 求积、矩阵、RHS、trace | 必须与 S3aa 完全相同 |

## ④ 预登记

在同一 canonical half-stage geometry/state 上同时运行：

1. frozen S3aa zero＋unit evaluator；
2. direct one-solve evaluator。

比较 body potential、全 wake P2 state、solved trace、actual weak residual、
free-state preservation、attachment、copy-counterfactual residual和输入不变性。
所有状态差 ≤ `2e-12` 才允许 S3ac repeated-insertion 使用 direct evaluator。

GO 只冻结执行等价性；任何物理 claim、阈值、求积和状态均不变。

## 执行结果

预登记双路径比较为 **GO**：

- 全局 body potential、全局 wake P2 state、solved current trace 与全部共享
  诊断的最大绝对差均为 `0.0`；
- actual matrix 与 RHS 逐位相同；
- 输入状态 mutation 为 `0.0`；
- frozen S3aa 证据构造使用 `12` 次 actual solve，direct evaluator 使用 `1`
  次；
- 没有缓存近似、求积变更、阈值放宽或物理方程变更。

因此仅将“direct one-solve 是 frozen S3aa 的严格执行等价路径”改写为
validated/frozen，并允许它进入 S3ac repeated-insertion 候选。该结果不允许
生产激活，也不新增气动物理 claim。
