# N3 S3d：旧 material wake 已知态／最新 trace 未知态分区

日期：2026-07-28  
Claim：`N3.1j3b6d18a`  
运行角色：无压力、无力的 affine equation/Kelvin ledger oracle。

## ① 病因定位

S3c 的多带空间方程通过，但所有 band 都被赋予当前 body jump；这只适用于
steady canonical。若把这套组装直接用于非定常过程，每次 solve 都会重写整个
wake，违反 material Kelvin 记忆。

病因挂在 `N3.1j3b6d18`：

> 旧带是已脱落材料态，应作为已知 wake influence 进入 RHS；只有最新 band
> 的当前行与当前 body cut jump 同属未知量。

## ② 学科机理

Krebs 明确区分 steady 与 unsteady：

- steady 可把相同 TE strength 沿整个 wake 传播；
- unsteady 时，单个 wake element 一旦脱落，其 strength 必须保持；
- 当前 surface 与 newly-created wake 相互影响，需要在当前 flow-tangency
  solve 中联立。

因此非定常方程是 affine partition，而不是每步重算全部 wake strength：

```text
A_body mu_current
+ A_new,current J_cut mu_current
= rhs_source
  - influence(old material rows)
  - influence(new previous/middle rows)
```

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 每步用当前 jump 覆盖所有 wake bands | 错件 |
| 把每条旧带继续放进未知向量 | 错件，破坏 material Kelvin |
| 旧带 affine RHS＋最新 current row 联立 | 缺件 |
| 用 `(old+current)/2` 自动补 midpoint | 当前禁止，时间积分尚未验证 |

## ④ 预登记

冻结一个 old band 和一个 active band。old band 三行显式为
`0.20/0.30/0.40·g(y)`；active 的 previous/middle 为
`0.40/0.10·g(y)`，current 行由 body solve 决定。再运行符号反转 history
和 zero-history comparator，检查：

- old rows bitwise 不变；
- material interface 和 current attachment；
- history 确实改变当前解；
- incident/history 的 affine superposition；
- rank、条件数、弱残差与 tip identity。

完整阈值已在实现前冻结于
`actual_boundary_unsteady_wake_partition_cases.yaml`。本门不推断 midpoint，
也不授权时间阶、pressure 或 force。

## 执行结果：GO

全部预登记门通过：

- old input rows 最大 mutation 为 0；
- old-current 到 active-previous interface 与 active-current 到 body jump
  attachment 均为 0；
- old wake 未知量为 0，但 known-wake RHS 范数为 `5.77e-4`，证明旧历史
  实际进入方程而非只存 manifest；
- 四个系统 rank deficiency 为 0，最大条件数 `110.82`，弱残差
  `9.80e-16`；
- old history 全部反号后，当前 body jump 最大变化 `0.11032`，通过
  `1e-4` 的 history-consumption 门；
- `solution(incident,history) = solution(incident,0) +
  solution(0,history)` 的最大误差 `4.65e-16`；
- tip jump 为 0。

因此 `N3.1j3b6d18a` 在 **known-old-state/current-active-row affine
partition** 范围 `validated/frozen`。父节点 `N3.1j3b6d18` 升为
`partial`。

仍未解决的是 active band 的 previous/middle rows 从何而来。下一
`N3.1j3b6d18b` 必须用显式 stage 方程生成 midpoint，并在相同物理时间窗做
`dt,dt/2,dt/4` Cauchy；禁止把端点平均冒充经过验证的 midpoint。通过前不允许
用 `∂mu/∂t` 进入 Bernoulli pressure。
