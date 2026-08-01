# N3 S3g：有向 body-cut ↔ material-wake trace attachment

日期：2026-07-28  
Claim：`N3.1j3b6d18c1a`  
当前状态：**EXECUTED / GO**  
角色：无压力、无力的有向 trace 身份 oracle。

## ① 病因定位

S3f 已经排除三类病因：

- 旧 straight path 与显式 straight history 完全一致；
- 曲 wake geometry 会改变当前 body 解，说明几何已被消费；
- 当 cut 顺序不被世界坐标旋转翻转时，底层积分客观到 `O(1e-15)`。

唯一随失败共同变化的是 cut 的**参数方向**。当前
`ClassifiedP2CutTopology` 为方便展示，从世界坐标字典序较小的端点开始遍历；
这不是 material identity。任意刚体旋转可改变起点，继而反转 wake span
参数、三角面法向和 double-layer 势跃的物理方向。

病因挂在已证伪的 `N3.1j3b6d18c1`，新节点
`N3.1j3b6d18c1a` 只允许补充有向 attachment，不允许改几何、积分或阈值。

## ② 学科机理

panel abutment、Krebs gapless material sheet 和 Pate orientable vortex
sheet 都要求接口身份包含拓扑方向。对于 double layer，

```text
phi_doublet ~ mu * partial(G)/partial(n)
```

反转面法向会反转 `partial(G)/partial(n)`；若要表示同一物理 sheet，`mu`
必须同步反号。因此 attachment 不是“两个数组长度相同”，而是：

```text
wake_mu = s_attach * P_cut_to_wake * (mu_upper - mu_lower)
```

其中 `P` 只能由明确的 body material vertex IDs 推导，`s_attach` 只能是
预先声明的 `+1/-1`。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| world-coordinate endpoint sort | 对物理接口是错件 |
| exact ordered material vertex IDs | 缺件 |
| 由 IDs 推导 P2 identity/reversal | 缺件 |
| 有向 jump sign | 缺件 |
| 按数值结果选择 sign | 禁止 |
| nearest-neighbour welding | 禁止 |

## ④ 预登记

候选 typed `MaterialWakeCutAttachment` 只含：

- `ordered_body_cut_vertex_indices`；
- `wake_jump_from_body_cut_sign ∈ {−1,+1}`。

运行四个独立门：

1. forward attachment 复现 legacy straight；
2. curved history 仍被消费且不变；
3. generic rigid transform 即使让 coordinate topology order 翻转也保持
   matrix/RHS/cut jump；
4. span 参数反向且 rows 同步反序、反号时，表示同一物理 double layer。

非法、重复、非 cut-chain IDs 与非二值 sign 必须在 assembly 前失败。全部阈值
冻结于 `actual_boundary_oriented_wake_attachment_cases.yaml`。

GO 也只允许回到三维 geometry interface；仍不授权 wake relaxation、压力、
力或 production。

## 执行结果：GO

四组预登记门全部通过：

- forward typed attachment 与 legacy straight 的 matrix/RHS/cut jump 差均
  为 `0`；
- curved geometry/strength mutation 为 `0`，相对 straight 的 cut-jump
  响应仍为 `1.516e-3`；
- generic rigid transform 确实反转了 coordinate-selected cut order，但
  material-ID permutation 后 matrix/RHS/cut-jump 客观误差仅
  `5.55e-17 / 1.78e-17 / 6.63e-15`；
- 反向 span 参数化配合 `s_attach=-1` 后，matrix/RHS/cut-jump gauge 差为
  `1.12e-14 / 4.88e-15 / 1.02e-12`；
- oriented current attachment、history interface 和 tip jump 误差均为
  `0`；
- 4 个系统 rank deficiency 为 `0`，最大弱残差 `3.82e-16`；
- 重复 IDs、非二值 sign、非 cut-chain IDs 三类非法输入全部 fail closed。

因此 `N3.1j3b6d18c1a` 仅在 **有向 material attachment identity** 范围
`validated/frozen`。S3f 的“无 orientation 仍充分”反例继续保留，不被本结果
删除。

该 GO 只解除 `N3.1j3b6d18c2` 的接口前置阻塞：现在可以把已验证的无核 Heun
wake geometry 推进结果重新送入 actual-boundary 方程，并验证 post-relaxation
surface/newborn-wake equilibrium。压力、力和 production 仍禁止。

