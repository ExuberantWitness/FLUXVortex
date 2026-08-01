# N3 S3z：point trace 与 weak release velocity 裁决

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c1b`  
状态：**EXECUTED / WEAK-GO**

## ① 病因定位

S3y 的 topology/state injection 全部通过，但 retaining-to-released row 的
one-sided local fit 留下两个不能吸收的指纹：

- pointwise relative residual `68.80`；
- amplitude-level projection residual 约 `55.2%`。

随后只读 approach audit 在同一 actual 四通道速度上把距离连续减半。翼尖
vertex 的 normal velocity：

```text
epsilon: 0.025   0.0125   0.00625   0.003125
value:   0.2337  0.2918   0.3479    0.4027
delta:           0.0581   0.0561    0.0548
```

差值没有趋零；外侧 edge midpoint 也保持约 `8e-3` 的漂移。三个内部 vertex
从不同 incident faces 逼近时，最细方向差最大 `0.0441`。这不是“拟合阶数
稍低”的指纹，而是 point trace 可能不存在或不唯一。

## ② 学科机理

Pozrikidis (JFM 2000) 对三维 vortex sheet 的表述是：marker 只需跟随流体
**法向**速度，切向运动是任意参数化。它并未赋予开放 sheet 边/角点一个天然
的有限节点速度。

DeVoria & Mohseni (JFM 2018) 直接指出 Birkhoff–Rott 自诱导方程的奇异性，
并通过有限 segment 的有界积分构造处理自诱导运动。这支持“先问积分量是否
存在”，不支持从越来越近的点值硬外推。

Dziuk–Elliott 和 Elliott–Venkataraman 的 evolving-surface FEM 则提供了无
经验常数的离散路线：把物理法向速度作为弱式右端，在当前 continuous-P1
surface space 中通过 consistent Gram/mass matrix 投影为有限 mesh velocity。
这不是平滑物理场，而是定义有限维几何状态所必需的 variational projection。

一手来源：

- https://doi.org/10.1017/S0022112000002202
- https://doi.org/10.1017/jfm.2018.645
- https://doi.org/10.1017/S0962492913000056
- https://doi.org/10.1002/num.21930

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| actual 四通道 interior velocity | 已验证，冻结 |
| P2 owner finite-part | 已验证，冻结 |
| attachment pointwise vertex trace | 强疑似错件，需正式证伪 |
| local vertex-star LS 的满秩/刚体性 | 数值恒等式正确，但不等于 point trace |
| global weak P1 normal projection | 缺件 |
| blob/core/epsilon/filter | 本病因下无证据，禁止 |

## ④ S3z 预登记

同一冻结场上并行执行：

1. **Point candidate**：使用未在 exploratory audit 中查看的
   `epsilon=0.0015625, 0.00078125` 继续既有序列，按预登记差值、收缩和
   incident-direction 门判定。
2. **Weak candidate**：在完整两带 wake 上用 owner quadrature
   `q=3/5/7` 组装 global continuous-P1 scalar normal-speed Gram system。
   几何速度为 `Σ N_i s_i n_i`。
3. weak operator 必须通过 manufactured FE recovery、rank/condition、
   weak orthogonality、quadrature contraction、finest change 和刚体协变。

只有 point 失败且 weak 全过才允许把 `c1b1` 记为 falsified、`c1b2` 记为
validated。若 weak 也失败，则转向 bounded finite-segment/thin-layer，而不是
引入未经锚定的 core radius。

## 执行结果

### Point candidate：FALSIFIED

两个未见细层继续了原有非收敛规律：

| 指标 | 结果 | GO 门 |
|---|---:|---:|
| tip 最细变化 | 0.053586 | ≤0.005 |
| tip 最后差值收缩 | 1.0085 | ≥1.3 |
| outer-edge 最细变化 | 0.007729 | ≤0.002 |
| outer-edge 最后差值收缩 | 1.0171 | ≥1.3 |
| incident-direction spread | 0.041898 | ≤0.005 |

翼尖序列最终达到 `0.4568→0.5103`，没有趋向有限 point value 的证据。
因此不能通过提高 local fit 阶数或缩小 epsilon 来继续该路线。

### Global weak P1 candidate：GO

| 指标 | 结果 | 门 |
|---|---:|---:|
| rank deficiency / condition | 0 / 11.105 | 0 / ≤1e5 |
| manufactured recovery | 1.94e-16 | ≤2e-12 |
| weak orthogonality | 2.02e-16 | ≤2e-12 |
| q3→5 / q5→7 change contraction | 3.567 | ≥1.05 |
| q5→7 finest relative change | 0.02519 | ≤0.05 |
| release speed max | 0.14621 | ≥0.001 |
| tangential leakage | 2.79e-17 | ≤2e-14 |
| rigid error | 8.60e-16 | ≤2e-12 |
| ledger / mutation | 0 / 0 | ≤2e-12 / 0 |

physical field 的 surface-L2 residual 随求积阶约为 `0.574/0.603/0.613`。
这个值没有被隐藏或当作拟合误差去优化：P1 是有限维 geometry state，本来不
应逐点复制含边界奇异的速度。获得收敛的是 weak coefficients 和其正交条件。

## Claim 改写

- `c1b1` point trace：falsified/frozen，禁止重走；
- `c1b2` global weak P1 normal mesh velocity：validated/frozen；
- `c1b` 表示裁决：validated/frozen；
- `c1` 以“weak normal release＋exact chronological injection”的证据版本
  validated/frozen；
- `c2` 只能消费 `c1b2` weak field，禁止调用 local point/vertex-star
  release 幅值作为物理边界速度。

这仍然只闭合 geometry representation，不产生 newborn strength、pressure
或 force。
