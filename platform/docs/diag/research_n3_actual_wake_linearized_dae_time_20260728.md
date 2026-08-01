# N3 S3v：actual fixed-geometry DAE 的显式/隐式时间裁决

日期：2026-07-28  
主 Claim：`N3.1j3b6d18c2b3b3b`  
反事实 Claim：`N3.1j3b6d18c2b3b3c`  
状态：**EXECUTED / EXPLICIT-GO**

## ① 病因定位

S3u 已证明实际 stage 有两条非零耦合：

```text
free material P2 y ──actual boundary solve──> body trace g
g_dot ──M_fb──> free P2 rate
```

因此 endpoint clamp 被证伪，但这不等于必须做 geometry fixed point。
需要先回答更窄的问题：在同一实际 fixed geometry 和 actual velocity
线性化上，constraint-consistent explicit RK 是否已经满足时间阶、约束和局部
稳定门？

## ② 学科机理

对 semi-explicit index-1 DAE，half-explicit RK 的关键不是“显式”二字，而是
每个 differential stage 都有相容的 algebraic stage。Arnold–Strehmel–Weiner
证明这种构造可保持 underlying explicit RK 的阶；Vijalapura 等则排除把普通
ODE splitting 加 endpoint projection 当作二阶替代。

Roccia 的 previous/present/averaged-time 分类给出方向性先验：previous-time
explicit 是合法候选，present/averaged-time 才必然引入迭代。Kovács–Power
Guerra 的 implicit ALE 结果作为刚性/稳定失败后的参考，不是默认生产答案。

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| endpoint solve/clamp | 错组成，已证伪 |
| actual free→body affine map | 缺少完整局部测量 |
| actual `M,C` | S3t fixed-stage 已可提供 |
| constraint-consistent explicit midpoint | 缺件候选 |
| implicit trapezoidal | 同问题反事实参考 |
| nonlinear geometry fixed point | 本门不裁决 |

## ④ S3v 预登记

用 36 个 free-P2 单位基向量直接重解 frozen actual-boundary 方程，得到精确仿射
映射

```text
g = G y + c.
```

单位幅值是线性叠加基，不是小扰动差分步长。另用全自由度 signed combination
做未参与建图的 direct-solve 反事实。

在 S3t actual `M,C` 上消去 algebraic trace：

```text
(M_ff + M_fb G) y_dot
  = -(C_ff + C_fb G)y - C_fb c.
```

同一初态、`T=0.05`、`2/4/8/16` 步族比较：

- constraint-consistent explicit midpoint；
- implicit trapezoidal；
- augmented affine matrix exponential 真值。

预登记门同时检查 algebraic/weak residual、二阶 Cauchy、真值误差、显式局部
稳定多项式缺陷和输入不可变。只有显式全部通过，才允许进入**非线性 actual
previous-time prototype**；这仍不验证生产压力或力。

## ⑤ 执行结果

冻结输入未改，10/10 门通过：

| 指标 | 结果 |
|---|---:|
| actual unit-basis solves | `36` |
| base / signed counterfactual trace 重建 | `0 / 1.749e-15` |
| actual-boundary weak residual | `4.318e-16` |
| effective mass rank deficiency / condition | `0 / 11.4558` |
| reduced spectral radius | `16.3678` |
| algebraic / free weak residual | `0 / 5.908e-16` |
| explicit Cauchy | `4.00359` |
| implicit Cauchy | `3.96612` |
| explicit / implicit 最细真值误差 | `1.456e-5 / 7.258e-6` |
| coarsest explicit polynomial defect | `1.1866%` |
| finest explicit–implicit difference | `2.182e-5` |
| input mutation | `0` |

显式 midpoint 与隐式 trapezoid 都表现为二阶；显式候选在最粗步长仍只有
`1.19%` 局部稳定多项式缺陷。依据预登记的分层决策，结果为
**EXPLICIT-GO**：当前 fixed-geometry actual DAE 没有给出必须 nonlinear
geometry fixed point 的证据。

## ⑥ Claim 边界

新增窄义 validated/frozen 子节点：

> actual fixed-geometry affine trace map、有效 DAE mass，以及
> constraint-consistent explicit midpoint local time oracle。

父 half-explicit claim 仍为 partial，因为本门冻结了 geometry 与 physical
velocity。下一门必须实际更新：

1. free-P1 geometry；
2. actual four-channel velocity；
3. global-P2 scalar；
4. 每个 stage 的 body algebraic constraint。

原 `55.20%` geometry projection residual 和 `0.3295` relative-normal
diagnostic 继续保留。implicit 分支没有被证伪；只有 nonlinear prototype 出现
残差漂移、步长敏感或刚性证据时才可晋升。
