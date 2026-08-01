# N3 S3u：actual-wake 时间耦合的 DAE 身份审计

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3`  
状态：**EXECUTED / COUPLED**

## ① 病因定位

S3t 只证明同一 fixed stage 内的 actual velocity 可以无歧义地进入：

- free-P1 法向 geometry velocity；
- body-essential geometry boundary；
- owner-aware global-P2 相对切向矩阵。

它没有证明把三个现成函数依次调用就是二阶时间算法。实际 scalar 方程含

```text
M_ff mu_dot_f = -(M_fb g_dot + C_ff mu_f + C_fb g),
```

而 `g` 不是规定函数，是 actual body–wake 势方程给出的 newest body trace。
因此必须先量化两条边：

```text
interior material-P2 state ──> algebraic body trace g
g_dot ──M_fb──> free material-P2 rate
```

任一为零，顺序算法可能解耦；两者均非零，问题就是 differential–algebraic
stage coupling，不能用末端 clamp 假装约束一致。

## ② 学科机理

Arnold–Strehmel–Weiner 的 half-explicit RK 理论说明：index-1 DAE 可以对
differential part 使用 explicit RK，但 algebraic component 必须在相应 stages
一致求解。Vijalapura–Strain–Govindjee 更直接证明，朴素的 differential /
algebraic sequential splitting 即使各自号称二阶，组合也会降为一阶。

另一方面，Roccia et al. Appendix C2 区分：

- previous-time wake convection：显式、不需 nonlinear iteration；
- present/averaged-time convection：需要迭代。

所以“必须 fixed point”和“无需耦合”都不是可先验接受的结论。要先识别当前
方程的耦合身份，再比较 constraint-consistent half-explicit RK 与真正的
present/averaged implicit residual。

一手来源：

- Arnold, Strehmel & Weiner (1993)：
  https://doi.org/10.1007/BF01388697
- Vijalapura, Strain & Govindjee (2005)：
  https://doi.org/10.1016/j.jcp.2004.08.015
- Roccia et al. (2024)：
  https://doi.org/10.5194/wes-9-385-2024
- Elliott & Venkataraman (2015)：
  https://doi.org/10.1002/num.21930

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| S3n actual algebraic body–wake solve | 已验证，冻结 |
| S3s actual topology / scalar bijection | 已验证，冻结 |
| S3t fixed-stage velocity discretization | 已验证，冻结 |
| S3m free/constrained P2 block | 已验证，冻结 |
| prescribed-velocity Heun / midpoint oracles | 各自在制造范围正确 |
| 把上述算子顺序调用后 endpoint clamp | 未验证；DAE 文献给出降阶风险 |
| actual constraint-consistent time stage | 缺件 |
| nonlinear geometry fixed point | 尚待裁决，非先验必需 |

## ④ S3u 预登记

只读 frozen S3n snapshot。选取一个对全部 open-boundary P2 DOF 为零的确定性
interior direction，以 `0.02/0.04` 两个幅值重解 actual body equation：

- body trace 响应必须可见；
- 两倍输入必须给两倍响应，证明测到的是线性代数耦合而非数值噪声。

再用 S3t 的同一 owner-aware P2 mass assembly、零 advection 隔离 `M_fb`：

- 正确 free/constrained block 残差应为 roundoff；
- full-rate-then-clamp 的归一残差应接近 1。

完整门已在执行前冻结于
`actual_wake_time_coupling_audit_cases.yaml`。本审计不推进时间、不评价力，
也不选择 explicit 或 implicit 候选。

## ⑤ 测量器审计与执行结果

首跑错误地绑定了“构造 actual solve 的 curved history”，而非预登记的
`base_solution.wake_history`。二者 geometry 相同，但前者还没有接收代数求解
后的 body trace，attachment identity 相差 `1.1056e-4`。该无效首跑已单独
保留；只修正状态绑定，不改阈值、方向或假设。

有效执行 8/8 通过：

| 指标 | 结果 |
|---|---:|
| global / body / free P2 DOF | `45 / 9 / 36` |
| strict interior P2 DOF | `21` |
| base attachment identity | `0` |
| free mass rank deficiency | `0` |
| `||M_fb||∞ / ||M_ff||∞` | `0.146536` |
| interior perturbation → body trace response | `9.22456e-3` |
| response directional rate | `0.461228` |
| 两倍输入线性相对误差 | `2.708e-14` |
| 正确 free/constrained block 残差 | `3.257e-16` |
| full-rate-then-clamp 残差 | `1.0` |
| actual-boundary weak residual | `3.914e-16` |
| 输入状态 mutation | `0` |

这不是“可能有一点耦合”：body-boundary mass block 已达 free block 的
`14.65%`，而 endpoint clamp 正好遗漏完整 `M_fb g_dot` 通道。

## ⑥ Claim 改写

`N3.1j3b6d18c2b3b3` 改为 partial，并拆成三个可证伪子命题：

1. 独立推进后 endpoint solve/clamp 足够：**falsified/frozen**；
2. 每个 RK stage 一致求代数态的 half-explicit 路线：**open**；
3. present/averaged-time 联立残差路线：**open**。

S3u 证明必须使用 constraint-consistent time method，但没有证明 nonlinear
fixed point 必需。下一门必须在同一实际问题、同一残差和同一时间族上比较
两条路线，禁止以迭代次数或总力选型。
