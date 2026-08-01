# N3 S3ae：unsteady-Kutta material birth-flux oracle

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3`  
状态：**EXECUTED / GO（局部表示）**

## ① 病因定位

S3ad 已把当前 trace-only newborn construction 证伪：

- `Delta g` 的时间步阶仅为 `0.0603`；
- newborn P2 片涡量按 `dt^-0.950` 发散；
- junction total velocity 按 `dt^-0.898` 发散；
- actual condition 最大仅 `55.07`，weak residual 最大
  `3.67e-16`。

因此病灶不是代数求解器，而是 newborn 值迹没有携带形成片的物质环量通量。
可动空间只在开放节点 `...b3`；actual boundary、S3aa、S3t、S3q 及生产力
均冻结不动。

## ② 学科机理

对局部正交片面坐标，

\[
\boldsymbol\gamma=\nabla_s\mu\times\boldsymbol n
=-\mu_{,s}\boldsymbol t_\eta+\mu_{,\eta}\boldsymbol t_s .
\]

若 newborn 长度
\(h=u_g\,dt+O(dt^2)\)，有限形成片强度要求

\[
g_{\rm body}-g_{\rm released}
=\chi\,u_g\gamma_g\,dt+O(dt^2)
=\chi\,\dot\Gamma_g\,dt+O(dt^2).
\]

Xia–Mohseni 的 finite-angle sharp-edge 守恒式给出

\[
\dot\Gamma_g=u_g\gamma_g
=\tfrac12(u_{2-}^2-u_{1+}^2),
\]

所以系数来自 Kutta、环量、质量和动量，而不是 panel length、core 或力。
完整推导、假设和自由度计数见仓库根目录 `DERIVATION_PACKAGE.md`。

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| trace-only actual solve 自动给出 `Delta g=O(dt)` | falsified/frozen |
| `mu` 的 P2 表面梯度到片涡量身份 | validated/frozen，不改 |
| finite-angle 方向、强度、相对速度守恒式 | validated/frozen，不改 |
| material circulation flux 到 newborn P2 trace | 当前缺件 |
| 把 flux 方程直接追加到当前方阵 | 错件，会多出 `m` 个方程 |
| finite NACA-2406 base 可用单 junction 表示 | falsified/frozen |

## ④ 方案与 go/no-go

先做无 actual-body、无力的局部制造门：

1. 使用 validated finite-angle formation oracle 生成
   `gamma_g/u_g/dotGamma_g`；
2. 以 `h=u_g*dt` 构造规则平面 newborn strip；
3. 在现有 temporal-P2 容器中写入
   `[g0, g0+0.5*dt*dotGamma, g0+dt*dotGamma]`；
4. 直接用冻结的 `sheet_vorticity_barycentric` 计算片涡量；
5. 对 `dt={0.004,0.002,0.001,0.0005}` 检查
   trace 一阶、vorticity 零阶、通量恒等、镜像和速度尺度协变。

GO 只验证 coincident finite-angle sharp junction 的局部 birth-flux
表示，并授权下一步研究 topology-derived quotient test space。它不授权
actual-body 联立、close evaluation、finite base、压力、力、118 工况或生产。

NO-GO 时只能检查取向、公式或表示映射；禁止增加 wake length、epsilon、
core、damping、least squares 或载荷目标。

## S3ae 执行结果

预登记三组 finite-angle 状态、四个时间尺度均原样通过：

| 指标 | 最大值 |
|---|---:|
| trace 一阶阶数误差 | `2.1094e-15` |
| sheet-vorticity 零阶阶数绝对值 | `1.6720e-16` |
| P2 涡量向量对 `gamma_g` 误差 | `0` |
| consistent-mass 环量通量残差 | `0` |
| midpoint trace 残差 | `0` |
| sheet normal 取向误差 | `0` |
| 镜像残差 | `0` |
| 速度尺度协变残差 | `0` |
| 输入 mutation | `0` |

以 `u1+=-2, u2-=-1` 为例，
`gamma_g=-1.05643`、`u_g=1.41987`、`dotGamma_g=-1.5`。
当 `dt` 从 `0.004` 减到 `0.0005` 时，`Delta g` 从 `-0.006`
线性减到 `-0.00075`，而 P2 片涡量严格保持 `-1.05643`，没有 S3ad 的
`1/dt` 发散。

因此局部命题
`N3.1j3b6d18c2b3b3b2c2b2b3a` 可判为
`validated/frozen`。更广的父命题保持 `partial`：该门没有识别 current
actual-body 方程中应被 Kutta 条件替换的 `m` 个 test modes，也没有计算
body–wake 联合速度。下一可动节点是 topology-derived quotient test space，
而不是生产全扫。

## 后续证据更新（S3af）

S3af 纠正了上述最后一段的自由度解释：完整 cut 有 9 个节点，但只有
7 个独立 jump 模态；body matrix 本身满秩 81。Euclidean 与 surface-L2
quotient 给出不同解且遗留非零完整 BIE residual，因此“删除 test modes”
已判 `falsified/frozen`。S3ae 的局部 birth-flux 结论不变；新的下一路径
是保留全部 body BIE，以独立 active-wake trace 加一套物理 Kutta closure
替代 Morino。
