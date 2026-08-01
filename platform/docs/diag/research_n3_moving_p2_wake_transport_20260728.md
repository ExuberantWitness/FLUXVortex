# N3 S3k：moving-curved multi-patch P2 材料输运

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b2b`  
状态：**EXECUTED / NO-GO**

## ① 病因定位

S3j 的 `M μ_dot+C μ=0` 只在静止平面上通过。S3h 已有移动几何 Heun
stage，N3.1j4c6 已有显式 patch interface，但当前没有组件把三者组合：

- 每个几何 stage 重建 `M(t),C(t)`；
- 只按声明接口焊接 P1/P2 DOF；
- 用同一个全局 μ vector 更新所有 patch trace。

缺的是 moving-geometry composition，不是再调空间算子。

## ② 学科机理

ALE evolving-surface FEM 的 basis transport 和 relative-velocity term 都定义在
当前 evolving triangulation；因此将静态矩阵留到下一 stage 是错件。
S3h 的 Heun stage geometry 与 c6 的显式接口身份已经验证，必须作为冻结输入。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| continuum ALE law / stationary P2 operator | 已验证，冻结 |
| stage-current `M(t),C(t)` | 缺件 |
| explicit-interface P2 weld | 缺件 |
| proximity weld / seam averaging | 错方向，禁止 |
| actual induced velocity / pressure / force | 后续，本门禁止 |

## ④ 预登记

规范面为四个命名 chronological strips，移动曲面
`X=[s,r,0.08 sin(pi s)sin(pi r)sin(1.3t)]`，相对材料速度由参数流
`s_dot=0.3sin(pi s), r_dot=-0.2sin(pi r)` 推前得到。`2/4/8` Heun
时间族检查 patch/monolithic 等价、时间收缩、共边与零边界、刚体客观性和面积。

阈值冻结于 `moving_p2_wake_transport_cases.yaml`。通过前不得接 actual
body/source/wake velocity。

## ⑤ 执行结果与病因裁决

不改冻结配置，结果只有一个失败门：

| 指标 | 结果 | 门 |
|---|---:|---:|
| patch/monolithic matrix/final | 0 / 0 | PASS |
| shared trace jump | 0 | PASS |
| Heun time Cauchy | 4.428 | ≥3.5 PASS |
| finest relative L2 | 0.002265 | ≤0.01 PASS |
| rigid geometry/scalar | 2.48e-16 / 3.47e-17 | PASS |
| minimum area ratio | 1.000004 | ≥0.95 PASS |
| zero boundary trace | **3.015e-4** | ≤2e-12 **FAIL** |

未约束 CG trial space 会更新显式 `zero` boundary DOFs；连续方程的特征线沿
边界并不能让离散边界子空间自动不变。故 `c2b2b` 作为“无 boundary-role
约束仍充分”falsified/frozen。

下一候选只能把 YAML/topology 已声明的 `zero` 与未来 `body-attachment`
trace 作为 essential scalar boundary。禁止清零事后结果、放宽阈值、加扩散或
把漂移归到曲率。
