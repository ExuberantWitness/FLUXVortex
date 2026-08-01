# N3 S3ad：actual wake birth-junction finite-velocity decision

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b2`  
状态：**EXECUTED / NO-GO**

## ① 病因定位

S3ac 的首个放大不是 algebraic matrix、P2 mass 或 off-plane：

- step 1 midpoint 的 actual matrix condition 为 `55.06`，势跳仅 `0.0926`，
  但 total velocity 已由 `0.990` 增至 `13.92`；
- step 2 initial 的势、几何和 actual residual 仍正常，body-doublet 与
  wake-sheet 却分别达到 `1.3171e7` 和 `1.3168e7`，相减后仍残留
  `1.4003e4`；
- 之后才出现势跳、geometry、mass condition 和投影残差的指数反馈。

数据把病因挂到 S3aa/S3t 的**组合适用边界**：S3aa 只保证 attachment
potential trace，S3t 只保证有限固定快照的分通道账；没有节点证明 newborn
厚度趋零时，body-bound sheet 与形成中 wake sheet 的涡量和速度保持有限。

可动空间只在开放的 birth-junction composition。冻结的 actual equation、
S3aa direct 等价、S3t 四通道身份及 S3q 片上有限部都不修改。

## ② 学科机理

Ramesh 的非定常薄翼研究指出，离散时刻的 shed wake 应是连续涡量分布；
涡量跨 trailing edge 连续时，尾缘速度有限、压差趋零。只让一个 point/blob
或值迹承担 Kutta 会产生跨尾缘涡量不连续。

Xia & Mohseni 对有限角尾缘进一步把形成片的方向、强度和相对速度与
unsteady Kutta、环量、质量和动量联立；因此 newborn geometry 与 strength
不能由事后 panel-length 规则替代。

即使连续问题已适定，Helsing–Ojala、QBX、三维 close-evaluation 以及
harmonic density interpolation 都说明：贴近 source surface 时，普通分项
求值会受 nearly-singular kernels 与 cancellation 破坏；应使用一侧局部展开、
密度减法或等价联合有限部表示。

一手来源：

- Ramesh, *On Satisfying the Kutta Condition in Unsteady Thin Aerofoil
  Theory*, arXiv:2205.08647；
- Xia & Mohseni, *J. Fluid Mech.* 830 (2017),
  https://doi.org/10.1017/jfm.2017.513；
- Maskew et al., NASA CR-159312 (1980),
  https://ntrs.nasa.gov/citations/19800018783；
- Helsing & Ojala, *J. Comput. Phys.* 227 (2008),
  https://doi.org/10.1016/j.jcp.2007.11.024；
- Klöckner et al., *J. Comput. Phys.* 252 (2013), arXiv:1207.4461；
- Pérez-Arancibia, Faria & Turc, *J. Comput. Phys.* 376 (2019),
  https://doi.org/10.1016/j.jcp.2018.10.002；
- Khatri et al., *J. Comput. Phys.* 423 (2020),
  https://doi.org/10.1016/j.jcp.2020.109798。

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| S3aa 一次 actual solve 的代数迹身份 | validated/frozen，不改 |
| S3t 有限固定快照四通道账 | validated/frozen，不改 |
| 值迹连续自动保证 birth-limit 有限速度 | 错件，已被 S3ac 反例证伪 |
| `g_new-g_old=O(dt)` / junction 涡量连续 | 尚未验证的物理缺件 |
| body 与 newborn 分开求近奇异场再相减 | birth limit 中不充分 |
| 联合密度减法/QBX/finite-part | 数值缺件，但必须排在物理连续性之后 |
| 调 wake panel length、core、epsilon、damping | 无权进入 |

## ④ 下一方案与 go/no-go

S3ad 只做无力尺度诊断：

1. 从同一冻结 S3n state 计算一次 release velocity；
2. 对 `dt = 0.004, 0.002, 0.001, 0.0005` 分别建立 half newborn geometry；
3. 每个尺度只做一次严格等价 direct actual solve；
4. 测量 attachment 值迹、`Delta g`、newborn P2 sheet-vorticity、actual
   condition/residual，以及当前分通道 total velocity；
5. 以 log-slope 判定 `Delta g=O(dt)`、vorticity/velocity 是否有界。

若 `Delta g` 不随 `dt` 消失或 vorticity/total velocity 随 `dt→0` 发散，则
当前 trace-only formation falsified，下一节点必须建立 unsteady-Kutta
vorticity-continuous birth condition。只有物理尺度门通过后，才允许以联合
密度减法/QBX 建 close-evaluation oracle。

本门不计算 pressure、force、LESP、flap cycle、118 cases 或结构动力学。

## S3ad 执行结果

预登记的四尺度族已原样执行：

| `dt` | newborn thickness | `|Delta g|∞` | newborn `|gamma|∞` | total `|u|∞` |
|---:|---:|---:|---:|---:|
| `0.0040` | `2.9241e-4` | `2.1974e-2` | `3.5815e2` | `1.0924e2` |
| `0.0020` | `1.4621e-4` | `2.0535e-2` | `6.7508e2` | `1.9664e2` |
| `0.0010` | `7.3103e-5` | `1.9770e-2` | `1.3087e3` | `3.6701e2` |
| `0.0005` | `3.6552e-5` | `1.9362e-2` | `2.5793e3` | `7.0668e2` |

拟合得到：

- `Delta g` 的 `dt` 阶仅为 `0.06025`，没有达到预登记的至少一阶；
- 片涡量发散阶为 `0.95001`；
- total velocity 发散阶为 `0.89810`；
- actual matrix 最大 condition 仅为 `55.0706`，最大 weak residual 为
  `3.674e-16`；
- attachment error 与输入状态 mutation 均严格为 `0`。

因此当前 trace-only newborn construction 的确具有非零 birth limit。
物理厚度按 `O(dt)` 消失，而势跳保持约 `O(1)`，所以表面梯度、片涡量以及
联合速度趋于发散。这不是线性方程病态、时间步太大或输入污染；S3ad
`NO-GO`，`N3.1j3b6d18c2b3b3b2c2b2b2` 判为
`falsified/frozen`。

下一可动节点只能是
`N3.1j3b6d18c2b3b3b2c2b2b3`：先建立 unsteady-Kutta/
vorticity-continuous formation condition，再允许研究联合 close
evaluation。禁止以 panel length、core、epsilon、clamp、damping 或
目标载荷吸收这个发散。
