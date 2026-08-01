# N3 从“给定环量切面”到“Kutta 选定环量”的必要桥接

日期：2026-07-28  
Claim：`N3.1j3b6d`  
前置证据：S2a 环量切面拓扑门为 GO。

## ① 病因定位

S2a 已证明闭合单值势 trace 不可能承载非零环量，而在分类 cut 复制两端
自由度后，给定 `Gamma` 可以同时恢复圆柱速度、压力和 Kutta–Joukowski 力。
但圆柱没有尾缘，`Gamma=0.8` 是解析输入，不是模型根据尾缘状态求出的未知量。

因此当前缺口不是“再做一个三维网格”，而是：

> actual-boundary 方程尚未证明能以尾缘 Kutta 条件选择环量。

若跳过这一层直接进入有限翼，任何环量误差都无法区分来自 cut 拓扑、
Kutta 符号、三维 tip wake、finite-base closure 还是离散压力，病因会重新
混在一起。

## ② 学科机理

Erickson 的 NASA TP-2995 将 lifting panel method 的势跃变、wake 与 Kutta
条件作为不同但耦合的组成部分；提高表面 doublet 阶次本身不会选择环量。

Hwang（CMAME 2000，DOI `10.1016/S0045-7825(00)00182-1`）以 Joukowski 和
van-de-Vooren airfoil 的解析解验证 Dirichlet 边界方法，并特别指出尖边附近
的奇异/近奇异积分是高阶边界方法必须独立处理的问题。

Xia–Mohseni（JFM 830, 2017，DOI `10.1017/jfm.2017.513`）区分了两种情形：

- cusped/sharp trailing edge 的新生片方向由尾缘切线确定，可先隔离 Kutta 门；
- finite-angle trailing edge 的方向不再唯一，必须把 unsteady Kutta、环量、
  质量和动量联立求解。

Le Provost 等（JFM 977, 2023，DOI `10.1017/jfm.2023.997`）进一步把
no-through-flow、Kelvin 和 unsteady Kutta 放在同一求解系统中，说明稳态
尖尾缘通过仍不能替代后续物质 wake。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| S2a 给定 `Gamma` 的圆柱门已验证 Kutta 选环量 | 错件 |
| 直接上三维有限翼能更快证明 body–wake 闭合 | 当前 NO-GO，病因不可辨识 |
| 尖尾缘解析 lifting-body Kutta 门 | 缺件 |
| 尖尾缘通过即可代表 NACA-2406 finite base | 错件，已由 N3.1j3b6c2 限定 |

## ④ 方案与 go/no-go

用无拟合 Joukowski 变换

\[
z=\zeta+\frac{b^2}{\zeta},\qquad
\zeta=\zeta_0+R e^{i\theta},\qquad R=b-\Re\zeta_0
\]

构造经过 `zeta=b` 尾缘 cusp 的有限厚度对称 airfoil。`alpha=5°` 时以

\[
\Gamma_K=-4\pi R U\sin\alpha
\]

令 `dW/dzeta` 与映射导数在 cusp 同时为零。只用同一 cut P2 势 trace
求导，执行一次 `Cp` 和一次压力积分，检查：

1. cut 跃变和闭路环量都等于解析 `Gamma_K`；
2. Kutta 分子在机器精度内为零；
3. 速度、`Cp`、`D=0`、`L=-rho U Gamma_K` 随网格收敛；
4. 上下尾缘侧 `Cp` 收敛到同一有限极限；
5. 常势平移不改变速度和力。

所有参数、网格、阈值和禁止项已在
`actual_boundary_joukowski_kutta_cases.yaml` 中于实现前冻结。通过只允许
进入 finite-angle/base wake 的下一预登记，不允许跳到三维生产或修改 N1。

## S2b 首次执行：局部 NO-GO

首次实现没有修改任何阈值。13 个门中 11 个通过：cut 跃变/环量、解析
Kutta 分子、全表面速度与 `Cp`、压力唯一成力、零阻、规范下的总力以及三族
收敛均通过。失败的是：

| 门 | 实际 | 阈值 |
|---|---:|---:|
| 常势平移后最大速度变化 | `8.419e-10` | `1e-12` |
| 256 面板上下尾缘侧 `Cp` 差 | `0.10957` | `0.02` |

该失败不是物理 Kutta 不连续。相同最近积分点上的解析上下侧 `Cp` 差只有
`1.936e-5`；全表面速度/Cp RMS 已分别为 `6.812e-4/1.247e-3`。普通 P2 的
误差集中在 cusp 最近点，并随 `32→2048` 面板从
`0.4719→0.01390`单调下降。

病因是 cusp 处

\[
\frac{d\phi}{d\theta}\rightarrow0,\qquad
\frac{dz}{d\theta}\rightarrow0,
\]

二者有由 Kutta 条件规定的共同一阶零点。普通端点 P2 同时插值三个值，却
没有编码端点导数为零；将两个独立近零导数相除会把局部截断和浮点规范相消
误差放大。这是**边界基缺 Kutta 约束**，不是网格或阈值问题。

## S2b1 有证据改写（实现前冻结）

只改 cusp 相邻的两个 trace 单元：

- cusp 在 `tau=0` 时用
  `N0=1-7tau²+6tau³, Nm=8tau²-8tau³, N1=-tau²+2tau³`；
- 该三次 enrichment 保留端点/中点/远端值并严格满足 `p'(0)=0`；
- 末单元镜像该基，严格满足 `p'(1)=0`；
- 几何与势使用同一 enriched basis，保持共同零点；
- 所有势导数先减去局部常势 anchor，再求导，规范量在除弧长前即消失；
- 其余单元仍为原 P2，Gamma、网格、积分、阈值和全部 gate 均不变。

这不是尾缘 epsilon 或压力修补，而是把解析 Kutta 正则性写入分类边界基。
完整 S2b1 预登记已追加到 YAML；若原门仍有一个失败，继续 NO-GO。

## S2b1 执行与 S2b2 规范状态分解

S2b1 将尾缘侧 `Cp` 差从 `0.10957` 降至 `1.942e-5`，明显低于原
`0.02` 门；速度/Cp/力也保持收敛。唯一剩余失败是常势平移后的速度变化
`1.443e-11 > 1e-12`。

虽然局部导数已使用 `phi_i-phi_anchor`，但若先在每个浮点节点上加
`7.25`，低位信息在相减前已经丢失；cusp 的小弧长再将该舍入误差放大。
因此 S2b2 的 claim 改写为：

> 全局势规范是独立 null-space 标量，不属于可微 trace 系数。表面梯度只消费
> gauge-free trace；需要势绝对值时才在求导后加规范。

该分解没有新增物理自由度，反而明确了后续三时刻 material potential history
必须保存的“可观测差值状态”和“时间规范状态”。S2b2 已在再次实现前登记，
原网格、公式、阈值和全部门保持不变。

## S2b2 最终结果

分离规范状态后，完整原始 S2b 门全部通过：

| 指标 | 256 面板结果 | 原阈值 |
|---|---:|---:|
| cut 跃变相对误差 | `0` | `1e-13` |
| 环量相对误差 | `1.126e-15` | `1e-13` |
| Kutta 分子残差 | `0` | `1e-13` |
| 表面速度 RMS | `1.634e-5` | `0.01` |
| 表面 Cp RMS | `4.078e-5` | `0.03` |
| 升力相对误差 | `4.892e-11` | `0.005` |
| 阻力绝对值 | `8.006e-11` | `0.002` |
| 尾缘两侧 Cp 差 | `1.942e-5` | `0.02` |
| 规范速度/力变化 | `0 / 0` | `1e-12 / 1e-12` |

速度、Cp、升力三族误差均随 `32/64/128/256` 面板严格下降。S2b 因此为
**GO**，但仅授权 steady sharp-cusp lifting-body bridge。结果同时冻结：

- 普通端点 P2 不含 Kutta 共同零点，不能直接承担 cusp 压力；
- classified cusp enrichment 与独立 gauge 状态是该边界表示的必要组成；
- NACA-2406 有实体 base 和两个有限角尾缘，不能继承这个尖尾缘 GO；
- 三维 wake、Kelvin 物质历史、LEV 释放和生产载荷仍保持未验证。
