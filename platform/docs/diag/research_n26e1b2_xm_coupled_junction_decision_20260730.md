# N2.6e1b2 有限角尾缘联立算子研究裁决

日期：2026-07-30  
父 Claim：`N2.6e1b`  
裁决：`组成部分错 / 仅改写尾缘空间算子 / V4.1 不动`

## ① 数据指纹与唯一病灶

`N2.6e1b1` 已在预登记的 `64/128/256 panels-per-side`、相同运动、
相同时间步和相同尾迹核下完成。所有 Kelvin、无穿透、Eq. (7)、Eq. (8)
和线性系统残差均不大于 `1.17e-12`，但 `128 -> 256` 时：

| 局部出生量 | 冻结变化 |
|---|---:|
| lower TE trace | 7.33% |
| upper TE trace | 9.16% |
| mean trace / newborn length | 8.20% |
| jump / newborn strength | 9.36% |
| newborn endpoint y | 4.77% |
| newborn circulation | 0.38% |

最后一项的小变化来自长度下降与强度上升的误差抵消，不能替代局部状态
收敛。故病灶唯一挂到 `N2.6e1b` 的“有限角尾缘空间离散”：

> 常源面板加一个全翼均匀束缚环量自由度，只能在邻近控制点读取两侧速度；
> 它没有上下尾缘各自独立的连续涡片极限。继续加密同一最近控制点定义已被
> `N2.6e1b1` 证伪。

本轮可动空间仅为这个空间算子。实际 NACA0015 几何、Kelvin 账、材料尾迹
身份、运动学、IBL、压力、V4.1、Figure 12 和 Fig17/18/19 全部冻结。

## ② 一手文献机理

### Xia--Mohseni 的正式 JFM 算法

Xia & Mohseni (2017), DOI `10.1017/jfm.2017.513` 的正式出版版
§6.1（pp. 459--461）明确给出：

1. 闭合翼面用 `N_B` 个**线性变化束缚涡面板**表示，共有
   `N_B+1` 个节点强度；几何共点的上下尾缘节点仍是两个独立自由度；
2. 当前形成的尾迹是一个常强度 `g` 面板，历史尾迹才转成物质点涡；
3. 上一步的两侧尾缘节点状态只用于确定本步 `g` 面板方向和相对形成速度；
4. 本步 `N_B+1` 个束缚节点强度与 `gamma_g` 由 `N_B` 个无穿透方程、
   一个 Kelvin 方程和一个有限角非定常 Kutta 方程组成的
   `(N_B+2)` 阶线性系统**同时求解**；
5. 两侧状态是排除 junction 自身奇性的 Birkhoff--Rott 极限，不是邻近
   面板中心速度。

正式论文 PDF 的本次审计 SHA256 为
`67a96824e7025fba99532052941f9a82c7990aedb4df97b230f05089edb74c49`。

### 独立复现与数值含义

Dumoulin, Eldredge & Chatelain (2023), DOI `10.1017/jfm.2023.997`
采用同一结构：线性节点束缚涡片、一个均匀 forming panel，以及
no-flow/Kelvin/unsteady-Kutta 的块联立系统。该独立工作说明这些部分不是
可任意拆开的多个“补丁”，而是一个有限角 triple-junction 离散组件。

Sun & Wu (2022) 的**稳态**高阶边界元分析进一步指出，有限角 corner 的
端点速度是奇异数值问题，需要把几何角决定的幂次显式写入局部表示；直接把
最近控制点值外推到数学端点并不是合法的出生律。稳态 Kutta 下精确端点还
可能是停滞点，因此“换成端点总速度”也不能替代形成算子。该论文只授权
空间角点表示这一窄义判断，不授权非定常 forming-sheet 时间闭合。

仓库已有 `finite_angle_sheet_formation()`，其 Xia--Mohseni
角度、强度、环量率和动量恒等式属于
`N3.1j3b6d9 = validated/frozen`。本轮只复用它，不重写公式。

## ③ 缺件还是错件

判定为 **组成部分错**，不是“再缺一个 LESP 标量”：

- 错件：Riziotis 最近控制点 `mean/jump` 在当前有限角 actual surface 上
  被当作连续 junction 状态；
- 正确替代件：actual-surface 线性节点涡片与 forming panel 在
  no-through/Kelvin/Kutta 下联立；
- 未改部分：Riziotis 的材料尾迹、后续 IBL/double-wake 和统一 Bernoulli
  压力仍属于父候选，不能由本门提前宣称验证。

这也解释了为何不能回到 LESP：LESP 可以描述起涡临界或剪切层供给，但不能
提供两侧空间涡态、形成方向、位置运动和统一面板压力所需的状态。

## ④ 唯一候选

唯一候选为 `N2.6e1b2-XM-coupled-junction`：

> 在冻结 actual NACA0015 壁面上新增 shadow 表示：线性节点束缚涡片、
> 一个有限角 forming panel、Kelvin 行和 Xia--Mohseni Kutta 行同解；
> 形成方向/速度复用冻结的 `finite_angle_sheet_formation()`。

第一门只裁决空间算子，不输出力，也不读取任何实验响应。Xia 与 Dumoulin
均只把端点裁去“很小的 epsilon”，未披露可移植数值，因此 `epsilon`
不得成为常数。它只作为 `epsilon/h_TE -> 0` 的预登记数值收敛轴；若结果
依赖某个固定比值，候选直接 NO-GO。

预执行符号审计还发现，无 `g`、只有 Kelvin 的 finite-corner 初始化端点
不满足 `+6 deg` formation 的符号域，不能冒充可用的上一时步状态。因此
第一门明确降格为“已冻结可容许 previous canonical 下的联立空间矩阵
oracle”；即使通过，也仍缺 actual-boundary 时间历史 provider。该缺件必须
在下一独立 claim 中验证，不能由本门结果自动继承。

## 禁止项与适用边界

- 禁止恢复 `N2.6e1b0/b1` 的最近控制点路线；
- 禁止端点外推、滤波、松弛、clamp、经验 core 或 epsilon；
- 禁止按 Figure 12、Fig17/18/19 或总力选择网格/epsilon；
- 禁止改写已冻结的 finite-angle 代数符号；
- 禁止把本候选解释成 LEV/动态失速、分离尾迹或压力模型；
- 文献验证域主要是附着、小到中等攻角；通过只授权进入移动壁面和材料尾迹
  时间门，不直接授权目标翼载荷。

## 一手来源

- X. Xia and K. Mohseni, “Unsteady aerodynamics and vortex-sheet
  formation of a two-dimensional airfoil,” *Journal of Fluid Mechanics*
  830 (2017) 439--478, DOI `10.1017/jfm.2017.513`, especially
  Eqs. (2.5)--(2.6), (3.5), (5.1), (5.4), (6.1), and (6.3)--(6.8).
- M. Dumoulin, J. D. Eldredge and P. Chatelain, “A lightweight vortex
  model for unsteady motion of airfoils,” *Journal of Fluid Mechanics*
  977 (2023) A20, DOI `10.1017/jfm.2023.997`, especially §2 and Eq. (2.7).
- S. Y. Sun and G. X. Wu, “Inviscid flow passing a lifting body with a
  higher order boundary element method,” *Engineering Analysis with
  Boundary Elements* 136 (2022) 144--157,
  DOI `10.1016/j.enganabound.2021.12.012`, source PDF:
  `https://discovery.ucl.ac.uk/10142211/1/EABE%28HOBEM%29.pdf`.

## 2026-07-30 结果后证据校正

初版把 Sun--Wu 的作者缩写、题名和稳态适用域写错；以上正文与来源条目已经
显式校正。该书目错误不改变已经执行的 `N2.6e1b2` 数值矩阵，但会缩窄其
解释边界：

- Sun--Wu 只能支持“有限角角点需要几何幂次相容的空间表示”，不能支持
  `N2.6e1b2` 的非定常出生律；
- Xia--Mohseni 的来源算法要求由实际上一时刻提供 forming 方向/速度，
  而本门只使用冻结 canonical previous state；
- 所有非退化 current outputs 均至少有一侧越出 Xia--Mohseni 的
  no-backflow 适用域。

因此 `N2.6e1b2 = falsified/frozen` 的精确含义保持为：当前
“裁切端点节点值 + 冻结 canonical previous state + 常强度 forming panel”
组合不是可连续推进的空间 provider。它不证伪连续 Xia--Mohseni 机理，也
不授权把任意新的角点基函数直接宣称为非定常 closure。
