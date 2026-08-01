# N3 S2d：有限钝尾缘不是一个有限角 junction

日期：2026-07-28  
Claim：`N3.1j3b6d`  
前置证据：S2c 单有限角尖锐 junction 守恒形成门为 GO。

## ① 病因定位

S2c 只证明：当两条入射边界涡片在同一个尖锐材料点相交时，可由守恒式联合
确定一条新生片的方向、强度和相对运动。NACA-2406 使用标准开尾缘厚度式，
其上、下尾缘角由实体 base 分开：

\[
h_{TE}/c=0.00126.
\]

因此直接复用 S2c 会删除两个已观测几何量：

1. 两个不同的新生剪切层起点；
2. 两起点之间的 base 控制体。

病因挂在 `N3.1j3b6d` 的 wake/base 表示拓扑，可动空间只允许只读几何、
可辨识性与 sharp-limit continuation。压力、环量、N1 和生产力均不可动。

## ② 学科机理

Kemp（NASA TM-80104）对钝尾缘分别给出 B-1 bisector、B-2 双角点方向与
B-3 非线性压力相等三种替代 closure；它们不是同一个恒等式。

Martinez-Cava 等（*Aerospace Science and Technology* 113, 2021,
106730；DOI `10.1016/j.ast.2021.106730`）明确描述：上、下边界层在钝尾缘
分别形成两条剪切层，在下游 confluence point 汇合，两者之间是低压、
低动量 base region。Thomareis–Papadakis（*Physics of Fluids* 29,
014101, 2017；DOI `10.1063/1.4973811`）的 DNS 进一步显示，blunt edge
产生独立的周期性全局涡脱落，并反向锁定上游剪切层；这不是一个尖尾缘片的
小位置误差。

Xia–Mohseni 的推导仍是同点单 junction 控制体。Wang 等的 pressure Kutta
则是非线性 circulation closure；它不能凭一个压力标量生成两个空间起点。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 把两角中点作为单一 wake 起点 | 错件，删除有限 base |
| 给单一 wake 规定 bisector | Kemp B-1 comparator，不是普适机理 |
| 用上下压力相等反推出全部 wake 状态 | 不可辨识，B-3 只是一项非线性 closure |
| 两个材料锚点＋base/confluence 状态 | 缺件 |
| 直接加入经验 base 压力 | 禁止，属于未立证 N2 黏性闭合 |

## ④ 方案与预登记

在实现前冻结标准 NACA-2406 开/闭尾缘系数之间的几何 continuation：

\[
a_4(f)=-0.1036+f(-0.1015+0.1036),\quad
f\in\{1,0.5,0.25,0\}.
\]

对每个算例解析计算两个角点和切向。对任意单一材料起点 \(q\)，其同时附着
两个角点的最优 minimax 残差严格为

\[
\min_q\max(\|q-q_u\|,\|q-q_l\|)
=\frac12\|q_u-q_l\|.
\]

所以非零 base 下该残差不可能为零；两个显式起点可使附着残差为零，但这并不
给出两条片的强度、方向、下游汇合、base 压力或物质历史。B-2 的单方向
minimax 角残差也分别报告，但不冒充一般 unsteady Kutta。

完整输入、解析身份、阈值、GO/NO-GO 与禁止项见
`actual_boundary_blunt_base_topology_cases.yaml`。本门通过也只允许打开
“双 front＋base/confluence”动力学 claim，不允许接入生产。

## S2d 执行结果与 claim 改写

四级 continuation 全部通过：

| base fraction | `h_TE/c` | 最优单起点残差/c | B-2 单方向最优错角 |
|---:|---:|---:|---:|
| 1.00 | `0.001260` | `0.000630` | `4.0042°` |
| 0.50 | `0.000630` | `0.000315` | `4.0758°` |
| 0.25 | `0.000315` | `0.0001575` | `4.1117°` |
| 0.00 | `0` | `0` | `4.1475°` |

所有非零 base 的归一单起点残差严格为 `0.5`，几何恒等式最大误差
`7.16e-18`；两个显式锚点的附着残差为 `0`。`f=0` 时角点严格重合，但
上下切向仍形成 `8.295°` 的有限角尖缘，因此不是把几何退化成平板，而是
干净回到 S2c 单 junction 范围。

据此：

- `N3.1j3b6d10`：“一个单 junction 新生片直接代表有限 base”，
  `falsified/frozen`；
- `N3.1j3b6d11`：“二维有限 base 至少需要双材料 front 或等价有限宽界面，
  并保留 base/confluence region”，仅在拓扑必要性范围
  `validated/frozen`。

这仍不验证两 front 的环量分配、方向、相对运动、汇合历史或 base 压力；
它们是下一项 `partial/open` 动力学 claim，而不是本门可补的常数。
