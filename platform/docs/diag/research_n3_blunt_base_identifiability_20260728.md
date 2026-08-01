# N3 S2e：双 front 仍缺 base-side 动力状态

日期：2026-07-28  
Claim：`N3.1j3b6d`  
前置证据：S2d 双 front＋base/confluence 拓扑必要性为 GO。

## ① 病因定位

S2d 只回答了“有几个材料起点”，没有回答每个起点的新生片方向、强度和相对
运动。S2c 的有限角守恒式在每个 junction 都消费两侧入射速度。对钝尾缘角点，
一侧来自外表面边界层，另一侧来自尚未建模的 base recirculation。

所以当前病因不是少一条几何线，而是：

> 外侧 N1/actual-boundary 势流是否足以唯一确定双 front，还是必须显式拥有
> base-side/confluence 动力状态？

该病因仍挂 `N3.1j3b6d`。可动空间只包括解析守恒族与可辨识性；禁止选择任何
base 速度、压力或 force 参数。

## ② 学科机理

Xia–Mohseni 的有限角方向关系同时依赖两个 incident-side 速度；其中一侧速度
为零只是单侧停滞极限，不是一般钝尾缘定律。

Martinez-Cava 等（AST 113, 2021, 106730）把 blunt-base 流动明确描述为：
上下两条剪切层围成低压、低动量区域并在下游汇合。Thomareis–Papadakis
（PoF 29, 014101, 2017）的 DNS 显示 blunt-edge global shedding 会反向锁定
分离剪切层。这两项证据都否定“base 侧永远静止、可从几何删除”的普适身份。

Wang 等的 pressure Kutta 是非线性联立条件；在没有 material potential
history 时，不能用压力相等从 witness family 中事后挑一个。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 外表面速度＋角点几何唯一闭合双 front | 待证伪命题 |
| base-side 速度恒为零 | 仅一侧停滞极限，不能当普适件 |
| 用 lift/thrust 选择 base 回流 | 补丁式拟合，禁止 |
| base-side/confluence 动力状态 | 候选缺件 |

## ④ 方案与预登记

固定标准开尾缘 NACA-2406 的两个解析角点及外侧无量纲速度 `1`。保持这些
可观测输入完全相同，只令未观测 base-side 速度比取
`{0, 0.25, 0.5, 0.75}`。这些比值只是非唯一性 witness，不是候选参数。

对每个成员使用已经 validated 的 S2c 守恒式，要求全部方程和非负片侧速度门
通过；同时预登记：

- 形成方向 spread 至少 `30°`；
- 片强 spread 至少 `0.5`；
- 相对速度 spread 至少 `0.1`。

若同一外侧输入存在多组守恒可接受、且输出明显不同的 front 状态，则
outer-only closure 被证伪；下一件必须是 base-region/confluence 状态或可观测
输入，而不是 wake angle 常数。完整门见
`actual_boundary_blunt_base_identifiability_cases.yaml`。

## S2e 执行结果与方向裁决

四组 witness 全部满足局部角度、方向、Kutta、环量率和 signed momentum
恒等式，最大归一残差 `6.34e-17`，片侧最小速度为 `0`。上下角解析 wedge
只差 `0.000555°`，两侧 witness family 最大差 `3.57e-6`。

在外侧速度严格相同的前提下：

- 形成角 spread：`35.4105°`；
- 片强 spread：`0.66117`；
- 相对速度 spread：`0.14561`。

因此 `N3.1j3b6d12`（outer-only 唯一闭合双 front）
`falsified/frozen`；`N3.1j3b6d13`（base-side/confluence 状态或等价
可观测量是动力学可辨识性的必要输入）只在必要性范围
`validated/frozen`。没有任何 witness 被选为物理解。

### 主线分流

精确有限 base 动力学需要黏性 base-region 状态，当前不具备，继续在这里写
closure 只会变成无锚选择。与此同时，`h_TE/c=0.00126` 的未闭合支路不应阻塞
“空间涡态—统一压力”的独立三维基础研究。

因此下一阶段采用严格隔离：

1. 生产 NACA-2406 开尾缘几何保持 validated/frozen，不改；
2. finite-base 生产激活继续 NO-GO；
3. 三维 material-wake 入口只使用另行预登记的**角点重合有限角 lifting
   canonical**，验证展向 Kelvin、wake jump 和压力守恒；
4. 该 canonical 通过也不能宣称 open-base 已解决，最终仍需回接
   base-region 状态或给出独立误差界。

这不是用 sharp limit 替换生产几何，而是把两个尚未耦合的问题拆开验证，
防止一个局部未辨识量拖住 LEV 空间态主线。
