# N3 S3o：目标点中心化的近奇异 P2 边积分

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3a1`  
状态：**PREREGISTERED / NOT EXECUTED**

## ① 数据指纹与可动空间

S3n 的四通道身份、账本、拓扑、逐 band 表示、刚体客观性和失败语义全部
通过，唯一失败是求积 Cauchy。误差集中在 Krebs 顶点侧内点：

- `lambda=(0.0333,0.0333,0.9333)`；
- body 最近主犯面的 `h/sqrt(2A)=3.237e-3`；
- body `q52→72=9.478e-3`；
- wake `q64→80=3.416e-3`；
- wake 继续到 `q192→256` 后仅 `1.186e-9`。

这说明有稳定极限，但普通全区间 Gauss 没有解析目标点附近的窄层。可动空间
只允许剩余一维边积分的坐标/权重，不允许改势跳、径向原函数、finite-part、
来源账、core、压力或载荷。

## ② 学科机理

Johnson CR-3079 用闭式 `H/F` 递推处理高阶多项式面元影响系数，说明“堆普通
面元 Gauss 阶数”并不是生产算子的合理定义。

Montanelli 等针对二次三角边界元证明，近奇异目标需要先定位参考域中的奇异
原像，再做消去/continuation 与 transplanted Gauss。Johnston 等则给出
projection-centered sinh–sigmoidal 变换；其作用是移动复平面近奇异点，
恢复 Gauss 的快速收敛，而不是改变核。

## ③ 缺件还是错件

| 命题 | 裁决 |
|---|---|
| 四通道 physical velocity ledger | 9/10 支持，保持 partial |
| P2 doublet / sheet-average 公式无稳定极限 | 数据否定 |
| 普通全区间 edge Gauss 对近顶点仍高效 | 错件 |
| 目标点中心化的近奇异坐标 | 缺件 |
| 加 core、offset、提高冻结阶次 | 禁止 |

## ④ 预登记方案

保持解析径向约化和 owner finite-part 完全不变。对每条边
`x(s)=x0+s e`：

1. 求目标点到闭线段的最近坐标 `s0`；
2. 取无量纲最近距离 `d=|target-x(s0)|/|e|`；
3. 用 `s=s0+d sinh(t)` 与精确 Jacobian 把 `[0,1]` 映回有限 `t` 区间；
4. 在 `t` 上用冻结阶次 Gauss；
5. 目标落在边上或退化边时 fail closed。

门覆盖变换测度、常强度解析涡环、二次强度独立高阶参考、普通远场等价、
实际 S3n body/wake Cauchy、刚体客观性和非法几何。完整输入和阈值在
`near_singular_p2_edge_quadrature_cases.yaml`，执行前已冻结。

## S3o 执行结果：NO-GO

简单 sinh 路径在实际 S3n 快照上显著改善并通过原冻结最高阶次：

- body 最终变化 `1.747e-9`；
- wake 最终变化 `6.319e-9`；
- ledger/逐 band 精确，刚体误差 `1.639e-13`。

但更强的预登记制造点揭示它不可迁移：

- `q16→24` 最大变化 `1.769e-2`；
- 常强度解析三角涡环误差 `5.252e-4`；
- 普通 `q192→256` 参考本身也未解析该近顶点层。

这不是把制造点阈值设得“太严”。常强度面元有独立解析涡环真值，已经直接
证明候选在近顶点处错误。Montanelli Theorem 2.1 的幂次结论解释了原因：
conformal transplant 对 `f^k` 在 `k<=1` 时可超指数收敛，而当前边界涡项
保留 `1/r^3`，只移动节点并没有先消掉强奇异幂。

因此 `N3.1j3b6d18c2b3a1` 冻结为 falsified。允许的下一步不是提高阶次，
而是利用“P2 势跳沿直边严格为二次多项式”这一结构，按 Johnson `H/F`
思路将完整二次边界涡积分闭式抽出；transplanted Gauss 只留给已经做过
finite-part 消去的面积涡余项。
