# N3 S3p：P2 边界涡闭式抽出

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3a2`  
状态：**PREREGISTERED / NOT EXECUTED**

## ① 病因

S3o 在实际快照通过，却在更靠近顶点的解析制造门失败。常强度 P2 没有面积
涡量，全部速度就是三条有限直涡段；其误差 `5.25e-4` 直接把主犯锁定为仍用
数值 Gauss 的 `1/r^3` 边界涡项。

## ② 机理

Johnson CR-3079 与 Krebs Appendix A 均把二次 doublet velocity 写成多项式
系数乘闭式 `H/F` 影响积分。Montanelli Theorem 2.1 则说明，transplant 对高
奇异幂仍需先做解析消去。

P2 在任一直边上严格为 `mu(s)=A s^2+B s+C`。以目标到直线的投影为原点，
边界涡核只需三个解析矩：

```text
I0 = integral du/(u^2+a^2)^(3/2)
I1 = integral u du/(u^2+a^2)^(3/2)
I2 = integral u^2 du/(u^2+a^2)^(3/2)
```

它们分别有 `u/(a^2 R)`、`-1/R`、`asinh(u/a)-u/R` 的端点原函数。

## ③ 判定

缺件是**完整二次边界涡解析影响**，不是新的物理涡或经验核。S3o 的简单
sinh 命题已 falsified，不重走。

## ④ 预登记

完整边界涡改用三个解析矩；原有面积涡 finite-part 公式不变，其余项继续使用
S3o 已定义的 target-sinh 坐标。制造近顶点、常强度解析涡环、实际 S3n、
普通远场、刚体和 edge fail-closed 门全部冻结在
`analytic_p2_boundary_vortex_cases.yaml`。本门无压力、无力、无结构。

## S3p 执行结果：窄义 GO / 总门 NO-GO

测量器首轮因 relative tolerance 提前停止而误报实际 body 绝对门，原文件保存为
`analytic_p2_boundary_vortex_results_initial_early_relative_stop.json`。
改成冻结绝对门后，结果为：

- edge P2 polynomial 重构 `1.11e-16`；
- constant-strength 解析 ring `2.84e-14`；
- regular field 等价 `5.38e-17`；
- actual body/wake 冻结阶次 `5.68e-12 / 6.32e-9`；
- rigid objectivity `1.47e-13`；
- 但 manufactured near-vertex `q16→24=6.977e-3`。

所以“完整 P2 边界涡闭式抽出”本身 validated/frozen；“它足以使完整
sheet-average 收敛”被数据否定。剩余误差只能来自面积涡 finite-part 的两类
边矩：

```text
(linear/quadratic numerator) / r^3
(constant/linear numerator) * log(r/L) / r^3
```

这些项同样有无参数端点原函数。下一门 S3q 只解析 `I0/I1/I2` 和
`J0/J1`，不再改边界涡或 target-sinh。
