# N3 S3q：线性面积涡 finite-part 的解析边矩

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3a2b`  
状态：**PREREGISTERED / NOT EXECUTED**

## ① 病因

S3p 已把完整 P2 边界涡解析抽出并通过解析 ring，但制造近顶点仍有
`6.98e-3` 的阶次变化。owner/coplanar 算子中唯一剩余数值项是线性面积涡的
finite-part 边积分。

## ② 机理

Montanelli 等对强奇异二次三角形使用 Taylor subtraction 与 continuation，
把奇异主体化为一维边界项；Johnson 的 `H/F` 递推说明这些多项式边矩存在闭式
端点表达。

对 P2 面，`gamma=grad_s(mu)×n` 为线性函数。以目标到边直线的投影为
`u=0`：

- `gamma_edge-gamma_point` 与径向矢量的叉积至多二次，使用
  `I0/I1/I2`；
- `gamma_point` 的 finite-part log 项至多一次，使用 `J0/J1`。

## ③ 判定

剩余的是数值积分缺件，不是物理状态缺件。边界涡 `a2a` 已冻结，禁止重写。

## ④ 预登记

实现五个端点矩，并用数值微分验证其原函数；随后对适中 owner 点与原 256 阶
算子对照，再检查制造近顶点的阶次不变性、实际 S3n、刚体和 edge failure。
完整阈值在 `analytic_p2_sheet_finite_part_cases.yaml`。仍无压力、力或结构。

## S3q 执行结果：GO

全部冻结门通过：

- 五个 `I/J` 端点原函数的数值导数相对误差 `4.74e-9`；
- 适中 owner 点与独立 standard-256 误差 `7.77e-16`；
- 制造近顶点 `q8` 与 `q24` 完全一致；
- constant ring `2.84e-14`；
- actual body/wake `5.68e-12 / 2.94e-10`；
- ledger closure `0`，逐 band 表示 `1.73e-14`；
- rigid objectivity `1.46e-13`；
- 两个 edge target 均 fail closed。

因此 `N3.1j3b6d18c2b3a2b` validated/frozen，且可将 `a2` 的完整“解析
boundary + analytic coplanar finite-part”数值组成冻结。该 GO 仍不是
四通道父 claim 的晋升证据；下一步必须用新算子完整重跑 S3n 的 10 项门。
