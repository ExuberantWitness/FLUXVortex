# N3 S3y：actual attachment release 与 newborn 状态注入

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c1`  
状态：**EXECUTED / GO（broad parent remains partial）**

## ① 病因定位

S3x 证明了固定两带 actual state 的非线性时间推进是二阶，但其算法和状态维数
始终不变：

- `band_count=2`；
- newest P1/P2 row 始终为 body-essential；
- `_geometry_history` 以 `zip(..., strict=True)` 重建相同数量的 bands；
- 不存在旧 P2 state 到新增 chronological rows 的扩维映射。

实际 S3n 指纹进一步排除了“步后直接 append”：

| 指标 | 结果 |
|---|---:|
| static body attachment speed | 0 |
| free-wake P1 speed max | 0.1157399 |
| 旧/新 body edge 重合时 append | `triangle is degenerate` |
| retiring edge one-sided release speed | 0.005488–0.192419 |

所以零面积不是需要 epsilon/regularization 的数值问题，而是生命周期顺序错误：
旧 body row 必须先转为 free/released，随后才有新的 body-essential row。

## ② 学科机理

Krebs (2021, §3.2–3.3) 明确把新 wake elements 定义在前一时刻和当前时刻
trailing edge 包围的区域；wake 是 gapless sheet，body circulation 连续进入
wake。Krebs, Bramesfeld & Cole (2022) 进一步说明：新生 row 的 strength 在
生成时刻求得，之后保持其 material identity，几何拉伸只改变其表示系数。

Roccia et al. (2024, Appendix C2) 的 previous-time 路线支持显式几何释放：
先有已求得的 circulation，再对流 wake 并传播新 segments。但它没有授权用
端点平均生成新生 strength。

Dumoulin, Eldredge & Chatelain (JFM 2023) 将 shed-sheet strength 与 body
no-through-flow、Kelvin theorem 和 unsteady Kutta condition 放在同一方程组。
因此 newborn middle/current rows 必须是后续 coupled stage 的具名未知量；
本门只能要求它们显式传入，不能猜测。

一手来源：

- Krebs dissertation (2021)，§2.1.3、§3.2–3.3；
- https://doi.org/10.3390/aerospace9010028
- https://doi.org/10.5194/wes-9-385-2024
- https://doi.org/10.1017/jfm.2023.997

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| S3s chronological P1/P2 topology | 正确，冻结 |
| S3t fixed-stage owner-aware velocity | 正确，冻结 |
| S3x fixed-band nonlinear midpoint | 正确，冻结 |
| 步后用同一静止 body edge append | 错生命周期，零面积 |
| retiring body row → free row 的角色迁移 | 缺件 |
| old 45 P2 DOFs → augmented 63 DOFs | 缺件 |
| newborn middle/current row 的 coupled solve | 下一缺件，不混入 S3y |
| epsilon offset / scalar average / smoothing | 无机理支持，禁止 |

## ④ S3y 预登记方案

在冻结 S3n 两带 actual state 上：

1. 用同一 owner-aware 四通道物理速度，对 retiring attachment 做单侧
   vertex-star 法向外推；该行不再使用 body velocity。
2. 分别构造 `dt/2` 与 `dt` 的有限 released edge；新 current body edge 保持
   规范体 cut。
3. 把旧 body row 变成 augmented topology 的内部 seam，并新增 current
   body-essential row。
4. old P1/P2 state 只按 chronological integer identity 精确注入；不按坐标
   搜索、不平均。
5. newborn upstream row 必须等于 old released trace；middle/current rows
   使用显式非平凡输入，只验证 round-trip，不在本门求解。
6. 执行通用刚体变换与三类 fail-closed 反例。

GO 只授权下一门的 coupled newborn midpoint/full strength 与多次插入时间推进；
不授权压力、力或生产模型。

## 执行结果

预登记门原样执行，`10/10 GO`：

| 指标 | 结果 |
|---|---:|
| bands / P1 / P2 | `2→3 / 15→20 / 45→63` |
| release speed min/max | 0.005488 / 0.192419 |
| release tangential leakage | 6.99e-18 |
| release fit condition number | 13.38 |
| naive coincident append failure | 1/1 |
| half/full newborn min area | 6.86e-6 / 1.37e-5 |
| P1/P2 injection | 0 / 0 |
| seam / P2 round-trip / role overlap | 0 / 0 / 0 |
| rigid geometry / release velocity | 0 / 6.66e-16 |
| invalid failures | 3/3 |
| inferred scalar / epsilon / mutation | 0 / 0 / 0 |

因此零面积病因被闭环：正确组成是 retiring-row release 和新的
body-essential row，而不是几何偏移。chronological state 扩维也不需要平均或
坐标焊接。

## 保留诊断与 claim 裁决

release least-squares 虽满秩且条件数良好，但最大逐点相对残差为 `68.80`；
这是局部样本穿越零值时被小分母放大的指标。以最大绝对残差除最大输入法向
速度的整体指纹约为 `55.2%`，与 S3t 已保留的 projection residual 一致。

这不推翻本门的拓扑/代数结论，但也不能证明 retiring edge 的物理速度已空间
收敛。因此：

- broad `N3.1j3b6d18c2b3b3b2c1` 保持 partial；
- `c1a` 仅冻结 full-rank normal operator、角色迁移和 P1/P2 注入恒等式；
- 新增 `c1b` 检验 one-sided boundary limit 的 query/span refinement 与独立
  reference；
- `c2` 必须依赖 `c1b`，不得直接使用 S3y 的 release 幅值进入长期时间推进。

这是 research-pipeline 对结果的实际影响：GO 没有被自动等同于 broad claim
晋升，未预先设门的高残差被保留并转成下一条可证伪命题。
