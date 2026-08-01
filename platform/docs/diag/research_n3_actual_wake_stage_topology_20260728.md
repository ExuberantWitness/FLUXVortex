# N3 S3s：typed actual-wake topology 与 chronological P2 双射

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b2a`  
状态：**EXECUTED / GO**

## ① 病因定位

S3r 已排除 actual-boundary 和四通道速度本身：weak residual 为
`3.493e-16`，ledger closure 为 `0`。直接组成失败的最前置原因是：

- 实际 newest wake edge 是单侧、由 body cut 给定的 external boundary；
- 既有 assembly topology 只认识 `zero` 或恰好两侧的 `interface`；
- global continuous-P2 state 与 `MaterialWakeBand.potential_jump_rows`
  之间没有双向映射。

这里存在一个可检验的维数恒等式。若有 `B` 个 chronological bands、每行
`n` 个 P1 span vertices：

```text
chronological scalar nodes = (2B+1)(2n-1).
```

把全部 band 作为一个连续 P2 三角网格，vertex 与 edge-midpoint DOF 的总数也
恰为 `(2B+1)(2n-1)`。所以问题不是缺少可调参数，而是缺少一个由
chronological integer identity 定义的**双射**。

## ② 学科机理

Krebs 的 DDE wake 是 gapless chronological elements；旧 element strength
保留 material identity，最新 row 与当前 TE 相连。Dziuk–Elliott 与
Elliott–Venkataraman 的 evolving-surface FEM 则要求 scalar FE state 明确附着
于当前 triangulation 的共享实体。两者共同排除：

- 按世界坐标排序或距离焊接；
- seam 两侧取平均；
- 只动 geometry 却假定 scalar representation 自动更新。

对本仓库的结构化 P2 band，可由拓扑直接建立映射：

- P1 vertex `(time row k, span vertex j)` → chronological `(2k, 2j)`；
- span-edge midpoint → `(2k, 2j+1)`；
- time-edge midpoint → `(2k+1, 2j)`；
- cell diagonal midpoint → `(2k+1, 2j+1)`。

这是一个 permutation，不含拟合、正则或物理常数。

一手来源：

- Krebs dissertation (2021)，§3.2–3.3；
- Krebs, Bramesfeld & Cole (2022)：
  https://doi.org/10.3390/aerospace9010028
- Dziuk & Elliott (2013)：
  https://doi.org/10.1017/S0962492913000056
- Elliott & Venkataraman (2015)：
  https://doi.org/10.1002/num.21930

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| frozen MaterialWakeHistory geometry/rows | 已验证输入 |
| closed assembly 的 two-sided interface validator | 在其原范围正确，冻结 |
| 用 closed validator 表示 one-sided body attachment | 错范围，不改原节点 |
| typed actual-wake P1/P2 topology | 缺件 |
| global P2 ↔ chronological rows 精确双射 | 缺件 |
| coordinate/proximity map 或 seam average | 错件，禁止 |
| owner-aware velocity与transport | 下一门，不混入 |

## ④ S3s 预登记

在 S3n 两带曲尾迹上冻结：

- 组合公式与完整 permutation；
- body/old/root/tip 的 disjoint typed boundary；
- baseline geometry/rows/face_mu round-trip；
- 非零内部 scalar 反事实 round-trip；
- 通用刚体变换拓扑不变；
- geometry seam、scalar seam、face pattern 三类失败语义。

只有全门 GO，才允许进入 owner-aware velocity/P2 transport。阈值与禁止项已在
实现前冻结于 `actual_wake_stage_topology_cases.yaml`。

## ⑤ 执行结果与 claim 裁决

冻结配置未改，9/9 门通过：

| 指标 | 结果 |
|---|---:|
| band / span vertices | `2 / 5` |
| chronological scalar layout | `5 × 9` |
| global P1 / P2 DOFs | `15 / 45` |
| P2 permutation error count | `0` |
| boundary-role overlap | `0` |
| baseline geometry / rows / face_mu | `0 / 0 / 0` |
| nonterminal scalar round-trip / face_mu | `0 / 0` |
| rigid geometry / topology mismatch | `0 / 0` |
| invalid histories failed closed | `3/3` |

三类失败分别是 `1e-3` geometry seam、`1e-3` scalar seam 和不受支持的
local-face parameterization；均在任何速度或推进前终止。

因此 `N3.1j3b6d18c2b3b2a` 在以下窄义范围
`validated/frozen`：

> chronological integer topology、one-sided external body boundary，以及
> global continuous-P2 state 与 material rows 的无参数精确双射。

这项 skill 指令使本阶段保持在“先证据、后组成”的路径上：没有为了让现有
closed assembly 通过而修改冻结节点，而是新增了一个范围独立、可证伪的
actual-wake topology。

速度查询、vertex-star projection、P2 transport matrix、geometry advance、
stage re-solve、压力和力均未进入。下一节点只能是
`N3.1j3b6d18c2b3b2b` 的 owner-aware stage 离散。
