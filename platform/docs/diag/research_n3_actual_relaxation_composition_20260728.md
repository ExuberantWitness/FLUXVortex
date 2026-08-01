# N3 S3r：actual relaxation 直接组合充分性审计

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b1`  
状态：**EXECUTED / NO-GO**  
角色：无压力、无力的只读组成审计。

## ① 病因定位

S3q 已使 S3n 的 actual body–wake 四通道 sheet-velocity ledger 完整通过，
因此 `c2b3b` 可以启动。但“有速度账”不等于“现有组件可以直接组成自由尾迹
stage”。逐接口检查发现三个互相独立的断点：

1. `material_wake_assembly()` 把最新 body attachment 表达为单侧
   `interface:external-body-cut`；既有 assembly 顶点星投影和 P2 transport
   都调用只接受**两侧成对接口**的 closed-topology validator。
2. actual ledger 的 on-sheet 极限需要显式 patch/face/barycentric owner；
   P2 transport 的 callback 只有裸 `points`，无法无歧义地请求同一物理速度。
3. transport 返回一个 global continuous-P2 vector；现有
   `MaterialWakeBand.material_update()` 只改 vertices，没有把该 vector 精确
   写回 chronological `potential_jump_rows` 的 typed inverse map。

一次探索性调用已在任何几何更新前触发
`interface 'external-body-cut:0' has 1 sides; exactly two are required`。它只是
预登记前的病因指纹，不作为正式裁决；正式门冻结在
`actual_relaxation_composition_audit_cases.yaml`。

可动空间因此不是 N1、诱导速度公式、涡核或压力，而仅是：

> actual material wake 的单侧外部边界身份、owner-aware stage velocity
> contract，以及 global-P2 与 chronological rows 的精确双向映射。

## ② 学科机理

Krebs 2021 的算法先求 surface/newborn strengths，随后每步按 local velocity
移动 wake vertices，再在移动后的几何上重新求 strengths。其 Fig. 3.8 中的
迭代对象是 circulation continuity 与 flow tangency；并不能推出 wake geometry
必须在同一 stage 做非线性 fixed point。论文版本也明确保留各 material wake
element 的历史强度。

Roccia 等 2024 的 UVLM Appendix C 进一步区分时间选择：

- 用 previous-time velocity 的显式一步推进不需要 nonlinear wake iteration；
- 使用 present-time 或 averaged-time velocity 才需要迭代；
- 任一选择的节点速度都必须包含 bound surface、完整 wake 与 freestream。

Tan–Wang 2013 则从 thick-panel/free-wake 耦合说明：TE condition 给出 shed
wake strength，而 wake field 又进入 body-panel 方程。由此，body attachment
与 field owner 都是方程数据，不能把真实 attachment 改标为 `zero` 来迎合
closed assembly。

一手来源：

- Krebs dissertation (2021)，仓库全文
  `KrebsTravis_2021_DDE.pdf/.txt`；
- Krebs, Bramesfeld & Cole (2022)：
  https://doi.org/10.3390/aerospace9010028
- Roccia et al. (2024)：
  https://doi.org/10.5194/wes-9-385-2024
- Tan & Wang (2013)：
  https://doi.org/10.1016/j.cja.2013.04.050

## ③ 缺组成部分，还是组成部分错误

| 命题 | 当前判定 |
|---|---|
| S3n 四通道速度组成 | 已验证，冻结 |
| c5/c6 顶点星法向几何极限 | 已验证，冻结 |
| S3m global-P2 ALE 与 typed inflow 方程 | 已验证，冻结 |
| 把真实 body attachment 当成普通 two-sided patch seam | 错组件候选 |
| 从坐标或距离猜 owner / body boundary | 禁止 |
| 固定四内点 ledger 可直接充当任意 P2 Gauss callback | 证据不足 |
| global-P2 transported state 自动回到 material rows | 缺组件候选 |
| “必须做 nonlinear geometry-strength fixed point” | 文献不支持为必要条件 |

因此当前不能直接实现 fixed point。先用 S3r 对“现有接口直接组合已经充分”
作可证伪审计；若 NO-GO，再把缺件拆成 typed open-boundary stage operator，
然后单独裁决显式 RK stage re-solve 与隐式 fixed point。

## ④ S3r 预登记

S3r 只复用冻结的 S3n canonical actual state，依次检查：

1. actual-boundary history re-entry 与 weak residual；
2. 四通道 ledger；
3. 单侧 body attachment 能否进入 vertex-star projector；
4. 同一 topology 能否进入 P2 transport；
5. transport callback 是否收到 owner identity；
6. transported P2 state 是否有精确 chronological-row reconstruction。

任一失败即 NO-GO，证伪“无需新 stage interface 即可直接组合”。本门禁止通过
重标 boundary、补虚假第二侧、坐标匹配、冻结非 Lagrangian `mu`、添加 core
或放宽 topology check 来过门。

正式阈值、GO/NO-GO 和 scope 已在执行前冻结于
`actual_relaxation_composition_audit_cases.yaml`。

## ⑤ 执行结果与 claim 裁决

预登记后未改配置。actual state 的两个上游健康门先通过：

| 指标 | 结果 | 门 |
|---|---:|---:|
| actual-boundary relative weak residual | `3.493e-16` | `≤2e-12` |
| S3n ledger closure | `0` | `≤2e-12` |

随后四个 direct-composition 接口全部按预登记失败：

| 接口 | 结果 |
|---|---|
| vertex-star geometry projection | one-sided `external-body-cut` 被 closed topology 拒绝 |
| global-P2 transport assembly | 同一 one-sided attachment 被拒绝 |
| transport velocity callback | 缺 patch/face/barycentric owner 参数 |
| scalar history write-back | `MaterialWakeBand` 只有 geometry-only `material_update` |

总计 `4` 个直接组成失败，冻结门要求 `0`，因此 S3r 为有效 **NO-GO**。
这不是 actual-boundary、四通道速度或求积失败，也不能通过把 attachment 改成
zero boundary 来绕过。

Claim 改写为：

- `N3.1j3b6d18c2b3b1` falsified/frozen：S3n、c5/c6、S3m 和 c2a 的现有
  public interfaces **不能直接组成** actual relaxation stage；
- 打开 `N3.1j3b6d18c2b3b2`：typed one-sided body attachment、owner-aware
  stage query、global-P2/chronological-row 双向映射；
- 时间算法不再预设为 nonlinear fixed point。待 b2 通过后，必须另设
  `b3` 比较“每个显式 RK stage 代数重解＋终态重解”和真正 implicit
  geometry–strength iteration，并以耦合残差和固定物理时间 Cauchy 裁决。

压力、力、LEV 和 production 继续禁止。
