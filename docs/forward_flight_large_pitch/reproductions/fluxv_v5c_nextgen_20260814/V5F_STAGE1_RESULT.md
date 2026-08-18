# FluxV v5f Stage 1 机械结果

## 结论

Stage 1 的“基线同源性”门通过；主动材料 LEV 和论文性能仍未实现、不得评分。

- hard-off 工厂直接返回原 `UVPMHybridSolver`；
- enabled-pristine 在 Yang 2025、AoA=15 deg、2 个周期、40 步、2×4 面板
  smoke 上，与 parent 的 bound circulation、全部 TE wake 顶点/强度/年龄、逐面元
  力矩、整机载荷和 195 个 VPM 粒子逐位一致；
- literal `[A B; H 0]` 代数 helper 的 inactive 和 `Gamma_L=0` 路径严格退化
  为 parent solve；合成 active system 的无穿透与约束残差小于 `1e-12`；
- 定向测试 `9 passed`，Black、Ruff 和 diff check 通过。

## Stage 2A 增量结果

真实 Ptera AIC 上的 bound-only 主动探针已通过独立复审：

- Hirato Eq.6 使用物理面元几何，Yang smoke 的 `c=0.130 m`、
  `delta_x=0.065 m`，排除了 Ptera 移动后缘的 advective extension；
- `Lcrit=0.03` 时四条带活动状态为 `[True, False, True, True]`；
- newborn 反向、pseudovortex 保持 bound orientation，共享前缘丝线残差严格为
  零；pseudovortex 后缘与 rearmost bound aft edge 逐位一致，因此增广未知量可直接
  使用 Eq.9 的 `+Gamma_L` 符号；
- `cond(A)=2.37`、`cond([A B;H 0])=74.62`、Schur 条件数 `1.074`；
  最大 `|Gamma_L|/(Uc)=0.05966`；无穿透和 LESP 残差分别为
  `1.11e-16`、`3.47e-18`；
- 极端病态/非有限求解已 fail closed；15/15 定向测试通过。

这里的 Stage 2A `B` 列特意使用 Ptera 自身的 Nguyen/Ramasamy 有限核，只用于
核对 AIC 排序、环向、符号和增广代数；它不是 Hirato Eq.25 的材料尾迹核，也不作为
后续时间推进的物理结果。时间推进必须改用 Eq.25 Lamb--Oseen cutoff，并在同一
cutoff 下同时计算 newborn 与 pseudovortex，保持共享前缘丝线严格相消。

原生载荷修正的纯函数也通过独立复审：front 修正恒为零；span 腿由本步表面释放
的展向差分给出；TE 使用本步表面释放减上一步 wake release；Eq.17 只在本步
active 条带使用 `(Gamma_L^n-Gamma_L^{n-1})/dt`。activation、continuation、
deactivation、inactive 与 step-0 语义共 24/24 测试通过。

live Ptera 四腿适配器随后通过独立复审：它只允许在 parent `_calculate_loads()`
返回后立即读取同一时层的四腿中心、向量和速度；step-0 真实 wake、已求解 bound
state、panel load freshness 均 fail closed。材料状态层也已通过：Eq.7 proposal、
candidate view、旧 Gamma 位级不可变、Eq.24 以及 shed+convect 原子发布均有定向测试。

## Stage 2A 时点的独立审计限制

1. Ptera bound core 与 Hirato cutoff 口径不同；材料尾迹阶段必须显式报告 core
   敏感性，不能用目标论文载荷选择 core。
2. surface-image 分支目前主要由源码同构保证，尚缺真实镜像工况测试。
3. 当时历史材料 LEV、共享 Kelvin TE row、Eq.25 velocity/KJ 与 load/wake commit
   尚未在一个 time march 中联通；这些接口随后在 Stage 2B/2C 接通，但 M5 仍失败。
4. 测试命令必须固定 `PYTHONPATH=src:platform` 并记录导入路径，避免误读另一
   worktree 的 `fluxvortex.solver`。

## Stage 2B/2C：多步材料时间推进与停止门

首事件之后的 continuation 已接通：Eq.7 remesh、旧材料 LEV 固定 RHS、连续脱落、
deactivation/reactivation、Eq.9 下一 TE row、Eq.17/TE 分账、故障回滚和 poisoned
fail-stop 均有真实 Ptera 定向测试。hard-off 和无事件路径继续逐位退化到 parent。

这使 M3/M4 的机械时间与账本门可执行，但随后 M5 时间/core refinement 明确失败：

- 固定 Yang 15 deg、`Lcrit=sin(5 deg)`、`2x4` 空间网格，在同一半周期比较
  `20/40/80/160` steps/cycle；
- 对三个预注册 `rc/d_min=0.10/0.25/0.49`，80→160 细化的峰值材料环量倍率为
  `2.192/8.260/4.681`；
- 三者均达到或超过 `1/dt` 级增长，M5 判定为 NO-GO；
- 同时无穿透、LESP 与 Eq.9 残差仍分别小于 `5.69e-14`、`1.59e-15` 和等于零，
  所以失败属于材料反馈不收敛，而不是代数残差未闭合。

正式报告见 `V5F_M5_REFINEMENT_STOP_REPORT.md`，可复现产物位于
`runs/20260814_fluxv_v5f_m5_refinement_final/`。按预注册停止规则，没有运行
Yang、Figure 14 或 Baik 的论文精度评分，也没有通过目标数据选择 core。

当前关键 SHA256：

- `fluxv_v5f_native_solver.py`: `a444e5ec0624cba4560b1e0e6deaa6c654bc8bf86a12cb2705a0cda4c76f4563`
- `test_fluxv_v5f_native_solver.py`: `a6a72a1ce8bb3a17e6003c37aed3e5987d0d8f8e99e5b1a4a0cf0ee62f1725c2`
- `test_fluxv_v5f_native_continuation.py`: `27b28b705b63c4deb717bd7cbb52b6c6a710fda6e48cdce0ec0d68a528be6009`

结论：v5f 当前候选归档为 mechanical NO-GO；不得以小残差、首事件闭合或单一 core
结果表述为下一代性能改进。

停止后的只读取证进一步表明：没有找到 RHS、Eq.9 或 Eq.24 双计；主要增益来自
卷起几何下 newborn 与 pseudovortex 的 LESP Schur 响应接近相消。实现还缺少
相邻 strip 的共享展向节点，但临时缝合后细化仍不收敛。下一条 native 路线若继续，
必须升级为 node-owned 连通高阶涡片并从独立 canonical 工况重新验收，而不是修补
当前常强度环。
