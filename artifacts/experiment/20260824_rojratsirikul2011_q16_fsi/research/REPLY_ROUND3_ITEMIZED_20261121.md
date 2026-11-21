# 逐条回复审计方（第三轮）：每一条的执行情况、实测值、未完成项

需求背景：A16（`t*`=21，2100 步）目前实测 **9.136 s/步 ⇒ 5.33 h**；目标 600 s ⇒ 0.286 s/步。
本文按审计方原文的条目顺序逐条回复；§B 为速度测量的全部结果。所有数字附数据文件（§C）。

---

# A. 逐条反馈

## A1. §1.1 `Cε` 的零拟合形式与 `C_s = 1`

**你的内容**：`ε_p = Σ_q (σ_p/σ_q)³ exp(−r²/2σ_q²)(J_pᵀ−J_qᵀ)Γ_q`；`C_d,p = min(1, |Γ|²/(Δt Γ·ε))`（`Γ·ε ≤ 0` 时取 0）；`C_s = 1` 由高斯滤波 Leonard 残差的二阶展开比值恒为 1 得到。

**仓库现状（已核实）**：`platform/rvpm3d.py:64` 有 `_sfs_estr(X, G, sig, J, S)`（numpy，`ZETA0 = 0.06349363593424097`），`sfs_cs=1.0` 常数系数、背散射裁剪；**该文件第 1 行为 `[CLOSED 2026-07-19 …彻底关闭;本文件保留不维护]`，`src/` 下无任何 import**。GPU/warp 版 SFS **不存在**；动态 `C_d` 流程**不存在**（`research_rvpm_arch.md:126` 为设计表，非代码）。

**执行情况**：**未实现**。理由记录如下（不是判断，是执行顺序）：我们先用实测确认了 `Cε` 是否仍被需要。
- 现配置（转置拉伸 + rVPM 重分布 + 冻结斜坡）在 **α=15 / α=19 的 1000 步**上 `max|cn|` 分别为 **1.229 / 3.738**；
- **不含任何拉伸**的带斜坡基线为 **1.242 / 4.443**。
两组的 `max|cn|` 均低于 5 的判据，且两者差距 0.4–1.4pp。因此在当前判据下，`Cε` 的收益无法用 `max|cn|` 观测。

**若你要我们做**：需要新增一个 O(N²) 的 GPU 成对 pass（当前只有 numpy 版），输入是每粒子的 `Γ_p`、`J_p`、`σ_p`。请说明你希望的验收量（`max|cn|`？`Γ_LEV` 累积？拟能？），因为现有判据观测不到它。

## A2. §1.2.1 转置拉伸（`JᵀΓ` vs `JΓ`）

**你的内容**：我们的 `pfield_torch_gpu.py:1033` 用第二指标缩并（经典 Helmholtz），rVPM 应用第一指标；并称 `rvpm_reference.py` 存的也是 `J[i,m]=∂u_i/∂x_m`（非转置）。

**执行情况**：**已实现并已提交**（`5e0f08b`）：`apply_vortex_stretching(..., transposed=True)` 为默认，经典式保留；`NativeV5MConfig.particle_stretching_transposed`；三处错误表述已改（`pfield_torch_gpu.py` docstring、`particles.py:150` 注释、`NativeV5MConfig` 注释）。

**新增判据**（仓库此前空缺）：`tests/test_pfield_stretching_conservation_gpu.py`（7 项）——两个等核粒子：
- 转置：`|S_p+S_q| ≤ 1e-12·max|S|`（实测成立）；
- 经典：`|S_p+S_q| > 1e-6·max|S|`（实测成立，断言写入测试以保证判据非空）；
- `Γ_q = −Γ_p` 的零环量云：转置下相对漏量 ≤1e-12，经典下 >1e-9。

**与你的引证不符之处（供你核对）**：`rvpm_reference.py:23-26` 的 docstring 为 `jacobian[:, i, j] is ∂u_i/∂x_j`，第 149-157 行的解析构建也一致；因此 `rvpm_transport.py:131` 的一致性来自"缩并转置指标"，而不是"该模块存转置"。我们此前 docstring 里写的"that module stores the transpose"是错的，已删除。

**实测效果**：α=19、300 步、斜坡后窗口——经典 5.557、转置 3.551、无拉伸基线 3.137（`/tmp/rvpm_300.json`、`/tmp/rvpmT_300.json`、`/tmp/ramped_300.json`）。

## A3. §1.2.2 `STAGE1_2` 的 `weight = 1.0` 使核收缩快 5 倍

**你的内容**：我们否证核演化时 `rvpm=False` ⇒ `weight=1.0` ⇒ `dσ/dt` 是 V4 规定速率的 5 倍，同时 `dγ/dt=+S` 无 `−3Zγ` 约束。

**执行情况**：**这一条成立，且推翻了我们的一个结论。** 我们的 `STAGE1_2` 运行命令为 `--stretching --core-spreading`（无 `--stretching-rvpm`）；查 `pfield_torch_gpu.py:1037` 为 `weight = RVPM_Z_WEIGHT if rvpm else 1.0`，故当时：
- `z = 1.0 · (S·Γ)/|Γ|²`（正确应为 0.2 倍）；
- `dσ/dt = −σ z`（5 倍速率）；
- `dγ/dt = +S`（无 `−3Zγ`）。
第 125 步的塌缩爆炸是该配置的结果。我们此前把它写成"核演化被否证"属过度归因，**已更正为"该配置下否证；`rvpm=True` 权重下的核演化未测"**。

**未完成**：`rvpm=True` + `core_spreading=True` 的联测**尚未执行**。代价：一次 300 步运行（α=15/19，约 18 min）。你要不要排在当前速度工作之后？

## A4. §1.2.3 `J` 缺前缘束缚涡环与尾迹环的贡献

**你的内容**：`jacobian_at_cuda` 只算粒子互诱导，漏掉 `J_ring(x_p)`。

**执行情况（已核实代码）**：`velocity_and_jacobian_self_cuda` 的 source 只有 `self.pos[:self.n]`（存活粒子），**不含 bound rings、不含 wake rings**。你的描述与代码一致。

**未实现**。补充一条我们的实测（供你判断优先级）：`particle_stretching` 开启后对 α=15 的 1000 步均值影响 0.4pp（0.5753 → 0.5782），对 `max|cn|` 影响 1.26 → 1.23。即当前判据下拉伸整体贡献很小，`J_ring` 是在此基础上的二阶修正。

## A5. §2.1 `bound_rate` 的 `Mf1` 双算

**你的内容**：`bound_rate` 的常量槽是全导数差分 `ρ(Γ_n−Γ_{n−1})/Δt`，已含 `Mf1 q̈ + Mf2_1 q̇`；而 stepper 仍另加 `velocity_force` 与 `Mf1@q̈`。

**执行情况**：**已用外部真值确认**。工具 `reproduce_yamano2020_q16_flux_v5m_native.py`（8 外步，`max|LESP|=0.0179<0.11` ⇒ 零粒子释放），真值为 Yamano 论文尖端位移：

| 配置 | 步1 | 步2 | 步3 | 步4 | 步8 |
|---|---|---|---|---|---|
| `bound_rate`（现行）| 15.25% | 18.68% | 24.87% | 30.22% | **51.62%**（完成）|
| `bound_rate` + 删 `Mf1`/`Mf2_1` | **0.17%** | 1.34% | 6.62% | 10.52% | 崩（Newton 线搜索）|
| `material` | 15.23% | — | — | — | 8 步内崩 |
| 三槽式（扣 `Mf1q̈`、留切线）| **2.03%** | — | — | — | 第 8 步未达 Newton 容差 |
| 历史记录 2026-08（material）| 1.425% | — | — | — | 27.92%（完成）|

**未解决**：`material` 现在步 1 是 15.23%，而 2026-08 的记录是 1.425%。两次的 `material` 与 `bound_rate` 在当前代码上几乎相同（15.23 vs 15.25）。该差异未定位。

## A6. §2.1 的 `R^2976` 混合量纲范数

**你的内容**：Q16 每节点含 N、N·m、N·m² 三类单位；5 点 Dirac 散射 vs 光滑求积的差异会被未加权范数放大。

**执行情况**：**已实现替代度量**（`d99eeae`）：`src/fluxvortex/warp_fsi/q16_load_metrics.py`
- `flexibility_energy_norm(r, K) = sqrt(rᵀK⁻¹r)`，用 `solve` 不建逆、取对称部、负二次型拒绝、可选正则化；
- `relative_flexibility_gap(a, b, K)`。
判据（7 项 CPU 测试全过）：真实单位变换（`r'=r/c, K'=K/c²`）下不变到 1e-12；同一变换下原始 L2 摆动 >100×。

**未完成**：**尚未用它重报膜翼双轨差异**（现有 `0.338/0.383` 仍是未加权 L2）。需要一次 A16 短跑 + 该函数接入 shadow consumer。

## A7. §2.2.1 用 Fig.14 裁决轨道 A/B，`MAE 0.130`

**你的内容**：轨道 A 在 `IZRA-15-015` 给反号推力 −0.133，MAE 0.130；轨道 B 得 0.0386–0.0409。

**我们的核对**：
- 0.130 是 `fluxv_v5m_izraelevitz2017_fig14_mandatory_v2` 的历史值；`REVIEW_IZRA_V4_20260828.md:14` 记录 v1/v2/v3/v4 = 0.210 / 0.130 / 0.036 / 0.039，即 v3 之后已是 0.036–0.039。
- **在同一刚翼状态上同时计算两种口径**（`diagnose_roj_load_track_arbiter.py`，α=5，300 步）：轨道 B `+0.3294`、轨道 A `+0.3294`、**A/B = 1.0001**（实验 0.3413）。

因此在这一工况上两种口径不可区分；0.130 对应的是另一处缺陷（`test_rigid_kj_loads_gpu.py:3-10` 记为"frozen wake / 缺尾迹速度导致推力反号"）。

## A8. §2.2.2 Goland ⇒ 隐式 `Mf1` 必留

**你的内容**：Goland 显示把 `Mf1` 当滞后外力显式施加会引入 σ≈−0.47 的人工阻尼；必须隐式并入有效质量。

**我们的实测支持这一点**：删掉 `Mf1`（`bound_rate_author_weak`）后步 1 最优（0.17%），但误差单调增长 0.17 → 1.34 → 6.62 → 10.52%，随后结构 Newton 线搜索失败。这与"`Mf1` 提供稳定化"一致。

## A9. §2.2.3 三槽统一式

**你的内容**：Slot1 = 弱常量 − `Mf1(q_n)q̈_n`；Slot2 = 仅网格运动 KJ，删 `Mf2_1`；Slot3 = 隐式 `+Mf1q̈`；收敛端点两者逐位抵消至 1e-16。

**执行情况**：**已实现 Slot1+Slot3 的配对**（`three_slot_mf1_cancel`，`667c313`）：在 RHS 常量槽减去一份 `anchor_am_matrix @ q̈_current`，`Mf1` 仍作为 `acceleration_load_action` 进入有效切线。
- 步 1：**2.03%**（对比现行 15.25%、删 `Mf1` 0.17%、历史 material 1.425%）；
- 8 外步：第 8 步 Newton 未达容差，残差单调下降 21.9 → 1.9e-5（容差 3e-7，用尽 128 次迭代）。
- **Slot2（删 `Mf2_1`）未实现**；`velocity_force(..., include_mf21=False)` 已具备，但尚未与三槽式组合实测。

## A10. §2.2 实验 E-1（`Q_5P` / `Q_panel` / `Qf_author` 三向量 + 两种物理范数）

**执行情况**：**未执行**。核查结果：`Q_5P`、`Q_panel`、`Qf_author` 这三个标识在仓库中不存在；现有 `verify_yamano2020_q16_native_load_gates.py`（0.83 s）只存 3 分量合力与标量功（`aic=3.007e-7`、`mf2=3.816e-17`、`qf_resultant=0.002894`、`mf1_common_field=0.3735`、`q16_work=−0.025859`）。柔度能量范数已具备（A6），wrench 已具备（`q16_rigid_body_wrench.py:30`，生产在 `case_runner.py:872-915` 用作 `cn_full_action`）。要执行 E-1 需先新增"同一状态上三个向量的并列导出"。

## A11. §2.2 实验 E-2（三轨迹终裁）

**执行情况**：**已执行**，结果见 A5 表。三条轨迹均跑了；其中两条（`material`、删 `Mf1`）在 8 步内中断；`bound_rate` 跑完（51.62%）；三槽式在第 8 步停。

---

# B. 速度：全部测量

## B1. 各窗口的 `step_wall_seconds_mean`（并标注仪器）

| 窗口 | 粒子数 | 单价 | 仪器 |
|---|---|---|---|
| 3 步 | 近空 | **2.08 s/步** | 无 |
| 15 步 | ~1.6k | 3.945 | monkeypatch 计数（µs/次）|
| 20 步 | ~5k | 4.35–4.64 | cProfile |
| **30 步** | **~8k** | **4.720** | **计时器插桩（主测量）** |
| 40 步 | ~11k | 10.52 | 采样线程（GIL 争用）|
| 60 步 | ~16k | 6.156 | 采样线程 |
| **150 步** | **~27k** | **9.136** | **无** |

计时器法与无插桩同区间相差 2–8%；采样线程法的两点明显偏离趋势线。

## B2. `.item()`（计时器，30 步）

| 量 | 值 |
|---|---|
| 次数 | 103,510 ⇒ **3450/步** |
| 累计 | **92.20 s = 65% wall** |
| 单次 | **891 µs** |

调用点（20 步 cProfile）：`_require_cuda64` 1240/步（36.8%）、`line_derivative` 688/步（20.4%）、`_reference_dense_nonsymmetric_solve` 292（8.7%）、`_integrate_structure` 228（6.8%）、`_acceleration_load_matrix` 195（5.8%）、`assemble_native_ring_geometry` 177（5.3%）、`material_ring_velocity_derivative_expanded` 172（5.1%）、`velocity_force` 172（5.1%）。
与 2026-08-30 profile 对比：1,524 次/步、22% → 3,450 次/步、65%。

## B3. 相位拆分（计时器，绝对秒，wall 141.6 s）

| 相位 | 绝对 | 占比 | 次数/步 |
|---|---|---|---|
| `_integrate_structure` | 128.05 s | 90.4% | 2.9 |
| `structural_step` | 112.19 s | 79.2% | 58.0 |
| `_reference_dense_nonsymmetric_solve` | 76.75 s | 54.2% | 104.9 |
| `propose` | 13.15 s | 9.3% | 2.9 |
| `refresh_reference_tangent` | 11.04 s | 7.8% | 1.0 |

## B4. 稠密求解内部

| 操作 | 次数/步 | 累计 | 占比 | 单次 |
|---|---|---|---|---|
| `solve_ex`（LU）| 104.9 | 1.47 s | 1.0% | 0.5 ms |
| ≥10⁶ 元素 gather | 364.0 | 0.11 s | 0.08% | 0.3 ms |
| 整个函数 | 104.9 | 76.75 s | 54.2% | **24.4 ms** |

⇒ 24.4 ms/次内：LU 2%、gather 0.4%、函数内 `.item()`（约 2.8 次/调用 ×891 µs）≈10%，**其余 ≈23.8 ms（97.6%）未定位**。

## B5. 粒子 BS target 集合

| 次数/步 | 调用者 | targets |
|---|---|---|
| 5.5 | `propose` | 450 |
| 2.8 | `propose` | 1800 |
| 多次 | `wake_velocity`（`q16_flux_v5m_native.py:995`）| **`n_rows×30×4`**（1 行 120、2 行 240 …）|
| 1 | `velocity_self_cuda` | n（=sources）|

稳态（n=27,371、300 行）：尾迹顶点 36,000 targets ⇒ 9.85×10⁸ 对（51%）；自诱导 7.49×10⁸ 对（39%）；配点 450×5.5 ⇒ 6.8×10⁷；中点 1800×2.8 ⇒ 1.4×10⁸；合计 **1.94×10⁹ 对/步**。

## B6. 已执行改动与实测

| 改动 | 提交 | 实测 |
|---|---|---|
| Stage 4a 指纹守卫 | `213c042` | parity 6/6；每次守卫 8 次 D2H + 2.25 MB 哈希 → 0 |
| Stage 4b 诊断合批 | `6f4be69` | 3 步 A/B：355.3 → 341.3 次/步；37 键逐位不变 |
| S1 延迟切线内有限性检查 | 未提交 | 2.081 → 2.087 s（无变化）；cn 逐位同；scope = 1.0 次/步 |
| A1 free 空间缓存 | 未提交 | cn 逐位相同（diff = 0.0）；速度未测（运行中止）|

## B7. 由数字直接算出的关系（各附假定）

1. `.item()` 归零上界 = 65% ⇒ 9.136×0.35 = **3.20 s/步 ⇒ 1.87 h**（假定稳态占比不低于 30 步窗口值）。
2. 600/2100 = **0.286 s/步**。
3. 稠密求解 24.4 ms/次 → 0.5 ms 量级 ⇒ wall −53% ⇒ **4.29 s/步 ⇒ 2.5 h**（假定其余 23.8 ms 可消除；来源未定位）。
4. 尾迹顶点按 free rows 限制 ⇒ 对数 −33%；但 `propose` 仅占 9.3% ⇒ 对总 wall 影响上界 9.3%。
5. 1+3 去重（重叠 0.542×0.65 = 0.35）⇒ wall 下降上界 0.84 ⇒ **1.46 s/步 ⇒ 51 min**（上界，依赖 1、3 的假定）。

## B8. 尚未测量

| 项 | 代价 |
|---|---|
| 150 步稳态的 `.item()` 占比（计时器法）| ~23 min 一次 |
| 稠密求解内 23.8 ms 的分段归因 | 需在该函数体内插桩（改代码）|
| `propose` 内细分（气动求解 / 沉积 / 载荷装配 / retention）| 需在 propose 内插桩（改代码）|
| 冻结行顶点诱导是否被下游消费 | 需读 `wake_velocity` 三个 RK3 结果的消费链 |

---

# C. 三问（需要你回答）

1. 两处 `isfinite().all().item()`（1928 次/步、57%）改为"设备端 `atomic_max` + 步末读一次"后，**异常抛出位置由"该次调用时"变为"该步末"**——是否改变你对该路径的验收含义？
2. 冻结行（`n_rows>100`）顶点的粒子诱导**是否仍被下游消费**？（`propose` 仅 9.3%，故此项对总 wall 影响上界 9.3%。）
3. `_reference_dense_nonsymmetric_solve` 每次 24.4 ms，其中 LU 0.5 ms、gather 0.1 ms、函数内 `.item()` ≈2.5 ms。**剩余 ≈23.8 ms 你认为落在哪些操作上？** 我们的插桩未覆盖该函数体。

# D. 数据文件
```
/tmp/a16_150.json       150 步（无插桩）9.1363 s/步
/tmp/measure4.py        计时器插桩（§B2–B4 的来源）
/tmp/where_items.py     .item() call-site
/tmp/count_targets.py   粒子 BS target 计数
/tmp/sample_prof.py     采样 profiler（只用于份额）
/tmp/rvpm_300.json /tmp/rvpmT_300.json /tmp/ramped_300.json   拉伸三对照（300 步）
/tmp/yam_*.json         Yamano 8 外步各配置
```
