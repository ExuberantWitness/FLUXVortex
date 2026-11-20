# 净前缘丝约束在当前几何中不成立：脱落量比 bound 环量大 2.2–2.6 倍（2026-11-21）

> 工具：`platform/warp_vpm/diagnose_roj_net_filament_feasibility.py`
> 证据：`artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/net_filament_feasibility_a15.json`
> 工况：A15、joint 路径、冻结配置、300 步（t*=3，爆前稳态段）

## 1. 结果（步 280–300，30/30 条带分离）

| 量 | 数值 | 说明 |
|---|---|---|
| `Γ_bound,LE`（max）| **0.0341967** m²/s | 与 pin 目标 `target` **逐位相同** → bound-only pin 按设计工作 |
| `λ_first`（max）| 0.081–0.093 m²/s | 最新脱落涡片强度 |
| `λ/Γ_bound,LE`（中位）| **2.16 – 2.60** | 每步脱落的环量是 bound 前缘环量的 2–3 倍 |
| `net = Γ − λ`（max）| 0.115–0.127 m²/s | 净前缘丝 |
| **`net/target`（中位）** | **−3.16 – −3.60** | 净前缘丝是临界值的 −3 倍（量级与符号都错） |

## 2. 判读（三条，逐条都有数字支撑）

1. **bound-only pin 本身工作正常**：`Γ_bound,LE` 与 pin 目标逐位相等，说明约束求解无 bug。此前怀疑的“pin 表示错配”不是数值失效，而是下文第 3 条的物理错配。
2. **净前缘丝关系被结构性违反**：`net/target ≈ −3.3`（恒定，非偶发），且 `λ/Γ ≈ 2.2–2.6`。这说明脱落片每步拿到的环量是 bound 前缘环量的 2–3 倍。
3. **由此推出：把 pin 从 bound 改到净前缘丝不是修复，而是加重**。净前缘丝约束要求 `Γ_bound,LE = crit + λ`；而 `λ ≈ 2.5·Γ`，于是 `Γ_bound,LE` 会被推高到 `3.5×crit` —— 正是我们观测到的 runaway 方向。**Step 3（E1-BUFFER-NETFILAMENT）在本几何下被本测量否证。**

## 3. 真正的缺口：脱落量不满足 Kelvin

`Γ_bound,LE` 被钉为常数，故每步 bound 环量的**变化量**只来自其余弦向行（远小于 `Γ_bound,LE`）。而实际脱落量 `λ ≈ 2.2–2.6·Γ_bound,LE`，比“环量变化”大 1–2 个量级 ⇒ **脱落量不是由环量守恒给出的**。这与两条既有证据完全一致：
- L2-1：`Γ_ribbon/Γ_flux = 22.9`，且随 Δt 翻倍（`γ ∝ 1/Δt`）；
- 队列分解：爆后最新一条 ribbon 占 LE 感应 100%。

## 4. 队列改道（依据证据，不是临时起意）

- **取消** Step 3（E1-BUFFER-NETFILAMENT）：其前提（净前缘丝在当前几何有意义）已被测量否证；
- **新增 Step 3′：E1-KELVIN** —— 脱落量由**环量守恒**给出（Kelvin：`λ_new = −Δ(ΣΓ_bound)`，与 TEV 的 `bound_rate` 同一原则），bound 求解仍保持 LE pin。理由：这是经典精确要求（无系数、无阈值、无调参），且直接针对本测量暴露的“量”而非“钉哪里”；
- 红测不变：A15 `step≥2 max|cn| < 5`；A5 保持 0.3425±0.003。

## 5. 边界

- 本判据用 `sign(Γ_bound,LE)` 作为 pin 目标符号的离散代理（diagnostics 只记录 LESP 幅值），符号约定需在 E1-KELVIN 实现时用有向单位基复核；
- `Γ_bound,LE` 与 target 逐位相等只证明 pin 的**代数**执行正确，不证明该物理量选得对；
- 本记录不改变任何生产数值（纯诊断）。
