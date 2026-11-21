# 速度诊断：A16 全窗 5.33 小时，20 分钟**在现配置下不可达**；两级改法与其真实靶点

> 实测入口：`reproduce_rojratsirikul2011_q16_flux_v5m_native.py --case ROJ11-A16`
> **注意：3 步测速会严重低估**（尾迹为空，2.08 s/步）；必须用稳态窗口（150 步）测。

## 1. 实测

| 量 | 值 |
|---|---|
| A16，150 步 | **9.136 s/步**（`step_wall_seconds_mean`），总 1370.8 s |
| 外推 2100 步（t\*=21 全验收）| **5.33 小时** |
| 3 步时的单价 | 2.08 s/步（尾迹几乎空）|
| `structural_substeps` | 10 |
| `proposal_count` / 150 步 | 447（**每外层步 ~3 次 propose**）|
| `reference_tangent_cache_refresh_count` / 150 步 | **20** |
| `reference_tangent_refresh_rtol` | 0.002（已生效）|
| `commit_count` / `formal_replay_count` | 150 / 150 |

## 2. 成本拆分（消融实测）

| 分量 | 每步 | 依据 |
|---|---|---|
| 气动 + 粒子（同 15×30、27k 粒子）| **~1.2 s** | 刚翼 900 步 / 18 min；α=16 刚翼 50 步仅 14 s（含建模）|
| Q16 结构 + FSI 耦合 | **~7.9 s（87%）** | 差额 |

**⇒ 20 分钟（0.57 s/步）在现配置下数学上不可达**：光气动地板 1.2 s/步 ⇒ 2100 步 = **42 分钟**，结构归零也到不了 20 分钟。

| 情景 | 全窗 |
|---|---|
| 现状 | 5.33 h |
| 结构侧修到理论下限 | ~50 min |
| 结构归零（不可能）| 42 min |
| 20 min | 需同时砍粒子侧（treecode / 减 N），或缩短验收窗 |

## 3. 两个被证伪的假设（记录下来，避免重复）

1. **S1（延迟有限性检查）收益 = 0**：同会话 A/B，`step_wall_mean` 2.081 → 2.087 s，cn 逐位相同。原因：那些 `synchronize_device` 的 self time 是 0.002 s，时间花在等真实 GPU 工作上，不是在等空转。
2. **S2（切线刷新节奏）几乎没有空间**：刷新已经被 rtol=0.002 门控到 **20/150 步**，最多省 0.35 s/步（4%）。

## 4. 真正的靶点：每次求解前后的 70 MB 张量搬运

每次 dense 求解摊到 **0.22–0.45 s**，而 2436² 的 LU 本身只要 5–8 ms。所以钱花在：

```
effective = tangent*(1+damp) + coef*M − inv*added_mass   # 3 × 70.9 MB
effective[free][:, free]                                 # 2 次 2976² → 2436² gather
solve_ex                                                  # 5–8 ms ← 便宜的
```

每步 20–40 次求解 ⇒ 每步搬运 **~5–10 GB**。

## 5. 两级改法（都可精确等价；但都改变求和顺序 ⇒ 不能逐位，只能 ≤1e-12）

**A1（低风险）**：把 β-无关项提到 refresh 时算一次，并把 added-mass 的两块**预先在 free 空间**算好：
```
am_free_a = anchor_am[free][:, free]        # 每次 refresh 1 次
dam_free  = am_slope[free][:, free]         # 每次 refresh 1 次
E_free(β) = base_free − inv·(am_free_a + β·dam_free)
base_free = (tangent*(1+damp) + coef*M)[free][:, free]
```
预计每次求解搬运量降 **~2–4 倍** ⇒ 全窗 **5.33 h → 1.3–2.7 h**。

**A2（真大头）**：在 `base_free` 上做**秩 ≤450 的 Woodbury**。
`dMf1 = ρ P (A⁻¹ N)`，`U = ρP`（dof×450）、`Vᵀ = A⁻¹N`（450×dof）⇒ 秩 ≤ 面板数 450。
每次求解变成：1 次回代 + 2 次 450×2436 matvec + 1 次 450² 解。
预计 **→ 45–60 分钟**。

**A2 的前置条件（必须记录）**：
- 有效切线**非对称**（Q16 切线 + Kelvin–Voigt 阻尼）⇒ Woodbury 只能作用在**解**上（`E0⁻¹` 用一次 LU），不能假设对称 Cholesky；
- Kelvin–Voigt 阻尼项复用缓存参考切线（`q16_structural_solver.py:512-517`），所以 `base_free` 必须包含它，否则等价性不成立；
- 验收：与生产路径的广义力/位移差 ≤1e-12（不是逐位），且 3 步 + 50 步墙钟下降、parity 6/6 仍过。

## 6. 判据（避免再自欺）
- 每级都测 **同 50 步窗口的墙钟**（不用 3 步，那是空尾迹）；
- A1 若 <1.5×，直接跳 A2；
- A2 若仍 >1 s/步，则这条路线在"不碰物理"的前提下无法满足 20 分钟，必须回到"砍粒子侧"或"缩短验收窗"的决策。
