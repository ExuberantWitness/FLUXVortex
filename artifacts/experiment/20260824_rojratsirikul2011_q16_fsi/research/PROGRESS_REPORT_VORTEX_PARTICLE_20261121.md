# FLUX-V5M 涡粒子方法：进展汇报（2026-11-21）

> 代码基线：分支 `run/q16-lev-tev-pc-fsi-20260821`，HEAD `97235c3`（全部已推送）
> 工作树：`FLUXV_RUNS/v5m-fa8eaca`
> 本报告自包含：①方法原理 ②已比对/验证的 case 与数字 ③当前问题（精度、发散、速度）④已否证的路线 ⑤下一步与判据

---

## 0. 一页纸结论

| 项 | 状态 |
|---|---|
| 能复现的 | **刚翼 AR=2、U=5 m/s、α=5°：Cn = 0.3425 vs 实验 0.3413（+0.35%）**，且是纯几何修复带来的；α=0° 得 0.000（对称应得 0）|
| 不能复现的 | α=15° 在 t*≈5–7 后**发散**；α=19° 有界但振荡、均值 **−28%**；膜翼 A16/A17 的 Cn **+34%** |
| 已确认的机理边界 | 失稳是**"LEV 由 LE 不穿透 + 临界 pin 求解"这一离散闭包在持续深分离下的内在不稳定**，不是几何重叠、不是 retention、不是远场感应、不是脱落量公式 |
| 已修复的真实缺陷 | TE 涡片几何连通（此前是假修复）、Mf1 面积/半系数、CPU 上的 Cn、每带 LEV 账本、谱分辨率误报 |
| 最慢的环节 | 膜翼 Q16 FSI ≈ **8.8–13.2 s/步**；刚翼 V5M 15×30 ≈ **1.4–2.5 s/步**（含 ~27k 粒子）|

---

## 1. 涡粒子方法计算原理（本求解器实际实现）

### 1.1 三种表示的混合
| 表示 | 承载什么 | 数值实现 |
|---|---|---|
| **bound 涡环**（V5M/UVLM）| 附着涡量，`nc×ns` 个四边形环（15×30=450）| 环量 `Γ`（m²/s），影响核 = 有限涡丝 + 四阶核修正（`core_fraction=1e-6`，自感项为负，硬门校验）|
| **LEV 高斯粒子** | 前缘分离涡 | 每条分离带每步沉积一条**相连 ribbon**（Gaussian，σ=0.04c，目标间距 0.018c）|
| **TEV / 自由尾迹涡环** | 后缘脱落与远尾迹 | 逐排在 TE 处插入新环行（`γ_tev = Γ_TE + γ_lev`），free rows 对流、older rows 冻结 |

### 1.2 每步的求解顺序（`Q16NativeV5MSolver.propose`）
1. **影响矩阵** `A`：`A[i,j] = n_i · u_j(单位环)`（1/m）。
2. **不穿透 RHS**：`rhs_i = n_i·(V_body − U∞ − V_wake − V_particles)` —— **已释放的 LEV 粒子感应就在 RHS 里**。
3. **无约束解** `γ_pre = A⁻¹ rhs` → **LESP**：`lesp = −γ_pre[LE行]/scale`，`scale = c·U·(θ+sinθ)/1.13`。
4. **分离判据**：`|lesp| > lesp_crit = 0.11`（唯一分离 owner）。
5. **分离求解（joint 路径）**：LE 行被**钉在临界值**（Dirichlet pin），新生 LEV 强度由 **LE 不穿透闭合方程**解出（用**真实沉积 ribbon 的感应基**，不是 AIC 代理列）。
6. **沉积**：ribbon 端点 = 物理 LE → frontier（位移 ≈0.005–0.0066c）；押入粒子场并记入每带账本。
7. **材料对流**：joint 路径用**同步 RK3**（源与目标每子级同步更新）。
8. **尾迹推进**：新 TEV 行（连接当前 TE 与**已对流的旧前沿**，保证涡片连续），超容量则截断并审计环量/线冲量。
9. **retention**：粒子按年龄上限回收，回收前记录环量与线冲量。
10. **载荷**：Kutta–Joukowski（4 条线力/面板，共享边强度取半差、TE 行取 `Γ_n−Γ_{n−1}`）+ 非定常 `ρ dΓ/dt A n`；同时统一 `Mf1 = ρ P(q) A⁻¹ N(q)` 作为加速度作用。
11. **两个 Cn 口径**：`cn_constant_current`（历史口径，只用常量压力部分）与 `cn_full_action`（把 constant+velocity+Mf1 全部分量经 `T_RBᵀ` 投影到弦法向）。

### 1.3 关键不变量与硬门（工程保证）
- 三相分离约束同解：LE pin + LE 不穿透 + 实体表面**实际沉积流场**不穿透（提交前校验，实测 3e-15–3e-14 m/s）；
- 载荷闭合：`Σ点力 = 总力`、`Σ r×f = 总矩`（≤1e-12 相对）；
- 功共轭：`Q·q̇ = Σ f·v`（6e-17，机器精度）；
- 默认生产路径**逐位不变**由 `tests/test_roj_unified_case_runner_parity_gpu.py` 守卫（6/6，每轮都跑）。

---

## 2. 已比对与验证的 case

### 2.1 纯气动（外部基准）
| Case | 内容 | 结果 |
|---|---|---|
| **Izraelevitz 2017 / Scherer 1968 Fig.14** | 12 个俯仰-升沉工况的推力系数 CT | MAE 0.0386 → **0.0409**（速度优化后），偏差方向不变 |
| Yamano 2020 载荷门 | 结构-气动载荷一致性 | 密度线性、闭合门全过 |

### 2.2 刚翼 AR=2、U=5 m/s（论文 Figure 9 刚性系列，数字化 CSV 为真值）
| α | 模型 Cn（t*=10）| 实验 | 判定 |
|---|---|---|---|
| 0° | 0.000（无释放）| 0.020 | 对称应得 0，OK |
| **5°** | **0.3425（恒定）** | **0.3413** | **+0.35%** ✅ |
| 15° | t*≈5–7 后**发散**（step≥2 max\|cn\| 9.8e10）| 0.70 | ❌ |
| 19° | 有界、强振荡，均值 0.5775 ∈ [0.24, 1.10] | 0.800 | ❌ **−28%** |

### 2.3 柔翼 Q16 FSI（论文 Figure 6 / 9 柔性系列）
| 工况 | 量 | 模型 | 实验 | 判定 |
|---|---|---|---|---|
| A16 (U=5, α=16°, t*=21) | zmax/c | 0.0451 | 0.0434 | ✅ +0.0018 |
| A16 | Cn | 1.2635 | 0.9200 | ❌ **+34%** |
| A17 (α=17°) | zmax/c | 0.0436 | 0.0440 | ✅ −0.0003 |
| A17 | Cn | 1.3165 | 0.9662 | ❌ +35% |

### 2.4 刚性尾流频率（论文 Figure 12/13/14/15）
| 门 | 内容 | 结果 |
|---|---|---|
| Figure 12 尾流谱主峰 | AR=2、α=15°、Re=48,700 | St = 0.545 vs 0.58 → **PASS**（−0.034）|
| Figure 13/15 St(α) | Re=24,300/36,500 | MAE 0.136 vs 门 0.03 → **FAIL**（15–19° 支路 5–10% 内，低角高频支路未复现）|
| Figure 6 曲线（膜翼 9 点）| zmax/c | MAE 0.0046 ≤ 0.006（但无 H7 收敛证书，标注 `UNQUALIFIED`）|

### 2.5 一致性/合同回归（每次提交都跑）
核心套件 32/32、几何契约 10/10、LEV 账本 5/5、通量律 8/8、窗口 7/7、绑定值 parity 6/6。

---

## 3. 目前存在的问题

### 3.1 计算精度不足（量化）
| 问题 | 量化 | 说明 |
|---|---|---|
| 膜翼 Cn 系统性偏高 | **+34% ~ +36%**（A16/A17）| 与刚翼对照同量级 ⇒ **势流类模型边界**（缺黏性前缘损失/失速），非膜耦合 |
| 刚翼 α=19° 偏低 | **−28%**（0.578 vs 0.800）| 与膜翼偏高**方向相反**，说明载荷侧仍有定义问题（TE 支路两种强度口径已实现但未裁决）|
| α=15° 无法到 t*=10 | 发散 | 见 3.2 |
| 尾流频率曲线 | MAE 0.136 | 低角高频支路未复现 |
| 谱分辨率曾被误报 | 真实 ΔSt = **0.267 / 0.400**（Welch 段长决定），此前按 1/T* 理解 | 已改为显式输出 `psd_grid_spacing_st` |

### 3.2 容易发散（本报告最重要的机理结论）
**现象**：α=15° 在步 635 前稳定（cn 0.4–1.1），随后指数爆炸到 1e8；α=19° 有界但大幅振荡。

**已定位的因果链**（全部有测量）：
1. `Γ_bound,LE` 被钉在临界值（0.03420，三角度相同），**逐步不变**；
2. 每步脱落量 `λ` 与它之比随攻角暴涨：α=5 → **0.36**，α=15 → **2.16**，α=19 → **2.62**；
3. 净前缘丝 `Γ−λ` 达到临界值的 **−3.3 倍**（量级与符号都错）；
4. 队列分解显示：爆前 LE 感应由**累积尾迹**主导（older 26%、6–10 排 25%、4–5 排 22%，最新带仅 3%，平均距离 1.3c）；
5. 但**把远场感应从 LE 约束里移除会立刻失稳**（window=3 第 50 步即 2.6e7）⇒ 远场感应是**稳定因素**。

**结论**：失稳是**"LE 钉在临界 + LE 不穿透解出 newborn"这一闭包在持续深分离下的内在不稳定**。文献（N-LEV、VoFFLE）对同类模型记载了同一失效模式，其补救均为**脱离/上限判据**（`Γ_LEV/(cU) ≲ 4.2`、或尾缘流动反转时脱离）。

### 3.3 计算慢（实测）
| 路径 | 每步耗时 | 主要成本 |
|---|---|---|
| 膜翼 Q16 FSI（A5 / A21）| **8.8 / 13.2 s/步** | Q16 Newton 子步 ×10、Mf1（71 MB）作用、稠密切线刷新、~1.5k host sync/步 |
| 刚翼 V5M 15×30（含 ~27k 粒子）| **1.4–2.5 s/步** | 粒子/尾迹 Biot–Savart（O(N·M)）、ribbon 沉积、LE 闭合的 k 次线性求解 |
| 一次 1000 步 × 2 角度刚翼红测 | **~45–70 分钟** | — |

已做过的加速（5×，膜翼 31.7→6.3 s/步）：融合 warp 环核、粒子远场早退、Mf1 同源映射顺带删除 1.47 GiB 临时量、切线批量预算。剩余大头是 Q16 结构求解与粒子感应。

### 3.4 观测/评分层的已知缺陷（不影响求解，但影响结论可信度）
- `kelvin_max_abs` 是**恒等式**（按定义恒 0），不能当证据；
- H2/H3/H4 都没有 H7 网格/时间步/记忆收敛证书，一律标注 `UNQUALIFIED`；
- 谱分辨率不足（ΔSt 0.27–0.40）不足以裁决 St 支路物理。

---

## 4. 已否证的路线（避免重复劳动）

| 路线 | 否证证据 |
|---|---|
| retention/age-cull 是发散根因 | age=200、**零回收**，α=15 仍发散（max\|cn\| 7.1e9）|
| 近场涡带"堆叠" | 爆前最新队列仅占 LE 感应 **3%**（500 步几乎不变），文献也判定 0.13–0.17σ 属重度重叠无害 |
| 净前缘丝 pin | `net/target = −3.3`、`λ/Γ = 2.2` ⇒ 会把 bound 推到 3.5×crit，**加重** runaway |
| 通量律脱落 `½U_s²Δt` | 有界（newborn ↓5 个量级、max\|cn\| 1.18）但**过预测升力**：A5 +9%、A15 +29% |
| Kelvin 脱落 `λ=−ΔΣΓ` | bound 已定常 ⇒ `λ = 0`，LEV 会完全消失 |
| 窗口化远场感应 | 100 步=构造性 no-op；3 步=**立即失稳** |

---

## 5. 下一步（带判据）

1. **L2-4 诊断（零新物理，先用已存轨迹）**：算每带 `Γ_LEV/(cU)` 与文献上限 4.2 的对比，确认发散前后是否越限、越限倍数；
2. 若越限 → 实现**脱离/上限判据**（flag 门控，文献值，不调参）：`Γ_LEV/(cU) > 4.2` 或尾缘流动反转时停止该带释放并转自由涡；
3. **红测（沿用）**：α=15 `step≥2 max|cn| < 5` 且 α=5 保持 `0.3425±0.003`；默认生产路径逐位不变（parity 6/6）；
4. 并行（不改数值）：**L2-3** 量化 α=19° 的 −28% 中载荷侧 TE 支路份额；**L2-5** 把 `kelvin_max_abs` 改名或换成跨时环量账本。

---

## 6. 复现与环境

```bash
cd FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1

# 刚翼四角度短窗/长窗（红测驱动；--window/--flux/--legacy 为实验开关）
python3 artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/p1_te_retest.py \
  --steps 1000 --angles 15,5 --out <out>.json

# 诊断工具
python3 platform/warp_vpm/diagnose_roj_te_connectivity.py --require-connected --output <new-dir>
python3 platform/warp_vpm/diagnose_roj_flux_consistency.py --alpha 15 --t-star 3.5 --out <out>.json
python3 platform/warp_vpm/diagnose_roj_near_le_stacking.py --alpha 15 --steps 700 --out <out>.json
python3 platform/warp_vpm/diagnose_roj_net_filament_feasibility.py --alpha 15 --steps 300 --out <out>.json

# 合同回归
python3 -m pytest tests/test_roj_unified_case_runner_parity_gpu.py -q   # 6/6 绑定值
python3 -m pytest tests/test_q16_flux_v5m_native_gpu.py tests/test_q16_joint_deposited_flow_gpu.py \
  tests/test_q16_lev_strip_ledger_gpu.py tests/test_vpm_te_connectivity_gpu.py -q
```

**文件索引**（均在 `artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/research/`）：
`THREE_SHEDDING_LAWS_ADJUDICATED_DEFECT_IS_THE_FEEDBACK_PATH_20261121.md`、
`WINDOW_REFUTED_FAR_FIELD_INDUCTION_IS_STABILIZING_20261121.md`、
`NET_FILAMENT_REFUTED_SHED_AMOUNT_EXCEEDS_BOUND_20261121.md`、
`E1_FLUX_REDTEST_RESULT_MECHANISM_CONFIRMED_ACCURACY_FAILS_20261121.md`、
`NEAR_LE_COHORT_DECOMPOSITION_RULES_OUT_STACKING_20261121.md`、
`A15_SUSTAINED_SEPARATION_DIVERGENCE_AND_NET_PIN_ATTEMPT_20261120.md`；
数据在 `artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/`。
