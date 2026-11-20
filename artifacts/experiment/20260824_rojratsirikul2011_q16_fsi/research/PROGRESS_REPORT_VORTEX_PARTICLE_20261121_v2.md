# FLUX-V5M 涡粒子方法：进展汇报（2026-11-21，第 2 版）

> 代码基线：分支 `run/q16-lev-tev-pc-fsi-20260821`，HEAD `34e0529`（全部已推送）
> 工作树：`FLUXV_RUNS/v5m-fa8eaca`
> 本报告自包含：①方法原理 ②已比对/验证的 case 与数字 ③当前问题（精度 / 发散 / 速度）④已被数据否证的路线 ⑤模型边界判定 ⑥下一步与判据
> 上一版：`PROGRESS_REPORT_VORTEX_PARTICLE_20261121.md`（`5cc74d1`）；本版新增 L2-4 与模型边界判定

---

## 0. 一页纸结论

| 项 | 状态 |
|---|---|
| **能复现的** | **刚翼 AR=2、U=5 m/s、α=5°：Cn = 0.3425 vs 实验 0.3413（+0.35%）**，且完全来自几何修复；α=0° 得 0.000（对称应得 0）|
| **不能复现的** | α=15° 在 t*≈5–7 后**发散**；α=19° 有界但振荡、均值 **−28%**；膜翼 A16/A17 的 Cn **+34~36%** |
| **机理判定（新）** | 深分离失稳是**"LEV = 每步一条 ribbon + LE 不穿透解 newborn" 这一表示的内在模型边界**，不是单个判据/系数能补的实现 bug |
| **已修的确定性缺陷** | TE 涡片几何连通（此前是假修复）、Mf1 面积/半系数、CPU 上的 Cn、每带 LEV 账本、谱分辨率误报 |
| **速度** | 膜翼 Q16 FSI **8.8–13.2 s/步**；刚翼 V5M 15×30（~27k 粒子）**1.4–2.5 s/步** |

---

## 1. 涡粒子方法计算原理（本求解器实际实现）

### 1.1 三种表示的混合
| 表示 | 承载 | 数值实现 |
|---|---|---|
| **bound 涡环**（V5M/UVLM）| 附着涡量 | `nc×ns = 15×30 = 450` 个四边形环，环量 `Γ`（m²/s），有限涡丝 + 四阶核（`core_fraction=1e-6`；自感项必须为负，硬门）|
| **LEV 高斯粒子** | 前缘分离涡 | 每条分离带**每步沉积一条相连 ribbon**（Gaussian，σ=0.04c，目标间距 0.018c，frontier 位移 0.005–0.0066c）|
| **TEV / 自由尾迹涡环** | 后缘脱落与远尾迹 | 逐步在 TE 插入新环行（`γ_tev = Γ_TE + γ_lev`）；free rows 对流、older rows 冻结；超容量截断并审计环量/线冲量 |

### 1.2 每步求解顺序（`Q16NativeV5MSolver.propose`）
1. **影响矩阵** `A`：`A[i,j] = n_i·u_j(单位环)`（1/m）。
2. **不穿透 RHS**：`rhs_i = n_i·(V_body − U∞ − V_wake − V_particles)` —— **已释放的 LEV 粒子感应已在 RHS 内**。
3. `γ_pre = A⁻¹ rhs` → **LESP**：`lesp = −γ_pre[LE行]/scale`，`scale = c·U·(θ+sinθ)/1.13`。
4. **分离判据**：`|lesp| > lesp_crit = 0.11`（3D 实际表面唯一 owner）。
5. **分离求解（joint 路径）**：LE 行钉在临界值（Dirichlet pin），**新生 LEV 强度由 LE 不穿透闭合方程解出**（用真实沉积 ribbon 的感应基，非 AIC 代理列）。
6. **沉积** ribbon → 粒子场，并记入每带账本（`lev_strip_last_release`）。
7. **材料对流**：joint 路径用**同步 RK3**（源与目标每子级同步）。
8. **尾迹推进**：新 TEV 行连接当前 TE 与**已对流的旧前沿**（保证涡片连续）。
9. **retention**：粒子按年龄回收，回收前记录环量/线冲量。
10. **载荷**：Kutta–Joukowski（4 线力/面板，共享边取半差，TE 行取 `Γ_n−Γ_{n−1}`）+ 非定常 `ρ dΓ/dt A n`；`Mf1 = ρ P(q) A⁻¹ N(q)`。
11. **两个 Cn 口径**：`cn_constant_current`（历史口径）与 `cn_full_action`（constant+velocity+Mf1 经 `T_RBᵀ` 投影）。

### 1.3 工程保证（硬门）
- 三相分离约束同解 + **实际沉积流场**不穿透（提交前校验，实测 3e-15–3e-14 m/s）；
- 载荷闭合 `Σ点力=总力`、`Σ r×f=总矩`（≤1e-12 相对）；
- 功共轭 `Q·q̇ = Σf·v`（6e-17）；
- 默认生产路径**逐位不变**由 `tests/test_roj_unified_case_runner_parity_gpu.py` 守卫（6/6，每轮都跑）。

---

## 2. 已比对与验证的 case

### 2.1 纯气动（外部基准）
| Case | 内容 | 结果 |
|---|---|---|
| **Izraelevitz 2017 / Scherer 1968 Fig.14** | 12 个俯仰-升沉工况的 CT | MAE 0.0386 → **0.0409**（提速后），偏差方向不变 |
| Yamano 2020 载荷门 | 结构-气动载荷一致性 | 密度线性、闭合门全过 |

### 2.2 刚翼 AR=2、U=5 m/s（论文 Figure 9 刚性系列；数字化 CSV 为真值）
| α | 模型 Cn（t*=10）| 实验 | 判定 |
|---|---|---|---|
| 0° | 0.000（无释放）| 0.020 | 对称应得 0，OK |
| **5°** | **0.3425（恒定）** | **0.3413** | **+0.35%** ✅ |
| 15° | t*≈5–7 后**发散**（step≥2 max\|cn\| 9.8e10）| 0.70 | ❌ |
| 19° | 有界强振荡，均值 0.5775 ∈ [0.24, 1.10] | 0.800 | ❌ **−28%** |

### 2.3 柔翼 Q16 FSI（论文 Figure 6 / 9 柔性系列）
| 工况 | 量 | 模型 | 实验 | 判定 |
|---|---|---|---|---|
| A16（U=5，α=16°，t*=21）| zmax/c | 0.0451 | 0.0434 | ✅ +0.0018 |
| A16 | Cn | 1.2635 | 0.9200 | ❌ **+34%** |
| A17（α=17°）| zmax/c | 0.0436 | 0.0440 | ✅ −0.0003 |
| A17 | Cn | 1.3165 | 0.9662 | ❌ +35% |

### 2.4 刚性尾流频率（论文 Figure 12/13/14/15）
| 门 | 内容 | 结果 |
|---|---|---|
| Figure 12 主峰 | AR=2、α=15°、Re=48,700 | St = 0.545 vs 0.58 → **PASS**（−0.034）|
| Figure 13/15 St(α) | Re=24,300 / 36,500 | MAE 0.136（门 0.03）→ FAIL；15–19° 支路 5–10% 内，低角高频支路未复现 |
| Figure 6 曲线（膜翼 9 点）| zmax/c | MAE 0.0046 ≤ 0.006，但无 H7 收敛证书 → 标注 `UNQUALIFIED` |

### 2.5 合同回归（每次提交都跑）
核心套件 32/32、几何契约 10/10、LEV 账本 5/5、通量律 8/8、窗口 7/7、绑定值 parity 6/6。

---

## 3. 目前存在的问题

### 3.1 计算精度不足
| 问题 | 量化 | 归因 |
|---|---|---|
| 膜翼 Cn 系统性偏高 | **+34% ~ +36%**（A16/A17）| 与刚翼对照同量级 ⇒ **势流类模型边界**（缺黏性前缘损失/失速），非膜耦合 |
| 刚翼 α=19° 偏低 | **−28%**（0.578 vs 0.800）| 与膜翼偏高**方向相反**；L2-3（TE 支路双候选）正在量化其中载荷定义的份额 |
| α=15° 到不了 t*=10 | 发散 | 见 3.2 |
| 尾流频率曲线 | MAE 0.136 | 低角高频支路未复现 |
| 谱分辨率 | 真实 ΔSt = **0.267 / 0.400**（Welch 段长决定）| 已改为显式输出 `psd_grid_spacing_st`，不再按 1/T* 误读 |

### 3.2 容易发散（本报告核心）
**现象**：α=15° 在步 635 前稳定（cn 0.4–1.1），随后指数爆炸到 1e8；α=19° 有界但大幅振荡（0.24–1.10）。

**已定位的因果链（全部有测量）**
1. `Γ_bound,LE` 被钉在临界值 **0.03420 m²/s**，三角度**完全相同**（因 `scale` 只含几何），且逐步不变；
2. 每步脱落量 `λ` 与之比随攻角暴涨：α=5 → **0.36**，α=15 → **2.16**，α=19 → **2.62**；
3. 净前缘丝 `Γ−λ` 达临界值的 **−3.3 倍**（量级与符号都错）；⇒ 把 pin 改到净前缘丝会把 bound 推到 **3.5×crit**，是加重而非修复（已否证）；
4. 队列分解：爆前 LE 感应由**累积尾迹**主导（older 26%、6–10 排 25%、4–5 排 22%，最新带仅 3%，平均距离 1.3c）；
5. 但**把远场感应从 LE 约束里移除会立刻失稳**（window=3 第 50 步即 2.6e7）⇒ 远场感应是**稳定因素**；
6. 累积 LEV 环量 `Γ_LEV/(cU∞)`：α=5 在步 300 达 **13.0**、α=15 达 **82.4**、α=19 达 **108.0**，而文献（N-LEV）脱离上限是 **4.2** ⇒ 超限 **3–25 倍**，越深分离越早越重（A15 约第 3 步越限）。

**判定**：这是**表示层的模型边界**——"每步一条 ribbon + LE 不穿透解 newborn"在持续深分离下没有稳定分支。文献对同类模型（N-LEV、VoFFLE）记载同一失效模式，其补救是**脱离判据/上限**，但该阈值**不可直接移植**（我们只能按 `U∞` 归一，且最好工况 A5 也超限 3 倍却完全稳定 ⇒ 强行加阈值会破坏已验证的 +0.35%）。

### 3.3 计算慢（实测）
| 路径 | 每步耗时 | 主要成本 |
|---|---|---|
| 膜翼 Q16 FSI（A5 / A21）| **8.8 / 13.2 s/步** | Q16 Newton 子步×10、Mf1（71 MB）作用、稠密切线刷新、~1.5k host sync/步 |
| 刚翼 V5M 15×30（~27k 粒子）| **1.4–2.5 s/步** | 粒子/尾迹 Biot–Savart（O(N·M)）、ribbon 沉积、LE 闭合的 k 次线性求解 |
| 一次红测（1000 步 × 2 角度，刚翼）| **45–70 分钟** | — |

已实现的 5× 提速：融合 warp 环核、粒子远场早退、Mf1 同源映射（顺带删掉 1.47 GiB 临时量）、切线批量预算。剩余大头是 Q16 结构求解与粒子感应。

### 3.4 观测/评分层缺陷（不影响求解，但影响结论可信度）
- `kelvin_max_abs` 是**恒等式**（按定义恒 0），不能当证据（待改名/替换）；
- H2/H3/H4 无 H7 网格/时间步/记忆收敛证书，一律 `UNQUALIFIED`；
- 谱分辨率不足（ΔSt 0.27–0.40）不足以裁决 St 支路物理。

---

## 4. 已被数据否证的路线（六条，避免重复劳动）

| 路线 | 否证证据 |
|---|---|
| retention / age-cull 是根因 | age=200、**零回收**，α=15 仍发散（7.1e9）|
| 近场涡带"堆叠" | 爆前最新队列仅占 LE 感应 **3%**；文献判定 0.13–0.17σ 属重度重叠无害 |
| 净前缘丝 pin | `net/target=−3.3`、`λ/Γ=2.2` ⇒ 会把 bound 推到 3.5×crit |
| 通量律脱落 `½U_s²Δt` | 有界（newborn ↓5 个量级、max\|cn\| 1.18）但**过预测升力**：A5 +9%、A15 +29% |
| Kelvin 脱落 `λ=−ΔΣΓ` | bound 已定常 ⇒ `λ=0`，LEV 会消失 |
| 窗口化远场感应 | 100 步=构造性 no-op（=age cap）；3 步=**第 50 步即失稳**（远场感应是稳定因素）|

---

## 5. 下一步（带预注册判据）

1. **L2-3（进行中）**：用已实现 flag `te_leg_candidate_diagnostic` + 逐步 ΔCn 记录，量化 α=19° 的 −28% 中「TE 腿按 `Γ_n−Γ_{n−1}` vs 按 `γ_tev`」的份额：
   - 若 ΔCn 显著（>0.1）⇒ −28% 部分属**载荷定义**，可实现 flag 修正 + 红测；
   - 若很小（<0.02）⇒ 归因气动解本身（与 §3.2 一致）。
2. **L2-5**：`kelvin_max_abs` 改名/替换为跨时环量账本，消除假证据。
3. **交付已验证范围**：α=0/5（U=5）的 **+0.35%**、几何/观测/载荷合同基础设施、Figure 12 主峰；
4. **α≥15 的正路**：表示层升级（近场相连缓冲 + LEV 正式定义 + `U_eff` 归一化），需先写推导与红测再动代码——数天量级，建议单独立项；**不再尝试调阈值/系数**。

---

## 6. 复现与环境

```bash
cd FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1

# 刚翼长短窗（红测驱动；--window/--flux/--legacy/--te-leg 为实验开关）
python3 artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/p1_te_retest.py \
  --steps 1000 --angles 15,5 --out <out>.json

# 诊断工具
python3 platform/warp_vpm/diagnose_roj_te_connectivity.py --require-connected --output <new-dir>
python3 platform/warp_vpm/diagnose_roj_flux_consistency.py --alpha 15 --t-star 3.5 --out <out>.json
python3 platform/warp_vpm/diagnose_roj_near_le_stacking.py --alpha 15 --steps 700 --out <out>.json
python3 platform/warp_vpm/diagnose_roj_net_filament_feasibility.py --alpha 15 --steps 300 --out <out>.json

# 合同回归
python3 -m pytest tests/test_roj_unified_case_runner_parity_gpu.py -q    # 6/6 绑定值
python3 -m pytest tests/test_q16_flux_v5m_native_gpu.py tests/test_q16_joint_deposited_flow_gpu.py \
  tests/test_q16_lev_strip_ledger_gpu.py tests/test_vpm_te_connectivity_gpu.py -q
```

**文件索引**（`artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/research/`）：
`L2_4_LEV_CIRCULATION_VS_LITERATURE_CAP_20261121.md`、
`WINDOW_REFUTED_FAR_FIELD_INDUCTION_IS_STABILIZING_20261121.md`、
`THREE_SHEDDING_LAWS_ADJUDICATED_DEFECT_IS_THE_FEEDBACK_PATH_20261121.md`、
`NET_FILAMENT_REFUTED_SHED_AMOUNT_EXCEEDS_BOUND_20261121.md`、
`E1_FLUX_REDTEST_RESULT_MECHANISM_CONFIRMED_ACCURACY_FAILS_20261121.md`、
`NEAR_LE_COHORT_DECOMPOSITION_RULES_OUT_STACKING_20261121.md`、
`A15_SUSTAINED_SEPARATION_DIVERGENCE_AND_NET_PIN_ATTEMPT_20261120.md`；
数据在 `artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/`。
