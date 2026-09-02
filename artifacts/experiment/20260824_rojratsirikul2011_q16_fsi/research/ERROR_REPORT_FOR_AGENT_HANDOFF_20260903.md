# Rojratsirikul 2011 Figure 6/9/12–15 复现误差报告（协作 Agent 交接版）

> 日期：2026-09-03
> 任务来源：`HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829.md`（本目录）
> 代码基线：`cc9bb3f`（分支 `run/q16-lev-tev-pc-fsi-20260821`，全部已推送）
> 运行树：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca/`
> 本报告目的：**自包含地给出全部已测误差、已定案/未定案的误差源、以及待协助定位的开放问题**。
> 本报告只陈述已测事实与已冻结裁决，不含新的计算。

---

## 1. 复现对象与合同（一段话）

按 handoff，Figure 6（膜翼时间平均最大位移 zmax/c）与 Figure 9（柔性/刚性 Cn）是 Q16–FLUX-V5M FSI 的直接 oracle；Figure 12（刚性翼尾流谱）、13/15（刚性有限翼脱涡 St/St·sinα）必须用独立刚性 AR=2 case + 尾流探针，**不得**用膜位移谱冒充。实验真值来自全曲线数字化包（`observations/figure_digitization_20260829/`，SHA 冻结，含不确定度列）。刚性对照的物理意义：刚翼与膜翼共用同一气动类（同网格/LESp/尾迹参数），其偏差即"模型类偏差"的干净测度。

## 2. 冻结配置（所有数据共用，无逐工况调参）

| 项 | 值 |
|---|---|
| 气动网格 / 方法 | V5M 15×30，separated LEV 强制、joint TEV、free wake、粒子，`wake_history=bound_rate` |
| LESP_crit | 0.11 |
| 尾迹 | wake rows 300 / free rows 100；粒子容量 32768 / age 100 |
| 结构 | Q16 5×10（2976 DOF，每节点 [r(3)|d(3)]，d=半厚×法向），Newmark，10 子步/气动步 |
| 材料 | E=2.2 MPa，eta=0.1（Kelvin-Voigt），预张力假设 0 |
| 时间步 | dt*=0.01；膜翼扫掠 t*=10（锚点 A16/A17 t*=21）；刚翼 U5 曲线 t*=30，Fig12 锚点 t*=40 |
| 硬件/精度 | RTX 4090，CUDA float64，GPU-only（Cn 已改 GPU 点积，`cc9bb3f`）|
| 几何/来流 | c=0.0688 m，b=0.1375 m（AR≈2），U=5/7.5/10 ↔ Re=24,300/36,500/48,700，ρ=1.208 |

## 3. 已完成计算与评分（截至 2026-09-03）

### 3.1 评分总表（scores.json，`comparison/` 目录）

| 门 | 结果 | 门限 | 判定 |
|---|---|---|---|
| H1 A16 zmax/c | 0.0451 vs 0.04338（+0.0018）| ±0.0045 | **PASS** |
| H1 A16 Cn | 1.2635 vs 0.9200（+0.3435）| ±0.08 | FAIL |
| H2 Figure 6（9 点 MAE）| 0.0046（U5）| 0.006 | 数值过门，**无 H7 证书标注 UNQUALIFIED** |
| H3 Figure 9 柔性（9 点）| MAE 0.444（U5）| 0.10 | FAIL |
| H4 Figure 9 刚性 | U5: 0.269（8 点）/ U10: 0.294（1 点）| 0.08 | FAIL |
| H5 Figure 12 谱峰 St | 0.5455 vs 0.58（−0.0345）| ±0.05 | **PASS** |
| H6 Figure 13/15 St | MAE 0.136（7 点）| 0.03 | FAIL |

### 3.2 膜翼 U=5 全曲线（9 点；zmax 括号为偏差，Cn 为模型值/实验值）

| α | zmax/c 模型 | 实验 | 偏差 | Cn 模型/实验 |
|---|---|---|---|---|
| 5 | 0.0358 | 0.0197 | **+82%** | 0.465 / 0.259（+80%）|
| 10 | 0.0399 | 0.0313 | +27% | 0.824 / 0.557（+48%）|
| 13 | 0.0414 | 0.0354 | +17% | 1.038 / ≈0.75 |
| 16* | 0.0451 | 0.04338 | **+4%** | 1.263 / 0.920（+37%）|
| 17* | 0.0436 | 0.04396 | **−1%** | 1.317 / 0.966（+36%）|
| 19 | 0.0433 | 0.0458 | −5% | 1.444 / ≈1.03 |
| 21 | 0.0439 | 0.04763 | −8% | 1.578 / ≈1.07（实验已失速）|
| 23 | 0.0445 | 0.04715 | −6% | 1.710 / 0.997 |
| 25 | 0.0451 | 0.04696 | −4% | 1.837 / ≈0.95（实验回落）|

\* A16/A17 为 t*=21 长窗锚点；其余 t*=10（fallback 窗，见 §5.3 窗敏感性）。

### 3.3 刚翼 U=5 曲线（含 U10 α15 锚点）

| α | Cn 模型/实验 | 偏差 | St 模型/实验 | St·sinα 模型 |
|---|---|---|---|---|
| 0 | 0.000 / 0.020 | — | 无峰（附流，诚实 NaN）| — |
| 5 | 0.363 / 0.199 | **+82%** | 0.600 / —（无 oracle）| 0.052 |
| 9 | 0.639 / 0.451 | +42% | **0.578 / 1.016（−43%）** | 0.090 |
| 11 | 0.773 / 0.552 | +40% | 0.851 / 0.949（−10%）| 0.162 |
| 13 | 0.906 / 0.647 | +40% | 0.901 / 0.658（+37%）| 0.203 |
| 15 | 1.036 / 0.700 | +48% | 0.590 / 0.645（−8.5%）| 0.153 |
| 17 | 1.163 / 0.7615 | +53% | 0.550 / 0.578（−4.8%）| 0.161 |
| 19 | 1.288 / 0.780 | +65% | 0.616 / 0.562（+9.6%）| 0.201 |
| U10 15 | 1.036 / ≈0.84 | +23% | **0.545 / 0.58（−6%，H5 PASS）** | 0.141 |

---

## 4. 误差源裁决总表（本报告核心）

### E1【已定案·主因】势流类载荷偏差：无黏性前缘分离损失/无失速

- **证据**：刚翼平板对照（同 oracle、同冻结参数）：ΔCn 从 +0.164@5° 单调增至 +0.488@19°；实验平板 ~21° 失速回落至 0.78 平台，模型线性上升无失速。低角相对偏差同样巨大：实验 Cn@5°=0.199 仅为薄翼势流值（0.54）的 37%（低 Re 尖前缘升力线斜率亏损），模型 0.363（67%）。
- **传播**：α=5 顺度比（见 E2 定义）=1.01 ⟹ 低角 zmax +82% **完全**由载荷 +80% 线性传播（结构不软）。
- **解释范围**：H3/H4 FAIL 主体；H6 部分经由载荷影响分离剪切层。
- **修法（已设计未实施）**：M3 黏性载荷修正——LESP 耦合的有效环量衰减（黏性减弯度），入轨条件：M1/M2 通过 + rigid A5 inviscid finite-wing oracle 通过（先定位 Cn=0.363 vs 0.199 中有限翼下洗/AIC 的贡献，**禁止**直接用黏性系数压数）。见 `MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830.md` §7。

### E2【未定案·强候选】顺度随载荷衰减过快（载荷链双消费者嫌疑）

- **现象**：顺度（zmax/Cn）比 模型/实验 = **1.01@5° → 0.49@25°**；实验顺度几乎不变（4.5–5.0×10⁻²），模型从 7.7 降到 2.5×10⁻²。高角 zmax "吻合"（±5%）是**载荷偏高×顺度偏低两误差相消**。
- **已发现的确定性 bug（载荷链双消费者）**：
  - 正式 Cn 历史上只消费 `constant_pressure` 合力；结构 RHS 消费 `constant + velocity + Mf1`；
  - 统一 5P packet 与 legacy 的**净合力**差：1.4e-5@步2 → **0.021@步60**（随发展增大）；
  - **广义投影差**（5P 即时映射 vs legacy constant-only，分解错位的比较，仅作规模参考）：0.93@t*≤0.6 → **1.23@t*1.2**（全部启动段样本）；
  - **two-tap 实测**（`cc9bb3f` 起每步落盘）：步 3 `cn_constant_current=0.0518` vs `cn_full_action=0.0399`，**输出定义差 23%**；
  - legacy 分解账本（t*0.5–1.2）：constant 范数 0.0145–0.019，velocity 0.0012–0.002，Mf1 action 0.0012–0.0018（后两者各约 constant 的 10%，不可忽略）。
- **传递骨架已验证无损**：5P→Q16 守恒传递力闭合 1.8e-19、矩闭合 8.7e-19、**点级功共轭 Q·q̇=Σfᵢ·vᵢ 相对误差 6e-17**（权重 CSR 转置重构点速度）。
- **未决**：E2（顺度异常）与双消费者投影差的**因果链未裁决**——需要 E2-L1（时间对齐双层载荷 oracle）/E2-L2（全局冲量 oracle）/E2-M（Mf1 受限物理子空间测试）+ 同 checkpoint A/B。规格见 `refine-logs/q16-v5m-gpu-load-contract-20260831/FINAL_PROPOSAL.md` §5.2–5.5。
- **文献先例**（5P 保守转置传递合法）：Farhat et al. 1998（CMAME）、Yamano et al. ICCFD12、Werter 2017（TU Delft 论文）。

### E3【已排除】统计窗启动污染

t*∈[1,10] vs [4,10] vs [7,10] 重算 zmax 变化 ≤0.005 且晚窗更大（膜仍在缓慢鼓胀）——方向与"启动膨胀"相反，排除其为 α=5 +82% 的原因。

### E4【已测·与 E1 同源】载荷空间分布前载过强

平均压力图弦向重心：α=16 → x/c=0.253（峰 0.10），α=5 → 0.431；实验侧前载指纹 ~0.35（max-camber 站位 x/c=0.35）。势流 LE 吸力峰未被黏性削减。

### E5【部分定位】尾流 St 选模错位

模型谱为准周期谐波族（基频 ~0.4 + 0.8/1.2 谐波），各角锁不同谐波：9° 锁 0.58（实验 1.02）、11°/13° 锁 0.85/0.90（实验 0.95/0.66）、15–19° 锁 0.55–0.62（实验 0.56–0.65，**吻合 5–10%**）。低角高频支路（侧缘/剪切层不稳定性）未被正确激发。α=0 附流无峰（诚实 NaN，非误差）。

### 已排除的其他假设

- **预张力/结构过软**：α=5 顺度比 1.01 证伪（若软则 >1）。
- **retention 假峰**：粒子平台平稳（stationary=True 全部工况），逐步计数序列已存档可查。

---

## 5. 关键量化证据（复核/引用用）

1. **顺度检验表**（zmax/Cn×10²，模型 vs 实验）：α=5: 7.69/7.62（比 1.01）；10: 4.84/5.62（0.86）；13: 3.99/5.17（0.77）；16: 3.57/4.72（0.76）；17: 3.31/4.55（0.73）；19: 3.00/4.45（0.67）；21: 2.78/4.51（0.62）；23: 2.60/4.73（0.55）；25: 2.46/5.03（0.49）。
2. **膜/刚对照（同 α 同 oracle）**：膜Δ vs 刚Δ：5°: +0.207/+0.164；13°: +0.298/+0.273；17°: +0.350/+0.402；19°: +0.390/+0.488 ⟹ 膜耦合本身贡献仅 ±0.05–0.1，偏差主体是气动类。
3. **载荷链测量**（§4 E2 引用的全部数字）来自 A16 真实运行 shadow（cadence 50）与 two-tap 记录。

## 6. 数据与文件地图（协作 Agent 入口）

```
运行树 = /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/
  DATA_MANIFEST.md                     ← 文件地图 + 复现命令 + 假设-字段映射
  membrane_sweep/ROJ11-*_T10.{json,z_history.npz}   ← 膜翼 9 点（含全场 z(x,y,t) 原始时序）
  cases/ROJR-RIGID-*.json (+_probe_history.npz)     ← 刚翼曲线（逐步 Cn/粒子数/12 探针速度时序）
  model_observables.csv / checkpoint.json           ← P2 标准输出 / 断点
  comparison/{7 张对比图, scores.json, case_failures.csv, progress.log}
  run.log / chain.log                  ← 完整运行时间线
artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current/ROJ11_{A16,A17_MODE}_FULL.*  ← 长窗锚点
artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/
  observations/figure_digitization_20260829/        ← 实验 oracle（SHA 冻结 + 不确定度）
  research/REPORT_ERROR_ANALYSIS_ROJ_20260830.md    ← 此前误差分析（本报告的前身）
  research/MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830.md  ← M0-M4/P1-P4 修改方案
refine-logs/q16-v5m-gpu-load-contract-20260831/FINAL_PROPOSAL.md   ← 载荷合同研究终案（G000/E0/E1/E2 规格）
refine-logs/roj-q16-v5m-repair-20260830/CLAIMS_FROM_M1_*.md        ← 两轮独立复核裁决
```

关键代码：`src/fluxvortex/runtime/case_runner.py`（runner/双 tap）、`src/fluxvortex/warp_fsi/q16_flux_v5m_native.py`（Q16 求解+packet 发布）、`q16_flux_v5m_native_fsi.py:339` 起（生产 RHS 消费链）、`rigid_flux_v5m_native.py`、`aero/v5m/load_packet.py`（统一 packet）、`q16_shadow_resolved_consumer.py`、`q16_rigid_body_wrench.py`。

## 7. 待协助定位的开放问题（按优先级）

1. **E2 因果裁决**：双消费者广义投影差（~1.0 量级、分解错位参考值）是否解释顺度衰减 1.0→0.5？需要实现 FINAL_PROPOSAL §5.4 的 E2-L1/L2（时间对齐区间 wrench 比较 + 独立全局冲量 oracle，从保存 primitives 独立重算，禁止调用生产 KJ/author 例行）。**当前 GO 状态：observer/dual-owner schema GO，consumer switch NO-GO。**
2. **Mf1 可表示性（E2-M）**：Mf1 与 H5 的 director 行/列合同；对预注册加速度方向 A 的 `Y=Mf1·S_f·A` 相对 `range(G=H5ᵀE_N)` 的 residual（GPU rank tolerance τ=eps·max(2976,450)·σmax，报告 0.1τ/τ/10τ 敏感性）。范围失败 ⟹ 永久转 hybrid；范围过而 C_phys 失败 ⟹ 禁 point-action 主张。
3. **A5 inviscid finite-wing oracle（M3 入轨门）**：在引入任何黏性闭合前，定位刚翼 Cn=0.363 vs 实验 0.199 中有限翼下洗/AIC/载荷公式的贡献（对照 lifting-line/Helmbold、冻结 Ptera attached solver、KJ 法向 vs 压力法向积分）。**禁止用黏性系数直接把 0.363 压到 0.199。**
4. **低角高频 St 支路（E5）**：9° 实验 St=1.02 的物理来源（侧缘涡脱落/剪切层）在当前 LEV 条带释放+自由尾迹中为何不被激发；建议从已存 `*_probe_history.npz` 做谱分解（主峰+谐波族）而非单 argmax。
5. **H7 收敛矩阵**：网格 15×30→21×42、Q16 5×10→7×14、dt*×2、wake 300→600、粒子 age 100→200——H2 的 UNQUALIFIED 标注等待此证书解除。
6. **性能（独立线，勿与科学线混提交）**：膜翼 ~9–13 s/步；1.59 GB `_normal_shape_matrix` 稠密临时量在测试串跑时 OOM（方案 P2-1）；涡粒子/尾迹增长导致的步时漂移（P3 空间加速）。

## 8. 复现命令

```bash
cd <运行树>
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1
# 膜翼单点（例 α=13）
python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-SWEEP-A13 --max-aero-steps 1000 \
  --output artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/membrane_sweep/ROJ11-SWEEP-A13_T10.json
# 刚翼队列（可恢复）
python3 platform/warp_vpm/queue_roj_rigid_fig9_12_13_15.py
# 全部对比图 + 评分
python3 platform/warp_vpm/compare_rojratsirikul2011_digitized_oracles.py
# G-M0 profile / 两步 parity
python3 platform/warp_vpm/profile_roj_q16_v5m.py --case ROJ11-A16 --steps 3
```

## 9. 裁决与红线（协作 Agent 必须遵守）

- 消费者切换 **NO-GO**；`5P + legacy` 直接相加 **PROHIBITED**；未经互斥证明的 `5P + Mf1` **PROHIBITED**。
- 禁止逐工况/逐图调参；禁止关闭 separated LEV/TEV/free wake；禁止 Q4/Q9/Ptera 生产化；禁止 CPU 数值回退（含 Cn——已修）。
- 失败工况必须留在 `model_observables.csv` / `case_failures.csv`，不得从评分集中删除。
- 数字化不确定度（zmax ±0.0007，Cn 柔 ±0.012/刚 ±0.02，St ±0.012）是**图读误差**，不是求解器容差，更不是实验置信区间。
