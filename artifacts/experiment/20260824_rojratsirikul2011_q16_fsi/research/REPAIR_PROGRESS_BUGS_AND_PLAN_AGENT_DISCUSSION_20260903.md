# Rojratsirikul 2011 复现：修复进展、Bug 台账与后续方案（协作 Agent 讨论版）

> 日期：2026-09-03　|　代码基线：`f566856`（分支 `run/q16-lev-tev-pc-fsi-20260821`，已推送）
> 运行树：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca/`
> 姊妹文档：**误差数值全表见本目录 `ERROR_REPORT_FOR_AGENT_HANDOFF_20260903.md`**（基线 cc9bb3f，本文不重复）；
> 原始任务合同：`HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829.md`。
> 本文目的：①对照 handoff 验收节点给复现进展；②完整的 bug 台账（已修/未修，含证据提交）；③合并 research pipeline 后的后续修改方案；④**给协作 agent 的 6 个具体待决问题**。

---

## 1. 复现进展（对照 handoff §12 验收节点 G0–G10）

| 节点 | 内容 | 状态 | 证据 |
|---|---|---|---|
| G0 | 数据来源（SHA/行数/唯一键）| ✅ PASS | `rojratsirikul2011_observations.py` + 9 冻结测试；A10 Cn 修正 0.50–0.52→0.5569（`0f321e5`）|
| G1 | 观察器定义（max(mean z)、Cn 投影、St* 闭合）| ✅ PASS | 同上 + `wake_probe_observer.py` 5 GPU 测试 |
| G2 | A16 正式运行双观测 | ⚠️ zmax PASS（+0.0018）/ Cn FAIL（+0.3435）| t*=21 长窗 payload；**但见 §2-B1：Cn 定义本身存疑** |
| G3 | U=5 关键工况（A10/A17/A23）| ⚠️ 全部完成、无调参，zmax 全 PASS、Cn 全 FAIL、A17 模式 oracle FAIL | t*=10 扫掠 payload（9 点）|
| G4 | U=5 全曲线 | ⚠️ 柔性 9 点 H2 MAE 0.0046 数值过门（无 H7 证书→UNQUALIFIED）；刚翼 9/13 点 | `comparison/scores.json` |
| G5 | rigid Figure 9 | ⚠️ U5 8 点 + U10 1 点，H4 MAE 0.269/0.294 FAIL（α=21..30 中断待续）| `model_observables.csv` |
| G6 | Figure 12 尾流谱 | ✅ **H5 PASS**（St 0.5455 vs 0.58±0.05；t*=40 正式探针谱）| `ROJR-RIGID-U10p0-A150.json`（`06d9f03`）|
| G7 | Figure 13/15 | ⚠️ H6 MAE 0.136 FAIL；15–19° 支路 5–10% 过、低角高频支路（9°: 0.58 vs 1.02）未复现 | 7 点 |
| G8 | U=7.5/10 泛化 | ⏸ **有意冻结**（载荷 owner 裁决前跑=复制偏差）| — |
| G9 | H7 收敛证书 | ❌ 未建立（网格/dt/尾迹记忆收敛矩阵未跑）| `h7_certificate.json` 不存在 |
| G10 | 论文级报告 | 部分（图 7 张 + scores + 失败表齐；owner 未决则 Cn 类结论不可写）| `comparison/` |

**关键原则（用户裁决 2026-09-03）**：最小验证先行——任何长工况（≥2h）前先跑 3 点短切片验证基本假设，否则全是垃圾计算。该原则已写入执行流程。

---

## 2. Bug 台账（初步分析，按严重度排序）

### B1【最高】载荷双消费者（Cn 定义与结构 RHS 不同源）——**已定量、未裁决**

- **事实**：正式 Cn 历史上只消费 `constant_pressure` 合力；而结构子步 RHS 消费 `constant + velocity_force + Mf1·a`（`q16_flux_v5m_native_fsi.py` `_integrate_structure`）。
- **3 点最小验证实测（t*=1.5，2026-09-03，α=5/16/25）**：

| α | cn_constant | cn_full_action | 差 |
|---|---|---|---|
| 5° | 0.5455 | 0.6544 | **+10.9%** |
| 16° | 1.3746 | 1.3816 | +0.7% |
| 25° | 1.6793 | 1.5749 | **−10.4%** |

- **含义**：双 tap 差沿曲线**变号**、±10% 量级——owner 裁决前所有 Cn 精度声明含 ±10% 未声明不确定度。基础设施已齐（two-tap 记录 `cc9bb3f`、全空间 T_RB wrench、E0 observer `f566856`）。
- **未决**：哪个定义正确（见 §4-Q1）。**修复状态：未修（按方案冻结，等裁决）**。

### B2【高】5P 与 legacy 的广义投影差 ~1.0–1.2（载荷空间分布失真候选，E2 根因嫌疑）

- shadow 实测：统一 5P KJ+dΓ packet 与 legacy 的**净合力**只差 1.4e-5@步2 → 0.021@步60，但**广义投影差 0.93→1.23**（启动段，t*≤1.2，分解错位比较，不可作载荷误差声明——两轮独立审阅已纠正）。
- 点级功共轭 6e-17、传递闭合 1e-17（守恒传递骨架无问题）。
- **嫌疑链**：E2（顺度衰减模型/实验=1.0→0.5）可能由 legacy 投影的载荷分布失真造成——**未裁决**，等 E2-L1/L2 oracle + A/B。
- 修复状态：shadow 基础设施完成（`997e791`→`60c92c7`→`be51a44`）；消费者切换 **NO-GO**。

### B3【高，物理类】势流载荷偏差（E1）：无黏性前缘损失/失速

- 刚翼平板对照定量：ΔCn +0.164@5° → +0.488@19°；实验 21° 失速平台 0.78，模型线性升。
- α=5 顺度检验（zmax/Cn 模型/实验=1.01）证明低角 zmax +82% **完全**由载荷偏差线性传播（结构不软、预张力假说被证伪）。
- 修复状态：**未修**——M3 黏性载荷修正待做（进入条件：owner 线与 A5 inviscid oracle 先过）。

### B4【中】历史 ghost separation 嫌疑（A5：LESPPmax=0.153>0.11、3D pin active、LEV release=0）

- 来自 `MODIFICATION_PLAN` §1.2 合同缺陷清单；M2（3D LEV/TEV 环量事务状态机）未实施。
- 修复状态：未修。

### B5【已修】工具链/合同 bug（记录在此供协作 agent 排除重复劳动）

| bug | 修复 | 提交 |
|---|---|---|
| Cn 经 `cpu().tolist()`→numpy 计算（违 GPU-only）；`cpu_fallback_count` 写死 0 | CUDA fp64 点积 + 真实计数 | `902d5b1` |
| CLI `window_selection=None` 崩溃；求解器/artifact 退出码不分 | 序列化修复 + 独立退出码 3 | `a561f19` |
| 评分混合 U5/U10 MAE；无 H7 时 H2 宣称 PASS | U 分组 + UNQUALIFIED 标注 | `a561f19` |
| H6 匹配失败（模型精算 Re 24325 vs 论文标称 24300）| 按来流速度映射 | `2ad2c2c` |
| rigid 两处 `NativeV5MLoad` 重建丢 packet（双视图不一致）| 两处传递 + material 回退链 | `60c92c7` |
| shadow 证据合同缺陷（吞错、无 t*、support 级恒等式冒充点级功）| 硬门 + 分解账本 + CSR 转置点级功（6e-17）| `be51a44` |
| 旧数据包 A10 Cn 锚点偏差（0.50–0.52 vs 0.5569）| 注册表/适配器双侧同步 | `0f321e5`/`32a26ae` |

### B6【中，性能】（不阻塞精度但限制实验规模）

- A21 13.17 s/步（膜翼 FSI）；三测试套件串联时 1.59 GB Q16 稠密临时映射 OOM；
- 粒子/尾迹源数增长近线性拖慢（α=15@U10：~20k 粒子 + 9000 环）。
- 计划 P1–P4（聚合审计、删 1.47 GB unit-force 构造、cell-list/treecode、批处理）未实施。

---

## 3. 后续修改方案（合并两条 plan 线，research pipeline 三轮审阅 READY）

**方案文件**：`refine-logs/q16-v5m-gpu-load-contract-20260831/FINAL_PROPOSAL.md`（载荷合同线，`G000/E0` 已授权并完成）+ 本目录 `MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830.md`（M/P 线）。裁决门见 FINAL_PROPOSAL §6 结果到决策表。

### 当前位置（2026-09-03）
`G000`（4/4 ✅）+ `E0`（4/4 ✅，observer on/off 逐位 parity）刚完成（`f566856`）。§8 分叉已解锁但按最小验证原则暂缓长跑。

### 线 A：载荷 owner 裁决（B1/B2 的解药，zmax/Cn 精度前提）
1. **factorized exact H5**：`x_5P=C5[Hq;Ht]q`，`Q_5P=[Hq;Ht]ᵀC5ᵀf_5P`（rear=−q/3+4t/3；保留稀疏算子不物化 dense）→ 验证 5P 作用点运动+离散虚功；
2. **Mf1 受限子空间裁决（E2-M）**：`G=H5ᵀE_N(q,t)`，预注册方向集 A，range/residual 测试（τ=eps·max(2976,450)·σ_max）→ range fail=hybrid（5P+sealed Mf1）；
3. **E2-L1/L2 双层载荷 oracle**：L1 区间 wrench 一致性（event-free 才判）；L2 独立全局 impulse GPU 实现 + finite-CV 叠加；
4. **同 checkpoint A16 A/B**（旧/新消费者，比较 Cn/合矩/zmax/形状/反力/功/耦合残差/尾迹）→ 通过才 A10/A23 holdout。

### 线 B：气动类修正（B3/E1 的解药，Cn 精度主项）
1. **rigid A5 inviscid oracle**（先定位 Cn=0.363 vs 0.199 中有限翼 AIC/下洗/载荷的份额，**禁止**直接用黏性系数压数）；
2. A15/A19 separated oracle（尾迹记忆收敛：rows 300→600 等）；
3. Reynolds-aware 黏性闭合（优先级：边界层积分 > Reynolds-LESP/viscous decambering > 更高保真分支），**只用独立 rigid 静极线标定，一组参数跨 U/α 冻结**。

### 线 C：M2 事务 + 尾流（B4 + E5）
- 3D LEV/TEV 环量事务状态机（ATTACHED→NEWLY_SEPARATED→SEPARATED_ACTIVE→REATTACHING）+ GPU 账本；
- Figure 13/15 低角高频支路：侧缘探针加密 + 谱分解评分（主峰+谐波族，不再单 argmax）。

### 线 D：性能 P1–P4（B6）与 H7 证书（G9）
- H7 最低矩阵：V5M 15×30→21×42（ΔCn≤0.03）、Q16 5×10→7×14（Δz/c≤0.002）、dt* 减半、wake/particle 记忆加倍——任何上游门失败停止下游 sweep。

### 执行纪律（已由两轮独立审阅 + 用户裁决固化）
- 最小验证先行（3 点/短切片）；每步 parity 门；**禁止**：`5P+legacy` 直接相加、逐图调参、toy 替代、关闭 LEV、CPU 数值回退、在 owner 裁决前跑长工况。

---

## 4. 请协作 agent 重点分析的 6 个问题

- **Q1（owner 语义）**：Cn 的正确科学定义是 `constant_pressure` 合力（历史/论文对照惯例）还是结构实际消费的全分解刚体 wrench（`T_RBᵀQ_aero`）？若论文实验的 Cn 由天平测得（全载荷），后者更忠实——但 velocity/Mf1 分量的物理归属（真实气动力 vs 数值人工制品）如何裁决？
- **Q2（E2-M 设计）**：Mf1（added-mass 矩阵）能否被 `G=H5ᵀE_N` 的物理子空间表示？预注册方向集 A（结构模态/whitened production q̈/seeded random）是否充分？rank tolerance τ 的敏感性协议（0.1τ/τ/10τ）有无遗漏？
- **Q3（E2-L2 oracle）**：独立全局 impulse 实现的 event 组合二选一（whole-interval pre/post vs micro-interval 累加+表示转换）哪个对本求解器（粒子 age-cull + wake truncation 混合 retention）更可判定？cull/truncation 用固定 CV 通量还是标 inconclusive？
- **Q4（M3 闭合选择）**：低 Re（24k–49k）尖前缘平板的黏性损耗：边界层积分法 vs Reynolds-aware LESP decambering，哪个与现有 LESP 分离框架（lesp_crit=0.11 已有）耦合代价最小、标定数据（独立 rigid 静极线）需求最少？
- **Q5（E5 物理）**：Figure 13 低角高频支路（St~1.0@9°→0.66@13°）的机理候选：侧缘涡脱落 vs 剪切层不稳定性 vs 展向行波？我们的尾流探针布置（1c/2c 下游 × ±b/4 展向）能否分辨？是否需要在侧缘附近加密探针？
- **Q6（优先级裁决）**：在"owner 裁决（线 A，~数天）"与"M3 黏性修正（线 B，Cn 误差主项 +34~65%）"之间，若资源只够一条线先行，哪条对"可信复现 Figure 6/9"的期望收益更大？（当前内部倾向：A 先——因为它决定一切 Cn 数值的定义；但 B 是误差幅度的主项。）

---

## 5. 关键索引

- 提交链（精度/合同线）：`0f321e5`→`06d9f03`→`32a26ae`→`a561f19`→`06d6a57`→`997e791`→`60c92c7`→`be51a44`→`902d5b1`→`cc9bb3f`→`f566856`
- 数据清单：`artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/DATA_MANIFEST.md`（文件地图+五假设-字段映射）
- 独立审阅：`refine-logs/roj-q16-v5m-repair-20260830/CLAIMS_FROM_M1_3_RESULTS.md`、`CLAIMS_FROM_M1_4_SHADOW_RESULTS_20260831_*.md`（两轮 claim_supported: no/supplement 的纠正已全部落实）
- 文献综合（传递先例）：`idea-stage/q16-v5m-gpu-load-contract-20260831/LITERATURE_SYNTHESIS.md`（Farhat 1998、Yamano、Werter 2017）
- 复现命令与验证脚本：见 DATA_MANIFEST §2 与 `tests/test_e0_rhs_observer_parity_gpu.py`（E0 门）
