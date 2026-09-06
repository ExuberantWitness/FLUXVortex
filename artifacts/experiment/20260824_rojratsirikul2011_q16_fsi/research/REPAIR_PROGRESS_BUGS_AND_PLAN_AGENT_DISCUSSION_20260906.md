# Rojratsirikul 2011 复现：修复进展、Bug 台账与后续方案（协作 Agent 讨论版 · 2026-09-06）

> 代码基线：`6db341f`（分支 `run/q16-lev-tev-pc-fsi-20260821`，全部已推送）
> 运行树：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca/`
> 原始任务合同：本目录 `HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829.md`
> 前版协作文档：`REPAIR_PROGRESS_BUGS_AND_PLAN_AGENT_DISCUSSION_20260903.md`（误差数值表见 `ERROR_REPORT_FOR_AGENT_HANDOFF_20260903.md`）
> 本文目的：**自包含地给出 2026-09-03 以来的修复进展、经多轮独立复核确认的 bug 台账、以及按复核固定顺序的后续方案与开放问题**。

---

## 1. 一句话状态

**载荷合同与观测层（P0/G000/E0）已完成并通过全部验证；分离/尾迹物理线（P1）经四步修复后三角度（A5/A15/A19）首次全部稳定且 A19 发散被根治——但全部在实验 flag 下（默认 OFF，生产路径逐位冻结）；Cn 类精度声明仍被冻结，等长窗验证与共享边强度账本。**

## 2. 复现进展（对照 handoff §12 节点 + 修复线）

### 2.1 评分面（数值自 09-03 未动——按裁决有意冻结）

| 门 | 结果 | 备注 |
|---|---|---|
| H1 A16 | zmax PASS（+0.0018）/ Cn FAIL（+0.3435）| Cn 定义本身有双消费者问题（§3-B1）|
| H2 Fig6（9 点）| MAE 0.0046（U5）数值过门 | 无 H7 证书 → UNQUALIFIED |
| H3 Fig9 柔性 / H4 刚性 | MAE 0.444 / 0.269 | FAIL（E1 类偏差，未修）|
| H5 Fig12 | St 0.545 vs 0.58 | PASS（谱统计资格待重验证：ΔSt=0.27–0.40 窗口问题）|
| H6 Fig13/15 | MAE 0.136 | FAIL（低角高频支路；观察方法先于物理重查）|

### 2.2 修复线（09-03 → 09-06，全部经独立复核）

| 提交 | 内容 | 验证 |
|---|---|---|
| `902d5b1`→`f566856` | G000 全 4 项（GPU Cn、T_RBᵀQ_aero 全空间 wrench、two-tap 双 Cn、production-RHS observer+实际功带）+ E0 门 | E0 4/4；parity 6/6 |
| `6527a6f`+`eb3e764` | **P0 完成**：Mf1 统一到同源压力映射 `ρPA⁻¹N`（机器精度 1.0000000000000000，旋转协变恢复）；1.47GB unit-force 临时量退役；实际功补 Mf1·a；功比值双口径 | 算子诊断脚本；16 测试 |
| `3a9618c` | P1-1：联合分离求解（增广系统、AIC LE 列代理基）flag 化 | A5 级缺陷修复验证 |
| `fae7f57` | 复核修正：**实际沉积流场不穿透门**（揭穿代数残差假通过：0.98→0.136 m/s 实际破缺；代理基与真实涡带差 ~105%）| 8+31+6 测试 |
| `53dc83a` | P1-2 步1：**真实材料涡带算子**替代代理基（单位强度涡带临时场求感应列 + k×k LE 闭合）| α=5 完美、α=15 稳定 |
| `d81a6e9` | 复核修正：**RK3 材料对流同步**（源/目标每级同步；此前 2/3 级源停旧位）| 方环平移机器精度 |
| `6db341f` | **TE 尾迹重连接**（新生行连接当前 TE 与已对流旧前沿；断口恰 U·dt=0.01c 是持续分离失稳根因）| **A19 发散根治** |

### 2.3 修复链累积效果（150 步短窗，joint flag 开；**启动段值非稳态精度**）

| α | 代理基时代 | +RK3 同步 | +TE 重连接 | 实验 Cn |
|---|---|---|---|---|
| A5 | 0.397 | 0.289 | 0.104 | 0.199 |
| A15 | 发散 | 0.510 | 0.595 | 0.70 |
| A19 | 发散 | 发散(4.45×10⁶) | **0.716** | 0.78 |

门全绿：沉积流场 Neumann 3e-14/3e-15、Kelvin 1e-16 量级（注意 Kelvin 仍是赋值恒等式）。

## 3. Bug 台账（多轮独立复核后的状态）

### 已修复并验证（协作方无需重复劳动）
| # | Bug | 修复 | 关键验证 |
|---|---|---|---|
| F1 | Mf1 半系数+面积不一致（=同方程压力积分的 0.5cosα）| `6527a6f`/`eb3e764` | 算子比值 1.0（±2e-16）全角度 |
| F2 | E0 功记录缺 Mf1·a | `6527a6f` | w_nonacc/w_acc/w_total 三项+双口径比值 |
| F3 | CPU Cn + 写死 fallback 计数 | `902d5b1` | GPU fp64 点积 |
| F4 | LE 行被 pin 替换后不穿透破缺（0.98–2.86 m/s）| `3a9618c`+`53dc83a` | 沉积流场门 3e-15 |
| F5 | A5 触发分离约束却零释放（节点掩码未激活→重合抵消边）| `53dc83a` | 真实释放 3730/25711 粒子 |
| F6 | RK3 源/目标不同步 | `d81a6e9`（复核方）| 方环平移机器精度 |
| F7 | TE 尾迹片断接 U·dt（持续分离发散根因）| `6db341f` | A19 4.45e6→0.716 |
| F8 | 工具链七项（window=None 崩溃、U 混合 MAE、H6 Re 匹配、rigid packet 视图、shadow 证据合同、A10 锚点、GPU 合同）| `a561f19`…`be51a44` | 各提交内嵌 |

### 已确认未修（优先级序）
| # | 问题 | 现状/证据 | 位置 |
|---|---|---|---|
| U1 | **Cn 双消费者**（constant-only vs 结构实际消费的 constant+velocity+Mf1）| 双 tap 差随攻角变号（同窗均值三角度均为正；末帧 +19.96%/+0.51%/−6.21%，复核校正后口径）；owner 裁决前 Cn 精度声明含此不确定度 | `case_runner.py` two-tap 记录已就位 |
| U2 | **共享边 TE/LE 强度关系未推导**（材料涡片 vs bound-only pin；Kelvin 现为赋值恒等式不能验证共享边）| 复核指定为下一项 | `q16_flux_v5m_native.py` 尾迹/LEV 段 |
| U3 | **bound_rate 与 Mf2_1/Mf1 时间分解的相容性**（重复计入嫌疑未裁决）| 需 `Q=Q_ref+JvΔv+JaΔa` 同方程增量线性化+时间离散推导；不可盲删 | author_loads + FSI 子步 |
| U4 | **载荷导数未对同一活动约束系统求导**（分离后环量用 separated_aic，导数用原 aic）| 复核清单第 3 项 | propose 载荷段 |
| U5 | **E1 势流类载荷偏差**（刚翼对照 +0.16@5°→+0.49@19°，无失速）| 解药是 M3 黏性闭合，进入条件：P1/P2 先过 | 独立线 |
| U6 | **谱窗口 ΔSt=0.27–0.40 太粗**（Welch 分段压缩所致）| 先离线重分析（长段/整窗/分块稳定性）再判物理；H5 过门保留历史记录但资格待重验证 | `wake_probe_observer.py` |
| U7 | 性能债（A21 13.2 s/步、粒子/尾迹增长、历史 1.59GB OOM 已随 F1 顺带解决一项）| P1–P4 独立线 | — |

## 4. 后续修改方案（按最近一轮复核固定顺序 + research pipeline）

```text
（当前完成：真实 B 算子 → RK3 同步 → TE 重连接——三角度稳定）
① 共享边 TE/LE 强度关系：从实际共享边推导（Bird et al. 2021 式 17–19
   的材料涡片共享前缘丝强度 vs bound-only pin 明确区分）
② 材料历史与 LESP/Kelvin 约束统一到同一拓扑；Kelvin 改为非赋值式验证
   （跨时间、跨表示的实际环量/冲量账本）
③ 载荷导数对同一活动约束系统计算（修 U4）
④ 长窗验证（≥t*=10 统计窗）三角度 + dt 减半趋势 → 评估开 flag
⑤ A16 同 checkpoint 对照（joint on/off；Cn/合矩/zmax/形状/反力/功/耦合残差/尾迹）
⑥ rigid 有限翼载荷 oracle（A5 的 Γ(y)/下洗/有效面积同帧分解定位）
⑦ 若仍有系统偏差 → 独立标定的低 Re 闭合（M3；Reynolds-LESP 最小耦合代价）
⑧ H7 收敛矩阵 → U7.5/U10 泛化 → 论文曲线
```

**纪律**（多轮裁决固化）：最小验证先行（3 点短切片否证假设，不用于验收稳态）；每步 parity 门；默认路径逐位冻结直到整线过门；禁止盲删项/调阈值压 Cn/`5P+legacy` 相加/逐图调参。

## 5. 请协作 agent 重点分析的 4 个问题

- **Q-A（U2 账本推导）**：材料涡片共享前缘丝的强度应取"bound LE 环量钉临界后的超额"（当前实现）还是 Bird 17–19 的共享丝连续性条件（片与 bound 在共享边上强度相等）？两者在持续分离下的差异如何用非赋值式 Kelvin/冲量账本裁决？
- **Q-B（U3 时间离散）**：`Q=Q_ref+Jv·(v−v_ref)+Ja·(a−a_ref)` 的 Jv/Ja 应从哪个离散方程推导（Newmark 平均速度层的 velocity_force 已用 predictor 冻结速度——E3 的 provisional_causal 语义）才能保证参考态严格重建且不与 bound_rate 的完整 ΔΓ/Δt 双计？
- **Q-C（长窗设计）**：三角度稳态验证的最小可信窗（t*≥10 够吗？A15/A19 短窗值 0.595/0.716 已近实验，但启动段口径不能外推）+ 何时允许开 flag 做 A16 柔性翼对照？
- **Q-D（优先级裁决）**：U2/U3/U4 完成前是否值得并行启动 rigid 有限翼 oracle（U5 的定位前置）？复核口径是"方程一致性先行"，但 rigid oracle 不依赖 flag 线，可并行——资源如何分配？

## 6. 关键索引

- 提交链（本窗口）：`902d5b1`→`cc9bb3f`→`f566856`→`6527a6f`→`eb3e764`→`3a9618c`→`fae7f57`→`53dc83a`→`d81a6e9`→`6db341f`
- 复核报告（本目录）：`LOAD_REPRODUCTION_DIAGNOSIS_AND_MODIFICATION_20260906.md`、`P0_REVIEW_AND_RIGID_DIAGNOSTIC_6527A6F_20260906.md`、`P1_STAGE2_OPERATOR_AND_LEDGER_REVIEW_20260906.md`、`P1_REAL_B_REVIEW_AND_SYNCHRONOUS_WAKE_FIX_20260906.md`
- 研究线文档：`refine-logs/q16-v5m-gpu-load-contract-20260831/FINAL_PROPOSAL.md`（载荷合同 GO/NO-GO 表）、`refine-logs/roj-q16-v5m-repair-20260830/`（两轮 claim 审阅）
- 数据清单：`artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/DATA_MANIFEST.md`
- 复现：算子诊断 `research/load_diagnosis_20260906/check_acceleration_operator.py`；P1 三角度驱动 `/tmp/p1_driver.py` 模式（joint flag + 150 步诊断提取）见各提交信息
