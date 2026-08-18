# 三论文复现目标、V5H12 执行效果审计与后续计划

日期：2026-08-16

仓库：`/tmp/fluxv-v5-nextgen`

性质：现状接管审计 / 后续执行报告

论文观测边界：本次审计未运行正式 `N=32/64/128` 矩阵，未读取 GT/scorer，未产生新的论文精度结果。

## 1. 结论先行

当前最终要比较的“三篇论文”是：

1. **Yang et al. (2025)**：刚性矩形扑翼六个安装攻角的周期均值升力/推力；
2. **Izraelevitz, Zhu & Triantafyllou (2017)**：Figure 14 中的 Scherer 1968 有限翼水槽实验，比较周期平均推力系数；
3. **Baik et al. (2012)**：W1--W4 大幅俯仰--非简谐升沉平板实验，比较相位 `CL/CD`。

需要特别区分：

- Ramesh 2014 和 Hirato 2019 是当前 V5H 科学链的**源模型、方程和机制门**，不是本轮“三论文最终评分矩阵”中的额外论文；
- Izraelevitz 2017 Figure 11 是作者数值曲线对比，不是实验真值；当前三论文中的 Izraelevitz 实验门是 Figure 14 / Scherer 1968；
- V5H12 目前只在修复 **Baik W2 内部 formal execution/artifact 协议**，尚未运行三论文精度比较。

对当前 agent 的评价是：

- **技术方向正确**：根因定位和最小 execution-only 修复基本正确，没有改动 IR-WRK3、coupling、stream、W2 参数或论文评分规则；
- **执行层修复有效**：真实 7 字段记录可被解析，source-before-stage 和 conversion STOP 坐标均已有定向测试；
- **晋级结论过早**：`PROGRESS_H0_H6_20260816.md` 的“H0--H6 全 PASS，可进入 H7”不成立。诚实状态应为 H4 未通过预注册的同进程联合命令、H6 缺独立可追踪审计工件，H3 的 exactly-once 文字强于现有实现。

因此当前结论是：**V5H12 修复候选可继续完善，但 H7 dependency re-sign、真实 smoke 和 formal A 仍应保持 BLOCKED。**

## 2. 三篇目标论文与具体工况

### 2.1 Yang et al. (2025)

题目：H.-H. Yang, S.-G. Lee, E.-H. Lee, J.-H. Han, “Numerical simulation framework of bird-inspired ornithopter in forward flight,” *Journal of Fluids and Structures* 133 (2025), 104263。

DOI：`10.1016/j.jfluidstructs.2024.104263`。

本地来源审计：`docs/forward_flight_large_pitch/reproductions/plev2025/source_data/DIGITIZATION.md`。

冻结工况：

| 项目 | 数值/定义 |
|---|---|
| 几何 | 单个矩形刚性翼，弦长 `0.130 m`，翼展 `0.250 m`，厚度 `0.001 m` |
| 翼根偏置 | 机构铰点至气动翼根 `0.080 m` |
| 来流 | `U=5.5 m/s` |
| 扑动频率 | `f=2.5 Hz`，周期 `0.4 s` |
| 介质参数 | `rho=1.23 kg/m^3`，`nu=1.47e-5 m^2/s` |
| 雷诺数 | 约 `4.86e4` |
| 安装攻角 | `0, 5, 10, 15, 20, 25 deg`，共 6 个条件 |
| 运动 | 论文公开四连杆参数重建的 nominal four-bar；不是未公开的激光实测时序 |
| 论文观测 | Figure 11 Type A 的周期均值 lift/thrust，单位 `gf` |
| 主指标 | 6 条件 lift/drag MAE；统一使用正阻力 `D=-T` |

证据边界：Figure 11 没有公开相位载荷，因而本地相位曲线只能做模型诊断，不能计算实验相位误差。数字化读图不确定度约 `±0.4 gf`，不是实验误差条。

### 2.2 Izraelevitz, Zhu & Triantafyllou (2017), Figure 14 / Scherer 1968

题目：J. S. Izraelevitz, Q. Zhu, M. S. Triantafyllou, “State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span,” *AIAA Journal* 55(4), 2017。

DOI：`10.2514/1.J055144`。

本地来源审计：`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/DIGITIZATION_FIG14.md`。

冻结工况：

| 项目 | 数值/定义 |
|---|---|
| 实验来源 | 论文 Figure 14 中的 Scherer 1968 水槽数据 |
| 翼型/几何 | NACA 63A015，矩形翼，`c=0.1016 m`，`b=0.3048 m`，`AR=3` |
| 俯仰轴 | `0.75c` |
| 运动 | `z=h cos(omega t)`；`theta=theta_max cos(omega t+psi)` |
| 无量纲参数 | `h/c=0.6`，`St=0.2`，`k≈pi/6≈0.5236`，`J'=6` |
| 俯仰幅值 | `theta_max=15 deg` 和 `25 deg` |
| 15° 条件 | `psi=15,30,45,60,75,90,105 deg`，7 个唯一条件；15°、75°各有一个重复实验点 |
| 25° 条件 | `psi=45,60,75,90,105 deg`，5 个唯一条件 |
| 实验样本量 | 12 个唯一条件、14 个实验 marker |
| 论文观测 | 周期平均推力系数 `CT` 及图示误差棒 |
| 主指标 | 14 个观测的 `CT` MAE/RMSE，同时报告 12 条件结构 |

来源冲突必须保留：Izraelevitz Figure 14 使用 `Cd0=0.057`，Scherer 原报告为 `0.027`。主结果固定忠实使用 `0.057`，`0.027` 只作为来源敏感性，不能根据误差选择。

Figure 11 可继续作为 ULLT/UVLM 数值复现门，但不能冒充 Figure 14 的实验真值。

### 2.3 Baik et al. (2012)

题目：Y. S. Baik, L. P. Bernal, K. Granlund, M. V. Ol, “Unsteady force generation and vortex dynamics of pitching and plunging aerofoils,” *Journal of Fluid Mechanics* 709 (2012), 37--68。

DOI：`10.1017/jfm.2012.318`。

主要可复现真值来自 Baik 博士论文 Figures 5.24--5.27 的 corrected-total 载荷。

本地来源审计：`docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/SOURCE_AUDIT.md`。

共同工况：

- 弦长 `0.076 m`，跨度 `0.600 m`，厚度比 `6.25%`；
- 前后缘圆角半径 `0.002375 m`；
- 四分之一弦俯仰轴；
- `Re=5000`；
- 水槽宽约 `0.61 m`、底部约 `1 mm` 间隙并有自由面端板，属于准二维壁面约束实验；
- Ptera 当前仅能使用自由端零厚度中面替代，不能声称已复现真实壁面/端板和厚圆边截面。

W1--W4：

| Case | `k` | `h0/c` | 标称 `St` | 表中俯仰幅值 | 周期 |
|---|---:|---:|---:|---:|---:|
| W1 | 0.5 | 0.50 | 0.16 | 13.16° | 7.13 s |
| W2 | 1.0 | 0.50 | 0.32 | 33.73° | 3.56 s |
| W3 | 1.0 | 0.25 | 0.16 | 13.16° | 3.56 s |
| W4 | 0.5 | 1.00 | 0.32 | 33.73° | 7.13 s |

W3 的 `k=1.0`；早期 AIAA 表中的 0.5 是误印。升沉位移不是正弦，而是从
`h_dot/U=-tan(alpha_pl,max sin(2 pi t/T))` 周期积分得到。有效攻角在四个关键相位为 `8,22,8,-6 deg`。

论文真值与评分：

- 每个工况 401 点数字化曲线，去掉 phase=1 重复端点后是 400 个唯一相位点；
- 同时保留 raw 和论文一致的 1 Hz sharp Fourier low-pass；
- 主指标为每个 case 的 `CL/CD` RMSE/MAE，再作四工况等权宏平均；
- 不允许相位平移、幅值乘子或均值拟合；
- `CD<0` 在论文定义中表示推力。

### 2.4 当前 V5H formal 内部门不是完整三论文矩阵

当前 V5H11/V5H12 formal A/B 只使用 **Baik W2**，且目的首先是验证科学执行链和收敛工件：

- transport `N=32,64,128`；
- 每个 N 有 3 个 release layers；
- source steps `4,5,6` 对应 Ptera steps `3,4,5`；
- A/B 两个 fresh 进程要求前 9 个 semantic files 字节一致、provenance 不同；
- 在此阶段 GT/scorer 仍封存。

即使该门通过，也只证明 inner mechanics/execution/convergence，可进入后续 outer gate；它本身不等于 Yang、Figure 14、Baik W1--W4 的论文精度复现。

## 3. 现有程序地图

### 3.1 论文工况、数字化和传统基线

| 作用 | 主要程序/工件 | 当前用途 |
|---|---|---|
| Yang/Izraelevitz 工况 | `platform/forward_flight_benchmarks/cases.py` | 冻结几何、运动和无量纲参数 |
| Yang PLEV 核 | `platform/forward_flight_benchmarks/yang_plev.py` | 作者 Eqs. 8--14 的独立实现；不是完整作者求解器 |
| Yang 六工况 runner | `platform/forward_flight_benchmarks/run_yang2025_crosscase.py` | 六攻角周期均值比较 |
| Figure 14 runner | `platform/forward_flight_benchmarks/run_izraelevitz_scherer_experiment.py` | 12 条件/14 marker 的 `CT` 比较 |
| Baik 工况/运动 | `platform/forward_flight_benchmarks/baik2012.py` | W1--W4 几何、非简谐运动、过滤 |
| Baik 主 runner | `platform/forward_flight_benchmarks/run_baik2012_benchmark.py` | W1--W4 相位 `CL/CD` 比较 |
| Baik 敏感性 | `platform/forward_flight_benchmarks/run_baik2012_sensitivity.py` | W2 数值/尾迹单因素敏感性 |
| 三论文聚合 | `build_fluxv_v5_all_conditions.py`、`fluxv_v5_all_conditions.py`、`finalize_fluxv_v5_all_conditions.py` | 汇总 Yang/Fig14/Baik 工件和统一指标 |
| 图件 | `plot_fluxv_v5_all_*.py`、`plot_fluxv_v5_final_report.py` | 论文对比图和最终报告图 |

这些程序已存在，但其中相当一部分仍是 untracked 研究工件；它们不能仅凭文件存在就视为已发布、已冻结或已完成新 V5H 比较。

### 3.2 V5H 科学核心与耦合链

| 层 | 主要程序 | 当前状态 |
|---|---|---|
| 2-D source parity | `platform/ldvm_fourier.py`、`v5h_dvm_source.py`、`v5h_dvm_node_placement.py` | Ramesh/作者 Fortran parity、LEV/TEV 事件与出生位置；mechanism evidence |
| rVPM reference/core | `src/fluxvortex/rvpm_reference.py`、`rvpm_ir_wrk3.py` | Gaussian-erf 核、invariant-reconstructed WRK3；M0/M1 机制测试通过 |
| streaming execution | `rvpm_ir_wrk3_stream.py` | 大 N 不保留完整 stage 数组，提供 compact record/failure journal |
| FD/Ptera adapter | `rvpm_ir_wrk3_fd_adapter.py`、`fluxv_v5h4_ptera_rvpm_transport.py` | same-stage parent field、中心差分、调用账本 |
| row ownership | `fluxv_v5h10_row_owner.py` | source/transport/advance 能力和事件链 |
| W2 coupling | `fluxv_v5h11_baik_coupling.py` | Ptera load owner + direct/FD/rVPM transport + support attestation |
| V5H12 executor | `fluxv_v5h12_baik_w2_executor.py` | 将 coupling/stream 结果转换为 artifact rows；本轮修复重点 |
| V5H12 runner | `run_fluxv_v5h12_baik_w2.py` | dependency preflight、12-file PASS/STOP bundle、A/B 语义校验 |

### 3.3 历史分支应如何使用

- V5H7/V5H8：manufactured oracle，验证时序拓扑/增量涡片；不能当论文结果；
- V5H9：live boundary/owner 事务机制；
- V5H10：首次进入真实 W2 三层耦合，但 additive-`sigma` 路线数值 STOP；
- V5H11：改为 invariant-reconstructed WRK3；真实 disposable `N32/layer1` smoke 通过，但 formal A 在 artifact ABI 转换首条记录处 STOP；
- V5H12：只修 V5H11 的 parser/source-order/STOP-coordinate 协议；尚无真实 V5H12 smoke 或 formal A/B。

## 4. 当前 agent 的方向是否正确

### 4.1 正确的部分

1. **分支边界正确**：另建 V5H12 runner/executor/tests，冻结 V5H11 历史；
2. **归因正确**：V5H11 formal-A 是 artifact integration failure，不是已经证明数值或模型失败；
3. **7+1 所有权正确**：observer payload exact 7 字段，normalized invariant 来自已 hash 的 stream compact record；
4. **source-before-stage 正确**：stage durable row 必须先有 source parent；
5. **STOP 语义改进正确**：conversion 第 1/第 2 条失败分别保留 `(32,1,4,3,1,1)` / `(32,1,4,3,1,2)`，`stage_began=false`；
6. **科学范围克制**：没有改 RK、Ptera、coupling、W2、阈值或论文目标，也没有拿 smoke 冒充论文结果。

### 4.2 报告与实际的出入

| 节点 | agent 报告 | 独立重放/代码审计 | 诚实状态 |
|---|---|---|---|
| H0 | PASS | 冻结 8 leaves、formal-A 工件和四个 V5H12 最终 SHA 匹配 | PASS |
| H1 | RED-before-GREEN PASS | 三个旧缺陷可重放；但没有原始 UTC/stdout 快照证明 RED 的时间顺序 | PROVISIONAL |
| H2 | parser PASS | actual coupling N=1 exact-7 记录、7+1、负矩阵均通过 | PASS |
| H3 | exactly-once PASS | 公共成功路径为 source→96 stages→commit；但 runner `commit_completed_layer` 仍保留“source 未预提交时隐式 append”的 fallback，且 pytest 中缺少以该合同命名的 identical/inconsistent/commit-without-source 定向闭环 | WARN / contract incomplete |
| H4 | 联合回归 PASS | 六个 suite 分进程分别为 `33/44/20/42/11/25` PASS；但 HANDOFF §9.3 的同进程 V5H12 executor→runner 实际为 **4 failed, 73 passed**，不是报告所称“与 V5H11 相同的 3 个继承失败”。第 4 个是 Ptera 模块残留导致 formal-entry isolation test 失败；V5H11 同进程为 3 failed/59 passed | FAIL against prereg command |
| H5 | static PASS | `py_compile`、Black、Ruff、diff-check 均通过 | PASS |
| H6 | fresh hostile review PASS | 没有 reviewer ID、request/response、trace、独立 audit 文件或 token；现有 durable 文件只有自述 | UNVERIFIED |
| H7 | 待授权 | H4/H6 尚未闭合，且 HANDOFF 中 CHECKLIST/TRACKER 哈希已陈旧 | BLOCKED |

当前四个实现/测试 SHA 与 `PROGRESS` 报告匹配：

- runner：`063725d1512df574b969f302b12199e4ff82c0ac52c6be15b34043cf640d282b`；
- executor：`5c74a9ffe245a0212aacf06067c477c6ddcc384e1c71b97a2aa1bd017bfb7053`；
- runner test：`8612d52ba330fbe9f287fdd57190aa3dcfee6d1d436e96a351c469269395d4a9`；
- executor test：`b155af9a224dfa7f2b7f32f5a6903ed0dbecdf7b9245a82d6df4b1dce2f9fbd6`。

但治理文件已继续变化，HANDOFF §5.1 的旧哈希不再代表当前内容：

- 当前 `CHECKLIST.md`：`553d205f4b88fa692a47a96bc71b51bcc92157896849d1eeb070e6861b6566e4`；
- 当前 `EXPERIMENT_TRACKER.md`：`c3d14da7af3bf754f896386d3243f592db3b939d51dba79c93f29b6ed0b6645b`。

### 4.3 当前任务实际取得了什么

已经取得：

- V5H11 formal-A 的三项 execution ABI 根因被准确分离；
- V5H12 候选代码可以消费真实 7 字段 coupling compact record；
- normalized invariant 的所有权和 hash 边界没有被改写；
- source-before-stage 正常路径和 conversion STOP 精确坐标有定向测试；
- 冻结 V5H11 科学文件没有发生字节漂移；
- 分进程 focused suites 和静态检查均为绿色。

尚未取得：

- 没有新的 dependency manifest/token；
- 没有 V5H12 真实 disposable N32/layer1 smoke；
- 没有 V5H12 formal A/B；
- 没有 outer moving-parent/source-release convergence；
- 没有 Yang/Figure14/Baik 新 V5H 精度结果；
- 没有论文 GT/scorer 解封，也没有论文复现成功结论。

因此当前效果应称为：**execution-repair candidate ready for evidence closure**，而不是“V5H12 已完成”或“三论文复现已进入正式运行”。

## 5. 后续计划与效果检验节点

### P0：先修正治理状态，不运行科学矩阵

必须完成：

1. 将 `PROGRESS_H0_H6_20260816.md` 的总状态改为 H4 FAIL、H6 UNVERIFIED、H7 BLOCKED；
2. 将 tracker 的 G1 complete/frontier 更新回证据真实状态；
3. 更新 HANDOFF 中 CHECKLIST/TRACKER 当前哈希，保留初始治理哈希时应明确标为 historical；
4. 为 H1 标注“缺原始时间顺序工件”，不要补造 RED 历史。

通过标准：报告、tracker、当前文件哈希和可执行命令互相一致。

### P1：闭合 H3 source exactly-once

优先方案：删除或禁止 `commit_completed_layer` 中“source 未预提交则隐式 append”的 fallback，使 completed-layer transaction 只验证已存在 source canonical bytes。

必须增加的定向测试：

1. source 未预提交时 commit 拒绝，sink 不变；
2. identical duplicate source append 拒绝；
3. inconsistent duplicate source append 拒绝；
4. 正常 source→96 stages→commit 后 `source_count=1`；
5. source append 失败时 stage/layer 均为 0，clean retry 成功。

若决定保留 fallback，必须先做治理 amendment，把“commit only verifies”降级为更窄的公共执行路径声明；不能让代码和报告各自正确。

### P2：闭合 H4 同进程联合回归

不得放宽 runner 的 runtime-inventory 门。应在测试生命周期中隔离/恢复真实 coupling test 写入的 `sys.modules` 状态，或把真实生命周期测试放进显式子进程，同时保证 HANDOFF §9.3 的原始联合命令最终全绿。

通过标准：

- V5H12 同进程 executor+runner 无 4 个失败；
- 冻结 V5H11 控制仍按预期；
- formal-entry 测试仍证明 dependency preflight 之前不加载 Ptera；
- 不能通过忽略 `pterasoftware` 或扩大 manifest 来“修测试”。

### P3：补真正的 H6 fresh hostile audit

生成独立、只读、可追踪的审计工件，例如：

- `H6_FRESH_HOSTILE_AUDIT_20260816.md`；
- 记录 reviewer/agent ID、输入四 SHA、命令、攻击矩阵、stdout 摘要、verdict 和审计文件 SHA；
- 攻击至少覆盖 legacy 第 8 字段、normalized tamper、source 重复/缺失、conversion 错坐标、generic source-only 语义、registry/dependency drift。

通过标准：无 must-fix；审计输入 SHA 与重签前最终四叶一致。

### P4：H7 dependency closure

只有 P0--P3 全部通过后：

1. 冻结最终 4 个 V5H12 叶 SHA；
2. 生成无自哈希环的 dependency manifest 与 external audit token；
3. 逐叶运行时重哈希；
4. 证明旧 V5H11 token 对 V5H12 新叶 fail closed；
5. 验证 runtime module inventory、路径 canonical、无 back-edge。

失败动作：发布 `dependencies_unbound` STOP；不构建 Ptera case。

### P5：V5H12 disposable real smoke

fresh process，仅 `N32/layer1`，无 formal sink、无 GT/scorer。旧 V5H11 smoke 不能代替，因为 runner/executor leaves 已变化。

检查：

- source/prehistory/owner/parent hashes；
- 96 stages、192 direct、192 center、576 offset，以及 FD raw call count；
- invariant/stability/sigma/near-zero gates；
- support/frontier counts；
- load/no-penetration/Kelvin；
- parent before==after；
- summary canonical SHA 与 observed runtime inventory。

### P6：formal A → B

1. fresh formal A：`N=32,64,128 × 3 layers`；
2. A 不完整或任一门失败：保存 12-file exact-prefix STOP，禁止 B；
3. A 完整 PASS 后运行 fresh B；
4. A/B 前 9 个 semantic files 字节完全相同，UUID/UTC/path/replicate 不同；
5. fresh artifact audit 复算 schema、hash DAG、stage gates、state/tracer/probe/load/convergence。

### P7：outer science gate

formal A/B 只完成 inner gate。随后仍需 inherited moving-parent/source-release outer convergence（B4）：

- fresh `P=32,64,128`；
- time-freeze/joint convergence；
- FD-J sensitivity；
- source release/owner continuity；
- 任一失败均 STOP，不调阈值、不读取论文目标。

### P8：三论文正式校验

只有 P7 和 unlock token 通过后，才进入：

1. Yang 2025：6 个攻角周期均值 lift/drag；
2. Izraelevitz Figure 14：12 个唯一条件、14 个实验 `CT` marker；
3. Baik 2012：W1--W4，每工况 400 个唯一相位的 raw 与 1 Hz filtered `CL/CD`；
4. 使用冻结 metric contract，不做相位/幅值/偏置拟合；
5. 与 frozen v4b、作者模型/论文参考作同口径比较；
6. 完成独立 result-to-claim 审计后再决定能否写论文精度结论。

## 6. 里程碑判断

| 层级 | 当前状态 | 距离完成 |
|---|---|---|
| 三篇论文的来源、工况和历史数字化 | 已有，可复用但有边界声明 | 近完成 |
| V5H source/rVPM/coupling 机制 | 大部分机械门已过，已有真实 V5H11 layer1 smoke | 先进但仍需 formal/outer |
| V5H12 execution repair | 功能候选完成，证据闭环未完成 | 还差 P0--P4 |
| Baik W2 inner formal convergence | 未运行 V5H12 A/B | 还差 P5--P6 |
| moving-parent/source-release outer convergence | 未运行 | 还差 P7 |
| 三论文新 V5H 精度校验 | 未解封、未运行 | 还差 P8 |

不宜用单一百分比描述，因为“代码完成度”和“论文证据完成度”差异很大。若按门禁计，当前还剩至少 **5 个执行/科学门层级**：证据闭环、dependency/smoke、formal A/B、outer convergence、三论文评分与 claim audit。

## 7. 建议给下一个 agent 的一句话任务

> 不运行 Ptera 正式矩阵、不读取 GT/scorer；先把 V5H12 的 H3/H4/H6 证据闭环做实，修正 progress/tracker 状态并生成可追踪 hostile-audit 工件。只有原预注册联合命令全绿、最终四叶重冻且新 token verified 后，才运行 fresh disposable N32/layer1 smoke。

## 8. 主要证据入口

- 当前自述报告：`refine-logs/v5h12_execution_repair/PROGRESS_H0_H6_20260816.md`
- 治理计划：`refine-logs/v5h12_execution_repair/PLAN.md`
- 门禁 tracker：`refine-logs/v5h12_execution_repair/EXPERIMENT_TRACKER.md`
- 接手说明：`refine-logs/v5h12_execution_repair/HANDOFF.md`
- V5H11 formal-A STOP：`/tmp/fluxv-v5h11-b3-formal-A-20260816`
- 三论文历史总报告：`docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/FINAL_REPORT_ZH.md`
- Yang 数字化：`docs/forward_flight_large_pitch/reproductions/plev2025/source_data/DIGITIZATION.md`
- Figure 14 数字化：`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/DIGITIZATION_FIG14.md`
- Baik 来源审计：`docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/SOURCE_AUDIT.md`
