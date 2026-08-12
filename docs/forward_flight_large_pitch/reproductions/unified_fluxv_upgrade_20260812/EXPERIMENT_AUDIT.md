# Experiment Audit Report

**Date**: 2026-08-12  
**Auditor**: GPT-5.6-Sol ultra（fresh same-family agent，read-only，provisional）  
**Project**: Izraelevitz 2017 / Yang 2025 unified exploratory FluxV upgrade

## Overall Verdict: WARN

## Integrity Status: warn — qualified claims only

没有发现伪造真值、自归一化、幽灵结果或结果数字不匹配。所有汇总指标均由独立审计器从最终 CSV 重算，最大差为 `0.0`；最终定向回归测试为 `38 passed`。警告来自实验设计和结论范围，而不是数值完整性失败：两个任务都已经参与方案设计，没有独立留出盲验；最终模型是周期双遍而非在线因果求解器；Yang 周期均值由 v1 固定拥有，因而不能用该指标验证 v2 新增 ULLT 相位机制。

### 2026-08-12 Figure-14 / v3 closeout addendum

本报告主体冻结的是 v2 审计。随后加入的 Izraelevitz Figure 14 / Scherer
真实实验门禁表明：冻结 v1/v2 的 `CT` RMSE 为 `0.22260`，显著差于旧
FluxV 的 `0.05115`，所以原“统一改进已泛化”结论被否决。看过该失败后
开发的 post-hoc v3，在正确的 `3/4c`、`Cd0=0.057` 分账下得到：Yang
升力/阻力 MAE `3.5791/3.8911 gf`；Figure 11 的 256 步 full 升力/阻力
RMSE `0.15462/0.31420`；Figure 14 实验 `CT` RMSE `0.04719`。这些准确性
门通过，但 smoke→full 的 Figure-14 最大变化 `0.00555 CT` 略高于预记
阈值 `0.005`，且 v3 是在看到实验失败后选择的周期所有权启发式。因此
最终完整性等级仍为 **WARN / qualified claims only**，不能视为独立盲验、
LEV 闭合或生产模型。

## Checks

### A. Ground Truth Provenance: PASS

- Yang `Test` 来自论文 Figure 11 的风洞周期均值数字化；`±0.4 gf` 是读图不确定度，不是实验误差条。
- Yang `Authors' Proposed` 是作者 modified-UVLM 数值结果，不是真实观测，也不是本地 PLEV 输出。
- Izraelevitz Figure 11 的作者 UVLM、ULLT 和 QS+added-mass 都是论文数值曲线，不是实验真值。
- 本地实现不读取风洞力残差，不含按论文或工况编号建立的修正表。

证据：`plev2025/source_data/DIGITIZATION.md:5-17`、`source_data/DIGITIZATION.md:5-64`、`uvlm_polar_correction.py:35-78`、`augmented_uvpm.py:36-55`。

### B. Score Normalization: PASS

- MAE、RMSE、bias 和最大绝对误差直接由原始残差计算。
- range-NRMSE 的分母是参考曲线范围，不是预测曲线自身统计量。
- 正峰和负峰的相位误差分别计算，没有做最优循环平移或幅值拟合。
- v1 的 22 行和 v2 的 30 行指标独立重算后与 `accuracy_metrics.csv` 完全一致。

证据：`run_unified_fluxv_upgrade.py:99-139`、`PLAN.md:18-20`、最终 `accuracy_metrics.csv`。

### C. Result File Existence And Consistency: WARN

通过项：

- 最终 v2 manifest 状态为 `complete`；所列环境、核心源码、base artifacts 和结果 artifacts 的 SHA-256 全部匹配。
- 汇总 JSON、均值 CSV、相位 CSV、精度 CSV、四张 PNG 和四张 PDF 均存在。
- 图件直接读取最终运行 CSV，未手录结果。
- 周期端点错误已经修复并重跑；Yang 15° ULLT/v2 的 lift closure jump 从约 `0.3002 N` 降为 `0.00102757 N`。

警告项：

- manifest 尚不是完整传递闭包，未哈希所有运动构造、绘图、测试和说明文件；本审计 JSON 额外冻结了关键输入和最终结果哈希。
- 尚未完成一次只改变一个因素的空间/时间/周期收敛矩阵。

证据：最终 `run_manifest.json:2-35,74,120-135`、`ullt_attached.py:216-253`、`test_ullt_attached.py:41-80`。

### D. Dead Code / Metric Identity: WARN

- UVLM force ledger 的总载荷、非定常项和环流项恒等式有测试覆盖；关闭单向 VPM 粒子的快速路径与原载荷逐点一致。
- 重复 phase-zero endpoint 和追加未来周期的两个回归测试有效，301→400 和 1024→1280 样本的选定周期结果最大差为 `0.0`。
- 但 v2 明确重新加回 v1 周期均值，因此 Yang 六攻角的 v1/v2 均值仅有浮点误差；Yang 均值指标对 v2 ULLT 机制是结构性失活指标。
- 旧 FluxV 和 Ptera prescribed UVLM 是同一载荷通道，不能作为两个独立模型的一致性证据。
- 本地一状态 ULLT 是独立可执行数值重建，不是独立实验验证。

证据：`augmented_uvpm.py:115-170,336-414`、`run_unified_ullt_extension.py:245-266,391-419`、`SUMMARY_REPORT_ZH.md:102-107,183-197`。

### E. Scope Assessment: WARN

- 15°–20°门限是在 v0 联合诊断失败后引入，属于 post-hoc exploratory design；不是预注册或确认性实验。
- 评估仅覆盖两个已查看任务：Yang 六个安装角周期均值，以及 Izraelevitz Figure 11 一个周期工况。
- Yang 没有公开实验相位历史，不能验证 v2 的相位改善。
- 当前算法为周期双遍，不能声称任意瞬态、在线因果或生产就绪。
- 没有独立新论文/新工况留出集，因此不能声称跨任务泛化、统计显著性或一般适用性。

证据：`PLAN.md:69-84`、最终 `run_manifest.json:12-17`、`SUMMARY_REPORT_ZH.md:15-20,179-207`。

### F. Evaluation Type: PASS

- Yang 周期均值：`real_gt_digitized`（风洞均值数字化）。
- Yang 相位历史：`simulation_only_no_gt_diagnostic`。
- Izraelevitz Figure 11：`simulation_only_digitized_numerical_reference`。
- 不属于 `synthetic_proxy`，也不是 `self_supervised_proxy`。

## Action Items

1. 冻结当前参数，在未参与设计的新前飞、大俯仰/扭转工况上做盲验。
2. 将周期双遍 ULLT/UVLM 所有权切换改造成因果在线状态递推。
3. 对网格、时间步、运行周期和尾迹保留长度逐项做单因素收敛。
4. 补齐运动构造、辅助脚本、绘图、测试和全部最终图件的传递哈希。
5. 在获得 Yang 相位真值之前，只把相位图当模型形状诊断。

## Claim Impact

- C1：同一探索性代码路径在两个已查看冻结任务上降低了指定点估计误差 — **supported with qualifier**。
- C2：Yang 周期均值比旧 FluxV 更接近数字化风洞参考 — **supported**，归因于 v1 UVLM/非线性极曲线均值通道。
- C3：v2 ULLT 提高了 Yang 准确率 — **unsupported**，均值指标对它失活且相位无真值。
- C4：Izraelevitz Figure 11 相对作者 UVLM 数值参考的相位 RMSE 降低 — **supported with qualifier**。
- C5：修改版已是通用、因果、生产 FLUXV — **unsupported**。

审计留痕：`.aris/traces/experiment-audit/2026-08-12_run01/`。
