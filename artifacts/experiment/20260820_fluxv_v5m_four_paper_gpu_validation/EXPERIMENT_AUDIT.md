# Experiment Audit Report

**Date**: 2026-08-20  
**Auditor**: GPT-5.6-Sol ultra（fresh same-family agent，read-only，provisional）  
**Project**: FLUX-V5M four-paper GPU validation

## Overall Verdict: FAIL

## Integrity Status: fail

## Checks

### A. Ground Truth Provenance: WARN

四篇均使用论文数字化实验数据而非模型输出生成 GT。Yang 只读实验升/阻力列；Izraelevitz 只读 `experimental_observation`；Mancini 验证冻结 CSV SHA；Baik 只读旧 scored CSV 的 `experiment` 列。

Baik 的问题是 GT 容器同时含历史模型列，且同一 400 个 unique GT 点因模型行被均匀重复 7 次（CL）/6 次（CD）。重复不改变当前 RMSE，但违反“历史输出完全排除”的宽泛措辞。当前 summary 也未绑定所有 GT 文件哈希。

### B. Score Normalization: PASS

Baik 使用原始 RMSE；Yang、Izraelevitz 使用原始 MAE；Mancini 使用 501 点 RMSE。没有使用预测最大值、均值或其他自归一化分母。力到系数只使用物理 `qS`。

### C. Result Existence And Hashes: FAIL

报告中的全部数值和百分比可由结果文件复算；两个 summary 文件哈希、嵌入源码哈希、三篇 content SHA 和 Baik 四个 payload SHA 均匹配。

失败项：

- `PLAN.md` 预先要求的 13 个扁平 metric keys 和 4 个 per-run GPU fields 没有完整落盘；
- 三篇 summary 没有原始预测曲线，Mancini RMSE 不能只靠产物独立重算；
- Baik payload SHA 未覆盖 phase，GT/依赖哈希不完整；
- worktree 尚未形成提交，部分关键文件为 untracked。

### D. CUDA Coverage And Dead Code: FAIL

真实 CUDA 执行成立：正式 runner 调用 CUDA subclass；AIC、wake influence、solve、loads 都有 CUDA 实现；24 工况记录每类 11,022 次计数；Nsight smoke 捕获 16,382 次 kernel launch、CUTLASS GEMM、GPU `getrf/trsm`。

但广义“CPU 只做调度/I/O/序列化”不成立：

- Baik macro mean 使用 NumPy；
- finite-wing gain/polar slope 与 strip area 有主机算术；
- Ptera prescribed-wake 坐标推进使用 NumPy，后续会改变诱导速度和载荷；
- Mancini 运动预处理、若干物理标量仍在主机；
- LDVM 用 GPU scalar `.item()` 形成 host bool 决定 LEV shedding。

因此只能称已验证热核为 CUDA，不能称四篇全部数值与指标 GPU-only。

### E. Scope: WARN

实际范围为 24 个确定性工况：Baik 4、Yang 6、Izraelevitz 12、Mancini 2；每工况一次，无 seed/repeat/不确定度或收敛研究。不能扩展为 robustness/generalization 结论。

### F. Evaluation Type: PASS

- 四篇论文评分：`real_gt`；
- CUDA 等价回归：`simulation_only` / frozen numerical oracle；
- ledger 单元测试：`synthetic_proxy`。

## Action Items

1. 将 Ptera wake coordinate advancement 和剩余会改变预测的主机算术迁移到 CUDA，或明确降级为 hybrid，不得继续称完整 GPU-only。
2. 用独立纯 GT 文件或去重/绑定的 Baik GT，记录四个 GT SHA。
3. 落盘计划要求的完整 metric contract、GPU fields、原始三篇预测/GT 曲线和容器哈希。
4. 修复后重新冻结、重跑并进行新的 fresh audit；旧 provisional 结果不得重标。

## Claim Impact

- “Baik/Yang/Izraelevitz 指标优于 V4B，Mancini PARTIAL”：**有条件支持**，数值算术成立。
- “四篇全部数值与指标 GPU-only”：**不支持**。
- “CUDA AIC/wake influence/solve/load/LDVM 热核实际执行”：**支持**。

