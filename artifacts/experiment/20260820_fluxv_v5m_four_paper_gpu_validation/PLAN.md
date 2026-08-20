# FLUX-V5M 四论文精度与 GPU 验证计划

## 1. Objective

- run id: `20260820_fluxv_v5m_four_paper_gpu_validation`
- selected idea: 独立重放提交 `fa8eaca` 的四项审计修复，并用同一冻结实现 fresh 复算 Baik 2012、Yang 2025、Izraelevitz 2017、Mancini 2017。GPU 结论必须由实际执行设备、CUDA 核路径和运行时监测共同支持。
- 用户核心要求：验证 P0/P1 修复效果；验证 FLUX-V5M 对四篇论文的精度并与 V4B/实验比较；主数值计算必须使用 CUDA GPU，不允许 CPU 数值后端或 CPU fallback。
- 非协商约束：不改变论文 GT、评分公式、工况或冻结参数；不借用旧 NPZ 充当 fresh 结果；不把“检测到 GPU”等同于“算法在 GPU 上执行”；CPU 只允许 Python 调度、文件 I/O、CUDA launch 和结果序列化，诱导速度、影响矩阵、线性求解、载荷/修正及时间推进必须由 CUDA 执行。
- research question: `fa8eaca` 是否闭合四项审计缺陷，四论文指标是否可 fresh 复现，实际计算中哪些主核在 GPU 上并行？
- null hypothesis: 至少一项修复、论文指标或 GPU 执行合同不能由 fresh 证据支持。
- alternative hypothesis: 四项修复全部通过，四论文指标达到冻结目标，且四篇论文的全部主数值核均有 CUDA 执行证据、无 CPU 数值 fallback。

## 2. Baseline And Comparability

- baseline id: `fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169` 的父提交 `3d4c45e...`；论文对比基线为 handoff 冻结 V4B 指标。
- baseline variant: FLUX-V5M frozen constants；不调 `lesp_crit`。
- dataset / split:
  - Baik W1–W4 digitized experimental samples；
  - Yang 2025 frozen mean-force digitization；
  - Izraelevitz 2017 Fig.14 frozen markers；
  - Mancini 2017 fast/slow pitch digitization。
- primary metric:
  - fixes: ledger closure、device selection、G0/G0b/G0c exit semantics；
  - Baik: macro CL/CD RMSE；
  - Yang: lift/drag MAE (gf)；
  - Izra: CT MAE；
  - Mancini: fast/slow lift RMSE。
- required metric keys:
  - `ledger_total_closure_abs`
  - `g0_cl`, `g0b_parity_abs`, `g0c_finite`, `g0c_reduced`, `g0c_active_strips`
  - `baik_cl_macro_rmse`, `baik_cd_macro_rmse`
  - `yang_lift_mae_gf`, `yang_drag_mae_gf`
  - `izra_ct_mae`
  - `mancini_fast_rmse`, `mancini_slow_rmse`
  - per-run `gpu_device`, `cuda_kernel_path`, `gpu_utilization_observed`, `gpu_memory_peak_mib`
- comparability risks: runner hardcoded旧输出路径、跨工况 cache、CPU/GPU浮点差异、2D LDVM 本身可能无 GPU 后端、历史输出污染、Mancini长运行。

## 3. Code Translation Plan

本轮先冻结 GPU-only 边界，再建立独立 CUDA 执行分支。不得为追求论文指标修改生产参数；GPU 后端必须与冻结 CPU 参考在小型机械算例上做数值等价回归，但 CPU 参考不得用于正式论文计分。

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/*` | frozen production baseline | 在独立分支新增严格 CUDA 后端；不改冻结论文参数 | 现有四论文入口的主求解仍为 CPU | 高 |
| 本 run 目录 | evidence | 新增计划、清单、日志、manifest、metrics、summary | 隔离 fresh 证据 | 低 |
| audit harness（如需） | read-only wrapper | 仅重定向输出/收集设备与指标 | 避免覆盖历史 `/tmp` 文件 | 中 |

## 4. Execution Design

- minimal experiment:
  1. commit/SHA/env/GPU preflight；
  2. `test_ledger_contract.py`；
  3. GPU-only 模式下 CUDA 正常、`cpu`/非法设备/无 CUDA 全部 fail-fast；
  4. G0/G0b/G0c；
  5. Baik W1 @0.19 回归与 CUDA 路径分类。
- smoke / pilot:
  - 运行修复门；
  - 对每篇论文先确认入口、输出和单工况；
  - 监测 `nvidia-smi` 与 Warp device。
- full run:
  - Baik W1–W4；
  - Yang 全冻结 AoA；
  - Izra Fig.14 全工况；
  - Mancini fast/slow；
  - 每篇独立进程、独立输出、独立 GPU 监测。
- expected outputs: `run_manifest.json`, `metrics.json`, `metrics.md`, `summary.md`, `claim_validation.md`, `runlog.summary.md`, raw logs, GPU snapshots, fresh result files。
- stop condition: 非有限结果、评分/GT漂移、历史文件覆盖风险、GPU OOM、任何主数值步骤落到 CPU、预计单项超过 45 分钟且没有进度证据。
- abandonment condition: 当前实现没有某篇论文的可重跑入口或缺失冻结 GT；该篇标为 inconclusive，不得引用历史成绩替代。
- strongest alternative hypothesis: 现有 CPU 指标可复现，但 GPU 只覆盖三维粒子诱导速度，2D LDVM/Ptera矩阵求解仍在 CPU；此类结果一律为诊断记录，GPU-only 论文验证为 FAIL/BLOCKED，不能给出精度 PASS。

## 5. Runtime Strategy

- smoke commands: 逐一运行 `test_ledger_contract.py`、G0/G0b/G0c 和设备负控。
- main commands: 经入口审计后冻结到 `run_manifest.json`。
- budget: 门测试 <10 分钟；四论文总预算 90 分钟，单 runner 硬上限 45 分钟。
- output: `artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/`
- safe efficiency levers: 独立进程、复用 CUDA 编译缓存但不复用工况状态、GPU监测1秒采样；不得改变网格/步数；不得用 CPU fallback 获得结果。
- tooling downgrade: 本会话无 `bash_exec`/artifact 接口，使用普通终端和本 run 目录保留等价证据；因此结果等级为开发验证而非正式 artifact registry 记录。

Monitoring:

- 短门直接等待；长跑 60/120/300/600 秒检查。
- 继续信号：有进度、GPU/CPU利用率与预期路径一致、输出有限。
- kill/relaunch：无输出且资源无活动、非有限、OOM、工况状态污染。

## 6. Fallbacks And Recovery

- runner 硬编码 `/tmp`：使用审计 wrapper 或隔离临时目录，不改算法。
- GPU 不可用：立即 STOP；不启动论文算例。
- 2D LDVM/Ptera 无 GPU：先实现并验证等价 CUDA 后端；完成前四论文门保持 BLOCKED。
- 长算例超预算：保留 pilot，完整项标未完成，不引用旧结果。

## 7. Checklist

- `CHECKLIST.md`
- next: 冻结文件/入口/环境并执行修复门。

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-20 19:10 CST | 建立持久 run worktree 与计划 | 原 `/tmp` worktree 被清理 | 无算法影响；证据路径更可靠 |
| 2026-08-20 19:17 CST | GPU 合同降级为“active-LEV 实证 + 四论文逐项分类” | 四个冻结论文入口均关闭 LEV，LDVM/polar 在 CPU；强开 LEV 会改变方法 | 精度仍可比较；“四论文全流程 GPU”不能在不改算法时成立 |
| 2026-08-20 19:42 CST | 用户明确收紧为 GPU-only 硬门 | 不允许 CPU 计算 | 撤销上一条降级；既有 CPU Baik/Yang 仅作诊断，Izra CPU 运行中止；先开发真实 CUDA 后端再正式复现 |
| 2026-08-20 20:20 CST | 严格 CUDA Ptera、LDVM、polar/ledger 后端及负控完成 | 四论文不得使用 CPU 数值 fallback | 11 项 GPU/P0 回归通过；开始冻结重跑 |
| 2026-08-20 20:55 CST | 冻结源码哈希后完成 Baik 与三篇三维正式重跑 | 闭合源码—结果 provenance | Baik/Yang/Izra 优于 V4B；Mancini 略逊，结论为 PARTIAL；详见 `REPORT_GPU_ONLY_20260820.md` |
| 2026-08-20 21:30 CST | fresh audit 判定首轮 FAIL，撤回 GPU-only 完成声明 | 发现 NumPy 尾迹推进、host LEV 分支和产物合同缺口 | 首轮数值只保留 provisional；启动 GPU-only v2 修复与全量重跑 |
| 2026-08-20 22:10 CST | GPU-only V2 全矩阵与第二次 fresh audit 完成 | 尾迹 age/坐标、LEV、finalize、修正、指标与产物合同闭合 | PASS_WITH_WARNINGS；限定于 24 个冻结工况，Mancini 保持 PARTIAL |
