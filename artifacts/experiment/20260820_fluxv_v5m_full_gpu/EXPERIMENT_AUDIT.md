# FLUX-V5M GPU 并行优化实验审计

**日期**：2026-08-21
**审计者**：GPT-5.6-Sol ultra（fresh same-family agent，只读复放）
**独立性**：same-family
**接收状态**：provisional

## 总体结论：PASS with WARNINGS

本结论支持：在当前授权的单机、单翼、无 image Ptera 模式和标准 Warp-FSI 数据面内，科学数值计算使用 CUDA float64；三篇 3D 论文完整矩阵相对冻结基线实现约 **1.300×** 的单卡端到端加速。

本结论不支持：整个 Python 进程零 CPU、任意 Ptera 拓扑均已 GPU 化、统计置信区间或鲁棒性/泛化声明。

## A–F

### A. Ground-truth provenance：WARN

Yang、Izraelevitz、Mancini 的数字化实验 GT 哈希均与结果记录匹配。Baik 当前评分文件 `scored_phase_samples.csv` 的 SHA-256 为 `66fbff1b6fb922440096dacd60e207f5b6220ef3cebb710dc360d57a60740ff2`；它和纯 GT `baik2012_w1_w4_corrected_total_cl_cd.csv`（`4de6b01cd8072959e5b780053f311efa92ab5a94f17940dd122df340ad638f2f`）的公共 3,200 点逐值一致。

Baik scored 容器含 17,600 个重复行，并缺少四工况 CL/CD 的 `phase=1` 周期重复端点。当前 RMSE 不受影响，但后续应直接绑定纯 GT 文件。

### B. Score normalization：PASS

四篇均使用原始 RMSE/MAE 和物理量纲分母，没有用预测结果自身最大值、均值或方差归一化。

### C. Results and hashes：PASS

- Baik summary：`f2fba13edd70f089f8e3d7da027e7d98c3ba178f228db06720919b9a94104592`
- 三篇 summary：`0cf3c056c3a591e1ec8570a24a21f3d9ec832e66d5b838537ed1a34442a2a797`
- CUDA metrics：`06ac52ebdb0dc38ee90d4b85664304bf5e9004d7eb54ba366b48c0aa2022a068`
- 三篇 canonical result hash：`b49b1d23...43f2`，独立复算匹配。

四个 Baik payload hash、七个论文指标、runner/backend/correction/LDVM/monitor 与三份 GT source map 均重算匹配。指标使用 CUDA float64 独立复算。

### D. GPU execution and dead-code coverage：PASS

- fresh focused suite：40 passed；
- Warp-FSI：12/12 PASS，正常 STRUCT_CG 为 145 iterations；
- 21 个目标文件：Black、Ruff、py_compile、`git diff --check` 全部 PASS；
- LDVM CPU、NumPy、float32、混合设备输入全部 fail-close；
- `FLUXV_DTYPE=float32` 的 V5M-FSI facade 非零退出；
- PCG `max_iter=1, tol=1e-30` 抛出 `RuntimeError`，不返回未收敛伪解。

### E. Scope：WARN

范围是 24 个确定性论文配置：Baik 4、Yang 6、Izraelevitz 12、Mancini 2。每个配置仅运行一次，没有 repeat/seed、不确定度或收敛区间。

### F. Evaluation classification：PASS

- 四篇论文：`real_gt`；
- CUDA/CPU 等价门：`simulation_only`；
- 单元 fixture：`synthetic_proxy`。

## 性能证据

| 路径 | 基线 | 最终 | 审计结论 |
|---|---:|---:|---|
| Ptera 38-step kernels | 35,575 | 21,725 | PASS |
| Ptera kernel time | 57.158837 ms | 35.968836 ms | PASS |
| Warp-FSI kernels | 78,758 | 40,475 | PASS |
| Warp-FSI sync API | 7,201 | 1,203 | PASS |
| Warp-FSI kernel time | 1.760184 s | 0.413734 s | PASS，4.25× 仅限 kernel time |
| 三篇 3D 完整矩阵 wall | 582.976084 s | 448.384719 s | PASS，1.30017× |
| 峰值显存 | 11,104 MiB | 6,556 MiB | PASS，组合配置 −40.96% |

最终 FSI profile：

- Nsight：`8b78a0e8e82b60387bb68470981b774d589fd2533a3016d37293523d9ec4a2b0`
- SQLite：`6dc31d45b01d1fa007bb398730476175d0fe8727fe056408b839590a8e3d1e80`

## 保留警告

1. Baik 后续应直接使用纯 GT 文件。
2. 性能数字是单次确定性测量；首次调用受本机 Inductor 磁盘缓存影响。
3. 显存下降只能归因于最终组合配置，不能只归因于 allocator。
4. 当前隔离 worktree 未携带默认 MATLAB fixture，12 层复放需显式绑定外部只读 fixture。
5. metrics 中 G0/G0b/G0c/ledger 是 legacy gate，不是本次 fresh 四论文 GPU 指标。
6. CPU 仍负责配置、第三方几何对象、控制流、I/O、序列化和遥测。

## Claim impact

- “当前授权范围内的 CUDA float64 科学数据面”：**supported**。
- “三篇 3D 完整矩阵单卡端到端约 1.30×”：**supported，development-level/provisional**。
- “Warp-FSI kernel 累计时间约 4.25×”：**supported；不得表述成端到端 4.25×**。
- “整个 Python 进程零 CPU”或“任意 Ptera 配置完全 GPU 化”：**unsupported**。
