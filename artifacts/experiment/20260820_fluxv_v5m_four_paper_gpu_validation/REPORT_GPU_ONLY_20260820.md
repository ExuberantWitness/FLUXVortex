# FLUX-V5M 四论文 GPU-only 验证报告

> **已撤回/仅作首轮 provisional 记录。** 2026-08-20 fresh audit 判定本报告的
> “完整 GPU-only”结论不成立，原因与证据见 `EXPERIMENT_AUDIT.md`。本文中的
> 精度数字可复算，但在 GPU-only v2 全矩阵重跑和第二次审计 PASS 前不得引用为
> 最终 GPU-only 结果。

日期：2026-08-20（Asia/Shanghai）  
基线提交：`fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169`  
执行设备：NVIDIA GeForce RTX 4090 D，24 GiB，SM 8.9，driver 580.173.02  
数值环境：Torch 2.11.0+cu130 / CUDA 13.0，Warp 1.14.0

## 1. 结论

四篇论文的正式 fresh 结果现已全部由 CUDA 数值后端生成。这里的合同是：CPU 仅负责 Python 调度、Ptera 几何对象更新、文件读取和 CUDA 结果序列化；诱导速度、AIC、尾迹影响、稠密求解、载荷、LDVM 时间推进、有限翼修正、阻力账本和误差指标均在 GPU float64 上计算。任何无 CUDA、显式 CPU 设备、非法设备或隐藏 CPU LAPACK/Numba 热路径均 fail-fast。

总体结果不是“四篇都优于 V4B”：Baik、Yang、Izraelevitz 优于冻结 V4B；Mancini 快/慢俯仰略逊于 V4B，但明显优于同一 GPU 三维底盘的 bare 结果。禁止把 Mancini 写成优于 V4B。

| 文献/工况 | GPU-only FLUX-V5M | 冻结 V4B | 相对变化 | 判定 |
|---|---:|---:|---:|---|
| Baik 2012 W1–W4，CL macro RMSE | 0.445204 | 0.6580 | 改善 32.34% | PASS |
| Baik 2012 W1–W4，CD macro RMSE | 0.284027 | 0.3450 | 改善 17.67% | PASS |
| Yang 2025 Fig.11，升力 MAE | 4.10483 gf | 4.55 gf | 改善 9.78% | PASS |
| Yang 2025 Fig.11，阻力 MAE | 1.51863 gf | 2.64 gf | 改善 42.48% | PASS |
| Izraelevitz–Scherer 2017 Fig.14，CT MAE | 0.0181885 | 0.0198 | 改善 8.14% | PASS |
| Mancini 2017 Fig.4.13b，快俯仰 CL RMSE | 1.25532 | 1.2184 | 变差 3.03% | PARTIAL |
| Mancini 2017 Fig.4.13b，慢俯仰 CL RMSE | 0.295094 | 0.2908 | 变差 1.48% | PARTIAL |

Mancini 的 GPU bare RMSE 分别为 1.44490 和 0.389158；加入冻结的 CUDA LDVM 分离增量后降至 1.25532 和 0.295094，说明机制修正有效，但尚未超过 V4B。

## 2. GPU-only 修复内容

1. `pfield.py` 现在只允许 CUDA；新增 Gaussian-erf 速度与解析 Jacobian 的 Warp CUDA kernel，移除 Jacobian CPU oracle 生产调用。
2. `warp_fsi/config.py` 新增 `FLUXV_GPU_ONLY=1` 硬门；CPU、非法设备、无 CUDA 均拒绝。
3. `batched_solver.py` 移除 CUDA 小批量 `B<=4` 时偷偷调用 `np.linalg.solve` 的分支；CUDA 请求始终使用 GPU LU。
4. 新增 `LDVM2DCuda`：Baik 与分离增量的尾迹诱导、Fourier 投影、TEV/LEV 联立求解、力和尾迹推进均用 Torch CUDA float64。
5. 新增受限 `CudaAttachedJointLEVTEVSolver`：只支持当前三篇三维工况实际使用的单机翼、无镜像面、`enable_lev=False` 路径。AIC、RHS、尾迹、`torch.linalg.solve`、Kutta–Joukowski/非定常载荷、力矩和 LESP 账本均在 CUDA；超出合同直接拒绝。
6. 新增 CUDA 有限翼 polar、LDVM separated-minus-attached 投影及 T1/T2/T3 阻力账本。

此限制不会改变本轮四篇论文的冻结方法：三篇三维论文原本就使用 `enable_lev=False`；Baik 的 LEV 由独立 CUDA LDVM 负责。旧 G0c 三维 active-LEV 门仍不是严格 GPU-only 路径，因此没有把其旧 CPU/PField 混合结果计入本报告。

## 3. P0/P1 修复复核

- P0-1：`ledger total == t1+t2+t3` 回归通过，CUDA 账本也以零容差验证闭合。
- P0-2：在 `fa8eaca` 自动选择基础上进一步收紧为 CUDA mandatory；`cpu`、bogus device 与 CPU Jacobian/CPU LAPACK 热路径攻击均被拒绝。
- P1-1/P1-2：`fa8eaca` 中 G0 阈值和 G0/G0b/G0c 非零失败退出逻辑保留；但这些 legacy 脚本包含 CPU Ptera 数值，因此只作为提交级修复事实，不作为本轮 GPU-only 正向科学证据。
- GPU 替代正门：小型三维运动的 CUDA Ptera 载荷与冻结 CPU oracle 最大相对差约 `1.1e-15`；运行时把 `np.linalg.solve`、Ptera Numba ring velocity、CPU cross 和 `process_solver_loads` 全部替换为“调用即失败”后仍通过。

## 4. 运行与计数证据

- Yang：6 个迎角，每个 512 步；每工况 AIC/wake/solve/load/ledger 均精确 512。
- Izraelevitz：12 个条件，每个 513 步；每工况五类计数均精确 513。
- Mancini：快/慢各 897 步；每工况五类计数均精确 897。
- 三篇三维矩阵合计每类 11,022 次 CUDA AIC/wake/solve/load/ledger；冻结重跑耗时 612.94 s。
- Baik：W1–W4 各 3×512 个 CUDA LDVM step；W1–W4 数组哈希均独立重算一致。
- 直接 Nsight CUDA Ptera smoke 捕获 16,382 次 `cudaLaunchKernel`，包含 CUDA vector/reduction/cross kernels、CUTLASS GEMM、`getrf_pivot` 和 `trsm`，证明稠密 solve 实际进入 GPU 库而非 CPU LAPACK。

Nsight 文件：

- `profile_output/v5m_gpu_only/ptera_cuda_backend.nsys-rep`
- `profile_output/v5m_gpu_only/ptera_cuda_backend_stats_cuda_gpu_kern_sum.csv`
- `profile_output/v5m_gpu_only/ptera_cuda_backend_stats_cuda_api_sum.csv`
- 较早 Warp UVLM smoke：`profile_output/v5m_gpu_only/uvlm_smoke.nsys-rep`

## 5. 测试与静态门

最终冻结源码上：

- GPU/P0 回归：`11 passed in 5.40s`。
- 其中包含设备负控、B=1 CPU LAPACK 攻击、Gaussian-erf Jacobian、CUDA LDVM、CUDA Ptera hot-path 攻击、polar/ledger oracle 与 P0-1 closure。
- Black：11 个目标文件全部 unchanged。
- Ruff：PASS。
- `py_compile`：PASS。
- `git diff --check`：PASS。

## 6. 产物与哈希

Baik：

- `fresh_results/baik_gpu_only/summary.json`
- 文件 SHA-256：`c798de28f4feecef8d37c5f74dc1660cdca7bd07b75fd4b3096247e6a241a5e9`
- runner：`c84861fc28c7f7f4e95c17c460b302885d0b1b48e1e7bb609156d7cf8cafe90e`
- LDVM CUDA：`4ea39c941e23942094648cdbc0564631bf82f77be200e2fc6ef6f66b53a67ba1`

三篇三维：

- `fresh_results/gpu_only_three_papers/summary.json`
- 文件 SHA-256：`cfe5032ac349d20229aa1b83706faee9a920960348bccf18ddbd963e67c7f2b4`
- 内容 result SHA-256：`bb1251110df35308ba6cc4169671c1161250d0cd6c40197fb484b1aac2b032d6`
- runner：`0b3f0218fce31f61ff9608e9e9021420fdb80717ccb20e92be6f774de50faea7`
- CUDA Ptera：`a11d825b8f40f4b21fe267344a6574b66405985adecf308046c9c7e3f67249ad`
- CUDA corrections：`9eb7988df45ca717d7b7471f2d617169667aaef1398e5d433955bfefe838d434`
- LDVM CUDA：`4ea39c941e23942094648cdbc0564631bf82f77be200e2fc6ef6f66b53a67ba1`

所有嵌入源码哈希均与冻结后实盘文件一致；三篇结果内容哈希、Baik 四个 NPZ 数据哈希均已独立复算一致。

## 7. 限制与后续

1. 这是开发级 GPU-only 验证，不是论文 artifact registry 的最终 A/B 审计包。
2. CPU 仍承担 Ptera 几何对象构造/更新和最终序列化；没有声称 Python 控制平面本身在 GPU。科学载荷数值链已在 CUDA。
3. 严格 CUDA Ptera 当前只覆盖冻结论文所需的 attached、单机翼、无镜像面路径；三维 active-LEV、镜像面和多机翼必须另行移植并新增门，不能静默复用本结果。
4. Mancini 两项未超过 V4B；后续若优化必须重新预注册，禁止用本轮实验曲线调 Lcrit 或参数。
5. 下一步应由 fresh reviewer 做结果/代码只读复核，然后提交独立分支；不要直接改写 `fa8eaca` 历史提交。

## 8. Instrumentation changelog

- 新增每步 CUDA AIC/wake/solve/load/ledger 计数并写入正式 JSON；这些计数只观测路径，不改变数值。
- 新增 Nsight Systems smoke profile；profile 运行独立于正式评分矩阵，不向模型注入 NVTX 或计时分支。
- runner 新增运行时源码 SHA 和设备/版本字段；只影响 provenance 与序列化，不影响力、系数或指标。
- 未增加长期监控守护进程；运行结束后没有后台 session。
