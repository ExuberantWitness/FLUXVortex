# FLUX-V5M GPU 并行迭代优化报告

日期：2026-08-21
设备：NVIDIA GeForce RTX 4090 D 24 GiB
基线提交：`f6251cd`
分支：`run/v5m-full-gpu-20260820`

## 1. 结论

本轮目标不是单纯证明“科学计算在 GPU 上”，而是在不改变四论文模型、网格、时间步、阈值和评分方法的前提下，提高 FLUX-V5M 的单卡迭代吞吐。

结果：四论文精度门全部保持。严格输入/精度/PCG 失败门闭合后的 fresh 复跑中，三篇 3D/Ptera 完整矩阵由 **582.976 s** 降至 **448.385 s**，端到端提速 **1.300×**（wall time 减少 **23.09%**）；峰值设备显存由 **11,104 MiB** 降至 **6,556 MiB**。门修复前的同一优化实现曾测得 443.006 s / 6,494 MiB，说明 fresh 复跑落在约 1.2% 的单次波动内。Baik 2D LDVM 每个 1536-step 工况约 3.6 s，指标与优化前完全一致。

这不是“任意 Ptera 配置都完全 GPU 化”的声明。当前生产范围仍是能力矩阵中授权的单机、单翼、无 image 工况，以及标准 Warp-FSI 数据面；多翼、多机和 image surface 继续在首步前 fail-close。

## 2. 实施内容

### 2.1 Ptera / UVLM / LEV

1. 四条环涡边由四次 Python 调用改为一次大批量 CUDA 求值；
2. 四组面板载荷目标点合成一个 `4N` 批量，只遍历一次 bound/wake/particle 源；
3. 固定与动态 wake 拓扑共用 shape-polymorphic `torch.compile` 融合核；
4. 只需要总诱导速度的路径在融合图中直接完成 ring 维归约，避免输出大展开张量；
5. 动态增长 wake 使用 expandable CUDA allocator，消除历史形状造成的缓存碎片。

### 2.2 Warp-FSI

1. PCG dot 改为每自由度一线程的并行归约；
2. 合并 `alpha + x/r`、Jacobi + `r·z`、`beta + p` 三组更新；
3. 收敛检查由每次迭代改为每 8 次一次，最终仍强制检查；
4. CPU/MATLAB 模块只用于离线 oracle，不进入 V5M 生产 facade。

### 2.3 已否决路线

多 Python 线程 + 多 CUDA stream 在同一张 4090 D 上没有收益：

- active 4×20 step：1.560 → 1.659 s；
- attached 4×64 step：4.357 → 4.837 s。

因此当前采用单进程内批量、融合、归约和编译缓存复用；未来若继续做多工况并行，应增加真正的 case 维张量批处理。

## 3. 性能证据

| 路径 | 基线 | 最终 | 变化 |
|---|---:|---:|---:|
| 环涡 96 targets × 96 rings | 0.736 ms | 0.227 ms（仅批量） | 3.24× |
| active-LEV 20 step | 约 0.390 s/工况 | 约 0.128 s warm | 约 3.05× |
| 38-step 生产 warm CUDA kernels | 35,575 | 21,725 | −38.93% |
| 38-step 生产 warm kernel time | 57.159 ms | 35.969 ms | −37.07% |
| Warp-FSI CUDA kernels | 78,758 | 40,475 | −48.61% |
| Warp-FSI synchronize calls | 7,201 | 1,203 | −83.29% |
| Warp-FSI kernel time | 1.760 s | 0.414 s | −76.49%（kernel time 4.25×） |
| 三篇 3D 完整矩阵 wall | 582.976 s | 448.385 s | −23.09%（1.300×） |
| 三篇 3D 峰值设备显存 | 11,104 MiB | 6,556 MiB | −40.96% |

融合核首次进程内调用实测约 2–3.4 s；它包含本机已有 Inductor 磁盘缓存的影响，因此不是严格的机器冷启动统计。正式参数扫描在同一进程运行并复用缓存。非常短的一次性诊断可以设置 `FLUXV_V5M_FUSE=0`，但正式论文矩阵保持默认开启。

上述 4.25× 是 CUDA kernel 累计时间之比；Nsight 全 trace 时长约为 1.08×，不能把 4.25× 表述为 Warp-FSI 端到端提速。Ptera 的 38-step baseline 与 final 使用相同生产步数，但 baseline 文件名不是 `warm`；表中按执行内容而不是文件名标注。显存下降是最终组合配置的实测结果，不单独归因于 expandable allocator。

## 4. 数值复验

最终 CUDA 重算结果：

| 文献工况 | 指标 | 最终值 | 优化前值 |
|---|---|---:|---:|
| Baik W1–W4 | CL macro RMSE | 0.42156276782081215 | 相同 |
| Baik W1–W4 | CD macro RMSE | 0.2897097942292014 | 相同 |
| Yang 2025 | lift MAE | 4.104826595775628 gf | 4.104826595775652 gf |
| Yang 2025 | drag MAE | 1.518625589657690 gf | 1.518625589657666 gf |
| Izraelevitz 2017 | CT MAE | 0.017452113111165457 | 0.017452113111165308 |
| Mancini fast | CL RMSE | 1.2553197460217826 | 1.2553197460217713 |
| Mancini slow | CL RMSE | 0.29509477610137613 | 0.29509477610137613 |

汇总指标差异约为 `1e-14`，完整曲线逐点最大差异约 `2.1e-11`；二者都属于浮点归约顺序变化，不改变任何结论或阈值判定。

其他执行门：

- GPU 合同/负攻击/数值测试：40 passed；
- Warp-FSI 12 层：全部 PASS；
- STRUCT_CG：145 iterations，GPU residual `4.178e-11`，CPU 相对误差 `3.531e-13`；
- LDVM CPU/混合输入、FSI float32 配置、PCG 耗尽三类 fresh hostile probe 均 fail-close；
- CUDA 指标独立复算：PASS。

## 5. 证据路径与 SHA-256

- 最终 Baik summary：`f2fba13edd70f089f8e3d7da027e7d98c3ba178f228db06720919b9a94104592`
- 最终三篇 summary：`0cf3c056c3a591e1ec8570a24a21f3d9ec832e66d5b838537ed1a34442a2a797`
- 最终 CUDA metrics：`06ac52ebdb0dc38ee90d4b85664304bf5e9004d7eb54ba366b48c0aa2022a068`
- 生产 warm Nsight：`8d3c434f747fe1c5ebaa757597b61249bd9a9891233feb1ae9f09c067da41f76`
- Warp-FSI final-gate Nsight：`8b78a0e8e82b60387bb68470981b774d589fd2533a3016d37293523d9ec4a2b0`
- Ptera baseline/final SQLite：`2c9ae29c13cf343b9562ca73e9a1b1d46eae2e80a28f2dd197777a76a5858815` / `7eb8d1c67667afba8f41c94132bd811bf6f2d5b9c9bd8a42482528a55722580b`
- Warp-FSI baseline/final SQLite：`510f2b4017137167678f07876853e85a708e560cd525fdcb813905e4d9e1feb6` / `6dc31d45b01d1fa007bb398730476175d0fe8727fe056408b839590a8e3d1e80`
- 生产 benchmark 源码：`c1489d9cdfd1177d6c002bb206f81a3f658f6a7eb7a45e309466dc5eb26049b0`
- Ptera CUDA backend 源码：`9c0eb2f57fcd8d7e8b0c4186286106149e0195b40bc2ed0c0a9a16916480fb3a`

## 6. 仪器化变更记录

为满足性能证据可追踪要求，本轮新增/生成：

- `benchmark_iteration.py`：只运行生产 `CudaJointLEVTEVSolver`，分离 compile-cold 与 warm 阶段；
- `v5m_production_iteration_warm_final.nsys-rep/.sqlite`：CUDA profiler API 截取的 warm-only profile；
- `v5m_warp_fsi_all_final_gate.nsys-rep/.sqlite`：融合 PCG 并闭合最终收敛门后的全层 profile；
- `four_paper_optimized_verified/`：最终四论文输出与 CUDA 指标重算。

仪器化不改变模型参数、网格、时间步、实验数据或评分公式。旧 profile 和中间优化输出保留，用于审计前后变化，没有覆盖。

## 7. 剩余性能边界

warm 38-step profile 仍有 21,725 个 CUDA kernels 和 5,147 次 stream synchronize。主要来源是第三方 Ptera 对象逐步发布、门控 `.item()`、面板/尾迹对象的 host 编排与结果序列化。继续压缩需要把整个 case/time 维状态改成持久 CUDA 张量或 CUDA Graph 兼容状态机，属于下一轮架构升级，不能以删除数值门或延迟错误为代价。

性能数字均来自单次确定性运行，不代表统计置信区间。Warp trace 中 memcpy 事件数下降但总传输字节几乎不变，说明下一阶段的主要空间是减少 host/device 数据往返总量，而不只是继续融合小 kernel。

下一优先级：

1. case 维真正批处理 AIC/solve/load；
2. 常驻 CUDA 的几何与 wake 状态，减少每步 H2D/D2H；
3. 批量异步诊断与最终一次性发布；
4. 在上述状态固定后，再评估整步 CUDA Graph。
