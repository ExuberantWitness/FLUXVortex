# FLUX-V5M 四论文 GPU-only V2 最终验证报告

日期：2026-08-20（Asia/Shanghai）  
基线提交：`fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169`  
执行设备：NVIDIA GeForce RTX 4090 D（24 GiB，SM 8.9）  
环境：Torch 2.11.0+cu130 / CUDA 13.0 / Warp 1.14.0

## 1. 最终结论

独立 fresh 只读审计结论为 **PASS，无 blocker，带非阻断 WARN**。

本结论严格限定为 24 个冻结工况：Baik 4 个、Yang 6 个、
Izraelevitz–Scherer 12 个、Mancini 2 个。正式科学数值计算使用 CUDA
float64；CPU 只承担 Ptera 几何/配置对象和容器管理、Python 调度、文件
I/O、GPU launch、序列化与遥测。这里不声称 Python 控制面本身运行在 GPU，
也不把范围扩展到整个 FLUX-V5M 或通用 Ptera。

| 文献/工况 | FLUX-V5M GPU V2 | 冻结 V4B | 相对变化 | 判定 |
|---|---:|---:|---:|---|
| Baik 2012 W1–W4，CL macro RMSE | 0.421563 | 0.6580 | 改善 35.93% | PASS |
| Baik 2012 W1–W4，CD macro RMSE | 0.289710 | 0.3450 | 改善 16.03% | PASS |
| Yang 2025 Fig.11，升力 MAE | 4.10483 gf | 4.55 gf | 改善 9.78% | PASS |
| Yang 2025 Fig.11，阻力 MAE | 1.51863 gf | 2.64 gf | 改善 42.48% | PASS |
| Izraelevitz–Scherer 2017 Fig.14，CT MAE | 0.0174521 | 0.0198 | 改善 11.86% | PASS |
| Mancini 2017 Fig.4.13b，快俯仰 CL RMSE | 1.25532 | 1.2184 | 变差 3.03% | PARTIAL |
| Mancini 2017 Fig.4.13b，慢俯仰 CL RMSE | 0.295095 | 0.2908 | 变差 1.48% | PARTIAL |

Mancini 的 GPU bare RMSE 分别为 1.444895 和 0.389158；加入冻结的
LDVM 分离增量后有改善，但仍未超过 V4B。因此不能写成“四篇均优于 V4B”。

## 2. GPU-only 执行边界

正式冻结入口中，以下影响预测的数值步骤均由 CUDA 完成：

- Ptera 诱导速度、AIC、尾迹影响、稠密线性求解、载荷和力矩；
- prescribed-wake 坐标推进与由行号、时间步生成的 wake age；
- strip-area reduction、LESP/阻力账本；
- LDVM Fourier 投影、TEV/LEV 联立、尾迹诱导与推进；
- 有限翼投影、polar residual、插值、滤波和最终误差指标；
- Baik 四工况的单例 RMSE 与 macro reduction。

为关闭首轮审计发现的 host 路径，V2 增加了：

1. CUDA prescribed-wake 坐标推进；
2. CUDA wake-age 派生，父类 CPU age 累加被覆盖；
3. 父类 NumPy final mean/RMS 汇总被禁止，`only_final_results=True`
   fail-closed；
4. CUDA masked LEV buffer，`step()` 内无 `.item()` 主机分支；
5. strip-area、有限翼增益、polar slope 与 Baik macro score 的 CUDA 化；
6. CPU LAPACK、Ptera CPU ring velocity/cross/load、父类 wake-point、
   wake-vortex 和 finalize 路径的“调用即失败”攻击回归。

允许的 CPU 工作只包括配置/几何对象构造、数据搬运、容器更新、Python
循环调度、文件读取、哈希/JSON/NPZ 序列化和 NVIDIA 遥测。这些工作不执行
气动力、诱导速度、方程求解、时间推进、修正或评分。

## 3. P0/P1 修复效果

| 项目 | 验证状态 | 证据边界 |
|---|---|---|
| P0-1 ledger 裁剪后 total 陈旧 | PASS | CUDA 账本以零容差验证 `total=t1+t2+t3`；提交级旧诊断 closure 也是 0 |
| P0-2 CUDA 硬编码/设备选择 | PASS | 本轮进一步收紧为 CUDA mandatory；CPU、非法设备、无 CUDA 均 fail-fast |
| P1-1 G0 chassis/阈值/CL 上界 | PASS（提交级） | `fa8eaca` 源码与旧正门记录为 CL=0.4850；legacy G0 含 CPU Ptera，因此未冒充本轮 GPU fresh 指标 |
| P1-2 G0/G0b/G0c 失败退出 | PASS（源码/负控） | 三个入口的失败路径均为非零退出；旧正门数值不纳入四论文 GPU claim |

`metrics_gpu_only_v2.json` 中的 G0/G0b/G0c/P0 字段明确标注为
`fa8eaca` 提交级诊断。四论文 GPU 科学结论只使用本轮 fresh 生成的七个论文指标。

## 4. 运行与路径证据

- Baik：W1–W4 各 3×512 个 CUDA LDVM step；GPU 峰值利用率 56%，
  峰值显存 1181 MiB，57 个遥测样本。
- Yang：6 个迎角，每工况 512 步。
- Izraelevitz–Scherer：12 个条件，每工况 513 步。
- Mancini：快/慢各 897 步。
- 三篇三维工况合计：AIC/wake/solve/load/ledger 各 11,022 次，
  wake-convection 11,002 次；完整重跑 582.80 s。
- 三篇矩阵 GPU 峰值利用率 75%，峰值显存 11,146 MiB，2,032 个遥测样本。

最终源码绑定的 8 步 Nsight smoke：

- report SHA-256：
  `6001d5900a0ff943cccacc1e54dd8dfe183baba2d303a9223b6629f01f636fa9`；
- 16,516 次 `cudaLaunchKernel`；
- 捕获 CUDA reduction、cross、CUTLASS GEMM、`getrf_pivot` 与上下三角
  `trsm`。

该 Nsight 文件只证明当前源码的内核机制，不冒充 24 个完整工况的 profile；
完整运行的 GPU 证据来自每工况计数与 NVIDIA 运行时遥测。

## 5. GT、评分与产物闭合

四篇均使用数字化实验观测，评分没有用预测自身的最大值、均值或范围作
归一化：

- Baik：scored 文件的 `experiment` 列按相位去重。去重后的 3200 点与纯
  GT 公共点逐位相等；纯 GT 只额外包含 8 个 `phase=1` 周期重复端点。
- Yang：只读 `test_lift_gf` 和 `test_thrust_gf` 实验列。
- Izraelevitz：只读 `experimental_observation` 标记。
- Mancini：校验 GT 文件 SHA、列、网格和实验数据角色。

最终主要哈希：

- Baik summary：
  `f2fba13edd70f089f8e3d7da027e7d98c3ba178f228db06720919b9a94104592`；
- 三篇 summary：
  `6ecd1a046f4d31391485c38ff92e585e7803cd03a815e975f55f90b19cd9abcb`；
- 三篇 canonical result：
  `c4cc9bcc635717351982f611747dcd069293463902e0721167fff049fd5ac437`；
- GPU 独立重算 metrics：
  `5479580ebf8fd0b86c357e3c006db75056aaef262f407ed92e0ff5a25d75bbf8`；
- 三篇完整日志：
  `bffb7af70a99bafc15c651d2b6d181689ff0179c319ec952f336a4e65bcb9c86`；
- Baik 完整日志：
  `bc9e4b41997d70c8b4250f48d2ee5ccd7b9ec34c2af09d8abbfb988094c50394`。

三篇 summary 保存原始预测与实验曲线；Baik 四个 NPZ 保存 128 点
phase/CL/CD，并分别绑定原始数组 digest。独立审计已从这些持久化数据重算
全部指标与摘要哈希。

## 6. 测试与静态门

- GPU/P0 回归：15/15 PASS；
- 13 个本轮冻结目标：Black、Ruff、py_compile PASS；
- `git diff --check` PASS；
- current-source Nsight oracle：1/1 PASS。

`test_ledger_contract.py` 是 `fa8eaca` 的旧 collection-time CPU 诊断脚本，
不属于上述 13 文件静态门，且仍有旧格式问题；本轮 GPU ledger closure 由新
CUDA 回归覆盖。因此不能声称整个 `platform/warp_vpm` 目录 style clean。

## 7. 非阻断 WARN 与不可支持声明

1. 有限翼投影以 `alpha_rad.device` 为锚；混合 CPU/CUDA 输入会先上传到
   CUDA 后计算，而不是 API 级拒绝。正式 runner 的五个输入均原生 CUDA，
   因此不影响本轮结果，但不能宣称所有 CPU 输入都 fail-close。
2. 每个冻结工况只运行一次，没有重复试验、网格/时间步收敛或不确定度传播。
3. 当前严格 CUDA Ptera 仅覆盖 attached、单机翼、无镜像面的 prescribed-wake
   入口；不覆盖三维 active-LEV、多翼、镜像面或自由尾迹。
4. 不支持“整个 FLUX-V5M/Ptera 全路径 GPU-only”“四篇全部优于 V4B”
   “鲁棒性/泛化性已证明”等扩大声明。

## 8. Instrumentation changelog

- 新增每步 CUDA AIC/wake/solve/load/ledger/wake-convection 计数；只观测
  执行路径，不参与预测。
- 新增 `gpu_runtime_monitor.py`，轮询 NVIDIA 设备利用率和显存；只记录
  设备遥测，不改变模型状态、时间步或数值顺序。
- 新增 current-source Nsight Systems smoke；独立于正式评分矩阵运行，未向
  模型注入影响结果的分支。
- runner 新增源码/GT SHA、原始曲线、设备/版本和 canonical result hash；
  这些仅影响 provenance 与序列化。
- 所有运行结束后均无后台 monitor/session。

## 9. 审计判定

独立 reviewer：same-family，acceptance status 为 provisional。A–F 结论：

- A GT provenance：PASS；
- B score normalization：PASS；
- C result existence/integrity：PASS；
- D dead code/execution path：PASS；
- E scope：PASS，但严格限定于 24 个冻结确定性工况；
- F evaluation type：`real_gt`。

完整审计见 `EXPERIMENT_AUDIT_V2.md` 与 `EXPERIMENT_AUDIT_V2.json`。
