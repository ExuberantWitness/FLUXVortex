# FLUX-V5M 四 CASE 复现基线计划

日期：2026-08-22  
路线：复用并复验（attach/reuse-and-verify）  
状态：Baik W1--W4 与 Mancini fast/slow fresh mandatory GPU 复现通过；Yang、Izraelevitz 待迁移

## 不可变要求

1. 当前目标是工程化复现 CASE，不继续开展周期机理、参数扫描或论文级泛化研究。
2. 科学数值数据面必须使用 CUDA float64；不允许 CPU fallback。
3. separated LEV、joint TEV 和 free wake 是生产 CASE 的强制组成，不允许通过关闭它们获得通过结果。
4. 不改变论文数据、几何、运动、评分定义或冻结参数来追逐指标。
5. Q16 FSI 是独立集成验证线，不作为四篇刚性气动 CASE 的替代指标。

## 复用对象

- 历史四 CASE GPU 对照（只作 reference，不是当前合格结果）：`artifacts/experiment/20260820_fluxv_v5m_full_gpu/fresh_results/metrics_gpu_only_v2.json`
- 运行与设备证据：`artifacts/experiment/20260820_fluxv_v5m_full_gpu/`
- 冻结基线提交：`f6251cd`
- 当前开发提交：`dc43d4cc0c22`

## CASE 与验收面

| CASE | 冻结指标 |
|---|---|
| Yang | lift/drag MAE（gf） |
| Scherer/Izraelevitz Figure 14 | CT MAE |
| Baik W1–W4 | filtered CL/CD macro RMSE |
| Mancini 2017 | fast/slow pitch-ramp RMSE |

历史三维 runner 使用 `enable_lev=False + prescribed_wake=True`，不满足当前模型定义。首轮只做三件事：把生产入口改为 reduced-mode fail-close、跑一个 mandatory-mode GPU-only 冒烟门、为四 CASE 建立不改变 GT/评分的生产适配入口。任何会改变默认气动数值的周期探索改动先隔离，不带入该基线。

### 当前执行入口

Mancini 2017：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
python platform/warp_vpm/reproduce_mancini_v5m_mandatory.py \
  --case fast_pitch --quality smoke --show-progress
```

smoke 通过后，用 `--quality full` 分别运行 `fast_pitch` 和 `slow_pitch`。
生产结果不再叠加 LDVM separation delta，因为三维 separated LEV 已由同一
联合尾迹承担，重复叠加会造成分离载荷双计。

历史执行已经证明常强度闭合环候选不满足此前提：fast smoke/full 和 slow
smoke 均未通过冻结 reference 门，且时间细化误差增大。不得继续运行 slow full，
也不得把矩阵残差闭合写成 CASE 复现通过。下一实现入口必须复用现有 v5h 的
`CUDA DVM source -> node-owned connected ribbon -> material rVPM -> Ptera RHS/load`
谱系；首先在 Mancini fast smoke 上要求 `Gamma_birth=O(dt)` 和误差不劣于当前
fast smoke，随后才允许跑 full。

当前已完成完整物理数据通路：`LDVM2DCuda(source_parity=True)` 的展向批处理
source bank 导出
新生 LEV/TEV 强度、出生点、LESP 残差和首 TEV 零持久化账本；
`CudaParticleField.add_connected_ribbon_particles()` 可按共享节点消去重复内边，
并以固定核半径/固定间距在 CUDA 上沉积。节点前沿、粒子和 Ptera 自由尾迹已
进入 predictor 分支事务；旧粒子与 newborn 均进入 Ptera RHS，活动分离行由
DVM/Ptera 状态联合保持，最终载荷唯一由 Ptera `KJ+dGamma` 拥有。Mancini
fast/slow full 已据此通过，结果见 `verification.md`。

Baik W1--W4 已有独立的生产入口：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
python platform/warp_vpm/reproduce_baik_v5m_mandatory.py
```

该入口要求 CUDA float64，逐 CASE 检查 separated LEV、TEV 历史和自由尾迹推进，
完整四工况还必须不劣于冻结 GPU reference；任一门失败均保存证据并非零退出。

## 已知风险与处理

- 当前工作树含未提交 Q16/FSI 工作：不清理、不覆盖，只隔离本轮周期探索造成的默认路径变化。
- 完整四 CASE 运行耗时较长：先执行有代表性的 mandatory-LEV GPU 冒烟门；冒烟不通过时不盲目启动全量复现。
- 当前会话没有 `baseline` 技能要求的 `bash_exec`/artifact/memory 接口：使用现有终端与 Markdown 留痕，结果不得据此升级为全局已确认 baseline。
- 已知 v5f 常强度环路线在 2026-08-14 的 M5 时间/core 门已是 NO-GO；本轮
  Mancini 结果是同一 `1/dt` 增长在新 CASE 上的复现，不再对该表示调 core、
  `Lcrit` 或网格。
- node-owned full 的主要吞吐瓶颈是累计粒子云的直接 WRK3 自诱导；fast/slow
  full 分别耗时约 `46.85/18.73 min`。后续性能工作应替换近二次近远场求和，
  不得通过减少论文步数、关闭 LEV 或更改核/间距追求墙钟。
