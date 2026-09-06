# HANDOFF — Rojratsirikul 2011 Q16–FLUX-V5M 复现：跨机继续开发交接（2026-09-06）

> 目的：在另一台电脑上无损继续本项目的修复与复现开发。
> 远端：`https://github.com/ExuberantWitness/FLUXVortex.git`
> 分支：**`run/q16-lev-tev-pc-fsi-20260821`**（唯一活动分支，全部工作已推送）
> 基线提交：`97e7c92`（本文档）；最新修复链末端 `6db341f`
> 本机运行树：`FLUXV_RUNS/v5m-fa8eaca`（即仓库工作树，下称 REPO）

---

## 1. 新机器启动步骤

```bash
git clone -b run/q16-lev-tev-pc-fsi-20260821 https://github.com/ExuberantWitness/FLUXVortex.git <workdir>
cd <workdir>
# 依赖：python3.11 + torch(CUDA) + warp-lang + numpy/matplotlib；GPU ≥ 12GB fp64（RTX 4090 实测 24GB）
# 所有正式运行的环境（写进 shell profile 或每命令前缀）：
export PYTHONPATH=$PWD/src:$PWD/platform:$PWD/platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 冒烟验证（~6 分钟，应 45+ 全过）：
python3 -m pytest tests/test_q16_flux_v5m_native_gpu.py tests/test_q16_joint_deposited_flow_gpu.py \
  tests/test_native_material_convection_gpu.py tests/test_e0_rhs_observer_parity_gpu.py -q
# 绑定值门（~4 分钟，默认路径逐位不变的最终守卫）：
python3 -m pytest tests/test_roj_unified_case_runner_parity_gpu.py -q
```

无 GPU 的机器：只能做文档/分析；正式求解 GPU-only 是硬合同（CLI 无 CUDA 即失败）。

## 2. 当前状态（详见 §5 文档索引，此处为执行摘要）

- **载荷合同/观测层（P0/G000/E0）完成**：GPU Cn、全空间 T_RBᵀQ_aero wrench、two-tap 双 Cn、production-RHS observer（实际功带含 Mf1·a）、Mf1 统一同源映射（机器精度）。
- **分离/尾迹物理线（P1，实验 flag `joint_separation_solve`，默认 OFF）**：真实材料涡带算子 → RK3 同步对流 → TE 尾迹重连接——三角度（A5/A15/A19）150 步全部稳定，**A19 发散已根治**（4.45e6 → 0.716 vs 实验 0.78）。验证数据入库：`artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/`。
- **生产路径逐位冻结**：一切实验物理在 flag 后面；`test_roj_unified_case_runner_parity_gpu.py` 是不可协商的守卫。
- **Cn 精度声明冻结**：双消费者（U1）未裁决前不得宣称精度改善。

## 3. 下一步工作（按已裁决顺序，勿跳步）

1. **共享边 TE/LE 强度关系**（Bird et al. 2021 式 17–19；区分材料涡片共享丝与 bound-only pin）；
2. 材料历史 + LESP/Kelvin 统一到同一拓扑；Kelvin 改**非赋值式**验证（跨时间/跨表示环量-冲量账本）；
3. 载荷导数对同一活动约束系统计算（分离后导数仍用原 AIC 的不一致）；
4. **长窗验证**（≥t*=10 统计窗）三角度 joint on/off A/B → 评估开 flag；
5. A16 柔性翼同 checkpoint 对照 → A10/A23 holdout；
6. rigid 有限翼载荷 oracle（A5 的 Γ(y)/下洗/有效面积同帧分解）→ 若仍有系统偏差做独立标定低 Re 闭合（M3）；
7. H7 收敛矩阵 → U7.5/U10 泛化 → 论文曲线恢复。

**纪律（多轮独立裁决固化）**：最小验证先行（3 点/短切片否证假设，不用于稳态验收）；每步过 parity；禁止盲删项/调阈值压 Cn/`5P+legacy` 相加/逐图调参/toy 替代/关 LEV/CPU 数值回退；≥2h 长工况前必须先 3 点短切片。

## 4. 常用命令

```bash
# P1 三角度短窗验证（joint flag；结果诊断含 full_surface_neumann_max_abs）
python3 artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/p1_driver.py

# 膜翼正式工况（注册表 case；shadow/observer/two-tap 证据自动入 payload）
python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py --case ROJ11-A16 --max-aero-steps N --output <path>.json

# 刚翼队列（可恢复；Figure 9r/12/13/15）
python3 platform/warp_vpm/queue_roj_rigid_fig9_12_13_15.py

# 全图刷新 + 评分（H1–H6，U 分组）
python3 platform/warp_vpm/compare_rojratsirikul2011_digitized_oracles.py

# 算子诊断（Mf1/载荷一致性）
python3 artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/research/load_diagnosis_20260906/check_acceleration_operator.py
```

## 5. 文档索引（全部在仓库内）

| 主题 | 路径（相对 REPO）|
|---|---|
| 原始任务合同 | `artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/research/HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829.md` |
| 协作状态+bug 台账（最新）| 同目录 `REPAIR_PROGRESS_BUGS_AND_PLAN_AGENT_DISCUSSION_20260906.md` |
| 误差数值全表 | 同目录 `ERROR_REPORT_FOR_AGENT_HANDOFF_20260903.md` |
| 四轮复核报告 | 同目录 `LOAD_REPRODUCTION_DIAGNOSIS_…20260906.md`、`P0_REVIEW_…6527A6F_20260906.md`、`P1_STAGE2_…20260906.md`、`P1_REAL_B_REVIEW_…20260906.md` |
| 修改总方案（M/P 线）| 同目录 `MODIFICATION_PLAN_ROJ_ACCURACY_PERFORMANCE_20260830.md` |
| 载荷合同研究方案 | `refine-logs/q16-v5m-gpu-load-contract-20260831/FINAL_PROPOSAL.md`（§9 GO/NO-GO 表）|
| 数据保全清单 | `artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/DATA_MANIFEST.md` |
| 实验真值（SHA 冻结）| `artifacts/experiment/…/observations/figure_digitization_20260829/`（loader：`src/fluxvortex/cases/rojratsirikul2011_observations.py`）|

## 6. 关键代码地标

| 内容 | 位置 |
|---|---|
| 联合分离求解（真实涡带基 + LE 闭合；flag 门控）| `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` propose 内 `joint_solve` 分支 |
| 同步 RK3 材料对流（实验路径）| `src/fluxvortex/warp_fsi/native_material_convection.py` |
| TE 重连接（实验路径）| 同 propose 内 `reconnect_wake` 段 |
| Mf1 同源映射（已修，默认路径）| `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py`（`pressure_map @ gamma_rate`）|
| 统一载荷 packet + shadow 消费者 | `src/fluxvortex/aero/v5m/load_packet.py`、`q16_shadow_resolved_consumer.py` |
| RHS observer/实际功带（默认开）| `q16_flux_v5m_native_fsi.py` `_integrate_structure`；runner 聚合 `rhs_observer_evidence` |
| two-tap 双 Cn + T_RB wrench | `case_runner.py`（`cn_constant_current`/`cn_full_action`）+ `q16_rigid_body_wrench.py` |
| 尾流探针（谱窗口问题 U6 待修）| `src/fluxvortex/warp_fsi/wake_probe_observer.py` |

## 7. 未决问题（接手时先读 §3 顺序与协作文档的 Q-A~Q-D）

U1 Cn 双消费者裁决 · U2 共享边强度账本 · U3 bound_rate/Mf2_1/Mf1 时间分解相容性 · U4 载荷导数同约束 · U5 势流类偏差（M3 前置 rigid oracle）· U6 谱窗口 ΔSt=0.27–0.40 · U7 性能（P1–P4）。

## 8. 本机遗留状态

- 无运行中 GPU 任务（全部验证完成即停）；
- `/tmp` 下一次性驱动脚本已备份入 `artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/`；
- 用户本人未提交的 `UNIFIED_Q16_V5M_FSI_REFACTOR_PLAN_20260826.md` 修改**未入库**（他人文件，勿动勿提交）。
