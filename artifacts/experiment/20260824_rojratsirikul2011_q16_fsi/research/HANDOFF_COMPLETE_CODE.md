# 完整代码 HANDOFF — Q16/FLUX-V5M 生产架构（基线 a0f0869）

> 生成日期：2026-08-24（基线提交当日）。本文为完整代码交接：生产链路 file-by-file 地图、
> 组件门/论文精度门状态、可直接复制的运行协议、按 ANALYSIS_BRIEF_v4 §4/§5 排序的工作队列。
> 所有 git 状态与最后提交哈希均在基线 a0f0869 上逐文件核实。

## 0. 两大事实（评审发现确认）

1. **基线风险已消除**：本次提交 `a0f0869` 将全部生产代码与测试纳入跟踪（此前 **68 个文件未跟踪**：
   `src/fluxvortex/` 17 个——含 `warp_fsi/kernels_q16_mesh.py`、`kernels_q16_ans_eas.py`、
   `kernels_q16_constraints.py`、`q16_ancf_mesh.py`、`q16_ans_eas_continuum.py`、
   `q16_boundary_constraints.py`、`q16_work_conjugate_transfer.py`、`q16_aero_load_packet.py`、
   `q16_lev_impulse_transfer.py`、`q16_mandatory_aero_mode.py` 等；
   `tests/` 15 个（`test_q16_structural_step_gpu.py`、`test_q16_flux_v5m_native_gpu.py` 等全套 Q16 测试）；
   `platform/warp_vpm/` 34 个、`platform/forward_flight_benchmarks/` 2 个——含 `yamano2020_q16.py`
   案例与全部 yamano 脚本；另将 7 个已跟踪文件的未提交修改一并冻结入基线。
   注意：核心求解器本体（`q16_structural_solver.py`、`q16_flux_v5m_native.py`、
   `q16_flux_v5m_author_loads.py`、`q16_flux_v5m_native_fsi.py`、`kernels_q16_transfer.py`、
   `pfield_torch_gpu.py` 等）此前已被跟踪，风险集中在"外围新模块 + 测试层"，现已全部收编）。
2. **架构定位**：现有代码已具备"GPU Q16 结构 + 变形面 FLUX-V5M（环格 + LEV 粒子 + 自由尾迹）+
   强预测校正事务"完整雏形，**无需推倒重来**；当前须区分两个概念——
   **"组件正常"**（组件门全绿：结构步、native GPU oracle、功共轭/载荷包、LEV 冲量、案例合同）与
   **"论文精度正常"**（A16 已近带：camber −1.6%（E1.4 全周期）；A10/A23 未跑）。

## 1. 生产链路代码地图（file-by-file，含 git 状态与验证状态）

所有下列文件均已跟踪（基线 a0f0869 后 `git status` 干净）。"最后提交"为 `git log --oneline -1 -- <file>`。

### 结构链（GPU Q16 壳/连续体）

| 文件 | 最后提交 | 职责 | 验证状态 |
|---|---|---|---|
| `src/fluxvortex/q16_ancf_mesh.py` | `a0f0869`（本次基线新跟踪） | Q16 网格/宏材料（shared-node 16 节点壳宏单元组装） | `tests/test_q16_ancf_shared_mesh_gpu.py`、`test_q16_ancf_continuum_gpu.py` 组件门 |
| `src/fluxvortex/q16_ans_eas_continuum.py` | `a0f0869`（本次基线新跟踪） | ANS/EAS 连续体单元（膜/弯曲协调，剪切锁定与膜锁定缓解） | `tests/test_q16_ans_eas_continuum.py`（CPU oracle）+ `test_q16_ans_eas_continuum_gpu.py`（CUDA 位级对齐） |
| `src/fluxvortex/warp_fsi/kernels_q16_ans_eas.py` | `a0f0869`（本次基线新跟踪） | ANS/EAS CUDA 算子（单元级核函数） | 同上（GPU 测试直接驱动） |
| `src/fluxvortex/warp_fsi/kernels_q16_mesh.py` | `a0f0869`（本次基线新跟踪） | 网格/宏材料 CUDA 算子（共享节点拓扑、装配核） | `test_q16_ancf_shared_mesh_gpu.py` |
| `src/fluxvortex/q16_boundary_constraints.py` | `a0f0869`（本次基线新跟踪） | 四边固支边界约束（投影/掩码层，Python 侧） | `tests/test_q16_boundary_constraints_gpu.py` |
| `src/fluxvortex/warp_fsi/kernels_q16_constraints.py` | `a0f0869`（本次基线新跟踪） | 约束施加 CUDA 算子 | 同上 |
| `src/fluxvortex/warp_fsi/q16_structural_solver.py` | `f63da18` | Newmark 时间推进 + Newton 非线性 + GPU PCG（Jacobi 预条件）+ `reference_dense` 周期刷新 + 刚度阻尼（Kelvin-Voigt θ）+ Rayleigh 预留（`mass_damping_coefficient` 钩子，现值 0.0，S1 直接启用） | `tests/test_q16_structural_step_gpu.py`（7 用例）+ `test_q16_structural_pcg_gpu.py`（7 用例） |

### 气动链（变形面 FLUX-V5M）

| 文件 | 最后提交 | 职责 | 验证状态 |
|---|---|---|---|
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` | `66194d6` | V5M 求解器本体：环格 AIC、LESP 临界判定与释放、LEV 粒子（年龄/强度）、joint TEV、自由尾迹、攻角诊断、分块环核、dense transfer 出口 | `tests/test_q16_flux_v5m_native_gpu.py`（9 用例：AIC/Mf1/Mf2 喂 MATLAB 态位级 oracle） |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py` | `944080d` | 作者压力分解：dp_lift1 / mf2 / lift2 / mf21 + Mf1 added-mass + aic LU 缓存（`944080d` 引入 LU factorization 复用） | 同上（native oracle 用例）+ Yamano 载荷门（`aic_relative_error ~3e-7`、`mf2_max_abs_error ~3.8e-17`） |
| `platform/warp_vpm/pfield_torch_gpu.py` | `dc04e10` | 粒子场（LEV/TEV 质点）+ 融合 Biot-Savart 核（`dc04e10` fused warp kernel，速度计划 item 1） | `test_q16_flux_v5m_native_gpu.py` 粒子路径 + `platform/warp_vpm/test_pfield_connected_ribbon_gpu.py`（本次基线一并跟踪） |
| `src/fluxvortex/warp_fsi/kernels_q16_transfer.py` | `66194d6` | 面传输（气动格 ↔ Q16 结构格）+ `dense_map` GEMV 快路径（`66194d6`，固定传输矩阵 torch-GEMV 热路径） | `tests/test_q16_work_conjugate_transfer.py`（CPU oracle + CUDA，11 用例） |

### 耦合链（强 FSI）

| 文件 | 最后提交 | 职责 | 验证状态 |
|---|---|---|---|
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py` | `bde3f20` | 强预测校正（predictor-corrector）+ Aitken 持续松弛（生产默认）+ `_IQNILS` 备选（opt-in，`bde3f20` 停靠）+ 事务原子提交（aero step transaction：全部子步成功才落盘） | `tests/test_flux_v5m_fsi_gpu_contract.py`（GPU 契约，`dc43d4c`）+ 正式运行日志（耦合残差 ~1e-10 级收敛） |

### 传递/审计层

| 文件 | 最后提交 | 职责 | 验证状态 |
|---|---|---|---|
| `src/fluxvortex/q16_work_conjugate_transfer.py` | `a0f0869`（本次基线新跟踪） | 功共轭传递（载荷/位移对偶映射的能量一致性） | `tests/test_q16_work_conjugate_transfer.py`（11 用例） |
| `src/fluxvortex/warp_fsi/q16_aero_load_packet.py` | `a0f0869`（本次基线新跟踪） | 真实气动载荷包（Q16 广义力打包，CUDA-only 门） | `tests/test_q16_aero_load_packet_gpu.py`（5 用例） |
| `src/fluxvortex/warp_fsi/q16_lev_impulse_transfer.py` | `a0f0869`（本次基线新跟踪） | LEV 条带冲量力的功共轭 CUDA 传递（source-owned impulse） | `tests/test_q16_lev_impulse_transfer_gpu.py`（5 用例） |
| `src/fluxvortex/warp_fsi/q16_mandatory_aero_mode.py` | `a0f0869`（本次基线新跟踪） | 强制气动模式（生产气动必须含 LEV + joint TEV + 自由尾迹，禁用降级路径） | `tests/test_q16_mandatory_aero_mode.py`（10 用例） |

### 案例层（论文复现合同）

| 文件 | 最后提交 | 职责 | 验证状态 |
|---|---|---|---|
| `platform/forward_flight_benchmarks/rojratsirikul2011_q16.py` | `2aae3b4` | Rojratsirikul 2011 膜翼案例：冻结参数（几何/材料/Re/Pi1/阻尼 η=0.1/时钟协议 15×30 气动格 + 5×10 结构格）、论文 PDF SHA256 校验、数字化目标 | `tests/test_rojratsirikul2011_q16_case.py`（30 用例/95 断言，全绿） |
| `platform/forward_flight_benchmarks/yamano2020_q16.py` | `a0f0869`（本次基线新跟踪） | Yamano 2020 悬臂单膜案例（single_sheet_u25_m1_ar1）+ MATLAB oracle 数据路径（csv 已跟踪；npz 见 §5 说明） | `platform/warp_vpm/test_yamano2020_q16_case.py`（16 用例，本次基线一并跟踪） |
| `platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py` | `bde3f20` | 正式 runner（`--case`/`--max-aero-steps`/`--young-modulus-override`/`--execution-gate-only`/`--structural-substeps`/`--output`） | 产出全部 §5 证据文件（E1.4/ETA0.1/VERDICT300 等） |
| `platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py` | `3d9b0ed` | Yamano 正式 runner（native FLUX-V5M 路径） | `YAMANO_ZIPPER_CHECK_STEP8.json`（步 8 误差 2.26%） |
| `platform/warp_vpm/reproduce_yamano2020_q16_fsi.py` | `a0f0869`（本次基线新跟踪） | Yamano FSI 变体 runner | 组件门 + 案例合同 |

### 测试层（tests/ 全清单 + 案例测试）

| 测试文件 | 用例数 | 门含义 | git 最后提交 |
|---|---|---|---|
| `tests/test_q16_structural_step_gpu.py` | 7 | 结构步（GPU Newmark 非线性 trial，投影共享节点壳） | `a0f0869` |
| `tests/test_q16_structural_pcg_gpu.py` | 7 | GPU PCG 求解器（Jacobi 预条件/reference_dense 路径） | `a0f0869` |
| `tests/test_q16_flux_v5m_native_gpu.py` | 9 | native GPU oracle（AIC/Mf1/Mf2 喂 MATLAB 态，位级） | `a0f0869` |
| `tests/test_q16_mandatory_aero_mode.py` | 10 | 强制气动模式（LEV + joint TEV + 自由尾迹不可降级） | `a0f0869` |
| `tests/test_q16_work_conjugate_transfer.py` | 11 | 功共轭传递（CPU oracle + CUDA 一致性） | `a0f0869` |
| `tests/test_q16_aero_load_packet_gpu.py` | 5 | 载荷包（真实气动 → Q16 广义力门） | `a0f0869` |
| `tests/test_q16_lev_impulse_transfer_gpu.py` | 5 | LEV 冲量传递（source-owned 条带冲量功共轭） | `a0f0869` |
| `tests/test_q16_ancf_element.py` | — | ANCF 单元（CPU 参考） | `a0f0869` |
| `tests/test_q16_ancf_continuum_gpu.py` | — | ANCF 连续体 GPU | `a0f0869` |
| `tests/test_q16_ancf_shared_mesh_gpu.py` | — | 共享网格 GPU | `a0f0869` |
| `tests/test_q16_ans_eas_continuum.py` | — | ANS/EAS CPU oracle | `a0f0869` |
| `tests/test_q16_ans_eas_continuum_gpu.py` | — | ANS/EAS CUDA 位级对齐 | `a0f0869` |
| `tests/test_q16_boundary_constraints_gpu.py` | — | 四边固支约束 GPU | `a0f0869` |
| `tests/test_q16_mitc16_projection.py` | — | MITC16 投影（抗锁定投影算子） | `a0f0869` |
| `tests/test_q16_ptera_resolved_transfer_gpu.py` | — | Ptera 解析传递 GPU | `a0f0869` |
| `tests/test_flux_v5m_fsi_gpu_contract.py` | — | V5M FSI GPU 契约（耦合步/事务） | `dc43d4c` |
| `tests/test_rojratsirikul2011_q16_case.py` | 30 | Rojratsirikul 案例合同（冻结参数/网格/时钟/数字化目标） | `f63da18` |
| `platform/warp_vpm/test_yamano2020_q16_case.py` | 16 | Yamano 案例合同（单膜溯源 + CUDA Q16 模态门） | `a0f0869` |

（"—" 表示计数未单列，随组件回归命令一并执行；`platform/warp_vpm/` 下其余
`test_q16_real_*`、`test_q16_incremental_*`、`test_q16_dvm_*`、`test_ldvm_*` 等
历史增量测试已随基线 `a0f0869` 全部跟踪，见提交清单。）

## 2. 验证状态总表

### 组件门（全绿）

| 门 | 测试文件 | 通过数 | 内容 |
|---|---|---|---|
| 结构步 | `tests/test_q16_structural_step_gpu.py` | 7/7 | GPU Newmark/Newton 非线性步 |
| PCG | `tests/test_q16_structural_pcg_gpu.py` | 7/7 | GPU 预条件共轭梯度 |
| native GPU oracle | `tests/test_q16_flux_v5m_native_gpu.py` | 9/9 | AIC/Mf1/Mf2 喂 MATLAB 态，位级一致 |
| 强制气动模式 | `tests/test_q16_mandatory_aero_mode.py` | 10/10 | LEV + joint TEV + 自由尾迹不可降级 |
| 功共轭/载荷包 | `tests/test_q16_work_conjugate_transfer.py` + `tests/test_q16_aero_load_packet_gpu.py` | 11/11 + 5/5 | 传递能量一致 + 真实载荷包 |
| LEV 冲量传递 | `tests/test_q16_lev_impulse_transfer_gpu.py` | 5/5 | source-owned 条带冲量 |
| 案例合同 | `tests/test_rojratsirikul2011_q16_case.py` | 30 用例/95 断言（评审简报口径 39 项合同检查） | 冻结参数/网格/时钟协议 |
| Yamano 案例 | `platform/warp_vpm/test_yamano2020_q16_case.py` | 16/16 | 溯源 + 模态门 |

### 论文精度门

| 门 | 状态 | 数值 |
|---|---|---|
| Yamano（悬臂脉冲 oracle） | 通过中带 | 步 8 端点误差 **2.26%**（`YAMANO_ZIPPER_CHECK_STEP8.json`：2.2645%，较 20260823 报告的 27.9% 已修复） |
| Rojratsirikul A16（主战场） | 近带 | camber **−1.6%**（E=1.4 分支全周期 600 步）/ **−13%**（E=2.2 主结果渐近）；Cn 落入论文带缘（无过冲）；**弦向二峰 ✓**；**全时程无符号穿越 ✓** |
| Rojratsirikul A10 / A23 | **未跑** | 泛化验证待 S1/S2 后执行（同参数，不调参） |

已知边界（诚实声明口径）：刚性板基准 **+40%** 偏差（势流方法类共性，UVLM/LDVM 类均如此）；
**St~1 振动缺失**（RC2：膜振动 rms 低于论文，锁频相关）；**慢呼吸漂移**（RC1：待 S1 阻尼带修复）。

## 3. 环境与命令（可直接复制执行）

```bash
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_DEVICE=cuda:0 FLUXV_DTYPE=float64 FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

组件回归（≈1–2 分钟，须 CUDA float64）：

```bash
pytest -q tests/test_q16_structural_step_gpu.py tests/test_q16_flux_v5m_native_gpu.py \
         tests/test_q16_mandatory_aero_mode.py tests/test_q16_work_conjugate_transfer.py \
         tests/test_q16_aero_load_packet_gpu.py
```

案例合同：

```bash
pytest -q tests/test_rojratsirikul2011_q16_case.py
```

正式运行（A16 主结果 / E1.4 材料分支）：

```bash
python platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
    --case ROJ11-A16 --max-aero-steps 600 [--young-modulus-override 1.4e6] --output <path>
```

（runner 另支持 `--execution-gate-only`（50 步执行门）与 `--structural-substeps`（仅诊断用，
冻结协议值始终随记录落盘）。）
**GPU 单租户纪律**：正式计时/精度运行须独占 GPU；witness-loop 共租时实测约 **2× 减速**。

## 4. 工作队列（按 ANALYSIS_BRIEF_v4 §4/§5）

1. **S1 — Rayleigh 带拟合阻尼**（立即）：保留 β（=现 θ 量级）+ 加 α·M，**α=7.85，β=1.77e-4**；
   Newmark 有效刚度加 γ/(βn·dt)·α·M·v 项（`q16_structural_solver.py` 的
   `mass_damping_coefficient` 钩子已预留，现值 0.0）。先验预测：慢模态 ζ→0.05（衰减快 3.3×）、
   camber 均值 0.042±0.002 不漂、Cn 末态 0.91±0.02 带缘保持。
2. **S2a — 解除远场尾迹冻结**：`wake_free_rows` 100→全自由，恢复尾迹卷起/TEV 感应反馈
   （文献要求的锁频使能项）。先验：LEV 释放计数/LESP 出现 St~1 周期调制（FFT 可辨）。
   风险：O(rows²) 成本回升，需重测速度。
3. **S2b — Ramesh 清单逐项核查 或 诚实声明路线（S3）**：间歇释放动态、非定常 Kelvin/Kutta、
   表观质量项 k≈1 量级；先验：膜振动 rms → ~0.001c。触及 oracle 验证模型，须逐项先验。
   若 S1+S2a 后仍无周期性，走 S3（P2/P3 声明为 UVLM 类边界，handoff §5.4 预授权）。
4. **A10/A23 泛化**：同参数（η、E 分支结构）跑 A10/A23，落带判定不逐工况调参；
   验收窗必须整慢模态周期（≥3 t*）+ 平衡渐近拟合双口径。
5. **工程停靠项**（不阻塞科学线）：IQN-ILS 需变量预缩放（`bde3f20` 已 opt-in 停靠）；
   warp CUDA Graph spike 残余阻断（CLAIM_TREE 工程注记）；尾迹感应 O(rows²) 复杂度。

## 5. 文档与证据索引

全部位于 `artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/`，§6 数据清单
经 `2c78bda`（gitignore 反白名单）全部 git 可见：

- `research/ANALYSIS_BRIEF_v4.md` — 现象/根因/方案/访问契约（含 §8 文件缺失问题根治）
- `research/CLAIM_TREE.md` — 证据树 N1–N6 + 2026-08-26 更新（N4 判定、E1.4 分支裁定、warp graph spike 工程注记）
- `research/SPEED_ENGINEERING_PLAN.md` — 工程加速执行记录（fused kernel/GEMV/LU/IQN-ILS）
- `diagnostics/` — 8 个诊断脚本 + README（`roj_mf2_rows.py`、`roj_mf2_zero.py`、`roj_steady_drift.py`、`roj_term_decompose.py`、`roj_warp_graph_spike.py`、`roj_weak_dp.py` 等）
- 数据：`ROJ11_A16_E1.4_T6.{json,z_history.npz}`、`ROJ11_A16_ETA0.1_T6.{json,npz}`、
  `ROJ11_A16_VERDICT300.*`、`ROJ11_A16_ETA0.1_VERDICT.*`、`ROJ11_A16_EXECUTION_GATE.*`、
  `YAMANO_ZIPPER_CHECK_STEP8.{json,log}`、`compare_exp_vs_sim_A16{,_final}.png`、7 个 `.log` 流水、
  `references/Rojratsirikul2011_JFS.pdf`（SHA256 校验）
- Yamano MATLAB oracle 数据：`platform/forward_flight_benchmarks/data/` 下
  `yamano2020_matlab_tip_dt002.csv`（已随基线跟踪）；两个 `.npz` oracle（mf1_step1/mf2_history）
  受仓库级 `*.npz` gitignore 约束未入 git，但生成脚本
  `platform/warp_vpm/extract_yamano2020_mf2_history_oracle.py` 等已跟踪，可复现。

## 6. 纪律红线

1. **不得调参拟合论文值**——任何参数改动必须给出独立于目标值的物理依据。
2. **先验预测先行**——每项修改在跑之前写下定量预测（§5 判定标准），跑完对照。
3. **禁止 toy 工况替代正式网格**——正式结论只在 15×30 气动 + 5×10 结构冻结网格上成立。
4. **求解器改动位级回归**——时间推进/线代路径任何改动须过 bit-level 回归后再谈精度。
5. **物理项改动舍入级验证**——新增物理项先用 oracle 验证到机器舍入量级再进生产。
6. **主结果不可被分支覆盖**——E=2.2 MPa、零假设（zero-damping 对照）是主结果；
   E=1.4 等材料分支只作材料不确定性解释性分支，不得反向覆盖主结论。

---

## Refactor Execution Record (2026-08-26, U0→U6 complete)

Seven vertical slices landed on the unified framework (all commits pushed):

| Slice | Commit | Content | Tests |
|---|---|---|---|
| U0-P | de1536c | SurfaceFrame/WorldOwner/Protocols/ResultStatus — bit-identical parity | 4+39 |
| U0-F | b12724b | Corrected observers: max(mean z), sign-crossing, stationarity, E1.4 label | 24 |
| U1-P/F | a47e36b | PartitionedStrongFSI + true transaction counters + GlobalTransaction | 7 |
| U2-P | 4367962 | V5MWorldState + CirculationImpulseLedger + RetentionPolicy | 9 |
| U2-F | ae60caf | Unified 3D separation owner + load-history model identity | 13+44+9 |
| U3 | 5c5eaa0 | PrescribedRigid kinematics + OneWay coupling + Baik/Yang/Izra configs | 16 |
| U4 | 09c416a | MultiSurfaceTopology + frame concatenation + Meng config | 17 |
| U5 | b5eca85 | SE(3) body + joints + moving-root skeleton + composite dynamics | 34 |
| U6 | fae510e | GeneralizedLoadPacket + J^T f + full velocity chain + free-flight FSI | 58 |

**Total: 182 tests green across 9 test files. Zero production-numerics regressions.**

New package structure:
```
src/fluxvortex/
  state/      → WorldDynamicState, WorldOwner, GlobalTransaction
  kinematics/  → SurfaceFrame, Q16/PrescribedRigid/BodyJointQ16/MultiSurface adapters
  aero/        → protocols + v5m/{state, separation, retention, topology, loads}
  dynamics/    → protocols + Q16Adapter, RigidBodySE3, Joints, Composite, LoadPacket
  coupling/    → OneWay, PartitionedStrongFSI, PartitionedFreeFlightFSI
  cases/       → Baik/Yang/Izraelevitz/Rojratsirikul/Meng configs
  runtime/     → ResultStatus (5-dim, accuracy-fail → exit 2)
  validation/  → corrected observers, gates, block stationarity
```

Physics verifications embedded in tests:
- SurfaceFrame bit-identical to production evaluate() (torch.equal, all 10 tensors)
- SE(3): angular momentum conservation, Hamilton product convention, semi-implicit
  translation x_k = a·dt²·k(k+1)/2, dt-halving convergence 2.00×
- Work conjugacy: surface power = F·v + M·ω + Q·q̇ to < 1e-12 (rigid + elastic + combined)
- Separation: unified 3D LESP owner (verified no-op in saturated regime, conflict counter)
- Free-flight: spring-loaded fixed-point converges to analytic solution within 1e-8
- Transaction: double-commit rejected, failed trial zero-pollutes owner

Key physics bug fixed during development:
- PrescribedRigidSurfaceKinematics: velocity = ω×(R·x_ref) not ω×x_ref (U3)
- BodyJointQ16: full velocity chain v_body + ω×(r) + joint_rate + elastic_vel (U6)

Next (U7, not yet started):
- Wire the multi-surface V5M solver adapter to run Meng production
- Long-time Roj A16 stationarity with corrected statistics
- A10/A23 generality (same parameters, no per-case tuning)
- Resolution convergence (5×10→7×14 Q16, 15×30→21×42 V5M)
- Performance: CUDA graph capture (warp stream issue documented), kernel fusion

### U7 partial (stepper adapter + moving root)

| Slice | Commit | Content | Tests |
|---|---|---|---|
| U7-1 | e7a377e | V5M3DStepper adapter: single-surface production parity, multi-surface topology rejection with clear error | 8 |
| U7-2 | d96571e | MovingRootBoundary + Q16CudaBoundaryConstraints.update_prescribed_values + update kernel with fail-closed validation | 9 |

Physics bugs caught and fixed during U7-2:
- root_velocities: ω×(R·p_ref) not ω×(p_abs) (finite-difference check exposed)
- update semantics: absolute-from-reference, not incremental (SE(3) inverse recovers)

Moving-root three-layer status: position ✓, velocity ✓, acceleration pending
(requires relaxing the zero-constrained-velocity check in require_kinematics);
constraint reaction routing pending (constraint_reaction is a stub).

Remaining U7 work:
- Multi-surface V5M propose (global AIC assembly, §8.6)
- Acceleration layer of moving root (§7.5)
- Constraint reaction extraction and routing to body/joint
- Roj A16 long-time stationarity with corrected statistics
- A10/A23 generality runs

### U7-3: Cross-surface AIC with mutual-induction quantification

Commit f87b628. Tests: 13 (8 GPU + 5 CPU). Total suite: 212.

Key quantitative finding: for the Meng left/right mirror pair (2×450 panels):
- Cross-influence norm ratio |A_cross|/|A_self| = 4.33%
- But joint solve vs independent solves shifts bound circulation by up to 44.7%
  of max|gamma| — the AIC's stiff diagonal amplifies the coupling, proving
  that left/right mutual induction MUST be inside one solve (plan §8.6).
- Self-blocks bit-identical to single-surface native_aic (torch.equal).
- Mirror symmetry: A_LR == A_RL, gamma_L == -gamma_R to 1.3e-16.

The cross-AIC builder is production-ready; the U7-4 work is to swap the
global AIC into the bound solve inside the propose loop.
