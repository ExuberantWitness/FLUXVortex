# LEV+TEV 预测校正、Q16 FSI 与论文复现开发交接

更新时间：2026-08-23（Asia/Shanghai）  
接手仓库：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca`  
当前分支：`run/q16-lev-tev-pc-fsi-20260821`  
当前 HEAD：`dc43d4cc0c2290ee34df942e51cbb05e13afbb0d`  
当前主 CASE：Yamano et al. 2020，*Influence of boundary conditions on a flutter-mill*，DOI `10.1016/j.jsv.2020.115359`，工况 `single_sheet_u25_m1_ar1`

## 1. 接手目标与完成定义

目标不是继续搭建中间示例，而是在同一正式论文 CASE 中完成并验证以下唯一生产通路：

`Q16(5×3) 高阶结构壳 → 结构运动 → 原生 FLUX‑V5M(15×10) → separated LEV + joint TEV + free wake → predictor/corrector 强耦合 → Q16 结构更新`

只有同时满足以下条件，才可以称为本阶段复现完成：

1. 正式 Yamano CASE 连续运行 8 个论文采样点，退出码为 0。
2. 第 1–3 点精度不退化；第 4 点和第 8 点端点位移相对误差均不超过 5%。
3. 每个外步 FSI 正式 replay 后的耦合残差不超过 `5e-7`，Q16 非线性残差不超过 `3e-7`。
4. separated LEV 始终集成，且只在 `|LESP| > Lcrit` 时物质释放；固定 `Lcrit=0.11`，不得调参。
5. LEV、joint TEV、free wake、bound circulation 与 predictor/corrector 共用同一可回滚事务；被拒绝的 trial 对 committed 状态零污染，正式接受时只推进一次。
6. 全部科学数值张量、矩阵装配、线性代数、载荷映射和结构积分均为 CUDA float64；不允许 CPU 数值 fallback。
7. 生产运行时不得加载 Ptera，不得建立 Q4/Q9 中间结构或载荷投影。

“8 步能跑完”只代表执行门通过，不代表论文精度门通过。

## 2. 不可违反的开发约束

- 只允许 Q16；禁止 Q4、Q9、低阶 toy、简化压力板和临时小网格替代正式 CASE。
- 气动模型是原生 FLUX‑V5M，不是 Ptera。文件名中仍含 `ptera` 的历史代码只作失败诊断，不得接回生产入口。
- 不得关闭 separated LEV 来推进任何任务。低 LESP 工况中“释放数为 0”是物理阈值结果，不等于关闭 LEV。
- LEV 释放条件固定为 `abs(LESP)>0.11`；不得为了让结果变化而降低 `Lcrit`。
- joint TEV 必须满足 `Gamma_TEV = Gamma_bound_TE + Gamma_LEV`，并通过 Kelvin/联合尾缘门。
- predictor 阶段必须推进真实的 trial LEV/TEV/自由尾迹；不能用冻结尾迹或事后补尾迹冒充预测校正。
- trial 必须从同一 committed parent 克隆；只有 formal replay 达到耦合容差后才能原子提交气动和结构状态。
- 不得经验缩放气动力、Mf1/Mf2、结构刚度、阻尼或位移来追论文曲线。
- 不得更改论文网格、材料、时间步、脉冲、误差定义或容差来隐藏失败。
- 用户要求用论文 CASE 快速暴露问题。遇到失败先读取 `.partial.json` 的首个失败外步/子步，再修底层契约；不要绕回与论文复现无关的长期探索。

## 3. 冻结论文工况

科学坐标：`+x` 为弦向/来流下游，`+y` 为展向，`+z` 为厚度方向。

| 参数 | 冻结值 |
|---|---:|
| 板长（弦向） | 1.0 m |
| 板宽（展向） | 1.0 m |
| 厚度 | 1.0e-3 m |
| 泊松比 | 0.3 |
| 降低速度 U* | 25 |
| 质量比 M* | 1 |
| 来流速度 | 10 m/s |
| 流体密度 | 1.225 kg/m³ |
| 结构无量纲时间步 | `dt*=0.002` |
| 气动外步 | `34×dt*=0.068` |
| 外加半正弦脉冲 | `0≤t*<0.2`，峰值 0.5 |
| Q16 网格 | 弦向 5 × 展向 3 个宏单元 |
| FLUX‑V5M 网格 | 弦向 15 × 展向 10 个面板 |
| Lcrit | 0.11 |

参数真源：`platform/forward_flight_benchmarks/yamano2020_q16.py`。  
外加载荷真源：`platform/warp_vpm/yamano2020_q16_pulse.py`。  
作者开源 MATLAB 程序：`FSI_by_FEM_and_UVLM/single_sheet/`。

注意：历史 `single_sheet/save/tip_displacement.csv` 使用了错误的 3-DOF 索引，不能作为位移 oracle。当前正确位移真值来自 `platform/forward_flight_benchmarks/data/yamano2020_matlab_tip_dt002.csv`，并在适配器中固定 SHA256。

## 4. 当前生产架构与事务语义

### 4.1 原生 FLUX‑V5M

主文件：`src/fluxvortex/warp_fsi/q16_flux_v5m_native.py`

- `Q16NativeV5MSurface`：从 Q16 高阶表面直接插值得到 1/4 弦环涡、面板配置点、前缘、尾缘及其速度；不存在 Q4/Q9 中间自由度。
- `Q16NativeV5MSolver.propose()`：从 committed `NativeV5MState.clone()` 开始，推进已有自由尾迹/粒子，计算 AIC、Neumann RHS、预分离 LESP、DVM 源、LEV 释放与约束、joint TEV、Mf2 历史、面板压力和 Q16 广义力。
- `Q16NativeV5MOwner.commit()`：校验 parent digest 和步数，只允许一次 `step+1` 提交。
- `NativeV5MState.digest()`：覆盖 bound gamma、wake rings/gamma、LEV source bank、particle field 等事务状态，用于检测 trial 污染。

LEV/TEV 的当前正式逻辑位于 `Q16NativeV5MSolver.propose()`：

1. 先解未约束 `gamma_pre` 并计算三维条带 `LESP_pre`。
2. `surface_separated = abs(LESP_pre) > 0.11`。
3. DVM source bank 在每个 trial 中真实推进；满足释放条件时向 particle field 写入新生 LEV。
4. 对活动前缘行施加 LESP pin 后重解 bound circulation。
5. 联合尾缘强制 `gamma_tev = gamma_bound_TE + gamma_lev`。
6. trial 中推进自由尾迹并追加新的 TEV wake row；若 trial 被拒绝，整个克隆状态随 proposal 丢弃。

### 4.2 Q16 结构与载荷传递

关键文件：

- `src/fluxvortex/q16_ancf_mesh.py`
- `src/fluxvortex/q16_ans_eas_continuum.py`
- `src/fluxvortex/warp_fsi/q16_structural_solver.py`
- `src/fluxvortex/warp_fsi/kernels_q16_*.py`
- `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py`
- `src/fluxvortex/warp_fsi/kernels_q16_transfer.py`

当前结构为共享节点 Q16 MITC16/ANS/EAS 宏壳，5×3 正式网格。面板压力和力经 Q16 高阶形函数转置直接形成广义力；力、矩、虚功均有独立门。Q16 Newmark/Newton 在 CUDA float64 上执行；`reference_dense` 是 GPU 上的缓存参考切线策略，不是 CPU 求解回退。Python 只负责拓扑和事务控制。

### 4.3 强 predictor/corrector FSI

主文件：`src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py`

每个外步的真实顺序为：

1. 保存 committed 结构 `q/v/a`、气动 digest 和 generation。
2. 由 committed 结构做运动预测。
3. 每次耦合迭代都调用 `owner.aerodynamic.propose(...)`，因此 LEV、TEV、自由尾迹随 trial 结构运动真实推进。
4. 用 proposal 的端点载荷在 34 个结构子步中进行 Q16 replay；`Mf1` 作为加速度作用项，`Mf2/lift2` 随实时结构速度更新。
5. 用结构状态/速度相对差计算强耦合残差，并以 Aitken 更新下一次 trial 运动。
6. 普通 trial 收敛后，再做一次 formal aerodynamic proposal 和 formal structural replay。
7. formal replay 仍满足 `5e-7` 后，先准备结构状态，再调用气动 `commit()`，最后原子更新 Q16 owner；generation 只增加 1。

这条链已经具备“LEV 和 TEV 一起进行预测校正”的必要事务条件。后续不得退回只预测结构、尾迹最后补推进的旧架构。

## 5. 2026-08-23 新修复的作者时钟 bug

作者冻结 Mf2 fixture 和 MATLAB 主循环证明：

- 启动气动事务位于 `t*=0.002`；
- 正常气动提交位于 `t*=0.070, 0.138, 0.206, 0.274, ...`；
- 论文比较点 `t*=0.068, 0.136, 0.204, 0.272, ...` 是相应 34 子步 replay 的第 33 子步状态；第 34 子步才是事务端点。

旧入口在 `0.068/0.136/...` 提前提交尾迹，导致论文采样时 wake/流体历史错一拍。当前未提交修改已实现：

- `Q16NativeStructuralCheckpoint`：允许从正式 replay 保存指定结构子步。
- `advance(..., load_betas, checkpoint_substep, author_startup)`：支持作者启动事务及显式子步载荷相位。
- 正式入口先执行 `t*=0.002` 启动事务，载荷 beta 为 `1/34`。
- 常规外步载荷 beta 为 `1/34...34/34`。
- 论文位移读取 `checkpoint_substep=33`，事务仍在第 34 子步完成并提交。

第一论文点的 wake ring 数由旧时钟的 10 改为 20，和作者事务步一致。必须保留该修复，并补一条专门的作者时钟回归测试；不能因后段误差仍高而回退正确时钟。

## 6. 当前证据与真实精度状态

### 6.1 已通过的硬门

载荷门文件：`artifacts/experiment/20260823_q16_fluxv5m_native_fsi/YAMANO_Q16_NATIVE_LOAD_GATES.json`

| 门 | 当前结果 | 判定 |
|---|---:|---|
| AIC 相对误差 | `3.0073528848739746e-7`，阈值 `1e-6` | 通过 |
| Mf2 step1–4 逐面板最大绝对误差 | `3.8163916471489756e-17`，阈值 `1e-13` | 通过 |
| Qf 合力最大相对误差 | `0.0028941429731128726`，阈值 `0.005` | 通过 |
| Q16 Mf1 共同场厚度向功 | `-0.02585925156893745`，作者 `-0.04127802486316853` | 物理符号通过 |
| Mf1 跨 9-DOF/6-DOF 共同场差异 | 37.35% | 仅 warning，不得冒充离散无关硬门 |

此前已验证原生气动和结构 GPU 核心测试共 16 项通过：

```bash
PYTHONPATH=src:platform/warp_vpm:platform \
pytest -q tests/test_q16_structural_step_gpu.py \
  tests/test_q16_flux_v5m_native_gpu.py
```

### 6.2 旧作者载荷时钟结果（只作对照）

文件：`YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_LOAD.json`

- 8 步完整执行，764.80 s。
- 误差：1.425%、1.757%、3.685%、7.633%、11.225%、18.656%、26.385%、27.918%。
- 所有耦合残差低于 `2e-8`。
- 最大 `|LESP|=0.01787<0.11`，因此释放数为 0；LEV 机制仍处于活动数据通路。

### 6.3 正确作者时钟 8 点最终结果

最终文件：`YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_CLOCK.json`  
状态：`completed`，退出码 0，总耗时 788.19 s，设备 NVIDIA GeForce RTX 4090 D，CUDA float64，旧运行时模块计数 0。

| 论文点 | t* | 当前位移/m | 作者位移/m | 相对误差 | wake rings |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.068 | 0.000385805 | 0.000390455 | 1.191% | 20 |
| 2 | 0.136 | 0.002644888 | 0.002591999 | 2.040% | 30 |
| 3 | 0.204 | 0.006886789 | 0.006621754 | 4.002% | 40 |
| 4 | 0.272 | 0.012027020 | 0.011139383 | 7.968% | 50 |
| 5 | 0.340 | 0.017590481 | 0.015762990 | 11.594% | 60 |
| 6 | 0.408 | 0.024088239 | 0.020237335 | 19.029% | 70 |
| 7 | 0.476 | 0.031414875 | 0.024779414 | 26.778% | 80 |
| 8 | 0.544 | 0.037679175 | 0.029357551 | 28.346% | 90 |

所有论文点 sample 均来自结构子步 33，事务端点为子步 34；启动事务位于 `t*=0.002`。第 8 点使用 5 次流固校正、6 次气动求值，formal replay 耦合残差为 `1.58e-9`。全程最大 `|LESP|=0.01782<0.11`，LEV 物质释放为 0，但 separated LEV 仍在每次 trial 内参与求解。

与旧时钟相比，正确时钟第 1 点误差从 1.425% 降至 1.191%，第 2–8 点分别增加约 0.28、0.32、0.34、0.37、0.37、0.39、0.43 个百分点。这说明时钟修复是真实 bug 修复，但不是后段偏差的唯一根因。

接手后先读取现有最终文件，不要直接重跑：

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca

jq '{status,completed_outer_steps,requested_outer_steps,final_output}' \
  artifacts/experiment/20260823_q16_fluxv5m_native_fsi/\
YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_CLOCK.partial.json

jq '{status,elapsed_seconds,records:[.records[]|\
{step:.outer_step,t:.time_star,error:.tip_relative_error_percent,\
residual:.coupling_residual,lesp:.aerodynamic.lesp_pre_max_abs,\
wake:.aerodynamic.wake_ring_count}]}' \
  artifacts/experiment/20260823_q16_fluxv5m_native_fsi/\
YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_CLOCK.json
```

只有最终 JSON 丢失或校验失败时才从头运行；当前入口尚未实现跨进程 resume，不能拼接不同进程的检查点。

## 7. 代码与证据索引

### 7.1 生产代码

| 对象 | 文件 |
|---|---|
| 正式 CASE 入口 | `platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py` |
| 冻结论文参数/Q16 建模/真值 | `platform/forward_flight_benchmarks/yamano2020_q16.py` |
| 正式 CUDA 脉冲 | `platform/warp_vpm/yamano2020_q16_pulse.py` |
| 原生 V5M、LEV/TEV/wake transaction | `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` |
| Q16↔V5M 强预测校正 FSI | `src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py` |
| 作者气动力分项/Mf1/Mf2 | `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py` |
| Q16 Newmark/Newton | `src/fluxvortex/warp_fsi/q16_structural_solver.py` |
| Q16 网格/宏壳 | `src/fluxvortex/q16_ancf_mesh.py`、`src/fluxvortex/q16_ans_eas_continuum.py` |
| Q16 CUDA 核 | `src/fluxvortex/warp_fsi/kernels_q16_*.py` |
| 载荷门生成 | `platform/warp_vpm/verify_yamano2020_q16_native_load_gates.py` |

### 7.2 首要测试

| 合同 | 测试 |
|---|---|
| 原生 AIC/Mf2/Qf/Mf1、事务、Lcrit、direct Q16 transfer、CUDA64 | `tests/test_q16_flux_v5m_native_gpu.py` |
| Q16 Newmark、非线性残差、事务失败清洁性、CUDA 所有权 | `tests/test_q16_structural_step_gpu.py` |
| 强制 separated LEV + joint TEV + free wake 模式 | `tests/test_q16_mandatory_aero_mode.py` |
| Q16 虚功/力/矩 | `tests/test_q16_work_conjugate_transfer.py` |
| 气动载荷包拒绝未解析 LEV 冲量 | `tests/test_q16_aero_load_packet_gpu.py` |
| LEV 冲量到 Q16 | `tests/test_q16_lev_impulse_transfer_gpu.py` |

包含 `ptera`、Q4 迁移路径的旧测试只可用于历史诊断，不能作为当前生产路线的通过证据。

### 7.3 报告与计划

- `artifacts/experiment/20260823_q16_fluxv5m_native_fsi/PLAN.md`
- `artifacts/experiment/20260823_q16_fluxv5m_native_fsi/CHECKLIST.md`
- `artifacts/experiment/20260823_q16_fluxv5m_native_fsi/YAMANO_Q16_NATIVE_CASE_REPORT_20260823.md`
- `artifacts/experiment/20260823_q16_fluxv5m_native_fsi/LOAD_GATE_DIAGNOSIS_20260823.md`

这些文件的旧 8 步表尚使用修复前时钟。接手者应在正确时钟 8 点运行完成后更新它们，不能并列保留两个“当前结果”。

## 8. 接手后的执行顺序

### P0：收口正确时钟证据和回归

1. 以 `YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_CLOCK.json` 为新的唯一轨迹基线。
2. 更新 `PLAN.md`、`CHECKLIST.md` 和正式报告中的旧时钟结果。
3. 给作者时钟补回归：至少锁定启动 `t*=0.002`、论文 sample 子步 33、事务端点子步 34，以及首个论文点 wake rings=20。

### P1：用正式 CASE 定位后段首个科学偏差

当前证据已经排除以下首要解释：AIC 方向错误、Mf2 step1–4 错误、Qf 合力严重错、耦合未收敛、初始结构刚度/质量完全错误。不要重新循环审这些已通过项。

在同一正式轨迹中做二选一判别：

1. **气动力判别**：在作者保存的 15×10 几何/速度/尾迹状态上，逐外步比较原生 FLUX‑V5M 的 `dp_lift1`、Mf2、速度相关 lift2、面板压力、合力和总功。首个偏离分项决定修复对象。
2. **结构判别**：若气动力同轨迹通过，把同一作者总广义载荷历史直接施加到 Q16，比较每个论文点的内能、恢复力功、动能和端点位移。若 `t*>0.2` 后仍系统性超调，修 Q16 恢复力/时间积分。

不要再尝试已经否决的 Mf1 扩支持经验修复：宏单元局部支持给出共同场功 `-0.06879`，完全全局装配为 `-0.15385`，均比当前路线更差，且没有独立科学依据。

MATLAB `dump_traj_long(0.55)` 在本机尝试时因 MathWorks Licensing Error 9 / `-9.2` 失败。它只是离线作者 oracle，不是生产求解器。优先使用已经冻结的 `.npz/.csv/.mat` 作者 fixture；不要让 MATLAB 许可证阻塞正式 CASE。若确实缺少某个同轨迹分项，再在有许可证环境补导出。

### P2：只修首个独立 oracle 失败的根因

- 若面板级压力/功先偏离：修复对应的 FLUX‑V5M 历史项或子步相位，并添加该外步的面板级回归。
- 若作者载荷下 Q16 先偏离：修复 Q16 恢复力、质量/约束或 Newmark 载荷时序，并添加能量/恢复力回归。
- 若两者单独都通过、耦合后才偏离：检查 formal proposal/replay 使用的端点运动与 Aitken 残差定义，尤其是气动载荷端点和结构 sample/commit 时刻是否混用。
- 修复必须保持所有既有科学门，不得通过放宽容差或重定义误差获得“通过”。

### P3：按 1→4→8 点回归

每个候选修复只跑同一正式入口：

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca

PYTHONUNBUFFERED=1 PYTHONPATH=src:platform/warp_vpm:platform \
python platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py \
  --outer-steps 1 \
  --output artifacts/experiment/20260823_q16_fluxv5m_native_fsi/\
YAMANO_Q16_5X3_V5M_15X10_STEP1_CANDIDATE.json
```

1 点通过后改为 4 点；4 点误差低于 5% 后才跑 8 点。8 点命令只改 `--outer-steps` 和输出名。不得另建 toy。

## 9. 验收节点

| 节点 | 必须满足的证据 | 状态 |
|---|---|---|
| H0 依赖纯度 | 正式进程 `runtime_legacy_module_count=0`；生产导入不含 Ptera/Q4/Q9 | 已通过 |
| H1 GPU 所有权 | 全部科学状态 CUDA float64，GPU 不可用时 fail-fast | 已通过 |
| H2 Q16 结构 | 5×3 Q16 MITC16/ANS/EAS；无 Q4/Q9 中间拓扑 | 已通过 |
| H3 传递合同 | Q16 直接映射的力、矩、虚功闭合 | 已通过 |
| H4 LEV/TEV/wake 事务 | 重复 trial 确定性、parent digest 不变、commit 仅推进一次 | 已通过 |
| H5 释放合同 | `≤0.11` 不释放，`>0.11` 释放且 LESP pin 通过 | 已通过 |
| H6 作者时钟 | 启动 0.002；sample=33；commit=34；首点 wake=20 | 实现通过，缺专门回归 |
| H7 执行门 | 正确作者时钟连续 8 点退出 0 | 已通过，788.19 s |
| H8 中程精度 | 第 4 点误差 ≤5% | 未通过，当前 7.968% |
| H9 长程精度 | 第 8 点误差 ≤5% | 未通过，当前 28.346% |
| H10 报告一致性 | PLAN/CHECKLIST/报告只引用正确时钟当前证据 | 待更新 |

## 10. 失败处理规则

### 如果程序非零退出

1. 先读同名 `.partial.json` 的 `failed_outer_step`、`error_type`、`progress_records`。
2. 判断是结构子步非收敛、耦合非收敛、事务 digest、GPU 所有权还是物理门失败。
3. 只复现该正式外步的同一路径，不新建小网格。
4. 数值仍单调收敛时可改进求解策略，但不得改 `3e-7/5e-7` 验收容差、材料或载荷。

### 如果程序完成但误差超门

这不是“运行成功即可接受”，也不是先调参。必须找首个分项/能量 oracle 偏离。当前第 3 点为 4.002%，第 4 点跳到 7.968%，因此首要观察窗是脉冲在 `t*=0.2` 结束后的第 3→4 外步，而不是第 8 点终值本身。

### 如果 LEV 释放数为 0

同时检查 `separated_lev_mandatory=true`、`lesp_release_condition=abs(LESP)>0.11` 和 `lesp_pre_max_abs`。本 CASE 前 8 点 LESP 约 0.018，物理上不应释放；禁止通过降低 Lcrit 制造 LEV。

## 11. 工作树与提交注意事项

当前工作树非常脏，包含用户和此前多轮 agent 的大量修改/未跟踪文件。不要执行 `git reset --hard`、`git checkout -- .`、批量清理或把所有文件一次性提交。

尤其注意：本次作者时钟修改和大量 Q16 原生文件尚未进入 HEAD；接手前先执行：

```bash
git status --short --branch
git diff -- src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py \
  platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py
git diff --check
```

若需提交，只暂存本任务明确核验过的文件和证据，先确认没有夹带旧 Ptera/toy 路线；未经用户明确要求不要 push。

## 12. 接手者最终汇报格式

最终不要只说“已修复”或“全部通过”，必须给出：

1. 正确作者时钟 1–8 点误差表及相对旧结果变化。
2. 第一个独立 oracle 失败位置、根因和对应代码修改。
3. LEV/TEV/wake predictor/corrector 事务证据：trial 无污染、formal commit 一次。
4. Q16 直接载荷的力/矩/虚功证据。
5. GPU 设备、dtype、无 CPU fallback、无 Ptera/Q4/Q9 运行时证据。
6. H0–H10 哪些通过、哪些仍失败；不得把未通过的精度门写成“复现完成”。

当前最重要的一句话：**生产架构和事务条件已经成立，当前真正未闭合的是 Yamano 正确作者时钟下第 4–8 点的论文位移精度；下一位 agent 应从正式 CASE 的首个同轨迹分项/能量偏离处修根因，而不是再造中间模型。**
