# 执行记录

## 2026-08-22 Mancini fast smoke 启动

- 命令：见 `PLAN.md` 的当前执行入口。
- 第一次启动在首个气动步前失败：`torch.cuda.reset_peak_memory_stats` 不接受
  `torch.device("cuda:0")`。
- 改用设备索引 `0` 后同一 API 仍报 `Invalid device argument`；没有改变或执行
  CASE 科学计算。
- 处理：停止该失败类的重试，删除可选显存峰值重置/读取。GPU 科学路径继续由
  CUDA float64 合同、LEV 粒子设备、joint-TEV 张量和 free-wake 计数验证。

## Mancini fast smoke 首次进入科学路径

- 求解推进到约 `14%` 后由 `spanwise impulse ledger does not close global
  impulse` 阻断。
- 根因类别：Q16 既有门只覆盖单侧翼；Mancini 对称全翼的左右条带冲量存在
  大量抵消，而闭合容差错误地只按抵消后的合力缩放。
- 修复：闭合误差仍不放宽为经验阈值，改用组成该归约的 free-particle 与
  bound-panel 冲量绝对值 L1 规模计算浮点前向误差界；force 门同样使用各条带
  力的 L1 规模。模型、载荷与总账定义不变。
- 空间冲量门修复后运行推进到约 `25%`，继而在时间差分 force 门停止。这里
  相邻两个大冲量相减产生小净力，因此 force 的前向误差界还必须包含
  `(abs(I_n)+abs(I_{n-1}))/dt`；已按各条带历史冲量的 L1 规模补全。
- 修复后 fast smoke `169/169` 步完成，mandatory 模式和 CUDA 计数通过；但
  `CL RMSE=572925.34`，结果为明确的科学 FAIL，禁止启动 full。保存曲线显示
  `t*=0.13` 首次超过 `|CL|=10`，并在 `t*=3.75` 达到约 `-7.25e6`。下一次
  运行只增加 bound/non-impulse 与 impulse 两个账本观测，不改变求解。
- 分量与 `Γ` 诊断证明 bound 路径失控来自全局 LEV eligibility：正式分辨率
  170 步前缀中，俯仰开始后的 42 步只有 3 步释放 LEV；其余步即使
  `LESP=10--41` 也因旧 `|LESP|<10 / same-sign` 启发式被全翼关闭。
- 修复：保留 CUDA 联合方程的逐条带 signed pin；达到启动步后每个条带独立按
  `abs(LESP)>Lcrit` 激活。高 LESP 或左右展向符号不同不再关闭 separated LEV。
- 正式前缀随后在 step 138 由 G3 pin residual 正确阻断，而非继续产生错误载荷；
  这把剩余问题定位为联合矩阵数值病态。下一诊断只在失败分支增加矩阵条件数、
  pin 误差和三组未知量尺度。
- 失败矩阵：`cond=1.035475e8`，`Gamma_bound=1.407e10`、
  `Gamma_TEV=1.535e11`、`Gamma_LEV=1.675e11`，是明确的 TEV/LEV 对消近零空间。
- 修复：使用 Kelvin 行精确消元 `Gamma_TEV=Gamma_TEpanel-Gamma_LEV`，把
  `N+2S` 联合系统凝聚为 `N+S`；Neumann、LESP pin 和恢复后的 Kelvin 合同
  保持不变，不引入正则化参数。
- Q16 两步冻结冲量相对旧联合系统的变化为 `7.21e-15` relative L2，最大分量
  绝对差 `1.05e-12`，属于求解归约顺序舍入；闭合门通过，更新了精确新哈希。

## Hirato 时间层修复与 Mancini 正式结果

- 代数消元仍在 step 138 失败，证明近零空间不是冗余未知量 alone。对照仓库
  v5f 原生合同后，改为本步联合求解 newborn LEV + Hirato pseudovortex，载荷后
  再按 `Gamma_TE,next=Gamma_bound,rear+Gamma_LEV` 提交 TE row；不再把未来 TE row
  当作本步独立未知量。
- fast smoke 完成 `169/169`：RMSE 从旧错误实现的 `572925.34` 降至
  `6.18865`，环量量级从 `1e10--1e11` 回到约 `1e-2--1e-1`，但仍未达标。
- fast full 首次越过旧 step 138 后在增长尾迹的未分块张量处 OOM。先修粒子目标
  的 bound/wake 归约，随后又发现自由尾迹节点对自由尾迹环的同类遗漏；两处统一
  改为 GPU-only `1024×1024` 分块累加。最终显存约 `2.1 GiB`、GPU 利用率
  `100%`，完整 `449/449` 步耗时 `444.66 s`。
- fast full 科学结果仍 FAIL：`RMSE=316.16453`、`MAE=117.42197`、预测峰值
  `3043.80@t*=4.70`；实验峰值 `4.83@t*=0.96`。slow smoke 同样 FAIL：
  `RMSE=4.25877`。
- 该 `smoke -> full` 恶化与既有 v5f M5 的 `Gamma_birth~1/dt` NO-GO 一致。
  停止 slow full，不再对常强度环路线做参数修补。runner 已加入冻结 reference
  RMSE 门，失败将保存证据并 `exit 2`。

## Baik W1--W4 fresh mandatory GPU 复现

- 新增 `platform/warp_vpm/reproduce_baik_v5m_mandatory.py`，不复用写死旧提交号的
  历史 summary；当前 `git_head`、runner、GPU LDVM、监控器和 GT 均写入证据哈希。
- 在 `LDVM2DCuda` 增加只读性质的尾迹推进计数；没有改变气动力数值路径。
- 四工况各执行 `3×512=1536` 步，全部检测到 separated LEV、TEV 历史和
  自由尾迹推进，单 CASE `physics_gate=true`。
- CL/CD 宏 RMSE 为 `0.4215627678 / 0.2897097942`，与冻结 reference 位级一致；
  `accuracy_gate=true`、总状态 `PASS`、进程退出码 `0`。
- 首次 GPU 证据：RTX 4090 D、CUDA 13.0、监测到利用率、峰值利用率 `56%`、
  峰值显存 `1167 MiB`。结果保存在 `results/baik/summary.json` 和四个 NPZ
  曲线文件；下节记录 source-only 修改后的最终复跑证据。

## CUDA DVM source-only 与节点涡带原语

- `LDVM2DCuda` 新增显式 `source_parity=True` 模式：GPU 直接导出新生 LEV/TEV
  强度和出生坐标、pre/post LESP、约束残差、TEV/LEV 数目；实现作者 v2.5 首步
  TEV“参与本步求解但不进入后续历史”的语义。默认 `source_parity=False`，Baik
  四条曲线的结果 SHA256 全部保持不变。
- `CudaParticleField.add_connected_ribbon_particles()` 接受 `(S+1)` 个 anchor/
  frontier 共享节点和 `S` 个 cell 环量，在 CUDA 上先做有符号边 incidence 归约，
  再按固定 `sigma` 和 target spacing 分段沉积。它不会把相邻 cell 的同一内边
  重复存成两个粒子，也不让 `sigma` 随出生边长或 `dt` 缩小。
- 新增 5 个 GPU-only 门，结果 `5 passed`：source parity/LESP 两门，uniform/
  nonuniform ribbon 和 host/overlap fail-close 三门。
- 修改 source 后再次执行 Baik 全矩阵，四条结果哈希与前次完全一致，总门仍
  `PASS`。最终 evidence 运行因 GPU 同机负载较高而耗时增加，但峰值利用率
  `100%`，进程级峰值显存 `4669 MiB`；不得将墙钟差异解释为模型变化。

## DVM node-owned 三维接入与 Mancini 修复

- 为 CUDA LDVM/source bank 增加可序列化状态；Q16 predictor fork 会保留全部
  DVM 尾迹、活动历史与节点前沿，只重绑定进程本地 CUDA stream。
- 新增展向批处理 `CudaLDVMSourceBank`，同一 CUDA kernel 批量推进 cell/node
  sectional sources；与独立 `LDVM2DCuda` lane 的最大差小于 `2e-13`。
- 接入 DVM node frontier、connected ribbon、newborn 当步 RHS、旧粒子 RHS/
  KJ 速度、Eq. 9 Ptera TE row 与真实 free wake；predictor fork 推进真实粒子与
  尾迹且不污染 parent。
- 修复三个科学合同错误：DVM 粒子改由最终物理束缚环量推进；DVM 模式禁止在
  `KJ+dGamma` 之外重复叠加自由+束缚 vortex impulse；DVM 新生事件与 Ptera
  分离边界状态分开，任一吸力判据仍超限时保留 LESP 行。
- 最后一项是精度转折点。只加入 LESP 行但用 source event 直接开关时，fast
  smoke 因 `LESP 0.11 -> 0.65 -> 0.11` 产生约 `+24/-19` 假峰，RMSE `3.07873`；
  状态拆分后假峰消失，RMSE 降至 `0.84303`，未改论文参数或阈值。
- fast full：`449/449` 步，RMSE `1.04886310`，`76312` 粒子，耗时
  `2810.92 s`，PASS。slow full：RMSE `0.22526573`，`62521` 粒子，耗时
  `1123.51 s`，PASS。两个 full 的自由尾迹均推进 `448` 次。
- 运行中设备证据为 RTX 4090 D、GPU 利用率 `100%`、总显存约
  `5.7/24.6 GiB`；没有 CPU 数值 fallback。受影响 DVM/Hirato/Q16/GPU-only
  回归最终 `41 passed`。
