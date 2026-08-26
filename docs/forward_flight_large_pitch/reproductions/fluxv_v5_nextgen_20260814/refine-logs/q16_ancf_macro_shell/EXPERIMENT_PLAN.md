# FLUX-V5M Q16 高阶剪切可变形 ANCF 宏壳：开发与验证计划

**问题**：现有四节点 Kirchhoff–Love ANCF 壳能描述薄翼全局变形，但其 4 节点/36 DOF 单元、统一厚度和按单元缩放 `E/rho` 的设计面，不能承担早期机翼拓扑的高阶刚度—质量分布优化，也没有完整覆盖横向剪切、厚度伸缩和真实分离流 FSI。

**方法主张**：采用唯一的 Q16（4×4 节点）三次插值、剪切与厚度可变形 ANCF 宏壳；在 CUDA float64 上以矩阵自由方式求解，并通过功共轭气动—结构传递，与始终开启的 separated LEV、真实自由尾迹和事务化 predictor–corrector 耦合；同一 Q16 节点场同时承载运动学与等效拓扑材料变量。

**日期**：2026-08-21

## 0. 决策冻结

### 0.1 唯一生产单元

- 单元：`Q16ShearDeformableANCFShellElement`。
- 面内节点：4×4 Gauss–Lobatto 节点，坐标 `{-1, -1/sqrt(5), +1/sqrt(5), +1}` 的张量积。
- 插值：固定三次 Lagrange 张量积，不实现通用 p-family。
- 每节点状态：`r ∈ R³` 与厚度梯度/导演向量 `g ∈ R³`，共 6 DOF。
- 每单元状态：16×6 = 96 DOF。
- 三维退化壳映射：

  `x(ξ,η,ζ) = Σ_a N_a(ξ,η) [r_a + ζ g_a]`, `ζ ∈ [-1,+1]`。

  参考态 `|2g_a|` 给出局部物理厚度；`g` 的转动、剪切和长度变化分别提供大转动、横向剪切与厚度伸缩能力。
- 应变：基于上述映射的完整 Green–Lagrange 六分量；不得用小转角替代生产路径。
- 积分：生产默认面内 6×6、厚度 3 点；8×8×4 只作为同一 Q16 方程的过积分敏感性检查。
- 锁死控制：对横向剪切与厚度法向应变使用固定的 ANS/EAS 投影；投影空间和 tying points 在 H0 冻结，之后不得按工况切换。

### 0.2 明确删除 Q9 路线

- 不实现 Q9 单元、Q9 网格、Q9 oracle、Q9 配置、Q9 测试或 Q9→Q16 升级器。
- 不建立可在 Q4/Q9/Q16 间运行时切换的通用单元模板。
- 所有新数学测试从 Q16 开始；独立 oracle 也必须重算 Q16 的 96-DOF 方程。
- 历史 Q4 代码只作为不可修改的旧结果隔离基线，不得成为新 FSI、性能或 co-design 证据。

### 0.3 气动和执行硬约束

- separated LEV 是生产模型不可关闭的一部分；Q16 FSI 配置中不暴露 `enable_lev=False`。
- 真实自由尾迹不可关闭；attached-only、prescribed-wake、wake-off 仅可存在于历史测试，不能进入 Q16 生产入口。
- predictor 迭代只能读取同一份 pre-step LEV/TEV/free-wake 快照；结构—气动收敛后，真实尾迹恰好提交一次。
- 任何结构、载荷、LEV、尾迹或收敛门失败均不得推进结构状态或真实尾迹；clean retry 必须与 fresh run 一致。
- 科学数据面必须为 CUDA float64。CPU 仅允许配置、I/O 和独立离线 oracle；生产时间循环禁止 `.numpy()`、隐式上传、CPU 力/矩阵装配及 CPU fallback。

### 0.4 设计语义

Q16 宏单元不显式画出蒙皮、梁、肋。每个 Q16 设计节点存储同构的宏材料变量，连续插值得到等效结构场：

- `h`：有效厚度；
- `φ_skin, φ_span, φ_chord, φ_core`：四类周期性/均匀化拓扑基的非负体积分数；
- `θ`：主正交方向；
- 可选阻尼参数仅在结构验证完成后解锁。

通过固定、可审计的均匀化映射生成正定的 `C_eff`、质量密度 `rho_eff` 和转动惯量。禁止继续把单元标量 `E_scale` 同时线性缩放膜与弯曲刚度，并把它称为“厚度优化”。真实厚度必须同时影响几何、质量、剪切、弯曲和厚度法向响应。

## 1. 当前基线与必须替换的路径

| 现状 | 证据位置 | Q16 处理 |
|---|---|---|
| 四节点、9 DOF/节点、36 DOF/单元、Kirchhoff–Love | `src/fluxvortex/ancf_shell.py` | 新建 Q16 模块；不原地扩写旧单元 |
| `set_distribution()` 只按单元缩放 `Dm/Dk/rho` | `src/fluxvortex/ancf_shell.py` | 改为 Q16 节点宏材料场与物理厚度/均匀化映射 |
| Warp 内核固定 36 DOF 和全局统一 `Dm/h` | `src/fluxvortex/warp_fsi/kernels_ancf.py` | 新建固定 96-DOF、矩阵自由 Q16 CUDA 内核 |
| 生产 ANCF 载荷传递已有形函数 Gauss 投影 | `src/fluxvortex/standalone_hybrid_solver.py` | 推广为任意气动点→Q16 表面的 Jacobian 转置投影 |
| 部分 differentiable 路径仍把面板力四等分到位置 DOF | `platform/diff_coupled_fsi.py`、`platform/diff_coupled_unsteady_gpu.py` | 删除 Q16 入口中的四等分路径，统一功共轭传递 |
| 某些 unsteady GPU 循环仍有 `.numpy()` 往返 | `platform/diff_coupled_unsteady_gpu.py` | Q16 正式循环完全设备驻留 |

## 2. 主张图

| 主张 | 为什么重要 | 最低可信证据 | 关联实验块 |
|---|---|---|---|
| C1：Q16 高阶剪切/厚度可变形 ANCF 宏壳能在法向气动载荷和大变形下给出收敛、无锁死且守恒的 GPU FSI | 解决当前四节点拓扑分辨率不足，并覆盖薄翼到中等厚度/夹芯等效翼的全局响应 | Q16 数学恒等式、法向压力/剪切/大转动基准、网格与积分收敛、GPU-oracle 一致、功共轭与能量门全部通过 | B1、B2、B3 |
| C2：Q16 节点宏材料场能在相同质量和同一完整 separated-LEV/free-wake FSI 下产生可信的刚度—质量拓扑 Pareto 改善 | 证明新增高阶设计自由度不是装饰，也不依赖显式肋位或关闭困难物理 | 相同 Q16 求解器下，二维宏场优于均匀与仅展向分布基线；梯度、约束、复算和多初值一致 | B4、B5 |

**反主张必须排除**：改善只是更多 DOF、关闭 LEV/尾迹、使用四等分载荷、改变质量、改变气动网格、CPU 后处理改分数，或用互不物理关联的刚度/质量标量作弊。

## 3. 计算架构

### 3.1 Q16 结构核

- 新模块建议：`src/fluxvortex/q16_ancf_shell.py`。
- 新 CUDA 模块建议：`src/fluxvortex/warp_fsi/kernels_q16_ancf.py`。
- 数组尺寸在编译期固定：16 nodes、96 element DOF、36×3 quadrature points。
- 每个 `(batch, element, quadrature)` 线程块计算运动学、应变、材料和内力局部贡献。
- 不装配 96×96 全局刚度块；提供 `internal_force(q,z)` 与一致的 `Jv(q,z,v)`。
- 动力学采用非线性 Newmark/generalized-alpha + Newton–Krylov；有效算子 `M + βΔt²K_t` 使用矩阵自由 PCG，在失去正定时 fail-closed，禁止静默改用 CPU 稠密解。
- 预条件器：GPU 上的节点/单元块 Jacobi；必须记录迭代数、最终残差和失败坐标。
- 反向路径：先实现同一算子的隐式 VJP，再解锁 co-design；不得维护与 forward 不同的简化结构方程。

### 3.2 锁死控制（2026-08-21 文献核验修订）

- Q16 原始运动学保持不变。
- 采用 Bucalem--Bathe MITC16 的固定协变 tying：`E11/E13` 用 3×4、
  `E22/E23` 用对称 4×3、`E12` 用 3×3；横向剪切在中面取样。
- `E33` 另走 Q16 节点 ANS，并叠加厚度线性 EAS 内部参数；EAS 必须满足
  Hu--Washizu 正交/patch 条件并逐单元静态凝聚。当前未实现，不得把 compatible
  `E33` 标为 ANS/EAS。
- 不采用 Q9、选择性降阶积分或按厚度比自动切换公式，避免零能模态和不可导分支。
- 验证覆盖 `h/L = 1e-3, 1e-2, 5e-2, 1e-1`；最后一档只验证全局响应，不宣称层间应力或局部屈曲精度。

### 3.3 功共轭气动—结构传递

- 气动表面位置由 Q16 映射在气动顶点/积分点直接评价，不把 ANCF 节点硬等同于 VLM 网格角点。
- 对任意气动力 `f_a`，广义力必须由 `Q_s = J_x(q)^T f_a` 得到。
- 若载荷作用在上下表面而非中面，Jacobian 必须包含导演向量 DOF 的力臂贡献。
- 生产门同时检查：总力、总矩、瞬时功与虚功。
- 所有 differentiable、predictor 和正式 runner 共用同一个传递算子及其 VJP。

### 3.4 separated LEV / 自由尾迹事务

每个物理步固定为：

1. 冻结 `q_n,dq_n,LEV_n,TEV_n,wake_n` 和它们的 hash；
2. predictor–corrector 在只读气动父状态上反复评价 trial geometry；
3. 结构和气动残差同时过门；
4. 以收敛几何重新计算一次正式载荷；
5. 原子提交 `q_{n+1},dq_{n+1}`、LEV/TEV 释放和自由尾迹对流；
6. 记录 before/after hash、调用账本、提交次数和功交换；
7. 任一失败回滚整个物理步，禁止“结构未提交但尾迹已推进”。

## 4. 实验块

### B1：Q16 数学与单元真实性

- **主张**：96-DOF Q16 方程实现正确，没有低阶 toy 依赖。
- **任务**：单单元和 2×2 Q16 网格；随机扭曲参考面；各向同性、正交各向异性和宏材料混合。
- **指标**：
  - partition of unity、Kronecker 性、导数和为零；
  - 三次位移场再现；
  - 刚体平移/转动内力与应变能；
  - 常膜、纯弯、横向剪切、厚度压缩 patch；
  - 能量方向导数=`Q_int·δq`；`Jv` 对中心差分；
  - 一致质量、总质量与刚体惯量。
- **成功门**：解析恒等式误差 `≤64 eps`（按尺度归一）；force/energy 方向导数相对误差 `≤1e-7`；`Jv` 相对误差 `≤5e-6`；所有错误输入在分配大数组或启动 CUDA 核前拒绝。
- **失败解释**：任何一项失败都停止 FSI 接入；不得通过放宽到工程误差掩盖单元公式错误。
- **优先级**：MUST-RUN。

### B2：法向载荷、剪切与大变形结构验证

- **主张**：Q16 在厚度方向气动压力下不锁死，并能覆盖目标翼的全局非线性响应。
- **算例**：
  - 简支/固支方板均布法向压力；
  - 厚/薄悬臂板端部法向剪力与分布压力；
  - 扭曲悬臂、复合方向刚度和大转动；
  - 已有 Yamano/Pazy-class 几何只在输入 provenance 完整时作为系统级结构基准。
- **比较**：解析解或同一连续体的高精度参考；Q16 的 1×1、2×2、4×4 h-refinement；6×6×3 与 8×8×4 积分。无 Q9。
- **指标**：位移、转角、应变能、反力、前六阶频率、厚度/剪切能占比、网格/积分收敛率。
- **成功门**：最终网格关键位移/能量对 reference `≤2%`；前六阶频率 `≤3%`；薄极限无剪切锁死趋势；过积分变化 `<0.5%`；大转动过程中能量和状态 finite。
- **边界**：不据此声明层间 `σ_zz`、脱层、连接区应力或局部屈曲精度。
- **优先级**：MUST-RUN。

### B3：GPU、载荷传递与完整 FSI 事务

- **主张**：Q16 不是“GPU 上能跑”，而是设备驻留、可并行扩展并与完整 separated-LEV/free-wake FSI 守恒耦合。
- **比较**：同一 Q16 NumPy 独立 oracle（仅小算例）与 CUDA；旧四等分传递仅作应当失败的负控。
- **指标**：GPU/CPU oracle 的内力、质量作用、Jv、一步 Newmark；虚功、总力、总矩；predictor 残差；LEV/TEV/wake 调用账本；Nsight kernel/memory 时间；batch/mesh scaling。
- **成功门**：
  - float64 GPU–oracle 相对误差 `≤1e-10`；
  - 虚功、总力、总矩归一残差各 `≤1e-11`；
  - separated case 的 LEV 释放/反馈计数非零；每物理步 wake commit 恰好 1；
  - predictor 失败时结构/LEV/TEV/wake hash 全不变，clean retry 与 fresh run 一致；
  - 正式时间循环无 `.numpy()`、host solve、隐式 CPU tensor 上传和 CPU fallback；
  - batch≥32 时 GPU 相对独立 CPU oracle 加速 `≥10×`，且 4 倍 batch 的时间增长显著低于 4 倍；若未达到，先 profile，不得降阶或关闭 LEV。
- **优先级**：MUST-RUN。

### B4：系统级柔性翼验证

- **主张**：结构升级不破坏已验证气动，并在柔性响应中给出时间/空间/耦合收敛。
- **工况**：零变形刚性极限、规定法向压力、阵风、俯仰/升沉柔性翼，以及冻结后的一个经典实验 FSI case。
- **统一要求**：所有正式工况使用相同 Q16、相同传递、separated LEV=mandatory、free wake=mandatory；不允许 attached-only 对照进入结果表。
- **指标**：`CL/CD/CM`、翼尖挠度/扭转、频率、峰值根部弯矩、FSI 功交换、PC 迭代数、时间步/气动网格/Q16 网格收敛。
- **成功门**：
  - 刚性极限与同网格 FLUX-V5M 气动在预注册浮点容差内一致；
  - 两级加密后 headline 气动和结构量变化 `<3%`；
  - FSI 每步残差 `<1e-8`，无未登记步长缩小；
  - 实验 case 的几何、材料、运动和测量不确定度先冻结，再计算误差，禁止事后调参。
- **优先级**：MUST-RUN。

### B5：Q16 宏拓扑刚度—质量 co-design

- **主张**：高阶二维宏场在相同质量下比均匀或仅展向分布更好，且改善来自真实 FSI。
- **只比较三类系统**：
  1. Q16 均匀宏材料；
  2. Q16 仅展向变化（限制设计变量，而非换单元）；
  3. Q16 完整弦向×展向宏材料场。
- **设计约束**：总质量严格相同；`h_min/h_max`、最小体积分数、SPD、本征频率、最大应变/挠度、场梯度和制造滤波均固定；所有候选使用同一气动和相同收敛预算。
- **目标**：阵风峰值根部载荷、翼尖响应、周期平均气动效率和控制功；输出 Pareto 面而非单一加权分数。
- **梯度门**：8 个随机方向的端到端方向导数对中心差分，相对误差中位数 `≤1e-4`、最大 `≤5e-4`；失败则禁止优化。
- **优化门**：至少 3 个固定初值；完整二维 Q16 场在相同质量下相对最强基线至少产生一个非劣 Pareto 点，并在一个主目标上改善 `≥5%`、其他主目标恶化 `≤2%`。否则结论应为“高阶场未显示额外收益”。
- **优先级**：MUST-RUN，但必须在 B1–B4 全部通过后启动。

## 5. 执行顺序与检验节点

| 里程碑 | 目标与产物 | 必跑项目 | GO 门 | 失败动作 | 估算 GPU 时 |
|---|---|---|---|---|---:|
| H0 合同冻结 | Q16 方程、节点序、积分、ANS/EAS、设计映射、CUDA/LEV/wake 边界 | tests-first、API/schema、输入 hash | 无 Q9；无可关闭 LEV/wake 的生产开关；独立 reviewer 签字 | 改合同，不写 kernel | 0–2 |
| H1 单元核 | NumPy Q16 oracle + CUDA residual/mass/Jv | B1 全部 | 数学门全部 PASS | STOP，不接 FSI | 8–16 |
| H2 结构求解 | matrix-free Newmark/Newton–Krylov、锁死控制 | B2 + 收敛失败负控 | 法向载荷/频率/积分门 PASS | 修结构核，不调气动 | 12–24 |
| H3 传递与事务 | Q16 geometry/load VJP，真实 LEV/wake 事务 | B3 守恒、失败回滚、GPU profiling | 传递/事务/GPU-only 全 PASS | STOP，不跑柔性实验 | 12–24 |
| H4 系统验证 | 刚性极限、结构基准、柔性翼时间/空间收敛 | B4 | 全局 FSI 门 PASS | 定位结构/气动/传递 owner，禁止补丁调分 | 20–50 |
| H5 可微 co-design | 同一 forward 的 VJP 与约束 | B5 梯度 | 梯度门 PASS | 只修 adjoint，不启动搜索 | 15–30 |
| H6 正式优化 | 3 初值 Pareto 搜索和 fresh 复算 | B5 正式矩阵 | 同质量 Pareto 门 PASS | 如实报告无收益 | 40–100 |
| H7 独立审计 | fresh process 重放、hash、A/B 重复、claim 审核 | 所有 MUST-RUN | reviewer PASS | 不发布结构/co-design claim | 8–20 |

**总预算初估**：115–266 RTX 4090 GPU-hours。H3 完成后用实测吞吐重估；预算不足时先减少优化迭代或 NICE-TO-HAVE 工况，禁止减少 Q16 节点、关闭 LEV/尾迹或改用 CPU。

## 6. 首批实现文件与测试边界

建议新建，不原地污染旧 Q4：

- `src/fluxvortex/q16_ancf_shell.py`
- `src/fluxvortex/warp_fsi/kernels_q16_ancf.py`
- `src/fluxvortex/warp_fsi/q16_structural_solver.py`
- `src/fluxvortex/warp_fsi/q16_work_conjugate_transfer.py`
- `platform/q16_separated_lev_fsi.py`
- `platform/q16_codesign.py`
- `tests/test_q16_ancf_element.py`
- `tests/test_q16_ancf_gpu.py`
- `tests/test_q16_work_conjugate_transfer.py`
- `tests/test_q16_lev_wake_transaction.py`
- `tests/test_q16_structural_benchmarks.py`
- `tests/test_q16_codesign_gradients.py`

首批三个红测：

1. Q16 三次场再现 + 刚体零能量 + 横向剪切/厚度压缩 patch；
2. 任意 Q16 trial deformation 下 `δq·Q = δx·f`，四等分传递必须失败；
3. predictor 第 2 次气动评价注入失败：LEV/TEV/free-wake/结构状态 0 commit，clean retry exact。

## 7. 风险与缓解

- **96 DOF 导致寄存器和局部内存压力**：采用矩阵自由 residual/Jv、按 quadrature 分块归约；以 Nsight 决定线程布局，不回退低阶单元。
- **剪切/厚度锁死**：ANS/EAS 合同先冻结，覆盖 4 个厚度比和过积分敏感性；不得靠选择性调参按 case 解锁。
- **高阶几何畸变**：入口检查 Jacobian 正定、最小奇异值和 director 方向；优化施加 geometry barrier。
- **宏材料变量非物理**：只允许正定基材料的凸组合和明确厚度映射；独立重算质量、惯量和能量正定性。
- **co-design 借质量或气动网格获益**：严格相同质量、相同气动离散、相同 LEV/wake 和相同迭代预算；artifact 绑定所有输入 hash。
- **现有 GPU FSI 含 host 往返**：Q16 路径建立独立 GPU-only runtime gate；不得把旧路径的“能运行”当作通过。
- **壳模型越界**：若目标变成层间应力、脱层、局部连接或接触，Q16 宏壳明确 STOP，另立 solid-shell/3D 子项目，不在本计划补丁式扩张。

## 8. 主文、附录与明确删减

### 主文必须包含

- Q16 运动学、锁死控制和宏材料映射；
- 法向载荷/大变形结构验证；
- 功共轭 + separated LEV/free-wake 事务；
- GPU scaling；
- 相同质量的三类 Q16 设计对比和 Pareto 结果。

### 附录可以包含

- 完整 patch-test 表；
- 6×6×3 vs 8×8×4 积分；
- 更多厚度比、初值和制造滤波敏感性；
- Q4 历史结果，仅用于说明旧证据边界。

### 明确不做

- Q9 或任何中间低阶单元；
- 显式蒙皮—梁—肋离散拓扑；
- attached-only/wake-off 的正式 FSI；
- CPU 生产结构求解；
- 以单元 `E_scale/rho_scale` 冒充物理厚度/topology；
- 层间应力、脱层、局部屈曲和连接细节声明。

## 9. 最终检查表

- [ ] Q16 是唯一新生产单元，仓库无新增 Q9 代码/测试/配置。
- [ ] 96-DOF Q16 数学、质量、内力和一致 Jv 通过独立 oracle。
- [ ] ANS/EAS 在薄到中厚范围无锁死和畸变 fail-open。
- [ ] GPU float64 正式循环无 host 科学计算、隐式上传或 CPU fallback。
- [ ] 所有气动—结构路径使用同一功共轭传递与 VJP。
- [ ] separated LEV 与真实自由尾迹在所有正式结构/优化工况中强制开启。
- [ ] predictor 只读 trial，真实 LEV/TEV/wake 每物理步恰好提交一次。
- [ ] 结构、气动、传递、尾迹任一失败均保留完整前态并可 clean retry。
- [ ] thickness、刚度、质量、剪切和惯量由同一宏材料设计映射产生。
- [ ] co-design 梯度通过，且优化比较严格同质量、同网格、同物理、同预算。
- [ ] fresh audit 通过前不宣称 Q16 FSI 或 topology co-design 有效。
