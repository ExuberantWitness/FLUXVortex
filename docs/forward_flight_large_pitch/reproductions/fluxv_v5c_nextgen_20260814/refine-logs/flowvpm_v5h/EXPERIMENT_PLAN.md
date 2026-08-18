# FluxV v5h：FLOWVPM 参考的三维自由涡量路线实验计划

生成时间：2026-08-14 18:43 CST  
计划状态：预注册草案；只允许按门推进，不授权直接跑三篇论文评分

## 1. Claim Map

### C1（主声明）

一个以 FLOWVPM 为数值 oracle、采用向量粒子和 source-faithful rVPM 时间推进的自由涡量后端，能够在给定有限三维释放源时，消除当前 v5f 的展向 seam 表示缺陷，并获得对时间步、沉积间距和核重叠率一致的有界收敛。

证据要求：B1–B3 全部通过；特别是固定物理源下 20/40/80/160 细化不再出现 `q` 或位移的 `1/dt` 增长，且 direct Julia/Python parity、守恒和场收敛同时满足。

### C2（支撑声明）

保守的 surface-to-particle ownership bridge 可以把自由粒子诱导速度反馈到 Ptera，同时保持 particles-off 位级退化、每个尾迹元素唯一 owner 和 Ptera 唯一翼面载荷 owner。

证据要求：B2、B4 全部通过；任何 ring+particle 双重拥有或第二载荷通道都否定本声明。

### 明确不主张

- FLOWVPM 本身能决定 LEV 何时、何处、以多大强度脱落；
- v5h 必然提高 Yang 2025、Izraelevitz Figure 14、Baik W1–W4 的精度；
- 三篇已检视 development cases 构成 held-out 泛化验证；
- FMM/SFS/核放大可以修复不适定的出生方程。

## 2. Paper Storyline

1. 现有 v4b 在三组数据上取得有条件的经验改善，但其二维条带修正和 load blending 不是完整三维材料尾迹。
2. v5f 原生材料 ring 在代数残差闭合时仍发生时间细化发散，说明问题不在求解残差，而在出生未知量和三维自由片表示。
3. FLOWVPM 提供经过独立代码验证的三维自由涡量输运框架，但不提供 separation source law。
4. v5h 将“出生源”和“出生后输运”分开：先证明输运与耦合，再资格审查 source law，最后才允许目标论文评分。
5. 如果 source law 不能通过独立门，论文结论应停在“可靠的三维输运基础设施”，而不是制造跨论文精度声明。

## 3. Baseline Families（最多三类）

1. `Ptera/FluxV native UVLM`：particles-off 的位级基线和唯一翼面载荷 owner。
2. `FluxV v4b qualified baseline`：只在最终 B5 作为三篇论文经验性能参考。
3. `Pinned Julia FLOWVPM oracle`：固定 commit 的 direct 数值真值，不作为实验数据拟合器。

v5f/v5f2 只作为失败机理和出生律诊断，不构成第四个性能 baseline。

## 4. Experiment Blocks

### B1 — 冻结 Julia oracle 与 source-faithful direct 后端

目标：证明 Python/Warp 的核、`U/J`、rVPM RHS、RK3 和 corrected relaxation 与固定 FLOWVPM 版本一致。

实施：

1. 在项目隔离环境安装 Julia 1.10 LTS，不修改系统 Julia；固定 `Project.toml/Manifest.toml` 和上游 commit。
2. 运行 FLOWVPM 官方测试，冻结 single-vortex-ring、leapfrog 和小粒子快照。
3. 新建隔离模块，建议 `src/fluxvortex/rvpm_reference.py` 与 `rvpm_transport.py`；不修改旧 `particles.py`。
4. 第一版仅 Float64、direct O(N^2)、Gaussian-erf、inviscid、no-SFS、corrected Pedrizzetti、LSRK3。
5. 为每个粒子建立稳定 ID 和 lineage sidecar；禁止以数组下标作为持久身份。

硬门：

- 随机快照 `U` 相对误差 `<=1e-12`，`J` 相对误差 `<=1e-11`；
- 每个 RK stage 的 `X/Gamma/sigma` 相对误差 `<=1e-11`；
- corrected relaxation 前后每粒子 `|Gamma|` 与 Julia 一致；
- single-ring 使用上游同口径并满足上游误差门，同时 Python–Julia 差异 `<=1e-10`；
- zero clips、zero NaN replacement；首次非有限立即 fail closed，并输出粒子 ID、stage 和 source ledger；
- 任何一项失败即停止 B2。

### B2 — 保守 TE surface-to-particle bridge

目标：先在已知、无分离歧义的 TE 尾迹上证明三维 ring/edge→vector-particle 映射和唯一 ownership。

实施：

1. 从 Ptera 全局有向边图合并共享边，禁止逐 panel 重复沉积。
2. 对每条边使用 `Gamma_particle_vector = gamma_edge * delta_l_vector`，保留源 panel/edge/step lineage。
3. 转换必须原子化：对应 ring edge 退出 Ptera 自由尾迹 owner 的同一时层，等量 particle 进入 rVPM owner。
4. 初始只做 shadow：ring 保持求解，粒子只比较场；通过后才做 exclusive replacement，不能同时把同一尾迹场加两次。

硬门：

- 离散边 incidence、Kelvin 和总 vector moment 残差 `<=1e-12`；
- particles disabled 时 AIC、bound Gamma、wake、逐面力和总力 `np.array_equal`；
- ring→particle 转换前后，在避开 core 的固定探针集上 finest relative `L2<=1%`，连续细化误差比 `>=1.5`；
- 每个 source element 的 owner 数恒为 1；
- 任一双重 owner、场跳变或非有限即停止 B3/B4。

### B3 — 制造的有限 LE source 与三维输运收敛

目标：把“出生强度问题”隔离掉，只检验三维自由涡量表示和输运是否消除 v5f 的 seam/卷起不稳定。

实施：

1. 使用解析、有限、`O(dt_release)` 的预设 spanwise source flux；不读 Yang/Fig14/Baik 数据、不运行翼面载荷。
2. 通过同一全局有向边图沉积向量粒子，显式处理连续区、间歇区和展向自由边。
3. 初始沉积采用 FLOWUnsteady static-sheet-inspired 固定半径/计数合同：先由独立空间网格给定 `sigma=lambda*h*`，再对每条边取 `n_e=ceil(L_e/h*)`。短的 `O(dt_release)` 闭合边不得反向缩小 `sigma`；该规则是项目 development transfer，不是通用动态 LEV core 定律。
4. repeated release 前必须先建立 `NodeFrontierFact`：continuous birth 只能消费同一 rVPM 时间推进已对流到当前时层的 newest frontier；不得复用未对流的上一出生点。
5. 分开三种收敛：
   - 固定初始粒子云，仅细化输运 `dt_transport`；
   - 固定物理沉积间距 `h`，只对子步细化；
   - 固定时间步，扫描 `lambda=sigma/h={1.5,2.0,2.125,2.5}`。
6. source-frozen baseline 取 `lambda=2.125` 作为最小 overlap 下界，实际短边可有 `sigma/h_e>2.125`；不得根据三论文表现选择。
7. 消融只比较三项：fixed-sigma cVPM、dynamic-sigma rVPM、rVPM+corrected relaxation。SFS/viscosity/FMM 暂不加入。

监测量：

- `Gamma*sigma^2` 的 reformulated invariant；
- Kelvin、总 vector moment、冲量代理和质心轨迹；
- `min_spacing/sigma`、`dt*||J||`、`dt*|u_self|/sigma`；另报含均匀来流的总平移指标，但不把可由伽利略变换消去的 `U_inf` 当作 stretching 稳定性证据；
- 粒子 ID/lineage、seam-free edge closure、非有限和 limiter 计数。

硬门：

- 20/40/80/160 时间层全部有限，clip count=0；
- 峰值量的 `160/80 <=1.25`，不得出现 `>=2x` 细化增长；
- finest 两级 probe velocity、质心和积分 vector moment 差异 `<=5%`；
- Kelvin/vector-moment 相对漂移 `<=1e-6`；
- seam incidence 残差 `<=1e-12`；
- 每个 continuous node 的 advected-frontier fact 唯一、时间层连续、不可重放；相邻释放层中心距 `<=h*`；
- 若仍出现 `q~1/dt` 或 newborn speed 发散，判定 source/bridge 仍不适定，停止 B4/B5。

### B4 — Ptera 双向机械耦合

目标：让旧自由粒子在同一时间层影响 Ptera 配点与载荷局部速度，同时保持 Ptera 唯一受力路径。

实施：

1. 粒子场只在明确时间层进入 Ptera RHS、KJ 各腿速度和原有 unsteady load 所需位置。
2. Ptera bound/near-wake 场在 rVPM 每个 RK stage 以同一 `U/J` provider 作用于粒子；不得只给速度而漏掉应变。
3. 冻结当步 newborn 的可见性：出生当步或下一步生效必须由离散方程决定，并以解析测试锁定。
4. Ptera KJ+dGamma 是唯一 surface force；关闭 particle impulse、独立 LEV force、LDVM additive、PLEV additive 和重复 polar residual。
5. direct 后端通过后才测试 FMM；固定容差、阶数和 commit，不依赖易漂移默认值。
6. 增加 restart：稳定 ID、lineage、owner、`X/Gamma/sigma`、RK 状态和 Ptera step token 必须连续。

硬门：

- no-release 和 disabled 两条路径逐步 bitwise 等于 native FluxV；
- particle induction 在 RHS/载荷各自只出现一次，静态 call-count 与数值 ledger 均闭合；
- Ptera 总载荷严格等于其逐面载荷求和，没有第二 force owner；
- direct/FMM 探针 `U/J` 相对误差 `<=1e-4`，积分载荷变化 `<=0.2%`；
- restart 前后逐步状态和载荷在浮点容差内连续；
- 先通过非目标 AR6 canonical/heaving-wing case，才允许 B5。

### B5 — Source-law 资格审查与全 22 工况验证

目标：只有在 B1–B4 全绿后，才把有限自由涡量 source 接到真实 LEV 事件，并评估三篇论文。

source-law 候选顺序：

1. Ramesh/LDVM v2.5 clean-room source-flux transfer：只借 LESP/Kelvin 释放通量，不叠加其完整二维力；明确 Ptera 时层与原 Fortran 的差异。
2. 若第一项不能在独立 2D/AR6 case 上收敛，再研究独立来源的三维 release law。
3. 任何条带 source 必须通过全局 conservative edge graph 沉积；禁止每条带独立生成不连续 span nodes。

source-law 先决门：

- 独立 source case 的释放强度为 `O(dt)`，`Gamma/dt` 有界并收敛；
- onset、Kelvin、birth geometry 与其来源模型一致；
- 不使用三篇目标观测选择 `Lcrit`、core、lambda、relaxation 或分支。

全 22 工况：

- Yang 2025：6 个 AoA，周期均值 L/D；
- Izraelevitz Figure 14：12 个唯一条件、14 个实验 marker，两种计权均报告；
- Baik W1–W4：4×400 phase CL/CD，按源 1 Hz 等效滤波口径评分，同时报告 raw。

性能门：

- 每篇 primary aggregate 均不劣于冻结 v4b；
- Yang L/D、Fig14 all14 与 unique12、Baik 8 个 case-channel 均报告，不允许只挑改善项；
- 任一论文 primary no-regression 失败，则 v5h 不晋级，保留 v4b；
- 即使三篇通过，也只能称 development-transfer，不能称 held-out 泛化。

## 5. Run Order and Milestones

### R-1（可选且便宜）

完成 v5f2 revised Ramesh birth geometry 的 M5。若仍失败，直接归档为 ring-source 反例；若通过，也只作为 source-law 候选，不取代 v5h 的三维输运验证。

### R0 — 环境和上游冻结

- 安装隔离 Julia；固定 FLOWVPM/FLOWUnsteady commit、包 manifest 和许可证归属。
- 跑官方 tests，导出小型 Float64 oracle 快照。
- 交付：environment manifest、oracle CSV/HDF5、hashes、原始 test log。

### R1 — B1 direct parity

- kernel U/J → one-step RK → multi-step RK → relaxation → ring/leapfrog。
- 任何 parity 失败都不进入 Ptera。

### R2 — B2 TE bridge

- 单环 → 相邻 panel 共享边 → 矩形翼 TE row → 多翼/间歇边界。
- shadow 通过后才做 exclusive ownership。

### R3 — B3 manufactured LE transport

- 先 source 固定的 direct runs，再做 time/h/lambda 正交矩阵。
- 达到收敛门后才能接 wing solver。

### R4 — B4 Ptera coupling

- no-release exact reduction → TE-only two-way → 非目标 AR6/heaving-wing。
- direct 通过后再评估 FMM；SFS、黏性和粒子合并仍默认关闭。

### R5 — B5 source law and target papers

- 独立 source qualification → all-22 一次冻结运行 → 全工况图和审计。
- 禁止中途查看目标误差后改参数再继续同一 claim。

## 6. Compute and Data Budget

| 阶段 | 估计 CPU 时间 | 主要存储 | 备注 |
|---|---:|---:|---|
| R0 Julia 环境/官方测试 | 2–4 CPU·h | <2 GB | 首次包构建可能较慢 |
| R1 direct parity | 4–12 CPU·h | 2–5 GB | 小粒子 Float64 快照 |
| R2 TE bridge | 8–20 CPU·h | 5–10 GB | direct probe matrices |
| R3 manufactured LE matrix | 24–60 CPU·h | 10–30 GB | 4 dt × 4 lambda × 3 formulations，分批早停 |
| R4 Ptera 非目标机械验证 | 30–80 CPU·h | 20–50 GB | direct 为主，FMM 后置 |
| R5 all-22 | 100–300 CPU·h | 50–150 GB | 只在所有前门通过后 |

首轮不需要 GPU。若 direct 粒子数超过可承受范围，优先缩小机械 case，而不是提前引入 FMM 改变真值。

## 7. Risks and Mitigations

1. **出生源仍不适定**：rVPM 不能修复。用制造有限源隔离输运；source law 单独早停。
2. **ring/particle 双计**：全局 owner ledger、原子转换、induced-field continuity gate。
3. **载荷双计**：Ptera 唯一 surface-force owner；禁止粒子力和旧残差叠加。
4. **U/J 时间层不一致**：stage-aware provider，同一 source set 同时计算 U/J。
5. **核/时间混杂**：独立控制 `dt_transport`、`h`、`sigma`、lambda。
6. **FMM 默认漂移**：direct oracle；固定上游 commit、容差和阶数。
7. **粒子删除破坏 lineage**：首轮不删除/合并；稳定 ID sidecar；后续单独验证。
8. **目标数据泄漏**：B1–B4 不读取目标 GT；B5 参数和 gate 在读取结果前冻结。
9. **低 AR 适用域**：先过非目标 AR6，再对 Yang AR≈1.92 明确作为外推验证。
10. **计算量膨胀**：逐门早停；不通过 direct 机械门就不做 FMM、full 或 all-22。

## 8. Completion Checklist

- [ ] 上游 commits、Julia manifest、许可证与 oracle hashes 冻结。
- [ ] FLOWVPM 官方 tests 在本机隔离环境通过。
- [ ] Python/Warp direct `U/J`、RK3、relaxation 与 Julia parity 通过。
- [ ] 生产路径零 clip、零 NaN replacement，失败 telemetry 完整。
- [ ] TE edge incidence、Kelvin、vector moment 和唯一 owner 通过。
- [ ] 时间、沉积间距和 lambda 三类收敛已解耦。
- [ ] 制造 LE source 的 20/40/80/160 收敛门通过。
- [ ] particles-off/no-release 位级退化通过。
- [ ] Ptera 唯一载荷 owner 和双向 U/J 时层通过。
- [ ] 非目标 AR6/heaving-wing 机械 case 通过。
- [ ] source law 在独立数据上资格审查通过。
- [ ] all-22 参数在评分前冻结，GT 读取顺序可审计。
- [ ] 三篇全工况曲线、raw/filtered metrics、manifest 和独立审计完成。
- [ ] 若任一硬门失败，停止并保留 v4b，不用 limiter/core 调参掩盖失败。
