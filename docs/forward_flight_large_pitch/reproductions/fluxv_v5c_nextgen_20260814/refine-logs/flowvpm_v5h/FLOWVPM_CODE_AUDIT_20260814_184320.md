# FLOWVPM.jl 代码审计与 FluxV 三维路线裁决

生成时间：2026-08-14 18:43 CST  
状态：只读源码审计完成；尚未实施 v5h，尚未进行论文工况评分

## 1. 审计范围与版本

- FLOWVPM.jl：commit `4f433fb09f6baad25db65c9905e0d9cbb09663ce`，版本 4.0.4，MIT。
- FLOWUnsteady：commit `b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7`。
- VortexLattice.jl：commit `63e8c363389f90b00176ff67675bdfd6f2498c58`。该公开分支没有找到可直接复用的 `PanelParticleWake`，因此计划不依赖未发布 API。
- 本地 FluxV：审查 `src/fluxvortex/particles.py`、`src/fluxvortex/solver.py` 及当前 v5f material-LEV 路径。
- 本机当前没有 Julia 可执行文件，故本轮只完成源码级核对；Julia 官方测试和数值逐步 parity 是实施阶段的第一道硬门。

上游固定链接：

- [FLOWVPM.jl 固定提交](https://github.com/byuflowlab/FLOWVPM.jl/tree/4f433fb09f6baad25db65c9905e0d9cbb09663ce)
- [FLOWUnsteady 固定提交](https://github.com/byuflowlab/FLOWUnsteady/tree/b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7)
- [FLOWVPM 低存储 RK3](https://github.com/byuflowlab/FLOWVPM.jl/blob/4f433fb09f6baad25db65c9905e0d9cbb09663ce/src/FLOWVPM_timeintegration.jl)
- [FLOWVPM corrected Pedrizzetti relaxation](https://github.com/byuflowlab/FLOWVPM.jl/blob/4f433fb09f6baad25db65c9905e0d9cbb09663ce/src/FLOWVPM_relaxation.jl)
- [FLOWUnsteady 尾迹到粒子的耦合实现](https://github.com/byuflowlab/FLOWUnsteady/blob/b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7/src/FLOWUnsteady_simulation.jl)

## 2. 总裁决

FLOWVPM 能解决的是：已生成三维自由涡量之后的正则化诱导、对流、涡伸长、核尺度演化和可扩展求和。它不提供 LEV 起始判据、LESP/Kelvin 联立出生强度，也不提供适用于 Yang、Izraelevitz 和 Baik 的通用分离闭合。

因此建议建立独立的 FluxV v5h：

1. Ptera/FluxV 继续唯一拥有束缚 AIC、无穿透、释放事件/Kelvin 账本和翼面载荷。
2. 新的 source-faithful rVPM 后端只拥有已经释放的三维自由涡量输运及其对翼面的诱导速度。
3. Julia FLOWVPM 作为离线数值 oracle；生产端采用隔离的 Python/Warp 实现，不做每个 RK stage 的 Python–Julia 在线共仿真。
4. 先用 direct O(N^2) 建立真值，FMM/GPU/SFS/黏性重构全部后置。
5. FLOWVPM 不能修复当前 v5f 的 newborn+pseudovortex Schur 相消或 `q~1/dt`；有限、守恒的出生源必须先独立成立。

## 3. FLOWVPM 中可迁移的机制

### 3.1 粒子状态和核

FLOWVPM 的粒子状态包含位置 `X`、三分量涡量系数 `Gamma`、核尺度 `sigma`、体积、速度、涡量和速度梯度。这里的 `Gamma` 是三维向量 RBF 强度，不是当前 v5f 标量闭合涡环的环量。

可迁移：

- Gaussian-erf 正则核及解析 `U/J`；
- reformulated VPM 中 `X/Gamma/sigma` 的联立演化；
- 同一 RK stage、同一 source ownership 下同时求 `U` 和 `J`；
- 低存储 RK3；
- corrected Pedrizzetti 方向松弛；
- direct 与 FMM 双后端的验证结构；
- 可选 core spreading、RBF remeshing、SFS，但这些不是第一阶段必需项。

不可直接迁移：

- 把向量粒子方程套到 v5f 的标量 ring/pseudovortex 未知量；
- 用核放大、SFS 或 relaxation 掩盖出生强度发散；
- 用粒子冲量力或独立 KJ 力叠加到 Ptera 载荷。

### 3.2 时间推进

上游每个 RK stage 先在同一 stage 状态计算 `U/J`，再统一更新 `X/Gamma/sigma`。本地 `particles.py` 当前先更新位置、再在新位置和旧强度上计算 `J`，不是相同的离散方程。

上游默认的 transposed stretching 使用 `J^T Gamma`。本地当前 `einsum('tij,tj->ti', J, gamma)` 计算的是 `J Gamma`，与注释和上游默认 formulation 不一致。

### 3.3 核重叠和空间—时间解耦

FLOWUnsteady 的收敛说明使用重叠率 `lambda=sigma/h`；`lambda` 接近 1 时容易失去重叠收敛，示例通常采用约 2.0，文档建议值约 2.125。当前 FluxV 令 `h=V_infinity dt` 且 `sigma=h`，即固定 `lambda=1`，同时把时间加密、粒子间距和核尺度三者绑在一起。因此现有“时间细化”不是纯时间收敛。

v5h 必须独立控制：

- 释放/沉积间距 `h`；
- 输运子步 `dt_transport`；
- 核尺度 `sigma=lambda h`；
- 物理 core 或扩散模型。

## 4. FLOWUnsteady 提供的耦合参考及边界

FLOWUnsteady 的有效参考是离散尾迹到粒子的守恒映射：共享边按全局边 incidence 合并，边强度乘以线段长度和方向形成向量粒子强度；粒子诱导速度反馈到 VLM 配点和载荷所需位置。

它不能直接作为本项目的分离模型：官方文档明确说明当前不释放 separation wake，失速区主要依赖 lookup polars。其力链还包括 KJ、可选非定常环量力和独立 polar drag，不适合作为 FluxV 的整套载荷替换。

FluxV 的边界应冻结为：

- Ptera 是唯一束缚求解和载荷 owner；
- 自由粒子只通过速度场改变 Ptera RHS/载荷所需局部速度；
- 同一尾迹元素在任一时刻只能由 ring 或 particle 中的一方拥有；
- 禁止额外叠加粒子 impulse、LDVM 力、PLEV 力或重复 polar residual。

## 5. 本地现有粒子路径的关键偏差

当前 `src/fluxvortex/particles.py` 不应直接修补后投入论文评分，主要原因如下：

1. RK3 stage 的 `X` 与 `J` 时间层不一致。
2. 默认涡伸长使用 `J Gamma`，而非上游默认的 `J^T Gamma`。
3. corrected Pedrizzetti 在覆盖 `Gamma` 后才计算归一化因子，和上游“用旧 Gamma 定标”不同。
4. 速度、伸长率、`|Gamma|` 和 `sigma` 存在硬裁剪；NaN/Inf 时只把 `Gamma` 清零。这些操作会让结果保持有限，却破坏 Kelvin、冲量和收敛证据。
5. 黏性项只是 `nu/sigma` 的简化项，不是上游 CoreSpreading+RBF 流程。
6. 当前 hybrid 传播时 `bound_velocity_func=None`，自由粒子没有获得翼/环场的速度与应变。
7. `solver.py` 采用 `dl=Vdt`、`sigma=dl`，固定在不利的 `lambda=1`。
8. 当前粒子仍是单向可视化通道，没有进入翼面 AIC/力闭合。

实施裁决：新建 feature-gated 后端；保留旧路径用于回归，不在其上继续叠 limiter 或宣称 FLOWVPM parity。

## 6. 与 v5f 数值不稳定的关系

v5f 已观察到随 20/40/80/160 steps-per-cycle 加密，`max|q|/(Uc)` 和 newborn 位移不收敛；同时无穿透、LESP 和 Eq9 残差仍接近机器精度。取证显示主因是卷起几何下 newborn 与 pseudovortex 的 LESP Schur 响应趋于相消，另有展向节点不连续的拓扑缺件。

rVPM 对出生后的连续三维涡量输运更合适，也天然避免以相邻标量环的共享节点表示整张自由片；但它不会自动给出有限、唯一的 LEV 出生通量。因此 v5h 必须先用已知有限源证明输运和耦合，后续再单独资格审查 Ramesh/LDVM 或其他 source law。

## 7. 许可证与复现边界

FLOWVPM 和 FLOWUnsteady 为 MIT 许可，可在保留许可证和归属的前提下参考实现。计划首轮采用 clean, source-faithful implementation，并记录固定上游 commit。Julia oracle、Python 后端、输入快照和输出 hashes 都必须进入 manifest。任何上游 FMM 容差、核、formulation、relaxation 和时间推进参数都必须显式记录，不能依赖默认值漂移。

## 8. 审计结论

- `GO`：把 FLOWVPM 作为三维自由涡量输运的数值参考，并建立 v5h 隔离后端。
- `GO`：先做 direct、小规模、无黏性、无 SFS 的 source parity 和守恒桥接。
- `NO-GO`：直接把 FLOWVPM 当作 LEV/LESP 出生模型。
- `NO-GO`：直接在当前 `particles.py` 上调 clip/core 后进行三论文评分。
- `NO-GO`：在出生通量仍随 `dt` 发散时用 rVPM/SFS/FMM 掩盖问题。
- `DEFER`：FMM、GPU、SFS、core spreading、粒子合并/删除和完整 restart，直到 direct 路径通过机械门。
