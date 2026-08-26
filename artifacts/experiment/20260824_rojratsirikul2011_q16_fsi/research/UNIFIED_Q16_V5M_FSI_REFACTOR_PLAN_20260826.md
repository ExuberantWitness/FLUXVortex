# Q16 / FLUX-V5M / FSI / 扑翼自由飞统一框架详细重构方案

日期：2026-08-26  
适用仓库：`FLUXV_RUNS/v5m-fa8eaca`  
当前生产分支：`run/q16-lev-tev-pc-fsi-20260821`  
重构起点：生产源码基线 `a0f0869`，代码地图见 `HANDOFF_COMPLETE_CODE.md`  
问题审计：`ANALYSIS_BRIEF_v4.md` 及其独立数据/代码复核

---

## 0. 执行结论

项目可以重构为一个同时支持以下任务的组合式框架：

1. Meng 刚性扑翼、固定机身、给定扑动/俯仰；
2. Baik、Yang、Izraelevitz–Scherer 三个刚性纯气动论文 CASE；
3. Rojratsirikul 2011 固定框架、Q16 柔性膜翼强 FSI；
4. 类 Meng 扑翼运动 + Q16 柔性翼 + FLUX-V5M + 机身六自由度自由飞。

但统一对象必须是：

- 状态和时间线；
- 翼面运动学输出；
- 气动 propose/commit 事务；
- 表面载荷到广义载荷的功共轭映射；
- 动力学子系统和耦合策略；
- CASE、观测量和验收门。

不得强行把 2D Baik 和 3D 多翼面自由飞塞进同一个数值气动内核。Baik 保留 2D LDVM 后端，但接入同一运行框架、状态机、结果协议和科学验收层。

推荐路线不是推倒重写，而是用适配器逐步从当前生产路径中抽出稳定边界；每一步均须由正式论文 CASE 回归约束。

---

## 1. 当前可信状态与重构前提

### 1.1 Q16 状态

当前判断：**组件级基本正常；Roj 工况物理级验证尚未完成。**

可信证据：

- Q16 16 节点张量三次插值、Kronecker 性和单位分解；
- 有限刚体运动零 Green–Lagrange 应变；
- 一致质量矩阵对称正定并恢复总质量；
- ANS/EAS 增强参数驻值；
- 内力是凝聚应变能的导数；
- 解析切线作用与内力有限差分一致；
- CUDA ANS/EAS 与独立参考实现对齐；
- GPU Newmark/Newton 结构步收敛并保持边界；
- 失败结构步不污染输入状态；
- 质量阻尼进入有效切线和端点功 ledger；
- 气动到结构的转置映射满足功共轭。

仍未证明：

- Roj 真实材料/厚度/固支条件下的独立模态；
- 变形平衡态附近的固有频率和阻尼；
- 正式 5×10 Q16 网格的空间收敛；
- 长时大变形能量和耗散稳定性；
- Q16 本构参数对实验的独立可识别性。

因此本重构不替换 Q16 单元，不改变现有形函数、ANS/EAS、本构或 Newmark 算法；只抽出接口并补验证。

### 1.2 当前 FSI 状态

当前判断：**预测校正和正式提交骨架正确；集成与长期物理事务不完整。**

可复用部分：

- committed aerodynamic state 不被普通 trial 修改；
- predictor/corrector 从 committed state 克隆真实 LEV、TEV、尾迹状态；
- trial 内实际推进粒子和尾迹；
- 固定点收敛后执行 formal replay；
- formal replay 再次收敛后原子提交结构和气动状态；
- Aitken 可作为默认耦合加速器，IQN-ILS 保持 opt-in；
- 结构子步和气动大步已显式调度。

缺口：

- `tests/test_flux_v5m_fsi_gpu_contract.py` 没有完整驱动当前 `Q16NativeV5MFSIStepper`；
- 缺少当前 native 路径的失败回滚、唯一提交和正式重放集成测试；
- 耦合残差仅证明运动学固定点，不证明环量、冲量或系统能量闭合；
- 运行结果中的 `rejected_trial_count=0` 与实际存在非正式 trial 的语义不一致；
- 长时粒子删除与尾迹截断没有进入完整事务 ledger。

### 1.3 当前气动状态

当前判断：**既有三个 CASE 是重要基线，但不等价于当前 native FSI 路径已完全验证。**

- Baik W1–W4 主要验证 2D CUDA LDVM 路径；
- Yang 和 Izraelevitz 历史最佳来自旧 Ptera GPU 底盘，尚未全部迁移到 mandatory separated-LEV + joint-TEV + free-wake 当前合同；
- Meng 当前正式复现是 Ptera/V4B transfer，不是当前 native V5M；
- 当前 Roj 使用 `Q16NativeV5MSolver`，其几何、状态所有权、远场处理和载荷历史并非前三个 CASE 的完全同一路径。

`separated_lev=True`、`joint_tev=True` 和 `free_wake=True` 是必要条件，但不能替代以下正确性：

- 3D LESP 与 source bank 释放状态一致；
- 释放强度、位置和符号正确；
- 粒子删除守恒；
- 尾迹截断守恒；
- 尾迹反馈未被不当冻结；
- 唯一载荷 owner 与历史项正确。

### 1.4 `ANALYSIS_BRIEF_v4.md` 对重构的直接约束

必须先纠正以下报告层问题，不能把错误结论固化进新框架：

1. 论文位移指标必须是 `max_xy(mean_t(z))`，不是 `mean_t(max_xy(z))`；
2. 不允许用末端短窗 `Cn≈0.913` 替代时间平均；
3. E=1.4 MPa 是由目标位移反推的敏感性分支，不是独立验证；
4. 正式 accuracy gate 全部为 false 时，运行不得仍标为 accuracy completed；
5. `instantaneous max(z)` 不能用来判断是否发生符号穿越；
6. 当前短记录不能确认精确阻尼比、三次谐波或 St≈1 模态缺失根因；
7. Rayleigh 阻尼只能先作为待验证方案，不能作为首个“低风险已确认修复”；
8. 先修 LEV/TEV/尾迹事务，再进行长时阻尼和论文精度裁定。

---

## 2. 重构目标、非目标与硬约束

### 2.1 重构目标

建立一个组合式运行框架，使下面对象彼此独立但可组合：

- CASE 定义与论文真值；
- 刚性、给定运动、Q16 弹性、自由刚体六自由度；
- 2D LDVM 与 3D FLUX-V5M；
- 单向气动、固定框架强 FSI、自由飞强耦合；
- GPU 执行策略、事务、检查点和结果记录；
- 科学指标和验收门。

### 2.2 非目标

本轮不做：

- 替换 Q16 为 Q4/Q9 或中间低阶单元；
- 新建 Ptera toy 作为科学验收；
- 关闭 separated LEV、joint TEV 或 free wake 获取稳定结果；
- 把 Baik 强行改造成自由端三维机翼；
- 用一个巨型 `if case == ...` solver 包含所有任务；
- 在框架重构中顺便调 `Lcrit`、E、阻尼或实验目标；
- 在 CPU 上执行正式科学数值计算；
- 在没有回归门时一次性重写全部运行器。

### 2.3 硬约束

1. 正式科学数据面必须 CUDA float64、fail closed、无 CPU numerical fallback；
2. 所有正式 3D V5M 模式必须保持 separated LEV + joint TEV + free wake；
3. 所有 trial 必须从同一 committed snapshot 派生；
4. 只有 formal replay 允许提交；
5. 每个外层时间步最多发生一次全局提交；
6. 气动载荷到结构、关节和刚体的传递必须功共轭；
7. 粒子删除、尾迹截断或合并必须守恒环量并审计线/角冲量；
8. CASE 参数、模型参数和执行参数必须分层，不允许在求解器中写死某篇论文；
9. 科学验收必须使用真实论文 CASE 和真实观测定义；
10. 组件单元测试可以使用数学构造，但不得把它冒充论文验证；
11. 源码、GT、配置、运行环境和分析脚本必须可重建；
12. 重构前后同一正式 CASE 的结果必须有明确 parity 口径。

---

## 3. 四类任务的统一分解

| 任务 | Body | Joint/motion | Elastic | Aero | Coupling |
|---|---|---|---|---|---|
| Meng 固定机身扑翼 | 固定 | 给定扑动+俯仰 | 刚性 | 3D V5M | 单向 |
| Baik | 固定 | 给定升沉+俯仰 | 刚性/准二维 | 2D LDVM | 单向 |
| Yang | 固定 | 给定刚性扑翼机构运动 | 刚性 | 3D V5M | 单向 |
| Izraelevitz–Scherer | 固定 | 给定升沉+俯仰 | 刚性 | 3D V5M | 单向 |
| Roj A10/A16/A23 | 固定 | 固定边框 | Q16 | 3D V5M | 强 FSI |
| 柔性 Meng 自由飞 | SE(3) 6DOF | 给定/受控扑动+俯仰 | 每翼 Q16 | 多翼面 3D V5M | 强耦合 |

统一的是外层状态机和数据契约；2D/3D、刚性/弹性、固定/自由是可替换组件。

---

## 4. 总体架构

### 4.1 唯一全局时间线

每个正式外层步执行：

```text
CommittedWorldState(n)
  │
  ├─ KinematicsGraph.evaluate(trial dynamic state)
  │      └─ SurfaceFrame(s): geometry + velocity + topology
  │
  ├─ Aerodynamics.propose(committed aero state, SurfaceFrame(s), dt)
  │      ├─ bound circulation
  │      ├─ separated LEV release/transport
  │      ├─ joint TEV/free-wake update
  │      ├─ surface force packet
  │      └─ proposed aero state (未提交)
  │
  ├─ GeneralizedLoadProjector
  │      └─ body wrench + joint torque + Q16 generalized loads
  │
  ├─ Dynamics.propose(committed dynamic state, loads, dt)
  │      └─ proposed body/joint/Q16 state
  │
  ├─ Coupling residual / accelerator / formal replay
  │
  └─ GlobalTransaction.commit_once()
         └─ CommittedWorldState(n+1)
```

任何失败均保留 `CommittedWorldState(n)`。

### 4.2 建议包结构

```text
src/fluxvortex/
  runtime/
    execution_policy.py
    clocks.py
    provenance.py
    result_schema.py
  state/
    world.py
    transaction.py
    circulation_ledger.py
  kinematics/
    frames.py
    graph.py
    fixed_rigid.py
    prescribed_joint.py
    q16_surface.py
    multibody_surface.py
  aero/
    protocol.py
    v5m/
      config.py
      topology.py
      geometry.py
      state.py
      solver.py
      loads.py
      wake.py
      separation.py
    ldvm2d/
      adapter.py
  dynamics/
    protocol.py
    fixed_body.py
    rigid_body_se3.py
    joints.py
    q16.py
    composite.py
  coupling/
    one_way.py
    partitioned.py
    accelerators.py
    residuals.py
  cases/
    protocol.py
    meng2025.py
    baik2011.py
    yang2025.py
    izraelevitz2017.py
    rojratsirikul2011.py
  validation/
    observers.py
    gates.py
    stationarity.py
    spectra.py
```

不要求一次性完成物理移动；先建立薄接口和适配器，再逐文件迁移。

---

## 5. 核心数据合同

### 5.1 标识和坐标系

每个对象必须具有稳定 ID：

- `body_id`：独立刚体；
- `joint_id`：机身与翼面或部件之间的运动副；
- `surface_id`：气动翼面；
- `edge_id`：可释放 LEV/TEV 的拓扑边；
- `elastic_id`：Q16 弹性子系统；
- `case_id`、`run_id`、`step_index`、`generation`。

显式坐标系：

- `I`：惯性/风洞世界系；
- `B`：机身系；
- `J`：翼根关节系；
- `S`：翼面参考系；
- `P`：面板局部弦向—展向—法向系。

所有位置、速度、力和力矩字段必须在名字或 metadata 中声明坐标系。禁止依赖隐式 Ptera 轴约定。

### 5.2 `RigidBodyState`

建议字段：

```python
@dataclass(frozen=True)
class RigidBodyState:
    body_id: str
    position_I: torch.Tensor       # (3,), float64 CUDA
    quaternion_IB: torch.Tensor    # (4,), unit quaternion
    linear_velocity_I: torch.Tensor
    angular_velocity_B: torch.Tensor
    mass: torch.Tensor             # scalar CUDA or frozen device constant
    inertia_B: torch.Tensor        # (3,3)
```

提交时检查：

- 四元数归一；
- 质量和转动惯量正定；
- 所有科学张量同一 CUDA device/float64；
- 姿态和速度有限；
- generation 单调增加一。

### 5.3 `JointState`

```python
@dataclass(frozen=True)
class JointState:
    joint_id: str
    parent_body_id: str
    coordinates: torch.Tensor
    rates: torch.Tensor
    prescribed: bool
```

Meng 首版使用给定扑动和俯仰；后续可扩展为力矩驱动，但不改变翼面运动学接口。

### 5.4 `ElasticState`

包装现有 Q16 state/velocity/acceleration，不改变底层 Warp 数组：

```python
@dataclass(frozen=True)
class ElasticState:
    elastic_id: str
    q: wp.array
    qd: wp.array
    qdd: wp.array
    boundary_generation: int
```

### 5.5 `SurfaceFrame`

这是统一架构的关键边界：气动求解器只接收翼面几何和速度，不知道它来自刚性、给定运动、Q16 或自由机身。

```python
@dataclass(frozen=True)
class SurfaceFrame:
    surface_id: str
    body_id: str
    panel_rings_I: torch.Tensor
    panel_ring_velocity_I: torch.Tensor
    collocation_I: torch.Tensor
    collocation_velocity_I: torch.Tensor
    normals_I: torch.Tensor
    areas: torch.Tensor
    leading_edge_I: torch.Tensor
    trailing_edge_I: torch.Tensor
    leading_velocity_I: torch.Tensor
    trailing_velocity_I: torch.Tensor
    chordwise_panels: int
    spanwise_panels: int
    topology_digest: str
```

所有字段必须驻留 CUDA float64；`topology_digest` 在运行中不可漂移，除非未来明确支持拓扑变化。

### 5.6 `AeroState`

将当前单翼 `NativeV5MState` 扩展为：

```python
@dataclass
class V5MWorldState:
    step_index: int
    generation: int
    surfaces: dict[str, V5MSurfaceState]
    wake_system: V5MWakeSystemState
    circulation_ledger: CirculationImpulseLedger
```

每个 `V5MSurfaceState` 拥有：

- bound circulation；
- previous bound circulation；
- 3D LESP state；
- source closure state；
- LEV frontier/particles；
- TEV release history；
- wake edge connectivity；
- load-history state。

禁止多个模块同时拥有同一个“是否分离”真值。

### 5.7 `AeroProposal`

```python
@dataclass(frozen=True)
class AeroProposal:
    parent_digest: str
    proposed_state: V5MWorldState
    surface_loads: tuple[SurfaceLoadPacket, ...]
    diagnostics: AeroStepDiagnostics
    proposal_digest: str
```

proposal 必须包含所有未来提交所需状态，不能依赖 solver 对象中的隐式 mutable cache。

### 5.8 `GeneralizedLoadPacket`

```python
@dataclass(frozen=True)
class GeneralizedLoadPacket:
    body_wrenches_I: dict[str, torch.Tensor]   # force+moment
    joint_torques: dict[str, torch.Tensor]
    elastic_forces: dict[str, wp.array]
    surface_work_rate: torch.Tensor
    generalized_work_rate: torch.Tensor
    relative_work_error: torch.Tensor
```

验收：

\[
\left|f_s^T v_s - Q_g^T \dot q_g\right| /
\max(|f_s^T v_s|, |Q_g^T \dot q_g|, \epsilon)
\le \varepsilon_{work}.
\]

### 5.9 `CirculationImpulseLedger`

每一步至少记录：

- bound circulation before/after；
- newborn LEV/TEV circulation；
- retained LEV/TEV/wake circulation；
- removed/merged circulation；
- Kelvin residual；
- linear impulse before/after；
- angular impulse before/after；
- culling/remeshing residual；
- load-impulse consistency residual。

粒子年龄上限和尾迹排数上限不能只做数组切片，必须通过 ledger owner 执行。

---

## 6. 核心协议接口

### 6.1 翼面运动学

```python
class SurfaceKinematicsProvider(Protocol):
    @property
    def surface_ids(self) -> tuple[str, ...]: ...

    def evaluate(
        self,
        world_state: WorldDynamicState,
        time: float,
    ) -> tuple[SurfaceFrame, ...]: ...

    def project_surface_loads(
        self,
        world_state: WorldDynamicState,
        loads: tuple[SurfaceLoadPacket, ...],
    ) -> GeneralizedLoadPacket: ...
```

实现：

- `FixedRigidSurfaceKinematics`；
- `PrescribedRigidSurfaceKinematics`；
- `Q16FixedFrameKinematics`；
- `BodyJointQ16Kinematics`。

### 6.2 气动步进器

```python
class AerodynamicStepper(Protocol):
    def initialize(self, surfaces: tuple[SurfaceFrame, ...]) -> AeroState: ...

    def propose(
        self,
        committed: AeroState,
        surfaces: tuple[SurfaceFrame, ...],
        dt: float,
    ) -> AeroProposal: ...

    def commit(self, owner: AeroOwner, proposal: AeroProposal) -> None: ...
```

实现：

- `LDVM2DStepperAdapter`：Baik；
- `V5M3DStepper`：Meng/Yang/Izraelevitz/Roj/自由飞。

### 6.3 动力学子系统

```python
class DynamicSubsystem(Protocol):
    def predict(self, committed: DynamicState, dt: float) -> DynamicState: ...

    def propose(
        self,
        committed: DynamicState,
        loads: GeneralizedLoadPacket,
        dt: float,
    ) -> DynamicProposal: ...
```

实现：

- `FixedDynamics`：固定刚性 CASE；
- `PrescribedMotionDynamics`：Meng/Yang/Izraelevitz/Baik；
- `Q16DynamicsAdapter`：当前 Roj；
- `RigidBodySE3Dynamics`：机身；
- `CompositeBodyJointQ16Dynamics`：自由飞柔性扑翼。

### 6.4 耦合策略

```python
class CouplingStrategy(Protocol):
    def advance(
        self,
        owner: WorldOwner,
        dt: float,
    ) -> CommittedStepResult: ...
```

实现：

- `OneWayPrescribedCoupling`；
- `PartitionedStrongFSI`；
- `PartitionedFreeFlightFSI`。

第一版自由飞继续使用 partitioned strong coupling，不在同一阶段引入全单体 Jacobian。

### 6.5 CASE 与验收

```python
class CaseDefinition(Protocol):
    case_id: str
    source_manifest: SourceManifest
    physical_config: PhysicalConfig
    numerical_protocol: NumericalProtocol
    expected_observables: tuple[ObservableContract, ...]

    def build_world(self, policy: ExecutionPolicy) -> WorldOwner: ...
    def observers(self) -> tuple[Observer, ...]: ...
    def gates(self) -> tuple[AcceptanceGate, ...]: ...
```

CASE 只定义真实几何、运动、材料、流场、观测和允许的数值协议；不得直接包含目标驱动调参逻辑。

---

## 7. 机身六自由度与多翼面设计

### 7.1 六自由度的正确所有权

通常是一架飞行器一个 `RigidBodyState`，而不是每个翼面复制一套机身六自由度：

```text
Aircraft Body B (one 6DOF owner)
  ├─ left_flap_joint  → left Q16 wing
  ├─ right_flap_joint → right Q16 wing
  ├─ tail_joint       → tail surface
  └─ fuselage aerodynamic surface (optional)
```

只有物理上独立的刚体才各自拥有六自由度。若“每个翼面引入额外机身六自由度”实际指多个独立飞行器，则每个 `body_id` 单独建状态；若是同一机身的多个翼面，必须共享一个状态。

### 7.2 运动学链

对于第 k 个 Q16 翼面：

\[
x_I = x_B + R_{IB}\,T_{BJ_k}(q_{J_k})\,
      \left(X_{S_k} + u_{e,k}\right).
\]

速度必须包含：

- 机身平动；
- 机身角速度叉乘；
- 关节运动速度；
- Q16 弹性速度。

不能只把机身位姿加到位置上而漏掉表面速度，否则非穿透条件和 added-mass 载荷错误。

### 7.3 载荷映射

统一使用运动雅可比转置：

\[
Q = J^T f_s.
\]

同一份表面力同时产生：

- 机身合力；
- 关于机身质心的力矩；
- 扑动/俯仰关节力矩；
- Q16 弹性广义力。

避免分别计算三份载荷造成重复计力。

### 7.4 刚体积分器

建议：

- 平动使用与耦合时间精度匹配的隐式中点或 generalized-α；
- 姿态使用四元数或 SO(3) 指数映射；
- 每个 accepted step 后规范化四元数；
- 刚体质量矩阵、Q16质量矩阵和关节惯性进入统一动态残差；
- 重力、推力/驱动力矩、约束反力作为显式 load owners；
- 气动力矩统一关于 body COM 计算。

不建议使用欧拉角作为积分状态。

### 7.5 移动翼根条件

当前 Roj 是四边固定；自由扑翼的 Q16 翼根需要：

- 根部 Q16 节点/导演与关节 frame 的时间一致约束；
- 约束位置、速度和加速度三层一致；
- 约束反力回传为机身/关节反力；
- 约束更新必须属于 trial，不得提前修改 committed boundary state。

首版可采用消元/投影的强约束；后续若需要柔性铰接再引入乘子或 augmented Lagrangian。

---

## 8. FLUX-V5M 重构细节

### 8.1 从 Q16 中抽离几何

当前：

- `Q16NativeV5MSurface` 同时负责 Q16 插值和 V5M 面板构造；
- `Q16NativeV5MSolver` 直接依赖该具体类。

修改：

1. 将 `NativeV5MGeometry` 移到 `aero/v5m/geometry.py` 并改名 `SurfaceFrame` 或建立零拷贝转换；
2. 将面板环格构造提取为 `StructuredSurfaceBuilder`；
3. Q16 只负责输出四分之一弦、LE、TE及其速度；
4. 刚性/给定运动 adapter 输出同一数据结构；
5. V5M solver 只读取 `SurfaceFrame`。

R1 阶段必须保证 Roj/Yamano 的 rings、collocation、normal、area、velocity 逐位或舍入级不变。

### 8.2 参数分层

拆分当前 `NativeV5MConfig`：

- `V5MPhysicsConfig`：LESP模型、核半径、载荷历史模型；
- `V5MDiscretizationConfig`：面板、时间步、粒子容量、尾迹离散；
- `V5MExecutionConfig`：device、dtype、kernel batching；
- `CasePhysicalConfig`：密度、来流、Re、论文材料和运动；
- `RetentionPolicy`：粒子/尾迹保留、合并、截断方法。

删除求解器中 `lesp_crit必须等于0.11` 的 Yamano 专用硬编码；改为 CASE 负责值和来源，solver 只验证其合法性。删除硬编码不代表允许调参。

### 8.3 唯一分离状态 owner

目标：3D 实际表面 LESP 是“是否分离”的唯一真值。

建议流程：

1. 在实际 3D geometry 上计算 `lesp_pre_3d`；
2. 用 CASE 冻结的 threshold 得到 `surface_separated`；
3. source closure 接收该 mask，计算已激活条带的释放强度/位置；
4. source closure 不再独立产生另一个有冲突可能的 `shed_lev`；
5. 记录 `surface_separated`、`new_release_mask` 和 `continuing_release_mask`；
6. pin mask 与释放状态的关系必须显式，不得简单取两个 owner 的并集；
7. 不一致直接 fail closed。

### 8.4 LEV/TEV 联合事务

每个条带执行：

- bound solve pre-state；
- LESP release decision；
- LEV circulation proposal；
- joint TEV proposal；
- Kelvin/Kutta约束；
- newborn induction加入同一次AIC；
- load计算；
- material convection proposal；
- formal commit。

LEV 与 TEV 不能分别在不同模块提前提交。

### 8.5 粒子与尾迹保留策略

禁止：

```python
particles = particles[age <= cap]
wake = wake[:max_rows]
```

替换为 `RetentionPolicy.apply(proposed_state, ledger)`：

- `NoCulling`：短时反事实/验证；
- `ConservativeMerge`：邻近粒子守恒合并；
- `FarWakeCondensation`：远场尾迹等效化；
- `AuthorKelvinEnforcedDeletion`：若有独立来源和oracle，可把删除环量进入 `kelv_enf`。

每个策略必须报告守恒误差和适用范围。

### 8.6 多翼面支持

将 AIC 按全局面板集合组装：

```text
[surface 0 panels]
[surface 1 panels]
...
```

保留 offsets：

- panel offset；
- strip offset；
- LE/TE edge offset；
- wake owner；
- particle source surface/strip。

多翼面必须包含相互诱导；不能把左右翼分别独立求解后简单相加。

### 8.7 唯一载荷 owner

必须明确选择并验证一个生产载荷公式 owner。`bound_rate` 与 `material` 是物理模型差异，不应只是为避免发散的运行开关。

要求：

- 相同历史输入下与作者 oracle 对齐；
- 刚性平板稳态/非定常极限正确；
- newborn LEV、TEV、wake history、added mass 只计一次；
- 表面力、总力和冲量导数 ledger 闭合；
- FSI 和刚性 CASE 使用同一 owner，除非 CASE 明确选择经验证的另一物理模型。

---

## 9. Q16 与结构层重构细节

### 9.1 保持不变的生产内核

首轮不得改变：

- Q16 shape functions；
- ANS/EAS 凝聚；
- Green–Lagrange 应变；
- 材料本构；
- CUDA 内力和切线核；
- Newmark 参数；
- Newton接受条件；
- 当前正式 Q16 网格。

### 9.2 `Q16DynamicsAdapter`

包装现有 `Q16CudaNewmarkStepper`：

- 输入 `ElasticState` 和 `elastic_forces[elastic_id]`；
- 输出 detached `DynamicProposal`；
- 保留现有失败异常和 work audit；
- 不要求气动 solver 是具体 `Q16NativeV5MSolver`；
- 结构 solver 不再知道 surface panel layout。

### 9.3 边界策略

新增：

- `FixedPerimeterBoundary`：Roj；
- `FixedRootBoundary`：Yamano；
- `MovingRootBoundary`：自由扑翼；
- `PrescribedFrameBoundary`：给定机身/关节运动验证。

边界策略输出 admissible projection 和约束值/速度/加速度，不直接修改结构 owner。

### 9.4 阻尼策略

将阻尼从 CASE runner 中抽出为：

- `NoDamping`；
- `KelvinVoigtReferenceTangentDamping`；
- `RayleighDamping`；
- `StartupOnlyNumericalDamping`。

每种策略记录：

- 物理/数值身份；
- 参数来源；
- 目标频带；
- 实际每模态阻尼（若已识别）；
- 是否在统计窗仍激活。

Roj 在气动事务修复前不以阻尼改变宣称论文复现。

---

## 10. 耦合与全局事务

### 10.1 `WorldOwner`

唯一 committed owner：

```python
@dataclass
class WorldOwner:
    dynamic_state: WorldDynamicState
    aero_state: AeroState
    previous_load: GeneralizedLoadPacket | None
    generation: int
```

### 10.2 trial 规则

每个 trial：

1. 读取同一 `WorldOwner` digest；
2. 生成 trial dynamic state；
3. 生成 surface frames；
4. 从 committed aero state propose；
5. 映射载荷；
6. 动力学 propose；
7. 计算残差；
8. 丢弃或进入 formal replay。

trial 数量必须真实记录：

- `aero_proposal_count`；
- `dynamic_proposal_count`；
- `discarded_trial_count`；
- `formal_replay_count`；
- `commit_count`。

### 10.3 formal replay

formal replay 必须：

- 仍从相同 committed parent 开始；
- 使用已收敛的 trial endpoint；
- 重新完整推进 LEV/TEV/wake；
- 重新计算载荷和动态响应；
- 通过耦合、守恒和功共轭门；
- 然后一次性提交。

### 10.4 残差集合

不能只看位移/速度残差。建议同时记录：

- `r_q`：结构/刚体配置；
- `r_v`：速度；
- `r_load`：界面载荷；
- `r_work`：功共轭；
- `r_kelvin`：环量；
- `r_impulse`：涡冲量；
- `r_constraint`：移动根/关节约束。

收敛主条件仍可使用加权运动学固定点，但其它残差必须作为 fail-closed physics gates。

### 10.5 单向耦合

刚性给定运动 CASE 不需要伪造结构迭代：

```text
prescribed state → surface frame → one aero proposal → gates → commit
```

它与 FSI 共享同一个 aero transaction，而不是共享不必要的结构 solver。

---

## 11. 各 CASE 的迁移方案

### 11.1 Meng 2025

现状：

- `meng2025_case.py` 依赖 Ptera 几何/运动对象；
- `run_meng2025_v4b_transfer.py` 是 V4B/transfer 路径；
- 尚未在当前 native mandatory V5M 中形成正式基线。

迁移：

1. 将真实几何与运动规律保留在 CASE 层；
2. 新建 `MengPrescribedKinematics`，直接输出左右翼 `SurfaceFrame`；
3. 固定 body pose；
4. 左右翼共享一个 body，使用对称 joint states；
5. 通过多翼面 V5M 计算相互诱导；
6. 保留 Fig.16 真实观测、单位转换和评分；
7. Ptera 只作为历史 reference，不作为新生产数据面。

验收：

- 几何面积、展长、弦长、关节轴和运动相位与 CASE manifest 一致；
- 固定机身下刚体运动速度解析对齐；
- 所有数值 CUDA float64；
- separated LEV/joint TEV/free wake 均激活且遵守释放条件；
- 周期均值与真实 Fig.16 按冻结口径评分；
- 不进行相位/幅值/偏置拟合。

### 11.2 Baik W1–W4

迁移：

- 保留当前 2D CUDA LDVM 生产后端；
- 封装为 `AerodynamicStepper`；
- 使用统一 `CaseDefinition`、时钟、transaction、provenance和gate；
- 不把水槽准二维 CASE 强制变成自由端 3D wing；
- 保留非简谐升沉定义和 W1–W4 相位数据。

验收：冻结同代码/GT/评分下宏 RMSE 不劣化超过舍入/已声明实现差异。

### 11.3 Yang 2025

迁移：

- 将四杆机构运动输出为 `PrescribedRigidSurfaceKinematics`；
- 从历史 Ptera GPU 路径迁移到 native multi-surface V5M；
- mandatory separated LEV/joint TEV/free wake；
- 保存每个安装迎角的周期均值和涡事务证据。

验收：六个安装迎角分别报告 lift/drag MAE，不允许只报汇总。

### 11.4 Izraelevitz–Scherer

迁移：

- 将椭圆翼面、升沉俯仰和相位偏置直接输出为 `SurfaceFrame`；
- 当前 post-hoc separation correction 必须改为 live solver state；
- 保留14-marker真实评分，不把重复marker误当独立工况。

验收：12个唯一运动条件、14个marker逐项保留，报告周期平均 CT MAE。

### 11.5 Rojratsirikul 2011

迁移：

- `Q16FixedFrameKinematics` 适配现有 Q16；
- 使用通用 V5M surface interface；
- 使用 `PartitionedStrongFSI`；
- 修正统计量、状态和退出码；
- E=2.2 为主分支，E=1.4 为校准敏感性分支；
- A16先完成事务与统计，再运行A10/A23。

验收：

- `max(mean z)` 与同窗口 mean Cn；
- block stationarity；
- zsd map 和空间模态；
- A16平均量与论文带；
- 约17°模态频率另建匹配论文定义的 CASE；
- A10/A23 同参数泛化；
- 不使用末端点作论文均值。

### 11.6 柔性 Meng 自由飞

构成：

- 一个 fuselage `RigidBodyState`；
- 左右翼 prescribed/controlled flap-pitch joints；
- 每翼一个 Q16 elastic subsystem；
- native multi-surface V5M；
- `PartitionedFreeFlightFSI`；
- gravity、actuator torque、aero loads和约束反力。

第一阶段不加入飞控器，只运行给定关节运动下的自由机身响应。控制器是后续独立层。

极限退化门：

1. 固定 body → Meng 风洞刚性/柔性 CASE；
2. Q16 刚度趋大 → 刚性 Meng；
3. 关节运动为零且固定 body → 常规柔性翼 FSI；
4. 固定四边 frame → 当前 Roj；
5. 关闭气动载荷但保留刚体惯性 → 解析/守恒刚体运动；
6. 左右翼对称运动 → 横侧向净力矩满足对称性；
7. 任何退化不得关闭 separated LEV，而是由物理释放条件自然决定是否释放。

---

## 12. 文件级修改地图

| 当前文件 | 动作 | 目标 |
|---|---|---|
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` | 分阶段拆分，先保留兼容入口 | 通用 surface 输入、独立 V5M state/solver/wake/separation |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py` | 抽出通用 coupling protocol | 不再 exact-type 绑定 Q16/native |
| `src/fluxvortex/warp_fsi/q16_structural_solver.py` | 保持数值内核，增加 adapter | `Q16DynamicsAdapter`、moving-boundary入口 |
| `src/fluxvortex/q16_work_conjugate_transfer.py` | 扩展 | body/joint/elastic统一 Jacobian 转置映射 oracle |
| `src/fluxvortex/warp_fsi/kernels_q16_transfer.py` | 扩展 CUDA 快路径 | 多翼面、body wrench、joint torque |
| `src/fluxvortex/warp_fsi/q16_aero_load_packet.py` | 泛化 schema | `SurfaceLoadPacket`/`GeneralizedLoadPacket` |
| `src/fluxvortex/warp_fsi/q16_lev_impulse_transfer.py` | 合并到全局 ledger | 避免 source-owned 与 surface-load owner 重复计力 |
| `platform/warp_vpm/flux_v5m_gpu.py` | 变成薄生产入口 | capability来自真实接口/测试，不再仅one-wing |
| `platform/forward_flight_benchmarks/meng2025_case.py` | 去除生产Ptera依赖 | 纯CASE数据+native kinematics builder |
| `platform/forward_flight_benchmarks/run_meng2025_v4b_transfer.py` | 保留历史只读 | 新增native runner，不覆盖历史结果 |
| `platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py` | 变成CaseRunner adapter | 修正统计、gate、provenance |
| `platform/forward_flight_benchmarks/rojratsirikul2011_q16.py` | 保留真值并修正oracle作用域 | A16均值与A17模态分离 |
| `platform/warp_vpm/bing_joint_ptera_gpu.py` | 历史reference/迁移来源 | 不作为最终多翼面自由飞核心 |
| `tests/test_flux_v5m_fsi_gpu_contract.py` | 保留旧facade门并重命名语义 | 避免冒充native Q16 FSI集成门 |
| 新增 `tests/test_q16_native_v5m_fsi_transaction_gpu.py` | 正式Roj网格集成测试 | parent/trial/formal/commit/rollback |
| 新增 `tests/test_v5m_circulation_impulse_ledger_gpu.py` | 正式涡事务门 | cull/truncate/merge守恒 |
| 新增 `tests/test_multisurface_v5m_gpu.py` | Meng/Yang真实几何 | 多翼面相互诱导和拓扑owner |
| 新增 `tests/test_body_joint_q16_work_transfer_gpu.py` | Meng真实几何+Q16 | 表面功=body+joint+elastic功 |
| 新增 `tests/test_freeflight_meng_reduction_gpu.py` | 正式退化门 | fixed/rigid/flexible/free-flight一致性 |

兼容入口至少保留一个迁移周期；旧runner继续调用新接口并输出旧schema兼容字段，同时增加新schema。

---

## 13. 结果与 provenance schema

统一结果必须区分：

```yaml
execution_status: completed | failed | aborted
numerical_status: converged | nonconverged
physics_gate_status: passed | failed
accuracy_gate_status: passed | failed | not_applicable
reproduction_status: passed | partial | failed
```

不得再用一个 `completed` 混淆执行成功和论文复现成功。

每次运行保存：

- clean git commit；
- 完整 patch digest 或确认无dirty source；
- CASE config digest；
- physics/numerical/execution config digests；
- GT和论文PDF哈希；
- GPU、CUDA、Torch、Warp版本；
- dtype/device；
- 每个正式开关；
- 统计窗的实际起止时间和样本数；
- analysis脚本路径和哈希；
- trial/formal/commit计数；
- 全部gate及失败原因。

如果 accuracy gate 失败，CLI 必须非零退出，除非用户显式运行 `--diagnostic-only`；diagnostic 文件必须标明不能作为正式复现结果。

---

## 14. 分阶段实施计划

### R0：真值和事务基线修复

目的：确保重构不会保留错误指标或不可见事务缺陷。

修改：

1. 修复Roj `max(mean z)`、统计窗、符号穿越和非零退出；
2. E1.4改为`post_hoc_calibrated_sensitivity`；
3. 增加真实 trial/commit计数；
4. 增加native Q16 FSI正式A16网格事务测试；
5. 增加LEV/TEV/尾迹环量冲量ledger；
6. 记录3D separation mask与source closure mask；
7. 在同一clean commit做100/101、300/301步正式A16反事实。

验收节点 H0：

- 指标独立重算一致；
- accuracy false → exit非零；
- trial失败后owner digest不变；
- 每步formal commit恰好一次；
- 删除/截断守恒误差进入结果；
- 不运行长时精度CASE，直到H0通过。

### R1：抽取 `SurfaceFrame`

目的：解耦V5M和Q16，不改变数值。

修改：

1. 提取通用geometry数据类；
2. `Q16NativeV5MSurface`变成provider adapter；
3. V5M solver接收provider输出；
4. 旧类名保留兼容包装。

验收节点 H1：

- Yamano正式geometry逐位/舍入级一致；
- Roj A16前8正式步rings/AIC/gamma/load/state digest一致；
- Q16结构结果一致；
- GPU驻留和性能无明显退化（目标≤3%）。

失败停止条件：无法说明任何数值变化来源，或科学输出超出冻结舍入容差。

### R2：统一V5M状态owner和保留策略

目的：修复RC2类问题并为多翼面准备。

修改：

- 唯一3D separation owner；
- LEV/TEV联合proposal；
- conservative retention policy；
- 全局 circulation/impulse ledger；
- load history owner显式化。

验收节点 H2：

- 当前无culling短窗与重构NoCulling结果一致；
- culling/merge前后环量/冲量误差低于预声明阈值；
- 100/101和300/301步无无解释跳变；
- separated LEV始终集成，释放仍由条件控制；
- 相同committed parent的重复proposal确定性一致。

### R3：统一刚性CASE运行层

目的：让三篇刚性论文和Meng使用统一CASE/runner/transaction。

修改：

- `FixedRigidSurfaceKinematics`；
- `PrescribedRigidSurfaceKinematics`；
- `OneWayPrescribedCoupling`；
- `LDVM2DStepperAdapter`；
- 统一observer/gate/result schema。

迁移顺序：

1. Baik adapter（不改2D内核）；
2. Izraelevitz single-wing native；
3. Yang机构运动native；
4. Meng左右翼multi-surface native。

验收节点 H3：

- Baik四CASE冻结评分保持；
- Yang六迎角完整迁移并记录mandatory涡证据；
- Izraelevitz 14 marker完整迁移；
- Meng Fig.16正式基线建立；
- 不允许历史Ptera结果冒充native新结果。

### R4：通用动力学与强耦合

目的：让现有Roj FSI不再被具体类锁死。

修改：

- `DynamicSubsystem`；
- `Q16DynamicsAdapter`；
- `PartitionedStrongFSI`；
- 通用Aitken/IQN-ILS；
- 多残差诊断；
- 全局WorldOwner事务。

验收节点 H4：

- 当前Roj A16前8步与旧native FSI路径一致；
- formal replay状态一致；
- failed iteration完整回滚；
- work/constraint/Kelvin/impulse gates全通过；
- 性能退化≤5%，否则先profile再继续。

### R5：机身SE(3)、关节与moving-root Q16

目的：形成自由飞所需动态组件。

修改：

- `RigidBodySE3Dynamics`；
- `PrescribedJointDynamics`；
- `MovingRootBoundary`；
- body/joint/Q16功共轭映射；
- 对称左右翼共享body state。

验收节点 H5：

- 使用Meng正式几何完成固定body退化；
- 刚度趋大退化到刚性Meng；
- 表面功与body+joint+elastic功闭合；
- 对称运动横滚/偏航力矩满足容差；
- 零气动下刚体动量/角动量符合外力矩ledger。

### R6：柔性扑翼自由飞强耦合

目的：组合body+joint+Q16+multi-surface V5M。

耦合未知量：

- body pose/velocity；
- joint state（若非完全给定）；
- Q16 state/velocity；
- aerodynamic load/state。

首版：给定关节运动，body和Q16参与强耦合。

验收节点 H6：

- 每步所有子系统只提交一次；
- 全局功、动量、环量、冲量ledger闭合；
- 固定body、刚性翼、零关节运动三种正式退化分别通过；
- 多周期无状态漂移、无隐藏CPU fallback；
- 结果包含body轨迹、姿态、wing deformation、joint work和气动功。

### R7：论文精度、泛化与性能

在架构和事务通过后运行：

1. Roj A16长时统计；
2. Roj A10/A23同参数泛化；
3. 约17°模态oracle；
4. Meng/Yang/Izraelevitz/Baik全部fresh基线；
5. 多翼面和Q16正式分辨率收敛；
6. CUDA profile、kernel fusion、graph capture和批处理。

性能优化不得早于physics parity；每项优化必须证明结果在预声明容差内。

---

## 15. 验证矩阵

### 15.1 组件数学门

- Q16形函数/刚体不变性/质量；
- ANS/EAS驻值、能量导数、切线；
- SE(3)姿态和动量；
- 运动学Jacobian；
- Jacobian转置功共轭；
- AIC、Mf1/Mf2、Biot–Savart oracle；
- Kelvin/impulse ledger。

### 15.2 正式CASE集成门

不使用Q4/Q9/Ptera toy：

- Roj A16正式15×30气动、5×10 Q16；
- Baik W1–W4真实运动；
- Yang Fig.11真实六迎角；
- Izraelevitz Fig.14真实12条件/14 marker；
- Meng Fig.16真实几何和运动。

短步门仍使用正式CASE和正式网格，只缩短时间长度，不改变科学对象。

### 15.3 科学门

- Roj：mean map、mean Cn、zsd、模态、stationarity；
- Baik：W1–W4逐CASE CL/CD相位RMSE及宏平均；
- Yang：六迎角lift/drag MAE；
- Izraelevitz：14-marker CT MAE；
- Meng：Fig.16各运动幅值平均lift/thrust；
- 自由飞：无对应实验前只宣称数值/物理契约验证，不宣称论文精度验证。

### 15.4 泛化门

- Roj A16定参数后A10/A23不得逐CASE调参；
- Yang安装迎角之间不得单点调参；
- Baik W1–W4共用模型合同；
- 自由飞退化到固定/刚性/无扑动极限。

### 15.5 性能门

- GPU-only counter和设备检查；
- 无科学tensor `.cpu().numpy()`参与后续数值；
- 数据常驻GPU；
- single-GPU单租户正式计时；
- R1/R4纯架构阶段性能退化阈值分别3%/5%；
- 优化后输出误差和kernel计数同时报告。

---

## 16. 优先级、依赖与并行边界

严格依赖：

```text
R0 truth/transaction
  → R1 surface interface
    → R2 aero ownership/retention
      → R3 rigid/meng migration
      → R4 generic FSI
        → R5 SE3/joint/moving root
          → R6 free-flight FSI
            → R7 long paper/performance
```

可并行但不能提前合并：

- R1接口设计与R0分析脚本修复；
- R3各CASE adapter在R2接口冻结后分别实现；
- R5刚体积分器和moving-root Q16可并行开发，但必须通过共同功共轭门后合并。

不可并行混做：

- R2物理事务修改与R1纯重构parity；
- Rayleigh阻尼实验与LEV/TEV事务根因实验；
- CUDA graph/fusion与核心状态owner迁移；
- 长时论文运行与仍在变化的状态schema。

---

## 17. 风险与控制

| 风险 | 后果 | 控制 |
|---|---|---|
| 大爆炸式重写 | 无法定位精度变化 | adapter/兼容入口/逐阶段parity |
| 抽象过宽 | 新框架比旧代码更复杂 | 只保留5个核心数据对象和小协议 |
| 把2D/3D强行统一 | 科学对象错误 | 统一runner，保留两个aero backend |
| 多翼面AIC显存/复杂度 | Meng/自由飞变慢 | 全局offset+分块CUDA，不改变物理 |
| 每翼复制body6DOF | 重复质量和错误力矩 | 每独立aircraft一个body owner |
| moving root漏惯性速度 | added-mass错误 | 完整位置/速度/加速度运动学链 |
| load重复计数 | FSI能量注入 | 唯一SurfaceLoad owner+J^T映射 |
| trial隐式修改cache | 回滚不完整 | proposal状态完备+digest+formal replay |
| culling/truncation丢环量 | 低频漂移/错误锁频 | conservative retention+impulse ledger |
| 架构重构夹带调参 | 无法归因 | 纯重构阶段参数/结果冻结 |
| 旧handoff错误结论复用 | 假复现 | 新result schema+accuracy fail closed |
| GPU性能优化过早 | 快速得到错误结果 | physics gate通过后才profile |

---

## 18. 提交和交付纪律

每个阶段至少一个独立提交，建议：

1. `fix(roj): correct observables, gates and provenance`
2. `test(fsi): add formal Roj A16 native transaction gate`
3. `fix(v5m): add circulation-impulse retention ledger`
4. `refactor(kinematics): extract SurfaceFrame without numerical change`
5. `refactor(v5m): unify separation owner and aero proposal state`
6. `refactor(cases): add common case/observer/result protocols`
7. `feat(v5m): migrate rigid multi-surface Meng/Yang/Izraelevitz`
8. `refactor(fsi): generalize WorldOwner and coupling strategy`
9. `feat(dynamics): add SE3 body, joints and moving-root Q16`
10. `feat(freeflight): compose flexible Meng free-flight FSI`

每个提交必须附：

- changed contract；
- unchanged contract；
- exact test command；
- exact artifact；
- numerical parity/expected scientific change；
- GPU evidence；
- known limitation。

禁止把数值重构、物理修改、参数修改和性能优化混在同一提交。

---

## 19. Definition of Done

### 19.1 Meng固定机身扑翼

- native V5M而非V4B/Ptera生产路径；
- 左右翼multi-surface相互诱导；
- mandatory separated LEV/joint TEV/free wake；
- Fig.16真实评分和完整provenance；
- 所有数值GPU float64。

### 19.2 三个刚性论文CASE

- Baik、Yang、Izraelevitz均使用统一CaseRunner和结果schema；
- Baik保留2D科学后端；
- Yang/Izraelevitz迁移到当前mandatory 3D V5M；
- 所有CASE分别报告真实指标，不合并虚假总分；
- fresh结果绑定当前源码hash。

### 19.3 当前Roj FSI

- Q16/native FSI事务直接集成门；
- LEV/TEV/wake删除守恒；
- 正确统计定义；
- A16长时stationarity；
- A10/A23同参数泛化；
- accuracy门失败时非零退出。

### 19.4 柔性扑翼自由飞

- 每个独立aircraft一个6DOF body owner；
- 多翼面通过关节共享body；
- 每翼可挂Q16；
- surface load唯一并功共轭映射到body/joint/elastic；
- body+joint+Q16+V5M在同一global transaction中formal commit；
- 固定/刚性/无扑动退化门通过；
- 多周期GPU运行稳定；
- 在缺乏实验GT时仅报告契约验证和预测，不声称实验复现。

---

## 20. 第一批实际开发切片

### Slice A：结果真值修复

涉及：

- `rojratsirikul2011_q16.py`；
- `reproduce_rojratsirikul2011_q16_flux_v5m_native.py`；
- 新增tracked analysis；
- result schema/status。

完成标准：独立重算与runner完全一致，旧错误报告不再能生成PASS。

### Slice B：native FSI正式事务门

涉及：

- `q16_flux_v5m_native_fsi.py`；
- `q16_flux_v5m_native.py`；
- 新增正式A16前1/8步GPU集成测试。

完成标准：parent不漂移、正式重放、失败回滚、一次提交、trial计数正确。

### Slice C：LEV/TEV/尾迹守恒owner

涉及：

- `q16_flux_v5m_native.py`；
- particle field；
- wake retention；
- circulation/impulse ledger。

完成标准：100/101与300/301正式A16反事实无未记账状态跳变。

只有A/B/C全部通过，才进入 `SurfaceFrame` 纯重构。这样不会把当前未闭合的气动事务包装成“统一框架正确性”。

---

## 21. 最终路线判断

当前真正的瓶颈不是Q16需要重写，也不是FSI predictor没有推进尾迹；瓶颈是：

1. CASE真值、统计和运行状态语义未统一；
2. V5M几何接口绑定Q16具体类型；
3. 3D LESP/source bank存在双状态owner；
4. 粒子/尾迹有限保留没有完整守恒事务；
5. 刚性、FSI和自由飞尚未共享同一WorldState和proposal/commit协议；
6. 多翼面和机身六自由度尚无统一功共轭映射。

因此最小收敛路线是：

```text
修真值/事务 → 抽SurfaceFrame → 修V5M状态owner →
迁移刚性CASE/Meng → 泛化FSI → 加SE3/关节/moving-root →
组合柔性扑翼自由飞 → 长时论文/性能验收
```

该路线最大限度复用现有Q16、V5M公式块、CUDA核和formal replay，同时避免继续形成MENG、刚性CASE、Roj FSI和自由飞四套互不兼容的专用底盘。
