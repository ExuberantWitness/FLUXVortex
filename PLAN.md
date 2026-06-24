# 当前主线 Plan:从单翼切片 → 整机 MAP-Elites co-design(发 AST)

> 目标(用户定):**基于元强化学习 + 刚柔分布 + 质量分布 + 设计参数的、基于 MAP-Elites 的 co-design**,
> 成果可发表到《Aerospace Science and Technology》一区 TOP 期刊。
> 本节是 2026-06 的诚实差距审计 + 待办路线图;下方"ANCF统一壳单元"等是更早的子计划,作为地基保留。

## 诚实现状审计(2026-06-22)

已完成的只是一个**单翼切片**:一块 0.4×0.3 m、根部夹死的平板柔性翼(6×4=24 ANCF 壳单元),
纯附着 ring-UVLM(无 LEV、无粘性),被**初速踢一脚**(非真实阵风),~6 ms 瞬态窗;设计 = 刚度/质量的
**3 点展向样条(非逐单元)**;控制 = **amortized SHAC 增益(非真 meta-RL)**。在此切片上验证了:可微强耦合
PC 前向(逐位 k=2/4/8)、控制器梯度精确(FD 7e-4)、设计梯度闸门(抓修 dk/dθ bug,4%)、72-niche 档案
(182/182 稳定,跨 seed 复现柔尖+尖轻最优)。**离 plan 整机愿景很远——本体与气动均降标。**

审计答复(对应用户 6 问):①逐单元:能力有(gE/gR 已验证)、co-design 实际用 3 点样条,**否**。
②meta-RL:**否**,是 amortized design-conditioned SHAC。③翼面/尾翼/舵面设计参数:**全无**(翼面写死矩形)。
④前缘涡:co-design 里**无**(纯附着 UVLM)。⑤数据校准:仅在**另一个**鸟级翼上做了量级 2× anchor,
**未与 co-design 模型打通、非逐点校准**。⑥plan:**远未完全执行**,只完成一个切片。
仓库存在更全模块(`aircraft_geom`/`lev_dvm`/`sectional_lev`/`aircraft_multibody`/`uvlm_aircraft`/`flex_aircraft`),
但 co-design **一个都没接入**。方向已定:**直接把 co-design 接到更全模型**。

## 整机基底现状(已读接口,2026-06-22;**未核实正确性**,仅知调用)

| 模块 | 关键接口 | 提供 | 可微? |
|---|---|---|---|
| `aircraft_geom.py` | `WingDesign`(span_profile/taper, stiffness_scale_fn, mass_scale_fn)、`TailDesign`、`Aircraft`(wing_dims/tail_dims/wing_components(sgn)/tail_components(sgn)) | 参数化 planform + 尾翼几何 | 几何(numpy) |
| `uvlm_aircraft.py` | `MultiSurfaceUVLM(surfaces).solve(poses,twists,V_inf)`、`build_aircraft_surfaces(ac,nc_box,nc_flap,ns_seg,nc_tail)`、`RigidSurfaceUVLM` | 多面 UVLM(翼盒+舵面+尾翼) | 前向 |
| `flex_aircraft.py` | `FlexAircraft(chord,span,nc,ns,flap_hz,trim_aoa_deg,...).step_window(wind)`;FlapEntry(elastic)+FlapUVLMProvider+WindowPredictorCorrector;状态 x/q/v/om;att_kp/kd | 6-DOF 刚柔扑翼 + PC 耦合 + 阵风入口 | 前向 |
| `aircraft_multibody.py` | `build(ac,spring_ke,spring_kd,surf_ke,surf_kd)`、`add_servo_body`、`ACIndex.n_actuators`、`wing_stiffness_profile` | Featherstone 多体 + 伺服扑动铰 + 14 舵面 | **wp.Tape ✅** |
| `flight_env.py` | `FlightEnv`、`rollout(env,policy)` | RL 飞行环境 | **grad ✅** |
| `aircraft_flight.py` | `AircraftFlight`(docstring:"NN + MOME co-design 架在其上") | 飞行循环(MOME 预留位) | 前向 |
| `p0_resonant_freeflight.py` | `build_model(spring_ke,spring_kd,requires_grad)`、`rollout`、`grad_tape_vs_fd` | 共振自由飞 + **tape 梯度已对 FD** | **wp.Tape ✅** |
| `sectional_lev.py` / `lev_dvm.py` | `SectionalLEV(surfs,lesp_crit,n_chord)`、`LDVM2D(U,c,n,lesp_crit).step(alpha,dalpha)`;`_validate`(45°+扑动脱 LEV) | 前缘舵面 LESP-LDVM LEV | 前向 |

## 细化待办路线图(A 为关键阻塞;红线:每件先对金标/FD 验证再用)

### A. 接更全模型(本体 + 气动)— 🔴 关键阻塞,先做
- [ ] **A1.1** 选基底:以 `flex_aircraft.FlexAircraft`(已有 6-DOF 刚柔 + PC + flap_hz + wind)为评估环境主体;读 `step_window(wind)`、`AircraftFlight` 的 MOME 预留接口,确认 rollout→指标通路
- [ ] **A1.2** 写 `FullModelEnv` 适配层:封装 reset / design→model / rollout→(抗风,效率),接口与现 `cq.Env` 同构,使 `fsi_codesign_qd`/`fsi_shac_controller` 能"换底"而 QD/控制逻辑不动
- [ ] **A1.3** 设计→模型:`aircraft_geom.WingDesign.stiffness_scale_fn/mass_scale_fn` → `flex_aircraft` 的 ANCF 翼 `set_distribution`;确认每单元 (E,ρ) 受设计驱动、与 `aircraft_multibody.wing_stiffness_profile` 一致
- [ ] **A1.4** 指标改造:从整机 rollout 取抗风(姿态/极限环偏差,见 D4)+ 效率(功率,见 D5),替换"6 ms 变形能量 J"
- [ ] **A1.5** 🔴红线:整机 env 单次 rollout 跑通、有限、量级合理;`flex_aircraft.verify()`/`aircraft_flight.verify()`/`p0_resonant_freeflight.main()` 金标不回归
- [ ] **A2.1** 暴露几何设计变量:根梢比 taper、展弦比 span/chord、弦长 chord(`WingDesign`)、尾翼上下反/掠/展弦比/根梢(`TailDesign`)、舵面比例(`build_aircraft_surfaces` 的 nc_flap/ns_seg)
- [ ] **A2.2** DesignMap:几何参数 → `Aircraft.wing_dims/tail_dims/*_components` → 重网格(`build_aircraft_surfaces` 多面 + ANCF 翼重建);几何变自动重映射设计场
- [ ] **A2.3** 🔴红线:重网格后面积/展长/弦长对解析一致;UVLM 板↔ANCF 单元角点对应不错位(力转移守恒)
- [ ] **A3.1** 把 `SectionalLEV(surfs,lesp_crit,n_chord)` 挂到前缘舵面条带,与 `MultiSurfaceUVLM` 附着解混合保真(主翼 bound 附着 + 前缘舵面 LDVM 脱 LEV)
- [ ] **A3.2** LDVM 截面力 → 条带 → ANCF 节点力(一致力转移,虚功守恒);与附着 UVLM 力叠加不重复计
- [ ] **A3.3** 🔴红线:`sectional_lev._validate`(45°+3Hz 扑动脱 LEV 加升力)通过;`lev_dvm` 对 Ramesh 2014 / DVM.m 金标(归入 F3)
- [ ] **A4.1** 截面极曲线粘性/型阻(复用螺旋桨 CCBlade 截面气动或 2D polar 代理),按条带 Re→C_D,可微极曲线
- [ ] **A4.2** 型阻进功率/效率口径(诱导 + 型阻);🔴红线:巡航 L/D 落 Re~1e5 文献量级

### B. 逐单元 刚柔 + 质量
- [ ] **B1.1** 设计参数化:`stiffness_scale_fn/mass_scale_fn`(展向场)→ 逐单元数组(可选弦向);保留光滑 + 边界(min gauge)约束
- [ ] **B1.2** 与 A2 几何重网格兼容:单元数随几何变,设计向量维度自适应
- [ ] **B2.1** 整机可微通路:把设计 (E,ρ) 接进 `aircraft_multibody`/`p0_resonant_freeflight` 已验证的 wp.Tape(requires_grad)
- [ ] **B2.2** 逐单元 ∂J/∂(E,ρ) 经整机 tape 反传;🔴红线:对 FD 验证 rel<1e-2(类比单翼 `verify_pc_grad`)
- [ ] **B2.3** 若整机 PC 耦合非端到端可微 → 用 IFT 伴随(复用单翼 PC 伴随数学 `coupled_unsteady_pc_grad`)或检查点 BPTT 扛长 rollout

### C. 真 meta-RL(替换 amortized 控制器)
- [ ] **C1.1** 策略网:Takens n=20 步观测堆叠 → 前馈网,替换 `GainPolicy` 线性增益;接 `flight_env.rollout(env,policy)` 的 policy 接口
- [ ] **C1.2** 适应机制:RL²/上下文为主(从交互推断形态、**不读设计参数**);MAML/PEARL 作对比
- [ ] **C2.1** 观测:IMU(姿态/角速度)+ 扑动相位/角 + 离散翼应变(少数传感器,à la Kim 2024)+ 风加速度/滚转线索(à la Reddy 2016)
- [ ] **C2.2** 可插拔传感器钩子:理想 → 噪声/延迟/偏置 + 域随机化(sim-to-real)
- [ ] **C3.1** PPO 起步(model-free):GPU 多环境向量化采样(Isaac-Gym 式),接 `FlightEnv`;🔴红线:策略学到非平凡抗风
- [ ] **C3.2** SHAC 升级:torch policy ↔ Warp env 梯度桥;长 rollout 检查点;🔴红线:∂/∂θ vs FD
- [ ] **C4.1** 控制空间:差动扑动电机力矩(`aircraft_multibody` 伺服铰,`ACIndex.n_actuators`)+ 14 舵面偏角
- [ ] **C4.2** 作动器动态 + 限幅(带宽/速率/力矩饱和)+ 电机电流模型(复用 DD 240525C/CORE_OPT_MCD,对标验证)

### D. 飞行物理(夹死悬臂 → 自由飞扑翼)
- [ ] **D1.1** 6-DOF 自由飞:用 `FlexAircraft`/`AircraftFlight` 刚体 + 柔性附件;30 m 高 / 10 m·s @45° 起,policy 拉平到 5–8 m·s 巡航(无显式配平)
- [ ] **D1.2** 发散 rollout 优雅截断为有限惩罚(QD/梯度可用,非 NaN 崩)
- [ ] **D2.1** 翼根扑动铰 = 伺服 + 扭簧(`aircraft_multibody` spring_ke;`p0_resonant_freeflight` 已有共振自由飞 + tape);弹簧刚度 = 设计变量(设共振频率)
- [ ] **D2.2** policy 命令电机力矩 → 扑动轨迹/频率/幅值/flap-glide 涌现(非预设运动学)
- [ ] **D3.1** 统一 Disturbance 抽象,接 `flex_aircraft.step_window(wind)`:1-cos → Dryden/von Kármán → 3D 湍流场
- [ ] **D3.2** 单轨迹注入(前段测效率、阵风段测抗风,一次 rollout 出两指标)
- [ ] **D4.1** 抗风指标 = 回扑动周期极限环(相位平均偏差/Poincaré 回归),替换定点变形能量;可微信号
- [ ] **D5.1** 效率 COT:功率模型 P=(P_ind+P_pro+P_par+P_iner)/η(η≈0.85,Zhong&Xu 验证)+ 电机/14 舵面功率
- [ ] **D5.2** COT = 总功率/(克重·速度),可微(从 rollout 算)→ SHAC 反传

### E. MAP-Elites 升级
- [ ] **E1** MOME 多目标 QD:quality=(抗风, 效率) Pareto,替换单目标 −J(复用 `codesign_qd_coupled` Archive/emitter 升 MOME)
- [ ] **E2** 行为描述子轴:动力系统轴(电机型号 + 弹簧刚度,如共振调谐度)× 翼面轴(几何 + 刚柔分布,如柔度/展弦比)
- [ ] **E3** 可实现性约束:min gauge / 刚度光滑 / 载荷下不破坏不屈曲(强度 + 屈曲校核),QD 以可行域处理
- [ ] **E4** G3 CUDA-graph 加速(整机 FSI 评估规模使能;本地 DEV → A100 多环境);复用已验证 record-once-reuse 协议

### F. 验证 / 校准(发表硬门)
- [ ] **F1** Goland 颤振复现 ~137 m/s(对 `benchmark_goland.py` 140.2)+ 收敛/网格无关研究
- [ ] **F2** 整机扑翼对真实飞行器**逐点**校准(HIT-Hawk Zhong&Xu 2022:span 1.6–1.8m/~520g/≤3Hz/5–10m·s;E-Flap):配平 + 功率曲线 + 攻角域,**非"量级 2×"anchor**
- [ ] **F3** LEV 模型对金标(`lev_dvm` vs Ramesh 2014 JFM / 用户 DVM.m / Ptera)
- [ ] **F4** 对标 SOTA:SHARPy(非可微/非GPU)、FLOWUnsteady(Julia/非可微)、伴随气弹优化
- [ ] **F5** 🔴打通:co-design 用的模型 = 校准用的模型(当前断裂:co-design 0.3 m 平板 vs 校准 1.6 m 鸟级翼)
- [ ] **F6** 三红线贯穿:整机 solver vs 金标逐位 / tape 梯度 vs FD / 批量一致

### G. 论文(AST)
- [ ] **G1** 重写论文:claim 从"单翼切片"升到"整机 刚柔+质量+几何+控制 co-design",所有数字来自整机模型
- [ ] **G2** 杀手图:MOME archive 进化(多样形态 × 控制铺开 抗风×效率前沿)
- [ ] **G3** 诚实保真边界界定(核心高保真 blade-rate/fp64/自由尾迹/PC + 周边原则性近似,各有依据)

## 依赖图(关键路径)
```
A1(总闸) → A2/A3/A4(模型完整度) → B1/B2(逐单元设计梯度) ┐
A1 → D1/D2/D3(飞行+扑动+阵风) → D4/D5(抗风/效率指标) ────┼→ E1/E2/E3(MOME co-design) → E4(A100 规模)
A1 → C1/C2 → C3/C4(meta-RL 控制) ───────────────────────┘                              ↓
F1/F3(单项金标,可并行) → F2/F5(整机校准+打通) → F4(对标) → G1/G2/G3(论文)
```
**最小可发表闭环(MVP)**:A1 → (A2 几何 + B1 逐单元) → D1+D3+D4+D5 → C3(PPO) → E1+E2 → F2+F5 → G1。
LEV(A3)、SHAC(C3.2)、桨/blade-rate、RL²(C1.2 高阶)可作迭代2 增强。

---

# (子计划,地基)Plan: ANCF统一壳单元 + BST基准 气动弹性实验矩阵

## Context

**核心目标**: 基于ANCF(绝对节点坐标公式)开发统一壳单元, 通过梯度降阶实现膜/壳/梁统一, 用于仿生气动弹性翼分析。

**ANCF核心优势**:
- 用位置+梯度向量代替转角自由度 -> 质量矩阵恒定(常数)
- Green-Lagrange应变 -> 几何非线性精确
- 梯度降阶 -> 同一单元可配置为膜/壳/梁
- C1连续 -> 气动载荷传递友好
- 无转角奇异 -> 大变形无问题

**技术路线**: ANCF壳(主) + BST壳(基准对比) + PM-force(辅助对比)

---

## 轴1: 结构单元 (4种)

| # | 结构单元 | 文件 | DOF/node | 力计算 | 状态 |
|---|---------|------|----------|--------|------|
| 1 | ANCF-shell | **待开发** `ancf_shell.py` | 9或12 | Green-Lagrange应变 + PK2应力 | **核心新开发** |
| 2 | BSTShell | `bst_shell.py` | 3 | CST膜+IBM弯曲+扭转弹簧 | 现有(基准) |
| 3 | PM-XPBD | `particle_mesh.py` | 3 | 位置约束投影 | 现有 |
| 4 | PM-force | `particle_mesh.py` | 3 | 弹性力计算 | 现有 |

## ANCF壳单元设计

### 节点坐标 (2种方案)

**方案A: 全参数化 (12 DOF/node)**
```
e_node = [r, r_x, r_y, r_z]  -- 位置 + 3个梯度向量
r = 位置向量 (3)
r_x = dr/dx1 面内梯度1 (3)
r_y = dr/dx2 面内梯度2 (3)
r_z = dr/dz  厚度梯度 (3)
```
- 优势: 自然包含膜+板+厚度变化, 可退化梁
- 劣势: DOF较多, 泊松锁闭需要EAS

**方案B: 梯度缺省 (9 DOF/node)**
```
e_node = [r, r_x, r_y]  -- 位置 + 2个面内梯度
```
- 优势: DOF较少, Kirchhoff-Love薄壳
- 劣势: 无厚度变形, 退化梁时精度受限

**推荐: 方案A (全参数化)**
- 符合"统一膜/梁/板"目标
- 通过梯度降阶自动退化为方案B

### 统一模式切换

| 模式 | 实现方式 | DOF/node | 用途 |
|------|---------|----------|------|
| 膜 | 弯曲刚度置零 (D_bend=0) | 12 (但只用面内) | 蒙皮 |
| 壳 | 默认 (膜+弯曲) | 12 | 完整翼面 |
| 梁 | 窄条mesh, 约束r_y梯度, 保留r_x(轴向)+r_z(扭转) | 等效6-8 | 翼梁 |

### 形函数 (4节点四边形)

```
r(xi,eta) = sum_i S_i(xi,eta) * e_i

S_i = 形函数矩阵 (3x12), 含双三次Hermite插值
xi, eta = 参数坐标 [-1,1]
```

### 应变-应力

```
Green-Lagrange应变: epsilon = (J^T * J - I) / 2
J = [r_x, r_y, r_z] = 变形梯度
PK2应力: S = D * epsilon  (D = 弹性本构矩阵)
弹性力: Q_e = integral_V (B^T * S) dV
```

### 质量矩阵 (常数!)

```
M = integral_V rho * S^T * S dV = CONSTANT
只计算一次, 每步复用
```

### 锁闭缓解

| 锁闭类型 | 缓解方法 | 实现 |
|---------|---------|------|
| 膜锁闭 | ECM (Enhanced Continuum Mechanics) | 分离膜/弯曲应变, 修正本构 |
| 剪切锁闭 | ANS (Assumed Natural Strain) | 边中点采样, 双线性插值 |
| 泊松锁闭 | EAS (Enhanced Assumed Strain) | 增加内部应变参数, 静力凝聚 |
| 厚度锁闭 | 假设横向应变 | 约束厚度方向应变 |

### 参考实现

- **Project Chrono** (C++/Python): `ChElementShellANCF_3423` -- 4节点梯度缺省壳, ANS+EAS
- **Exudyn** (Python/C++): ANCF beam/cable, 可扩展壳
- **MATLAB ANCF shell**: 4节点四边形, 数学推导完整

---

## 轴2: 求解器 (9隐式 + 2显式 = 11)

| # | 求解器 | 积分格式 | K策略 |
|---|--------|---------|-------|
| a | Newmark-fd | beta=1/4, gamma=1/2 | fd_direct |
| b | Newmark-jfnk | beta=1/4, gamma=1/2 | jfnk |
| c | Newmark-ibm | beta=1/4, gamma=1/2 | ibm_precond |
| d | Euler-fd | 后退Euler | fd_direct |
| e | Euler-jfnk | 后退Euler | jfnk |
| f | Euler-ibm | 后退Euler | ibm_precond |
| g | GenAlpha-fd | rho_inf=0.8 | fd_direct |
| h | GenAlpha-jfnk | rho_inf=0.8 | jfnk |
| i | GenAlpha-ibm | rho_inf=0.8 | ibm_precond |
| j | Explicit VV | Velocity-Verlet | N/A |
| k | XPBD | 位置约束 | N/A |

## 相乘: 4单元 x 11求解器

| 单元 \ 求解器 | a:NM-fd | b:NM-jfnk | c:NM-ibm | d:EU-fd | e:EU-jfnk | f:EU-ibm | g:GA-fd | h:GA-jfnk | i:GA-ibm | j:VV | k:XPBD |
|--------------|:-------:|:---------:|:--------:|:-------:|:---------:|:--------:|:-------:|:---------:|:--------:|:----:|:------:|
| ANCF-shell   | Y       | Y         | Y        | Y       | Y         | Y        | Y       | Y         | Y        | Y    | -     |
| BSTShell     | Y       | Y         | Y        | Y       | Y         | Y        | Y       | Y         | Y        | Y    | -     |
| PM-XPBD      | -       | -         | -        | -       | -         | -        | -       | -         | -        | -    | Y     |
| PM-force     | Y       | Y         | Y        | Y       | Y         | Y        | Y       | Y         | Y        | Y    | -     |

**有效组合**: 3x9(隐式) + 3(VV) + 1(XPBD) = **31种**

## x2耦合: 31 x 2 = 62实验

| 单元 | 求解器数 | x耦合 | 小计 |
|------|---------|-------|------|
| ANCF-shell | 9隐+1VV=10 | x2 | 20 |
| BSTShell | 9隐+1VV=10 | x2 | 20 |
| PM-XPBD | 1 | x2 | 2 |
| PM-force | 9隐+1VV=10 | x2 | 20 |
| **总计** | | | **62实验** |

62实验 x 3速度(V=10,30,50) = **186次运行**

---

## 核心代码结构

### 新文件

| 文件 | 内容 | 行数估计 |
|------|------|---------|
| `src/fluxvortex/ancf_shell.py` | ANCF壳单元 (形函数+应变+力+质量) | ~600行 |
| `src/fluxvortex/structural_adapter.py` | `StructuralAdapter` + 3个适配器 | ~200行 |
| `src/fluxvortex/implicit_solver.py` | `ImplicitSolver` (从BSTImplicitGPU泛化) | ~400行 |
| `src/fluxvortex/ancf_aero_coupling.py` | ANCF + UVLM耦合器 | ~300行 |
| `tests/run_experiment_matrix.py` | 统一测试框架 | ~500行 |

### ANCF-shell实现步骤

1. **形函数**: 4节点四边形, 双三次Hermite插值, 12 DOF/node
2. **应变计算**: Green-Lagrange应变 `epsilon = (J^T J - I)/2`
3. **弹性力**: `Q_e = integral_V (B^T S) dV`, 数值积分 (2x2 Gauss)
4. **常数质量矩阵**: `M = integral_V rho S^T S dV`, 初始化时计算一次
5. **锁闭缓解**: ECM (膜锁闭) + ANS (剪切锁闭)
6. **模式切换**: 膜(D_bend=0), 壳(默认), 梁(约束梯度+窄条)
7. **BC**: 绝对坐标, 固定节点直接冻结所有12 DOF
8. **GPU加速**: Warp核函数加速力计算

### ANCF + UVLM耦合

- UVLM面板角点 -> ANCF节点位置 (直接对应)
- 气动力 -> ANCF节点力 (一致力转移, 用形函数积分)
- ANCF位移 -> UVLM面板变形 (节点位移直接映射)
- 表面法向量: `n = r_x x r_y` (从梯度向量直接计算)

---

## 实施顺序

1. **ANCF壳单元** -- `ancf_shell.py` (形函数+应变+力+质量+锁闭缓解)
2. **ANCF单元测试** -- 悬臂梁弯曲 vs 解析解, 膜拉伸 vs 解析解
3. **StructuralAdapter** -- BSTShellAdapter + ANCFShellAdapter + PMForceAdapter
4. **ImplicitSolver** -- 从BSTImplicitGPU泛化, 支持任意适配器
5. **ANCF + UVLM耦合** -- `ancf_aero_coupling.py`
6. **统一测试框架** -- `run_experiment_matrix.py`
7. **Phase 1**: BSTShell x 10 x 2 = 20实验 (验证ImplicitSolver泛化)
8. **Phase 2**: ANCF-shell x 10 x 2 = 20实验 (ANCF验证)
9. **Phase 3**: PM-force x 10 x 2 + PM-XPBD x 2 = 22实验
10. **汇总**: 62实验 x 5速度对比表

---

## Goland Wing 参数

```
chord=1.8288m, semi_span=6.096m, EI=9.773e6 N*m^2, GJ=0.988e6 N*m^2
h=0.3m, Ex=Ey=2.16e9 Pa, G_xy=6.01e7 Pa, nu_xy=0.3
Mesh: 4x8 (45 nodes / 135 DOF for BST, 540 DOF for ANCF)
dt=0.003s, wake=20-30 chords, V=[10,30,50] m/s
```

## 验证清单

- [ ] ANCF壳: 形函数+Green-Lagrange应变实现
- [ ] ANCF壳: 悬臂梁均布载荷 vs 解析解 (< 5%误差)
- [ ] ANCF壳: 膜模式(纯拉伸) vs 解析解
- [ ] ANCF壳: 锁闭缓解验证 (膜锁闭消失)
- [ ] ANCF壳: 常数质量矩阵验证 (能量守恒)
- [ ] `StructuralAdapter`: ANCF + BST + PM-force 各自通过常力验证
- [ ] `ImplicitSolver`: 泛化版本对每个适配器 Newmark-fd收敛
- [ ] ANCF + UVLM耦合: 力转移能量守恒验证
- [ ] BSTShell: 9隐式精度一致 + VV发散(已知基准)
- [ ] ANCF vs BST: 同mesh下梢位移精度对比
- [ ] PM-force + PM-XPBD: 与理论对比
- [ ] 全部62实验 x 3速度(V=10,30,50)完整对比表
