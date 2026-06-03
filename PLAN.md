# Plan: ANCF统一壳单元 + BST基准 气动弹性实验矩阵

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
