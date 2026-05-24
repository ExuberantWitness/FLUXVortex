# IDEA REPORT: GPU-Accelerated Aeroelasticity for Vortex Method Solvers

**Date**: 2026-05-24
**Pipeline Stage**: 1 — Idea Discovery
**Project**: FLUXVortex (E:\DATA\vscode\VLM\FLUXVortex)
**Direction**: 给当前气动力计算方法翼面引入弹性，模拟碳纤维材料，进而模拟气动弹性问题

---

## 1. 研究背景与已有成果

### FLUXVortex 已有能力

| 能力 | 状态 | 细节 |
|------|------|------|
| UVLM 环形涡格法 | ✅ 已完成 | 基于 PteraSoftware，涡环面板气动力求解 |
| 混合面板-粒子尾涡 | ✅ 已完成 | 近场 N_keep 行面板 + 远场 VPM 粒子 |
| GPU 加速 (NVIDIA Warp) | ✅ 已完成 | Biot-Savart 全部 GPU 化，2.4-11× 加速 |
| 自由尾涡 (rVPM) | ✅ 已完成 | RK3 + reformulated VPM stretching |
| 扑翼/沉浮翼验证 | ✅ 已完成 | CL corr=0.9998 vs PteraSoftware |

### 文献调研关键发现

| 来源 | 核心贡献 | 与本工作的关系 |
|------|---------|--------------|
| **Murua, Palacios & Graham (2012)** | UVLM + 非线性梁耦合 aeroelasticity 综述 (304 引用) | 理论框架基础 |
| **SHARPy (Imperial College)** | UVLM + 几何精确梁，CPU only | 最接近的竞品，无 GPU |
| **OpenAeroStruct (U. Michigan)** | VLM + 梁FE，纯 Python，MDO 优化 | 轻量参考实现 |
| **Dussler & Palacios (2023)** | 增强 UVLM + 非线性梁 + 机身气动 | 最新进展 |
| **Schubert et al. (2024)** | 拟牛顿加速 UVLM-结构耦合 | 算法加速参考 |
| **Pudasaini et al. (2026)** | 变体翼 UVLM aeroservoelasticity | 最新应用方向 |

### 关键空白 (The Gap)

1. **无 GPU 加速的 aeroelastic UVLM+VPM 求解器** — FLUXVortex 已有 GPU UVLM+VPM 气动，但无结构耦合
2. **无 PteraSoftware aeroelastic 扩展** — PteraSoftware 纯气动，FLUXVortex 将是首个
3. **无复合材料翼面在中等保真度涡方法中的建模** — VABS 等工具存在但未与 GPU UVLM 集成

---

## 2. 候选方案 (按推荐排名)

### Idea 1: GPU-Accelerated UVLM + Euler-Bernoulli Beam Aeroelastic Solver

**假说**: 将 GPU 加速的 UVLM 气动求解器与 Euler-Bernoulli 梁 FE 结构求解器耦合，实现中等保真度 aeroelastic 仿真，并在速度上比 CPU 方案 (如 SHARPy) 快 10-100×。

**方法**:
1. 实现 1D Euler-Bernoulli 梁 FE（弯曲 EI + 扭转 GJ）
2. 分区 (partitioned) 耦合：UVLM 气动力 → 梁节点载荷 → 梁变形 → 更新翼面几何
3. 梁节点位于 WingCrossSection 位置
4. 力映射：展向积分面板力 → 截面升力/力矩
5. 几何更新：梁挠度/扭转 → WingCrossSection Lp_Wcsp_Lpp + angles → 重新网格化
6. 欠松弛保证收敛

**验证**: Goland Wing 颤振边界预测（矩形翼，均匀梁特性，大量参考解）

**预期效果**:
- 颤振速度预测误差 < 10% vs 参考解
- 计算速度 10-100× 快于 CPU UVLM+梁 (如 SHARPy)
- 单步耗时目标：0.1-1s (GPU) vs 1-10s (CPU)

**新颖性**: **高** — 首个 GPU 加速的 aeroelastic UVLM+VPM 求解器。NVIDIA Warp kernel 同时用于气动 Biot-Savart 和梁 FE 组装/求解。

**可行性**: ★★★★★ — Euler-Bernoulli 梁是最简单的结构模型，UVLM 气动力已完整可用

**Pilot**: 实现 Goland Wing 线性颤振分析，对比 SHARPy/OpenAeroStruct 参考解

---

### Idea 2: Hybrid Panel-Particle Wake Effects on Aeroelasticity

**假说**: FLUXVortex 的混合尾涡（近场面板 + 远场 VPM 粒子自由卷起）对 aeroelastic 行为有显著影响，特别是在大变形和颤振边界附近。

**方法**:
1. 在 Idea 1 基础上，将混合尾涡引入 aeroelastic 耦合
2. 对比纯涡环尾涡 (prescribed) vs 混合尾涡 (VPM free wake) 的颤振预测
3. 研究尾涡卷起对翼面气动反馈的影响
4. Timoshenko 梁（剪切变形）+ 几何非线性

**验证**: Pazy Wing 大变形静气动弹性 + Goland Wing 颤振（两种尾涡对比）

**预期效果**:
- 混合尾涡在颤振边界附近提供更准确的气动反馈
- 自由尾涡卷起可能影响颤振速度预测 ±5-15%
- 几何非线性对大展弦比翼至关重要

**新颖性**: **高** — 混合尾涡对 aeroelastic 行为的影响从未被研究过。现有工作 (SHARPy, MACH) 均使用纯涡环尾涡。

**可行性**: ★★★★ — 需要先完成 Idea 1，额外工作是混合尾涡集成 + 几何非线性梁

---

### Idea 3: Composite Cross-Section Integrated Aeroelastic Solver

**假说**: 将碳纤维复合材料截面分析 (VABS 风格) 集成到 GPU 加速的 UVLM 气动求解器中，实现复合材料翼面的 aeroelastic 仿真。

**方法**:
1. 解析/数值复合材料截面分析：给定铺层序列 → 等效梁刚度矩阵
2. 各向异性梁刚度：弯曲-扭转耦合（非对称铺层）
3. 集成到 Timoshenko 梁模型
4. UVLM 气动力 + 复合材料梁耦合

**验证**: 复合材料箱梁 vs Abaqus/ANSYS 参考解

**预期效果**:
- 弯曲-扭转耦合改变颤振速度和模态形状
- 碳纤维的高刚度-重量比影响气动弹性优化

**新颖性**: **中-高** — 复合材料在中等保真度涡方法中的建模是新的，但 VABS 本身成熟

**可行性**: ★★★ — 需要 VABS 集成或自建复合材料截面分析，复杂度中等

---

### Idea 4: GPU Quasi-Newton Aeroelastic Coupling

**假说**: 将 Schubert et al. (2024) 的拟牛顿耦合算法迁移到 GPU，实现快速收敛的紧耦合 aeroelastic 仿真。

**方法**:
1. 气动 Jacobian ∂F_aero/∂u 和结构 Jacobian ∂F_struct/∂u 均在 GPU 上计算
2. 拟牛顿算法在 GPU 上迭代
3. 每步仅需 1-2 次 UVLM + 梁求解（vs 显式耦合可能需要 5-10 次）

**验证**: 对比显式/隐式耦合的收敛速度和精度

**新颖性**: **高** — GPU 拟牛顿用于 aeroelastic 耦合是全新的

**可行性**: ★★ — Jacobian 计算复杂，需要 UVLM AIC 矩阵对网格变形的灵敏度

---

## 3. 排名

| 排名 | 方案 | 新颖性 | 可行性 | 影响力 | Pilot 风险 |
|------|------|--------|--------|--------|----------|
| **1** | **GPU UVLM + Euler-Bernoulli 梁** | 高 | 极高 | 高 | 低 |
| 2 | 混合尾涡 aeroelasticity | 高 | 高 | 高 | 中 |
| 3 | 复合材料截面集成 | 中-高 | 中 | 中 | 中 |
| 4 | GPU 拟牛顿耦合 | 高 | 低 | 高 | 高 |

**推荐路径**: Idea 1 → Idea 2 → Idea 3 逐步递进。Idea 4 可作为 Idea 1 的加速方案整合。

---

## 4. Idea 1 详细实验设计

### 4.1 实现架构

```
┌─────────────────────────────────────────────────┐
│ AeroelasticSolver (继承 UVPMHybridSolver)        │
│                                                  │
│  run() override:                                 │
│    for step in range(num_steps):                 │
│      1. _collapse_geometry()                     │
│      2. _calculate_wing_wing_influences()        │
│      3. _calculate_freestream_wing_influences()  │
│      4. _calculate_wake_wing_influences()        │
│      5. _calculate_vortex_strengths()            │
│      6. _calculate_loads()        ─── panel forces│
│      7. [NEW] _solve_structure()  ─── beam FE    │
│         ├── map_panel_forces_to_beam_nodes()     │
│         ├── assemble_beam_fe()                    │
│         ├── solve_beam_system()                   │
│         └── update_wing_geometry()               │
│      8. _populate_next_airplanes_wake()          │
│                                                  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ BeamFE (新模块)                                  │
│  - 节点: WingCrossSection 位置                    │
│  - DOF: heave (z) + twist (θy) per node          │
│  - 单元: Euler-Bernoulli 梁单元                   │
│  - 材料属性: EI, GJ, m, x_ea (弹性轴偏移)         │
│  - 质量/刚度矩阵组装 + Newmark-β 时间积分          │
└─────────────────────────────────────────────────┘
```

### 4.2 Goland Wing 验证配置

| 参数 | 值 |
|------|-----|
| 弦长 | 6.0 ft (1.8288 m) |
| 半展长 | 20.0 ft (6.096 m) |
| 展弦比 | 10 |
| 弹性轴位置 | 33% chord |
| EI | 23.65 × 10⁶ lb·ft² |
| GJ | 2.39 × 10⁶ lb·ft² |
| m | 0.746 slugs/ft |
| x_α (CG-EC offset) | 0.20 chord |
| 参考颤振速度 | ~450 ft/s |

### 4.3 成功标准

- 颤振速度预测误差 < 10% vs Goland 参考解
- 线性响应收敛（亚临界速度下衰减振荡）
- 颤振频率预测误差 < 15%
- 单步 GPU 耗时 < 1s

### 4.4 Pilot 实验计划

1. 实现 BeamFE 模块（~200 行 Python）
2. 实现 AeroelasticSolver（~150 行，继承 HybridSolver）
3. 运行 Goland Wing 多速度扫描（V = 300-500 ft/s）
4. 检测颤振边界（位移发散点）
5. 对比参考解

---

## 5. 参考文献

1. Murua, Palacios & Graham (2012), "UVLM in Aircraft Aeroelasticity and Flight Dynamics", Prog. Aero. Sci.
2. Dussler & Palacios (2023), "Enhanced UVLM for Nonlinear Flexible Aircraft", AIAA Journal
3. Schubert et al. (2024), "Accelerating Aeroelastic UVLM by Inexact Newton", arXiv:2403.15286
4. Yang, Xie & Yang (2020), "Geometrically Exact VLM in Static Aeroelasticity", Proc. IMechE Part G
5. Pudasaini et al. (2026), "Morphing Aerial Vehicles Mid-Fidelity Aeroservoelastic", arXiv:2605.02076
6. SHARPy: github.com/ImperialCollegeLondon/sharpy
7. OpenAeroStruct: github.com/mdolab/OpenAeroStruct
8. Goland & Luke (1948), "The Flutter of a Uniform Wing", JAM
9. Hodges (2006), "Nonlinear Composite Beam Theory", AIAA Education Series
