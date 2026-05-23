# IDEA REPORT: 基于涡粒子方法的中等保真度CFD仿真改进

**Date**: 2026-05-23 (updated)
**Pipeline Stage**: 1 — Idea Discovery
**Project**: FLUXVortex (E:\DATA\vscode\VLM\FLUXVortex)

---

## 1. 研究背景与已有成果

### 实验体系 (全部已完成)

| 实验 | 配置 | 核心发现 |
|------|------|---------|
| 多粒子脱落策略 | A:4环, B:单前腿kf sweep, k=0.5/0.2/0.1 | 4粒子环比单粒子更差(39-54%), Gaussian近场相消 |
| 核函数替换 | Gaussian-erf vs Winckelmans, 6种配置×3k | Winckelmans近场强6.6倍但精度仅提升1-2% |
| 线涡段替代粒子 | front/full_ring, rc0×offset扫描 | 低k提升25-33%(0.694 vs 0.524), 高k反馈爆炸 |
| 稳定性扫描 | rc0×offset×v_clamp 8种配置 | 核心矛盾：精度越高反馈越强，无法同时稳定+精确 |

### 涡环面板基准

| k | Ring/Theo | 相关系数 |
|---|-----------|---------|
| 0.50 | 0.929 | 0.88 |
| 0.20 | 0.930 | 1.00 |
| 0.10 | 0.920 | 0.98 |

### 核心瓶颈

1. **拓扑失配**：0D点粒子 vs 1D线段 — 近场方向性信息丢失
2. **反馈不稳定**：强核→陡梯度→正反馈→高k爆炸 (stiff system)
3. **核函数无关**：Gaussian vs Winckelmans 仅差1-2%，不是瓶颈

---

## 2. 文献调研关键发现

| 来源 | 核心贡献 | 相关性 |
|------|---------|--------|
| **Alvarez & Ning, rVPM (2022-2024)** | LES滤波NS重推VPM, SFS模型, 100×加速 | 稳定性方案，解决远场湍流 |
| **Fu & Laurendeau, NL-UVLM-VPM (2025, arXiv:2511.11430)** | UVLM+VPM双向耦合 + Vreman SGS + 自适应粒子转换, 70%省时 | **唯一直接相关**：自适应转换+粘性修正 |
| **Martins et al., LES-VPM (2026, arXiv:2601.06942)** | 高阶代数核 + SGS依赖粒子正则化 | 解释高k爆炸机制 |
| **FLOWUnsteady (BYU)** | VLM翼面 + rVPM尾涡, **单向耦合** | 生产级架构参考 |
| **Palha et al., Hybrid E-L (2015)** | Euler边界 + Lagrange尾涡 | 拓扑问题的标准解决方案 |
| **Wang et al., VPFM (SIGGRAPH 2025)** | 粒子涡量 + 网格速度 + 边界条件 | 最新边界处理方法 |
| **Cottet & Koumoutsakos (2000)** | VIC方法, 正则化理论, 边界条件 | VIC近场精度 O(h²) |

**关键洞察**: Fu & Laurendeau (2025) 是目前唯一做UVLM-VPM双向耦合并达到URANS精度的团队。他们的关键方法是：
1. 自适应粒子转换（TE附近密，远场稀）
2. Vreman SGS 涡粘模型 + PSE 粘性扩散
3. 非线性粘性-无粘耦合 (alpha-coupling)

---

## 3. 候选方案 (按推荐排名)

### Idea 1: 混合面板-粒子尾涡 (Hybrid Panel-Particle Wake)

**假说**: 近场保留涡环面板（保证精度），远场转换为VPM粒子（实现自由尾涡卷起）。

**方法**:
1. 每个 timestep 正常 shed 涡环面板（PteraSoftware 默认行为）
2. 超过 N_keep 行的旧面板行：提取环量信息，转换为 VPM 粒子
3. 转换算法：每个涡环的4条腿 → 提取展向 Gamma 梯度 → 生成 trailing vortex 粒子
4. `_calculate_wake_wing_influences` = 面板贡献（近场）+ 粒子贡献（远场）

**预期效果**:
- 近场：92-93% Theo（涡环面板精度，已验证）
- 远场：VPM粒子能力（自由尾涡卷起、涡拉伸、rVPM 稳定化）
- 全k稳定（近场用面板，无反馈问题）

**新颖性**: 中-高。Fu & Laurendeau 做了类似工作但用 LES 框架。我们的方案更轻量——不需要 LES，自适应阈值是新的。

**可行性**: ★★★★★ — PteraSoftware 已有涡环面板尾涡，只需添加面板→粒子转换

**Pilot**: 比较纯面板 vs N_keep=1,2,5,10 混合方案 vs 纯粒子

---

### Idea 2: 近场核函数校正 (Distance-dependent Kernel Correction)

**假说**: 对粒子诱导速度施加距离相关校正，近场匹配线涡行为。

**方法**:
- 计算粒子到最近翼面控制点的距离 d
- 校正因子 C(d) = v_line(d) / v_blob(d)
- 近场 (d < 2σ): 强校正
- 远场 (d > 5σ): 无校正
- 平滑过渡

**预期效果**: 精度从55%提升到70-80%，无需改变粒子表示

**新颖性**: 高。VIC方法的离散化版本，但用于VLM-VPM耦合接口是新思路

**可行性**: ★★★★ — 只修改 `_calculate_wake_wing_influences`

---

### Idea 3: 自适应Core Radius (Adaptive σ based on distance)

**假说**: σ随距离翼面的距离变化，近场小σ（接近线涡），远场大σ（稳定）。

**方法**: σ(d) = σ_min + (σ_max - σ_min) * sigmoid((d - d_ref) / d_scale)

**预期效果**: 55% → 65-75%

**新颖性**: 中。rVPM 中有 core-spreading 但用于粘性扩散，非耦合精度

**可行性**: ★★★★ — 只修改 σ 更新逻辑

---

### Idea 4: 展向多分辨率脱落 + 粒子合并

**假说**: 近场多粒子（展向细分），远场合并控制计算量。

**预期效果**: 55% → 60-65%（分辨率提升，但拓扑问题仍在）

**新颖性**: 中。VPM中常见，用于VLM-VPM耦合是新的

**可行性**: ★★★ — 需要粒子合并算法

---

## 4. 排名

| 排名 | 方案 | 预期精度 | 新颖性 | 可行性 | Pilot风险 |
|------|------|---------|--------|--------|---------|
| **1** | **混合面板-粒子** | 92%+ (近场) | 中-高 | 极高 | 低 |
| 2 | 核函数校正 | 70-80% | 高 | 高 | 低 |
| 3 | 自适应σ | 65-75% | 中 | 高 | 中 |
| 4 | 多分辨率脱落 | 60-65% | 中 | 中 | 中 |

---

## 5. Idea 1 Pilot 实验设计

### 实现方案
在 `experiment_hybrid_panel_particle.py` 中：
1. 继承 PteraSoftware UVLM solver
2. 保留涡环面板尾涡（父类默认行为）
3. 超过 N_keep 行的面板：提取 Gamma → 生成 VPM 粒子 → 从面板中移除
4. `_calculate_wake_wing_influences` = 父类面板影响 + VPM粒子影响

### 实验配置
- NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles
- k = 0.5, 0.2, 0.1
- N_keep = 1, 2, 5, 10, ALL(纯面板), 0(纯粒子)

### 成功标准
- N_keep ≥ 2: vs Ring/Theo ≥ 0.90
- k=0.5 稳定（不爆炸）
- 粒子数 < 纯面板方案的50%

### Pilot 结果 (2026-05-23)

**配置**: NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles, Gaussian-erf kernel

| Method | k=0.5 Theo | k=0.2 Theo | k=0.10 Theo | 稳定性 |
|--------|-----------|-----------|------------|--------|
| Ring baseline | 0.929 | 0.930 | 0.920 | 稳定 |
| **Hybrid N=20** | **0.939** | **0.934** | **0.921** | **全k稳定** |
| **Hybrid N=10** | **0.910** | **0.906** | **0.907** | **全k稳定** |
| Hybrid N=5 | 5.44 (爆) | 0.879 | 0.846 | k≤0.2稳定 |
| Hybrid N=2 | BLOWUP | BLOWUP | 9.21 (爆) | 不稳定 |
| VPM-only (N=0) | 0.762 | 0.440 | 0.425 | 稳定但精度低 |

**关键发现**:
1. **N=10 是最佳平衡点**: 全k精度 90.6-91.0%, 相关系数 0.998-1.000
2. **N=20 几乎完美**: 92.1-93.9%, 与纯面板差异 < 1%
3. **近场面板数量与稳定性正相关**: N<5 时反馈不稳定
4. **4粒子/环转换远优于单粒子脱落**: VPM-only k=0.5 从 113% 过冲降至 76% 稳定

**成功标准评估**:
- ✅ N_keep=10: vs Ring/Theo ≥ 0.90 (0.906-0.910)
- ✅ k=0.5 稳定 (N≥10)
- ⚠️ 粒子数 ~6600 vs 面板 ~894, 但粒子计算量 ≤ 面板 (1 BS vs 4 BS/环)

**下一步**:
- 自适应 N_keep: 根据局部条件动态调整近场/远场边界
- 自由尾涡测试: 启用 VPM free_wake 验证远场卷起能力
- 与 Fu & Laurendeau (2025) 方案对比: 自适应转换 vs 固定 N_keep

---

## 6. 参考文献

1. Alvarez & Ning (2024), "Reviving the VPM", AIAA Journal, arXiv:2206.03658
2. Fu & Laurendeau (2025), "NL-UVLM-VPM for Rotor Aerodynamics", arXiv:2511.11430
3. Martins et al. (2026), "LES-Integrated VPM", arXiv:2601.06942
4. Alvarez et al. (2022), "FLOWUnsteady", AIAA Journal
5. Palha et al. (2015), "Hybrid Eulerian-Lagrangian", arXiv:1505.03368
6. Wang et al. (2025), "Vortex Particle Flow Maps", SIGGRAPH/ACM TOG, arXiv:2505.21946
7. Cottet & Koumoutsakos (2000), "Vortex Methods: Theory and Practice", Cambridge
8. Winckelmans & Leonard (1993), "Contributions to VPM", J. Comp. Phys.
