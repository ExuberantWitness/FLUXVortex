# Goland 颤振：GPU FLUX-V5M 气动 + Beam 结构（最终计划，无中间版本）

## 架构（一次到位）

```
保留：GPU FLUX-V5M 气动核心（已在 Yamano 8步验证，CUDA float64）
     = src/fluxvortex/warp_fsi/q16_flux_v5m_native.py 的气动部分
     - AIC (torch CUDA)
     - Γ 求解 (torch CUDA)
     - 尾迹对流 RK3 (torch CUDA)
     - 5 分量压力 (torch CUDA)
     - LEV/TEV 粒子场 (Warp CUDA)

替换：Q16 结构 → Beam FE（已验证 140.2 m/s）
     = src/fluxvortex/beam_fe.py
     - Euler-Bernoulli 弯扭耦合
     - Newmark-β 时间积分

接口（新增，唯一需要写的代码）：
     Beam→面板：beam w(y),θ(y) → 面板顶点 z 偏移 + 绕弹性轴旋转
     面板→Beam：GPU 压力 → 展向积分 → station lift/moment → beam 节点力
```

## 具体实现步骤（每步完整，不简化）

### Step 1: 从 Q16NativeV5MSurface 提取气动核心
- 复制 Q16NativeV5MSolver 中与结构无关的部分：
  AIC 组装、Γ 求解、尾迹对流、5分量压力计算
- 删除 Q16 形状函数依赖，改为接受外部提供的面板几何

### Step 2: Beam→面板几何映射
- Beam 给出 w(y), θ(y)（Hermite 插值到面板站位）
- 面板 z 坐标 += w(y) + (x - x_ea)·sin(θ(y))
- 面板 x 坐标 += (x - x_ea)·(cos(θ(y))-1)

### Step 3: 面板压力→Beam 力映射
- GPU 压力 (n_panels,) → 按展向 station 求和 → lift_j, moment_j
- lift_j = Σ_chord p_ij × A_ij, moment_j = Σ_chord p_ij × A_ij × (x_ij - x_ea)
- BeamFE.distribute_force_to_nodes(y_st, lift, moment)

### Step 4: 时间步进循环
```
for step:
    1. 用当前 beam 状态变形面板 (Step 2)
    2. GPU FLUX-V5M 求解 AIC→Γ→尾迹→5分量压力 (Step 1)
    3. 压力→beam 力 (Step 3)
    4. Beam Newmark 步进
    5. 记录 tip_w, tip_θ
```

### Step 5: Goland 扫速
- V = 120, 130, 140, 145, 150, 160
- 包络增长率 σ → V_f 零交叉
- 对比 140.2 (lagged) / 137 (Goland-Luke)

## 预期

- 全部计算 GPU（beam 27 DOF CPU 可忽略）
- 每步 < 0.01s（GPU AIC 128 面板）
- 完整 100 步 < 5s/速度点
- 扫 6 个速度 < 1 分钟（vs 当前 CPU 路径 ~50s）

## 不做什么

- 不做 KJ-only
- 不做 partial pressure（缺少任何分量）
- 不做 simplified Mf1
- 不用 Pterra CPU UVLM
- 不做 Q16（Newton 尺度问题未解）

## 更新（2026-08-24）：Beam 也上 GPU

原计划 Beam FE 留 CPU → 改为全 GPU：

- K, M, C 矩阵: numpy → torch CUDA float64（一次性组装后传输）
- np.linalg.solve → torch.linalg.solve（每步 GPU 求解）
- 状态向量 d, v, a → torch CUDA
- 力映射 distribute_force_to_nodes → torch CUDA

满足 FLUX-V5M 合同："全部科学数值 CUDA float64，无 CPU fallback"
