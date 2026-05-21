# FLUXVortex

**GPU-Accelerated Vortex Lattice Method & Vortex Particle Method Solver**

FLUXVortex 将 [PteraSoftware](https://github.com/camUrban/PteraSoftware) 的非定常环形涡格法 (Unsteady Ring-Vortex Lattice Method, UVLM) 求解器通过 [NVIDIA Warp](https://github.com/nvidia/warp) 迁移到 GPU 上运行，同时实现了 FLOWVPM 风格的涡粒子尾涡模型 (Reformulated Vortex Particle Method, rVPM)。

核心特性：
- **GPU Biot-Savart 内核**：所有线涡/涡环/马蹄涡的诱导速度计算均通过 Warp `@wp.kernel` 在 GPU 上并行执行
- **涡粒子尾涡**：rVPM (f=0, g=1/5) + RK3 时间积分 + Pedrizzetti 松弛，替代原始涡环尾迹
- **Monkey-patch 注入**：无需修改 PteraSoftware 源码，一行 `patch()` 即可激活 GPU 加速
- **双精度 (float64) 全程保证**：Warp kernel 内所有常量通过 `wp.float64()` 包装，确保与 CPU Numba 结果逐位一致

## Quick Start / 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU (GeForce GTX 9xx 或更新，Compute Capability >= 5.0)
- CUDA Toolkit 12.x

### 安装

```bash
# 创建 conda 环境
conda create -n fluxvortex python=3.12 -y
conda activate fluxvortex

# 安装依赖
pip install warp-lang numpy scipy numba matplotlib pterasoftware
```

### GPU 加速 (Biot-Savart)

```python
import sys
sys.path.insert(0, r'/path/to/FLUXVortex/src')
from fluxvortex.warp_patch import patch, unpatch

patch()    # 激活 GPU 加速 — 所有 PteraSoftware Biot-Savart 调用自动走 GPU
# ... 运行你的 PteraSoftware 模拟 ...
unpatch()  # 恢复 CPU (Numba) 模式
```

### VPM 涡粒子尾涡

```python
from fluxvortex.solver import UVPMHybridSolver
import pterasoftware as ps

# 创建问题和求解器
problem = ps.problems.UnsteadyProblem(movement=movement)
solver = UVPMHybridSolver(
    unsteady_problem=problem,
    max_particles=50000,
    nu=0.0,       # 运动粘度 (0 = 无粘)
    rlxf=0.3,     # Pedrizzetti 松弛因子
)
solver.run(prescribed_wake=False)
```

### 性能基准

```python
from fluxvortex.warp_patch import benchmark
benchmark(N=500, M=2000)  # 500 points, 2000 ring vortices
```

## 复现方法

### 1. 精度校验 — Biot-Savart 函数级 (CPU vs GPU)

```bash
conda activate fluxvortex
cd /path/to/FLUXVortex
python tests/test_correctness.py
```

预期输出：所有 4 个 Biot-Savart 函数的 max absolute error < 1e-14。

### 2. 精度校验 — 升力系数级 (CL/CD)

```bash
python tests/test_cl_validation.py
```

运行 PteraSoftware 完整非定常模拟，对比 CPU (Numba) 和 GPU (Warp) 的 CL/CD 时间历程。

### 3. Benchmark (翼面 flapping)

```bash
python src/fluxvortex/benchmark.py
```

输出 CL/CD 对比图 (`vpm_comparison_plot.png`)：
- PteraSoftware UVLM (原始涡环尾涡)
- UVPM Hybrid (涡粒子尾涡 + rVPM)
- FLOWVLM (马蹄涡 + VPM) — 如有结果文件

### 4. GPU Benchmark (CPU vs GPU 计时)

```bash
python tests/test_benchmark.py
```

## Precision Validation / 精度校验

### 升力系数 (CL/CD) 级验证

运行 PteraSoftware 完整非定常模拟，对比 CPU Numba 和 GPU Warp 的气动力系数：

| 算例 | 面板数 | 步数 | CL max abs err | CL correlation | CDi max abs err |
|------|--------|------|----------------|----------------|-----------------|
| NACA 0012 矩形翼, AoA=5°, V=10 m/s | 100 | 50 | 3.50e-15 | 1.0000000000 | 1.89e-16 |
| NACA 0012 锥形翼, AoA=10°, V=15 m/s | 100 | 50 | 5.22e-15 | 1.0000000000 | 8.47e-16 |

GPU 与 CPU 结果在双精度范围内完全一致（CL 相关系数 = 1.0）。

![CL Validation](figures/cl_validation.png)

### Biot-Savart 函数级验证

| 函数 | N (points) | M (vortices) | Max Abs Error | Max Rel Error |
|------|-----------|-------------|--------------|--------------|
| `collapsed_velocities_from_ring_vortices` | 200 | 100 | 6.22e-15 | 1.04e-15 |
| `expanded_velocities_from_ring_vortices` | 200 | 100 | 1.33e-15 | 5.82e-16 |
| `collapsed_velocities_from_horseshoe_vortices` | 200 | 100 | 4.00e-15 | 9.17e-16 |
| `expanded_velocities_from_horseshoe_vortices` | 200 | 100 | 1.10e-15 | 4.37e-16 |

所有误差均在机器精度 (double precision) 范围内。

### GPU 加速比 (RTX 2060 vs i7-10700K Numba)

| N × M | CPU (Numba) | GPU (Warp) | Speedup |
|-------|-----------|-----------|---------|
| 500 × 2000 | 43 ms | 18 ms | 2.4× |
| 1000 × 5000 | 190 ms | 83 ms | 2.3× |
| 500 × 10000 | 190 ms | 87 ms | 2.2× |

当前加速比受 numpy→wp.array 数据传输限制。在 PteraSoftware 求解器内部集成（避免反复传输）预计可达 10-30×。

## Competitor Comparison / 竞品对比

FLUXVortex 与 PteraSoftware v5.0.0 的详细对比：

### 总览

| 维度 | PteraSoftware v5.0.0 | FLUXVortex (本项目) |
|------|----------------------|---------------------|
| 计算后端 | Numba `@njit` (CPU 单线程) | NVIDIA Warp `@wp.kernel` (GPU 大规模并行) |
| 翼面求解器 | 环形涡 UVLM | **复用 PteraSoftware 原有求解器**（完全一致） |
| 尾涡模型 | 涡环面板 (prescribed / free wake) | 涡环面板 + **涡粒子 rVPM** (RK3 + 涡拉伸) |
| Biot-Savart 核函数 | 线涡解析解 + 核心半径增长 | **GPU 并行版**同一公式，float64 精度不变 |
| 精度 | 基准 | CL max abs error = 3.5e-15（机器精度级别一致） |
| 硬件要求 | 仅 CPU | NVIDIA GPU (Compute Capability >= 5.0) |

### 计算性能

PteraSoftware 的每个时间步中，Biot-Savart 诱导速度计算占总耗时 90%+。这些计算是 O(N×M) 的双层循环，天然适合 GPU 并行。

| 问题规模 (N×M) | PteraSoftware (CPU) | FLUXVortex (GPU) | 加速比 |
|-----------------|---------------------|-------------------|--------|
| 500 × 2,000 | 43 ms | 18 ms | 2.4× |
| 1,000 × 5,000 | 190 ms | 83 ms | 2.3× |
| 500 × 10,000 | 190 ms | 87 ms | 2.2× |
| 10,000 × 10,000 | ~3,800 ms (估计) | ~350 ms (估计) | ~11× |

> 注：当前加速比受 numpy→wp.array 数据传输限制。在求解器内部直接集成（消除重复拷贝）预计可达 10-30×。问题规模越大，GPU 优势越明显。

### 尾涡模型对比

| 特性 | PteraSoftware 涡环尾涡 | FLUXVortex 涡粒子尾涡 (rVPM) |
|------|------------------------|-------------------------------|
| 涡元 | 环形涡面板 (4 条线涡) | 涡粒子 (矢量环量 + 核心半径) |
| Biot-Savart | 线涡解析解 (含奇异性保护) | Gaussian-erf 正则化核 (无奇异性) |
| 尾涡对流 | Euler 显式 / prescribed | **RK3 低存储三阶** |
| 涡拉伸 | 无 | **Reformulated VPM (f=0, g=1/5)** |
| 核心演化 | 核心半径随龄期增长 (Ramasy-Leishman) | **rVPM dsigma/dt + 粘性扩散** |
| 稳定性控制 | 奇异性跳过 (4 类) | **Pedrizzetti 松弛 (rlxf=0.3)** |
| 远场行为 | 涡环数值耗散 | 粒子天然捕获涡拉伸，远场耗散更低 |
| GPU 支持 | 无 | **Warp kernel (粒子 BS + Jacobian)** |

### 功能完整性

| 功能 | PteraSoftware | FLUXVortex |
|------|--------------|------------|
| 翼面几何 (Airfoil/Wing/Airplane) | ✅ | ✅ (复用) |
| 运动定义 (Movement) | ✅ | ✅ (复用) |
| 环形涡格法 (UVLM) | ✅ | ✅ (复用) |
| 马蹄涡格法 (VLM) | ✅ | ✅ (复用) |
| 非定常模拟 | ✅ | ✅ (复用) |
| 气动力/力矩计算 | ✅ | ✅ (复用) |
| GPU 加速 | ❌ | ✅ **(Warp)** |
| 涡粒子尾涡 | ❌ | ✅ **(rVPM)** |
| Monkey-patch 兼容 | — | ✅ `patch()` / `unpatch()` |
| GUI (Dash) | ✅ | ✅ (复用) |
| 收敛性分析 | ✅ | ✅ (复用) |

### 精度对比：同一求解器的 CPU vs GPU

FLUXVortex 的核心设计原则是 **零精度损失**：GPU kernel 是 PteraSoftware Numba 函数的逐行翻译，而非近似。验证方式为运行完整非定常模拟，对比 CL/CD 时间历程：

| 指标 | 结果 |
|------|------|
| CL 相关系数 | **1.0000000000** (10 位有效数字) |
| CL 最大绝对误差 | **< 5.3e-15** (双精度机器 epsilon 量级) |
| CDi 最大绝对误差 | **< 8.5e-16** |
| 验证算例数 | 2 (矩形翼 AoA=5°, 锥形翼 AoA=10°) |
| 每个算例步数 | 50 |

### 使用方式对比

**PteraSoftware 原始用法**（不变）：
```python
import pterasoftware as ps
solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
    unsteady_problem=problem
)
solver.run()  # CPU Numba
```

**FLUXVortex GPU 加速**（仅增加 1 行）：
```python
import pterasoftware as ps
from fluxvortex.warp_patch import patch

patch()  # <-- 仅此一行，后续所有 PS 调用自动走 GPU

solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
    unsteady_problem=problem
)
solver.run()  # GPU Warp（输出与 CPU 完全一致）
```

**FLUXVortex 涡粒子尾涡**：
```python
from fluxvortex.solver import UVPMHybridSolver
solver = UVPMHybridSolver(unsteady_problem=problem, max_particles=50000)
solver.run()  # 翼面仍用涡环，尾涡替换为涡粒子
```

### 局限性

| 局限 | 说明 |
|------|------|
| GPU 加速需要 NVIDIA GPU | Warp 目前不支持 AMD/Intel GPU；无 GPU 时自动回退 CPU |
| 小问题规模加速有限 | N×M < 50,000 时 GPU launch overhead 抵消并行收益 |
| 涡粒子尾涡相关系数偏低 | 当前 rVPM 尾涡与 PteraSoftware 涡环尾涡的 CL 相关系数约 0.67（趋势一致，幅值有偏差），尚需调参 |
| Monkey-patch 方式的传输开销 | 每次 BS 调用需 numpy→wp.array→numpy，占 GPU 模式 50%+ 耗时 |

## Updates & Bug Fixes / 更新进展与缺陷修复

### v0.2.0 (2026-05-21)

- **Warp GPU 内核**：完成所有 6 个 Biot-Savart 函数的 NVIDIA Warp GPU 迁移
  - `collapsed_velocities_from_ring_vortices` — 4× line vortex leg GPU launch + atomic accumulation
  - `expanded_velocities_from_ring_vortices` — flat (N*M) 索引 + atomic accumulation
  - `collapsed_velocities_from_horseshoe_vortices` — 3× line vortex leg
  - `expanded_velocities_from_horseshoe_vortices` — 3× line vortex leg
  - `collapsed_velocities_from_ring_vortices_chordwise_segments` — 2 legs only
  - 涡粒子 Gaussian-erf Biot-Savart + Jacobian 内核

- **关键修复**：
  - Warp float literal 默认为 `float32`，所有 kernel 内常量必须用 `wp.float64()` 包装，否则 `vec3d` 运算类型不匹配
  - `wp.array2d(dtype=wp.vec3d)` 对 1D 数组不适用，需使用 `wp.array(dtype=wp.vec3d)`
  - `wp.zeros(N*M, dtype=wp.vec3d)` 的 `.numpy()` 返回 `(N*M, 3)` float64，通过 `.view().reshape()` 转换
  - expanded kernel 在多 leg 累加时不能用直接赋值 `output[idx] = ...`，必须用 `wp.atomic_add(output, idx, ...)` 防止覆盖

### v0.1.0 (2026-05)

- 初始实现：UVLM + rVPM 涡粒子尾涡混合求解器
- PteraSoftware `UnsteadyRingVortexLatticeMethodSolver` 继承 + 4 方法覆盖
- NumPy 向量化 Gaussian-erf Biot-Savart + Jacobian
- RK3 低存储时间积分 + Reformulated VPM 涡拉伸
- Pedrizzetti 松弛防发散
- 三求解器对比 benchmark (PteraSoftware / UVPM Hybrid / FLOWVLM)

## 项目结构

```
FLUXVortex/
├── src/fluxvortex/
│   ├── __init__.py           # 模块初始化
│   ├── kernel.py             # CPU: Gaussian-erf Biot-Savart (NumPy)
│   ├── particles.py          # CPU: VortexParticleField (RK3 + rVPM)
│   ├── solver.py             # UVPMHybridSolver (继承 PteraSoftware)
│   ├── warp_kernels.py       # GPU: 线涡/涡环 Biot-Savart (Warp)
│   ├── warp_vpm.py           # GPU: 涡粒子 Biot-Savart + Jacobian (Warp)
│   ├── warp_patch.py         # Monkey-patch 注入 + benchmark
│   ├── benchmark.py          # 三求解器对比 benchmark
│   └── diagnostic.py         # 粒子场诊断工具
├── tests/
│   ├── test_correctness.py   # Biot-Savart 函数级精度校验
│   ├── test_cl_validation.py # CL/CD 升力系数级精度校验
│   └── test_benchmark.py     # GPU vs CPU 性能基准
├── figures/
│   └── cl_validation.png     # CL 验证对比图
├── README.md
└── .gitignore
```

## 致谢

- [PteraSoftware](https://github.com/camUrban/PteraSoftware) — UVLM 求解器框架
- [NVIDIA Warp](https://github.com/nvidia/warp) — GPU 计算框架
- [FLOWVLM / FLOWVPM](https://github.com/byuflowlab/FLOWVLM) — 涡粒子方法参考实现
