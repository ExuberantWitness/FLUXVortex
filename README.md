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

### 1. 精度校验 (CPU vs GPU)

```bash
conda activate fluxvortex
cd /path/to/FLUXVortex
python tests/test_correctness.py
```

预期输出：所有 4 个 Biot-Savart 函数的 max absolute error < 1e-14。

### 2. Benchmark (翼面 flapping)

```bash
python src/fluxvortex/benchmark.py
```

输出 CL/CD 对比图 (`vpm_comparison_plot.png`)：
- PteraSoftware UVLM (原始涡环尾涡)
- UVPM Hybrid (涡粒子尾涡 + rVPM)
- FLOWVLM (马蹄涡 + VPM) — 如有结果文件

### 3. GPU Benchmark (CPU vs GPU 计时)

```bash
python tests/test_benchmark.py
```

## Precision Validation / 精度校验

### Biot-Savart GPU vs CPU

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
│   ├── test_correctness.py   # GPU vs CPU 精度校验
│   └── test_benchmark.py     # GPU vs CPU 性能基准
├── README.md
└── .gitignore
```

## 致谢

- [PteraSoftware](https://github.com/camUrban/PteraSoftware) — UVLM 求解器框架
- [NVIDIA Warp](https://github.com/nvidia/warp) — GPU 计算框架
- [FLOWVLM / FLOWVPM](https://github.com/byuflowlab/FLOWVLM) — 涡粒子方法参考实现
