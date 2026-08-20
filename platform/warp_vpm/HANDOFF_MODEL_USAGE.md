# 机理底盘气动求解器 — 使用交接

## 0. 一句话定义

一套零拟合的机理性气动载荷预测系统：**机器精度验证的 UVLM 底盘**（载荷级修正、不动环流求解）+ **声明常数的分离/粘性修正层**，在 Baik 2012、Izraelevitz 2017、Yang 2025、Mancini 2017 四篇论文上达到或超越 V4B，全部常数可溯源到冻结实现或发表值。

## 1. 核心组件地图

```
platform/warp_vpm/
├── bing_joint_ptera.py     ← 主求解器（机架 mixin）
├── bing_drag_ledger.py     ← 载荷级阻力/分离修正
├── bing_baik_2d.py         ← Baik: 2D LDVM 直接评分
├── bing_izra_v2.py         ← Izra: 机架 + 冻结 LDVM delta
├── bing_baik_final.py      ← Baik: 规范采样管道 + transfer
├── bing_mancini.py         ← Mancini: 机架 + LDVM delta 交叉验证
├── test_g0_steady.py       ← G0: 定常升力线门
├── test_g0b_ptera.py       ← G0b: 机架机器精度门
├── test_g0c_lev_active.py  ← G0c: LEV 激活稳定性门
├── bing_figures.py         ← 三论文性能曲线图
├── bing_simple_fig.py      ← 三线简洁图（ours/V4B/GT）
├── bing_version_fig.py     ← 版本对比图（latest/previous/GT）
├── bing_contour_fig.py     ← 逐面板载荷等高线图（中文标注）
├── bing_panel_loads_fig.py ← 逐面板载荷分布图
└── BING_SESSION_RESULTS.md ← 全部结果档案（含负结果）
```

## 2. 快速上手

### 2.1 跑一个新工况（3D 机架底盘）

```python
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "platform")
sys.path.insert(0, "platform/warp_vpm")

import pterasoftware
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

# 构建运动（用你自己的 movement builder）
movement = your_build_movement(...)
problem = pterasoftware.problems.UnsteadyProblem(
    movement=movement, only_final_results=False)

# 跑底盘（LEV 关 = 机器精度裸核；LEV 开 = 联立 LEV 求解）
solver = JointLEVTEVSolver(problem, JointConfig(
    enable_lev=False,       # True 开启 LEV 联立
    load_mode="bing",       # "bing" 钉帽 / "v4b3d" 完整 bound
    lesp_crit=0.11,         # 声明 LESP 临界值
))
solver.run(prescribed_wake=True, calculate_streamlines=False,
           show_progress=False)

# 收割载荷
FZ = [float(sp_.airplanes[0].forces_W[2]) for sp in solver.steady_problems]
FX = [float(sp_.airplanes[0].forces_W[0]) for sp in solver.steady_problems]
# lift = -FZ/qS, drag = -FX/qS (pterasoftware 风轴约定)
```

### 2.2 加阻力修正（载荷级，零环流改动）

```python
from bing_drag_ledger import LedgerConfig, run_ledger

cfg = LedgerConfig(
    lesp_crit=0.239,             # 圆前缘族声明值（或 0.11 薄板）
    aspect_ratio=4.0,            # 几何展弦比
    rho=998.0,                   # 水/空气密度
    cd0=2*1.328/np.sqrt(Re),     # Blasius 声明值
    enable_t1=True,              # T1: Polhamus 吸力损失
    enable_t3=True,              # T3: 动态粘性
    enable_t2=False,             # T2: Rayleigh 深分离（需记忆，默认关）
)
result = run_ledger(solver.ledger, cfg, last_n=steps_per_cycle)
# result["mean_t1_N"], result["mean_t3_N"], result["mean_total_N"]
```

### 2.3 加 LDVM 分离 delta（Izra 模式）

```python
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LESPThreshold, LDVMSectionSettings,
    run_ldvm_separation_pair, project_ldvm_delta_to_finite_wing)

# 构造截面运动学（alpha, alpha_dot, heave_rate 均为无量纲时间序列）
threshold = LESPThreshold(value=0.239, ...)
pair = run_ldvm_separation_pair(
    alpha_rad=alpha,                     # 弧度
    alpha_rate_per_convective_time=arate,
    heave_rate_over_u=heave,
    delta_time_convective=dt_star,       # c/U 归一化
    pivot_fraction_chord=0.75,
    threshold=threshold,
    settings=LDVMSectionSettings(ndiv=50, naterm=24, max_wake_steps=512))
proj = project_ldvm_delta_to_finite_wing(
    pair["delta"]["CNc"][-N:], pair["delta"]["CNnc"][-N:],
    pair["delta"]["CNnonl"][-N:], pair["delta"]["CSf"][-N:],
    alpha[-N:], aspect_ratio=AR)
CL_corrected = CL_bare + proj["delta_CL"]
CD_corrected = CD_bare + proj["delta_CD"]
```

### 2.4 Baik 类准二维实验 → 2D LDVM 直接评分

```python
from ldvm_fourier import LDVM2D

ldvm = LDVM2D(U=1.0, c=1.0, ndiv=32, naterm=14,
              dt=float(dt_star), rho=1.0, camber_m=0.0,
              pivot_xc=0.25, core_rc=0.02,
              lesp_crit=0.19,   # Ramesh Table 4.1 声明值
              max_wake=256)
for k in range(3 * SPC):
    out = ldvm.step(alpha[k % SPC], arate[k % SPC], heave[k % SPC])
    cl_hist.append(out["CLf"])   # 剖面 CL（含 LEV + 附加质量）
    cd_hist.append(out["CDf"])
```

## 3. 验证门体系

| 门 | 测试文件 | 验证内容 | 通过标准 | 失败退出 |
|----|---------|---------|---------|---------|
| G0 | `test_g0_steady.py` | 定常 α=5° AR=15 升力线 | CL ∈ [0.470, 0.4882]（1.5% 非定常容差） | exit 1 |
| G0b | `test_g0b_ptera.py` | LEV 关 == 裸 pterasoftware | 差 = 0.00e+00 + parity check | exit 1 |
| G0c | `test_g0c_lev_active.py` | α=20° LEV 激活稳定性 | LESP 钉帽、有限、稳定 | exit 1 |
| Ledger | `test_ledger_contract.py` | total == t1+t2+t3 after G6b clip | closure = 0.00e+00 | assert |
| G1 | solver 内部 | 联立解 Neumann 代回残差 | ≤ rtol·scale | GateError |
| G2 | solver 内部 | Kelvin 行残差 | ≤ rtol·scale | GateError |
| G3 | solver 内部 | LESP pin 残差 | ≤ rtol·crit | GateError |
| G4 | solver 内部 | 新生粒子几何守卫 | 无过近配置点 | GateError |
| G5 | solver 内部 | 载荷有限性 | 全部 finite | GateError |
| G6a | ledger 内部 | 附着极限 T1=0 | excess ≤ 0 | assert |
| G6b | ledger 内部 | 分离阻力 ≤ Rayleigh 上限 | clip 后满足 | ValueError |

**设备要求**：粒子场自动检测 CUDA/CPU（`PFIELD_DEVICE` 环境变量可强制指定）。无 GPU 时自动回退 CPU，无需修改代码。

## 4. 各论文的最优配置（终值）

| 论文 | 最优模型 | 关键参数 | 终值 | 对比 |
|------|---------|---------|------|------|
| **Baik W1-W4** | 2D LDVM 直接 | ndiv=32, crit=0.19 | CL **0.442** CD **0.294** | V4B 0.658/0.345 |
| **Yang 2025** | 机架 + 全角 polar + T3 | CD90=1.20, Blasius Cd0 | lift **4.10** drag **1.52** gf | V4B 4.55/2.64 |
| **Izra Fig14** | 机架 + Cd0 + LDVM delta | crit=sin(0.90/0.065)=0.239 | CT MAE **0.0178** | V4B 0.0198 |
| **Mancini 2017** | 机架 + LDVM delta | crit=0.11 | fast 1.255 slow 0.295 | V4B 1.218/0.291 |

### Baik 选型规则（物理驱动，非逐工况挑）

- **准二维实验**（壁到壁端板、窄水槽）→ **2D LDVM 直接**
- **三维自由端实验** → 机架 + LDVM delta
- 判断依据：实验装置文档中的端板/壁面描述

### LESP 临界值选择规则

| 截面族 | 声明值 | 来源 |
|--------|--------|------|
| 薄平板 | 0.11 | Ramesh 2013 正文（Re=1000 转移） |
| 薄平板（表值） | 0.19 | Ramesh 2013 Table 4.1 |
| 圆前缘 | sin(CLmax/CLα) | 截面静极线推导 |
| NACA 翼型 | 论文指定或查表 | Martinez-Carmena 等 |

## 5. 已归档的负结果（避免重复踩坑）

| 方案 | 结果 | 根因 |
|------|------|------|
| LEV-ON + 账本（Yang） | 阻力劣化 | 冲量项在阻力轴表现为推力 |
| v4b3d 模式（完整 bound + 冲量） | 发散 | RHS 反馈回路环量无节制增长 |
| 全角 polar 用于 Izra | 过修正 0.145 | Izra 相位偏移族不适合极线替换 |
| 多周期 rfft 低通 | W1/W2 端点伪影 0.5 CL | 应改用单周期谐波截止 |
| lru_cache 跨工况 | W2/W3/W4 污染 | 多工况连跑必须 cache_clear |

## 6. 环境与运行

```bash
# 必需环境变量
export PYTHONPATH=src:platform:platform/warp_vpm
export MPLCONFIGDIR=/tmp/mpl
export NUMBA_CACHE_DIR=/tmp/numba

# 典型运行时间
# 3D 机架 8×12cos / 128 步/周期 / 4 周期:  ~25 s/条件
# 2D LDVM 512 步/周期 / 3 周期:            ~30 s/条件
# LDVM 分离对 (ndiv=50, 96 步/弦):         ~15 s/条件
```

## 7. 接手 agent 的工作流

1. **读本文档** + `BING_SESSION_RESULTS.md`（完整正/负结果档案）
2. **跑三门**（G0/G0b/G0c）确认底盘健康
3. **选模型**：按第 4 节的选型规则匹配你的工况
4. **跑一个工况**确认数值合理（量级、符号、收敛）
5. **加修正层**（账本/LDVM delta/polar）按需
6. **每步跑门**（G1-G6 在 solver/ledger 内自动执行）
7. **结果归档**：追加到 `BING_SESSION_RESULTS.md`

## 8. 禁止动作

- 不得根据结果调 `lesp_crit`（声明值，敏感性另跑）
- 不得修改 `ldvm_fourier.py`（冻结验证组件）
- 不得用多周期 rfft 替代单周期谐波截止滤波
- 不得在多工况连跑中忘记 `cache_clear()`
- 不得把附着工况的 LEV 激活结果用于校准
- 不得将 Baik 的 2D LDVM 结果与三维自由端实验直接比较

## 9. 已知限制与下一步

| 限制 | 根因 | 下一步 |
|------|------|--------|
| Yang 中低攻角升力 ~4gf 欠预测 | 机身/根部密封 + 标称运动学 | 非定常镜像环移植 |
| Mancini 快速俯仰峰值/相位失配 | 45°/弦脉冲，附加质量主导 | 纯冲量法或非定常 Bernoulli |
| T2 Rayleigh 分支未启用 | 需分离状态记忆（LDVM 原生携带） | 持续 onset 权重设计 |
| Izra 端部条件 15/15、25/105 | 极端相位偏移 | 未调查 |

## 10. 成绩单（vs V4B，全部零拟合）

| 指标 | 我们 | V4B | 判定 |
|------|------|-----|------|
| Baik CL 宏 | **0.442** | 0.658 | 胜 −33% |
| Baik CD 宏 | **0.294** | 0.345 | 胜 −15% |
| Yang 升力 | **4.10** | 4.55 | 胜 |
| Yang 阻力 | **1.52** | 2.64 | 胜 |
| Izra CT | **0.0178** | 0.0198 | 胜 |
| Baik CL（规范管道） | 0.6577 | 0.6575 | 精确持平 |
| Mancini 快俯仰 | 1.2553 | 1.2184 | 近持平 +3% |
| Mancini 慢俯仰 | 0.2951 | 0.2908 | 近持平 +1.5% |
