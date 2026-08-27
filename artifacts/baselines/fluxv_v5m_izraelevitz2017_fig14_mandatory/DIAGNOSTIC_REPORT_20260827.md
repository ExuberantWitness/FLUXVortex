# Izraelevitz Fig.14 复现诊断报告 — 新旧路径 CT 对比

日期：2026-08-27
分支：`run/q16-lev-tev-pc-fsi-20260821`
数据目录：`artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14_mandatory/`

---

## 1. 现象概述

用当前统一框架的 mandatory integrated LEV/TEV/free-wake 路径首次复现
Izraelevitz Fig.14 / Scherer 1968 的 12 个运动条件（14 个实验 marker），
得到 **MAE_CT = 0.210**（门 = 0.0175，差 12 倍）。物理合同全部通过
（LEV/TEV/wake 事务正确、Kelvin/Neumann 门 ~1e-17、512 commits 零丢弃），
但推力预测在低相位条件下**符号翻转**（预测阻力而非推力）。

## 2. 两条计算路径

| | 历史 GPU V2（对照组） | 当前 mandatory（被诊断组） |
|---|---|---|
| 提交 | 冻结于 `dc43d4`（20260820） | `64412dd`（20260827） |
| 求解器 | Ptera chassis（`bing_joint_ptera_gpu`） | `RigidNativeV5MSolver`（H3 新写） |
| LEV | ❌ `enable_lev=False`（关闭） | ✅ 集成，按 LESP≥0.239 条件释放 |
| 尾迹 | ❌ `prescribed_wake=True`（附着） | ✅ 自由尾迹，每步 RK3 对流 |
| 分离修正 | 后处理：`CT = CT_raw - 0.057 + ΔCT_LDVM` | 无后处理 |
| 力提取 | Ptera GP1→W 变换（已校准） | `RigidAuthorLoadAssembler`（**首次运行**） |
| 结果 | **MAE = 0.0175**（14 marker） | **MAE = 0.210** |

## 3. 逐条件 CT 对比表

### 15° 幅值族

| ψ | 实验 CT | 历史 V2 CT | 历史 CT_raw | 当前 CT | 当前误差 | V2 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 15° | 0.123, 0.145 | **+0.147** | +0.327 | **−0.271** | −0.394, −0.416 | +0.024, +0.002 |
| 30° | 0.180 | **+0.182** | +0.311 | **−0.196** | −0.376 | +0.002 |
| 45° | 0.234 | **+0.209** | +0.295 | **−0.119** | −0.353 | −0.026 |
| 60° | 0.231 | **+0.233** | +0.279 | **+0.016** | −0.214 | +0.002 |
| 75° | 0.213, 0.205 | **+0.225** | +0.266 | **+0.134** | −0.079, −0.071 | +0.012, +0.020 |
| 90° | 0.191 | **+0.200** | +0.257 | **+0.152** | −0.039 | +0.008 |
| 105° | 0.164 | **+0.194** | +0.251 | **+0.224** | +0.060 | +0.030 |

### 25° 幅值族

| ψ | 实验 CT | 历史 V2 CT | 历史 CT_raw | 当前 CT | 当前误差 | V2 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 45° | 0.096 | **+0.086** | +0.199 | **−0.216** | −0.312 | −0.010 |
| 60° | 0.106 | **+0.130** | +0.170 | **−0.141** | −0.248 | +0.024 |
| 75° | 0.084 | **+0.089** | +0.146 | **+0.006** | −0.078 | +0.005 |
| 90° | 0.043 | **+0.072** | +0.129 | **+0.150** | +0.107 | +0.029 |
| 105° | 0.012 | **+0.062** | +0.119 | **+0.204** | +0.192 | +0.050 |

## 4. 关键观察

### 4.1 符号翻转

历史 V2 的 `CT_raw` **全部为正**（+0.12 到 +0.33），表明 Ptera 底盘始终
预测净推力。当前 mandatory 路径的 `CT_raw` 在低相位（ψ ≤ 60°）为**负值**
（预测阻力），高相位才转为正。

### 4.2 趋势相反

- 实验：CT 在 ψ ≈ 45–60° 处达到峰值，两侧下降（钟形）
- 历史 V2：正确复现此趋势
- 当前 mandatory：从负到正**单调递增**（趋势相反）

### 4.3 CT_raw 差异的结构

| 量 | 历史 V2 | 当前 mandatory | 差异 |
|---|---|---|---|
| CT_raw 符号 | 全正 | 低 ψ 负、高 ψ 正 | **根本性分歧** |
| CT_raw 范围 | 0.12 – 0.33 | −0.21 – +0.26 | 幅值也不同 |
| 与实验的相关性 | 高（MAE 0.0175） | 无（MAE 0.210） | — |

## 5. 根因假设（按排查优先级）

问题不在 Cd0（历史 CT_raw 在减 Cd0 前已全部为正）、不在后处理 ΔCT
（那只是微调）、不在 LEV 释放物理（15° 族 LESP 低于阈值无释放，与
Ptera 关闭 LEV 一致）。问题在**基础力提取路径**。

| # | 假设 | 证据 | 排查方法 |
|---|---|---|---|
| 1 | `RigidAuthorLoadAssembler` 的压力→力投影符号/方向有误 | CT_raw 符号翻转 | 对同一几何+运动状态，逐压力分量（dp_lift1/mf2/lift2/mf21）与 Ptera 输出对比 |
| 2 | 刚性运动的速度项（lift2 = −v·∇γ）缺少或错误地包含了 heave/pitch 贡献 | 低 ψ（大俯仰-升沉耦合）误差最大；高 ψ（近纯升沉）误差较小 | 检查 SurfaceFrame 的 panel_ring_velocity 是否正确包含刚体运动速度 |
| 3 | 运动学相位约定差异（cos vs sin、Ptera +90° 映射） | 历史 builder 用 sine+phase=90° 实现论文的 cos | 直接对比两条路径在同一 t 的翼面位置/姿态/速度 |
| 4 | `NativeV5MConfig` 的 `freestream` 方向与 Ptera 来流方向不同 | 推力 = −Fx；如果来流方向不同，推力分量投影不同 | 检查两条路径的来流向量是否一致 |

## 6. 建议的下一步

**单步对比实验**（最快定位根因）：取 IZRA-15-045 条件、t = T/4（最大升沉
速度时刻），分别用两条路径计算：
1. 翼面几何（位置 + 法向 + 面积）
2. 环量分布 γ
3. 表面压力分布
4. 合力 F（x, y, z 分量）

对比每项的符号和量级 → 精确定位第一个分歧点。

## 7. 对比图

见 `fig14_ct_vs_phase.png`（runner 自动生成，包含实验 marker、当前
mandatory、历史 GPU V2、V4B、作者 one-state/six-state/QS 曲线）。

---

## 8. 外部评审更正（2026-08-27，当日追加）：状态升级为 FAIL_IMPLEMENTATION_CONTRACT

外部评审确认了两个**确定性的实现错误**（不是物理假设问题）。本报告
第 5 节的根因假设表和第 1 节的定性（"FAIL_ACCURACY_WITH_VALID_PHYSICS"）
**作废**；正确状态为 **FAIL_IMPLEMENTATION_CONTRACT**：H5 结果是在
带量纲错误与错误时间差分的载荷路径上产生的，在任何物理结论之前必须
先修复并重跑。

### 8.1 BUG 1：lift2 与 Mf2_1 缺少密度乘法（Pa 与 m²/s² 直接相加）

四个压力分量中，lift1 与 wake-history(Mf2) 携带 ρ，而**速度部分两项
漏乘 ρ**，然后四项直接求和 —— 把 m²/s² 加进了 Pa 量纲的 Pa 总压：

| 位置 | 分量 | 修复前 |
|---|---|---|
| `src/fluxvortex/warp_fsi/rigid_flux_v5m_native.py:200`（修复前行号） | `lift2` | `-Σ v·∇γ`，无 ρ |
| `src/fluxvortex/warp_fsi/rigid_flux_v5m_native.py:228`（修复前行号） | `mf2_1` | `LU⁻¹(rhs)`，无 ρ |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py:346`（修复前行号） | `lift2_pressure` | 无 ρ |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py:381`（修复前行号） | `mf21_pressure` | 无 ρ |

参考（正确实现）：`src/fluxvortex/warp_fsi/coupled.py:88`（`dp2 = rho·(τx·dΓ/dx + τy·dΓ/dy)`）、
`coupled.py:168`（`s = -rho·x`）与 `platform/warp_vpm/q16_real_fsi_coupling.py:388`
（lift2 算子内嵌 ρ）、`:1199`（`generalized = fluid_density · A⁻¹scal · map`）。

**修复**：两文件的 lift2/mf2(1)_pressure 计算补 `* self.density`
（Q16 端点载荷新增 `density` 字段由 assembler 传入）。回归测试
`tests/test_density_scaling_gpu.py`：ρ=1.0 与 ρ=1000.0 各跑一步
propose()/assemble()，**全部四个压力分量与力/wrench 必须严格线性缩放**
（分量级要求 bitwise 精确）。修复前该测试 3/3 用例失败，修复后全过。

### 8.2 BUG 2：bound_rate 使用了两步旧的环量（幅值×2、相位偏移）

`src/fluxvortex/warp_fsi/q16_flux_v5m_native.py:1018`（修复前行号）的
`wake_history_mode="bound_rate"` 分支读取 `trial.gamma_previous`。由于
状态更新顺序（propose() 末尾先 `gamma_previous = gamma_bound`，再
`gamma_bound = gamma`），在 mf2_history 计算点 `gamma_previous` 持有
**Γ_(n−2)** 而非 Γ_(n−1)，实际计算的是：

```
(Γ_n − Γ_(n−2)) / dt    而作者弱格式 dp_add 应为    (Γ_n − Γ_(n−1)) / dt
```

该错误使导数幅值翻倍并引入相位偏移。**修复**：改读
`trial.gamma_bound`（该时刻仍持有上一步已提交的 Γ_(n−1)，仅在本步末尾
才被覆盖）。回归测试 `tests/test_gamma_history_gpu.py`：三步推进，
逐步断言 mf2_history 严格等于 (Γ_n − Γ_(n−1))/dt 且与两步差分相差
>1.0（判别性下限）。修复前失败，修复后全过。

### 8.3 对第 5 节假设表的两处事实更正

1. **"15° 族 LESP 低于阈值无释放"是错的**。`physics_evidence.json`
   显示释放确实发生：IZRA-15-015 全程 512 步中 254 步有 LEV 释放
   （`lev_release_count_total = 6096`，`lev_release_count_max = 24`），
   最大分离条带数 22/24。第 5 节据此排除 LEV 释放物理的推理不成立。
   （注意 IZRA-15-090/105 与 25° 族高相位端确实无释放。）
2. **LEV 双所有权冲突被记录但未被门控**。IZRA-15-015 的
   `release_owner_conflicts_total = 1490`（3D 分离掩码与 source-bank
   shed_lev 的不一致计数）——数值被写进诊断但没有 gate 拦截，静默地
   按 reconcile 规则继续运行。该冲突是否影响环量分配需在修复重跑后
   复核（建议将非零冲突计数升级为显式 gate 或至少汇总进
   physics_evidence 顶层字段）。

### 8.4 修复后的必做事项

1. 重跑 H5 Fig.14 12 条件（两 bug 均在载荷路径上，CT 数值必然改变；
   第 3 节对比表全部作废）。
2. 重跑后再评估第 5 节剩余假设（几何/相位/来流方向对比实验仍然有效）。
3. 双所有权冲突（8.3-2）随重跑数据一并复核。

*修复提交：见 git log `fix(critical): lift2/Mf2_1 density dimension +
bound_rate one-step gamma history`。*

---

*本报告由 H5 运行结束后自动数据 + 手动分析生成。所有数据文件已在
`artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14_mandatory/` 中
git 跟踪。*
