# DATA MANIFEST — Rojratsirikul 2011 Figure 6/9/12–15 复现数据保全

> 目的（用户指令 2026-08-30）：**仔细保留全部数据，方便后续定位与分析原因。**
> 本目录所有产物均已入 git（含 `.npz` 原始时序，`.gitignore` 已加白名单例外）。
> 冻结参数：aero 网格 15×30、lesp_crit 0.11、wake_history=bound_rate、
> wake rows 300/100、粒子容量 32768 / age 100、dt*=0.01、CUDA float64、
> E=2.2 MPa、eta=0.1。所有工况共用，无逐工况调参。

## 1. 文件地图

| 路径 | 内容 | 用途 |
|---|---|---|
| `membrane_sweep/ROJ11-*_T10.json` | 膜翼扫掠正式 payload（α=5,10,13,21,19,23,25，t*=10）：逐门评分、统计窗、`mean_zmax_over_c`、`mean_Cn`、retention/守恒证据 | Figure 6/9 柔性曲线主数据 |
| `membrane_sweep/ROJ11-*_T10.z_history.npz` | **原始全场位移时序** `z_history_over_c(steps,15,31)` + `time_star` + `mean_pressure_map(15,30)` | 复算任意统计量/模态/PSD；定位窗口协议敏感性的唯一原始材料 |
| `membrane_sweep/ROJ11-*_T10.partial.json` | 运行中检查点（完成后冗余，保留供崩溃取证） | 中断恢复与异常定位 |
| `cases/ROJR-RIGID-*.json` | 刚翼工况正式 payload：`cn_history`（逐步）、`particle_counts`、`wake_rows`、`st_per_probe`、`dominant_psd_st/dominant_psd`、retention 审计 | Figure 9 刚性 + 13/14/15 曲线主数据 |
| `cases/ROJR-RIGID-*_probe_history.npz` | **12 探针 × 全时步速度时序**（u,v,w）+ 探针坐标 + aero_dt（2026-08-30 后新增的工况） | 尾流谱任意复算；retention 假峰取证 |
| `model_observables.csv` | P2 标准输出表（每工况一行，含失败行） | 评分与绘图输入 |
| `comparison/*.png` | 7 张仿真–实验叠加图（标题含 MAE/n） | 交付图 |
| `comparison/scores.json` / `case_failures.csv` | H1–H6 评分 / 失败留档 | 验收 |
| `comparison/progress.log` | 监视器逐点追加的 MAE 历史 | 回放"何时发现的" |
| `run.log` / `chain.log` | 刚翼队列 / 膜翼链的完整运行日志 | 时间线取证 |
| `checkpoint.json` | 队列恢复标记（gitignore，本地） | 断点续跑 |

对照锚点（长窗 t*=21 的膜翼正式运行）在
`../fluxv_v5m_rojratsirikul2011_unified_current/ROJ11_{A16,A17_MODE}_FULL.{json,z_history.npz}`。

## 2. 复现命令

```bash
cd <repo>
# 膜翼某点（例 α=13）
PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-SWEEP-A13 --max-aero-steps 1000 \
  --output artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current/membrane_sweep/ROJ11-SWEEP-A13_T10.json
# 刚翼队列（可恢复）
PYTHONPATH=... python3 platform/warp_vpm/queue_roj_rigid_fig9_12_13_15.py
# 全部图 + 评分刷新
PYTHONPATH=src:platform python3 platform/warp_vpm/compare_rojratsirikul2011_digitized_oracles.py
```

## 3. 原因定位的既知入口（按假设分类）

1. **统计窗协议敏感**：α=5/10/13/19/21 短窗（t*∈[1,10] fallback）未过平稳门
   （payload `window_selection.stationary_window_found=false`）。用
   `z_history.npz` 重算 t*∈[4,10] / [1,10] 两种窗可量化启动段污染
   （A16 先例：窗选择改变 Cn ~0.015）。zmax 低角 +82% 有多少来自此？
2. **势流类边界（Cn 平台 +34%）**：刚翼 α=15 U10 同窗 Cn +23%；
   对照 `cases/ROJR-RIGID-U10p0-A150.json`。
3. **低角膜响应偏软**：α=5 膜 +80% vs 刚翼 α=5 +6%（同 oracle）——
   气弹正反馈过强候选：预张力假设 0、E 名义 2.2 MPa（handoff E 分支）、
   Kelvin-Voigt 阻尼 (eta=0.1)。payload `assumption_ledger` 有冻结值。
4. **retention 谱污染取证**：`particle_counts`/`wake_rows` 逐步序列 +
   `*_probe_history.npz`——检查粒子 age-cull（每 100 步）时刻的谱跳变。
5. **实验侧不确定度**：oracle 包
   `observations/figure_digitization_20260829/`（含不确定度列与 SHA）。

## 4. 评分口径

H1–H6 门算术在 `platform/warp_vpm/compare_rojratsirikul2011_digitized_oracles.py`
顶部常量，与 handoff §9 一致；"数字化不确定度 ≠ 工程容差 ≠ 模型误差"三分。
