# 保留 UVLM 的 FLUXV 双论文统一改进与交叉比较报告

## 1. 结论先行

已经实现并完整运行了同一套、无论文编号分支的探索性 `FluxV periodic v2`，最初覆盖：

1. Izraelevitz、Zhu 与 Triantafyllou (2017)，*State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span* 的 Figure 11 工况；
2. Yang 等 (2025)，*Numerical simulation framework of bird-inspired ornithopter in forward flight* 的刚性矩形翼六个安装攻角工况。

该版本的底层 v1 继续求解原 UVLM 环量、尾迹和非定常载荷通道；最终 v2 保留其周期均值与分离区交变量，并在附着区用 ULLT 交变载荷替换 UVLM 输出，同时引入两个共享机制：

- 附着流阶段：由本地复现的一状态 ULLT 提供周期交变载荷形状；
- 分离流阶段：由 UVLM 加有限翼全攻角极曲线残差提供载荷；UVLM/极曲线始终拥有周期均值。

在这两个开发期冻结任务上，v2 的指定点估计误差相对旧 FLUXV 均数值降低：

- Yang 2025：升力 MAE 从 `6.855 gf` 降到 `3.952 gf`，阻力 MAE 从 `12.922 gf` 降到 `2.062 gf`；分别降低 `42.35%` 和 `84.04%`。
- Izraelevitz 2017 Figure 11：无相位平移的升力 RMSE 从 `2.4310` 降到 `0.1546`，阻力 RMSE 从 `0.9531` 降到 `0.2293`；分别降低 `93.64%` 和 `75.94%`。

随后补入了此前缺失的真实实验检验：Izraelevitz 2017 的实验对比不是 Figure 11，也不存在该论文 Figure 17；正确对象是 **Figure 14 的 Scherer 1968 水槽实验**。该附加实验门禁否决了“统一改进已经泛化”的结论：

- Scherer Figure 14 的 14 个实验观测上，旧 FLUXV 的平均推力系数 RMSE 为 `0.05115`；
- 改进 FLUXV v1/v2 的周期均值相同，RMSE 反而增至 `0.22260`；
- 本地一状态 ULLT 的 RMSE 为 `0.04636`，是当前本地入口中最小；
- 作者一状态/六状态 ULLT 的数字化 RMSE 分别为 `0.06321/0.06704`。

该失败随后被定位为周期载荷所有权错误：v1/v2 将瞬时静态全攻角极曲线的正阻力均值直接叠加到零均值大幅升沉--俯仰运动上，重复占用了本应由非定常 ULLT/UVLM 环量与尾迹描述的推进载荷。基于这一诊断新增了无 `case_id` 的探索性 `FluxV periodic v3 persistent owner`：用周期持续迎角比控制 ULLT 与 UVLM/极曲线的均值所有权，同时控制交变载荷的分离门。三个 full 门禁结果为：

- Yang 2025：升力/阻力 MAE 为 `3.579/3.891 gf`，仍优于旧 FLUXV 的 `6.855/12.922 gf`，但阻力不如 v2 的 `2.062 gf`；
- Izraelevitz Figure 11：升力/阻力原相位 RMSE 为 `0.1546/0.3142`，仍显著优于旧 FLUXV 的 `2.4310/0.9531`；升力与本地 ULLT 一致，阻力较冻结 v2 的 `0.2293` 退步；
- Scherer Figure 14 的 14 个实验观测：`CT` MAE/RMSE 为 `0.03395/0.04719`，小幅优于旧 FLUXV 的 `0.03484/0.05115`；只修均值透传、不把 ULLT 设为零持续迎角所有者的消融 RMSE 为 `0.05204`。

因此当前正确结论更新为：**v3 在三个既有门禁上均通过预定的非退化条件，但它是在看过 Figure 14 失败后提出并选择的 post-hoc 探索性修复，不是独立盲验，也不能合入生产默认路径。** 它仍是周期双遍模型，不是因果在线/瞬态 FLUXV；Figure 14 只验证周期平均推力，且 `p=0` 时 v3 完全选择一状态 ULLT，不能据此声称 LEV 吸力、动态失速或 v3 瞬时载荷已经解决。

## 2. 两个验证任务及参考数据

### 2.1 Yang 2025 刚性翼

- 单个矩形刚性翼：弦长 `0.130 m`、翼展 `0.250 m`；铰点至气动翼根 `0.080 m`；
- 来流 `5.5 m/s`，扑动频率 `2.5 Hz`；
- 安装攻角：`0°、5°、10°、15°、20°、25°`；
- 风洞 Test 与作者 Proposed 均来自 Figure 11 的周期均值数字化；
- 本地运动采用论文公开四连杆参数重建的 nominal four-bar，不能等同于未公开的激光位移传感器时序；
- 统一使用正阻力 `D=-T`。

这一任务的风洞周期均值是真实实验参考，但 `±0.4 gf` 仅表示读图不确定度，不是实验误差条。论文没有公开相位载荷，因此本报告中的 Yang 相位图只能比较模型形状，不能计算相位精度。

### 2.2 Izraelevitz 2017 Figure 11

- 使用论文 Figure 11 的有限翼升沉--俯仰工况；
- 作者 UVLM、六状态 ULLT、一状态 ULLT 和 QS+added-mass 曲线直接从源 PDF 矢量路径提取；
- 所有相位指标使用原始相位，未做最优循环平移或幅值拟合。

这里的作者 UVLM 是数值参考，不是实验真值，因此结果只能说明对论文数值基准的复现/逼近能力。

### 2.3 Izraelevitz 2017 Figure 14 / Scherer 1968 实验

- Izraelevitz 2017 只有 Figure 1--15，没有 Figure 17；唯一直接实验对比是 Figure 14；
- 实验翼来自 Scherer 1968：NACA 63A015、弦长 `4 in (0.1016 m)`、翼展 `12 in (0.3048 m)`、`AR=3`、略圆翼尖、`3/4c` 俯仰轴；
- 运动为 `z=h cos(ωt)`、`θ=θmax cos(ωt+ψ)`，`h/c=0.6`、`J'=6`、`St=0.2`、`k=π/6=0.5236`；
- 实验比较包含 `θmax=15°` 与 `25°`，相位差 `ψ=15°--105°`；实验只公开周期平均推力系数及误差棒，不提供瞬时升力/阻力真值；
- Figure 14 文本使用 `Cd0=0.057`，而 Scherer 原报告静态试验写 `CD0=0.027`。主结果忠实采用 `0.057`，并固定输出 `0/0.027/0.057` 来源敏感性，不按观测误差择优；
- 实验方框、误差棒和作者三条模型曲线从 PDF 矢量图提取；重复实验点作为独立观测保留。

来源敏感性不会改变 v1/v2 失败的判定：

| `Cd0` | 旧 FLUXV `CT` RMSE | 改进 v1/v2 `CT` RMSE | 本地一状态 ULLT `CT` RMSE |
|---:|---:|---:|---:|
| 0.000 | 0.10054 | 0.17886 | 0.09021 |
| 0.027（Scherer 原报告） | 0.07451 | 0.19824 | 0.06552 |
| 0.057（Izraelevitz Figure 14） | 0.05115 | 0.22260 | 0.04636 |

## 3. 改进算法

### 3.1 UVLM 载荷分账

旧 FLUXV 的载荷实际来自 prescribed-wake UVLM；现有 VPM 粒子是单向诊断量，不反馈翼面 AIC 或载荷。新求解器保留完整 UVLM 解，并逐时刻审计：

\[
\mathbf F_{\mathrm{UVLM}}=\mathbf F_C+\mathbf F_{AM},
\]

其中 `F_AM` 按 Ptera 原生非定常 Bernoulli 项由环量时间差直接恢复，`F_C` 为总载荷减去该项。粒子关闭的快速路径已经由成对回归试验证明与粒子开启时的升阻力逐点一致。

### 3.2 v1：UVLM 周期载荷修正

v1 保留 UVLM 周期均值，只修正两类交变载荷并加入分离极曲线残差：

\[
\mathbf F_{v1}=\overline{\mathbf F}_{\mathrm{UVLM}}
+K_{AM}(AR)(\mathbf F_{AM}-\overline{\mathbf F}_{AM})
+G_H(AR)(\mathbf F_C-\overline{\mathbf F}_C)
+\Delta \mathbf F_{polar}.
\]

- `K_AM(3)=0.85`、`K_AM(6)=0.95`，中间按展弦比线性插值；
- `G_H` 使用 Izraelevitz 2017 Eq. (42)，`K=13.5`；
- 极曲线残差使用实际几何和局部四分之一弦速度：

\[
\Delta C_L=w(\alpha)a_{3D}[\sin\alpha\cos\alpha-\alpha],\qquad
\Delta C_D=w(\alpha)C_{D90}\sin^2\alpha,
\]

其中 `a_3D` 为 Prandtl 有限翼升力线斜率，`C_D90=1.20`，`w` 是在 `15°--20°` 之间由 0 平滑过渡到 1 的共享门函数。

### 3.3 v2：ULLT 附着态与 UVLM 分离态统一

本地一状态 ULLT 使用论文给出的 Wagner 状态、倾斜半无限 horseshoe AIC、有限展弦比附加质量和 Eq. (42) lifting-line-to-surface 修正。v2 对周期交变量进行统一所有权分配：

\[
\mathbf F'_{v2}=(1-w)\mathbf F'_{ULLT}+w\mathbf F'_{v1},
\]

随后重新去均值，并加回 v1 的 UVLM/极曲线周期均值。这样：

- 附着流由 ULLT 状态模型提供幅相；
- 分离流由 UVLM/全攻角极曲线提供；
- 底层 v1 仍求解 UVLM 环量和尾迹；v2 保留 v1 周期均值和分离区交变载荷，但附着区交变载荷由 ULLT 替换，因此不能称为全相位 UVLM 载荷保真；
- 两篇论文使用完全相同的参数、门函数和代码路径；
- 实现不读取风洞力残差，也没有 `paper_id`/`case_id` 修正表。

### 3.4 v3：持续迎角控制的周期载荷所有权

Figure 14 失败后新增的 v3 不再允许瞬时分离门单独移动周期均值。对每个条带定义

\[
p_j=\frac{|\overline{\alpha_j}|}{\overline{|\alpha_j|}},\qquad
p=\frac{\sum_j S_jp_j}{\sum_j S_j},
\]

其中 `p` 仅来自几何和周期运动学；没有论文编号或观测残差输入。全局瞬时分离门 `s(t)` 被衰减为 `s_eff(t)=p s(t)`，周期均值和交变量分别为

\[
\overline{\mathbf F}_{v3}=(1-p)\overline{\mathbf F}_{ULLT}
+p\overline{\mathbf F}_{v1},
\]

\[
\mathbf F'_{v3}=[1-s_{eff}(t)]\mathbf F'_{ULLT}
+s_{eff}(t)\mathbf F'_{v1},
\]

并对混合交变量重新去均值。因此 `p=0` 时均值和交变量都严格退化到本地一状态 ULLT，不会保留一个去均值后的静态极曲线阻力历史；`p=1` 时退化到既有 ULLT/v1 分离混合。

full 工况的 `p` 为：Yang 六攻角依次 `0.00009/0.42142/0.72398/0.88727/0.96661/0.99602`；Figure 11 为 `0`；Figure 14 的 12 个唯一工况全部为 `0`。这解释了 v3 的行为，但也构成明确限制：Figure 14 的通过主要验证“零持续迎角应由 ULLT 拥有周期载荷”这一 post-hoc 所有权假设，没有验证新的 LEV 物理。

此外，Figure 14 的 `Cd0=0.057` 已与非线性极曲线代理分账，按论文真实 `3/4c` 俯仰轴处的局部速度单独计算并只加一次。旧/v1/v2 冻结指标仍保留原四分之一弦代理，因此最终主图不再把它们与 v3 当成同口径曲线叠画；主图改用同一 `3/4c` 阻力账本下的均值透传消融作为公平基线。

## 4. Yang 2025 六攻角性能

### 4.1 聚合误差

| 模型 | 升力 MAE (gf) | 阻力 MAE (gf) | 说明 |
|---|---:|---:|---|
| 作者 modified UVLM | 3.350 | 1.933 | 作者 PLEV+AWS+free-wake/core 整体结果，不是本地 PLEV |
| 旧 FLUXV | 6.855 | 12.922 | 当前 UVLM 载荷通道 |
| 改进 FLUXV v1 | 3.952 | 2.062 | 改进周期均值 |
| 改进 FLUXV v2 | 3.952 | 2.062 | 均值继承 v1；ULLT 改变尚无真值的相位形状 |
| **改进 FLUXV v3** | **3.579** | **3.891** | 持续迎角同时控制均值所有权和 AC 分离门；post-hoc exploratory |
| 本地一状态 ULLT | 8.339 | 12.774 | 缺少高攻角分离物理 |
| Ptera free-wake UVLM | 6.278 | 13.070 | 仅自由尾迹不足以解决阻力问题 |
| RoboFalcon2 系数迁移 | 73.889 | 8.905 | 跨几何/跨雷诺数诊断，不能视为原生验证 |

v2 的升力和阻力误差均接近作者完整 modified UVLM，但两通道合并看仍没有超过作者：作者平均为 `2.642 gf`，v2 为 `3.007 gf`。Yang 的周期均值由 v1 明确拥有，所以 v2 与 v1 的六攻角均值逐位相同；Yang 上已经验证的改进来自 UVLM/全攻角极曲线均值通道，新增 ULLT 只改变尚无实验真值的相位形状。20°--25°升力饱和和阻力修复不能归因于 ULLT。

### 4.2 随安装攻角变化的周期均值

表中每格为 `升力/阻力 (gf)`：

| 攻角 | 风洞 Test | 作者 modified UVLM | 旧 FLUXV | 改进 FLUXV v2 |
|---:|---:|---:|---:|---:|
| 0° | 2.00 / -0.50 | 0.00 / -0.00 | 0.18 / -6.56 | 0.18 / -2.21 |
| 5° | 17.40 / 0.10 | 15.80 / 1.40 | 13.72 / -6.02 | 12.50 / -0.87 |
| 10° | 31.50 / 5.10 | 29.80 / 5.20 | 27.11 / -4.39 | 24.16 / 3.24 |
| 15° | 38.70 / 14.10 | 40.20 / 10.80 | 40.22 / -1.69 | 34.63 / 9.82 |
| 20° | 42.90 / 21.00 | 48.20 / 17.50 | 52.90 / 2.02 | 43.51 / 18.26 |
| 25° | 45.30 / 27.80 | 53.30 / 24.90 | 65.02 / 6.71 | 50.28 / 28.62 |

v3 在六个攻角的升力/阻力为 `0.07/-6.44、13.42/-3.79、25.25/1.16、35.48/8.53、43.93/17.73、50.36/28.53 gf`。它改善了旧 FLUXV，也把高攻角趋势保留下来，但低到中攻角阻力较 v2 退步；不能用“三门均通过”替代逐工况曲线判断。

## 5. Izraelevitz 2017 Figure 11 性能

以下为相对作者 UVLM 的原始相位 RMSE，单位是论文缩放系数：

| 模型 | 升力 RMSE | 阻力 RMSE | 说明 |
|---|---:|---:|---|
| 作者六状态 ULLT | 0.1443 | 0.3430 | 论文曲线 |
| 作者一状态 ULLT | 0.1029 | 0.2957 | 论文曲线 |
| 作者 QS + added mass | 0.9296 | 0.8849 | 论文曲线 |
| 本地一状态 ULLT | 0.1546 | 0.3142 | 无真值拟合的本地复现 |
| 旧 FLUXV | 2.4310 | 0.9531 | 幅值和相位均明显偏离 |
| 改进 FLUXV v1 | 2.2800 | 0.9208 | 仅小幅改善，不能通过强统一结论 |
| **改进 FLUXV v2** | **0.1546** | **0.2293** | 附着态幅相由 ULLT 状态接管，保留 UVLM 均值 |
| **改进 FLUXV v3** | **0.1546** | **0.3142** | `p=0`，256 步 full 离散下完全由本地 ULLT 拥有；post-hoc exploratory |
| Ptera free-wake UVLM | 2.3553 | 0.9711 | 自由尾迹未解决核心偏差 |

v2 的升力接近本地一状态 ULLT，阻力 RMSE 低于作者一状态 ULLT；后者部分来自 v2 保留了旧 UVLM 的周期均值，不能解释为本地 ULLT 全面超过论文方法。

### 5.1 Figure 14 实验门禁修复结果

| 模型 | MAE `CT` | RMSE `CT` | Bias `CT` |
|---|---:|---:|---:|
| 本地一状态 ULLT（冻结四分之一弦 `Cd0` 代理） | 0.03388 | 0.04636 | 0.01503 |
| **FluxV v3（真实 3/4c `Cd0` 速度）** | **0.03395** | **0.04719** | **0.01634** |
| 旧 FLUXV | 0.03484 | 0.05115 | 0.02698 |
| v3 均值透传消融 | 0.03495 | 0.05204 | 0.02830 |
| 作者一状态 ULLT | 0.04584 | 0.06321 | 0.04539 |
| 作者六状态 ULLT | 0.05014 | 0.06704 | 0.05014 |
| 冻结 FluxV v1/v2 | 0.18331 | 0.22260 | -0.16786 |

v3 对旧 FLUXV 的 RMSE 改善只有约 `0.00396`，没有统计显著性解释。更重要的是，该结构是看到 Figure 14 失败后才提出；所以这里只能报告“已修复已知门禁上的载荷所有权错误”，不能报告独立泛化证据。

数值敏感性仍有警告：开发期 smoke→full 同时改变网格、时间步和周期数，不是正式收敛序列；Figure 14 各条件的最大成对变化为 `0.00555 CT`，略高于预先记录的 `0.005 CT` 开发阈值。因此只把 full 结果作为主结果，但不声称已数值收敛。

## 6. 图件

1. Yang 2025 周期均值升力/阻力随安装攻角变化：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/figures/yang2025_mean_lift_drag_vs_aoa.png`
2. Yang 2025 典型 15° 工况的相位升力/阻力：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/figures/yang2025_15deg_phase_lift_drag.png`
3. Izraelevitz 2017 Figure 11 升力/阻力相位曲线：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/figures/izraelevitz2017_fig11_lift_drag_phase.png`
4. 两篇论文的误差汇总：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/figures/crosspaper_accuracy_summary.png`
5. Izraelevitz Figure 14 / Scherer 1968 实验平均推力随相位差变化：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/figures/izraelevitz2017_fig14_scherer_experiment.png`
6. v3 Figure 14 实验、作者模型、同一 `3/4c` 账本消融与 v3 随相位差对比：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full/figures/izraelevitz2017_fig14_v3_experiment_comparison.png`
7. v3 Yang 2025 Test、作者、旧/v2/v3 升阻力随攻角对比：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full/figures/yang2025_v3_mean_lift_drag_vs_aoa.png`
8. v3 Izraelevitz Figure 11 作者 UVLM/一状态、旧/v2/v3 相位升阻力：
   `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full/figures/izraelevitz2017_fig11_v3_lift_drag_phase.png`

相同目录中同时提供矢量 PDF。

## 7. 可复现入口

核心实现：

- `platform/forward_flight_benchmarks/augmented_uvpm.py`
- `platform/forward_flight_benchmarks/uvlm_polar_correction.py`
- `platform/forward_flight_benchmarks/ullt_attached.py`
- `platform/forward_flight_benchmarks/periodic_load_ownership.py`
- `platform/forward_flight_benchmarks/run_periodic_v3_regression.py`
- `platform/forward_flight_benchmarks/plot_periodic_v3_regression.py`

运行命令（在 FLUXV 根目录执行）：

```bash
PYTHONPATH=src:platform python -m forward_flight_benchmarks.run_unified_fluxv_upgrade \
  --quality full \
  --output-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v1_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.run_unified_ullt_extension \
  --base-run docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v1_full \
  --output-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.plot_unified_fluxv_upgrade \
  --run-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.run_izraelevitz_scherer_experiment \
  --quality full \
  --output-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_scherer_fig14_experiment_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.plot_izraelevitz_scherer_experiment \
  --run-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_scherer_fig14_experiment_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.run_periodic_v3_regression \
  --quality full \
  --output-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full

PYTHONPATH=src:platform python -m forward_flight_benchmarks.plot_periodic_v3_regression \
  --run-dir docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full
```

开发期两个任务的完整数值结果位于：

`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full`

Figure 14 实验交叉验证结果位于：

`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_scherer_fig14_experiment_full`

v3 三门 full 结果位于：

`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v3_persistent_full`

## 8. 当前可以和不可以声称的结论

可以声称：

- 同一探索性周期 v2、同一参数和同一代码路径，在两个开发期冻结任务上均得到更低的指定点估计误差；
- 底层 v1 的 UVLM 环量/尾迹继续求解，v2 保留其均值和分离区交变量；附着区交变量由 ULLT 接管；
- Yang 的高攻角升力饱和和阻力趋势在当前数字化周期均值上更接近风洞参考；
- Izraelevitz Figure 11 的附着流幅相由本地一状态 ULLT 数值重建，并在该作者数值参考上降低了误差；
- Figure 14 的真实实验交叉验证已完成，并明确显示 v1/v2 的均值修正不能迁移到该工况。
- v3 在当前三个 full 门禁上均优于旧 FLUXV 或满足原相位 RMSE 阈值；Figure 14 的 `Cd0` 使用真实 3/4c 速度分账并只加一次。

不能声称：

- 已经完成可直接替换生产求解器的在线、因果、任意瞬态 FLUXV；
- 已完整复现 Yang 作者的 PLEV/AWS/modified-UVLM；
- Yang 相位载荷已被实验验证；
- Izraelevitz Figure 11 对比是实验验证；
- 15°--20°门函数已经通过独立盲验，或当前结果是预注册确认性结果；
- v2 总体精度已经超过 Yang 作者模型；
- Yang 上新增 ULLT 机制已经提高准确率（均值指标对它结构性不敏感，相位又无公开真值）；
- 旧 FLUXV 与 Ptera prescribed UVLM 是两个独立模型。
- 当前 v1/v2 已经在新的实验工况上优于旧 FLUXV或具有统一泛化能力。
- v3 是独立盲验、确认性验证或已证明可泛化的生产模型；
- v3 已解决 LEV 吸力、动态失速或 Figure 14 的瞬时升阻力（该实验未公开瞬时载荷）；
- v3 全面优于 v2 或作者模型；Yang 阻力和 Figure 11 两通道均较 v2 退步。

## 9. 下一步

生产化的优先任务不是继续对这两套曲线调参，而是：

1. 将周期双遍的 ULLT/UVLM 所有权切换改造成因果状态递推；
2. 对 15°--20°过渡区做载荷和导数连续性约束，消除典型 15°相位曲线中的快速门切换；
3. 将 v3 作为 post-hoc 候选冻结，不再用 Figure 14 调结构或参数；
4. 在未参与当前探索的新前飞大俯仰/扭转工况上冻结参数做真正盲验，重点包含公开瞬时力或 LEV 数据；
5. 分别进行空间网格、时间步、周期数和尾迹保留长度的一因素收敛；
6. 只有上述门槛通过后，才考虑将 v3 的因果版本合入生产 FLUXV 默认路径。

## 10. 验证与独立审计

- 最终定向回归：`38 passed`；覆盖 UVLM 分账、粒子快速路径等价、共享极曲线、一状态 ULLT、周期所有权、重复周期端点和追加未来周期不变性。
- 图件独立复核：`PASS`；单位、`D=-T` 符号、模型身份和相位真值限制均已明确。
- 实验完整性审计：`WARN / qualified_only`，无完整性 `FAIL`。所有指标从最终 CSV 独立重算后最大差为 `0.0`；警告来自 post-hoc 设计、无独立留出盲验、周期双遍非因果，以及 Yang 均值不能验证新增 ULLT 相位机制。
- 审计文件：`EXPERIMENT_AUDIT.md`、`EXPERIMENT_AUDIT.json`。
