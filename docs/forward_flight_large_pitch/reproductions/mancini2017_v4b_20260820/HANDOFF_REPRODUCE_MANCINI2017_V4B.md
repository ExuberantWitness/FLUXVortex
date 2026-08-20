# Mancini 2017 / FluxV v4b 独立复现交接

## 0. 任务身份与接手原则

本交接用于让一个**没有本次对话上下文**的 agent，在现有工作区中独立重放
Mancini 2017 有限翼快速机动实验的 FluxV v4b 零拟合迁移测试。

- 仓库根目录：`/tmp/fluxv-v5-nextgen`
- 复现目录：
  `docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820`
- 论文：Peter Mancini, *Experimental Investigation into Unsteady Force
  Transients on Rapidly Maneuvering Wings*, 2017 博士论文。
- 目标工况：AR=4 矩形有限翼，绕前缘从 0° 俯仰到 45°，快、慢两种运动。
- 方法：冻结 FluxV v4b；不允许根据 Mancini 实验修改 `Lcrit`、增益、相位、
  网格、轴位或运动参数。
- 结论边界：本任务只评分整翼 `C_L(t*)`，不能声称验证展向载荷分布。

接手 agent 的首要职责是**复现现有结果**，不是继续优化。若任一冻结输入、数据
或机械测试不一致，应停止并报告，不得一边修代码一边把新结果当成原结果。

当前文件均位于未提交工作树中。不得用 `git clean`、`git reset`、覆盖原运行目录
或其他破坏性操作。新运行必须写入新的 `runs/replay_<UTC>_*` 目录。

本次原执行使用 `experiment`/`write` 工作流的科学门，但环境缺少托管
`bash_exec`、artifact 和 memory 接口，因而用普通终端保存命令、SHA 和日志。
接手环境若提供这些接口，应优先使用它们；科学阈值和结果门不得改变。

## 1. 不在本任务范围内的论文

以下文献已由用户明确排除，不得替换或混入本复现：

1. *Experimental study and modelling of unsteady aerodynamic forces and
   moment on flat plate in high amplitude pitch ramp motion*；
2. *All you need is time to generalise the Goman-Khrabrov dynamic stall
   model*；
3. *Experimental characterization of static and dynamic stall noise in an
   anechoic wind tunnel*；
4. *Modelling the unsteady lift of a pitching NACA 0018 aerofoil using
   state-space neural networks*；
5. *Transient dynamics of stall and reattachment at low Reynolds number*；
6. *Experimental characterization of dynamic stall of the FFA-W3-211 wind
   turbine airfoil*。

前五项被排除是因为展向载荷表现不突出、近似二维；第六项因翼型过厚而排除。

## 2. 先读文件与代码地图

按以下顺序阅读，不要从聊天记忆恢复：

1. `PLAN.md`：冻结工况、指标和停止条件；
2. `CHECKLIST.md`：已完成证据面；
3. `RESULTS_REPORT_ZH.md`：结果、差距和结论强度；
4. `source_data/DIGITIZATION_MANCINI2017.md`：图线提取、像素标定和数据边界；
5. `source_data/mancini2017_case_matrix.csv`：快、慢工况矩阵；
6. `platform/forward_flight_benchmarks/mancini2017.py`：几何、运动、UVLM 和 v4b
   适配；
7. `platform/forward_flight_benchmarks/run_mancini2017_v4b.py`：预测先于实验加载、
   评分、中文图和运行清单；
8. `platform/forward_flight_benchmarks/digitize_mancini2017.py`：确定性数字化；
9. `platform/tests/test_mancini2017.py`：几何、运动、数据角色和禁调参数负门；
10. `platform/forward_flight_benchmarks/ldvm_uvlm_correction.py`：实际复用的 v4b
    分离增量与有限翼投影。

`platform/warp_vpm/bing_v4b_refined.py` 是 Baik 专用运行脚本，会绑定旧工况数据，
不能直接作为 Mancini runner。这里“直接使用 v4b”是指冻结算法和参数后进行
零拟合迁移，而不是把 Baik 脚本连同旧观测一起执行。

## 3. 冻结输入与控制 SHA

在做任何执行前，于仓库根目录重算以下 SHA-256。任一不一致即 `H0 FAIL`：

| 文件 | SHA-256 |
|---|---|
| Mancini PDF | `afd15346f7177b90828b826d026560ad6887673563484ecfe52f502926ccee24` |
| 数字化实验 CSV | `8feecd8469bc5b7dff00c761d773d91464143e394c244cbfe0502a473df4db7c` |
| `digitize_mancini2017.py` | `bfe98b43b91957a16f4158060881f5a1dbb985d1219c0f6d03a92888f915c4a5` |
| `mancini2017.py` | `a8c33220d7be2f39af0f2d2e96ca9b49b421ba036cbcdbd3033fa87c48e8b9ad` |
| `run_mancini2017_v4b.py` | `27ae05e4c47ef5a93d7e5626ea89abc96c219fa7e15a4247e85ae25d8283e7f0` |
| `ldvm_uvlm_correction.py` | `d72d794242ab702bcd95ab1f5aaf6e623a87722eb1474f073124f6f60031db13` |
| `bing_v4b_refined.py` | `80d6bb5b18f356516439670e01d5f09a88662cb96911b317de5ba6982782c271` |
| `test_mancini2017.py` | `cecb9593e754c74f5a9624e7594dd9262c3f25e44d4f49424c04361a060b1537` |
| `PLAN.md` | `08f75d08f6c2900ff69aa61fd3dc6d31285271f2f2408ae875072a01f3a261e3` |
| `CHECKLIST.md` | `2129beddd062222576ea951c1c1e2ee406357d6923b1e5e604a6a84033088093` |
| `RESULTS_REPORT_ZH.md` | `7ef294e9df773d910f7b778ec9b8e17d87d4d1b6de325b03f3ff13ce0ef897d7` |

PDF 路径：

```text
researchpaper/experimental_validation_candidates_20260820/13_mancini_2017_unsteady_force_transients_dissertation.pdf
```

实验 CSV 路径：

```text
docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820/source_data/mancini2017_fig4_13b_pitch_lift_digitized.csv
```

原始成功运行的不可覆盖证据：

| 文件 | SHA-256 |
|---|---|
| JIT smoke `summary.json` | `74215bf56ebc341d7295690455384133c061db20f345aee8423fb4331dbaafcd` |
| JIT smoke `run_manifest.json` | `c01441cc0ae78708667fe9f143fe924690d0166208c0055f218edc2cc431a198` |
| full `summary.json` | `ba3f4f9b1e45b3037a492a30ef4b91cdb64a5bf782685571dcdb6b1f95c05279` |
| full `run_manifest.json` | `9bbaea11291beaca56b99d6bc669f86ec6f080ffc4c98f74b68ff387fb56a841` |

原始 full summary 内登记了 8 个结果文件的 SHA；H0 还必须逐项重哈希这些文件，
不能只核 summary 自身。

## 4. 冻结工况与计算设置

共同实验条件：

- 矩形有限翼平板，`AR=4`；
- 弦向 `c=0.0762 m`，展向 `b=0.3048 m`；
- `t/c=0.05`，圆钝前、后缘；
- `U_inf=0.26 m/s`，`Re=20,000`；
- 绕前缘轴从 0° 俯仰到 45°；
- 快俯仰：`s_a/c=1`、`k=0.39`、`a=15`；
- 慢俯仰：`s_a/c=6`、`k=0.065`、`a=4`；
- 共同评分窗口：`0 <= t*=tU_inf/c <= 5`。

坐标用词必须保持一致：`x/c` 是**弦向**，`y/b` 是**展向**。论文评分数据是
整翼总升力，不是展向分布。

计算设置：

| 级别 | 弦向面元 | 半翼展向面元 | UVLM 步/弦 | v4b 步/弦 |
|---|---:|---:|---:|---:|
| smoke | 2 | 6 | 24 | 24 |
| full | 4 | 12 | 64 | 96 |

full 应为完整翼 96 面元、449 个时刻、最大尾迹 384 行。smoke/full 只有两档，
不构成网格收敛证明。

## 5. 复现环境

原运行环境：

- Python `3.11.7`；
- Linux `6.8.0-124-generic`、glibc `2.35`；
- `PYTHONPATH=platform:src`；
- `NUMBA_CACHE_DIR=/tmp/numba-mancini`；
- `MPLCONFIGDIR=/tmp/mpl-mancini`；
- full 使用 Numba JIT，`NUMBA_DISABLE_JIT` 未设置；
- 中文字体文件：
  `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。

JIT smoke 与禁用 JIT smoke 的逐点预测最大差为约 `6.93e-14`，v4b 增量逐点
完全一致。除非专门审计 JIT 等价性，不必重跑耗时约 162 秒的 no-JIT smoke。

## 6. 验收节点

### H0：输入与工作树完整性

1. 记录 `git status --short`，确认不会删除已有未跟踪文件；
2. 核第 3 节所有冻结 SHA；
3. 解析 full summary，逐项重算其中 `result_hashes`；
4. 确认原 `runs/20260820_mancini2017_v4b_*` 目录保持只读、不覆盖。

通过条件：所有冻结哈希完全相同。否则停止。

### H1：数字化数据独立重放

```bash
PYTHONPATH=platform:src python \
  platform/forward_flight_benchmarks/digitize_mancini2017.py \
  researchpaper/experimental_validation_candidates_20260820/13_mancini_2017_unsteady_force_transients_dissertation.pdf \
  /tmp/mancini2017_digitized_replay.csv

sha256sum /tmp/mancini2017_digitized_replay.csv
cmp /tmp/mancini2017_digitized_replay.csv \
  docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820/source_data/mancini2017_fig4_13b_pitch_lift_digitized.csv
```

通过条件：临时 CSV SHA 必须为
`8feecd8469bc5b7dff00c761d773d91464143e394c244cbfe0502a473df4db7c`，
`cmp` 无差异；1301 行数据覆盖 `t*=0...13`。若失败，不得进入求解。

### H2：机械测试与静态门

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_CACHE_DIR=/tmp/numba-mancini \
PYTHONPATH=platform:src \
pytest -q platform/tests/test_mancini2017.py

python -m py_compile \
  platform/forward_flight_benchmarks/digitize_mancini2017.py \
  platform/forward_flight_benchmarks/mancini2017.py \
  platform/forward_flight_benchmarks/run_mancini2017_v4b.py \
  platform/tests/test_mancini2017.py

python -m black --check \
  platform/forward_flight_benchmarks/digitize_mancini2017.py \
  platform/forward_flight_benchmarks/mancini2017.py \
  platform/forward_flight_benchmarks/run_mancini2017_v4b.py \
  platform/tests/test_mancini2017.py

python -m ruff check \
  platform/forward_flight_benchmarks/digitize_mancini2017.py \
  platform/forward_flight_benchmarks/mancini2017.py \
  platform/forward_flight_benchmarks/run_mancini2017_v4b.py \
  platform/tests/test_mancini2017.py
```

通过条件：`8 passed`，其余命令退出码为 0。测试明确覆盖：几何/运动、周期载体
在评分窗内不污染、整翼数据角色、阈值禁止调节、实验 CSV 不进入 v4b 预测、
移除增量恢复 UVLM。

### H3：快速工况 JIT smoke

输出目录必须使用新的 UTC 标识，例如：

```bash
PYTHONPATH=platform:src \
NUMBA_CACHE_DIR=/tmp/numba-mancini \
MPLCONFIGDIR=/tmp/mpl-mancini \
python -m forward_flight_benchmarks.run_mancini2017_v4b \
  --quality smoke \
  --cases fast_pitch \
  --output docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820/runs/replay_<UTC>_fast_smoke
```

预期值，允许因平台/JIT 舍入出现 `1e-10` 绝对误差：

- UVLM RMSE：`1.4804654209360024`；
- v4b RMSE：`1.2368967436876148`；
- RMSE 变化：`-16.452169284331877%`；
- 501 个评分点；
- `v4b_improves_uvlm=true`；
- 所有载荷、LDVM 增量和指标有限。

原 JIT smoke 耗时 13.8 秒；首次编译或不同 CPU 可以更慢，耗时不是科学门。

### H4：正式快、慢双工况

只有 H0--H3 全通过才执行：

```bash
PYTHONPATH=platform:src \
NUMBA_CACHE_DIR=/tmp/numba-mancini \
MPLCONFIGDIR=/tmp/mpl-mancini \
python -m forward_flight_benchmarks.run_mancini2017_v4b \
  --quality full \
  --cases fast_pitch slow_pitch \
  --output docs/forward_flight_large_pitch/reproductions/mancini2017_v4b_20260820/runs/replay_<UTC>_full
```

原运行耗时 `2043.82 s`，约 34.1 分钟。建议新复现以 45 分钟作为操作预算；
超过预算或进程异常时保留现状并报告 STOP，不得降低分辨率后仍称 full。

预期 headline，允许 `1e-9` 绝对误差：

| 工况 | UVLM RMSE | v4b RMSE | v4b 相对变化 |
|---|---:|---:|---:|
| 快俯仰 | `1.4004268692194932` | `1.2184155280412512` | `-12.996847259841802%` |
| 慢俯仰 | `0.3777715428377396` | `0.2907950175610002` | `-23.023577854327048%` |

额外检查：

- 每个工况/模型各 501 个评分点；
- 快工况实验峰值约 `4.8289@t*=0.96`，v4b 预测峰值约
  `7.0426@t*=0.03`；
- 慢工况实验峰值约 `2.0482@t*=4.97`，v4b 预测峰值约
  `2.2055@t*=4.46`；
- v4b 对两工况 RMSE 均改善，但快速工况的幅值和相位仍明显失败。

### H5：独立结果重算与文件审计

不得只相信 stdout。至少完成：

1. 从 replay `scored_samples.csv` 独立按工况/模型重算
   `sqrt(mean((prediction_CL-experiment_CL)^2))`；
2. 与 replay `summary.json` 比较，容差 `5e-15`；
3. 重算 replay summary 中全部 `result_hashes`；
4. 检查 `accuracy_metrics.csv` 的 RMSE、MAE、bias、correlation 和峰值；
5. 确认正式输出包含：
   `digitized_experiment.csv`、`scored_samples.csv`、`accuracy_metrics.csv`、
   `run.log`、`summary.json`、`run_manifest.json`、两张 PNG 和两张 PDF；
6. 检查 manifest 的 `numba_disable_jit=unset`、命令、Python 和运行时间。

原 full 的独立重算与 summary 最大差为 `2.22e-16`；summary 登记的 8 个结果
文件哈希全部一致。

### H6：中文图与结论边界

人工打开：

- `figures/mancini2017_lift_comparison_zh.png`；
- `figures/mancini2017_geometry_chord_span_zh.png`。

必须确认：

- 中文没有方框或缺字；
- `x/c` 明确标为弦向；
- `y/b` 明确标为展向；
- 实验、UVLM、冻结 v4b 图例不混淆；
- 快/慢工况各自成图，不池化；
- 报告没有把整翼 `C_L` 写成展向载荷分布。

## 7. 停止条件与禁止动作

出现下列任一情况立即 STOP，并保留日志：

- 冻结 SHA 不匹配；
- 数字化 CSV 不能逐字节重放；
- H2 任一测试或静态门失败；
- 运动端点、约化俯仰率、前缘轴或 AR=4 几何漂移；
- `Lcrit` 不再严格等于 `0.11`；
- 预测在实验 CSV 加载后才开始，或实验观测进入参数选择；
- 任何 NaN/Inf、点数不是 501、full 设置不是 4×12/64/96；
- full 超过 45 分钟仍未完成；
- 中文图无法正确显示弦向/展向；
- 新结果只改善池化分数、却隐藏某个工况回归。

禁止：

- 调整阈值、增益、相位、滤波、轴位、攻角或实验曲线；
- 用用户排除的六篇近二维/厚翼文献替换 Mancini；
- 覆盖现有 canonical run；
- 将 smoke/full 两档称作网格收敛；
- 将“v4b 比 UVLM 好”升级为“快速俯仰定量吻合”；
- 将 PIV 的三维观察升级为“已验证展向载荷”。

## 8. 接手 agent 的最终交付

在复现目录新增 `REPLAY_REPORT_<UTC>.md`，至少记录：

1. H0--H6 的 PASS/FAIL/STOP；
2. 实际命令、环境、运行时间和新输出目录；
3. 数字化、测试、smoke、full 的实际结果；
4. 新 summary/manifest SHA 及全部 result-hash 复核；
5. 与本交接期望值的最大数值差；
6. 中文图人工检查结果；
7. 最终结论只能从以下三项选择：
   - `REPRODUCED_DIAGNOSTIC_PASS`：机械、数据、两工况指标均重现；
   - `REPLAY_STOP`：外部预算/环境阻断，不能签数值结论；
   - `REPRODUCTION_FAIL`：冻结条件下结果或证据不一致。

若为 `REPRODUCED_DIAGNOSTIC_PASS`，允许重复的结论仅为：冻结 v4b 在 Mancini
快、慢俯仰整翼 `C_L` 上分别把 RMSE 相对 UVLM 降低约 13.0% 和 23.0%；快速
工况仍有重大峰值/相位失配，且没有展向载荷或网格收敛证据。

## 9. 后续研究计划（不属于本次复现）

复现完成后如要继续开发，必须另立计划并重新冻结：

1. 至少三级空间/时间网格矩阵，建立实际收敛而非 smoke/full 对照；
2. 单独登记快速俯仰的初始峰值幅值、峰值时间和后续振荡指标；
3. 引入能表达 5% 厚度与圆钝前缘的模型，或把该误差保留为明确限制；
4. 寻找同类有限翼实验的展向 PIV/分段载荷，才能评价展向结构；
5. 若修改 v4b/三维 LEV 耦合，必须建立新方法版本，不得回写本次零拟合证据。
