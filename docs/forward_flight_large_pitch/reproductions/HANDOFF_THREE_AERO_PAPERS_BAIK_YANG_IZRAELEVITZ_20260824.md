# 三篇纯气动论文复现交接：Baik、Yang、Izraelevitz–Scherer

日期：2026-08-24（Asia/Shanghai）  
目标模型：FLUX-V5M，CUDA float64 生产数据面  
接手目标：用真实论文工况和真实实验数据，复现三篇纯气动 CASE；先复核冻结结果，再完成当前 mandatory separated-LEV + joint-TEV + free-wake 路径的迁移与验收。

---

## 0. 先读结论

本交接中的“三篇”固定为：

1. Yeon Sik Baik，低雷诺数俯仰—升沉平板，W1–W4；项目历史名称常写“Baik 2012”，但权威实验载荷源是 2011 年博士论文 Fig. 5.24–5.27。
2. H.-H. Yang 等，2025，刚性仿鸟扑翼，Figure 11 Type A。
3. J. S. Izraelevitz、Q. Zhu、M. S. Triantafyllou，2017，Figure 14；真实实验点来自 Scherer 1968。

不包含 Mancini 2017。历史脚本 `run_three_papers_gpu_only.py` 的“三篇”实际指 Yang + Izraelevitz + Mancini，这是文件名遗留陷阱；原始三篇论文矩阵应读作 Baik + Yang + Izraelevitz。

当前状态不是“三篇均已由现生产路径验证”：

| CASE | 冻结实验复现 | 当前 mandatory 生产合同 | 当前可发布状态 |
|---|---|---|---|
| Baik W1–W4 | 已完成 | separated LEV、TEV 历史、自由尾迹、CUDA 均有 fresh PASS | **已合格，但当前树仍须 fresh 重跑绑定新源码哈希** |
| Yang Fig. 11 | 历史 GPU V2 已完成 | 历史三维底盘为 `enable_lev=False + prescribed_wake=True` | **待迁移，不得把历史数值冒充当前生产结果** |
| Izraelevitz Fig. 14 | 历史 GPU V2 已完成 | 历史底盘同样不是当前 mandatory 路径，且分离增量为 post-hoc | **待迁移，不得把历史数值冒充当前生产结果** |

“当前 SOTA”在本文中只表示**本项目、同一真实数据、同一评分口径下的最佳已保存结果**。三篇没有统一的公开全球排行榜，工况、数据与指标也不统一，因此禁止写成“领域全球 SOTA”。

---

## 1. 不可变合同

### 1.1 科学与工程约束

1. 只跑论文真实工况；不得新建 Q4、Q9、Ptera toy、缩尺假翼、虚构运动或假 GT。
2. Yang 与 Izraelevitz 的生产复现必须同时开启 separated LEV、joint TEV 和 free wake；任一关闭即 FAIL。
3. 影响气动力、涡诱导、矩阵求解、尾迹推进、载荷修正和误差评分的数值计算必须在 CUDA float64 上完成；不允许 CPU 数值 fallback。
4. CPU 只允许承担操作系统、Python 调度、论文几何对象构造、文件 I/O、哈希和 JSON/NPZ 序列化；不得参与科学数值预测。
5. 不得通过改变 GT、符号、相位、幅值、均值、`Lcrit`、`Cd0`、步数或网格来追指标。
6. 不得做预测—实验最优相位对齐、幅值拟合、偏置修正或按预测自身范围归一化。
7. separated LEV 必须遵守释放条件；“LEV 始终存在”不等于“每步无条件释放”。必须保存 pre/post LESP、释放布尔量、LEV/TEV 状态和尾迹推进证据。
8. 当前三篇任务是刚性纯气动 CASE。Q16/FSI 是另一条集成线，不得用 Q16 结构门替代这三篇的气动精度门，也不得把 Q4/Q9 中间单元带入本任务。

### 1.2 坐标和符号

- 弦向：从前缘到后缘，记为 chordwise / `x`。
- 展向：从翼根到翼尖，记为 spanwise / `y`。
- 厚度/法向：翼面法向，记为 thickness-normal / `z`。
- Yang 数据文件给的是升力和推力；评分阻力必须取 `drag_gf = -test_thrust_gf`。
- Baik 使用论文 `CL`、`CD` 定义；`CD < 0` 表示净推力，不能擅自重命名系数。
- Izraelevitz Figure 14 使用周期平均推力系数 `CT`。

---

## 2. 仓库、环境和冻结身份

仓库根目录：

```text
/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
```

当前工作树包含未提交 Q16/FSI 和 mandatory 气动开发，不得清理、覆盖或回退用户修改。历史复现必须另建隔离 worktree。

### 2.1 GPU 环境

```bash
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0
export FLUXV_DEVICE=cuda:0
export FLUXV_GPU_ONLY=1
export FLUXV_DTYPE=float64
export FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

冻结 GPU V2 环境为 RTX 4090 D、Torch 2.11.0+cu130、CUDA 13.0。换 GPU、CUDA、Torch 或融合核后允许墙钟和最后若干舍入位变化，但 GT、评分定义和物理开关不能变化。

### 2.2 冻结源码快照：使用 `dc43d4`，不要误用 summary 的 `fa8eaca`

历史 summary 写有：

```text
base_commit = fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169
```

该字段表示早期科学基线来源，不是完整的可重建源码快照。四个 GPU 后端和两个 runner 在 `fa8eaca` 尚不存在。经 Git blob 哈希核验，能完整匹配冻结 V2 源码的提交是：

```text
dc43d4cc0c2290ee34df942e51cbb05e13afbb0d
```

冻结哈希：

| 文件 | SHA-256 |
|---|---|
| `run_baik_gpu_only.py` | `2e7d43d63e248bb293ce5ff23801cc0128c81e66c5b183e238fc30e5e1d80b13` |
| `run_three_papers_gpu_only.py` | `600b351402d2a6d25c1d6c18aa0005926da424251f8b192fe9d87f5e9b515e27` |
| `bing_joint_ptera_gpu.py` | `9c0eb2f57fcd8d7e8b0c4186286106149e0195b40bc2ed0c0a9a16916480fb3a` |
| `bing_gpu_corrections.py` | `62fdb2e331446ea710cb47f441b052284c92d36925771eaa95589e44240bf954` |
| `ldvm_torch_gpu.py` | `31f19712679efa5af92d9bf1b9b9d3c998d6a11a421cf67c4f975ced5bb7b6d1` |
| `gpu_runtime_monitor.py` | `8ebd06525ae68f3a4a3bb3ebce6537acd5c508c1d52d25577369606330e4d83d` |

隔离复核：

```bash
git worktree add /tmp/fluxv-v5m-three-paper-frozen \
  dc43d4cc0c2290ee34df942e51cbb05e13afbb0d
cd /tmp/fluxv-v5m-three-paper-frozen
```

开始计算前逐项执行 `sha256sum`；任何源码或 GT 哈希不匹配都先停止，不得把不匹配运行称作冻结复现。

---

## 3. 三篇工况总览

| 论文/数据 | 真实观测量 | 完整工况数 | 主指标 | 冻结项目最佳值 |
|---|---|---:|---|---:|
| Baik W1–W4 | 周期相位 `CL/CD` | 4 个工况、8 条曲线 | 每工况相位 RMSE，再作四工况宏平均 | `CL 0.42156277`，`CD 0.28970979` |
| Yang Fig. 11 Type A | 周期平均升力/推力 | 6 个安装迎角 | 六点 MAE，升力和阻力分开 | `4.10482660 / 1.51862559 gf` |
| Izraelevitz Fig. 14 / Scherer | 周期平均 `CT` | 12 个唯一运动条件、14 个实验 marker | 14-marker MAE | `0.0174521131` |

禁止把 gf、`CL/CD` 和 `CT` 归一后拼成一个总分。三篇必须分别报告。

---

## 4. CASE A：Baik W1–W4

### 4.1 论文和实验源

- 权威源：Yeon Sik Baik，*Unsteady Force Generation and Vortex Dynamics of Pitching and Plunging Airfoils at Low Reynolds Number*，University of Michigan 博士论文，2011。
- 官方页面：<https://deepblue.lib.umich.edu/items/e1cec17c-27e7-46c6-8956-704134cb257a>
- PDF 直链：<https://deepblue.lib.umich.edu/bitstream/handle/2027.42/84556/yeonb_1.pdf?isAllowed=y&sequence=1>
- 预期 PDF：29,993,055 bytes，SHA-256 `2efbf3becd339df61cc9275e2e933700ef75216504580d5f4e5cca1e80eadc0a`。
- 实验 GT：论文 Fig. 5.24–5.27，PDF 物理页 204–207；W1/Fig. 5.24、W2/Fig. 5.25、W3/Fig. 5.26、W4/Fig. 5.27。
- 不得替换为早期 AIAA relative-load 曲线；早期曲线删除了稳态水动力且没有 W4。

### 4.2 几何和真实运动条件

共同几何：弦长 `c=0.076 m`，跨度 `b=0.600 m`，厚度比 `t/c=6.25%`，前后缘圆角半径 `0.002375 m`，俯仰轴 `0.25c`，`Re=5000`，水密度 `998.2 kg/m³`。试验翼近乎横跨 `0.61 m` 水槽，底部约 `1 mm` 间隙并有自由面端板，因此是壁面约束的准二维实验，不是自由端 AR=7.895 机翼。

| Case | `k` | `h0/c` | 标称 `St` | 表中俯仰幅值 | 周期 `T` |
|---|---:|---:|---:|---:|---:|
| W1 | 0.5 | 0.50 | 0.16 | 13.16° | 7.13 s |
| W2 | 1.0 | 0.50 | 0.32 | 33.73° | 3.56 s |
| W3 | 1.0 | 0.25 | 0.16 | 13.16° | 3.56 s |
| W4 | 0.5 | 1.00 | 0.32 | 33.73° | 7.13 s |

W3 的 `k=1.0`；早期 AIAA 表中的 `0.5` 是误印。升沉位移不是正弦，应按：

```text
h_dot/U = -tan(alpha_pl,max * sin(2*pi*t/T))
```

周期积分得到位移。`alpha_pl,max` 为 W1/W3 `27.182110°`，W2/W4 `47.755954°`；四个关键相位的有效迎角为 `8°, 22°, 8°, -6°`。

### 4.3 真实数据取值与哈希

权威公共 401 点文件：

```text
docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/source_data/
  baik2012_w1_w4_corrected_total_cl_cd.csv
```

SHA-256：`4de6b01cd8072959e5b780053f311efa92ab5a94f17940dd122df340ad638f2f`。每个工况相位为 `0, 0.0025, ..., 1.0`，`phase=1` 是周期重复端点；评分使用 400 个唯一相位。

| 通道 | 点数（含重复端点） | 最小值 | 最大值 | 公共网格均值 |
|---|---:|---:|---:|---:|
| W1 CL | 401 | -0.440000 | 2.860000 | 1.038764 |
| W1 CD | 401 | -0.310891 | 0.297030 | 0.032518 |
| W2 CL | 401 | -0.544118 | 4.647059 | 2.108575 |
| W2 CD | 401 | -2.307789 | 1.694724 | -0.124738 |
| W3 CL | 401 | -0.333333 | 2.751244 | 1.139484 |
| W3 CD | 401 | -0.231167 | 0.382456 | 0.127297 |
| W4 CL | 401 | -0.548544 | 4.286408 | 1.369043 |
| W4 CD | 401 | -1.914392 | 0.606700 | -0.302072 |

冻结 GPU scorer 实际读取：

```text
docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/
  20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv
```

其 SHA-256 为 `66fbff1b6fb922440096dacd60e207f5b6220ef3cebb710dc360d57a60740ff2`。必须只读 `experiment` 列并按 `(case_id, quantity, phase)` 去重；不得误读同文件的历史 `prediction` 列为 GT。

源实验约有 `±0.02` 系数不确定度；读图不确定度另为：W1/W3 `CL ±0.02, CD ±0.004`，W2 `CL ±0.03, CD ±0.023`，W4 `CL ±0.03, CD ±0.020`。这些不确定度不能用于移动或缩放预测。

### 4.4 参考程序

| 角色 | 路径 |
|---|---|
| 工况、非简谐运动和源合同 | `platform/forward_flight_benchmarks/baik2012.py` |
| 历史完整 benchmark | `platform/forward_flight_benchmarks/run_baik2012_benchmark.py` |
| 数值敏感性 | `platform/forward_flight_benchmarks/run_baik2012_sensitivity.py` |
| 冻结 GPU V2 scorer | `artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/run_baik_gpu_only.py` |
| 当前 mandatory runner | `platform/warp_vpm/reproduce_baik_v5m_mandatory.py` |
| GT 提取 | `docs/.../baik2012_w1_w4/source_data/extract_baik2012_fig524_527.py` |
| Theodorsen 提取 | `docs/.../baik2012_w1_w4/source_data/extract_baik2012_fig528_531_theodorsen.py` |
| 当前 mandatory 结果 | `artifacts/baselines/fluxv_v5m_four_case_20260822/results/baik/summary.json` |

### 4.5 冻结数值设置和评分

- `LDVM2DCuda`，`cuda:0`，float64。
- `ndiv=32`，`naterm=14`，`3 × 512` 步，`max_wake=256`，`pivot=0.25c`，`core_rc=0.02c`。
- 冻结最佳使用 `Lcrit=0.19`。正文转移值 `0.11` 与表值 `0.19` 存在来源冲突；不得把 `0.19` 写成已由 Baik 数据独立识别的材料常数。
- 取最后一周期，512 点每 4 点抽取为 128 点，再按实验的 1 Hz 上限做单周期 sharp harmonic filter。
- 预测在实验相位作周期线性插值，不做相位优化。

评分：

```text
RMSE(W,q) = sqrt(mean_i((prediction(W,q,phase_i)-experiment(W,q,phase_i))^2))
macro_RMSE(q) = mean_W(RMSE(W,q)), W in {W1,W2,W3,W4}
```

宏平均是四个单 CASE RMSE 的算术平均，不是把 1600 个点混池后再算一次 RMSE。

### 4.6 当前项目最佳与对照

| Case | CL RMSE | CD RMSE | LEV 释放次数 | 尾迹推进步 |
|---|---:|---:|---:|---:|
| W1 | 0.3814217554 | 0.1762887278 | 517 | 1536 |
| W2 | 0.4224478190 | 0.3530825198 | 524 | 1536 |
| W3 | 0.3853954206 | 0.2172295402 | 479 | 1536 |
| W4 | 0.4969860763 | 0.4122383891 | 484 | 1536 |
| 宏平均 | **0.4215627678** | **0.2897097942** | — | — |

对照：V4B 为 `CL 0.6580 / CD 0.3450`；论文 standard Theodorsen 的 lift-only 宏 RMSE 为 `0.82062`。当前 V5M 结果不等于“达到实验不确定度”，只是同口径下项目最佳。

当前 mandatory artifact 的 `ldvm_torch_gpu.py` 记录哈希为 `4b4f34...`，当前工作树文件已继续变化。因此保存结果有效，但接手者必须在当前树 fresh 重跑并记录新源码哈希；不得仅引用旧 summary 就声称当前源码通过。

---

## 5. CASE B：Yang et al. 2025 Figure 11 Type A

### 5.1 论文和观测身份

- H.-H. Yang、S.-G. Lee、E.-H. Lee、J.-H. Han，*Numerical simulation framework of bird-inspired ornithopter in forward flight*，Journal of Fluids and Structures 133 (2025), 104263。
- DOI：<https://doi.org/10.1016/j.jfluidstructs.2024.104263>
- 本地正式 PDF：

```text
/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/docs/
forward_flight_large_pitch/literature/candidates_20260807/
yang_et_al_2025_jfs_plev.pdf
```

- 使用 Figure 11 的 Type A rigid wing。
- `Test` 是风洞周期均值，是真实 GT。
- `Proposed` 是作者完整 modified-UVLM 数值结果，包含 PLEV、AWS 和作者自由尾迹/涡核处理；不是 PLEV-only 消融，也不是真实观测。

### 5.2 几何、运动和流场

刚性矩形单翼：弦向 `c=0.130 m`，展向 `b=0.250 m`，厚度 `0.001 m`，翼根偏置 `0.080 m`；来流 `U=5.5 m/s`，频率 `f=2.5 Hz`，空气密度 `1.23 kg/m³`，运动粘度 `1.47e-5 m²/s`。安装迎角为 `0°, 5°, 10°, 15°, 20°, 25°`。

四杆机构名义参数：

| 参数 | 数值 |
|---|---:|
| `phi0`（论文符号） | -14.5° |
| 固定杆 | 0.0479 m |
| 曲柄 | 0.0084 m |
| 连杆 | 0.0457 m |
| 摇杆 | 0.0144 m |
| 目标下扑/上扑极值 | -30° / 40° |

论文数值计算使用未公开的激光测量运动历史。本地只能由论文四杆长度重建名义运动，所以这是明确的可比性限制；禁止用 Figure 11 载荷反求一条“更好”的运动曲线。

### 5.3 六个真实 GT 点

GT 文件：

```text
docs/forward_flight_large_pitch/reproductions/plev2025/source_data/
  yang2025_fig11_rigid_digitized.csv
```

SHA-256：`0351a59d601513a2c0e1605863f3afeeb41bf9d0dc57e73cc54b5b9167d2a8ea`。每个纵坐标读图不确定度 `±0.4 gf`，不是实验误差条。

| 安装迎角 | Test lift [gf] | Test thrust [gf] | 评分 drag = -thrust [gf] | 作者 Proposed lift [gf] | 作者 Proposed thrust [gf] |
|---:|---:|---:|---:|---:|---:|
| 0° | 2.0 | 0.5 | -0.5 | 0.0 | 0.0 |
| 5° | 17.4 | -0.1 | 0.1 | 15.8 | -1.4 |
| 10° | 31.5 | -5.1 | 5.1 | 29.8 | -5.2 |
| 15° | 38.7 | -14.1 | 14.1 | 40.2 | -10.8 |
| 20° | 42.9 | -21.0 | 21.0 | 48.2 | -17.5 |
| 25° | 45.3 | -27.8 | 27.8 | 53.3 | -24.9 |

### 5.4 参考程序

| 角色 | 路径 |
|---|---|
| 工况 dataclass | `platform/forward_flight_benchmarks/cases.py` 的 `Yang2025RigidCase` |
| 四杆运动/翼面 builder | `platform/forward_flight_benchmarks/ptera_adapter.py` |
| 作者 PLEV/AWS 参考实现 | `platform/forward_flight_benchmarks/yang_plev.py` |
| 作者模型交叉复现 | `platform/forward_flight_benchmarks/run_yang2025_crosscase.py` |
| 冻结 GPU V2 scorer | `artifacts/.../run_three_papers_gpu_only.py` 的 `run_yang()` |
| 当前生产迁移模板 | `platform/warp_vpm/reproduce_mancini_v5m_mandatory.py` |
| GT 说明 | `docs/.../plev2025/source_data/DIGITIZATION.md` |

`reproduce_mancini_v5m_mandatory.py` 只能复用 mandatory node-owned LEV/TEV/free-wake 的工程模式，不能复用 Mancini 的几何、运动、步数、阈值或评分。

### 5.5 冻结评分与历史最佳

- 六个安装迎角都使用论文完整运动，不允许只跑 `15°` 后宣称 Figure 11 复现。
- 冻结网格 `8` 个弦向面元 × `12` 个展向面元，`128` 步/周期，4 周期；评分周期均值。
- 升力与阻力分别计算六点 MAE：

```text
MAE_lift = mean_a(abs(predicted_lift_gf(a)-test_lift_gf(a)))
MAE_drag = mean_a(abs(predicted_drag_gf(a)+test_thrust_gf(a)))
```

冻结 V5M GPU V2 预测：

| 迎角 | 预测 lift [gf] | GT lift | 预测 drag [gf] | GT drag |
|---:|---:|---:|---:|---:|
| 0° | -0.149347 | 2.0 | -1.283694 | -0.5 |
| 5° | 12.277693 | 17.4 | -0.036691 | 0.1 |
| 10° | 24.044423 | 31.5 | 4.028620 | 5.1 |
| 15° | 34.629377 | 38.7 | 10.610495 | 14.1 |
| 20° | 43.594446 | 42.9 | 19.091972 | 21.0 |
| 25° | 50.436660 | 45.3 | 29.522456 | 27.8 |

聚合：`lift MAE=4.1048265958 gf`，`drag MAE=1.5186255897 gf`。V4B 为 `4.55 / 2.64 gf`。作者 Proposed 为 `3.350 / 1.933 gf`：作者在升力上更好，冻结 V5M 在阻力上更好，不能笼统写“全面超过作者模型”。

历史 V2 路径的 `_run_chassis()` 明确使用 `JointConfig(enable_lev=False)`，因此上述数值只是迁移前精度 reference，不是当前 mandatory V5M 合格结果。

---

## 6. CASE C：Izraelevitz 2017 Figure 14 / Scherer 1968

### 6.1 论文和真实实验身份

- J. S. Izraelevitz、Q. Zhu、M. S. Triantafyllou，*State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span*，AIAA Journal 55(4), 2017, 1279–1294。
- DOI：<https://doi.org/10.2514/1.J055144>
- MIT 作者稿入口：<https://dspace.mit.edu/handle/1721.1/120112>
- 本地 PDF：

```text
/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/docs/
forward_flight_large_pitch/literature/candidates_20260807/
izraelevitz_zhu_triantafyllou_2017_state_space_ullt.pdf
```

只用 Figure 14 的 Scherer 1968 open-square 实验点作为真实 GT。Figure 11 是作者 UVLM 数值参考，不是实验；论文没有 Figure 17。

### 6.2 几何、运动和流场

- NACA 63A015，矩形有限翼，翼尖略圆。
- 弦向 `c=4 in=0.1016 m`，展向 `b=12 in=0.3048 m`，`AR=3`。
- 俯仰轴 `0.75c`。
- `U=10 ft/s=3.048 m/s`，水密度 `1000 kg/m³`，运动粘度 `1e-6 m²/s`。
- `h/c=0.6`，`St=0.2`，`k≈pi/6≈0.5236`，`J'=6`，对应 `f=5 Hz`。
- 运动：`z(t)=h cos(omega t)`，`theta(t)=theta_max cos(omega t+psi)`。
- `theta_max=15°` 和 `25°`；相位差见真实数据表。
- 2017 Figure 14 对所有无粘预测加 `Cd0=0.057`；Scherer 原始静态表给出 `0.027`。主复现固定跟随 2017 的 `0.057`，`0.027` 只能作为预声明敏感性，不能按误差择优。

### 6.3 14 个真实实验 marker

GT 文件：

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/
  source_data/izraelevitz2017_fig14_digitized.csv
```

SHA-256：`993f410c5d4857a221e57c616bf45beb5eaef5391a2deafb0b6e48e6d083b3cf`。评分只取 `data_role=experimental_observation`。

| `theta_max` | `psi` | replicate | 实验 `CT` | 下误差条 | 上误差条 |
|---:|---:|---:|---:|---:|---:|
| 15° | 15° | 1 | 0.123091624 | 0.011190235 | 0.011190029 |
| 15° | 15° | 2 | 0.144850206 | 0.011190269 | 0.011190064 |
| 15° | 30° | 1 | 0.179661361 | 0.011187596 | 0.011192668 |
| 15° | 45° | 1 | 0.234370817 | 0.011189755 | 0.011190509 |
| 15° | 60° | 1 | 0.230641242 | 0.011192805 | 0.011190064 |
| 15° | 75° | 1 | 0.212612201 | 0.011189687 | 0.011190509 |
| 15° | 75° | 2 | 0.205152638 | 0.011190235 | 0.010568313 |
| 15° | 90° | 1 | 0.191475779 | 0.011190235 | 0.011190029 |
| 15° | 105° | 1 | 0.163500449 | 0.011190269 | 0.011190064 |
| 25° | 45° | 1 | 0.095981895 | 0.011160582 | 0.011160779 |
| 25° | 60° | 1 | 0.106249906 | 0.011160952 | 0.011160459 |
| 25° | 75° | 1 | 0.084374756 | 0.011160410 | 0.011161001 |
| 25° | 90° | 1 | 0.043303596 | 0.010714543 | 0.011160607 |
| 25° | 105° | 1 | 0.012053646 | 0.011160779 | 0.011160632 |

`15°/15°` 和 `15°/75°` 的重复 marker 必须分别保留；主 MAE 权重是 14 个观测，不是先平均成 12 个点。原论文没有定义误差条的统计含义，不得称作标准差、置信区间或 z-score 权重。

### 6.4 参考程序

| 角色 | 路径 |
|---|---|
| 工况 dataclass | `platform/forward_flight_benchmarks/cases.py` 的 `IzraelevitzSchererCase` |
| 运动/翼面 builder | `platform/forward_flight_benchmarks/ptera_adapter.py` |
| 冻结 GPU V2 scorer | `artifacts/.../run_three_papers_gpu_only.py` 的 `run_izraelevitz()` |
| 早期 Figure 14 完整实验 runner | `platform/forward_flight_benchmarks/run_izraelevitz_scherer_experiment.py` |
| 当前生产迁移模板 | `platform/warp_vpm/reproduce_mancini_v5m_mandatory.py` |
| GT 说明 | `docs/.../unified_fluxv_upgrade_20260812/source_data/DIGITIZATION_FIG14.md` |
| 独立审计 | `docs/.../unified_fluxv_upgrade_20260812/FIG14_EXPERIMENT_AUDIT.md` |

### 6.5 冻结评分与历史最佳

- 冻结网格 `8 × 12`，`128` 步/周期，4 周期，底盘每工况 513 步。
- 12 个唯一运动条件产生 12 个周期平均预测；与 14 个 marker 逐一匹配。
- 主评分：

```text
MAE_CT = (1/14) * sum_j(abs(prediction(theta_j,psi_j)-CT_experiment_j))
```

- 冻结 2017 实现采用 `CT = CT_raw - 0.057 + delta_CT`；当前 mandatory 迁移不得在 integrated separated-LEV 载荷之外再次叠加同一 LDVM separation delta，否则是分离载荷双计。

冻结 V5M GPU V2 预测：

| `theta_max/psi` | 预测 `CT` | 实验 `CT` |
|---|---:|---|
| 15/15 | 0.147064489 | 0.123091624, 0.144850206 |
| 15/30 | 0.182160362 | 0.179661361 |
| 15/45 | 0.208679944 | 0.234370817 |
| 15/60 | 0.232853334 | 0.230641242 |
| 15/75 | 0.224968182 | 0.212612201, 0.205152638 |
| 15/90 | 0.199706341 | 0.191475779 |
| 15/105 | 0.193885954 | 0.163500449 |
| 25/45 | 0.086285230 | 0.095981895 |
| 25/60 | 0.130480751 | 0.106249906 |
| 25/75 | 0.089259705 | 0.084374756 |
| 25/90 | 0.071809003 | 0.043303596 |
| 25/105 | 0.061688658 | 0.012053646 |

聚合 `MAE_CT=0.0174521131`；V4B 同一 MAE 口径为 `0.0198`。作者 one-state ULLT 为 `0.045836`，six-state ULLT 为 `0.050136`。早期报告中的 `0.02595` 是 RMSE，不得与这里的 MAE 直接比较。

历史路径先运行 `enable_lev=False` 的 attached/prescribed-wake 底盘，再 post-hoc 加 LDVM 分离增量，所以同样只是迁移前 reference。

---

## 7. 精确复核冻结结果

### 7.1 Baik

在 `dc43d4` 隔离 worktree：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
PFIELD_DEVICE=cuda:0 FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
FLUXV_DTYPE=float64 \
python artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/\
run_baik_gpu_only.py
```

预期输出：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
fresh_results/baik_gpu_only/summary.json
```

必须重算四个 NPZ 的 128 点 `phase/CL/CD` 与每 CASE RMSE。运行时间、GPU 遥测样本数不参与逐位复核。

### 7.2 Yang 与 Izraelevitz

冻结 runner 没有 paper selector，而且会额外跑 Mancini。为了保持冻结源码哈希，精确 replay 阶段不要编辑它：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
PFIELD_DEVICE=cuda:0 FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
FLUXV_DTYPE=float64 FLUXV_V5M_FUSE=1 \
python artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/\
run_three_papers_gpu_only.py
```

只从输出的 `yang` 与 `izraelevitz` 字段提取本交接结果。不要把额外 Mancini 计入三篇矩阵。

预期输出：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
fresh_results/gpu_only_three_papers/summary.json
```

历史整体验证：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
python artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/\
validate_gpu_only_v2.py
```

该验证器覆盖四论文历史包；在本交接中只把 Baik、Yang、Izraelevitz 对应项目作为验收证据。

### 7.3 冻结复核通过标准

1. 六个源码哈希和三个 GT 哈希全部匹配第 2、4、5、6 节。
2. Baik 四个曲线 `result_sha256` 分别为：
   - W1 `b6d0287c28b1821bdb646c65d406c760761cac6f3b6e2dde4bb22daadf12f3f6`
   - W2 `dcbda5fd91807bdb7ef68922e8015060428b3376ffb09f3b830f7e4e49cc50c7`
   - W3 `e73d9b04e2d5c0bc506b41496cde1f9edd33f7a14c9a063e5c3114ed06e02b47`
   - W4 `950b5970b1342ea26a0385e60ff188f73ccb45656491fc02df5d58962b739f3f`
3. 同 GPU/软件栈要求主要指标绝对差 `<=1e-12`；不同兼容 GPU 栈可放宽到 `<=1e-9`，但必须解释并保存曲线差，而不是只比聚合数。
4. `numerical_device` 必须是 `cuda:0`，监测到 GPU 利用率；任何 CPU fallback 或非零数值 CPU oracle 均 FAIL。
5. 禁止比较包含 elapsed time/遥测采样数的整个 JSON 哈希，因为墙钟和遥测会自然变化；比较科学曲线、指标和源码/GT 哈希。

---

## 8. 当前 mandatory 路径迁移计划

### 节点 H0：输入与评分冻结

交付：一个 manifest，记录三篇论文身份、全部 GT 哈希、工况表、运动 builder、评分函数和当前 Git HEAD。

通过标准：

- 三个 GT 哈希逐位匹配；
- Yang 符号测试确认 `drag=-thrust`；
- Izraelevitz 保留 14 marker/12 condition；
- Baik 保留 400 个唯一相位/通道；
- 不读取任何预测列作为 GT。

### 节点 H1：冻结基线 replay

按第 7 节执行完整矩阵。该节点只回答“历史 reference 是否可重建”，不回答当前 mandatory 三维模型是否合格。

### 节点 H2：Baik 当前树 fresh mandatory

直接运行完整 W1–W4，不跑 toy：

```bash
PYTHONPATH=src:platform:platform/warp_vpm \
PFIELD_DEVICE=cuda:0 FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
FLUXV_DTYPE=float64 \
python platform/warp_vpm/reproduce_baik_v5m_mandatory.py
```

通过标准：

- 四个真实 CASE 全部完成；
- 每个 CASE `LEV release count > 0`、`TEV history count > 0`、`wake_convection_steps=1536`；
- `CL macro RMSE <= 0.4215627678` 且 `CD macro RMSE <= 0.2897097942`；
- runner、LDVM、GT、Git HEAD 新哈希写入 summary；失败非零退出。

### 节点 H3：Yang 当前 mandatory 完整六工况

新建或扩展一个正式 paper runner，参考：

- 工况/评分：冻结 `run_three_papers_gpu_only.py::run_yang()`；
- mandatory 物理数据通路：`reproduce_mancini_v5m_mandatory.py`；
- 唯一生产入口和能力矩阵：`platform/warp_vpm/flux_v5m_gpu.py`。

不得复用历史 `_run_chassis()` 的 `enable_lev=False`。必须用 node-owned DVM connected ribbon、integrated separated LEV、joint TEV、free wake，并由唯一 surface-load owner 产出载荷；不能再叠加第二份 LEV impulse/separation load。

通过标准：

- 六个真实迎角全部完成，不以单角 smoke 代替；
- 每个工况保存 LEV 释放、TEV、free-wake、GPU 计数和完整周期平均；
- `lift MAE <= 4.1048265958 gf`；
- `drag MAE <= 1.5186255897 gf`；
- 最大单点误差不得无报告恶化；冻结参考最大值约为 lift `7.456 gf`、drag `3.490 gf`；
- 若精度失败，仍保存所有六点和残差，非零退出，先定位载荷 owner/运动/符号，不准调 GT 或阈值。

### 节点 H4：Izraelevitz 当前 mandatory 完整 12 条件/14 marker

同样直接跑完整论文矩阵，不建立缩减 toy。integrated separated-LEV 路径必须取代历史 post-hoc LDVM delta；`Cd0=0.057` 只能记一次。

通过标准：

- 12 个真实运动条件全部完成并映射到 14 个 marker；
- `MAE_CT <= 0.0174521131`；
- 最大 marker 绝对误差不高于冻结参考约 `0.04964`，否则必须逐点报告；
- 两个重复 marker 保持独立权重；
- 不按误差条加权，不补造 `25°/15°` 或 `25°/30°` 实验值；
- 失败非零退出并保存 12 个预测。

### 节点 H5：三篇联合验收

最终报告至少包含：

1. 三篇全部条件的预测—GT 表与中文图；几何图必须标明弦向、展向、厚度/法向。
2. Baik 八条周期曲线；Yang 六点升力/阻力；Izraelevitz 两个幅值族的 `CT-psi` 曲线及误差条。
3. 项目历史 reference、当前 mandatory 结果、V4B、作者参考四列；不同指标不混算。
4. GPU 型号、CUDA/Torch、峰值显存、利用率、每工况步数、源码与 GT 哈希。
5. mandatory 证据：LEV 释放条件、TEV 非空、free-wake 推进、predictor/commit 状态一致、无双重载荷 owner。
6. 失败项和差异，不得用“总体气动精度较好”掩盖局部失败。

联合状态定义：

- `PASS_CURRENT_MANDATORY`：H0–H5 全通过。
- `PASS_FROZEN_ONLY`：历史 replay 通过，但 Yang 或 Izraelevitz mandatory 尚未达精度门。
- `FAIL_PROVENANCE`：GT/源码/论文身份不闭合。
- `FAIL_PHYSICS_CONTRACT`：LEV/TEV/free-wake/GPU 任一未满足。
- `FAIL_ACCURACY`：物理合同满足但论文指标超门。

---

## 9. 故障定位顺序

复现失败时按以下顺序查，不要先调参数：

1. **GT 身份**：哈希、列、实验/数值角色、重复点权重。
2. **符号和归一化**：Yang 推力/阻力反号，Baik `CL/CD`，Izraelevitz `CT` 和 `Cd0`。
3. **运动**：四杆重建、相位、俯仰轴、非简谐升沉、步数/周期。
4. **载荷 owner**：是否同时加了 Ptera surface load、LDVM delta 和 vortex impulse，造成双计。
5. **事务**：predictor 是否推进真实 LEV/TEV/free wake，reject 后 parent 是否未污染，accept 后是否提交同一状态。
6. **释放条件**：是否用 `abs(LESP)>Lcrit` 的声明条件，是否把“分离状态保持”和“本步新生释放”混为一谈。
7. **数值分辨率**：只在前六项闭合后做论文分辨率的时间/网格诊断；不得用 toy 网格替代。
8. **模型误差**：若以上全部通过，再将差异判为当前 V5M 模型能力不足并保存失败证据。

---

## 10. 明确禁止

- 禁止把 Mancini 计入本交接三篇。
- 禁止把 Izraelevitz Figure 11 数值曲线称作实验。
- 禁止把 Yang `Proposed` 称作 PLEV-only 或实验。
- 禁止把 Baik 早期 relative-load 曲线换入 corrected-total GT。
- 禁止关闭 separated LEV 后发布 V5M 结果。
- 禁止 `prescribed_wake=True` 后发布当前生产结果。
- 禁止先跑 Q4/Q9/Q16 toy 或 Ptera toy，再把它当论文 CASE 进度。
- 禁止用 CPU 气动求解或 CPU fallback。
- 禁止以关闭 LEV、减少论文步数、截短尾迹、改变 `Lcrit/Cd0` 来加速。
- 禁止逐工况挑阈值、相位或修正模型。
- 禁止只给矩阵残差/守恒门而不给论文真实误差；机械门通过不等于 CASE 复现通过。

---

## 11. 最小交付清单

接手 agent 完成后，应新增一个独立 artifact 目录，至少包含：

```text
MANIFEST.json
SUMMARY_ZH.md
metrics.csv
per_condition_predictions.csv
gpu_evidence.json
source_and_gt_sha256.txt
figures/
  baik_w1_w4_cl_cd_zh.png
  yang_fig11_lift_drag_zh.png
  izraelevitz_fig14_ct_zh.png
results/
  baik/*.npz
  yang/*.npz-or-json
  izraelevitz/*.npz-or-json
logs/
```

`SUMMARY_ZH.md` 必须明确回答：

- 三篇各自是否达到当前 mandatory 物理合同；
- 三篇各自精度相对冻结 V5M、V4B 和作者参考如何变化；
- 变化来自科学模型、源码迁移还是仅运行环境；
- 哪些结论是实验验证，哪些只是数值参考；
- 是否存在未闭合的源码哈希、数据来源、收敛或重复性问题。

---

## 12. 关键证据入口

- GPU V2 总报告：`artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/REPORT_GPU_ONLY_V2_20260820.md`
- GPU V2 审计：`artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/EXPERIMENT_AUDIT_V2.md`
- Baik 当前 mandatory 核验：`artifacts/baselines/fluxv_v5m_four_case_20260822/verification.md`
- Baik 数据来源：`docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/source_data/DIGITIZATION_AND_PROVENANCE.md`
- Baik 总结：`docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/SUMMARY_REPORT_ZH.md`
- Yang 数据来源：`docs/forward_flight_large_pitch/reproductions/plev2025/source_data/DIGITIZATION.md`
- Izraelevitz Figure 14 数据来源：`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/DIGITIZATION_FIG14.md`
- Izraelevitz 实验审计：`docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/FIG14_EXPERIMENT_AUDIT.md`
- V5M 使用合同：`platform/warp_vpm/HANDOFF_MODEL_USAGE.md`

最后原则：**用完整论文 CASE 快速暴露真实问题；修底层物理与数据合同，不修表面分数。**
