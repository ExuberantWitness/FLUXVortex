# Izraelevitz et al. (2017) Figure 14 / Scherer (1968) 实验复现 HANDOFF

日期：2026-08-27（Asia/Shanghai）

仓库：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca`

当前审计 HEAD：`2f43ba33677b671f19e988736f53ae147ca4ca59`

任务性质：刚性有限翼、纯气动、前进来流、12 个真实运动工况、14 个实验 marker

目标：先可重复复核历史 GPU 结果，再把同一论文 CASE 迁移到当前 FLUX-V5M mandatory integrated separated-LEV + joint-TEV + free-wake 数据通路。

---

## 0. 接手者先读结论

1. 本 HANDOFF 只复现 **Izraelevitz et al. (2017) Figure 14 中 Scherer 1968 的 open-square 实验数据**。Figure 11 是数值 UVLM 比较，不是真实实验；论文只有 Figure 1--15，没有 Figure 17。
2. 真实对象是有前进来流的 **NACA 63A015、AR=3、矩形有限翼**，不是椭圆翼、不是悬停、不是二维翼型，也不是一端贴壁的半翼。完整物理翼跨中对称，左右两个外翼尖均为自由翼尖。
3. 共有 **12 个唯一运动条件、14 个实验 marker**。`15 deg/15 deg` 与 `15 deg/75 deg` 各有两个实验 marker；主评分必须让 14 个 marker 分别计权，不能先平均为 12 点。
4. 项目保存的历史最佳为 `CT MAE=0.01745211311116545`。这是 CUDA float64 的历史 V5M GPU V2 结果，但计算路径是 `enable_lev=False + prescribed_wake=True + post-hoc LDVM delta`，**不满足当前 mandatory 物理合同**，只能用作回归基线。
5. 当前统一架构的 `src/fluxvortex/cases/izraelevitz2017.py` 仍是空注册表占位，而且错误写成“elliptical wing / hovering flapper”。接手者必须先纠正科学对象，再接通生产 V5M；不能在错误占位上继续补丁。
6. 本 CASE 是刚性纯气动，不需要 Q16、Q9、Q4 或 FSI。不得建立任何缩减翼、toy 网格或假 GT。代码级小测试只验证接口，论文验收必须直接覆盖全部 12 个真实工况。

当前状态建议标记为：

```text
PASS_FROZEN_HISTORICAL
PENDING_CURRENT_MANDATORY
```

即：历史结果和真实数据已经闭合；当前 mandatory separated-LEV/TEV/free-wake 生产复现尚未闭合。

---

## 1. 论文、实验源和本地材料

### 1.1 2017 论文

- J. S. Izraelevitz, Q. Zhu, M. S. Triantafyllou, “State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span,” *AIAA Journal*, 55(4), 1279--1294, 2017。
- DOI：<https://doi.org/10.2514/1.J055144>
- MIT 作者稿入口：<https://dspace.mit.edu/handle/1721.1/120112>
- 本地 PDF：

```text
/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/docs/
forward_flight_large_pitch/literature/candidates_20260807/
izraelevitz_zhu_triantafyllou_2017_state_space_ullt.pdf
```

- PDF 大小：`1,009,350 bytes`
- PDF 页数：`16`
- PDF SHA-256：

```text
68d1a5fca17479eb857d327b632ab6762a7cf6b363633a9157d5b50400077304
```

Figure 14 位于 PDF 第 13 页。只把 open black squares、图例 `Scherer 1968` 作为实验 GT。CSV 中作者的 six-state ULLT、one-state ULLT、quasi-steady + added-mass 曲线只是数值参考，禁止当成实验点。

### 1.2 Scherer 1968 原始实验报告

- J. O. Scherer, “Experimental and Theoretical Investigation of Large Amplitude Oscillation Foil Propulsion Systems,” Hydronautics, Inc., Technical Report `TR-662-1-F`, 1968。
- DTIC accession：`AD0673776`
- 公开目录记录：<https://trid.trb.org/View/783>
- 原始报告当前未保存在本仓库；本地数字化说明记录了用于闭合几何和工况的原报告 Figure 23、Figure 29 及静态表。

Scherer Figure 23 对应 `theta_max=15 deg, J'=6`，Figure 29 对应 `theta_max=25 deg, J'=6`。2017 论文把这些实验点重绘到 Figure 14。

### 1.3 权威本地 GT 和数字化说明

GT CSV：

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/
source_data/izraelevitz2017_fig14_digitized.csv
```

SHA-256：

```text
993f410c5d4857a221e57c616bf45beb5eaef5391a2deafb0b6e48e6d083b3cf
```

数字化协议：

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/
source_data/DIGITIZATION_FIG14.md
```

独立完整性审计：

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/
FIG14_EXPERIMENT_AUDIT.md
```

CSV 共 56 行（含表头），同时包含实验 marker 和三条作者数值曲线。评分时必须显式过滤：

```python
row["data_role"] == "experimental_observation"
```

作者曲线标识为 `data_role=numerical_reference`；不得将二者混池评分。`theta_max=25 deg` 的 QS+added-mass 曲线在 `psi=15 deg` 超出图框，CSV 没有该数值，禁止补造。

### 1.4 数字化方法与误差条限制

Figure 14 是 PDF 矢量图。现有数据用 PyMuPDF 直接读取路径、square marker 中心与误差条端点，不是截图后人工光标取点。坐标映射为：

```text
top panel:    y=134.1401825 -> CT=0
              y=56.2197571  -> CT=0.35
bottom panel: y=241.9563904 -> CT=0
              y=164.4504700 -> CT=0.25
x axis:       x=67.7190552  -> psi=0 deg
              x=276.8464050 -> psi=120 deg
```

Figure 14 没有定义误差条是标准差、标准误还是置信区间。因此：

- CSV 只把上下半跨度记为 `ct_error_minus/plus`；
- 主评分不按误差条加权；
- 不得称残差为 z-score 或显著性；
- 可附报“预测是否落入所画误差条”，但它不是主门。

---

## 2. 科学工况合同

### 2.1 几何和边界

| 参数 | 数值/定义 | 来源和实现说明 |
|---|---:|---|
| 翼型 | NACA 63A015 | 对称、15% 厚；现有 UVLM 只使用其零弯度中面 |
| 平面形 | 矩形有限翼，翼尖略圆 | 现有 builder 采用矩形翼尖近似 |
| 弦长 `c` | `4 in = 0.1016 m` | chordwise，前缘到后缘 |
| 全展长 `b` | `12 in = 0.3048 m` | spanwise，左自由翼尖到右自由翼尖 |
| 半展长 | `0.1524 m` | 仅为镜像建模尺度，不是物理壁面 |
| 展弦比 `AR=b/c` | `3.0` | 矩形翼 `S=bc` |
| 参考面积 `S` | `0.03096768 m^2` | 全翼面积 |
| 俯仰轴 | `x/c=0.75` | 四分之三弦线 |

坐标语义固定为：

```text
x: chordwise，前缘 -> 后缘
y: spanwise，跨中 -> 外翼尖（全翼为 -b/2 ... +b/2）
z: thickness-normal / heave normal
```

现有 Ptera builder 用半翼加 `symmetric=True` 镜像得到完整翼。这个 `y=0` 是数学镜像面，不是风洞壁面，也不是固定端。迁移到统一 `SurfaceFrame` 时，推荐直接展开成完整 `[-b/2,+b/2]` 网格，明确保存两个自由翼尖，避免把计算镜像条件误写成物理墙面。

NACA0012 只可作为 Ptera 中“相同零弯度中面”的适配对象；不能在报告中把实验翼型改名为 NACA0012，也不能声称当前薄面 UVLM 已解析 15% 厚度边界层。

### 2.2 来流和流体

| 参数 | 数值 |
|---|---:|
| 前进来流 `U` | `10 ft/s = 3.048 m/s` |
| 密度 `rho` | `1000 kg/m^3` |
| 运动粘度 `nu` | `1.0e-6 m^2/s` |
| Reynolds 数 `U c / nu` | `309676.8` |
| 动压面积 `qS=0.5 rho U^2 S` | `143.84958068736 N` |

这不是 Wu/Zimmerman 类无来流悬停扑翼；任何 case 描述出现 `hovering` 都是错误。

### 2.3 运动规律

论文运动定义：

```text
z(t)     = h * cos(omega*t)
theta(t) = theta_max * cos(omega*t + psi)
```

参数：

| 参数 | 数值 |
|---|---:|
| `h/c` | `0.6` |
| `h` | `0.06096 m` |
| `St = h*omega/(pi*U) = 2fh/U` | `0.2` |
| `J' = U/(f c)` | `6` |
| `f` | `5 Hz` |
| `T` | `0.2 s` |
| `omega` | `31.41592653589793 rad/s` |
| `k=omega*c/(2U)` | `pi/6 = 0.52359877559830` |
| `theta_max` | `15 deg` 或 `25 deg` |

Ptera 的运动接口使用 sine 表达，所以现有 builder 用 `heave phase=90 deg` 实现 `cos`，用 `pitch phase=psi+90 deg` 实现论文的 pitch law。迁移后必须直接按解析式同时生成位置、姿态、线速度和角速度；不得只更新几何而留下旧速度，也不得做预测—实验最优相位平移。

### 2.4 12 个唯一运动条件

```text
theta_max=15 deg: psi = 15, 30, 45, 60, 75, 90, 105 deg
theta_max=25 deg: psi =         45, 60, 75, 90, 105 deg
```

不存在 `25/15` 和 `25/30` 的实验 GT。不得为凑矩阵补造这两个点。

---

## 3. 14 个实验 marker：不得平均重复观测

| `theta_max` | `psi` | replicate | 实验 `CT` | 下误差条 | 上误差条 |
|---:|---:|---:|---:|---:|---:|
| 15 | 15 | 1 | 0.123091624 | 0.011190235 | 0.011190029 |
| 15 | 15 | 2 | 0.144850206 | 0.011190269 | 0.011190064 |
| 15 | 30 | 1 | 0.179661361 | 0.011187596 | 0.011192668 |
| 15 | 45 | 1 | 0.234370817 | 0.011189755 | 0.011190509 |
| 15 | 60 | 1 | 0.230641242 | 0.011192805 | 0.011190064 |
| 15 | 75 | 1 | 0.212612201 | 0.011189687 | 0.011190509 |
| 15 | 75 | 2 | 0.205152638 | 0.011190235 | 0.010568313 |
| 15 | 90 | 1 | 0.191475779 | 0.011190235 | 0.011190029 |
| 15 | 105 | 1 | 0.163500449 | 0.011190269 | 0.011190064 |
| 25 | 45 | 1 | 0.095981895 | 0.011160582 | 0.011160779 |
| 25 | 60 | 1 | 0.106249906 | 0.011160952 | 0.011160459 |
| 25 | 75 | 1 | 0.084374756 | 0.011160410 | 0.011161001 |
| 25 | 90 | 1 | 0.043303596 | 0.010714543 | 0.011160607 |
| 25 | 105 | 1 | 0.012053646 | 0.011160779 | 0.011160632 |

一个运动条件只产生一个预测 `CT_pred(theta,psi)`；若该条件有两个 marker，同一个预测分别与两个实验值形成两项误差。

主评分：

```text
MAE_CT = (1/14) * sum_j |CT_pred(theta_j,psi_j) - CT_exp,j|
```

附加诊断可报告：

```text
RMSE_CT = sqrt((1/14) * sum_j error_j^2)
bias_CT = (1/14) * sum_j error_j
max_abs_error_CT = max_j |error_j|
```

不得做：

- 先把两个重复 marker 平均为一个点；
- 用 12 个 condition mean 代替 14-marker MAE；
- 按误差条反比加权；
- 对预测做相位、幅值或均值拟合；
- 用作者数值曲线替代 Scherer 实验。

---

## 4. 推力、阻力和载荷所有权

历史 runner 使用：

```text
CT_raw = mean(force_W[last_cycle, 0]) / (0.5*rho*U^2*S)
```

在该历史坐标约定中，`force_W[:,0]` 的正方向被当作推力。迁移后必须用一个真实工况做坐标审计，确认 GP1/body/world/wing 的变换与历史定义一致；不能看到符号不合就直接取绝对值或反号。

### 4.1 `Cd0` 来源冲突

- Izraelevitz 2017 Figure 14 的无粘预测统一采用 `Cd0=0.057`；主复现固定采用这一值。
- Scherer 1968 原始静态表记录 `Cd0=0.027`；只允许作为预声明敏感性。
- 不得根据 Figure 14 误差选择二者。

历史 V2 计算式为：

```text
CT = CT_raw - 0.057 + delta_CT_posthoc_LDVM
```

当前 mandatory 迁移要求 separated LEV 在主求解器中成为实时状态和唯一载荷数据通路。因此迁移后：

1. 不得再叠加历史 `delta_CT_posthoc_LDVM`，否则分离贡献双计；
2. `Cd0=0.057` 只进入一次；若生产 ledger 已经拥有该项，scorer 不得再次减去；
3. manifest 必须记录 `surface_load_owner`、`separated_load_owner`、`profile_drag_owner`；
4. predictor/proposal 必须推进实际 bound circulation、LEV、TEV 和 detached/free-wake 状态；只有 accepted proposal 可以 commit。

---

## 5. 已有代码与工件清单

### 5.1 权威工况和运动代码

| 角色 | 路径 | 当前用途 |
|---|---|---|
| 几何/流场 dataclass | `platform/forward_flight_benchmarks/cases.py::IzraelevitzSchererCase` | 权威参数源，可复用 |
| Ptera 运动 builder | `platform/forward_flight_benchmarks/ptera_adapter.py::build_izraelevitz_scherer_movement` | 历史几何/运动 oracle，可复用合同，不作为最终统一入口 |
| 早期完整 Figure 14 runner | `platform/forward_flight_benchmarks/run_izraelevitz_scherer_experiment.py` | CPU/Ptera/ULLT 历史诊断，不是当前 GPU 验收 |
| 早期绘图 | `platform/forward_flight_benchmarks/plot_izraelevitz_scherer_experiment.py` | 可复用图形布局 |
| 工况合同测试 | `platform/tests/test_forward_flight_benchmarks.py` | 只验证几何/运动接口，不代表论文精度通过 |
| 14/12 计数测试 | `platform/tests/test_fluxv_v5_all_conditions.py` | 防止重复 marker 被误平均 |
| `0.75c` 源项测试 | `platform/tests/test_fluxv_v5c_ledger.py` | 防止俯仰轴回退 |

`build_izraelevitz_scherer_movement()` 的默认 `quality="full"` 仍是 `4 chordwise x 12 spanwise-per-semispan`；冻结 GPU V2 明确覆盖成：

```python
settings=(8, 12, 128, 4)
```

即 8 个弦向面元、每半翼 12 个展向面元、128 步/周期、4 周期。物理全翼相当于 24 个展向面元；Ptera movement 含初始状态，所以每工况保存 513 个状态、512 次时间推进。

### 5.2 历史探索脚本：只能考古，不能作为当前验收

| 路径 | 已做过什么 | 不能作为当前验收的原因 |
|---|---|---|
| `platform/warp_vpm/izra_runner.py` | V5H15 bare-core 12 条件试跑 | 硬编码 `/tmp/fluxv-v5-nextgen`、使用 `smoke`、prescribed wake，输出到临时目录 |
| `platform/warp_vpm/bing_izra_v2.py` | attached chassis + 冻结 LDVM delta，首次把 MAE 降到约 0.0178 | `JointConfig(enable_lev=False)`、prescribed wake、post-hoc 分离增量 |
| `platform/warp_vpm/bing_p3_izra_baik.py` | 动态 viscous ledger 探索 | 同样关闭 integrated LEV；属于方案探索 |
| `platform/warp_vpm/BING_SESSION_RESULTS.md` | 记录 0.0386 -> 0.0260 -> 0.0178 的历史演化 | 会混合多个不同阶段/指标，只作考古 |

`bing_izra_v2.py` 对应提交：

```text
43b696c4558017c1b012e5fc56f92db06badcdb3
feat: Izra gap closed — chassis + frozen LDVM delta beats V4B
```

该提交定位到历史精度改进来自 Scherer 静态极限推导的 `Lcrit` 和 LDVM 分离增量，但这不是 integrated separated-LEV 的生产实现。

### 5.3 冻结 GPU V2 scorer 和结果

主历史 runner：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
run_three_papers_gpu_only.py::run_izraelevitz()
```

结果：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
fresh_results/gpu_only_three_papers/summary.json
```

结果文件当前 SHA-256：

```text
63bd07fa49c8d761e45e58f99869031478ffdb9d059fc4da05d10527057ee5b8
```

GPU scorer 复核器：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
validate_gpu_only_v2.py
```

重要事实：`run_three_papers_gpu_only.py` 没有 paper selector，会一起运行 Yang、Izraelevitz 和 Mancini。为了保持冻结源码哈希，精确 replay 时不要修改这个文件；后续另建聚焦 Izraelevitz 的正式 runner。

历史 `_run_chassis()` 明确包含：

```python
JointConfig(enable_lev=False)
solver.run(prescribed_wake=True, ...)
```

最新保存结果中每个 Izraelevitz 工况的：

```text
dvm_source_steps = 0
dvm_ribbon_shed = 0
dvm_frontier_advance = 0
particle_* = 0
```

因此该结果不能证明 current mandatory separated LEV、joint TEV 或自由尾迹已经在 Figure 14 上通过。

### 5.4 早期完整实验复现工件

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/
runs/20260812_scherer_fig14_experiment_full/
```

其中：

- `mean_thrust_vs_phase.csv`：所有实验与模型值；
- `accuracy_metrics.csv`：14-marker 指标；
- `profile_drag_sensitivity.csv`：固定 `Cd0` 敏感性；
- `run_manifest.json`：几何、运动、每条件审计；
- `summary.json`：汇总；
- `run.log`：运行日志。

独立审计给出的早期基线：

| 模型 | MAE CT | RMSE CT | Bias CT |
|---|---:|---:|---:|
| local one-state ULLT | 0.033879 | 0.046361 | 0.015025 |
| old FluxV | 0.034841 | 0.051147 | 0.026983 |
| authors' one-state ULLT | 0.045836 | 0.063212 | 0.045392 |
| authors' six-state ULLT | 0.050136 | 0.067037 | 0.050136 |
| authors' QS + added mass | 0.097814 | 0.112079 | 0.097814 |
| failed periodic v1/v2 | 0.183314 | 0.222598 | -0.167865 |

这些数值的意义是历史比较，不是 current mandatory 验收。

### 5.5 V4B 参考

```text
docs/forward_flight_large_pitch/reproductions/unified_fluxv_v4_ldvm_stevens_20260812/
runs/20260812_fluxv_v4b_crosspaper_full/
izraelevitz2017_fig14_v4_mean_thrust.csv
```

同一 14-marker MAE 口径约为 `0.0198`。V4B 是本项目参考，不是实验 GT。

### 5.6 当前 mandatory 生产模板和统一架构

| 角色 | 路径 | 接手时的正确理解 |
|---|---|---|
| mandatory LEV/TEV/free-wake 模式模板 | `platform/warp_vpm/reproduce_mancini_v5m_mandatory.py` | 只复用工程合同，不能复用 Mancini 几何、运动、`Lcrit=0.11`、步数或评分 |
| V5M GPU 入口 | `platform/warp_vpm/flux_v5m_gpu.py` | 目标生产入口/能力注册 |
| prescribed rigid kinematics | `src/fluxvortex/kinematics/prescribed_rigid.py` | 应输出准确 `SurfaceFrame` 与速度 |
| one-way coupling | `src/fluxvortex/coupling/one_way.py` | 刚性论文 CASE 的 predictor/commit 控制 |
| V5M stepper | `src/fluxvortex/aero/v5m/stepper.py` | 当前仍依赖 raw structural-state bridge，需接通 rigid native 3D 路径 |
| case protocol | `src/fluxvortex/cases/protocol.py` | 统一 case contract |
| Izraelevitz case placeholder | `src/fluxvortex/cases/izraelevitz2017.py` | 当前错误且未实现，必须修复 |
| U3 测试 | `tests/test_u3_rigid_cases.py` | 当前反而断言 `IZRA_CASES == {}`，迁移后应改成 12 条件合同 |

`src/fluxvortex/cases/izraelevitz2017.py` 当前两处科学错误必须删掉：

```text
elliptical wing
hovering flapper
```

正确描述应为：

```text
rectangular AR=3 finite wing in forward water flow,
prescribed heave/pitch about x/c=0.75,
12 unique motion conditions and 14 Scherer markers
```

---

## 6. 历史 GPU V2 最佳结果

### 6.1 逐条件预测

| `theta/psi` | 历史预测 `CT` | 实验 `CT` | marker 绝对误差 |
|---|---:|---:|---:|
| 15/15 | 0.147064489010 | 0.123091624, 0.144850206 | 0.023972865, 0.002214283 |
| 15/30 | 0.182160361626 | 0.179661361 | 0.002499001 |
| 15/45 | 0.208679943782 | 0.234370817 | 0.025690873 |
| 15/60 | 0.232853334144 | 0.230641242 | 0.002212092 |
| 15/75 | 0.224968182229 | 0.212612201, 0.205152638 | 0.012355981, 0.019815544 |
| 15/90 | 0.199706340807 | 0.191475779 | 0.008230562 |
| 15/105 | 0.193885953856 | 0.163500449 | 0.030385505 |
| 25/45 | 0.086285230479 | 0.095981895 | 0.009696665 |
| 25/60 | 0.130480751086 | 0.106249906 | 0.024230845 |
| 25/75 | 0.089259705214 | 0.084374756 | 0.004884949 |
| 25/90 | 0.071809003074 | 0.043303596 | 0.028505407 |
| 25/105 | 0.061688657531 | 0.012053646 | 0.049635012 |

聚合指标：

```text
N marker          = 14
MAE_CT            = 0.01745211311116545
RMSE_CT           = 0.02198753256178541
bias_CT           = +0.01239675057702133
max_abs_error_CT  = 0.04963501153092067 at theta=25, psi=105
```

最大残差在 `25/105`，同时整体 bias 为正。current mandatory 结果如果整体恶化，优先检查载荷坐标、`Cd0`/分离贡献是否双计和尾迹事务，而不是先调释放阈值。

### 6.2 运行环境证据

最新保存的三论文联合运行：

```text
GPU: NVIDIA GeForce RTX 4090 D
Torch: 2.11.0+cu130
CUDA: 13.0
dtype: float64
GPU utilization peak: 100%
GPU peak memory: 13889 MiB
combined elapsed: 2981.939 s
```

这是 Yang + Izraelevitz + Mancini 的总运行时间，不得写成单独 Figure 14 的成本。`dc43d4` 原始冻结联合工件记录约 `448.385 s`；后续源码和监测范围变化使墙钟不同，但 Izraelevitz 12 个预测和 MAE 保持一致。

### 6.3 退化与修复历史

Q16/FSI 开发曾改坏 `bing_joint_ptera_gpu.py` 的 GP1 -> W 载荷变换，使 Izraelevitz `CT MAE` 从约 `0.0175` 恶化到约 `0.465`。根因不是 `Lcrit`，而是共享三维底盘的坐标变换被替换。

修复提交：

```text
7b46905ca0c6f363d972e0c5428f0ef7eaf06f4e
fix: restore Pterra GP1-to-W force transform in _calculate_loads
```

修复报告：

```text
artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/
THREE_PAPER_REGRESSION_FIX_20260824.md
```

这段历史给出明确故障优先级：若 Figure 14 突然大幅退化，先查共享 frame transform 和 load owner，不要先改论文工况或 LEV 阈值。

---

## 7. 冻结历史结果的精确 replay

### 7.1 为什么使用 `dc43d4` 而不是 summary 的 `fa8eaca`

历史 summary 的：

```text
base_commit=fa8eaca9bcaa4b963ecf41683bf77d3c9e3df169
```

只是早期科学基线标签，不是包含全部 GPU runner/backend 的完整源码快照。原始冻结 V2 源码可重建提交为：

```text
dc43d4cc0c2290ee34df942e51cbb05e13afbb0d
```

该提交中的关键 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `run_three_papers_gpu_only.py` | `600b351402d2a6d25c1d6c18aa0005926da424251f8b192fe9d87f5e9b515e27` |
| `bing_joint_ptera_gpu.py` | `9c0eb2f57fcd8d7e8b0c4186286106149e0195b40bc2ed0c0a9a16916480fb3a` |
| `bing_gpu_corrections.py` | `62fdb2e331446ea710cb47f441b052284c92d36925771eaa95589e44240bf954` |
| `ldvm_torch_gpu.py` | `31f19712679efa5af92d9bf1b9b9d3c998d6a11a421cf67c4f975ced5bb7b6d1` |
| `gpu_runtime_monitor.py` | `8ebd06525ae68f3a4a3bb3ebce6537acd5c508c1d52d25577369606330e4d83d` |

### 7.2 隔离 worktree 和 GPU 环境

不要回退当前开发树；使用隔离 worktree：

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
git worktree add /tmp/fluxv-v5m-izra-frozen-dc43d4 \
  dc43d4cc0c2290ee34df942e51cbb05e13afbb0d
cd /tmp/fluxv-v5m-izra-frozen-dc43d4

export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0
export FLUXV_DEVICE=cuda:0
export FLUXV_GPU_ONLY=1
export FLUXV_DTYPE=float64
export FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

先核对源码和 GT 哈希，再计算：

```bash
sha256sum \
  artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/run_three_papers_gpu_only.py \
  platform/warp_vpm/bing_joint_ptera_gpu.py \
  platform/warp_vpm/bing_gpu_corrections.py \
  platform/warp_vpm/ldvm_torch_gpu.py \
  platform/warp_vpm/gpu_runtime_monitor.py \
  docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv
```

运行原始冻结 runner：

```bash
python artifacts/experiment/20260820_fluxv_v5m_four_paper_gpu_validation/\
run_three_papers_gpu_only.py
```

注意：该命令还会运行 Yang 和 Mancini。精确 replay 阶段不要为节约时间编辑冻结 runner；需要单论文快速迭代时另建正式聚焦 runner，并把它视为新的源码身份。

### 7.3 replay 通过标准

- GT SHA-256 必须为 `993f410c...b3cf`；
- 12 个唯一条件、14 个 marker；
- 12 个 `CT` 预测与上表相符；
- 同一冻结软件/硬件栈建议逐条件容差 `1e-12`；其他兼容 GPU 可放宽到 `1e-9`，但必须报告每条件差异；
- 14-marker `MAE_CT=0.01745211311116545`；
- 不比较整个 JSON 文件哈希作为数值等价条件，因为 elapsed、GPU UUID、监测样本数会变化；
- replay 只证明历史基线可复现，不证明 current mandatory 物理合同通过。

---

## 8. 当前统一 FLUX-V5M 的实现计划

### H0：冻结科学输入和评分

完成条件：

- manifest 记录论文 PDF、GT CSV、数字化说明的路径和 SHA-256；
- `data_role` 分离实验和数值参考；
- 注册 12 个 motion case，不注册 14 个运动 case；
- scorer 按 14 marker 计权；
- 固定 `Cd0=0.057` 为主设置，`0.027` 仅为显式 sensitivity；
- 所有输入错误、缺列、重复点结构变化都非零退出。

### H1：修复统一 case 的科学对象

修改：

```text
src/fluxvortex/cases/izraelevitz2017.py
tests/test_u3_rigid_cases.py
```

`IzraCaseConfig` 至少应冻结：

```text
case_id
theta_max_deg
phase_offset_deg
chord_m
span_m
area_m2
pivot_fraction_chord
freestream_m_s
rho_kg_m3
nu_m2_s
heave_amplitude_m
frequency_hz
profile_drag_coefficient
ground_truth_path
ground_truth_sha256
```

建立 12 个配置，例如：

```text
IZRA-15-015 ... IZRA-15-105
IZRA-25-045 ... IZRA-25-105
```

测试必须从 `assert IZRA_CASES == {}` 改为：

- 恰好 12 个配置；
- 两个幅值族/相位集合精确；
- `span/chord=3`、`pivot=0.75c`、`h/c=0.6`；
- forward flow 非零；
- GT 映射后 14 marker；
- 不出现 `elliptical` 或 `hovering`。

### H2：接通完整刚性翼 `SurfaceFrame`

使用 `PrescribedRigidSurfaceKinematics`，但参考几何必须是完整 AR=3 矩形有限翼，明确 chordwise/spanwise/normal 三个方向。推荐展开全翼，避免统一框架把中面镜像误解释为墙面边界。

必须用解析运动同时提供：

```text
position/orientation
linear velocity
angular velocity
panel-node velocity
```

对照 oracle：

```text
platform/forward_flight_benchmarks/ptera_adapter.py::
build_izraelevitz_scherer_movement
```

对照只检查几何和运动合同；最终载荷不能绕回 CPU Ptera 路径。

### H3：让刚性 `SurfaceFrame` 直接驱动 native V5M stepper

当前 `V5M3DStepper` 仍要求 `set_structural_state()` 的 Q16 raw-state bridge。这对刚性论文 CASE 是不合适的耦合泄漏。应实现一个生产级刚性 native 3D 适配路径，使 V5M 从 `SurfaceFrame` 直接获得：

- 气动节点和面元；
- 节点速度；
- TE/LE 拓扑；
- bound circulation 的 parent/proposal 状态；
- LEV/TEV/free-wake 的 detached state；
- 唯一表面载荷 owner。

不要为此创建 Q4/Q9/Q16 中间结构或假翼。该 paper runner 与 Q16 FSI 共享的是 `SurfaceFrame + aero transaction` 合同，不共享结构单元。

### H4：实现聚焦 Figure 14 的正式 mandatory runner

建议新文件：

```text
platform/warp_vpm/reproduce_izraelevitz2017_fig14_v5m_mandatory.py
```

它应直接跑 12 个真实工况，并：

- `cuda:0`、float64；无 CUDA 时 fail-fast，禁止 CPU 数值 fallback；
- `enable_lev=True`；
- `joint_tev=True`；
- `prescribed_wake=False`；
- predictor/proposal 推进真实 bound/LEV/TEV/free-wake 状态；
- rejected proposal 不污染 parent；accepted proposal 只 commit 一次；
- LEV 始终集成，但只在声明的释放条件满足时新生；不得每步无条件释放；
- 保存每步 pre/post LESP、release event、LEV/TEV 数量、wake convection、load ledger；
- 只从唯一 integrated load owner 生成 `CT`；
- `Cd0=0.057` 只加一次；
- 不调用 post-hoc `run_ldvm_separation_pair_cuda()` 给最终载荷再加 `delta_CT`。

`reproduce_mancini_v5m_mandatory.py` 只能借鉴上述工程模式。绝对不能复制 Mancini 的：

```text
geometry
kinematics
Lcrit=0.11
time grid
score definition
```

### H5：直接运行完整 12 条件论文矩阵

正式科学运行固定：

```text
8 chordwise panels
12 spanwise panels per semispan / 24 full-span
128 steps per cycle
4 cycles
score last complete cycle
```

不得以 `smoke`、单工况或缩减网格代替论文验收。如果调试阶段崩溃，可以复现某个真实条件的失败，但它不是独立科学 gate；最终必须一次性生成全部 12 个预测和 14-marker 指标。

### H6：对比和结论

同一报告并列四类结果：

1. Scherer 实验 GT；
2. authors' one-state/six-state/QS 数值参考；
3. 历史 V4B 与 GPU V2；
4. current mandatory V5M。

不同计算路径必须标清，不能把历史 GPU V2 写成 current mandatory，也不能只因 MAE 更小就把 post-hoc 路径宣称为更物理。

---

## 9. current mandatory 验收节点

### Gate G0：输入闭合

- PDF/GT 哈希正确；
- 12 conditions / 14 markers；
- 两个重复 marker 保留；
- 几何为矩形 AR=3；
- 前进来流 `U=3.048 m/s`；
- `theta/z` 运动和 `0.75c` 轴精确；
- 失败非零退出。

### Gate G1：GPU 数值合同

- 所有 AIC、诱导速度、线性求解、涡状态推进、载荷与评分在 CUDA float64；
- CPU 只做调度/I/O/哈希；
- 记录 GPU 名、Torch/CUDA、峰值显存、利用率、每工况墙钟；
- 检测到 CPU 数值 fallback 立即 FAIL。

### Gate G2：LEV/TEV/free-wake 物理合同

- `enable_lev=True` 且 separated LEV 是主求解状态；
- LEV 新生严格服从声明 release condition；某些步或某工况 release count 为零不等于关闭，但必须有逐步证据；
- joint TEV 被求解并有非空历史；
- `prescribed_wake=False`，每个 accepted step 推进真实自由尾迹；
- predictor 内 bound/LEV/TEV/wake proposal 与 commit 是同一份状态；
- 没有 post-hoc LDVM 分离载荷双计；
- 任一项不满足，状态为 `FAIL_PHYSICS_CONTRACT`，即使 MAE 很好也不能通过。

### Gate G3：论文完整性

- 12/12 条件均完成；
- 每个条件最后一完整周期 128 个样本；
- 14 个 marker 全部评分；
- 失败时仍保存已完成条件、残差和失败原因，runner 非零退出；
- 不以 toy/smoke 结果填补失败条件。

### Gate G4：数值精度

项目冻结参考：

```text
MAE_CT <= 0.01745211311116545
max_abs_error_CT <= 0.04963501153092067
```

这是 current mandatory 的目标门，不保证换成真实 integrated LEV/free-wake 后自然达到。若物理合同全部通过但超门，应标记：

```text
FAIL_ACCURACY_WITH_VALID_PHYSICS
```

并保存逐条件差异，不能关闭 LEV、切回 prescribed wake 或重新叠加 post-hoc delta 来“修指标”。

建议同时报告相对 V4B `0.0198`、作者 one-state `0.045836`、authors six-state `0.050136` 的变化，但主门仍是当前项目冻结最佳。

### Gate G5：回归隔离

Figure 14 修改后至少运行：

```bash
PYTHONPATH=src:platform:platform/warp_vpm pytest -q \
  platform/tests/test_forward_flight_benchmarks.py \
  platform/tests/test_fluxv_v5_all_conditions.py \
  platform/tests/test_fluxv_v5c_ledger.py \
  tests/test_u3_rigid_cases.py
```

这些测试只验证合同。科学 PASS 必须来自正式 12-condition CUDA runner。

由于历史上共享载荷变换曾同时破坏 Yang 和 Izraelevitz，若修改 `bing_joint_ptera_gpu.py`、frame transform 或 surface-load owner，还必须重跑已有三篇刚性/纯气动回归矩阵，不能只看本 CASE。

---

## 10. 正式输出工件要求

建议输出目录：

```text
artifacts/baselines/fluxv_v5m_izraelevitz2017_fig14_<YYYYMMDD_HHMMSS>/
```

至少保存：

```text
manifest.json
predictions.csv
metrics.json
physics_evidence.json
summary.json
run.log
fig14_ct_vs_phase.png
fig14_ct_vs_phase.pdf
```

`predictions.csv` 每行至少包含：

```text
case_id
theta_max_deg
phase_offset_deg
replicate
ct_prediction
ct_experiment
signed_error
abs_error
ct_error_minus
ct_error_plus
```

`physics_evidence.json` 每工况至少包含：

```text
enable_lev
joint_tev
prescribed_wake
release_condition_source
lesp_pre/post extrema
lev_release_count
tev_shed_count
free_wake_convection_count
proposal_count
accepted_commit_count
rejected_proposal_count
parent_state_unchanged_on_reject
surface_load_owner
profile_drag_owner
posthoc_separation_delta_applied=false
cuda kernel counters
```

图形要求：

- 上图 `theta_max=15 deg`，下图 `theta_max=25 deg`；
- 横轴 `psi [deg]`，纵轴 mean `CT`；
- Scherer marker 和上下误差条按原值绘制；
- current mandatory、历史 GPU V2、V4B、作者数值曲线分开图例；
- 另附几何示意，中文标注“弦向 x”“展向 y”“厚度/法向 z”“前进来流 U”“0.75c 俯仰轴”；
- 不把数学镜像面画成风洞壁面。

manifest 必须记录：

- Git HEAD、dirty 状态和改动文件；
- runner、V5M backend、case config、GT 的 SHA-256；
- GPU/Torch/CUDA；
- 网格、步数、周期数；
- 所有物理开关和 release-condition 来源；
- `Cd0` 所有权；
- 12 条件完成状态；
- 总指标和最大残差位置。

---

## 11. 失败定位顺序

出现 FAIL 时按以下顺序查，禁止先调参：

1. **GT 身份**：CSV 哈希、`data_role`、12/14 计数、重复 marker。
2. **科学对象**：是否误用椭圆翼、悬停、半翼壁面、NACA0012 名称或错误参考面积。
3. **运动**：`cos`/`sin` 转换、`psi` 单位、`0.75c` 轴、节点速度与几何是否同步。
4. **坐标和归一化**：GP1 -> W 变换、推力正号、全翼 `S`、`qS`。
5. **载荷所有权**：surface load、profile drag、LEV impulse/LDVM delta 是否双计。
6. **事务**：predictor 是否推进真实 LEV/TEV/free wake，reject 是否污染 parent，commit 是否重复。
7. **释放条件**：LEV 是否集成但按条件释放；不得以“本步无释放”误判为关闭，也不得复制 Mancini `Lcrit=0.11`。
8. **数值分辨率**：前七项闭合后才检查周期收敛、尾迹长度和网格；不创建 toy 替代。
9. **模型能力**：合同全部通过后，剩余误差才可判为 current V5M 模型误差。

最重要的历史教训：大幅精度退化曾由共享载荷坐标变换导致，而不是 LESP 阈值。不要用调 `Lcrit` 掩盖 frame/load-owner bug。

---

## 12. 接手完成定义

只有同时满足以下条件才可写“Figure 14 已由当前 FLUX-V5M 复现”：

- [ ] 2017 PDF、Scherer 身份和 GT 哈希闭合；
- [ ] `src/fluxvortex/cases/izraelevitz2017.py` 已纠正为矩形有限翼、前进来流并注册 12 条件；
- [ ] 完整翼几何、解析刚性运动和节点速度进入统一 `SurfaceFrame`；
- [ ] native V5M 直接消费该 rigid `SurfaceFrame`，没有 Q16 raw-state 假桥；
- [ ] CUDA float64，无 CPU 数值 fallback；
- [ ] separated LEV 100% 集成并按 release condition 释放；
- [ ] joint TEV 和 free wake 在 predictor/commit 中真实推进；
- [ ] 唯一载荷 owner，`Cd0=0.057` 一次且无 post-hoc separation delta；
- [ ] 12/12 条件、14/14 marker 完成；
- [ ] `MAE_CT <= 0.01745211311116545`，最大 marker 误差无未报告恶化；
- [ ] 结果、逐点残差、物理证据、GPU 证据、源码/GT 哈希完整保存；
- [ ] 所有门失败时非零退出，不删除失败工件；
- [ ] 共享底盘相关刚性翼回归没有被破坏。

若只完成冻结 replay，状态只能写：

```text
PASS_FROZEN_HISTORICAL / PENDING_CURRENT_MANDATORY
```

若 current mandatory 物理合同通过但精度没达到历史门，状态写：

```text
PASS_PHYSICS_CONTRACT / FAIL_ACCURACY
```

不得为了得到绿色总状态而关闭 separated LEV、改用 prescribed wake、引入 toy、修改 GT 或叠加历史 post-hoc 修正。
