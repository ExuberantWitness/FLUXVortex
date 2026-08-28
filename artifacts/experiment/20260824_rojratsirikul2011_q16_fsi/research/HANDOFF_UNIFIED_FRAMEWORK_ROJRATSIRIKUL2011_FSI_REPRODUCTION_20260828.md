# Rojratsirikul et al. (2011) 统一框架 Q16–FLUX-V5M FSI 复现 HANDOFF

日期：2026-08-28（Asia/Shanghai）  
仓库：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca`  
分支：`run/q16-lev-tev-pc-fsi-20260821`  
编写前 HEAD：`1056b36`  
目标读者：正在继续统一框架 U7 及其后续工作的开发 agent

---

## 0. 先读结论

此前选定的流固耦合实验论文是：

> P. Rojratsirikul, M. S. Genc, Z. Wang, I. Gursul, “Flow-induced vibrations of low aspect ratio rectangular membrane wings,” *Journal of Fluids and Structures*, 27(8), 1296–1309, 2011. DOI: `10.1016/j.jfluidstructs.2011.06.007`。

这是一个**四边固定的三维矩形乳胶膜翼，在恒定前进来流和恒定几何攻角下产生流致变形与振动**的双向 FSI 实验。它不是规定升沉/俯仰的扑翼，也不是一端悬臂，不存在外加周期运动规律。

统一框架已经完成 U0–U6，并完成 U7 的若干基础切片，但 **Rojratsirikul CASE 尚未完成统一框架下的正式长时复现**：

- `SurfaceFrame`、`WorldOwner`、`GlobalTransaction`、`ResultStatus`、修正后的 observer、`PartitionedStrongFSI`、`V5M3DStepper` 单翼适配器已经存在；
- 当前 `V5M3DStepper` 的单翼 Q16 路径仍通过 raw structural-state bridge 调用原生求解器；
- 当前 `PartitionedStrongFSI` 仍是旧生产 `Q16NativeV5MFSIStepper` 的薄封装；
- 尚无真正完成的统一 `CaseRunner` 驱动 Roj A16 长时统计；
- `src/fluxvortex/cases/rojratsirikul2011.py` 目前只注册了 A16 主分支和 E1.4 敏感性分支，尚未容纳论文所需的 A10/A17/A23；
- 旧 runner 仍绕开统一的 `ResultStatus`、正确事务计数和部分修正后的统计门。

因此本任务不是重跑旧脚本，而是：

```text
Roj CaseDefinition
  -> Q16SurfaceFrameAdapter
  -> V5M3DStepper（mandatory separated LEV + joint TEV + free wake）
  -> Q16DynamicsAdapter
  -> PartitionedStrongFSI
  -> WorldOwner + GlobalTransaction
  -> corrected observers + block stationarity + ResultStatus
  -> 论文 A16/A17/A10/A23 正式评分
```

旧程序只作为输入、几何和同代码状态的差异 oracle。不得把旧专用 runner 继续扩展成第二套生产框架。

---

## 1. 论文和本地来源

### 1.1 权威 PDF

- 本地 PDF：`artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/references/Rojratsirikul2011_JFS.pdf`
- 作者公开 URL：<https://purehost.bath.ac.uk/ws/files/227159/Gursul_JFS_2011.pdf>
- SHA-256：

```text
c9d8f59b4fefafd846fae77fdda6376424b70032db6ae6c40f1f28d51aa9a6a4
```

正式 runner 必须在开始前校验该哈希。不能只记录 URL。

### 1.2 独立数值复现参考

- Gordnier & Attar, 2014, *Journal of Fluids and Structures*, DOI `10.1016/j.jfluidstructs.2013.10.004`。其高保真计算采用 `Re=24,300` 下约 `alpha=10°,16°,23°`，可用于理解 A16 平均态和 A10/A23 响应，但不能替代实验 GT。
- Li, Jaiman & Khoo, 2021, *JFM* 929 A33 / arXiv `2011.11422`。可作为高保真 FSI 机理参考；其数值模型、材料和边界解释必须与本 CASE 分开记录。

这些文献只用于解释差异和检查机理，不允许把其计算值写成 Rojratsirikul 2011 的实验观测。

### 1.3 实验数据所在图

| 论文图 | 用途 |
|---|---|
| Figure 2 | 矩形翼、刚性框和主要尺寸 |
| Figure 5 | `U=5 m/s` 时若干攻角的时间平均位移场 |
| Figure 6 | `zmax/c` 随攻角、来流速度变化 |
| Figure 7/8 | 中展向面平均流线、速度与脉动强度 |
| Figure 9 | 时间平均法向力系数 `Cn` |
| Figure 10 | 位移标准差 `zsd/c` 场及弦向/展向峰数 |
| Figure 11 | 最大 `zsd` 测点的位移频谱和主 Strouhal 数 |

当前图读数主要冻结在 `platform/forward_flight_benchmarks/rojratsirikul2011_q16.py` 中。接手者应把这些值同步导出为带 `figure/page/source_role=digitized_approx` 的 CSV/JSON；不能继续只靠 Python 常量传播 GT。

---

## 2. 科学对象、坐标和外部运动

### 2.1 科学坐标

- `+x`：弦向，从前缘到后缘；
- `+y`：展向；完整翼有两个自由外翼尖，但乳胶膜的四周均连接刚性框；
- `+z`：弦面法向/膜面外位移方向；
- 来流沿统一惯性系 `+x`；
- 正攻角通过将参考膜翼绕前缘展向轴刚性旋转实现，来流向量不倾斜；
- 法向力必须投影到旋转后弦面的单位法向，而不是固定世界 `+z`。

### 2.2 外部运动规律

本 CASE 没有任何扑动运动：

```text
frame_pose(t) = constant
frame_velocity(t) = 0
alpha(t) = alpha0
U_inf(t) = U0              # 实验真值
```

数值启动允许使用冻结的半余弦来流斜坡：

```text
t* = t U0 / c
U(t*)/U0 = 0.5 [1 - cos(pi t*)],  0 <= t* < 1
U(t*)/U0 = 1,                      t* >= 1
```

该斜坡只是初始瞬态协议，不是实验运动输入，且不得进入统计窗。

膜位移、平均鼓包、RMS、模态和频率全部必须由 FSI 自然产生：

```text
z(x,y,t) = mean_z(x,y) + z'(x,y,t)
```

禁止预设 `z(t)`、强制施加目标频率、给膜翼添加升沉/俯仰或用带通滤波制造论文峰值。

---

## 3. 实验几何、材料和流动参数

### 3.1 论文直接报告值

| 参数 | 值 | 来源/说明 |
|---|---:|---|
| 平面形状 | 矩形 | 正文/Figure 2 |
| 弦长 `c` | `68.8 mm = 0.0688 m` | 正文 |
| 翼展 `b` | `137.5 mm = 0.1375 m` | Figure 2；与正文 `AR=2` 一致到约 0.07% |
| 展弦比 | `AR≈2` | 正文 |
| 参考面积 `S=bc` | `0.00946 m²` | 推导值 |
| 膜材料 | 黑色乳胶橡胶 | 正文 |
| 膜厚 `t` | `0.2 mm = 2.0e-4 m` | 正文 |
| 厚弦比 `t/c` | `0.002907` | 推导值 |
| 杨氏模量 `E` | `2.2 MPa` | 正文；正式主分支 |
| 膜密度 `rho_m` | `1 g/cm³ = 1000 kg/m³` | 正文 |
| 面密度 `rho_m t` | `0.2 kg/m²` | 推导值 |
| 面内刚度基量 `Et` | `440 N/m` | 推导值 |
| 刚性框架 | 不锈钢、尖边朝内 | 正文/Figure 2 |
| 风洞 | `760 mm` 直径开口射流工作段 | 正文 |
| 端板间距 | `450 mm` | 正文 |

刚性框架首轮只作为四边不可动边界，不显式进入 Q16 结构域。不得从 `c×b` 膜域中擅自扣除框宽；若以后显式建框，必须作为独立几何分支。

### 3.2 流动工况

| `U_inf` | 论文 `Re` | 论文 `Pi1=(Et/qc)^(1/3)` |
|---:|---:|---:|
| `5.0 m/s` | `24,300` | `7.51` |
| `7.5 m/s` | `36,500` | `5.73` |
| `10.0 m/s` | `48,700` | `4.73` |

首轮正式复现固定 `U=5 m/s, Re≈24,300`。现有适配器使用：

```text
nu_air = 1.4142e-5 m²/s   # 由论文三组 U/Re 反推，不是正文直接打印值
rho_air = 1.208 kg/m³     # 由论文 Pi1 交叉反推，不是正文直接打印值
q_inf = 15.10 Pa
```

这些推导值必须在 manifest 标为 `derived_from_printed_pairs`。

### 3.3 论文没有报告的结构参数

| 参数 | 当前基线/分支 | 纪律 |
|---|---|---|
| 泊松比 | `nu_s=0.49` | 近不可压乳胶假设；必须检查锁死/条件数 |
| 初始预张力 | `0 N/m` | 论文未给；不得据结果添加 |
| 初始松弛/过长量 | `0` | 论文未给 |
| 初始几何缺陷 | 平面 | 非零扰动必须单独声明 |
| 本构 | 当前 Q16 各向同性几何非线性 | 不能把缺失超弹性藏进拟合 `E` |
| 边界 director | 四边位置+director 六自由度固定 | 是高阶边界解释，不是论文直接真值 |
| 结构阻尼 | 论文未报告 | 必须作为假设/敏感性，而不是实验输入 |

`E=2.2 MPa` 是论文主分支。已有 `E=1.4 MPa` 是看过目标响应后形成的校准敏感性，只能标为：

```text
post_hoc_calibrated_sensitivity_E1.4
```

它不能覆盖主结果，不能用于宣布零拟合复现。

当前旧适配器把乳胶损耗因子 `eta=0.1` 作为文献型假设。由于论文未报告阻尼，而且该分支是在旧结果出现慢振荡后引入，统一结果必须明确标记为 `assumed_literature_sensitivity`。至少保留 `eta=0` 的结构模型基线；任何采用 `eta=0.1` 的论文结果都必须在 A10/A17/A23 共用，禁止逐工况调整。

---

## 4. 正式实验 CASE 和观测量

### 4.1 必须区分 A16 平均量和 A17 模态量

旧交接曾把 A16 与论文约 17° 的动态模态混在一起。统一框架必须拆开：

| CASE ID | `U` | `alpha` | 主要验收对象 |
|---|---:|---:|---|
| `ROJ11-A16` | `5 m/s` | `16°` | 主平均量：`max(mean_t z)/c` 与同窗 `mean Cn` |
| `ROJ11-A17-MODE` | `5 m/s` | `17°` | 弦向二峰、展向峰基本消失、`St≈0.85` |
| `ROJ11-A10` | `5 m/s` | `10°` | 弦向三峰、展向三峰、`St≈1.10` |
| `ROJ11-A23` | `5 m/s` | `23°` | 弦向二峰主导、`St≈0.83`、振幅小于 17° |

### 4.2 冻结的图读数近似值

以下均为项目从论文图读取的近似值，不是作者数据表：

| CASE | `zmax/c` | 物理位移 | `Cn` | 动态 oracle |
|---|---:|---:|---:|---|
| A10 | `≈0.032` | `≈2.20 mm` | `≈0.50–0.52` | 3 chordwise peaks、3 spanwise peaks、`St≈1.10` |
| A16 | `≈0.043` | `≈2.96 mm` | `≈0.92–0.95` | 不把 A17 的 `St` 强加给 A16 |
| A17 | `≈0.044–0.045` | `≈3.0–3.1 mm` | `≈0.97` | 2 chordwise peaks、spanwise peaks no longer visible、`St≈0.85` |
| A23 | `≈0.047–0.048` | `≈3.2–3.3 mm` | `≈0.98–1.02` | 2 chordwise peaks、`St≈0.83` |

补充高动态压扩展（前三/四个主 CASE 通过后再做）：

| `U, alpha` | `zmax/c` | `Cn` |
|---|---:|---:|
| `7.5 m/s, 16°` | `≈0.055–0.056` | `≈1.07` |
| `10 m/s, 16°` | `≈0.075–0.076` | `≈1.23–1.26` |

### 4.3 实验统计定义和不确定度

- DIC：1500 fps，持续 1 s，得到 1500 个瞬时变形场；
- 面外位移测量不确定度：约 `0.04%c`；
- 法向力：3000 Hz，持续 20 s；
- `Cn` 测量不确定度：约 `2%`；
- 数字化图读数还会引入额外误差，必须与实验仪器不确定度分开。

论文位移指标是：

```text
z_paper = max_xy(mean_t(z(x,y,t)))
```

不是：

```text
mean_t(max_xy(z))
max_xy,t(z)
最后一个时间点的 zmax
```

`Cn` 必须与位移使用同一统计窗。论文模态是 `zsd(x,y)` 的空间峰数，频谱取最大 `zsd` 附近的测点；不是 Q16 真空固有模态编号。

---

## 5. PIV/流动拓扑 oracle

这些是定性科学门，不应直接变成经验载荷修正：

- `alpha=5°`：基本附着；
- `alpha=10°`：膜翼平均流大体附着，但近表面存在明显速度脉动；刚性平板已大范围分离；
- `alpha=16°`：膜翼已分离，但剪切层仍比刚性翼更贴近表面；
- `alpha=23°`：剪切层远离翼面，翼尖涡影响减弱，弦向二峰响应主导；
- 低/中攻角：LEV/剪切层脱涡和翼尖涡共同激励弦向、展向响应；
- 高攻角：翼尖涡远离膜面，展向峰减弱，弦向二阶响应占主导。

FLUX-V5M 是势流/离散涡类模型，不应假装解析黏性边界层。但如果连释放间歇性、尾迹反馈、翼尖互作、平均载荷和空间模态都不对，不能只用“模型边界”结束诊断。

---

## 6. 当前统一框架状态

### 6.1 已有统一组件

| 层 | 当前生产/统一文件 | 状态 |
|---|---|---|
| 世界状态/事务 | `src/fluxvortex/state/world.py`, `state/transaction.py` | 已有 `WorldOwner` / `GlobalTransaction` |
| Q16 表面运动学 | `src/fluxvortex/kinematics/q16_surface.py` | `Q16SurfaceFrameAdapter` 已有 |
| V5M 状态 | `src/fluxvortex/aero/v5m/state.py` | `V5MWorldState` 已有 |
| V5M 统一适配 | `src/fluxvortex/aero/v5m/stepper.py` | 单 surface 已有；Q16 仍有 raw-state bridge |
| Q16 动力学适配 | `src/fluxvortex/dynamics/q16_adapter.py` | 已有 |
| 强耦合 | `src/fluxvortex/coupling/partitioned.py` | 当前是旧生产 FSI stepper 的薄封装 |
| 观测 | `src/fluxvortex/validation/observers.py` | `max(mean z)`、同窗 Cn、符号穿越、实际窗记录已修正 |
| 平稳性 | `src/fluxvortex/validation/stationarity.py` | block stationarity 已有 |
| 结果状态 | `src/fluxvortex/runtime/result_schema.py` | execution/numerical/physics/accuracy/reproduction 已拆分 |
| Roj 统一 case | `src/fluxvortex/cases/rojratsirikul2011.py` | 仅 A16 两分支，内容不完整 |

### 6.2 当前已验证组件门

2026-08-28 在 RTX 4090 D、当前工作树上执行以下 7 个测试文件：

```text
tests/test_rojratsirikul2011_q16_case.py
tests/test_u0f_corrected_observers.py
tests/test_u1_fsi_transaction_gpu.py
tests/test_u2f_separation_owner_gpu.py
tests/test_u7_v5m_stepper_adapter_gpu.py
tests/test_density_scaling_gpu.py
tests/test_gamma_history_gpu.py
```

结果：`87 passed in 141.95 s`。

这只证明组件/合同测试通过，不证明 Roj 论文精度通过。

### 6.3 当前仍未闭合的问题

1. 尚无统一 `CaseRunner` 正式运行 A16 长时矩阵。
2. 现有 Roj 旧 runner 不使用统一 `ResultStatus`，accuracy fail 仍可能退出 0。
3. 现有旧 runner 的 `rejected_trial_count=0` 是硬编码，不是真实事务计数。
4. 现有统计窗 `t*=4–6` 明显不平稳，不能用末态或不同窗口拼接结论。
5. separated LEV 当前始终集成，但实际新生释放仍是 `3D mask AND 2D trigger` 的两阶段合同；提交 `f451e4f` 只是增加了 `3D-only/2D-only/newly-separated/separated-never-shed` 诊断，没有从科学上消除该不闭合。
6. `particle_max_age_steps=100`、`wake_max_rows=300`、`wake_free_rows=100` 是数值保留策略，不是论文输入；长时结果需要 circulation/impulse ledger 证明删除和截断没有污染载荷。
7. `wake_history_mode=bound_rate` 是当前模型身份；它已修复为单步 `Gamma_n-Gamma_(n-1)`，但不能因为结果更稳定就称其为唯一物理真值。

---

## 7. 旧结果必须隔离，不能作为当前 baseline

旧 A16 结果生成于当前关键修复之前：

- `55916e0`：Q16/刚性载荷密度量纲和 `bound_rate` 单步 Gamma 历史修复；
- `f451e4f`：释放分歧、新分离和 separated-never-shed 诊断；
- 统一 `ResultStatus`、observer、事务计数也未被旧 runner 使用。

更严重的是，旧工件内部已有不闭合：

```text
ROJ11_A16_ETA0.1_T6.json
  payload mean_Cn                    = 1.1217068785
  从同文件 records 按 t*>=4 重算    = 1.2484093368

同一运行配套 z_history，t*>4：
  payload mean_zmax/c                = 0.0246776886
  正确 max_xy(mean_t z)/c            = 0.0411540519
```

`E=1.4 MPa` 旧结果按 `t*>4` 重算得到约：

```text
max(mean z)/c = 0.05353
mean Cn       = 1.4173
```

这与旧分析把某个完整慢周期位移和末态 `Cn≈0.913` 拼成“接近实验”的说法不同。旧窗口本身仍在慢振荡过程中。

因此以下目录只能作为考古材料：

```text
artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/ROJ11_A16_*.json
artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/ROJ11_A16_*.npz
artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/research/ANALYSIS_BRIEF_v4.md
```

接手者必须在新的输出目录从当前源码重新运行，禁止覆盖这些工件，也禁止把它们与新状态拼接。

---

## 8. 可参考但不宜直接继续使用的程序

### 8.1 科学参数和观测 oracle

| 文件 | 可复用内容 | 不能直接照搬之处 |
|---|---|---|
| `platform/forward_flight_benchmarks/rojratsirikul2011_q16.py` | PDF 哈希、几何、材料、A10/A16/A23、边界选择、`Cn`、FFT、峰数 | 阻尼假设和 A16/A17 作用域需重新标注；应迁移到统一 case schema |
| `tests/test_rojratsirikul2011_q16_case.py` | 参数、四边约束、面积/法向、静止运动合同 | 主要是合同测试，不是论文精度测试 |
| 本 HANDOFF §3–§5 | 必要实验数据和定义 | 图读数仍需独立 CSV 固化 |

### 8.2 旧生产实现，只作差异 oracle

| 文件 | 可参考内容 | 禁止事项 |
|---|---|---|
| `platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py` | 正式网格构造、启动斜坡、GPU-only 检查、原子 partial、输出字段 | 不得继续作为独立生产架构；统计/status/counter 已过时 |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py` | 已运行过的 predictor/corrector、formal replay、Aitken | 通过 adapter 复用，不能复制公式到新 coupling |
| `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` | 原生 Q16–V5M 数值内核 | 不能在 CaseRunner 再实现一套 AIC/LEV/TEV/wake |
| `src/fluxvortex/warp_fsi/q16_structural_solver.py` | Q16 Newmark/Newton/PCG | 不能复制成 Roj 专用结构求解器 |
| `platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py` | Q16/native GPU 组件调用方式 | Yamano 是悬臂/不同工况，不是 Roj 论文替代物 |

旧 runner 对应实现可通过 git 历史恢复；无需复制成 `legacy_v2` 等新垃圾文件。

### 8.3 统一框架应直接使用的代码

```text
src/fluxvortex/state/world.py
src/fluxvortex/state/transaction.py
src/fluxvortex/kinematics/q16_surface.py
src/fluxvortex/aero/v5m/stepper.py
src/fluxvortex/aero/v5m/state.py
src/fluxvortex/aero/v5m/separation.py
src/fluxvortex/aero/v5m/retention.py
src/fluxvortex/dynamics/q16_adapter.py
src/fluxvortex/coupling/partitioned.py
src/fluxvortex/validation/observers.py
src/fluxvortex/validation/stationarity.py
src/fluxvortex/runtime/result_schema.py
```

---

## 9. 统一框架实现计划

### P0：冻结 case 和 GT，不跑长时

1. 将完整实验定义迁入 `src/fluxvortex/cases/rojratsirikul2011.py`，至少注册 A10、A16、A17-MODE、A23。
2. 明确字段角色：`paper_printed`、`derived`、`digitized_approx`、`model_assumption`、`numerical_protocol`。
3. 生成并哈希冻结观测 CSV；每行带 Figure、攻角、速度、metric、值/带、证据角色。
4. 保留 `E=2.2 MPa` 主分支；E1.4 标为 post-hoc calibrated sensitivity。
5. 阻尼、预张力、泊松比、边界 director、retention、load-history 各有独立模型身份。

验收：旧 platform adapter 与新 case 在共同字段上完全一致；A16 和 A17 的 oracle 不再混用。

### P1：把旧 Roj runner 改成统一 CaseRunner 的薄 CLI

不得新增第二套长期生产 runner。目标是让现有：

```text
platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py
```

只负责参数解析、调用统一 CaseRunner、保存结果和返回 `ResultStatus.exit_code`。

统一运行内部必须组合：

```text
Q16SurfaceFrameAdapter
V5M3DStepper
Q16DynamicsAdapter
PartitionedStrongFSI
WorldOwner
GlobalTransaction
```

旧数值块只能被 adapter 委托，不得复制。

### P2：同一正式 A16 的迁移 parity

在当前 HEAD、同一 `5×10 Q16 + 15×30 V5M`、同一前 8 个气动步上比较旧调用和统一调用：

- Q16 state/velocity/acceleration；
- rings、collocation、normal、area、panel velocity；
- AIC、rhs、bound Gamma；
- LEV/TEV/wake/particle state digest；
- panel pressure、总法向力、Q16 generalized load；
- coupling residual 和迭代数。

迁移前后若物理模型未变，要求位级或预声明舍入级一致；不能先跑论文值再解释迁移误差。

### P3：FSI 和涡事务正式门

在完整 A16 网格验证：

- 每个 coupling trial 都从同一 committed parent 分叉；
- discarded trial 不污染结构、bound Gamma、LEV、TEV、wake、particle 或 retention ledger；
- formal replay 恰好一次；
- global commit 恰好一次；
- `discarded = proposals - formal_replay` 由真实计数产生，不能硬编码；
- 失败的结构子步/FSI 步使所有 owner 回滚；
- 重复 proposal 确定性一致。

### P4：气动—结构传递门

每个正式步保存并检查：

- 气动面元合力与 Q16 接收合力；
- 关于同一参考点的合矩；
- `surface force · surface velocity` 与 `Q_structural · qdot` 的虚功；
- 压力法向、膜法向和 `Cn` 投影坐标一致；
- separated contribution、surface pressure、profile/viscous correction 不能双计。

### P5：LEV/TEV/free-wake 释放与 retention 门

separated LEV 永远集成，禁止关闭。每步至少记录：

```text
lesp_pre_3d
surface_separated
raw_2d_release
actual_release
release_3d_only_count
release_2d_only_count
newly_separated_count
continuing_separated_count
strips_with_existing_lev_circulation
separated_never_shed_count
Gamma_bound / Gamma_LEV / Gamma_TEV
wake/particle counts
circulation + linear impulse + angular impulse ledger
```

当前 AND 合同若继续保留，必须诚实称为“3D separation eligibility + 2D release trigger”，不能称实际 release 已单 owner。若要实现真正 3D 单 owner，新分离条带的强度/位置闭合必须有独立 source oracle，不能仅强制释放导致发散。

长时运行前，particle cull、wake truncate、far-wake freeze 必须进入 retention ledger。仅统计删除数量不算守恒。

### P6：A16 正式执行门

仍使用正式网格，只缩短时长：

```text
Q16 macros = 5 x 10
V5M panels = 15 x 30
dt* = 0.01
structural substeps = 10
steps = 110                 # 完成 t*=1 启动并进入 10 个恒定来流步
CUDA float64 only
```

该门只证明可以运行、收敛、回滚和保存，不作论文精度结论。

### P7：A16 长时 stationarity 和平均精度

默认完整协议至少运行到：

```text
t* >= 21     # t*=1 startup + 至少 20 convective times
steps >= 2100
```

不得固定使用 `t*=4–6`。统计窗由数据确定，但必须：

1. 完全排除 startup；
2. 覆盖至少多个慢模态周期和大量 `St≈1` 周期；
3. 对 `Cn`、指定点位移、`max(mean z)` 前的空间均值场分别做 block stationarity；
4. 记录实际窗口起止、样本数和完整慢周期数；
5. 加长窗口后均值、RMS、主频稳定。

同一窗口计算：

```text
mean_map = mean_t(z)
zmax_over_c = max_xy(mean_map) / c
mean_Cn = mean_t(Cn)
zsd_map = std_t(z)
```

项目图读数门可冻结为：

- `abs(zmax/c - 0.043) <= 0.005`；
- `mean Cn` 位于图读数带的 ±10% 扩展范围；
- accuracy fail 必须 `exit=2`，不能因为执行完成返回 0。

这里的 ±0.005/±10% 是项目数字化容差，不是论文作者给出的仪器不确定度。

### P8：A17、A10、A23 泛化

A16 冻结全部模型参数后，不允许逐工况调节：

1. A17-MODE：弦向二峰、展向峰基本消失、`St=0.85±0.08`；
2. A10：弦向三峰、展向三峰、`St=1.10±0.10`；
3. A23：弦向二峰、`St=0.83±0.08`，振幅小于 A17；
4. 同时报告各 CASE 的平均 `zmax/c` 和 `Cn`，不能只报模态成功。

峰数必须来自二维 `zsd/c` 场沿论文定义方向的峰数；人工选点、带通后峰数和真空结构模态均不能替代。

### P9：正式高阶分辨率收敛

只允许：

```text
5x10 Q16 / 15x30 V5M
  -> 7x14 Q16 / 21x42 V5M
```

不得使用 Q4、Q9、梁、压力板或缩小膜翼。报告 `mean Cn`、`zmax/c`、`zsdmax/c`、主 `St`、峰数和运行成本变化。

---

## 10. 验收节点

| Gate | 必须满足 | 当前状态 |
|---|---|---|
| H0 Source | PDF/DOI/SHA、观测 CSV、证据角色完整 | PDF 已冻结；独立观测 CSV 待建 |
| H1 Case | A10/A16/A17/A23 统一注册；A16/A17 不混用 | 待完成 |
| H2 GPU purity | CUDA float64；无 CPU 数值 fallback；无 Ptera/Q4/Q9 | 组件已有门，本 CASE 待 fresh 证明 |
| H3 Migration parity | 当前 HEAD 下旧调用与统一调用前 8 步一致 | 待完成 |
| H4 Transaction | trial/formal/commit/rollback 全 owner 闭合 | 组件测试通过，正式 A16 待证 |
| H5 Transfer | 合力、合矩、虚功闭合 | 组件测试通过，正式 A16 待证 |
| H6 Vortex/retention | LEV/TEV/free-wake 实时推进；cull/truncate ledger 闭合 | 诊断已有，守恒长时门未闭合 |
| H7 A16 execution | 正式网格 110 步、退出 0、finite、无污染 | 旧结果不能代替 fresh 结果 |
| H8 A16 stationarity | 正确统计窗 block-stationary | 未通过 |
| H9 A16 accuracy | 同窗 `zmax/c` 与 `Cn` 通过项目门 | 未通过/无当前结果 |
| H10 A17 mode | 二峰、spanwise 消失、`St≈0.85` | 未运行 |
| H11 A10 generalization | 3×3 峰、`St≈1.10` | 未运行 |
| H12 A23 generalization | 弦向二峰、`St≈0.83` | 未运行 |
| H13 Resolution | Q16/V5M 高阶收敛 | 未运行 |
| H14 Provenance | 当前源码哈希、dirty files、配置 hash、完整结果可重算 | 待完成 |

只有 H0–H14 中与所宣称范围相关的门全部通过，才能写“Rojratsirikul 2011 已复现”。执行成功、组件测试通过或 A16 某个末态接近实验都不等于论文复现。

---

## 11. 接手后的第一组命令

### 11.1 环境

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca

export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0
export FLUXV_GPU_ONLY=1
export FLUXV_DEVICE=cuda:0
export FLUXV_DTYPE=float64
export FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

GPU 不可用时必须 fail-fast，禁止 CPU fallback。

### 11.2 当前组件门

```bash
python -m pytest -q \
  tests/test_rojratsirikul2011_q16_case.py \
  tests/test_u0f_corrected_observers.py \
  tests/test_u1_fsi_transaction_gpu.py \
  tests/test_u2f_separation_owner_gpu.py \
  tests/test_u7_v5m_stepper_adapter_gpu.py \
  tests/test_density_scaling_gpu.py \
  tests/test_gamma_history_gpu.py
```

### 11.3 统一 runner 完成后的目标命令

不要覆盖 20260824 旧目录。建议新输出目录：

```text
artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current/
```

执行门：

```bash
PYTHONUNBUFFERED=1 python \
  platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 \
  --execution-gate-only \
  --max-aero-steps 110 \
  --output artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current/ROJ11_A16_EXECUTION.json
```

正式 A16：

```bash
PYTHONUNBUFFERED=1 python \
  platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 \
  --output artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current/ROJ11_A16_FULL.json
```

这些命令只有在该 CLI 已改成统一 CaseRunner 薄入口后才是正式命令；当前旧实现直接运行只能叫 legacy diagnostic。

---

## 12. 正式输出 schema

至少保存：

```text
paper/doi/pdf_sha/GT_sha
git_head/dirty_files/config_digest
device/device_name/dtype/cpu_fallback_count/legacy_module_count
case_id/U/Re/alpha/c/b/S
E/nu_s/rho_m/t/prestress/damping_model/damping_parameters
q16_grid/aero_grid/dt_star/structural_substeps
startup_window/actual_statistics_window/block_stationarity
ResultStatus 五维状态和 CLI exit code

proposal_count/discarded_trial_count/formal_replay_count/commit_count
parent/trial/formal/committed digests
coupling residuals/Newton iterations/PCG iterations

force_transfer_error/moment_transfer_error/virtual_work_error
surface pressure force/separated contribution/profile correction owner

LESP/release decomposition/Gamma ledger/impulse ledger
wake rows/particles/cull/truncate/freeze events

mean_map/zsd_map/zmax_over_c/zsdmax_over_c/mean_Cn
dominant_St/chordwise_peak_count/spanwise_peak_count
all accuracy gates and failure reasons
```

长时运行每 10 个气动步原子写 `.partial.json` 和必要状态摘要。若没有完整 checkpoint/restart 状态，电脑重启后不得把两个进程的结构/尾迹轨迹拼接成一条正式运行。

保存后的最终文件必须进行内部重算检查：summary 中每个均值、频率、峰数都要能从原始 records/NPZ 重算；任何不一致使 H14 失败。

---

## 13. 失败时的诊断顺序

### 13.1 非零退出或不收敛

1. 找到同一正式 A16 的首个失败 step/trial/substep；
2. 检查 GPU ownership 和非有限量；
3. 检查结构 Newton/PCG；
4. 检查 FSI residual、Aitken 和 formal replay；
5. 检查 parent/trial/commit digest；
6. 检查 particle/wake capacity 和 retention ledger；
7. 只在同一正式网格复现该失败，禁止另建 toy。

### 13.2 平均力/位移失败

依次检查：

1. `q_inf S`、法向、`Cn` 符号和参考面积；
2. 四边约束和 rotated reference director；
3. Q16 质量、体积、`Et` 和近不可压锁死；
4. 气动到结构的合力/合矩/虚功；
5. 同窗统计和 stationarity；
6. 密度量纲、Gamma 历史、load-history identity；
7. LEV/TEV/release/retention；
8. 最后才讨论未报告的阻尼、本构、预张力和框架厚度。

禁止先改 `E`、阻尼或 `Lcrit` 使单个数落带。

### 13.3 模态/频率失败

1. 确认 CASE 攻角：A17 的 `St=0.85` 不能用 A16 验收；
2. FFT 使用平稳统计窗、最大 `zsd` 测点和原始位移；
3. 检查快峰是否只是慢呼吸的整数谐波；
4. 同时检查二维 `zsd` 场的弦向和展向峰数；
5. 检查 LEV 释放是否饱和为每步全展向释放；
6. 检查 far-wake freeze、particle cull 是否删除了相位反馈；
7. 不得强制施加实验频率。

---

## 14. 禁止事项

- 禁止 Q4/Q9、梁、压力板、悬臂或缩小翼面替代 Q16 正式 CASE；
- 禁止 Ptera 作为生产气动求解器；
- 禁止关闭 separated LEV；零释放只能由物理阈值自然产生；
- 禁止把 A17 模态 GT 用在 A16；
- 禁止用末态 Cn 与另一时间窗的位移拼接；
- 禁止用 E1.4 覆盖 E2.2 主结果；
- 禁止逐工况调阻尼、E、预张力、Lcrit、retention 或载荷比例；
- 禁止在新框架旁复制第二套 Roj solver；
- 禁止覆盖用户当前对 `UNIFIED_Q16_V5M_FSI_REFACTOR_PLAN_20260826.md` 的未提交修改；
- 禁止清理或重置当前脏工作树。

---

## 15. Definition of Done

接手者只有在以下事实同时成立时才能结束本任务：

1. Roj A10/A16/A17/A23 都是统一 case schema 的正式成员；
2. 旧 runner 已成为统一 CaseRunner 的薄 CLI，而非第二套底盘；
3. Q16、V5M、FSI、LEV/TEV/free-wake 全部在同一 global transaction 中；
4. rejected trial 对所有 owner 零污染，formal replay/commit 各一次；
5. A16 达到长时 stationarity，并用 `max(mean z)` 与同窗 mean Cn 评分；
6. A17/A10/A23 使用同一冻结参数重现论文模态和主频趋势；
7. E2.2 主结果和所有未报告参数假设被诚实区分；
8. 只使用 Q16/V5M 正式高阶网格完成分辨率说明；
9. 所有结果从当前源码 fresh 生成，旧 20260824 工件未混入；
10. execution、numerical、physics、accuracy、reproduction 五类状态分别报告；
11. accuracy/physics/numerical 失败时 CLI 非零退出；
12. 原始 records、场数据、摘要、图和 manifest 可以互相重算闭合。

最重要的执行原则：**不要先完善一套与论文无关的中间框架；直接让正式 `ROJ11-A16` 驱动统一 CaseRunner 的下一纵向切片，用它暴露第一个真实失败，修复后立即进入长时 A16，再用 A17/A10/A23 检查泛化。**
