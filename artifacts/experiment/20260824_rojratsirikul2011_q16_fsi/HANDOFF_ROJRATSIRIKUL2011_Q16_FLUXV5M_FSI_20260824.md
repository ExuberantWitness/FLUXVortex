# Rojratsirikul et al. (2011) Q16–FLUX‑V5M FSI 复现交接

更新时间：2026-08-24（Asia/Shanghai）  
接手仓库：`/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca`  
当前分支：`run/q16-lev-tev-pc-fsi-20260821`  
记录时 HEAD：`d36cdeebc7ae082f1d31a84795d6eb0b0d4ab56a`  
本任务工件目录：`artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/`

## 0. 接手者先读：本 CASE 是什么、又不是什么

目标论文：

> P. Rojratsirikul, M. S. Genc, Z. Wang, I. Gursul, “Flow-induced vibrations of low aspect ratio rectangular membrane wings,” *Journal of Fluids and Structures*, 27(8), 1296–1309, 2011. DOI: `10.1016/j.jfluidstructs.2011.06.007`.

- 作者公开 PDF：<https://purehost.bath.ac.uk/ws/files/227159/Gursul_JFS_2011.pdf>
- 期刊页面：<https://www.sciencedirect.com/science/article/pii/S0889974611001010>
- 独立高保真数值复现：R. E. Gordnier, P. J. Attar, 2014，DOI `10.1016/j.jfluidstructs.2013.10.004`，<https://www.sciencedirect.com/science/article/pii/S0889974613002235>

这是一个**有来流、三维有限展长、四周框定乳胶膜的固定翼流致振动 CASE**。它没有外加升沉、俯仰、扑动或翼根转动。正确的输入是恒定来流和恒定攻角，膜翼的平均鼓包、瞬态振动、振型和频率全部是双向 FSI 输出。

禁止给本 CASE 添加 `h(t)`、`pitch(t)` 或 flapping law。若程序需要外部运动对象，必须传入严格静止的边界运动，而不是零幅值扑动的历史兼容分支。

## 1. 本轮目标与完成定义

在当前唯一生产架构中实现并验证以下正式数据通路：

`四周固定 Q16 高阶膜/壳 → 静止几何 + 恒定来流 → 原生 FLUX‑V5M → separated LEV + joint TEV + free wake → predictor/corrector 强耦合 → 流致 Q16 变形与振动`

复现完成至少要求：

1. 正式 `ROJ11-A10`、`ROJ11-A16`、`ROJ11-A23` 三个 `Re=24,300` 工况全部运行；不得用低阶或缩小物理问题替代。
2. `ROJ11-A16` 的平均最大位移和平均法向力进入论文图读数容差带。
3. `ROJ11-A10` 重现弦向三峰与展向三峰响应；`ROJ11-A23` 重现弦向二峰主导及相应频率带。
4. separated LEV、joint TEV、free wake 在每次 trial 内真实推进；被拒绝 trial 对 committed 状态零污染，formal replay 只提交一次。
5. Q16 与 V5M 之间的合力、合矩和虚功传递门全部通过。
6. 全部科学数值计算使用 CUDA float64；无 CPU 数值 fallback、无 Ptera、无 Q4/Q9 中间拓扑。
7. 最终报告必须区分“执行成功”“机理门通过”“论文精度通过”，不能把前两者冒充论文复现完成。

## 2. 不可违反的项目约束

- 结构只允许 Q16；禁止 Q4、Q9、压力板 toy、梁替代和低阶临时网格。
- 气动只允许原生 FLUX‑V5M；禁止 Ptera 生产路径。
- separated LEV 是强制组成部分，不能关闭。是否发生物质释放只由 `abs(LESP)>Lcrit` 决定。
- 薄膜/薄平板基线 `Lcrit=0.11`，在获得独立材料/前缘证据前不得调参。
- LEV、joint TEV、bound circulation、自由尾迹和粒子场必须属于同一可回滚气动事务。
- predictor 必须推进真实 trial 尾迹；禁止冻结尾迹、事后补推进或只在 corrector 释放涡。
- 不能经验缩放压力、法向力、结构刚度、阻尼、位移或时间。
- 不得用论文结果反调 `E`、预张力、泊松比或阻尼后再称为零拟合复现。
- 只允许在同一完整论文 CASE 上由短时执行门推进到长时统计门；不得另建中间 toy。

## 3. 坐标、几何和边界条件

科学坐标统一为：

- `+x`：弦向，前缘 `x/c=0` 指向后缘 `x/c=1`；
- `+y`：展向，一侧翼尖 `y/c=-1` 指向另一侧翼尖 `y/c=+1`；
- `+z`：垂直翼弦面的厚度方向/膜位移方向；
- 来流从前缘流向后缘，攻角为固定几何攻角 `alpha0`。

### 3.1 论文直接报告的几何/材料真值

| 参数 | 冻结值 | 证据等级 |
|---|---:|---|
| 平面形状 | 矩形 | 正文 |
| 弦长 `c` | `68.8 mm = 0.0688 m` | 正文/图2 |
| 翼展 `b` | `137.5 mm = 0.1375 m` | 图2；正文以 `AR=2` 表述 |
| 展弦比 | `AR=2` | 正文 |
| 参考面积 `S=bc` | `0.00946 m²` | 由论文尺寸推导 |
| 膜厚 `t` | `0.2 mm = 2.0e-4 m` | 正文 |
| 厚弦比 `t/c` | `0.002907 = 0.2907%` | 推导 |
| 膜材料 | 黑色乳胶橡胶 | 正文 |
| 杨氏模量 `E` | `2.2 MPa` | 正文 |
| 膜密度 `rho_m` | `1 g/cm³ = 1000 kg/m³` | 正文 |
| 面密度 `rho_m*t` | `0.2 kg/m²` | 推导 |
| 面内刚度基量 `E*t` | `440 N/m` | 推导 |
| 刚性框架 | 不锈钢，截面约 `5 mm × 2 mm`，尖边朝内 | 正文/图2 |
| 刚性对照翼 | `1 mm` 不锈钢平板，`t/c=1.45%` | 正文/表1 |

图2还给出约 `6.5 mm` 支撑直径、`50 mm` 近翼连接段和 `60 mm` 支撑偏置。风洞为直径 `760 mm` 的圆形开口射流试验段，端板间距 `450 mm`。

### 3.2 正式结构边界解释

论文说明乳胶膜附着在四周刚性框架上，但没有报告粘接层、边界转角或高阶 director 条件。首个可审计基线应采用：

1. Q16 结构域为论文给出的完整 `c × b` 矩形；不要自行从弦长和翼展中扣除两侧 `5 mm` 框宽。
2. 选择四条周边上的全部 Q16 节点。
3. 使用现有 `make_clamped_q16_nodes(...)` 固定这些节点的六个 Q16 自由度，即位置与半厚度 director 均保持参考值。
4. 将“全六自由度周边固定”明确记录为**边界解释假设**，不是论文直接给出的高阶自由度真值。
5. 若后续独立 oracle 明确表明 director 过约束，才允许实现“仅平移固定”作为单独分支；不得在看到位移偏小后无证据地放松边界。

刚性框架可以先作为不可变边界而不显式加入 Q16 结构。框架的气动厚度效应属于后续独立分支；论文主基线不得为了加快运行而改变外轮廓 `c × b`。

## 4. 流动参数与外部运动合同

### 4.1 论文工况

| `U_inf` | `Re=U_inf*c/nu` | 论文 `Pi1=(Et/qc)^(1/3)` | 由 `Pi1` 反推 `q` |
|---:|---:|---:|---:|
| `5.0 m/s` | `24,300` | `7.51` | `≈15.10 Pa` |
| `7.5 m/s` | `36,500` | `5.73` | `≈34.00 Pa` |
| `10.0 m/s` | `48,700` | `4.73` | `≈60.43 Pa` |

统一使用：

- `nu = 1.414e-5 m²/s`，由三个论文 `Re` 反推；
- `rho_air = 1.208 kg/m³`，由三个论文 `Pi1` 交叉反推；
- 攻角从 `0°` 到失速后区域，论文曲线最高约 `25°–30°`；
- 法向力系数 `Cn = Fn/(0.5*rho_air*U_inf²*S)`。

### 4.2 外部运动必须是静止的

```text
U_inf(t)      = U0
alpha(t)      = alpha0
frame_pose(t) = frame_pose(0)
frame_vel(t)  = 0
```

结构响应由 FSI 得到：

```text
z(x,y,t) = mean_z(x,y) + fluctuation_z(x,y,t)
```

不能给 `fluctuation_z` 预设正弦函数。论文只提供观测频带和模态，频率是结果而不是输入。

### 4.3 启动与统计时钟

论文没有报告风洞启动过程。数值基线允许使用只影响初始瞬态的光滑来流启动，但必须冻结并报告：

- 推荐 `0 <= t* <= 1` 使用半余弦从 `0` 增长到 `U0`；`t*=t U_inf/c`。
- 启动段不得进入平均值、RMS 或 FFT 窗口。
- 推荐气动步 `dt*=0.01` 或更小；该值在 `U=5 m/s` 下约为 `1.376e-4 s`，可给论文约 `60–100 Hz` 响应每周期至少约 `70–120` 个气动采样点。
- 正式统计至少在启动后覆盖 `20` 个对流时间；若均值或主频未稳定，延长同一 CASE，不得换模型。
- 结构子步数由 Q16 非线性收敛和最高频率决定，不能为节省时间降低到不满足时间收敛的程度。

这些时钟是数值复现协议，不是论文原始试验输入，必须在结果中标记。

## 5. 首批正式 CASE 与论文 oracle

### 5.1 必做三个 CASE

| CASE ID | `U_inf` | `Re` | `alpha` | 目的 |
|---|---:|---:|---:|---|
| `ROJ11-A10` | `5 m/s` | `24,300` | `10°` | 低/中攻角三维 LEV–翼尖涡–膜振动耦合 |
| `ROJ11-A16` | `5 m/s` | `24,300` | `16°` | 主精度 CASE；明显分离、较大平均变形和独立数值复现 |
| `ROJ11-A23` | `5 m/s` | `24,300` | `23°` | 深分离、弦向二阶主模态、LEV/TEV/free-wake 长时事务 |

Gordnier–Attar 的独立高保真计算采用 `Re=24,300` 下 `alpha=10°,16°,23°`，所以本阶段不要先做论文未被独立复现的参数扩展。

### 5.2 正文明确给出的动态 oracle

论文使用：

```text
St_m = f_m*c/U_inf
```

并给出刚性翼尾迹近似关系：

```text
f_s*c*sin(alpha)/U_inf ≈ 0.17
```

该尾迹关系只能用于结果校验，禁止作为强制涡脱落频率输入。

| 工况 | 论文响应 | `St_m` | 对应频率 |
|---|---|---:|---:|
| `U=5, alpha=10°` | 弦向三峰 + 展向三峰 | `≈1.10` | `≈79.9 Hz` |
| `U=5, alpha=17°` | 展向峰基本消失，弦向二峰 | `≈0.85` | `≈61.8 Hz` |
| `U=5, alpha=23°` | 弦向二峰，振幅较 `17°` 小 | `≈0.83` | `≈60.3 Hz` |
| `U=7.5, alpha=4°` | 小幅一阶 | 论文未给单值 | — |
| `U=7.5, alpha=12°` | 弦向/展向混合 | `≈0.9` | `≈98 Hz` |
| `U=7.5, alpha=15°` | 弦向二峰 | `≈0.7–0.8` | `≈76–87 Hz` |
| `U=10, alpha=9°` | 明显弦向二峰 | `≈0.6–0.7` | `≈87–102 Hz` |

“弦向/展向模态阶数”是论文按位移标准差图中的峰数定义的，不是 Q16 真空固有模态编号。

### 5.3 从论文图读取的平均量

下表均为图读数，必须在字段名中包含 `digitized_approx`，不得伪装为作者原始表格。

| 工况 | `zmax/c` 约值 | 物理位移 | `Cn` 约值 |
|---|---:|---:|---:|
| `U=5, alpha=10°` | `0.032` | `2.20 mm` | `0.50–0.52` |
| `U=5, alpha=16°` | `0.043` | `2.96 mm` | `0.92–0.95` |
| `U=5, alpha=17°` | `0.044–0.045` | `3.0–3.1 mm` | `≈0.97` |
| `U=5, alpha=20°` | `≈0.049` | `≈3.37 mm` | 峰值 `≈1.08` |
| `U=5, alpha=23°` | `0.047–0.048` | `3.2–3.3 mm` | `≈1.00` |
| `U=7.5, alpha=16°` | `0.055–0.056` | `≈3.8 mm` | `≈1.07` |
| `U=7.5, alpha≈19°` | `≈0.061` | `≈4.20 mm` | 峰值 `≈1.17` |
| `U=10, alpha=16°` | `0.075–0.076` | `≈5.2 mm` | `≈1.23–1.26` |
| `U=10, alpha≈16–17°` | `≈0.076` | `≈5.2 mm` | 峰值 `≈1.26` |

平均鼓包在展向上大体对称，最大位移位置略位于半弦前方。最高来流速度下膜翼 `Cn,max≈1.26`，约为刚性对照翼最大值的 `1.5` 倍。

### 5.4 PIV/流动拓扑 oracle

- `alpha=5°`：膜翼基本附着。
- `alpha=10°`：刚性平板已经大范围分离；膜翼分离剪切层更贴近表面。
- `alpha=16°`：膜翼明显分离，但剪切层比刚性翼更靠近翼面；增大 `U_inf` 后更贴近膜面。
- `alpha=23°`：剪切层远离表面；`U=10 m/s` 膜翼上方仍有闭合回流区。
- 低/中攻角下 LEV 与翼尖涡共同激励弦向和展向响应。
- 高攻角下翼尖涡远离膜面，展向模态减弱，弦向二峰主导。

FLUX‑V5M 不必复现黏性边界层细节才能通过执行门，但论文精度报告必须诚实说明其涡模型能否重现上述分离剪切层/回流拓扑。

## 6. 论文未报告参数与冻结假设台账

| 参数 | 论文状态 | 首轮基线 | 纪律 |
|---|---|---|---|
| 泊松比 | 未报告 | `nu_s=0.49`，近不可压乳胶的声明假设 | 不得据结果调节；必须检查 Q16 锁死/条件数 |
| 初始预张力/预应变 | 未报告 | `0` | 不得看到欠变形后直接添加预张力 |
| 初始松弛/过长量 | 未报告 | `0` | 任何非零值必须有独立来源 |
| 结构阻尼 | 未报告 | `0` | 先让流体和材料数值耗散自然决定响应 |
| 初始几何缺陷 | 未报告 | 完全平面 | 若出现对称分岔问题，使用机器量级可复现扰动且单独报告 |
| 乳胶本构曲线 | 只报告 `E` | 当前 Q16 各向同性几何非线性基线 | 不得把超弹性缺失藏进拟合 `E` |
| 边界 director | 未报告 | 四周六自由度固定 | 单列为边界解释假设 |
| 框架气动厚度 | 只给图2截面 | 首轮以固定边界/外轮廓表示 | 显式框架只能作为后续正式分支 |
| 入口湍流度/转捩 | 未报告 | 不引入经验湍流调参 | 在限制中声明 |

重要风险：后续数值文献报告过 `E=2.2 MPa` 可能低估实验变形。这不能成为直接把 `E` 改为 `1.5 MPa` 的理由。首轮必须保存 `E=2.2 MPa` 的真实结果；只有在结构、载荷传递、时间步和边界 oracle 全部通过后，才可将不同 `E` 作为明确标注的材料不确定性分支，不能替代主结果。

## 7. 当前生产代码地图与应新增文件

### 7.1 必须复用的现有生产组件

| 对象 | 文件 |
|---|---|
| 原生 Q16–V5M、LEV/TEV/free-wake 事务 | `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py` |
| Q16↔V5M 强 predictor/corrector FSI | `src/fluxvortex/warp_fsi/q16_flux_v5m_native_fsi.py` |
| Q16 Newmark/Newton/PCG | `src/fluxvortex/warp_fsi/q16_structural_solver.py` |
| Q16 网格和宏材料 | `src/fluxvortex/q16_ancf_mesh.py` |
| Q16 MITC16/ANS/EAS | `src/fluxvortex/q16_ans_eas_continuum.py` |
| Q16 周边约束 | `src/fluxvortex/q16_boundary_constraints.py` |
| Q16 力/矩/虚功传递 | `src/fluxvortex/q16_work_conjugate_transfer.py` |
| mandatory 气动模式 | `src/fluxvortex/warp_fsi/q16_mandatory_aero_mode.py` |
| 生产 GPU 气动能力矩阵 | `platform/warp_vpm/flux_v5m_gpu.py` |
| 当前原生 CASE 参考入口 | `platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py` |

### 7.2 本任务应新增，而不是污染 Yamano 适配器

建议新增：

```text
platform/forward_flight_benchmarks/rojratsirikul2011_q16.py
platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py
tests/test_rojratsirikul2011_q16_case.py
artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/
```

参数适配器至少提供：

- 论文真值及证据等级；
- `ROJ11-A10/A16/A23` case registry；
- Q16 四周节点集合；
- 静止边界运动；
- `Cn`、`zmax/c`、`zsd/c`、FFT、弦向/展向峰数计算；
- 运行输出 schema 和验收门；
- 缺失参数假设与唯一配置摘要/hash。

不要修改 Yamano 参数文件来容纳本 CASE；两篇论文的边界、时钟和物理对象不同。

## 8. 正式网格、运行输出与性能要求

### 8.1 首轮正式网格

首轮使用近等尺寸宏单元/面板：

- Q16：弦向 `5` × 展向 `10` 个 Q16 宏单元；
- FLUX‑V5M：弦向 `15` × 展向 `30` 个气动面板；
- 每个方向每个 Q16 宏单元对应约 `3` 个气动面板；
- 不允许先跑 Q4/Q9 或一两个低阶宏单元；
- 短时运行只能缩短同一正式 CASE 的时间窗，不能降低结构阶次或换物理模型。

若该网格因显存/时间失败，先修真正的 GPU 批量、wake 存储和事务问题；不能自动回退 CPU 或退成 toy。

### 8.2 每个输出必须记录

```text
case_id, git_head, dirty_state_digest
device_name, device, dtype
cpu_fallback_count, runtime_legacy_module_count, ptera_loaded
q16_macro_chord, q16_macro_span, aero_nchord, aero_nspan
rho_air, nu_air, U_inf, Re, alpha, c, b, S
E, nu_s, rho_m, thickness, prestress, damping
dt_star, structural_substeps, startup_window, statistics_window
separated_lev_mandatory, Lcrit, joint_tev, free_wake
trial_count, rejected_trial_count, commit_count, parent_digest
force_transfer_error, moment_transfer_error, virtual_work_error
mean_Cn, mean_zmax_over_c, zsd_map, dominant_St
chordwise_peak_count, spanwise_peak_count
wake_ring_count, lev_release_count, max_abs_lesp_pre
```

长时运行必须定期写原子 `.partial.json`，以便电脑死机后定位最后已完成的正式时间窗。若尚无跨进程状态恢复，不得把不同进程的尾迹/结构状态拼接成一条轨迹。

## 9. 开发和执行顺序

### P0：冻结来源与参数合同

1. 下载作者 PDF 到本工件目录的 `references/`，记录 SHA256。
2. 建立 `rojratsirikul2011_q16.py`，将正文值、图读数和假设分成不同字段。
3. 新增单元测试锁定 `AR`、`t/c`、`Re`、`Pi1`、`rho_air`、`nu_air`、三个 CASE ID 和静止运动合同。
4. 确认四周节点集合包含四条边且没有内部节点。

### P1：接入同一正式 CASE 的短时执行门

1. 从现有原生 Yamano 入口复用事务和 GPU 所有权，不复用其悬臂边界/脉冲时钟。
2. 构造 `5×10` Q16、`15×30` V5M 的 `ROJ11-A16`。
3. 在完整网格上推进最少 `50` 个气动步，验证结构、气动、LEV/TEV/wake 和 FSI 事务均能执行。
4. 该节点只叫“执行门”，不能报告论文精度。

### P2：A16 主精度 CASE

1. 运行到统计量稳定。
2. 比较 `mean_zmax/c≈0.043`、`mean_Cn≈0.92–0.95`。
3. 输出平均位移等值图、`zsd/c` 图、法向力时间序列、LESP/释放/尾迹历史。
4. 若失败，先检查首个独立合同：边界、载荷传递、法向力符号、Q16 体积/质量、时间收敛；禁止先调材料。

### P3：A10 与 A23 模态 CASE

1. `A10`：检查弦向三峰、展向三峰、`St≈1.10`。
2. `A23`：检查弦向二峰主导、`St≈0.83`，并检查深分离 LEV/TEV/free-wake 事务。
3. 使用与 A16 相同的网格、时钟、材料和缺失参数假设。

### P4：高动态压扩展

三个独立复现工况通过后，再运行 `U=10 m/s, alpha=16°`：

- `Re=48,700`；
- `zmax/c≈0.075–0.076`；
- `Cn≈1.23–1.26`；
- 检查更强的二阶/混合振动和近表面剪切层。

### P5：只用 Q16/V5M 正式分辨率做收敛

- 结构候选：`5×10` → `7×14` Q16；
- 气动候选：`15×30` → `21×42` V5M；
- 不允许用 Q4/Q9 作收敛中间点；
- 对 `mean_Cn`、`mean_zmax/c`、主 `St` 和模态峰数报告变化。

## 10. 验收节点

| 节点 | 必须满足的证据 | 初始状态 |
|---|---|---|
| H0 来源冻结 | PDF URL/SHA256、DOI、图读数来源和假设台账完整 | 待完成 |
| H1 参数合同 | 几何、材料、`Re/Pi1`、静止运动测试通过 | 待完成 |
| H2 依赖纯度 | 原生 V5M；无 Ptera/Q4/Q9/CPU fallback | 生产架构已有门，本 CASE 待证 |
| H3 GPU 所有权 | CUDA float64；GPU 不可用时非零退出 | 生产架构已有门，本 CASE 待证 |
| H4 Q16 周边约束 | 四周边界固定；内部节点全自由；约束反力可提取 | 待完成 |
| H5 传递合同 | 合力、合矩、虚功闭合至现有生产阈值 | 组件已有门，本 CASE 待证 |
| H6 LEV/TEV/wake 事务 | trial 无污染、formal commit 一次、释放条件不变 | 组件已有门，本 CASE 待证 |
| H7 A16 执行门 | 正式网格至少 50 步，退出 0、全状态 finite | 待完成 |
| H8 A16 平均精度 | `zmax/c` 距 `0.043` 不超过 `0.005`；`Cn` 相对图读数误差不超过 `10%` | 待完成 |
| H9 A10 模态 | 弦向/展向均为三峰；主 `St` 距 `1.10` 不超过 `0.10` | 待完成 |
| H10 A23 模态 | 弦向二峰主导；主 `St` 距 `0.83` 不超过 `0.08` | 待完成 |
| H11 长时统计 | 均值、RMS、主频对加长窗口稳定；尾迹/事务无漂移 | 待完成 |
| H12 分辨率 | 只用 Q16/V5M 高阶网格完成收敛说明 | 待完成 |
| H13 报告一致性 | 明确哪些是实验真值、图读数、推导值和假设 | 待完成 |

H8–H10 的容差是本项目针对图像数字化误差制定的复现门，不是论文作者声明的误差界。论文报告的测量不确定度为：`Cn` 约 `2%`、速度约 `2% U_inf`、面外位移约 `0.04%c`；图读数会额外引入误差。

## 11. 失败处理规则

### 程序非零退出

1. 先读同名 `.partial.json` 的首个失败时间步、trial 和结构子步。
2. 判断失败属于 GPU 所有权、Q16 非线性收敛、FSI 耦合、事务 digest、尾迹容量还是 LEV/TEV 门。
3. 在同一完整 `ROJ11-A16` 网格和状态上复现该失败；不得另建小模型。
4. 允许优化求解器、预条件和 GPU 批量；不得放宽科学残差门或改物理参数隐藏失败。

### 程序完成但平均位移/力失败

按以下次序查首个独立 oracle：

1. `Re`、`q`、参考面积、力轴和 `Cn` 符号；
2. 四周 Q16 节点与 director 约束；
3. Q16 质量、体积、`Et` 和厚度；
4. 气动→Q16 合力、合矩和虚功；
5. 时间步/统计窗收敛；
6. LEV/TEV/free-wake 状态与提交次数；
7. 最后才讨论论文未报告的材料、本构、预张力和框架气动不确定性。

### 模态/频率失败

- 先确认 FFT 使用启动后位移，且采样点是论文定义的 `zsd,max` 附近。
- 同时输出二维 `zsd/c` 场；不能只靠单点 FFT 判模态。
- 峰数沿弦向和展向分别统计；不能把 Q16 真空模态编号当成论文峰数。
- 检查尾迹截断、重采样和窗口是否改变主频，再检查物理模型。
- 禁止强制施加论文频率或用带通滤波制造目标峰。

### LEV 释放数为零

检查 `separated_lev_mandatory=true`、`Lcrit=0.11`、`joint_tev=true`、`free_wake=true` 和 `max_abs_lesp_pre`。若物理上未越阈值，零释放是合法结果；禁止降低 `Lcrit` 制造 LEV。反之若越阈值却不释放，则是必须修复的生产 bug。

## 12. 测试和目标命令

接手后先运行现有生产门：

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca

export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0
export FLUXV_GPU_ONLY=1
export FLUXV_DEVICE=cuda:0
export FLUXV_DTYPE=float64
export FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pytest -q \
  tests/test_q16_structural_step_gpu.py \
  tests/test_q16_flux_v5m_native_gpu.py \
  tests/test_q16_mandatory_aero_mode.py \
  tests/test_q16_work_conjugate_transfer.py \
  tests/test_q16_aero_load_packet_gpu.py \
  tests/test_q16_lev_impulse_transfer_gpu.py
```

新入口完成后的目标命令应为：

```bash
PYTHONUNBUFFERED=1 python \
  platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 \
  --output artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/ROJ11_A16.json
```

短时执行门必须是同一命令增加显式参数，例如 `--max-aero-steps 50 --execution-gate-only`；不得指向另一套 solver。

## 13. 工作树和提交纪律

当前工作树包含大量用户和此前 agent 的修改/未跟踪文件。接手前执行：

```bash
git status --short --branch
git diff --check
```

- 不得执行 `git reset --hard`、`git checkout -- .`、批量删除或清理工作树。
- 不得覆盖现有 Yamano、三篇纯气动复现和 GPU 优化证据。
- 只暂存本 CASE 明确核验过的新增/修改文件。
- 未经用户明确要求不要 push。
- `/tmp/fluxv-v5-nextgen` 已在电脑重启后消失，不要把新结果写回临时路径。

## 14. 接手者最终汇报格式

最终汇报必须包含：

1. 三个主 CASE 的输入摘要/hash、GPU、dtype、网格和统计窗。
2. `Cn`、`zmax/c`、主 `St`、弦向/展向峰数对照表。
3. 平均位移、位移标准差、法向力和频谱图。
4. LEV/TEV/free-wake 事务证据：trial 无污染、formal commit 一次、释放条件原样执行。
5. Q16 直接载荷合力、合矩、虚功闭合证据。
6. H0–H13 的逐项状态；任何未通过项必须保留为 FAIL。
7. 首个独立 oracle 偏离、根因、修改位置及修改前后对比。
8. 论文未报告参数的假设和敏感性分支，禁止写成实验真值。

当前最重要的一句话：**不要再搭中间模型；直接在 `ROJ11-A16` 的正式 Q16 5×10、FLUX‑V5M 15×30、mandatory separated-LEV/joint-TEV/free-wake 强耦合路径上快速暴露首个失败，再沿独立 oracle 修根因。**
