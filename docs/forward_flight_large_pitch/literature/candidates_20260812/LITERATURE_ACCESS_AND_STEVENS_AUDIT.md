# LEV 候选文献获取状态与 Stevens–Babinsky 2017 复现审计

审计日期：2026-08-12

## 结论先行

1. **Stevens & Babinsky (2017) 是真实实验，而且可以建立一个有来流、45° 大幅瞬态俯仰的升力/LEV 校核工况。**几何、无量纲运动、测力流程和主要观测曲线基本闭合。
2. 它**不能作为升阻力双通道基准**：实验虽然使用两分量天平，但论文明确只公开升力；没有公开阻力曲线。
3. 它也不是完全独立的作者模型盲验：作者 ROM 的相对涡对流速度由同一组 PIV 实验提供。前缘轴结果很好，中弦轴在俯仰阶段明显高估；删除 Magnus 项后才改善。
4. 因而建议把它纳入 FLUXV 的 **瞬态 `CL(s/c)` + LEV 环量/轨迹机制门禁**，不把它包装成周期前飞升阻力总体验证。
5. Li et al. (2023) 的**计算模型确实是二维 DVM**；实验是 `AR=5` 的有限展长平板配上下端板形成的准二维流动。作者明确承认实验仍存在不可避免的三维 LEV 变形。因此它很适合完善 FLUXV 的截面 LEV/二次涡闭合，但不能单独验证有限翼三维诱导效应。

## PDF 获取结果

| 文献 | 当前状态 | 本地文件/替代源 | 备注 |
|---|---|---|---|
| Stevens & Babinsky (2017), *Experiments to investigate lift production mechanisms on pitching flat plates* | 已从 Cambridge 官方仓储下载 | `stevens_babinsky_2017_pitching_flat_plates.pdf` | 17 页，CC BY 4.0；可公开引用和按许可再利用 |
| Thielicke, Kesel & Stamhuis (2011), *Reliable Force Predictions for a Flapping-Wing Micro Air Vehicle: A “Vortex-Lift” Approach* | 已从 University of Groningen 官方仓储下载 | `thielicke_kesel_stamhuis_2011_vortex_lift.pdf` | 15 页；与 `candidates_20260807` 既有文件 SHA-256 完全一致 |
| Ramesh et al. (2014), *Discrete-vortex method with novel shedding criterion…* | JFM 正文仍无稳定匿名直链 | `ramesh_2013_phd_ldvm_foundation.pdf` + `../ramesh_ldvm_v2_5_source.zip` | Glasgow Enlighten 明确写着全文当前不可用；ResearchGate 作者上传页存在但直接文件返回 403。NC State 官方博士论文完整覆盖 LDVM 理论和工况，另有作者 Fortran 源码 |
| Martínez-Carmena et al. (2022), *Modulation of the Leading-Edge Vortex Shedding Rate…* | 已取得作者公开上传的完整会议稿；AIAA 官方正文仍受限 | `martinez_carmena_et_al_2022_lev_shedding_rate.pdf`；补充源 `martinez_carmena_2023_phd_includes_2022_dvm.pdf` | 会议稿 18 页；Glasgow 官方博士论文第 5 章（pp. 89–122）还给出更完整的 SVDVM 方程、算例和讨论。EPFL 官方记录本身只有 LICENSE bundle、没有 PDF bundle |
| Li et al. (2023), *Lift generation mechanism of the leading-edge vortex for an unsteady plate* | 项目原有完整出版社 PDF，已复制到统一目录 | `li_et_al_2023_lev_unsteady_plate.pdf` | Cambridge 官方端点在匿名请求中返回访问页而非 PDF；当前本地 PDF 有效、28 页 |

官方链接：

- Stevens 2017：<https://api.repository.cam.ac.uk/server/api/core/bitstreams/6bfe0ca7-39e3-43ec-a50b-dc88fa3321cb/content>
- Thielicke 2011：<https://pure.rug.nl/ws/portalfiles/portal/57154251/1756_8293.3.4.201.pdf>
- Ramesh 2014 出版记录：<https://eprints.gla.ac.uk/99370/>；开放博士论文：<https://repository.lib.ncsu.edu/bitstreams/66ace1bb-cef8-45b3-84b9-afdc340eefd7/download>
- Martínez-Carmena 2022 DOI：<https://doi.org/10.2514/6.2022-2416>；作者上传页：<https://www.researchgate.net/publication/357589781_Modulation_of_the_Leading-Edge_Vortex_Shedding_Rate_in_Discrete-Vortex_Methods>；开放博士论文：<https://theses.gla.ac.uk/83534/2/2023martinezcarmenaphd.pdf>
- Li 2023 官方记录：<https://doi.org/10.1017/jfm.2023.569>

这些 PDF 是本地研究材料。除明确 CC BY 的 Stevens 论文外，不应默认将其他出版社 PDF 或仓储副本提交到公开 GitHub；公开仓库优先保存引用、来源链接、数字化数据及自行实现的代码。

## 文件完整性

| 文件 | 页数 | 字节数 | SHA-256 |
|---|---:|---:|---|
| `stevens_babinsky_2017_pitching_flat_plates.pdf` | 17 | 6,222,584 | `aa41b2311916fe6500cbb199bb1230aefc2b3ff23863660ed38df7bec8653b93` |
| `thielicke_kesel_stamhuis_2011_vortex_lift.pdf` | 15 | 350,058 | `5cd4ddf94e7c01ee38fe73ca69dc4763559e5a597c5a15458160bde92613bb44` |
| `ramesh_2013_phd_ldvm_foundation.pdf` | 186 | 68,823,845 | `735f0cb9af7636bf3fa3e21845f4a722b40a55004c409891af0687fa87f740c4` |
| `martinez_carmena_et_al_2022_lev_shedding_rate.pdf` | 18 | 1,956,357 | `80bf445864f014ba094677480410d64ccd5f3abe3f9791720e7eebb5e99dbf01` |
| `martinez_carmena_2023_phd_includes_2022_dvm.pdf` | 202 | 70,651,330 | `aad40937d9b9f92506291d0840e984bf5c9f5df5d1683884bb11dd0141575cfd` |
| `li_et_al_2023_lev_unsteady_plate.pdf` | 28 | 3,141,936 | `af2e9a005c658db4c2f0b8ab23f3dabf4b68be2c6975ff2f33476078da27d626` |

## Stevens & Babinsky 2017：实验与工况闭合度

### 真实实验链

- Cambridge University Engineering Department 水平拖曳水槽；来流由模型以恒速拖曳形成。
- 二分量 Flow Dynamics 天平，采样 `1 kHz`；已用已知砝码校准。
- 每个载荷工况重复 `10` 次，论文曲线再做 `100` 点移动平均。
- 用同一运动在空气中测得机构/翼惯性载荷并从水中信号扣除。
- 平面 PIV、染色流动显示、LEV/TEV 轨迹和 Lamb–Oseen 拟合环量与直接测力同步构成观测链。
- PIV 自报总误差相当于来流速度的约 `2.7%`；保持 `Re=10,000` 的误差约 `±2.5%`。

### 几何

- 碳纤维有限展长矩形平板；圆钝前缘。
- 弦长 `c = 0.12 m`。
- 厚度 `t/c = 2.5%`，即约 `3 mm`。
- 有效展弦比 `AR = 4`。
- 水面 skim plate 作为对称面，故有效展长是物理半翼的两倍：有效全展长约 `0.48 m`，实际浸水半翼展长约 `0.24 m = 2c`。
- PIV/染色测量平面在距自由翼尖 `2c` 处，即对称中剖面。

**未闭合项：**论文没有给出圆钝前缘的精确半径/截面坐标，也没有单独给出尾缘厚度细节。这不会阻止建立矩形有限翼和二维平板 ROM 工况，但会限制对分离起始时刻的严格几何一致性。

### 有来流的大幅运动

- 弦长雷诺数 `Re = 10,000`；水温变化时调整拖曳速度以维持该值。
- 起始几何攻角 `0°`，随后俯仰到 `45°` 并保持平移。
- 俯仰在一个弦长的平移距离内完成，即 `0 < s/c < 1`。
- 论文定义的恒速段俯仰约化频率为
  `k = alpha_dot c / (2 U_inf) = 0.392`。
- 运动采用 Wang–Eldredge 平滑 ramp，平滑参数 `a_s = 11`。
- 两个实验轴位：前缘轴 `x_p/c = 0`；中弦轴 `x_p/c = 0.5`。

**未闭合项：**平滑 ramp 的显式函数没有在本文重印，需要按 Wang & Eldredge (2012) 的定义实现；拖曳速度只通过 `Re` 给出，没有冻结一个不随水温变化的米制数值。对无量纲 ROM/UVLM 比较不构成障碍。

### 公开观测量

适合优先数字化的实验图：

1. Figure 13：两个轴位的 `CL(s/c)`，范围约 `0–20`；
2. Figure 14：俯仰区 `0–4` 的 `CL(s/c)` 放大图；
3. Figure 18：LEV/TEV 的无量纲 `x,z` 轨迹；
4. Figure 20：实验 `Gamma_LEV/Gamma_inf` 随 `s/c`；
5. Figure 21：实验 `CL` 与 ROM 及各载荷分量；
6. Figure 12：无来流对照，用于单独核对 added-mass（不属于当前前飞主工况）。

PDF 检查表明载荷、环量和轨迹图是矢量图元，只有染色/PIV 图像主要是栅格图。因此曲线可以高质量数字化，并冻结为来源明确的 CSV。

### 数据缺口

- 未发现论文随附原始数组、PIV 矢量场数据包或 supplementary dataset。
- 没有公开 drag；论文明确写着 “Only lift is reported”。
- 没有给出直接测力的不确定度条或 `CL` 置信区间；只有重复次数、滤波方法、Re/PIV 误差。
- 两分量天平虽测到切向力，未发表的阻力数据不能从升力图反演。
- Figure 13/14/21 的实验升力曲线存在重复表达，不能当作三组独立观测。

## 作者 ROM 的自身验证是否通过

作者模型由以下分量组成：

- 非环量/虚质量；
- 基于修正 Wagner 函数的 LEV 环量增长；
- vortex-growth 与 vortex-advection impulse 项；
- pitch-rate/Magnus（virtual-camber）项。

关键限制是，LEV/TEV 的相对对流速度取自同一 PIV 实验的近似值：前缘轴约 `0.5 U_inf`，中弦轴约 `0.6 U_inf`。因此 Figure 21 是**实验辅助闭合后的自验证**，不是完全独立的盲预测。

- 前缘轴：模型能很好捕捉实验升力历史形状，俯仰段略高估；可认为通过定性/半定量自验证。
- 中弦轴：模型在俯仰段明显高估。论文自己指出去掉 Magnus 项后相关性显著改善；不能声称原始模型对两个轴位都验证通过。

## 对 FLUXV 的推荐用法

### 可以做

- 建立两条严格分开的 case：`LE-axis` 与 `mid-chord-axis`。
- 首先只对比 Figure 13/14 的实验 `CL(s/c)`，保留初始峰值、俯仰结束峰值、后续衰减和 `s/c≈7–8` 次峰等相位特征。
- 再用 Figure 18/20 检查 FLUXV 的 LEV 对流速度、环量增长和脱落时序，避免只靠总升力补偿错误物理。
- 作者 ROM 同时报告 `as-published` 与 `mid-chord without Magnus` 两个版本，不应把后者冒充原论文默认模型。

### 不可以做

- 不可给出 Stevens 实验 `CD` 误差；不存在公开真值。
- 不可把 Figure 21 当成与 PIV 无关的独立校核。
- 不可把这个一次性 pitch-up/hold 工况描述为周期扑翼前飞。
- 不可用 Figure 13、14、21 的同一升力测量重复计权来放大样本量。

### 复现闭合度评级

| 维度 | 评级 | 说明 |
|---|---:|---|
| 真实实验 | 5/5 | 直接测力 + PIV + 染色 + 空气惯性扣除 |
| 几何 | 4/5 | `c/t/AR/轴位` 齐全；前缘半径/精确截面缺失 |
| 运动/来流 | 4/5 | `Re/k/45°/1c/a_s` 齐全；ramp 公式需追溯二级来源 |
| 升力真值 | 4/5 | 高质量矢量曲线可数字化；无原始数组/力误差条 |
| 阻力真值 | 0/5 | 未发表 |
| LEV 机制真值 | 4/5 | 环量和轨迹均有图；无原始 PIV 场 |
| 作者模型独立性 | 2/5 | 使用同实验的对流速度；中弦默认模型未通过 |

总判断：**值得复现，优先级中高；定位为“有来流、大幅瞬态俯仰的升力与 LEV 机制 case”，而不是完整升阻力 benchmark。**

## Li et al. 2023：二维性的准确表述

- DVM 本体是二维点涡/涡片模型，使用 200 个固定 bound vortices，并以二维涡量冲量计算载荷。
- 实验平板 `c=120 mm`、span `=600 mm`，故几何 `AR=5`；上下端板用于压制翼尖流，目标是形成准二维中剖面流动。
- 平板前/后缘削成 `60°` 尖缘，厚度 `5 mm`。
- 有非零来流；`Re=24,000`、`St=0.04`、`k=0.3`；前缘轴同步 pitch–plunge；`alpha_eff,max=16/24/32/40°`。
- 实验包含直接测力和 100 周期相位平均 PIV，但论文主要发布升力而非完整升阻力双通道。
- 作者明确将模型/实验差异的一部分归因于实验 LEV 在对流中不可避免的三维变形及 spanwise flow。

因此推荐把它用于：

1. 校准/检验截面 LEV shedding、二次涡、LEV 脱落和高攻角升力衰减；
2. 给 FLUXV 的三维 UVLM/ULLT 主体提供二维截面物理闭合；
3. 不用它单独决定有限翼 spanwise LEV 稳定化或诱导阻力修正。
