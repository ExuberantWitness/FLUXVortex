# FLUXV 修改版后续实验验证候选文献

## 筛选边界

本清单只保留同时满足或接近以下条件的文献：

1. 有非零来流；
2. 有真实测力/压力积分实验，而非只用 CFD 互证；
3. 几何、运动和无量纲工况足以重建；
4. 自身核心方法是准定常、升力线、状态空间、离散涡或其他低阶模型，不以 CFD-CSD 为主；
5. 排除 clap-and-fling、多翼强干扰和悬停专用工况。

## 2026-08-12 获取状态

- Ramesh 2014 的 JFM 正文仍没有稳定、合法的匿名 PDF 直链；可执行复现不受阻：本地已有作者 `LDVM v2.5` Fortran 源码，并已取得完整覆盖该算法与验证工况的 NCSU 博士论文。
- Thielicke 2011、Martínez-Carmena 2022、Li 2023 和 Stevens 2017 均已有经过页数/校验和检查的本地 PDF。
- 完整下载路径、官方来源、文件校验值与 Stevens 专项审计见：
  `docs/forward_flight_large_pitch/literature/candidates_20260812/LITERATURE_ACCESS_AND_STEVENS_AUDIT.md`。
- 这些 PDF 仅作为本地研究资料；公开 GitHub 提交只保存官方链接、自行数字化的数据、报告和复现代码，不默认再分发出版社 PDF。

## 第一优先级：建议用户先选四篇

### A. Ramesh et al. (2014), JFM

**Discrete-vortex method with novel shedding criterion for unsteady aerofoil flows with intermittent leading-edge vortex shedding**  
DOI: <https://doi.org/10.1017/jfm.2014.297>

- 低阶模型：LESP 调制的离散涡方法（LDVM），不是 CFD；
- 实验：SD7003，`Re=30,000`，大幅 pitch-hold-return 至 `25 deg`，公开相位 `CL/CD` 和流场；
- 另有明确的周期 pitch-plunge 工况；
- 本地已有作者 Fortran 源码与完整替代论文来源：
  `docs/forward_flight_large_pitch/literature/ramesh_ldvm_v2_5_source.zip`；
  `docs/forward_flight_large_pitch/literature/candidates_20260812/ramesh_2013_phd_ldvm_foundation.pdf`；
- 复现难度：低；主要风险是 `LESPcrit` 具有翼型/Re 依赖。

**建议用途：** 最快建立动态失速/LEV 的源码级低阶基线，并直接检验 FLUXV 相位升阻力。

### B. Li et al. (2023), JFM

**Lift generation mechanism of the leading-edge vortex for an unsteady plate**  
DOI: <https://doi.org/10.1017/jfm.2023.569>

- 低阶模型：改进 DVM，含 LE/TE 涡、涡核和二次涡机制；
- 实验：水洞直接测力 + 100 周期相位平均 PIV；
- 几何/工况：平板 `c=120 mm, span=600 mm`，`Re=24,000, St=0.04, k=0.3`，前缘轴 pitch-plunge，最大有效攻角 `16/24/32/40 deg`；
- 本地 PDF：`docs/forward_flight_large_pitch/literature/candidates_20260812/li_et_al_2023_lev_unsteady_plate.pdf`；
- 复现难度：低至中；原始数组未公开，需要矢量图数字化。

该论文的 DVM 是严格二维模型；实验翼 `AR=5`，配合上下端板形成准二维中剖面，但作者仍观察到不可完全消除的三维 LEV 变形。因此它适合截面 LEV/二次涡与升力校核，不应单独用于有限翼诱导阻力或展向稳定化验证。

**建议用途：** 对 FLUXV 的高攻角升力饱和、LEV 增升和二次涡机制做最直接的实验门禁。

### C. Thielicke, Kesel & Stamhuis (2011)

**Reliable Force Predictions for a Flapping-Wing Micro Air Vehicle: A “Vortex-Lift” Approach**  
DOI: <https://doi.org/10.1260/1756-8293.3.4.201>

- 低阶模型：叶素法 + Polhamus 型 vortex-lift + 诱导攻角；
- 实验：真实有限整翼慢速前飞，力天平测竖直/水平力；每工况 18 周期、3 次重复；
- 工况：翼展约 `0.33 m`、平均弦长约 `40 mm`、`AR=8.3`，来流 `2.28/2.57/2.84 m/s`，频率 `3.5--9.5 Hz`，`J=0.6--1.7`；
- 本地 PDF：`docs/forward_flight_large_pitch/literature/candidates_20260812/thielicke_kesel_stamhuis_2011_vortex_lift.pdf`；
- 复现难度：中；运动会随柔性变形，只有部分完整运动学示例，主要公开周期均值。

**建议用途：** 当前最好的新有限翼/整翼实验闭环，物理对象比二维 foil 更接近 FLUXV。

### D. Izraelevitz & Triantafyllou (2014), JFM

**Adding in-line motion and model-based optimization offers exceptional force control authority in flapping foils**  
DOI: <https://doi.org/10.1017/jfm.2014.7>；作者稿：<https://jsi.scripts.mit.edu/jsi/wp-content/uploads/2015/11/IzraelevitzJFM2014_VOR.pdf>

- 低阶模型：Wagner/线性非定常升力 + 准定常阻力 + 附加质量；
- 实验：拖曳水槽、六分量测力和 PIV，并有同一翼静态极曲线；
- 几何/工况：NACA0013，`c=55 mm, AR=6.5, Re=11,000`，`h/c=1`，`St=0.1--0.5`，含 surge-heave-pitch；
- 复现难度：中；优先选解析运动 I--III，避免使用经过实验残差迭代优化的 IV--VI 作为盲验。

**建议用途：** 验证非定常状态、附加质量以及流向运动对推力/侧向力的共同影响。

## 第二优先级：机制补充

### E. Martínez-Carmena et al. (2022), AIAA SciTech

**Modulation of the Leading-Edge Vortex Shedding Rate in Discrete-Vortex Methods**  
DOI: <https://doi.org/10.2514/6.2022-2416>

- DVM/LDVM 低阶方法，以剪切层速度调制 LEV 环量供给率；
- NACA0015 风洞实验，压力积分相位升力/力矩 + TR-PIV；
- `c=0.3 m, U=30 m/s, Re=5.5e5`，均值攻角 `20 deg`、幅值 `8 deg`、`k=0.025--0.15`；
- 本地会议稿：`docs/forward_flight_large_pitch/literature/candidates_20260812/martinez_carmena_et_al_2022_lev_shedding_rate.pdf`；
- 公式更完整的博士论文：`docs/forward_flight_large_pitch/literature/candidates_20260812/martinez_carmena_2023_phd_includes_2022_dvm.pdf`；
- 风险：高 Re 尾缘分离可能超出 LDVM 假设。

### F. Stevens & Babinsky (2017), Experiments in Fluids

**Experiments to investigate lift production mechanisms on pitching flat plates**  
开放全文：<https://link.springer.com/article/10.1007/s00348-016-2290-x>

- 解析 ROM：Wagner/LEV 环量、虚质量、pitch-rate/Magnus 和涡对流；
- 直接升力、PIV、染色、LEV 环量；
- AR=4 平板、`c=0.12 m, Re=10,000`，在一个弦长内从 `0` pitch 至 `45 deg`，前缘轴/中弦轴两组；
- 本地 PDF：`docs/forward_flight_large_pitch/literature/candidates_20260812/stevens_babinsky_2017_pitching_flat_plates.pdf`；
- 几何、`Re/k/45°/1c` 运动、两轴位、测力与 PIV 流程足以重建；Figure 13/14 可数字化瞬态 `CL(s/c)`，Figure 18/20 可校核 LEV 轨迹与环量；
- 风险：论文明确只公开升力，没有可用于误差计算的阻力曲线；作者 ROM 的涡对流速度取自同一 PIV，且中弦轴默认 Magnus 项明显高估，不能称为完全独立盲验。

**建议用途：** 中高优先级的“一次性大幅 pitch-up/hold 升力与 LEV 机制门”，不是周期扑翼升阻力双通道 benchmark。

### G. Han, Chang & Han (2017), Bioinspiration & Biomimetics

**An aerodynamic model for insect flapping wings in forward flight**  
DOI/摘要：<https://pubmed.ncbi.nlm.nih.gov/28362636/>

- 半经验准定常模型，含 potential/vortex/rotation/added-mass 分账；
- 147 个 `J-alpha` 标定点，另有人工翼拍和真实 hawkmoth 翼拍的直接力/矩验证；
- 与 FLUXV 分账结构高度相关；
- **条件性候选：** 必须先取得全文并冻结 planform、根部偏置和参考速度定义，不能从已有摘要猜测。

### H. Bhowmik, Das & Ghosh (2013)

**Aerodynamic modelling of flapping flight using lifting line theory**  
DOI: <https://doi.org/10.1108/20496421311298134>

- 广义 Prandtl 升力线准定常模型，含扭转、弯度、诱导阻力和型阻；
- 风洞 ATI Nano-43 测力，`U=0--7 m/s`、`f=4.36/7.03 Hz`，有平均竖直力和推力曲线；
- 总翼展 `0.50 m`、根弦约 `0.12 m`、椭圆翼；
- **条件性候选：** 被动扭转的实验瞬时翼形未测，理论设计扭转不一定等于真实翼形。

## 低风险但信息量较弱的备选

- Floryan et al. (2017), **Scaling the propulsive performance of heaving and pitching foils**, DOI <https://doi.org/10.1017/jfm.2017.302>：超过千组水洞均值数据和准定常/附加质量标度，但主要公开周期均值，且幅值较小。
- Faure et al. (2022), **Flapping wing propulsion: comparison between discrete vortex method and other models**, DOI <https://doi.org/10.1063/5.0083158>：DVM/动态失速/升力线与水洞实验齐全，但主工况接近纯升沉，偏离“大幅俯仰/扭转”主目标。

## 不建议进入本轮

- Aerobat/Sihite：虽有实验和 Wagner 升力线，但十关节几何、连杆尺寸与完整运动不足，不能唯一重建；
- Hirato/Kumar：主要用 RANS/Euler/UVLM 数值互证，缺独立实验载荷；
- Mayo：CFD-CSD 主路线；
- Armanini/Caetano：四翼 clap-and-fling；
- Sane/Wang/Nabawy 当前公开验证工况：以悬停或静止流体为主。

## 推荐选择

- 若优先完善 **LEV/动态失速物理**：`Ramesh 2014 -> Li 2023 -> Martínez-Carmena 2022`；
- 若优先增加 **有限翼/真实整翼前飞证据**：`Thielicke 2011 -> Han 2017（先补源） -> Bhowmik 2013`；
- 若优先验证 **状态空间、附加质量、surge-heave-pitch**：`Izraelevitz & Triantafyllou 2014 -> Stevens 2017`。
