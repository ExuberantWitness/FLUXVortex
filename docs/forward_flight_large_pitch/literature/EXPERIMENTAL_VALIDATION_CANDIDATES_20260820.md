# FLUXV/V5H 后续实验文献调研与复现候选（2026-08-20）

## 1. 结论摘要

现有三篇主验证文献已经覆盖了低雷诺数非定常升力、升沉/俯仰涡动力学和一个整翼/前飞方向，但还没有系统覆盖以下四个盲区：

1. **单次高幅 pitch-ramp 的轴位和俯仰速率扫描**；
2. **有限翼翼尖涡、展向流和动态失速的共同作用**；
3. **变来流（surge）与俯仰同步时的非线性耦合**；
4. **从低雷诺数到工程高雷诺数的公开动态失速数据库**。

本轮检索去除了已有清单中的 Ramesh 2014、Li 2023、Thielicke 2011、Izraelevitz 2014、Martínez-Carmena 2022、Stevens 2017、Han 2017、Bhowmik 2013、Floryan 2017 和 Faure 2022，得到 **10 个新增候选**。其中建议立即推进的不是 10 篇全部，而是下面四项：

1. **Yu et al. 2018**：与当前大幅俯仰、刚性平板和载荷分解最贴合；
2. **Yu 2016**：最直接的有限翼 `AR=4`、`0°→45°`、测力与 stereo-PIV 校验；
3. **Strangfeld et al. 2025**：补齐同步 surge–pitch 和瞬时来流耦合；
4. **Unsteady Aerodynamics Open Data Set**：先建立机器可读的多翼型动态失速回归库。

Kiefer 2022、Chellini 2026 和 PIBE 数据库应作为第二阶段的高雷诺数/厚翼型/压力数据外推门，不能直接替代当前低雷诺数 V5H 主工况。

## 2. 沿用的筛选标准

候选文献必须满足或接近以下标准：

1. 有非零来流或明确的平动/变来流；
2. 有真实测力、表面压力积分或 PIV，而不是只用 CFD 相互验证；
3. 几何、运动学、参考速度和无量纲参数足以唯一重建；
4. 优先低阶模型、离散涡、升力线、状态空间或解析非定常理论；纯实验论文可作为独立验证源；
5. 排除 clap-and-fling、多翼强干扰、悬停专用和几何不闭合工况；
6. 原始数组公开优先；其次是可数字化矢量图；只给定性流场的论文降级；
7. 模型中若使用同一实验数据标定，必须把“标定集”和“盲验集”分开；
8. 先冻结文献、工况和指标，再运行模型；不得根据目标曲线调阈值或参数；
9. 载荷、流场、LEV/TEV 位置和环量应分通道评价，不能用单一综合分数掩盖失配；
10. 文献相关不等于可复现：数据访问、坐标定义、参考量和误差带必须另设 intake 门。

### 2.1 排序评分

每项按 0–2 分评价：

- `E`：独立实验质量；
- `C`：几何/运动/参考量闭合；
- `D`：数据可获得性；
- `K`：与当前大幅前飞运动学的接近程度；
- `O`：对现有三篇证据的正交增益。

总分只用于安排 intake 顺序，不是科学结果评分。

## 3. 第一优先级：建议正式立项的新增文献

### P1. Yu, Amandolese, Fan & Liu (2018), JFM

**Experimental study and modelling of unsteady aerodynamic forces and moment on flat plate in high amplitude pitch ramp motion**  
DOI: <https://doi.org/10.1017/jfm.2018.271>  
官方页面：<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/experimental-study-and-modelling-of-unsteady-aerodynamic-forces-and-moment-on-flat-plate-in-high-amplitude-pitch-ramp-motion/FA019649DC22BD3AAD8E4A1544352D75>

- 实验：风洞刚性平板，直接测量升力、阻力和俯仰力矩；
- 工况：`Re≈1.45×10^4`；约化俯仰速率 `0.01–0.18`；最大角 `30°/45°/60°/90°`；轴位从前缘扫到后缘；另比较近似 `AR≈8` 有限翼与准二维构型；
- 方法：Wagner/阶跃响应叠加、非线性定常输入和速率相关附加环量的时变低阶模型；
- 价值：同时检验非循环项、环量建立、轴位效应、载荷过冲和力矩，和当前 V5H 的误差分账最贴合；
- 数据状态：正文曲线可数字化；本轮未发现公开原始数组；
- 风险：原始测量数组不可直接下载，必须先做矢量图/坐标轴质量审计。

评分：`E2 C2 D1 K2 O2 = 9/10`。

**建议首个工况：** `45°` pitch-ramp，先选前缘轴和中弦轴各一例，固定一个中等约化速率；同时评价 `CL/CD/CM`，避免只看升力。

### P2. Yu (2016), Journal of Aeronautics, Astronautics and Aviation

**Low Reynolds Number Aerodynamics of Finite Wing at Low Reduced Pitch Rates**  
DOI: <https://doi.org/10.6125/15-1123-870>  
可访问全文页面：<https://www.researchgate.net/publication/304254510_Low_Reynolds_Number_Aerodynamics_of_Finite_Wing_at_Low_Reduced_Pitch_Rates>

- 实验：矩形平板有限翼，直接测力、染色流动显示和 lens-shift stereo-PIV；
- 工况：`AR=4`，前缘轴，`0°→45°` ramp/hold，`Re=0–13,000`，约化俯仰速率 `0–0.13`；正文含 `Re=8.9k, K=0.065` 的分相位流场；
- 对照：准定常势流在失速前可描述升力，但不能描述含翼尖涡贡献的阻力；
- 价值：这是新增候选中最接近“当前二维/准二维方法向有限翼外推”的直接实验门；
- 数据状态：全文和图可访问，未发现原始数组；
- 风险：图像数据需数字化，且 `Re=0` 静水加速分支不能和非零来流分支混在同一评分中。

评分：`E2 C2 D1 K2 O2 = 9/10`。

**建议用途：** 先复现 `Re≈8.9k, K=0.065`，把升力、阻力、翼尖/三分之四展向 PIV 分开验收；它比继续增加二维平板论文更能揭示当前有限翼闭合的真实缺口。

### P3. Strangfeld et al. (2025), JFM

**Airfoil synchronous surging and pitching**  
DOI: <https://doi.org/10.1017/jfm.2025.220>  
开放全文：<https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E7685333F34EA99AC8740307458B8F78/S0022112025002204a_hi.pdf/airfoil-synchronous-surging-and-pitching.pdf>

- 实验：NACA 0018，非定常风洞，压力积分载荷与沿弦压力分布；
- 工况：平均 `Re=3.0×10^5`，来流振幅 `51%`，攻角 `2°±2°`，约化频率 `0.097`，surge 与 pitch 相位差 `0°/90°/180°/270°`；
- 方法：一般非定常翼型理论和时变束缚涡片，分解 Joukowsky 与 impulsive-pressure 升力；
- 数据质量：每个实验约 `1328` 个周期，相位压力最大不确定度约 `ΔCp=±0.0047`；正文开放，曲线可数字化；
- 价值：当前三篇和 V5H 大多冻结父来流，本工况能直接检验“不能把 surge 和 pitch 简单独立叠加”的时变源项；
- 限制：攻角小、非大幅失速，是线性/弱分离边界的正交校验，而非大幅 LEV 主验证。

评分：`E2 C2 D1 K1 O2 = 8/10`。

**建议用途：** 作为 source-release / moving-parent 的首个实验门，优先比较四个相位差的相位 `CL` 与压力分布，而不是把它纳入大幅动态失速平均分数。

### P4. Unsteady Aerodynamics Open Data Set (2018), Zenodo

数据 DOI：<https://doi.org/10.5281/zenodo.1135424>

这是与多份同行评议论文和实验报告绑定的机器可读数据库，而不是单篇新模型论文：

- Glasgow：NACA 0015 / NACA 0030 正弦俯仰动态失速；
- NREL/OSU：LS(1)-0417MOD、NACA 4415、S809 正弦俯仰；
- CENER/DTU：NACA 643-418 的俯仰、襟翼和俯仰–襟翼组合；
- ForWind：DU00-W-212 在层流、固定网格和动态网格湍流下的气动力/压力；
- 文件：ASCII、NetCDF 和转换 notebook，总记录约数 GB；关联 Glasgow 数据库报告 DOI `10.5525/gla.researchdata.464`、三份 NREL 报告和 AIAA 2017 panel-code 论文。

评分：`E2 C1 D2 K1 O2 = 8/10`。这里 `C=1` 是因为各子库的参考量和运动约定不同，必须逐库 intake，不能把数据库 DOI 本身当成完整工况定义。

**建议用途：** 先建立自动数据解析、单位/符号/相位一致性回归；选一个 NACA0015 case 做 blind baseline，再扩到厚翼型和湍流入口。它最适合做持续集成数据集，不适合直接替代高幅平板主工况。

## 4. 第二优先级：高价值外推和机制候选

### P5. Kiefer et al. (2022), JFM

**Dynamic stall at high Reynolds numbers induced by ramp-type pitching motions**  
DOI: <https://doi.org/10.1017/jfm.2022.70>  
官方页面：<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-stall-at-high-reynolds-numbers-induced-by-ramptype-pitching-motions/B11424A7CA1CFF79BAEA504506B70833>  
公开数据：<https://doi.org/10.34770/b3vq-sw14>

- NACA 0021，高压风洞、低马赫数；
- `Re_c=0.5×10^6–5.5×10^6`，约化频率 `k=0.01–0.40`，变化均值角和振幅；
- 表面压力瞬态数据；数据公开；
- 主要机制是渐进尾缘失速，动态失速涡约在中弦形成；
- 价值：检验模型能否越过当前低 Re/尖前缘假设；
- 风险：物理机制与低 Re 平板 LEV 不同，只能作为外推失败/适用域边界，不应拿来调低 Re 模型。

评分：`E2 C2 D2 K1 O2 = 9/10`。

### P6. Chellini, De Tavernier & von Terzi (2026), Wind Energy Science

**Experimental characterization of dynamic stall of the FFA-W3-211 wind turbine airfoil**  
论文 DOI：<https://doi.org/10.5194/wes-11-753-2026>  
数据 DOI：<https://doi.org/10.4121/92716ecf-b075-41a1-ab93-8e2af785a404>

- FFA-W3-211 厚风机翼型，四分之一弦轴正弦俯仰，压力采样 `300 Hz`；
- 静态范围 `Re_c=5×10^5–3.5×10^6`；动态测试最高到 `2×10^6`；
- 公开数据覆盖 `k=0.023–0.365`，论文明确比较 `0.046/0.091/0.182` 等频率与 Reynolds 数组合；
- 正、负失速区和线性区均有实验；动态结果通常平均 50 个周期；
- 价值：公开、覆盖广，适合作为工程厚翼型和 reattachment 门；
- 风险：翼型、Re 和尾缘分离远离当前 V5H 主域，先做数据解析和失败边界，不先要求精度通过。

评分：`E2 C2 D2 K1 O2 = 9/10`。

### P7. Toppings & Yarusevych (2025), JFM

**Transient dynamics of stall and reattachment at low Reynolds number**  
DOI: <https://doi.org/10.1017/jfm.2025.348>  
官方开放页面：<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/transient-dynamics-of-stall-and-reattachment-at-low-reynolds-number/F97A76B0038B843B58BDC6625B67A8F2>

- NACA 0018，二维近似翼型和 `AR=2.5` 有限翼；弦长 `0.2 m`；
- `Re_c=8×10^4–1×10^5`；攻角在 `10°↔13°` 间变化；约化俯仰率 `3×10^-5–5×10^-3`；
- 同步直接测力与 PIV，并同时研究改变 Reynolds 数触发的失速/再附；
- 数据可向通讯作者申请，未直接公开下载；
- 价值：同一几何下比较二维/有限翼、失速/再附和速度改变；
- 风险：幅值较小且靠近 LSB 临界态，更适合验证延迟与再附时标，不是高幅 LEV 载荷基准。

评分：`E2 C2 D1 K1 O2 = 8/10`。

### P8. Damiola et al. (2024), JFM

**Modelling the unsteady lift of a pitching NACA 0018 aerofoil using state-space neural networks**  
DOI: <https://doi.org/10.1017/jfm.2024.148>  
官方页面：<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/modelling-the-unsteady-lift-of-a-pitching-naca-0018-aerofoil-using-statespace-neural-networks/0BB2A98D2A2092C76682035087C20262>

- NACA 0018，`c=0.3 m`、`AR=1.8`、端板，绕中弦俯仰；`Re=2.8×10^5`；
- 47 个中剖面压力孔、`200 Hz`，压力积分升力；
- 比较来流湍流强度 `0.3%` 与 `8.2%`；可执行简谐、sine sweep 和 random-phase multisine；
- 验证含最高攻角 `28°` 的非线性正弦运动；
- 价值：适合把当前确定性模型与数据驱动状态空间模型做正交比较，并检验入口湍流敏感性；
- 风险：本轮未发现原始训练/验证时序公开入口；若只有曲线，模型复现会受训练集定义影响。

评分：`E2 C2 D1 K1 O2 = 8/10`。

### P9. PIBE NACA 63(3)418 dynamic-stall database (2024), Zenodo

数据 DOI：<https://doi.org/10.5281/zenodo.10638882>  
关联论文：<https://doi.org/10.1016/j.jsv.2022.117144>

- 仪器化 NACA 63(3)418，静态与动态俯仰；
- HDF5 原始包公开，动态文件约 `263 MB`；
- 含相位平均升力、壁面压力谱和远场声学谱；公开动态子集为均值攻角 `15°`、振幅 `15°`；
- 价值：载荷和压力的机器可读数据非常适合回归；额外声学通道可留作远期；
- 风险：当前任务不做气动声学，且必须从项目报告中进一步冻结 Re、频率、轴位与归一化定义后才能成为正式 benchmark。

评分：`E2 C1 D2 K1 O2 = 8/10`。

### P10. Ayancik & Mulleners (2022), JFM Rapids

**All you need is time to generalise the Goman–Khrabrov dynamic stall model**  
DOI: <https://doi.org/10.1017/jfm.2022.381>  
开放页面：<https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/all-you-need-is-time-to-generalise-the-gomankhrabrov-dynamic-stall-model/5DAFFF508F0811B159C53499531D1BDB>

- 用物理时标替代 Goman–Khrabrov 模型的手调常数；
- 用三套实验动态失速数据验证，覆盖不同翼型、`Re=75,000–1,000,000`、正弦和 ramp-up；
- 报告总体 `R²>0.85`，重点评价首个升力峰时间与失速延迟；
- 价值：适合建立“不按每个频率重新调参”的低维状态模型对照；
- 风险：论文是跨数据集建模总结，不是新的统一原始实验库；必须追溯三套底层数据，防止把同一实验既标定又盲验。

评分：`E1 C1 D1 K1 O2 = 6/10`。

## 5. 机制补充，但暂不进入主载荷评分

### 5.1 Granlund, Ol & Bernal (2013), JFM

**Unsteady pitching flat plates**  
DOI: <https://doi.org/10.1017/jfm.2013.444>

- 水洞刚性平板，直接升阻力与定性流动显示；
- 约化速率 `0.01–0.5`，攻角 `0–90°`，轴位从前缘到后缘；
- 高度匹配，但与 Yu 2018 的参数扫描重叠较大；
- 建议作为 Yu 2018 的跨设施一致性复核，而不是单独先立一个完整分支。

### 5.2 Buchner et al. (2012), Experiments in Fluids

**Stereoscopic and tomographic PIV of a pitching plate**  
DOI: <https://doi.org/10.1007/s00348-011-1218-8>

- pitch-hold-return 平板，`Re=7,500`、无量纲俯仰率约 `0.93`；
- 3C-2D 与 3C-3D PIV，显示 LEV/TEV 中的展向流和三维涡丝；
- 没有同等强度的直接载荷通道，适合做流场/三维性机制门，不进入主 `CL/CD` 评分。

### 5.3 Mancini (2017), University of Maryland dissertation

**Experimental Investigation into Unsteady Force Transients on Rapidly Maneuvering Wings**  
开放论文与 DOI：<https://doi.org/10.13016/M25717R55>

- 低 Re 水洞平板，包含静止起动、非零来流俯仰和大襟翼偏转；
- 有非定常测力和时间分辨速度场；
- 适合补“起动–俯仰–襟翼”三个基础脉冲响应；
- 需要先从 41 MB 论文中整理工况矩阵和数据出处，本轮不把摘要层信息当成可执行工况。

## 6. 与现有三篇验证的互补关系

| 证据盲区 | 首选新增文献 | 主要可观察量 | 不能替代的现有证据 |
|---|---|---|---|
| 高幅 pitch-ramp、轴位扫描 | Yu 2018 | `CL/CD/CM`、峰值/相位、轴位响应 | Baik 的 pitch–plunge LEV 轨迹 |
| 有限翼翼尖涡/展向流 | Yu 2016 | 升阻力、stereo-PIV、翼尖/展向平面 | 现有准二维流场一致性 |
| 时变来流与俯仰耦合 | Strangfeld 2025 | 相位 `CL/Cp`、四相位差 | 冻结父来流下的积分器阶次 |
| 多翼型动态失速数据回归 | Zenodo 1135424 | 时序载荷/压力、多翼型 | 单一大幅运动的机制验证 |
| 工程高 Re 外推 | Kiefer 2022 / Chellini 2026 | 动态压力、法向力、stall/reattach | 低 Re 尖前缘 LEV 机制 |
| LSB、入口湍流和再附 | Toppings 2025 / Damiola 2024 | 力、PIV、压力、时标 | 大幅俯仰的载荷过冲 |

## 7. 推荐执行顺序与效果检验节点

### G0：资料 intake，不运行 FLUXV

1. 下载或登记官方全文/数据，不使用来历不明转载；
2. 对每个文件记录 URL、DOI、许可证、字节数和 SHA-256；
3. 建立参数闭合表：几何、轴位、`Re`、`k` 定义、参考面积/速度、相位零点、滤波/平均方式；
4. 原始数组不可得时，登记待数字化图号、坐标轴、线型、误差带和分辨率。

**通过条件：** 工况可从报告唯一生成，或明确标记为 `BLOCKED_DATA_REQUEST`；不得用猜测补参数。

### G1：开放数据机械回放

先做 Zenodo `1135424` 的一个 NACA0015 case，再做 Kiefer/Chellini 中一个公开压力 case：

- 只写 parser、单位转换和参考量校验；
- 不调模型；
- 原始文件 SHA、解析后数组 SHA、时间/相位单调性和周期计数进入 artifact。

**通过条件：** 两次 fresh parse 字节一致；随机改单位、相位、符号或一行数据会 fail-closed。

### G2：第四篇正式复现——Yu 2018

- 先冻结 `45°`、两个轴位、一个约化速率；
- 指标预注册：相位 `CL/CD/CM` 的相对 L2、峰值幅值、峰值时刻、冲量；
- 循环/非循环/LEV/压力项分别输出，禁止只给合力；
- 数字化不确定度单独传播，不把图像误差算成模型误差。

**通过条件：** A/B fresh run 语义产物逐字节一致；所有阈值在读取目标残差前冻结。

### G3：有限翼门——Yu 2016

- 固定 `AR=4, 0→45°, Re≈8.9k, K=0.065`；
- 载荷和 PIV 分开判定；
- 记录翼尖和三分之四展向位置；
- 若二维模型过升力但阻力失败，只能得出“二维载荷适用域受限”，不得通过重新标定吞掉翼尖涡误差。

### G4：moving-parent/source-release 门——Strangfeld 2025

- 固定四个 surge–pitch 相位差；
- 使用论文给定 `51%` 速度振幅、`2°±2°`、`k=0.097`；
- 与纯 surge、纯 pitch 和简单叠加同时比较；
- 重点评价相位误差和 `Cp(x/c,t)`，不只比较周期均值。

### G5：外推门

只有 G2–G4 通过后，才运行 Kiefer 2022、Chellini 2026、Toppings 2025 或 PIBE：

- 这些工况用于界定适用域，不用于反向调低 Re 模型；
- 任一高 Re/厚翼型失败不自动否定低 Re V5H，但必须明确记录机制边界。

## 8. 当前推荐组合

若资源只允许新增三项，建议：

1. **Yu 2018**：方法最贴合、信息量最高；
2. **Yu 2016**：有限翼证据最正交；
3. **Strangfeld 2025**：moving-parent/变来流最正交。

同时把 Zenodo `1135424` 作为基础设施数据集接入，但不要把“解析了公开数据库”当成完成第四篇论文复现。

## 9. 本轮没有晋级的候选类型

- 只有 CFD、LES 或 UVLM 对照、没有独立实验载荷；
- 只有定性流场图、没有可定量对齐坐标或载荷；
- 悬停/静止流体为唯一工况；
- 多翼 clap-and-fling 或强干扰，无法隔离单翼机制；
- 几何/运动由闭源控制器或未公开柔性变形决定；
- 只给训练后的 ROM 误差、没有实验输入/输出时序；
- 数据库名义公开但文件受限、无许可证或缺参考量。

## 10. 调研范围声明

- 本报告是候选文献与数据源的 **intake 研究**，不是复现实验结果；
- 本轮没有运行 FLUXV、Ptera、GT 或 scorer，也没有读取论文目标数据来调参数；
- 没有下载或重新分发出版社 PDF；所有链接均指向出版者、机构库或公开数据仓库；
- 公开数据是否真的包含所需字段，仍需在 G0/G1 以文件级校验确认；
- 2026 年文献属于当前新资料，正式引用前还应固定最终版本和数据版本。

