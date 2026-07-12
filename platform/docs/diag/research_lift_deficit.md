# 机理裁定：全工况升力亏空 gap_L ≈ −0.52·f − 0.071·f·α − 0.25 [N]（准定常静态失速切割缺乏动态失速延迟）

**指纹回顾（28 工况，测量−仿真，RoboEagle 膜翼，U=6–8 m/s，f=1.4–2.6 Hz，c≈0.28 m，半展 0.8 m，k=πfc/U≈0.15–0.45，Re≈1.5e5，Mylar 膜+碳梁，NACA2406-like 弯度）：**
- 仿真升力在**全部 28 个工况都低于**实测。gap_L ≈ −0.52·f − 0.071·f·α − 0.25 [N]（f 单位 Hz，α=平均攻角 deg），rms 0.26 N 解释 99% 方差。
- 纯攻角项外推至 f→0 ≈ **零**（+0.021 N/deg）：**静态/准定常升力线斜率基本正确**。亏空是一个**非定常（拍动锁定）机制**。
- 主导项是 **f·α 交叉项**：亏空的攻角斜率随 f **线性变陡**（f=1.4:−0.087 → f=2.6:−0.183 N/deg）。
- 存在**纯 f 项**：即使 α=0，亏空也随 f 增长（f1.4:−0.95 N → f2.6:−1.21 N，约 35% 相对亏空；翼有弯度）。
- **扭转（feathering，降低外侧有效攻角）缓解亏空**（+0.03..+0.04 N/扭转度）。
- **U 标度（仅 2 点）**：f=1.4 时近 U 无关；f=2.3 时强 U 相关（U6:−0.84 N vs U8:−2.47 N）。

**嫌疑通道：** 每条带按**瞬时**有效攻角（含拍动速度贡献）施加的准定常 CL_max 饱和 `factor = min(1, sin(11°)/|sin(α_eff)|)`——**无滞后、无缩减频率依赖**。求解器已有 Goman–Khrabrov 一阶滞后基础设施（τ = τ*·c/(2·U_rel)，用于 LE 吸力分离态），但**未作用于该失速饱和因子**。

---

## 1. 机理裁定（按证据强度排序）

**裁定：主犯是嫌疑通道本身——准定常静态失速切割缺失"动态失速延迟"（Kramer 效应 / LEV 维持附着）。真实翼在拍动行程内把瞬时攻角带出静态失速角以外仍保持升力，而无滞后的 CL_max 切割把这部分升力错误抹掉，产生一个随 f 增长（延迟带 Δα∝r∝f 变宽）、随平均攻角增长（更长/更深地进入超阈段）、f→0 时消失（r<0.01 退化为准静态）、feathering 卸掉外侧攻角即缓解的亏空——与指纹逐项同构。次犯是膜动态弯度缺失（求解器固定 NACA2406 弯度，缺 Ae∝f⁻² 的拍动诱导弯度增益），是唯一能解释高 f 段 U 相关性的补充通道。**

### M1（主裁定，证据最强，4 条 CONFIRMED 独立律 + 精确指纹命中）：CL_max 饱和缺少动态失速延迟滞后

**指纹逐项命中：**
- **f→0 斜率为零 ⇔ 延迟阈值 r0=0.01。** Sheng–Galbraith–Coton 判据（DS-1，CONFIRMED）：失速起始角延迟 Δα=D1·r 仅在 r=α̇c/(2U)≥r0≈0.01 时激活，r<r0 退化为准静态。RoboEagle 拍动诱导的逐条带 r≈0.05–0.09∝f，f→0 时 r<0.01，延迟带闭合，切割不再误伤——**与纯攻角项 f→0 外推≈0 精确一致**。
- **f·α 交叉项主导 ⇔ 延迟带线性宽化 × 超阈占空比。** 延迟带宽 Δα=D1·r∝f（DS-1 线性律）；平均攻角越高，行程内落在（α_ss, α_ss+Δα）超阈段的比例越大、越深。二者相乘即给出亏空的攻角斜率**随 f 线性变陡**（观测 −0.087→−0.183 N/deg）——这是判别性证据：sqrt(r) 型（Gormont）会给 √f 斜率，而观测是**线性**，直指 Sheng 线性判据 / GK 速率移位。
- **纯 f 项（α=0 仍有亏空，~35%）⇔ 弯度截面在拍动诱导攻角下的动态失速。** 翼有 NACA2406-like 弯度，α=0 时零升攻角非零，拍动诱导 α_eff(t) 仍周期性超阈；GK 速率移位 α_ds=α_ss+τ₂·α̇（DS-4/GK-3，CONFIRMED）在 α=0 基态上仍抬高动态失速角，切割仍误伤 → 纯 f 亏空。
- **扭转缓解（+0.03..0.04 N/deg）⇔ feathering 降低外侧 α_eff 与 α̇。** 扭转把外侧有效攻角拉到失速阈以下、并降低 α̇（降 r），延迟带超阈段缩短，被误伤的升力减少——正是 +0.03..0.04 N/扭转度的符号与量级。

**为何是嫌疑通道而非别处：** 静态斜率已正确（f→0 纯攻角项≈0）直接排除"静态升力线斜率错误"；亏空是幅值损失而非相位错误，且求解器核心是自由尾迹 UVLM（本已解析尾迹/诱导角）——UVLM 核心不删升力（Fritz & Long 2004，uvlm C5，CONFIRMED：纯附着势流，无失速模型，任何 CL_max 帽都是外挂）。故亏空是**失速外挂的伪影**，与 DeLaurier 谱系"瞬时攻角对固定静态失速角比较、无滞后无 k 依赖"（uvlm C1，CONFIRMED）的已知失效模式**结构完全一致**。

**零拟合可实现性（判别性优势）：** 求解器已携带 GK 一阶滞后基础设施（τ=τ*·c/(2·U_rel)）——修正就是把同一滞后/速率移位加到失速饱和的输入角上，**τ 常数全部有文献锚**（见 §3）。

### M2（次选，证据中等，唯一能解释 U 相关性）：膜动态弯度缺失（固定 NACA2406 弯度）

Gehrke & Mulleners（MW-4，CONFIRMED，PNAS 2025）与 Li & Jaiman（MW-5，CONFIRMED，AIAA J 2023）给出闭式弯度律：气弹数 **Ae = E^s·h/(½ρU²c) = E^s·h/(2ρf²φ̂²R2²c) ∝ f⁻²**，归一化力 **C*_n = C_n/(Ae·sin(α̂))**，弯度 δ_max/c = 0.3178·(C_n/Ae)^0.3212 ≈ (We)^(1/3)。
- **纯 f 项兼容：** Ae∝f⁻² → f↑ → Ae↓ → 动态弯度↑ → 升力增益↑ → 固定弯度求解器漏掉的部分随 f 增长。
- **f·α 兼容：** C*_n∝sin(α̂) → 增益随攻角增长。
- **扭转缓解兼容：** feathering 降有效攻角 → 增益缩小。
- **f→0 兼容：** 拍速→0 → Ae→∞ → 无动态弯度 → 与静态斜率正确一致。
- **判别性证据——U 相关性：** Ae∝1/(U²c)，**U↑ → Ae↓ → 动态弯度增益↑ → 固定弯度求解器漏掉更多 → 亏空随 U 增大**。这与观测 f=2.3 时 U8(−2.47 N) > U6(−0.84 N) 同向。而**纯失速延迟机制预测相反**（U↑→k↓→r↓→延迟少→亏空缩），单靠 M1 无法解释高 f 段 U 相关性。故 M2 是必要补充通道。
- Gordnier 2009（MW-2，CONFIRMED）：**静态膜翼相对等效弯度刚翼无显著优势**——增益来自振荡/动态耦合。这直接证明求解器的固定 NACA2406 弯度**恰是"等效弯度刚翼"参考**，残余动态增益必是拍动锁定项，f→0 消失、静态斜率保持——与指纹一致。

### M3（弱，机制支持但非零拟合独立数）：LEV 周期均值增益 / 旋转翼延迟失速

Hubel & Tropea 2010（lev C1，CONFIRMED，鸟类尺度、根部拍动、k 与 Re 双重覆盖）：拍动峰值 Cz 超固定翼最大值 +7%(Re1.33e5)…+56%(Re2.8e4)，且随 Re↓（U↓）增强；Usherwood & Ellington 2002（lev C4，CONFIRMED）：旋转翼 α≈40–45° 仍 CL>1.5 无失速断裂，LEV 供~2/3 下冲程升力；Lentink & Dickinson 2009（lev C5，CONFIRMED）：低 Rossby 数（Ro≈3）离心/Coriolis 稳定 LEV，外侧低 Ro 段维持附着——这些**机制性证明**外侧超静态失速段仍产生附着流量级力，正是准定常 CL_max 切割删掉的物理。但均为定性/峰值百分比，非周期均值 Newton 亏空的零拟合定量律，故仅作机制佐证，不作实现常数。

### M4（排除，见 §4）：静态弯度斜率增陡（f→0 不消失，矛盾）；纯附加质量（∝f²，与 f/f·α 标度矛盾）；条带尾迹亏损（求解器已解析尾迹）；剖面速度阻力（升力通道非阻力）。

---

## 2. 各角度幸存证据 与 机理解释力

### 各角度证据强度小结

| 角度 | 幸存 CONFIRMED | 核心贡献 | 指纹匹配 |
|---|---|---|---|
| **dyn-stall-onset** | DS-1..DS-7（DS-1/2/4/5/6 CONFIRMED，DS-3/7 CONFIRMED with 校正） | 失速延迟律 + 一阶滞后实现 + pitch-plunge 等价 + Kramer 历史锚 | **HIGH**（f→0=0, f·α 线性, 扭转缓解全命中） |
| **goman-khrabrov** | GK-1..GK-8（除 GK-7 PLAUSIBLE 全 CONFIRMED） | 状态方程 + 物理 τ₁/τ₂ + 速度合成 α_eff 先例 + LLT 嵌入先例 | **HIGH**（零拟合 τ，已有基础设施） |
| **lev-cycle-mean** | C1,C2,C4,C5（CONFIRMED），C3 CONFIRMED with 校正，C7 PLAUSIBLE | LEV 增益机制 + Gormont-Berg 闭式 sqrt 律 | **HIGH（机制）/HIGH（Gormont 实现）** |
| **membrane-camber** | MW-1..MW-6 全 CONFIRMED（部分常数带 caveat） | Ae∝f⁻² 动态弯度律 + LEV 抑制 + 静态膜无优势 | **HIGH（唯一解释 U 相关）** |
| **uvlm-stall-practice** | C1,C3,C4,C5,C7（CONFIRMED），C2 CONFIRMED | 嫌疑通道即 DeLaurier 静态帽；GK 是标准修法；UVLM 核心不删升力 | **HIGH（确证嫌疑通道结构）** |

### 三特征解释力对照

| 特征 | M1：失速延迟滞后缺失（GK/Sheng） | M2：膜动态弯度缺失（Ae∝f⁻²） | M3：LEV 周均增益 | M4-排除项 |
|---|---|---|---|---|
| **f·α 交叉项主导（斜率随 f 线性变陡）** | **强**。Δα=D1·r∝f × 超阈占空比∝α → 线性 f·α。Sheng 线性判据精确给线性斜率 | **中**。C*_n∝Ae⁻¹sin(α̂)，f 与 α 均单调，但 Ae∝f⁻² 更接近平方而非纯交叉 | 中。机制对，无闭式 f·α 数 | 附加质量 ∝f²、无 α 交叉 → 排除 |
| **f→0 纯攻角项≈0（静态斜率对）** | **强**。r<r0=0.01 延迟带闭合；GK 速率移位 α̇→0 归零 | **强**。拍速→0→Ae→∞→无动态弯度；静态膜≈等效刚翼（Gordnier） | 强。拍动→0 无 LEV 增益 | 静态弯度增陡在 f→0 仍在 → 排除该子机制 |
| **扭转缓解（+0.03..0.04 N/deg）** | **强**。feathering 降 α_eff 与 α̇→r↓→延迟带超阈段缩 | **强**。降有效攻角→C*_n 增益缩 | 中。降外侧攻角→LEV 增益缩 | 无明确机制 |
| **纯 f 项（α=0，~35%）** | **中-强**。弯度截面零升角非零，拍动诱导 α_eff 仍周期超阈 | **强**。α=0 时膜仍随动压鼓起产生动态弯度 | 中 | — |
| **U 相关性（f=2.3 时 U8≫U6）** | **弱/反向**。U↑→k↓→r↓→延迟少→亏空应缩（与观测反） | **强（唯一正确方向）**。Ae∝1/U²→U↑→弯度增益↑→亏空↑ | 中。Hubel: 增益随 Re↓（U↓）增强——方向反 | — |

**判别结论：** M1 命中 f·α 线性、f→0=0、扭转缓解、纯 f 四项，且是**已识别的嫌疑通道 + 零拟合 + 已有基础设施**，为**首选单变量修正**。M2 是**唯一能解释高 f 段 U 相关性**的补充通道，但需新增 FSI 弯度耦合（非单变量、非现成），列为二阶。M1 修完后若残余亏空仍随 U 增长且集中在高 f，则启用 M2。

---

## 3. 实现配方（单变量零拟合，首改 + 排序备选）

### 首改（PRIMARY）：把 CL_max 饱和的输入角从"瞬时 α_eff"改为"速率移位/滞后后的 α_eff"

**改动本质：** 保留求解器自身的 α_ss=11°（即 sin(11°) 阈），**只改传入 `min(1, sin(11°)/|sin(·)|)` 的角度**——从瞬时 α_eff 改为动态失速延迟后的角度。单变量、替换（非叠加）、复用既有 GK 滞后代码。

**实现变体 A1（首选，GK 速率移位，Ayancik–Mulleners 物理常数，零拟合）：**
- 动态失速角移位：**α_ds = α_ss + τ₂·α̇_eff**（GK-3 / DS-4 eq 3.1，CONFIRMED）。等价地，把饱和因子对 (α_eff − τ₂·α̇_eff) 求值。
- τ₂ 由普适失速延迟律（eq 3.5，R²=0.978，Re 7.5e4–1e6 覆盖我们 1.5e5）：
  **Δt_ds·U∞/c = 0.0815·(α̇_ss·c/(2U∞))^(−7/9) + 4.24**
  正弦运动映射（eq 3.4）：**τ₂ = (2α₁/α̇_ss)·sin(πf·Δt_ds)·cos(πf·Δt_ds)**。
- 常数：**0.0815，指数 −7/9，加项 4.24**（全部文献值，airfoil/Re/ramp-vs-sinusoidal 无关）。
- 出处：Ayancik & Mulleners 2022，JFM **942:R8**（非 934:A33），DOI 10.1017/jfm.2022.381（arXiv:2110.08516），eqs 3.1/3.4/3.5/3.6。

**实现变体 A2（等价、最省事，直接复用求解器 τ 基础设施）：**
- 对 α_eff 施加一阶低通滞后后再切割：**τ₁ = 4.24·c/U_rel = 8.48 半弦**（GK-2 eq 3.6，St=0.235，CONFIRMED），即在求解器 τ=τ*·c/(2U_rel) 约定下 **τ* = 8.48**。仅一个常数、直接插入现成 GK 滞后层。
- 物理自洽性检查：RoboEagle 半冲程时长 π/(2k)≈3.5–10 c/U，与 τ₁=4.24 c/U 同量级 → 分离/切割在冲程内**来不及达到准定常深度** → 定量支持 35% 相对亏空随 f 增长（GK-2 fingerprint）。

**为何 A 优于备选：** (1) 保留 α_ss=11° 不动，只加速率移位，是最小改动；(2) 常数 airfoil 无关，绕开"NACA2406 无直接失速数据"问题；(3) 复用已验证的 GK 滞后代码；(4) 给**线性** f 斜率（与观测 −0.087→−0.183 线性变陡一致），优于 Gormont 的 √f。

### 备选（FALLBACK，按优先级）

**F1（Gormont–Berg 闭式 sqrt 律，结构最贴嫌疑通道，弯度截面有直接常数）：**
- **α_ref,L = α_eff − K1·Y_L·√(|T_u·α̇|)·sign(α̇)**，T_u = c/(2·V_rel)；在 α_ref,L 上做静态 CL_max 切割/查表（lev C2，CONFIRMED）。
- 常数：**K1 = 1.0（pitch-up）/ 0.5（pitch-down）；Y_L = 1.4 − 6.0·(0.06 − t/c)，t/c=0.06（NACA2406-like）时 Y_L = 1.4**（Y_M=1.0）。
- 出处：Gormont 1973 USAAMRDL TR 72-67（DTIC AD0767240）+ Berg 校正；常数经 QBlade Gormont-Berg 文档、OpenFAST/AeroDyn Boeing-Vertol、Energies 2015 三路核实。
- 取舍：闭式、发布常数、**直接就是"移位角查表"**（与嫌疑通道结构同型，改动最直观）；但延迟 ∝√r → √f 斜率，与观测线性 f 斜率略逊，故列 F1 而非首选。t/c 说明：以气动弦厚比 0.06 锚定（Y_L 分段的低 Mach γ_max 值），Ø8mm 圆杆前缘的钝度未计入。

**F2（An–Williams 常数，最贴 RoboEagle 运行包络）：** τ₁=3.75、τ₂=4.375 c/U（NACA-0009，Re=4.9e4，K=πfc/U 验证至 0.51，相关系数 0.956）。GK-4 CONFIRMED。用作 A2 的 τ 交叉校核——独立证实两个 GK 滞后均≈4 c/U。

**F3（Rezapour–Mulleners，最贴 Re）：** τ₁=3.57 c/U，(t_ds−t_ss)U∞/c = 0.06·(α̇c/2U∞)^(−0.77)+3.57（NACA0018，Re=6e4，最接近我们 1.5e5 以下）。GK-5/DS-5 CONFIRMED。用作 A1 常数的下包络（bracket，非替换）。

**F4（二阶补充通道，仅当 M1 修完残余仍随 U 增长时启用）：** 膜动态弯度。乘 C*_n = C_n/(Ae·sin(α̂)) 增益或按 δ_max/c = 0.3178·(C_n/Ae)^0.3212 引入拍动诱导弯度增量（MW-4/MW-5 CONFIRMED）。Ae = E^s·h/(½ρU²c)，输入为实测预张力/膜厚，零拟合但需新增 FSI 耦合——非单变量，故排最后。

**硬警告（防重复计权）：** GK/Sheng 自述"**不适用平板前缘失速**"（DS-4/GK-1，原文 CONFIRMED）。膜锐缘截面上，失速延迟滞后**只许作用于附着环流部分**；LEV 载荷必须留在既有 LESP 门控模块（见 research_les_suction.md 的吸力通道修正），两通道边界即 LESP 阈值。修 M1 时严禁把滞后再叠到 LESP 通道上。

---

## 4. 常数来源：文献锚定 vs 外推

**文献锚定（可零拟合直接用）：**
- Ayancik–Mulleners 普适延迟律 **0.0815 / −7/9 / 4.24**（R²=0.978，Re 7.5e4–1e6，ramp+正弦）——JFM 942:R8。**Re 覆盖我们 1.5e5。**
- 物理 τ₁ = 4.24 c/U（St=0.235）；OA209 最佳拟合 τ₁=0.026(9) s vs 预测 0.0254 s。
- Sheng–Galbraith–Coton 阈值 **r0=0.01**、线性判据 Δα=D1·r、per-airfoil (α_ds0, T_alpha) 表（NACA23012 族 α_ds0≈17.2–18.1°、T_alpha≈4.0–6.1 半弦，最贴薄弯度截面）——JSEE 128(4):461-471。
- An–Williams τ₁=3.75、τ₂=4.375 c/U（Re=4.9e4，K≤0.51）；Rezapour τ₁=3.57 c/U（Re=6e4）——GK-4/GK-5。
- Gormont–Berg K1=1.0/0.5、Y_L=1.4@(t/c=0.06)——TR 72-67 + QBlade/OpenFAST 三路核实。
- 膜律 δ_max/c=0.3178·(We)^0.3212、Ae 定义、C*_n=C_n/(Ae·sinα̂)、CL_flex=2.43@(Ae=1.70, α̂=55°)、k=0.42、Re=2800–9300——MW-4/MW-5 逐字核实。

**外推 / 存疑（须披露，不可当锚）：**
1. **r≈0.05–0.09 略超 Sheng 测试域**（r≤~0.05）。DS-1 fingerprint 自陈此外推；用 An–Williams（K≤0.51）与 Rezapour（r≤0.04）交叉包络缓解。
2. **NACA2406 无直接失速数据**：DS-2 的 α_ds0/T_alpha 用 NACA23012 族代理（薄弯度最近邻），非该翼型实测。变体 A1/A2 保留求解器自身 α_ss=11° 只加速率移位，绕开此外推——**这是选 A 而非需要 per-airfoil α_ds0 的关键**。
3. **GK/Sheng 明文不适用前缘失速**：膜锐缘上仅作用于附着环流部分，属外推；LEV 段边界须由 LESP 阈值划定。
4. **Gormont Y_L 是低 Mach γ_max 值**，圆杆钝前缘厚度效应未计入（lev C2 caveat）。
5. **pitch→plunge 等价**：RoboEagle 拍动是 plunge 主导，DS-1/DS-4 的 pitch-rate 律经 Miotto & Wolf 2023（DS-6，CONFIRMED，SD7003 Re=6e4）与 Sedky 2020（GK-7，PLAUSIBLE，速度合成 α_eff 先例）证明可由 flap-induced α_eff 速率驱动——**但 DS-6 Re 仅在我们 ~2.5× 内，GK-7 仅 PLAUSIBLE**，属受支持外推。
6. **膜"~20% 升力增益"数值未证**（MW-4 verdict：press release 仅定性）；C*_n / Ae 公式与 CL=2.43 已证，但 20% 不可引。
7. **U 相关性仅 2 点**（f=2.3），判别杠杆弱；M2 的 U↑→亏空↑ 方向对但样本不足，须补点确认。

**一句话合规性：** 首改 A1/A2 的全部常数（0.0815/−7/9/4.24 或 τ₁=4.24c/U）对**用户数据零拟合**、airfoil/Re 无关、Re 覆盖 1.5e5；唯一外推是"pitch-rate 律驱动 plunge-induced α_eff"（DS-6 支持）与"作用于膜锐缘附着部分"（须以 LESP 阈值门控）。

---

## 5. nc4 / nc12 门禁验证判据

**改完 A1/A2 后必须检查，破门则回退：**

**必须改善（nc12 = 高 f·α / 高扭转激进带，亏空最大处）：**
1. **f·α 交叉项系数从 −0.071 N/(Hz·deg) 塌向 ≈0**（|残余| ≲ 0.02），且亏空攻角斜率不再随 f 线性变陡（f=1.4 与 f=2.6 斜率差 <10%）。
2. **纯 f 项 −0.52 N/Hz 显著收缩**（目标 |残余系数| ≲ 0.15 N/Hz）；α=0 的 ~35% 相对亏空降到 <15%。
3. **总 rms 亏空从 0.26 N 降到 <0.12 N**（解释度从"负偏"转为无偏残差）。
4. **扭转缓解率符号保持、量级收敛**：修后模型自身应内生 +0.03..0.04 N/扭转度的部分，扭转项残差缩小。

**必须不回退（nc4 = 温和/低 f·α 带 + 已达标战果）：**
5. **f→0 / 低 k 附着流区不动**：纯攻角项 f→0 外推须保持 ≈+0.02 N/deg（静态升力线斜率不得被滞后误伤）。r<r0=0.01 时延迟带须严格闭合——代码须验证 r<0.01 分支下 factor 与原瞬时切割一致。
6. **tw22.5° H16 巡航升力战果不破**：升力通道对失速因子的敏感度须复跑确认 0.95× 达标保持（sibling research_les_suction §3 同款回归门）。
7. **tw0 推力平衡不破**（与 research_bern_twist 门禁一致）：**tw0 推力 MAE ≤ 0.45 N**——失速延迟只许改升力/环流通道，不得扰动已验证的推力平衡；若 A1 破此门，退变体 A2 或改为只对扭转扰动敏感度线性化（防 tw0 by-construction 零变化）。
8. **防重复计权**：LESP/LEV 通道诊断量（吸力平台、涡冲量相位）不得因加滞后而二次改变；nc4/nc12 的 LESP 超临界占空比须与修前一致。

**分诊判据（若首改未完全塌缩）：**
- 若 f·α 项**只降一半** → M1 常数域外推（r>0.05）低估延迟，切到 F2/F3 下包络 τ 重跑。
- 若残余亏空**仍随 U 增长且集中在高 f（nc12 的 U8 点）** → M2 膜动态弯度未计，启用 F4（Ae∝f⁻² 增益），叠加时弯度增益只乘一次。
- 若残余**变成 ∝f²** → 转查附加质量；若出现 **k 依赖相位错** → 转查尾迹分辨率。
- 若残余集中在 **LEV 形成段相位** → 转查 LESP 模块（非本通道），见 research_les_suction。

---

## 附：全部幸存题录（逐条对抗核实）

### 角度 dyn-stall-onset

- [DS-1] **CONFIRMED** 低 Mach 俯仰翼动态失速起始角随缩减俯仰速率线性延迟 Δα=D1·r（r=α̇c/2U≥r0≈0.01），r<r0 准静态；13 翼型 ramp + 正弦 k=0.075/0.124 验证。α_ds=D1·r+α_ds0；per-airfoil (α_ds0, T_alpha)：NACA0012 18.73/3.90 … GUVA10 15.82/5.70；D1=T_alpha·180/π。
  来源: Sheng, Galbraith, Coton (2006), J. Solar Energy Eng. 128(4):461-471, DOI 10.1115/1.2346703
  URL: https://eprints.gla.ac.uk/3381/1/Stall-Onset_Criterion_Speed_Dynamic-Stall.pdf
- [DS-2] **CONFIRMED** 同判据可实现为攻角一阶滞后（与求解器现有 GK 滞后同型）：Δα'(s)=Δα(s)(1−exp(−s/T_alpha))，s=2Ut/c，滞后角超 α_ds0 时判失速；τ=T_alpha·c/(2U_rel)。T_alpha=3.90–6.95 半弦；NACA23012 族 α_ds0≈17.2–18.1°、T_alpha≈4.0–6.1（最贴 NACA2406）。
  来源: Sheng, Galbraith, Coton (2006), 同上，eq 5
  URL: https://eprints.gla.ac.uk/3381/1/Stall-Onset_Criterion_Speed_Dynamic-Stall.pdf
- [DS-3] **CONFIRMED（校正）** k=0.09→0.27 动态失速角 ~16°→~30° 单调增。校正：16° 是 k=0.09 的**动态**失速角非静态；Re=4.5e3 水槽，k 上界 0.27。
  来源: Yu, Leu, Miau (2017, online 2016), J. Visualization 20(1):31-44, DOI 10.1007/s12650-016-0366-6
  URL: https://link.springer.com/article/10.1007/s12650-016-0366-6
- [DS-4] **CONFIRMED** 失速延迟普适幂律 Δt_ds·U/c=0.0815·r^(−7/9)+4.24（R²=0.978），跨 NACA0015/0018/OA209、ramp+正弦、Re 7.5e4–1e6；τ₁=4.24c/U（St=0.235），α_ds=α_ss+τ₂·α̇_ss。
  来源: Ayancik & Mulleners (2022), J. Fluid Mech. 942:R8, DOI 10.1017/jfm.2022.381 (arXiv:2110.08516)
  URL: https://arxiv.org/abs/2110.08516
- [DS-5] **CONFIRMED** 独立 EPFL 实验 tripped NACA0018 Re=6e4（静失速 13.3°）复现同型幂律 Δt_ds·U/c=0.06·r^(−0.77)+3.57（r=1e-4..0.04），延迟由俯仰速率主导、加速度为次要。τ₁=3.57c/U。
  来源: Rezapour & Mulleners (2025/2026), arXiv:2508.10647, AIAA J DOI 10.2514/1.J066234
  URL: https://arxiv.org/abs/2508.10647
- [DS-6] **CONFIRMED** Pitch-plunge 等价穿越动态失速起始：SD7003 LES Re=6e4，以准定常薄翼几何有效攻角匹配的 plunge 与 pitch 在 DSV 起始/输运期流拓扑、载荷、壁压/摩擦高度相似——pitch 律可由 flap(plunge) 速度诱导 α_eff 驱动。
  来源: Miotto, Wolf, Gaitonde, Visbal (2023), AIAA Journal 61(1):174-188, DOI 10.2514/1.J061507
  URL: https://arc.aiaa.org/doi/10.2514/1.J061507
- [DS-7] **CONFIRMED** Kramer 效应（1932）：突增有效攻角时 CL_max 随攻角增率成正比增加——线性-in-rate 失速延迟历史锚；数值常数 γ 本轮不可验，故用 DS-1/DS-4 常数替代。
  来源: Kramer, M. (1932), ZFM 23:185-189; 英译 NACA TM-678
  URL: https://ntrs.nasa.gov/citations/19930094738

### 角度 goman-khrabrov

- [GK-1] **CONFIRMED** GK(1994) 一阶 ODE：τ₁dX/dt+X=X0(α−τ₂α̇)，X∈[0,1]，Kirchhoff Cl=(dCl/dα)|0·sinα·((1+√X)/2)²；明文仅适用尾缘渐进分离，**不适用平板前缘失速**。嫌疑通道正是其 τ₁=τ₂=0 的未滞后 X0 分支。
  来源: Goman & Khrabrov (1994), J. Aircraft 31(5):1109-1115, DOI 10.2514/3.46618（方程经 Ayancik & Mulleners 2022 复述核实）
  URL: https://arxiv.org/abs/2110.08516
- [GK-2] **CONFIRMED** 物理 τ₁=4.24 c/U∞（eq 3.6，St=0.235，结论取 0.25）；OA209 最佳拟合 τ₁=0.026(9)s vs 预测 0.0254s，无运动非定常性依赖。半冲程 π/(2k)≈3.5–10 c/U 与 τ₁ 同量级 → 分离在冲程内达不到准定常深度。
  来源: Ayancik & Mulleners (2022), JFM 942:R8, DOI 10.1017/jfm.2022.381
  URL: https://arxiv.org/abs/2110.08516
- [GK-3] **CONFIRMED** τ₂ 零拟合普适律（eq 3.1/3.3/3.4/3.5），Re 7.5e4–1e6 独立、ramp+正弦；正弦 τ₂=(2α₁/α̇_ss)sin(πfΔt_ds)cos(πfΔt_ds)。RoboEagle Re~1.5e5 在验证带内，τ₂α̇ 移位 ∝ 速率 → 保留升力 ∝ f·α 超阈，命中主导交叉项。
  来源: Ayancik & Mulleners (2022), JFM 942:R8（数据用 Le Fouest et al. 2021, JFS 104:103304）
  URL: https://arxiv.org/abs/2110.08516
- [GK-4] **CONFIRMED** NACA-0009（c=245mm, U=3m/s, Re=4.9e4, pivot 0.15c）τ₁=3.75、τ₂=4.375 c/U；正弦 K=0.05–0.128 + quasi-random K≤0.51 验证（相关 0.956）。**最贴 RoboEagle 运行包络**，独立佐证两 GK 滞后≈4 c/U。
  来源: An, Williams, Eldredge, Colonius (2021), Exp. Fluids 62:11, DOI 10.1007/s00348-020-03105-3 (arXiv:2005.01870)
  URL: https://arxiv.org/abs/2005.01870
- [GK-5] **CONFIRMED** NACA0018 Re=6e4 τ₁=3.57 c/U + 幂律 0.06/−0.77/3.57；标记一阶模型对非单调/不规则运动的退化——正弦/扭转分布 α(t) 须配 GK-3 正弦 τ₂ 映射用。
  来源: Rezapour & Mulleners (2026), AIAA J DOI 10.2514/1.J066234 (arXiv:2508.10647)
  URL: https://arxiv.org/abs/2508.10647
- [GK-6] **CONFIRMED** Williams et al. 实践中 τ₁/τ₂ 每翼型手调一次后跨全频率固定；Ayancik & Mulleners 指此经验选取为 GK 主要缺陷。支持零拟合策略：一次取文献 τ，频率依赖由 ODE 自身涌现。厚弯度/静态迟滞截面可能需两支 x0 扩展。
  来源: Williams, Reissner, Greenblatt, Mueller-Vahl, Strangfeld (2017), AIAA J 55(2):403-409, DOI 10.2514/1.J054937
  URL: https://arc.aiaa.org/doi/10.2514/1.J054937
- [GK-7] **PLAUSIBLE** Sedky et al. 用横向阵风的**速度诱导有效攻角**驱动改进 GK 状态方程（与拍翼 flap-velocity α_eff 同构），UMD 水拖曳池标定并做升力调节——直接先例：GK 滞后可作用于速度合成 α_eff。arctan 精确形式与 pre/post-stall 调节未证（付费墙）。
  来源: Sedky, Jones, Lagor (2020), AIAA Journal 58(9):3788-3798, DOI 10.2514/1.J059127
  URL: (Crossref/OpenAlex abstract)
- [GK-8] **CONFIRMED** 2026 J. Aircraft 把 GK 嵌入经典升力线（每段分离态 + Wagner/Kuessner 滞后态 + plunge/pitch 结构 DOF），对 DLM/NASTRAN/CFD 验证 GK-in-LLT 改善近失速升力**幅值与相位**——正是"给每条带/panel 附 GK 分离态调制 sectional 升力"的发布先例。
  来源: AbuNawas & Qawasmeh (2026), Journal of Aircraft, DOI 10.2514/1.C038542
  URL: (Crossref)

### 角度 lev-cycle-mean

- [C1] **CONFIRMED** 鸟类尺度根部拍动翼维持 LEV，周期峰值 Cz 超固定翼 +7%(Re1.33e5)…+56%(Re2.8e4)，k≥0.07 最显著、随 Re↓ 增强。k=0.05–0.3、Re 2.8e4–1.33e5 双重覆盖我们 28 工况。
  来源: Hubel & Tropea (2010), J. Exp. Biol. 213(11):1930-1939, DOI 10.1242/jeb.040857
  URL: https://journals.biologists.com/jeb/article/213/11/1930
- [C2] **CONFIRMED** Gormont/Boeing-Vertol(-Berg)：α_ref,L=α−K1·Y_L·√(|T_u·α̇|)·sign(α̇)，T_u=c/(2V_rel)，在 α_ref,L 查静态系数；延迟 ∝√r。K1=1.0/0.5；Y_L=1.4−6.0(0.06−t/c)，t/c=0.06 时 Y_L=1.4，Y_M=1.0。
  来源: Gormont (1973) USAAMRDL TR 72-67 (DTIC AD0767240) + Berg 校正；常数据 QBlade Gormont-Berg / OpenFAST / Energies 2015 三路核实
  URL: https://docs.qblade.org/src/theory/aerodynamics/dynamic_stall/GOR_stall.html
- [C3] **CONFIRMED（校正）** 俯仰翼 k=0.09→0.27 动态失速角 ~16°→~30° 单调延迟。校正：16° 为 k=0.09 动态值非静态；Re=4.5e3。
  来源: Yu, Leu, Miau (2017), J. Visualization 20(1):31-44, DOI 10.1007/s12650-016-0366-6
  URL: https://link.springer.com/article/10.1007/s12650-016-0366-6
- [C4] **CONFIRMED** 旋转（根部拍动、不反向）翼延迟失速、附着 LEV，α≈40–45° 时 CL>1.5 无失速断裂（CL_max=1.75@41°）；LEV 供~2/3 hawkmoth 下冲程升力。Re 1100–26000。
  来源: Usherwood & Ellington (2002), J. Exp. Biol. 205(11):1547-1564 & 1565-1576；Ellington et al. (1996), Nature 384:626-630
  URL: https://journals.biologists.com/jeb/article/205/11/1565
- [C5] **CONFIRMED** LEV 由离心/Coriolis 加速度在低 Rossby 数稳定，Ro≈3 为跨昆虫/种子/鸟收敛高升力解（离心/Coriolis∝1/Ro）；外侧低 Ro 段维持附着——机制上支持外侧超静态失速仍产力，且合理化扭转缓解。
  来源: Lentink & Dickinson (2009), J. Exp. Biol. 212(16):2705-2719, DOI 10.1242/jeb.022269
  URL: https://journals.biologists.com/jeb/article/212/16/2705
- [C7] **PLAUSIBLE** 准定常/静态失速力限制在扑翼/俯仰翼上低估非定常升力（Kramer/LEV），超量随 k 增长——单调-in-f、f→0 消失，与本案指纹一致；发布零拟合量化式为 GK/Ayancik-Mulleners（C3）。Gormont γ 常数本轮未数值核实。校正：Ol et al. 2009 而非 Amiralaei 为 "shallow/deep dynamic stall" 一文作者。
  来源: Ol, Bernal, Kang, Shyy (2009), Exp. Fluids 46:883-901; Ayancik & Mulleners (2022) arXiv:2110.08516; Gormont TR 72-67
  URL: https://arxiv.org/abs/2110.08516

### 角度 membrane-camber

- [MW-1] **CONFIRMED** 柔性膜翼相对等效刚翼延迟失速最多 ~10°、抬升 CL_max（Song 2008），增益含 (a) 自适应弯度增陡斜率 + (b) 延迟失速；Ae=E^s·h/(½ρU²c)，动压升 Ae 降弯度增。**仅 (b) 子机制与我们正确静态斜率兼容**（(a) 在 f→0 仍在，被排除）。
  来源: Song, Tian, Israeli, Galvao, Bishop, Swartz, Breuer (2008), AIAA J 46(8):2096-2106, DOI 10.2514/1.36694
  URL: https://arc.aiaa.org/doi/10.2514/1.36694
- [MW-2] **CONFIRMED** 静态膜翼相对等效**弯度刚翼无显著气动优势**；升力增益/失速延迟来自膜振荡与均值弯度的耦合（非定常机制）。直接支持：求解器固定 NACA2406 弯度=等效弯度刚翼参考，残余亏空必是振荡/动态项，f→0 消失、静态斜率保持。
  来源: Gordnier (2009), J. Fluids & Structures 25(5):897-917, DOI 10.1016/j.jfluidstructs.2009.03.004
  URL: (ScienceDirect; 经 Tiomkin & Jaworski arXiv:2204.01204 复述核实)
- [MW-3] **CONFIRMED** 拍动膜翼诱导弯度使 LEV 沿弯曲上表面附着滑移（覆盖大部弦长、维持高力），刚翼 LEV 脱落对流走致力降；低预张力（弯度~0.25c）推力增 ~40%。
  来源: Gopalakrishnan & Tafti (2010), AIAA J 48(5):865-877, DOI 10.2514/1.39957
  URL: (AIAA)
- [MW-4] **CONFIRMED** 高变形拍动膜翼在悬停抑制 LEV 而升力更高，Ae=E^s·h/(½ρU²c)=E^s·h/(2ρf²φ̂²R2²c) ∝ f⁻²，C*_n=C_n/(Ae·sinα̂)；CL_flex=2.43@(Ae=1.70, α̂=55°)，k=0.42，Re=2800–9300。**Ae∝f⁻²→f↑增益↑；∝1/U²→U↑增益↑（唯一解释我们 U 相关性）**。注：~20% 增益数未证。
  来源: Gehrke & Mulleners (2025), PNAS 122:e2410833121 (arXiv:2410.01670); Tiomkin & Gehrke (2026), JFM 1026:R2 (arXiv:2509.00666)
  URL: https://arxiv.org/abs/2410.01670
- [MW-5] **CONFIRMED** 膜鼓起弯度零拟合幂律 δ_max/c=0.3178·(C_n/Ae)^0.3212≈(We)^(1/3)（Waldman 1/3 律），气动增益为延迟深失速/高峰值法向力；深失速角 40°→45°（柔 vs 刚）。校正：拟合为 Li & Jaiman，AoA 域 4°–90°；"非低攻角斜率"子句原文未陈。
  来源: Li & Jaiman (2023), AIAA Journal 61, DOI 10.2514/1.J063004 (arXiv:2212.12112)；引 Waldman & Breuer (2017) JFS 68:390-402
  URL: https://arxiv.org/abs/2212.12112
- [MW-6] **CONFIRMED** 膜振动锁定到尾迹涡脱频率、经非定常耦合抬升均值升力并延迟失速（后失速段二阶模主导）；St=fc/U 表征锁定，理论共振 k_n=nπ√(C_T/8μ)。我们 k~0.15–0.45 远低于首阶流固共振（k1~1.8 @ C_T=2.5）→ 线性附着鼓起单独欠预测，LEV/失速延迟耦合为操作通道。
  来源: Rojratsirikul, Wang, Gursul (2009), Exp. Fluids 46:859-872; Rojratsirikul et al. (2011), JFS 27(8):1296-1309; Tiomkin & Jaworski (2022) arXiv:2204.01204
  URL: https://arxiv.org/abs/2204.01204

### 角度 uvlm-stall-practice

- [C1-uvlm] **CONFIRMED** 经典扑翼条带码（DeLaurier 1993 + FullWing）用**瞬时**（含俯仰速率项）有效攻角对固定静态失速角比较、**无滞后无 k 依赖**——正是本案嫌疑通道。(α_stall)max=±20°（Eq.25），后失速 CD=1.98（Hoerner）。差别只在帽角（20° vs 本案 11°），失效模式相同。
  来源: DeLaurier (1993), Aeronautical Journal 97(964):125-130; Djojodihardjo (2018) IJAAE 3:017 Eqs.25-33; Mau (2003) UTIAS
  URL: https://vibgyorpublishers.org/content/ijaae/ijaae-3-017.pdf
- [C2-uvlm] **CONFIRMED** Kim et al.(2011) 显式把修正条带理论扩展加入 plunge+pitch 组合的真动态失速判据，据 Scherer 振荡翼数据标定（瞬时 CL_max 可达静态 ~2×）——社区直接证据：纯静态帽对大幅拍动不足。
  来源: Kim, Lee, Han (2011), AIAA Journal 49(4):868-872, DOI 10.2514/1.J050556
  URL: (Semantic Scholar / SCIRP)
- [C3-uvlm] **CONFIRMED** 文献标准物理修法是 GK 一阶状态模型，动态失速角随瞬时俯仰/攻角率抬高 α_ds=α_ss+τ₂α̇；Ayancik & Mulleners 给零拟合 airfoil 无关闭式（0.0815/−7/9/4.24）——可直插求解器现有 GK 滞后（τ₁=4.24c/U→τ*≈8.5 半弦）。GK 不适用平板前缘失速。
  来源: Goman & Khrabrov (1994) J. Aircraft 31(5):1109-1115; Ayancik & Mulleners (2022) JFM 942:R8
  URL: https://arxiv.org/abs/2110.08516
- [C4-uvlm] **CONFIRMED** 业界 UVLM 气弹码（SHARPy 谱系；Murua & Palacios）为无粘势流，失速仅作**静态 2D 剖面极曲线**逐条带 α 迭代黏性修正——捕捉静态分离但**无 k/动态失速滞后**，与本案帽同型结构缺陷。
  来源: Ritter & Hilger (2022), AIAA 2022-0177, DOI 10.2514/6.2022-0177; Murua, Palacios, Graham (2012), Prog. Aerosp. Sci. 55:46-72
  URL: (DLR elib / AIAA)
- [C5-uvlm] **CONFIRMED** 扑翼 UVLM 基线（Fritz & Long 2004）为纯附着势流、**无任何失速模型**——任何 CL_max 帽都是外挂，过激静态封顶（非 UVLM 核心）是系统性升力亏空的合理来源，与近正确准定常斜率一致。
  来源: Fritz & Long (2004), Journal of Aircraft 41(6):1275-1290, DOI 10.2514/1.7357
  URL: (Penn State / AIAA)
- [C7-uvlm] **CONFIRMED** 准定常/静态失速力限制在扑翼/俯仰翼上系统低估非定常升力（动态失速延迟/Kramer/LEV 让截面带升力到远超静失速角），超量随 k 增长——本案亏空单调-in-f、f→0 消失的确切行为。发布零拟合量化式=GK(C3)；Gormont 为经典 √ 替代但 γ 未数值核实。
  来源: Ol, Bernal, Kang, Shyy (2009) Exp. Fluids 46:883-901; Ayancik & Mulleners (2022) arXiv:2110.08516; Gormont (1973) TR 72-67
  URL: https://arxiv.org/abs/2110.08516

---

## 附:墓园(REFUTED,勿再起诉)

- [per-strip-LB-governs-error] **REFUTED** 声称一个耦合逐条带简化 Leishman-Beddoes 的扑翼 UVLM 发现纯拍动中甚至展中也大面积分离、且动态失速载荷估计相对无粘(Katz/Joukowski)显著改变力——即剖面失速处理而非尾迹主导扑动区力误差。
  裁定理由:引文真实且外围事实成立(三载荷法逐条带、L-B 在纯拍动零/负静态俯仰下给更好阻力幅值但引入恒定负阻力偏置),但**中心断言与原文相反**:耦合 L-B **未**发现展中大面积分离——原文明言"Leishman-Beddoes technique does not identify any significant flow separation",有效攻角因下洗停在 −9°..1°、分离点从不前移过 0.87c;"展中大面积分离"是从 ref[14](Razak & Dimitriadis 2014)**实验流场可视化**引入的观测,非模型输出。"剖面失速而非尾迹主导力误差"非原文结论(原文对负俯仰问题"不清楚原因…可能是失速",纯推测)。且该简化 L-B **无** Wagner/indicial 附着子模型、**无** LEV/DSV 子模型、**无** Tp/Tf/Tv 时间常数("time delays were not applied in this work",NACA6409 无值故**刻意省略**);实际所用为 Kirchhoff 静态拟合参数 α₁=10.31°、S₁=0.02、S₂=0.043、cn0=0.5709、LE 吸力效率 η~0.95-0.97。
  来源: Lambert, Abdul Razak, Dimitriadis (2017), 'Vortex Lattice Simulations of Attached and Separated Flows around Flapping Wings', Aerospace 4(2):22, DOI 10.3390/aerospace4020022
  URL: https://doi.org/10.3390/aerospace4020022
