# 文献研究综合报告:gap_T = c(U) + 0.40·f² 残差律的机理裁定

## 1. 机理裁定(按证据强度排序)

**裁定:组合机理,主犯是"超临界期间吸力保持在临界值"的 LDVM 假设,从犯是"亚临界期间 eta_s=1 全实现"假设。"逐段瞬时速度剖面阻力"不被本轮文献支持为 f² 项来源。**

**主犯(证据强度:最高,4 条独立 CONFIRMED 证据链):LESP 超临界期间吸力并未保持在临界值——真实黏性吸力在前缘分离时跌至近零。**
- 我们的模型继承了 Ramesh et al. 2014 (JFM 751:500-538) 的处理:超临界时 A0 被钉在 ±LESP_crit,吸力通道以 c_s = 2π·A0² 全程激活,平台值 2π·LESP_crit²(UNSflow 源码 postprocess.jl 逐行核实,Faure et al. 2019 AMM 69:32-46 独立复现)。
- 但该处理的物理依据被 Hirato (NCSU 2016 博士论文) 明确标注为**未经证明的假设**,可溯源至 Katz 1981 的"proposed flow behavior"——不是黏性证明。
- 黏性真相:Narsipur, Hosangadi, Gopalarathnam, Edwards 2020 (JFM 900, A25) 用黏性 CFD+实验(Re 1e4–1e5,正覆盖我们的 Re_c≈1.2e5)证明:**"viscous LESP 在前缘分离时跌至近零值,与 LEV 是否形成无关"**,机理是前缘流线曲率丧失。这是 Gopalarathnam 团队(LDVM 原作者组)自己的修正结论。
- 独立实验佐证:Deparday et al. 2022 (JFM 941, A60) PIV 直接测量,吸力与前缘剪切层高度负相关——分离越多吸力越弱。
- 宏观后果佐证:Mackowski & Williamson 2015 (JFM 765:524-543) 直接测力证明 Garrick 线性理论在大频率/振幅下**显著高估平均推力**,LEV 削减推力且线性理论不捕捉。

**从犯(证据强度:高,闭式方法完整):附着(亚临界)阶段 eta_s=1 也偏高,实现率应由几何+Re 决定。**
- DeLaurier 1993 扑翼模型谱系(根在 DeLaurier 1983, J. Aircraft 20(10):882-886 "partial leading-edge suction")一贯用 eta_s<1 计入黏性削减——即"全实现"从来不是扑翼低阶模型的默认。
- 零拟合的定量方法存在:Carlson, Mack & Barger 1979 (NASA TP-1500) 的 attainable thrust 因子 K_T = c_t*/c_t,闭式地由厚度比 t/c、前缘半径比 r/c、法向 Mach、弦长 Re、理论吸力水平算出,机制是用极限压力 C_p,lim(随 Re 单调,Re→∞ 趋真空极限,Re→0 趋零)截断吸力峰。

**加重情节:a0_crit=0.27 高于全部文献记录值。** Ramesh 2014 验证集(经 Faure Table 1 与 Gelado & Ramesh 2022 双重核实):SD7003 在 Re=1e5 时 LESP_crit=**0.14**,且随 Re 上升而降(0.21@1e4→0.14@1e5);平板 0.11、NACA0015 0.19。我们取 0.27 意味着模型把过多的周期时间判为"亚临界全吸力",进一步放大虚构推力。(缓和因素:Ramanathan & Gopalarathnam 2023 证明临界值随前缘变圆而增大,我们的 Ø8mm 圆杆前缘确实钝;但 Kay et al. 2022 同时证明临界值强烈依赖俯仰速率与 Re,"单一常数 0.27"无文献锚。)

## 2. 各机理对三个特征的解释力

| 特征 | 机理A:超临界吸力摧毁(近零化) | 机理B:亚临界 eta_s<1 (Carlson K_T) | 机理C:剖面速度阻力 |
|---|---|---|---|
| **b≈0.40 N/Hz²,U 无关** | **强**。Garrick 纯挥拍吸力 ∝ ½ρU²c·2πA0²,A0∝ḣ/U∝f/U,故吸力∝ρc(fh)²——f²、U 无关,与残差律精确同构。挥拍主导段恰是 LESP 高、分离段:模型在此处保留了(实际上近零的)吸力,虚构量继承 Garrick 标度 | **中**。同一 f² 标度,但 K_T 只削减一部分而非归零;单靠它解释不了满额 0.40 | **弱**。剖面阻力 ∝ 合速度²,含 U²、Uf 交叉项,必然 U 相关,与"f² 系数 U 无关(0.409 vs 0.403)"矛盾 |
| **c(U) 随 U 下降** | **中**。U↑→k=πfc/U↓→有效攻角与超临界占空比降→虚构窗口缩小;且 LESP_crit 随 Re 升(Kay 2022),分离更少 | **中**。C_p,lim 随 Re 上升→K_T 升→附着段虚构份额降 | c(U) 里可能确有真实剖面阻力成分(圆杆+膜的型阻不在无粘模型内),但方向是随 U 增大,只能解释 c(U) 的一部分,非其下降趋势 |
| **gap 随扭转增长** | **强**。扭转↑→有效攻角幅值↑→超临界时长与深度↑→被虚构保留的吸力更多 | **强**。Carlson 明文:K_T 随理论吸力需求(局部攻角)增大而**下降**——攻角越大实现率越低,虚构越多 | 无明确机制 |

**常数来源(零拟合合规性):**
- 机理A:分离期吸力→近零,直接引 Narsipur 2020 的黏性观测(定性上是"near-zero",非可调参数);替代实现是 Polhamus 吸力类比(NASA TN D-3767, 1966; J. Aircraft 8(4):193-199, 1971)——分离后吸力**大小保留但转到法向**(变涡升力),弦向推力分量归零,同样无自由常数。
- 机理B:K_T 由 TP-1500 式(1)-(10) 闭式算出,输入仅我们的几何(t/c≈8/287≈0.028, r/c≈4/287≈0.014)与 Re=1.2e5。注意披露:TP-1500 式(9)-(10) 的系数本身是 Carlson 对翼型程序+二维实验数据的拟合——对**用户**是零拟合(无一常数对我们的数据调),但非第一性原理。
- LESP_crit:文献锚为 SD7003@Re=1e5 的 0.14(Ramesh 2014 验证集),而非 0.27。

## 3. 推荐实现方案与验证预测

**改动(按优先级):**

1. **改超临界吸力通道(核心)**:LESP 超临界/LEV 脱落期间,前缘吸力的弦向(推力)分量不再保持 2π·LESP_crit²,改为二选一:
   - **方案 3a(首选,证据最直接)**:分离期弦向吸力 → 0(Narsipur 2020 "near-zero when separated");
   - **方案 3b(经典替代)**:Polhamus 旋转——吸力大小保留、方向转为面法向计入涡升力,弦向推力分量自动消失(Polhamus 1966/1971)。3b 同时给升力一个涡升力补偿,对扭转工况的法向力更保真。
   - 两者的推力后果相同(弦向吸力清零),差别在法向力。建议实现 3b 并以 3a 为消融。
2. **改亚临界吸力实现率**:附着段乘 Carlson K_T(TP-1500 闭式,输入 t/c=0.028, r/c=0.014, Re_c=1.2e5),替换 eta_s=1。
3. **改 LESP_crit 敏感性检查**:以 0.14(Re=1e5 文献锚)为对照跑一组;0.27 无文献出处,若保留须给出圆杆前缘的独立论证(Ramanathan 2023 方向上支持钝前缘更高,但无定量值)。

**验证预测(改完后必须检查,否则回退):**
- **GAP 律的 f² 系数应从 0.40 塌缩至 ≈0**(残差 b 的绝大部分来自被虚构保留的 Garrick 型吸力);零扭转工况模型净流向力应翻号为净阻力,落入实测 −0.5..−3 N 区间的量级。
- 残余 gap 应退化为纯 c(U) 型(圆杆+膜的黏性型阻,无粘模型本就不含),且不再随扭转显著增长。
- 若 f² 系数只降一半左右→机理B份额被低估,检查亚临界段占空比与 K_T 值。
- **tw22.5 的 H16 战果**:升力通道 L = F_N·cosα + F_S·sinα,吸力只经 sinα 小项进入升力,且 F_S ∝ A0² 本身远小于 F_N;预期巡航升力变化为个位数百分比以内,0.95× 达标结论**大概率保持但必须复跑确认**。若采用 3b(Polhamus 旋转),分离段升力还会获得涡升力补偿,H16 更安全。注意 Hirato 论文 p.94 的警告:45° 大攻角初始 LEV 生长期 CFD 吸力反而**超过**平台值——若 tw22.5 存在深失速相位,3a 可能在该相位低估法向载荷。

## 4. 文献不支持或存疑的部分(明确列出)

1. **"逐段瞬时速度剖面阻力"作为 f² 项来源:不支持。** 本轮无任何 CONFIRMED 证据;其标度必含 U 相关项,与 f² 系数 U 无关(0.409/0.403)矛盾。DeLaurier 谱系的型阻放大因子 K(max 4.4, Scherer 1968)是半经验值且出处为水翼实验,用它违反零拟合约束。它只可能贡献 c(U) 的一部分。
2. **eta_s 的具体数值无文献锚。** 核实结论:Djojodihardjo 2018 只列 "ηs: Efficiency Coefficient",无范围、无取值;坊间的 eta_s≈0.98 未获验证。所以"直接取某文献的 eta_s 值"这条路走不通——必须走 Carlson K_T 闭式路线。另注:该文 eq.(31) 还把 DeLaurier 括号项的平方排印丢了,引用时应直引 DeLaurier 1993 原文。
3. **Narsipur "near-zero" 的可迁移性存疑。** 其结论来自光滑前缘翼型的俯仰运动(黏性 CFD+实验,Re 1e4–1e5);我们的前缘是 Ø8mm 圆杆+膜,几何差异大。方向(分离→吸力崩塌)可信,"精确为零"是外推。
4. **Carlson TP-1500 是定常、整机翼方法。** 其常数是对定常翼型数据的拟合,用于扑动瞬时剖面属于准定常外推;且对用户零拟合 ≠ 第一性原理。
5. **Polhamus 旋转在 LDVM 家族中无先例(已核实的否定结论)。** SureshBabu/Hirato/Kambampati 均未实现分离后吸力旋转;采用 3b 是对 LDVM 谱系的自主偏离(尽管 Polhamus 本身是经典文献)。Liu et al. 2017 (AIAA J 55(2)) 的前缘处理因付费墙未能核实。
6. **吸力封顶的符号并非总是"高估"。** Hirato 2016 p.94:大攻角初始 LEV 生长期,CFD 吸力持续增长而模型被迫平台化——封顶在该 regime 反而**低估**吸力。我们的修正在深失速相位可能过狠。
7. **LESP_crit=0.27 与 0.14 之争未决。** 文献同时支持"Re=1e5 应取 0.14"(Ramesh 验证集)与"钝前缘临界值更高"(Ramanathan 2023,无定量);且 Kay 2022 证明临界值随俯仰速率分段线性变化,恒定常数假设本身在低速率区受挑战。此项只能靠敏感性扫描裁决,不能靠文献直接定值。
8. **Ramesh 2014 原文闭源**,其吸力处理经 UNSflow 源码、Faure 2019、Gelado & Ramesh 2022、Hirato 论文四路间接核实,一致性极高,但原文 eq.(2.30)/(2.31) 未直接目验。

**一句话决策**:把超临界期间的弦向前缘吸力从"钉在临界值"改为"Polhamus 旋转归零(或直接置零)",亚临界段乘 Carlson TP-1500 闭式 K_T,LESP_crit 用 0.14 做对照——三处改动全部有文献出处、零自由参数;预期 f² 虚构推力项塌缩、零扭转翻号为净阻力,H16 巡航升力战果预期保持但需复跑确认。

---

## 附:全部幸存题录(逐条对抗核实)

- [thrust-deficit-experiments] **CONFIRMED** Garrick linear potential theory over-predicts mean thrust of oscillating foils at large frequency and amplitude.
  来源: Mackowski and Williamson 2015 JFM 765 524-543
- [ldvm-suction-treatment] **CONFIRMED** In the original LDVM (Ramesh et al. 2014), when LESP exceeds the critical value, discrete leading-edge vortices are shed with strength computed such that the instantaneous LESP is brought back to and 
  来源: Ramesh, Gopalarathnam, Granlund, Ol, Edwards (2014), 'Discrete-vortex method with novel shedding criterion for unsteady aerofoil flows with intermittent leading-edge vortex shedding', J. Fluid Mech. 751:500-538; corroborated by Gelado & Ramesh (2022)
  URL: https://doi.org/10.1017/jfm.2014.297
- [ldvm-suction-treatment] **CONFIRMED** The LDVM force calculation splits forces into a normal force (from A0, A1, A2 Fourier coefficients plus a wake-induced nonlinear term) and a chordwise leading-edge suction force with coefficient c_s =
  来源: Ramesh's open-source LDVM implementation UNSflow (KiranUofG/UnsteadyFlowSolvers.jl), calc_forces in src/lowOrder2D/postprocess.jl (comments cite eqns (2.30)/(2.31) of Ramesh et al. 2014 JFM); independently reproduced in T. M. Faure, L. Dumas, V. Drou
  URL: https://github.com/KiranRamesh-Aero/UnsteadyFlowSolvers.jl/blob/master/src/lowOrder2D/postprocess.jl
- [ldvm-suction-treatment] **CONFIRMED** The physical justification in the LDVM lineage for retaining suction during shedding is an unproven postulate traced to Katz (1981): a separated leading edge is assumed to still support a certain amou
  来源: Hirato, Y. (2016) PhD thesis 'Leading-Edge-Vortex Formation on Finite Wings in Unsteady Flow', NC State University, citing J. Katz (1981) 'A discrete vortex method for the non-steady separated flow over an airfoil', J. Fluid Mech. 102:315-328
  URL: https://www.lib.ncsu.edu/resolver/1840.20/33340
- [ldvm-suction-treatment] **CONFIRMED** Viscous CFD and experiments show the realized leading-edge suction does NOT stay at the critical value during/after LEV shedding: the viscous LESP drops to NEAR-ZERO whenever the flow is separated at 
  来源: Narsipur, Hosangadi, Gopalarathnam, Edwards (2020), 'Variation of leading-edge suction during stall for unsteady aerofoil motions', J. Fluid Mech. 900, A25 (doi:10.1017/jfm.2020.467)
  URL: https://doi.org/10.1017/jfm.2020.467
- [ldvm-suction-treatment] **CONFIRMED** Experimental PIV-based measurement of the leading-edge suction parameter independently confirms suction weakens with LE separation: suction magnitude is negatively correlated with the LE shear-layer h
  来源: Deparday, He, Eldredge, Mulleners, Williams (2022), 'Experimental quantification of unsteady leading-edge flow separation', J. Fluid Mech. 941, A60 (arXiv:2110.08384)
  URL: https://arxiv.org/abs/2110.08384
- [ldvm-suction-treatment] **CONFIRMED** Documented critical-LESP values from the Ramesh 2014 validation set are all well below the user's a0_crit=0.27 at comparable Re, and decrease with Re for the SD7003: 0.21 (Re=10,000), 0.18 (Re=30,000)
  来源: Ramesh, K., Gopalarathnam, A., Granlund, K., Ol, M.V. & Edwards, J.R. (2014) "Discrete-vortex method with novel shedding criterion for unsteady aerofoil flows with intermittent leading-edge vortex shedding", J. Fluid Mech. 751, 500-538; as tabulated 
  URL: https://crea.ecole-air-espace.fr/wp-content/uploads/2021/01/ARTICLES-5.pdf
- [ldvm-suction-treatment] **CONFIRMED** Ramesh (2020) resolves the LE singularity with matched asymptotic expansions: leading-edge velocity is finite and inversely proportional to the square root of the leading-edge radius, with suction mea
  来源: Ramesh, K. (2020), 'On the leading-edge suction and stagnation point location in unsteady flows past thin aerofoils', J. Fluid Mech. 886, A13 (doi:10.1017/jfm.2019.1070)
  URL: https://eprints.gla.ac.uk/206131/7/206131.pdf
- [ldvm-suction-treatment] **CONFIRMED** The critical-LESP criterion itself is airfoil-shape and Reynolds-number dependent; critical LESP increases as the leading edge is made rounder, which motivated replacing it with a boundary-layer-criti
  来源: Ramanathan, H. & Gopalarathnam, A. (2023), 'Prediction of leading-edge-vortex initiation using criticality of the boundary layer', Theoretical and Computational Fluid Dynamics (doi:10.1007/s00162-023-00648-z)
  URL: https://repository.lib.ncsu.edu/server/api/core/bitstreams/b653ded2-3167-417f-abc4-d3585e03f66d/content
- [ldvm-suction-treatment] **CONFIRMED** In the 3D extension of the LESP concept (UVLM + LESP for finite wings — the architecture closest to the user's model), the LE suction force channel C_S = 2*pi*Sum(c*A0^2*dy)/S is retained and LESP is 
  来源: Hirato, Y. (2016) PhD thesis, NC State University; journal version: Hirato, Shen, Gopalarathnam, Edwards (2019), 'Vortex-Sheet Representation of Leading-Edge Vortex Shedding from Finite Wings', Journal of Aircraft 56(4) (doi:10.2514/1.C035124)
  URL: https://doi.org/10.2514/1.C035124
- [ldvm-suction-treatment] **CONFIRMED** The constancy/universality of critical LESP is further challenged at low pitch rates and across Re: critical LESP shows a strong, roughly piecewise-linear dependence on reduced pitch rate and increase
  来源: Kay, N.J., Richards, P.J., Sharma, R.N., "Low-Reynolds Number Behavior of the Leading-Edge Suction Parameter at Low Pitch Rates," AIAA Journal, Vol. 60, No. 3, 2022, pp. 1721-1729 (published online Sept 2021), doi:10.2514/1.J060733
  URL: https://researchspace.auckland.ac.nz/server/api/core/bitstreams/dddcf47d-03fb-4c43-9311-73f203465415/content
- [ldvm-suction-treatment] **CONFIRMED** The Gopalarathnam group's own successor model for dynamic stall (2023) explicitly augments inviscid unsteady airfoil theory with time-dependent VISCOUS effects (time-varying trailing-edge separation i
  来源: Narsipur, Gopalarathnam, Edwards (2023), 'Low-Order Modeling of Dynamic Stall on Airfoils in Incompressible Flow', AIAA Journal 61(1):206-222 (doi:10.2514/1.J061595)
  URL: https://doi.org/10.2514/1.J061595
- [ldvm-suction-treatment] **CONFIRMED** The classical alternative treatment for separated leading edges is the Polhamus leading-edge suction analogy: after LE separation the suction force magnitude is reassigned but ROTATED to act normal to
  来源: Polhamus, E.C. (1966) 'A Concept of the Vortex Lift of Sharp-Edge Delta Wings Based on a Leading-Edge-Suction Analogy', NASA TN D-3767; Polhamus, E.C. (1971) 'Predictions of Vortex-Lift Characteristics by a Leading-Edge Suction Analogy', J. Aircraft 
  URL: https://ntrs.nasa.gov/api/citations/19680022518/downloads/19680022518.pdf
- [suction-efficiency-eta] **PLAUSIBLE** In DeLaurier's model, the leading-edge suction efficiency eta_s is a scalar in [0,1] that multiplies Garrick's leading-edge suction thrust term to 'account for viscosity effects' on the potential-flow
  来源: DeLaurier, J.D. (1993), 'An aerodynamic model for flapping-wing flight,' The Aeronautical Journal 97(964):125-130 (original squared LE-suction equation with ηs); Djojodihardjo, H. (2018), Int J Astronaut Aeronautical Eng 3:017, eq. (31) reproduces it
  URL: https://vibgyorpublishers.org/content/ijaae/ijaae-3-017.pdf
- [suction-efficiency-eta] **CONFIRMED** The provenance of eta_s is DeLaurier's own partial-leading-edge-suction treatment (rooted in Polhamus's suction analogy and Garrick's plunge-suction theory), NOT Scherer 1968. Scherer 1968 is cited in
  来源: Djojodihardjo, H. (2018) Int. J. Astronaut. Aeronautical Eng. 3:017; and Djojodihardjo & Ramli, 'Kinematic and Aerodynamic Modelling of Bi- and Quad-Wing Flapping Wing MAV' (SciRP, cites Scherer 1968 Hydronautics for K, eta_s listed separately).
  URL: https://file.scirp.org/Html/2-2850028_72656.htm
- [suction-efficiency-eta] **CONFIRMED** DeLaurier's partial-leading-edge-suction concept predates and feeds the 1993 flapping model: DeLaurier (1983) 'Drag of wings with cambered airfoils and partial leading-edge suction,' Journal of Aircra
  来源: DeLaurier, J.D. (1983), Journal of Aircraft 20(10):882-886; cited in Harmon, R.L. (2008) M.S. thesis, Univ. of Maryland (dir. J.E. Hubbard).
  URL: https://drum.lib.umd.edu/bitstreams/28747e88-459a-498f-a931-87086f96443d/download
- [suction-efficiency-eta] **CONFIRMED** A ZERO-FITTING, geometry+Reynolds-anchored way to set eta_s<1 already exists: Carlson, Mack & Barger's 'attainable leading-edge thrust' method defines a thrust factor KT = ct/Ct = fraction of theoreti
  来源: Carlson, H.W., Mack, R.J., Barger, R.L. (1979), NASA Technical Paper 1500, 'Estimation of Attainable Leading-Edge Thrust for Wings at Subsonic and Supersonic Speeds,' NASA Langley.
  URL: https://ntrs.nasa.gov/api/citations/19800001866/downloads/19800001866.pdf
- [suction-efficiency-eta] **CONFIRMED** In the Carlson attainable-thrust method the attained fraction KT DECREASES with increasing theoretical thrust (i.e. increasing local angle of attack / higher suction demand) and INCREASES with thickne
  来源: Carlson, Mack & Barger (1979), NASA TP-1500.
  URL: https://ntrs.nasa.gov/api/citations/19800001866/downloads/19800001866.pdf