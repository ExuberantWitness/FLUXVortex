# 机理裁定：扭转线性推力亏空(-0.20 N/deg、频率解耦、环流幅值过预测2.5–4×)

## 1. 机理裁定(按证据强度排序)

### M1(主裁定,证据最强):分离态下环流敏感度衰减缺失 —— Kirchhoff/LB 因子未作用于扭转诱导载荷

**定量指纹直接命中。** LB/Kirchhoff 环流衰减因子 ((1+√f)/2)² 的界为 [1/4, 1](Bangga et al. WES 2020, Eq.13, CONFIRMED),深失速下 f 落至地板值 0.04,因子 = ((1+0.2)/2)² = 0.36,即无衰减模型对环流敏感度过预测 **1/0.36 ≈ 2.8×**——正落在归档诊断的 2.5–4× 区间中央,上界 4× 恰是 Kirchhoff 界的算术极限。这不是巧合级的吻合。

**为何对扭转线性:** 扭转诱导的环流载荷扰动 δN = (∂N/∂α)·Δα_tw,而 Δα_tw ∝ tw。误差 = (1−η)·(∂N/∂α)_model·Δα_tw,只要分离态 η 主要由拍动(而非扭转)设定且近似恒定,误差严格 ∝ tw,斜率恒定。

**为何对频率解耦(这是判别性证据):**
- 扭转是几何攻角变化,准定常环流力 ∝ ½ρU²c·a₀·tw,**不含频率因子**(不同于沉浮的 ḣ/U ∝ f);
- 中下冲程 70% 展位拍动诱导攻角 ≈ atan(V_flap/U) ≈ 38°(f=2.3)/42°(f=2.6),两者都深超 α_ss=12° 十几度以上 → f'' 在两个频率下**同样饱和于地板**,η 相同 → 斜率相同;
- k = 0.26 vs 0.29,|C(k)| 差 <2%,环流通道本身几乎无频率杠杆。
- 反向排除:若是附加质量过预测,gap 应 ∝ f²(2.6/2.3 应差 28%);若是尾迹滞后,应有 k 依赖。观测的频率解耦直接排除这两类。

**为何升力亏空共增长:** 扭转(feathering,与拍动 90° 相位)峰值与拍动速度峰值同相,即扭转在中下冲程"卸掉"攻角。模型以全敏感度(≈2.8× 真实值)扣除法向力,而真实翼此时大面积分离、对攻角扰动的响应被衰减——真实翼被 feathering 卸掉的力比模型少。中下冲程法向力同时投影到升力(向上)与推力(前倾),故升力、推力同时被模型低估,均 ∝ tw。上冲程升力方向误差部分对消、推力误差同号累加,故推力 gap(22.5° 处约 4.5 N)大于升力亏空(1.5–2.6 N)——与数据的不对称一致。

**反证的正确读法(反而强化门控式方案):** Strangfeld(JFM 2025)、Daliri(PoF 2023)、Ol(ExF 2009)一致表明附着流/一般趋势下 Theodorsen 幅值**不**过预测——所以 2.5–4× 的过预测必然是**分离态特异性**的,衰减必须由分离状态门控,绝不能做全局 knockdown。

### M2(次选,证据中等):膜被动卸载/气弹洗出

Tiomkin & Jaworski(arXiv:2204.01204, CONFIRMED)给出膜等效 Theodorsen 函数:低 k 下跟随刚板趋势但"幅值降低、相位滞后增大"——方向正确、闭式零拟合(常数为张力系数 C_T 与质量比 μ,均为物理量)。频率解耦兼容(低 k 段平缓)、线性兼容(线性理论)。**但**:低 k 衰减量级通常是百分之十几,单独扛不动 2.8×;且若流固加载共振落在 2.3–2.6 Hz 附近则**反向放大**。定位:补充项,非主因。

### M3(弱):尾迹亏损/3D 下洗不足

Bird et al.(AIAA J 2022, CONFIRMED)证明条带理论因缺尖涡下洗而过预测幅值——但我们的模型是**自由尾迹 UVLM,本已解析尖涡下洗**;且该文中即使 ULLT 的过预测也只是 "slightly",承载不了 2.5–4×,且应随 k 变化。仅剩尾迹分辨率/核半径的收敛性检查价值。

### M4(排除):附加质量(∝f²,与频率解耦矛盾);平板 CN blend(cd_form=1.98,已证伪在案,不再提)。

## 2. 零拟合实现、常数出处、tw0 风险

**M1 常数(全部文献值/物理反演,零拟合):**
- 静态分离曲线 f(α):由静态极曲线 Kirchhoff 反演 f_s = min{[2√(C_l/(C_lα(α−α₀)))−1]², 1}(OpenFAST AeroDyn 理论文档, CONFIRMED),直接复用模型已有的 α_ss=12°、宽度 16°——无新自由度;或 LB Eq.(14) 形式(α₁=α_ss,S₁/S₂ 由失速宽换算)。
- 非定常滞后:T_p=1.7、T_f=3.0 半弦(LB 原始/OpenFAST 默认, CONFIRMED);或 GK 路线 τ₁=4.24c/U(St=0.235 物理来源)、τ₂ 由失速延迟幂律(Ayancik & Mulleners JFM 2022, R²=0.978);Re=6e4 tripped 版本 τ₁=3.57c/U(Rezapour & Mulleners)更贴近我们 Re。
- **实现变体 A(标准 LB 式):** 用 ((1+√f'')/2)² 替换现有"仅升力、准定常"的 Kirchhoff 失速,作用于**投影前的条带环流法向力**——注意是替换不是叠加(防重复计权);LEV-impulse 模块保留,承担 LB 中"涡升力"补偿角色。**tw0 风险:中-高**——tw0 时中下冲程同样深失速,整力衰减会动已验证的推力平衡;能否守住取决于 LESP-LEV 通道是否已承载该处主力。必须设回归门:tw0 推力 MAE ≤ 0.45 N。
- **实现变体 B(保守):** 衰减只作用于扭转扰动敏感度 δN(围绕 tw0 线性化),tw0 **by construction 零变化**。代价:非标准形式,需明示线性化假设。

**M2:** 膜等效 Theodorsen 幅值比 |C_memb(k)|/|C(k)| 乘在条带非定常环流幅值上;先算 C_T(实测预张力)验证 C_T>1.73(免发散)、μ 与首阶流固共振位置。tw0 风险:同时衰减拍动通道,会扰动 tw0 战果,且量级不足——不作首选。

**M3:** 尾迹加密/核半径收敛检查,无新常数,tw0 风险低,预期收益小。

**硬警告(防重复计权):** GK 自述"不适用平板前缘失速"(Ayancik & Mulleners, 原文 CONFIRMED)。薄膜锐缘截面上,Kirchhoff/GK 只许衰减**附着环流部分**;LEV 载荷必须留在既有 LESP 门控模块,两者边界即 LESP 阈值。

## 3. 推荐实现与验证预测

**推荐:** 变体 A(f'' 静态反演 + T_f=3.0 滞后,替换式接入环流法向力通道;LE 吸力塌缩、LEV impulse、附加质量均不动),tw0 回归门破门则退变体 B。

**可证伪预测:**
1. 扭转律斜率从 −0.20 N/deg 塌至 |s| ≲ 0.05 N/deg(衰减因子 ~0.36 → 残差 ≈ 原 gap 的 1/3 以内),且 2.3/2.6 Hz 斜率仍一致(差 <10%)。若修后残差 ∝ f² → 转查附加质量;∝ f → 转查尾迹。
2. tw≥22.5° 升力亏空从 1.5–2.6 N 收敛到 <0.8 N,与推力**同因、同比例**塌陷(同一通道单一修改,双指标必须同时动)。
3. 瞬时诊断:环流幅值比 model/measured 从 2.5–4× 落入 1.0–1.3×,且中下冲程相位段改善最大。
4. 回归门:tw0 推力 MAE ≤ 0.45 N(现 0.39)。
5. 若斜率只塌一半:剩余部分启用 M2 膜衰减(先算共振,叠加时衰减只乘一次)。

## 4. 文献不支持/存疑(如实列出)

- **f(α) 曲线出处翼型不匹配:** LB 常数源自 NACA0012/HH-02/SC-1095(圆头旋翼翼型);薄膜锐缘截面无直接验证,属外推。GK 明文排除前缘失速情形。
- **Otomo et al.(ExF 2021, CONFIRMED)是最强张力:** 2D 大幅俯仰、大分离下,**未衰减**薄翼理论+涡冲量记账仍准确。这暗示我们的 2.5–4× 可能部分来自 LEV 冲量相位/记账缺陷而非纯缺衰减——若修后幅值比只降到 ~1.5× 且残差集中在 LEV 形成段,应转查 LESP 模块相位。
- **Ol et al.:** 深失速下 Theodorsen"无明显失败"(仅定性)——反对任何非门控的整体 knockdown。
- 膜理论(Tiomkin-Jaworski)为线性、2D、小振幅;±45° 拍动远超其适用域,共振位置尚未对我们的膜算出。
- UVLM 幅值过预测**无文献定量数**;流传的 "−17%/+21%" 无源,禁引(已核实无出处)。
- 频率解耦证据仅 2.3/2.6 Hz 两点(相差 13%),判别杠杆偏弱;建议若有条件补一个更低频点。
- Daliri(PoF 2023)仅摘要级核验(AIP 付费墙);且其"coupled 俯仰-沉浮幅值预测准确"的结论削弱"运动耦合本身导致过预测"的备选解释——反向支持分离归因。

---

## 附:幸存题录(对抗核实)

- [amplitude-overprediction] **CONFIRMED** 3D finite-wing downwash makes real wings produce LESS unsteady lift than 2D strip/Theodorsen theory, and inviscid strip-based methods over-predict the lift amplitude as aspect rati
  来源: Bird, H.J.A., Ramesh, K., et al. (2021), 'Applying inviscid linear unsteady lifting-line theory to viscous large-amplitude problems,' arXiv:2104.06091 (AIAA). Heaving NACA0008, Re=10,000, k=0.4, aspect ratios inf/6/3/1; 
  URL: https://arxiv.org/abs/2104.06091
- [amplitude-overprediction] **CONFIRMED** Theodorsen's lift AMPLITUDE prediction is accurate (not over-predicting) for pure-pitch and pure-plunge in attached flow at moderate k, but degrades for uncoupled/coupled combined 
  来源: A. Daliri, M. J. Maghrebi, and M. R. Soltani, 'Experimental assessment of Theodorsen's function for uncoupled pitch–plunge motion,' Physics of Fluids 35(3), 037114 (2023). DOI: 10.1063/5.0139918
  URL: https://pubs.aip.org/aip/pof/article/35/3/037114/2882146
- [amplitude-overprediction] **CONFIRMED** COUNTER-EVIDENCE / boundary condition: for small-amplitude, attached-flow oscillation at moderate k, Theodorsen predicts the unsteady-lift amplitude essentially EXACTLY (no meaning
  来源: Strangfeld, C., Müller-Vahl, H.F., Nayeri, C.N., Paschereit, C.O., Greenblatt, D. (2025), 'Airfoil synchronous surging and pitching,' Journal of Fluid Mechanics, 1009, A41 (arXiv:2408.10675, 2024). Re = 3.0x10^5, k = 0.0
  URL: https://arxiv.org/abs/2408.10675
- [amplitude-overprediction] **CONFIRMED** COUNTER-EVIDENCE / nuance: even in deep dynamic stall at flapping-cruise reduced frequency, Theodorsen's lift-coefficient time history shows 'no clear failure' vs experiment for th
  来源: Ol, M.V., Bernal, L., Kang, C.-K., Shyy, W. (2009), 'Shallow and deep dynamic stall for flapping low Reynolds number airfoils,' Experiments in Fluids 46(5), 883-901. DOI: 10.1007/s00348-009-0660-3
  URL: https://link.springer.com/article/10.1007/s00348-009-0660-3
- [amplitude-overprediction] **PLAUSIBLE** UVLM/potential-flow flapping-wing solvers reproduce lift TRENDS and net magnitude with only 'reasonable/good' accuracy and structurally cannot represent leading-edge separation — t
  来源: Urban, C., 'Validating an Open-Source UVLM Solver for Analyzing Flapping Wing Flight', Mechanical Engineering and Materials Science Independent Study 135, Washington University in St. Louis, 2020 (https://openscholarship
  URL: https://openscholarship.wustl.edu/mems500/135/
- [separation-attenuation] **CONFIRMED** The Leishman-Beddoes (LB) model attenuates the CIRCULATORY normal force under separation by the Kirchhoff/Helmholtz factor ((1+sqrt(f))/2)^2: C_N^visc = (dC_N/dalpha) * ((1+sqrt(f_
  来源: Bangga, Lutz, Arnold, 'An improved second-order dynamic stall model for wind turbine airfoils', Wind Energy Science 5, 1037-1058 (2020), Eq. (13); original: Leishman & Beddoes, J. Am. Helicopter Soc. 34(3), 1989. Verifie
  URL: https://wes.copernicus.org/articles/5/1037/2020/
- [separation-attenuation] **CONFIRMED** The LB static separation-point curve is f_n = 1 - 0.3*exp((alpha_n - alpha_1)/S_1) for alpha <= alpha_1, and f_n = 0.04 + 0.66*exp((alpha_1 - alpha_n)/S_2) for alpha > alpha_1, whe
  来源: Bangga et al., WES 5:1037 (2020), Eqs. (14) and surrounding text (verified in PDF, lines 307-333); inversion formula from OpenFAST AeroDyn Unsteady Aerodynamics theory documentation (verified via WebFetch).
  URL: https://openfast.readthedocs.io/en/main/source/user/aerodyn/theory_ua.html
- [separation-attenuation] **CONFIRMED** The unsteady (lagged) separation state f'' in LB uses two first-order lags with literature-fixed dimensionless time constants: a leading-edge pressure lag T_p = 1.7 semi-chords and
  来源: Bangga et al., WES 5:1037 (2020), Table 2 (verified in PDF, lines 1502-1530); OpenFAST AeroDyn UA theory docs (verified via WebFetch). Lag form: D_fn = D_fn-1 exp(-ds/T_f) + (f_n - f_n-1) exp(-ds/2T_f).
  URL: https://wes.copernicus.org/articles/5/1037/2020/
- [separation-attenuation] **CONFIRMED** The Goman-Khrabrov (GK) state-space model: tau_1 dX/dt + X = X_0(alpha - tau_2 alpha_dot), with X in [0,1] the attached-flow fraction (separation-point location), X_0 the STATIC se
  来源: Ayancik & Mulleners, 'All you need is time to generalise the Goman-Khrabrov dynamic stall model', J. Fluid Mech. 942, R8 (2022) / arXiv:2110.08516. Verified from downloaded PDF full text (gk_paper.txt).
  URL: https://arxiv.org/abs/2110.08516
- [separation-attenuation] **CONFIRMED** HARD CAVEAT from the same primary source: the GK model targets trailing-edge-separation dynamic stall and 'is not applicable to flat plate leading-edge stall cases'. For our sharp/
  来源: Ayancik & Mulleners, JFM 942 R8 (2022), Section 1 (verified from PDF full text).
  URL: https://arxiv.org/abs/2110.08516
- [separation-attenuation] **CONFIRMED** Independent replication with updated constants at Re = 6e4: for a TRIPPED NACA0018 (water channel, EPFL), static stall angle alpha_ss = 13.3 deg, stall-delay power law (t_ds - t_ss
  来源: Rezapour & Mulleners, 'Dynamic Stall Characteristics and Modelling of Time-Varying Pitching Kinematics', arXiv:2508.10647. Verified from downloaded PDF full text (rezapour.txt).
  URL: https://arxiv.org/abs/2508.10647
- [separation-attenuation] **CONFIRMED** COUNTER-EVIDENCE that must be addressed: for large-amplitude pitching of a NACA0018 at Re = 3.2e4, suitably adapted Theodorsen / unsteady thin-aerofoil theory predicts the measured
  来源: Otomo, Henne, Mulleners, Ramesh, Viola, 'Unsteady lift on a high-amplitude pitching aerofoil', Experiments in Fluids 62:6 (2021). Verified from University of Glasgow repository abstract (fetched via curl).
  URL: https://eprints.gla.ac.uk/231548/
- [separation-attenuation] **CONFIRMED** Membrane-specific attenuation mechanism with closed-form, zero-fit constants: linear aeroelastic theory for membrane aerofoils yields membrane-equivalent Theodorsen/Sears functions
  来源: Tiomkin & Jaworski, 'Unsteady aerodynamic theory for membrane wings', arXiv:2204.01204v2 (JFM style). Verified from downloaded PDF full text (memb.txt).
  URL: https://arxiv.org/abs/2204.01204