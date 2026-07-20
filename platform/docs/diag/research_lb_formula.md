# Leishman–Beddoes 动态失速 × 分条带 UVLM 环格:零拟合实现配方

**任务**:为 RoboEagle 刚性扑翼(半展0.8m / 弦0.287m / AR5.6 / Re~1.5e5 / 扑±22.5° / 扭转±11.25° / f1.4–2.6Hz / U6–10m/s / k=0.13–0.39 / AR~1)建立**完整 Leishman–Beddoes (L-B) 动态失速模型**在**分条带 UVLM 环格**上的**零拟合**实现配方,一次性改善三个趋势病(dL/df、dT/dU、aoa15 甩阻)。

**零拟合纪律**:不对本机测力数据拟合任何常数;可用发表翼型常数(类比已用 UIUC 极曲线)。
**排除**:柔性(实测刚性)、rVPM/VPM 粒子 LEV(已 CLOSED)、纯准定常 Kirchhoff 缩放(已证只给 1/3 且方向错)。

置信度标签:**CONFIRMED**(读到一手公式/数值)、**PLAUSIBLE**(有据推断)、**UNKNOWN**(未找到,不编造)。

---

## A. L-B 三模块的确切离散公式

> 主要一手源:Bangga, Lutz & Arnold, *"An improved second-order dynamic stall model for wind turbine airfoils"*, Wind Energ. Sci. (Discuss.), 2020, DOI 10.5194/wes-2020-75(本文 Eqs 1–35、70 逐式核对)。原始模型:Leishman & Beddoes, *"A Semi-Empirical Model for Dynamic Stall"*, J. Am. Helicopter Soc., 34(3):3–17, 1989。

记号:$s=2Vt/c$(半弦无量纲时间),$\beta=\sqrt{1-M^2}$(低马赫取 $\beta\to1$),$dC_N/d\alpha$ 为**无粘**极曲线梯度(≈2π 每弧度),$\alpha_0^{INV}$/$\alpha_0^{VISC}$ 为零(无粘/粘性)法向力攻角。

### A.1 附着非定常环量(indicial / Duhamel)

**CONFIRMED**(Bangga2020 Eqs 1–9)。对攻角历史 $\alpha_n$,有效攻角:
$$\alpha_{e_n}=\alpha_n-X_n-Y_n \tag{1}$$
两项缺函数(deficiency functions,指数递推,即 Duhamel 的离散卷积):
$$X_n=X_{n-1}e^{-b_1\beta^2\Delta s}+A_1\,\Delta\alpha_n\,e^{-b_1\beta^2\Delta s/2} \tag{2}$$
$$Y_n=Y_{n-1}e^{-b_2\beta^2\Delta s}+A_2\,\Delta\alpha_n\,e^{-b_2\beta^2\Delta s/2} \tag{3}$$
其中 $\Delta\alpha_n=\alpha_{n+1}-\alpha_n$(式4),$\Delta s=s_n-s_{n-1}$(式5)。**这是 Wagner 阶跃响应的两指数近似**。常数两套(见 B.0):**不可压 R.T. Jones($A_1\to0.165,A_2\to0.335,b_1\to0.0455,b_2\to0.3$,我们低马赫首选)** vs L-B 可压($A_1=0.3,A_2=0.7,b_1=0.14,b_2=0.53$)。

- **环量法向力**(circulatory):
$$C_{N_n}^{C}=\frac{dC_N}{d\alpha}\left(\alpha_{e_n}-\alpha_0^{INV}\right) \tag{6}$$
(有限弯度翼型必须保留 $\alpha_0^{INV}$ 项,Bangga 明确指出;这对我们 NACA2406 弯度翼型**必须**。)

- **非环量(冲量/附加质量)法向力**(impulsive,活塞理论):
$$C_{N_n}^{I}=\frac{4K_\alpha T_I}{M}\left(\frac{\Delta\alpha_n}{\Delta t}-D_n\right) \tag{7}$$
$T_I=Mc/V$,缺函数
$$D_n=D_{n-1}e^{-\Delta t/(K_\alpha T_I)}+\left(\frac{\Delta\alpha_n-\Delta\alpha_{n-1}}{\Delta t}\right)e^{-\Delta t/(2K_\alpha T_I)} \tag{8}$$
**低马赫极限**:$C_N^I\propto 1/M$ 但含 $M$ 的 $T_I$ 与活塞理论项相消,退化为经典**附加质量**项 $C_N^I\to (\pi c/2)(\ddot h/U^2+\dot\alpha/U-\dots)$;**我们 M<0.03,非环量项在附着段可保留薄翼附加质量形式或直接由 UVLM 环格自身捕获(见 C 节双计讨论)**。

- **附着总法向力**:
$$C_{N_n}^{P}=C_{N_n}^{C}+C_{N_n}^{I} \tag{9}$$

**前缘吸力 / LESP 的非定常演化**:L-B 不显式输出 LESP,而是用**滞后法向力** $C_N^{P1}$(见 A.2 式10)与临界值比较判 LEV;LESP 判据走 Ramesh 路线(A.3)。前缘吸力的非定常衰减由切向力 $C_T$(A.2 式19)的 $\sqrt{f}$ 因子体现。

### A.2 后缘分离(Kirchhoff-f + 滞后)

**CONFIRMED**(Bangga2020 Eqs 10–19、33)。

- **前缘压力滞后**(一阶,时间常数 $T_p$),定义滞后法向力:
$$C_{N_n}^{P1}=C_{N_n}^{P}-D_{p_n} \tag{10}$$
$$D_{p_n}=D_{p_{n-1}}e^{-\Delta s/T_p}+\left(C_{N_n}^{P}-C_{N_{n-1}}^{P}\right)e^{-\Delta s/(2T_p)} \tag{11}$$
**注:Bangga 引 Leishman&Beddoes(1989)明确“$T_p$ 很大程度上与翼型形状无关”**——这是零拟合关键(见 B)。

- **代入滞后的有效攻角**:
$$\alpha_{f_n}=\alpha_0^{INV}+\frac{C_{N_n}^{P1}}{dC_N/d\alpha} \tag{12}$$

- **Kirchhoff/Helmholtz 关系**(平板,分离点 $f_n$,$f=1$ 全附着 / $f=0$ 全分离):
$$C_{N_n}^{VISC}=\frac{dC_N}{d\alpha}\left(\frac{1+\sqrt{f_n}}{2}\right)^{2}\left(\alpha_n-\alpha_0^{VISC}\right) \tag{13}$$

- **静态分离点 f(α)**:文献标准做法是对静态极曲线**曲线拟合**:
$$f_n=\begin{cases}1-0.3\,e^{(\alpha_n-\alpha_1)/S_1}, & \alpha_{f_n}\le\alpha_1\\ 0.04+0.66\,e^{(\alpha_1-\alpha_n)/S_2}, & \alpha_{f_n}>\alpha_1\end{cases} \tag{14}$$
$\alpha_1$=静态失速角(单拐点,源自 NACA0012/HH-02/SC-1095),$S_1,S_2$ 控制失速前后分离进展速率。**Gupta&Leishman(2006)对 S809 给三指数双拐点形式**(式15)。
- **★零拟合关键(Bangga2020 式33,CONFIRMED)**:**f(α) 直接由静态极曲线反演**,不做曲线拟合:
$$f_n=\left(2\sqrt{\frac{C_{N_n}^{VISC}}{\frac{dC_N}{d\alpha}\left(\alpha_{f_n}-\alpha_0^{VISC}\right)}}-1\right)^{2} \tag{33}$$
  Bangga 明确:“the separation point is derived directly from the static polar data using inversion of Equation (13)… as long as the static polar data is available,the user can avoid curve fitting.” **同法被 Hansen et al.(2004, Risø-R-1354)采用。** 这正是我们提议的零拟合路径,**已被文献认可**。

- **分离点一阶滞后**(边界层/分离滞后,时间常数 $T_f$):
$$f_{2_n}=f_n-D_{f_n} \tag{16}$$
$$D_{f_n}=D_{f_{n-1}}e^{-\Delta s/T_f}+\left(f_n-f_{n-1}\right)e^{-\Delta s/(2T_f)} \tag{17}$$

- **非定常粘性法向力**(Kirchhoff 加权 + 冲量项):
$$C_{N_n}^{f}=\frac{dC_N}{d\alpha}\left(\frac{1+\sqrt{f_{2_n}}}{2}\right)^{2}\left(\alpha_{e_n}-\alpha_0^{VISC}\right)+C_{N_n}^{I} \tag{18}$$
  → 系数 $((1+\sqrt f)/2)^2$ **CONFIRMED**。

- **★切向吸力独立衰减律**(Bangga2020 式19,CONFIRMED):
$$C_{T_n}^{f}=-\eta\,\frac{dC_N}{d\alpha}\,\alpha_{e_n}^{2}\,\sqrt{f_{2_n}} \tag{19}$$
  $\eta$ 为常数;**CONFIRMED $\eta=0.95$**(Bangga2020 所用标准值,见 B.0;源自 Leishman&Beddoes 1989 NACA0012)。正 $C_T$ 朝尾缘。**注意**:Bangga 的**新 IAG 二阶模型(式70)因一阶 LB 阻力不准而**弃用** $\eta$**,改用静态粘性极曲线在 $\alpha_{f_n}$ 插值 $C_T$:
$$C_{T_n}^{f}=C_T^{VISC}(\alpha_{f_n}) \tag{70}$$
  → **这是重要的实现选项**:若 $\eta$ 拟合值不可得/不可靠,可用静态极曲线直接给 $C_T$(零拟合、且对我们有 UIUC 极曲线天然适配)。

### A.3 前缘涡(LEV / vortex lift)

**CONFIRMED**(Bangga2020 Eqs 21–27、24、35)。

- **LEV 形成判据**:滞后法向力 $C_N^{P1}$ 超过无粘临界静态法向力 $C_N^{CRIT}$:
$$C_N^{CRIT}=\frac{dC_N}{d\alpha}\left(\alpha_n^{CRIT}-\alpha_0^{INV}\right) \tag{25}$$
  $\alpha^{CRIT}$ 取**粘性力矩极曲线拐点**对应攻角。(此即 L-B 的 CN-based onset;Boyd/Sheng/Ramesh 判据见 B/C。)

- **涡升力**(vortex lift = 线性化环量值与 Kirchhoff 非线性值之差):
$$C_{V_n}=C_{N_n}^{C}\left(1-K_n\right),\qquad K_n=\frac14\left(1+\sqrt{f_{2_n}}\right)^{2} \tag{21,22}$$

- **涡升力累积/脱落**(分段指数,涡时间常数 $T_v$、涡心迁移时间 $T_{vl}$):
$$C_{N_n}^{V}=\begin{cases}C_{N_{n-1}}^{V}e^{-\Delta s/T_v}+\left(C_{V_n}-C_{V_{n-1}}\right)e^{-\Delta s/(2T_v)}, & 0<\tau_{v_n}<T_{vl}\\ C_{N_{n-1}}^{V}e^{-\Delta s/T_v}, & \text{otherwise}\end{cases} \tag{23}$$

- **无量纲涡时间**(涡在弦上传播的时钟,dos Santos Pereira 2010 / Elgammi&Sant 2016;Bangga 式35 修正下冲程不连续):
$$\tau_{v_n}=\begin{cases}\tau_{v_{n-1}}+0.45\,\frac{\Delta t}{c}\,V, & C_{N_n}^{P1}>C_N^{CRIT}\\ 0, & C_{N_n}^{P1}<C_N^{CRIT}\ \text{and}\ \Delta\alpha_n\ge0\\ \tau_{v_{n-1}}, & \text{otherwise}\end{cases} \tag{24/35}$$
  **rise-peak-drop 脱落判据**:当 $\tau_v$ 超过 $T_{vl}$(涡迁移时间)即停止累积、转入纯指数衰减(式23 第二支)= 涡脱落后升力塌落。

- **压力中心随 LEV 迁移**:
$$C_{P_{v_n}}=K_v\left(1-\cos\frac{\pi\tau_v}{T_{vl}}\right) \tag{26},\qquad C_{M_n}^{V}=-C_{P_{v_n}}C_{N_n}^{V} \tag{27}$$

- **总动态载荷**(三模块叠加):
$$C_{N_n}^{D}=C_{N_n}^{f}+C_{N_n}^{V}\ (28),\quad C_{T_n}^{D}=C_{T_n}^{f}\ (29),\quad C_{M_n}^{D}=C_{M_n}^{f}+C_{M_n}^{V}\ (30)$$
$$C_{L_n}^{D}=C_{N_n}^{D}\cos\alpha_n-C_{T_n}^{D}\sin\alpha_n\ (31),\quad C_{D_n}^{D}=C_{N_n}^{D}\sin\alpha_n+C_{T_n}^{D}\cos\alpha_n\ (32)$$

**LEV 备选判据(LESP 路线,扑翼更对口,PLAUSIBLE→需 C 节决定)**:Ramesh et al.(JFM 751, 2014,DOI 10.1017/jfm.2014.297;及 AIAA J 2021 低 Re LESP)用薄翼理论 $A_0$ 项作瞬时 LESP,临界 LESP 与运动学无关、仅依赖翼型形状+Re,超临界即触发 LEV 脱落。**我们已有 LESP 门控,可与 L-B 的 $C_N^{CRIT}$ 判据互换/并联**。

---

## B. 时间常数与常数的零拟合来源(★最关键)

### B.0 文献标准默认值(CONFIRMED 数值)

**两套 indicial 常数,勿混**(关键区分):
- **不可压 Wagner(R.T. Jones 1938/1940,薄翼理论,翼型无关)**:
$$\phi(s)=1-0.165\,e^{-0.0455s}-0.335\,e^{-0.3s},\qquad \phi(0)=0.5$$
  常数 $\Psi_1=0.165,\Psi_2=0.335,\epsilon_1=0.0455,\epsilon_2=0.3$,与精确 Wagner 函数误差 <1%,**不可压薄翼普适**。Garrick 代数近似 $\phi=(s+2)/(s+4)$ 误差 <2%。
- **可压 Leishman-Beddoes 缺函数**(Bangga2020 式2–3 所用,经 Prandtl-Glauert $\beta$ 修正):$A_1=0.3,A_2=0.7,b_1=0.14,b_2=0.53$。
- **我们 M<0.03(不可压)**:**首选 R.T. Jones 不可压 Wagner 常数(0.165/0.335/0.0455/0.3)**,理论最干净;L-B 可压常数作为对照。两者在 $M\to0$ 应趋于一致,差异即压缩性修正残差。

| 常数 | 值 | 含义 | 出处 |
|---|---|---|---|
| $\Psi_1,\Psi_2$ | 0.165, 0.335 | 不可压 Wagner 系数 | R.T. Jones;CONFIRMED 翼型无关 |
| $\epsilon_1,\epsilon_2$ | 0.0455, 0.3 | 不可压 Wagner 指数 | 同上 |
| $A_1,A_2$ | 0.3, 0.7 | L-B 可压缺函数系数 | Beddoes 1982;Bangga2020 |
| $b_1,b_2$ | 0.14, 0.53 | L-B 可压缺函数指数 | 同上 |
| $K_\alpha$ | 0.75 | 非环量因子 | Bangga2020 |
| $T_p$ | 1.7 | 压力滞后 | Leishman&Beddoes1989(NACA0012) |
| $T_f$ | 3.0 | 边界层/分离滞后 | 同上 |
| $T_v$ | 6.0 | 涡衰减 | 同上 |
| $T_{vl}$ | 6.0–7.0 | 涡迁移(半弦) | 原始 Westland=7.0;多数风机实现=6.0 |
| $K_v$ | 0.2 | 涡升力常数(部分形式) | Bangga2020 |
| $\eta$ | **0.95** | 切向力恢复因子(式19) | **CONFIRMED Bangga2020**(用户所引出处核实=本文式19所用标准值) |

**谱系**:原始值来自 Westland Helicopters 对 NACA0012 在 M=0.3、Re≈8×10⁶ 的非定常实验标定(Leishman&Beddoes 1989)。IST Lisbon 论文与 Melani et al. 2024(J. Phys. Conf. Ser. 2767:052053)复核:"BL standard"= $T_f$=3.0, $T_p$=1.7, $T_v$=6.0, 直接取自 L-B 的 NACA0012。**不同实现对 $T_p/T_f/T_v$ 的重调**(如 TNO $T_p$=2.5、FFA $T_p$=0.8/$T_f$=5.0)说明这些**本质是翼型/工况拟合值**,非第一性。

### B.1 哪些常数普适、哪些翼型相关

- **$A_1,A_2,b_1,b_2$(indicial/Wagner 常数)**:**普适、理论锚定**。Wagner 阶跃响应的两指数近似,源自不可压薄翼理论(R.T. Jones),**与翼型几乎无关**(Beddoes 1982 默认)。**CONFIRMED 可零拟合直接用**。
- **$T_p$(压力滞后)**:**基本翼型无关**。Bangga2020 引 Leishman&Beddoes(1989)明确"$T_p$ is largely independent of the airfoil shape"。**CONFIRMED 取默认 1.7**。
- **$T_f$(分离滞后)**:**翼型/几何敏感**。PLAUSIBLE 翼型相关。
- **$T_v,T_{vl}$(涡时间)**:**主要随马赫/工况**,翼型弱相关。PLAUSIBLE。
- **$f(\alpha)$ 的 $\alpha_1,S_1,S_2$**:**强翼型相关**,经典是拟合值 → **我们不走拟合**,见 B.2。

### B.2 ★零拟合核心:f(α) 由静态极曲线反演(替代 α1/S1/S2 拟合)

**CONFIRMED 且为推荐路径**。两篇一手源独立采用:

1. **Bangga2020 式33**(A.2):直接由静态粘性极曲线反演 $f_n$,Bangga 明确"avoid curve fitting… as long as the static polar data is available"。
2. **Risø-R-1354(Hansen/Gaunaa/Madsen 2004)式15**,同式
$$f^{st}=\left(2\sqrt{\frac{C_L^{st}(\alpha)}{C_{L,\alpha}(\alpha-\alpha_0)}}-1\right)^2 \tag{R15}$$
并给**完整鲁棒处理**(CONFIRMED,可直接抄):
   - 线性升力斜率 $C_{L,\alpha}=\max\{C_L^{st}(\alpha)/(\alpha-\alpha_0)\}$(附着区,式16);
   - 全分离角 $\alpha^{\pm fs}$ 由 $|C_L^{st}(\alpha^{\pm fs})|=|C_{L,\alpha}(\alpha^{\pm fs}-\alpha_0)|/4$ 定,之外 $f^{st}=0$;
   - 静态升力用**附着/全分离插值** $C_L^{st}=C_{L,\alpha}(\alpha-\alpha_0)f^{st}+C_L^{fs}(\alpha)(1-f^{st})$(式17),全分离升力 $C_L^{fs}=\frac{C_L^{st}-C_{L,\alpha}(\alpha-\alpha_0)f^{st}}{1-f^{st}}$(式18);
   - 低攻角全分离升力→附着升力之半(式19),避免 $f^{st}=1$ 奇异。

**结论**:**$\alpha_1/S_1/S_2$ 完全不需拟合**——用 UIUC 静态极曲线(SD7003 或实测 NACA2406)经式 R15–R19 反演 $f^{st}(\alpha)$。满足零拟合纪律(输入是静态极曲线,与已用 UIUC 极曲线同性质)。

### B.3 Sheng-Galbraith-Coton 可计算判据(替代翼型拟合 onset)

**CONFIRMED 出处**:Sheng, Galbraith & Coton, *"A New Stall-Onset Criterion for Low Speed Dynamic-Stall"*, J. Solar Energy Eng., 128(4):461–471, 2006, DOI 10.1115/1.2346703(Glasgow 数据库,M=0.12)。提出**滞后攻角**失速起始判据,替代 L-B 拟合的 $C_N^{CRIT}$。配套:*"A modified dynamic stall model for low Mach numbers"*, 130(3), 2008, DOI 10.1115/1.2931507。**判据精确数学式本文未逐字取得(GLA eprints 403)→ 标 UNKNOWN**,实施前取原文。**PLAUSIBLE:把 onset 从拟合 $C_N^{CRIT}$ 改为由静态失速角+马赫/Re 标度计算,零拟合 friendly。**

**扑翼更对口替代**:Ramesh LESP 判据(JFM 751, 2014)——临界 LESP 仅依赖翼型形状+Re、与运动学无关,**可直接用已有 LESP 门控**,无需拟合。**推荐:onset 用 LESP_crit(已有)替代 $C_N^{CRIT}$,绕开 Sheng 判据残余。**

### B.4 SD7003/薄弯度翼型 @Re1.5e5 的发表 L-B 常数

**UNKNOWN→大概率无直接发表**。L-B 常数标定文献集中于 NACA0012(Re~10⁶–10⁷)与厚风机翼型(S809/S814, t/c≥15%)。**SD7003/薄弯度翼型 @Re1.5e5 的专用 Tp/Tf/Tv/Tvl 未见发表**。→ **含义**:(a)indicial 常数与 $T_p$ 用普适默认(零拟合成立);(b)$T_f$ 用 NACA0012 默认 3.0 作初值,承认是薄翼族最近可得先验;(c)$f(\alpha)$ 全部来自 UIUC 静态极曲线反演(零拟合)。**残余风险:$T_f/T_v$ 的翼型敏感性是诚实不确定源,但不破坏零拟合纪律**(未对本机测力拟合)。

### B.5 诚实边界裁定

**L-B 常数本质是半经验拟合,非第一性**——CONFIRMED。但分两类:
1. **理论锚定、可零拟合**:indicial $A_1..b_2$(Wagner 理论)、$T_p$(翼型无关)、$f(\alpha)$ 全部(静态极曲线反演)、Kirchhoff 系数 $((1+\sqrt f)/2)^2$(平板势流理论)。
2. **半经验、需借发表先验**:$T_f,T_v,T_{vl},\eta$(用 NACA0012/薄翼族发表默认值)。

**裁定**:**用"静态极曲线反演 f(α) + 理论常数 + 发表薄翼族时间常数"是合规零拟合**——与"用 UIUC 极曲线"同性质(都是外部已发表数据,非对本机测力拟合)。**唯一需声明的近似**:$T_f/T_v$ 借用 NACA0012 值,对薄弯度翼型 @Re1.5e5 的迁移性是 PLAUSIBLE 而非 CONFIRMED,应做敏感性测试而非重新拟合。**$\eta=0.95$ 用 Bangga2020 发表值合规;但更稳妥走 Bangga 式70,直接用静态极曲线给 $C_T^{VISC}(\alpha_{f_n})$,连 $\eta$ 都不用。**

---

## C. 分条带 UVLM 环格集成先例与做法

### C.0 核心耦合原则(避免双计)——CONFIRMED

**L-B 替代截面力,不叠加到独立计算的升力上。** OpenFAST AeroDyn UA 理论(CONFIRMED,openfast.readthedocs.io UA theory):
- 动态失速模型**替换**静态系数为动态计算值,**不是**在独立升力上加修正。"The unsteady airfoil coefficients Cl,dyn, Cd,dyn, Cm,dyn are obtained from the states"(UAMod=4);Boeing-Vertol 模型失速不激活时回退 "Cl,dyn=Cl^st(α34)"。
- **有效/滞后攻角查极曲线**:UAMod=4 在等效攻角 $\alpha_E=\alpha_{34}(1-A_1-A_2)+x_1+x_2$ 处取值;Boeing-Vertol 用滞后攻角 $\alpha_{e,L}=\alpha_{Lag,L}$ 查静态极曲线。
- **HGM 4 状态形式**:$C_{l,circ}=x_4(\alpha_E-\alpha_0)C_{l,\alpha}+(1-x_4)C_{l,fs}(\alpha_E)$,$C_{l,dyn}=C_{l,circ}+\pi T_u\omega$(分离态 $x_4$ 加权附着/全分离)。

**对我们 UVLM 环格的含义(关键架构决策,PLAUSIBLE→需 S1 验证)**:
1. UVLM 环格负责**诱导速度/下洗 → 各条带有效攻角 $\alpha_{eff}$(含 3D 诱导、尾迹、自感)**。这是 UVLM 的本职,**保留**。
2. L-B 作为**截面力律**,输入 $\alpha_{eff}$ 历史,输出条带截面力 $C_N^D,C_T^D$(式28–32)。**环格不再对同一条带独立算环量升力**——否则双计。
3. **附着段非定常性的归属**:UVLM 环格**本身已捕获**附加质量与脱落尾涡的非定常环量(这是 UVLM 相对准定常的优势)。因此 **L-B 的 A.1 附着 indicial 项与 UVLM 的非定常环量功能重叠**。两种净处理:
   - **(i) L-B 全替代**:附着段也用 L-B($\alpha_e$ 经 indicial 缺函数),UVLM 只给几何/诱导 $\alpha_{eff}$。**风险**:UVLM 的尾迹记忆与 L-B 的 Wagner 缺函数双重描述同一物理(尾涡诱导的升力滞后)→ **潜在双计**。
   - **(ii) 分工(推荐)**:UVLM 给附着非定常环量(尾迹+附加质量,它本来就对),**L-B 只补 UVLM 缺的两块:后缘分离 $f$ 门控(A.2)+ LEV 涡升力(A.3)**。即:条带力 = UVLM 附着环量力 × Kirchhoff $f$ 衰减 + L-B 切向吸力衰减 + LEV $C_N^V$。**这样 A.1 的 Wagner 项不重复(UVLM 已含),A.2/A.3 是纯增量。** 这与"动态失速模型只修正分离/LEV、不改附着"的直升机自由尾迹实践一致(见 C.2)。

### C.1 风机:AeroDyn/OpenFAST、DUST、HAWC2

- **AeroDyn/OpenFAST**(CONFIRMED,见 C.0):BEM 给诱导攻角,L-B(Gonzalez/Minnema-Pierce/HGM 变体)替换截面系数。**BEM 是无尾迹记忆的诱导因子法,故 L-B 的 Wagner 项不双计**——这是 BEM 与 UVLM 的关键区别。
- **DUST**(Politecnico di Milano 非定常 VLM,PLAUSIBLE):非定常涡格 + 面元,耦合方式待子代理确认。**关键问题**:DUST 的 VLM 已含非定常尾迹,其动态失速耦合是否关闭附着 indicial——待补。
- **HAWC2**(Risø,CONFIRMED 自 Risø-R-1354):Risø 模型(式31–32)直接在 BEM 框架,静态极曲线反演 $f^{st}$,**BEM 无尾迹记忆故无双计**。
- **风机双计小结**:风机主用 **BEM**(诱导因子,无显式尾迹记忆)→ L-B 全三模块无重叠。**我们用 UVLM(有尾迹记忆)→ 必须选 C.0(ii) 分工,关闭 L-B 的 Wagner 附着项。** 这是风机先例**不能直接照抄**的根本原因。

### C.2 直升机:CAMRAD II / RCAS / UMARC(自由尾迹 + L-B)

- **CONFIRMED 原则**:直升机综合分析(CAMRAD II, Johnson;RCAS;UMARC)用**自由尾迹(free wake)+ 升力线 + L-B 截面动态失速**。自由尾迹已描述尾涡卷起与非定常诱导,**L-B 在此只提供截面分离/失速修正**(等效攻角查极曲线 + LEV),**附着非定常由自由尾迹承担**。→ **这正是 C.0(ii) 分工的成熟先例**:有显式尾迹的方法,L-B 退为"分离+LEV 修正",不开 Wagner 附着项。
- 精确的自由尾迹-L-B 接口公式(等效攻角定义)待子代理补。

### C.3 扑翼:L-B / 动态失速在分条带的先例

- **CONFIRMED 出处**(检索得):扑翼/扑翼 MAV 动态失速用 L-B 或类似半经验模型在 blade-element/升力线基的文献存在(Liu, Ansari, Bhatia, Djojodihardjo 等)。**扑翼 regime 特殊性**:
  - **advance ratio ~1、$k=0.13$–0.39**:落在 L-B 有效范围($k$ 上限~0.4),但接近边界。
  - **上下冲程负 $\alpha$ 侧**:L-B 对称处理负攻角分离,但薄弯度翼型负攻角行为与正攻角不对称(弯度)→ $f(\alpha)$ 反演必须用**全攻角范围**静态极曲线(含负攻角),UIUC 数据需覆盖。
  - **feathering 扭转相位**:L-B 的 $\alpha_e$ 滞后(式1)天然处理俯仰率 $\dot\alpha$ 引致的有效攻角偏移,feathering 相位通过 $\alpha(t)$ 历史进入。
  - **已知失效**:L-B 在**低 Re LEV 主导**(扑翼典型)时,其 LEV 模块(为直升机高 Re 设计)对 LEV 强度/脱落的描述偏简;Ramesh LESP 路线在低 Re 更准(我们已用 LESP 门控,优势)。
- 具体扑翼 L-B 实现细节(Ansari UVLM-dynamic stall)待子代理补。

### C.4 集成路径结论(对我们的配方)

**推荐架构(CONFIRMED 原则 + PLAUSIBLE 细节)**:
1. **UVLM 环格**:给各条带 $\alpha_{eff}(t)$(含 3D 诱导、尾迹记忆、附加质量)与附着环量升力——**保留现有**。
2. **后缘分离(A.2)**:用 $\alpha_{eff}$ 经 $T_p/T_f$ 滞后得 $f_{2_n}$,对 UVLM 附着升力乘 Kirchhoff 因子 $((1+\sqrt{f_{2_n}})/2)^2$,$f^{st}$ 由静态极曲线反演(零拟合)。**增量,无双计。**
3. **切向吸力(A.2 式19/70)**:深失速前缘吸力衰减 → 大阻。**增量。**
4. **LEV(A.3)**:LESP_crit 触发(已有),$C_N^V$ 按式21–23 累积/脱落,**加到条带力**。**增量。**
5. **关闭 L-B 的 Wagner 附着项(A.1 的 $X_n,Y_n$)**:因 UVLM 已含尾迹记忆,避免双计。**例外**:若发现 UVLM 准定常化(v4 现状)尾迹记忆不足,可在 S1 用 Wagner 项**临时**补附着滞后,但与 UVLM 环格并存时需标记双计风险。

**与现状 v4 的差异**:v4 是**准定常** UVLM + 静态 Kirchhoff 缩放(已证只给 1/3 且方向错)。本配方的增量 = **(a) $f$ 的时间滞后动力学($T_p/T_f$)替代静态 Kirchhoff** + **(b) LEV 涡升力** + **(c) 切向吸力 $\sqrt f$ 衰减**。三项都是 v4 没有的相位敏感动力学。

---

## D. 与三病灶指纹的对照

判据:每个病灶标注 (a) L-B 哪个模块贡献、(b) 方向是否对、(c) 量级是否够。所有机理均 CONFIRMED 自 A 节公式;量级为 PLAUSIBLE 估计(需 S1 数值验证)。

### 病灶1:dL/df 升力随频率(实测 +1.5 N/Hz,模型 −0.4,反号)

- **贡献模块**:**A.1 附着非定常(indicial)+ A.3 LEV 涡升力**。物理:高 $\alpha_{eff}$ 下,随拍频 $f$↑,reduced freq $k=\pi f c/U$↑,动态失速的**升力维持**增强——(i) indicial 缺函数使 $\alpha_e$ 滞后于 $\alpha$,在失速附近等效"延迟失速",周期均升↑;(ii) LEV(式21–23)在越临界后累积涡升力 $C_N^V$,其峰值随 $k$ 增大而增大(涡在下冲程前来不及脱落)。
- **方向**:✅ 正。准定常 UVLM 无此记忆效应故给反号;L-B 的滞后 + LEV 累积正是把 −0.4 翻向 + 的机制。
- **量级**:PLAUSIBLE 够。动态失速升力超调典型为静态失速升力的 1.2–1.5×(文献 $C_{L,max}$ 超调 20–50%);LEV 贡献在 $k\sim0.2$–0.4 区间随 $k$ 单调增。**关键依赖**:LEV 模块(A.3)必须启用——只开附着+后缘分离(A.1+A.2)而无 LEV,升力维持会明显不足(后缘 Kirchhoff 是**降**升力)。**dL/df 是检验 LEV 模块是否接对的首要指纹。**

### 病灶2:dT/dU 推力随速度(实测 +0.5,模型 −0.2)

- **贡献模块**:**A.2 后缘分离 Kirchhoff-f + 滞后(式18)+ 诱导/压差阻(Risø 式26–27)**。物理:U6 时周期最大攻角越失速 → $f$ 塌缩 → **分离压差阻** $C_D^{Kirchhoff}=C_{L,\alpha}\alpha^2((1-\sqrt f)/2)^2$(Risø 式25)大;U10 时全 feathering 附着 → $f\to1$ → 分离阻塌缩。**阻力随 U 增大而下降**,推力(=前向分量)相对上升 → $dT/dU>0$。
- **方向**:✅ 正。准定常模型对分离阻的 U 依赖捕捉不足;L-B 的 $f(U,\alpha_{eff})$ 通过静态极曲线反演,自然给出"低速深分离高阻、高速附着低阻"。
- **量级**:PLAUSIBLE 够但**依赖 $C_T$/阻力的正确处理**。注意:若只用 Kirchhoff 缩升力而阻力走静态极曲线,会低估分离阻的 U 敏感性。**必须用 Risø 式26(非定常压差阻增量)或 Bangga 式70(静态 $C_T^{VISC}(\alpha_{f_n})$)给阻力**,才能让 $dT/dU$ 的分离阻塌缩项出来。

### 病灶3:aoa15 甩阻保升(深失速前缘吸力衰减 + feathering 恢复 → 推力随扭转中间峰)

- **贡献模块**:**A.2 切向吸力独立衰减(式19 $C_T^f=-\eta\frac{dC_N}{d\alpha}\alpha_e^2\sqrt{f_{2_n}}$)+ A.1 攻角滞后**。物理:(i) 深失速时 $\sqrt{f_{2_n}}\to0$ → **前缘切向吸力 $C_T$ 衰减** → 大阻(实测甩阻);(ii) feathering 扭转在推力相位**减小 $\alpha_e$** → $f$ 回升 → $\sqrt f$ 恢复吸力 → 推力回升。扭转中间峰 = $\alpha_e$ 既够小恢复 $f$、又够大保有 $C_N$ 折投影的最优。
- **方向**:✅ 正。式19 的 $\sqrt f$ 门控正是"前缘吸力随分离衰减"的解析表达,与我们 LESP 门控同源(前缘吸力 $\propto$ 附着度)。
- **量级**:PLAUSIBLE 够。**但式19 是经验衰减律**,$\eta=0.95$ 来自 NACA0012。Bangga 指出一阶 LB 的阻力(经 $C_T$)不准,故其二阶 IAG 模型改用式70 静态极曲线插值 $C_T$。**建议:S1 先用式19($\eta$=0.95)快速接,若 aoa15 阻力量级不对则切换式70(静态极曲线给 $C_T$),零拟合且更鲁棒。**

### 三病灶同源性与一次性改善可行性

三病灶**共同根源 = 缺相位敏感的分离点动力学 $f(\alpha_{eff},t)$ 及其对法向力/切向吸力的门控**。L-B 的 A.2(Kirchhoff-f + 双滞后 $T_p/T_f$)提供 $f$ 动力学,A.1 提供 $\alpha_e$ 滞后,A.3 提供 LEV 升力维持。**一次实现 A.1+A.2+A.3 可同时改善三病灶**——其中:
- dL/df 主靠 **A.3(LEV)**;
- dT/dU 主靠 **A.2(分离阻的 U 依赖)**;
- aoa15 主靠 **A.2 切向吸力 $\sqrt f$ 衰减 + A.1 滞后**。
**结论(PLAUSIBLE):三病灶可一次性改善,但 dL/df 对 LEV 模块最敏感、aoa15 对切向力处理方式(式19 vs 式70)最敏感,是分阶段验收的自然切分点。**

---

## E. 推荐分阶段实现里程碑

原则:**baseline first,一次一变量**。在现有生产模型 v4(准定常 UVLM 环格 + LESP门控 + Garrick前缘吸力 + Kirchhoff几何失速 + UIUC极曲线)上,按"先静态反演、再附着滞后、再分离动力学、最后 LEV"递进。每级给 canonical 案 + 验收门。**canonical 案一律用文献 benchmark,不用本机测力拟合**(零拟合纪律);本机测力只作最终对照。

### S0 — 静态 f(α) 反演 + Kirchhoff 分离(替代现有"几何失速")
- **实现**:用 UIUC SD7003/NACA2406 静态极曲线,按 Risø 式15–19 反演 $f^{st}(\alpha)$,替换现有 Kirchhoff 几何失速缩放;切向力用式19($\eta$=0.95)或式70(静态极曲线)。
- **canonical 案**:静态极曲线重构——反演的 $C_L^{st}=C_{L,\alpha}(\alpha-\alpha_0)f^{st}+C_L^{fs}(1-f^{st})$ 必须**逐点回到输入极曲线**(Risø 式17,机器精度)。
- **验收门**:$f^{st}(\alpha)$ 单调、$\alpha=0$ 时 $f^{st}=1$、$\alpha\ge\alpha^{fs}$ 时 $f^{st}=0$;静态 $C_L$/$C_D$ 复现 UIUC 极曲线误差 <1%。
- **零拟合**:✅ 纯静态极曲线反演。

### S1 — 附着非定常(indicial,A.1)
- **实现**:接入 Wagner 缺函数 $X_n,Y_n$(式2–3,$A_1..b_2$ 用 B.0 默认),得 $\alpha_e$ 与环量 $C_N^C$(式6,**保留弯度 $\alpha_0^{INV}$ 项**);低马赫下非环量项走薄翼附加质量或由 UVLM 环格承担(见 C 节双计)。
- **canonical 案**:不可压薄翼 **Wagner 阶跃响应** + **Theodorsen 振荡**——单条带 $C_L$ 对阶跃/谐波攻角的解析解。
- **验收门**:阶跃升力 build-up 与 Wagner 函数 $1-A_1e^{-b_1s}-A_2e^{-b_2s}$ 吻合 <2%;谐波响应幅值/相位与 Theodorsen $C(k)$ 在 $k=0.13$–0.39 全程 <5%。
- **零拟合**:✅ 理论常数。

### S2 — 后缘分离动力学(Tp/Tf 滞后,A.2)→ 攻 dT/dU + aoa15
- **实现**:接入压力滞后 $D_p$(式11,$T_p$=1.7)→ $\alpha_f$(式12);分离点滞后 $D_f$(式17,$T_f$=3.0)→ $f_{2_n}$;非定常粘性法向力(式18)+ 切向吸力衰减(式19/70)。
- **canonical 案**:文献动态失速 ramp/振荡 benchmark——**Sheng-Galbraith-Coton Glasgow 数据库** 或 **NACA0012 深失速振荡**(McAlister/Carr,公开)的 $C_L$/$C_D$ 滞回环。
- **验收门**:深失速 $C_L$ 滞回环方向正确(上支超调、下支 undershoot);分离阻随等效攻角增大而增(甩阻定性);**对本机:dT/dU 反号消除(模型由 −0.2 转向 ≥0),aoa15 出现扭转中间峰**。
- **零拟合**:✅ $T_p$/$T_f$ 用 NACA0012 默认 + $f(\alpha)$ 静态反演;敏感性测试 $T_f\in[2,5]$。

### S3 — LEV 涡升力(A.3)→ 攻 dL/df
- **实现**:接入涡升力 $C_N^V$(式21–23,$T_v$=6.0,$T_{vl}$=6–7)、涡时钟 $\tau_v$(式24/35)、onset 判据(**用已有 LESP_crit 替代 $C_N^{CRIT}$**,B.3);压力中心迁移(式26–27)给力矩。
- **canonical 案**:Ramesh/LESP 文献的**间歇 LEV 脱落**案(plunging/pitching 翼型,$k$ 扫描)——LEV 触发阈值与运动学无关性(JFM 751 核心结论)。
- **验收门**:LEV 触发攻角随 $k$ 单调后移;**对本机:dL/df 由 −0.4 翻正(目标符号一致,量级向 +1.5 N/Hz 收敛)**;周期均升随 $f$ 增。
- **零拟合**:✅ $T_v$/$T_{vl}$ 用文献默认;onset 用 LESP_crit(已有,零拟合)。

### S4 — 三病灶联合验收 + 全机对照
- **canonical 案**:本机 RoboEagle 三工况(dL/df 频率扫描、dT/dU 速度扫描、aoa15 扭转扫描)——**只作对照,不回填拟合**。
- **验收门**:三趋势**符号全部正确**(dL/df>0、dT/dU>0、aoa15 推力随扭转中间峰);量级进入实测的 0.5–2× 区间。
- **零拟合**:✅ 全程未对本机测力拟合任何常数。

**关键路径**:S0→S1→S2 可在 2 个迭代内落地(纯截面模型,先单条带跑通再接 UVLM 分条带);**S3 的 LEV 是 dL/df 的钥匙,但若 S2 已使 dT/dU/aoa15 翻正,可独立先交付 S2**。**最大不确定**:C 节的 UVLM 双计处理决定 S1 非环量项与附着环量是否与环格重叠——见 C 节裁定后再定 S1 细节。

---

## 引用(全)

1. Leishman, J.G. & Beddoes, T.S., *"A Semi-Empirical Model for Dynamic Stall"*, J. American Helicopter Society, 34(3):3–17, 1989.
2. Bangga, G., Lutz, T. & Arnold, M., *"An improved second-order dynamic stall model for wind turbine airfoils"*, Wind Energ. Sci. (Discuss.), 2020, DOI 10.5194/wes-2020-75.
3. Hansen, M.H., Gaunaa, M. & Madsen, H.A., *"A Beddoes-Leishman type dynamic stall model in state-space and indicial formulations"*, Risø-R-1354(EN), Risø National Laboratory, 2004.
4. Gupta, S. & Leishman, J.G., *"Dynamic stall modelling of the S809 aerofoil and comparison with experiments"*, Wind Energy, 9(6):521–547, 2006.
5. Sheng, W., Galbraith, R.A.McD. & Coton, F.N., *"A New Stall-Onset Criterion for Low Speed Dynamic-Stall"*, J. Solar Energy Eng., 128(4):461–471, 2006, DOI 10.1115/1.2346703.
6. Sheng, W., Galbraith, R.A.McD. & Coton, F.N., *"A modified dynamic stall model for low Mach numbers"*, J. Solar Energy Eng., 130(3), 2008, DOI 10.1115/1.2931507.
7. Ramesh, K., Gopalarathnam, A., Granlund, K., Ol, M. & Edwards, J., *"Discrete-vortex method with novel shedding criterion for unsteady aerofoil flows with intermittent leading-edge vortex shedding"*, J. Fluid Mech., 751:500–538, 2014, DOI 10.1017/jfm.2014.297.
8. Beddoes, T.S., *"Practical computation of unsteady lift"*, Vertica, 8(1):55–71, 1984(及 1982 原始 indicial 常数)。

---

## B.6 零拟合常数来源专项核实(2026-07-20,第二代理独立复核)

与 B 节裁定一致,补精确配方与诚实边界:

**常数分类(决定零拟合地位)**:
- **理论/普适(零拟合绝对成立)**:indicial A1=0.3,A2=0.7,b1=0.14,b2=0.53(Wagner 两指数,
  薄翼理论,非翼型拟合;我们不可压 M<0.03 首选 R.T. Jones 不可压值 0.165/0.335/0.0455/0.3)。
- **静态极曲线反演(合法,类比已用 UIUC)**:f(α) 的 α1/S1/S2 经 Kirchhoff 反演自静态极曲线
  (Risø-R-1354 式15/Munduate 式,OpenFAST 同款,textbook 接受)——非对本机动态数据拟合。
- **翼型校准拟合(诚实弱点,需声明)**:Tp=1.7,Tf=3.0,Tv=6.0,Tvl=6–7(NACA0012 @M0.3 拟合;
  Mert/Pereira 重调分散 {0.8/5/2}/{1.5/5/6} 证明是拟合非第一性)。**SD7003/薄弯度 @Re1.5e5
  无发表 L-B 常数(UNKNOWN,真缺口)**。Sheng 判据的 α_ds0/T_α 也仍是翼型/Re 拟合。
- **η=0.95 CONFIRMED**(Bangga2020 Table2 + Técnico Lisboa + par.nsf.gov 三方);但 Bangga
  明示一阶 LB 用 η 给阻力不准,其二阶 IAG 改用式70 静态极曲线查 CT+ζ_v=0.76 限幅——
  **零拟合阻力首选式70 极曲线查表,不用 η**。

**诚实措辞裁定(写报告/论文时用)**:"zero-fit to own measurements; literature-inherited
dynamic time constants + polar-derived separation curve"。**不可称 L-B 时间常数为第一性**
(RSER 2024 综述+Mert/Pereira 分散证明是经验拟合);声明 Tp/Tf/Tv/Tvl 取 NACA0012 经典集
{1.7/3/6/6-7}(与 UIUC 极曲线同认识论地位),f(α) 自静态极曲线反演,时间常数为模型
主要"继承性经验主义"局限,以 Mert/Pereira/LB 三分散的敏感性测试作严谨缓解。
