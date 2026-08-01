# N2 弦向分离压力与统一面板载荷：一手文献扩展裁决

日期：2026-07-29  
范围：只做步骤②学科机理与步骤③方向裁决；不改代码、YAML、常数或 claim 状态。

## 0. 裁决摘要

1. **N2.5 当前生产 NO-GO。** 本轮没有找到 NACA 2406、
   \(Re\approx1.5\times10^5\) 的一手实验静态 \(C_T(\alpha)\) 或
   \(C_p(s,\alpha)\) 数据。Bangga Eq. (62) 不是无数据的弦向力模型，而是
   直接查目标静态黏性 \(C_T\) 极曲线；Eq. (64) 也仍需目标静态
   \(C_D\)，并含观测得到的限幅常数。缺少目标极曲线时，该路线没有输入，
   不能用代理翼型、XFOIL 或 Fig18 残差补出。
2. **即使以后取得目标静态 \(C_T\)，N2.5 也只能是截面级诊断/过渡替代件。**
   一个 \(C_T\) 标量不能唯一决定上下表面的 \(C_p(s)\)，因而不能满足
   co-design 所需的逐面板载荷、力矩和结构虚功。
3. **现有 N2.2 是错件。** 全局风升力向的 separation chop 不能代表实际
   厚翼表面上的弦向分离压力，也不能生成唯一的压力中心；它至多保留为
   V4.1 的基线闭合，不应被继续修补。
4. **下一合法生产方向属于 N2.6，而非 N2.5。** 一手文献支持的是：
   双侧、实际厚度、非定常黏性边界层状态通过质量亏损/位移效应强耦合回同一
   外流求解器；N2 负责壁面状态和分离释放，N3 负责已经释放的空间涡态；
   二者只通过一份统一 Bernoulli 面板压力形成载荷，不各自添加第二套力。
5. **LESP 的角色必须锁定。** \(A_0/\mathrm{LESP}\) 可描述附着薄翼的前缘
   吸力以及起涡临界/剪切层供给门，但不能直接充当持续 LEV 载荷幅值。
   持续涡载荷还需要涡环量、位置及相对翼面运动。

## 1. NACA 2406 目标静态数据审计

### 1.1 检索范围

本轮检索了以下一手/官方入口：

- NASA NTRS 的 NACA 技术报告；
- UNT Digital Library 的美国政府 NACA 文献镜像；
- UIUC Low-Speed Airfoil Tests 的已发表实验翼型清单；
- 以 `NACA 2406`、`150000`、`pressure distribution`、`Cp`、
  `drag`、`polar` 为联合条件的全文/题名检索；
- 公开翼型极曲线库中数据来源的实验/计算身份。

“未找到”严格表示：**在上述可审计公开一手来源中，本轮没有发现满足目标
翼型、目标 Reynolds 数及静态 \(C_T/C_p\) 三项联合身份的数据**；不声称
世界上从未进行过此类试验。

### 1.2 找到的直接 NACA 2406 一手数据

Jacobs 与 Ward 的 NACA TN 404（后纳入 NACA TR 460）给出了 NACA 2406 的
升力、阻力和四分之一弦矩表，但条件是

\[
Re=3.12\times10^6,
\]

约为目标 Reynolds 数的 20 倍。该表是积分 \(C_L/C_D/C_m\)，不是上下表面
\(C_p(s)\)。因此它既不能锚定低 Reynolds 数的层流分离泡、转捩和压差阻，
也不能提供逐面板压力。

UIUC 低速实验目录覆盖约 \(6\times10^4\) 到 \(5\times10^5\)，但其已发表
测试翼型清单不含 NACA 2406。常见 AirfoilTools 目录也没有 NACA 2406 的
实验条目；其他检索到的 NACA 2406 低 Reynolds 数曲线是 XFOIL/CFD 产物，
不是本候选要求的一手实验锚。

### 1.3 对 N2.5 的直接后果

当前禁止以下替换：

- 用 NACA 2408、2412、SD7003 或其他翼型的 \(C_T\) 代替；
- 把 \(Re=3.12\times10^6\) 的 NACA 2406 极曲线外推到
  \(Re\approx1.5\times10^5\)；
- 把 XFOIL 输出称为“静态数据锚”；
- 从 Fig17/18/19 的力残差反演静态 \(C_T\)；
- 以 Sheng 的 \(E_0\)、Bangga 的 0.76/1.2 或新的常数吸收残差。

以上任一做法都会把“有来源的机理候选”改成“以文献符号包装的目标拟合”。

## 2. Bangga Eq. (62)–(64) 的方程级核验

Bangga、Lutz 与 Arnold 的 IAG 模型首先记录了原 L-B 弦向力

\[
C_T^f
=-\eta\frac{\mathrm d C_N}{\mathrm d\alpha}
\alpha_e^2\sqrt{f_2},
\tag{19}
\]

并指出阻力对 \(\eta\) 高度敏感。其替代式是

\[
C_T^f=C_T^{\mathrm{VISC}}(\alpha_f).
\tag{62}
\]

这里 \(C_T^{\mathrm{VISC}}\) 是**输入的静态黏性极曲线**，不是由
\(f_2\)、LESP 或外流解现场推导出的量；\(\alpha_f\) 只是滞后攻角。论文还
明确说明相关动态失速模型以静态极曲线和动态参数为输入，并以静态
\(C_N^{\mathrm{VISC}}\) 反演 separation factor。

Eq. (63) 定义

\[
\zeta_n=
\frac{1}{\pi}\frac{\mathrm dC_N}{\mathrm d\alpha}
\left(\frac{1+\sqrt{f_n}}{2}\right)^2,
\tag{63}
\]

但它只用于决定何时启用经验 drag limiter，并不生成缺失的 \(C_T\)。
Eq. (64) 在小角度区域把动态阻力限制到静态阻力或其 1.2 倍：

\[
C_D^D\leftarrow C_D^{\mathrm{VISC}}
\quad\text{或}\quad
1.2\,C_D^{\mathrm{VISC}},
\tag{64}
\]

门限采用 \(\zeta_v=0.76\)。作者明确写明：

- 0.76 可随翼型的 vortex-lift 对阻力影响而改变；
- 实现时建议增加 relaxation 以避免不连续；
- 关键失速角仍从静态极曲线中的矩突变或阻力上升读出。

因此，在没有目标 \(C_T^{\mathrm{VISC}}\) 和
\(C_D^{\mathrm{VISC}}\) 时：

- Eq. (62) 无法求值；
- Eq. (63) 不能补出 Eq. (62) 的输入；
- Eq. (64) 仍无法求值；
- 0.76 与 1.2 不能被当作跨翼型机理常数。

### 2.1 已验证适用域

Bangga 的跨翼型比较使用 S801、NACA4415、S809 和 S814，厚度比为
13.5%–24%，带前缘 grit，\(Re\approx7.5\times10^5\)。主要频率研究为
\(k=0.036,0.073,0.111\)，并以二维俯仰翼型的积分
\(C_L/C_D/C_m\) 验证。

RoboEagle 目标则是 6% 厚 NACA 2406、低 Reynolds 数、自然转捩与显著三维
扑翼/扭转，且需要逐面板载荷。几何、转捩、Reynolds 数、运动学和输出层级
均越出 Bangga 的验证域。

### 2.2 严格裁决

Bangga 支持的命题只有：

> 原 L-B 的 \(\eta\)-型弦向力公式对阻力高度敏感；当目标静态黏性极曲线
> 已知时，可以把 lagged-static \(C_T\) 用作一个工程截面闭合。

Bangga 不支持：

> 在没有目标静态极曲线时，Eq. (62)–(64) 可以预测 NACA 2406 的低-Re
> 弦向分离压力或逐面板压力。

故 N2.5 保持生产 NO-GO。

## 3. Sheng 对“旧件错误”的证据，而非可移植候选

Sheng、Galbraith 与 Coton 将原 L-B 弦向力改写为

\[
C_C=
\eta C_{N\alpha}(\alpha-\alpha_0)^2
\left(\sqrt{f'}-E_0\right),
\tag{5b}
\]

并由

\[
C_d=C_N\sin\alpha-C_C\cos\alpha+C_{d0}
\tag{5c}
\]

得到阻力。引入 \(E_0\) 的目的，是允许 fully separated deep stall 中出现
负弦向力；文中 \(E_0\approx0.2\) 是对多种 NACA 翼型的经验观察，
\(\eta\) 仍可调。

这篇一手论文能够支持：

- 原 L-B 的 chordwise component 在大范围分离下结构不充分；
- normal 与 chordwise force 必须分别建账。

它不能支持：

- 将 \(E_0=0.2\) 移植到 NACA 2406、目标 Re 和三维扑翼；
- 将一个截面 \(C_C\) 唯一分配到实际厚翼面板；
- 在已有 UVLM 压力、前缘吸力和 profile drag 上再叠加 \(C_C\)。

因此 Sheng 是 N2.2/N2.5 的**反证和边界证据**，不是本轮候选。

## 4. LESP、前缘吸力与持续涡载荷必须分账

### 4.1 薄翼前缘边界项

Garrick 薄翼理论中的前缘涡片奇性可写为

\[
S=\lim_{x\rightarrow0}\frac{1}{2}\gamma(x)\sqrt{x},
\qquad
F_S=\pi\rho S^2,
\qquad
C_S=2\pi A_0^2.
\]

该 \(A_0^2\) 项是零厚度附着势流的前缘边界项。在稳态势流中，它与薄翼法向
压力的流向投影共同满足零阻力。它不是一条弦向 separated-pressure 分布，
也不能任意涂抹到若干前缘面板。

### 4.2 LESP 的起涡角色

Ramesh 等的 LESP 离散涡方法在瞬时 \(A_0\) 超过临界值时触发前缘放涡；
一旦涡已经释放，模型显式推进该涡的环量和位置，并从整个涡系统求压力/力。
这正好把两个角色分开：

- \(A_0/\mathrm{LESP}\)：边缘临界和剪切层供给条件；
- 已释放 LEV：具有独立 \(\Gamma_v\)、位置和运动历史的空间状态。

Deparday 等进一步表明，有限厚度前缘、停滞点和剪切层高度都会改变 LESP 的
解释，分离还是渐进过程。这排除了“一个固定 \(A_0\) 直接给持续正涡力”的
普适用法。

### 4.3 对 FLUXV 的所有权约束

- 若生产仍使用零厚度中弧面压力跳，则 Garrick/LESP suction 必须是明确的
  **edge boundary term**，并与 N3 的已释放涡载荷分开。
- 若生产升级为实际厚度上/下表面 Bernoulli 压力，则有限鼻部压力已包含
  前缘吸力效应；不得再额外叠加 \(2\pi A_0^2\)。
- 在两种表示之间必须有互斥 ownership flag；不允许同时记账。

## 5. N2 separated-pressure 与 N3 vortex-load 的唯一边界

Hirato 的有限翼离散涡片模型给出了必要的空间状态和压力通路：

- 自由 LEV 对 no-penetration 的诱导速度进入同一外流解；
- 活跃条带上的势跳包含自由 LEV 环量与 bound circulation；
- 非定常压力含 \(\mathrm d\Gamma_{\mathrm{LEV}}/\mathrm dt\) 和
  \(\mathrm d\Gamma_b/\mathrm dt\)；
- 自由涡环再按局部诱导速度对流。

因此 N3 的涡载荷不是 “LESP 乘一个幅值”，而是空间
\(\Gamma+\)几何+\)时间历史通过统一压力算子产生的载荷。

另一方面，Terrington、Hourigan 与 Thompson 对移动无滑移固壁给出

\[
\boldsymbol{\sigma}
=\hat{\boldsymbol s}\times
\left[
\frac{\mathrm d\boldsymbol u_{\mathrm{wall}}}{\mathrm dt}
+\nabla\left(\frac{p}{\rho}+\Phi_g\right)
\right],
\tag{4.30}
\]

其结论是：相对无黏加速度/切向压力梯度和壁面加速度建立界面环量，黏性负责
把环量从界面层转移进流体。该式是 wall-vorticity ledger，不是由涡量库存
反求同一压力的闭合。如果一边用压力梯度生成 N2 涡量、另一边又用该涡量
直接补回相同压力，而没有联立外流—边界层方程，就形成循环论证。

因此生产所有权应为：

| 节点 | 拥有的状态/事件 | 不允许拥有 |
|---|---|---|
| N2.6 | 双侧边界层质量/动量亏损、壁面剪切、转捩、分离流形、释放通量 | 释放后 LEV 的持续空间载荷；直接残差力 |
| N3 | 已释放剪切层/LEV 的 \(\Gamma\)、位置、形变和对流 | 壁面黏性库存；第二套独立总力 |
| 统一压力算子 | N1/N2/N3 一致速度势与唯一双侧 \(p(s,t)\) | 从总力反演压力分布 |

在 N2/N3 接口上，环量、质量和动量释放必须守恒；在载荷端只积分一次压力和
剪切牵引。

## 6. 为什么 impulse/vortex-force 不能替代面板压力

Li 与 Wu 的 vortex-force 研究表明，同一涡的力贡献随其相对翼型的位置而变；
LEV 可形成上表面吸力，TEV 的诱导速度还会改变下表面压力。故环量标量本身
不够。

Graham、Pitt Ford 与 Babinsky 的冲量方法进一步指出，只有不完整的外场
涡量时，body vortex sheet 对正确总力有重要贡献。冲量表达给出的是整体
力/矩约束；对多物体时甚至不能自动分解到每一物体，更不能唯一反演每个面板
的 \(C_p\)。

所以 impulse/vortex-force 可以作为：

- 独立总力守恒 observer；
- N3 空间涡态的因果诊断；
- 统一压力积分的交叉核验。

它不能作为：

- 将一个总力重新分配到面板的算法；
- 与 Bernoulli 面板压力并列的第二力源；
- N2 缺失 pressure closure 的替代品。

## 7. 下一合法方向：N2.6 强黏性—无黏性相互作用

Riziotis 与 Voutsinas 的一手方法把 panel/vortex-wake 外流与积分边界层强耦合：

- 边界层质量亏损/位移厚度形成 transpiration boundary condition；
- 该边界条件返回同一外流面板系统；
- 分离位置由边界层状态产生，并释放第二条分离尾迹；
- 分离尾迹的诱导速度再反馈表面压力；
- 内外流方程非线性同时求解，而不是求完势流后加一个力系数。

在二维符号下，其核心耦合可概括为

\[
w_e\cdot n=\frac{\mathrm dM}{\mathrm ds},
\]

其中 \(M\) 是边界层/尾迹质量亏损；三维移动曲面上必须推广为带面积
Jacobian 的表面散度/ALE 守恒式。该路径能自然得到局部表面压力和壁面剪切，
因而在输出层级上符合 co-design。

但这篇方法的验证主要是中高 Reynolds 数二维俯仰翼型，使用积分边界层剖面、
转捩、摩擦和耗散闭合；论文自身也报告阻力仍可低估。它支持的是
**强耦合架构**，不证明其中任一经验闭合可以直接移植到 RoboEagle。

### 7.1 对 claim tree 的方向判定

- N2.2：组件角色错误；不再继续修补 global wind-lift chop。
- N2.5：缺目标静态输入，生产 NO-GO；以后有数据也只作截面诊断/过渡件。
- N2.6：缺生产组件，方向 GO；现有三维 IBL 守恒骨架属于正确层级，下一
  证据缺口是独立场数据约束的物理闭合及其与统一实际厚度压力算子的强耦合。
- N3：空间涡态方向不变；不得把 N2 的 wall/IBL 状态与 N3 的 free-vortex
  状态合并成一个旧标量。

## 8. 可证伪机理命题与预登记边界

### P1：N2.5 静态极曲线候选

> 目标 NACA 2406、目标 Re 的静态 \(C_T\) 在 lagged angle 处足以构成
> 生产逐面板载荷。

当前状态：**生产 NO-GO**。  
理论反证：一个积分 \(C_T\) 对 \(C_p(s)\) 的映射非唯一。  
数据反证：目标静态 \(C_T/C_p\) 锚尚未找到。

即使以后取得静态极曲线，也只能验证截面总量，不能使该命题晋升为 co-design
生产载荷。

### P2：LESP 持续载荷命题

> \(A_0/\mathrm{LESP}\) 可直接决定已经形成的 LEV 持续载荷幅值。

状态：**falsified，禁止重走**。  
Ramesh/Hirato 的载荷还显式依赖释放后涡的环量、位置和运动；有限厚度及
剪切层高度还改变 LESP 本身。

### P3：N2.6 强相互作用候选

> 在固定 N1/N3 状态下，经独立场数据验证的双侧 IBL 质量/动量亏损状态，
> 通过守恒 transpiration 边界条件耦合回同一实际厚度外流算子，可以形成
> N2 所缺的局部分离压力变化，而不需要添加目标力系数。

允许的最小 shadow 必须预先通过：

1. 零 IBL 亏损严格返回原 N1 压力，且新增力为零；
2. 上/下表面、前缘边界项和 N2/N3 释放所有权唯一；
3. 面板压力积分、冲量 observer 与虚功账在离散误差内一致；
4. 网格、时间步和内外流迭代收敛；
5. 用独立 \(C_p(s,t)\)、近壁速度/厚度/剪切指标验机理，不以 Fig17/18/19
   的 L/T 作为闭合输入；
6. 只有场级门通过，才允许进入三图代表工况，再决定晋升或 falsified。

以下任一情况立即证伪该具体候选：

- 需要由目标推力/升力选择系数或边界层状态；
- 只有总力改善而 \(C_p\) 拓扑、相位或压力中心错误；
- N2 与 N3/前缘吸力出现重复记账；
- 结果对网格、尾迹切口、时间步或内外流迭代不收敛；
- 目标域外的二维/高-Re 闭合被无证据移植。

### P4：VES 必要性

DeVoria 与 Mohseni 的 vortex-entrainment sheet 具有面质量、内禀切向动量和
法向速度跳，可承载压力跳；零 entrainment 时退化为传统无质量涡片。它说明
普通涡片在需要质量/动量卷吸时可能缺状态，但不证明 RoboEagle 的 N2 必须
立即启用完整 VES。

当前裁决：**保持 OPEN，不是下一单一候选。** 只有质量无关的 N2.6 强相互
作用/释放片通过收敛门后，独立场数据仍显示非零面质量、卷吸或内禀动量缺项，
才允许把 VES 晋升为待测候选。

## 9. 一手来源

- Jacobs, E. N. & Ward, K. E., *Tests of N.A.C.A. Airfoils in the
  Variable-Density Wind Tunnel, Series 24*, NACA TN 404 (1932),
  [NACA 2406 数据页](https://digital.library.unt.edu/ark:/67531/metadc54043/m1/15/).
- Jacobs, E. N., Ward, K. E. & Pinkerton, R. M., *The Characteristics of
  78 Related Airfoil Sections from Tests in the Variable-Density Wind
  Tunnel*, NACA TR 460 (1933),
  [NASA NTRS](https://ntrs.nasa.gov/citations/19930091108).
- UIUC Applied Aerodynamics Group,
  [Low-Speed Airfoils Tested](https://m-selig.ae.illinois.edu/uiuc_lsat_airfoilsTested.html).
- Leishman, J. G. & Beddoes, T. S., *A Semi-Empirical Model for Dynamic
  Stall*, Journal of the American Helicopter Society 34(3) (1989),
  DOI: [10.4050/JAHS.34.3](https://doi.org/10.4050/JAHS.34.3).
- Sheng, W., Galbraith, R. A. McD. & Coton, F. N., *A Modified Dynamic
  Stall Model for Low Mach Numbers*, Journal of Solar Energy Engineering
  130, 031013 (2008),
  DOI: [10.1115/1.2931509](https://doi.org/10.1115/1.2931509).
- Bangga, G., Lutz, T. & Arnold, M., *An Improved Second-Order Dynamic
  Stall Model for Wind Turbine Airfoils*, Wind Energy Science 5,
  1037–1058 (2020),
  [full text](https://wes.copernicus.org/articles/5/1037/2020/).
- Garrick, I. E., *Propulsion of a Flapping and Oscillating Airfoil*,
  NACA TR 567 (1936),
  [NASA NTRS](https://ntrs.nasa.gov/citations/19930091642).
- Ramesh, K., Gopalarathnam, A., Granlund, K., Ol, M. V. & Edwards, J. R.,
  *Discrete-Vortex Method with Novel Shedding Criterion for Unsteady
  Aerofoil Flows with Intermittent Leading-Edge Vortex Shedding*,
  JFM 751, 500–538 (2014),
  DOI: [10.1017/jfm.2014.297](https://doi.org/10.1017/jfm.2014.297).
- Deparday, J., He, X., Eldredge, J. D., Mulleners, K. & Williams, D. R.,
  *Experimental Quantification of Unsteady Leading-Edge Flow Separation*,
  JFM 941, A60 (2022),
  DOI: [10.1017/jfm.2022.319](https://doi.org/10.1017/jfm.2022.319).
- Hirato, Y., Shen, M., Gopalarathnam, A. & Edwards, J. R.,
  *Vortex-Sheet Representation of Leading-Edge Vortex Shedding from
  Finite Wings*, Journal of Aircraft 56(4), 1626–1640 (2019), DOI:
  [10.2514/1.C035124](https://doi.org/10.2514/1.C035124).
- Hirato, Y., Shen, M., Gopalarathnam, A. & Edwards, J. R.,
  *Flow Criticality Governs Leading-Edge-Vortex Initiation on Finite
  Wings in Unsteady Flow*, JFM 910, A1 (2021),
  DOI: [10.1017/jfm.2020.896](https://doi.org/10.1017/jfm.2020.896).
- Li, J. & Wu, Z.-N., *A Vortex Force Study for a Flat Plate at High
  Angle of Attack*, JFM 801, 222–249 (2016),
  DOI: [10.1017/jfm.2016.349](https://doi.org/10.1017/jfm.2016.349).
- Graham, W. R., Pitt Ford, C. W. & Babinsky, H., *An Impulse-Based
  Approach to Estimating Forces in Unsteady Flow*, JFM 815, 60–76 (2017),
  DOI: [10.1017/jfm.2017.45](https://doi.org/10.1017/jfm.2017.45).
- Terrington, S. J., Hourigan, K. & Thompson, M. C., *Vorticity Generation
  and Conservation on Generalised Interfaces in Three-Dimensional Flows*,
  JFM 936, A44 (2022),
  DOI: [10.1017/jfm.2022.91](https://doi.org/10.1017/jfm.2022.91).
- Gehlert, P. & Babinsky, H., *Boundary-Layer Vortex Sheet Evolution
  Around an Accelerating and Rotating Cylinder*, JFM 915, A50 (2021),
  DOI: [10.1017/jfm.2021.121](https://doi.org/10.1017/jfm.2021.121).
- Riziotis, V. A. & Voutsinas, S. G., *Dynamic Stall Modelling on Airfoils
  Based on Strong Viscous–Inviscid Interaction Coupling*, International
  Journal for Numerical Methods in Fluids 56, 185–208 (2008),
  DOI: [10.1002/fld.1525](https://doi.org/10.1002/fld.1525).
- DeVoria, A. C. & Mohseni, K., *The Vortex-Entrainment Sheet in an
  Inviscid Fluid: Theory and Separation at a Sharp Edge*, JFM 866,
  660–688 (2019),
  DOI: [10.1017/jfm.2019.134](https://doi.org/10.1017/jfm.2019.134).
