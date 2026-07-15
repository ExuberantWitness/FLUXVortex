# 文献研究综合报告:d_para 零拟合替换 —— 机构组件阻力构建、滑翔极曲线锚定与"随 U 递减残差"的机理裁定

> 证据口径说明:本轮编排脚本的 claims JSON 未能注入,本报告的题录由本 session 已下载并全文核实的一手文献重建。状态标注:**CONFIRMED** = 本地全文逐段核实;**PLAUSIBLE** = 仅摘要/二手来源核实(出版社页面/检索摘要),定量引用时须以原文复核。跨 track 结论(T1 吸力通道、Narsipur 塌缩等)只作定性交叉引用。

## 1. 机理裁定(按证据强度排序)

**裁定:d_para=3.0N 已由滑翔锚定判死(隐藏拟合常数,吸收扑动态模型缺陷);物理机构阻力 = 0.5×(U/8)² N(几何构建 + 滑翔极曲线锚定,双路一致)。随 U 递减的残差(物理口径 +3.51/+2.15/+0.52 N @U6/8/10)在符号上排除一切"真实阻力遗漏"解释,唯一同号的候选是随 k=πfc/U 增长的模型过推力——即翼的非定常通道(吸力过供)问题,归 T3b/T1 管辖,禁止再用任何阻力常数吸收。**

**裁定一(证据强度:最高,项目内直接实验 + 文献方法先例):d_para 的物理份额只有 ~0.5N@U8。**
- 滑翔锚(flap_amp=0 静态极曲线 vs 源论文 Drones 2025,9,535 Fig15):d_para=3.0 时模型静态系统阻力 3× 超实(aoa5:3.54N vs 实测反推 ~1.1N);改 d_para=0.5×(U/8)²+Blasius 摩擦后,L/D=6.46 vs 实测 6.8 @aoa5,巅峰点命中(gap_t3_dpara.md §5)。
- 几何构建交叉验证(Fig14a 照片 + 文献 Cd,见 §2a):机构钝体阻力正算区间 0.45–0.65 N@U8,与滑翔锚 0.5N 吻合。
- 方法先例:Ayancik et al. 2019 (JFM 871) 在自推进扑动实验的 BEM 对标中,**明文用"均匀流中圆柱的 Cd–Re 关系"(引 Munson 教材)把驱动杆阻力作为附加阻力施加**——"机构件按已发表钝体 Cd(Re) 正算并叠加"是该领域的标准操作,非我们发明。

**裁定二(证据强度:高,符号论证 + 4 条独立文献链):随 U 递减的残差不可能来自任何被遗漏的真实阻力。**
- 符号论证:一切真实阻力力值随 U 单调增(钝体 ∝U²、Blasius ∝U^1.5、Senturk 层流 offset 折合力 ∝U^1.5)。模型若漏了它,gap(模型−实测)应随 U **增大**;实测 gap 随 U **减小**(+3.51→+0.52)。方向相反,一票否决。
- 同号候选=随 k 增长的项:U↓(f 固定)→ k=πfc/U ↑(0.18→0.29)→ 模型过推力 ↑。文献中周期平均流向力确有纯 k 依赖成分:Moored & Quinn(AIAA J, doi:10.2514/1.J056634)把 Garrick 理论精确分解为"加质量产推力 + 环量致阻力"两项,**环量致(诱导)阻力项是 k 的纯函数,与振幅无关,恒存在**:C_T^G(k) = c1 − c2·w(k),理论精确值 c1=3π³/32, c2=π³/8。
- Garrick 高 k 高估推力的实测确认(交叉引 T1):Mackowski & Williamson 2015 (JFM 765) 直接测力;Moored & Quinn 自己的非线性数值也系统低于 Garrick 线。
- 尾迹侧独立佐证:Bohl & Koochesfahani 2009 (JFM 620:63–88) MTV 全控制体分析证明,仅用平均速度剖面估计的平均推力**显著偏高**,横向速度脉动引起的下游压降是随 k 急剧增长的"隐性阻力"项;阻力→推力翻转实际发生在 k≈9 而非平均速度法给出的 k≈6,且涡列排向翻转(k=5.7)不等于推力翻转。低阶模型若不含该压力项,高 k 处必然过推——与我们指纹同向。
- 结论:残差是**翼的非定常通道误差**(主嫌:plateau 吸力在深超临界的过供,Narsipur 2020 塌缩区,交叉引 T1 裁定),移交 T3b 案;d_para 不得回填。

**裁定三(证据强度:中,量纲分析 + 文献):残差的"非纯 k"细节指向混合 U×f 依赖。**
- U8 的 f 线(k 0.15→0.29)平坦而 U 线陡(案卷 §5)→ 残差非 w(k) 单变量。Moored & Quinn 的修正式 C_T(k,A*) = c1 − c2·w(k) − c3·A* 中,LEV/TEV 形阻项 D_form ∝ ρ s f²A³(定振幅定 f 时为 U 无关常数力)可解释 gap 的**水平项**而非斜率;Floryan et al. 2017 (JFM 822:386–397) 证明黏性阻力在推力系数上是**近似常数负偏置**;Senturk & Smits 2019 (AIAA J 57(7):2663–2669) 证明该偏置及阻推翻转随 **Re^(−1/2)** 标度(层流)。三者合起来:残差 = k 项(斜率主导)+ 常数形阻项(水平)+ 小 Re^(−1/2) 项,须 U×f 全网格分解(§5)。

## 2. 三个子问题逐一裁定

### 2a. 机构组件阻力构建配方(照片估计几何 + 文献 Cd + 干扰)

| 组件(Fig14a 照片) | 几何估计 | Cd(文献) | 适用域 | D@U8 (q=39.2Pa) |
|---|---|---|---|---|
| 机构本体(曲柄箱+电机,锐边箱体) | 迎风面 ~0.10×0.15 m = 0.015 m² | 0.85–1.05(Hoerner FDD Ch. III,锐边棱柱体;Re 无关区) | Re_D≈6.4e4∈[1e4,1e6],锐边分离固定 | 0.50–0.62 N |
| 外露碳管段(Ø8mm,根部外露 ~2×0.1m) | 0.0016 m² | ≈1.0(圆柱亚临界,Munson/Hoerner;Re_D≈4.3e3) | 1e3<Re_D<2e5 | ~0.06 N |
| 干扰阻力(杆-箱结合部) | — | +10%(Hoerner FDD Ch. VIII 量级,**外推标记**:本轮未逐页核书) | — | +0.06 N |
| 翼根遮蔽修正 | — | −10~−20%(**外推标记**,无文献定值) | — | −0.06~−0.12 N |
| **合计** | | | | **0.45–0.65 N** |

- 天平在风洞开口段外/整流罩内的部分不计(照片判读,**外推标记**)。
- 构建法本身的文献先例:Ayancik et al. 2019(驱动杆按圆柱 Cd(Re) 正算叠加,CONFIRMED,全文核实);Hoerner 常数为教材级标准表值(PLAUSIBLE,本轮无书面核对)。
- 锐边钝体 Cd 与 Re 无关 ⇒ **d_para ∝ U² 严格成立**(U5–10 全域),这正是 Fig15 三速极曲线重合所要求的性质——自洽。

### 2b. 滑翔极曲线锚定程序(已执行,予以规范化)

1. **锚的合法性**:Fig15(双翼水平滑翔,L/D_max 6.8@AoA5°,5/8/10 m/s 三速重合)是**静态**测量,与 118 个扑动工况不相交;用户指令明确其为"free static anchor"。锚定后 d_para 仍是全模型唯一标定常数,但语义从"扑动残差吸收器"变为"静态系统阻力配平",且被几何正算区间(0.45–0.65N)约束——非自由拟合。
2. **程序**:flap_amp=0,AoA 扫 0–20°,U∈{5,8,10};令模型 L/D@AoA5 命中 6.8 的 d_para 值,且要求落入几何正算区间,且三速曲线重合(∝U² 检验)。结果:0.5 N@U8,三条件同时满足。
3. **行业口径**:正规做法是 wind-on tare(去翼装机构吹风)或镜像法(Barlow, Rae & Pope, *Low-Speed Wind Tunnel Testing*,tare-and-interference 章节;PLAUSIBLE,教材级,本轮未核页码)。源论文无 tare、无机构图纸,滑翔极曲线是数据集内唯一可用锚。
4. **co-design 语义(关键)**:d_para 属于**实验装置模型**,不属于翼模型。co-design 预测(自由飞/换构型)时必须置零或换成新装置的构建值;它对任何翼几何/运动学**无预测职能**。

### 2c. k/Re 依赖翼阻候选 vs 三特征解释力

| 特征 | 候选A:k 依赖过推力(plateau 吸力过供 + Garrick 环量致阻缺失) | 候选B:LEV/TEV 形阻 ∝ρsf²A³ | 候选C:Re^(−1/2) 层流偏置(Senturk)/Blasius |
|---|---|---|---|
| gap 随 U 递减(f 固定,斜率 ≈−0.75 N/(m/s)) | **强**。k=πfc/U 随 U 降而升;深超临界占空比 ↑;w(k) 环量致阻纯 k 函数(Moored Eq.18);BK2009 压力项亦随 k 陡增 | 弱。定 f 时 f²A³ 为 U 无关常数,无斜率 | **反号排除**。力值 ∝U^1.5,漏掉它 gap 应随 U 增 |
| gap 全正(+0.5~+3.5N,模型过推) | **强**。与 T1 裁定(plateau 在 Narsipur 塌缩区过供吸力)同源 | **中**。无粘模型不含涡致形阻,方向正确,可解释 ~+0.5–1N 水平项 | 弱。Blasius 已开(visc=True),Senturk 偏置与之同物理,不得叠加 |
| U8 f 线平坦(k 0.15→0.29) | 中(存疑)。纯 w(k) 应在 f 线上也显形;平坦说明过推力≈c·f²·(1/U 型)混合标度,需 U×f 网格分解 | 中。f²A³ 在 f 线上应二次增长——与平坦矛盾,除非与吸力项对消 | 无预测 |

**常数来源(零拟合合规性)**:候选A的 w(k) 与 c1=3π³/32、c2=π³/8 为理论精确值(Garrick 1936 NACA Rept. 567 经 Moored & Quinn 重排);其拟合值 c1=2.89, c2=4.02, c3=0.39 是**对他们自己 NACA0012 数值数据的最小二乘,作者明言"不太可能是普适数"**——只可用作诊断读数,禁止移植进我们模型(违反零拟合精神的"借来的拟合")。

## 3. VERDICT:co-design-ready 配方

**总流向力 = D_wing(几何/Re 的预测函数) + D_rig(仅限本风洞比对)**

| 项 | 公式/方法 | 常数 | 适用域 | 锚定状态 |
|---|---|---|---|---|
| D_rig | 0.5×(U/8)² N | 0.5 N@U8(几何构建 0.45–0.65N 区间 ∩ 滑翔锚) | U∈[5,10] m/s,仅此装置、仅风洞比对;co-design 预测时置零 | 锚定(唯一幸存标定常数,受构建区间约束) |
| 翼摩擦阻 | Blasius:C_f=1.328/√Re_c,湿面积双面 | 无自由常数 | 层流附着,Re_c<5e5(我们 1.5e5 ✓);visc=True 已开 | 文献(Blasius/Schlichting) |
| 翼诱导阻 | UVLM 自带;实现方法须核查 Katz(压力+前缘吸力分解)vs Joukowski 两法收敛性 | 无 | 弦向面板收敛核查(有拱度翼 Katz 收敛更快,收敛率近乎与 k 无关) | 文献(Simpson 2013;Lambert & Dimitriadis) |
| 翼非定常周期平均项 | **不加任何新阻力项**。残差(+3.5→+0.5N)整案移交 T3b:吸力通道修正(T1 方案:Narsipur 塌缩/Polhamus 旋转 + Carlson K_T) | — | — | 交叉引 T1 裁定 |
| 诊断读数(非模型项) | C_T(k,A*)=c1−c2·w(k)−c3·A*,w(k)=3F/2+F/(π²k²)−G/(2πk)−(F²+G²)(1/(π²k²)+9/4)(实现时以 Moored & Quinn Eq.(18) 原排版为准) | 理论 c1=3π³/32,c2=π³/8;拟合 2.89/4.02/0.39 仅限诊断 | 小振幅、2D、尾迹平面化假设 | 文献(理论值)/外推(拟合值) |

**明确排除**:(i) Senturk Re^(−1/2) 偏置不叠加(与 Blasius 同物理,双计);(ii) Moored 拟合系数不进模型;(iii) 任何用我们 118 工况力数据回调的常数。

## 4. 实现配方(单变量步骤,防双计)

1. **[已完成] d_para=3.0 → 0.5×(U/8)²**,同步 visc=True。验证:静态极曲线 aoa5 命中(L/D 6.46 vs 6.8)。
2. **诱导阻力口径审计**(纯核查,不改数):确认 UVLM 流向力用的是 Katz 分解还是 Joukowski;跑弦向面板收敛曲线(我们是有拱度膜翼——文献预期 Katz 收敛更快);若两法差 >5% 则以收敛者为准。改动量:0 个常数。
3. **T3b 移交**:扑动过推力残差由吸力通道修正解决(T1 的 Narsipur/Polhamus + Carlson K_T),**本 track 不新增阻力项**。防双计红线:吸力修正落地后,禁止回头调 d_para;若静态极曲线因此漂移,说明吸力修正误伤了静态通道(静态时 LESP 亚临界,本不应受影响)——回退排查。
4. **co-design 出口**:预测自由飞/新构型时 d_para:=0;新装置需重新走 §2a 构建 + 可用静态锚。

## 5. 判别实验设计(哪些扫掠点分离 rig 份额与翼份额)

- **D1 静态 U 线(rig 定标线)**:flap_amp=0, aoa∈{0,5}, U∈{5,6,8,10}。f 无关 ⇒ 全部 U 依赖属于 rig+静态翼;检验 D(U)/q = 常数(∝U² 律)。任何偏离 → rig 模型错,且只能在此线上修。
- **D2 定 U 变 f 线(翼份额纯化线)**:U∈{6,10} 各扫 f∈{1.0,1.5,2.0,2.5}(U8 已有)。**rig 阻力对 f 严格常数** ⇒ f 方向的一切 gap 变化 100% 是翼的非定常份额。这是分离 rig/翼的最锋利的一刀。
- **D3 等 k 对角线(Re 判别线)**:k≈0.225:(U6,f1.5)、(U8,f2.0)、(U10,f2.5)。此线上加质量推力与 rig 阻力都 ∝U²(退化),但 Re^(−1/2) 项使 gap/qS ∝ U^(−0.5):**若 gap/qS 沿线平坦 → 纯 k 机理;若 ∝U^(−0.5) 下降 → 存在层流 Re 项**(Senturk 判据)。
- **D4 定 f 变扭转 @U6 与 U10**:分离"有效攻角幅值"依赖与 k 依赖(深 feathering 下 plateau 过平的 T1 附案在两个 k 端点的显形差)。
- 退化警告:等 k 线上 k 与 U² 不可分(两者都 ∝U²),**不要**试图用等 k 线分离 rig 份额——rig 份额只能由 D1(静态)与 D2(f 方向导数)钉死。

## 6. 验证门(118 工况电池)

- **G1(必须保持)**:静态极曲线 aoa5:L/D∈6.8±10% 且 U5/8/10 三速重合(Fig15 复现)。这是 d_para 的锚,漂移即回退。
- **G2(必须改善,T3b 落地后)**:U6/U10 扑动线的 curve-mean gap 由 +3.51/+0.52N 收敛到 |gap|<1N;全 118 工况推力 curve-mean MAE ≤ 0.84N 不得回退。
- **G3(不得回退)**:升力 MAE ≤ 0.87N。d_para 与摩擦阻不触升力通道;若升力变化 >2%,即双计/误伤警报。
- **G4(零拟合审计)**:改动全程无任何常数向我们的力数据回调;d_para 只能经 §2a 几何重估或新的静态锚变更。

## 7. 常数:文献锚定 vs 外推(逐项披露)

| 常数 | 值 | 状态 |
|---|---|---|
| 锐边箱体 Cd | 0.85–1.05 | 文献表值(Hoerner Ch.III;PLAUSIBLE,未核页) |
| 圆柱 Cd(亚临界) | ≈1.0(Re_D 4e3) | 文献表值(Munson/Hoerner;Ayancik 同款用法全文核实) |
| 干扰 +10%、遮蔽 −10~20% | — | **外推**(量级判断,无定值文献) |
| d_para=0.5N@U8 | 构建区间∩滑翔锚 | 锚定(唯一标定常数;锚为静态独立数据) |
| Blasius 1.328/√Re | — | 文献(精确解) |
| c1=3π³/32, c2=π³/8 | 理论精确 | 文献(Garrick 经 Moored & Quinn) |
| c1,2,3=2.89/4.02/0.39 | 他人数据拟合 | **外推,禁入模型**,仅诊断 |
| Senturk Re^(−1/2) 律 | Re≲3.2e4 数值实验 | **外推**至我们 Re 1.5e5;且与 Blasius 双计,不采用 |

## 8. 文献不支持或存疑的部分(明确列出)

1. **Hoerner 干扰阻力公式未逐页核实**:结合部干扰的闭式(FDD Ch. VIII)本轮未取得书面文本,+10% 是量级外推;取得 Hoerner 复印件后应回填闭式。
2. **Senturk & Smits、Floryan 仅摘要级核实**(出版社页/检索摘要),其精确方程与系数未目验;二者在本方案中只承担"排除/定性"职能,不引入常数,风险受控。
3. **Moored & Quinn 是 2D、小振幅、无 LEV 的势流数值**;其 w(k) 环量致阻在我们深 feathering、LEV 脱落工况只有定性指向力。其 3D 版(Ayancik 2019)的形阻项同样是对自家数据的拟合。
4. **BK2009 的 k 范围(至 11.5)远高于我们(0.15–0.29)**;其"压力项随 k 增长"的定量曲线不可直接内插到我们的 k 区间,只作机理方向证据。
5. **残差的"非纯 k"未决**:U8 f 线平坦与 U 线陡的矛盾说明单变量 k 律不成立,必须等 D2/D3 网格数据裁决;在此之前任何"翼阻闭式"都为时过早——这正是本 verdict 拒绝新增阻力项、整案移交吸力通道(T3b)的原因。
6. **翼根遮蔽与天平整流罩判读依赖照片**,机构图纸未发表,几何区间(0.45–0.65N)的 ±20% 不可再压缩。

**一句话决策**:d_para 定格为 0.5×(U/8)² N(几何构建区间 ∩ 滑翔锚,∝U² 严格、仅限本装置、co-design 预测置零),Blasius 摩擦保持,诱导阻力只做方法审计;随 U 递减的残差在符号上排除一切真实阻力解释、指向随 k 增长的吸力过供,整案移交 T3b/T1,由 D1 静态线 + D2 定 U 变 f 线 + D3 等 k 对角线三把刀分离 rig/翼/Re 份额——全程零新增拟合常数。

---

## 附:幸存题录(本 session 核实)

- [rig-buildup-precedent] **CONFIRMED** 自推进扑动实验/BEM 对标中,驱动杆阻力按"均匀流圆柱 Cd–Re 关系"(引 Munson 教材)正算并作为附加阻力施加——机构件组件构建法的领域先例。
  来源: Ayancik, Zhong, Quinn, Brandes, Bart-Smith & Moored (2019), 'Scaling laws for the propulsive performance of three-dimensional pitching propulsors', J. Fluid Mech. 871
  URL: https://arxiv.org/abs/1810.11170
- [k-dependent-wake-drag] **CONFIRMED** Garrick 理论可精确分解为加质量(产推)与环量(致阻)两项:C_T^G(k)=c1−c2·w(k),c1=3π³/32、c2=π³/8 为理论精确值;环量致(诱导)阻力项是 k 的纯函数、与振幅无关、恒存在;理论系统性略高估非线性数值推力;LEV/TEV 形阻修正 D_form∝ρsf²A³ 给出 C_T(k,A*)=c1−c2w(k)−c3A*,拟合值 2.89/4.02/0.39 作者明言非普适。
  来源: Moored, K.W. & Quinn, D.B. (2019), 'Inviscid Scaling Laws of a Self-Propelled Pitching Airfoil', AIAA Journal 57(9), doi:10.2514/1.J056634
  URL: https://arxiv.org/abs/1703.08225
- [wake-survey-thrust] **CONFIRMED** 仅用平均速度剖面的动量法显著高估平均推力;完整控制体(含速度脉动与下游压力项)给出的平均力显著更低,与黏性计算吻合;阻→推翻转在 k≈9(完整式)而非 k≈6(仅平均速度);涡列排向翻转(k=5.7)不决定推力翻转。
  来源: Bohl, D.G. & Koochesfahani, M.M. (2009), 'MTV measurements of the vortical field in the wake of an airfoil oscillating at high reduced frequency', J. Fluid Mech. 620:63–88, doi:10.1017/S0022112008004734
  URL: https://doi.org/10.1017/S0022112008004734
- [uvlm-induced-drag] **CONFIRMED** UVLM 诱导阻力两种算法(Katz 压力+前缘吸力分解 vs Joukowski)收敛性不同:2D 无拱度 Joukowski 快;3D 有拱度翼 Katz 显著更快,差距随拱度增大;收敛率近乎与 k 无关。
  来源: Lambert, T. & Dimitriadis, G., 'Induced Drag Calculations with the Unsteady Vortex Lattice Method for Cambered Wings', AIAA Journal(技术注记);完成 Simpson, Palacios & Murua (2013), AIAA Journal 51(7):1775–1779, doi:10.2514/1.J052136 的分析
  URL: https://doi.org/10.2514/1.J052136
- [viscous-thrust-offset] **PLAUSIBLE**(摘要级)黏性阻力在扑动翼推力系数上表现为近似常数的负偏置,对慢运动/低 St 影响最大。
  来源: Floryan, Van Buren, Rowley & Smits (2017), 'Scaling the propulsive performance of heaving and pitching foils', J. Fluid Mech. 822:386–397, doi:10.1017/jfm.2017.302
  URL: https://arxiv.org/abs/1704.07478
- [re-scaling-offset] **PLAUSIBLE**(摘要级)俯仰翼推力/功率标度系数与阻推翻转随 Re^(−1/2) 标度(层流边界层);Re>1.6e4 后推力/功率对 Re 不敏感,但阻力(效率)仍显著 Re 依赖。
  来源: Senturk, U. & Smits, A.J. (2019), 'Reynolds Number Scaling of the Propulsive Performance of a Pitching Airfoil', AIAA Journal 57(7):2663–2669, doi:10.2514/1.J058371
  URL: https://arc.aiaa.org/doi/10.2514/1.J058371
- [tare-practice] **PLAUSIBLE**(教材级,未核页码)风洞标准做法为 wind-on tare / 镜像法扣除支撑与干扰;源论文无 tare ⇒ 数据集内静态滑翔极曲线是唯一合法锚。
  来源: Barlow, Rae & Pope, 'Low-Speed Wind Tunnel Testing' (3rd ed.), tare-and-interference
- [hoerner-constants] **PLAUSIBLE**(教材表值,未核页)锐边棱柱体 Cd≈0.85–1.05(Re 无关区)、亚临界圆柱 Cd≈1.0–1.2(1e3<Re_D<2e5)、法向平板 2D 1.98/方板 1.17。
  来源: Hoerner, S.F. (1965), 'Fluid-Dynamic Drag', Ch. III(钝体表)、Ch. VIII(干扰,量级)
- [交叉引 T1] **CONFIRMED**(track 1 全文核实)分离期黏性前缘吸力跌至近零(Narsipur et al. 2020, JFM 900 A25);Garrick 大 k/大振幅显著高估平均推力(Mackowski & Williamson 2015, JFM 765)——本 track 残差的主嫌机理出处。
- [项目内证据] **CONFIRMED** 滑翔锚判决与物理重构电池(d_para=0.5+visc 在 aoa5 命中 L/D 6.46 vs 6.8;扑动残差 +3.51/+2.15/+0.52 @U6/8/10,k 0.29/0.22/0.18;U8 f 线平坦)。
  来源: platform/docs/diag/gap_t3_dpara.md §4–§5(2026-07-14/15)
