> **⚠ 口径标注(2026-07-13,kinematics_audit.md)**:本报告的定量指纹(−0.18 N/deg、
> −3.8N@tw22.5、±45°扑/±22.5°扭)采集于运动学口径审计**之前**。正确口径(±22.5°扑/
> ±11.25°扭、正确条件配对)下扭转行残差为 **−1.48N@标称22.5°**。机理裁定与常数菜单
> 仍然有效;实现配方排序(R1-R4)须在 v2 基线 step-1 重拟合后重排——尤其 R1 的
> "CD90(AR)≈1.20 给 −3.7~−4.1N 正中所需"的量化匹配已失效。

# 文献研究综合报告:Case A "扭转过阻"(gap_T = −0.18 N/°) 的机理裁定与重定向方案

**案情指纹**:RoboEagle 扑翼膜翼,U=8 m/s,f=2.3–2.6 Hz,±45° 挥拍,展向 feathering 扭转 ψ = A_t·(y/span)·cos(Ωt),A_t=0–45°,Re≈1.5e5,k≈0.25–0.45。实测升力与净推力在 A_t: 0→22.5° 间**近乎持平**(L≈7.8 N, T≈−1.1 N),>~25° 升力才跌落。模型的附着流 Bernoulli 力幅值随扭转 +25%,方向锁死在离升力轴 ~67° 的后仰角,升力投影跟得上、阻力投影失控:gap_T = −0.18 N/°(tw22.5 刚性差 −3.8 N,柔性回放释放 ~1.3 N)。过阻上下冲程对称,深扭转时 |Fx| 峰值移向冲程反转点。

---

## 1. 分角度幸存证据表(全部经对抗核实,verdict=CONFIRMED)

### 角度一:twist-measurements(扭转实测同类案)

| ID | 一句话结论 | 匹配 | 文献 |
|---|---|---|---|
| C2 | Carlson 部分旋转规则就是"甩阻保升"的精确闭式:吸力矢量 c_t/cosΛ 旋转 θ=arccos(K_T),保留部分为纯弦面推力(抵消阻力投影),旋走部分变法向力 ΔC_N=(c_t/cosΛ)·√(1−K_T²)(=Polhamus 涡升力)。生产模型卡在 θ→90°(K_T→0),增量落成钝体法向力带大后仰投影 | 高 | Carlson, Mack & Barger, NASA TP-1500 (1979), 'Lift and Drag for a Flat Wing', fig.9;Polhamus NASA TN D-3767 (1966);J. Aircraft 8(4):193-199 (1971) |
| C3 | DeLaurier 改型片条理论:截面流向力 dFx = ηs·dTs − dDcamber − dDf(3c/4 相对流角),扭转设计准则=保持截面附着以实现前缘吸力→合力前倾成推力;其气弹扭转 ornithopter 实测"平台后失速跌落"形状 | 中高 | DeLaurier, Aeronaut. J. 97(964):125-130 (1993);同刊 97(965):153-162(单作者) |
| C5 | 鸽翼 CFD 扭转扫掠 0→40°:扭转降有效攻角→削弱下冲程 LEV→"压力合力前转",C_T −0.128→−0.024 单调甩阻,C_L 仅 −20%(<25° 区段约 −10%)——即"近平升力+甩阻"响应,机制是分离缓解+吸力式前倾,非气弹 washout | 中高 | Wu, Wang, Jia, Ding, Biomimetics 10(5):328 (2025), doi:10.3390/biomimetics10050328 |
| C6 | 弯扭耦合复合材料扑翼(LES+FSI):被动扭转 +77% 周期均推力、升力 6.9 vs 1.39 N;k>~0.3 后被动扭转过大触发分离、升力回落——同时复现"中等扭转甩阻"与"过扭升力塌" | 中 | Kumar, Samanta, Padhee, arXiv:2505.23372 (2025);J. Fluids Struct. (2025) |
| C7 | 风洞实测前缘扭转 FWMAV:过度连续扭转把截面打到**负有效攻角**(上冲程 −23.1°..−54.8°)产生负升力——深扭转的增量是被"卸载/反号"(dumped),不是继续后仰;升力(非推力)是最终塌掉的量 | 低中 | Yang et al., Biomimetics 8(2):134 (2023), doi:10.3390/biomimetics8020134 |

### 角度二:poststall-vector(失速后力矢量定律)

| ID | 一句话结论 | 匹配 | 文献 |
|---|---|---|---|
| A2-C1 | Leishman–Beddoes 的 Kirchhoff **矢量**分裂:法向通道 CN∝((1+√f)/2)²(下限 0.25 倍附着斜率),切向吸力通道 CT=−η·(dCN/dα)·α_e²·√f 独立衰减(η=0.95),CD=CN·sinα+CT·cosα。对整个力矢量套单一 Kirchhoff 幅值因子**不是**文献模型——文献只强衰减吸力通道,故甩阻保升 | 高(结构) | Leishman & Beddoes, JAHS 34(3):3-17 (1989);Bangga, Lutz & Arnold, Wind Energ. Sci. 5:1037-1058 (2020), Eqs.14/18/19/32, Table 2 |
| A2-C2 | Hoerner/Lindenburg 深失速法向力饱和律:CN = CD90·sinα/(0.56+0.44·sinα)(饱和于 CD90),切向仅剩小圆前缘吸力(γ=0.28·√(r_nose/c) rad,约 2–4°)+半层流摩擦;深失速截面的**增量攻角几乎不再加力**——增量被"dumped" | 高 | Lindenburg, ECN-RX--01-004 (2001), Sec.2.2.1-2.2.3;前缘律引自 Hoerner 'Fluid-Dynamic Lift' p.21-1 |
| A2-C3 | 有限展弦比封顶:2D 平板 CD90=1.98 必须换成 CD90(AR)≈1.20–1.23 @AR=6(四个独立发表拟合一致:Viterna 1.11+0.018AR;Montgomerie 1.98−0.81(1−e^(−20/AR));Hibbs-Radkey 1.98−0.81·tanh(12.22/AR);StC 2.0−0.82(1−e^(−17/AR)))——对 Hoerner 2D 常数打 **0.61×** | 高(定量) | Lindenburg ECN-RX--01-004 Sec.1.1.4;Viterna & Janetzke (1981);数据源 Hoerner Fluid-Dynamic Drag p.3-16 fig.28 |
| A2-C4 | Viterna 失速后外插:CD = CDmax·sin²α + B2·cosα,CL = (CDmax/2)·sin2α + A2 项,CDmax=1.11+0.018AR;失速后阻力增量 d(CD)/dα = CDmax·sin2α ≈ 0.02/° @25°——阻力投影**饱和**,远低于附着格 67° 后仰增长率 | 高 | Viterna & Corrigan (1982);NASA TM-82944;Mahmuddin, JSOse Vol.6 (2016) Eqs.1-9 |
| A2-C5 | LESP 吸力饱和闭式:C_suction = 2π·LESP²,超临界时钉在 2π·LESP_crit²(NACA0015@Re1100: 0.19 → C_s,max≈0.23),溢出涡量的力贡献是弦法向 | 中 | Ramesh et al., JFM 751:500-538 (2014);Ramesh, Murua & Gopalarathnam, JFS 55:84-105 (2015);Gelado & Ramesh, AIAA 2022-4105 |
| A2-C6 | 黏性 LESP(Narsipur/Gopalarathnam):Cf 拐点判据定义临界;前缘分离后黏性吸力跌至近零而积分法向力经 DSV 延续——切向(吸力)份额是分离的**第一受害者**,等效于 Garrick/Carlson 吸力项上乘 ηs(t)<1 | 中 | Narsipur et al., JFM 900:A25 (2020), doi:10.1017/jfm.2020.467;Narsipur, TCFD 36:845-863 (2022) |
| A2-C7 | Polhamus 重定向律:分离消灭前缘吸力时,吸力矢量转 90° 到**面法向**("normal force becomes the resultant force"),CL = Kp·sinα·cos²α + Kv·sin²α·cosα,ΔCD=CL·tanα;且 camber/twist 可"recover the leading-edge thrust"。文献的重定向目标永远是**局部弦法向**,不是风轴升力轴——但深扭转+拍动尖部的局部弦法向本身已前倾,远前于 67° | 中高 | Polhamus, J. Aircraft 8(4):193-199 (1971), doi:10.2514/3.44254;NASA TN D-3767 (1966) |

### 角度三:les-retention(前缘吸力保持)

| ID | 一句话结论 | 匹配 | 文献 |
|---|---|---|---|
| C1 | **主机理(Garrick)**:振荡翼净推力=前缘吸力+法向力流向投影(eq.13, P_x=πρS²+αP+βP_β;纯 plunge 推力全部来自吸力)。附着格若给扭转增量法向载荷却**不给增量记吸力**,增量全额投到阻力轴(67° 后仰)——这正是 gap_T 超增而升力跟踪的直接解释 | 高 | Garrick, NACA Report 567 (1936), eq.13;c_t=2πα²/√(1−M²) 见 Carlson TP-1500 p.6 |
| C2 | **生产分裂是静态的**:Carlson K_T 是定常巡航标定关联(eqs.7-10),含 Cp,lim(M,Re) 无任何折减频率项;k~0.3 下结构性**低估**俯仰/feathering 截面实际可获吸力(无法表达前缘分离的非定常延迟)。这就是准定常 Kirchhoff 只给 ~1/3 缓解的原因——它继承了 Carlson 的静态天花板 | 高 | Carlson, Mack & Barger, NASA TP-1500 (1979), eqs.(7)-(10), pp.7-11 |
| C3 | **速率效应**:俯仰截面周期内最大前缘吸力**超过**静态最大值,LEV 脱落临界随非定常度上升;失速前 LESP 完全由瞬时有效攻角驱动。k~0.3 的 feathering 外段附着(保吸力)到比静态关联更高的有效攻角 | 高 | He, Deparday, Siegel, Henning & Mulleners, AIAA J. 58(12):5146-5155 (2020);Deparday & Mulleners, Phys. Fluids 31:107104 (2019)。注意:实测 k=0.1,k~0.3 属外推 |
| C4 | LESP 临界门:给定翼型+Re,LEV 起始发生在近乎运动学无关的临界 LESP;门下附着、全推力方向弦向吸力保留(K_T→1)。修正常数:LESP_crit≈0.11–0.21(SD7003: 0.21@1e4/0.18@3e4/0.14@1e5;平板 0.11@1e3),**不是 0.07–0.13** | 中高 | Ramesh et al., JFM 751 (2014);TCFD 32:109-136 (2018);caveat: Kay et al. AIAA J. (2021), Deparday et al. JFM 941:A60 (2022) |
| C5 | **Polhamus 路由**:吸力保不住时不变阻力——变弦法向力(涡升力),对小/中等入射的 feathering 截面弦法向≈升力轴。正确做法:对扭转增量用**增量自己的弦法向**做 (1−K_T) 旋转,而不是给增量套基载荷的 67° 后仰 | 高 | Polhamus NASA TN D-3767 (1966);Carlson TP-1500 p.12-13(twist/camber 扩展:旋转矢量须沿局部 camber 切向角 δ_n) |
| C7 | feathering 保效率的力矢量基础:部分 feathering 保持截面附着→整周期保留前缘吸力→合力周期均前倾;Lighthill Θ=1 零推力、Θ=0 强 LEV 低效,高效区在中间。高效带 α_max≈15–25°, St≈0.16–0.35 | 高/中 | Lighthill feathering 参数(Paniccia et al., Sci. Rep. 11:22297, 2021);Read, Hover & Triantafyllou, JFS 17:163-183 (2003) |

### 角度四:uvlm-twist-practice(格法失速处理实践)

| ID | 一句话结论 | 匹配 | 文献 |
|---|---|---|---|
| C1 | Polhamus 吸力类比 = 经典"甩阻保升"律:分离时吸力大小保留、从弦面旋到法向;全分离极限 CD=CL·tanα;Kp/Kv 全部来自势流升力面理论,零拟合 | 高 | Polhamus TN D-3767 (1966);J. Aircraft 8(4):193-199 (1971);阻力式 TN D-4739 (1968) |
| C2 | 生产自家机器 TP-1500 本身就含重定向(亏缺吸力→法向,保法向、只调弦面投影),且**明文警告**扭转/弯度翼的部分旋转处理"需要进一步研究"——正是本案 regime。扭转扩展式:dCA=−c_t·cos(θ−δ_n), dCN=(c_t/cosΛ)·sin(θ−δ_n), δ_n=arctan(tanδ/cosΛ) | 高 | Carlson TP-1500 (1979), 'Extension to Wings With Twist and Camber', p.13 |
| C3 | Mukherjee–Gopalarathnam decambering:双变量 δ1/δ2 把片条势流 (Cl,Cm) 拉到黏性极曲线——本质是**降环量的法向/升力修正**,不旋转弦面投影;这解释了纯 decambering(Kirchhoff 矢量缩放)阻力欠缓解 | 中 | Mukherjee & Gopalarathnam, J. Aircraft 43(3):660-668 (2006), doi:10.2514/1.15149 |
| C4 | 失速片条载荷经格间诱导转移内侧**有条件物理**:文献记录了锯齿 spanwise Cl 伪影与多解(需 stalled/unstalled 标记选物理支)——重分配式阻力修正必须正则化,不可裸用 | 中 | Mukherjee NCSU 博士论文 (2004) Ch.3-4;Paul & Gopalarathnam, IJNMF 76(4):199-222 (2014) |
| C5 | DeLaurier ηs:最字面的"保法向、只调吸力"旋钮——ηs 只乘弦向吸力项,法向力不动;但它是标量效率非旋转,单用可能欠缓解(ηs≈0.98 数值无一手出处,勿硬编码) | 中高 | DeLaurier, Aeronaut. J. 97(964):125-130 (1993);复现 Djojodihardjo & Alif (2018) eqs.30-33 |
| C7 | 附着 UVLM 的弦向力对力评估方案高度敏感(Katz-Plotkin vs Joukowski 低 AR 定常阻力收敛性迥异),深入射下附着格系统性错置弦向投影;业界通行补救=在 UVLM 上嫁接 Polhamus/LESP 吸力类比 | 中 | Fritz & Long, J. Aircraft 41(6) (2004);Lambert, Abdul Razak & Dimitriadis, Aerospace 4(2):22 (2017);Nguyen et al., J. Aircraft 53(6) (2016), doi:10.2514/1.C033456 |

### 角度五:aeroelastic-relief(柔性释放 ~1.3 N 的机理边界)

| ID | 一句话结论 | 匹配 | 文献 |
|---|---|---|---|
| A5-1 | 高变形膜翼(hover)以气弹数 Ae=Eh/(½ρŪ²c) 标度:被动 camber 使前缘对齐诱导流→面法向压力前转(甩阻保升),最优 Ae 处 LEV 被贴体剪切层取代;Ae 过低反而升损阻增——**非单调**,对应 >~25° 升力跌 | 高 | Gehrke & Mulleners, PNAS 122(6):e2410833121 (2025) |
| A5-2 | 最优 Ae 处膜被动 camber 15–20%,柔性收益集中在**减速半冲程/近反转**——与深扭转时 \|Fx\| 峰移向冲程反转的相位结构一致 | 中 | Gehrke, Richeux, Uksul, Mulleners, Bioinspir. Biomim. 17(6):065005 (2022) |
| A5-3 | 匹配运动学刚柔对照实测:弦向柔性同时提升推力与效率,PIV 显示高效工况 LEV 更弱——"匹配运动学下推力上升"就是流向阻力投影被甩掉 | 中高 | Heathcote & Gursul, AIAA J. 45(5):1066-1079 (2007) |
| A5-4 | 展向柔性:适度柔性 St>0.2 时推力小增+功率小降;**过度柔性有害**——释放只在有界柔度窗内成立 | 中 | Heathcote, Wang & Gursul, JFS 24(2):183-199 (2008) |
| A5-5 | 气弹数标度律:力与效率随无量纲尖部变形单参数塌缩;力最优略低于共振,效率最优约半固有频率(Π1∝Eh³/(ρ_f U²c³)) | 中 | Kang, Aono, Cesnik & Shyy, JFM 689:32-74 (2011) |
| A5-6 | 柔性收益是**相位/形状(力重定向)效应**而非幅值效应:最优在共振外(~0.7ω0),靠弹性储放能调形状相位、重排气动力方向 | 中 | Ramananarivo, Godoy-Diana & Thiria, PNAS 108(15):5964-5969 (2011) |

---

## 2. 裁定(VERDICT)

**主裁定:深 feathering 下"甩阻力投影、保升力"是三段闭式机理的组合,全部有一手文献:**

1. **附着段(主犯之因)——Garrick 吸力抵消缺失**:附着流增量的弦向反作用本应被前缘吸力抵消(Garrick NACA 567 eq.13:P_x = πρS² + αP;les-retention C1)。生产格把扭转增量当纯法向压力载荷、锁死 67° 后仰,却用 **Carlson 静态 K_T** 给增量记吸力——TP-1500 是定常巡航标定,无折减频率项(les-retention C2),而 k~0.3 的 feathering 截面实测在周期内保持**高于静态**的吸力(He/Deparday & Mulleners,les-retention C3;LESP 门 C4)。→ 模型对增量**系统性欠记吸力**,弦向投影全额漏成阻力。
2. **分离段(增量去向)——Polhamus 路由**:吸力保不住的部分不进阻力轴,而是旋 90° 进**局部弦法向**(涡升力;Polhamus 1966/1971,A2-C7/les-C5/uvlm-C1)。深扭转+拍动尖部的局部弦法向本身前倾,故增量在实验室系里落向升力轴——"redirected toward the lift axis"有精确闭式:Carlson 部分旋转 dCN=(c_t/cosΛ)√(1−K_T²)(twist-C2),扭转翼须用局部 δ_n(uvlm-C2)。
3. **深失速段(>~25° 平台上界)——法向力饱和**:Lindenburg/Viterna 律 CN=CD90·sinα/(0.56+0.44sinα) 饱和于有限 AR 的 CD90≈1.2,深失速截面增量攻角几乎不再加力("dumped",A2-C2/C4),负有效攻角卸载(twist-C7)+过柔窗外气弹失效(A5-1/4)共同给出升力最终塌落。柔性回放的 ~1.3 N 属被动 camber/washout 相位重定向(A5 全系+twist-C6),量级与文献一致,是从犯不是主犯。

**反面确认(墓地)**:Garrick 线性理论本身预测推力对 feathering **强烈敏感**(REFUTED-2)——实测平台**不能**归因于附着线性理论,必须靠上述黏性/分离重定向;生产 K_T 的 "Re 地板塌缩到 0" 一说系公式误引,实际 K_T≈0.1–0.5(REFUTED-1),故问题不在 K_T 数值崩溃,而在其**静态性**与**增量旋转方向**。

**三个失败修复为何失败(全部由 A2-C1 矢量分裂定律解释):**
- **失败 (1) 平板 CN 混合超冲(−6.1..−6.7 N)**:用了 2D Hoerner 常数 1.98。有限 AR 封顶 CD90(AR=6)≈1.20(0.61×,A2-C3)给 −3.7..−4.1 N——几乎恰好是所需 −3.8 N;且缺 Lindenburg 饱和分母,深攻角继续线性加阻。
- **失败 (2) α_eff-Kirchhoff 幅值因子治推力杀升力(−5 N)**:把单一标量因子套在整个 3c/4 载荷上。文献模型(L-B Eqs.18/19)**分两通道**:法向保 ((1+√f)/2)²(下限 0.25),只有切向吸力按 η√f·α_e² 强衰减。标量幅值因子无法旋转力矢量——治阻必杀升,结构性必然。
- **失败 (3) 准定常几何失速 Kirchhoff 矢量缩放只给 ~1/3**:decambering/Kirchhoff 家族本质是法向/升力修正(uvlm-C3),不旋转弦面投影;且继承 Carlson 静态天花板(les-C2),无非定常吸力保持。

---

## 3. 实现方案(IMPLEMENTATION RECIPE)——单变量、零拟合、按优先级

**通道记账原则(防双算)**:附着片条走既有 K_T/LESP 通道**不动**(保证 tw0 推力不变);候选改动只作用于 geo_stall_vec 判定为分离/超几何失速的片条或扭转增量载荷,且 R1/R2 互斥(同一片条只能走一条失速后定律),R3 只改旋转**方向**不加力,R4 只改门控不加力。

**R1(首选,定量命中)——有限 AR 封顶复活失败修复 (1)**:单变量改动,把平板混合中的 Hoerner 常数替换:

    CD90(AR) = 1.98 − 0.81·(1 − exp(−20/AR))        [Montgomerie;AR=6 → 1.199]
    dp_sep   = q · CD90(AR) · sin²(α_eff)             [替换原 q·1.98·sin²]

  可选加 Lindenburg 饱和形式(同一文献族,仍零拟合):CN = CD90(AR)·sinα/(0.56+0.44·sinα),深攻角自动"dumped"。**预测**:tw22.5 过阻从 −6.1..−6.7 N 落到 −3.7..−4.1 N ≈ 所需 −3.8 N(A2-C3 算术)。AR 是量到的几何,无自由常数。
**R2(结构正解)——geo_stall_vec 换成 L-B 两通道矢量分裂**:废除标量矢量缩放,分离片条改为

    CN_sep = (dCN/dα)·((1+√f)/2)²·(α_e−α0)          [法向,下限 0.25×附着]
    CT_sep = −0.95·(dCN/dα)·α_e²·√f                  [切向吸力,η=0.95, L-B Table 2]
    CD     = CN·sinα + CT·cosα;  CL = CN·cosα − CT·sinα

  f(α) 用 Kirchhoff 分离点律 f = 1−0.3·exp((α−α1)/S1) / 0.04+0.66·exp((α1−α)/S2),S1/S2/α1 取自翼型**静态极曲线**(发表翼型属性,非对我们数据拟合)。治阻同时法向下限 0.25 保升——正是失败 (2) 缺的那一半。
**R3(方向修正)——扭转增量按 TP-1500 twist 扩展旋转**:对扭转增量的未获吸力,旋转目标从全局阻力轴改为**增量的局部弦法向**:

    δ_n = arctan(tanδ / cosΛ_le)                      [δ = 局部扭转/弯度角]
    dCA = −c_t·cos(arccos(K_T) − δ_n);  dCN = (c_t/cosΛ_le)·sin(arccos(K_T) − δ_n)

  纯几何分解,零常数;直接消除"增量继承基载荷 67° 后仰"这一 bug 级方向错置。与既有 K_T 通道兼容(只改它的旋转方向)。
**R4(门控修正)——LESP 门下不施加任何阻力旋转**:瞬时 LESP < LESP_crit 的片条,扭转增量按 K_T=1 全吸力记账(Ramesh 门:附着即全弦向吸力,les-C4);LESP_crit 用文献锚 0.14(SD7003@Re=1e5)做敏感性对照(0.11–0.21 带)。利用既有 LESP 通道,不新增力项。非定常上修(He 2020: 动态临界>静态)方向明确但 k=0.1→0.3 属外推,只作为 R4 的定性辩护,不引入速率修正常数。

**执行顺序**:先 R1 单独跑(一个常数改动,可直接对 gate);若斜率残余 >|0.06| N/°,叠加 R3(方向,与 R1 无重叠);R2 作为 R1 的结构化替代(互斥消融);R4 最后做门控敏感性。

## 4. 常数来源清单

**文献锚定(可直接引用)**:η=0.95(L-B Table 2, Bangga 2020);Kirchhoff 法向下限 0.25;Lindenburg 分母 0.56/0.44、γ=0.28√(r_nose/c)、半摩擦 0.5×0.0075;2D CD90=1.98(Hoerner);CD90(AR) 四拟合(1.11+0.018AR 等,@AR6→1.20±0.02);Viterna CDmax·sin²α 结构;K_T 闭式(TP-1500 eqs.7-10,输入 t/c、r/c、Re);δ_n 几何式(TP-1500 p.13);Polhamus Kv≈π、ΔCD=CL·tanα;C_s=2π·LESP²;LESP_crit: SD7003 0.21/0.18/0.14 @Re 1e4/3e4/1e5,平板 0.11@1e3,NACA0015 0.19@1100。

**外推/须声明(不得当作已证)**:L-B f 律常数 0.3/0.04/0.66 系 NACA0012/HH-02/SC-1095 家族拟合,非普适;Carlson K_T 属定常→扑动准定常外推(其系数对**我们**零拟合但非第一性);He/Mulleners 速率保吸力实测于 k=0.1,k~0.3 外推;膜+圆杆前缘无发表静态极曲线,S1/S2/α1 需用最接近薄翼型代理并披露;LESP_crit 运动学无关只是一阶近似(Kay 2022 低速率反例);ηs≈0.98 无一手出处,禁用。

## 5. 验收门(nc4/nc12 双网格,全过才收)

1. **扭转斜率**:gap_T 斜率从 −0.18 N/° 收敛到 **|slope| < 0.06 N/°**(tw 0→22.5 扫掠,nc4 与 nc12 均须满足且两网格差 < 网格研究既有容差)。
2. **升力保持**:全扭转扫掠升力 MAE ≤ **0.8 N**(不得复现失败 (2) 的 −5 N 屠升)。
3. **零扭转不变**:tw0 净推力不变(改动只经分离/增量通道生效,附着基线原样)。
4. **U6 常数不动**:U=6 m/s 的 c(U) 常数项不受影响(本案改动不得触碰与扭转无关的 U 标定残差)。
5. 失败回退:若 R1 后斜率仍 >|0.06|,按 §3 顺序叠加 R3 再验;任何一门失败即回退该单变量改动并记录消融。

---

## 附:核实状态说明

以上五角度共 26 条 claim 全部经对抗核实为 CONFIRMED(题录含逐条 corrected_citation,存档于 claim 验证记录);另有 3 条 REFUTED 入墓地:(i) "K_T Reynolds 地板塌缩到 ~0"(公式误引:指数写成乘子、丢平方、10^(4−3Mn) 写死为 10;实际 K_T≈0.1–0.5,且生产代码 `platform/_v2_robo.py` les_sep=='kt' 已实现正确指数形式);(ii) "Garrick 理论预言推力对 feathering 不敏感"(相反:Garrick 预测强敏感,θ=0→1 推力 100%→0——平台必须靠黏性重定向解释);(iii) "Kirchhoff-AIC VLM 系 Gopalarathnam 组"(实为 dos Santos & Marques 2018, J. Aircraft 55(2):887-891;Gopalarathnam 组是 decambering 家族)。

---

## 病灶#3 快测:les_att 三门全破 + 元判断(2026-07-20,GAP闭环)

力通道确诊 aoa15/f2.6 过推力主犯 = T_lesp(前缘吸力)深失速下给满额 +4.7N(D_prof 分离阻只 0.84 偏小)。测现成 les_att(dTs×fsep_le 深失速衰减):

| | v4 | les_att | 实测 |
|---|---|---|---|
| 巡航 aoa5/tw0 | T+0.39 | T−0.16(**dT−0.56**) | ~0 |
| aoa15/tw0 | +0.05 | −1.34 | −4.46 |
| aoa15/tw25 | −0.98 | −2.81 | −3.39 |
| aoa15/tw45 | −1.65 | −3.87 | −4.18 |

**三门全破**:①破坏巡航(−0.56,巡航下扑也高 α_eff 误触发);②衰减不够(tw0 −1.34 vs −4.46 差 3N);
③**斜率单调降**(−1.34→−3.87),实测中间峰(甩阻保升),**方向反**。

**根因(硬)**:准静态 fsep 门控 → "twist 增大→更失速→更衰减"→ 单调降;实测甩阻保升要
"twist feathering 在**推力相位**减 α_eff→恢复吸力",需**相位敏感动态失速**(R2 完整版),
非准静态开关能给。印证 caseA"准定常 Kirchhoff 只给 1/3 且方向错"(失败3)。

**★元判断:三病灶同墙**。病灶#1(dL/df)、#2(dT/dU)、#3(aoa15 甩阻)全部指向同一物理——
**动态失速/LEV 的相位敏感力矢量**(升力维持/分离阻塌缩/切向吸力甩阻),而 v4 是准定常
UVLM。逐病灶开关都撞这堵墙。统一正解 = 动态失速闭合升级(L-B 完整矢量分裂 or LDVM),
但那是大工程,且 LEV 粒子路线(rVPM)已 CLOSED。**v4 已逼近准定常框架能力边界。**
