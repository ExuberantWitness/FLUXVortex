# 文献研究综合报告:LEV 闭合去 ad-hoc 化(track 2)——放置/馈送规则、吸力保持-塌缩转变判据、附着阻力记账

范围:替换三处无文献锚的闭合选择——`lev_rollh=0.5`(规定卷起高度形状)、
`les_sep='plateau'` 的深顺桨(deep-feathering)语义(T2-deep 案:实测净推力在标称扭转 30
之外快速恶化,模型几乎不响应)、`attached_drag='faure'` 的选型依据。约束:co-design 就绪
(几何/Re 的预测函数或带适用域的原则性方法),零拟合(禁止对本组力数据调常数)。
交叉引用:`research_les_suction.md`(track 1,a0_crit 与吸力封顶)、`research_dsv_closure.md`
(lev_place='ansari' 忠实性已裁定)、`gap_t1_thrust_growth.md`(plateau 换代判决)。

## 1. 裁定(按三个子案)

### 子案 A:LEV 放置/馈送规则——谱系标准规则完备,`lev_rollh=0.5` 无文献锚

**裁定:放置与馈送在 Ramesh-LDVM/Hirato 谱系内有完整的、几何无关的运动学规则,可整体
替换我们的规定卷起;卷起(roll-up)在全部已核实文献中是涌现的(自由对流),从无"规定
高度形状"先例。**

- **1/3 放置规则(Ansari 规则)**:新 LEV 离散涡放在"前缘剪切点到上一枚已脱 LEV 的 1/3
  距离处"。UNSflow 源码逐行核实(`place_lev`, calcs.jl L409-427):
  `xloc = x_LE + (1/3)(x_lev_prev − x_LE)`;TEV 同规则对后缘。Hirato 博士论文 eq.(4.48)
  给出同式并明确归源 Ansari et al.:`x_{L,n−1} = (2/3)x_LE + (1/3)x_{L,n−2}`,且给出到
  涡环离散(vortex-ring)的转换(切分前一 LEV 面板)与前缘伪涡环(pseudo vortex ring)
  修正。**适用域:任何 DVM/涡环离散的 LEV 脱落,规则纯运动学,与翼型/平面形状/Re 无关
  → co-design 安全。**
- **脱落事件首涡**:放在 LE 下游 `0.5·v_LE·Δt`(v_LE = 前缘点当地速度,含运动+诱导),
  UNSflow 源码核实(levflag==0 分支)。同规则用于首枚 TEV(0.5·u·Δt)。
- **涡核半径 0.02c**(UNSflow `TwoDVort(...,0.02c,...)`):谱系标准数值正则化常数。
- **馈送(feeding)**:新 LEV 强度由"把瞬时 LESP 压回 LESP_crit"隐式确定;Hirato
  eq.(4.47) 给出两点割线迭代式。这与我们的 LESP 门实现同构,无需改动。
- **卷起是涌现的**:Hirato §4.8——卷起=涡量对流的 Cauchy 问题,由当地速度场驱动
  (自由流+镜像+尾迹+自诱导),数值配方是 Euler/多步法 + 局部时间步(§4.8 引 Takami/
  Summa,Lamb-Oseen 核黏性扩散正则化)。**"规定卷起高度形状 0.5"在 Hirato 论文与全部
  已核实谱系文献中不存在**——Fig.11 是观测到的卷起形态,不是闭合输入。
- `lev_place='ansari'`(LEV 片锚在前缘、覆于吸力面)已由 research_dsv_closure.md 裁定为
  忠实实现(Hirato Fig.4/5/11),本轮维持。

### 子案 B:吸力保持-塌缩转变判据(T2-deep 的平台语义边界)

**裁定:plateau(Katz-1981 保持假设,标准 LDVM 语义)与塌缩(Narsipur-2020 近零)不是
二选一,而是 LEV 生命周期的两个阶段;文献给出可计算的阶段转换判据——LEV 形成数
(formation number)T* ≈ 4 封顶馈送窗,窗内保持、窗外塌缩;恢复(重附着)由与俯仰速率
无关的临界 LESP 阈值触发。**

- **保持派(窗内)**:Katz 1981 假设分离前缘仍支持一定吸力(经 Hirato 论文溯源,标注为
  未证明假设;track 1 已录);Ramesh 2014 标准语义 = 剪切期间 A0 钉在 ±LESP_crit。我们
  v2 正确运动学下的再裁定(gap_t1 §6)证明 plateau 在中等扭转(≤tw30)横扫全部验收门
  ——**平台语义在"LEV 正在形成并馈送"的窗内是对的**。
- **塌缩派(窗外)**:Narsipur et al. 2020 (JFM 900, A25)——黏性 LESP 在前缘分离时跌至
  近零(track 1 已录,交叉引用)。Deparday & Mulleners 2019 (Phys. Fluids 31, 107104,
  OA209@Re=1e6):吸力演化由剪切层高度经"有效弯度+有效攻角"双通道决定;**失速发生
  (=吸力崩塌起点)的运动学无关判据是剪切层法向高度与 LEV 正环量的临界值**(其 Fig.7c-d,
  在两阶段转换点 t_t 处对所有运动恒定),LESP_crit 本身随俯仰非定常度上升——单常数
  LESP_crit 不是普适转变判据。
- **可计算的窗口封顶:涡形成数 T\* ≈ 4**。Gharib et al. 1998 谱系;俯仰平板直接值:
  Onoue & Breuer 2016 (JFM 793:229-247) LEV 环量在 **T\* = 3.7±0.3** 达最大;加速平板
  pinch-off ≈4 (Ringuette et al. 2007);孔口涡环统一到 ≈4 (Limbourg & Nedić 2021)。
  以上经 de Guyon/Mulleners 组综述文本核实(Exp. Fluids 64:128, 2023)。**这是文献中
  唯一跨构型稳健(O(4)±10%)、且在无粘求解器内可计算的 LEV 馈送终止判据。**
- **辅助判据(仅诊断,不作主门)**:
  - Widmann 博士论文(TU Darmstadt;平板,Re 1.7e4-2.8e4,k≈0.5):两种脱离机制
    ——bluff-body 型(几何长度限制,覆盖比 κ=2r_LEV/c→~1 时 LEV 停止生长)与边界层
    喷发型(黏性/无粘干扰,无几何长度);κ=f(k,Re,剪切层参数),**κ_trans 无解析值**;
    低 k 由弦长限制、高 k 由运动学限制。
  - Kissing et al. 2020 (Exp. Fluids 61:208):二次结构在涡雷诺数 Re_v=Γ_LEV/ν 达阈值时
    出现(TUDA 装置 3500-3900,均值≈3700±150;**BUAA 装置数值不同——非普适常数**);
    二次结构出现且剪切层角不再增大 → LEV 停止累积环量。
  - Sudharsan & Sharma 2024 (arXiv:2402.00990, LES@Re 1e4/6e4):LESP 反映失速起始,但
    低 Re 全弦长涡脱事件只有 BEF(边界熵流)能定位——LESP 门在低 Re 的空间局域性有限。
- **恢复(平台/塌缩→附着)判据**:Mulleners 组 2026 (arXiv:2505.12798,"Dynamic stall
  reattachment revisited";OA209,Re=9.2e5,k=0.05-0.10):攻角降到静态失速角以下是必要
  非充分条件;**重附着一致地在 LESP 回穿临界值 A0\* 时启动,A0\*∈[0.07,0.13],与俯仰
  速率无明确相关**(取上界 0.13 保证全工况);作者明示该值随 Re/翼型变化,非普适。
  恢复过程三阶段(反应延迟→波传播→松弛),延迟随非定常度减小。
- **对 T2-deep 的解释力**:深顺桨(tw>30,根部俯仰 ±15° 以上)= 超临界驻留时间长 →
  形成窗(T\*≈4)在驻留中段耗尽 → 真实吸力塌缩(Narsipur),而纯 plateau 把吸力钉在
  临界值直到 LESP 自然回落 → 模型推力几乎不随扭转恶化。**三态机在中等扭转下窗口极少
  耗尽(plateau 战果保持),在深顺桨下自动切换塌缩——单一机制同时解释"平台在 tw≤30 赢"
  与"tw>30 失真"。**

### 子案 C:附着阻力记账选型

**裁定:UVLM 附着诱导阻力两种记账(Katz 压力法 vs Joukowski 逐段 K-J 力)收敛到同一答案,
选型是收敛速度/稳健性问题;对我们的三维带弯度膜翼,文献明确判给 Katz 压力法。**

- Lambert & Dimitriadis(Liège,UVLM 诱导阻力注记,核实全文):2D 无弯度 Joukowski 快
  (复现 Simpson, Palacios & Murua 2013 结论);**3D 带弯度翼 Katz 压力法收敛显著更快,
  优势随弯度增大**(Joukowski 每面板 4 个取速点中 2 个远离弯度线,3D 有 M(N+1) 个这类点);
  收敛速度几乎与折合频率无关(测试 k≤0.8)。
- Faure et al. 2019 (AMM 69:32-46) 的 2D 条带记账(eq.22:D = F_N·sinα − F_A·cosα,
  F_A=轴向/吸力分量)与 Ramesh 2014 力学一致——我们 `attached_drag='faure'` 即该式的
  条带版,**与 Katz 压力法在 2D 条带极限一致,维持现状即合规**;唯一纪律是吸力奇点只
  经 LES 通道进一次(压力记账不得再含前缘奇性贡献),现实现已满足(通道分账核实于
  gap_t1 §2 表)。
- **适用域**:小-中振幅振荡 UVLM(文献测试域);我们的大振幅扑动属外推(标注),但
  选型结论(弯度→Katz)与振幅无关,风险低。co-design 规则:带弯度截面→Katz 压力法;
  对称/平板截面→Joukowski 亦可(仅收敛差异)。

## 2. 幸存题录表(本 track 逐条对抗核实;标注★=本会话全文/源码直验)

| # | 主张 | 状态 | 来源 |
|---|---|---|---|
| 1 | LDVM 新 LEV/TEV 放置:首涡 0.5·v_edge·Δt,后续 1/3 规则,涡核 0.02c | ★CONFIRMED(源码) | UNSflow `place_lev/place_tev`, calcs.jl L375-451, github.com/KiranRamesh-Aero/UnsteadyFlowSolvers.jl |
| 2 | 1/3 规则归源 Ansari;涡环离散转换+前缘伪涡环;馈送=LESP 割线迭代 | ★CONFIRMED | Hirato 2016 NCSU 博士论文 eq.(4.47)/(4.48), Fig.4.17-4.18;Ansari, Żbikowski & Knowles 2006, Proc IMechE G 220(2) |
| 3 | 卷起在 UVLM/LDVM 谱系是涌现的(当地速度场对流,局部时间步稳定化);无规定高度先例 | ★CONFIRMED | Hirato 2016 §4.8;期刊版 Hirato et al., J. Aircraft 56(4) 2019 / AIAA J 62:3356 2024 |
| 4 | 失速期吸力演化由剪切层高度决定(有效弯度+有效攻角);失速起始的运动学无关判据=临界剪切层高度+临界环量;LESP_crit 随非定常度增大 | ★CONFIRMED | Deparday & Mulleners 2019, Phys. Fluids 31:107104 (doi:10.1063/1.5121312), OA209@Re=1e6 |
| 5 | LEV 环量封顶于形成数 T\*≈4(俯仰平板 3.7±0.3;加速平板≈4;涡环≈4) | ★CONFIRMED | Onoue & Breuer 2016 JFM 793:229;Ringuette et al. 2007;Gharib et al. 1998;综述核实于 de Guyon & Mulleners 组, Exp. Fluids 64:128 (2023) |
| 6 | 两种 LEV 脱离机制(bluff-body vs 边界层喷发);κ=2r/c 覆盖比;κ_trans 无解析值,κ=f(k,Re,剪切层);低k弦限/高k运动学限 | ★CONFIRMED | Widmann 2015 TU Darmstadt 博士论文 §2.6, eq.(2.12), 摘要 |
| 7 | 二次结构onset在 Re_v=Γ/ν≈3700±150(TUDA;设施相关非普适);伴随剪切层角停增 → LEV 停止累积环量 | ★CONFIRMED | Kissing et al. 2020, Exp. Fluids 61:208 (arXiv:2003.13763) |
| 8 | 重附着由临界 LESP 阈值 A0\*∈[0.07,0.13] 触发,与俯仰速率无关;非普适值,随 Re/翼型变化;恢复三阶段 | ★CONFIRMED | Mulleners 组 2026, arXiv:2505.12798v3, OA209@Re=9.2e5 |
| 9 | 低 Re 下 LESP 反映失速起始但漏检全弦涡脱;BEF 可空间定位 | ★CONFIRMED | Sudharsan & Sharma 2024, arXiv:2402.00990 (LES, SD7003, Re 1e4/6e4) |
| 10 | 3D 带弯度翼诱导阻力:Katz 压力法收敛远快于 Joukowski,优势随弯度增;k 无关 | ★CONFIRMED | Lambert & Dimitriadis, Univ. Liège(注记,完成 Simpson et al. 2013 AIAA J 51(12) 的弯度扩展) |
| 11 | 2D LDVM 力记账 L/D 由 F_N/F_A 旋转,与 Ramesh 一致(faure 记账合法性) | ★CONFIRMED | Faure, Dumas, Drouet & Montagnier 2019, Appl. Math. Model. 69:32-46, eq.(22) |
| 12 | 深顺桨区(15°-45° 俯仰)LEV 合并/延迟形成,高 k 时枢轴位置影响增大 | CONFIRMED(定性) | Aravind, Seshadri & De, arXiv:2108.06275 (NACA0012, Re=3000, k=0.1/0.5) |
| 跨 | 超临界吸力保持=Katz-1981 未证明假设;黏性真相=分离时近零(Narsipur);a0_crit 文献值 0.14@Re1e5(SD7003) | 跨 track 引用 | 见 research_les_suction.md 题录(track 1) |

## 3. VERDICT:co-design 就绪的推荐闭合

**三态吸力状态机(逐条带,单变量可实现,零新拟合常数):**

```
状态 S0 附着:    LESP < LESP_crit          → 全吸力(现有通道,含饱和帽)
状态 S1 脱涡馈送: LESP ≥ LESP_crit 且 T̂ < T* → plateau(吸力钉在临界值,标准 LDVM 语义)
状态 S2 脱离塌缩: T̂ ≥ T*                    → 弦向吸力 → 0(Narsipur 近零)
恢复 S2→S0:      LESP 回穿 A0*(结构文献锚,值需按截面校验;过渡期用 LESP<LESP_crit 代)
其中 T̂ = ∫ ũ_LE dt / c(自脱落事件起算的涡形成时间,ũ_LE=前缘剪切馈送速度,
求解器现成量),T* = 4(带 3.7±0.3 文献带宽)。
```

- **中等扭转(tw≤30)**:窗口不耗尽,S1 全程 → 退化为现产 plateau,已过全部验收门。
- **深顺桨(tw>30/T2-deep)**:超临界驻留 > 形成窗 → S2 塌缩 → 推力随扭转恶化,方向
  与实测一致。
- **放置/馈送**:维持 `lev_place='ansari'` + LESP 馈送;卷起改涌现(自由对流+局部子步
  +Lamb-Oseen 核 0.02c),`lev_rollh=0.5` 降级为可选数值正则化并标注"无文献锚";若
  涌现卷起在 nc4 数值不稳,允许保留 rollh 作纯数值限幅,但不得进力学记账。
- **附着阻力**:维持 faure/Katz 压力记账(2D 条带=Katz 极限);若未来升三维整装 UVLM
  阻力,选 Katz 压力法(弯度翼判决,题录 10),禁止与 Joukowski 逐段力混用(双计)。
- **co-design 适用域**:规则 1/3、0.5Δt、0.02c、T\*≈4 均与平面形状/截面/质量刚度布局
  无关,可直接随任意设计点携带;LESP_crit(截面函数)由 track 1 方案供给(Ramesh 表 /
  Ramanathan-2023 边界层临界法);A0\* 与剪切层临界值属截面+Re 特定,跨设计点须重校验
  (结构可携带,数值不可)。

## 4. 实现配方(单变量步进,不与现有通道双计)

1. **步 R1(卷起涌现)**:关 `lev_rollh` 力学参与,LEV 片自由对流(现有 Biot-Savart 基建),
   局部子步稳定化;对照 tw22.5/f2.3 守卫钉(r["L"]=+5.3736 / r["T"]=+8.6010,
   L_wind=+6.3364 / T_wind=−1.9430,±0.15)。此步只许动卷起,平台语义不动。
2. **步 R2(形成数门)**:在 les_sep='plateau' 上加 T̂ 累积器(每条带,自 LESP 上穿起算,
   LESP 回落归零),T̂≥4 → 该条带弦向吸力置零(S2)。单开关 `les_sep='plateau_fn'`。
   预期:tw≤30 全部条件逐位不变(窗口不耗尽=判据惰性),tw45 推力显著下修。
3. **步 R3(恢复迟滞,可选)**:S2→S0 用 A0\* 迟滞替代即时回落;A0\* 无本截面值,先用
   LESP_crit 本身(单参数,不新增常数),标注外推;仅当 R2 后 tw45 相位仍失真才开。
4. **记账纪律**:S2 塌缩只作用于弦向(推力)吸力分量;法向力照旧走 Bernoulli/A0-A1-A2
   通道(Hirato 路线,research_dsv_closure.md);吸力奇点只进 LES 通道一次;d_para(T3 案)
   不动——本 track 改动若把 U6 常数偏置改型,即越界,回退。

## 5. 常数清单:文献锚 vs 外推

| 常数 | 值 | 性质 |
|---|---|---|
| 1/3 放置 | 1/3 | 文献锚(Ansari 2006;Hirato eq.4.48;UNSflow 源码) |
| 首涡放置 | 0.5·v_edge·Δt | 文献锚(UNSflow 源码;Ramesh 2014 谱系) |
| 涡核 | 0.02c | 文献锚(UNSflow;数值正则化,非物理参数) |
| 形成数 T\* | 4(带 3.7±0.3) | 文献锚(Onoue & Breuer 2016 俯仰平板直接值;Gharib 谱系跨构型);**Re 外推**:直接测量在 Re≲1e4-1e5,我们 Re≈1.5e5,方向稳健、带宽已标 |
| A0\*(重附着) | 0.07-0.13(OA209@Re9.2e5) | **结构文献锚、数值外推**——作者明示非普适;本模型如用取 LESP_crit 代,零新常数 |
| κ_trans, Re_v≈3700 | — | 仅诊断,不入闭合(κ 无解析值;Re_v 设施相关) |
| lev_rollh=0.5 | 0.5 | **无文献锚**,降级为数值限幅或删除 |
| 阻力记账选型 | Katz 压力法(弯度翼) | 文献锚(Lambert & Dimitriadis;Simpson 2013);大振幅属外推(选型结论振幅无关) |

## 6. 验收门(118 条件扫掠)

**必须不回退(带内保持):**
- 巡航 tw0/f2.3 双通道 |Δ|<0.15(不可回退门);
- U8/aoa5 频率线推力 gap 全带内(现 +0.16/+0.17/+0.14/+0.08/−0.30);
- 扭转行 tw15/22.5/30(现 +0.15/+0.08/−0.07)——R2 的形成数门在这些点必须惰性
  (逐位一致或 |Δ|<0.05),否则 T\* 门参数化错误,回退;
- 升力 MAE ≤1.0N(现 0.89)、推力 MAE ≤0.8N(现 0.67);
- U6 保持"干净常数偏置"结构(~+1.9N 两频率同值)——归 T3/d_para,本案不得改其形状。

**必须改善:**
- T2-deep:tw45 净推力方向性——模型须再现"tw30 之外实测净推力快速恶化"的趋势
  (现产 plateau 几乎平坦);量化门:tw30→tw45 的 ΔT 符号与实测一致,幅值达实测降幅
  的 50% 以上(方向门先行,不设精确数值门以免诱导拟合);
- tw45 回放爆炸(案 D-aero)必须消失(涌现卷起或 S2 环量封顶其一应治愈);
- aoa15/f2.6 门(现 plateau +0.44/+0.43)不得因 S2 门恶化超 0.3。

## 7. 存疑与墓地(明确列出)

1. **`lev_rollh=0.5` 作为文献方法:REFUTED**。Hirato Fig.11 是结果图非闭合输入;谱系内
   卷起一律涌现。保留只能以"数值限幅"身份。
2. **plateau 普适于任意顺桨深度:REFUTED(本案立案依据)**。保持假设(Katz 1981)只在
   LEV 馈送窗内成立;窗外与 Narsipur 塌缩、Deparday 剪切层证据冲突;T2-deep 实测即反例。
3. **Re_v≈3700 作为普适转变常数:REFUTED**。Kissing 原文自证设施相关(TUDA vs BUAA
   不同);仅可作诊断。
4. **κ_trans 可解析定值:不支持**。Widmann 原文明示无解析值且依赖运动学。
5. **A0\* 数值跨截面携带:不支持**。原文明示随 Re/翼型变化;只携带"存在速率无关阈值"
   这一结构。
6. **形成数 T\*=4 在 Re~1.5e5、圆杆前缘膜翼的定量精度:外推**。跨构型稳健性(环/平板/
   俯仰板皆 O(4))是采纳理由,精确值±10% 带宽必须随敏感性扫描报告。
7. **本报告题录来源说明**:上游裁定器的 kept/refuted JSON 未随任务送达(模板未插值);
   题录表由本会话验证组下载的全文/源码直验重建(★项均为本地全文核实),证据强度不低于
   转录,但与裁定器编号系统的对应关系未核。
