# FLUXV RoboEagle 气动模型 — Claim Chain 树

> **代码本体（2026-07-27 用户裁定重构）**: 仿真结构已重构为 claim chain 树形式。
> `platform/claim_nodes/*.yaml` = DevReady 资产（id/claim/state/freeze/evidence/refs/memory/guard/children）；
> `platform/claim_dag.py` = ClaimNode + ClaimDAG（遍历/验证/修改检查）。
> **修改规则**: validated 节点（N1/N4）freeze 禁改；已证伪节点（N5/N6）禁重走；
> partial/open 节点可动但需证据+归因（四步流程）。
> 本文档 = 树的人类可读视图，代码本体 = 树的机器可执行形式。

> 纪律(memory [[feedback_claim_chain_research]],用户裁定 2026-07-26):
> 模型 = 命题树。每个节点带证据状态;一切修改 = 树的有证据改写;validated 节点 = 不可动空间。
> 状态:`✓validated` `~partial` `○open` `✗falsified(部分或全部)` `∎dead-end 留档`。
> 每 Phase 结束更新节点状态 + 证据链接。证伪路径永久留档,禁止重走。

```
R: UVLM + L-B 修正层在 4 参数空间(f/tw/aoa/U)复现 RoboEagle 实测 L/D
│   (验收:轨迹相似性 v3 八族捕获 + MAE 门,哲学=规律捕获为主)
│
├─ N1 UVLM 附着流主引擎                     ✓validated
│     证据: 稳态 96%;引擎对 FLUXVortex 0.1%;le 网格生产区间收敛(㊶㊴);
│           尾迹 1 周期截断充分(㊾③,推力免疫/升力±4% 有界);运动学 v2 口径(㊷)
│     禁动: 主解算器/尾迹/网格/运动学
│
├─ N2 Kirchhoff 分离砍量(L-B 双时滞)        ~partial
│   ├─ N2.1 f(α) 零拟合反演 + Tp/Tf 双时滞   ✓validated (dL/df 治愈 Pearson −0.07→+0.93,㊽)
│   ├─ N2.2 砍量方向 = 全局风升力向           ✗ 力闭合不完整
│   │       风向施加构造性零推力(_v2_robo.py 精确相消)，与 BL/Sheng 的
│   │       normal/chordwise 分立簿记不一致；但 Fig18 源图审计发现 U6/U10
│   │       推力曲线反标，旧“dT/dU 反号→N2.2”因果边已撤销，不再视为 D1 主犯。
│   └─ N2.3 a3d 2D→3D 失速移位              ○open (有效参数,禁跨截面外推,㊹)
│   └─ N2.4 hybrid 回补无 Kirchhoff 衰减      ✗ E1 嫌疑 (2026-07-27)
│           证据(LB_DIAG2 通道分解,aoa15/f2.6/tw22.5): 中外侧条带 f_qs 被 clip 在 1
│           (aeff 2-13° < 失速角) → 判"永附着" → loss≈0.32 砍量被 F_LB 准定常**满额**回补
│           (周期均 +4.64N,峰 +17.6N),叠在 UVLM 力上 → L=+25.1 vs 缓存 +15.6/实测 ~15。
│           物理上 L-B hybrid 的回补应同样过 Kirchhoff K(f2) 衰减;现实现 = 满额准定常
│           截面力,附着条带近似 E2(有效),但叠加在"UVLM 尾迹反旋已抑制有效环量"的力上
│           → 双计嫌疑。**候选 F 缓存无此病灶** → 07-25 调用含未重建要素
│           (主嫌 lb_lesp_crit=0.23:aoa0/aoa15 两角区方向全对,但 origin 无记录)。
│           裁决待定: (i) 0.23=v4 时代遗留生产值 → 缓存=closure+0.23,病灶=配错常数;
│           (ii) 0.18 也是生产值 → 缓存含更深要素,E1=结构错件。
│   └─ N2.5 滞后静态极曲线弦向力闭合          ○open (2026-07-27)
│           IAG/Bangga: 从 lagged-alpha 静态极曲线取得 CT；只能替代现有
│           attached_drag/UIUC，禁止叠加。先过唯一簿记与量级门，非 D1 吸收器。
│   └─ N2.6 移动壁面环量—黏性库存—三维释放链 ~partial
│       ├─ N2.6a 双侧移动壁面环量源           ~partial
│       ├─ N2.6b 曲面三维IBL/矢量库存         ○open
│       │   ├─ N2.6b2 无闭合矢量守恒账        ✓validated❄
│       │   ├─ N2.6b3 三方程张量IBL骨架       ✓validated❄
│       │   └─ N2.6b4 横流/剪切/耗散/转捩闭合 ○open
│       │       ├─ N2.6b4d 有限IBL矩唯一决定VES ✗falsified❄
│       │       ├─ N2.6b4e profile-edge同源投影 ✓validated❄
│       │       └─ N2.6b4f 近壁剖面/层边缘动力闭合 ○open
│       │           ├─ N2.6b4f1 积分量回归=完整profile/edge ✗falsified❄
│       │           ├─ N2.6b4f2 四加二状态条件充分性 H0 ○open
│       │           ├─ N2.6b4f3 无载荷近壁场数据契约 ✓validated❄
│       │           │   ├─ N2.6b4f3a 公开场资产联合资格审计 ✓validated❄
│       │           │   └─ N2.6b4f3b 目标域合格场取得/独立审计 ○open
│       │           │       ├─ N2.6b4f3b1 PBFM表面时空场≠三维体速度场 ✓validated❄
│       │           │       └─ N2.6b4f3b2 第二轮公开性审计：压力可验而空间场仍缺 ✓validated❄
│       │           └─ N2.6b4f4 受矩约束时序profile-edge decoder ○open
│       └─ N2.6c 三维分离流形上的守恒释放     ○open
│           ├─ N2.6c1 物质spike分离流形       ○open
│           │   ├─ N2.6c1a Cf=0/Hcrit生产流形 ✗falsified
│           │   ├─ N2.6c1b 三维物质曲率脊参考oracle ○open
│           │   │   ├─ N2.6c1b1 材料面Weingarten变化身份 ✓validated❄
│           │   │   ├─ N2.6c1b2 独立场驱动材料面流映射 ○open
│           │   │   │   ├─ N2.6c1b2a 无载荷flow-map数值身份 ✓validated❄
│           │   │   │   └─ N2.6c1b2b 独立场插值/材料面推进 ○open
│           │   │   │       └─ N2.6c1b2b0 原始二维PIV线性时间插值 ✗falsified❄
│           │   │   └─ N2.6c1b3 正值主曲率变化backbone ○open
│           │   │       ├─ N2.6c1b3a backbone微分条件身份 ✓validated❄
│           │   │       └─ N2.6c1b3b 多层连通/场级验证 ○open
│           │   └─ N2.6c1c IBL驱动生产spike近似 ○open
│           └─ N2.6c2 涡量—卷吸新生面带      ○open
│               ├─ N2.6c2a 纯DDE是完整分离层 ✗falsified❄
│               ├─ N2.6c2b DDE+VES状态身份   ✓validated❄
│               └─ N2.6c2c 光滑前缘release junction ○open
│                   └─ N2.6c2c1 移动流形相对IBL通量 ✓validated❄
│
├─ N3 ds 涡升增强 = LEV 均值升力             ~partial
│   ├─ N3.1 LEV 触发、供给与有限寿命事件状态 ~partial (2026-07-27)
│   │       验证区 aoa5-15/U8-10 ✓;角区 aoa0/U6 无鉴别力过供(D2 回归)
│   │       死路留档: signed 驱动(候选 G,aoa0 过抵消 −0.77/杀 aoa5)、
│   │               f2gate(loss_frac 随 f 降压平 f 依赖)、全矢量(x 投影归因错)
│   │       精确 D3 归因:tw0→45 N1 −0.930N、N2 −0.869N，唯 N3 +4.589N 翻转总趋势
│   │       学科裁决:A0/LESP 是 onset/feed proxy，不是已形成 LEV 的持续力幅值
│   │       ├─ N3.1.0 |A0|−crit 直接作为持续正涡力 ✗falsified
│   │       ├─ N3.1a 形成数 T_hat<T*=4 只门控馈送 ✗ D3主因证伪（B1激活门失败）
│   │       ├─ N3.1b LEV 事件翼面身份记忆      ✗ D3修复证伪（B2在当前A0上惰性）
│   │       ├─ N3.1c 三维展向相干性            ○open（无力映射前仅诊断）
│   │       ├─ N3.1d LESP 起始与持续载荷分离   ~partial
│   │       │   ├─ N3.1d1 alpha_kin 替代 A0    ✗falsified（B4 二者同向上升）
│   │       │   └─ N3.1d2 A0 仅作 onset/feed   ~partial（文献支持，待本案判别）
│   │       ├─ N3.1e 无符号分离亏损→有限事件 ✗falsified（D4 双 NO-GO）
│   │       ├─ N3.1g signed CV/CNv翼面—展向抵消 ✗falsified（D5）
│   │       ├─ N3.1h LEV位置/运动或等价角冲量  ~partial（D9后开放D10影子态）
│   │       │   ├─ N3.1h1 强度+位置/运动闭合总力 ~partial（只诊断）
│   │       │   └─ N3.1h2 有限矩唯一重构面压力 ✗falsified❄
│   │       ├─ N3.1i 空间涡片→逐面板压力       ○open（co-design生产方向）
│   │       │   ├─ N3.1i0 现有Hirato已忠实     ✗falsified❄
│   │       │   ├─ N3.1i1 有限翼涡片→面板压力  ○open
│   │       │   │   ├─ N3.1i1a Eq.6 LESP算子身份 ✓validated❄
│   │       │   │   ├─ N3.1i1e Case1/2网格收敛拓扑 ~partial（11/12）
│   │       │   │   ├─ N3.1i1b Eq.7拓扑/自由态 ~partial（无力shadow）
│   │       │   │   │   ├─ N3.1i1b1 首环=LE+U∞dt ✗falsified❄
│   │       │   │   │   ├─ N3.1i1b2 速度型首环3D适配 ~partial
│   │       │   │   │   └─ N3.1i1b3 单一固定核有文献身份 ✗falsified❄
│   │       │   │   ├─ N3.1i1c Eq.9/17压力账   ~partial（仅恒等式）
│   │       │   │   │   ├─ N3.1i1c1 TEV首排=1.0Udt ✗falsified❄
│   │       │   │   │   ├─ N3.1i1c2 Eq.24=current Euler ✗falsified❄
│   │       │   │   │   └─ N3.1i1c3 固定核族+历史速度 ○open
│   │       │   │   └─ N3.1i1d 完整场→压力晋升 ○open
│   │       │   │       ├─ N3.1i1d1 pressure-only输出门≠空间态身份 ✓validated❄
│   │       │   │       └─ N3.1i1d2 P0动态压力几何—相位—守恒观测账 ✓validated❄
│   │       │   ├─ N3.1i2 面板载荷→结构H^T     ~partial
│   │       │   └─ N3.1i3 空间基ROM替代涡片    ○open（需跨设计域场证据）
│   │       ├─ N3.1j 高阶连续涡面空间态          ~partial
│   │       │   └─ N3.1j3 连续涡面→统一面板压力 ○open
│   │       │       └─ N3.1j3b 双侧移动面Bernoulli ~partial
│   │       │           ├─ N3.1j3b3 弯曲多patch势/物质率 ✓validated❄
│   │       │           ├─ N3.1j3b4 压力跳唯一决定双侧Cp/厚翼Ct ✗falsified❄
│   │       │           ├─ N3.1j3b5 双侧Cp→gauge-free ΔCp/Cn观测 ✓validated❄
│   │       │           └─ N3.1j3b6 实际厚度均势/源面+黏性双侧压力 ~partial
│   │       │               ├─ N3.1j3b6a 实际厚度气动几何先决件 ✓validated❄
│   │       │               ├─ N3.1j3b6b 厚度/升力压力分别成力后相加 ✗falsified❄
│   │       │               ├─ N3.1j3b6c 原始N1环场+source条件化 ✗falsified❄
│   │       │               │   ├─ N3.1j3b6c1 N1→实际厚壳只读速度场 ✓validated❄
│   │       │               │   ├─ N3.1j3b6c2 薄面Kutta自动传递到有限base ✗falsified❄
│   │       │               │   └─ N3.1j3b6c3 多残差+守恒base-wake状态 ~partial
│   │       │               └─ N3.1j3b6d N1环量约束的实际边界source-doublet重表示 ~partial
│   │       │                   ├─ N3.1j3b6d1 Morino实际边界势方程oracle ✓validated❄
│   │       │                   ├─ N3.1j3b6d2 常元ring普通值直接给物面Cp ✗falsified❄
│   │       │                   └─ N3.1j3b6d3 连续P2 Galerkin+片上有限部梯度 ~partial
│   │       │                       ├─ N3.1j3b6d3a 连续P2 trace+弱方程代数 ✓validated❄
│   │       │                       ├─ N3.1j3b6d3b 独立tensor-Duffy相邻弱积分 ✗falsified❄
│   │       │                       ├─ N3.1j3b6d3c 单源面径向势自动闭合外积分 ✗falsified❄
│   │       │                       ├─ N3.1j3b6d3d 同面/共边/共点成对消奇 ✓validated❄
│   │       │                       └─ N3.1j3b6d3e 附着球P2/P1势—速度—Cp门 ✓validated❄
│   │       │                   ├─ N3.1j3b6d4 单值闭合trace可承载非零环量 ✗falsified❄
│   │       │                   ├─ N3.1j3b6d5 分类势切面/wake跃变拓扑缺件 ✓validated❄
│   │       │                   ├─ N3.1j3b6d6 普通端点P2直接解析Kutta cusp ✗falsified❄
│   │       │                   ├─ N3.1j3b6d7 Kutta-enriched cusp+独立gauge ✓validated❄
│   │       │                   ├─ N3.1j3b6d8 印刷two-minus有限角动量式直接联用 ✗falsified❄
│   │       │                   ├─ N3.1j3b6d9 signed-control-volume有限角形成 ✓validated❄
│   │       │                   ├─ N3.1j3b6d10 单junction可直接代表有限base ✗falsified❄
│   │       │                   ├─ N3.1j3b6d11 双front+base/confluence拓扑必要性 ✓validated❄
│   │       │                   ├─ N3.1j3b6d12 外侧势流唯一闭合双front ✗falsified❄
│   │       │                   └─ N3.1j3b6d13 base-side/confluence可辨识性缺件 ✓validated❄
│   │       └─ N3.1f 显式 rVPM/粒子 LEV       ∎dead_end❄
│   ├─ N3.2 Tv=6 记忆累积                    ✓validated (文献锚 L-B/Bangga;瞬时式=低通过敏)
│   └─ N3.3 面板法向施加 = 涡诱导阻          ✓validated (aoa15 推力 +2.3→+0.4 来源)
│
├─ N4 推力簿记(CT 吸力一致式)              ✓validated-诊断
│     证据: v4 时代双重记账 +3.6N 证伪;一致形式 −(1−√f2)²/4·ηCLaα² 推导严格零新常数;
│           完整 (1−√f2) 形式过拖阻 5× 证伪。候选 F 中 CT off(该诊断是地基)
│
├─ N5 扭转耦合响应                           ✗ D3 证伪 (2026-07-26)
│     证据: 模型升力随扭转单调升(chop-release 嫌疑),实测 tw15-22 峰后滚落
│           (19|d aoa0 tw35 +4.7N;17|b tw15 峰 miss −2.1N)
│     归因已定: N3.1 派生观测失败；N5 只作守卫，不提供独立力学旋钮
│     └─ N5.1 实测扭转—扑动相位符号身份       ~partial（输入已闭合、物理未闭合）
│           ├─ N5.1a +90与Figure10在代码坐标同义 ✗falsified
│           ├─ N5.1b 仅翻转-90恢复完整twist族   ✗falsified
│           └─ N5.1c 装配角→theta/psi坐标审计   ✓validated❄（D9: -90）
│
└─ N6 d_para 钝体阻                          ∎对 T3b 证伪 (2026-07-17 ㊺,2026-07-26 复核)
      证据: 物理值 0.5×(U/8)²(照片正算∩滑翔极曲线双锚);T3b 指纹随 U 衰减,方向反
```

## 当前病灶 → 节点映射(v4.2 战役,2026-07-26;E1/E2 2026-07-27)

| 病灶 | 节点 | 方向预判 | Phase |
|---|---|---|---|
| D1 推力阻力量级不足、dT/dU 同号但偏弱 | N2.5 open；实验系统边界 open | 先分离 wing/rig 账；N2.5 只作替代式弦向闭合 | A重定向 |
| D2 ds 角区过供(U6/aoa0 回归) | N3.1 缺位置/运动与相位状态 | D6 先判同装置半周期指纹；禁旧标量重排 | B |
| D3 扭转扫升力形状 | `A0/|CV|/signed CV` 均单调；N5 为派生观测 | D6 对账 Figure16 瞬时峰与半周期有效攻角 | B |
| **E1 closure 预设 aoa15 角区 +10N**(代码 v41 L=+25.1 vs 缓存 +15.6) | N2.4 hybrid 回补 | 先裁决(i)/(ii);若(ii)则回补过 K(f2) 衰减 | −1.3b |
| **E2 closure 预设 aoa0 角区 −3N**(代码 L=+1.7 vs 缓存 +4.8) | N3.1 驱动/crit | 与 E1 同源嫌疑(crit=0.23 全角区方向对) | −1.3b |

**E1/E2 性质**:晋升验证关口发现的**代码-缓存不一致**——非缓存数据的模型病灶
(07-25 缓存 aoa15 = +15.59 贴实测 15.14,无病灶)。118 重扫(lb_sweep118.py 已备,
12 点快环已跑)在全 118 之前**被本病灶阻塞**(用户裁定 2026-07-27:先归因 E1 再扫)。
缓存 json 仍为 v4.1/v4_legacy 官方数值基线。

## 已证伪候选总账(禁重走)

- chop 方向: body-z(sin(aoa) 泄漏 +1.5N@aoa15)、全矢量(两通道耦合不清,x 投影仅 0.3-0.5N)
- ds: signed 驱动(G)、f2gate、全矢量、瞬时式(无 Tv 记忆)
- CT: |CT| 前向记账(双计 +3.6N)、完整 (1−√f2)(过拖 5×)
- sep_drag f² 型(候选 C 死因:dT/df 形状 0.995→−0.03)
- Prandtl cla3d 修 chop(Kirchhoff 比值对 CLa 缩放自相似,近零效)
- d_para 常数吸收 T3b(用户红线:禁常数吸收)
- 杂交: hybrid=1(量级 18.9 过冲)、纯砍无 ds(斜率够量级缺 3N)

## 更新日志

- 2026-07-28: **N3.1j3b6d S2e有限base双front动力学可辨识性**。
  固定标准开尾缘几何与外侧无量纲速度`1`，只把未观测base-side速度比作为
  `{0,.25,.5,.75}`非唯一性witness；禁止把比值当参数。四组均通过S2c
  局部角/方向/Kutta/环量率/signed动量门，最大归一残差`6.34e-17`、
  片侧速度非负；上下角family差仅`3.57e-6`。然而相同外侧输入产生形成角、
  片强、相对速度spread `35.41°/0.6612/0.1456`。故outer-only唯一闭合
  N3.1j3b6d12 falsified/frozen；base-side/confluence或等价可观测状态的
  必要性N3.1j3b6d13 validated/frozen，但未给任何closure。finite-base
  生产仍NO-GO；为避免局部base问题拖住主线，三维material-wake研究只可在
  另行预登记的角点重合finite-angle canonical上继续，禁止借此替换冻结的
  NACA-2406开尾缘生产几何。
- 2026-07-28: **N3.1j3b6d S2d有限钝尾缘拓扑可辨识性裁决**。
  S2c只闭合同点单junction，故在实现前冻结NACA-2406标准开/闭尾缘
  `a4=-0.1015→-0.1036`的`f={1,.5,.25,0}`continuation。标准开尾缘
  `h/c=0.00126`时，一个材料起点同时附着两角的最优minimax残差严格为
  `h/2=0.00063c`；全部非零base归一残差均为`0.5`，两个显式锚点残差为
  `0`，且绝对残差单调趋向sharp limit。`f=0`两角重合后仍保留
  `8.295°`有限角，准确回到S2c适用域。故“中点单junction直接代表有限
  base”N3.1j3b6d10 falsified/frozen；“双front或等价有限宽interface+
  base/confluence region”N3.1j3b6d11仅在二维拓扑必要性范围
  validated/frozen。两front动力学、base压力、material history、3D及
  production仍未授权。
- 2026-07-28: **N3.1j3b6d S2c有限角尖缘形成恒等式裁决**。
  在实现前冻结`40°`楔角、对称/镜像/单侧停滞/尺度六个速度族及角度、
  方向、Kutta强度、环量率、动量、非负出流、镜像与协变门。首次执行只有
  Xia--Mohseni印刷Eq.4.19的two-minus动量残差失败
  (`0.09523>1e-13`)；按论文自身`u1+,u2-<=0`入流和
  `ug+,ug->=0`出流约定，印刷式三项必然同号，故不是数值误差。
  预登记后仅从原始signed mass与切/法向动量控制体式重新消元，得到
  plus-flux残差；其余公式、算例和阈值均不变。重跑后角/方向/Kutta/
  环量率/signed动量最大残差分别为
  `0/1.85e-17/0/0/4.11e-18`，镜像`5.55e-17`，尺度、bisector与
  切向极限均为0。因此印刷直用路径N3.1j3b6d8 falsified/frozen，
  守恒形成身份N3.1j3b6d9仅在二维高Re单尖锐junction范围
  validated/frozen。有限base、物质势/Kelvin历史、三维junction及生产
  压力仍未解决，父节点保持partial。
- 2026-07-28: **N3.1j3b6d S2b尖尾缘Kutta选环量—统一压力门**。
  S2a只接受给定Gamma，故先以Joukowski有限厚度尖尾缘解析解隔离“Kutta
  如何选环量”，拒绝直接跳三维。首次普通P2虽通过全局速度/Cp/力，却在
  cusp出现尾缘侧Cp差`0.1096>0.02`和规范速度变化
  `8.42e-10>1e-12`；相同点解析Cp差仅`1.94e-5`，故
  N3.1j3b6d6 falsified/frozen。预登记后只在两个cusp单元加入保留三节点值
  且严格施加共同导数零点的Kutta enrichment，并把势规范从可微trace系数
  分离；全部原门随即通过。256面速度/Cp RMS
  `1.63e-5/4.08e-5`、尾缘Cp差`1.94e-5`、升力误差`4.89e-11`、
  drag`8.01e-11`，规范速度/力变化均为0。故N3.1j3b6d7只在steady
  sharp-cusp范围validated/frozen；finite-base、Kelvin、三维wake和生产
  压力仍未授权。
- 2026-07-28: **N3.1j3b6d S2a非零环量势切面拓扑裁决**。
  在实现前冻结二维圆柱`8/16/32/64`曲线P2面板、`Gamma=0.8`、解析
  势/切向速度/Cp、Kutta–Joukowski力、规范门和阈值。闭合单值trace的
  最大环量仅`7.22e-16`，证明其梯度闭路积分按拓扑严格望远镜相消；
  提高阶次或网格不能让它承担非零环量，故N3.1j3b6d4
  falsified/frozen。复制分类cut两端自由度后，势跃变/环量相对误差
  `2.78e-16/1.11e-15`；64面板速度RMS`6.45e-4`、升力相对误差
  `9.67e-8`、阻力`5.61e-16`，常势平移后速度/力变化
  `8.04e-14/1.45e-14`。全部预登记门通过，故N3.1j3b6d5仅在“3D lifting
  domain必须显式拥有classified wake jump”的拓扑范围内
  validated/frozen。有限翼wake、finite-base Kutta、Kelvin、物质历史和
  生产压力仍未验证。
- 2026-07-28: **N3.1j3b6d S1连续P2成对弱算子与附着球压力门**。
  普通独立tensor-Duffy相邻积分和精确单源径向势+普通外积分分别因固定
  网格Cp阶次变化`0.009519`和near-edge Cauchy`24.25`失败，故
  N3.1j3b6d3b/c falsified/frozen。依Erichsen–Sauter与Generalized
  Taylor–Duffy把同面/共边/共点target×source作为一个积分域后，面积分割
  误差≤`8.88e-16`，共边/共点Cauchy约`1.4e-5`；level3的1280面球
  势/速度/Cp误差`0.008596/0.038973/0.013847`，原冻结阈值全部通过。
  因此成对算子N3.1j3b6d3d与附着球门d3e validated/frozen；父d3仍因
  环量、wake、Kutta和物质势历史缺失保持partial。
- 2026-07-28: **N3.1j3b6d S0实际边界势方程与物面压力拆分**。
  在实现前冻结单位球20/80/320面、source符号、内部Dirichlet方程、解析势/
  速度/Cp和阈值。S0实际边界常元source-doublet解的内部势、外侧身份和
  source flux最细残差分别`1.13e-15/1.24e-15/5.38e-18`，离体势误差
  `0.3494→0.1105→0.02932`单调收敛，条件数≤`2.343`，故仅限势方程的
  N3.1j3b6d1 validated/frozen。相同解若用逐面常doublet周界ring普通值直接
  形成物面速度，320面法向残差`0.03778>0.03`、Cp RMS
  `0.56681>0.08`，预登记pressure门失败；阈值未放宽，N3.1j3b6d2
  falsified/frozen。病因不是Green符号，而是缺连续势跳与片上有限部/一致
  表面梯度；新N3.1j3b6d3仅允许actual-boundary连续P2 Galerkin、分类物理
  暴露边和material-potential历史，仍禁生产压力与总力拟合。
- 2026-07-28: **N3.1j3b6c 通量通道/涡丝拓扑归因与方向改写**。
  预登记后把闭壳通量拆成固定物理通道：代表点 freestream
  `-3.47e-17`、bound direct `+0.0610303`、bound image `-0.0312306`、
  wake direct `-7.06e-5`，故生产总通量`+0.0297291`由bound表示主导，
  不是wall体积通量`-7.48e-6`或wake。158→1246面加密时direct/image
  仍为`O(1e-2–1e-1)`且不单调。对共置反向线段先按有向环量消去后，
  production仍有17条活动束缚涡丝穿壳：direct命中trailing_base/root/tip
  各`9/5/4`次，image在root-base角命中1次；wake命中0。特别是9条
  trailing-base穿越恰为`ns+1`条末排弦向涡丝，排除“只有半翼根盖伪影”。
  NASA TP-2995说明constant-doublet/ring等价只绑定同一面板周界，连续
  doublet才消去非物理边界线涡；Dusto–Epton把厚翼source/doublet置于
  实际物面。因此“原始N1环场+单值source条件化”N3.1j3b6c
  falsified/frozen，而其只读重放接口N3.1j3b6c1仍validated/frozen。
  新live方向N3.1j3b6d只把N1当环量不变量/初值候选，在实际移动边界上把
  source、连续doublet和TE/base wake与无穿透、Kelvin、质量、动量及Kutta
  联立；若规定N1环量冲突则触发freeze review，禁core/offset/source容差
  或总力拟合。生产路径未改。
- 2026-07-28: **N3.1j3b6c N1运行输入与实际厚度尾缘裁决**。
  预登记后实现只读 adapter：制造环场对真实 Warp fp64/fp32 核误差分别
  `5.80e-16/2.24e-7`；代表 `v41 U8/aoa15/f2.6/tw11.25` 的120帧
  wake重放误差`2.16e-7`、N1 collocation法向残差`1.85e-7`，81顶点/
  158面实际壳闭合且开启观测前后全部数值字段bitwise相同。因此
  N3.1j3b6c1 validated/frozen。实际壳source解无穿透`1.29e-16`，但
  source-flux相对残差`0.006205`；镜像wake物理候选还会改变壳面入射速度
  RMS `0.2792 m/s`，二者均未偷换进生产身份。NACA-2406标准开尾缘
  `h_TE/c=0.00126`（根部`0.36162 mm`）具有上下两角+实体base，而N1的
  单一wake来自薄涡格rear bound-ring line；该线在几何尾缘之后`0.25`个
  末面板长度，实测偏置`0.0366743c`。Kemp NASA TM-80104的B-1
  bisector、B-2双角、B-3非线性
  等压条件互不等价，且其数值净source added mass不属于真实分离wake；
  Xia–Mohseni JFM 830要求有限角非定常涡片方向/强度/速度与环量、质量、
  动量联立。故“薄面Kutta自动传递”N3.1j3b6c2 falsified/frozen；新
  N3.1j3b6c3仅允许预登记的pressure/direction/flux/wake-formation多残差
  与sharp-limit continuation，禁Kutta常数、wake角或base力拟合。因一致
  material potential history尚缺，厚体压力与生产晋升仍NO-GO。
- 2026-07-28: **N3.1j3b6 厚体均势场方向裁决**。资产审计确认
  `N2.6d1` 已冻结与 N1 中弧面共置的 NACA-2406 双侧气动壳，当前缺口是
  fluid source/mean-potential/no-penetration 而非结构几何。Morino、
  Dusto–Epton、Bristow、NASA TP-2995 和 NACA TM-1023 支持实际表面边界积分
  与势/速度级 source-doublet 组合。预登记的带环量圆柱解析门证明：若厚度与
  升力压力分别成力后相加，会漏掉 Bernoulli 交叉项并丢失全部
  Kutta–Joukowski 升力；三个 k 工况漏失 `0.6283/2.1991/4.3982`，
  统一压力误差≤`3.55e-15`。故 N3.1j3b6b falsified/frozen；当时唯一 live
  方向暂定为 N3.1j3b6c：冻结 N1 作为规定环量/入射场，source 解总无穿透残差，
  势率/速度先合并、Bernoulli 与成力各一次。厚体 Kutta/canonical/Cp/118
  门未过前只准 diagnostic shadow。随后预登记 G1a-e 全部通过：三角面核对
  独立 Duffy 积分误差`3.13e-15`；1280面球速度误差`0.252%`、Cp RMS
  `0.00612`；无穿透/源通量约`1e-15`；Galilean力误差`4.45e-16`、
  added-mass误差`3.74%`；NACA-2406半翼壳2894面闭合且N1中面变化0。
  因厚体Kutta、N1条件化运行输入、双侧Cp与118尚未执行，节点当时仅升
  partial；后续通量/涡丝拓扑证据已将该原始环场条件化路线降为
  falsified/frozen并转向N3.1j3b6d。生产公式未改。
- 2026-07-27: **D10c 首生环身份/尺度裁决**。Hirato Eq.7 只规定已有旧 LEV
  后的 1/3 切分；历史 `LE+U_inf*dt` 无首环文献身份，N3.1i1b1
  falsified/frozen。Ansari 半步当地边速度与 Ramesh LESP 首涡式拆为显式候选，
  后者经正交局部弦向—吸力法向基在 `nc8/ns24/spc120` 产生
  `ell_min/c=0.00477–0.00487`，历史整步为0.025。旧 `0.01c` 核下限由此变为
  `rc/ell_min=2.05–2.10`，违反 Hirato `<0.5` 范围；N3.1i1b3
  falsified/frozen。P-A 因完整当地边速度三维 adapter 缺失暂不具 live 资格；
  P-R 仅作为下一 implementation-ready 候选，需跑
  `rc/ell_min={0.10,0.25,0.49}` 的守恒/场形/分辨率门，未晋升。
- 2026-07-27: **D10d live-shadow 底层离散身份复审**。Hirato dissertation
  §4.5.3 的新 TEV offset 是 `0.3U_inf dt`，历史 `ug.shed_kernel` 为整步；
  Journal Eq.24 是当前/上一当地速度平均，不是现 shadow 的 current-only Euler。
  两个同义命题 N3.1i1c1/c2 均 falsified/frozen。Eq.25 的 `r_c` 结合
  dissertation 的 singular-radius 解释，预登记为每个 run 固定、随分辨率族变化，
  禁止逐环动态变核；N3.1i1c3 保持 open，live shadow 必须自有 TEV/LEV 账和
  每个 material vertex 的历史速度。
- 2026-07-27: **D10b 完整敏感性揭示半翼自由尾迹边界缺项**。仅镜像 bound AIC、
  未镜像 TEV 的 RHS/自由对流时，Case1 首次位置随细化漂到
  `0.146–0.203` 半展长；Case2 半展向仍稳。只读 `hirato_probe` 加完整镜像场后，
  `nc8/12` 两 Case 的时空门全部通过，Case1 根部误差收敛到0.016–0.021；
  但预登记12点仅11点通过，`nc4/spc240` Case1 `t*=1.600`、误差0.110，超门
  0.010。故不放宽容差、不删除粗格：N3.1i1a只冻结Eq.6公式身份，新建
  N3.1i1e partial；“中细网格为最低合格域”待下一轮预登记。生产N1未改。
- 2026-07-27: **D10a Eq.6 起涡门 GO，病因锁定为观测算子错配**。旧 canonical
  runner 实为平板；换官方 UIUC SD7003 中弧线仅使 Case1/2 起涡各推迟
  `0.025t*`，不是主因。Hirato Eq.6 要求最前环 `Gamma1` 与同一格实际
  `delta_x1` 配对并以 `U_inf` 归一化；旧分支把 `Gamma1` 与固定 `0.10c`
  分母错配。只读原式在 `nc4/ns16/spc120` 给 Case1 `t*=1.650`（目标1.710，
  root-first）、Case2 `t*=1.500`（目标1.575，y=0.531/0.594），均通过预登记
  粗格时间/展向门且未改 `LESPcrit=0.27`。新增 N3.1i1a validated/frozen；
  Eq.7/9/10/17/23-24 和空间压力仍未闭合，N3.1i1 不晋升。
- 2026-07-27: **D9 输入身份 GO；D10 空间载荷预登记**。Figure10 上止点后的
  正 nose-down twist 经代码旋转矩阵唯一映射为 `twist_phase_deg=-90`；左右翼镜像
  和 Figure14 风轴变换不改变该结论，N5.1c 转 validated。D8 的 phase-only
  NO-GO 仍有效，故生产默认/缓存/118未改。面向结构 co-design，有限
  `Gamma+centroid` 只准总力诊断；“有限矩唯一重构面压力”证伪，新增 N3.1i
  空间涡片→面板压力→`H^T`结构载荷路线，详见
  `research_n3_spatial_loads_20260727.md`。
- 2026-07-27: **D10 Hirato 方程审计：现有 H14 身份 NO-GO**。规范 Case 1 中
  `hirato+ansari` 在 `param_rings=0/wake_LEV=0` 时仍改变 CL，证明其效果只来自
  当步束缚重解；代码另缺 Eq.9 LEV–TEV Kelvin 联立、Eq.17 `dGamma_LEV/dt`，
  且伪涡环和规定卷起几何不符论文。新增 N3.1i0 falsified/frozen；下一候选必须是
  隔离的 `hirato_exact` shadow，详见
  `research_n3_hirato_equation_audit_20260727.md`。
- 2026-07-27: **D7 lift门 GO、D8 完整族 NO-GO**。显式 -90 下 f2.0 的
  L(tw0/15/22.5/45)=`6.228/6.485/6.924/6.778N`，恢复峰后下降但峰仍错在22.5；
  T=`+0.034/−0.054/−0.151/−0.576N`，没有实验的初段改善。N5.1 改为 partial：
  +90 同义命题与 phase-only 充分命题分别 falsified；N5.1c 开放做 Figure7/10
  装配角到代码旋转矩阵的坐标审计。生产默认和118基线均未改。
- 2026-07-27: **D6 NO-GO + N5.1 相位身份异常**。Figure16 匹配工况 f2.0 下，
  tw0/22.5/45 的 lift-dominant `|alpha_kin|=7.60/11.06/14.54°`、N3 mean
  `0.84/2.75/5.10N`、正峰 `19.33/22.63/25.77N`，均与论文“深扭转降有效攻角
  且压低双峰”相反。原页 Figure10 视觉核对显示上止点后 flap下降、twist由0向正；
  代码 +90 在同一相位由0向负。因 +90 是交接铁律，只登记 N5.1 open，并预登记
  `-90°` 显式隔离诊断，禁止静默改默认。
- 2026-07-27: **D5 NO-GO + 位置—冲量文献裁决**。signed `I_CV`
  `+680.89/+873.52/+1042.82` 仍单调；signed `I_CNv`
  `−63.75/−245.46/−342.56` 不适合作正增升，N3.1g 证伪。Wang–Eldredge、
  Fernandez-Feria、Siala–Liburdy、Li–Wu 和 Chowdhury–Ringuette 共同支持
  “Γ之外还需位置/运动或角冲量状态”，但现成点涡路线触及 N3.1f dead-end，
  三维旋翼总力式又会与 N1 双计。D6 转查 Meng Figure16 的半周期指纹。
- 2026-07-27: **D4 双 NO-GO**。生产网格 tw0/22.5/45 的 `I_|CV|`
  `1799.69/2944.04/4114.94` 仍单调；`tau_v>4.24` 的生产 N3 占比
  `10.58%/11.25%/11.93%`，tw45 仅为 tw22.5 的1.06×且未达30%。
  N3.1e 无符号分离幅值替换和事件寿命主因均证伪，不接入力。D5 预登记转查
  L-B 自身 signed `CV/CNv` 是否包含翼面—展向抵消；这不复活 signed-A0 候选 G。
- 2026-07-27: **research-pipeline 扩展景观重置路线**。Ramesh、Narsipur、
  Deparday、Sheng、Onoue、Li、Menon、Joshi 等一手来源共同否定“只缺一个门”
  的窄诊断：LESP 可判起涡/供给，持续载荷还取决于分离/环量、事件年龄、涡位置和
  三维拓扑。N3.1.0 改判 falsified；`alpha_kin` 简单替代也由 B4 证伪。
  首个幸存方向 N3.1e 只允许先做 D4 只读对账，禁止调整 `cds/Tv/crit`。
- 2026-07-27: **B2 event-surface-memory NO-GO**。生产网格事件序参量
  tw15/22.5/45=`+0.447/+0.379/+0.266`，虽向另一翼面单调移动但未跨零；
  且与瞬时符号序参量逐点相同，事件记忆在当前 A0 超临界区间内惰性。未接入力，
  禁止复活 signed 候选。病因继续收窄为 N3.1 的 `|A0|-crit` 是否是深扭转下错误代理。
- 2026-07-27: **B1 formation-only NO-GO**。预登记后以纯诊断时钟跑生产网格：
  tw0/22.5/45 的 `T_hat≥4` drive 加权占比为 3.28%/6.71%/7.79%；
  tw45 仅比 tw22.5 高1.08个百分点、1.16×，未达 `+0.10` 且 `2×` 激活门。
  N3.1a 不接入力、不降低 T*；形成数保留为物理边界，但证伪为 D3 主修复路径。
- 2026-07-27: **D3 精确通道归因 + Phase B 文献裁决**。生产网格
  `U8/aoa5/f2.6/tw=0,22.5,45` 与冻结缓存误差≤0.025N；tw0→45 时 N1
  `−0.930N`、N2 `−0.869N`，唯 N3 `+4.589N` 把总趋势翻转。N5 改判为 N3.1
  派生观测失败。Gharib/Onoue/Manar 支持有限形成窗，Meng 原文支持上/下翼面身份迁移；
  N3.1a/B1 已在 `research_n3_twist_gate.md` 预登记，禁止修改 cds/Tv/crit。
- 2026-07-27: **D1 数据身份纠偏**。源论文 PDF 第17页 Fig18(a/c)视觉核对确认，
  数字化资产将推力 U6/U10 曲线反标；`correct_fig18_curve_identity.py` 已交换
  四对记录的测量 `x/exp`。实测 `dT/dU` 从伪 `+0.508` 更正为约 `−0.508`
  N/(m/s)，V4.1 为 `−0.276`，趋势同号、量级偏弱。撤销 D1→N2.2 因果边；
  证据见 `d1_fig18_curve_identity_audit.md`。纠正后六族 `dT/dU` Pearson=0.970。
- 2026-07-27: **运行时计算图 v1**。`claim_nodes/*.yaml` 增加实现绑定、数据
  `requires/provides`、closure profile、互斥关系、运行角色和 frozen 源码指纹；
  `claim_runtime/` 执行拓扑校验、状态/角色检查与唯一 `ForceLedger`。每次
  `gpu_run_twist()` 返回 `claim_manifest/claim_contributions/claim_guards`，力账不闭合
  则拒绝返回。v41 拓扑 `N1→N2→N3→N4→N5→N6`，v4_legacy 使用隔离的
  `LEGACY` 兼容节点。validated UVLM Warp 内核及气动公式未改。
- 2026-07-27: **E1/E2 CLOSED（调用身份审计）**。从原始 Claude session transcript
  恢复 07-25 `lb_sweep118_F.py`：候选 F 真身为 `nc=12,ns=16,n_cycle=4`,
  `spc=spc_of(U,f)`, `CFG_PRESETS["H16"]` 加显式覆盖，以及
  `lb_hybrid=0,lb_lesp_crit=0.18,lb_cla3d=True`。三点端到端复现门的最大
  `|Δ(L,T)|=0.100 N`（容差 0.15 N）。裁决：缓存有效；`hybrid=0` 假设确认；
  无需用无锚 `lb_lesp_crit=0.23`；E1/E2 是预设与后补粗网格调用字典失真，
  不是新结构病灶。可执行证据：`platform/verify_v41_repro.py`。
- 2026-07-26: 树建立(v4.2 战役 Phase −1)。D1/D2/D3 入树;方向预判待各 Phase ② 文献确认。
- 2026-07-27: Phase −1.3 晋升提交(de715bf,closure= 预设)。晋升验证发现 **E1/E2**:
  closure 预设不 reproduce 07-25 缓存(aoa15 +25.1 vs +15.6;aoa0 +1.7 vs +4.8;验证区
  aoa5/U 全族近缓存)。LB_DIAG2 归因:hybrid 满额回补(N2.4 嫌疑)+ ds 幅值驱动(N3.1)。
  已排除:网格/spc/sym/swept_axis/a0_mode/a3d/cla3d/cds=0/f2gate/signed/运动学幅值全组合。
  主嫌 = 07-25 调用含 lb_lesp_crit=0.23(未记录)。**118 重扫阻塞于 E1 裁决**。
- 2026-07-27: **代码本体重构完成**(521c523,用户裁定"仿真结构必须重构为 claim chain 树形式"):
  `claim_nodes/*.yaml` DevReady 资产 + `claim_dag.py` DAG 本体。修改规则:
  validated 节点(N1/N4)freeze 禁改;已证伪节点(N5/N6)禁重走;partial/open 节点可动但需证据+归因。
- 2026-07-28: **N3.1j3b6d↔N3.1j4 S3a三维body-cut—material-wake
  trace接口门**。在实现前冻结闭合full-wing diamond翼、四段分类TE cut、
  `mu(y)=1-y²`、P2 material TEV band、规范与N1/DDE符号硬门。执行后
  20顶点/36面物理几何变化为0，boundary/nonmanifold/orientation mismatch
  全为0；连续body P2的74 DOF只复制3个内部TE顶点和4个TE边中点，所有
  非cut边trace mismatch为0。body jump、wake current-edge attachment、
  gauge变化、N1/DDE round-trip及wake内部trace jump均为0；翼尖jump为0，
  对称导数误差`3.18e-15`。故“watertight几何迫使势trace单值”
  N3.1j3b6d14 falsified/frozen；共享几何＋分类势复制＋同一material
  jump接口N3.1j3b6d15只在topology/trace范围validated/frozen。下一门只准
  联立body无穿透、gauge、Kelvin与wake jump；有限base、压力、力和生产仍
  NO-GO。
- 2026-07-28: **N3.1j3b6d16 S3b三维actual-boundary—steady wake
  联立门NO-GO**。单长带逐pair组装得到81×81满秩、无独立wake幅值、弱残差
  `4.08e-16`和严格attachment，但zero-alpha假jump`5.76e-5`、反对称
  `1.15e-4`、mirror`5.44e-5`均失败；q4→q5 root Cauchy`0.269`，
  x4→x8 far-wake Cauchy`0.315`。高阶只读指纹在固定x8时趋向稳定
  (`q12 root=-0.12651`)，但固定q10把同一band拉到x16/x32反而漂到
  `-0.11972/-0.08157`，锁定“低阶共边积分＋单带失去shape regularity”
  而非代数欠定。N3.1j3b6d16 falsified/frozen；新N3.1j3b6d17只允许
  shape-regular chronological P2多带＋显式interface＋首带共边联立，
  仍禁pressure/force/production。
- 2026-07-28: **N3.1j3b6d17 S3c shape-regular多带wake联立门GO**。
  以`0.5c` chronological bands替代单长带后，x12的22带最大长宽比为1，
  history time/geometry/trace gap全0，body共边奇异pair始终仅最新带8对。
  未知量仍为81个body势DOF、wake独立幅值0，rank deficiency 0、弱残差
  `3.83e-16`。zero-alpha/反对称/mirror归一误差随q6→8→10单调下降；
  q8→q10 root Cauchy`9.23e-5`，x8→x12 far-wake Cauchy
  `1.153e-3`，相对S3b单带0.315降低约273倍。故N3.1j3b6d17仅在steady
  fixed-geometry、重复material trace、equation/refinement范围
  validated/frozen；新N3.1j3b6d18开放验证非恒定body jump的material
  Kelvin历史，仍禁势率、pressure、force与production。
- 2026-07-28: **N3.1j3b6d18a S3d old-known/new-current Kelvin账GO**。
  一个old band和一个active band的预登记affine门中，old rows mutation、
  old→active interface、active→body attachment和tip jump均为0；旧wake
  未知量0但known RHS范数`5.77e-4`。反号history使当前jump变化`0.11032`，
  incident/history仿射叠加误差`4.65e-16`，rank deficiency 0、弱残差
  `9.80e-16`。故N3.1j3b6d18a只在旧材料态冻结、最新current row联立范围
  validated/frozen，父d18升partial；d18b继续开放显式midpoint和
  `dt/dt2/dt4`时间门，通过前禁止`dmu/dt` pressure、force与production。
- 2026-07-28: **N3.1j3b6d18b S3e fixed-body显式midpoint时间门GO**。
  三个预登记时间步族共28个half/full actual-boundary—wake系统中，old
  material strength、规定几何对流、history时空/trace接口、midpoint identity、
  current attachment和tip误差均为0；rank deficiency 0，最大弱残差
  `3.84e-16`。body-jump Cauchy收缩`5.665`、最细相对变化`1.530%`；
  离面wake场收缩`1.767`、最细相对变化`4.616%`。`mu_mid`与端点平均最大差
  `0.01740`。故d18b只在fixed body＋prescribed uniform convection范围
  validated/frozen，不宣称统一二阶；新d18c开放moving/deforming geometry、
  wake relaxation与步内surface/newborn-wake equilibrium，仍禁pressure、
  LEV、force和production。
- 2026-07-28: **N3.1j3b6d18c1 S3f 无有向attachment的显式三维wake接口
  NO-GO**。straight legacy equivalence与geometry/strength mutation均为0，
  曲尾迹使cut jump改变`1.516e-3`，rank/interface/topology均通过；但通用刚体
  旋转令世界坐标cut排序反转，matrix/RHS/jump客观误差扩大到
  `1.889e-3/3.422e-4/3.842e-2`。保持cut顺序的x轴旋转以及恢复原material
  顺序的反事实均把误差降回`O(1e-15)`，故底层积分客观，缺件是ordered body
  material IDs、P2 permutation与`wake_mu=s_attach*body_jump`有向身份。
  d18c1以“无orientation仍充分”falsified/frozen；新d18c1a开放，仍禁
  relaxation、pressure、force和production。
- 2026-07-28: **N3.1j3b6d18c1a S3g 有向material attachment门GO**。
  forward typed attachment与legacy matrix/RHS/jump差全0；generic rigid
  transform虽反转coordinate cut order，有向ID/P2 permutation后误差仅
  `5.55e-17/1.78e-17/6.63e-15`。反向span参数化配合`sign=-1`的
  matrix/RHS/jump gauge差为`1.12e-14/4.88e-15/1.02e-12`；attachment、
  interface、tip误差0，三类非法identity全部fail closed。故c1a仅在
  ordered material IDs＋P2 permutation＋signed jump范围validated/frozen；
  S3f反例保留。c2现可启动wake Heun后的步内equilibrium，仍禁pressure、
  force和production。
- 2026-07-28: **N3.1j3b6d18c2a S3h body-attached material-wake
  Heun语义门GO**。在预登记的制造仿射速度和给定body-cut轨迹下，`2/4/8`
  步族free-vertex Cauchy收缩`3.995`、最细相对变化`1.61e-5`；newest-edge
  attachment、history seam、duplicate-seam velocity与material-strength
  mutation均为0。通用刚体变换误差`8.88e-16`，最小face-area ratio
  `0.958`，人为seam-velocity不一致按预登记fail closed。故c2a仅在
  “制造速度＋prescribed body cut＋无压力/无力”的积分语义范围
  validated/frozen；c2b继续开放实际body+wake诱导速度与relaxation后
  surface/newborn-wake equilibrium，仍禁pressure、force和production。
- 2026-07-28: **N3.1j3b6d18c2b1/b2 S3i 切向规范—材料势跳输运门
  GO**。代码指纹表明c5/c6只验证法向几何速度，而现
  `MaterialWakeBand.material_update`对任意几何移动冻结P2 rows。预登记的
  二维切向制造流中，三种mesh gauge的ALE residual最大`1.67e-16`，材料
  轨迹/跨规范标量误差均`4.44e-16`，通用刚体误差`3.04e-16`；但
  normal-only+frozen-mu仍产生`0.0567519`场误差，边界速度仅`3.67e-17`。
  故c2b1 falsified/frozen；c2b2只在
  `d_mu/dt|mesh+(u_bar,t-w_gauge,t)·grad_s(mu)=0`连续身份范围
  validated/frozen。离散P2 transport与实际诱导速度仍open，pressure、
  force和production仍禁。
- 2026-07-28: **N3.1j3b6d18c2b2a S3j continuous-P2 ALE空间门GO**。
  在stationary planar规范中，`3/6/12` cell families的relative L2 error
  为`3.43e-4/5.11e-5/9.56e-6`，收缩比`6.71/5.35`；mass rank
  deficiency 0、constant-rate residual `1.79e-14`、共边jump 0。
  通用刚体matrix/终态误差`2.26e-17/2.22e-15`，q10/q12终态差
  `4.44e-15`。故c2b2a仅在consistent mass、无稳定化的平面半离散空间
  算子范围validated/frozen；moving/curved/history-seam composition及
  actual induced velocity仍open。
- 2026-07-28: **N3.1j3b6d18c2b2b S3k moving-curved multi-patch门
  NO-GO**。patch/monolithic matrix与终态差0、shared trace jump 0、
  Heun time Cauchy`4.428`、finest relative L2`0.2265%`、刚体
  geometry/scalar误差`2.48e-16/3.47e-17`及area门均通过；但显式zero
  boundary从0漂到`3.015e-4`，失败冻结`2e-12`门。病因是unconstrained
  CG trial space把物理boundary DOF当自由输运量，不是曲率/时间阶。
  故“无essential trace仍充分”falsified/frozen；新c2b2b1只准以显式
  boundary roles分块stage solve，禁止事后清零、距离猜测、扩散或放宽门。
- 2026-07-28: **N3.1j3b6d18c2b2b1 S3l essential P2 trace门
  NO-GO / parent partial**。free/constrained block的partition、rank、
  zero/nonzero trace、patch trace、Heun时间收缩`4.398/3.812`、最细
  L2误差`0.2165%/0.06740%`、刚体`2.98e-16`及非法输入门均通过。
  首跑刚体2.0追踪为制造label在逆旋转后越过`tan(pi*r/2)`端点分支，
  已保留invalid-oracle原始结果；只修测量器后不影响NO-GO。唯一有效失败是
  clamp-only终态free差`3.499e-5 < 1e-4`。病因是预登记
  `c_s=a sin(pi s)`在body edge严格为0，名义attachment不是inflow，实验
  无法唯一识别boundary block。禁止降阈值；c2b2b1保持partial，新
  c2b2b1a只允许预登记非零typed body inflow、free old-edge outflow与
  characteristic tips，通过前仍禁actual induced velocity、pressure和force。
- 2026-07-28: **N3.1j3b6d18c2b2b1a S3m typed inflow/outflow门
  GO**。`ds/dt=-1`使body edge成为真实inflow，old edge为outflow，tips
  characteristic；17/17/14个P2 DOF角色完整且只有body受约束。correct
  block归一残差`6.66e-16`、rate injection free响应`0.0724`，而
  clamp-only rate残差`1.0`、终态free误差`1.052e-3`。传播时间收缩
  `4.048`、最细L2误差`2.7765%`，trace/patch/rigid(`3.12e-17`)及三类
  非法角色全部通过。故c2b2b1a validated/frozen；父c2b2b1同步改写为
  role-aware inflow/outflow/characteristic命题并validated/frozen。只授权
  c2b3实际速度组成与relaxation残差预登记，仍禁pressure、force和production。
- 2026-07-28: **N3 S3ah 在正式数值执行前 CIRCULAR-RANK-NO-GO**。
  原计划用
  `G_A=L_M^T(I-C Phi_g)J_P^-1 M_a Q` 的 rank 判断零态或七维
  forming state。独立公式审计证明
  `H=I+C B^-1 W` 正是已冻结满秩 Morino block 对 `B` 的 Schur 补；
  在 `J_P/M_a/Q/L_M` 同样可逆的预登记前提下，
  `rank(G_A)=rank(H)=7` 与 pressure closure、dt、history basis 无关。
  故 rank7 是前提的代数结果，不是缺态证据；正式 guard 未执行、无结果
  JSON、无 production/force 修改。该判据节点
  `N3.1j3b6d18c2b3b3b2c2b2b3e3a` falsified/frozen。
- 2026-07-28: **下一可动节点改为 reachable-manifold dynamic
  obstruction**。对完整兼容系统 `K=(R_B,A)=0` 和未强加动力残差 `D`，
  只允许由前一兼容解、Kelvin/material transfer及actual运动学产生
  `delta u=E_R delta eta`，真正判据为
  `Omega_D=(D_u-D_x K_x^-1 K_u)E_R`。任意 previous-trace P2 注入被
  禁止。Xia–Mohseni只作为无质量forming-geometry前向候选；
  DeVoria–Mohseni 2019 的三维VES只有在质量、动量、entrainment库存独立
  解释剩余 obstruction 后才可进入，禁止自由zeta/residual slack。
- 2026-07-28: **S3ai-v1 在正式执行前 PROTOCOL-AUDIT-NO-GO**。
  时间戳冻结预登记保持原样，未运行正式 guard、未生成结果 JSON。独立审计
  发现 direct-BIE 重解自证、Kelvin 望远镜恒等、body/direct-W 求积阶混用、
  material attachment trace 未独立核对、零解严格单调误判及 uncertainty
  量纲混账。开放节点
  `N3.1j3b6d18c2b3b3b2c2b2b3e3b` 状态不变；下一门必须使用存储 stage
  potential、独立 material circulation/反号负对照、共同 finest anchor 与
  同 residual-space Richardson error ball。该裁决不授权状态、力或生产。
- 2026-07-28: **S3ai-v2 定义冻结，尚未正式执行**。D16撤回“当前S3e
  已有独立global Kelvin equation”的过强角色：open cut与prescribed
  convection没有具名随流闭合曲线；只允许从`surface.face_mu`独立恢复
  previous/current完整P2边界，形成attachment/orientation inventory，并
  用错误birth sign负对照。动态主观察固定为stored
  `global_body_potential`上的
  `R=M(g_cur-g_prev)+dt*P_mid`；direct-W重解只作cross-observer，不回写
  material history。冻结共同anchor
  `(epsilon,dt,q)=(0.0025,0.0625,12)`、epsilon/dt Richardson、q tail、
  完整相邻2x2x2 mixed cube、独立`Z0/V0`误差球与31条history定义。此门
  没有空间mesh h/p细化，任何结果仅是fixed-space named-law witness；
  空间复现前禁止在massless forming geometry与finite VES之间裁决。
- 2026-07-28: **S3ai-v2.1 在任何正式history前补冻**。独立定义审计发现
  v2的q plateau与总式会双计`u_round`，且negative-control幅值、mirror
  map、condition norm与实际S3e计数不唯一。v2时间戳资产保持不改；
  v2.1固定每个axis allowance不含floor、总`U`只计一次round，active
  space严格7维/`M`为bit-exact symmetric SPD，stored-phi污染为指定DOF
  加`2^-10`，并冻结wrong sign/permutation/row-surface/local-cancellation
  数值负对照。现有API每条history各跑一个pre-step，故正式计数为31条、
  380 measured steps、411 total S3e steps、822 half/full solves及791个
  observed stages。仍未执行物理门、未生成result JSON。
- 2026-07-28: **S3ai-v2.2 在正式history前替换错误的zero/mirror
  observation**。`alpha=0` entrance prestep 的complete cut trace在
  q8/10/12下由`2.060e-6 → 6.298e-7 → 2.048e-7`收缩，几乎全为
  span-odd；correct-sign `face_mu` inventory仍精确为0。旧
  `max|v-Jv|/max|v|`却趋近2，证明它在continuum-zero极限是误差除以误差，
  不能作硬门。D17证明物理trace/pressure/residual为even，trace必须用
  primal `M` norm、weak pressure/residual用dual `M^-1` norm；v2.2冻结
  `Pi±=(I±J)/2`的同空间q/error intervals、stagewise odd
  noncancellation和exact even/odd负对照。一个`[-dt,0]` prestep保留但
  排除于measurement window；禁止无机理长prehistory/burn-in。`L_->0`
  才是resolved symmetry failure，`L_-=0`只称no resolved violation。
  当前仍未执行正式门、未产生pressure-law/force/production/Fig结果。
- 2026-07-29: **R0 图级周期归约契约晋升 validated/frozen**。
  条件性184运行在`6_2.6_27.5_5`暴露`0.1890927018 N`旧
  `uvlm_remainder`。代数和三调用复现证明它严格等于
  `R(total)-M(total)`，不是N1或其他物理通道漏项。第一次未经控制验证的
  跨进程bitwise门按原文NO-GO并保留；第二次预登记同调用不变量门中，十个
  既有气动力字段在ClaimGraph前后逐bit不变，两个anchor相对冻结基准最大
  偏差`0.006172 N < 0.15 N`，probe R0与旧差值误差
  `1.08e-15 N`，物理未分类力为0。R0现为v41图末端diagnostic节点，
  N1指纹不变；co-design只消费physics/necessary_physics角色，禁止把R0
  当作空间面板载荷。该晋升不计作气动精度改进。
  子节点`R0.1`明确冻结“状态/力双源＋跨进程历史bitwise充分”的失败路线；
  `R0.2`仅晋升唯一canonical源与同调用十字段不变量，避免把第一次NO-GO
  事后改判或重走。
- 2026-08-01: **双 scope 契约终裁（fig19_cd 频率身份，Phase 1.1）**。
  `fig19_cd_frequency_identity_exhaustion_20260729.md` 一手资产穷尽(PDF/JATS/EPUB/TIFF/
  supplement 探测全负)= UNRESOLVED 终态;终裁采用双 scope 契约:
  `confirmed`(42 曲线/151 条件)= **晋升域**;`conditional_fig19_cd`(8 曲线/48 条件)=
  **诊断域**(禁晋升/证伪 claim,H_shared_2p0 只作 plausible inference 不写 ground truth)。
  `fig171819_benchmark.py` `FIG19_CD_FREQUENCY_STATUS='conditional_scope'`,
  `promotion_eligible` 只看 confirmed 域 → **promotion_eligible=True、blockers 空**。
  同步更新 frozen 哈希链(benchmark→confirmed_compare→claim_attribution 三层),
  `test_fig171819_claim_attribution.py` 47/47 绿。M0 节点新增终裁证据。
  数值验证:当前工作树 v41 在 lb_sweep184 BASE(H16+kelvin+a0_crit=0.27+visc=False)
  下三点复现缓存最大差 0.100N < 0.15N 门;184(105013) 与旧缓存 118 点零差。
