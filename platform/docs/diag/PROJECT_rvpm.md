# 立项:rVPM 升级——解析涡动力学替代 LEV 闭合模型(2026-07-17,用户拍板)

## 动机(为什么必须做)

- **L2 案终判**:LEV 周期平均升力(dL/df 缺 +1.5~1.9 N/Hz)机理已证,但闭合模型
  路线撞零拟合墙(可实现配分系数文献不存在)——**DSV 升力必须从解析动力学中涌现**;
- 历史同源:H15"DSV lift did not emerge"(粗离散环)、vimp 反相、vnf 过供——
  所有闭合尝试的失败共同指向分辨率/动力学保真度;
- co-design 泛化:闭合常数(a0_crit 有效值等)不可跨构型外推;解析涡动力学是
  换翼型/换布局仍可信的根本投资。

## 资产盘点(2026-07-17)

- `src/fluxvortex/warp_vpm.py`:粒子 Biot-Savart + Jacobian GPU 核(高斯-erf 正则,
  fp64,已验证)✓;
- `_v2_robo.py` part_lev 路径:LEV 粒子剪切(LESP 门控,网格无关强度)+ 互诱导
  平流卷起 ✓;**缺:涡拉伸、reformulated 输运、SFS、发散松弛、粒子场力评估**;
- 环格 TEV 尾迹、LESP 门(保留为馈送判据)、fp32/chunk 性能模式、Warp autodiff
  经验(四坑清单在案);分支名 aero-rvpm-lev 即原定路线。

## 分级里程碑(架构细节待 research_rvpm_arch.md 回填)

- **S1 粒子输运升级**:涡拉伸 + reformulated VPM 方程(Alvarez-Ning f/g 选型)+
  核扩散;门:单涡环自诱导速度 vs 解析、双环蛙跳(经典验证);
- **S2 混合架构**:束缚 UVLM 保留,TEV/LEV 统一转粒子(转换时机/规则按文献);
  门:静态翼极曲线回归(v4 基线逐位带内)+ 巡航点不回退;
- **S3 LE 馈送**:LESP 门控馈送进粒子场(保留已验证判据);门:2D 俯仰平板
  canonical DSV 案(Eldredge/Baik 基准 CL 环,定量目标待文献回填);
- **S4 力评估**:束缚面压 + 粒子诱导(避开无界冲量陷阱,按文献实践);门:S3 案
  的力相位 + 周期均值;
- **S5 翼级验证**:118 工况重跑——**主门:dL/df 涌现(aoa10 目标 +1.5±0.4 N/Hz,
  L2 案指纹)**,推力 f² 律与巡航锚不回退;
- **S6 性能与可微**:N² 上限评估(FMM 决策)、fp32 化、autodiff 四坑逐条过。

## 纪律

零拟合(常数=文献);每级门禁带 canonical 案定量目标;失败如实入档;
电池参照同口径铁律;网格收敛状态标注。

## 架构裁定回填(2026-07-18,research_rvpm_arch.md,28 CONFIRMED)

- **历史悬案结案**:"DSV 未涌现"= 公式缺陷非分辨率——经典 VPM dσ/dt=0 违反
  角动量+质量守恒(固有不稳)+ 无 impulse-matching 喂涡 + 零散度控制,三病因
  各有零拟合修复;
- **架构**:束缚 UVLM+LESP 门原样保留;粒子只进 RHS 不进 AIC(DUST 模式);
  TE 经 ≥1 排隐式环缓冲后按弧长比例转粒子(永不在 TE 面对裸粒子);
- **LEV 喂入**:每步一粒、1/3 放置、2×2 闭式(A0 钉 crit+Kelvin)定强、
  σ=1.3×脱落间距、环量冻结;
- **输运**:rVPM f=0, g=1/5(σ 演化 + transposed 拉伸)+ 动态 C_d SFS
  (Re1.5e5 必开)+ corrected-Pedrizzetti 散度松弛;f=g=0 一键消融证涌现归因;
- **力评估**:束缚格 KJ + 粒子诱导入 V;Noca 通量式仅诊断;
- **数值**:LSRK3 + 直算 N²(1e4-1e5 粒子在发表包络内),FMM 延后;
  autodiff 三对撞点已配光滑替身方案;
- **验证梯定量目标**:S1 涡环自诱导/leapfrog(+cVPM 爆断消融);S3 2D ramp
  Case A/B/C + SD7003 CL 环(峰 2.3-2.4 @α≈21°);S5 dL/df +1.5-1.9 N/Hz 涌现。

## 状态

- **S1 过门(2026-07-18)**:warp_vpm_rvpm.py(rVPM 离散输运 f=0,g=1/5 + LSRK3 +
  corrected-Pedrizzetti + 消融开关)。G1 涡环:约定无关斜率门 **+0.5%**(严格),
  Saffman 值(a=√2σ 高斯核约定)−3.7~−5.8%(薄环渐近式 O(a/R) 自身退化,注记);
  G2 leapfrog t=40 真实穿越(R∈[0.75,1.23]),消融分化:cVPM |α| 增长 1.21 vs
  rVPM 1.08(3Z 泄流方向正确;硬爆断留待 S2 持续应变场)。
- S2 设计中:2D 条带测试台(平板束缚涡 + LESP 门 + 粒子喂入按裁定规则:每步一粒/
  1/3 放置/2×2 闭式/σ=1.3×间距/环量冻结 + KJ 力)→ Eldredge ramp Case A/B/C 门
  (CL 峰目标已录)+ SD7003 plunge 门(CL 峰 2.3-2.4 @α≈21°)。
