# 文献研究综合报告:rVPM 混合架构升级——让 DSV 升力从解析涡动力学中涌现

**目标**:为 RoboEagle 扑翼膜翼数字孪生(Re≈1.5e5, k=0.13–0.39, 拍幅 ±22.5°)裁定一条从"UVLM 束缚格 + LESP 闭合"到"reformulated-VPM 解析 LEV/尾迹"的分段升级路线。动机是两个未决物理案:(L2) 实测升力随频率增长(dL/df = +1.5–1.9 N/Hz,α 门控、U 超线性)= LEV 周期均值升力,闭合式修正全部失败 → DSV 升力必须**涌现**;(历史案) 粗糙离散环"DSV 升力未涌现"、material-impulse 账本反相。本轮 30 条主张逐条对抗核实,28 条 CONFIRMED、1 条 PLAUSIBLE、2 条 REFUTED(入坟场)。

---

## 1. 总裁定(先说结论)

**历史"DSV 未涌现"案结案:这不是分辨率问题,是公式缺陷。** 经典 VPM 固定核尺寸 (dσ/dt=0) 在拉伸场中同时违反角动量与质量守恒,涡量非物理增长且方法自身耗散压不住 → 数值爆炸/失真(rvpm-01,Alvarez & Ning AIAA J 62(2):637-656, 2024,机理在论文中为 hedged 假设但守恒论证完整)。我们的粗环无拉伸、无 σ 演化、无 SFS、无散度控制、且喂入合并涡时无 impulse-matching(A3-C3 确诊反相账本的第二病因)——四项缺陷每一项都有已发表的、常数零拟合的修复。

**升级路线不是"加经典拉伸"(已知不稳,A5-C4),而是 rVPM 全家桶**:f=0, g=1/5 的守恒推导输运方程 + 动态系数 SFS + Pedrizzetti 松弛,全部常数文献锚定,且 FLOWVPM.jl 提供 f=g=0 的一键退化开关做"涌现归因于公式"的零成本消融(rvpm-01/02)。目标架构(UVLM 束缚格 + LESP 门 + 粒子 LEV 尾迹)本身已有发表先例 VoFFLE(A3-C4),其失败模式(LEV 偏弱、脱落后爆炸)恰是 rVPM 缺失所致——正是我们的差异化空间。

---

## 2. 分角度题录表

### 2.1 rVPM 公式 (rvpm-01…08,全 CONFIRMED)

| ID | 主张一句话 | 对架构的意义 |
|---|---|---|
| rvpm-01 | 经典 VPM dσ/dt=0 违反角动量+质量守恒→固有不稳;FLOWVPM 有 f=g=0 退化开关 | **结案历史失败**;σ 必须成为粒子状态;免费消融开关 |
| rvpm-02 | rVPM 输运方程:f=0, g=1/5(守恒定值),Γ∥拉伸的 3/5 移入核收缩,dσ/dt=−(1/5)(σ/|Γ|)[(Γ·∇)ū]·Γ̂;transposed scheme 为默认 | **核心改动**:一次 Warp kernel 编辑 + 一个新状态数组(S 用现有 Jacobian 缩并) |
| rvpm-03 | SFS 是各向异性结构模型 E_str = Σ_q ζ_σq(x−x_q)(Γ_q·∇)(ū(x)−ū(x_q)),与速度/Jacobian 同一 N-body pass | Re 1.5e5 下前缘剪切层 enstrophy 级串的物理保障;融合进现有 kernel ~1.3–1.5× 成本 |
| rvpm-04 | C_d = ⟨Γ·L⟩/⟨Γ·m⟩ 动态算出(零拟合);α=0.667, rlxf=0.005, C∈[0,1], backscatter 裁剪 | 无需对 118 工况调参;裁剪/夹持 = 已知 autodiff 断伴随模式,需 soft-clip 替身 |
| rvpm-05 | Pedrizzetti 松弛控散度;corrected 变体保 |Γ|(修复强度衰减缺陷);默认 rlxf=0.3 每步 | 我们目前**无**散度控制——DSV 生长-脱落周期中的散度漂移是反相账本候选共犯;选 corrected 变体防环量泄漏 |
| rvpm-06 | 三种黏性方案;CSM dσ²/dt=2ν + β=1.5 RBF 重置;与 rVPM σ 收缩是独立叠加过程 | Re 1.5e5 下分子扩散次要:生产跑 Inviscid+SFS,CSM 只用于涡环黏性验证门;RBF 重置排除出可微路径 |
| rvpm-07 | 我们已验证的 Gaussian-erf 正则化 = FLOWVPM 默认核,bit 级同族 | **诱导侧零迁移风险**;fp32 尾迹快路径继续有效(改的是 ODE 右端不是 pairwise 结构) |
| rvpm-08 | 发表验证梯:孤立环→leapfrog 环→湍射流→旋翼悬停(~100× mesh-LES / 10× uRANS / 1000× DES)+ 螺旋桨-机翼(J. Aircraft 10.2514/1.C037279) | 我们分段门的模板;**注意**:发表案例全是自由涡/尾迹动力学,无壁面分离 DSV——pitching-plate 门是我们对梯子的未发表延伸 |

### 2.2 混合架构 (H2–H8;H8 为 PLAUSIBLE,余 CONFIRMED)

| ID | 主张一句话 | 对架构的意义 |
|---|---|---|
| H2 | rVPM 常数全守恒推导(质量-管 + 角动量-球),零拟合合规;束缚涡经 actuator 模型浸没、沿指定 TE 脱落 | 可微 endgame 的方程是 (x,Γ,σ) 的光滑 ODE |
| H3 | FLOWUnsteady ASM:束缚涡以压力型弦向分布浸没成粒子涡片,最小化过翼型中线的质量通量;σ_surf=b/100, overlap=2.125, σ_tbv=t/c·c/100 | Stage-2+ 当解析 LEV 粒子贴膜面掠过时,束缚格必须对粒子诱导表现为闭合固面,否则 DSV 吸力穿透中线 |
| H4 | DUST:Kutta 条件在线性系统内定第一排隐式尾迹板;粒子只进 RHS 不进 AIC;静态矩阵预分解;Rosenhead-Moore 核 + Cartesian FMM 阶数 2 足够 | **裁定束缚解耦设计**:AIC 纯格-格、可预分解、autodiff 简单;TE 保一排隐式环做 Kutta 缓冲,粒子永不贴 TE 奇点 |
| H5 | DUST 压力:旋转流中放弃平 Bernoulli,解 Bernoulli 多项式 B 的 BIE,源项 div(ω×u),复用板矩阵(去 Kutta 行);KJ 载荷仅限 vortex-lattice 元 | 膜面分布 Δp(FSI 需要)的原理性路径:LEV 粒子经 ∇G·(ω×u) 体积源进入表面压力 |
| H6 | Fu & Laurendeau 2025:TE 尾迹先直线环排、若干步后转粒子,上游线冻结防 TE 环量间断;粒子数正比流向弧长;保近三阶时间收敛,粗分辨率下均匀转换 8–10 圈发散、自适应 20 圈稳定 | **裁定转换时机**:≥1 排环缓冲后转换,永不在 TE;弧长比例粒子数直接可移植到扑翼(根/梢 TE 弧长随冲程强变) |
| H7 | 谱系正名:Willis-Peraire-White FastAero (MIT 2005-07) = BEM+粒子尾迹+快速求和的原型;非定常板处理只在 Theodorsen 滞后显著的频率才改变答案——"正是高频强迫扑翼的情形" | 独立支持 L2 前提:k=0.13–0.39、α 门控区,机理载体是解析非定常尾迹而非准定常闭合 |
| H8 (PLAUSIBLE) | 2024 状态空间耦合板-VPM:TE 后加 ghost+dummy 两排辅助板衔接粒子(ghost 板心=第一排粒子位);整个混合系统写成单一一阶光滑 ODE,解析线性化 0.3–30 rad/s 验证 | 近场配方(粒子首排放 ghost 板心)+ 存在性证明:格+粒混合可写成可微 endgame 需要的光滑状态空间;**细节未过原文页,勿引为 verbatim** |

### 2.3 LESP 喂入与验证案 (A3-C1…C8,全 CONFIRMED)

| ID | 主张一句话 | 对架构的意义 |
|---|---|---|
| A3-C1 | LESP 门控喂入强度律闭式:|A0|>LESP_crit 时脱 LEV,新 LEV+TEV 环量由 "A0 钉在 crit + Kelvin" 联立;Faure 2019 化为每步 2×2 线性系统,无迭代 | **喂入强度定律**:非迭代、fp64 友好、比 Newton 环路对 Warp autodiff 干净得多 |
| A3-C2 | 新脱涡放在"边→前一脱涡"矢量的 1/3 处(Ansari 规则);Vatistas 核 r_c=1.3×平均涡距=0.02c | 粒子释放位置 + σ 的文献锚(1.3× 局地脱落间距 → 我们的 Gaussian-erf σ) |
| A3-C3 | 变强度/合并喂涡若无 impulse-matching 速度修正必产生虚假力(Wang-Eldredge 2013;Darakananda-Eldredge JFM 2019 环量转移程序"消除虚假力") | **确诊反相账本**:粗环吞并 LE 环量无修正 → 力误差与真实 LEV 喂入反相。对策:每步一粒、此后环量冻结(免修正,首选);任何合并/变强度核则必须加修正 |
| A3-C4 | VoFFLE (AIAA 2021-1196):UVLM 20×45 + LESP 门 + 正则化粒子尾迹已发表;26k 粒/7 弦程,桌面 75 s vs CFD 2200 核时;失败模式:LEV 偏弱、卷起不全、脱落后爆炸 | **同架构先例**,strip-wise 2D LESP_crit 上翼合法(Hirato 2019/JFM 2021);其失败 = 我们 rVPM 升级的靶子;Re 1.5e5 涌现 DSV 仍是开放可主张结果 |
| A3-C5 | 2D 验证门 #1:Eldredge 正则 pitch-ramp 家族 Case A/B/C(LESP_crit=0.18/0.14 文献值,Δt*=0.015);经典 LDVM 618 LEV+750 TEV 即涌现 | S2 门的定量靶;O(10³) 元素足够 2D 涌现 → 失败归因喂入/正则化而非分辨率 |
| A3-C6 | 2D 验证门 #2:SD7003 深失速 plunge(Re=6e4, ωc/U=0.5 即 Visbal 约定 k=0.25, h0/c=0.5, α0=8°):CL 峰 2.3–2.4 @ α_eff≈21°,CD 峰≈0.35;LES 造价 ~1e5 CPU 时 | 最贴 RoboEagle 区间的 DSV 案(plunge 驱动、频率增长、α 门控 = L2 同物理);**注意文献本身有 k 定义因子 2 笔误,复现时用 ωc/U∞=0.5** |
| A3-C8 | 3D canon:Garmann-Visbal 旋转翼内稳外脱两区 LEV;Re=250 DNS 已需 0.57–3.4×10⁸ 谱点 → Re 1e5 解析 DSV 必走 LES 型 SFS 而非蛮力 | S3 定性门(±22.5° 拍幅恰好扫过内/外转换);3D 定量力靶仍缺开放数据(唯一 canon 缺口) |

### 2.4 力评估 (A4-C1…C6,全 CONFIRMED)

| ID | 主张一句话 | 对架构的意义 |
|---|---|---|
| A4-C1 | 生产混合求解器**只在束缚格上**算力:KJ 每束缚丝 F=ρΓ(V×l),V = V_vpm+V_vlm(ign_infvortex)+V_inf+V_kin;粒子尾迹永不被积分求力 | **Stage-1 力评估模板**:复用已验证粒子 Biot-Savart 在束缚丝中点求 V;DSV 升力经粒子诱导上洗/下洗改变 V 而涌现;结构上消灭无界尾迹 impulse 求和 |
| A4-C2 | 非环量项 F=−ρ(dΓ/dt)A·n̂ 默认**不计入总力**(噪声大、周期平均相消),单独存档 | L2 是周期均值:dL/df 斜率必须由环量 KJ 通道承载——告诉我们涌现该出现在账本哪一栏;瞬时力验证时该项单独记录 |
| A4-C3 | DUST 独立确认:1/4 弦控制点、条带 ΔΓ、粒子诱导速度显式进入、非定常项确定性加入 | 第二独立确认 + A/B 备选(确定性表观质量项 vs FLOWUnsteady 排除式) |
| A4-C4 | DUST Bernoulli-BIE:粒子以正则化 ∇G·(ω×u) 体积源进入表面压力,复用影响系数(**非**复用分解;新 master 中该管线休眠) | 膜 FSI endgame 的分布 Δp 路径;粒子项用粒子速度 → 我们的互诱导平流速度直接复用 |
| A4-C5 | Noca 有限 CV 谱系:对截断 CV 的动量体积分做时间差分正是虚假力振荡源;单时刻 Reynolds-transport 重排消振(**注意主张中 Eq.16 两个体积项符号曾抄反,已在核实中改正**) | 反相账本的第三机理确认;Noca 通量式只作 canon 案上的独立交叉核对门,**永不做生产力路径** |
| A4-C6 | 双计数是被显式防护的坑:半无限马蹄腿排除 (ign_infvortex);束缚涡镜像进粒子场时强度置 1e-14 掩蔽 | Warp 实现清单:每个尾迹元素恰经一种表示进入束缚丝速度;格转粒子后即刻退出解析格求和;*=1e-14 的 in-place 掩蔽是 autodiff 坑 → 用独立掩蔽 buffer |

### 2.5 GPU 与可微性 (A5-C1…C6,全 CONFIRMED)

| ID | 主张一句话 | 对架构的意义 |
|---|---|---|
| A5-C1 | 发表机翼级 rVPM 在 O(1e5–5.4e5) 粒子,16 核 CPU FMM 1 小时完赛 | 我们 1e4–1e5 粒子预算在发表包络正中;GPU 直算近实时 |
| A5-C2 | GPU 直算 N² 在 N≲5e4–1e5 优于树码(2010 硬件 66 FLOP/交互、14.4e9 int/s;现代卡更快,break-even 更高) | Stage-1 留在现有直算 Warp kernel(含 fp32 快路径先例);FMM 是延后机器,其伴随复杂度正好回避 |
| A5-C3 | FLOWVPM 生产积分器 = Williamson 低存储 RK3 (0,1/3)(−5/9,15/16)(−153/128,8/15),位置/强度/σ 同阶联合更新;松弛在整步后 | 积分器定案:3 次 Biot-Savart/步、2 寄存器/态、无自适应分支(AD 友好) |
| A5-C4 | 经典拉伸直加必不稳(同 rvpm-01);rVPM Z-耦合只加一条 σ ODE,光滑无门 | 核心物理去险的 GPU 确认 |
| A5-C5 | Green & Ning 2024:反向 AD + 手写解析 pullback(N-body 交互不上带)已在 VPM 上发表,"significantly faster" | 可微 endgame 的直接先例,与我们 Warp 经验同路:pairwise 环路写解析伴随(已有 Jacobian kernel = 前向一半),其余交给 tape |
| A5-C6 | Fu-Laurendeau 转换须"尺度一致"(粒子 σ/间距系于脱板几何)否则毁时间阶;推力/扭矩 ≤1% 是 vs 细分辨率参考解而非 URANS | UVLM→VPM 接缝的可引用设计规则;LESP 门保留为喂入判据、粒子接管下游动力学 |

### 2.6 坟场(REFUTED,禁止引用)

1. **"FLOWUnsteady 无环排缓冲立即转粒子 + 门控阈值只是 gate"** — 机制大部分属实,但 `unsteady_shedcrit` 是**模式开关**非门:默认 −1.0 时只脱 trailing、从不脱 unsteady;实际是每步两次 shed_wake(解前 trailing、解后 unsteady @0.01)。且 p_per_step 只细分 trailing。按原表述实现会行为反转。
2. **"Baik JFM 709 峰值 CL ≈ 高于 Theodorsen 50%"** — 论文自己的结论是与线性势流理论"reasonable agreement"(LEV 相下用 C(k)=1 反而更好);+50% 无出处。**不得**把它当作解析 LEV 必须复现的定量签名;St 控峰值与 k 延迟 LEV 定时这两点仍然成立可用。

---

## 3. 架构裁定(ARCHITECTURE VERDICT)

每条选择后括注文献锚。

**V1 — 束缚层:保留 UVLM 环格 + LESP 门,原样。** 束缚 AIC 纯格-格、静态部分预分解;粒子诱导只进 RHS(控制点/束缚丝中点),永不进矩阵(H4 DUST;A3-C4 VoFFLE 同构先例;H2 rVPM 自己也是 actuator 浸没束缚 + 指定线脱落)。LESP 门保持已验证状态,仅作**喂入判据**(A5-C6, A3-C4)。

**V2 — TE 尾迹转换:保 ≥1 排隐式 Kutta 环缓冲,之后转粒子;粒子数按 TE 段流向弧长比例;上游线冻结。** 永不在 TE 面对裸粒子(H4, H6);近场配方可选用 ghost 板心放首排粒子(H8, PLAUSIBLE,作工程参考不作引用依据)。转换尺度一致:新粒子 σ、间距系于脱板几何(A5-C6);现有 ring-lattice TEV 结构保留为缓冲区,只是缓冲后的命运从"环链"改为"rVPM 粒子"。

**V3 — LEV 喂入:每步每展向条带一粒,位置 1/3 规则,强度 2×2 线性系统(A0 钉 crit + Kelvin),σ_new = 1.3×局地脱落间距,环量此后冻结。**(A3-C1, A3-C2)冻结环量 = 免 impulse-matching 修正的合法路径(A3-C3);**任何**未来的粒子合并/变强度核必须同时实现 impulse-matching,否则重演反相账本。

**V4 — 粒子输运:rVPM f=0, g=1/5 + transposed stretching + σ 演化;一键 f=g=0 消融。** 具体即 FLOWVPM 离散式:Z=[(f+g)/(1+3f) S·Γ]/|Γ|²,ΔΓ=dt(S−3ZΓ−Cϵ),Δσ=−dt·σ·Z(rvpm-02)。S 由现有已验证 Jacobian kernel 缩并,新增仅 σ 数组 + 代数项。|Γ|→0 守卫用光滑分支(已知条件门断伴随坑)。

**V5 — SFS:开,fp64 先行。** Re 1.5e5 的解析 LEV/尾迹必然是 LES;E_str 与速度/Jacobian 融合进同一 pairwise pass(~1.3–1.5×),C_d 动态程序 α=0.667 预设 + backscatter 裁剪(rvpm-03/04)。分段引入:G1/G2 涡环门可先 SFS off(层流案),G3 起 SFS on(湍射流一级的物理在 FLOWVPM 中就是靠它)。黏性方案 = Inviscid(rvpm-06;分子扩散在 O(10) 对流时间的 DSV 周期内次要,湍耗散由 SFS 承担);CSM 只在 G1 黏性环子门用。

**V6 — 散度控制:correctedpedrizzetti(保 |Γ| 变体),rlxf=0.3 每步。**(rvpm-05)理由:我们现在为零控制,而普通 Pedrizzetti 的强度衰减缺陷会从账本漏环量;renormalization 的除法在可微段用门控旁路。

**V7 — 力评估:生产路径 = 束缚格 KJ,V 含粒子诱导;非环量项单独记录不入总力;双计数防护清单照抄。**(A4-C1/C2/C3/C6)周期均值 L2 验证直接用 KJ 通道;瞬时 CL 环验证(S2 门)时把 dΓ/dt 项作为独立通道叠加检查。交叉核对 = Noca 单时刻通量式,仅 canon 案、仅诊断(A4-C5,注意符号修正版 Eq.16)。膜 FSI 分布 Δp = DUST 式 Bernoulli-BIE,S4 之后再上(A4-C4)。

**V8 — 数值机器:Williamson LSRK3 + 直算 N²(fp64 主,fp32 尾迹快路径保留)+ 无 FMM。**(A5-C1/C2/C3)1e4–1e5 粒子在发表包络内,直算在现代卡上每步毫秒级;FMM/树码及其伴随复杂度全部延后。

---

## 4. 分段计划输入(S1–S5,每段一个 canon 案 + 定量门)

| 阶段 | 内容 | 验证案 | 定量门 | 文献锚 |
|---|---|---|---|---|
| **S1** | rVPM 输运落地:σ 状态 + 3/5 投影 + Z 耦合 + LSRK3 + corrected-Pedrizzetti;SFS off | G1 孤立涡环(平流 + CSM 黏性子门);G2 leapfrog 双环 | 环速/半径 vs 解析与 FLOWVPM 发表结果;**G2 关键消融:f=g=0 复现爆断,f=0,g=1/5 长时稳定**——涌现归因于公式的证据 | rvpm-01/02/06/08, A5-C3 |
| **S2** | LESP 喂入接 rVPM 粒子(2D 条带模式)+ SFS on;KJ 力通道 | G3a:Eldredge ramp Case A/B/C(LESP_crit=0.18/0.14, Δt*=0.015);G3b:SD7003 plunge Re=6e4, ωc/U=0.5 | Case A:CL 峰≈6→平台≈3;Case B:峰≈2.8–3.1 + 回程二次瞬态;Case C:CL 贴 CFD;SD7003:**CL 峰 2.3–2.4 @ α_eff≈21°(相位 ψ≈100–120°)、CD 峰≈0.35、CL_min≈−0.15** | A3-C1/C2/C5/C6 |
| **S3** | 3D:全翼束缚格 + 弧长比例 TE 转换 + strip-wise LESP 喂入;ASM 式闭面检查 | 旋转/pitch-up 有限翼 | 定性门:内稳外脱两区 LEV + 梢涡展向调制复现;(定量 3D 力靶 = 已知 canon 缺口,допdig 需另行找数据) | A3-C8, H3, H6 |
| **S4** | RoboEagle:先 118 工况子集(扫 f, α, U 角点),后全集 | L2 案本身 | **dL/df 涌现:+1.5–1.9 N/Hz、α 门控、U 超线性,零拟合复现**;历史反相账本案复跑翻正 | L2 实测, A4-C1/C2 |
| **S5** | 可微 endgame:pairwise 解析伴随 + 光滑替身(soft-clip C_d、光滑 |Γ| 守卫、松弛旁路) | 梯度 vs 有限差分核对,再接 co-design 栈 | 伴随梯度与 FD 在 canon 案上 <1% 偏差 | A5-C5, H8, rvpm-04 |

前进规则沿用 baseline-first / 一次一变量:每门未过不得进下段;S2 失败优先查喂入与正则化(A3-C5 证明 O(10³) 元素已够 2D 涌现,分辨率不背锅)。

---

## 5. 常数账本(零拟合合规性)

**文献锚定(直接取值,不许调):**

| 常数 | 值 | 出处 |
|---|---|---|
| (f, g) | (0, 1/5);h=(1−3g)/(1+3f)=2/5;∥ 移除系数 3/5 | Alvarez & Ning 2024 守恒推导;FLOWVPM.jl L85/L120 |
| SFS 动态程序 | α=0.667, rlxf=0.005, C_d∈[0,1], backscatter 裁剪 | FLOWVPM_subfilterscale.jl L188 |
| ζ(0) | (2π)^(−3/2)(Gaussian-erf,与我们现核同族) | rvpm-07 |
| Pedrizzetti | rlxf=0.3, 每步, corrected 变体 | FLOWVPM.jl L169-170 |
| LSRK3 | (0,1/3)(−5/9,15/16)(−153/128,8/15) | FLOWVPM_timeintegration.jl |
| LESP 喂入 | Δt*=0.015;1/3 放置;r_c=1.3×间距=0.02c;LESP_crit=0.18/0.14(案对应值,SD7003 谱系) | Ramesh 2014 / Faure 2019 / Gelado-Ramesh 2022 |
| CSM(仅验证门) | β=1.5, itmax=15, tol=1e-3 | FLOWVPM_viscous.jl |
| ASM 浸没(S3+) | σ_surf=b/100, overlap=2.125, σ_tbv=(t/c)·c̄/100 | FLOWUnsteady blownwing-asm |

**我们的选择(非文献定值,须披露并做敏感性):**转换缓冲排数(≥1,H6 只给"用户定义");每条带每步粒子数基数 n_t(弧长比例规则是文献的,基数是我们的);SFS 在 G3 起开的档位选择;fp32 快路径适用范围;RoboEagle 自己的 LESP_crit(0.27 之争见 research_les_suction.md ——本架构继承该文件结论:0.14 对照跑,0.27 需独立论证)。

---

## 6. 风险与已知坑

1. **Autodiff × rVPM 新项的三处对撞**(全部对应我们已知 Warp 四坑):backscatter 裁剪 + C_d 夹持 = 条件门断伴随 → 可微段用 soft-clip;corrected-Pedrizzetti 的 b² 除法 = 除累加器 NaN → 可微段旁路松弛;束缚镜像掩蔽 *=1e-14 = in-place 不可微 → 独立掩蔽 buffer。原则:primal 生产跑用硬裁剪(与文献一致),可微段用光滑替身并验证两者 primal 差异可忽略。
2. **发表 rVPM 梯子没有壁面分离 DSV 案**(rvpm-08 caveat):G3 门是我们对梯子的延伸,成功即差异化战果,失败没有文献兜底——所以 S2 前置了 2D LDVM 案(该处 canon 完整)分摊风险。
3. **粒子数与多周期累积**:单周期 1e4–1e5 在包络内(A5-C1/C2);118 工况 × 多周期若需累积则触发合并——而合并立刻触发 A3-C3 的 impulse-matching 强制条款。先用"尾迹截断/远场衰减"策略,合并留作最后手段。
4. **σ 演化 × CSM × 转换尺度的三方耦合**:rVPM 收缩与 CSM 增长是独立叠加(rvpm-06),再叠上转换处 σ 由脱板几何定(A5-C6)——σ 的三个来源必须在账本里分开记录,否则调试无从归因。
5. **两条坟场教训**:FLOWUnsteady 的 shed 开关语义按原文实现会行为反转(照 2.6 修正版实现);不要给解析 LEV 设"+50% 超 Theodorsen"的伪定量靶。
6. **性能兜底**:SFS 融合 pass ~1.3–1.5×、LSRK3 3× 求值/步;若 S4 全集吞吐吃紧,fp32 快路径对 rVPM 仍然合法(rvpm-07:pairwise 结构未变),但 S1/S2 门必须 fp64 先过。

**一句话决策**:束缚 UVLM + LESP 门原样保留;TE 经 ≥1 排环缓冲按弧长比例转 rVPM 粒子,LEV 每步一粒、1/3 放置、2×2 闭式定强、环量冻结;粒子输运换 f=0,g=1/5 的 rVPM(σ 演化 + transposed 拉伸)+ 动态 SFS + corrected-Pedrizzetti,全常数文献锚定零拟合;力走束缚格 KJ(粒子诱导入 V),Noca 只做诊断;验证梯 涡环→leapfrog(f=g=0 消融)→2D ramp/SD7003 CL 环→3D 两区 LEV→L2 dL/df 涌现→可微化。

---

## 附:题录裁定总表

| Angle | CONFIRMED | PLAUSIBLE | REFUTED |
|---|---|---|---|
| rvpm-formulation (rvpm-01…08) | 8 | 0 | 0 |
| hybrid-arch (H2…H8) | 6 | 1 (H8 ghost/dummy 细节未过原文页) | 0 |
| lev-feeding-validation (A3-C1…C8) | 8 | 0 | 0 |
| force-eval (A4-C1…C6) | 6 | 0 | 0 |
| gpu-autodiff (A5-C1…C6) | 6 | 0 | 0 |
| 坟场 | — | — | 2(FLOWUnsteady shed 开关语义;Baik "+50%" 伪签名) |

关键引文修正(核实中发现、引用时必须用改正版):AIAA J 页码 637-**656**(非 -654);rvpm-01 逐字引语出自 FLOWUnsteady 文档而非论文本体(论文为 hedged 表述且是 **angular** momentum);A4-C5 的 Noca 重排式 Eq.16 两个体积项符号(F = −∫_VCV ρ ∂u/∂t dV **+** d/dt ∫_Vb ρu dV + 面通量);A3-C6 的 k 定义因子 2 笔误(用 ωc/U∞=0.5 ⇔ Visbal k=0.25);A5-C6/H6 的 ≤1% 比较对象是细分辨率涡法参考解而非 URANS。全部 30 条完整 verdict(含 sources_checked 与 corrected_citation)存于 pipeline 的 VERIFIED CLAIMS 记录。
