# P2-S5 强耦合移植:文献锚定调研报告(2026-07-06,research-pipeline)

> 方法:4 个问题并行 web 调研(题录逐条核实,未核实项标注)。定位:S5 移植实现依据。
> 按用户裁定跳过外部 LLM 评审,以解析/收敛门禁为准。

## 核心结论(一段话)

实测环增益 1.6 与 CGN 理论单侧最低模预测 ρ_f·c/(π·ρₛh)=1.61 **数值吻合**,dt 无关性是
added-mass 型失稳的定义性指纹——机理归属闭环。我们已做的"条带 M_a 上 LHS + a_lag 滞后
补偿"在文献里有正统名字:**结构侧广义 Robin 传输条件**(Badia-Nobile-Vergara 2008 式 15-16)
/ **解析 added-mass 代理的 quasi-Newton**(Gerbeau-Vidrascu 2003);它"单独不够"的病根也
有理论解释:**收敛率由 ‖M_true−M_model‖ 控制**,单标量条带近似残差太大。S5 正解 =
**UVLM 自洽的 added-mass 矩阵**(从 AIC 非环量块提取,Lefrançois 2016/2017 面元法直接
先例:~6 次迭代、与 ρ_f 无关)上结构 LHS + **窗口级 Picard 迭代到收敛 + Aitken**
(Küttler-Wall 配方,μ≈5 预期 8-15 次/步)。

## 1. 失稳理论定位(CGN/FWR 谱系,全部题录核实)

- **Causin-Gerbeau-Nobile 2005**(CMAME 194:4506, DOI 10.1016/j.cma.2004.12.005,ICES
  预印本全文核实):线性势流+薄壁结构,added-mass 算子 M_A 紧自伴正;显式交替格式
  **无条件不稳定 iff ρₛh/(ρ_f·μ_max)<1**(Prop.3, Eq.26),判据不含 δt("irrespectively
  of the time step")。环增益显式式:**gain_i = ρ_f·μ_i/(ρₛh + aδt²)**(Prop.4 证明内)。
  子迭代收敛 iff **ω < 2/(1+ρ_f μ_max/ρₛh)**;ω=1 的 Gauss-Seidel 与显式同判据。
- **Förster-Wall-Ramm 2007**(CMAME 196:1278):**任何顺序交错方案都存在临界质量比**
  (§4.3 通用定理);C_inst 表(0阶/1阶/2阶预测器 × BE/BDF2:3 → 1/6,精度越高越不稳);
  "Δt 越小失稳越早"属粘性 NS 语境(结构刚度稳定贡献 ∝Δt² 消失更快)。
- **van Brummelen 2009**(J.Appl.Mech 76:021206):不可压/势流 added-mass 与 Δt 无关
  (椭圆瞬时响应),可压缩 ∝Δt——解释为何经典可压气弹传统从未遭遇、轻膜+UVLM 必遭遇。
- **本案定位**:μ=ρ_f c/(ρₛh)≈5.06;单侧最低模 m_a=ρ_f c/π=0.112 kg/m² → gain=1.61
  (**实测 1.6**);双侧/刚板估计 3.2-4.0。两遍窗口 PC=截断在 2 次的不收敛定点迭代,必炸。
  失稳模式判别法(留作诊断规范):残差发散/振荡+减 Δt 无效=added-mass 型(降 ω/升级 IQN);
  先降后停滞=非线性型(减 Δt 有效)。

## 2. 强耦合迭代配方(Küttler-Wall/Degroote,原文核实)

- **Aitken Δ²**(KW 2008, Comput.Mech 43:61;公式经 Degroote 2010 全文交叉核实):
  `ω_k = −ω_{k−1}·(r_{k−1}·(r_k−r_{k−1}))/‖r_k−r_{k−1}‖²`;**步首 ω 继承上步末值,
  保号+限幅** `ω⁰=sign(ωₙ)·min(|ωₙ|, ω_max)`(首步 ω_max;推荐 ω_max=0.5,发散降 0.1);
  迭代内不限幅(负 ω 与 |ω|>1 是加速来源)。
- **判据**:界面残差相对该步初始残差降 3 个量级(rel 1e-3,验证期收紧 1e-4~1e-5);
  ⚠️ 加速度量纲含 1/Δt²,**绝对容差不可移植,必须用相对范数**。
- **迭代数期望**(Degroote 2010 Table 2/3 实测):μ=10 → Aitken 9.9 次/步;μ≈1 → 26.7;
  **μ≈5 预期 8-15 次/步**。固定 ω 的 Picard 在大半参数域直接发散(Uekermann 论文实测)。
- **IQN-ILS 升级时机**(Degroote 2009, C&S 87:793):Aitken 平均 >15-20 次/步或偶发发散;
  强 added-mass 下 26.7→6.6(4×);实现=V/W 差分矩阵+economy QR+filter 1e-2+reuse 2-10。
- **加速度定点 ≡ 位移定点**(分析结论+Chow&Ng 2016 先例):Newmark 仿射双射 + Aitken
  零次齐次 ⇒ 迭代序列完全相同。**修正此前误判:加速度定点本身无免疫力**;co-design
  平板栈稳定靠 μ=0.27+Aitken。全部 KW/Degroote 理论直接适用于库内驱动。
- **每子步 vs 每窗**:每时间步定点是主流;窗口化必须配 waveform 插值(preCICE/Rüth 2021)。
  我们的"窗口+子循环+线性力插值+窗口级 Picard 到收敛"= preCICE 隐式窗口耦合的既有形态 ✓。

## 3. added-mass-safe 构件的正统形态(BNV/Lefrançois,全文核实)

- **Badia-Nobile-Vergara 2008**(JCP 227:7027,全文读):结构侧广义 Robin 条件(式 15)
  `(ρ_f M/Δt²)η^{n+1} + n·T_s·n = (ρ_f M/Δt²)(2η^n−η^{n−1}) − p̂` ——**与我们的
  M_eff=M+M_a、F≈F_w−M_a(a−a_lag) 逐项同构**;标量近似 γμ_max·I(式 16)即条带 m_a 的
  文献形态;**收敛率由 ‖M_true−M_model‖ 控制**(单标量不够的病根)。RN 方向(结构惯性
  进流体 BC)对 UVLM 侵入性大,不取。
- **Lefrançois 系(面元法直接先例)**:Brandely & Lefrançois 2016;"How an added mass
  matrix estimation may dramatically improve FSI calculations for moving foils",
  Appl.Math.Model 2017——**非定常面板法,从 AIC 后处理提取 added-mass 矩阵**驱动强耦合:
  ~6 次迭代收敛、与 ρ_f 无关(经典方案 ρ_f>8 即发散)。**这就是 S5 的施工图**。
- **气弹传统正当性**:非环量(表观质量)隐式进结构算子 + 环量滞后迭代 = Theodorsen/p-k、
  Peters 有限状态、扑翼 ROM(Schwab-Reade-Jankauski 2022)的标准分工。
- 命名规范(论文用):"structure-side Robin/added-mass preconditioned strong coupling"
  或 "quasi-Newton subiteration with an analytic added-mass surrogate";引 BNV 2008 +
  Gerbeau-Vidrascu 2003 + Lefrançois 2017。⚠️ "mass shifting" 一词未证实存在,弃用。

## 4. UVLM 实现实践(SHARPy 源码级核实)

- **SHARPy DynamicCoupled**:每时间步块 GS 强耦合;`fsi_substeps=70(上限)/最少 3 次/
  容差 1e-5(q 与 q̇ 相对变化)/力松弛 0.2 起`;**dΓ/dt = 子迭代内当前 Γ 对上个收敛步的
  一阶后向差分**(方案 b,主流);added-mass 困难的代码证据:子迭代内非定常力 0→1 ramp
  (`pseudosteps_ramp_unsteady_force`)、前几步禁非定常力、Γ̇ Wiener 滤波。
- **Roccia-Preidikman 2017**(AIAA J 55:1806):同步强耦合,3-7 次迭代到 1e-10;∂φ/∂t
  在迭代环内重估。
- **Mavroyiakoumou-Alben**(JFM 2020/2022):涡格+膜+added-mass 定量的唯一算例族,轻膜
  用 Broyden 全隐式——**S5 定量对标锚**(质量比×预张力的膜颤振参数空间)。
- dΓ/dt 差分基准**永远是上个收敛步**,不是上一子迭代;每子迭代从收敛态整步重做。

## S5 移植施工图(定案)

1. **修/验 madd → UVLM 自洽 added-mass 矩阵**(Lefrançois 路线):provider 现有 madd
   正是此意图,实测不定号 → 对标验证:刚性平板算例 vs 解析 ρπc²/4 分布(符号/量级/
   对称部分 PSD);修复后 M_eff = M − madd 上结构 LHS,配 a_lag 滞后补偿(BNV 式 15,
   管路已就位:`forces.payload['a_lag']`)。条带标量 M_a 降级为 madd 不可用时的回退。
2. **窗口级 Picard 迭代到收敛 + Aitken**:coupler 已支持 iterations+adaptive_tol;
   Aitken 换掉固定 ω(公式见 §2;ω 步首继承保号限幅 0.5;残差用界面位移量纲相对范数
   rel 1e-3 起步);最少 3 次(SHARPy);上限 20,超限 → ω_max 减半重试一次 → 报告。
   dΓ/dt 每迭代对上个收敛窗差分(coupler Picard 天然如此 ✓)。
3. **启动保护**:前 ~5 窗非定常力 ramp(SHARPy 同款,已有 load_ramp 管路);全幅起步
   (provider 需要运动);Γ̇ 噪声大时加时间滤波。
4. **门禁**:①刚板 madd vs 解析(修复验证);②ρ_f 扫描收敛性(Lefrançois 判据:迭代数
   应与 ρ_f 近无关);③迭代数监控(均值 >15 → IQN-ILS);④Mavroyiakoumou-Alben 膜颤振
   参数空间定量对标(可选,S6 级);⑤A2/A3 锚点(刚性 4.2N/实测 7.79N 升力量级)。
5. **预期成本**:8-15 次迭代/窗 → 相对两遍 PC 约 5-8×;结构子步 ~30ms×20/窗,总量级
   可接受(单工况 2 周期 ~20-40 min)。

## 修正记录(如实)

- "加速度定点天然对 added-mass 免疫"——**错**,与位移定点谱等价;稳定性来自 μ 小+Aitken。
- "窗口松耦合加欠松弛可救"——**错**,g>1 时 (1−ω)+ωg>1,固定窗间松弛数学上不可能稳。
- "Smith-Shyy 1995 是势流+膜"——**错**,是层流 NS+膜;势流+膜的对标应使用
  Mavroyiakoumou-Alben。
- Stanford&Beran 2010 实际题名为 "Analytical Sensitivity Analysis of an Unsteady
  Vortex-Lattice Method for Flapping-Wing Optimization"(J.Aircraft 47:647)。

(四份原始调研含 ~50 条源链接,存 agent 转录;题录核实状态逐条见各节标注。)
