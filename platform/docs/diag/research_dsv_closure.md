# 文献调研产出(2026-07-04):DSV/LEV 升力闭合 + 分离延迟 —— 步骤 2

来源:① 本地论文精读(Hirato 2019、Ramesh×2、LEV-lift-mechanism、dvm3d、mavs、flap)
② Web 检索(OpenFAST/Boeing-Vertol、Goman-Khrabrov、Polhamus NASA TN、Ramesh LDVM 系列)

## 核心结论:我们的 hirato 路径实现**不忠实**,忠实实现自闭合两个 GAP,零新增系数

### 1. DSV 升力闭合 —— 论文用路线 (a),不是 Polhamus 旋转(路线 b)
- **Hirato Eq.13/17/22**:脱出 LEV 的升力 = 非定常 Bernoulli 表面压力,LEV 以两种方式进入:**① v_L(Biot–Savart 诱导速度)在 `(v∞+vm+vW+vL)·γ` 项里;② ∂Γ_L/∂t(LEV 环量增长率本身就是压力/升力项)**。"不脱时 Γ_L=0,无贡献"。
- **Hirato 原文明确否定 Polhamus 类比**(§II.C):"LESP 与 Polhamus 的前缘吸力类比**几乎没有关系**……Polhamus 是给后掠尖前缘三角翼的**定常** LEV 用的;LESP 是识别 LEV 形成起止的。"
- `k_v=2π/(1+2/AR)` 在 Polhamus 原 NASA TN 里**查无此闭式**(Agent② 标记存疑,系教科书近似)。
- **含义**:`lev_vnf`(Polhamus 旋转分支)不在论文里;若同时开 v_L(Bernoulli)与 vnf → **双重计 LEV 升力**。代码注释自承此张力。

### 2. LESP 法没有 α_dynamic(k)、没有 Kirchhoff、没有 Goman-Khrabrov τ
- 六篇论文**无一**给出 α_dynamic(k) 公式或 GK 滞后。**LESP_crit 取代全部**。
- "分离延迟"在 LESP 法里是**涌现**的:`LESP(t)=A₀(t)` 因携带非定常(加质量+尾迹诱导)内容,穿过 LESP_crit **晚于**准定常 α>α_static 触发 —— 正是 GAP-2 需要的。
- DSV 升力(GAP-1)**涌现**自 (Bernoulli v_L + ∂Γ_L/∂t 脉冲),无任何涡升力系数。∝twist² 来自 `F_S=πρc·U_rel²·A₀²` 的 `U_rel²·sin²α_eff` 标度(扑动速度²·扭转²)。

### 3. **关键代码缺陷定位**:lev_place='wake' 是误实现
- **忠实放置 = `lev_place='ansari'`**:LEV 是一张**锚在前缘、覆在吸力面上的独立涡片**(Hirato Fig.4/5/11),只经约束 INF 矩阵 + Bernoulli v_L 进入求解。
- **`lev_place='wake'`(我们的 HIRATO_COMMON 默认)**:把 LEV 环丢进 TEV 尾迹数组 → ① 进入 bound 求解 RHS 污染环量 ② 对流出后缘。**代码注释自承**:"wake = old (trails off the back, wrong)"。
- **这解释了我们 P0-P2 的全部 hirato 病症**:脱环近场反馈增益>1(P0 跑飞)、A0 亚临界塌缩(尾迹环持续压制解)、tw45 顺桨相位虚构载荷 —— 全是 'wake' 放置的副作用。`lev_vnf` 当初正是为补 'wake' 环诱导太弱而加的数值补丁。
- **kelvin 路径(PROD)用 `lev_place='ansari'`** → 无此污染 → 这就是为什么 kelvin 路径数值干净(但无 LESP 约束)。

## 候选实现 H14(忠实 Hirato,零新系数)
`lev_shed_mode='hirato'`(隐式 LESP=LESP_crit 约束)+ `lev_place='ansari'`(锚定 LE 涡片)+ `lev_vnf=False`(去 Polhamus 双计)+ 终止=LESP 门(`lev_detach_deg=90`)+ **不开** kirch_cn/stall/fsep_lag(LESP 门即延迟)+ a0_crit=0.27(SD7003@Re2e4 物性)。
预测:DSV 升力(GAP-1)与分离延迟(GAP-2)同时涌现;P0 角落跑飞因无尾迹污染自消。若离散环诱导在 nc4 仍太弱(Agent① 警告),是网格细化问题,非加系数许可。

## 备选(GAP-2 若未完全闭合)
Boeing-Vertol 动态失速角闭式(OpenFAST 生产默认,对厚度零拟合):
`α_dyn = α_eff − k1·γ_L·√|α̇·T_u|`,`γ_L=1.4−6δ`,`δ=0.06−t/c`,NACA-2406(t/c=0.24)→ γ_L=2.48。
Goman-Khrabrov 滞后 `df/dt=(f_qs−f)/T_f`,`T_{f,0}=3` 半弦对流(Hansen-Gaunaa-Madsen/OpenFAST 默认,Re~1e6 标定;低 Re 可能需增大)。

## 引用(实现时入报告)
- Hirato, Ramesh, Murthy, Gopalarathnam, "Vortex-Sheet Representation of LEV Shedding from Finite Wings," AIAA J 62:3356, 2024(本地 PDF 即此,2019 预印本)
- Ramesh et al., "Discrete-vortex method with novel shedding criterion…," J Fluids Struct 65:102, 2017(SD7003 Re=30k,LESP 固定预测多工况 = 零拟合规杆)
- Ramesh, "On the LE-Suction and Stagnation-Point Location…," J Fluid Mech 886 A5, 2020(LESP 闭式)
- Goman & Khrabrov, J Aircraft 31(5):1109, 1994(状态空间滞后)
- Polhamus, NASA TN D-3767, 1967(吸力类比原版,K_v 查图)
- OpenFAST AeroDyn v3.5.3 theory §4.2.1.8(Boeing-Vertol γ_L 闭式)
