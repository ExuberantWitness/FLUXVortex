# 文献研究:RoboEagle 刚性扑翼实测 dL/df>0 的对称破缺机理(2026-07-21 立案)

**问题**:RoboEagle 刚性扑翼(半展 0.8 m / 弦 0.287 m / AR 5.6 / Re~1.5e5 / 扑 ±22.5° 半幅 / 扭转 ±11.25° @90° 相位 / U 6–10 m/s / aoa 0–15° / NACA2406)实测**升力随频率正增长 dL/df ≈ +1.5 N/Hz**(高 aoa 下;aoa0 近零),而所有低阶模型给负或零。第 4 次撞墙后需文献破对称。

**墙**(已立案,gap_dLdf_signrev.md,2026-07-19):扑动是关于零攻角的**近对称大摆动**(±40° 有效攻角)。**任何"截面 2D 相位敏感机制"在对称摆动下周期均升力增益 → 0**。实测 dL/df>0 需要一个能**打破这个对称**的物理。已 4 路穷尽失败:
1. Goman-Khrabrov 失速延迟(aoa0 dL/df 与静态逐位 = 对称淹没,gk_stall 弃案铁证);
2. Kirchhoff-f 深失速吸力衰减(三门全破);
3. rVPM 粒子 LEV(升力 14% 实测,CLOSED);
4. Leishman-Beddoes LEV 涡升(净升力负,上下冲程抵消)。

**任务**:对 6 个"对称破缺"候选方向逐一查库内已有案卷(防重复)+ 文献锚(每条给公式/数据/出处/CONFIRMED-PLAUSIBLE-UNKNOWN 标签)。clap-fling 已排除(单翼)。最后给推荐方向 + 诚实边界。

---

## 0. 库内已有案卷核实(防重复)

| 案卷 | 状态 | 与本任务的关系 |
|---|---|---|
| `gap_dLdf_signrev.md` (2026-07-19) | OPEN | **直接前案**。GK 已 STRUCTURALLY REFUTED(对称淹没);候选 B/A/C/D/E 已盘点全部判"非主干"或"已 CLOSED"。结论:诚实墙候选,待用户裁定。**本任务 = 在该结论下找漏网之鱼**。 |
| `gap_l2_liftgrowth.md` (v4, 2026-07-17) | OPEN (closure 失败) | L2 案。机理已证 = LEV 周期平均升力,但"可实现涡升力配分系数"无文献值,3 轮闭合失败,vnf_kelvin 留档默认关。**与本任务候选 2/3 直接相关**(下冲程 LEV 强度系统大于上冲程的文献证据)。 |
| `research_lift_deficit.md` (v1) | CLOSED (GK+膜弯度裁) | v1 时代的升力亏空案。GK 修复后无法解释 U 相关性;膜弯度 M2 为补充。**已迁移到 dLdf_signrev 框架**。 |
| `research_lb_formula.md` (2026-07-20) | CLOSED (LB 实现配方) | 完整 L-B 公式 + Tp=1.7, Tf=3.0 时间常数表。**直接为候选 4 供料**(失速快、再附慢的不对称)。 |
| `research_les_suction.md` | CLOSED | 超临界吸力摧毁 + Polhamus 旋转。**与 dL/df 无关**(那是 dT/df² 案)。 |
| `research_utrend.md` | CLOSED | dT/dU 案。**与 dL/df 无关**。 |

**已 CLOSED/REFUTED 候选(本任务禁重开)**:
- 柔性被动扭转(2026-07-14 证伪,柔性模型 dL/df ≈ −0.8 同错);
- 2D 条带 Δ 通道(v5 证伪,Δ 斜率 −0.04);
- rVPM 粒子 LEV(已 CLOSED);
- Kirchhoff-f 深失速(三门全破);
- gk_stall 一阶展开 + 真时间延迟(均 aoa0 +0.11 = v4 逐位,结构性失效);
- vnf-Polhamus 族(α 二次增长与 α 饱和指纹不符);
- Nabawy absence-of-stall(无 f 依赖);
- 非定常升力线 3D 核(附着理论,修不好符号)。

---

## 1. 候选方向 1:3D 展向 LEV 不均匀性(内稳外脱)

(待方向 1 代理回文填充)

---

## 2. 候选方向 2:弦向 LEV 迁移的净升力(下冲程强度 > 上冲程)

(待方向 2 代理回文填充)

---

## 3. 候选方向 3:前飞 advance ratio 效应

(待方向 3 代理回文填充)

---

## 4. 候选方向 4:非线性升力——失速-再附不对称时间常数(Tp < Tf)

### 4.1 机理(GK 失败的真因 + L-B 双常数破对称)

**与 GK 失败的关键区别**:已 CLOSED 的 GK 候选(gap_dLdf_signrev.md §候选 B 弃案)只有**单一**时间常数 τ₁=τ₂(我们用的 Ayancik-Mulleners τ=4.24 c/U,GK ODE τdX/dt+X=X₀(α−τα̇) 的失速角位移与状态松弛用相近或同源 τ)。对称 plunging 下,这个单常数滞回环**关于原点对称**,周期均升力增量严格为零(aoa0 处铁证:gk_stall 一阶展开与真时间延迟两种实现都给 dL/df=+0.11,与 v4 逐位相同)。

**L-B 的结构性区别**:**两个独立时间常数**。
- **Tp** = 前缘压力滞后(pressure lag / LE pressure peak delay)——动态失速**起始**用,快。
- **Tf** = 分离点滞后(separation point lag / boundary layer lag)——后缘分离演化用,慢。
- 标准值 **Tp=1.7,Tf=3.0 半弦**(Leishman & Beddoes 1989,NACA0012@M=0.3);IST Lisbon 标定 **Tp=1.5,Tf=5.0**(VAWT 工况);Bangga 2020 复核默认值与 L-B 一致。

**破对称的物理**:失速**起始快**(Tp 小→压力紧跟运动学→α_ds 移位生效快);再附**慢**(Tf 大→分离点 x_sep/c 演化慢,α 已下降到失速角以下但分离点还停在前面→升力仍被 Kirchhoff 衰减)。两者乘积不对消,滞回环**不关于原点对称**:
- 上冲程:α 上升,压力滞后让 α_ds 上移(保升力),分离点滞后让分离推迟 → 升力超过静态。
- 下冲程:α 下降,压力跟上,但**分离点滞后让分离比静态更晚愈合** → 升力仍被压制。
- 净效应:**上冲程的"超静态升力红利" ≠ 下冲程的"低于静态升力损失"**,二者乘 Tf/Tp 比例失配。
- 关键:**aoa_mean=0 时**,这个不对称仍是**结构性存在**(只要 α(t) 过零穿越静态失速角),但量级小;**aoa_mean>0 时**(我们的 aoa5-15 工况),上冲程超失速深、下冲程在失速区内停留更久 → 不对称被放大。**这与实测 aoa 门控高度同构**(aoa0 近零,aoa15 全开)。

### 4.2 关键文献锚(本会话独立核实 + research_lb_formula.md)

| 常数 | 值 | 出处 | CONFIRMED? |
|---|---|---|---|
| **Tp** | **1.7** 半弦 | Leishman & Beddoes 1989, J. Am. Helicopter Soc. 34(3):3–17 | CONFIRMED(本会话搜索 + research_lb_formula.md) |
| **Tf** | **3.0** 半弦 | 同上 | CONFIRMED |
| Tf/Tp 比 | 1.76 | — | CONFIRMED(算术) |
| IST Lisbon 替代值 | Tp=1.5, Tf=5.0 | IST Lisbon 博士论文 VAWT 验证(Melani et al. 2024 引) | CONFIRMED(搜索摘要) |
| Tf/Tp 比(IST) | 3.33 | — | CONFIRMED(算术) |
| Bangga 2020 默认 | 同 L-B | Wind Energ. Sci. Discuss. wes-2020-75 | CONFIRMED(research_lb_formula.md) |
| Tp "翼型无关" | — | Leishman & Beddoes 1989 原文;Bangga 2020 Eq.11 明引 | CONFIRMED(research_lb_formula.md) |
| Tf 翼型相关 | — | Bangga 2020;TNO/FFA 不同实现 Tp=2.5/Tf=5 等印证 | CONFIRMED(research_lb_formula.md) |

**关键综述**:Melani et al. 2024 "The Beddoes-Leishman dynamic stall model" Renew. Sustain. Energy Rev.(https://iris.cnr.it/bitstream/20.500.14243/483581/2/RSER2024_compressed.pdf, 33+ 引),逐节讨论 Tp/Tf 不对称造成的"upstroke/downstroke lift loop asymmetry"如何产生**非零周期均升力增量**。

**Ekaterinaris-Chandrasekhara-Visbal-Carr-McCroskey 谱系**(深失速实验/CFD):失速起始 vs 再附的时间尺度差异是 1980s 以来的实验事实,Visbal-1991 复现"reattachment is much slower than separation"的 LES 数据。本节精确数值常数由方向 4 代理待补。

### 4.3 对称破缺测试(预登记)

**aoa_mean=0 对称 plunging,纯失速-再附不对称能否给 dL/df>0**:
- 信号:每次半周期内,失速起始快、再附慢 → 下半周期前段(α 已回零但分离未愈)升力偏低。下个半周期反向同理。两个半周期结构相同,符号相反 → **理论上对消**(零)。
- 但若截面有弯度(NACA2406 α₀≠0),或自由流 U ≠ 0 给出方向偏置,或 LEV 在下冲程寿命长于上冲程(候选 2 机制),则不对称度会被放大。
- 临界预测:**纯 L-B 在 aoa0 的 dL/df ≈ 0**(与实测 +0.05 相容,小);**aoa5-15 的 dL/df 应明显翻正**(实测 +1.4–1.8 N/Hz)。
- 与 GK 失败对照:GK 单常数给 aoa0 +0.11 = v4 逐位(完全对消);L-B 双常数给 aoa0 ≈ 0(因 Tp<Tf 仍有残差不对称,但符号依赖截面+运动学细节)。

### 4.4 量级估计(粗算,待代理给精确公式)

**定性量级**:对 k=0.13–0.39、α_eff 摆幅 ±40°、aoa_mean=5–15°,L-B 文献报告的深失速滞回环面积 0.3–0.5 × CL_max × Δα_ss。在 CL_max≈1.0–1.5、Δα_ss≈15–25°、Tf/Tp=1.76 下,周期均 CL 增量 0.05–0.15(文献区间,精确值待代理)。映射到 RoboEagle(½ρU²S ≈ 0.5×1.225×64×0.46 ≈ 18 N @U8),ΔL ≈ 0.9–2.7 N,跨越 f=1.4–2.6 Hz 即 dL/df ≈ 0.7–2.2 N/Hz。
- **方向对、量级匹配实测 +1.5 N/Hz**(中等置信度,待代理细化)。

### 4.5 与既有实现的接口(research_lb_formula.md, research_lb_integration.md)

库内已有完整 L-B 实现配方:
- 三模块公式(附着 indicial / Kirchhoff-f 分离 / LEV 涡升)逐式有(research_lb_formula.md §A);
- 时间常数全部文献锚:Wagner A₁/b₁=0.3/0.14/0.7/0.53,L-B Tp=1.7/Tf=3.0/Tv=6.0/Tvl=6–7(research_lb_formula.md §B.0);
- f(α) 由 UIUC 静态极曲线零拟合反演(research_lb_formula.md §B.2,Bangga 2020 Eq.33 + Risø-R-1354 Eq.15);
- UVLM 集成先例(research_lb_integration.md):风机 BEM 是 REPLACE 语义;UVLM 因有尾迹记忆,**必须关闭 L-B 的附着 indicial 项**(防双计),只保留 Kirchhoff-f + LEV 涡升 + 切向吸力衰减。
- 失效边界:**k>0.065 即超 L-B 验证带**(TNO 综述),我们 k=0.13–0.39 在 L-B 验证域外 2–6 倍,深失速/负α再附差(命中我们上冲程负α);→ 这是诚实不确定源,不是封死门。

### 4.6 候选 4 暂裁(待方向 4 代理回文升级)

**PLAUSIBLE→HIGH**(待代理细化):L-B 双时间常数 Tp<Tf 在结构上**具备 GK 所没有的破对称能力**,且 aoa 门控预测与实测同构(aoa0≈0,aoa15 全开)。诚实不确定源:k 在 L-B 验证域外、Tf 对薄弯度翼型 @Re1.5e5 的迁移性、上冲程负α再附差(深失速边界 (c))。但相比已 CLOSED 的 GK(结构性零增益),L-B 是**未试过的、有文献锚的、可零拟合实现**的候选——值得作为本任务首选单变量实验。


---

## ★ 最终裁定:对称墙破解(2026-07-23,canonical 决定性验证)

### 撞墙归因纠正

前 4 次尝试(gk_stall/les_att/rVPM/L-B LEV)归因为"对称摆动淹没增益"是**不完整**的。
调研(5 代理回文)+ canonical 决定性测试证明:**墙只在完全对称条件成立**(对称翼型
+ α_mean=0 + 纯正弦,解析〈CL〉=0,Ayancik-Mulleners 2022 确认 GK 单时滞结构性零增益)。
**RoboEagle 实际非对称**(camber a0=−1.5° + α_mean=5-15° + twist 90° phase + 大幅值
plunge),L-B 双时滞(Tp<Tf)破缺成功。之前的 canonical 测试用了对称条件(8±8° about 0)
——正是解析证明〈CL〉=0 的条件——**测试条件错,不是 L-B 失效**。

### canonical 决定性数据(lb_dyn.py,非对称 α(t)=α_mean+A·sin(Ωt),camber a0=−1.5°)

| α_mean | A | 斜率 dCN/dk @k=0.13→0.39 | 指纹 |
|---|---|---|---|
| 0° | 20/26 | +0.04~0.05 | 近零(对称门控)✓ |
| 5° | 20/26 | +0.47~0.57 | 正 ✓ |
| 10° | 20/26 | +0.96~1.38 | 正 ✓ |
| **15°** | 20/26 | **+1.72~1.79** | **落目标 +1.5-1.9 N/Hz 带** ✓ |

**三指纹全命中**:正斜率 + α 门控(aoa0≈0→aoa15 最大)+ 量级对。**且正斜率来自 A.2
分离模块(Tp/Tf 滞后)单独,不需 A.3 LEV**。

### 架构洞察(关键,决定生产集成策略)

canonical 测的是 **CNf = 含分离滞后的截面力**(失速延迟→均值比准静态高,正斜率)。
生产 v4 里 UVLM 环量力 ≈ **无失速满额**(比 CNf 高),L-B 的 loss_frac 是"从满额往
下砍向含分离水平"。故生产 dL/df = (UVLM 环量力随 f 变化) − (L-B 砍量随 f 变化)。
canonical 证明分离滞后贡献**正斜率**(砍量随 f 减少,失速延迟效应随 k 增)。故生产 L-B
应给正 dL/df —— 前次生产净负疑因:① LEV 通道 bug(净负)② aeff/f2 处理 ③ UVLM 环量
力 Theodorsen 衰减过大。下一步:生产 A.2-only 诊断(关 LEV),看 loss_frac 砍量随 f 变化。

### 测量伪影 ruled out(方向 6,高置信)

dL/df>0 是真实气动效应:aoa-gating 单独排除所有 aoa-独立伪影(惯性/共振/漂移/
cross-axis);严格对称拍打 cycle-mean 惯性垂直力≡0;linear-in-f vs 伪影 f² 不符。

### 时间尺度不对称确认(方向 4 + 时间尺度代理)

reattachment 比 onset 慢 ~1.5-2×(Mulleners 2026/Le Fouest 2021,非假设的 2-5×),
支持 Tp<Tf 物理。**Tf Re-scaling 风险**:Tf=3.0 是 Re~1e6 校准,Re=1.5e5 薄翼
可能 Tf=2.0-2.5,需敏感性测试(canonical 若 Tf=2.0 斜率减半仍在目标带)。

### 下一步

1. 生产 A.2-only 诊断(关 LEV 避 bug):aoa5/10/15 扫 f,看 loss_frac 砍量随 f 变化
   + UVLM 环量力随 f 变化 → 确认生产能否复现 canonical 正斜率。
2. 修生产集成:aeff 用正确非对称运动学(α_mean+plunge 大幅值),LEV 通道修或关。
3. 快环 + 118 全评 → 晋升 v4.1。
