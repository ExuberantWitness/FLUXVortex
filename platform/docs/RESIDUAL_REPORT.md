# RoboEagle data.md 复现 — 诚实残差报告(v3,2026-07-04)

目标:完整复现 MDPI Drones 2025《Flapping–Twist Coupled…》全部实测(Fig17/18/19,122 唯一工况/322 点;Fig16 瞬时 6 曲线)。
红线:零拟合(所有常数 = 翼型/Re/几何物性);可切换不破坏 kelvin 生产路径;残差如实报告。

> v3(2026-07-04):按用户方法论 **GAP 规律分析 → research-pipeline 文献 → 实现 → 验证** 重做。本轮发现均值复现曾靠两层虚构对消;定位缺失机理 = 动态失速(DSV);忠实 Hirato 路线在本架构证明不稳定(nc8 发散);**涡冲量力(Li/Feng)路线网格无关、解决推力、部分解决升力**。残差从"未知"推进到"定性归因 = 分离流法向力模型(下周期)"。

## 0. 方法论与更正清单

**方法论(用户要求,记忆 feedback_gap_research_loop)**:修模型走 ① GAP 规律分析(工况维+相位维)→ ② research-pipeline 文献查缺失机理 → ③ 实现验证。禁止开关式随机试错。本轮严格遵循。

**v1→v2→v3 更正**:
| # | 旧说法 | 事实 |
|---|---|---|
| C1 | "H4 开 geo_stall 是角落稳定器" | H4 无 geo_stall;稳定器是 a0_crit 0.27 |
| C3 | "平板网格无弯度" | 弯度早在网格里(robowing_real) |
| C4 | 回归"逐位不变"判据 | GPU 跑间噪声 ±0.15N,按带判 |
| C5 | 6/25 commit"瞬时峰值已吻合" | 与 v3 滤波匹配协议矛盾,存疑 |
| C9(v3) | "hirato 路径不忠实(lev_place=wake)" | 部分对:wake 放置确污染;但**忠实 ansari+LESP 约束在 nc8 发散**——根因是约束折入 solve 的近场反馈,非放置 |

## 1. 步骤① GAP 规律(`gap_analysis.py` → `docs/diag/gap_laws.md`)

H13(当时最优)vs 实测,周期内谐波分解(Fig16,8m/s/AoA5/2Hz):
- **升力缺口 = 1/rev,中下冲程峰,∝twist²**:tw45 时 1/rev 幅 24.1N,t/T=0.30(中下冲程)。= 动态失速涡(DSV)升力缺失。
- **推力缺口 = 2/rev,中冲程峰,−8.7·twist²[rad]**:准定常 Kirchhoff 在 α_eff 峰值给全额静态失速阻力,实测小。= 分离延迟缺失。
- 两缺口同根:**动态失速未建模**。

## 2. 步骤② 文献调研(`docs/diag/research_dsv_closure.md`)

本地论文精读(Hirato 2019、Ramesh×2、LEV-lift-mechanism)+ Web(OpenFAST/Boeing-Vertol、Goman-Khrabrov、Polhamus、Ramesh LDVM 系列):
- **DSV 升力闭合**:Hirato/Ramesh 用**路线 a**(非定常 Bernoulli 表面压力,LEV 经 v_L 诱导 + ∂Γ_L/∂t),非 Polhamus 旋转(路线 b,Hirato 原文否定)。
- **分离延迟**:LESP 法无 α_dynamic(k)/Kirchhoff/GK τ——**LESP_crit 门取代全部**,延迟涌现(A₀ 携带非定常内容,穿过 crit 晚于静态)。
- **GAP-1 涌现**:DSV 升力应从 Bernoulli v_L + ∂Γ_L/∂t 涌现,无系数。
- **关键代码定位**:忠实放置 = `lev_place='ansari'`(LEV 片锚定前缘);`lev_vnf`(Polhamus)是离散环诱导太弱的工程补丁,与 Bernoulli v_L 双计。

## 3. 步骤③ H14(忠实 Hirato:ansari + LESP 约束 + vnf 关)—— 判定性负面

实现:约束对 ansari 生效(L633 拓宽)+ wake 写入门限(L1188)+ 预设(`lev_place='ansari'`, `lev_vnf=False`,无闭合套)。kelvin 回归 ✓。
- nc4:推力 tw22.5 dMean +0.22、tw45 −2.30(最优)——**GAP-2(分离延迟)证实涌现**。升力 tw45 RMSE 29(无 DSV 涌现)。
- **nc8:发散**(T RMSE 206N、L 101N)。证伪"细化网格 DSV 涌现"假说。
- **根因**:LESP=LESP_crit 隐式约束必须把 LEV 诱导折进 bound solve(rhs_f = rhs − INF@gL);离散环 UVLM 里该折入是近场反馈(nc 配点近 LE 环 → 需更大 gL → 诱导更强 → 跑飞)。代码注释 L616-619 早警告此。**忠实 Hirato 路线 a 在本架构任何实用网格都不稳定**。

## 4. 步骤④ H16(Li/Feng 涡冲量力,用户选项 B)—— 验证通过

文献给出网格无关的等价路线:`F_LEV = −d/dt[ρ Σ Γ_j·A_j]`(涡冲量导数,A_j=环矢量面积),与 Bernoulli 压力数学等价但**不依赖面板分辨**。实现:ansari 涡片每步末算 I_LEV,F=−dI/dt,加法累加到 Fzb_tot;关掉 Vlev_a 进 Vcol(去双计);脱落用稳定 kelvin 路径;`lev_impulse` 开关默认关。常数:a0_crit=0.27(SD7003@Re2e4 物性)。零新增常数。

**Fig16 结果(nc4)**:
| | tw0 | tw22.5 | tw45 |
|---|---|---|---|
| H16 推力 dMean | +2.84 | **+0.25(全场最优)** | **−1.94(最优,方向对)** |
| H16 升力 RMSE | 10.0 | 16.6 | 23.67(K0 28.44,部分改善) |

**网格无关性验证(nc8,选 B 的核心理由)**:
| H16 | nc4 | nc8 |
|---|---|---|
| T tw45 dMean | −1.94 | −1.74(≈,稳定) |
| T tw22.5 dMean | +0.25 | +0.13(≈) |
| L tw45 RMSE | 23.67 | 21.56(≈,略好) |

对比 H14@nc8 爆炸(T 206N)。**涡冲量路线网格无关、稳定**。

**结论**:
- **GAP-2(推力/分离延迟)= 解决**:H16 涡冲量,网格无关、第一性原理、无不稳定约束。
- **GAP-1(DSV 升力)= 部分解决**:冲量通道加法涌现(tw45 升力 28→24),但 bound 附着流幅值虚构主导。

## 5. 剩余残差(如实,定性归因 = 下周期机理工作)

### 5.1 升力幅值(深失速分离流法向力模型缺失)
H16/H17 实验共同确认:bound 附着流 Bernoulli 力在深失速是**一个后倾矢量**(升力+阻力同体),幅值虚构 ±18N。任何标量/矢量帽(stall、kirch_cn)scale 该矢量 → 修升力必损推力(H17:升力 RMSE 10.8 最优但推力 +3.90 退)。**需分离流法向力模型**(力重定向为阻力主导,CN·n 非 scale-Fb)——经典附着 UVLM 限制,是独立机理工作,非开关可解。

### 5.2 U6/静阻截距(rig 类外,d_para 语义)
H16 下 U6 残差退化为干净常数截距(符号已对)。d_para(唯一标定常数,rig 支撑板阻力)重标与标度律(U² vs 常数)在胜者全量残差上终裁。

### 5.3 aoa0 升力欠预测
弯度已在网格(C3 更正)。真因未结;候选:弯度 nc4 欠分辨、根部支撑板间隙。

### 5.4 data.md 图题勘误
Fig18/19 的 (c)/(d) 图题与列头相反;列头权威。

## 6. 记分卡(阶梯:趋势>符号>>50%误差数><20%数>MAE)[H16 全量扫后回填]

v1(K0/H2/H4,`SCORECARD_full.md`):K0 推力符号 23%、H4 73-86%(但靠对消)。H16 全量 + 终版对比图待回填。

## 7. 回归状态
- kelvin 生产路径冒烟:L=+5.99/T=+1.14 vs 参考 +5.9123/+1.1586,在跑间噪声带(±0.15N)内 ✅
- Hirato Fig.15:off 4.31✓ on峰 3.48✓ ✅
- H16 网格无关:nc4≈nc8,无发散 ✅(对比 H14 nc8 发散)

## 8. 本轮产出文件
`gap_analysis.py` + `docs/diag/gap_laws.md`(步骤①)、`docs/diag/research_dsv_closure.md`(步骤②)、`_v2_robo.py` H14/H16 改动(约束 ansari 生效、wake 门限、涡冲量通道、Vlev_a 去重)、`diag_runaway.py`/`diag_component.py`/`fig16_compare.py`、H14-H17 预设。
