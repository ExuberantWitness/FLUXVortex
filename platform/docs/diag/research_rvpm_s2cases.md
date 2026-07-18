# S2 门禁案定义核实(Ramesh 2014 验证案,2026-07-18)

**任务**:钉死 PROJECT_rvpm S2 门(G3a Eldredge ramp 家族)的确切运动学、LESP_crit、
数值细节与力公式——旧档 A3-C5 摘要("Case A/B/C, LESP_crit=0.18/0.14")粒度不足且归属有误。

**一手来源**:JFM 751 排版版不可得(付费墙;NCSU 镜像已死)。改用**作者本人博士论文**
K. Ramesh, *Theory and Low-Order Modeling of Unsteady Airfoil Flows*, NCSU 2013
(handle 1840.16/9203,官方 PDF),其 Ch.4 与 JFM 2014 逐段同文;交叉验证:JFS 2015
排版版(Elsevier)、UNSflow 源码(KiranRamesh-Aero/UnsteadyFlowSolvers.jl)、教程 README
(引 JFM DOI)、Gelado & Ramesh arXiv:2206.11597、AIAA 2013 perching。

## 案表(thesis Table 4.1, p.105;LESP_crit 以正文为准,见下方勘误)

| Case | 运动 | 翼型 | Re | LESP_crit | 我们的门 |
|---|---|---|---|---|---|
| 1 | Eldredge ramp-hold-return, 25°, K=0.11, a=11, 枢轴 LE | SD7003 | 3e4 | **0.18** | G3a "B_sd25" |
| 2 | Eldredge ramp-return, 90°, K=0.4, a=2, 枢轴 TE | SD7003 | 1e5 | **0.14** | 未用 |
| 3A/3B/3C | 正弦 pitch+plunge / 纯 plunge (k=0.393, h0/c=0.5, α=4°) | SD7003 | 1e4 | **0.21** | 未用(G3b 是 Visbal Re6e4 案,另源 A3-C6) |
| 4 | Kinsey-Dumas 能量提取正弦 | NACA0015 | 1100 | **0.19** | 未用 |
| 5A | Eldredge pitch-up 0→90°, K=0.2, a=11, 枢轴 LE, 至 t*=5 | 平板 2.3% | 1e3 | **0.11** | G3a "C_flat90" |
| 5B | Eldredge pitch-up 0→45°, K=0.4, a=11, 枢轴 LE, hold 至 t*≈9 | 平板 2.3% | 1e3 | **0.11** | G3a "A_flat45" |

**勘误(CONFIRMED)**:thesis Table 4.1 的 Case 4/5 行 LESP_crit 印刷错位(印 0.21/0.19);
正确值 Case 4=0.19(JFS 2015 p.89 逐字)、Case 5=0.11(thesis 正文三处 + UNSflow 教程
README 逐字)。**旧档修正**:"0.18/0.14" 中 0.14 属 Case 2(SD7003@1e5),不是平板案;
平板@1e3 = 0.11。"flat plate Re=1e4 的 25°/45°" 属 TCFD 2013(LAUTAT 验证,无 LEV),
不在 JFM 案集内。

## CL 定量靶(读图,thesis 图页视觉核对)

- **Case 5B**(A_flat45):LOM CL 尖峰 ≈6 @t*≈1.5(apparent-mass 主导;CFD≈5.5),
  坡末骤降至 ≈2,hold 期 von Kármán 交替脱落 CL≈1.5–3 振荡(均值 2–2.5);LOM 与
  couplevpm 吻合至 t*≈3.5,其后幅相漂移趋势同(Fig 4.21)。
- **Case 1**(B_sd25):峰 ≈2.7–2.8 @t*≈2.5–3.0;回程起点 t*≈4 apparent-mass 向下尖刺
  (至 ≈1.0–1.2);回程末 t*≈6 下探 ≈−0.5;后归 ≈0.1(Fig 4.3)。
- **Case 5A**(C_flat90):先小峰 ≈2.4 @t*≈1.2,主峰 ≈4.0 @t*≈2.5,t*=5 时 ≈1.5;
  Cd 峰 ≈4.2;LOM 与 couplevpm 几乎重合(Fig 4.19)。

## 运动学精确形式(thesis Appendix C + UNSflow kinem.jl,CONFIRMED)

- Ramp-hold-return:G=ln[cosh a(t−t1)·cosh a(t−t4)/(cosh a(t−t2)·cosh a(t−t3))],
  α=A·G/maxG;**t2=t1+A/2K,t3=t2+πA/4K−A/2K,t4=t3+A/2K**;t1=1(代码硬编码);
  maxG 数值求于 dt 网格。
- Pitch-up(EldUpDef):α=(K/a)·ln[cosh a(t−t1)/cosh a(t−t2)]+A/2,t2=t1+A/2K。
- 教程 vortexShedding 用分数式 a_frac=0.8(↔a_s≈12.57),论文 a=11(a_frac≈0.771)——
  已知小出入,门用论文值 11。

## 数值细节(thesis §4.1.5 + UNSflow 源码,CONFIRMED)

- Δt\*=0.015 基准;`find_tstep`:Eldredge 案 Δt\*=min(0.015·0.2/K, 0.015) → K=0.4 时
  **0.0075**;正弦案 0.015·0.2/(k·amp)。
- 涡核 = **Vatistas n=2**:u=γ/(2π)·r/√(r⁴+rc⁴);**rc=0.02c 硬编码**(=1.3×UΔt\*@0.015;
  Δt 细化后代码不缩核)。TEV/LEV/束缚一律 0.02c。
- 放置:1/3 规则(Ansari)"边→前一脱涡的 1/3 处";**首涡按边缘速度**:首 TEV=TE+0.5·u·Δt
  (x 向),首 LEV=LE+0.5·v_LE·Δt(v_LE 含自由流+俯仰速率+诱导;每次重入超临界段重放首涡)。
- LEV 判据:每步先只加 TEV 解 Kelvin→算 A0;|A0|>crit 则加 LEV 联立解(2×2),
  **A0 钉在 ±crit(取瞬时符号)**,超临界期间每步一枚;ndiv=70、naterm=35、一阶推进。

## 力公式(thesis eqs 4.26–4.32 = JFS 2015 eqs 12–14;UNSflow calc_forces 逐行一致)

- F_N = ρπcU[(U cosα + ḣ sinα)(A0+A1/2) + c(¾Ȧ0+¼Ȧ1+⅛Ȧ2)]
  + ρ∫(∂φ_wake/∂x)γdx(实现为 Σ_i (u_ind cosα − w_ind sinα)·Γ_b,i,**仅尾迹诱导**)
- F_S = ρπcU²A0²;C_l = C_N cosα + C_S sinα;C_d = C_N sinα − C_S cosα
- 矩式 Ȧ1 系数:thesis/JFS=**11/64**,UNSflow 代码=3/16 —— 分歧记录在案(S2 未用矩)。
- Case 4 频率:thesis 表 k=0.377 vs 正文+JFS f*=0.14(k≈0.44)——又一处表格排印误。

## 对 S2 实现的落地与移植审计(2026-07-18,platform/ldvm_fourier.py)

**测试台两代**:
1. flap_ldvm.py 加装裁定规则(集中涡束缚层)——**被离散无关性研究否决**:n=80→140
   加密下 C 案 CL 峰 6.06→164(逐点配置解对尾迹近距遭遇越奇异);逐点压力通道在
   plunge 穿尾迹时亦有 O(30-100) 尖刺(中位数健康)。保留为 legacy 条带闭合,不动。
2. **ldvm_fourier.py LDVM2D:UNSflow `ldvm` 逐行保真移植**(S2 正式测试台)。
   薄翼 Fourier 束缚层 naterm=35 << ndiv=70 = 内建低通,加密稳定。

**移植期抓出的三个真错 + 两处保真项**(全部对 UNSflow 源码逐行核对后修正):
- 非线性尾迹项符号:集中涡解的 gb 是 ccw 正(负=正升力),薄翼 Γ 是正升力正——
  混用导致重 LEV 尾迹时 CL 翻负;
- **plunge downwash 符号**:正确为 +ḣcosα(ḣ 上正,UNSflow z-up 同);legacy flap_ldvm
  的 −ḣcosα 是 h 下正的自洽约定(其阶梯 Knoller-Betz 推力∝ḣ² 测不出符号)——
  LDVM2D 按参考修正;⚠ legacy 条带驱动链的 h 约定为下正,已端到端验证,勿"顺手修";
- **率项语义(中段升力的来源)**:Ȧ0..Ȧ2 = (TEV-only 未钉系数 − 上步已钉系数)/dt;
  超临界期间 Ȧ0 = LESP 超额/dt > 0 持续——若用钉后差分则 Ȧ0≡0,中段升力塌 1-1.5;
- camber 项乘全局部弦向速度(含尾迹诱导),非只 U cosα;
- A_n 求积:θ=linspace(0,π,ndiv) 含端点 + trapz(集中涡版的 gradient(θ(pvor)) 系统
  低 7%,已另在 flap_ldvm 加 a0_quad='exact' 选项)。

**G3a 结果(2026-07-18,全过)**:
| 案 | 靶 | 得 |
|---|---|---|
| A_flat45 | 尖峰~6@1.5;hold 振荡 1.5-3(均 2-2.5) | 6.35@1.69;[1.42,3.43] 均 2.63(坡末 2.79 vs 读图~2,注记) |
| B_sd25 | 峰 2.7-2.8@2.5-3.0;回程谷~−0.5;终~0.1 | 2.84@2.76;−0.58;0.30(回程始下探 1.54 vs 读图 1.0-1.2,注记) |
| C_flat90 | 首峰~2.4@1.2;主峰~4.0@2.5;Cd 峰~4.2 | 2.29@1.2;4.22@2.46;Cd 峰 4.19-4.30 |

**加密无关性**(用户铁律):C 案跨 (ndiv,dt,naterm)=(70,0.015,35)→(140,0.0075,70)
CL 峰 4.17-4.23(±1.5%)、首峰逐位同(2.29)——对照集中涡版 164 爆掉,截断承重。

**读图勘误**:此前"C 案 t*=5 时 CL≈1.5"为误归属——α(5)≈89.7° 时公式上
CL=CS=2π·0.11²≈0.08(cnc∝cosα→0),参考 LOM 同公式不可能给 1.5;该读数应属
同图 Cd 曲线或 couplevpm 线。不作门禁判据。

**G3b 结果(2026-07-18,SD7003 plunge Re=6e4 物理门,靶=Visbal 系 LES,A3-C6)**:
| 特征 | 靶 | crit=0.14 | crit=0.18(敏感性) |
|---|---|---|---|
| CL 峰 | 2.3-2.4 | **+2.89** @ψ=102° | +2.82 @ψ=108° |
| 峰相位 ψ | 100-120° | 102° ✓ | 108° ✓ |
| 峰处 α_eff | ~21° | 21.7° ✓ | 21.4° ✓ |
| CL_min | ~−0.15 | −0.13 ✓ | −0.15 ✓✓ |
| CD 峰 | ~0.35 | 0.28 | 0.19 |

判读:相位/α_eff/CL_min 定量命中;**CL 峰 +20% 超调对 crit∈{0.14,0.18} 双锚点稳健
= 无粘 LDVM 对 LES 的系统性偏差**(非参数伪影),CD 峰同向低 20-45%。此门与 G3a
性质不同(G3a=移植保真门,全过;G3b=对 LES 的物理门)——峰值偏差正是 rVPM 升级
(SFS/粘性)在 S4-S5 的靶点,如实注记不吸收。注意 Re=6e4 无发表 LESP_crit
(0.18@3e4 / 0.14@1e5 之间),两锚点都跑过。图:rvpm_s2_gates.png。

本地存档:核实用 PDF/源码在 session scratchpad ramesh2014/(会被清);本文件为权威记录。
