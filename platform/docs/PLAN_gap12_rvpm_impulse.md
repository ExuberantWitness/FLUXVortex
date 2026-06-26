# Gap①② (AoA=0 +29% / AoA=15 +10%) — 实验经验 + 总系统冲量重构计划

**目标**:RoboEagle 扑翼气动,off-design 攻角边缘(AoA=0/15°)升力标定到 5%。巡航(5-10°)+ 推力已 solid。

## 一、实验经验:8+ 方案为什么失败(别再走这些)

测试工装:`gpu_run_twist(nc=5,ns=16,U=8,flap±45°,twist=0,freq=2,real_geom,sym,root_off=0.05)`。
实测靶(8m/s 2Hz):AoA 0/5/10/15° = **2.92 / 7.44 / 11.7 / 14.0 N**。attached(无LEV)= **1.1 / 3.95 / ~7 / 9.1 N**(≈实测一半;LEV 要补另一半 ~1.8/3.5/4.7/4.9N)。

| # | 方案 | 结果 | 失败根因 |
|---|---|---|---|
| 1 | GK X-态加性(per-strip 分离态) | AoA=0 **+77% 更糟** | 加性力 dF∝ρ·Vcol·∂Γ/∂x,Vcol 上/下扑不同→即使 Γ 奇也不抵消 |
| 2 | LEV 核侧翻转(随 sign(sinα) 翻吸力侧) | AoA=0 −76% 过冲、cruise −29% 破 | lag 与翻转位置耦合 + NACA-2406 弯度使上下非真镜像 |
| 3 | 守恒折叠(固定核进 solve 削 bound Γ) | **数值爆炸**(升力 16→50N) | 固定核 0.10c 近场诱导太强→bound 正反馈暴涨 |
| 4 | odd-kernel 直接增量(Polhamus sin²cosα·sign) | AoA=0 过抵消(0.9=−70%)**且 cruise 也被抵消(−47%)** | ±45°扑动使 5°base 的 kernel 也近抵消(downstroke 不主导) |
| 5 | post-stall Kirchhoff on bound(综合 agent 荐) | **破 cruise**(5/10/15 全降 ~40%) | per-instant α_eff gate:±45°扑动使每个 base 的 mid-stroke 都 α_eff 高→到处触发 |
| 6 | abar 判别器(signed cycle-mean sinα≈base AoA) | AoA=0 +30%→+18% 但 cruise −15% | 判别太弱:±45°扑动把 abar 压到 camber-bias 地板,AoA=0 与 cruise 分不开 |

**核心墙(2 轮 research workflow 数值证实)**:**±45° 大幅扑动 dominates → base-AoA 判别性根本性弱**。per-instant gate 到处触发(破 cruise);cycle-mean 被 flap 平均 + 弯度偏置冲淡(分不开)。任何能让 AoA=15 弯下来的 gate 必然 stroke-非对称→破 AoA=0。**单 sectional 项做不到**(综合 agent 5 种 rolloff gate 全证伪)。

**reframe(综合 agent 有价值发现)**:AoA=15 的"rolloff"其实是**单调 sub-linear 上升**(slopes 0.98→0.86→0.43 N/deg),不是 CL_max-then-drop。

## 二、rVPM 方向(用户定向,FLOWVPM 移植已在 `warp_vpm.py`+`particles.py`)

**做对的**(`part_lev` flag,默认 OFF,commit a3c177e on `aero-rvpm-lev`):
- `shed_lev_particles_kernel`:网格无关 kinematic 强度 `gmag=-klev·sgn·vmag·clen·(|sinα|-sin_crit)`,**signed**,吸力侧随冲程翻转。
- **LESP 速率门**:仅当 |LESP| 上升时脱涡(Ramesh/flap_ldvm up-stroke gate)→自调整:AoA=0 对称、AoA>0 downstroke 主导。
- 粒子稳定脱涡+对流+mutual rollup 跑通(~60s/run,n_cycle=3 spc=80,O(N²))。
- `lev_cons`:守恒耦合(粒子诱导进 solve RHS 削 bound)——**稳定**(对流粒子近场诱导衰减,不像固定核爆炸)。

**LEV 力耦合是墙(本计划要攻克)**:
- **Bernoulli 面力**:LEV **弱 ~10×**(+0.2~0.4N vs 需 ~3N)。诊断:自由对流粒子吹离吸力面;且 dp=ρVcol·∂Γ/∂x 只抓 **LEV×bound 交叉项**,漏 LEV 自身 KJ 升力 ρ·U·Γ_lev。核大小(0.02→0.10c)无影响→确认是力机制错,不是正则化。
- **朴素涡冲量** I=ρΣx_p×α_p(全局原点):**狂野**(AoA=0 −12N,AoA=15 +43N=+925%,随 AoA/klev 暴涨)。根因:**原点依赖**——LEV 净环量 Σα≠0,且翼 flap 在 z 振 ±0.5m,全局原点 z=0 ≠ 运动翼 LE→ Σα·Δz 假力矩。klev 现在有效(冲量抓到 LEV 了),但记账错。

## 三、总系统冲量重构计划(正确的 LEV 力)

**物理**:体受力 = −dI/dt,I=(ρ/2)∫x×ω dV 对**全涡系**(bound+wake+LEV)。**Kelvin 闭合**:从静止起 Στ=0(bound 补偿脱出)→**I 原点无关**→力良定义。

**关键修正(对朴素冲量)**:把 LEV 冲量参考到**脱涡点(运动翼 LE)**,因为脱出的环量由 bound 在该处补偿(偶极子):
```
I_lev = ρ Σ_p (x_p − x_le_ref) × α_p     # x_le_ref = 当前翼 LE 质心(随 flap 运动)
L_lev = −d(I_lev)_z/dt,  D_lev = −d(I_lev)_x/dt
```
参考到运动 LE 消去 z-flap 假力矩 + Σα 原点项(脱涡在 LE→ 关于 LE 的矩 ~0)。

**实现步骤**(`_v2_robo.py`,part_lev 路径):
1. 每步算 `le_ref` = 当前所有 strip LE 中点的质心(`rings` 的 c0,c1 均值)。
2. `I_lev = RHO * Σ cross(pp[:np]-le_ref, pa[:np])`(3-vec)。
3. `Lh_imp[t] += -(I_lev[2]-I_prev[2])/dt`;`Xh_imp[t] += -(I_lev[0]-I_prev[0])/dt`。
4. 移除 Bernoulli Vpart(避免双算);lev_cons 可选(冲量已含 LEV 力,bound 不必削——但 Kelvin 一致性需 bound 知道脱出,待验)。
5. d/dt 含运动 LE 项:`d/dt[(x_p-le_ref)×α_p]` 有 `−lė_ref×α_p`(翼运动/added-mass 项);先测 le_ref 参考是否驯服狂野,再决定是否补运动项。

**验收**:AoA 0/5/10/15 → 2.92/7.44/11.7/14.0(±5%),**同一 klev/sin_crit 跨频率(1.4/2.6Hz)+ 跨风速(6/10)** 不破 cruise/推力。klev/sin_crit/lev_off/core 是自由常数。

**风险/未知**:
- 冲量 d/dt 可能噪声(离散脱涡/丢粒子的 spike)→ 需平滑或子步。
- 运动-LE 参考的 d/dt 运动项可能仍留残差→ 可能需真正的全系统冲量(bound 环也进 I)= 大重构。
- O(N²) 慢(60s/run)→ 标定迭代慢;后续加粒子封顶/合并 + stretching。
- 若 le_ref 参考仍狂野 → 退到"全系统冲量"(bound rings 的 Γ·A·n + wake + LEV 一起,Σγ=0 严格)。

## 四、冲量重构实测(2026-06,本轮)

- **LE-参考冲量**(I=ρΣ(x_p−x_le)×α_p,x_le=运动翼LE质心):**仍狂野**,与全局原点几乎一致(AoA 0/5/10/15 @klev0.5 = −12/6.3/24.6/42.9N)。
- **根因(确诊)**:狂野 = **累积对流项 ρ·U·Σα_p**。冲量对**所有**脱出粒子求和,粒子累积(Σα 线性增长),每个对流粒子对 dI/dt **永久**贡献 ρU×α。LE-参考只移原点,不消远尾迹的假力。klev 现在有效(冲量抓到 LEV)但量级随累积 Σα 暴涨(AoA=15 给 +450%)。
- **这正是冲量必须用全系统(Σγ=0)的原因**:只有 bound+wake+LEV 一起且总环量守恒,远尾迹的稳定对流贡献才相消。

## 五、深层结论(诚实)

**自由对流粒子是"附着型动态失速 LEV"的错模型**:真实 LEV **停在前部吸力面上**(被分离泡 held),而粒子以 ~U 吹向下游、离开翼面。结果:
- Bernoulli 面力:粒子离面→诱导弱~10×。
- 冲量力:对所有累积粒子求和→远尾迹假力,狂野。
- lev_cons(粒子削bound):粒子离面→对bound诱导弱→bound 几乎不响应→弱。
- **三种力法都不对**,因为 LEV 物理上不在翼面上(模型让它吹走了)。

**真正的解(研究级,三选一)**:
1. **Held-LEV**:让 LEV 核停在翼前部(像 lev_merge 固定核,诱导强),但强度用 **LESP 速率门对称生成**(downstroke build / detach),力用 Bernoulli。难点=固定核+lag 仍破 AoA=0 对称(记忆㉝);需把核的"生成-detach"做成对称且力记账对。
2. **全系统冲量重构**:整个力计算换成 I=ρΣ_all x×Γ(bound rings 的 Γ·A·n + wake + LEV,Σγ=0 严格)的 −dI/dt。大重构、会动已标定 cruise/thrust(KJ力)、风险高。
3. **粒子封顶+子步去噪 + stretching 保相干**:让 LEV 核相干停留更久(stretching 抗耗散)+ 封顶去远尾迹累积。仍不解决"粒子离面"本质,可能只缓解。

**当前判断**:Gap①② 是真研究墙(本会话 8+方案+2 research workflow+rVPM 4 力法全界定)。thrust 是 solid 交付。LEV 力记账是 UVLM+VPM 混合的开放难题。

**进度锚**:thrust(Gap④)已 solid 同步 `meta-codesign`;Gap①② 在 `aero-rvpm-lev`(WIP=稳定脱涡+冲量诊断;本 plan=完整实验经验)。
