# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Anchor Check

- 原始瓶颈仍是 equilibrium/transient load ownership，而不是完整三维尾迹。
- v5a 只改条带级时域所有权；UVLM、LDVM数值核、2D→3D投影与实验适配全部冻结。
- v5b、apparatus adapter和variable-rate shedding仍在本轮范围之外。

## Simplicity Check

- 主贡献仍只有一个：strip-local equilibrium–transient exclusive ledger。
- 删除重复的 `a_i` gate；paired LDVM 本身提供 onset，单个对流状态负责慢变分量移除。
- 唯一新增主参数是 `lambda_tau=1`，且明确为 development hypothesis，不冒充来源参数。

## Changes Made

1. 删除 `a_i`，保留完整的 `r-m` recovery tail。
2. 状态方程改写为局部对流坐标的指数精确离散。
3. equilibrium residual 改为同一局部有效迎角下 `2D equilibrium − 2D attached`。
4. 明确 section drag baseline，不触碰 UVLM induced drag。
5. 明确 exact attached reduction 的三个同时条件。
6. 冻结 v4b component-resolved 2D→3D投影，避免同时改变空间与时间模型。
7. 为第四篇盲测预先冻结最低效应门。

## Revised Proposal

# Research Proposal: FluxV v5a — 条带级 equilibrium–transient 排他载荷所有权

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Technical Gap

v4b 用一个展向平均的 global persistence 混合已经积分后的完整载荷。它让静态 polar 与完整 paired-LDVM discrepancy 同时拥有准稳态分离：Yang 中低迎角被过度扣升力；Figure 14 的15°组改善但25°组退化；Baik owner几乎不随相位变化。共同修复应发生在**条带局部系数账本**，而不是继续调整全局权重或阈值。

## Method Thesis

> FluxV v5a 保留 UVLM 为唯一有限翼/非循环骨架，以同一局部有效迎角下的二维 equilibrium–attached 截面差量修正稳态饱和和型阻，再以 paired LDVM 差量的对流高通部分表示瞬态 LEV/吸力变化；两者在定义上排他，因此无需 global persistence 或第二个 gate。

## Frozen vs New

### Frozen

- UVLM bound solve、三维 induced downwash、induced drag和added-mass载荷；
- v4b paired-LDVM section solver、`Lcrit`来源假设和component-resolved 2D→3D projection；
- 三篇几何、运动、数值分辨率、GT、滤波与评分；
- ULLT仅作Figure 11 attached oracle，不进入生产载荷。

### New

- 每条带一个二分量低通状态 `m_N,m_D`；
- equilibrium residual 与 transient residual 的显式 ledger；
- 一个冻结的无量纲对流时间 `lambda_tau=1.0`。

## Exact strip-local formulation

从 UVLM 当前步诱导速度得到局部二维有效速度 `V_2D` 与有效迎角 `alpha_eff`。所有 residual 先在条带法向/阻力方向计算，再沿已有 v4b component projection 投影并展向积分：

\[
F_i^{v5a}=F_i^{UVLM}+R_{N,i}^{eq}\hat n_i+R_{D,i}^{eq}\hat d_i
+R_{N,i}^{tr}\hat n_i+R_{D,i}^{tr}\hat d_i.
\]

### Equilibrium section residual

\[
R_{N,i}^{eq}=q_iS_i
\left[C_{N,eq}^{2D}(\alpha_{eff,i},Re_i,\mathcal G_i)
-C_{N,att}^{2D}(\alpha_{eff,i},Re_i,\mathcal G_i)\right],
\]

\[
R_{D,i}^{eq}=q_iS_i
\left[C_{D,eq}^{section}(\alpha_{eff,i},Re_i,\mathcal G_i)
-C_{D,baseline}^{section}(\alpha_{eff,i},Re_i,\mathcal G_i)\right].
\]

- `alpha_eff` 已包含 UVLM 的三维 induced downwash；
- 两个被减函数都是同一二维截面、同一局部迎角和Re，绝不拿有限翼slope减二维polar；
- `C_D_baseline`逐benchmark在manifest声明：零、已有source profile term或provider的attached profile drag；同一profile drag只能出现一次；
- UVLM induced drag不进入这个减法，也不被 residual 替换；
- attached branch上两个差量严格为零。

### Paired-LDVM raw discrepancy

相同 kinematics、TE wake、core、step 和2D→3D投影下运行：

\[
r_i^n=C_i^{LDVM,LESP,n}-C_i^{LDVM,attached,n}.
\]

未触发分离时两解逐位相同，因此 `r=0`；材料LEV的出生、发展、对流和恢复均已包含在 `r` 中，不再乘第二个 activity gate。

### Convective equilibrium removal

定义局部对流增量：

\[
\Delta\chi_i^n=\max(|V_{2D,i}^n|,V_{floor})\Delta t/c_i,
\]

并用指数精确离散：

\[
m_i^{n+1}=m_i^n+
\left(1-e^{-\Delta\chi_i^n/\lambda_\tau}\right)(r_i^n-m_i^n),
\]

\[
R_i^{tr,n}=q_i^nS_i(r_i^n-m_i^n).
\]

- `lambda_tau=1.0` 是评分前冻结的 development hypothesis；`0.5/2.0`仅作不替换主模型的敏感性；
- `V_floor=1e-6 U_ref` 仅防除零，并随速度floor做敏感性；
- `m(0)=0`，至少运行到最后两个周期的状态差与载荷均值差都低于 `1e-4`；否则结果不评分；
- `lambda→0` 时 transient residual趋零；`lambda→∞,m0=0` 时恢复raw paired discrepancy；
- no-LESP且`m0=0`时 `r=m=0`逐位成立；
- 恒定`r`最终 `m→r`，只剩equilibrium residual；
- 周期残差在局部系数账本闭合，但姿态/动压投影仍允许产生非零平均全局推力和功率。

## Exact load ownership

| Component | Sole owner |
|---|---|
| finite-wing bound circulation/downwash/induced drag | UVLM |
| added mass/noncirculatory pressure | UVLM |
| 2D equilibrium saturation and section pressure/profile drag residual | equilibrium provider |
| transient LEV hysteresis and suction change | convectively high-passed paired LDVM discrepancy |
| attached oracle | ULLT, evaluation only |

Attached exact reduction只有在 `r=0,m=0,R_eq=0` 同时成立时声称。Figure 11 regression必须逐项检查这三个条件，不能只检查LES P未触发。

## Parameter provenance and restrictions

- Yang：沿用v4b的`Lcrit=sin(5°)`公开映射假设；
- Figure 14：沿用`sin(CLmax/CLa)=0.2393`；Cd0=.057只加一次且所有相关速度点为0.75c；
- Baik：主线0.11，0.19只作来源冲突敏感性；v4b/v5a公平使用同一free-tip surrogate；
- 新增`lambda_tau=1.0`仅称development hypothesis，不称source-derived；
- 禁止`case_id`、逐AoA系数、观测残差表或从三论文选择最优lambda。

## Validation gates

### Yang 2025

- L/D MAE≤`4.327/2.512 gf`；
- 每个攻角相对v4b恶化≤0.4 gf；
- 5°/10°升力和15°阻力实质改善；20°/25°饱和不丢失。

### Scherer/Izraelevitz Figure 14

- 14-marker RMSE≤0.0230；15°组≤0.02268；25°组≤0.02284；
- 12-condition RMSE<0.02751；max error≤0.050；单点恶化≤0.0112。

### Baik W1–W4

- filtered CL/CD macro RMSE各比v4b改善≥5%；
- 8/8 case-channel不退化；W2 Q1和W3 Q1–Q2 CL改善；
- 任一相位象限RMSE恶化≤5%；raw与1Hz filtered同时报告。

### Attached and numerical limits

- Figure 11 CL/CD RMSE相对现有attached result退化≤2%；
- module-off逐位等于v4b输入基线；ledger residual<1e-12 relative；
- time/chord/span/wake单因素加密后headline变化<5%；
- lambda sensitivity只报告，不选优。

## Ablations

首轮只有三项：`equilibrium-only`、`transient-only`、`full v5a`。比较对象只有old FluxV、v4b和v5a；ULLT/作者模型是外部参考，不冒充同一production path。

## v5b go/no-go

只有当v5a通过owner hard gates、但Baik仍有>5%的wake-retention变化且残差与材料LEV对流龄期系统相关，才启动v5b。v5b改为shared TE/LE wake在UVLM AIC中产生动态力，LDVM仅提供shedding law，不能继续添加force discrepancy。

## Generalization protocol

三论文只能支持“retrospective multi-paper Pareto improvement”。冻结v5a commit、全部参数、GT哈希、评分脚本与失败门后，再选第四篇未查看载荷的实验。盲测预注册通过门：主CL/CD指标至少一项比v4b改善≥5%，另一项不得恶化>2%；任何被论文明确报告的子组均不得恶化>5%。失败必须保留并终止“泛化提升”声明。

## Feasibility

- v5a公式/单元测试：2–4天；
- 三论文smoke和ledger消融：1–2天；
- full与单因素敏感性：2–4天；
- v5b若触发：另计2–4周。

