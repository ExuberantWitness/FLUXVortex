# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Anchor Check

- v5a解决三论文的Pareto与载荷所有权问题；Problem Anchor中的“统一Kelvin”是最终v5目标，不冒充v5a已实现。
- v5a只承诺UVLM自身Kelvin、两套LDVM各自Kelvin与最终force ledger闭合。
- 若用户要求一个共同UVLM–LDVM circulation system，必须通过go/no-go后的v5b实现，不能靠文案提前声称。

## Simplicity Check

- 主贡献：strip-local equilibrium–transient exclusive ledger。
- 无第二gate、无生产ULLT、无shared wake、无apparatus adapter、无variable-rate分支。
- `R_eq`和`R_tr`的空间映射彻底分开，避免最后一处实现歧义。

## Changes Made

1. `R_eq`固定为直接section-to-strip面积积分；输入包含UVLM downwash的局部`alpha_eff`，不使用LDVM的`g/g²/added-mass/suction`增益。
2. `R_tr`单独保留冻结v4b component-resolved LDVM projection。
3. v5a守恒声明降级为底层各自Kelvin regression + force-ledger closure；共同Kelvin仅属v5b。
4. 第四篇盲测的provider选择规则也在查看载荷前冻结。

## Revised Proposal

# FluxV v5a: Convective equilibrium–transient load ownership

## Goal and scope

在不改变UVLM bound solve和wake topology的首轮中，仅修复v4b已被三论文共同否定的global load blend。v5a必须同时优于Yang 2025、Scherer/Izraelevitz Figure 14和Baik W1–W4上的v4b；否则假设被证伪。三论文均为已查看开发证据，不能叫held-out。

## One-sentence thesis

保留UVLM为唯一有限翼与非循环骨架，把稳态截面非线性写成直接section residual，把动态LEV效应写成paired-LDVM discrepancy的局部对流高通部分，使两类修正在定义上互斥并删除global persistence。

## Frozen components

- UVLM bound circulation、三维downwash、induced drag和added mass；
- v4b paired-LDVM求解器、LESP来源假设及其component-resolved 2D→3D transient projection；
- 三论文几何、运动、GT、滤波、分辨率与评分；
- ULLT仅为attached oracle；
- Baik仍用v4b相同free-tip surrogate，apparatus adapter不计入v5a收益。

## Force ledger

对条带 (i)：

\[
F_i^{v5a}=F_i^{UVLM}+R_i^{eq}+P_{LDVM}(R_i^{tr,2D}).
\]

### Equilibrium path

从UVLM当前诱导速度得到`V_2D`和`alpha_eff`。在同一二维截面、同一`alpha_eff/Re`下：

\[
R_{N,i}^{eq}=q_iS_i(C_{N,eq}^{2D}-C_{N,att}^{2D}),
\quad
R_{D,i}^{eq}=q_iS_i(C_{D,eq}^{section}-C_{D,baseline}^{section}).
\]

该路径直接按条带面积和局部法/阻方向积分：

- 不使用LDVM的`g/g²/added-mass/suction`增益；
- 不修改UVLM induced drag；
- `C_D_baseline`在manifest逐benchmark明确，profile drag一次且仅一次；
- attached branch上两个差量均严格为零。

### Transient path

运行相同kinematics、TE wake、core和step的LESP/attached LDVM pair：

\[
r_i^n=C_i^{LDVM,LESP,n}-C_i^{LDVM,attached,n}.
\]

定义局部对流增量和指数状态：

\[
\Delta\chi_i^n=\max(|V_{2D,i}^n|,10^{-6}U_{ref})\Delta t/c_i,
\]

\[
m_i^{n+1}=m_i^n+(1-e^{-\Delta\chi_i^n/\lambda_\tau})(r_i^n-m_i^n),
\quad
R_i^{tr,2D}=q_iS_i(r_i^n-m_i^n).
\]

只对该transient residual使用冻结的v4b `P_LDVM` component projection；其circulatory/noncirculatory/nonlinear/suction的`g/g²`账本不变。`R_eq`绝不经过这个projection。

`lambda_tau=1.0`在评分前冻结为development hypothesis；0.5和2.0只做敏感性，不选优。`m0=0`，运行至连续两周期状态与均值差均<`1e-4`后才评分。

极限：

- paired LDVM不分离时`r=m=0`；
- 恒定`r`时`m→r`，transient项消失；
- `lambda→0`时transient项消失；
- `lambda→∞,m0=0`时恢复raw paired discrepancy；
- 只有`r=m=R_eq=0`同时成立时才声称逐位退化到attached UVLM。

## Ownership and conservation claims

| Item | Owner / claim |
|---|---|
| finite-wing bound/downwash/induced drag | UVLM |
| added mass/noncirculatory load | UVLM |
| 2D equilibrium saturation/profile residual | direct section-to-strip path |
| transient LEV/suction discrepancy | convective high-pass + frozen LDVM projection |
| UVLM circulation conservation | UVLM regression only |
| separated/attached LDVM circulation conservation | each 2D solver regression separately |
| combined force | exact component-sum ledger |

v5a不声称UVLM和LDVM构成共同Kelvin circulation system。若后续v5b启动，LDVM只提供shedding law，共享wake才成为唯一动态力来源。

## Source parameters and no-fit rule

- Yang `Lcrit=sin(5°)`沿用公开分离角映射假设；
- Figure14 `Lcrit=sin(CLmax/CLa)=0.2393`，Cd0=.057只加一次，velocity/incidence reference均为0.75c；
- Baik主线0.11，0.19只作来源冲突敏感性；
- 禁止case-ID、逐攻角系数、目标残差拟合和从0.5/1/2中挑最佳lambda。

## Promotion gates

- **Yang**：L/D MAE≤`4.327/2.512 gf`；各点恶化≤0.4gf；5°/10°L与15°D改善，20°/25°饱和保留。
- **Figure14**：14-marker RMSE≤0.0230；15°≤0.02268；25°≤0.02284；12-condition RMSE<0.02751；单点恶化≤0.0112。
- **Baik**：filtered CL/CD macro各改善≥5%；8/8 case-channel不退化；W2 Q1和W3 Q1–Q2 CL改善；任何象限恶化≤5%。
- **Attached limit**：Figure11 CL/CD RMSE退化≤2%，并逐项证明`r=m=R_eq=0`。
- **Numerics**：module-off逐位退化；force-ledger relative residual<1e-12；time/chord/span/wake单因素加密后headline变化<5%。

只有`equilibrium-only`、`transient-only`、`full v5a`三个必要消融。

## v5b go/no-go

若v5a通过owner gates但Baik仍有>5% wake-retention变化，且残差与LEV对流龄期系统相关，则启动2–4周v5b。v5b中禁止继续添加LDVM force discrepancy；LDVM只提供onset/shedding constraint，shared TE/LE wake通过UVLM AIC产生唯一动态力。

## Blind generalization

冻结v5a commit、provider表、GT hash、runner和失败门后才选第四篇。其section provider规则也先冻结：只接受已有provider覆盖，或在查看载荷前由公开静态polar/分离角唯一确定；不闭合则fail-closed。盲测门为主CL/CD至少一项改善≥5%，另一项恶化≤2%，论文明确子组恶化≤5%。失败必须保留并终止泛化声明。

## Timeline

- v5a与单元测试：2–4天；
- smoke/ledger消融：1–2天；
- full/单因素敏感性：2–4天；
- v5b仅在go/no-go后另计2–4周。

