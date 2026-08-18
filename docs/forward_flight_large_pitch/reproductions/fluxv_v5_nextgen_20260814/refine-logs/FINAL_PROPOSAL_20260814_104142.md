# Research Proposal: FluxV v5a

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 混合完整 UVLM/LDVM 和全攻角极线载荷，equilibrium 与 transient 分离载荷所有权重叠。
- **Non-goals**：不做 case-ID/逐工况残差拟合；不把 ULLT、polar、LDVM、PLEV 四套完整载荷相加；不以 CFD 替代低阶模型；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留 UVLM；同一代码路径覆盖三篇；只使用公开几何、运动、Re、截面及独立来源参数；所有载荷分量可审计且不双计。
- **Success condition**：三篇冻结评分均严格优于 v4b、关键子组不退化、attached 极限与数值账本通过，并在冻结后的第四篇未查看实验上通过盲测。

## Method Thesis

FluxV v5a 保留 UVLM 为唯一有限翼与非循环载荷骨架，用直接的二维 equilibrium–attached 截面差量修正稳态饱和和型阻，再用 paired-LDVM discrepancy 的局部对流高通部分表示瞬态 LEV、滞回与吸力变化；两条 residual 在定义和空间映射上互斥，因此删除 v4b 的 global persistence。

## Frozen Components

- UVLM bound circulation、三维 induced downwash、induced drag 和 added mass；
- v4b paired-LDVM section solver、LESP 来源假设及 component-resolved transient projection；
- 三篇论文的几何、运动、GT、滤波、数值分辨率和评分；
- ULLT 只作为 attached-flow oracle；
- Baik 使用与 v4b 相同的 free-tip surrogate，装置边界不计入 v5a 通用核收益。

## New State and Parameter

- 每个条带仅新增二分量低通状态 `m_N,m_D`；
- 唯一新物理假设为 `lambda_tau=1.0` 个局部对流时间；
- `lambda_tau=0.5/2.0` 只作预注册敏感性，不从结果中选优。

## Strip-local Force Ledger

对条带 (i)：

\[
F_i^{v5a}=F_i^{UVLM}+R_i^{eq}+P_{LDVM}(R_i^{tr,2D}).
\]

### 1. Equilibrium path

从 UVLM 当前诱导速度得到局部二维速度和有效迎角。对同一二维截面、同一 `alpha_eff/Re`：

\[
R_{N,i}^{eq}=q_iS_i(C_{N,eq}^{2D}-C_{N,att}^{2D}),
\]

\[
R_{D,i}^{eq}=q_iS_i(C_{D,eq}^{section}-C_{D,baseline}^{section}).
\]

约束：

- 直接做 section-to-strip 面积积分；
- 不使用 LDVM 的 `g/g²/added-mass/suction` projection；
- 不修改 UVLM induced drag；
- `C_D_baseline` 在 manifest 中明确，profile drag 一次且仅一次；
- attached branch 上两个 residual 均为零。

### 2. Transient path

使用完全相同的 kinematics、TE wake、core 与时间步运行 LESP/attached LDVM pair：

\[
r_i^n=C_i^{LDVM,LESP,n}-C_i^{LDVM,attached,n}.
\]

局部对流增量：

\[
\Delta\chi_i^n=\max(|V_{2D,i}^n|,10^{-6}U_{ref})\Delta t/c_i.
\]

指数状态更新与 transient residual：

\[
m_i^{n+1}=m_i^n+(1-e^{-\Delta\chi_i^n/\lambda_\tau})(r_i^n-m_i^n),
\]

\[
R_i^{tr,2D}=q_iS_i(r_i^n-m_i^n).
\]

只有该 transient residual 使用冻结的 v4b `P_LDVM`。实现必须选择唯一单位合同：优先先投影无量纲系数、再乘局部 `qS` 一次，单元测试禁止重复乘入动压或面积。

## Exact Limits

- paired LDVM 未分离时 `r=m=0`；
- 恒定 `r` 时 `m→r`，transient 项消失；
- `lambda→0` 时 transient 项消失；
- `lambda→∞,m0=0` 时恢复 raw paired discrepancy；
- 仅当 `r=m=R_eq=0` 同时成立时，逐位退化到 attached UVLM；
- `m0=0`，必须运行到连续两周期状态与载荷均值变化均 `<1e-4` 才评分。

## Load Ownership

| Physical component | Sole owner |
|---|---|
| finite-wing bound circulation/downwash/induced drag | UVLM |
| added mass/noncirculatory pressure | UVLM |
| 2-D equilibrium saturation/profile residual | direct section-to-strip provider |
| transient LEV hysteresis/suction change | convective high-pass paired LDVM |
| attached-flow oracle | ULLT evaluation only |

v5a 只声明：UVLM 自身 Kelvin regression、separated/attached LDVM 各自 Kelvin regression和最终 force-ledger 闭合；不声称三者构成共同 circulation system。

## Source Parameters and No-fit Rule

- Yang：沿用 v4b `Lcrit=sin(5°)` 公开映射假设；
- Figure 14：沿用 `Lcrit=sin(CLmax/CLa)=0.2393`；`Cd0=.057` 只加一次，incidence/profile velocity 都使用 0.75c；
- Baik：主线 0.11，0.19 仅作来源冲突敏感性；
- 禁止 `case_id`、逐攻角系数、观测残差表及从 lambda 敏感性中挑选最佳结果。

## Promotion Gates

- **Yang**：L/D MAE `≤4.327/2.512 gf`；单点恶化≤0.4 gf；5°/10°升力和15°阻力改善，20°/25°饱和保留。
- **Figure 14**：14-marker RMSE≤0.0230；15°组≤0.02268；25°组≤0.02284；12-condition RMSE<0.02751；单点恶化≤0.0112。
- **Baik**：filtered CL/CD macro RMSE 各改善≥5%；8/8 case-channel不退化；W2 Q1和W3 Q1–Q2升力改善；任一象限恶化≤5%。
- **Attached limit**：Figure 11 CL/CD RMSE退化≤2%，且证明 `r=m=R_eq=0`。
- **Numerics**：module-off逐位退化；force-ledger relative residual<1e-12；time/chord/span/wake单因素加密后 headline 变化<5%。

必要消融仅三个：`equilibrium-only`、`transient-only`、`full v5a`。

## v5b Go/No-go

只有 v5a 通过 owner gates、但 Baik 仍有>5% wake-retention变化且残差与 LEV 对流龄期系统相关时，才启动 v5b。v5b 中 LDVM 只提供 onset/shedding constraint；shared TE/LE wake 进入 UVLM AIC 并成为唯一动态力来源，禁止继续添加 LDVM force discrepancy。预计另需2–4周。

## Generalization Protocol

三篇论文只能支持“retrospective multi-paper Pareto improvement”。冻结 v5a commit、provider 表、GT hash、runner和失败门后再选第四篇。其 section provider 也须在查看载荷前由已有表或公开静态 polar/分离角唯一确定，否则 fail-closed。盲测门：主 CL/CD 至少一项改善≥5%，另一项恶化≤2%，任何论文明确子组恶化≤5%；失败必须保留并终止泛化声明。

