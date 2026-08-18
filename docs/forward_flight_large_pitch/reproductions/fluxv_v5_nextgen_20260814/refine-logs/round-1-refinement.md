# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Anchor Check

- **Original bottleneck**：v4b 的 global load blend 让 equilibrium、transient LEV 和 attached flow 的所有权不清，导致 Yang 中低迎角退化、Figure 14 的 25°子组失效和 Baik 局部相位退化。
- **Why the revised method still addresses it**：v5a 只重构 equilibrium/transient 的条带级所有权，恰好对应三论文共同缺陷；不再把长期 shared-wake 和 apparatus solver 当首轮必要条件。
- **Suggestions rejected as drift**：本轮不实现完整三维 shared wake、通用截面阈值回归或 Baik 壁面求解器；这些只有在 v5a 残差给出明确证据后才启动。

## Simplicity Check

- **Dominant contribution**：一个来源约束的局部 LESP owner，把 equilibrium section residual 与 transient LDVM discrepancy 排他分账。
- **Components removed or merged**：删除生产 ULLT owner、四状态机、global persistence、shared-wake feedback、variable-rate shedding、动态 edge switching 和 apparatus adapter。
- **Why this is the smallest adequate route**：它只改变 v4b 已被三篇证据直接否定的 load ownership，不改变 UVLM 求解器或新增另一套完整载荷。

## Changes Made

1. 把原“大一统 v5”拆成 v5a 和条件触发的 v5b；当前提案只承诺 v5a。
2. 用可执行的一阶对流平衡移除器替代未定义的 `LDVM equilibrium solution`。
3. ULLT 仅作为 Figure 11 attached oracle 和回归地标，不进入生产力通道。
4. Baik 主比较保持 v4b/v5a 相同的 free-tip surrogate；apparatus adapter 若开发，必须公平地同时用于两者并单列结果。
5. 明确列出仅有的新增无量纲参数、来源与敏感性，不允许从三论文真值选择。

## Revised Proposal

# Research Proposal: FluxV v5a — 条带级 equilibrium–transient 排他载荷所有权

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Technical Gap

v4b 的主要共同缺陷不是“分离修正太弱”，而是 equilibrium 和 transient 载荷混在两个完整模型中：

- Yang 的 polar 已成功修复20°/25°饱和和分离阻力；完整 LDVM 差量又在六个攻角全部扣除升力，使0°–15°退化。
- Figure 14 的 LDVM 差量显著改善15°组，却使25°组略差；其中无 shedding 条件没有可靠 attached fallback。
- Baik 的 global persistence 几乎恒定在0.76–0.78，无法识别 W2 Q1、W3 Q1–Q2 的局部相位失败。

因此首轮只需把两类分离载荷变成**条带级、因果、互斥的 residual**，无需先改 UVLM wake topology。

## Method Thesis

> 保留 UVLM 为唯一有限翼和非循环载荷骨架；用一个明确的 equilibrium section residual 修正稳态饱和/型阻，再用 LESP 触发且经对流平衡移除的 paired-LDVM transient residual 表示动态超调和吸力变化，从而删除 v4b 的 global persistence 和准稳态双计。

## Contribution Focus

- **Dominant contribution**：条带级 equilibrium–transient 排他载荷账本。
- **Supporting contribution**：完整 module-off / constant-incidence / no-shedding 极限及三论文 Pareto gate。
- **Explicit non-contributions**：v5a 不是三维材料 LEV wake；不引入 ULLT 生产载荷；不建立 Baik apparatus solver；不声称通用 (L_{crit}) 函数已解决。

## Proposed Method

### Frozen backbone

- 原 FluxV/Ptera prescribed-ring UVLM bound solve、有限翼诱导速度及 noncirculatory/added-mass load；
- 当前 clean-room paired LDVM section solver；
- 现有 section/Re-specific `Lcrit` provenance table；
- 三论文冻结 kinematics、geometry、GT 和评分。

### Strip-local force ledger

在条带局部法向/切向坐标先分账，再投影到全局并积分：

\[
F_i^{v5a}=F_i^{UVLM}+R_i^{eq}+a_i\,R_i^{tr}.
\]

#### 1. Equilibrium residual

\[
R_i^{eq}=q_iS_i
\left[
(C_{N,eq}-C_{N,att})\hat n_i+
(C_{D,eq}-C_{D,owned})\hat d_i
\right].
\]

- `C_N,att` 是与 UVLM 小迎角极限一致的有限翼 attached slope；
- `C_N,eq` 和 `C_D,eq` 来自不读取目标载荷的 source-defined/analytic section polar；
- `C_D,owned` 明确扣除已经在 source ledger 中单独加入的 profile drag，保证 profile/skin drag 一次且仅一次；
- equilibrium residual 不修改 UVLM induced drag；
- 在小迎角 attached 区，`R_eq=0`；恒定大迎角只保留该项。

#### 2. Raw transient LDVM discrepancy

同一运动、同一 TE wake、同一数值参数运行两套 section state：

\[
r_i=C_i^{LDVM,LESP}-C_i^{LDVM,attached}.
\]

这仍含慢变 equilibrium 成分，因此不直接加到总载荷。

#### 3. Causal equilibrium removal

在条带法向/切向系数上维护一个低通状态：

\[
\tau_i\dot m_i=r_i-m_i,
\qquad
R_i^{tr}=q_iS_i(r_i-m_i),
\qquad
\tau_i=\lambda_\tau c_i/|V_{2D,i}|.
\]

- `λτ=1.0` 在查看三论文 v5a 输出前冻结，表示一个局部对流时间；
- 只做预注册 `0.5/1/2` 敏感性，不以其中最优值替换主结果；
- constant residual 时 `R_tr→0`，消除与 equilibrium polar 的长期双计；
- 在运动条带坐标做低通，即使系数残差周期均值接近零，随姿态投影后仍可产生物理周期平均推力/功率效应。

#### 4. Local causal owner

`a_i` 不再是全局 persistence，而直接由当前及过去 LDVM 材料 LEV 状态确定：

\[
a_i(t)=\mathrm{clip}\left(
\frac{|\Gamma_{LEV,active,i}|}
{|\Gamma_{bound,i}|+\epsilon},0,1\right),
\]

并在没有活跃 LEV 时以同一个对流时间衰减到零。为了避免新阈值，首版可直接使用 `a_i=1` whenever paired discrepancy is nonzero，`a_i=0` otherwise；软强度只作后续预注册消融。所有组合发生在条带级，不能先展向平均 owner 再混合总载荷。

### Exact ownership contract

| 物理项 | v5a owner |
|---|---|
| bound circulation / finite-wing induced effects | UVLM |
| added mass / noncirculatory pressure | UVLM |
| equilibrium lift saturation | `R_eq` replacement |
| section/profile drag | source ledger, once |
| transient LEV hysteresis / suction change | high-passed paired LDVM residual |
| attached-flow oracle | ULLT evaluation only; no force addition |

### Parameter table

首轮不声称统一 `Lcrit(Re,t/c,rLE/c)` 已建立；只复用 v4b 已公开且未由本轮真值拟合的来源假设：

| Benchmark | Main `Lcrit` source | Status |
|---|---|---|
| Yang | `sin(5°)` from published separation-angle mapping | development hypothesis |
| Figure 14 | `sin(CLmax/CLa)=0.2393` from Scherer static data | source-derived hypothesis |
| Baik | `0.11` Ramesh flat-plate source; `0.19` source-conflict sensitivity | cross-Re/thickness transfer |

唯一新增主参数是 `λτ=1.0`。`ε` 仅为数值尺度，以机器精度和局部 circulation norm 定义，不作为物理拟合量。

### Correctness fixes before scoring

1. Figure 14 的 profile velocity 与 incidence reference 都固定为论文的 `0.75c`；
2. Baik 主比较继续使用 v4b/v5a 相同的 free-tip surrogate；apparatus boundary 不进入通用核增益；
3. Yang 只评分周期均值，不使用内部相位曲线调参；
4. no-LESP、`R_eq=0`、`λτ→0/∞` 和 module-off 极限都建立单元测试。

## v5b go/no-go boundary

v5b shared TE/LE wake 不属于本轮方法。仅当 v5a 完成后同时满足以下诊断才启动：

- Baik headline 对 wake-retention 仍变化超过5%；
- 残差与 LEV convection phase/age 系统相关；
- Yang/Figure 14 的剩余误差不能由 owner/ledger 修复解释。

若启动 v5b，LDVM 只提供 onset/shedding law，不再提供 force discrepancy；shared wake 通过 UVLM AIC 产生唯一动态力，避免三重双计。开发预算按2–4周估计，不再宣称3–5天完成。

## Claim-Driven Validation

### Claim 1: v5a 的排他 ledger 在三论文上形成 Pareto 改进

- **Yang hard gate**：L/D MAE ≤ `4.327/2.512 gf`（至少比v4b各改善5%）；每个攻角恶化≤0.4 gf；必须改善5°、10°升力和15°阻力，保留20°/25°饱和。
- **Figure 14 hard gate**：14-marker RMSE≤0.0230；15°组≤0.02268；25°组≤0.02284；12-condition RMSE<v4b 0.02751；单点恶化≤0.0112。
- **Baik hard gate**：filtered CL/CD macro RMSE分别比v4b至少改善5%；8/8 case-channel不退化；W2 Q1和W3 Q1–Q2 CL必须改善；任一相位象限恶化≤5%。
- **Attached limit**：Figure 11 的 CL/CD RMSE 相对现有 attached result退化≤2%。

### Claim 2: 改善来自 owner/ledger 而非额外自由度

三个必须消融：`equilibrium-only`、`transient-only`、`full v5a`。另比较 global v4b owner 与 strip-local v5a，记录参数数量、ledger residual及极限退化。

## Failure Interpretation

- 若 Yang改善而Figure 14/Baik不改善：equilibrium residual有效，transient owner仍错误；禁止调polar去补动态误差。
- 若 Figure 14改善而Yang中低迎角退化：transient normal-force residual泄漏，需重审section ledger，不调`Lcrit`。
- 若 Baik仍对wake retention高度敏感：v5a达到边界，按go/no-go启动v5b。
- 若三者无法在同一`λτ`方向上改善：该一阶状态假设被证伪，不得发布“统一算法”。

## Generalization Protocol

三篇都已被开发过程查看，只能称 retrospective multi-paper robustness。v5a代码、参数表、GT哈希、评分脚本和失败判据冻结到一个 commit 后，再选择第四篇尚未查看载荷的实验做一次真正盲测；盲测失败也必须保留并报告，不能回头调整v5a。

