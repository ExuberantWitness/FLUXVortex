# Research Proposal: FluxV v5 — 守恒、状态分辨的 UVLM–LEV 统一算法

## Problem Anchor

- **Bottom-line problem**：在保留有限翼 UVLM 的前提下，把 FluxV v4b 改进为一条统一的非定常气动算法，使其在 Yang 2025 刚性扑翼、Scherer/Izraelevitz Figure 14 俯仰–升沉实验和 Baik 2012 W1–W4 三篇论文上均比 v4b 更准确，而不是以牺牲某一论文换取总体平均分。
- **Must-solve bottleneck**：v4b 仍用全局 persistence 在“UVLM+独立二维 LDVM 差量”和“全攻角极线”之间混合；材料 LEV 不反馈 UVLM，尾迹账本不统一，分离阈值依赖未经同域验证的常数，Baik 的壁面/端板边界也尚未建模。
- **Non-goals**：不以逐 case/逐攻角残差表拟合实验；不把三篇论文的参数分别调到最好；不以 CFD 替代低阶模型；不把 PLEV、LDVM、极线和 ULLT 四套完整载荷相加；不对 Yang 未公开的相位载荷作精度声明。
- **Constraints**：保留现有 UVLM 几何和有限翼骨架；只使用论文公开几何、运动、Reynolds 数、截面和独立来源的静态/LESP 信息；同一代码路径覆盖三论文；所有载荷分量必须可审计并避免双计；Baik 的装置边界只能作为显式可关闭的 apparatus adapter。
- **Success condition**：三论文冻结评分均严格优于 v4b，关键子工况不退化；无 LEV 触发时精确退化为附着流解；Kelvin、功率、符号和载荷总账闭合；网格、时间步、涡核与尾迹保留敏感性通过；冻结后再在第四篇未参与开发的实验上盲测通过。

## Technical Gap

v4b 已证明 Ramesh 式 LESP/LDVM 差量有价值，但现有组合方式不是统一物理模型：

1. **Yang 2025**：v4b 升/阻力 MAE 为 `4.555/2.644 gf`，明显优于 old FluxV，却逊于 v1 的 `3.952/2.063 gf`。0–15°升力全部比 old 更差，20–25°的大幅修复掩盖了中低迎角退化。独立 LDVM 的周期平均升力差量在六个攻角全部为负，说明它同时扣除了 polar 已经拥有的准稳态升力饱和。
2. **Figure 14**：总体 RMSE `0.02595` 很好，但 25°分组没有稳健改善；当前 LESP 门在部分高速旋转工况不触发，而论文归因恰是 LEV/前缘吸力损失。无 LEV 时也没有一致地回到已验证的一状态 ULLT attached owner。此外，profile drag 的局部速度参考必须使用论文的 3/4 弦而非旧的 1/4 弦路径。
3. **Baik W1–W4**：v4b 的 CL/CD 宏平均 RMSE 为 `0.6575/0.3452`，仅比 old 改善 `5.4%/15.3%`。全局 persistence 在所有工况与相位近乎常数，不能识别附着、增长、脱落、反转和恢复。W2 尾迹从 0.50 延长到 0.75 周期带来的 CL 变化大于 v4b 的净改进，说明硬截断材料尾迹尚未收敛。实验是壁面/端板准二维装置，当前却用自由翼尖、零厚度有限翼替代。

因此，缺失的不是另一个经验增益，而是三个共同机制：**局部分离状态、共享守恒尾迹、互斥载荷所有权**。

## Method Thesis

- **One-sentence thesis**：FluxV v5 以 UVLM 为唯一有限翼环量和非循环载荷骨架，用条带级 LESP 状态机控制一个与 TE 尾迹共享 Kelvin 账本的材料 LEV row，并以“动态分离相对同迎角平衡态的差量”替换而非叠加准稳态载荷，从而统一稳态饱和、动态超调和尾迹记忆。
- **Why this is the smallest adequate intervention**：只新增一个局部分离状态和一个共享 LEV wake state；ULLT、静态极线和 LDVM 不再作为并列完整模型，而分别提供 attached memory、equilibrium manifold 和 dynamic discrepancy。
- **Why no foundation-model primitive is used**：本问题是守恒、因果和边界条件控制的确定性气动力问题。LLM/VLM/RL 不提供更自然的物理接口，强行加入只会增加不可审计参数和泛化风险。

## Contribution Focus

- **Dominant contribution**：守恒、状态分辨、互斥所有权的 UVLM–LEV 耦合。
- **Supporting contribution**：把实验装置边界和截面/Re 参数来源从通用气动核中分离为可关闭、可审计的 adapter/parameter provider。
- **Explicit non-contributions**：不声称完整复现 Yang 作者 PLEV/AWS；不声称 Ramesh 二维 LDVM 本身是三维模型；不把三篇开发论文当真正 held-out 泛化证据。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**：现有 Ptera/FluxV UVLM 网格、运动学、bound-circulation solve、added-mass/非循环载荷、已有 one-state ULLT 地标、clean-room LDVM Fourier/LESP primitive、三论文数据与评分 runner。
- **New components**：
  1. `LocalSeparationState`：每条带的 attached / LEV-growth / separated / recovery 因果状态；
  2. `ConservativeLEWakeState`：展向 LEV row、TE temporal edge、涡龄、核增长和 Kelvin 删除账本。
- **Intentionally excluded**：case-ID 网络、观测残差拟合、独立 PLEV 与 LDVM 同时启用、完整二维 LDVM 力直接相加、全局 persistence 标量、固定步数尾迹硬删除。

### System Overview

```text
moving wing mesh + U∞ + section/Re/boundary metadata
        |
        v
UVLM bound solve using previous TE/LE material wake induction
        |
        +--> noncirculatory / added-mass ledger (UVLM sole owner)
        |
        v
strip local normalwash -> A0/LESP + equilibrium polar state
        |
        v
causal separation state machine
 attached | growth | separated | recovery
        |
        v
joint TE/LE newborn circulation solve
 [no-penetration + Kelvin + LESP/shedding-rate constraint]
        |
        v
commit TE/LE material wake -> convect / core growth / conservative truncation
        |
        v
single force ledger: attached circulation + dynamic LEV discrepancy
                    + noncirculatory + one profile/separation drag owner
```

### Core Mechanism A: 条带级平衡态–动态态渐近匹配

对每个展向条带 (i)，不再直接使用完整的 `LDVM separated − LDVM attached`。定义：

\[
\Delta F_{i,\mathrm{dyn}}
=F_{i,\mathrm{LDVM}}(\text{history})
-F_{i,\mathrm{LDVM,eq}}(\alpha_i,Re_i,\text{section}) .
\]

最终条带载荷为：

\[
F_i=F_{i,\mathrm{UVLM}}
+R_{i,\mathrm{eq\ polar}}
+s_i\,\Delta F_{i,\mathrm{dyn}},
\]

其中 (R_{eq\ polar}) 只替换 UVLM 的准稳态线性饱和/型阻账本；(s_i) 是局部因果状态，而非积分后总载荷的全局混合权重。

强制三个极限：

1. 未触发 LESP：(s_i=0)，严格退化到附着 UVLM/ULLT；
2. 恒定大迎角：动态差量趋零，只保留 equilibrium full-angle closure；
3. 快速往复：只保留相对同迎角平衡态的动态超调、滞回和吸力变化，避免与 polar 重复扣升力/加阻力。

### Core Mechanism B: 局部 LESP/LEV 状态机

状态输入仅为局部物理量：

- (A_0/L_{crit}) 及其时间导数；
- 局部 3/4 弦 normalwash、迎角变化方向与 pitch-rate；
- 距上次脱落的对流时间和最近 LEV 涡龄；
- (Re_i,t/c,r_{LE}/c,) section family；
- 动态 upstream-edge role，处理换向时几何 LE/TE 角色切换。

状态转移：

```text
attached --onset--> LEV_growth --feed/convect--> separated
   ^                                           |
   +---------------- recovery/reversal --------+
```

首版采用源可追溯的常 LESP 作为 `parameter provider`，但接口从一开始冻结为：

\[
L_{crit}=\mathcal{F}(Re,t/c,r_{LE}/c,\text{section family}) .
\]

若截面阈值来源不闭合则 fail-closed，不允许从三篇载荷真值反标。第二个消融分支采用 Martínez-Carmena 的 variable-rate shedding criterion，和 constant-cap LDVM 互斥比较，不叠加。

### Core Mechanism C: 共享 Kelvin 账本与材料尾迹反馈

- 当前步 UVLM 只看已存在的前排 wake history；
- load 后由 Kelvin/LESP 约束求 temporal TE edge 与 newborn LEV circulation；
- 写入下一状态，保持 Ptera 时间层因果一致；
- LEV/TEV 诱导速度从下一步开始进入 UVLM AIC、局部 normalwash 和载荷；
- 尾迹按对流龄期/长度保留，涡核随龄增长；截断前先把被删除环量写入守恒账本；
- 禁止现有单向 VPM 粒子被误当作 temporal shed circulation。

### Load Ownership Contract

| 载荷分量 | 唯一所有者 |
|---|---|
| 有限翼 bound circulation / induced downwash | UVLM |
| added mass / noncirculatory pressure | UVLM |
| attached wake memory | ULLT state applied to UVLM circulatory AC only |
| 准稳态高迎角饱和 | equilibrium polar replacement |
| 动态 LEV 超调/滞回/吸力变化 | state-gated LDVM/LEV discrepancy |
| profile drag | section/Re provider，一次且仅一次 |
| apparatus wall/endplate | boundary operator，不计入通用核 |

每步输出完整 ledger，并断言分量和等于最终力、模块关闭逐位退化、同一物理项没有两个 owner。

### Boundary and Section Adapters

- **Yang**：薄木板/薄膜截面；名义四连杆只作为公开输入重建，LDS 缺失进入运动不确定性说明。
- **Figure 14**：NACA 63A015、3/4 弦枢轴；profile velocity 与 incidence 均按源定义的 3/4 弦；`Cd0=0.057` 主线、`0.027` 来源敏感性。
- **Baik**：6.25% 厚圆头圆尾板、壁面和端板准二维实验。新增 image/endplate boundary operator；自由翼尖 AR=7.895 保留为 secondary diagnostic，不能作为 apparatus fidelity 主结果。

这些 adapter 只能读取几何、Re 和装置信息，不能读取实验载荷或 `case_id`。

### Failure Modes and Diagnostics

- **阈值不可迁移**：以 source provenance + leave-one-paper-out 参数冻结检测；不闭合则 fail-closed。
- **尾迹仍未收敛**：逐一改变时间步、涡核、保留对流长度和 span/chord 网格；禁止同时改变多个因素后称收敛。
- **双计载荷**：ledger identity、module-off exact reduction、常迎角/小振幅解析极限。
- **动态边角色错误**：输出每条带 upstream-edge role；换向低置信区关闭单边 LEV birth。
- **Yang 无相位真值**：只评周期均值；相位只检查闭合、连续、功率符号和无数值尖峰。
- **Baik apparatus adapter 过拟合**：adapter 参数完全由水槽/端板几何决定，并和自由翼尖结果并列报告。

## Novelty and Elegance Argument

最接近的现有组件各自只覆盖问题的一部分：UVLM 提供三维有限翼但缺分离；ULLT 提供附着流记忆；Ramesh LDVM 提供二维 LESP/LEV；静态极线提供平衡态饱和。v5 的贡献不是再增加一个模型，而是把它们压缩为一个守恒状态空间中的互斥角色：**一个 bound solve、一个分离状态、一个共享 wake、一个 force ledger**。这比 v4b 的 load-level blend 更小、更可证伪，也更容易解释跨论文成功或失败的原因。

## Claim-Driven Validation Sketch

### Claim 1: 统一局部分离状态可在三论文上形成 Pareto 改进

- **Minimal experiment**：同一冻结 v5 参数提供规则运行 Yang 六攻角、Figure 14 全部条件、Baik W1–W4。
- **Baselines / ablations**：old FluxV、v4b、equilibrium-only、dynamic-only、无 shared-wake、constant-cap 与 variable-rate 二选一。
- **Metric**：Yang L/D MAE与逐角误差；Figure 14 14-observation及12-condition RMSE；Baik filtered/raw CL/CD macro与逐case/相位象限RMSE。
- **Expected evidence**：三论文主指标均严格优于 v4b，且关键分组不退化。

### Claim 2: 改善来自守恒状态与载荷所有权，而非更多参数

- **Minimal experiment**：保持相同阈值/极线，比较 v4b global blend、v5 local state、v5 local state+shared wake；做 module-off 和平衡态极限。
- **Metric**：误差、Kelvin residual、ledger residual、wake-retention敏感性、参数数量和运行成本。
- **Expected evidence**：local-state 先修复中低迎角/相位退化，shared-wake 再降低 Baik 尾迹敏感性；删除任一核心机制会在预注册子工况失败。

## Experiment Handoff Inputs

- **Must-prove claims**：三论文 Pareto 改进；无触发时退化；shared wake/owner contract 是增益来源。
- **Must-run ablations**：v4b；equilibrium-only；LDVM transient-only；local state without feedback；full shared wake；constant-cap vs variable-rate。
- **Critical datasets / metrics**：Yang 6×L/D means；Figure 14 14/12 mean CT；Baik W1–W4 raw+1Hz filtered phase CL/CD。
- **Highest-risk assumptions**：截面/Re 的 LESP provider；Baik wall/endplate operator；material-wake数值收敛；Yang nominal four-bar 与 LDS 差异。

## Compute & Timeline Estimate

- **Stage A, ledger/local-state prototype**：1–2天；优先重放冻结历史，不改 UVLM。
- **Stage B, shared-wake smoke**：3–5天；先单条带和 Ramesh golden case，再三论文代表工况。
- **Stage C, full matrix/sensitivity**：2–4天；CPU 数十小时，串行峰值内存约1–2 GB/工况。
- **Stage D, frozen external confirmation**：1–2天数据重建 + 正式运行；不得回头调三论文参数。

