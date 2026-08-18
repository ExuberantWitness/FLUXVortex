# Experiment Plan

**Problem**：验证 FluxV v5a 能否在三篇开发论文上形成不牺牲任何一篇的 Pareto 改进。  
**Method Thesis**：UVLM + direct equilibrium section residual + convectively high-passed paired-LDVM transient residual。  
**Date**：2026-08-14

## Claim Map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1 三论文 Pareto 改进 | 三篇全部通过预注册 hard gate | B2, B3 |
| C2 改善来自排他 ledger | three-way ablation + exact limits | B1, B2 |
| Anti-claim：只靠更多参数/后验选择 | lambda主值预冻、敏感性不选优、无case-ID | B1, B4 |

## Experiment Blocks

### B0: Formula and ledger sanity — MUST

- module-off逐位等于v4b输入；
- `qS`只乘一次；equilibrium与transient projection身份单测；
- `r=m=R_eq=0` attached reduction；
- constant/no-shed/lambda极限；
- UVLM和两套LDVM各自Kelvin regression；force-ledger residual<1e-12。

### B1: Frozen-history representative smoke — MUST

使用已有冻结历史，先不重跑昂贵UVLM：

- Yang：5°、15°、25°；
- Figure14：15°/45°、25°/45°、25°/90°、25°/105°；
- Baik：W2、W3。

对比 `v4b / equilibrium-only / transient-only / full v5a`。任何NaN、相位突跳、ledger错误或两篇以上主方向退化即停止。

### B2: Three-paper full matrix — MUST

- Yang：6 AoA的周期均值L/D及逐点误差；
- Figure14：14 observations、12 unique conditions、15°/25°分组；
- Baik：W1–W4 raw与1Hz filtered CL/CD、逐case及Q1–Q4。

不做相位、幅值或offset拟合。主结果只使用`lambda_tau=1.0`。

### B3: One-factor numerical sensitivity — MUST

逐一改变：time step、chord grid、span grid、LDVM wake retention、core radius。每次只变一个因素；headline变化须<5%。lambda=0.5/2只报告敏感性，不改变canonical。

### B4: Frozen blind experiment — MUST for generalization claim

在代码和provider冻结后选择第四篇未查看载荷的实验；先记录几何、section provider、Lcrit来源、GT hash和pass gate，再解封载荷评分。

## Promotion Gates

| Benchmark | Gate |
|---|---|
| Yang | L/D MAE≤4.327/2.512 gf；单点退化≤0.4 gf；指定中低AoA修复 |
| Figure14 | overall≤0.0230；15°≤0.02268；25°≤0.02284；12-condition<v4b |
| Baik | CL/CD macro各改善≥5%；8/8不退化；W2/W3指定象限修复 |
| Figure11 | attached CL/CD RMSE退化≤2% |
| Blind | 一项改善≥5%，另一项退化≤2%，子组退化≤5% |

## Run Order

| Milestone | Runs | Decision gate | Estimate |
|---|---|---|---|
| M0 | B0 unit/golden tests | all exact identities pass | 0.5–1 day |
| M1 | B1 representative smoke | no structural regression | 1–2 days |
| M2 | B2 full matrix | all three paper gates pass | 2–4 days |
| M3 | B3 sensitivity | headline changes<5% | 1–3 days |
| M4 | freeze + B4 blind | blind gate pass | 1–3 days plus data prep |

## v5b Stop/Go

只有 M2 通过且 M3 证明 Baik 对 wake-retention 仍>5%敏感、残差又与 LEV 对流龄期一致时，才启动 v5b。否则停止扩展，不为了论文堆叠 shared-wake 模块。

## Compute Budget

- GPU：0；
- CPU：smoke约小时级，full+敏感性预计数十CPU小时；
- 峰值内存：沿用现有串行runner，约1–2 GB/工况；
- 最大风险：provider/Lcrit可迁移性、Baik尾迹未收敛、Yang缺LDS时序。

