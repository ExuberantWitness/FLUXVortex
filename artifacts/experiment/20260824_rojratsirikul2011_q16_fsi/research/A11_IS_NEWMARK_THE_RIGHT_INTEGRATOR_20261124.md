# A11 文献调研：有没有比 Newmark 更适合膜翼 FSI 的时间积分格式？（2026-11-24）

> 起因：用户问"为什么不用预测–校正而用 Newmark"，并要求做文献调研、评估潜在提升。
> 读过**原文**的：Bathe 2007（MIT 开放 PDF，9 页，`/tmp/lit_bathe2007/`）。
> 只有检索摘要、**未读原文**的：Causin/Gerbeau/Nobile 2005、van Brummelen（附加质量不稳定性）。
> 本记录用**我们自己的全长数据**做判据，不以文献论断为准。

## 1. 先澄清两个层次

"Newmark" 是**时间离散格式**，"预测–校正"是**耦合/求解策略**，不互斥。我们三层都在用：

| 层 | 现状 | 位置 |
|---|---|---|
| 时间离散 | Newmark，**β = 1/4，γ = 1/2**（平均加速度法） | `q16_structural_solver.py:541-542` |
| 子步内 | 预测（用 t_n 载荷积分）→ 用预测速度重算气动 → 校正 | `q16_flux_v5m_native_fsi.py`（`predictor` / `result` 两次 `step`） |
| 流固耦合 | 定点迭代 + Aitken 松弛，残差 ≤ 5e-7 后 formal replay 再提交 | 同上 `:743` 起 |

β=1/4、γ=1/2 ⇒ **无数值阻尼**。这一点此前没确认过，现已从代码读出。

## 2. 候选格式逐个评估

| 候选 | 文献说法 | 对我们 |
|---|---|---|
| **显式预测–校正 / 中心差分** | Bathe 2007 原文："only conditionally stable and requires a very small time step … for structural vibration analyses, the use of an implicit technique can be much more effective" | **不适用**。四边夹支薄壳的最高阶频率远高于扑翼频率，显式步长被最高频率锁死 |
| **HHT-α / 广义-α** | 引入数值阻尼抑制高频 | 引入**幅度衰减**，会直接污染要对论文的 `zsd` 与 `St`。未读原文，不定论 |
| **Bathe 复合格式**（梯形 + 三点 Euler 后退，两子步） | 见 §3 | **机制上对口，但我们的数据显示触发的那个病不存在**，见 §4 |
| **能量–动量守恒格式（EM）** | 非线性下严格守恒 | 目标是长期守恒；我们 t*=21 的窗口没有观测到漂移，见 §4 |

## 3. Bathe 2007 复合格式（原文）

- **动机**（摘要）："analyses in which the widely used Newmark time integration procedure is not conserving energy and momentum, and is unstable"；正文："when large deformation dynamic analyses are pursued that require time integrations over relatively long time intervals, the trapezoidal rule can become unstable … the response grows, and energy and momentum are not conserved"。
- **格式**：每步两子步。前半步用梯形法（即 Newmark β=1/4, γ=1/2），后半步用三点 Euler 后退。原文式 (8)-(9)：`U̇ = (U_t − 4U_{t+Δt/2} + 3U_{t+Δt})/Δt`，加速度同理。
- **精度/稳定**："fully implicit second-order accurate"；"remains stable even in large deformation and long time response solutions when the trapezoidal rule fails"。
- **成本**："requires per step about twice the computational effort as the trapezoidal rule … but less steps can frequently be used"。
- **副作用**："second-order accurate scheme with **small amplitude decay** and period elongation"；而 "the trapezoidal rule does not give an amplitude decay"。
- **无参数**："has no parameter to choose or adjust"。
- **它演示梯形法失效的算例**：弹性单摆（大转动、10–400 周期）。**没有膜/壳算例，也没有 FSI 算例。**

## 4. 用我们自己的数据判：触发 Bathe 格式的那个病，我们有没有

`ROJ11_A16_FULL_20261123_CNCONV.json`（legacy，2100 步，t*=21）：

| 步段 | 牛顿迭代 均值 / max | 耦合迭代均值 | 平均 \|zmax\|/c |
|---|---|---|---|
| 1–100 | 49.7 / 60 | 1.970 | 0.01287（启动） |
| 400–500 | 63.8 / 80 | 2.000 | 0.04952 |
| 900–1000 | 63.6 / 80 | 2.000 | 0.04653 |
| 1400–1500 | 60.0 / 60 | 2.000 | 0.04552 |
| 1900–2000 | 60.0 / 60 | 2.000 | 0.04547 |
| 2000–2100 | 60.0 / 60 | 2.000 | 0.04560 |

- **振幅无漂移**：每 400 步一块的平均 |zmax|/c 为 0.0454 / 0.0444 / 0.0455 / 0.0457 / 0.0455，首尾差 **+0.25%**。能量增长会表现为单调上升趋势，没有。
- **牛顿迭代不发散**：全程最大 80（上限 128），**没有一步 ≥ 100**；收敛窗里恒为 60。
- 耦合迭代恒为 2.0，没有出现附加质量导致的不收敛。
- `stationary_window_found = True`。

**所以：在 t*=21 的窗口上，Newmark 没有表现出 Bathe 所针对的能量增长/不稳定。** 换格式的收益在当前窗口里**没有可测的对应物**。

## 5. 一个顺带查清的事实：每子步恒 6 次牛顿迭代

收敛窗里 `structural_newton_iterations` 恒为 60 = 10 子步 × 6。这不是封顶（收敛即 `break` 并返回真实计数，`q16_structural_solver.py:2018-2040`），是**真实值**。含两次 `step`（预测 + 校正）⇒ **每次 `step` 平均 3 次牛顿迭代**。

这说明：**我们的结构求解不是"牛顿迭代很贵"的问题，而是"每次迭代/每次 step 的固定开销很贵"**。S2b 实测 `struct_step` 1849.6 ms/步 ÷ 59.6 次 = 31 ms/次，其中稠密解只占 6 ms。换积分格式（Bathe 两子步）会让 `step` 调用数**翻倍**，按这个分解**预期是变慢**，不是变快。

## 6. 潜在提升的量化

| 方向 | 精度 | 速度 | 稳定 |
|---|---|---|---|
| 换 Bathe 复合格式 | 无可测收益（无能量漂移可修）；**引入小幅衰减**，可能让 zsd/St 变差 | **负**：每步两次 `step`，按 §5 的分解约 **−100%**（变慢一倍） | 仅在更长时窗（t* ≥ 50，计划 L4）或更强非线性时才可能有意义 |
| 换广义-α（加阻尼） | 同上，且引入参数 | 中性 | 抑高频，但我们没有高频问题 |
| 换显式预测–校正 | 取决于步长 | 步长被最高频率锁死，**负** | 有条件稳定 |
| **保持 Newmark，降每次 step 的固定开销** | 不变（逐位） | **正**：这是 S3c / P0-3 已经在做的事 | 不变 |

**结论：在当前问题与当前窗口上，没有比 Newmark(1/4, 1/2) 更合适的格式；换格式的潜在提升是零到负。** 真正的速度空间在"每次 step 约 31 ms 的固定开销"里，不在积分格式里。

## 7. 什么时候该重新考虑

1. **L4（t* ≥ 50）长时窗**：如果届时出现振幅漂移或牛顿迭代数上升，Bathe 格式是对口的候选——那时它是**修一个存在的病**。
2. **更强非线性**（更大变形、E 更低）：Bathe 的失效算例正是大转动。
3. **附加质量不稳定性**：若未来出现耦合迭代不收敛（现在恒为 2.0），该查的是 Causin–Gerbeau–Nobile 的那套分析（**未读原文**），而不是换时间格式。

## 8. 边界

- Bathe 2007 读了原文；HHT-α、广义-α、能量–动量格式**都没读原文**，表中关于它们的说法是通用认识，不是文献核对结果。
- "振幅无漂移"用的是 |zmax|/c 的分块均值，不是严格的能量范数。严格的判据应逐步记录结构能量（应变能 + 动能）与气动做功的收支——payload 里没有这个量，**没测**。
- 牛顿迭代"恒 6 次/子步"只看了 payload 的累加值；逐子步的分布没打印。
- Causin–Gerbeau–Nobile 针对的是不可压 CFD 的分区耦合；我们是势流涡格，附加质量的量级不同，其结论不能直接搬。
