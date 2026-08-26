# FLUX-V5M Mancini 回归事故检讨

日期：2026-08-22
范围：mandatory GPU 三维气动路径，重点为 separated LEV、joint TEV、自由尾迹和
DVM node-owned connected ribbon 的集成。
性质：开发过程复盘，不替代 `verification.md` 的正式精度结论。

## 1. 结论先行

这次“改坏”不是实验参数不合适，也不是 GPU 舍入误差，而是我在推进新气动数据
通路时，没有先冻结跨模块的**状态、时间层和载荷所有权**。局部模块可以各自通过
残差、Kelvin、GPU 和 predictor 一致性测试，但组合后仍然同时存在以下错误：

1. 把“本步是否释放新 LEV”误当成“本步是否处于分离边界状态”；
2. 新生 DVM 涡片进入了本步方程，但粒子/前沿推进仍读取分离前的束缚环量；
3. Ptera 的 `KJ+dGamma` 已经是表面载荷 owner，却又叠加了一次自由+束缚涡冲量力；
4. 更早阶段还临时复用了已有时间细化 NO-GO 的常强度 Hirato 闭合环表示。

因此，问题的本质是**科学对象接错**，其次才是数值实现问题。最严重的流程错误是：
我曾把“机械合同通过”近似当作“论文 CASE 可以运行”，而没有在第一次完整接线前
建立独立的论文曲线非退化门和明确的 owner 表。这个判断不充分，导致错误一路传播
到完整工况，浪费了一次 449 步 full 运行。

当前 Mancini fast/slow 已修复并通过冻结 RMSE 门，但不能据此宣称整个 FLUX-V5M
已经全部修好：Yang、Izraelevitz 尚未迁移到同一 mandatory node-owned 路径，直接
WRK3 粒子自诱导的性能门仍为 FAIL，历史 Q16 长时精确粒子日程门也需要按新时间层
语义更新。

## 2. 事故影响

### 2.1 结果退化轨迹

| 阶段 | fast RMSE(CL) | 结果 | 暴露的问题 |
|---|---:|---|---|
| 初始 mandatory Hirato smoke | `572925.34` | FAIL | 全翼 LEV eligibility 错误关闭，随后环量失控 |
| Hirato 时间层修复后 smoke | `6.18865` | FAIL | 常强度闭合环仍不满足论文精度和时间细化合同 |
| Hirato 时间层修复后 full | `316.16453` | FAIL | 网格/时间细化后严重恶化，峰值 `CL=3043.80` |
| DVM node-ribbon 初接入 | `3.42122` | FAIL | 载荷、边界和运输 owner 尚未闭合 |
| 去除重复冲量载荷 | `2.06526` | FAIL | 载荷方向改善，但 Ptera 分离边界仍未正确耦合 |
| 加入 LESP 行但混用事件状态 | `3.07873` | FAIL | 边界逐步开关，制造 `+24/-19` 假峰 |
| 分离“释放事件/边界状态” | `0.84303` | PASS | fast smoke 非退化门恢复 |
| 最终 fast full | `1.04886` | PASS | 低于冻结门 `1.25532` |
| 最终 slow full | `0.22527` | PASS | 低于冻结门 `0.29509` |

这条轨迹说明，最后的 PASS 不是调 `Lcrit`、core、网格、运动学或评分代码得到的；
决定性变化来自状态语义和所有权修正。

### 2.2 直接成本

- 运行了一次本应由 smoke/细化门提前否决的 449 步 Hirato full；
- 在“线性系统残差很小”的假象下继续诊断了错误的联合未知量结构；
- 先后产生多份失败产物，增加了结果身份和报告维护成本；
- 当前正确 fast full 因直接粒子自诱导耗时 `2810.92 s`，错误路线的试跑成本并不低。

失败产物被保留，不能覆盖或删除；它们是本次根因判断的必要证据。

## 3. 为什么会改坏

### 3.1 第一层：在已知 NO-GO 表示上继续集成

`V5F_M5_REFINEMENT_STOP_REPORT.md` 已经证明，旧常强度闭合环/Eq.7/
pseudovortex 路线在 `80 -> 160 steps/cycle` 时出现至少 `1/dt` 级材料释放增长，
并且相邻展向 strip 不共享节点。这个结果已经把该表示限定为 NO-GO。

我最初为了尽快把 separated LEV、joint TEV、free wake 接入 mandatory GPU runner，
仍把 Hirato 环路线当成过渡生产路径。虽然随后修复了全翼 eligibility、联合方程和
时间层，但这些修复只能消除实现性爆炸，不能使已被时间细化门证伪的离散表示重新
成立。fast smoke `6.19` 到 full `316.16` 的恶化正是已有 NO-GO 在论文 CASE 上的
再次出现。

应有做法：在写 runner 前先查版本与停止报告；已 NO-GO 的对象只能作为失败对照，
不能进入生产晋级链。

### 3.2 第二层：没有先建立跨模块 owner 表

新路径同时包含 DVM source bank、Ptera bound solve、粒子场、自由尾迹和结构载荷
接口。它们都使用“环量”“active”“force”等相似名称，但物理含义不同。我没有
先把每个量的唯一生产者、消费者和时间层写成强制合同，导致局部正确量被错误组合。

正确 owner 关系应为：

| 科学量 | 唯一含义/owner | 合法消费者 |
|---|---|---|
| `cell_active` | DVM 本步新生 LEV 释放事件 | 新生 ribbon 沉积、事件计数 |
| `ptera_separated` | Ptera 预解吸力是否仍超 `Lcrit` | 本步分离边界选择 |
| `pin_active` | `cell_active OR ptera_separated` | Ptera 前缘 LESP 行替换 |
| `gamma_bound` | 本步 coupled solve 的最终物理束缚环量 | 面板状态、粒子/前沿推进、KJ、`dGamma` |
| vortex impulse | 当前 DVM 模式的独立诊断量 | 闭合诊断，不进入生产总力 |
| `KJ+dGamma` | 当前 DVM 模式唯一表面载荷 owner | 气动力输出、Q16 resolved load packet |

没有这张表时，代码中的一个布尔量或旧缓存虽然类型和形状完全正确，仍然可以代表
错误的科学对象。

### 3.3 第三层：把“释放事件”错当“持续状态”

DVM 的 `shed_lev` 回答的是“本步是否产生新材料”，不是“该 strip 是否仍需执行
分离吸力边界”。当某一步没有新释放，但 Ptera 预解 LESP 仍超限时，旧实现取消了
前缘 pin，下一步又重新激活，形成：

`LESP 0.11 -> 0.65 -> 0.11`

这不是物理再附着，而是状态机抖动。由于 Ptera 载荷包含 `dGamma/dt`，布尔状态的
单步跳变被放大成约 `+24/-19` 的虚假升力峰，RMSE 从 `2.06526` 在初次 LESP
耦合后反而恶化到 `3.07873`。

现在代码明确使用：

```python
cell_active = result["shed_lev"][:span_count]
pin_active = cell_active | ptera_separated
```

这次修复的关键不是“把阈值调松”，而是恢复两个不同状态变量的科学定义。

### 3.4 第四层：同一步内存在两个束缚环量真值

新生 ribbon 已经进入本步 Neumann RHS 和 LESP 行，随后得到的 `gamma_bound` 才是
coupled solve 的最终物理状态。旧实现却让粒子和节点前沿继续读取 pre-separation
的 `gamma`。结果是：

- 面板载荷在一个速度场中计算；
- 材料 LEV 在另一个“影子附着流场”中推进；
- predictor 虽然能复制并推进尾迹，但推进的是错误状态。

这类错误不会被“parent 未污染”“fork A/B 一致”发现，因为两个分支可以完全一致
地执行同一个错误。现在 DVM 模式强制粒子运输读取最终 `gamma_bound`；旧 Hirato
路径的冻结行为未被顺手改写。

### 3.5 第五层：同一分离环量被计算了两次载荷

在当前 DVM/Ptera 耦合中，粒子速度已经进入 Neumann RHS，改变最终束缚环量；
Ptera 的 `KJ+dGamma` 随后给出表面力。旧实现又将自由+束缚 vortex impulse 的
时间导数加入总力，相当于让同一分离环量通过两条路径进入载荷，而且冲量路径没有
配套的结构作用点/力矩定义。

这既破坏唯一载荷 owner，也破坏气动—结构功共轭边界。现在 vortex impulse 仍被
计算用于诊断，但 DVM 模式下生产总力只由 `ptera_kj_plus_dgamma` 拥有；Hirato 的
历史 reduced-load 合同保持隔离，避免一次修复无意改变另一条基线。

### 3.6 第六层：测试证明了“自洽”，没有证明“科学正确”

之前已有测试主要覆盖：

- CUDA float64 与无 CPU 数值 fallback；
- Kelvin/Eq.9、Neumann 和 pin 残差；
- monolithic 与 incremental 一致；
- predictor fork 不污染 parent；
- 粒子非空、自由尾迹确实推进。

这些门很重要，但它们只能证明实现按同一合同运行。错误合同也能得到机器精度残差，
错误的 monolithic 和 predictor 也能位级一致。旧 V5F 停止报告已经给出反例：最大
无穿透残差约 `5.68e-14`、Kelvin 残差为零，但时间细化仍明显发散。

我没有尽早加入以下独立判别器：

- 释放事件与持续分离状态的反例序列；
- 载荷 owner 的排他性测试；
- 粒子运输状态必须等于最终 bound state 的身份测试；
- 论文曲线短窗非退化门；
- 时间步细化门。

这是“测试很多仍然改坏”的根本原因。

## 4. 为什么现在的修复可信

当前可信度来自多类独立证据的交叉，而不只是最终 RMSE：

1. **科学参数未调节**：未改变论文几何、运动学、GT、`Lcrit`、core、spacing 或
   评分公式；
2. **状态身份可检查**：DVM 模式下粒子运输状态等于最终 `gamma_bound`；
3. **载荷 owner 排他**：生产冲量计数为零、unresolved impulse 为零，同时保留
   非零诊断冲量；
4. **边界残差闭合**：fast/slow full 最大 LESP pin 残差分别为
   `3.153e-14/1.955e-14`，保留 Neumann 行分别为
   `4.663e-15/2.887e-15`；
5. **事务路径真实推进**：predictor fork 推进真实 DVM source、粒子、frontier 和
   free wake，并保持 parent 不漂移；
6. **论文 CASE 通过**：fast/slow full RMSE 分别为 `1.04886/0.22527`，低于冻结
   门 `1.25532/0.29509`；
7. **回归隔离**：受影响 DVM/Hirato/Q16/GPU-only 测试为 `41 passed`，旧 Hirato
   合同没有被 DVM owner 修复顺手重定义。

但 fast full 相关系数只有 `0.5357`，说明“RMSE 过门”不等于波形已经完全复现；
后续报告必须同时给出 RMSE、相关性、峰值位置和性能状态。

## 5. 我在开发方法上的具体失误

1. **先接线、后定义科学合同。** 应当先写 owner/time-layer 表，再写模块桥接；
2. **版本考古太晚。** 直到联合系统失败后才回看 v5f 原生时间层，而相关 NO-GO
   报告本应在开始前成为硬约束；
3. **把残差 PASS 过度解释为物理 PASS。** 线性代数闭合只是一层门；
4. **让变量名掩盖科学差异。** `active` 同时承担 release event 和 separated state，
   类型系统无法发现这种错误；
5. **验收顺序不严。** 在表示尚未通过 smoke/细化判别时启动了 full；
6. **诊断先围绕数值病态，后检查抽象身份。** 条件数很高是真实症状，但不是全部
   根因；代数凝聚无法修复错误时间层和错误状态机；
7. **没有把载荷—结构边界同时纳入气动修复。** 重复总力即使暂时改善某条曲线，
   也无法定义对应力矩和结构功，不能进入 FSI。

这些不是“模型复杂所以难免”，而是开发门序和抽象纪律没有执行到位。

## 6. 防复发措施

### 6.1 每次跨模块修改前必须冻结六列合同

任何 LEV/TEV/wake/FSI 新通路，先写清：

`科学量 -> 唯一 owner -> 时间层 -> 允许消费者 -> 守恒/功合同 -> 独立 oracle`

缺任一列，不得进入论文 runner。

### 6.2 门禁顺序固定

1. 单元身份门：符号、单位、节点共享、事件/状态反例；
2. 事务门：predict/rollback/commit，parent 无漂移；
3. 所有权门：同一状态和载荷只能有一个生产 owner；
4. 机械门：Neumann、LESP、Kelvin、总账、有限性；
5. 代表性 smoke：必须通过冻结论文非退化门；
6. 时间步/网格细化：不得出现 `Gamma_birth~1/dt`；
7. full CASE；
8. 性能 profiling。

任何一级失败，都不得通过调实验阈值、core、粒子间距或评分窗口绕过。

### 6.3 永久回归状态

已有直接自动化门：

- DVM 模式 `_cuda_particle_bound_strengths == _cuda_bound_strengths`；
- DVM 模式 `load_owner == ptera_kj_plus_dgamma`，生产 impulse 为零、诊断
  impulse 仍可观测；
- predictor fork 后 DVM step、粒子数、frontier 和 free-wake 都前进，parent 哈希
  不变；
- paper-mesh fused/eager CUDA 结果在紧容差内一致。

还应补成独立最小反例，不只依赖 Mancini 间接覆盖：

- `cell_active=False` 且 `ptera_separated=True` 时，LESP pin 必须保持；
- 先预热 compiled cache 再跑 small-grid，结果必须与冷启动一致；
- 论文 smoke 精度失败时，runner 必须以非零退出，队列必须因此阻止 full。

### 6.4 报告边界

- 只能说“Baik 与 Mancini 当前 mandatory GPU 路径通过”；
- 不得说“四篇论文已全部验证”；Yang、Izraelevitz 仍 pending；
- 不得把系统总显存约 `5.7 GiB` 写成当前进程峰值；
- 不得把 `41 passed` 写成全仓测试全部通过；
- 不得把性能门 FAIL 隐去；直接 WRK3 自诱导仍需 tree/FMM 或等价 near/far 方法。

## 7. 后续执行计划

1. 先把本报告中的 owner/time-layer 表转成 Yang 与 Izraelevitz runner 的迁移
   checklist；
2. 保持 GT、几何、运动和评分不变，只迁移 mandatory DVM node-ribbon 数据通路；
3. 每篇先跑代表性 smoke，失败即保存证据并停止 full；
4. smoke 通过后再跑 full，同时报告 RMSE/MAE、相关性、峰值时刻、粒子数、
   free-wake 步数和 GPU 证据；
5. 四 CASE 科学门稳定后再做树/FMM 性能优化；性能优化不得关闭 separated LEV、
   joint TEV 或 free wake；
6. 最后再把同一唯一载荷 owner 接到 Q16 FSI，先验收功共轭和 predictor 事务，
   不用 FSI 结果反向掩盖气动 CASE 失败。

## 8. 当前状态

- Mancini fast/slow mandatory node-owned GPU：**PASS**；
- Baik W1--W4 mandatory GPU：**PASS**；
- Yang、Izraelevitz mandatory node-owned 迁移：**PENDING**；
- 受影响回归：`41 passed`，不是全仓结论；
- 性能门：**FAIL**；
- Git：修复尚未 commit/push，工作树包含其他既有改动，不能整体提交或清理。

正式结果和证据入口：

- `verification.md`
- `execution.md`
- `results/mancini/summary.json`
- `platform/warp_vpm/bing_joint_ptera_gpu.py`
- `platform/warp_vpm/test_q16_dvm_node_ribbon_transaction.py`
