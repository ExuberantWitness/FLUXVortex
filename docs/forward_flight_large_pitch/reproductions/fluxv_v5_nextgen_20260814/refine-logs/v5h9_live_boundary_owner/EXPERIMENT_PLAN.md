# 实验计划：FluxV v5h9 live-boundary 净状态 owner

**问题：** v5h8 证明了在相同粒子支撑上，旧 downstream 贡献与新 upstream 反向贡献可以先以 gross clone 表示、再精确折叠；但 gross counter-pair 会随释放次数恶化条件数，且不能安全跨过非线性方向松弛。生产候选必须只运输一个净物理态，同时在支撑不兼容时诚实停止。

**方法主张：** 在每次释放事务内，由唯一的 live-boundary slice writer 原位更新边界粒子的当前涡量向量，仅追加新面板的侧边和 downstream 粒子；不可变事件账本保存 before/add/after 证据。任何 RHS、LSRK3 或 Pedrizzetti 松弛只能在净状态提交后执行。

**日期：** 2026-08-15

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1. 当旧 downstream 与新 upstream 具有逐粒子 exact material-basis 双射时，原位净更新与 v5h8 clone-then-collapse 在物理状态、诱导场、Jacobian、冲量和 rVPM 推进上等价，同时不保存 counter-pair。 | 这是把 v5h8 机械 oracle 变成有界物理状态 owner 的必要步骤；否则粒子数和 cancellation condition 随释放持续增长。 | releases 1--4 exact reduction；真实 `lsrk3_step_direct` 的 1/4/16 步及逐 stage replay；更新后 clone count=0、只改 live slice、事务回滚逐位、Pedrizzetti 前已净化。 | B1, B2, B3 |
| C2. 支撑不兼容可在提交前被完备识别并路由为显式 remesh-blocked；仅匹配环量、冲量和一阶矩不足以晋级 remesh。 | 这阻止把近似配对或 moment fitting 冒充为三维核场正确性。 | x/sigma/count/order/ID/tangent 的 1-ULP 与拓扑失配全部 fail-closed；两组 moment-twin 必须被近场 U/J 门识别；只有独立 holdout U/J 加 h 收敛才可解除 remesh blocker。 | B3, B4 |

**要排除的反主张：** 看似稳定只是因为先运输 gross counter-pair 再临时 collapse、依赖相消误差，或因为 remesh 只拟合了低阶矩而没有保持正则化核场。

## 论文证据主线

- 主证据只回答两个问题：exact compatible update 是否是 clone oracle 的真正净态归约；incompatible support 是否会在任何物理推进前停止。
- 附录可记录 conditioning 扫描、cap 边界、moment-twin 反例和未来 remesh 的可行性。
- 本阶段明确删除：Ptera feedback/load、Yang/Izraelevitz/Baik 评分、目标数据、长时生产稳定性、自动调 core、为了过门而改变阈值。
- 本方法不引入 LLM/VLM/Diffusion/RL 等 frontier primitive；无需人为设置 frontier-necessity 实验。

## 冻结 owner 与算子顺序

1. ring/incidence ledger 仍是唯一物理拓扑与 circulation owner；v5h9 仅是 `pre_release_t_n -> post_release_t_n` 的粒子 live-slice 唯一写者。
2. 对 compatible boundary，旧粒子的 `x`、`sigma`、count、particle ID、birth lineage 均逐位不变；只更新 current gamma 与 state epoch：

   `gamma_after = gamma_before * (1 - Gamma_current / Gamma_previous)`。

   该式等价于添加同支撑 clone `gamma_add = -(Gamma_current/Gamma_previous)*gamma_before` 后立即折叠，但物理态中禁止保存 clone。
3. 新面板只沉积两个侧边和新 downstream；upstream 必须由 live slice 更新承担，fresh upstream deposition call count 必须为零。
4. 不可变 event ledger 保存 parent/live digest、Gamma previous/current、before/add/after gamma digest、changed slice、支持兼容证明和事务状态；它是审计控制面，不参与诱导场。
5. 唯一合法次序：`validate -> net update + append -> atomic commit -> RHS/LSRK3 -> Pedrizzetti relaxation`。
6. `gross -> RHS/relaxation -> collapse` 是必败负控。特别是 Pedrizzetti 的方向非线性不与 collapse 交换。
7. 若 `Gamma_previous == 0`、比例非有限、符号/active lifecycle 不允许复用、或任何 material-basis 字段不 exact，则不得近似配对；结果为 `remesh_required` 且物理状态不变。

## 实验块

### B1. 事务、schema 与 v5h8 exact reduction

- **检验主张：** C1。
- **任务：** 复用冻结 v5h8 zero/affine manufactured sheet，release 1--4。
- **比较系统：** v5h8 gross clone + canonical collapse；v5h9 local net update。
- **指标：** positions/gamma/sigma/IDs/lineage，changed-index exact set，U/J、impulse、RHS，event prefix digest，clone/collapse/delete/remesh counters。
- **设置：** `U=2 m/s`、`dt=0.02 s`、span `0.60 m`、`dGamma/dt=0.8 m^2/s^2`、`sigma_birth=0.085 m`、spacing `0.02 m`，沿用 v5h8 fixed probes。
- **成功标准：** 物理数组与 oracle 净态逐位相等或统一 `tau=64 eps max(1, ||ref||inf)` 内；仅 live boundary gamma 改变；物理态 clone count=0；失败注入后 parent/event registry 逐位不变且原 proposal clean retry 等于 fresh run。
- **失败解释：** STOP v5h9；不能建立唯一净态 owner。
- **产物：** 主机械表与 owner ledger。
- **优先级：** MUST-RUN。

### B2. 真实 rVPM 推进与算子顺序

- **检验主张：** C1。
- **任务：** exact-coincident gross pair 与立即净化后的 net state，加入外部粒子，运行 `lsrk3_step_direct` 1/4/16 步；正式矩阵覆盖真实边界、多个 Gamma 比率和每个 RK stage。
- **比较系统：** `collapse -> LSRK3 -> relaxation`；`gross -> LSRK3 -> collapse` 诊断；`gross -> relaxation -> collapse` 必败负控。
- **指标：** 每 stage 的 x/gamma/sigma、U/J、RHS、impulse、sigma positivity、stage trace；松弛前后状态差。
- **成功标准：** 合法净态与逐 stage gross-collapse oracle 在预注册容差内；gross 云从未进入 relaxation；必败负控确实显示非交换而被 gate 拒绝。
- **已知反证种子：** `alpha=0.3` 的 Pedrizzetti microcase 中，错误次序的 gamma/U/J 相对差约 `0.8892/0.7114/0.9726`，不得被归类为通过。
- **失败解释：** 若净态本身与 rVPM 不交换，STOP；若只有 gross 路径失败，则确认必须即时净化。
- **产物：** 算子顺序图和 stage residual 表。
- **优先级：** MUST-RUN。

### B3. 支撑兼容分类与 conditioning/cap

- **检验主张：** C1、C2。
- **任务：** x/sigma/gamma 1 ULP、count/order/subdivision/ID/tangent/edge/release/time/wing mismatch、`ceil` count crossing、曲线/non-affine midpoint；release `n=4,16` 并推进至 cap 边界。
- **指标：** compatibility decision、first mismatch field、state/event/registry rollback、clean retry、particle overhead、条件数代理 `kappa=(||g_old||+||g_add||)/||g_net||=2n-1`。
- **成功标准：** exact 支撑才允许 update；所有失配在分配和提交前返回 `remesh_required`；物理云 counter-pair overhead 恒为零；cap-1/cap/cap+1 都先验、原子、可重放。
- **失败解释：** 任一近似配对、silent weld、旧 ID/lineage 改写或 cap 后验失败均立即 STOP。
- **产物：** fail-closed 矩阵和 conditioning 图。
- **优先级：** MUST-RUN。

### B4. Moment-twin 反证与 remesh 晋级门

- **检验主张：** C2。
- **任务：** 先证明低阶矩门会接受错误核场；仅在 B1--B3 通过后设计独立 constrained remesh feasibility。
- **反例 A：** `sigma=.085`；cloud A 位于 `x=+/-0.05 e_x`、各 `gamma=.5 e_y`；cloud B 位于原点、`gamma=e_y`。两者 `sum gamma`、impulse 和完整一阶矩相同，但冻结 probes 上 U/J 相对差应约 `0.20221/0.22922`。
- **反例 B：** 完全相同 x/gamma，只令 sigma `.085` 与 `.12`；U/J 相对差应约 `.56152/.61752`。
- **未来 remesh 门：** 约束 probes 与评估 probes 严格分离；评估距离 `0.25,0.5,1,2,4 sigma` 加固定随机 holdout；同时验证 U、J、impulse、sum gamma 和 `h,h/2,h/4` 自收敛。
- **成功标准：** 两个 moment-twin 均被 U/J gate 判 FAIL；remesh 只有在 holdout 与收敛全部通过时才可从 `blocked` 升为候选。
- **失败解释：** 低阶矩通过但近场/J 不收敛意味着 remesh NO-GO，不得用拟合 probes 评分自己。
- **产物：** 反例表；remesh 仅作后续附录候选。
- **优先级：** MUST-RUN 反例；remesh 实现 NICE-TO-HAVE 且受前置门控制。

### B5. Fresh artifact 与独立审计

- **检验主张：** C1、C2。
- **任务：** 两个独立 fresh 进程、strict JSON artifact、raw 重算和 fresh reviewer。
- **指标：** semantic digest、run provenance、源码闭包、所有数值/owner/trace/cap gate 重算、target/Ptera/load/write counters。
- **成功标准：** semantic payload 逐字节相同且 UUID/time/path 不同；源码/result SHA 闭合；零目标读取、零 Ptera solver/load、零 feedback/parent write；篡改后严格 STOP。
- **失败解释：** 仅保留开发诊断，禁止晋级。
- **产物：** 审计 bundle。
- **优先级：** MUST-RUN。

## 执行顺序与里程碑

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 冻结 schema/owner/transaction | bootstrap + wrong-type/finite/identity/rollback attacks | 构造即有效；失败零提交；唯一 writer | local CPU，分钟 | 事件账本与物理态循环证明 |
| M1 | 建立 exact local reduction | v5h8 zero/affine release 1--4 | 净态/场/J/冲量/RHS 归约；clone_count=0 | local CPU，分钟 | gamma 比例符号或 lineage 语义错误 |
| M2 | 验证真实算子次序 | LSRK3 1/4/16 + Pedrizzetti 正/负控 | 净化必须先于 RHS/relaxation | local CPU，分钟 | 非线性松弛放大 counter-pair |
| M3 | 完备拒绝与资源门 | mismatch matrix，n=4/16/cap | exact-only update；不兼容路由 blocked；先验 cap | local CPU，分钟 | 近似 welding 或后验 OOM |
| M4 | 证伪 moment-only remesh | 两组 twin + 独立 probes | 低阶矩假阳性被 U/J 捕获 | local CPU，分钟 | 测试与评估 probe 泄漏 |
| M5 | artifact/fresh audit | 两次 fresh + independent audit | strict reproducibility，零越界 owner | local CPU，分钟 | provenance 不完整 |
| CUT | production promotion | Ptera/target/long-time/remesh production | 本阶段永不执行 | 0 | scope creep |

## 计算与数据预算

- GPU：`0` 小时。
- 数据：完全 manufactured/synthetic；不读取目标论文观测、载荷或拟合数据。
- 人工评价：无。
- 首轮 M0--M3 预算：本地 CPU 不超过 15 分钟；M4--M5 另不超过 15 分钟。
- 最大瓶颈：直接 O(N^2) 核场/Jacobian 重放和 mismatch/cap 攻击矩阵。
- 任一预算上限触发即写 STOP artifact；不得靠删粒子、collapse gross 物理云或降低门限继续。

## 风险与缓解

- **counter cancellation：** gross 仅作为瞬时 oracle，不能成为 transport state；生产物理云永远是 net state。
- **非线性算子不交换：** 强制 commit net state 后才允许 RHS、LSRK3、Pedrizzetti。
- **旧粒子 provenance 与当前状态混淆：** particle ID/birth lineage 不变；current gamma/state epoch 放可审 sidecar。
- **支撑差异被浮点容差焊接：** compatibility 使用 exact dtype/layout/count/order/ID/x/sigma 事实；1 ULP 即拒绝。
- **moment-only remesh 假阳性：** 固定 twin 反例、隔离 holdout、U 与 J 同验、空间收敛必须通过。
- **资源攻击：** 计数与 cap 必须在任何数组分配前完成。
- **所有权扩张：** ring ledger 保留物理 topology/circulation owner；无 Ptera/load/feedback 权限。

## 即时 STOP 条件

- 物理态在 commit 后仍含任何 old/counter exact pair，或 clone_count 非零。
- gross pair 到达一次 RHS、LSRK3 stage 或 relaxation。
- compatible update 修改 x、sigma、particle ID、birth lineage 或非 live-boundary slice。
- x/sigma/count/order/ID/tangent/time/wing/source 任一失配却仍提交。
- event prefix 改写、proposal replay、失败后 registry/state 变化或 clean retry 不等于 fresh。
- moment-twin 被低阶矩门错误晋级；holdout U/J 或 h 收敛失败。
- sigma 非有限/非正、粒子数 cap 后验触发、任何 owner/write/target/Ptera/load 计数非零。

## 最终检查表

- [ ] C1 的 v5h8 exact reduction 覆盖 releases 1--4
- [ ] 真实 LSRK3 的 1/4/16 步与逐 stage replay 通过
- [ ] Pedrizzetti 错序负控稳定失败，且 gross 云从未进入物理推进
- [ ] mismatch/1-ULP/count crossing 全部事务式 `remesh_required`
- [ ] 物理粒子态 clone/counter overhead 恒为零
- [ ] moment-twin 两组反例被 U/J 门识别
- [ ] cap 在分配前 fail-closed，失败后 clean retry 等于 fresh
- [ ] 两次 fresh artifact、raw 重算、源码闭包与独立审计通过
- [ ] target/Ptera/load/feedback/parent-write 均为零
- [ ] remesh、production、target scoring 保持 blocked，除非另立预注册实验
