# 实验计划：v5h11 IR-WRK3

**Problem**: v5h10 的固定 N64 Williamson LSRK3 路线在 Baik W2 第 2/3
释放层不能满足状态、探针与每粒子不变量门，且不得通过升级 N、放宽阈值或
回写积分器来挽救。

**Method Thesis**: 对 `f=0,g=1/5` 的 reformulated rVPM，在每个 Williamson
stage 只推进 `(X,Gamma)`，并从宏步起点冻结的
`I=|Gamma|*sigma^2` 重建正的 `sigma`，可在不改变连续 RHS 的条件下得到
三阶、三次场求值、守恒叶上的独立积分器。

**Date**: 2026-08-16

**Governance**: 这是另名的 v5h11 分支，不是 v5h10 amendment。v5h10
`TERMINAL_DECISION.md` 保持 STOP；所有 M0--M3 在不读取 W2 论文 observation
的条件下执行。旧 v5h10 数值只用于确定研究问题和复用同一 N32/64/128 公平
矩阵，不能用于调整本计划的算法、候选角色或阈值。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1. IR-WRK3 在平滑、active `Gamma!=0`、`f=0,g=1/5` 的 autonomous frozen-parent ODE 上为三阶，并在每个 stage 将 `sigma` 保持为正、将 `|Gamma|sigma^2` 保持到预冻结 roundoff 门。 | 这是新积分器唯一的机制贡献；若不成立，后续 W2 计算没有意义。 | 独立 affine matrix-exponential oracle；三类非对称 J；观测阶 `p>=2.8`；zero/near-zero、错误参数和伪实现负控；无生产 reducer 参与 oracle。 | B0, B1 |
| C2. 在真实 W2 三层 frozen-parent inner solve 中，IR-WRK3 能保持物理云与 material/frontier tracer 同 stage，并通过预冻结的状态、探针、载荷和稳定性收敛门。 | 只有这一层通过，方法才有资格进入外层 Ptera 时间细化；它仍不是论文精度结论。 | fresh N32/64/128 完整三层；N64 唯一 candidate、N128 reference；独立 A/B raw artifact；所有 owner/call/hash 门闭合。 | B2, B3 |

**Anti-claims**:

- C1 不等于 source-faithful 同步长 FLOWVPM 离散轨迹；IR-WRK3 是新的离散
  oracle。
- inner N32/64/128 只验证 frozen-parent autonomous solve，不证明移动 Ptera
  parent 的非自治三阶精度。
- 不证明 Baik 曲线、长期 wake、inactive/restart/remesh、production 或 target
  accuracy。

## Paper Storyline

- 主文必须证明：解析三阶与不变量机制；真实 W2 无 GT inner convergence；
  随后独立 outer-time convergence 和 raw artifact 闭包。
- 附录可以支持：log-sigma、whole-step projection、q-freeze、普通 LSRK3 的
  负控；近零与 underflow 攻击矩阵。
- 明确 CUT：N128/256 结果驱动晋级、clipping/floor、stretch-off、core tuning、
  每子步 Pedrizzetti、在 raw freeze 前读取 paper observation。

## Experiment Blocks

### B0: RHS 恒等式与解析三阶 oracle

- Claim tested: C1。
- Why this block exists: 排除“只把 sigma 投影成正值”的伪三阶方案。
- Dataset / task: 无 GT manufactured affine field
  `U(X)=JX+b`；`J` 分别为 diagonal stretch、skew rotation、non-normal shear；
  每类含至少三个非共线 Gamma、三个 sigma 尺度及一个 exact-zero row。
- Compared systems: IR-WRK3；原 additive-sigma WRK3 仅作连续 ODE fine-step
  对照；log-sigma、whole-step projection、q-vector freeze 为必须失败负控。
- Metrics:
  - 独立 `scipy.linalg.expm` 解析解的 X/Gamma/sigma relative-L2；
  - `p=log2(E_h/E_h/2)`，在 N=4,8,16,32 的两个最细区间均 `p>=2.8`；
  - N32 每通道 relative-L2 `<=1e-7`；
  - 用 scaled norm `r=m*sqrt(sum((Gamma/m)^2))`，每 stage 的无量纲残差
    `abs(log(r/r*)+2log(sigma/sigma*)) <= 512*eps*Slog`，其中
    `Slog=max(1,abs(log(r/r*)),2abs(log(sigma/sigma*)))`；
  - chain-rule residual 定义为
    `abs(sigma_rate/sigma + .5*dot(Gamma,gamma_rate)/r^2)` 除以
    `max(1,abs(sigma_rate/sigma),.5*abs(dot(Gamma,gamma_rate)/r^2))`，要求
    `<=512*eps`；
  - 每个 macro `invariant_reference_freeze_count=1`，任何 inner substep rebase
    都是 STOP。
- Negative-control gates are frozen now: log-sigma 必须因非零 sigma-storage/
  stage-invariant trace 被拒；whole-step projection 必须因 projection call 或
  intermediate-stage invariant 失败被拒；q-vector freeze 必须在 non-normal
  shear 的方向/解析阶门失败。exact-zero rows 单独计分，不进入 relative-L2
  denominator。
- Setup details: horizon `T=0.11125 s`；Float64；field 不调用生产
  IR reducer；oracle 不读取任何 W2 output。固定 translation
  `b=(.7,-.03,.02)`，active initial positions
  `[(.1,.2,.3),(-.2,.05,.15),(.03,-.11,.07)]`，Gamma
  `[(.4,-.2,.3),(-.1,.35,.2),(.22,.08,-.31)]`，sigma
  `(.085,.070,.120)`；另设 exact-zero row
  `X=(-.04,.09,.21), Gamma=(+0.,-0.,+0.), sigma=.095`。三个固定 J 为
  `diag(1.2,-.4,-.8)`、
  `[[0,-1.1,.2],[1.1,0,-.3],[-.2,.3,0]]`、
  `[[.7,3,0],[0,-.2,1.5],[0,0,-.5]]`。错误但同 `c=.2` 的
  `(f,g)=(.1,.16)` 也必须拒绝。
- Success criterion: 所有正例和负控按上述门通过；三阶不能由 W2 ratio 代替。
- Failure interpretation: 算法定义或推导错误，v5h11 直接 STOP。
- Table / figure target: 方法附录的解析 order 表。
- Priority: MUST-RUN。

### B1: API、浮点边界与事务完整性

- Claim tested: C1。
- Why this block exists: 守恒公式正确并不保证实现不会在 near-zero、重放或
  callable drift 时 fail open。
- Dataset / task: exact-zero Gamma；`nextafter(sqrt(float64.tiny),0/inf)`；
  极大/极小 sigma；wrong `(f,g)`；NaN/Inf；stale/copy/replay；cap；生产 callable
  与 transitive NumPy/global drift。
- Compared systems: clean IR-WRK3 与每一种单点篡改。
- Metrics: 错误在首个 field call/array materialization 前拒绝；owner/report/parent
  bitwise unchanged；same input clean retry；exact-zero Gamma bitwise zero、sigma
  bitwise unchanged；active underflow 严格 STOP。
- Near-zero rule: `m=max(abs(Gamma))`；`m=0`（包括任意 signed-zero 组合）走
  exact-zero branch；`0<m<=sqrt(float64.tiny)` 在首个 field call 前 STOP；仅
  `m>sqrt(tiny)` 用 scaled norm。重建使用 `log(sigma/sigma*)=.5log(r*/r)`，
  结果超出 finite positive Float64 范围即 STOP。
- Setup details: public frozen dataclasses、bytes-backed readonly Float64 arrays、
  exact type/tree/lineage、运行时 dependency closure。
- Success criterion: 全部攻击被拒且 clean retry 闭环；无 clipping/rebase。
- Failure interpretation: provenance 或事务层 FAIL，不得进入 W2。
- Table / figure target: appendix integrity matrix。
- Priority: MUST-RUN。

### B2: Self + analytic external + physical/tracer same-stage

- Claim tested: C1、C2 的接口前提。
- Why this block exists: 验证新方法没有只守住物理粒子而让 material/frontier
  tracer 使用不同 source state 或 RK storage。
- Dataset / task: 小型非退化 particle cloud；direct Gaussian-erf self field；
  analytic affine external U/J；material subdivision 与 9-node frontier tracer。
- Compared systems: B2a 用独立手写 Gaussian-erf direct kernel、独立 RHS 和
  `solve_ivp(method='DOP853',rtol=2e-13,atol=2e-15)` fine reference；不得调用
  生产 direct/RHS/reconstruction。B2b 单独验证 mock-Ptera centered-FD adapter
  的 6 offset calls/stage。wrong stage-source、wrong parent token、未重置 tracer
  storage、stage 后修复为负控。
- Metrics: 每 stage 必须先缓存 physical 与 tracer 的全部 stage-pre 场，之后
  才能更新任何 X/Gamma/Y；source-state/parent/RK coefficient hash exact；
  stage-1 physical/tracer storage exactly zero；fine errors按约 8 倍衰减；
  final support attestation exact；A/B deterministic。B2a 支持独立时间/RHS
  正确性；B2b 只支持 adapter/call-ledger 正确性。
- Call ledger per layer: physical direct `3N`、Ptera-center `3N`、Ptera-offset
  `18N`；tracer direct `3N`、Ptera-center `3N`、offset `0`；总计 direct `6N`、
  center `6N`、offset `18N`；stage-pre reconstruction `3N`、stage-post
  reconstruction `3N`、physical RHS `3N`、macro invariant freeze `1`、
  sigma-storage update `0`、relaxation `0`。
- Success criterion: 所有计数、哈希、order、support、rollback 门通过。
- Failure interpretation: same-stage coupling 不成立，W2 STOP。
- Table / figure target: mechanics call/trace table。
- Priority: MUST-RUN。

### B3: Baik W2 三层 frozen-parent inner convergence

- Claim tested: C2。
- Why this block exists: 这是进入论文 case 外层时间验证前的最小真实尺度门。
- Dataset / task: frozen W2 `2x8`，source `4,5,6` 对 Ptera `3,4,5`，
  `sigma_birth=.00152 m`，spacing `.0007152941176470589 m`；不读 observation。
- Compared systems: fresh full N=32,64,128；N64 是唯一 candidate，N128 是
  verification reference；不得升级或新增 N。
- Metrics:
  - 每 stage finite、sigma>0、`h max||J_total||F<=1.5`、
    `h max|U-U_gal|/sigma<=.5`；
  - invariant log residual满足 B0 的 roundoff 门；
  - ID-aligned X/Gamma/sigma/material/frontier N64->128 relative-L2 各
    `<=1e-6`；
  - 三个固定 self-field probes 的 U/J 各 `<=1e-4`；
  - layer 2、layer 3 以及三层 stacked 的 force 与 moment **分别**
    `<=.002`；不混合 N 与 N*m；
  - absolute difference ratio `d32,64/d64,128>=1.5`，仅称 empirical
    convergence；两差均 `<=1e-14` 才豁免；
  - Kelvin `<=1e-10`、no-penetration `<=1e-12`、Ptera parent unchanged。
- Comparison formula: `d(a,b)=||q_a-q_b||2`，
  `e(a,b)=d(a,b)/max(1e-15,||q_b||2)`，其中 b 是 fine reference；CL/CD
  只报告，不是 selector。阈值继承来源由 prereg freeze 中的 v5h frontier
  mechanical gate 和 direct/FMM plan SHA 锁定。
- Setup details: 每 N fresh solver/owner；固定执行顺序 32->64->128；完整
  12-file STOP/PASS bundle；失败保留精确 completed prefix；独立进程 A/B。
  A/B 的 semantic payload 必须逐字节相同，而 UUID、UTC、输出路径与运行身份
  必须不同。
- Success criterion: 所有层、通道、调用分账、artifact closure 与 fresh audit
  通过。
- Failure interpretation: IR-WRK3 对真实 inner problem 仍不足，严格 STOP；
  不读 GT、不改方法、不调 N。
- Table / figure target: 主文 inner-convergence table；不是 paper score table。
- Priority: MUST-RUN。

### B4: Outer-parent/source-release 联合离散收敛与 paper-data 解封门

- Claim tested: C2 的边界，不新增 inner-order claim。
- Why this block exists: frozen-parent inner convergence 不能替代移动 Ptera
  parent、source birth 和 load history 的联合离散误差。本块只声称 joint
  self-convergence，不声称隔离出的 Ptera 非自治阶数。
- Dataset / task: W2 32/64/128 steps per cycle，从相同物理初态运行相同周期数；
  source 与 Ptera 采用各自分辨率的一步时钟并在共同 dyadic phase 节点比较；
  不插值、不相位拟合、不读 observation。
- Compared systems: outer P32/P64/P128；P64 是唯一 outer candidate、P128 是
  reference；inner 固定使用 B3 已验证的 N64。另在 P128 运行一次 inner N128
  contamination control。若 inactive/restart/support 触发 remesh_required，
  则记录机制 STOP 而非绕过。
- Metrics: common-phase raw CL/CD、force、moment、三个 probes、impulse；
  64->128 relative-L2 `<=.002`，absolute-difference ratio `>=1.5`；周期积分
  与 source/Kelvin ledger闭合；所有 load 仍由 Ptera 唯一写入。对每个 decisive
  channel，P128 上的 inner N64->N128 absolute difference 必须 `<=10%` 的
  outer P64->P128 difference；若 outer difference `<=1e-14`，inner difference
  也必须 `<=1e-14`，否则停止 outer attribution。
- Ptera FD-J rule: nominal epsilon 固定为
  `2^-10*min(layer_initial_sigma,c_ref)`；P64 raw 另做 `2^-9`、`2^-11`
  非选择性 sensitivity，force/moment/probe 相对变化各 `<=.002`，否则 STOP。
- Setup details: full raw cycle先独立 A/B冻结；评分进程只能消费通过审计的
  immutable raw artifact，不能回写/重跑 candidate。
- Success criterion: outer gate、lifecycle/remesh gate、A/B artifact audit 全过，
  才将 paper observation access 从 sealed 改为 score-only。
- Failure interpretation: 继续实现明确的 lifecycle/remesh owner 或提高 outer
  resolution需另行预注册；当前 paper comparison 保持 BLOCKED。
- Table / figure target: outer convergence appendix；通过后才生成 paper score。
- Priority: MUST-RUN BEFORE PAPER COMPARISON。

### M4 score contract frozen before GT access

- Ground-truth literal:
  `baik2012_w1_w4_corrected_total_cl_cd.csv`, SHA-256
  `4de6b01cd8072959e5b780053f311efa92ab5a94f17940dd122df340ad638f2f`。
- Metric-semantics reference only: `run_baik2012_benchmark.py`, SHA-256
  `4d3d05c6c2ed5f9d2735f0e60d87d4c738d5e7fc9999f5589e4b155a9167675b`。
  它会自行读取 GT/运行旧模型，**不是** v5h11 scorer，M4 不得执行它。
  filter semantics reference `baik2012.py`, SHA-256
  `a8305a6058d02aad12b9acb376b8a38d217789bc767fcc91a15b3fa07434a86d`。
- Planned scorer path:
  `platform/forward_flight_benchmarks/score_fluxv_v5h11_baik_w2.py`，只允许
  `--raw-artifact-a --raw-artifact-b --audit-token --ground-truth --output-dir`；
  禁止 solver/source imports 或 candidate execution。该文件尚未实现，源码 SHA
  必须在 raw A/B+fresh audit 后、首次打开 GT 前写入独立
  `SCORE_UNLOCK_FREEZE.json`；SHA 缺失时 fail before open。
- Frozen semantics: 400 phase samples；source-equivalent 1 Hz ideal sharp Fourier
  low-pass；禁止 phase/amplitude/offset fit。
- PASS thresholds: RMSE CL `<=1.1869728057492306`，RMSE CD
  `<=.7314313811433125`，Q1 RMSE CL `<1.5656896456956615`，Q1 RMSE CD
  `<.5598942841503666`。
- Unlock token 必须绑定 raw artifact A/B SHA、semantic equality、不同运行身份、
  source closure 和 fresh-audit PASS。缺任一字段时 scorer 在打开 GT 前拒绝。
  评分失败只归档 STOP，不允许回调算法、N、P、核或阈值。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 数学/阶数 sanity | B0 全矩阵 | order、invariant、zero branch 全过 | 秒至 2 分钟 CPU | 同源 oracle；以 matrix exponential 独立实现缓解 |
| M1 | 实现与 same-stage mechanics | B1+B2 | provenance、rollback、call ledger、fine limit 全过 | 2--10 分钟 CPU | near-zero 与 tracer 时序 |
| M2 | 真实 W2 inner gate | B3 N32/64/128 fresh A/B | 所有冻结门通过；任一失败即 STOP | 约 30--60 分钟/轮 CPU | self-stretching 强、内存 stage ledger |
| M3 | 外层时间/lifecycle gate | B4 outer 32/64/128 | common-phase/load/lifecycle 全过 | 约 1--4 CPU 小时 | inactive/restart/remesh 尚可能缺失 |
| M4 | raw freeze 后评分 | 只读 scorer | raw A/B+fresh audit 通过后才首次读 GT | 分钟级 | 数据泄漏；进程与目录隔离 |

## Compute and Data Budget

- GPU-hours: 0；当前均为 CPU direct-kernel/Ptera 诊断。
- M0/M1 预算: `<=15 min`。
- M2 预算: 两次独立完整矩阵合计 `<=2 h`；超时写 STOP prefix，不缩矩阵。
- M3 预算: `<=8 CPU h`；超时或 lifecycle 缺口为 STOP。
- Data preparation: 只复用冻结 W2 movement/source；M0--M3 不打开 corrected
  observation CSV。
- Human evaluation: 无。
- Biggest bottleneck: layer-3 self-stretching direct U/J 与未完成的完整周期
  lifecycle/remesh，而不是 Ptera load计算本身。

## Risks and Mitigations

- 风险：IR-WRK3 只消除 invariant drift，Gamma 方向误差仍大。
  缓解：B0/B2 先证阶，B3 固定 N64 仍失败则停止，不调参。
- 风险：`Gamma -> 0` 的约化坐标奇异。
  缓解：exact-zero 独立层；active near-zero 预检 STOP；scaled norm/log重建。
- 风险：inner convergence 被误写成全耦合时间精度。
  缓解：B4 为 paper comparison 前强制门，claim 中显式写 frozen-parent。
- 风险：layer1 完全相同稀释 stacked load误差。
  缓解：layer2/3 force、moment 各自另设 `<=.002` 门。
- 风险：结果驱动方法选择。
  缓解：算法、N 角色、门、run order 在第一条 IR-WRK3 数值前冻结；失败只允许
  另名新实验。

## Final Checklist

- [ ] B0 independent analytic oracle and `p>=2.8` pass
- [ ] B1 zero/near-zero/provenance/transaction attacks pass
- [ ] B2 physical/material/frontier same-stage evidence passes
- [ ] B3 fresh N32/64/128 inner artifact and audit pass
- [ ] B4 outer time/lifecycle convergence passes
- [ ] Raw A/B artifact is immutable before any paper observation read
- [ ] Main paper claims remain limited to evidence actually passed
- [x] Frontier/LLM/VLM contribution is explicitly not claimed
- [x] Nice-to-have alternatives are separated from must-run blocks
