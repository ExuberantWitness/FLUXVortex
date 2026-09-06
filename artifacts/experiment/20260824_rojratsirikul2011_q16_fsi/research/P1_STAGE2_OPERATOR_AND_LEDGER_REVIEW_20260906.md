**P1 阶段 2：先纠正新生涡算子与验收，再实现材料涡片账本（2026-09-06）**

审计基线：`3a9618cd13d3526e429432f0ef6ccb9fdc85e4a9`。本轮保留原配置、实验目标和 legacy 默认路径；新增真实沉积流场的运行时检查、显式开启 joint flag 的测试、两份诊断脚本。未实现完成持续分离账本，未开启生产开关，未提交或推送。

结论：当前不能把 P1 阶段 1 判为“A5 物理三门通过”，也不能把 A15/A19 发散定为已证实的单一 lg 增量错误。新增实测证明：联合系统的 LE AIC 代用列与沉积源不是同一个算子，第一步已违反真实不穿透；A5 的节点掩码还使整条新生涡带退化。先解决这两个问题，才能定义和检验时序环量账本。

**1. 复现实测。**

U=5 m/s，15×30 网格，dt*=0.01，采用冻结刚翼队列的完整 LEV/TEV/free-wake 配置。以下是启动诊断，不是统计均值或实验复现评分。新增 observer 在 author load 调用时观察已沉积 LEV、尚未插入下一 TEV 行的同一流场，重新计算全部表面行的速度残差。

| 攻角 | 时步 | 原 joint 报告的代数残差 m/s | 实际沉积后的全表面残差 m/s | 激活前缘节点 |
|---|---:|---:|---:|---:|
| 5° | 1 | 2.587e−15 | 0.136034 | 0 |
| 5° | 10 | 3.231e−15 | 0.152967 | 0 |
| 15° | 1 | 7.189e−15 | 1.438188 | 31 |
| 15° | 10 | 2.220e−14 | 8.357528 | 31 |
| 19° | 1 | 1.117e−14 | 1.940334 | 31 |
| 19° | 10 | 1.776e−14 | 9.299581 | 31 |

A5 第一帧有 28 条带被标记释放，生成 224 个粒子，但 source-bank lg 全零且 31 个前缘节点全部 inactive。`joint_shed[:ns]` 只覆盖 cell mask，node mask 仍来自 2D bank；`_deposit_dvm_ribbon` 把 inactive frontier 放回 leading edge。由此沉积的 anchor/frontier 两条展向边位置重合、强度相反；粒子数大于零不代表非退化的真实 LEV 出生。

代码位置（本轮补丁后）：

- `src/fluxvortex/warp_fsi/q16_flux_v5m_native.py:885–898`：cell/node 分开读取，inactive frontier 回落到 leading edge。
- 同文件 `:1090`：新生 LEV 列直接取 `aic[:, active_indices]`。
- 同文件 `:1099`：约束仅钉 bound LE。
- 同文件 `:1171`：只改 cell mask，node mask 未随 3D 拓扑重建。
- 同文件 `:1209–1213`：TEV 定义后立刻用同一表达式相减；这里只是关系式的舍入误差检查，不是独立的全历史 Kelvin 证明。

**2. 为什么“LE AIC 列就是新生 LEV 列”不成立。**

令 E 选择 bound 系统的前缘列，当前顶块实际是：

`A Γ + A E q = r`。

因为 A 可逆，它等价于 `Γ + E q = Γ_pre`。非前缘 Γ 必须维持附着预解，pin 仅决定如何把前缘环量分成两份。这验证的是同一个面板环上的代数拆分。

实际释放却使用 `add_connected_ribbon_particles(leading_edge, frontier, -q)`：位置、闭合边、遍历符号、Gaussian 核和空间离散都不同。可复用的是现有 Biot–Savart/粒子核，不能直接复用另一组几何的数值 AIC 列。可逆和条件数良好本身不证明物理等价。

本轮对每条活跃带沉积单位强度，使用现有粒子速度核构造真实 B；然后用联合解的组合强度重新独立沉积，检验组合流场。节点修正仅用于冻结算子探针，没有推进到生产轨迹。

| 攻角 | 原始 B 与 A_LE 的相对 Frobenius 差 | 修正 node mask 后的相对差 | 修正 B 的范数 / A_LE 范数 | 真实 B 重解后的实际流场残差 m/s |
|---|---:|---:|---:|---:|
| 5° | 1.0000 | 1.05379 | 0.05936 | 2.78e−15 |
| 15° | 1.05349 | 1.05349 | 0.05904 | 2.35e−14 |
| 19° | 1.05331 | 1.05331 | 0.05884 | 2.69e−14 |

A5 原始 B 范数 / A_LE 范数仅 5.20e−17，符合退化粒子带抵消。修正后矩阵条件数约 729–736，第一帧可以解；组合沉积与单位列线性组合相差不超过 2.6e−14 m/s。

这仅证明当前几何下存在一致的瞬时算子，不证明当前放置、闭合涡带与 LESP pin 已构成正确的材料涡片模型。替换 B 后第一帧 `max|q|` 从 0.00944/0.09538/0.12880 增为 0.09311/1.07604/1.48277 m²/s，不能把该冻结结果直接接成长期轨迹并宣告稳定。

**3. lg 的真实语义与待推导内容。**

`platform/warp_vpm/ldvm_source_bank_gpu.py:384–389` 的 old_circulation 是已有 TEV 点涡、LEV 点涡和截断账之和；`:491–492` 把本次求解的 LEV 强度追加进 lg。它保存每次释放的 2D 点涡强度历史，并非一个名为 lg 的“超额总量差分器”。其完整算法是历史流场 + 新 TEV/LEV 的耦合方程。

native joint 路径只覆盖返回字典中的 cell strength；bank 已更新的 lg、tg、Fourier 系数和尾迹对流仍属于原 2D 解，node strength 也仍来自它。因此 bank 尚未真正成为 placement-only owner。这个状态不一致需要解决，但“简单减去上一帧 q”没有被现有证据支持：第一帧历史为零时，算子和沉积已经不一致；而之后的 RHS 本来就包含旧 3D 粒子诱导。

必须先确定未知量是哪一种：

- **点涡单次释放量**：旧涡显式留在 RHS，新涡强度由包含全历史的 Kelvin 方程求出。不能再无条件减一次旧强度。
- **材料涡片面板强度/势跳**：相邻面板公共边的真实丝强度由有向 incidence 给出，可能表现为相邻时层强度之差。此时须先切分/连接旧面板并保存上一层 λ；不能把每层 λ 当作独立点涡强度反复追加。

2D lg 的标量总和不能直接充当 3D 涡环/粒子的 Kelvin 账，`particle_field.circul.sum()` 也会重复计数同一条细分边上的强度。

**4. P1-2 的具体修改方案。**

首先沿现有 3D 有限翼模型明确并冻结材料涡片拓扑，再写同一拓扑下的方程。建议以已在仓内讨论过的连续 LE 涡片/近场环缓冲为基线：

1. 新增独立的 CUDA 材料 LE 状态：每条带的上一层片强度、连接节点、自由片几何/强度、出生和移除记录；clone/commit/checkpoint/digest 均覆盖这些状态。3D 强度从这一账本读取，2D bank 不再决定释放或保存另一份“物理真值”。
2. 对流旧片后切分与 LE 相连的近场片，保证公共边连接与方向一致；从 cell mask 重建全部节点的 OR 邻接 mask。附着、持续分离、停止释放和再分离都必须有明确拓扑操作，不能用“未触发就把旧 λ 清零”删除旧涡量。
3. 用实际新生源的几何与核构造 B，联合求解 `A Γ + B λ = r`。约束行必须由实际共享前缘边的涡丝强度推导。不能为了非奇异而任意在 bound-only pin 与 total pin 间切换。
4. 对 TEV 规定求解/力评估/出生的时间层，按同一片强度和公共边关系提交。验收从提交后的实体源或有向边账独立重建；保留删除源的流出账，不能用赋值恒等式当全历史 Kelvin 门。
5. 载荷、pressure ledger、KJ 腿速度、LESP 和加速度/速度 Jacobian 使用相同活动约束与时层。新生源进入验收却不进入 pressure external_flow 的做法也必须消除。

Bird 等的有限翼方法提供一个可核查的具体例子：LE 片先切分并继承原强度，随后按临界前缘丝强度求新片；前缘丝是 bound 与相邻 LE 片强度之差，式 (19) 使用真实 LE 片影响矩阵。该构造有近场环缓冲，不能用 LE 面板的复制列代替。[原作者预印本，§III.D–E、式 (17)–(19)](https://hjab.co.uk/pdf_files/Bird2021_VoFFLE_preprint.pdf)。

若继续采用 Hirato 的 bound-only pin，则要同时实现与之配套的 pseudovortex 几何和势跳，而不是只借用 pin 形式。仓内 `platform/docs/diag/research_n3_hirato_equation_audit_20260727.md` 与 `platform/warp_vpm/bing_joint_ptera_gpu.py:_joint_vortex_solve_cuda` 已记录/实现过此类区分；它们可供移植核查，不能自动作为本次 native 路径的独立验证。本轮尝试下载 Hirato 学位论文返回 HTML，未将其当作已读论文证据。

**5. 本轮已实施的代码及验证。**

- 在 `q16_flux_v5m_native.py:1179–1208` 新增真实沉积流场检查。它独立调用 ring/particle 速度核，检查全部表面行；A5/A15/A19 的错误联合 proposal 均在第一步被拒绝，尚未提交给 owner。
- joint 路径通过后，用同一 newborn particle velocity 更新压力的 external_flow，避免力评估读取出生前流场。
- 分开记录 algebraic residual 与实际验收范围；legacy 保留原来的 retained-row 语义与计算顺序。
- 新增 `tests/test_q16_joint_deposited_flow_gpu.py`：显式 flag=True；三攻角重复拒绝且父状态/所有 bank tensor 不变；A0 附着路径正常通过且与 legacy 一致；开关须为确切 bool。共 **8 项通过**。
- 现有 native/E0 parity/密度/probe **21 项通过**。初次新测试 fixture 没采用冻结刚翼 spacing，触发现有 minimum-overlap 检查；修正 fixture 为队列的 0.018 后新测试通过，未放宽运行时容差。
- A5/A15/A19 各 3 步，补丁前后 legacy 的合力、合矩、两种 Cn 和既有 state digest **逐位一致**。digest 本身未包含 bank tensor；新拒绝测试单独验证了所有 bank tensor 不变，不将旧 digest 夸称完整账本证明。

历史 `tests/test_q16_flux_v5m_native_gpu.py` 的 solver fixture 没有打开 joint flag，所以旧的 13/13 或 native 回归不能作为联合分离路径的专项通过证据。新测试明确覆盖 ON 路径，但它验证的是拒绝错误结果，不能表述成联合物理已修复。

**6. 下一次放行条件。**

每个角度从首步开始，需同时满足：实际沉积后全表面不穿透、同一流场 LESP、非退化且正确连接的 LE 出生、独立时序环量/公共边账、proposal 重试和 checkpoint 重启一致。随后才做 150 步三角度持续释放、时间步对照及载荷窗口比较。默认 OFF 继续保留，当前不具备“开门”条件。

本轮没有把研究性材料涡片重写混入验收补丁，没有增加 dt、强度帽、经验阻尼或调整 LESPcrit 来掩盖发散。P1-2 仍未完成；已完成的是定位首步可证伪的算子错误、阻止错误验收、验证真实 B 构造方向并明确材料账本的实现依赖。

**产物与复现。**

输出目录：`artifacts/baselines/roj_p1_stage2_20260906/`。

- `before/A05.json`、`A15.json`、`A19.json`：原 `3a9618c` 开关 ON 的 10 步。
- `birth_operator.json`：第一帧单位真实粒子列、node 修正与独立组合沉积检查。
- `legacy_before/`、`legacy_after/`、`legacy_parity.json`：默认路径各 3 步前后对照。
- `guarded/`：新增运行时门在三个角度第一帧的明确拒绝。
- `validation.json` 与日志：检查结果和代码哈希。

从 run repo 执行，输出路径须不存在；joint CLI 拒绝时仍保存 manifest/JSON，并返回非零状态：

```bash
PYTHONPATH=src:platform:platform/warp_vpm FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 python3 -m pytest -q tests/test_q16_joint_deposited_flow_gpu.py
PYTHONPATH=src:platform:platform/warp_vpm FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 python3 platform/warp_vpm/diagnose_roj_joint_separation.py --joint --steps 1 --output /tmp/roj_joint_guard_new
PYTHONPATH=src:platform:platform/warp_vpm FLUXV_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 python3 platform/warp_vpm/diagnose_roj_joint_birth_operator.py --output /tmp/roj_joint_birth_new.json
```
