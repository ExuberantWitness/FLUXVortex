# N1 前缘吸力—N2 分离弦向压力可辨识性 witness 预登记

**日期**：2026-07-29  
**阶段**：研究流程①，病灶归因；只读观测，不是候选实现  
**状态**：PRE-REGISTERED；未运行 GPU；不得修改 V4.1、claim YAML 或任何力公式

## 1. 唯一问题与 claim 边界

本轮只回答：

> Fig17/18/19 的推力正偏差，能否在现有 V4.1 账中区分为 N1
> `leading_edge_suction` 已有前向吸力过强，还是 N2.2 把分离修正强制投到全局
> 风升力方向后，缺失了 N2.5 所需的分离弦向压力端口？

N1 已 `validated/frozen`，本轮只观察，禁止修改。N2.2 已 `falsified`，但其当前
生产行为仍需如实记录。N2.5 仍为 `open`，本轮不实现静态极曲线、LESP、LEV、
阻力系数或任何新的弦向力候选。

## 2. 运动学身份硬门

N5.1c 已 `validated/frozen`：Meng Figure10/16 的实验运动在当前代码坐标中唯一
对应 `twist_phase_deg=-90°`；当前生产 V4.1 使用 `+90°`，两者不得混称为同一
实验身份。

runner 在任何 GPU 初始化前必须同时验证：

1. `N5.1c.state == validated` 且 `freeze == true`；
2. 上止点 `phase_solver=pi/2` 时，`-90°` 给出 `dpsi/dt>0`，`+90°`
   给出 `dpsi/dt<0`；
3. Figure16 主判别的非零扭转 case 只能使用 `-90°`；
4. mean witness 的 `+90° production_baseline` 与
   `-90° experiment_identity_corrected` 使用不同 case identity、不同结果标签；
5. `tw=0°` 时两相位物理等价，只运行一次并同时挂两个角色，禁止重复浪费计算。

任何一项不满足即硬停，不允许回退到 `+90°` Figure16 对照。

## 3. 固定物理 witness

所有 nominal twist 继续采用生产口径
`solver twist_amp_deg = nominal_twist_deg / 2`。第一轮是 **G0 exploratory
quick identity**，不是 production-grid 精度结果：

```text
nc=4, ns=8, n_cycle=2,
steps_per_cycle=240 (=60*nc), wake_rows=240,
closure=v41
```

G0 只判断 N1/N2 通道是否具有可分辨的数据指纹，不允许报告为 Fig17/18/19
最终精度。只有 G0 给出唯一 claim 方向后，才另行预登记最小 3 点
`nc=12, ns=16, n_cycle=4, spc_of(U,f)` production-grid G1；本 runner 不会
自动升级网格。

| ID | U | f | nominal twist | AoA | 角色 |
|---|---:|---:|---:|---:|---|
| W1 | 6 | 2.6 | 22.5 | 5 | Fig18(c) U 低边界；与 W2 构成 U contrast |
| W2 | 10 | 2.6 | 22.5 | 5 | Fig18(c) U 高边界、f 高边界；U/f 共享峰端 |
| W3 | 10 | 1.4 | 22.5 | 5 | Fig18(c) f 低边界；与 W2 构成 f contrast |
| W4 | 8 | 2.6 | 22.5 | 5 | Fig17 扭转转折邻域、Fig18 共享中心 |
| W5 | 8 | 2.6 | 22.5 | 0 | Fig19 AoA 低边界 |
| W6 | 8 | 2.6 | 22.5 | 15 | Fig19 AoA 高边界 |

六点都运行两个明确分支：

- `production_plus90`；
- `experiment_identity_corrected_minus90`。

Figure16 的主相位 witness 固定为
`U=8 m/s, AoA=5°, f=2.0 Hz, nominal twist=0/22.5/45°`，非零扭转全部使用
`-90°`。这里不运行 `+90°` 作为实验主判别；生产错相位的历史结果只能作为
明确标注的外部背景，不能进入同组误差或相位对齐。

物理条件和分支先生成再按完整求解调用去重。预期唯一 solver calls 为
12 个 mean calls 加 3 个 Figure16 calls，共 15 个。Figure16 的 tw0 只运行
一次；`+90/-90` 在零扭转时物理等价。另有每个新进程一次、明确排除于科学
样本的 current-source preconditioner，因此全新 G0 session 的实际 solver
invocations 为 **16 = 1 excluded + 15 witnesses**；resume session 为
`1 excluded + 尚未完成的 witnesses`。

## 4. 当前源码身份，不复现旧 baseline

本轮不读取或比较旧 `s6_sweep_v41*.json` 数值。首次建 run 时，对 runner、
预登记、V4.1 求解依赖、全部 executable claim Python/YAML、N5.1c 证据输入及
Figure16 数字化源建立内容哈希闭包。

闭包至少显式包含 `claim_dag.py`、`diff_solve.py`、`rvpm3d.py`、
`research_n3_spatial_loads_20260727.md`、N2 弦向压力一手文献裁决以及
Meng 论文本地一手 PDF。N5 YAML 必须从一次读取的同一字节串完成解析和哈希，
其哈希必须等于闭包成员哈希；禁止“先解析、后重读哈希”的 TOCTOU。

- 每个 case 前后重算闭包；任何新增、删除或内容漂移均硬停；
- resume 必须与创建 run 时的闭包逐文件相同，并逐项匹配首次 session 的
  Python、NumPy 构建、Warp、CUDA driver/runtime、GPU UUID/架构、
  `FLUXV_DTYPE/FLUXV_DEVICE/PYTHONHASHSEED` 和数值线程环境；实际导入的
  `fluxvortex` 必须解析到本仓库 `src/fluxvortex` 且文件哈希属于同一闭包；
  任何差异均在新科学调用前硬停；
- 每个新进程先执行一个固定、排除于科学样本之外的 current-source
  preconditioner；它不与旧 baseline 比较。resume 的 preconditioner 必须与
  首个 session 的 L/T 分别相差不超过 `tau_F=0.15 N`，且 claim graph identity
  完全相同，否则在任何新科学 case 前硬停；
- case 的 NPZ、schema JSON、evidence JSON 与 campaign manifest 均使用
  同目录原子替换，并记录文件 SHA256 与数组内容哈希。

## 5. 必须保存的只读证据

每个 case 只通过空的 `claim_raw_out` observer 运行一次，保存最终周期：

- N1 的 panel pressure/force、环量、局部速度及完整 booked ledger；
- N1 `leading_edge_suction` 的逐步 body-force 序列；
- N2 的 `f2/K/CV/CNv`、分离 panel candidate、实际风升力向 booked force、
  profile-drag 与总 ledger；
- N3 的状态、panel force 与 booked ledger；
- 总 body/wind force trace、claim manifest、全部 claim guards 和
  `claim_contributions`；
- N1+N2+N3 raw ledger、graph ForceLedger、逐通道 mean 的独立闭合误差。

其中 recorder 中的 N1 总通道按 `total-N2-N3` 定义，所以
`N1+N2+N3=total` 只能作为代数记录一致性检查，不能作为 N1/N2 物理归因证据。
可用于归因的完整性证据仅包括 panel→Bernoulli、具体通道→graph 的独立核对，
以及跨固定 witness 的数据指纹。

scorer 不得信任 evidence JSON 中已汇总的 model total、N1 或 N2 数值；必须从
raw NPZ 重新计算 arithmetic/robust 总力、Q1 和单位化 Q2，并要求全部字段、
240 步形状、有限值及 schema 与实际数组逐项一致。artifact 路径必须局限在
run directory 内，拒绝绝对路径、`..` 与 symlink escape，并以一次打开所得字节
同时完成哈希和解析，避免“先哈希、后重开”的替换窗口。

Figure16 数字化只从 `docs/datav2.md` 解析一次，保存为独立 NPZ，并标记为
`published_filtered_gt`。模型 raw 不 clipping、不滤波、不相移。保存
`phase_solver`、`phase_paper`、解析扑动/扭转运动学和实验 `t/T`，但禁止用模型
力互相关选择相移；跨域 alignment 在取得实测运动学 trace 前保持
`unresolved_external_kinematics`。

scorer 必须再次从闭包内 `docs/datav2.md` 独立解析 Figure16，并与归档的 12 个
time/force 数组逐字节一致；manifest 自报 hash 不能自证 ground truth。

`n2.separation_panel_candidate_force_body_N` 的身份必须随 artifact 保存：它只是
旧过程式代码中、在改投全局风升力方向之前形成的 **内部 counterfactual /
支持域 observer**。full-vector 路线已经证伪，因此：

- 不得把该数组或 `candidate-booked` 差的牛顿量级解释成 N2.5/N2.6 的“缺力”；
- 不得把它相加到生产力或据其幅值直接制造候选；
- 只允许用它观察相位、方向和与 N1 前缘吸力的共线性/可辨识性。

## 6. 本轮输出边界

runner 只产出诊断 artifact：

- 不向求解器增加力；
- 不设置候选 closure；
- 不显式覆盖 `lb_lesp_crit/lb_cds/lb_Tv/d_para` 等生产参数；
- 不计算候选 MAE，不晋升 N2.5，不回写 claim 状态；
- 不把旧 `separation_panel_candidate` 当作可恢复的物理力；
- guards 或账本失败即停止，失败本身不得解释为气动机理。

后续归因只能使用预登记 contrasts 和 phase trace，且必须分别报告
`+90 production` 与 `-90 experiment identity`。只有在同一身份分支中，
N1 前缘吸力与 N2 candidate/booked 差的 U/f/twist/phase 指纹才允许进入
“组成部分错/缺组成部分”的裁决。

## 7. 结果出现前冻结的归因规则

只允许 `-90° experiment_identity_corrected` 分支参与 claim 归因。`+90°`
只诊断历史生产基线受到多大运动学身份污染，实验误差不得反向选择相位。

固定阈值沿用
`fig171819_active_disease_prereg_v3_20260729.md`：

```text
tau_F = 0.15 N
tau_c = 0.30 N
```

它们是数值复现及两点 contrast 传播门，不是实验置信区间。实验端只读取
`docs/data.md` 的以下 canonical endpoint，不允许 force 插值或 alias 投票：

- W1--W4：Figure18(a,b) 对应 U 的频率端点；
- W5--W6：Figure19(a,b) 对应 AoA 的频率端点；
- Figure16：本 runner 冻结的 published-filtered trace 周期均值。

数字化横坐标只可按 `fig171819_benchmark.py` 已冻结的 endpoint tolerance 投影到
公开轴端点，必须逐点留痕。固定 residual 为
`r = experiment - model`，固定零和 contrasts 为：

```text
C_U   = W2 - W1
C_f   = W2 - W3
C_A   = W6 - W5
C_tw1 = Figure16(tw22.5) - Figure16(tw0)
C_tw2 = Figure16(tw45) - Figure16(tw22.5)
```

独立 family 只有 `{U, f, AoA, twist}` 四类；两个 twist contrast 合计只算一个
family。只有 residual contrast 任一 `L/T` 分量严格超过 `tau_c` 才是 material。

两个预先方向模板为：

- `Q1 = -N1 leading_edge_suction` 的周期均值双翼风轴力。这只表示若将来证明
  edge term 重复记账时的撤回方向，绝不授权修改 `validated/frozen` N1；
- `Q2` 对每一步 N2 separation-panel observer 的风轴 `(T,L)` 向量先单位化，
  observer 范数不大于 `1e-12 N` 的步记零，再做周期均值。因此只保留方向和
  时间支持，彻底丢弃其牛顿幅值。

一个 material residual contrast 仅当模板至少一个 material 分量同号、且没有
material 分量反号时才支持该模板。Q1 和 Q2 同时支持不叫唯一归因。Figure16
跨域相位保持 unresolved：禁止互相关相移；只用周期均值以及同一 solver phase
下的零相移方向/共线性检查。任何结论若在 raw mean 与 robust mean 间改变，
固定输出 processing-sensitive NO-DECISION。

对全部 material contrast 的 Q1/Q2 列分别归一化后组成两列矩阵；其二范数条件数
严格大于 `20`（非有限值同样处理）即判为共线不可辨识。`20` 在本轮结果出现前
冻结，只是可辨识性门，不是气动力拟合参数。

决策顺序冻结为：

1. 闭包、N5.1c、artifact hash、guards 或账本任一失败：
   `INVALID_EVIDENCE`；
2. `-90°` 没有 material residual contrast：
   `NO_DECISION_OFFSET_ONLY`；若只有 `+90°` 存在则另标
   `NO_DECISION_KINEMATIC_CONTAMINATION`；
3. Q1/Q2 共线、同时解释、处理敏感：
   `NO_DECISION_COLLINEAR_OR_PROCESSING_SENSITIVE` 或
   `NO_DECISION_MULTIPLE_EXPLANATIONS`；
4. 至少两个独立 family 只支持 Q2、且 Q1 至少在一个 material contrast
   反向：`ACTIVE_N2_MISSING_PRESSURE_HYPOTHESIS`。它只授权下一轮 N2.6
   场级 shadow 预登记；N2.5 仍 NO-GO；
5. 至少两个独立 family 只支持 Q1、且 Q2 至少在一个 material contrast
   反向：`N1_LEDGER_AUDIT_REQUIRED`。N1 继续冻结；必须另找 edge-term 与
   Bernoulli panel pressure 的物理所有权矛盾；
6. 两模板都不支持：`NO_DECISION_MISSING_OR_STATE_MEDIATED`；不同 family
   各指向一个节点：`NO_DECISION_MIXED_SOURCE`。

“删掉某节点后 MAE 变好”、常数偏置、旧 observer 的牛顿差值，均不得越过上述
矩阵。没有唯一结果就是正式的 `NO_DECISION`，禁止硬造 N1 或 N2 候选。

scorer 还必须读取 claim YAML 做语义门：N1 保持 `validated/frozen`、N2.2
保持 `falsified`，N2.5 保持 `open/unfrozen`，N2.6 保持可动的
`partial/unfrozen`。N2.5 的“生产 NO-GO”来自本轮冻结的一手文献裁决与预登记，
不是伪造一个 YAML 状态；任何 G0 结果最多授权下一轮 N2.6 shadow
**预登记**，本轮永不直接授权候选实现。
