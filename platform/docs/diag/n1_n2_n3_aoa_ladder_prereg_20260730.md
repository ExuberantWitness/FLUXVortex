# N1/N2/N3 AoA 梯度归因 witness 预登记（G0c）

**日期**：2026-07-30  
**阶段**：研究流程①；G0c supplementary AoA attribution  
**状态**：PRE-REGISTERED；尚未运行本 campaign 的 GPU science calls  
**运行角色**：只读诊断；零气动公式修改；不产生候选力

## 1. 唯一问题、已有证据与可动边界

上一轮 G0 的 U/f/twist witness 不能仅凭总力偏差把 Fig17/18/19 的误差唯一归到
N1 或 N2。G0c 不回到常数扫描，也不实现新的压力公式；它只增加一个正交的数据
指纹：

> 在 Figure 19(a,b) 的两个实测频率端点上，AoA 从
> `0 -> 5 -> 10 -> 15 deg` 的三个相邻 residual 梯度，究竟唯一跟随
> N1 前缘吸力撤回方向、旧 N2 分离面板 observer 的方向变化，还是 N3 已记账
> 直接涡力的撤回方向？

claim 边界冻结如下：

- N1 继续 `validated/frozen`。`Q1` 仅是账本归因模板；无论结果如何，本轮都
  不授权修改 N1。
- N2.2 的旧 full-vector 路线已 `falsified`。旧 separation-panel 数组只能作
  单位方向 observer，牛顿幅值不是“缺失力”。
- N2.5 继续 production `NO-GO`；G0c 不把缺少目标静态极曲线的经验闭合复活。
- N2.6 是唯一可能继续预登记的 N2 场级方向，但本轮结果最多授权其下一轮
  shadow **预登记**，不授权实现。
- N3 的当前 booked direct vortex force 只读。既有 falsified N3 候选全部禁
  重走，LESP 仍只可解释起涡临界/供给，不能被重新解释为持续涡力幅值。
- N5.1c 运动学身份保持 `validated/frozen`；所有 science calls 固定使用实验
  身份 `twist_phase_deg=-90 deg`。

因此本轮可动空间不是生产模型，而只是 claim 归因：`N1 ledger audit`、
`N2.6 shadow preregistration`、`N3 spatial-state audit` 或正式
`NO_DECISION` 四类后续方向。

## 2. 固定 science calls

全部工况使用原样 `closure="v41"`，不显式覆盖任何 L-B、LESP、DS 或阻力常数。
nominal twist 遵守现有运动学口径：

```text
solver twist_amp_deg = nominal_twist_deg / 2 = 11.25 deg
twist_phase_deg = -90 deg
U = 8 m/s
AoA = {0, 5, 10, 15} deg
f = {1.4, 2.6} Hz
nc = 4, ns = 8, n_cycle = 2
steps_per_cycle = wake_rows = 240
```

精确的八个、且只有八个 science case 为：

| case_id | U | f | nominal twist | solver twist | AoA | phase |
|---|---:|---:|---:|---:|---:|---:|
| `aoa_f1p4_A0` | 8 | 1.4 | 22.5 | 11.25 | 0 | -90 |
| `aoa_f1p4_A5` | 8 | 1.4 | 22.5 | 11.25 | 5 | -90 |
| `aoa_f1p4_A10` | 8 | 1.4 | 22.5 | 11.25 | 10 | -90 |
| `aoa_f1p4_A15` | 8 | 1.4 | 22.5 | 11.25 | 15 | -90 |
| `aoa_f2p6_A0` | 8 | 2.6 | 22.5 | 11.25 | 0 | -90 |
| `aoa_f2p6_A5` | 8 | 2.6 | 22.5 | 11.25 | 5 | -90 |
| `aoa_f2p6_A10` | 8 | 2.6 | 22.5 | 11.25 | 10 | -90 |
| `aoa_f2p6_A15` | 8 | 2.6 | 22.5 | 11.25 | 15 | -90 |

每个进程在任何新 science case 前另运行一次固定的 current-source
preconditioner：

```text
U=8, f=2.6, nominal twist=0, AoA=5, phase=-90,
同一 v41 closure 与同一 G0c 网格
```

它只作当前数值运行时和 claim graph 锚点，明确排除于科学指标。因此 fresh
session 的 solver invocations 精确为 `9 = 1 excluded + 8 science`；resume
session 为 `1 excluded + 尚未完成的 science cases`。

本轮网格只允许做 G0c 归因，不是 production-grid Fig17/18/19 精度声明。

## 3. Figure 19 实验端点合同

实验端只从闭包内 `platform/docs/data.md` 读取 Figure 19(a,b) 的原始数字化
表。每个 AoA 分别取公开频率轴的首、末端样本：

- low-frequency context：标注为 `f=1.4 Hz` 的首个表格样本；
- high-frequency context：标注为 `f=2.6 Hz` 的末个表格样本。

数字化横坐标因读图存在约 `1.4/2.6 Hz` 的小偏移；只允许沿
`fig171819_benchmark.py` 的已冻结 endpoint tolerance 确认它属于哪个公开轴
端点。必须保留原始 `(f_digitized, force_digitized)` 和投影留痕。**禁止对实验
力插值、外推、平滑、重采样或跨 AoA alias 投票。** grams-force 只按固定
`9.80665e-3 N/gf`（冻结 measurement parser 的标准重力常数）转成 N。

固定 residual 定义为：

```text
r(f, AoA) = experiment(f, AoA) - model(f, AoA)
```

每个频率 context 独立形成三个相邻 AoA contrasts：

```text
C_f_5_0   = r(f, 5)  - r(f, 0)
C_f_10_5  = r(f, 10) - r(f, 5)
C_f_15_10 = r(f, 15) - r(f, 10)
```

每个 contrast 的分量顺序固定为 `(T, L)`。两个频率 context 是两份独立复现
环境，不能把六个 contrast 混成一个“样本数”来绕过跨频率一致性。

## 4. 三个冻结方向模板

所有模板均从最终周期 240 步 raw recorder 独立重算。`Delta` 始终表示同一
频率下 `high-AoA case - low-AoA case`。

### Q1：N1 前缘吸力撤回方向

先把半翼 solver accumulator
`diagnostic.n1.leading_edge_suction_solver_accumulator_body_force_N`
乘 2，逐步变换到各自 AoA 的 `(T,L)` 风轴，再作周期 reduction：

```text
Q1 = -Delta(N1 leading-edge-suction force)
```

负号只代表未来“若被证明重复记账则撤回”的方向，不是删除 N1 的许可。

### Q2：旧 N2 separation-panel observer 的纯方向差

每一步先将
`n2.separation_panel_candidate_force_body_N` 对面板求和、乘 2、变换到当前
case 的 `(T,L)`。对每步向量 `v=(T,L)`：

```text
u(v) = v / ||v||2,  if ||v||2 > 1e-12 N
u(v) = (0,0),       otherwise
Q2 = Delta(cycle-reduction of u(v))
```

这一定义先单位化再作时间 reduction，彻底丢弃 observer 的牛顿幅值。Q2 只
回答方向/支持域是否随 AoA 改变；不得解释为 N2.5/N2.6 missing force，不得
加入总力。

### Q3：N3 已记账直接涡力撤回方向

把半翼
`n3.ds_booked_solver_accumulator_N` 乘 2，逐步变换到各自 AoA 的 `(T,L)`：

```text
Q3 = -Delta(N3 booked direct vortex force)
```

它只检验“现有 N3 直接涡力是否沿 residual 梯度过供”的方向。即使 Q3 被唯一
支持，也只授权 N3 空间涡态/账本归因的下一轮预登记，禁止重开任何已证伪
LESP 标量幅值补丁。

## 5. raw、robust 与数值完整性门

runner 必须保留上一轮已经审计的 `claim-raw-ledger-v1` 精确 92 字段、每字段
240 步形状、`G0_exploratory_quick_identity` raw stage 以及
`post_force_pre_shed` snapshot 身份，以便复用严格 validator。G0c 自身另用
`n1-n2-n3-aoa-ladder-witness-v1` campaign schema 和
`G0c_supplementary_aoa_attribution` stage，二者不得混称。

每个 case 只调用一次原求解器并通过空 `claim_raw_out` 观察：

- raw NPZ、schema JSON、evidence JSON 和 manifest 内容寻址；
- 所有数值字段有限，时间、网格、panel-to-ledger、node ledger、reported-pair
  与 graph ForceLedger 身份全部通过；
- claim guards 全部通过；
- 八个 case 与同进程 preconditioner 的 claim graph identity 完全一致；
- 每个 resume 进程的 Python、NumPy build、Warp/CUDA、GPU UUID/架构、
  solver dtype/device、仓库内 `fluxvortex` 导入位置和线程环境与首个 session
  逐项相同；
- resume preconditioner 的 `L_wind`、`T_wind` 分别与首 session 相差不超过
  固定 force gate `tau_F = 0.15 N`。

scorer 只能信任经过 confinement、`O_NOFOLLOW`、同一 opened bytes 哈希/解析
的 raw artifacts；manifest/evidence 的 summary 不能投票。

两套固定 processing view 必须都计算：

1. `raw`：逐步序列的算术周期均值；
2. `robust`：沿用 V4.1 已固定的中位数/MAD 8-sigma clipping 后周期均值。

实验端在两 view 中完全相同。对 model total、Q1、Q2、Q3 的每个 case/contrast，
若 raw 与 robust 相差严格超过 `tau_F=0.15 N`，或两 view 的 material、
支持/反向、唯一列、condition-number 或最终决策不一致，固定输出
`NO_DECISION_PROCESSING_SENSITIVE`。不得挑选更有利的 view。

## 6. material、支持、反向与共线门

固定阈值：

```text
tau_F = 0.15 N
tau_c = 0.30 N
condition_limit = 20
numeric_zero = 1e-12 N
```

`tau_c` 不是实验置信区间，只作用于带 N 单位的 residual：一个 contrast 只有
至少一个 `(T,L)` 分量满足 `abs(C_r) > tau_c` 才 material。模板只提供方向；
在 residual 的 material 分量上，模板对应分量绝对值必须严格大于
`numeric_zero` 才能投票。这样不会把 `0.30 N` 错用于无量纲 Q2。零范数模板列
不支持任何 residual。

对于一个 material residual contrast：

- 某模板至少在一个 residual-material 分量上非零且与 residual 同号，并且没有
  任一 residual-material、模板非零分量反号，称为 `support`；
- 至少一个 residual-material、模板非零分量反号，称为该模板的
  `reverse evidence`；
- 同一 contrast 支持两个或三个模板，立即记为 `multi-support conflict`，不得
  事后按相关系数择一。

对每个 processing view，把所有 material contrast 的 Q1/Q2/Q3 `(T,L)` 依次
堆成三列矩阵；每列整体 L2 单位化后计算二范数条件数。任一列零范数、非有限，
或 `cond2 > 20`，固定输出 `NO_DECISION_COLLINEAR`。

## 7. 冻结决策规则

每个频率 context 必须单独通过以下“唯一列”门：

1. 至少有一个 material residual contrast；
2. 同一候选列在至少一个 contrast 获得 support，且在该 context 内没有
   reverse evidence；
3. 其余两列在该 context 内均不得获得 support，并且各自至少有一个
   reverse-evidence contrast；
4. 不存在任何 multi-support conflict，也不存在不同 contrast 分别唯一支持
   不同列的 mixed-source conflict。

只有 low/high 两个 frequency contexts 都唯一支持**同一列**，raw/robust 两
view 完全一致，且三列 condition gate 通过，才允许输出 active hypothesis：

- 唯一 Q1：`N1_LEDGER_AUDIT_REQUIRED`。N1 仍冻结，只授权另行预登记物理所有
  权审计。
- 唯一 Q2：`ACTIVE_N2_6_SHADOW_PREREG_ALLOWED`。N2.5 保持 NO-GO；只授权
  下一轮 N2.6 shadow 预登记，不授权实现。
- 唯一 Q3：`ACTIVE_N3_SPATIAL_STATE_AUDIT_REQUIRED`。只授权新的空间涡态/
  统一压力所有权预登记，不授权调 LESP、DS 常数或复活 falsified 候选。

失败语义冻结：

- 闭包、N5.1c、runtime、artifact、schema、guard、ledger 或 endpoint 身份失败：
  `INVALID_EVIDENCE`；
- 任一频率没有 material residual：`NO_DECISION_OFFSET_ONLY`；
- raw/robust 不一致：`NO_DECISION_PROCESSING_SENSITIVE`；
- condition gate 失败：`NO_DECISION_COLLINEAR`；
- 任一 multi-support：`NO_DECISION_MULTIPLE_EXPLANATIONS`；
- 两频率唯一列不同：`NO_DECISION_FREQUENCY_DEPENDENT_MIXED_SOURCE`；
- 有 signal 但不满足唯一列和反向证据：`NO_DECISION_INSUFFICIENT_UNIQUENESS`。

任何 `NO_DECISION` 都是有效研究结果。禁止用 MAE 改善、单个工况、常数偏置或
旧 observer 的牛顿幅值越过上述规则。

## 8. 源码闭包、resume 与输出边界

首次建 campaign 时建立独立、精确内容闭包，至少包括：

- 本预登记、新 runner、约定的新 scorer 及二者测试；
- 已审计的旧 N1/N2 runner/scorer；
- `data.md`、`fig171819_benchmark.py`、claim DAG 和全部 claim YAML；
- `_v2_robo.py`、L-B/UVLM/FSI/几何求解依赖、`src/fluxvortex`；
- Meng 2025 本地一手 PDF与当前 N2/N3 一手文献裁决。

N5 YAML 必须从一次读取的同一字节同时解析和哈希，并绑定闭包成员哈希。每个
preconditioner/science call 前后重算完整闭包；新增、删除或漂移均 fail-closed。
resume 必须拒绝额外 case、缺失/漂移 artifacts、runtime 漂移、preconditioner
force/graph 漂移，并在任何新 science call 前完成全部门。

runner 明确禁止：

- 修改或 monkeypatch 求解器全局变量；
- 修改气动力、常数、网格、运动学或 claim YAML；
- 读取旧 V4.1 cache 作为当前数值真值；
- 自动执行 scorer、自动实现候选或自动回写 claim；
- 把 G0c 结果称作 Fig17/18/19 完整精度验证。

本轮结束时最多产生一个只读 evidence bundle 和一个 scorer 决策。只有 active
hypothesis 才能进入下一轮单候选 shadow 预登记；没有 active hypothesis 就停止，
不得补丁式试错。
