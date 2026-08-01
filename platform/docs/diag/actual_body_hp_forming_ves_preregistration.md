# Actual-body h/p → massless forming → VES 条件化预注册

**冻结时间**：2026-07-28T17:37:08+08:00  
**状态**：`PREREGISTERED / EXECUTION=false / PRODUCTION=false / CLAIM_STATE_CHANGE=false`  
**用途**：S3ai-v2.2 正式结果之后的唯一合法研究分支定义；不是数值结果  
**直接上游节点**：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
**最终生产目标**：空间涡态 → 同一实际翼面统一压力 → 图级 `ForceLedger` 一次成力  

冻结时，PID 922252 的正式 31-history one-shot 仍在运行；永久 attempt
marker 已存在，canonical result 与 latest result 均不存在。本文没有修改
one-shot、guard、runtime、claim YAML、formal authorization 或任何气动常数，
也不授权在正式结果之前执行本矩阵。

## 0. 研究问题与四步纪律

### ① 病因、数据规律、claim 节点与可动空间

固定空间 S3ai 只可能回答：具名零额外状态压力律

\[
R_P^n=M_a(g^{n+1}-g^n)+\Delta t\,P^{n+1/2}
\]

在同一可达 material-history 路径上是否留下超出冻结误差球的 residual。
它没有空间 \(h/p\) 细化，因此任何正 witness 都可能来自：

- actual-body/source-doublet/wake-cut 离散误差；
- 只有 P2 的表示选择；
- close evaluation、同阶段压力或 common-space projection 错件；
- massless forming geometry 缺件；
- 更后置的有限质量/卷吸状态缺件。

当前直接可动节点是 open 的
`N3.1j3b6d18c2b3b3b2c2b2b3e3b`、`N3.1j2`、`N3.1j3` 和
actual formation 节点；validated/frozen 的材料 P2 恒等式、统一压力代数、
finite-angle birth-flux 身份和 VES 状态身份均不得扩大解释范围。

### ② 学科机理

- Xia–Mohseni 只授权二维、高 Re、零厚度、finite-angle sharp junction
  中的无质量形成片方向、强度、相对速度与环量率联立。
- DeVoria–Mohseni 的 VES 由片面质量、片内动量、涡量与 entrainment 守恒
  定义；一个 pressure residual 或自由松弛变量不能自称 VES。
- 连续自由涡态只能通过同阶段诱导速度、势跳及其材料历史进入唯一
  Bernoulli 面板压力；不得另加 vortex-force/impulse production 总力。
- actual-body 空间误差没有被排除前，不能用 forming 或 VES 吸收数值误差。

### ③ 缺件还是错件

当前确定的首要缺件不是一个可调常数，而是一套真正的 actual-body
\(p=1,2,3\) 同阶离散族。现有 body solve、classified cut、material wake
transport 和 coupled body-wake 路径均为 P2 专用。现有 sphere \(h\) oracle、
fixed-diamond 时间 Cauchy 和 NACA-2406 shell 几何不能证明 actual-body
unsteady pressure 的 \(h/p\) 收敛。

因此判定顺序固定：

1. 先排除 observer/数值组成错误；
2. 正 witness 必须在 actual-body \(h/p/\Delta t/q\) 下重现；
3. 再测试边界类型正确的无质量 forming geometry；
4. 只有无质量几何仍留下稳定、非零且有独立质量/动量/entrainment 证据的
   cokernel，才允许测试 VES necessity。

### ④ 机理方案

本文件预注册一个结果条件化的 13 点离散矩阵、三种边界 forming 分支和 VES
necessity 判据。任何方案均不读取 Fig17/18/19、升力或推力来选择阶次、网格、
核、base 条件、形成角、状态维数或 VES 参数。

## 1. 冻结时已有证据与尚未证明的内容

### 已有窄义证据

- 连续 P2 actual-boundary Galerkin 在 attached unit-sphere 解析
  potential/velocity/\(C_p\) oracle 上具有 \(h\) 收敛证据。
- fixed diamond 的 P2 material wake/body trace 有保存的时间 Cauchy 门。
- `surface.face_mu` 是 primary material trace；row cache 不是状态来源。
- same-stage observer 先读保存的 \(\phi\) 和 material trace，direct solve
  只作独立审计。
- 具名压力通道先在 pressure level 相加，再做一次面板成力的代数身份已冻结。
- NACA-2406 运动 shell 和壁面运动学可以生成，但尚未生成 actual-NACA
  coupled pressure。

### 尚未证明

- 不存在 actual-body P1/P3 body、cut、wake、transport 与 pressure provider；
- 不存在同一实际空间的 coupled pressure \(h/p\) 收敛族；
- 不存在联合 \(h/p/\Delta t/q\) uncertainty ball；
- 不存在无人工 root cap 的镜像焊接 full-wing NACA-2406 S3ai parity 网格；
- 非零 trailing base 的 two-front/base-confluence 状态尚未闭合；
- 不存在 moving actual-NACA same-stage pressure；
- 不存在 actual-geometry pressure 到单一图级 `ForceLedger` 的验证；
- 不存在 forming geometry、VES necessity、生产力、三点、118 或 Fig17/18/19
  的本轮结果。

## 2. Claim chain 边界

| 节点 | 当前状态 | 本预注册允许的含义 |
|---|---|---|
| `N3.1j3b6d18c2b3b3b2c2b2b3e3b` | open | 正式结果目标；即使有 witness，最多在独立审计后 open→partial，且只限 fixed-space named-law obstruction |
| `N3.1j3b6d` | partial | actual-boundary source-doublet 方向；不等于 actual-body pressure 已闭合 |
| `N3.1j2` | open | material continuous sheet advancement 尚未成为生产 provider |
| `N3.1j2a` | validated/frozen | 只冻结单元内仿射材料 P2 Kelvin–Helmholtz 恒等式 |
| `N3.1j3` | open | 统一压力的物理 provider 仍未闭合 |
| `N3.1j3a` | validated/frozen | 只冻结压力级合并、一次成力的代数身份 |
| `N3.1j4` | partial | P2 distributed-doublet 空间候选，仍是 diagnostic shadow |
| `…b3a` | validated/frozen | 只冻结 coincident finite-angle sharp-junction birth-flux 身份 |
| `…b3c` | open | actual body-newborn physical-Kutta formation solve |
| `N2.6c2b` | validated/frozen | 只冻结 VES 状态与守恒方程身份，不证明 FLUXV 需要 VES |
| `N2.6c2c` / `N3.1j4b5b` | open | smooth-LE VES junction/release 均未授权 |
| `N3.1f` | dead_end/frozen | 纯 rVPM/粒子 LEV 生产替换路线禁止复活 |
| `N3.1i2a` | validated/frozen | 只冻结已有面板力后的功共轭传递身份；当前阻塞不在结构模型 |

当前树还缺少独立的 `actual-body spatial h/p convergence`、
`boundary-specific massless forming coverage` 和 `VES necessity discriminator`
命题槽。它们只能在正式结果终止后按分支新增，不能在本预注册中伪装为已存在或
已验证的 runtime 节点。

## 3. 执行解锁条件

以下条件必须同时满足：

1. 正式 one-shot 已终止并生成 canonical result；不得静默重跑已消耗的
   permanent attempt。
2. result interpretation contract 的十五个 aggregate checks 全为 `true`。
3. canonical/marker/receipt/hash/accounting/31 histories/22 fields 均通过
   确定性复算。
4. fresh independent result review 已完成。
5. 正式分支是已解析的 zeroth-order 或 first-order obstruction。
6. P1/P3 与 P2 使用同一物理弱式、body/cut/wake/transport/pressure 语义，
   且各自先通过 manufactured/analytic oracle。
7. common-space transfer 是守恒投影、common refinement 或 mortar；
   禁止 coordinate-nearest-neighbour matching。

十五个上游 aggregate checks 为：

```text
registry_accounting
all_history_hard_guards
wrong_birth_negative_control
wrong_attachment_negative_control
projected_omega_even_convergence
projected_omega_odd_convergence
projected_zero_sum_even_convergence
projected_zero_sum_odd_convergence
projected_zero_noncancellation_even_convergence
projected_zero_noncancellation_odd_convergence
matched_stage_projected_q_families
manufactured_parity_controls
omega_odd_interval_includes_zero
zero_sum_odd_interval_includes_zero
zero_stagewise_odd_interval_includes_zero
```

分支授权：

| 正式结果 | 本预注册的执行 |
|---|---|
| `PROTOCOL-NO-GO` | 0 个 h/p history；返回失败的 observer/数值节点 |
| `ZEROTH-ORDER NAMED-LAW OBSTRUCTION` | 13 个 zero history；不得构造 centered tangent 或 forming/VES |
| `FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS` | 13 点 × `{zero,+epsilon,-epsilon}` = 39 histories |
| `NO RESOLVED WITNESS` | 默认 0；只可另行批准 39-history continuum upper-bound audit，且不得晋升 claim 或搜索状态 |

## 4. Actual-body h/p/dt/q 矩阵

继承且不拟合：

- `epsilon = 0.0025`；
- `dt ∈ {0.25, 0.125, 0.0625}`；
- `q ∈ {8,10,12}`；
- 相同 S3ai physical window、一个 excluded prestep、同一 parity 与负对照；
- 新气动常数数量为 0。

这里 \(p\) 是 body potential、cut trace、material wake 与 pressure projection
的一致多项式阶次，不是 quadrature order \(q\)。

| ID | h | p | dt | q | 作用 |
|---|---:|---:|---:|---:|---|
| C01 | H1 | 2 | 0.1250 | 12 | finest mixed cube |
| C02 | H1 | 2 | 0.0625 | 12 | finest mixed cube |
| C03 | H1 | 3 | 0.1250 | 12 | finest mixed cube |
| C04 | H1 | 3 | 0.0625 | 12 | finest mixed cube |
| C05 | H2 | 2 | 0.1250 | 12 | finest mixed cube |
| C06 | H2 | 2 | 0.0625 | 12 | finest mixed cube |
| C07 | H2 | 3 | 0.1250 | 12 | finest mixed cube |
| C08 | H2 | 3 | 0.0625 | 12 | common anchor |
| AH0 | H0 | 3 | 0.0625 | 12 | 第三个 h 水平 |
| AP1 | H2 | 1 | 0.0625 | 12 | 第三个 p 水平 |
| ADT | H2 | 3 | 0.2500 | 12 | 第三个 dt 水平 |
| AQ8 | H2 | 3 | 0.0625 | 8 | q-tail |
| AQ10 | H2 | 3 | 0.0625 | 10 | q-tail |

`C08` 同时是 q12 anchor，因此唯一离散点总数为 13。

### M0：continuum localization

- `H0/H1/H2` 是同一 frozen S3a full-wing diamond/cut 的三层精确嵌套均匀细化；
- 几何和拓扑身份不变；
- 只回答 named pressure-law witness 是否在 continuum limit 保持；
- 不宣称该 diamond 是 RoboEagle 生产几何。

### M1：actual RoboEagle transfer

只有 M0 稳定 witness 后才定义：

- 镜像焊接、无人工 root cap 的 full-wing NACA-2406；
- 明确 upper/lower/base/cut/front ownership；
- 非零 open base 使用 two-front + base/confluence 状态；
- 建议嵌套 `nc/ns_half = 4/8, 8/16, 16/32`，但最终网格不能由力误差选择；
- P1/P2/P3 必须有同阶 body/wake trace、压力与 common-space provider；
- 禁止拟合 Kutta、base speed、wake strength 或近场 core。

## 5. h/p 硬守卫

### 几何与拓扑

- nested-mesh identity、watertight manifold、正面积/体积；
- root mirror/weld identity，无 artificial root-cap load；
- typed upper/lower/base/cut/front ownership；
- zero-tip 与 span-reflection parity；
- physical open base 缺 two-front state 时 fail closed。

### 状态、历史与同阶段压力

- primary trace 只读 `surface.face_mu`；
- row-cache mutation 不得改变 primary trace；
- surface mutation、seam、tip、chronology、material-time gap 必须可检测；
- body、wake、wall motion、\(\phi\)、trace、velocity 和 pressure 使用同一
  half/full stage；
- primary pressure 使用 stored \(\phi\)，direct solve 只作 audit；
- moving-wall material potential rate 必须存在；
- 禁止以 endpoint average 代替 midpoint state。

### 收敛与决定

- \(h,p,\Delta t\) 三轴在共同物理范数内收缩；
- q-tail 收缩或进入冻结 round plateau；
- finest \(2\times2\times2\) mixed difference 进入联合 uncertainty ball；
- zero 和 odd-parity lower bounds 必须包含 0；
- 只有 continuum even lower bound 在全部
  \(h/p/\Delta t/q/\)repeat/cross-observer allowance 后仍大于 0，才称为
  persistent witness；
- 不使用 L/T/Fig17/18/19 选择方法。

### 压力与力账边界

本门的 method selection 只使用状态与 pressure-law residual。若随后进入统一
压力验证，所有 bound/free/source/viscous pressure channel 必须先在 pressure
level 相加，只做一次 surface integration，并且逐面板和等于唯一图级
`ForceLedger` contribution。不得并存 `dCN_ds`、impulse-force 或
vortex-force production 力。

## 6. Boundary-specific massless forming

只有 first-order witness 在 h/p 后持续时，才进入与实际边界类型匹配的分支。

### Sharp single junction

局部独立量是 \(\theta_g,\gamma_g,u_g\)，
\(\dot\Gamma_g=u_g\gamma_g\) 是依变量。Xia–Mohseni 公式只作二维
finite-angle oracle；通过它不等于 actual 3D wing formation 已闭合。

### Finite trailing base

单一 Xia junction 不适用。必须保留 upper/lower 两个 front 与
base/confluence region；外侧势流不能唯一识别 base state。不得把见证算例的
`base_speed` 比值、六个局部标量或任何固定维数晋升为生产参数。

### Smooth leading edge

必须先有：

- `N2.6b4f` 的独立近壁 profile/edge 动力学；
- `N2.6c1c` 的场级 material spike/separation manifold；
- `N2.6c2c1` 的相对 IBL inventory flux；
- wall-normal、surface \(h/p\)、时间窗与 held-out motion 的联合收敛。

随后先测试不带 \(\rho_s,q\) 的三维光滑前缘 massless forming sheet。二维
sharp-edge 成功不得代替该场级门。

## 7. VES necessity 门

只有正确的 massless forming geometry 仍留下 h/p-stable nonzero residual，
且独立近壁数据支持质量、动量与 entrainment 缺口，才定义：

\[
k=\dim\ker(T_G^T M^{-1}),
\]

其中 \(T_G\) 是 massless geometry tangent。\(k\) 必须从 refinement 后测得，
不得预设。再以独立守恒状态构造 VES tangent \(T_V\)，在 held-out histories
检查其是否覆盖同一个 cokernel。

VES 每个曲面点的 raw inventory 可写为
\(\boldsymbol{x}_s(3),\rho_s(1),\boldsymbol v(3),
\boldsymbol\gamma_{\mathrm{tan}}(2),q(1)\)，共 10 个 raw components；
这不是全局自由度维数声明。mesh、gauge、边界和守恒约束后的维数必须另行
预注册并测得。

VES GO 还必须同时满足：

- 面质量、面动量、\(\boldsymbol\alpha\) 演化与 release ledgers 守恒；
- on-sheet principal-value/self-advection 收敛；
- held-out field histories 覆盖，而不是拟合总力；
- \(\rho_s\to0\) 时退化到 \(q=0\) 与 pressure jump \(=0\)；
- 最多先成为 coupled diagnostic state，不直接进入 production。

若 \(k=0\)、residual 收敛到 0、独立 field evidence 缺失，或 \(T_V\) 不能覆盖
held-out 缺口，则 VES 保持 open 并停止。

## 8. 预注册决策树

```text
formal protocol failure
  → observer / numerical node wrong; stop h/p hypothesis search

zeroth-order obstruction
  → 13 zero histories
  → audit named pressure law / same-stage observer
  → no forming tangent, no VES

first-order obstruction
  → 39 h/p histories
  → witness disappears: numerical/observer component wrong; stop forming/VES
  → witness persists: enter boundary-correct massless forming
      → T_G covers residual: missing geometry component; no VES
      → stable nonzero cokernel + independent mass/momentum/entrainment evidence
          → held-out T_V coverage + zero-mass limit: VES diagnostic candidate
          → otherwise unresolved; no production

no resolved witness
  → no state search and no claim promotion
  → optional continuum upper-bound audit only after a separate post-result approval
```

## 9. 治理实现前置

在新增任何可执行 h/p/forming/VES 节点前，先修复并测试：

1. `ClaimDAG` 对所有深层 child 的递归索引、freeze/state/guard 治理；
2. 区分 `requires_claim`、`evidence_from`、`supersedes/falsified_by` 与
   runtime `requires/provides`；
3. `ClaimGraph` 对缺失 claim/runtime dependency fail closed，禁止静默过滤；
4. actual-boundary 深层节点具有显式 implementation/requires/provides/
   enabled_in/runtime_role；
5. validated/frozen 子节点继续独立保存实现指纹。

这些是正式 one-shot 终止后的治理前置，不授权现在修改 provenance-bound
`claim_runtime/core.py`、claim YAML 或生产入口。

## 10. Claim effect 与硬 nonclaims

本预注册本身不改变任何 state/freeze。未来各门的最大效力：

- 正式 resolved obstruction：目标节点最多 open→partial；
- continuum h/p 重现：新建的 actual-body h/p 命题最多 open→partial；
- massless forming coverage：只更新对应 boundary-specific 新命题；
- VES necessity 通过：最多把新 VES necessity 命题设为 partial；
- `N2.6c2b` 的 validated/frozen 身份范围不变；
- `N3.1f` 永不复活。

本文明确不声称：

- 已有 actual-body h/p 结果；
- 已证明 massless forming geometry 正确；
- 已证明需要 VES 或识别了状态维数；
- 已验证 actual panel pressure、force、production 或结构模型；
- 已通过 v41 三点、118、趋势记分卡或 Fig17/18/19；
- 已提升模型精度。

## 11. 冻结输入

| 文件 | SHA256 |
|---|---|
| `platform/actual_wake_reachable_pressure_obstruction_v22_one_shot.py` | `ddcb2dccfe315c4dfd978cc04f17fdfbdf99dcc8cf8172f6b3c8d9cea76b428c` |
| `platform/actual_wake_reachable_pressure_obstruction_v2_guard.py` | `d2f05dd9a4951c082ed3949f59d95dec10f9a052885394d24a6621ec1b295b73` |
| `platform/claim_runtime/reachable_pressure_symmetry.py` | `595df927549158801f27af26bd5e8049a260d3d689bc77d4637b9590b75581f6` |
| `platform/claim_runtime/actual_wake_reachable_pressure.py` | `1bd97ae35c5ec76b400ace4e42ea3fb3286af56d101cf1263fd0f996821946f9` |
| `platform/docs/diag/actual_wake_reachable_pressure_obstruction_cases_20260728_134229.yaml` | `3d662a69c1da80a1452b6b05c67107b188070871bb1b594977467ab7384e0b27` |
| `platform/docs/diag/actual_wake_reachable_pressure_result_interpretation_contract.md` | `1179c89f2690034c044a7328177aed6a3c8d1b877ee30ae576eafe1e5acf6494` |
| `platform/docs/diag/actual_wake_reachable_pressure_result_interpretation_contract.json` | `6e95b9ec4c51f47f432ce0e3e96c7647e5bf5e9949d877e4be4750ac5ef643cd` |
| `platform/claim_nodes/n3_ds_vortex.yaml` | `991752d97566a843bd854e4de159096e7ccfe28d0afd7ba42d57b5e30d20f9a3` |
| `platform/claim_nodes/n2_kirchhoff.yaml` | `cb392c95ad0338c97f7845bc722c801fe5b0497d1eef3e43653ba54f44560f48` |
| `platform/docs/diag/research_n3_finite_forming_zone_sources_20260728.md` | `145fa10913fbfd00d0417d8b533cac6c9e012d46c36f251f2025f0275d1cb909` |
| `platform/docs/diag/research_n3_panel_pressure_conservative_transfer.md` | `e7e8486b05bf457f486d1594306c9e005cb395849ac21f8895441b69f6edbadb` |
