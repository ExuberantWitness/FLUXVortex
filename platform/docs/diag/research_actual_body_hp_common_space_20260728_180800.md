# Actual-body h/p common-space 与可认证求积：文献裁决

**生成时间**：2026-07-28T18:08:00+08:00  
**状态**：`RESEARCH EVIDENCE / EXECUTION=false / PRODUCTION=false / CLAIM_STATE_CHANGE=false`  
**技能**：`research-lit`  
**范围**：S3ai-v2.2 正式结果之后的 actual-body P1/P2/P3 数值归因；不修改正式 one-shot、气动公式、常数、claim YAML 或 Fig17/18/19  
**上游预注册**：`actual_body_hp_forming_ves_preregistration_20260728_173708.md`

## 0. 裁决摘要

本轮高质量一手文献将原来的“13 点 h/p/dt/q 矩阵”收紧为以下可执行边界：

1. **当前代码不是 h/p 离散族。** 它只有连续 P2 actual-body/body-cut/material-wake/pressure 路径；P1、P3、共同参考空间和曲面几何阶次都缺失。
2. **P1/P2/P3 必须只改变场空间。** 弱式、RHS、body/cut/wake 拓扑、连续/跳跃语义、法向、时间层和压力观察器必须完全相同。
3. **场阶 \(m\)、几何阶 \(\ell\)、网格 \(h\)、时间步 \(\Delta t\) 和求积误差必须分账。** 在 M1 上若继续用平面 P1 几何，P2/P3 很可能被几何/法向误差封顶；不能把场阶和几何阶一起变化称作 p 收敛。
4. **不同离散空间不能用 nearest-neighbour 比较。** 必须在同一冻结物理曲面上用 common-refinement/supermesh 的 cross-mass，或稳定 mortar/L2 投影；载荷传递必须与位移传递构成功共轭伴随。
5. **`q={8,10,12}` 只是一条 Cauchy 诊断，不是严格误差上界。** 没有 topology/kernel-specific 的可计算误差证书或区间 enclosure 时，q 轴必须 `UNRESOLVED/NO-GO`，不得进入联合 uncertainty ball。
6. **因此这首先是缺少数值/observer 组成部分，而不是已经证明缺少 VES 物理。** 文献最多授权未来建立一个 `open` 的 “actual-body consistent P1/P2/P3 common-space” claim 槽；本文件不创建节点、不改变状态。

对上游预注册的唯一实质修正是：

> “q-tail 收缩或进入浮点平台”不再足以通过 q 门。AQ8/AQ10/C08 可保留为诊断点，但只有可计算误差界 \(U_q\) 或 validated enclosure 才能为 h/p 差值提供数值不确定度。

正式 31-history one-shot 已冻结且仍应完成；它的结果可按冻结 contract 完整复算，但在上述 q 证书缺失时只能称为 **fixed-P2 contract-internal evidence**，不能单独授权物理 claim 晋升、forming 或 VES。

## 1. ① 病因：代码指纹、挂树与可动空间

### 1.1 当前数据和实现指纹

代码审计显示：

- `actual_boundary_p2_galerkin.py` 只实现 P2 actual-boundary Galerkin；
- `classified_p2_cut_topology.py`、`actual_boundary_body_wake.py` 和
  `material_wake_time_march.py` 的 topology、trace、history 均绑定 P2；
- `actual_wake_reachable_pressure.py` 使用 active P2 line mass；
- 正式 guard 固定完整 trace 9、active trace 7、active rows 1..7、tip rows 0/8
  以及同一个 \(7\times7\) P2 mass；
- 31-case registry 只有 \(\epsilon,\Delta t,q,\mathrm{role}\)，没有 \(h\) 或 \(p\)；
- `global_p1_mesh` 一类符号若出现，只描述几何/速度插值，不能视作 P1 body
  potential provider。

所以当前正式任务只能回答固定 P2 空间中的可达 pressure-law residual，不能回答：

- witness 是否随 \(h\) 消失；
- P1/P2/P3 是否趋向同一 continuum observable；
- P3 是否被几何、求积或条件数污染；
- actual RoboEagle 曲面上的压力是否收敛。

### 1.2 Claim tree 挂接

本轮不写 YAML，只冻结未来可动边界：

| 位置 | 当前含义 | 文献允许的未来动作 |
|---|---|---|
| 正式 deep target `…b3e3b` | fixed-P2 named-law obstruction target，open | 正式结果只形成固定空间证据；q 未认证时不得晋升 |
| `N3.1j3b6d` | actual-boundary source-doublet 方向，partial | 增加同一弱式的 P1/P2/P3 family 与独立 geometry axis |
| `N3.1j3` | 唯一统一面板压力 provider，open | 各阶共享同阶段 pressure observer |
| `N3.1j2` | material sheet advancement，open | 各阶共享相同 cut/wake ownership 与材料历史 |
| `N3.1i2a` | 功共轭传力身份，validated/frozen | 只允许用其伴随映射审计 common-space transfer |
| `N2.6c2b` | VES 状态身份，validated/frozen | 不因 h/p residual 而解释为 VES necessity |
| `N3.1f` | pure-rVPM production，dead_end/frozen | 保持禁止复活 |

未来若正式门和治理递归索引均通过，可新增的命题只能先是：

```text
actual-body consistent P1/P2/P3 common-space discretization
state = open
role  = numerical/observer prerequisite
```

这不是新的气动力来源，也不是 VES 节点。

### 1.3 可动空间

允许：

- 相同弱式下实现 P1/P2/P3 trial/test、cut trace、wake trace、transport 和 pressure；
- 建立冻结 master surface、common-refinement cross-mass 和 mortar/L2 transfer；
- 分别扫描 \(h,m,\ell,\Delta t,q\)；
- 为 touching、disjoint-close、disjoint-far 和 curved pair 建立类型化求积证书；
- 记录自然范数、surface pressure norm、局部误差位置、矩阵条件数和守恒账。

禁止：

- 用 L/T/Fig17/18/19 选择 p、q、网格、曲面阶或 close-evaluation 阈值；
- 用 nearest-neighbour 作为 field/load provider；
- 把 P1 平面几何、P2 二次几何、P3 三次几何的混合变化称为 p 收敛；
- 因 P3 改善小就判定缺物理；边/角/cut/front 正则性可能限制均匀 p；
- 把三点 q-tail 的观测收缩率当严格 remainder bound；
- 在数值轴未解析前用 massless forming 或 VES 吸收 residual。

## 2. ② 学科机理：一手文献

### 2.1 h/p BEM 与几何误差

Stephan–Suri 的 p-version Galerkin BEM 理论说明，P1/P2/P3 必须是**同一算子和
同一变分方程**上的离散子空间，误差首先应在算子自然范数中比较，而不是只看点值
或总力。Babuška–Guo–Stephan 以及 Heuer–Maischak–Stephan 的指数 hp 结果依赖
边角附近的几何级加密和相匹配的局部阶次；它们不授权在均匀 H0/H1/H2 上预设
指数率。Bespalov–Heuer 进一步表明，开曲面边缘奇异性会限制准均匀网格上的
收敛，因而 “P3 相对 P2 改善不大” 可能只是 regularity-limited。

Dölz 等人的 isogeometric BEM 说明，高阶场可以在同一精确 NURBS 几何上比较。
Faria–Marchand–Montanelli 则明确拆分 solution degree \(m\) 与 geometry order
\(\ell\)：对光滑静态曲面，双层势误差同时含场误差与几何误差项；该论文还明确
不包含 quadrature error。其具体阶不能外推到移动、锐边、cut/wake 或表面压力，
但足以否定“固定平面几何上升 p 就是纯 p 试验”。

因此：

- **M0 diamond**：几何是精确分片平面，先做同一弱式的 P1/P2/P3
  continuum localization；若误差集中在 edge/corner/cut/front，另行预注册
  graded-h + local-p，不能改物理。
- **M1 RoboEagle/NACA-2406**：要么所有 p 使用同一个解析/CAD master surface；
  要么显式记录 \((m,\ell)\)，做 fixed-\(\ell\) 的 p 扫描与 fixed-\(m\) 的
  geometry 扫描。upper/lower/base/cut/front ownership 和法向定义跨阶冻结。

### 2.2 奇异和近奇异求积

Sauter–Schwab、Erichsen–Sauter 和 Schwab 的结果表明，共面、共边、共点等
touching pair 需要按相交拓扑做正则化坐标、几何分割和随 hp 精度增长的 composite
Gauss；不存在一个脱离 p、几何和核类型的 universal fixed q。

Reid–White–Johnson 的 Taylor–Duffy 适用于共享 1/2/3 顶点的平面三角形乘积域，
但论文明确不处理 nearly coincident triangles。因此：

- same-face/shared-edge/shared-vertex 可走经过验证的 topology transform；
- disjoint-close 不能用 touching rule 兜底；
- curved touching/close 不能先压成平片再沿用平面证书。

Montanelli–Aussal–Haddar 与 Montanelli–Collino–Haddar 对曲面弱/强奇异积分要求
先求目标点在曲面参考单元中的最近 preimage，再做 singularity subtraction、
continuation 和 transplanted Gauss。Khatri 等人及 af Klinteberg–Tornberg 表明，
普通高阶 Gauss 在目标逼近边界时没有统一精度，算法选择应由可计算的误差估计
驱动，而不是拟合一个 \(\eta=d/h\) switch。

所以 q 路由必须 fail closed：

```text
touching planar
  → topology/kernel/p/test/trial/Jacobian/PV 类型化规则
disjoint far
  → ordinary Gauss + 本规则自己的可计算误差证书
disjoint close
  → closest-point/preimage-aware close evaluator
curved touching/close
  → 保留 curved map/Jacobian/normal，weak/strong/hypersingular 分别认证
```

`q={8,10,12}` 的
\(\delta_1=\|I_{10}-I_8\|\)、\(\delta_2=\|I_{12}-I_{10}\|\)
只能报告 Cauchy tail。即便 \(\rho=\delta_2/\delta_1<1\)，
\(\delta_2/(1-\rho)\) 也依赖“未来误差比不超过 \(\rho\)”这一额外未证假设。
严格 q-GO 只有两条：

1. topology/kernel-specific 的可计算上界 \(U_q\) 低于预注册容差；或
2. validated/interval quadrature 给出足够窄的 enclosure。

两者都没有时，q 轴是 `UNRESOLVED`。浮点平台也必须由更高精度、directed rounding
或独立求和顺序给出的 roundoff envelope 证明，不能由 \(\delta\approx0\) 自证。

### 2.3 Common-space 与守恒传递

Jiao–Heath 的 common refinement 和 Jaiman 等人的非匹配 FSI 研究表明，逐点或
nearest-neighbour transfer 会产生振荡、overshoot 和积分不守恒；曲面接口需要在
真实 overlap cells 上积分。Farhat–Lesoinne–Le Tallec 的虚功原则要求运动映射
\(u_a=H u_s\) 与载荷映射 \(f_s=H^T f_a\) 成对出现。mortar/dual-mortar 文献给出
非匹配空间的稳定约束和 inf-sup/rank 边界。

P1/P2/P3 的无偏比较应在同一冻结参考面 \(\Gamma_\star\) 上用交叉质量矩阵：

\[
\|u_i-u_j\|^2_{\Gamma_\star}
=u_i^T M_i u_i+u_j^T M_j u_j-2u_i^T B_{ij}u_j ,
\]

其中 \(B_{ij}\) 在两个离散曲面的 common-refinement intersection cells 上积分。
不能先把一方 nearest-neighbour 到另一方再算差。

nearest-neighbour 的一维反例已经足够：

```text
Γ=[0,1], donor P1: p(x)=x, nodes={0,1}
target nodes={0,0.25,1}
nearest-neighbour values={0,0,1}
```

其分片线性积分为 0.375，而真值积分为 0.5；因此连 P1 patch 和零阶矩都不守恒。
它只可作 broad-phase 搜索 seed、可视化或负对照，不能作为 provider。

## 3. ③ 判定：缺组成部分还是组成部分错

### 已确定的缺件

1. P1/P3 body potential、classified cut、material wake、transport 与 pressure；
2. 冻结 master reference surface 与跨阶 lift/pullback；
3. common-refinement/cross-mass 或稳定 mortar；
4. topology/kernel-specific quadrature error certificate；
5. M1 的独立曲面几何阶 \(\ell\) 与法向/Jacobian 误差账；
6. 各轴共同的自然范数、压力范数、条件数和局部误差观察器。

### 已确定的错件/不足

1. 把 q8/q10/q12 Cauchy tail 当 joint uncertainty upper bound；
2. 把“q-tail 收缩或浮点平台”作为充分 q-GO；
3. 若用 nearest-neighbour 做跨阶比较或载荷 provider，则方法本身错误；
4. 若场阶与几何阶同步变化后称为 p 收敛，则归因错误。

### 尚不能判定

- fixed-P2 residual 是否为真实 named-law obstruction；
- 是否缺 massless forming geometry；
- 是否需要具有片面质量、动量和 entrainment 的 VES；
- 新空间模型是否提升 Fig17/18/19。

这些都必须等待正式结果、严格复算、fresh independent audit，以及上述数值缺件
补齐后的 held-out 证据。

## 4. ④ 机理方案：后续预注册边界

### 4.1 Phase HP0：同一弱式 family

P1/P2/P3 必须满足同一离散命题：

\[
B_\Gamma(\mu_h^{(p)},v_h)=F_\Gamma(v_h),
\qquad \mu_h^{(p)},v_h\in V_h^{(p)},\quad p=1,2,3.
\]

以下量跨 p 冻结：

- \(B_\Gamma\)、RHS、gauge 和 Kutta/formation 状态；
- body/cut/wake 拓扑与 side ownership；
- 连续/不连续语义、tip zero 与 mirror parity；
- material chronology 和 half/full same-stage；
- pressure observable 与唯一 ForceLedger 边界。

每个 p 先独立通过 manufactured/analytic oracle；P3 失败时先分解
discretization、quadrature、geometry 和 conditioning，不得改气动模型。

### 4.2 Phase HP1：common-space

必须同时通过：

- `reference_surface_identity`；
- `overlay_accounting`：intersection cells 无缺失、无重复、定向一致；
- `mixed_mass_quadrature`：平面 P3×P3 至少精确覆盖 degree 6；曲面还需
  Jacobian q-certificate；
- `polynomial_patch_P0_P1_P2_P3`；
- `moment_conservation`：总量与一阶空间矩；
- `symmetric_common_norm`；
- `adjoint_virtual_work`；
- `mortar_inf_sup_rank`；
- `discontinuity_ownership`；
- `no_nearest_neighbour_provider`；
- `transfer_error_below_hp_ball`。

### 4.3 Phase HP2：四轴分离与 q 认证

- h scan：固定 \(m,\ell,\Delta t\)，且每层 q 已认证；
- p scan：固定 h、同一几何/ownership、\(\ell,\Delta t\)；
- geometry scan：固定 h、\(m,\Delta t\)，只改 \(\ell\)；
- q scan：固定 h、\(m,\ell,\Delta t\)，按 pair topology 与 kernel 分账；
- dt scan：固定 h、\(m,\ell,q\)，保持 same-stage observer。

对两个空间 observable 的差
\(D=\|O_a-O_b\|\)，只有

\[
D>U_{q,a}+U_{q,b}
\]

时才称该空间变化已从积分噪声中解析；否则为 `UNRESOLVED`。三层 h/p 在扣除每层
\(U_q\) 后仍需严格收缩；否则不做 Richardson 外推。

### 4.4 结果路由

| 证据 | 后续动作 |
|---|---|
| 正式任一 protocol check 失败 | 回到对应 observer/数值节点；不执行 h/p |
| 正式 contract 内有 witness，但 q 无严格证书 | 记录 fixed-P2 evidence；claim 不晋升，先建 q certificate |
| witness 随合格 h/p/geometry 消失 | 数值/observer 组成错误；停止 forming/VES |
| witness 在合格 continuum ball 后持续 | 才允许 boundary-specific massless forming |
| massless forming 后仍有稳定 cokernel，且独立场证据给出质量/动量/entrainment 缺口 | 才允许打开 VES diagnostic candidate |
| 任一阶段靠 L/T/Fig 选方法 | PROTOCOL-NO-GO |

## 5. 文献库存与不可外推边界

本轮由三个只读分片分别审计 high-order/curved BEM、singular/close quadrature 和
common-space/conservative transfer；主代理合并去重。标准本地
`papers/`、`literature/` 路径不存在，本轮使用外部一手来源。核心来源如下：

| 来源 | 本轮使用的窄义命题 |
|---|---|
| Stephan & Suri 1989, DOI `10.1090/S0025-5718-1989-0947469-5` | 同一 Galerkin BEM 的 p-version 自然范数 |
| Babuška–Guo–Stephan 1990, DOI `10.1002/mma.1670120506` | 指数 hp 依赖角点附近几何加密 |
| Heuer–Maischak–Stephan 1999, DOI `10.1007/s002119900082` | 3D 开曲面边角奇异与 graded hp |
| Bespalov–Heuer 2010, DOI `10.1093/imanum/drn052` | 准均匀网格受边缘正则性限制 |
| Dölz et al. 2018, DOI `10.1016/j.cma.2017.10.020` | 精确 NURBS geometry 与高阶 field 分离 |
| Faria–Marchand–Montanelli 2025, arXiv:2507.13955 | \(m/\ell\) 分离；理论明确不含求积误差 |
| Cattarossi et al. 2026, DOI `10.1016/j.camwa.2026.01.021` | 任意阶 CAD-aware lifting-surface BEM 工程先例 |
| Sauter–Schwab 1997, DOI `10.1007/s002110050311` | touching Galerkin pair 类型化 composite quadrature |
| Erichsen–Sauter 1998, DOI `10.1016/S0045-7825(97)00236-3` | 按相交几何与目标精度自动 cubature |
| Reid–White–Johnson 2015, DOI `10.1109/TAP.2014.2367492` | Taylor–Duffy 适用 touching planar，不覆盖 nearly coincident |
| Montanelli–Aussal–Haddar 2022, DOI `10.1137/21M1462027` | curved weak/near-singular preimage-aware quadrature |
| Montanelli–Collino–Haddar 2024, DOI `10.1137/23M1605594` | curved strong/near-singular extension |
| Khatri et al. 2020, DOI `10.1016/j.jcp.2020.109798` | close evaluation 误差随距边界恶化 |
| af Klinteberg–Tornberg 2017, DOI `10.1007/s10444-016-9484-x` | 用可计算 error estimate 选 close evaluator |
| Jiao–Heath 2004, DOI `10.1002/nme.1147` | common-refinement 保守传递 |
| Jaiman et al. 2005/2006, DOI `10.1002/nme.1434`、`10.1016/j.jcp.2006.02.016` | 非匹配/曲面接口的精度、稳定性与能量 |
| Farhat–Lesoinne–Le Tallec 1998, DOI `10.1016/S0045-7825(97)00216-8` | 运动/载荷伴随与虚功守恒 |
| Wohlmuth 2000, DOI `10.1137/S0036142999350929` | dual mortar 稳定离散 |
| Klöppel et al. 2011, DOI `10.1016/j.cma.2011.06.006` | dual mortar FSI |

这些来源都**没有证明**：

- moving-body/material-wake 的 FLUXV 时间收敛；
- same-stage unified Bernoulli pressure 已闭合；
- Kutta birth、two-front/base-confluence 或 VES necessity；
- 任一固定理论阶可以直接成为 M0/M1 硬阈值；
- Fig17/18/19 精度一定提升。

## 6. 本轮 non-claims

- 没有执行 h/p、forming、VES、三点、118 或 Fig17/18/19。
- 没有实现 P1/P3、曲面高阶、common-refinement 或严格 q certificate。
- 没有修改任何气动公式、常数、网格、运动学或 claim 状态。
- 没有否定正在运行的 formal contract；只限制其结果可以支撑的科学外推。
- 没有授权 nearest-neighbour、经验 \(\eta\) 阈值、q-tail remainder 或总力拟合。

