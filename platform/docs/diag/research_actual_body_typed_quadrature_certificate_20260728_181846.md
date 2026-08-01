# Actual-body 类型化求积证书与 verified-solve 传播：研究裁决

**生成时间**：2026-07-28T18:18:46+08:00  
**状态**：`RESEARCH EVIDENCE / EXECUTION=false / PRODUCTION=false / CLAIM_STATE_CHANGE=false`  
**技能**：`research-lit`  
**范围**：S3ai-v2.2 fixed-P2 actual-body/body-wake 算子的数值可辨识性；不修改正式
one-shot、runtime、claim YAML、气动公式、常数或 Fig17/18/19  
**上游**：
`research_actual_body_hp_common_space_20260728_180800.md`、
`actual_body_hp_forming_ves_preregistration_20260728_173708.md`

## 0. 裁决摘要

本轮把“提高 quadrature order”改写为一个可证伪的端到端命题：

> 每个 actual-body/body-wake 面元对必须按
> `operator × kernel × topology × geometry × trial/test degree`
> 唯一分类，产生包含真积分的 enclosure；这些局部 enclosure 必须经 verified
> linear solve、同阶段压力和时间观察器传播，最终给出 pressure-law residual 的
> 严格上下界。任一必需面元对、浮点半径或传播步骤无证书，整个 continuum
> obstruction gate fail closed。

当前代码没有这条链。它已有若干窄义解析/多项式精确子算子，但 body–body 与
body–wake 的四维 Galerkin 面元对积分仍以 Gauss 结果的 Cauchy 差判断；随后
`np.linalg.solve` 只解已组装的浮点矩阵，没有把矩阵/RHS 积分不确定度传播到
\(\phi\)、\(g\)、压力和动态 residual。因此：

1. 这首先是**缺数值证书组成部分**，不是已证明缺 VES 或新气动力；
2. `q8/q10/q12` 仍可执行正式冻结合同，但只形成 fixed-P2 contract-internal
   evidence，不能充当严格 \(U_q\)；
3. 压力线和 P2 line mass 在当前 straight-edge/flat-P1 geometry 上不是主要
   truncation bottleneck；真正缺口是 Galerkin \(B/W/b\) 面元对、roundoff、
   verified solve 和时间误差分账；
4. 推荐路线是“三层证书”：解析/多项式精确层、类型化面元对 enclosure 层、
   verified system-to-observable 传播层；
5. 新 claim 只能先预登记为 `open` 的数值/campaign prerequisite；本文件不写
   YAML，也不授权生产。
6. 严格 \(U_q\) 只关闭求积/浮点/求解误差，不自动关闭 Galerkin 空间离散误差。
   P1/P2/P3 cross-space 差仍只是 Cauchy 证据；future continuum gate 还必须有
   可计算的空间离散误差上下界，或保持 `UNRESOLVED`。

## 1. ① 病因：数据指纹、claim 节点与可动空间

### 1.1 当前算子链

当前实际方程由 `actual_boundary_body_wake.py` 组装：

\[
A\phi=r,\qquad
A=B+W,\qquad
r=-b_{\mathrm{source}}-b_{\mathrm{known\ wake}} .
\]

- \(B\)：body double-layer Galerkin block，含 \(0.5M\) jump；
- \(W\)：body-test / material-wake-trial double-layer block，wake trace 已通过
  body/cut 关系消元；
- \(b_{\mathrm{source}}\)：Hess–Smith P0 source potential 经 P2 test 弱投影；
- \(b_{\mathrm{known\ wake}}\)：历史 wake 常量项；
- \(\phi\)：连续 P2 body potential；
- \(g\)：由 cut map、material history 和同阶段 observation 得到的 P2 trace；
- \(P\)：straight-edge P2 weak pressure；
- \(R=M(g^{n+1}-g^n)+\Delta t P^{n+1/2}\)：正式 active residual。

正式 q family 同时改变 body target/source 与 direct-W target/source 的
quadrature order；pressure line 固定为 q12。因此总量
\(\|R_{q=12}-R_{q=10}\|\) 无法定位是哪一种 kernel/topology 的误差。

### 1.2 算子级证据矩阵

| 算子 | 当前规则 | 已证明的窄义内容 | 未证明的内容 |
|---|---|---|---|
| `consistent_p2_line_mass` | 解析 `length/30` P2 mass | straight segment 上多项式截断精确 | 浮点 enclosure；curved arc |
| `_weak_active_pressure_from_state` | 1D GL q12 | P2 test × P2 speed-squared 为 P4，精确算术下 GL q≥3 精确 | 输入状态区间、roundoff、curved edge |
| Hess–Smith constant source | 解析边公式 | P0 source potential/jump 无数值面积求积截断 | edge/log 浮点 enclosure、外层 target test 积分 |
| P2 owner analytic sheet | boundary-vortex + linear area finite-part | 其严格适用域内的 on-sheet velocity identity | 完整 body-wake Galerkin \(B/W\)、nonowner/close pair |
| same/shared-edge/shared-vertex pair | 4D Taylor–Duffy-like transformed Gauss | partition 与 q-Cauchy sanity checks | 变换后积分 remainder、w→0 取消、roundoff |
| disjoint pair | target×source tensor-Duffy Gauss | separated/sphere 数值交叉核对 | disjoint-close 分类、距离下界、严格 Gauss remainder |
| direct independent W | 与 primary 路径同核/同规则重组装 | 代数 trace/factorization 一致性 | 共同 quadrature bias |
| `np.linalg.solve` | double precision | 已组装浮点系统的 backward residual | 连续算子误差与 forward enclosure |
| dt family | 三层 Richardson/Cauchy | 时间敏感性诊断 | 可计算时间 truncation bound |

特别需要闭环的实现—证明不一致：

- `paired_p2_triangle_integral` 对 identical face 与 shared edge 都走 6 个
  subregion；既有预注册文字曾写 identical/common-edge/common-vertex =
  3/6/2。partition test 通过不等于当前 6-region identical 规则已有来源证明。
- topology 目前主要由共享 vertex ID 判断。几何相交但 ID 不共享，以及
  disjoint-close，都进入 ordinary disjoint rule。
- 当前几何全部是 flat-P1 triangle；flat touching 证书不得外推到 curved
  touching/close。

### 1.3 数据规律

现有结果中的以下“GO”不是严格证书：

- q4/q6/q8 或 q8/q10/q12 Cauchy contraction；
- sphere 或 separated pair 与更高 q 的一致；
- direct-W 与 primary trace bitwise/near-bitwise 一致；
- `condition_number × componentwise_backward_error` 较小；
- 双精度和更高精度显示相同数字。

它们都是有价值的 sanity/oracle，但都不能证明真实积分落在一个给定半径内。
尤其两个实现若共享同一 quadrature rule，只能证明实现一致，不能发现共同 bias。

### 1.4 Claim tree 指纹

只读递归审计得到：

- 7 个 YAML 顶层、229 个唯一 claim；
- 当前 `ClaimDAG.nodes` 只治理 7 个顶层，222 个深节点未被递归索引；
- 222 个深节点中有 77 个 `validated/frozen`；
- 深节点只有 2 个带 guard、3 个带 runtime implementation；
- 缺 guard metric 当前会 fail open；
- runtime topology 会静默过滤不存在/未启用的 dependency；
- 单 run `ClaimGraph` 没有 campaign aggregation phase。

所以在正式 run 终止前不写新 YAML；终止后也必须先修复递归索引、唯一 ID、
typed edge、missing-guard failure 和 campaign graph，否则新增 d19 仍只是装饰
元数据。

### 1.5 可动空间

允许：

- 建立 typed pair registry 与 exact-cover guard；
- 对解析/多项式精确子算子生成 machine-radius certificate；
- 对 touching、disjoint-close、disjoint-far 分别生成 interval/ball enclosure；
- 用 Krawczyk/radii-polynomial 或等价 verified solve 传播矩阵/RHS区间；
- 在冻结 master surface 上建立 P1/P2/P3 common-space campaign；
- 将旧 q-tail 降格为 diagnostic，并保留其正式历史 provenance。

禁止：

- 拟合统一 \(q\)、\(\eta=d/h\) switch 或经验安全系数；
- 用 `abs_tol/rel_tol`、Gauss–Kronrod 差、q-tail 或裸高精计算自称严格界；
- 用 Fig17/18/19、L/T 或目标误差选择 topology rule、subdivision 或 tolerance；
- 把 flat-P1 certificate 外推到 curved geometry；
- 在 q/solve/time error 未分账前授权 forming、VES 或生产模型。

## 2. ② 学科机理：一手文献裁决

| 文献 | 类型 | 方法与结论 | 对 FLUXV 的授权 | 明确限制 |
|---|---|---|---|---|
| Sauter & Schwab, 1997, *Numerische Mathematik*, DOI 10.1007/s002110050311 | 同行评审 | hp-Galerkin BEM 的 topology-aware quadrature 与一致离散误差分析 | touching pair 必须与 p、几何和核共同设计 | 不是当前代码的自动严格 enclosure |
| Erichsen & Sauter, 1998, *CMAME*, DOI 10.1016/S0045-7825(97)00236-3 | 同行评审 | 为 3D Galerkin BEM 给出面向精度的自动 cubature order 分析 | far/near 路由应由可计算误差条件驱动 | 仍需把论文假设映射到当前核、几何和浮点实现 |
| Reid, White & Johnson, 2015, *IEEE TAP*, DOI 10.1109/TAP.2014.2367492 | 同行评审 | generalized Taylor–Duffy 降维处理共享顶点/边/面三角形乘积积分 | 可作为 flat touching evaluator 的解析正则化骨架 | 不覆盖 nearly coincident disjoint triangles；数值 convergence 不等于 interval proof |
| Montanelli, Aussal & Haddar, 2022, *SIAM JSC*, DOI 10.1137/21M1462027 | 同行评审 | curved weak/near singular：closest preimage、subtraction、continuation、transplanted Gauss | 未来 curved/close evaluator 必须保留 map/Jacobian/normal | 数值精度实验不是当前 P2 double-Galerkin 的严格 Uq |
| Montanelli, Collino & Haddar, 2024, *SIAM JSC*, DOI 10.1137/23M1605594 | 同行评审 | curved strong/near singular 的正则化算法 | hypersingular/strong observer 必须单独认证 | 不可用 weak-kernel certificate 代替 |
| Adelman, Gumerov & Duraiswami, 2016, *IEEE TAP*, DOI 10.1109/TAP.2016.2546951 | 同行评审 | separated pair 用 multipole/local expansion，touching 用 scaling/symmetry 与递归分解 | 可作为独立 production/oracle 候选 | 论文误差常数含经验选择，不能在本项目无锚使用 |
| Gumerov, Kaneko & Duraiswami, 2024, *SIAM JSC*, DOI 10.1137/23M1547688 | 同行评审 | flat triangle、constant density 的 Laplace layer Galerkin double integral 解析降维 | 支持“解析递归优先”的方向与 P0 oracle | 只覆盖 constant density，不能直接宣称 P2/P2 已解析 |
| Rump, 2010, *Acta Numerica*, DOI 10.1017/S096249291000005X | 同行评审综述 | 用 directed rounding/interval/Krawczyk 得到浮点可验证结果 | 矩阵/RHS enclosure 后必须 verified solve | 不替代 BEM-specific integral enclosure |
| Johansson, 2017, *IEEE Transactions on Computers*, DOI 10.1109/TC.2017.2690633 | 同行评审 | Arb midpoint-radius ball arithmetic | 可作为独立高精 ball oracle/实现后端 | 仅使用高 precision 而不保留 radius 仍不是证明 |
| Zalewski & Mullen, 2009, *IJNME*, DOI 10.1002/nme.2490 | 同行评审 | interval BEM 与 parametric interval linear solve 给出局部离散解界 | 支持 operator enclosure 后继续传播到解，而非只看 q-tail | 不是当前移动 body-wake/P2 pressure 的现成证书 |
| Kurz et al., 2021, *Numerische Mathematik*, DOI 10.1007/s00211-021-01188-6 | 同行评审 | functional a posteriori BEM estimates 给出 potential error 的可计算上下界 | 为 future \(U_h/U_p\) 提供比跨阶 Cauchy 更严格的研究方向 | 论文示例与直接证明范围不能直接外推到移动 wake、cut trace 或 surface pressure |

文献一致支持以下分工：

1. **正则化公式负责消除奇异结构；**
2. **类型化误差分析或 interval/ball arithmetic 负责 enclosure；**
3. **verified linear algebra 负责把 operator error 传播到状态；**
4. **物理 observer 只消费已认证状态，不自行重新拟合误差。**

文献不支持“选一个更大的统一 q 就视为解决”。

还必须区分两类误差：

- 本文件 Layer A–C 关闭的是 quadrature、roundoff、assembly、linear solve 和
  observable propagation；
- Galerkin 空间本身相对 continuum 的 \(U_h/U_p\) 需要独立 a priori 或
  functional a posteriori certificate。Kurz 等人的 potential-energy
  majorant/minorant 是可研究方向，但必须重新证明其对 moving exterior
  body-wake、cut discontinuity 和 surface pressure functional 的适用性。

因此三个 p 或三个 h 的数值收缩不能自行生成严格 continuum ball。

## 3. ③ 判定：缺组成部分还是组成部分错

### 3.1 缺失组成部分

1. typed pair registry：
   `operator/kernel/topology/geometry/p_test/p_trial/side/PV/finite-part`；
2. topology exact-cover、几何相交与 certified \(d_{\min}\)；
3. analytic/polynomial exact certificate 与 directed-rounding radius；
4. touching transformed-integrand interval enclosure；
5. disjoint-close/far 各自的可计算 remainder；
6. entrywise \(E_B,E_W,E_b\) 与来源 ledger；
7. verified solve 对 \(\phi,g,P,R\) 的传播；
8. 时间误差的独立严格账；
9. 空间 Galerkin discretization 的可计算 \(U_h/U_p\) 或明确 unresolved 状态；
10. run artifact 到 campaign common-space artifact 的不可变 provenance；
11. campaign execution phase 与 fail-closed claim governance。

### 3.2 已确定的错件

1. 把 q-Cauchy tail 或 \(\delta_2/(1-\rho)\) 命名为严格 error bound；
2. 仅以共享 vertex ID 代表完整几何 topology；
3. 未分类的 disjoint-close pair 使用 ordinary tensor quadrature；
4. 只报告浮点 solve backward residual，而未传播 operator integration error；
5. 若把 cross-run h/p campaign value 放进单 timestep `StepContext`，运行层级错误；
6. 若 forming authorization 作为物理 provider/ForceLedger 值，治理角色错误。

### 3.3 尚不能判定

- fixed-P2 formal witness 是否在 continuum limit 保持；
- named pressure-law 是否真实缺物理状态；
- massless forming geometry 是否足够；
- VES 是否必要；
- 空间涡态—统一压力是否提升 Fig17/18/19。

这些结论均不得由本案卷提前写入 claim 状态。

## 4. ④ 机理方案：三层端到端证书

### 4.1 Layer A：解析/多项式精确证书

每个证书必须分别报告 mathematical truncation radius 与 machine radius：

```text
operator_id
geometry_sha256
applicability_predicate
algebraic_degree_or_identity
truncation_radius = 0
roundoff_ball
independent_oracle
```

第一批应覆盖：

- straight P2 line mass；
- straight-edge P4 weak pressure moment；
- flat owner self double-layer PV=0 的严格适用域；
- Hess–Smith P0 source potential/jump；
- P2 boundary-vortex 与 linear-area finite-part 的已有窄义解析域。

“truncation radius=0”不等于浮点 radius=0。需要 directed rounding、ball
arithmetic 或经过证明的 error-free/compensated primitive 给出 machine radius。

### 4.2 Layer B：类型化面元对 enclosure

每个 required pair 先冻结：

```text
operator_id, geometry_sha256,
target_face_id, source_face_id,
kernel_id, singularity_class,
same_face/shared_edge/shared_vertex/disjoint_close/disjoint_far,
flat/curved, side/PV/finite_part,
p_test, p_trial, rule_id, rule_version
```

输出：

\[
I_{ij}\in[I_{ij}]=\widehat I_{ij}\pm e_{ij}.
\]

路由：

- **flat touching**：按 same/edge/vertex 选 Sauter–Schwab 或
  Taylor–Duffy 正则化；先代数消去 \(w\to0\) 的共同因子/0÷0，再对 regular
  integrand 自适应 subdivision + interval/Taylor-model enclosure；
- **disjoint far**：严格三角形距离下界、shape regularity 与核导数界共同给出
  Gauss remainder；只有证明 \(U_{ij}\le budget_{ij}\) 才走 far rule；
- **disjoint close**：closest preimage/距离变换、singularity subtraction、
  continuation 或独立解析递归，再用 interval/ball enclosure；
- **curved touching/close**：保留 map、Jacobian、normal；weak、strong、
  hypersingular 分别认证；当前 flat path 一律 NO-GO；
- **未知/相交/距离无法认证**：NO-GO，不回退普通 q。

所有 pair 形成 exact cover；同一个 pair 不能被重复簿记，也不能遗漏。局部
enclosure 无取消地组装为：

\[
|B-\widehat B|\le E_B,\quad
|W-\widehat W|\le E_W,\quad
|b-\widehat b|\le E_b .
\]

### 4.3 Layer C：verified solve 与 observable 传播

设

\[
\widehat A=\widehat B+\widehat W,\qquad
|A-\widehat A|\le E_A=E_B+E_W ,
\]

近似解为 \(\widehat\phi\)，近似逆为 \(R_A\)。实现必须用 interval Krawczyk、
radii polynomial 或等价 verified solver，而不是把下面的条件当浮点启发式。
设中心 RHS 为 \(d_c\)、其分量半径为 \(e_d\)，一个可执行的分量半径骨架为：

\[
K_A=|I-R_A\widehat A|+|R_A|E_A ,
\]

\[
e=|R_A(d_c-\widehat A\widehat\phi)|
  +|R_A|\left(e_d+E_A|\widehat\phi|\right).
\]

必须以 directed arithmetic 验证以下二者之一：

- 某个 induced norm 的严格上界满足 \(\|K_A\|<1\)；或
- 一个经过验证的 spectral-radius upper bound 满足
  \(\overline\rho(K_A)<1\)。

仅计算浮点 \(\rho(K_A)\) 不够。通过后，以 verified nonnegative solve 求

\[
r_\phi=(I-K_A)^{-1}e ,
\qquad
\phi\in[\widehat\phi-r_\phi,\widehat\phi+r_\phi],
\]

或直接验证 Krawczyk inclusion \(K(X)\subset\operatorname{int}(X)\)。
任一系统无法验证则该 history `CONDITIONING-NO-GO`。

之后逐状态传播：

\[
[v]=[c]-[G_v][\phi],\qquad
[P]=\mathcal B([v_{\rm upper}],[v_{\rm lower}],[g]),
\]

其中 straight-edge P4 line moment 可用 Layer A 精确积分，但速度/trace 的
interval dependency 必须保留。material history 的每一个 half/full stage 都
继承前一 verified enclosure，禁止只在最后一步补误差半径。

动态 residual 使用带几何/浮点 enclosure 的 \(M_{\rm dyn}\)：

\[
[R]=[M_{\rm dyn}]([g^{n+1}]-[g^n])
    +\Delta t[P^{n+1/2}] .
\]

用来定义 dual norm 的 \(M_{\rm metric}\) 必须是另一个冻结、经验证 SPD 的参考
metric。它可以与 \(M_{\rm dyn}\) 具有同一解析定义，但证书角色必须分开：

- `[M_dyn]` 参与 residual interval propagation；
- `[M_metric]` 必须有 verified SPD/eigenvalue 与 inverse/Cholesky enclosure；
- 不能把二者的浮点 midpoint 当作 exact 后共用。

对 active \(n=7\) residual box：

- 若 \(Q=M_{\rm metric}^{-1}\) 是一个固定、精确定义且已验证的 SPD metric，
  上界可枚举 \(2^7\) box vertices；下界可用 box-constrained convex QP；
- 若只有 interval inverse \([Q]\)，必须对
  \([R]\times[Q]\) 做 validated quadratic optimization，或用
  \(Q_c\pm\epsilon_Q I\) 的经验证 spectral enclosure 修正上下界；其中
  lower-side \(Q_c-\epsilon_Q I\) 必须 verified PSD/SPD，否则下界只能使用
  validated global optimization，或保守退回 0；
- 未计入 metric radius 时，128 个顶点和普通 QP 都只能叫 diagnostic；
- interval dependency 只允许使界更保守，不能事后缩半径；
- zero/odd 对照的区间必须含 0；even witness lower bound 必须严格大于 0。

### 4.4 时间误差单独分账

正式 dt 三层继续执行冻结 contract，但对 future continuum gate：

- Richardson/Cauchy 只报告 diagnostic；
- 严格 \(U_{\Delta t}\) 需 validated time integrator、可计算高阶导数界或
  另一个有证明的 local defect enclosure；
- q、h、p、geometry \(\ell\)、dt 五个半径分别报告，禁止用一个联合经验球掩盖。

### 4.5 预算规则

不得先看最终 witness 再分配误差预算。执行前冻结：

\[
U_{\mathrm{num}}
=U_q+U_{\mathrm{solve}}+U_{\mathrm{round}}
+U_h+U_p+U_\ell+U_{\Delta t}+U_{\mathrm{transfer}} .
\]

局部 pair budget 可由矩阵/observable 灵敏度做保守预分配，但不得由 Fig/L/T
误差拟合。若某条昂贵 rule 无法在预算内签证，应报告成本/NO-GO，而不是放宽
容差。

## 5. 新 claim 预登记草案（不写 YAML）

建议在 `N3.1j3b6d` 下新增 sibling group
`N3.1j3b6d19 actual-body h/p 数值可辨识链`，不扩大 validated/frozen 的现有
P2 节点。

| ID 简称 | 状态/角色 | execution phase | 命题 |
|---|---|---|---|
| d19a | open；numerical prerequisite | run | 每个 actual operator pair 都有唯一类型与可计算 enclosure，并传播成 \(E_B,E_W,E_b\) |
| d19b | open；audit family member | run | P1/P2/P3 只改变 field order，共享弱式、几何、拓扑、history 和 pressure；每个 member 绑定 d19a certificate 与 family-contract hash |
| d19c | open；observer | campaign | 消费不可变 run manifests，用 common-refinement/cross-mass 或稳定 mortar 在冻结参考面比较 |
| d19d | open；decision gate | campaign | 只有 h/p/geometry/dt/q 全分账、空间误差有严格 certificate 且 continuum even lower bound>0，才授权 boundary-specific forming 研究 |

d19c/d19d 不是逐时步物理节点，不能写入 `StepContext` 或 `ForceLedger`。
`forming_branch_authorization` 是治理/调度 artifact，不是气动力输入。

建议 typed claim edges：

- `requires_claim`：逻辑前置；
- `evidence_from`：窄义证据，不自动继承状态；
- `supersedes_scope`：只限制未来解释范围，不改写历史事实；
- `execution_phase: run|campaign`；
- runtime/campaign `requires/provides` 分离。

旧 q-tail 规则只在“future continuum/physical authorization”范围内被 d19a/d19d
取代；正式 v2.2 contract 仍按其冻结语义完成，validated/frozen 节点不降级。

## 6. Fail-closed guards

### d19a

- pair registry exact cover、无重复 provider；
- topology、kernel、PV/finite-part、side、geometry、trial/test typing 完整；
- triangle self-intersection/near-intersection 检测；
- certified \(d_{\min}\) 与 shape-regularity；
- touching/close/far 各自有 enclosure；
- transformed \(w\to0\) cancellation 已解析；
- analytic identity 与 roundoff ball 分账；
- independent high-precision ball oracle 包含 production center；
- 独立高精 ball oracle；
- local→matrix/RHS→solve→observable 传播闭合；
- `M_dyn` residual enclosure 与 `M_metric` verified-SPD norm certificate 分离；
- wrong-topology、underintegration、shared-bias negative controls；
- `q_tail_not_bound=true`；
- 无 Fig/L/T target selection。

### d19b

- weak-form/RHS/gauge/Kutta hash 跨 p 相同；
- master surface、topology、cut/side/tip/mirror、chronology相同；
- field p 与 geometry \(\ell\) 独立；
- P1/P2/P3 manufactured/analytic oracle；
- rank、condition 与 verified solve；
- same-stage pressure；
- 每个 history 绑定 d19a certificate；
- P1 geometry 不得冒充 P1 field。

### d19c

- reference-surface identity；
- overlay exact accounting/orientation；
- \(B_{ij}=B_{ji}^T\)；
- mixed-mass quadrature certificate；
- P0–P3 patch；
- total 与 first-moment conservation；
- self-zero/symmetric common norm；
- roundtrip/idempotence；
- adjoint virtual work；
- mortar inf-sup/rank；
- discontinuity ownership；
- nearest-neighbour provider 禁止；
- transfer error 不超过 h/p uncertainty ball。

### d19d

- formal provenance、31 histories、22 fields 与 15 checks 先通过；
- fresh independent formal-result audit；
- 所有 campaign levels 均 q/solve/time certified；
- `M_metric` 的 SPD、inverse 和 quadratic-bound radius 已验证；
- \(U_h/U_p\) 来自适用域已证明的 a priori/a posteriori certificate；仅跨阶差则
  `UNRESOLVED`；
- h/p/\(\ell\)/dt/q 独立三层；
- 差异只有在 \(D>U_a+U_b\) 时 resolved；
- joint finest cube、conditioning、repeat 与 cross-observer；
- zero/odd lower interval 含 0；
- continuum even lower bound 严格大于 0；
- no post-hoc rule/tolerance/state selection；
- 任一失败即 `NO-GO/UNRESOLVED`，不授权 forming/VES。

## 7. 实施优先级与 go/no-go

### Q0：治理与 registry（正式 one-shot 终止后）

1. ClaimDAG 递归索引 229 个唯一 ID；
2. missing guard metric → FAIL；
3. typed evidence/runtime/campaign edges；
4. runtime missing dependency → explicit error；
5. 建立只读 `ClaimCampaignGraph` 或等价 campaign artifact validator；
6. 建立 pair registry 与 topology exact-cover dry run。

**GO**：所有现有 claim 身份不变，229/229 受治理，旧正式 artifact 可重放。  
**NO-GO**：任何深节点遗漏、依赖静默过滤或 guard fail-open。

### Q1：先签发零 truncation 子算子证书

实现 P2 mass、P4 pressure line、Hess–Smith 与已有 analytic-sheet 适用域的
ball certificate；用独立高精 ball 后端交叉核对。

**GO**：每个 certificate 同时含 applicability、zero truncation proof 与
nonzero machine radius。  
**NO-GO**：仅输出双精度值或“与 q48 相同”。

### Q2：M0 flat touching/far/close pair certificate

先在小型 closed diamond 上签发完整 \(B/W/b\) pair ledger；touching 与
disjoint-close 分别认证。DIRECTFN/高 q 可作独立 numerical oracle，但不能作
唯一证书。

**GO**：每个 required pair 有 enclosure，组装 interval exact-cover。  
**NO-GO**：任一 pair 回退 ordinary q、未知 topology 或经验 \(\eta\)。

### Q3：verified solve-to-residual

将 interval \(B/W/b\) 传播到一个短 history 的 \(\phi/g/P/R\)，通过
zero/odd/even manufactured controls。

**GO**：verified solver 收缩、对照含 0、人工注入 witness 被下界检出。  
**NO-GO**：只报告 backward residual 或 condition estimate。

### Q4：P1/P2/P3 common-space campaign

只有 Q0–Q3 通过才执行 d19b–d19d。common-space 差首先用于定位；随后必须建立
适用于当前 moving body-wake/cut/pressure functional 的空间误差 certificate，
或以严格 upper-bound audit 保持 `UNRESOLVED`，其后才能重判 fixed-P2
obstruction 是否 continuum-qualified。

### Q5：物理方向

只有 d19d continuum witness 通过，才进入：

```text
boundary-specific massless forming
→ 若仍有经认证 cokernel，再审 VES necessity
→ 同一 actual-surface Bernoulli pressure
→ 图级 ForceLedger 一次成力
→ 三点、118、Fig17/18/19
```

## 8. Hard nonclaims

本文件没有证明：

- 正式 31-history result 已完成或通过；
- 当前 fixed-P2 witness 是物理 obstruction；
- 当前 q12 数值错误；
- P1/P3 已实现；
- curved geometry 已认证；
- h/p cross-space 差已经形成严格 continuum error bound；
- massless forming 或 VES 必要；
- 新模型已实现；
- 三点、118 或 Fig17/18/19 已运行；
- 任何 claim 可晋升；
- 任何气动参数可调整。

## 9. 正式运行快照

在 2026-07-28T18:17:53+08:00：

- PID 922252 仍为 `Rsl`；
- elapsed `01:58:05`；
- 24 threads，aggregate CPU 约 1799%，major faults=0；
- canonical result 不存在；
- latest result 不存在；
- formal log 为 0 bytes。

本案卷及其 JSON 不属于 frozen formal input。正式结果仍必须按原 interpretation
contract 复算，且不得用本文件事后重写其 fixed-space decision。
