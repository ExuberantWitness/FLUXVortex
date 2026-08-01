# N2.6e1bc 预登记执行前独立审计

日期：2026-07-30  
审计对象：`N2.6e1bc-SVI-WUK-WPJ`  
裁决：`DIRECTION CONDITIONALLY ADMISSIBLE / PREREG NOT IMPLEMENTATION-READY`

## 1. 审计身份

审计在 A0 正式结果产生前完成。被审快照为：

- `research_n26e1bc_weak_uk_decision_20260730.md`：
  `f9fe769dc44947c27f22ef52f701af3cb4d3fe0645594e73010df7889a5140d0`
- `n26e1bc_strong_vi_weak_uk_prereg_20260730.md`：
  `05bf121a7736fb133206fbcfae9bd41f682a1d828fdb934414b65b8ea96a1d8b`
- `claim_nodes/n2_kirchhoff.yaml`：
  `3b158b3e076fbbd0ea377cfaa4404466e540ad7cb97989c3c77fe68d8541a420`

审计与候选属于同一研究家族，因此独立性标记为 `same-family`；
结论只授权修订预登记，不提供物理验证。

## 2. 已确认的病因边界

`N2.6e1b1` 的 raw TE trace 不收敛与有限实体角下一 Laplace 模态高度
一致，且积分出生环量明显比端点迹稳定。这足以否定“继续加密最近点
trace”作为出生状态，但尚不能单凭两级百分比把所有误差唯一归因于该
角点模态。后继表述必须使用“高度一致”，并增加多级模态幅值/符号及
body--wake compatibility 检查。

候选文字没有直接重走 steady pressure-residual-only、删除满秩 BIE 行、
endpoint epsilon、双 Kutta closure 或载荷拟合路线；但当前定义缺失会
使实现实际退回这些已证伪路线。

## 3. MUST-FIX

### M1：Kelvin 必须使用规范不变的 wake circulation

冻结 `s_w` 从 TE 指向下游，定义

\[
\mu_w=\phi_w^- - \phi_w^+,\qquad
\gamma_w=\partial_{s_w}\mu_w .
\]

有向 edge `e=(i,j)` 沿 TE 到下游时，

\[
\Gamma_{w,e}=\mu_{w,j}-\mu_{w,i},\qquad
\Gamma_w^{sheet}=\sum_e(\mu_{w,j}-\mu_{w,i}).
\]

正确的 Kelvin 账为

\[
R_K=\Gamma_b+\Gamma_w^{sheet}
    +\sum_q\Gamma_{v,q}-\Gamma_{\mathrm{total},0}.
\]

其中远尾迹点涡另列；禁止 `sum_k(mu_w,k)`。该表达必须对
`mu_w -> mu_w+C` 不变。当前步新生 edge 为 `0 -> 1`，
`Gamma_birth=mu_1^{n+1}-mu_0^{n+1}`；旧 material 约束只覆盖保留的
`k>=1` 节点，不能冻结新生 `k=0`。

### M2：weak-UK 必须从明确控制体守恒重新推导

草稿

\[
\dot\Gamma_b+\mathcal J_{\omega}^{out}
-(p_L-p_U)/\rho=0
\]

尚未证明。修订必须逐段冻结 moving CV 的边界、朝向、局部涡量库存、
对流通量、壁面生成/黏性扩散和压力头项，并说明每一项是否已包含在
`Gamma_b` 或 wake birth 中。若 `J_omega` 直接取
`Gamma_birth/dt`，它只是 Kelvin 的时间差分，不能再作为独立闭合。

`J_omega` 必须来自独立 field/sheet weak provider；压力头必须来自同一
trial state 和冻结势历史的 Bernoulli provider，禁止 weak-UK backsolve。

### M3：控制体量必须可由状态观测

现有 `delta_star/theta/xi` 有限 IBL 矩不能唯一恢复二维尾缘
`omega/u/tau` 或控制体动量库存。修订只能二选一：

1. 增加具名、独立验证的 near-wall/profile/CV 状态；或
2. 把弱式和 kill guard 改写成可由现有 trace/sheet 弱变量直接计算、
   且不会由求根方程代数自证的泛函。

在此之前，`R_theta` 和“独立 CV force”不可作为可执行 A2 门。

### M4：冻结 Bernoulli 的 trial 角色和历史

Newton/DAE residual 内的 pressure head 必须在每个 trial state 上由同一
Bernoulli observation 计算；“统一压力只计算一次”只表示收敛后不再
添加第二套压力或力补丁。修订须冻结 inertial/body-frame 形式、势规范、
移动壁面项和时间离散。若 IBL/pressure 使用 BDF2，必须提供 `n-1,n-2`
或等价 stage history。

### M5：有限角弱空间和 body--wake compatibility 必须显式化

修订至少要冻结：

- 由实体角决定的 enriched/weighted potential 或 velocity basis；
- Kutta 消去的首奇异模态和保留的下一模态；
- weak test functional、quadrature 和自由度；
- newborn jump 与 body cut trace 的连续极限；
- `g-Cphi` 等 unforced compatibility、joint finite-part 或联合近场
  有界门。

这些是表示/兼容性守卫，不能暗中成为第二套同维 closure。

### M6：冻结完整 residual map 和物理工况

“每个 stage 至少包含”及 `R_M/R_E/R_xi` 占位符不可执行。v2 必须给出
完整未知向量、残差向量、维数、block ordering、body BIE、gauge、
wake-current unknown、唯一 closure slots、IBL 边界/停滞点处理、量纲
和 mass scaling。还必须冻结 `Re/U/rho/nu`、pitch pivot、运动窗、
初始攻角和时间格式。

### M7：空间、时间和 CV 轴必须独立

余弦网格的 terminal `h_TE` 约按 `N^-2`，而原 ramp `dt` 只按 `N^-1`；
同时取 `r_CV=2h_TE` 会使 `U dt/r_CV` 随加密增大，三种误差无法区分。
v2 必须分别冻结空间、时间、CV-radius 三条收敛轴，或保持明确的无量纲
比例不变，并在共同最细锚点交叉。

### M8：唯一性只限 reference-connected local branch

一条 continuation 和局部满秩 Jacobian 不能证明全局唯一根。v2 只可
声称预登记 tube 内的 reference-connected local uniqueness，并冻结
tube、初态、步长、局部唯一性半径或多初值/deflation 反证协议。

### M9：协议失败和物理证伪必须分开

- schema、实现、历史、非有限量或数值健康失败：
  `PROTOCOL/IMPLEMENTATION-NO-GO`，物理 claim 保持 `open`；
- 定义和数值健康门均通过后出现预登记物理反例：
  `PHYSICS-NO-GO`，该具体候选才 `falsified/frozen`；
- 所有物理门通过：
  才允许 `validated/frozen`。

### M10：claim DAG 语义必须闭合

`e1b/e1c provides_to e1bc` 与 `e1bc.depends_on` 不一致；已证伪的
point-provider 也不能成为新候选前提。应拆出“方程角色/材料历史定义”
prerequisite，再让 `e1bc` 依赖它。`e1d` 必须继承 integrated/weak
junction state，禁止恢复 `gamma1/gamma2` endpoint provider。

当前 `claim_dag.py` 只把 YAML 根节点放入 `ClaimDAG.nodes`，没有递归
验证 nested dependency、未知边和 cycle；在该守卫修复前，nested
`depends_on/provides_to` 只能视为未执行元数据，不能宣称已由 DAG 验证。

## 4. 当前执行边界

允许保留一个不选状态、不求解、不输出力的 typed formula oracle，用于
验证：

- 有向 edge Kelvin 的规范不变性；
- 新生节点与旧 material history 的身份分离；
- closed-CV、pressure 和 vorticity-flux provenance 的 fail-closed；
- 单位、符号和制造恒等式。

该 oracle 不是 A0 物理通过、不是 solver，也不授权 A1/A2。完成 M1--M10
并形成新的 v2 预登记后，才允许恢复最小求解实现。

## 5. 结论

研究方向保留为 `conditionally admissible`；原预登记被本审计取代并暂停
实现。下一步不是调常数或跑 Fig17/18/19，而是把唯一候选收缩为一个
可观测、无循环、规范不变且方程计数闭合的 v2。若无法做到，候选保持
`open/unresolved`，不能把实现失败伪写成物理证伪。
