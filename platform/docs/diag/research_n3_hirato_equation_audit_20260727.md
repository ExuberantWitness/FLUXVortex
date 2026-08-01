# N3.1i Hirato 空间涡片方程审计（2026-07-27）

## 0. 审计边界

本案只判断仓内 `lev_shed_mode='hirato'` 的两个实验放置分支是否构成 Hirato et al.
(2019) 的有限翼 LEV 涡片模型，并预登记下一实现。它不改变 V4.1 生产闭合、118 基线、
运动学、网格或任何 validated/frozen 节点。

审计对象挂在可动的 `N3.1i` 子树：

- `N3.1i0`：当前 `hirato+wake/ansari` 已忠实实现论文空间涡片；
- `N3.1i1`：方程忠实的有限翼涡片可生成面板压力；
- `N3.1h1`：有限矩只作总力影子诊断，不因本案获得面板载荷身份。

## 1. ① 病因：代码输出与“空间涡片”身份矛盾

### 1.1 规范数据指纹

Hirato Case 1（矩形翼 AR=6、SD7003、Re=20,000、`K=0.3`、0→45° pitch ramp），
`nc=4, ns=8, spc=120`，关闭 L-B、Polhamus/vnf、冲量、黏阻和压力帽，仅保留
UVLM、前缘吸力及待审 LEV 支路。以下 `CL` 顺序对应
`t*=[1.0,1.7,2.0,2.5,2.9]`：

| 分支 | `CL(t*)` | `CLmax/final` | 真实空间 LEV 状态 |
|---|---|---|---|
| LEV off | 2.244 / 3.529 / 4.174 / 4.166 / 4.465 | 4.478 / 4.478 | 0 |
| `hirato+wake` | 1.803 / 1.922 / 1.765 / 0.726 / 0.913 | 2.056 / 0.916 | wake LEV 952，`max|Γ|=0.10431` |
| `hirato+ansari` | 2.244 / 3.110 / 3.113 / 1.864 / 2.062 | 3.178 / 2.068 | **param ring=0，wake LEV=0** |

`ansari` 分支虽然显著改变 `CL`，但整个计算没有生成一枚持久 LEV 环。其变化只能来自
当前时步的隐式 LESP 约束和束缚环量重解，不能被解释为“空间涡片产生的面板载荷”。

直接病因在 `_v2_robo.py`：隐式求解已把脱涡条带的 post-solve `A0` 压到临界值，
随后持久片更新又用严格条件 `abs(A0)>a0_crit` 选条带，故选择集为空。正确事件身份
至少应沿用 pre-constraint `shed_m/hirato_A0pre`，但单独改这一行仍不足以恢复论文模型，
因为下面还有独立的方程缺失。

新增只读返回通道 `n3_hirato_audit` 对同一 120 步算例给出：

| 分支 | pre/post 脱涡条带事件数 | 持久片/非零 wake-LEV births | `max|遗漏 dGamma_L/dt|` | Eq.9 raw-sign 最大相对残差 |
|---|---:|---:|---:|---:|
| `ansari` | 507 / **0** | 0 / 0 | 7.295 | 30.6% |
| `wake` | 474 / 1 | 0 / 474 | 5.676 | 33.1% |

随后增加 `current/previous rearmost bound` 双通道，得到更窄而更强的数据指纹：
Case 1 probe 全时序

`max|Gamma_TEV-Gamma_bound,current| = 0`,

而

`max|Gamma_TEV-Gamma_bound,previous| = 1.156937e-2`.

因此历史 TEV kernel 的原始符号与当前后缘束缚环量完全一致；主错不是一个未经证明的
整体符号翻转，而是**时间索引使用 n 而非 n-1，并遗漏
`Gamma_L^(n-1)`**。论文—求解器符号边界仍须在 `hirato_exact` adapter 显式登记，
但不得再用“符号可能不同”掩盖已由数据分离出的时间账错误。pre/post 事件计数则直接
证明 `ansari` 的事件在持久状态入口被完全抹除。

### 1.2 规范输入与 Eq.6 观测算子审计

进一步核对发现，旧 Case 1/2 runner 标称 SD7003，实际 `real_geom=False` 调用的是
平板格。现已从 UIUC Airfoil Data Site 保存官方 `sd7003.dat`，先以几何前缘到上下
尾缘中点定义弦坐标作刚体平移、旋转和弦长归一化，再对上下表面取中弧线。这个变换
只恢复论文输入身份，不拟合任何气动力。

但四点隔离结果表明，平板不是早起涡的主犯。`nc=4, ns=16, spc=120` 下，把平板
换成 SD7003 只使 Case 1/2 的旧算子起涡各推迟 `0.025 t*`：

| 输入/观测算子 | Case 1 首次起涡 | Case 2 首次起涡 | 论文目标 |
|---|---:|---:|---:|
| flat + 旧 fixed-xref | 1.375 | 1.225 | 1.710 / 1.575 |
| SD7003 + 旧 fixed-xref | 1.400 | 1.250 | 1.710 / 1.575 |

真正的式级错误在 Hirato Eq.6 的离散配对。论文定义

`A0 = 1.13 Gamma_1 / [U_inf c (theta_1 + sin(theta_1))]`,

其中 `Gamma_1` 是最前方束缚环，`theta_1` 必须由**同一最前方格**的实际
`Delta x_1/c` 决定。历史分支虽然读取 `Gamma_1`，分母却固定使用
`xref=0.10c`。当前 `nc=4` 的实际前缘格宽为 `0.25c`，因此该错配把 LESP 大幅
放大并提前触发。论文还明确以 `U_inf` 归一化，而不是局部 `Urel_le`。

新增只读 `A0_eq6_paper/shed_eq6_paper/delta_x1_over_c` 后，在完全不改
`LESPcrit=0.27` 的条件下：

| 规范案 | Eq.6 首次起涡 | 时间误差 | 首次展向条带中心 | 预登记门 |
|---|---:|---:|---|---|
| Case 1, SD7003, 0° twist | `t*=1.650`, α=22.35° | −0.060 | 0.031…0.531，含根部带 | GO |
| Case 2, SD7003, +10° tip twist | `t*=1.500`, α=17.20° | −0.075 | 0.531 / 0.594 | GO |

这两个粗网格点通过实现前冻结的 `|Delta t*|<=0.10` 和展向误差 `<=0.125` 门，
但按预登记不能替代完整敏感性矩阵。由此先裁定：

1. SD7003 输入身份是必须修复的次级问题；
2. `Gamma_1` 与固定 `0.10c` 分母错配是当前起涡时刻的主病因；
3. 禁止改 `LESPcrit` 吸收该离散错误；
4. `N3.1i1a`（Eq.6 LESP 观测算子身份）可窄义晋升
   `validated/frozen`；
5. Case 1/2 的网格收敛拓扑另挂 `N3.1i1e`，完成全矩阵后裁决；
6. 这只验证 onset observer，不验证 Eq.7 后的涡片演化或压力，`N3.1i1`
   仍为 open。

### 1.3 网格反例与半翼自由尾迹边界

按 `hirato_canonical_cases.yaml` 预登记运行
`(nc,ns)=(4,16)/(8,24)/(12,32)`、`spc=120/240`、Case 1/2 共 12 点。第一轮
暴露：虽然 `sym=True` 在束缚 AIC 中镜像了另一半翼，但 TEV 尾迹对 RHS 的诱导和
自由尾迹对流均没有镜像另一半翼。该“不完整对称边界”下：

- Case 1 的时间随弦向细化接近论文，但首次起涡位置从粗格根部漂到
  `0.146–0.203` 半展长，违反 root-first 门；
- Case 2 的半展向拓扑仍保持。

这给出明确的数据指纹：缺失的不是 LESPcrit，而是有限翼自由涡场的另一半翼。新增
仅供 `hirato_probe` 的完整镜像 RHS/对流后，重新运行同一 12 点：

| grid | spc | Case 1 `t*/span error` | Case 2 `t*/span error` |
|---|---:|---|---|
| 4×16 | 120 | 1.625 / 0.031, GO | 1.500 / 0.031, GO |
| 4×16 | 240 | 1.600 / 0.031, **NO-GO** | 1.4875 / 0.031, GO |
| 8×24 | 120 | 1.700 / 0.021, GO | 1.575 / 0.021, GO |
| 8×24 | 240 | 1.6875 / 0.021, GO | 1.5625 / 0.021, GO |
| 12×32 | 120 | 1.725 / 0.016, GO | 1.600 / 0.016, GO |
| 12×32 | 240 | 1.7125 / 0.016, GO | 1.5875 / 0.016, GO |

完整镜像把 Case 1 的根部拓扑恢复并使中细网格稳定收敛；这支持“漏镜像自由尾迹是
根部漂移病因”。但 12 点中只有 11 点通过：`nc=4,spc=240` Case 1 的时间误差
`0.110`，比门限多 `0.010`。预登记规则不允许事后删除粗格或放宽容差，因此：

- `N3.1i1a` 只冻结 Eq.6 的公式身份；
- `N3.1i1e`（全网格时空拓扑）保持 partial；
- “中细网格是最低合格域”只能作为下一轮新假设另行预登记；
- 生产 N1 不因该只读 canonical adapter 被修改。

### 1.4 论文方程—当前代码逐项对照

| 论文必要项 | 论文定义 | 当前实现 | 裁决 |
|---|---|---|---|
| Eq.7 新环放置 | 在 LE 到上一 LEV 位置的 1/3 处切分上一环；旧环随后自由对流 | `wake` 近似构造新环但未证明拓扑切分；`ansari` 用 `fpos=0.45(1-exp(-2.2a))` 和 `h=lev_rollh*c*(a+0.06)` 重造几何 | 不忠实 |
| Fig.5 伪涡环 | 前缘到**最后方束缚环的后缘**，使 Eq.6 无需修改 | 使用 `cc0[:ns]`，即最前方束缚环后缘 | 错组件 |
| Eq.9 Kelvin | `Γ_W^n = Γ_b,max^(n-1) + Γ_L^(n-1)`，TEV 与 LEV 联立记账 | 普通 `shed_kernel` 精确复制当前 `Γ_b,max^n`；未使用 `n-1`，也未消费上一时步 `Γ_L` | 错时间索引 + 缺项 |
| Eq.10 边界条件 | `v_b+v_inf+v_m+v_W+v_L=0` | `wake` LEV 进入 RHS；`ansari` 明确只进 Bernoulli force、不进 RHS | `ansari` 违反 |
| Eq.16–17 势跳 | 活跃脱涡时 `φ_u-φ_l=Γ_L+Γ_b,x`，压力含 `ΔΓ_L/Δt+ΔΓ_b,x/Δt` | 面板压力只用 `(g-gprev)/dt`，没有 `ΔΓ_L/Δt` | 缺项 |
| Eq.23–24 卷起 | 所有自由环由 bound+TEV+LEV 当地速度对流 | `ansari` 按规定弦向年龄和高度函数放置，不是自由对流 | 错组件 |

“LEV 和 TEV 是否存储在同一个数组”本身不是物理错误；只要类型标签、相互诱导、
Kelvin 账和压力势跳正确，共享存储可以成立。旧案卷把 `wake` 的错误归结为“放进 TEV
数组”过于宽泛。真正错误是上述方程和账本没有闭合。

## 2. ② 学科机理

Hirato 模型不是“先算总 LEV 力再摊到翼面”，而是：

1. LESP 超临界决定新增 LEV 环量；
2. 可对流涡环通过 Biot–Savart 速度进入无穿透边界条件 Eq.10；
3. LEV 环量同时进入 Kelvin 条件 Eq.9；
4. 活跃前缘涡片改变翼面势跳 Eq.16，其时间导数通过 Eq.17 进入逐面板压力；
5. 面板压力积分得到法向力 Eq.18，前缘吸力另由 Eq.20–21 进入，最终按 Eq.22 旋转。

因此空间环几何、`Γ_L` 历史、TEV 联立账和压力势跳是一个不可拆散的组成部分。只实现
“把 A0 拉回临界值”，或只把某个参数化涡环速度加到压力式，都不是该模型。

Bird et al. 的 UVLM+规则化涡粒子方法与三维 vortex-force map 也支持同一边界：
总力需要空间涡结构；结构 co-design 所需的局部压力更不能由有限矩事后分配。

## 3. ③ 方向裁决

本案不是单一“缺组件”或“组件错”，而是两者并存：

- **已有实验组件错误**：当前两个 Hirato 分支违反或遗漏 Eq.7/9/10/17/23；
- **生产运行时缺组件**：V4.1 没有一个方程忠实、可输出逐面板压力的空间 LEV 节点。

裁决：

1. `N3.1i0 current_hirato_is_faithful = falsified/frozen`；
2. 禁止把 H14、`ansari` 参数片或 `wake` 分支直接晋升到 V4.1；
3. 不原地重定义旧开关，新增隔离的 `hirato_exact` 影子实现，旧分支仅保留历史复现；
4. 不复活已封存的全 rVPM `N3.1f`；有限翼环片仍是最贴合 N1 的首选。

## 4. ④ `hirato_exact` 预登记

### 4.1 必须实现的状态和方程

- 每条带保存当前/上一时步 `Γ_L`，以及自由 LEV 环的节点、强度、连接关系和类型；
- Eq.7 通过切分上一 LEV 环构造新环，不用 `lev_rollh/fpos/lev_fmax` 规定形状；
- 伪涡环后缘使用最后方束缚环后缘；
- TEV 强度显式满足 Eq.9，并输出逐条带 Kelvin residual；
- 所有旧 LEV 环按 Eq.23–24 的完整当地速度自由对流；
- Eq.17 的 `ΔΓ_L/Δt` 与 `ΔΓ_b/Δt` 分通道进入每个脱涡条带的面板压力；
- 力只来自面板压力和一次前缘吸力；`lev_vnf=False`、`lev_impulse=False`，无总力归一化；
- 输出面板 `dp/Cp`、环几何、`Γ_L`、涡心、二阶矩和拓扑，有限矩只作观测。

### 4.2 规范 GO 门

1. 每条带 Kelvin residual 相对主环量 `<1e-10`；
2. Eq.17 的 `dΓ_L/dt + dΓ_b/dt` 分解逐面板重组误差 `<1e-12`；
3. 脱涡后真实 LEV 环数大于零，旧环位置由局部速度推进，不含规定高度曲线；
4. Case 1 出现约 `t*=1.7` 的根部先起涡，并通过网格/时间步敏感性；
5. Case 2 扭转翼的起涡位置按论文向外展移动；
6. 面板力之和、总矩和压力中心与 ledger 一致，不与 N1/前缘吸力双计；
7. 先在规范算例通过，再以 `-90°` RoboEagle 只读 shadow 运行；未通过空间证据前不得接入
   118 生产扫。

规范输入和图注锚已冻结在 `hirato_canonical_cases.yaml`。其中：

- Case 1：AR=6、无扭无后掠，`t*=1.710, alpha=24.15 deg` 根部先起涡；
- Case 2：+10 deg tip twist、无后掠，`t*=1.575, alpha=19.51 deg` 在
  `|y|/(b/2)≈0.5` 先起涡；
- 共同条件：SD7003、Re=20,000、`LESPcrit=0.27`、根部四分之一弦轴、
  0→45 deg、`K=0.3`。

预登记数值带为 `|Delta t*|<=0.10`、起涡展向位置误差 `<=0.125` 半展长；这是实现前
冻结的离散化容差，不是文献物性常数。Case 1/2 的场拓扑门不能被总 `CL` 匹配替代。

### 4.3 NO-GO 门

- 只修正最终 `CL` 而 Eq.9/17 残差不闭合；
- 从 post-solve `A0` 丢失脱涡事件；
- 用规定 `rollh/fpos` 代替局部速度对流；
- 用 Polhamus/vnf、冲量总力或目标曲线归一化补足涡片压力；
- Meng 根部总力被用作面压力或空间涡拓扑验证。

### 4.4 当前实现进度：只读拓扑内核

已新增三个与 V4.1 力链隔离的模块/资产：

- `airfoil_geometry.py` + UIUC `sd7003.dat`：官方截面经几何弦坐标归一化得到
  SD7003 中弧面；
- `claim_runtime/hirato_equations.py`：Eq.6 LESP、Eq.9 Kelvin、Eq.17 势跳率、
  Fig.5 最后方束缚环后缘和 pre-constraint 脱涡事件的精确恒等式；
- `claim_runtime/hirato_shadow.py`：按 Fig.4/Eq.7 切分最近旧环、保留旧环强度、
  创建新环；以显式 ledger 分列 `freestream/bound/TEV/LEV` 的规则化
  Biot–Savart 当地速度，并用总当地速度自由对流。

shadow 明确不包含力、压力或第一枚环的经验位置。无旧环时，调用方必须显式提供
`first_aft_edges`；这是为了把“第一涡放置律”留作独立可验证命题，禁止把经验高度
悄悄写进空间状态。其 `core_radius` 同样只能由调用方提供，不拥有隐藏默认。当前单测
已覆盖：

1. SD7003 的几何前缘/尾缘弦坐标与中弧面身份；
2. Eq.6 使用同一 `Gamma_1/Delta x_1` 配对；
3. 伪涡环取最后方束缚环；
4. Eq.9 只有包含上一时步 `Gamma_L` 才闭合；
5. Eq.17 的 `dGamma_L/dt` 按条带广播到弦向面板；
6. post-solve LESP 不能恢复脱涡事件；
7. Eq.7 切分连接残差为零且旧环强度不变；
8. 所有自由顶点只按调用方提供的当地速度推进；
9. `freestream+bound+TEV+LEV=total` 的逐顶点速度账和半翼镜像；
10. `Gamma/centroid/covariance` 是空间状态的只读观测量。

这一步已证明 Eq.6 identity，并以 11/12 的矩阵证据支持完整半翼镜像场对 Case 1/2
时空拓扑的必要性；粗格反例使 `N3.1i1e` 保持 partial。数据结构和账本可以独立
实现；完整当地速度也已成为无力 reference，但**尚未证明**首生环适配、LEV 自身加入
后的场形或压力准确性，
故 `N3.1i1` 保持 open。

生产隔离守卫：修改后运行 `verify_v41_repro.py`，三点
`aoa=15/0/5` 的 `L/T` 最坏缓存偏差为 `0.100 N < 0.15 N`，通过；说明只读审计
没有扰动 V4.1 数值身份。

### 4.5 首生 LEV 环与核尺度：独立预登记

#### 4.5.1 病因和文献身份

Hirato Eq.7 规定的是：当已经存在上一枚 LEV 时，新分割点位于几何前缘到上一 LEV
位置的 `1/3` 距离处。论文和 Hirato dissertation 均没有赋予“第一枚 LEV =
`LE + U_infinity*dt`”这一身份。历史 `_v2_robo.py` 却在 `lev_first==1` 时直接使用
整步自由来流位移，故该组件不能由 Eq.7 辩护。

两项一手来源给出速度型首涡位置：

1. Ansari, Zbikowski & Knowles (2006) Part 2：若脱落边局部复速度为 `q`，
   第一涡使用半步位置 `delta z = 0.5 conjugate(q) dt`。把它推广到有限翼的候选
   `P-A` 定义为每个前缘端点
   `delta r = 0.5 v_edge,local dt`。复平面到物理三维边速度的等价性是 adapter
   假设，不冒充原论文恒等式。
2. Ramesh et al. 的 LESP-DVM：第一枚 LEV 在其二维坐标中采用
   `delta(x,z) = U_inf A0 dt / sqrt(2) [sin(alpha), cos(alpha)]`。有限翼候选
   `P-R` 只把这两个分量嵌入显式、正交的局部弦向/吸力面法向基；基的方向和
   `A0` 符号必须逐条带输出。

Hirato Eq.25 对 Lamb–Oseen cutoff 只规定
`r_c` 为预期最小涡环尺度的小比例、通常 `<0.5`，没有唯一 `r_c/c`。因此当前
`lev_roll_core=0.01c`、`lev_overlap` 或任何单一比例都不能因“跑稳了”成为物理常数。

#### 4.5.2 Claim 裁决

- `N3.1i1b1`: “首环固定为 `LE+U_inf*dt` 且由 Hirato Eq.7 支持”
  = `falsified/frozen`；
- `N3.1i1b2`: “速度型首涡律可忠实适配到有限翼环端点”
  = `partial`；`P-A/P-R` 都只是待裁决候选；
- `N3.1i1b3`: “固定单一核半径已经有文献身份”
  = `falsified/frozen`；数值核必须通过明确尺度族的收敛性审计。

#### 4.5.3 候选与不可调输入

同一 canonical Case 1/2、同一 SD7003、`LESPcrit=0.27`、完整半翼镜像和 Eq.6
observer 下，只比较：

- `P-A`: Ansari 半步局部边速度；
- `P-R`: Ramesh LESP 首涡二维式 + 显式局部弦/法向三维 embedding。

两者不得更改 `LESPcrit`、起涡时刻、`Gamma_L` 解、Eq.9、运动学或压力公式。核尺度
只取无量纲审计族

`r_c / ell_min in {0.10, 0.25, 0.49}`,

其中 `ell_min` 是该时步该候选实际最小涡环边长；这三个数是覆盖论文允许域的
**敏感性采样点**，不是拟合常数或晋升后的默认值。

#### 4.5.4 GO / NO-GO

公式级 GO：

1. `P-A` 必须逐点满足 `delta r - 0.5 v_edge dt = 0`；
2. `P-R` 必须先在二维基逐点满足已发表分量式，再通过显式正交局部基嵌入；
3. 任何未归一或非正交局部基、未知吸力面方向、NaN/零面积/翻转首环均立即失败。

空间级 GO：

1. 首环加入后 Eq.6 active residual `<1e-10`，Eq.9 residual 相对主环量
   `<1e-10`；不得以核正则化替代守恒；
2. 首环与翼面无穿透/相交，连接和环向一致，`Gamma_L` 不因几何候选被目标力缩放；
3. Case 1/2 的 onset 时刻和展向位置仍通过既有预登记门；
4. 对三个允许核比，首个事件后的 `Gamma_L`、LEV centroid、诱导速度和 LESP
   sensitivity condition number 全部报告；只有随网格/时间细化出现共同极限的候选
   才能进入 live shadow。不得事后挑一个最接近 `CL` 的核比；
5. canonical 场形比较先于 `CL/CM`，RoboEagle 总力不得参与 `P-A/P-R` 选择。

空间级 NO-GO：

- 固定 `LE+U_inf*dt` 被改名为“Ansari”；
- 把 `P-R` 的二维坐标分量未经基/符号审计直接写入全局 xyz；
- 因近奇异而添加未登记 offset、damping、pressure cap；
- 用 Meng 总力、Figure 17/18/19 或 V4.1 缓存选择首环位置或核尺度。

当前 `hirato_equations.py` 只实现 `P-A/P-R` 的公式级函数和显式二维到三维 embedding；
它们不接 V4.1、没有默认候选，也不生成力。

#### 4.5.5 首次 canonical 公式/几何结果

运行 `hirato_first_ring_guard.py --nc 8 --ns 24 --steps 120`，读取 Eq.6 probe 的真实
首次事件翼面姿态，但不插入 LEV、不解 `Gamma_L`、不算压力或力：

| Case | onset | 历史整步 `ell_min/c` | `P-A-kinematic` | `P-R` |
|---|---|---:|---:|---:|
| 1 | `t*=1.700`, strip 0–8 | 0.025000 | 0.011860 | 0.004791–0.004874 |
| 2 | `t*=1.575`, strip 11–15 | 0.025000 | 0.011793–0.011845 | 0.004774–0.004813 |

局部弦向—法向基的最大范数/正交残差分别为 `0` 和 `5.54e-17`，公式和非零环面积门
通过。但这一步暴露了一个耦合病因：历史 `lev_roll_core=0.01c` 对 `P-R` 首环给出
`r_c/ell_min=2.05–2.10`，不仅不是“小于 0.5”，甚至大于首环自身最小边。也就是说，
把首环改为文献尺度却沿用旧核下限会把正确几何重新抹平，属于“修一件、复活另一错件”。

方向裁决：

- 历史整步首环的几何尺度反例进一步支持 `N3.1i1b1 falsified/frozen`；
- `P-A-kinematic` 只验证半步公式能生成有限几何，因缺完整当地边涡速度和复平面—
  三维映射，标记 `partial-ineligible`，不得进入 live shadow；
- `P-R` 与 LESP-DVM 的状态身份直接相连，且不需要在奇异前缘另造一个未定义局部
  速度，因此成为下一步**唯一 implementation-ready 候选**；这不是 validated，
  仍须同时跑 `r_c/ell_min={0.10,0.25,0.49}` 的 Eq.6/9、场形和分辨率门；
- 不设置新的固定 `rc/c`，也不允许旧 `0.01c` floor 覆盖尺度族。

### 4.6 live shadow 前的第二轮离散方程审计

首环尺度缩小后重新通读 Hirato dissertation 的 UVLM 实现，发现不能把现有 TEV
几何和 Euler 卷起当作无关底层细节：

1. dissertation §4.5.3 明确把新 TEV 环沿尾缘扫掠轨迹放在
   `Delta x_W=0.3 U_infinity Delta t`；当前 `ug.shed_kernel` 使用
   `1.0 U_infinity Delta t`。该整步几何可以保留给历史 N1，但在
   `hirato_exact` 身份下为错件；
2. Journal Eq.23 给 `Delta x=v_r Delta t` 的概念式，紧接着 Eq.24 明确采用
   `Delta x=0.5(v_r^n+v_r^(n-1))Delta t`。现 `HiratoSheetShadow.convect`
   是 current-only Euler，只能算拓扑 scaffold，不能声称 Eq.24 已实现；
3. dissertation §4.8.2 把 `r_c` 解释为由离散分辨率决定的 singular radius，
   并指出小于半个面板尺度以下不应期待高于边界离散的精度。结合 Journal 的
   “anticipated smallest vortex-ring dimension”，最少假设的解释是：
   每个敏感性 run 先由预期全局最小环尺度确定一个**固定** `r_c`，而不是让核随
   每个环的瞬时变形改变。

因此原计划“把 P-R 接上现有 shadow 后直接自由对流”被否决。新增命题：

- `N3.1i1c1`: 全步 TEV offset 可冒充 Hirato 实现 = falsified/frozen；
- `N3.1i1c2`: current-only Euler 等价 Eq.24 = falsified/frozen；
- `N3.1i1c3`: 固定分辨率核族 + 每顶点历史当地速度 = open。

更新后的 live shadow 必须同时拥有：

- 自己的 TEV/LEV 自由场和 Eq.9 账，不能借用历史生产 TEV；
- `0.3 U_inf dt` 的 TEV 几何；
- P-R 首环与 Eq.7 后续切分；
- `freestream/bound/TEV/LEV` 当前当地速度账；
- 与每个 material vertex 对齐的上一时步速度；
- Eq.24 两级推进；
- `r_c/ell_min={0.10,0.25,0.49}` 三个固定-run 分支。

`ell_min` 在 run 开始前取 P-R 临界首环尺度
`U_inf LESPcrit dt/sqrt(2)`、最小展向边、`0.3U_inf dt` TEV offset 和最小束缚
面板尺度的最小值。该定义可随网格/时间步改变，但同一次 run 内不变；任何动态
core growth/diffusion 都属于另一个待研究组件，不能混入本轮。

### 4.6.1 新生环对流时序与 Eq.24 启动缺口

正式期刊与博士论文并排核对后，纠正了 reference shadow 的一个时间错件：

- Journal Fig.4(c) 把新环状态明确标为 “after LEV calculation, before LEV
  convection”；
- dissertation Fig.4.1 的每步顺序为新 LEV 求解、重新求束缚环量、计算力，
  随后在同一循环执行 TEV/LEV convection；
- 因此“出生步只记录速度、到下一步才位移”会人为制造一整步相位滞后，
  `N3.1i1c4` 记为 falsified/frozen。

同时，Journal Eq.24 的
`0.5*(v_r^n+v_r^(n-1))*dt` 对新生顶点和 Eq.7 新切分顶点缺少
`v_r^(n-1)` 的物质点定义；论文没有给出启动律。这里不能伪称文献已闭合。
无力 reference 预登记零参数 `P-E` 启动候选：

```text
有真实历史的旧顶点: 0.5*(v_now+v_previous)*dt
新生/重网格顶点:   v_previous := v_now，仅第一次退化为 v_now*dt
```

实现必须逐顶点输出 `bootstrap_vertex`，使这一适配与真正 Eq.24 历史区分。
`P-E` 只能由时间步收敛、Kelvin 和场拓扑门裁决，禁止用 Meng 总力或
Fig17/18/19 误差选择。它保持 `N3.1i1c5 partial`，不因公式单测而晋升。

### 4.6.2 Eq.25 身份纠正与 live shadow 反例

原 shadow 曾沿用生产 `vseg` 的 van-Garrel 型分母核。逐式核对期刊 Eq.25 后，
该身份被纠正：Hirato 使用无核有限线段 Biot–Savart 速度乘以
`1-exp[-(r/r_c)^2]`；`r` 是到涡丝直线的垂距。`r_c` 不进入
`|r1 cross r2|^2+r_c^2|r0|^2` 分母。NumPy 解析实现与 CPU Warp 逐通道实现已
互为 oracle；冻结 N1 AIC 和 V4.1 生产核未改动。

纠正 Eq.25 后，无力 canonical live shadow 仍出现比论文早得多的起涡：

- Case 1，`nc=4/ns=16/spc=120`；
- `r_c/(U_inf dt)=0.10/0.25/0.49` 均在约 `t*=0.75–1.125`
  越过阈值，而论文为 `t*=1.710`；
- 在尚无有意义 LEV 的 `t*=1.10–1.125`，TEV 当地速度从约 `68 m/s`
  增至 `123 m/s`，产生束缚解和 LESP 的非物理跳变。

这是一条有判别力的数据指纹：异常先发生在 TEV 自诱导/束缚近场，不是
`Gamma_LEV`、LEV 位置或后续压力项。因此禁止把 `LESPcrit`、首 LEV 位置或
`r_c` 调到论文起涡时刻。`0.3 U_inf dt` 是新 TEV 与尾缘的间隙，不等于唯一
最小面元尺度；更不能用未来 P-R 首 LEV 尺度从 `t=0` 改变 TEV 核。

Claim 裁决：

- `N3.1i1c6`：Eq.25 Lamb–Oseen 乘性截断身份 = `validated/frozen`；
- `N3.1i1c7`：常强度离散环可无附加模型稳定通过 canonical long-rollup
  = `falsified/frozen`；
- `N3.1i1c8`：dissertation §4.8.3 的涡片撞翼面/slip-wall 是缺件，
  但反例发生在 LEV 撞翼之前，故不能作为事后“修复” c7；
- `N3.1i1f`：Hirato 常强度环直接作为最终生产空间状态
  = `falsified/frozen`。方程/拓扑 reference 保留且不接入力。

### 4.7 方向判定：缺少高阶连续自由涡面

Hirato 论文自身把当前实现称为 proof of concept，并把稳健高阶涡片、长时多片
交互列为后续需求。新的高质量一手来源把病因收敛到同一组件：

1. Bramesfeld 的 DVE 用抛物展向环量和线性涡片强度构成连续尾迹，明确以消除
   流向离散线奇异及核尺度敏感性为目的；但论文/博士论文主要是稳态
   force-free relaxed wake，不能冒充扑翼非定常算法。
2. Kandil、Chu 与 Tureaud 的 NASA 非定常高阶涡面板，在近场使用一阶涡量
   面元，对相邻面元施加涡量兼容，并在跨时步自由面元上施加 Kelvin/Helmholtz
   约束；翼面压力由同一势跳/涡量状态的非定常 Bernoulli 计算。
3. Mracek 的非定常非平面涡面板采用线性涡量三角面和自适应 wake
   redistribution，说明“连续空间状态—滚起—连续压力”不是总力后分摊路线。
4. Krebs、Bramesfeld 与 Cole 的 DDE 把势跳扩展为三角面上的完整二次式
   `Gamma(xi,eta)=A1*eta^2/2+A2*eta+B1*xi^2/2+B2*xi+C2*eta*xi+C3`；
   wake 物质强度在生成后保持，几何伸缩后重解系数，并在非定常 rotor/突风算例
   验证。这比仅有展向多项式的 relaxed DVE 更接近 FLUXV 所需状态。

因此是**缺组成部分**，不是旧标量或单一常数错误。新增 `N3.1j`：
高阶连续自由涡面只向统一束缚面板压力提供诱导速度和势跳历史，禁止自身直接加
经验 LEV 力。第一阶段只实现 DVE/连续势跳基函数、共享边兼容、拉伸下 Kelvin
守恒和无力 canonical 场门；DVE 的稳态证据只支持基函数，不支持非定常晋升。

预登记 NO-GO：

- 以 `CL/CT` 选择基函数阶次、核、平滑或重网格阈值；
- 把稳态 relaxed DVE 直接命名为非定常扑翼模型；
- 允许内部公共边残留未配对涡丝或 `div(omega)` 源；
- 空间场未收敛即接 Eq.17/面板压力；
- 复活已经封存的 rVPM 生产路径。

### 4.7.1 DDE 最小可执行内核

`claim_runtime/distributed_doublet.py` 现只实现无力 P2 物质面：

- 六节点顺序为三顶点加三边中点，精确重现任意面内二次势跳与线性涡片；
- 等价涡片向量定义为 `grad_s(Gamma) cross n`；若
  `Gamma=A+B*eta+C*eta^2`，则流向强度严格为 `B+2*C*eta`；
- 相邻三角形公共边的两个端点和中点必须相同，因而整条二次 trace 连续；
- 仿射拉伸/旋转只移动物质节点，不改变六个势跳值，Kelvin residual 为零；
- 离面诱导速度直接积分 Krebs Eq.3.2，不引入核。符号和 `1/(4*pi)` 由常强度
  面元退化为同向三角涡环的恒等式固定；
- 面内自影响/主值尚未实现时明确抛错，禁止用 denominator epsilon 冒充。

当前 10 项测试覆盖二次完备性、梯度/涡片方向、公共边反例、非流形反例、物质
Kelvin、积分收敛和常强度涡环极限。此结果把 `N3.1j4` 置为 `partial`，而不是
validated：它还缺面内主值、自对流、LE/TE 新生拓扑、canonical field gate 和统一
面板压力。

## 5. 一手证据

- Hirato, Shen, Gopalarathnam & Edwards, *Journal of Aircraft* 56(4), 2019,
  DOI `10.2514/1.C035124`，本地全文
  `researchpaper/Vortex-Sheet Representation of Leading-Edge Vortex Shedding from Finite Wings.pdf`。
- Ansari, Zbikowski & Knowles, *Proc. IMechE Part G* 220 (2006), Part 2,
  DOI `10.1243/09544100JAERO50`，首涡半步局部边速度与后续 1/3 放置。
- Ramesh, Gopalarathnam, Granlund, Ol & Edwards, AIAA 2012-3027，
  DOI `10.2514/6.2012-3027`，LESP-DVM 第一 LEV 二维位置式。
- Bird, Ramesh, Otomo & Viola, AIAA SciTech 2021, UVLM with a regularized vortex-particle
  leading-edge wake.
- Li, Zhao & Graham, *JFM* 900 (2020), DOI `10.1017/jfm.2020.515`,
  three-dimensional vortex-force maps.
- Bramesfeld, *A Higher Order Vortex-Lattice Method with a Force-Free Wake*,
  Penn State PhD thesis (2006), Ch.3；以及 Bramesfeld & Maughmer,
  *J. Aircraft* 45 (2008), DOI `10.2514/1.31665`.
- Kandil, Chu & Tureaud, *Steady and Unsteady Nonlinear Hybrid Vortex Method
  for Lifting Surfaces at Large Angles of Attack*, NASA/NTRS `19820019347`
  (1982).
- Mracek, *A Vortex Panel Method for Potential Flows with Applications to
  Dynamics and Controls*, Virginia Tech PhD thesis (1988).
- Krebs, Bramesfeld & Cole, *Aerospace* 9 (2022) 28,
  DOI `10.3390/aerospace9010028`；Krebs, *A Distributed Doublet-Based Method
  for Unsteady Aerodynamic Analysis with Relaxed Wakes*, Ryerson PhD thesis
  (2021).
