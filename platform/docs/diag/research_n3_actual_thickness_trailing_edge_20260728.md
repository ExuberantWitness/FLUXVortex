# N3 实际厚度尾缘：从“单一 Kutta 数”改为多残差与 wake 拓扑案件

日期：2026-07-28  
Claim：`N3.1j3b6c1-c3`，后续改写至 `N3.1j3b6d`  
运行角色：read-only diagnostic shadow

## ① 病因定位

### 数据指纹

冻结 N1 的逐时步 adapter 已按预登记在真实 `v41` 工况执行：

- 制造环场对 Warp 生产核的 fp64/fp32 相对误差分别为
  `5.80e-16`、`2.24e-7`；
- `frames_out` 开关前后全部数值结果逐字段 bitwise 相同；
- 代表点 `U8/aoa15/f2.6/tw11.25` 得到
  `L_wind=15.7668 N`、`T_wind=-3.9602 N`；
- N1 collocation 的冻结生产身份重放残差为 `1.85e-7`；
- 真实双侧闭壳 source 解的无穿透残差为 `1.29e-16`；
- 但 source-flux 相对残差为 `0.006205`；
- 若把镜像 wake 纳入“物理对称候选”，真实壳面入射速度 RMS 改变
  `0.2792 m/s`。它没有被偷换进生产 N1。

这说明首要病因已不再是“拿不到 N1 场”，而是：

> 冻结薄面 N1 的单一 rear-bound-ring-line wake 与实际厚度闭壳的尾缘拓扑及
> Neumann 兼容条件是否相容，尚未证明。

### 几何身份

`N2.6d1` 使用标准 NACA 四位数开尾缘厚度式（末项 `-0.1015`）。
对 NACA-2406：

\[
\frac{h_{TE}}{c}=2(0.00063)=0.00126,
\]

根部 `c=0.287 m` 时基底厚度为 `0.36162 mm`。因此当前计算壳具有：

1. 上尾缘角线；
2. 下尾缘角线；
3. 二者之间的实体 base 面；
4. N1 wake 从薄涡格最后一排的 rear bound-ring line 生成；该线不是
   几何尾缘，而是位于最后一个弦向面板之后 `0.25` 个末面板长度处。

它不是一个 upper/lower 相交的 cusped 或 finite-angle sharp edge。把
“薄面 Kutta 已过”直接当成“厚体 Kutta 已过”是拓扑错件。

### Claim 节点与可动空间

- `N3.1j3b6c1`：N1 只读速度场 adapter，可冻结；
- `N3.1j3b6c2`：薄面 Kutta 自动传递到有限 base，证伪；
- `N3.1j3b6c3`：实际厚度 TE/base 的多残差和 wake 形成状态，保持 partial；
- 不可动：N1 环量、N1 wake、N1 force、N2.6d1 几何身份；
- 当前可动：只读 residual、网格/时间收敛、sharp-limit continuation；
- 禁止：Kutta 常数、wake 角度、base 压力或 source-flux 拟合。

## ② 学科机理

### Kemp：钝尾缘至少有三种不等价条件

Kemp, *A Vector-Continuous Loading Concept for Aerodynamic Panel
Methods*, NASA TM-80104 (1979), 对同一个 blunt trailing edge 比较：

- B-1：closing panel 控制点处，离开流沿尾缘 bisector；
- B-2：在上下两个尾缘角分别施加方向条件；
- B-3：上下尾缘角压力相等，以完整非线性形式迭代。

三者是不同 closure，不是同一恒等式。更关键的是，Kemp 对 blunt 模型的
总 source 账发现了从数值尾缘“流出”的 added mass，并明确指出真实分离
wake 不会添加这部分质量，因此没有把相应力增量保留在结果中。

对本项目的直接含义是：`0.006205` source-flux 指纹不能被解释为真实
尾迹质量、推力或升力，更不能通过一次力修正消掉。

### Xia–Mohseni：非定常尾迹方向不能任意指定

Xia & Mohseni, JFM 830 (2017), DOI `10.1017/jfm.2017.513` 表明，即便是
单一 **finite-angle sharp edge**，非定常形成涡片的方向、强度和相对速度
也必须把：

1. unsteady Kutta；
2. circulation conservation；
3. mass conservation；
4. momentum conservation

联立。任取 bisector 或某一侧切线都不是一般非定常解。当前 RoboEagle
还是两个角加一个 base，不能反而用更弱的单标量条件。

### 压力 Kutta 不是后处理

Wang, Abdel-Maksoud & Song, *Ocean Engineering* 130 (2017),
DOI `10.1016/j.oceaneng.2016.12.009` 指出，Morino 线性 wake-doublet
关系仍可能产生非物理尾缘压差；零压差的 pressure Kutta 是对 wake/body
doublet 的非线性耦合条件。它不能在冻结环量算完后仅作为一个“误差标签”
就宣称闭合。

Poling & Telionis, *AIAA Journal* 24 (1986),
DOI `10.2514/3.9244` 的高频实验进一步表明经典非定常 Kutta 有适用域，
不能把某一种形式当成所有 reduced frequency 下的普适边界。

## ③ 方向判定

| 候选 | 判定 | 证据 |
|---|---|---|
| 冻结 N1 薄面 Kutta 自动保证厚体尾缘 | 错组件，falsified | 几何有两个角和 base；Kemp 三种条件不等价 |
| 只检查 `p_upper=p_lower` | 不充分 | B-3 是非线性 circulation closure；不能唯一给出 unsteady wake 方向/强度 |
| 只检查 bisector 出流 | 不充分 | B-1 仅为一种 closure；Xia–Mohseni 要求守恒联立 |
| 把非零 source flux 当 wake 质量 | 禁止 | Kemp 明确该 added mass 不属于真实分离 wake |
| 立刻修改 N1 wake | 未授权 | N1 frozen；必须先做 read-only 冲突诊断 |
| 多残差＋sharp-limit continuation | 当前 GO | 不改 N1，可判定是小厚度离散误差还是缺 base-wake 组件 |

因此当前不是“缺一个更好的 Kutta 常数”，而是可能缺少：

> 与实际厚度尾缘匹配的双角剪切层/base-wake 空间状态，或者在有证据证明
> `h_TE/c→0` 平滑后采用受控 sharp-TE 极限。两者都不能靠总力修正替代。

## ④ 机理方案与预登记

`thick_body_trailing_edge_cases.yaml` 在残差实现前冻结五个账：

1. `R_geometry`：两个尾缘角、base、厚度和首 wake 间距；
2. `R_pressure`：统一 Bernoulli 后的上下尾缘总压力差；
3. `R_direction`：上切线、下切线、bisector 三个方向残差分别报告；
4. `R_flux`：闭壳 Neumann 兼容和 source flux，禁止转成力；
5. `R_wake_formation`：单 rear-ring-line wake 与双角拓扑的方向、强度、相对运动及
   circulation/mass/momentum 联立缺口。

执行顺序：

- 先通过 cusped/static canonical；
- 再做 finite-angle sharp 的守恒形成门；
- 再复现 Kemp B-1/B-2/B-3 不等价性；
- 做 NACA-2406 base fraction `{1, 0.5, 0.25, 0}` continuation；
- 最后才运行 RoboEagle 动态代表点。

当前 `R_pressure` 仍缺 N1 doublet/source 的一致 material potential history，
所以不能伪造压力 Kutta 数值。若多残差系统性要求改变 N1 环量，必须停止
并另开 N1 freeze review；不得在 N3 内暗改。

## 本轮结论

N1→真实壳速度 adapter 已有资格冻结为 validated read-only interface；实际
厚体压力模型仍然 NO-GO。新的主病灶是有限 base 的尾缘/wake 拓扑与非定常
守恒闭合，不是结构模型，也不是待调常数。

## 后续 K4a 通量/涡丝拓扑裁决

预登记 `K4a_channel_flux_and_filament_shell_topology` 后，代表点显示：

- production 闭壳通量 `+0.0297291`；
- bound direct/image 分别为 `+0.0610303/-0.0312306`；
- wake direct 仅 `-7.06e-5`，wall volume 仅 `-7.48e-6`；
- 158→1246 面加密时 bound 通量不单调趋零；
- 消去共置零净环量线段后，bound direct 仍有 16 条活动线段穿壳，其中
  `trailing_base/root/tip = 9/5/4` 次；
- bound image 在 root/base 共角处命中 1 次；wake direct 命中 0。

9 次 trailing-base 命中对应 `ns+1` 条末排弦向涡丝，说明问题不是只由
半翼 root cap 造成。冻结 N1 的 rear bound-ring line 在几何尾缘之后
`0.0366743c`，而实际 base 仅 `0.00126c`；原始 ring 支撑确实穿过真实壳。

因此：

- “薄面 Kutta 自动传递”仍为 falsified；
- 更强的“原始 N1 ring 场可由实际壳单值 source 条件化”也被证伪；
- `N3.1j3b6c` 已 falsified/frozen；
- 新 live 方向 `N3.1j3b6d` 不移动、不加核、不删除这些涡丝，而是停止把
  它们当实际物面外势：N1 只提供环量约束候选，lifting doublet/source 与
  TE/base wake 必须在实际边界拓扑上重新联立。

这正是“组成部分错”而非“缺一个更好常数”。预登记见
`actual_boundary_circulation_representation_cases.yaml`。
