# N3.1j3b6：冻结 N1 下的实际厚体均势场裁决

日期：2026-07-28  
范围：气动载荷模型；不开发结构求解器，不改 N1，不拟合常数。

## 结论先行

当前缺口不是双侧几何，也不是一个可加到 N1 总力上的“厚度力”。`N2.6d1`
已经冻结了与 N1 中弧面共置的 NACA-2406 上/下表面。真正缺失的是：

1. 由**全部入射场**条件化的实际厚体无穿透 source/mean-potential 解；
2. source、N1 bound/wake、自由涡态及壁面运动在势率/速度级的统一；
3. 在统一状态上只执行一次 Bernoulli 和一次表面积分；
4. 与 N2.6 黏性库存/分离压力在压力级、而非总力级闭合。

允许的第一步是 diagnostic shadow。它可消费冻结 N1 的状态，但不得改写 N1
环量、wake、实现或基线贡献。厚体尾缘 Kutta 并不会因为 N1 薄面解已通过就
自动成立，因此在 canonical、Kutta、附着基线和双侧 Cp 外验通过前不得晋升
生产路径。

## ① 病因定位

### 1.1 数据/表示指纹

P0/P1 公开动态压力账已经证明：

- `delta_Cp = Cp_lower - Cp_upper` 可无规范地重建薄面 `Cn`；
- 上下表面同加公共压力模态时，`delta_Cp/Cn` 不变；
- 同一公共模态会改变实际厚翼的 `Ct`；
- 因而 N1 薄面压力跳不能唯一恢复绝对双侧 Cp 或厚翼弦向压力力。

这把病因挂到 `N3.1j3b4/N3.1j3b6`：不是旧标量权重不佳，而是状态空间缺少
mean-potential/thickness 模态。

### 1.2 几何资产审计

`platform/claim_runtime/viscous_shell_geometry.py` 已经：

- 由 NACA 四位数定义生成 2406 上/下表面；
- 与 N1 共享 `(xi,eta)`，算术中面严格回到 N1 中弧面；
- 冻结厚度、director、材料配对和刚体运动学。

其预登记明确排除 boundary layer、pressure、force 和 production claim。
因此：

> `N2.6d1` 闭合的是气动壁面几何先决件；它没有闭合厚体气动算子。

### 1.3 当前实现缺件

`dual_side_bernoulli.py` 仅在调用者已给出以下量时验证代数身份：

- mean-potential material rate；
- mean velocity；
- potential-jump rate/gradient；
- wall velocity 和公共 Bernoulli gauge。

生产 N1 当前不提供实际厚体上的 mean-potential/source/no-penetration 解。
所以现有代码不是实际双侧压力模型，只是给定输入后的观察器。

## ② 学科机理

### 2.1 实际厚体必须在边界条件层求解

Morino, *A General Theory of Unsteady Compressible Potential Aerodynamics*,
NASA CR-2464 (1974),
https://ntrs.nasa.gov/citations/19750004821 ：任意运动升力体的表面势与其
由边界条件给定的法向导数通过边界积分方程关联；厚度趋零时算子会出现薄翼
奇异极限。因此薄面解不能直接解释为实际厚体双侧势。

Dusto & Epton, NASA CR-152323 (1980),
https://ntrs.nasa.gov/citations/19800017771 ：非定常 source/doublet
分布位于厚体的实际平均表面。

Bristow, NASA CR-3234 (1980),
https://ntrs.nasa.gov/citations/19800007773 ：实际构型表面采用组合
source-doublet；法向速度边界与未知 doublet 通过边界方程闭合，再由合成
表面速度求压力。报告同时指出单独 source 或单独 vortex/doublet 在薄、
高载荷或前缘问题上会产生数值缺陷。

NASA TP-2995, *Panel Methods—An Introduction* (1990),
https://ntrs.nasa.gov/api/citations/19910009745/downloads/19910009745.pdf ：
source 造成法向速度跳，doublet 造成势和切向速度跳；一致的高阶表示必须
匹配这些跳量。

Keune, NACA TM-1023 (1942),
https://ntrs.nasa.gov/citations/20030069008 ：厚翼势流可由均线上 source
与 vortex 分布的势场叠加构造。这个结论授权的是势/速度线性叠加，不授权
压力或力线性叠加。

### 2.2 为什么压力不能分别计算后相加

令实际厚体表面的总势为

\[
\phi=\phi_\infty+\phi_{N1}+\phi_{free}+\phi_\sigma ,
\]

其中 \(\phi_\sigma\) 是为实际壁面无穿透而求得的单值 source 势。非定常
Bernoulli 使用总状态：

\[
\frac{p}{\rho}=C(t)-\left[
\frac{\partial\phi}{\partial t}+\frac12|\nabla\phi|^2
\right].
\]

即使各势满足线性 Laplace 方程，平方速度仍包含
\(\nabla\phi_i\cdot\nabla\phi_j\) 交叉项。因此不存在一般成立的

\[
p_{total}=p_{thickness}+p_{lifting}.
\]

### 2.3 带环量圆柱的精确反例

对半径 \(R\)、来流 \(U\) 的圆柱，定义
\(k=\Gamma/(2\pi R U)\)。表面切向速度为

\[
\frac{v_\theta}{U}=-2\sin\theta+k,
\qquad
C_p=1-\left(-2\sin\theta+k\right)^2.
\]

若分别计算非环量压力与环量模态二次压力再相加，会漏掉

\[
C_{p,cross}=4k\sin\theta .
\]

该交叉项积分后恰好给出 Kutta–Joukowski 升力。预登记的三个 \(k\) 工况
结果为：

| \(k\) | 最大漏失 \(|\Delta C_p|\) | 统一压力 \(C_L\) | 压力级相加 \(C_L\) |
|---:|---:|---:|---:|
| 0.10 | 0.4 | -0.6283185 | 约 0 |
| 0.35 | 1.4 | -2.1991149 | 约 0 |
| 0.70 | 2.8 | -4.3982297 | 约 0 |

解析与积分最大误差 \(3.55\times10^{-15}\)，均匀压力规范改变力的误差同阶。
详见：

- `thick_body_pressure_coupling_cases.yaml`
- `thick_body_pressure_coupling_guard.py`
- `thick_body_pressure_coupling_results.json`

## ③ 改进方向判定

### 3.1 判定表

| 候选解释 | 判定 | 原因 |
|---|---|---|
| 缺少双侧几何 | 否 | `N2.6d1` 已 validated/frozen |
| 需要先做结构模型 | 否 | 当前刚性 RoboEagle 的气动壳和运动学已足够定义流体边界 |
| 缺一个可独立加力的厚度修正 | 错组件，falsified | Bernoulli 交叉项会被漏掉，解析反例丢失全部升力 |
| 缺实际厚体 mean-potential/source | 是，缺组件 | P1 公共模态与面元法边界积分共同支持 |
| 必须立刻重算 N1 环量 | 尚未授权 | 会碰 validated/frozen N1；先做条件化 shadow 判断耦合误差 |
| 可把冻结 N1 当作规定环量/入射场 | 有条件可行 | source 解必须消费总入射法向速度，压力必须统一；厚体 Kutta 未验证 |

### 3.2 可动空间

可动节点只有 `N3.1j3b6c` 及其后续 open 子节点：

- 输入：冻结 N1 bound/wake state、通过空间门的 free-vortex state、
  `N2.6d1` 实际双侧壳、壁面运动；
- 未知：附加单值 source/mean-potential；
- 约束：实际壁面总无穿透、闭体源通量兼容、远场衰减；
- 输出：总势率、总表面速度、统一双侧 Cp shadow；
- 禁止：写回 N1、分别成力、用厚度吸收 drag、拟合实验总力。

## ④ 机理方案与 go/no-go 预登记

### 4.1 条件化 Neumann shadow

在实际双侧闭合壳 \(S\) 上，以全部已知入射场构造

\[
g_n =
\left[
\boldsymbol V_{wall}
-\nabla(\phi_\infty+\phi_{N1}+\phi_{free})
\right]\cdot\boldsymbol n .
\]

求单值 \(\phi_\sigma\) 使

\[
\nabla^2\phi_\sigma=0,\qquad
\frac{\partial\phi_\sigma}{\partial n}=g_n
\quad \text{on }S,
\]

并检查闭体兼容条件、远场衰减和规范。然后在势率/速度层组合全部通道，
只执行一次 Bernoulli 和一次压力积分。

该算子是**条件化 shadow**，不是“geometry-only thickness correction”：
N1/free vortex 一变，source RHS 就必须变。

### 4.2 G1：算子 canonical 门

在接触 RoboEagle 数据前必须全部通过：

1. 闭合网格方向、watertight、法向和体积检查；
2. source 总通量与 Neumann 兼容残差；
3. 球/圆柱均匀来流的无穿透和解析表面速度/Cp；
4. 稳态无环量闭体的 d’Alembert 零阻；
5. 带规定环量圆柱保持 \(\Gamma\)，统一压力恢复 Kutta–Joukowski；
6. 均匀压力/势规范不改变压力梯度与总力；
7. 面板加密收敛、矩阵条件数和 null-space 显式处理；
8. 刚体平移/加速的 Galilean/objective 门与解析 added-mass；
9. N1 `Gamma/wake/n1_force` 输入前后 bitwise 不变；
10. 所有势通道在 Bernoulli 前合并，ForceLedger 只出现一次压力成力。

任何一项失败：`N3.1j3b6c = NO-GO`，不得用松阈值或常数补偿。

### 4.3 G2：RoboEagle shadow 门

G1 通过后才允许：

1. NACA-2406 二维/展向均匀截面的上、下 Cp 独立基准；
2. 实际三维双侧壳的逐面板无穿透、闭体通量与网格收敛；
3. 开口/有限厚尾缘 base 与 wake 出口拓扑检查；
4. 厚体尾缘 Kutta 残差；不得假设薄面 N1 自动满足；
5. 静态附着点不得破坏 N1 冻结守卫；
6. P1 公开动态双侧 Cp 只作输出外验，不反演 source 或经验参数；
7. 118 工况、趋势卡和 Fig17/18/19 在 shadow 与生产基线间并列比较；
8. 未获得满足 frozen spatial contract 的场数据前，不晋升 LEV 空间状态。

### 4.4 晋升规则

- G1 未全过：保持 `open`；
- G1 全过、G2 未全过：`partial + diagnostic shadow`；
- G1/G2 全过且厚体 Kutta 与 N1 freeze 无冲突：才可提出生产晋升；
- 若厚体 Kutta 系统性要求改变 N1 环量：停止。必须另开“是否重审 N1 freeze”
  的 claim 案件，不能在 N3 中暗改。

### 4.5 G1 执行结果

预登记后实现了独立的 diagnostic source shadow，未接入生产图：

- `claim_runtime/thick_body_neumann_shadow.py`
- `thick_body_neumann_shadow_guard.py`
- `thick_body_neumann_shadow_results.json`

五级 canonical 门全部通过：

| 门 | 结果 |
|---|---|
| G1a 常源面核 | 势/速度对独立 48 阶 Duffy 积分最大相对误差 `3.13e-15`；法向跳误差 0 |
| G1b 闭体球 | 80→320→1280 面速度误差 `1.076%→0.532%→0.252%`；最细 Cp RMS `0.00612` |
| G1b 守恒 | 无穿透 `3.11e-15`，源通量 `7.85e-17`，d’Alembert 力 `2.54e-16` |
| G1c 条件化入射 | 内部偶极改变 20% 时 source 解改变 20%；输入变化 0；source 势跳 0 |
| G1d 非定常 | Galilean 压力力变化 `4.45e-16`；三时级物质率误差 `1.63e-15`；added-mass 误差 `3.74%` |
| G1e 几何适配 | NACA-2406 半翼 `1449` 顶点/`2894` 面；boundary/nonmanifold/orientation 均 0；N1 中面变化 0 |

因此 `canonical_complete=true`，但在当时这只把 `N3.1j3b6c` 从 open
推到 partial；下面的真实 N1 通道/拓扑审计随后推翻了该运行表示。
结果文件仍明确：

- `model_comparison_executed=false`
- `production_activation_allowed=false`
- `promotion_gate=NO-GO`

剩余的第一真实病灶不再是 source 核或几何闭合，而是：

> 如何从冻结 N1 的逐时步 bound/wake 状态，在实际双侧壳每个面元上得到完整
> 入射速度与一致的 material potential history；以及规定的薄面 circulation
> 是否满足实际有限厚尾缘/base 的 Kutta 条件。

这两项必须先作为只读 adapter/Kutta residual 诊断，不能直接用新压力替换生产力。

## 本轮树改写

- `N3.1j3b6`：`open → partial`，明确几何已闭合、流体算子仍缺；
- `N3.1j3b6a`：validated/frozen，双侧气动几何先决件；
- `N3.1j3b6b`：falsified/frozen，压力/力级独立相加；
- `N3.1j3b6c`：此处记录的是当时的 partial 判定；后续已因活动涡丝穿壳
  和束缚场闭壳通量不收敛而改写为 falsified/frozen。

本轮没有修改生产气动公式、常数、网格、运动学、N1 或 V4.1 数值路径。

## 后续真实 N1 证据：条件化 shadow 被证伪

在只读 adapter 通过 Warp 核重放、N1 bitwise 不变和真实壳运动学门之后，
按预登记把代表点闭合面通量拆成：

| 通道 | 闭合面通量 |
|---|---:|
| freestream | `-3.47e-17` |
| bound direct | `+0.0610303` |
| bound image | `-0.0312306` |
| wake direct | `-7.06e-5` |
| production total | `+0.0297291` |
| wall volume | `-7.48e-6` |

壳体从 158 加密到 1246 面时，bound direct/image 通量仍在
`O(1e-2–1e-1)` 非单调波动；wake 始终约 `O(1e-5)`。这排除了
“wall 运动学”“wake 主导”和“简单粗网格误差”。

随后把每个环展开为有向线段，只按机器精度合并共置支撑并先消去零净环量，
活动 production 涡丝仍有 17 条穿壳：

- bound direct：`trailing_base/root_cap/tip_cap = 9/5/4`；
- bound image：root/base 共角处 1 次；
- wake direct：0。

其中 9 条 trailing-base 穿越正是 `ns+1` 条末排弦向涡丝。因此即使改用
完整翼、去掉半翼 root cap，真实有限 base 冲突仍存在。

NASA TP-2995 的精确关系是：constant doublet 与沿**同一面板周界**的
ring vortex 等价；连续 doublet 分布会消除面板边缘的伪线涡。它不授权把
薄格 ring 穿过另一套实际厚体边界。Dusto–Epton NASA CR-152323 则把
非定常 source/doublet 明确置于实际厚体表面。

故方向改写为：

1. `N3.1j3b6c` 原始 ring 场 + 单值 source conditioner：
   **falsified/frozen**；
2. `N3.1j3b6c1` 只读速度重放接口：仍 **validated/frozen**，因为它只声称
   忠实观察 N1，不声称物理厚体相容；
3. 新 `N3.1j3b6d`：N1 只作为环量不变量/初值候选，在实际边界上联立
   source、连续 doublet（或等价 bound sheet）与 TE/base wake；
4. 若实际边界无穿透、Kelvin、质量、动量和 Kutta 系统与规定 N1 环量冲突，
   触发 N1 freeze review，不得用 core、offset、source flux 或总力拟合。

完整 G4a-f 预登记见
`actual_boundary_circulation_representation_cases.yaml`。这一改写仍未触碰
生产 V4.1 气动公式。
