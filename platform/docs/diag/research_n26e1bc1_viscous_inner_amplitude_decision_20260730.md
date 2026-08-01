# N2.6e1bc1：有限角黏性内层受控幅值裁决

日期：2026-07-30  
父问题：`N2.6e1bc`  
唯一候选：`N2.6e1bc1-VA-TE`  
状态：`MECHANISM GO / SOURCE-LIMIT A0 REQUIRED / FINITE-ANGLE EQUATION OPEN /
PRODUCTION OFF`

> 后续预执行裁决：`n26e1bc1_preexecution_feasibility_result_20260730.md`
> 证明本文件原定的 “A0 先行” 不在生产关键路径上。`B_TE-only` 直接闭合
> 已 NO-GO；A0 因 `B_e` 数值资产和 Eq.26/27 身份未闭合而
> `PRECONDITION-NO-RUN`。本文件保留为候选形成时的文献裁决记录，执行顺序
> 以后续结果文件为准。

## 1. 病因、节点和可动空间

冻结的 `N2.6e1bc0` 六工况门在所有协议健康项通过时得到

\[
p_K=0.9606123,\qquad
p_{\rm regular}=\frac{1+\beta}{1-\beta}=1.1292006 .
\]

最细 local order 为 `0.9717438`，而 `128 -> 256` 的
lower/upper/mean regular modal 变化为
`2.662%/0.668%/1.216%`。因此病因不是“网格还不够”，而是：

> 消去首奇异模态后的有界 regular 外区缺少承担 generic
> \(O(\Delta t)\) Kelvin 出生环量的领先阶自由度。

该问题唯一挂到 `N2.6e1bc1`。可动空间只允许新增一个由局部黏性内层
选择的 signed amplitude `B_TE(t)`。冻结 N1/N4、LESP 力幅值、目标曲线、
endpoint gamma、epsilon/cutoff 和全局三项式 weak-UK 均不可动。

## 2. 一手机理

### 2.1 Taha--Rezaei：黏性内层可以选择非零尾缘奇异幅值

Taha & Rezaei, *JFM* 868 (2019),
DOI `10.1017/jfm.2019.159`，放松经典 Kutta 条件，在薄翼压力分布中保留
可积的尾缘奇异项，并以 triple-deck 内外匹配确定其幅值。其物理角色不是
附加升力公式，而是：

1. 无黏外区本身不能决定该幅值；
2. 黏性尾缘内区通过 Blasius 层到 Goldstein wake 的边界条件转换选出幅值；
3. 该幅值改变环量发展及压力相位。

来源尺度和常数包括

\[
\epsilon=Re^{-1/8},\qquad \lambda=0.332,
\]

以及 Chow--Melnik lower-deck 数值函数 \(B_e(\alpha_e)\)。来源给出的
有效域是薄翼、小扰动，且 unsteady 等价使用
\(0<k<O(Re^{1/4})\)。当 \(\alpha_e\to0.47\) 时出现 trailing-edge
stall 渐近线；对 `Re=10^4--10^6`，来源给出的实际角约
`3.1--4.2 deg`，越界表示 upstream separation，不允许外推。

### 2.2 dos Santos--Rezaei--Taha：幅值必须进入环量系统和统一压力

dos Santos, Rezaei & Taha, *Physics of Fluids* 33, 103606 (2021),
DOI `10.1063/5.0065293`，把同一黏性幅值转换为附加 bound-circulation
distribution，和 no-through、Kelvin、新生 wake circulation 一起求解，再用
同一 unsteady Bernoulli panel pressure 积分 lift/moment。该文支持：

- `B_TE` 应进入气动状态而非后处理总力；
- 任意时间运动和变形 camber 的数值嵌入是可行的；
- pressure、wake 和 load 应来自同一环量解。

但其“arbitrary shape”仍是薄面/camber-line UVLM 表示，不能自动授权
20.595° 实体尾缘或双侧厚翼压力。

来源间有一个必须显式记录的常数版本差异：2019 `JFM` 主推导印作
`\lambda=0.332`，2021 `PoF` 摘要公式使用 `\kappa=0.334`。候选物理定义
采用同行评审主推导的 `0.332`；若复现 2021 离散结果，必须把 `0.334`
标为 source-profile 身份并报告灵敏度，禁止混合后按目标响应选值。

### 2.3 Riziotis--Voutsinas：保留为上游 provider，不能原样重走

Riziotis & Voutsinas (2008), DOI `10.1002/fld.1525`，提供 actual
NACA0015、双侧 IBL、transpiration、非定常压力和 double wake 的强相互作用
拓扑，是后续厚翼统一压力的必要上游 provider。

但其最近上下 control point 的 TE birth 已在 `N2.6e1b1` 显示
`7--9%` 非 Cauchy。`\delta^*,\theta,n/C_tau` 是有限亏损矩和流态记忆，
不能唯一选择局部 TE profile 或非regular模态幅值。因此 Riziotis-alone
不是新候选；只保留其 outer/IBL 方程。

### 2.4 VES：物理范围更完整，但当前不是最小可观测候选

DeVoria & Mohseni, *JFM* 866 (2019),
DOI `10.1017/jfm.2019.134` 的 vortex-entrainment sheet 可携带面质量、
面动量、卷吸、速度跳和压力跳，适合有限 forming zone。可是当前模型没有
独立 provider 给出

\[
\rho_s,\quad \rho_s v,\quad q,\quad \gamma,\quad [[p]],\quad X_s .
\]

仓内 profile 反例也已证明有限 IBL 亏损矩不能唯一恢复这些状态。现在启用
VES 只能引入未登记 edge convention、profile 或目标反演，新增秩远大于
本病因所需的一个幅值。因此 VES 保留为后继余核候选，不在本轮并行尝试。

## 3. 缺件/错件裁决

- **缺组成部分**：缺少“黏性尾缘内层选择一个非regular outer-mode
  amplitude”的闭合。
- **组成关系错**：把 regular inviscid TE provider 先单独推进，再把出生
  交给 IBL/pressure 的顺序不成立。`B_TE` 必须与 outer、IBL、Kelvin、
  material WPJ 和压力历史处于同一个 trial state。

不是 `A0/LESP` 阈值、`f2`、网格点、core 或压力增益的问题。

## 4. 唯一候选及其运行角色

每个二维 strip 只新增一个 signed scalar `B_TE(t)`：

```text
requires:
  local Re, wall motion, same-trial outer/IBL state,
  pressure/circulation projection, TE angle, previous potential and material wake

provides:
  controlled finite-angle circulation basis,
  B_TE and matching residual,
  gauge-invariant newborn WPJ,
  same-trial dual-side panel pressure
```

对实体尾缘外角

\[
\Omega=2\pi-\tau,\qquad
\lambda_1=\frac{\pi}{\Omega},
\]

`B_TE` 是 classical least-singularity 条件会消去的首个 corner mode 的
受控幅值。它在外区可以非regular，但必须由有限尺度黏性 inner solution
正则化并通过匹配/adjoint solvability 唯一确定。禁止直接把平板
`B_v/sqrt(1-x^2)` 搬到厚翼。

`B_TE` 只改变气动状态；力仍由同一总势、总速度和势历史计算的唯一 panel
traction 产生。禁止另开 `Delta L/Delta T` 通道。A0/LESP 只作前缘拓扑与
适用域 guard，不提供幅值。

## 5. 顺序门

### A0：薄翼 source-limit oracle

先复现 2019/2021 来源限定问题，验证 \(B_v\) 公式、附加环量列、Kelvin、
压力和力矩账。至少覆盖：

- `Re=10^4, alpha=3 deg` Wagner step；
- `Re=10^5` flat-to-NACA2412 camber step；
- `Re=10^5` harmonic frequency response 与 `k=1` wake；
- 来源的非谐波 maneuver。

A0 不读取 Fig17/18/19，不使用 NACA0015 有限角，不晋升生产。

### A1：有限角 inner matching

只有 A0 通过后，才允许推导并实现：

1. actual wedge geometry 的 inner variables、尺度和 boundary conditions；
2. body no-slip 到 wake zero-stress 的局部 BVP；
3. 与 `r^{lambda_1}` outer corner mode 的双侧 matching；
4. 选择 `B_TE` 的 adjoint/Fredholm solvability；
5. `tau -> 0` 时回收来源 \(B_v\)；
6. 独立于 Kelvin backsolve 的 inner circulation flux。

在上述方程闭合前，A1 为 `IMPLEMENTATION NO-GO`。

## 6. 预登记硬门

- 来源算例的 `B_v`、panel pressure、lift/moment 在独立 `h/dt` 加密下，
  末级变化分别不超过 `2%/5%/2%`；
- BIE/Kelvin/WPJ 残差 `<=1e-9`，panel traction 与 ForceLedger 一次闭合；
- 角度反演保持符号对称，`Re -> infinity` 时 `B_TE -> 0`；
- A1 必须拥有 finite-angle BVP/solvability，且 `tau -> 0` 回收 A0；
- 在与 e1bc0 相同的 NACA0015、`U=9 m/s`、`t*=0.2 s` 和独立
  `h/dt/inner-resolution` 轴上，
  `|p_birth-1|<=0.03`、`B_TE` Cauchy `<=2%`、`Cp` relative L2
  `<=5%`、Kelvin mismatch `<=1e-8`。

缺有限角方程、直接搬平板公式、用 Kelvin 反解幅值、越过
`\alpha_e=0.47` 仍外推，或 `p_birth` 仍约 `1.129`，均为明确
`PHYSICS/DOMAIN NO-GO`。A0/A1 全过只授权 Fig17/18/19 代表点，不授权
完整 184。

## 7. 明确拒绝的旁路

- endpoint `gamma_TE`、epsilon、core、TE offset、压力 cap/gain；
- 从 Fig17/18/19、总力或 V4.1 残差选择 `B_TE`；
- 把 `B_TE` 直接变成力；
- 全局三项式 weak-UK；
- 同时启动 full VES 或涡粒子生产模型；
- 2026 arXiv `2604.16501` 的直接厚翼/三维外推或 profile-drag tuning。

该 2026 预印本只能作为“标量可嵌入 vortex framework”的低等级旁证；
它未提供有限实体角 inner matching，且未同行评议，不改变本裁决。
