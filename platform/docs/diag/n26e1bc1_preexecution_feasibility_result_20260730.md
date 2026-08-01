# N2.6e1bc1 有限角黏性幅值：预执行可行性裁决

日期：2026-07-30  
对象：`N2.6e1bc1-VA-TE`  
裁决：

```text
OUTER ONE-MODE COORDINATE: MATHEMATICALLY COHERENT
B_TE-ONLY VISCOUS CLOSURE: PRE-EXECUTION NO-GO / FALSIFIED
FULL FINITE-ANGLE VISCOUS INNER: OPEN
SOURCE-LIMIT A0: PRECONDITION-NO-RUN / OFF CRITICAL PATH
PRODUCTION / FIG17-18-19: OFF
```

本裁决写在任何 `B_TE` 数值实现、来源曲线数字化或 Fig17/18/19 候选运行
之前。它不读取目标曲线来选择公式或常数。

## 1. e1bc0 实际证明了什么

冻结的 e1bc0 门得到

\[
p_K=0.9606123,\qquad
p_{\rm regular}=1.1292006 .
\]

因此 bounded regular outer corner mode 单独不能承担 generic
\(O(\Delta t)\) Kelvin 出生。该结果只确定“缺一个领先阶库存/自由度”，
不确定该缺件一定是 Taha--Rezaei \(B_v\)，也不证明完整状态秩为一。

## 2. 一个 outer coordinate 合理，但不是完整黏性闭合

对实体尾缘角 \(\tau\)，流体外角

\[
\Omega=2\pi-\tau
\]

上的局部齐次 Neumann 模态为

\[
\phi_n=A_n r^{\lambda_n}\cos(\lambda_n\theta),\qquad
\lambda_n=\frac{n\pi}{\Omega}.
\]

closed NACA0015 的

\[
\tau=20.595196772^\circ,\qquad
\lambda_1=0.5303401669
\]

给出

\[
u_1\sim B_{TE}r^{-0.4696598331}.
\]

首模态的 outer coefficient 确实是一维；当 \(\tau\to0\) 时，
\(\lambda_1\to1/2\)，在模态类型上回收平板逆平方根奇异性。下一模态
\(\lambda_2=1.0606803339\) 才是 e1bc0 使用的 bounded regular mode。

但 outer 缺陷空间的维数不等于 viscous inner 状态的维数。要选择
\(B_{TE}\)，最低限度仍需：

- 上、下两侧各自的入射边界层剖面或经证明充分的状态；
- inner \((u,v,p)\) 或等价 \((\psi,\omega)\) 场及历史；
- 两个 moving no-slip wall 条件；
- 下游两半 wake 的速度、应力、方向和势跃历史；
- 双侧 inner--outer matching；
- 去除压力/势 gauge 后的唯一稳定分支；
- 证明线性化算子的 adjoint cokernel 为一，且幅值列不与其正交。

现有 \(\delta^*,\theta,n/C_\tau\) 是有限亏损矩，仓内反例已经证明它们不能
唯一恢复上述剖面和 forming-zone 状态。Kelvin/WPJ 只能检查 inner 输出的
环量通量，不能反解 \(B_{TE}\) 后再把它冒充独立黏性闭合。

## 3. Taha--Rezaei 平板来源不能直接延拓到实体角

Taha & Rezaei, JFM 868 (2019), DOI `10.1017/jfm.2019.159` 的来源域是
零厚度、小扰动、Blasius--Goldstein 平板尾缘和
\(\epsilon=Re^{-1/8}\) triple deck。其压力修正还含由尾迹历史决定的
\(a_{0v}(t)\)，并不是无历史的
`local Re, alpha -> B_v` 代数标量。

有限楔 triple-deck 使用

\[
\beta_w=Re^{1/4}\lambda^{-1/2}\tau,\qquad \lambda=0.332 .
\]

Gajjar & Turkyilmazoglu, *Phil. Trans. R. Soc. A* 358 (2000),
DOI `10.1098/rsta.2000.0699` 报告 wedged trailing edge 在
\(\beta_w\approx2.56\) 已进入分离/失稳区。对 closed NACA0015 和
RoboEagle 相关 \(Re\approx1.1\times10^5\)--\(1.9\times10^5\)，使用完整
实体夹角得到

\[
\beta_w\approx11.5\text{--}13.0 .
\]

即使按半角解释，仍远大于 2.56。因此 Chow--Melnik
\(B_e\)、Taha \(B_v\) 和平板 \(Re^{-1/8}\) 公式不能被直接搬到该 shadow。
Riley & Stewartson, JFM 39 (1969), DOI
`10.1017/S0022112069002114` 也把 finite wedge 作为独立匹配问题处理，
没有提供本工况所需的一标量非定常闭合。

本数值只针对 NACA0015 source shadow。生产 N1 目前是 NACA2406
mean-camber thin sheet；即使以后构造 closed NACA2406 双侧壳，也必须用
其自身厚度/尾缘约定重新求角度和 inner 拓扑，禁止复用 NACA0015 的角度。

## 4. A0 来源门为何现在不开跑

两份独立来源资产审计得到：

1. JFM 2019 和 dos Santos--Rezaei--Taha, PoF 33, 103606 (2021),
   DOI `10.1063/5.0065293` 都只给 \(B_e(\alpha_e)\) 图，没有数值表、
   插值规则或公开作者代码；
2. PoF 明示原始数据只能向通讯作者索取；
3. PoF Eq.26 中 \(\Gamma_b/\Delta\Gamma_b\) 的列身份与 Eq.27 压力式的
   \(\Gamma_b\) 语义必须通过 PDF 视觉转录和 Kelvin/压力账独立冻结；
4. JFM `lambda=0.332` 与 PoF `kappa=0.334` 是两个来源 profile，禁止平均
   或按目标表现择值；
5. 公开 Fig6--9 只能经预登记双流程数字化成为 source-consistency 证据，
   不能成为独立硬真值，更不能授权厚翼生产。

故 `N2.6e1bc1a` 保持 open，但状态是
`PRECONDITION-NO-RUN`。未来若完整 finite-angle operator 先闭合，A0 可作为
\(\tau\to0\) 回归门；它不再是当前主线第一步。

## 5. 结果回写

证伪并冻结的窄命题是：

> Taha--Rezaei 平板 \(B_v\) 或当前有限 IBL 亏损矩可以直接变成一个即时
> \(B_{TE}\) 标量，从而闭合 NACA0015 固定实体角的黏性出生与压力。

没有被证伪的是：

> 一个完整、双侧、带历史的 finite-angle viscous inner 可能最终只向 outer
> 暴露一个 \(B_{TE}\) coupling coordinate。

后者在写出目标域有效的 operator、边界条件、matching、尺度、稳定分支和
Fredholm/index 证据前保持 open 且 implementation-off。下一候选不得再是
source-only 平板 oracle；必须拥有通向实际双侧压力和 Fig17/18/19
代表工况的生产桥。

