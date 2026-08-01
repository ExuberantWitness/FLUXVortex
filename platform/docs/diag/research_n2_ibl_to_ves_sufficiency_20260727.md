# N2.6b4 → N2.6c2c：IBL 亏损矩不足以唯一确定 VES 释放态

## 1. 病因、节点与可动空间

`N2.6b3` 已冻结三维表面 IBL 的守恒骨架，储存量是切向质量流亏损
`M` 与动量亏损张量迹 `tr(T)`；`N2.6c2b` 已冻结 vortex-entrainment
sheet（VES）的状态身份。尚未闭合的是两者之间的 `N2.6b4 → N2.6c2c`
接口。

问题不是总力如何分配，也不是结构传力。问题是信息充分性：

```text
IBL deficit moments
    ? -> gamma, q, rho_s, rho_s*v, [[p]], edge geometry/history .
```

可动空间仅为开放的 `N2.6b4` 近壁剖面闭合和 `N2.6c2c` 光滑前缘
release junction。N1、统一 Bernoulli 面板压力、ForceLedger、冻结的
DDE/VES 守恒恒等式均不改。

## 2. 学科机理

Zhang（MIT PhD, 2022）的四方程 3D IBL 以

```text
Q_IBL = {delta*_1, delta*_2, theta_11, theta_12}
```

为主变量。它们是外流与真实近壁速度之差的有限个厚度矩。论文明确指出，
法向积分丢失的剖面信息必须由 closure 补回；四方程相对三方程的价值正是
增加独立横流表征，而不是从旧矩中重排出该信息。

DeVoria & Mohseni（JFM 866, 2019）定义

```text
rho_s   = integral rho dn
rho_s*v = integral rho*u dn
gamma   = n cross [[u]]
q       = -n dot [[u]] .
```

在其 Falkner–Skan 映射中，边界层边缘由 99% 规则定义，`rho_s=rho*delta`；
`gamma` 和 `q` 还显式依赖边缘形状及边缘法向速度。因此 VES 保存的是
有限层内实际质量、动量和两侧速度跳，而标准不可压 IBL 保存的是相对外流的
亏损矩。两者不是同一组状态。

一个直接的 gauge 反例是：在已达到外流速度的剖面外侧增加一段完全均匀的
外流平台。该平台对所有速度亏损矩贡献严格为零，却给 `rho_s` 和
`rho_s*v` 增加非零量。故没有明确、经场数据验证的层边界约定时，IBL
亏损矩不能唯一确定 VES 面质量与内禀动量。

即使固定“末点速度等于外流”，有限个矩也不能唯一恢复整个剖面或层厚：
存在两条单调、同端点剖面具有相同 `M` 与 `tr(T)`，但具有不同层厚、面质量
和内禀动量。`q` 还额外依赖边缘法向、边缘速度及时间/空间演化。

## 3. 缺件还是错件

### 错件：有限 IBL 亏损矩是完整 VES release state

该命题为 **NO-GO**。从 `M,tr(T)` 代数反演 `rho_s`、`rho_s*v` 或 `q`
必然选择一个未登记的剖面/层边界先验；若再以 L/T 或压力残差选择该先验，
就是常数吸收。

### 缺件：显式且可验证的近壁剖面—边缘状态

`N2.6b4` 至少必须提供：

1. 双侧切向速度剖面或能唯一重构它的独立形状状态；
2. 经独立场数据定义的层边界位置、法向和运动；
3. 边缘两侧速度，特别是法向分量；
4. 密度/质量与一阶动量积分；
5. 足以推进物质 spike 的空间梯度和时间历史。

这些量先产生 IBL 守恒矩与 bound VES 状态的同源投影，再由
`N2.6c2c` 在分离流形处做具名守恒 junction。它们不能由总力反推。

## 4. 首个预登记 oracle

实现一个纯局部、纯运动学的 profile-collapse oracle。输入显式法向坐标、
密度、层内速度、外流速度、两侧速度、片法向和调用者给出的 edge convention；
输出：

- IBL `M` 与 `T`；
- VES `rho_s`、`rho_s*v`、`v`、`gamma`、`q`；
- edge convention 的原样审计记录。

它必须通过：

1. 解析二维线性剖面的矩；
2. 刚体旋转客观性；
3. 外流平台 gauge：IBL 矩不变而 VES 质量/动量变化必须可见；
4. 同 `M,tr(T)` 的两条单调剖面产生不同 VES 质量/动量；
5. 非法层坐标、密度、切向外流或空 edge convention 明确失败。

该 oracle 不接收压力、力、LESP、实验目标或结构量，不决定 99%/其他
edge convention，也不生成自由 LEV。通过仅允许：

- 证伪“I​​BL 亏损矩唯一确定 VES”；
- 冻结“显式 profile/edge state 是 release 接口的必要输入身份”；
- 保持具体 `N2.6b4` 动力闭合和 `N2.6c2c` 生产 release 为 open。

## 5. 原始来源

- Zhang, S., *Three-dimensional Integral Boundary Layer Method for
  Viscous Aerodynamic Analysis*, MIT PhD thesis, 2022, Chapter 2,
  Appendix A.
- Zhang, Drela, Allmaras, Galbraith & Darmofal, *Closure Modeling for
  Three-dimensional Integral Boundary Layer using Physics-constrained
  Neural Network and Model Inversion*, AIAA 2022-1078.
- Drela, M., *Three-Dimensional Integral Boundary Layer Formulation for
  General Configurations*, AIAA 2013-2437.
- DeVoria, A. C. & Mohseni, K., *The vortex-entrainment sheet in an
  inviscid fluid: theory and separation at a sharp edge*, JFM 866
  (2019), 660–688, doi:10.1017/jfm.2019.134.
