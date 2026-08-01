# N2.6c2 释放态裁决：纯 DDE 不能同时保存环量、卷吸质量与法向动量

## 1. 病因指纹与可动空间

`N2.6b2/b3` 已分别冻结曲面矢量库存账和三维 IBL 守恒骨架，
`N2.6c1` 的生产分离流形也已限定为客观的有限时间物质 spike。
但当前 `N2.6c2` 要求穿过分离流形的环量、质量卷吸和动量共同生成自由
P2 DDE 面带，下游实际状态却只有势跳：

```text
mu_DDE -> gamma = grad_s(mu_DDE) cross n .
```

这只能表示切向速度跳和积分环量。它没有独立自由度表示法向速度跳、片面
质量和片内动量。数据指纹是一个表示秩缺失，而不是 LESP、核半径或释放
系数不对：

- 纯涡片诱导场由切向涡片强度 `gamma` 产生；
- 卷吸由独立的法向速度跳 `q=-[[u_n]]` 产生；
- 在平面片轴线上，两者分别产生切向和法向速度，不能互相重参数化；
- 因而用 `mu_DDE` 吸收卷吸会破坏速度跳身份或引入未命名源项。

可动空间只包括开放的 `N2.6c2` 和它提供给 N3 的新生自由片状态。冻结的
N1、统一压力恒等式、DDE 材料 Kelvin/Helmholtz 恒等式和 ForceLedger 均不改。

## 2. 学科机理

DeVoria & Mohseni（JFM 866, 2019）把有限厚度黏性层压缩成
vortex-entrainment sheet（VES）。该二维流形在三维空间中保存：

```text
rho_s = integral rho dn                 片面质量
v                                         片面内禀速度/动量
gamma = n cross [[u]]                    切向涡片强度
q = -n dot [[u]]                         卷吸强度
[[p]]                                    两侧压力跳
```

其质量与动量方程为

```text
D_s rho_s/Dt + rho_s div_s(v)
  = -[[rho (u-v) dot n]]

rho_s D_s v/Dt - (div_s(T_s) + rho_s f_s)
  = -[[rho (u-v) ((u-v) dot n)]] - [[p]] n + [[tau]] .
```

外流诱导速度同时含两个线性独立的表面核：

```text
u(x) = 1/(4 pi) integral_S
       [gamma cross (x-x_s) - q (x-x_s)] / |x-x_s|^3 dA .
```

当 `rho_s=q=0` 时，质量方程强制法向速度连续，动量方程使自由片压力跳为
零，模型才退化为普通无质量涡片。因此“普通 DDE 足以携带卷吸和法向动量”
是错组成部分；DDE 仍是 VES 的 `gamma`/势跳子状态，不应被删除。

Terrington、Hourigan & Thompson（JFM 936 A44, 2022）给出移动三维界面
上的环量生成与守恒，区分壁面相对加速度产生界面环量和黏性向流体内部
转移。它支持 N2.6a/b 的上游环量账，但不提供分离后的质量/动量状态。

Xia & Mohseni（JFM 830, 2017）在尖锐尾缘控制体上联立 Kutta、环量、质量和
动量，说明新生片的方向、强度与相对速度不是一个 LESP 幅值能够决定。
然而该结果依赖尖锐边几何，不能直接移植到 RoboEagle 光滑厚翼前缘。

## 3. 缺件/错件裁决

### 错件

**纯 P2 DDE 作为完整分离层状态：NO-GO。**

它可以且应继续保存连续势跳和涡片强度，但不能表示非零 `q`、`rho_s v` 或
由法向动量平衡支持的压力跳。给 DDE 增加核、LESP 幅值或总力标定不能修复
这个状态秩缺失。

### 缺件

在 DDE 旁增加 VES companion state：

1. `q_sep`：分离片两侧法向速度跳/卷吸；
2. `rho_s`：从有限厚度黏性层压缩得到的片面质量；
3. `rho_s v`：片内切向与法向动量；
4. `gamma`：由 DDE 势跳梯度保存的切向涡量；
5. `[[p]]`：只由片面法向动量和统一两侧外流共同决定；
6. 分离流形处 IBL 库存到上述状态的具名守恒 junction。

`q` 的 source-sheet 诱导速度必须进入 N1–N2.6–N3 隐式组。它不是新增
气动力；所有作用仍经同一束缚解和统一 Bernoulli 面板压力成力一次。

## 4. 证据边界

DeVoria–Mohseni 给出三维表面理论及 Falkner–Skan 边界层映射，但公开的自由
分离演示主要是二维尖锐边。故当前允许：

- 冻结 VES 的状态身份、诱导核、质量/动量守恒和普通涡片退化极限；
- 把纯 DDE 完整态命题判为 falsified；
- 把 VES companion state 设为 partial。

当前不允许：

- 宣称已有光滑前缘 `N2.6c1 -> N2.6c2` 释放闭合；
- 从总 L/T 或压力误差反演 `q`、`rho_s` 或片内速度；
- 把尖锐边 Kutta 条件作为光滑前缘生产边界；
- 在 `q` 的片上主值、自由面推进和场级验证前进入 V4.1 生产。

## 5. 首个预登记门

首个 CPU 方程 oracle 只验证状态表示和守恒语义：

1. **核分解**：平面对称片轴线上，`gamma` 与 `q` 分别产生正交的切向/
   法向诱导速度，叠加严格线性；
2. **旋转客观性**：源面、强度和观测点共同刚体旋转后速度同样旋转；
3. **远场卷吸**：总卷吸 `Q=integral q dA` 的远场恢复 source monopole
   `-Q r/(4 pi |r|^3)`；
4. **质量/动量账**：制造解逐项满足 VES Eq. (6)–(7)，遗漏卷吸或压力跳
   必须留下可见残差；
5. **退化极限**：`q=rho_s=[[p]]=0` 时严格退化为普通 DDE/涡片场；
6. **释放 junction**：IBL 输入与新生 VES 的环量、质量、动量和卷吸分别
   守恒，任何通道不得互相吸收。

这些门通过只允许把“VES 是必要且可守恒表达的 companion state”晋升，
不验证 `N2.6b4` 闭合、spike 位置、光滑前缘释放量或任何载荷。

## 6. 原始来源

- DeVoria, A. C. & Mohseni, K., *The vortex-entrainment sheet in an
  inviscid fluid: theory and separation at a sharp edge*, JFM 866 (2019),
  660–688, doi:10.1017/jfm.2019.134.
- Xia, X. & Mohseni, K., *Unsteady aerodynamics and vortex-sheet formation
  of a two-dimensional airfoil*, JFM 830 (2017), 439–478,
  doi:10.1017/jfm.2017.513.
- Terrington, S. J., Hourigan, K. & Thompson, M. C.,
  *Vorticity generation and conservation on generalised interfaces in
  three-dimensional flows*, JFM 936 (2022), A44,
  doi:10.1017/jfm.2022.91.

