# N2.6e1b1 来源忠实尾缘渐近加密预登记

日期：2026-07-30  
Claim：`N2.6e1b1`  
状态：`PREREGISTERED / NOT EXECUTED`

## ① 病因定位

`N2.6e1b` 的首个独立 panel/time/core 门为 **NO-GO**。七个冻结工况均
完成，Kelvin、无穿透、线性系统及 Eqs. (7)--(8) 的代数残差均不大于
`O(1e-12)`，`core=0.02c -> 0.01c` 也通过。但是：

- `32 -> 64` panels/side 时，下/上尾缘控制点速度分别变化
  `7.87% / 9.16%`；
- 二者均值变化 `8.49%`，因此 Eq. (7) 新生段长度同样变化；
- 二者差值变化 `9.28%`，因此 Eq. (8) 新生片强度同样变化；
- 长度与强度乘积的部分抵消使出生环量只变化 `1.58%`，故 Kelvin
  账本通过不能证明出生状态收敛；
- 诊断性的 `64 -> 128` 计算仍给出约 `7.60% / 9.14%` 的两侧速度变化，
  暂未显示进入快速收敛区。

病因唯一挂到 `N2.6e1b` 的“尾缘新生涡段离散”子命题。当前可动空间只有
面板加密层级；环量方程、尾迹核、时间步、运动学、实际翼面、IBL 闭包和
任何 RoboEagle 数据均冻结。

## ② 学科机理与来源边界

Riziotis (2003) 博士论文给出的离散身份是明确的：

- 印刷页 7.23，Eqs. (7.94)--(7.95)：上下尾缘速度的均值给出新生段
  增长率，差值给出新生涡片强度；
- 印刷页 7.28：每个直线面板的中点定义为 control point，并要求尾缘
  更密的离散以更好逼近尾缘速度和尾迹量；
- 印刷页 7.40，Eq. (7.126)：尾缘上下速度近似为最邻近两个 control
  point 的相对切向速度；翼面速度只在 control points 计算；
- 印刷页 7.42，Eq. (7.131)：double-wake 继续使用相同定义。

Riziotis--Voutsinas (2008) 文章页 190 的 Figure 3 也在尾缘上下相邻位置
标出 `(c.p.)`，随后 Eqs. (7)--(8) 使用其均值和跳量。因此本候选不能把
采样点改成 cusp 极限、外推值、有限部或经验偏移。

与之独立，Ardonceau (2009), DOI `10.1016/j.crme.2009.05.004` 指出：
有限尾缘角的二维局部压力具有奇异演化，半网格处的局部 Kutta 条件会产生
显著网格依赖，简单外推不能恢复正确极限。这只解释为何必须做更深的渐近
检验，不授权本候选改写 Riziotis 离散。

## ③ 缺件还是错件

当前证据尚不能区分：

1. `32/64/128` 仍处在前渐近区，来源指定的最近控制点定义在更细网格上
   会进入 Cauchy 区；或
2. 对当前闭合 NACA0015 有限角尾缘和常源/均匀环量表示，最近控制点
   出生状态不存在可用的独立空间收敛。

因此本轮判定为“数值渐近证据缺件”，不是物理公式或常数错误。

## ④ 单一候选与 go/no-go

候选 `N2.6e1b1`：

> 完全保持来源指定的最近上下控制点 Eq. (7)--(8) 离散，只把
> cosine-clustered NACA0015 网格推进到 `64/128/256 panels/side`，
> 固定 `32` 个 ramp steps、`core=0.02c` 和既有
> `U=9 m/s, alpha: 0 -> 6 deg, 0.4 s` 半余弦运动。

禁止项：

- 不读取 Fig17/18/19 或 Figure 12 响应；
- 不改公式、符号、采样位置、网格分布、时间步、核半径或阈值；
- 不加外推、滤波、松弛、epsilon、速度钳位或目标常数；
- 不修改已冻结的 `N2.6e1a`；
- 本门只检验空间渐近性，不把通过解释为 IBL、压力或载荷验证。

冻结观测量：

- lower/upper nearest-control-point downstream trace；
- mean trace、Eq. (8) jump；
- Eq. (7) newborn length、newborn sheet strength、newborn circulation；
- newborn endpoint `x/c,y/c`；
- bound/wake circulation、wake signed centroid 和 first moment；
- Kelvin、normal-BC、Eq. (7)、Eq. (8) 及线性系统残差。

对每个观测量，定义

```text
score(middle,fine) =
    abs(fine-middle) / max(abs(fine), 0.02*physical_scale)
```

速度/片强度的 `physical_scale=U`，长度/位置为 `c`，环量为 `Uc`，
一阶矩为 `Uc^2`。判据：

1. 三个唯一网格级别全部完成且无 branch ambiguity；
2. 所有代数残差 `<=1e-9`；
3. 每个观测量的 `64->128` 与 `128->256` score 均有限；
4. 每个观测量的 `128->256` score `<=2%`；
5. 每个观测量的最后一步 score 不大于前一步 score（允许
   `1e-12` 浮点裕量）。

任一失败即 `N2.6e1b1 = falsified/frozen`，禁止继续靠增加固定网格或放宽
门槛重走。只有全部通过，才保留来源离散并另行预登记时间收敛门；本门本身
不授权进入 `N2.6e1c`。

## 一手来源

- V. A. Riziotis, *Aerodynamic and aeroelastic analysis of stall on wind
  turbine rotors*, 2003, DOI `10.12681/eadd/16690`, printed pp. 7.23,
  7.28, 7.40, 7.42.
- V. A. Riziotis and S. G. Voutsinas, *International Journal for Numerical
  Methods in Fluids* 56 (2008) 185--208, DOI `10.1002/fld.1525`,
  article p. 190.
- P. Ardonceau, *Comptes Rendus Mécanique* 337 (2009) 208--217,
  DOI `10.1016/j.crme.2009.05.004`.
