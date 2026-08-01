# N3 实际边界 P2 Galerkin 成对奇异算子裁决

日期：2026-07-28  
Claim：`N3.1j3b6d3`  
范围：附着势方程的数值组成部分；不接 wake、力或生产路径。

## ① 病因定位

S1 的连续 P2 trace、弱方程代数残差和 prescribed-source 通量已经通过，
但物面压力对积分阶数敏感。固定 320 面时，order 6→10：

- 离体势误差只变化 `1.98e-5`；
- 物面速度 RMS 变化 `0.00417`；
- `Cp` RMS 变化 `0.00952`，超过预登记归因上限 `0.005`。

S1c 又把“单个源三角形的 P2 doublet 势是否算错”排除：

- 分离点 line-reduced 与 48 阶面积分相对误差 `4.93e-15`；
- 常强度面板恒等式相对误差 `4.41e-16`；
- 但靠近非 owner 公共边时，最高两阶相对变化仍为 `24.25`。

因此病因不在 P2 多项式、Green 符号或单面径向原函数，而在把
`target triangle × source triangle` 拆成两个互相独立的积分。共享边/点时，
内积分虽对每个固定 target 有定义，外积分仍面对随 target 移动的奇异层。

## ② 学科机理

Erichsen–Sauter 的 3D Galerkin BEM 工作把面元对按几何交集分类，并在积
分前进行专门变量变换。Reid–Johnson–White 的 generalized Taylor–Duffy
推导更明确：

- common triangle、common edge、common vertex 分别使用 3、6、2 个
  子域；
- source 与 target 共用一个径向尺度；
- 该尺度的 Jacobian 与弱奇异核同步消去奇异性；
- 这是积分域的精确重参数化，不是 core、offset 或平滑。

Seibel 进一步用 Sauter–Schwab 变换把非空相交的三角面元对先消奇，再做
半解析积分；同面和共边可继续降维。三条来源共同否定“只提高两个独立
面元的 Gauss 阶数”。

一手来源：

- Erichsen & Sauter, *Efficient automatic quadrature in 3-d Galerkin
  BEM*, CMAME 157 (1998), DOI:
  https://doi.org/10.1016/S0045-7825(97)00236-3
- Reid, Johnson & White, *Generalized Taylor-Duffy Method for Efficient
  Evaluation of Galerkin Integrals in Boundary-Element Method
  Computations*, IEEE TAP 63 (2015):
  https://arxiv.org/abs/1312.1703
- Seibel, *Almost Complete Analytical Integration in Galerkin Boundary
  Element Methods*, SIAM JSC 45 (2023), DOI:
  https://doi.org/10.1137/22M1534857

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| 连续 P2 势跳与同阶弱方程 | 尚未被当前数据证伪，保持 partial |
| 两面各自 tensor-Duffy 后做乘积 | 错件 |
| 精确单面径向积分自动闭合外层弱奇异性 | 错件 |
| 按同面/共边/共点分类的成对消奇积分 | 缺件，进入 S1d |
| 调高独立积分阶数、core、offset、平滑 | 禁止 |

## ④ 机理方案与预登记

S1d 直接组装每个 target/source 面元对：

1. 根据网格拓扑精确识别 common triangle/edge/vertex/disjoint；
2. 对相交面元对先应用共享径向尺度的精确子域变换；
3. 同时组装 double-layer 左端和 prescribed-source single-layer 右端；
4. 同面 planar double-layer 主值保持严格为零；
5. 分离面元仍用普通乘积积分，不扩大改动范围；
6. 先过面积分割、独立分离面 oracle、三种相交拓扑 Cauchy 门；
7. 再过单位球原阈值；任何失败只允许定位到具体拓扑/核。

完整阈值在
`actual_boundary_p2_galerkin_cases.yaml` 的
`S1d_paired_singular_galerkin_preregistered_after_S1c_before_implementation`
中冻结。S1d 仍是无力 diagnostic；通过也不授权 N1 环量、Kutta/wake、
非定常 Bernoulli 或 RoboEagle 生产载荷。

## S1d 执行结果

首轮 guard 的面积分割通过，但 common-edge doublet 的 order 6→8 变化为
`0.00907`，高于冻结 `0.005` 门。对照 Reid 等 Appendix A/Figure 2 后发现，
实现把参数原点错放在对边顶点；论文的 `V1,V2` 必须是公共边两端。只修正
这一坐标身份后，在原阈值和原几何上重跑：

- 面积分割最大相对误差 `8.88e-16`；
- 分离面 doublet/source oracle：`1.02e-13/1.21e-15`；
- 共边 doublet/source Cauchy：`1.41e-5/1.71e-6`；
- 共点 doublet/source Cauchy：`1.39e-5/1.11e-6`；
- 同面 source Cauchy：`4.60e-6`。

所以“相交面元对必须成对消奇”的机理通过，其坐标身份也被 guard 独立抓住。
单位球 level-0 的 order 6→8 势/速度/`Cp` 变化分别只有
`5.10e-7/2.70e-7/2.60e-6`，证明当前剩余误差不是积分阶数漂移。

但 80 面 level-1 仍为：

- 离体势误差 `0.12566 > 0.02`；
- 物面速度 RMS `0.15952 > 0.08`；
- `Cp` RMS `0.08898 > 0.08`。

因此 S1d 总 stage 仍为 NO-GO；窄命题“相交面 paired weak operator”可以
冻结为通过，完整 attached-sphere pressure 尚不能晋升。下一步 S1e 已在
执行前冻结为 level-2/order-8 网格归因，不改变方程或阈值。

## S1e 网格归因结果

320 面、paired order-8：

- 离体势误差 `0.03375`，仍未过 `0.02`；
- 物面速度 RMS `0.07857`，已过 `0.08`；
- `Cp` RMS `0.03393`，已过 `0.08`；
- weak/source/continuity 残差分别
  `6.13e-16/5.38e-18/0`，条件数 `10.87`。

三项均从 80 面严格下降，且压力误差从 `0.08898` 降至 `0.03393`。这排除
“paired 算子修好后压力仍不收敛”的假设；S1e 的唯一失败是离体势网格精度。
level-1→2 的势误差降低因子为 `3.72`，与 P1 几何统一加密相符。

S1f 因此预登记为一次 level-3 延拓。为避免把算力代替证据，先在 level-2
比较 order-6 与 formal order-8，三项差均须小于 `0.002`；通过后才运行
level-3/order-6。原 `0.02/0.08/0.08` 门不变。

## S1f 最终结果

level-2 order-6/8 的势、速度、`Cp` 绝对差分别为
`1.87e-7/1.20e-6/4.98e-6`，远低于预登记 `0.002`，所以 level-3 执行有效。

1280 面、2562 个 P2 自由度、paired order-6：

- 离体势误差 `0.008596 < 0.02`；
- 物面速度 RMS `0.038973 < 0.08`；
- `Cp` RMS `0.013847 < 0.08`；
- weak/source/continuity：`1.13e-15/8.47e-18/0`；
- condition number `11.59`；
- 成对拓扑账：同面 `1280`、有向共边 `3840`、有向共点 `11460`。

S1f 为 GO。由此可冻结两个窄命题：

1. 同面/共边/共点成对消奇是正确的 Galerkin 数值组成部分；
2. 连续 P2 势跳 + prescribed source + 成对弱方程 + 一致表面梯度，在足够
   解析的闭合 P1 几何上可同时恢复附着球的势、速度和压力。

父节点 `N3.1j3b6d3` 仍为 partial，因为该 GO 不包含 N1 环量兼容、有限厚度
尾缘/base wake、Kelvin/Kutta、材料势时间导数、分离压力或 RoboEagle 力。
