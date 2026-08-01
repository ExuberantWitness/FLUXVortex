# G0e Fig15 静态滑翔身份判别

日期：2026-07-30  
裁决：`NO_STATIC_CONSTANT / N2.6e1_SOURCE_METHOD_GO /
N2.6e2_TARGET_CONDITIONAL`

## 1. 问题与预登记边界

corrected Fig18 在 `f=1.4 Hz, twist=22.5 deg, AoA=5 deg` 的冻结 V4.1
推力残差 `model-experiment` 随来流为
`+1.110/+1.805/+2.038 N @ U=6/8/10 m/s`。G0e 只问：

1. 当前生产 V4.1 是否已经满足 Meng Fig15 的静态滑翔锚；
2. 一个已有文献身份的附着摩擦通道可解释多少 Fig18 缺口；
3. 是否允许把 Fig18 缺口继续解释成一个额外静态阻力常数。

不以 Fig15 选择新常数，不改 V4.1，不以总力反演压力或分离状态。

## 2. 观察身份

Meng et al. (2025) §4.1 将 Fig15 定义为双翼水平滑翔。正文写来流
`8 m/s`，图例同时列出 `6/8/10 m/s`；报告的最大升阻比为
`L/D=6.8 @ AoA=5 deg`。按论文式 (11) 的风轴定义，静态无驱动时

\[
D=-T_{\rm wind},\qquad L/D=L_{\rm wind}/(-T_{\rm wind}).
\]

必须区分两个已有配置：

- A：冻结 Fig17/18/19 V4.1，manifest 身份为 `visc=False`；
- B：历史静态物理锚，`d_para=0.5` 且 `visc=True`。

B 不是当前冻结生产身份，不能把它的 Fig15 结果写成 A 的验证结果。

## 3. 数值方法

共同配置：

- `closure=v41`，`H16`，`a0_crit=0.27`，`d_para=0.5`，
  `attached_drag=uiuc`；
- `fsep_lag=False`，`cosine_chord=le`，`les_sep=plateau_fn`，
  `geo_stall=False`；
- `real_geom=True`，`sym=True`，`les_suction=True`；
- `flap_amp=0`，`twist_amp=0`，`AoA=5 deg`，`f=2 Hz`；
- `nc=12, ns=16, n_cycle=4`；
- `steps_per_cycle=wake_rows=spc_of(U,f)`，即 U=5/8/10 分别为
  480/720/900。

B 的三个速度均独立运行。A 在 U=8 独立运行；U=5/10 的 A 值由同一 B
运行已经报告的 additive `viscous` body-force channel 精确去除得到。风轴
变换为

\[
\delta L=F_{z,\nu}\cos\alpha-F_{x,\nu}\sin\alpha,\qquad
\delta T=-(F_{x,\nu}\cos\alpha+F_{z,\nu}\sin\alpha).
\]

U=8 的去除结果与 A 独立运行相差 `<1.1e-9`，验证了该
counterfactual。

## 4. 结果

| 身份 | U (m/s) | L (N) | D (N) | L/D | Fig15 `6.8±10%` |
|---|---:|---:|---:|---:|---|
| B: `visc=True` | 5 | 2.968280 | 0.448280 | 6.62148 | pass |
| B: `visc=True` | 8 | 7.614967 | 1.072770 | 7.09841 | pass |
| B: `visc=True` | 10 | 11.903603 | 1.622749 | 7.33546 | pass |
| A: frozen `visc=False` | 5 | derived | derived | 7.79066 | fail |
| A: frozen `visc=False` | 8 | 7.626600 | 0.938105 | 8.12979 | fail |
| A: frozen `visc=False` | 10 | derived | derived | 8.30916 | fail |

B 的周期 2→4 收敛变化 `<5e-5 N`；末周期力振幅在 U=5 已
`<0.004 N`，U=8/10 更小，故上述值不是启动平均伪影。

U=8 时 B−A 为

\[
\Delta L=-0.011633\ {\rm N},\qquad
\Delta T=-0.134665\ {\rm N}.
\]

所以已有附着摩擦通道只提供 `0.134665/1.805=7.46%` 的 corrected Fig18
同速度推力缺口。它不能解释主体，也不授权按剩余误差增加摩擦系数。

壁钟时间：B 的 U=5/8/10 分别为 `71.365/145.951/196.621 s`；A 的 U=8
为 `147.165 s`。

## 5. 运行时账本缺陷

原样运行 B 时，气动力总量已经包含 `_channels["viscous"]`，但 V4.1 的
`_v41_booked` 和 `UVLMComponent.channel_names` 没有登记该 channel，导致
claim ledger 在输出阶段拒绝：

- U=5/8/10 的未分类物理力为
  `0.066785/0.135166/0.188905 N`；
- 只在测试进程内把已有 `viscous` channel 加入 ledger 后，force-ledger
  残差为 `8.88e-16/1.78e-15/1.11e-16 N`，旧气动力输出 0-bit 不变；
- `unclassified_physical_force` 仍忠实暴露分类元组遗漏；
- A 的 U=8 force-ledger/unclassified 残差均为 `1.78e-15 N`。

这是运行时来源记账缺陷，不是新的气动力候选。冻结 V4.1 使用
`visc=False`，故它不改变当前 184 工况基线数值；不得借修账本暗中改变基线。

## 6. 裁决与 claim 归因

1. **额外静态阻力常数：NO-GO。** B 已在无新常数下满足 Fig15，继续以
   `d_para`、pressure offset 或摩擦倍率吸收 Fig18 违反红线。
2. **冻结生产身份缺口：M0/N2 配置边界。** A 不满足自己的 Fig15 guard，
   说明历史“V4.1 已过静态锚”并非当前 manifest 身份；必须在观察/配置边界
   记录，不能重写冻结基线。
3. **附着壁面摩擦不是主体。** U=8 只解释 0.135 N，远小于 1.805 N。
4. **分离压力/强 VI 机理仍与数据相容，但非唯一可辨识。** 动态机构 tare
   未公开，故 N2 与 M0.2 仍存在不可辨识性；本实验不能把 Fig18 残差全部
   判给翼面。
5. **授权层级。** `N2.6e1` 来源论文二维强耦合双尾迹复现可以实施；
   `N2.6e2` 目标条带 shadow 仍须先通过来源压力/分离门及目标表示门。

因此下一步不是恢复 `visc=True` 后重跑三图，也不是增加静态阻力，而是先
完成不看 RoboEagle 总力的 SVI-DW 来源方法复现。
