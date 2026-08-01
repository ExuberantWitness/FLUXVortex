# N3 S3m：P2 wake 的 typed inflow/outflow trace

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b2b1a`  
状态：**EXECUTED / GO**

## ① 病因定位

S3l 的 free/constrained block 通过 partition、rank、zero/nonzero trace、
patch trace、时间收敛、精度、刚体和 fail-closed 门，但 clamp-only 终态
free 差只有 `3.499053e-5`，未过冻结的 `1e-4` 可辨识门。

数据指纹不是“block 方程错误”，而是：

```text
c_s = a sin(pi s)
c_s(s=1) = 0
```

名义 body-attachment edge 没有把材料势跳送入 wake domain。该边在 S3l 是
characteristic，不是真实 inflow。

可动空间仅限于制造相对边界通量、显式边界角色和 trace provider。S3j
consistent P2 assembly、S3k moving/curved patch composition、S3l block
equation与Heun语义均冻结。

## ② 学科机理

一阶输运只在

```text
Gamma_in = {x on boundary: c_t dot nu_boundary < 0}
```

给定进入域内的状态；outflow 不能当作同一个 essential trace。
Demkowicz–Gopalakrishnan 对 transport boundary-value problem 显式采用该
定义。

对 wake，Xia–Mohseni 将 forming sheet 解释为从 shedding edge 释放的
material streakline，并由守恒律确定强度和相对速度；DeVoria–Mohseni
进一步指出 edge pressure jump 会生成新涡量。故 body attachment 是新生
sheet state 的 source/inflow，old edge 是历史离开计算带的 outflow。

来源：

- Demkowicz & Gopalakrishnan, CMAME 199 (2010)：
  https://doi.org/10.1016/j.cma.2010.01.003
- Xia & Mohseni, JFM 830 (2017)：
  https://doi.org/10.1017/jfm.2017.513
- DeVoria & Mohseni, JFM 866 (2019)：
  https://doi.org/10.1017/jfm.2019.134

## ③ 缺件还是错件

| 组成部分 | 裁决 |
|---|---|
| S3l free/constrained block | 未被证伪，直接门通过 |
| S3l nominal attachment | 错误实验角色：零通量 characteristic |
| nonzero typed body inflow | 缺件 |
| old edge essential trace | 错件；应为 free outflow |
| characteristic tips | free material boundary |
| 降低 S3l clamp 阈值 | 禁止 |

## ④ 预登记方案

在同一 moving-curved 四条 chronological strips 上固定

```text
ds/dt = -1, dr/dt = 0
```

因此：

- `s=1` body edge：`c·nu=-1`，唯一 typed essential inflow；
- `s=0` old edge：`c·nu=+1`，free outflow；
- `r=0,1` tips：`c·nu=0`，free characteristic。

先做两个代数激励，分别隔离 `M_fb*g_dot` 与 `C_fb*g`。尤其在全零状态下
给非零 `g_dot=q(r)`：正确 block 必须产生 nonzero free response，而
full-update-then-clamp 的 free response 为零，其归一 free residual 应接近1。

再推进平滑 turn-on：

```text
g(t,r) = t^2 r(1-r)
tau = max(t-(1-s), 0)
mu_exact = tau^2 r(1-r)
```

时间族固定为 `8/16/32`，验证 body trace、free old/tips、Cauchy、L2、
clamp 反事实、patch trace和刚体客观性。全部阈值已冻结在
`p2_wake_inflow_trace_cases.yaml`；执行前不得试跑或改门。

本门仍明确禁止 actual induced velocity、pressure、force、LEV、target
load和任何结构动力学。

## ⑤ 执行结果

冻结输入和阈值一次执行得到 **GO**：

| 通道 | 结果 |
|---|---:|
| role partition / constrained nonbody | `0 / 0` |
| body / old / tip flux error | `0 / 0 / 0` |
| free mass rank deficiency | `0` |
| correct block normalized residual | `6.66e-16` |
| rate-injection free response | `7.2405e-2` |
| clamp-only rate residual | `1.0` |
| prescribed / shared trace error | `0 / 0` |
| propagation time Cauchy | `4.04799` |
| finest propagation relative L2 | `2.7765%` |
| clamp-only final free error | `1.05184e-3` |
| rigid scalar error | `3.12e-17` |
| invalid-role failures | `3` |

S3m 把 S3l 未能识别的因果关系拉开超过一个数量级：
`clamp-only final error` 是冻结门 `1e-4` 的 10.5 倍；瞬时
rate-injection residual 则从正确 block 的 roundoff 变为 `1.0`。

## ⑥ Claim 裁决

`N3.1j3b6d18c2b2b1a` 在以下严格范围内 validated/frozen：

- explicit global P2 material IDs；
- body edge 是唯一 essential inflow；
- old edge 是 free outflow；
- tips 是 free characteristic；
- consistent mass free/constrained block；
- moving-curved multipatch、Heun、无稳定化。

联合 S3l 已通过的 block/trace/time/objectivity 证据，父
`N3.1j3b6d18c2b2b1` 同步改写为上述 role-aware 命题并
validated/frozen。

没有验证 actual body-cut trace provider、actual induced velocity、
relaxation、pressure、force或 production。下一节点只能是 `c2b3` 的实际
速度组成与具名耦合残差预登记。
