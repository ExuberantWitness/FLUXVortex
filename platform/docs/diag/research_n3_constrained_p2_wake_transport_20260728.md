# N3 S3l：P2 材料输运的 essential trace

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b2b1`  
状态：**EXECUTED / NO-GO**

## ① 病因定位

S3k 唯一失败是 zero boundary 从0漂到 `3.015e-4`。其余 moving geometry、
patch/monolithic、时间阶、刚体与面积门全部通过，故病因已经锁定为：

> unconstrained CG 把物理 boundary trace 当作自由 DOF。

实际 wake 还存在非零 body-attachment trace，所以“每步算完后清零”既不一般，
也会漏掉边界对 interior weak equation 的耦合。

## ② 学科机理

ALE surface FEM 中 essential trace 属于 trial/test space。将 DOF 分为 free
`f` 与 prescribed `b` 后，正确方程是

```text
M_ff mu_dot_f =
  -(M_fb g_dot + C_ff mu_f + C_fb g),
mu_dot_b = g_dot.
```

S3g 已证明 body-cut/wake scalar trace 必须基于有向 material IDs；S3h 已证明
attached geometry edge 必须在每个 stage 精确给定。scalar trace 应采用同一类
typed stage boundary，而非坐标推断。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| moving P2 matrices / Heun stages | S3k 其余门已通过 |
| essential free/constrained block | 缺件 |
| update-all then clamp | 错件，登记反事实 |
| coordinate/proximity boundary inference | 禁止 |
| pressure/force | 本门禁止 |

## ④ 预登记

沿用 S3k 曲面和 `2/4/8` 时间族，同时验证：

- 全零物理边界 `mu0=s(1-s)r(1-r)`；
- 非零 body edge `s=1`，`mu0=s*r(1-r)`；
- typed DOF partition、free mass rank、stage trace、time Cauchy、刚体；
- clamp-only 反事实必须产生至少 `1e-4` free-interior error；
- 三类非法 constraint 必须 fail closed。

阈值冻结于 `constrained_p2_wake_transport_cases.yaml`。

## ⑤ 执行结果与测量器审计

S3l 保持冻结阈值执行，最终判定为 **NO-GO**。直接候选门的结果是：

- typed partition failure `0`，free mass rank deficiency `0`；
- prescribed trace 与 shared patch trace 误差均为 `0`；
- zero/nonzero case 的 Heun Cauchy 收缩分别为 `4.3983/3.8122`；
- 最细 relative L2 error 分别为 `0.2165%/0.06740%`；
- 三类非法 constraint 均 fail closed。

第一次执行出现刚体差值 `2.0`。逐量追踪证明这不是 transport
operator：逆刚体变换把名义 `r=1` 的制造 material label 移到
`1+O(1e-16)`，使 `tan(pi*r/2)` 穿过解析分支，制造边界值被错误算成
`[-1.5,2]`。首次结果完整保存在
`constrained_p2_wake_transport_results_initial_invalid_oracle.json`。测量器只在
确认 material label 位于 `[-2e-12,1+2e-12]` 后投影回其闭区间身份；
候选方程、状态、阈值和生产代码均未改变。复跑后刚体差值为
`2.98e-16`。

唯一有效失败是：

```text
clamp-only final free-interior difference
  observed = 3.499053e-5
  frozen gate = 1.0e-4
```

因此不能降低阈值后宣布 GO。

## ⑥ 新病因：attachment 反事实没有被激励

当前制造速度

```text
c_s = a sin(pi s)
```

在名义 body edge `s=1` 恰为零。该边是 characteristic boundary，而非把
新生势跳输送进 wake domain 的真实 inflow。虽在 `t=0` 测得 constrained
与 full-update rate 的 free-DOF 最大差为 `1.4666e-4`，积分终态差被压到
`3.4991e-5`，所以 S3l 不能唯一识别 `M_fb*g_dot+C_fb*g`。

这不是“essential block 错件”的证据，而是**实验组成部分缺少非零
attachment inflow flux**。父 claim 保留 partial，不得进入 actual induced
velocity。

输运理论把边界分为
`Gamma_in={x on boundary: c_t dot nu_boundary < 0}` 与 outflow；给定状态只应
进入 inflow。DeVoria–Mohseni 的 sharp-edge vortex-entrainment sheet
同时指出 edge pressure jump 生成新涡量并进入新生 sheet。因此下一门必须：

- body attachment 是显式 typed inflow，并有非零相对边界通量；
- old edge 是 outflow，不施加 essential scalar trace；
- span tips 是 characteristic/material 边；
- 仍用同一 free/constrained block，并把 clamp-only 作为冻结反事实；
- 先预登记再执行，不改 S3l 的输入或阈值。

文献：

- Demkowicz & Gopalakrishnan, *A class of discontinuous
  Petrov–Galerkin methods. Part I: The transport equation*，以
  `beta·n<0` 定义 global inflow boundary：
  https://doi.org/10.1016/j.cma.2010.01.003
- DeVoria & Mohseni, *The vortex-entrainment sheet in an inviscid
  fluid: theory and separation at a sharp edge*，JFM 866 (2019)：
  https://doi.org/10.1017/jfm.2019.134
