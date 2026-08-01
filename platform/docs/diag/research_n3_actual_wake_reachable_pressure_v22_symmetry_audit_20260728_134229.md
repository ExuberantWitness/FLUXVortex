# S3ai-v2.2：continuum-zero initializer 与零安全展向对称门

时间：2026-07-28 13:42:29 +08:00  
节点：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
裁决：**v2.1 的 exact-zero/relative-mirror 观察件错误；正式执行继续
NO-GO，先冻结 v2.2**

## ① 数据指纹、树节点和可动空间

本次没有执行 31 条正式 history，也没有产生 pressure-law 物理结果。
实现审计只运行了 `alpha=0` 的 entrance prestep：

| body/direct-W q | `maxabs(c)` | 旧 relative mirror |
|---:|---:|---:|
| 8 | 2.0602143e-6 | 1.97260 |
| 10 | 6.2977665e-7 | 1.99068 |
| 12 | 2.0476205e-7 | 1.99915 |

与此同时，correct-sign `face_mu` attachment inventory 精确为零。q10 下
`dt=0.25,0.125,0.0625` 的 `maxabs(c)` 只从
`7.8919e-7 → 7.0531e-7 → 6.2978e-7`，不呈 midpoint 二阶；而 q8→q12
下降约十倍。完整向量又几乎全是 span-odd。

因此异常优先挂到开放节点的**数值观察/共享边求积**分支，而不是物理
forming history：

- 连续对称问题的 exact initializer 是零；
- 有限 q 的 body/wake BIE 在共享边求积下产生 span-odd 离散误差；
- `max|v-Jv|/max|v|` 在真值为零时是“误差除以误差”，误差趋零仍可趋近
  常数 2；
- 现有诊断中的 trace `dual-mass` 数只作定位指纹，不能当正式误差证明；
  trace 的正确度量是 primal mass norm。

唯一可动空间是更正测量协议：保留相同 q/dt/历史和物理方程，改成 typed
even/odd 投影与同空间误差区间。禁止通过长 prehistory、burn-in、core、
clamp、压力或力目标消除该指纹。

## ② 学科机理

这是离散对称性和空间角色问题，不是新增气动经验律。

canonical cut 的反射是数组反向 `J9/J7`。几何反射不交换 upper/lower，
所以对 span-even 标量势有

```text
C Q = J9 C,       J9 c = c.
```

弱压力的上/下定义也不交换；反射只改变速度的 y 分量，速度模平方不变，
所以 `J7 P=P`。active consistent P2 mass 在实际 canonical 上满足

```text
J7 M J7 = M,      J7 M = M J7
```

且当前检查残差逐位为零。由此

```text
R = M Δg + dt P,  J7 R = R.
```

故 trace、weak pressure、step/window residual 和 centered tangent 的
物理目标全是 **even**；应趋零的是 odd 部分，而不是完整向量。

令

```text
Pi+ = (I+J)/2,    Pi- = (I-J)/2.
```

trace/release/inventory 是 P2 系数场，用
`||g||_M=sqrt(g^T M g)`；weak pressure/residual 是测试协向量，用
`||r||_(M^-1)=sqrt(r^T M^-1 r)`。二者不可混用。因为 `JM=MJ`，偶/奇
子空间在两种度量下都正交。

## ③ 缺组成部分还是组成部分错

本轮裁决是：**观察组件错，不是已有数据证明缺 forming state**。

- exact-zero 目标没有错；错的是要求有限 q 数组逐点为零；
- reflection map `[::-1]` 没错；错的是 near-zero relative ratio；
- 一个 `[-dt,0]` prestep 仍有必要，用于取得 stored phi 和完整 material
  state，但它不进入 pressure window；
- 固定长 prehistory 没有文献或控制体时间尺度，会改变 wake age/length，
  等价于用 burn-in 吸收离散误差，禁止采用。

若 v2.2 的完整向量 q-tail 不收缩，或 odd 零区间排除零，下一病灶只能是
symmetry-preserving body/wake BIE/共边求积；不得据此跳到 Xia forming
geometry 或 finite VES。

## ④ 有证据的方案与 go/no-go

时间戳冻结资产为
`actual_wake_reachable_pressure_obstruction_cases_20260728_134229.yaml`。
它不修改 v2/v2.1，而以 addendum 方式固定：

1. t=0 只硬验 BIE、typed attachment、inventory、seam/tip、chronology
   与不变性，不再要求有限 q trace 精确为零；
2. 对每个既有 epsilon/dt/q/mixed/repeat/direct-observer family，先按
   物理空间投影，再分别构造 `U+` 与 `U-`；
3. odd 区间
   `I-=[max(0,||Pi-v||-U-), ||Pi-v||+U-]`；
4. `L->0` 是 resolved symmetry failure；`L-=0` 只称
   `NO RESOLVED SYMMETRY VIOLATION`，不夸称 certified；
5. 只有 `L+>0` 才可报告 `H-/L+`，禁止用观测到的 near-zero 值作分母；
6. round floor 从投影前 operands 定标，并在总区间中只加入一次；
7. 对 zero history 另算 stagewise
   `V-=sum ||Pi- R_n||_(M^-1)`，禁止 window cancellation；
8. 冻结 exact even/odd manufactured controls、wrong reflection、exact
   zero 与 `2^-40` near-zero controls。

在 v2.2 runner、负对照和独立代码审计完成前，正式执行保持 fail-closed。
即使后续通过，本门最多产生 fixed-space named-pressure-law witness；仍需
单独的 actual-body h/p 复现，才可裁决 massless forming geometry 与 finite
VES。

## 边界

- 本文没有改气动公式、常数、网格、运动学或 claim 状态。
- 没有运行 force、production、118 sweep 或 Fig17/18/19。
- D16 的结论不变：`face_mu` inventory 是表示守卫，不是独立 Kelvin
  equation。
- 新 D17 只推导 reflection parity、primal/dual 正交分解和零安全区间。
