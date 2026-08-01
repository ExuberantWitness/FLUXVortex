# N3 S3b：三维实际边界—物质 wake 联立方程门

日期：2026-07-28  
Claim：`N3.1j3b6d16`  
运行角色：无压力、无力的 equation/identifiability oracle。

## ① 病因定位

S3a 已证明共享物理几何可以承载分类 P2 势切口，并把 body jump 精确交给
material wake。但是它没有回答：

> wake 的 double-layer 影响进入 actual-boundary 方程后，系统是否仍然可辨识，
> 还是必须人为再指定一次 circulation？

若 body jump 与 wake strength 被分别当成未知量，会重复同一个环量账；若只建
body cut 而不把 wake 影响放入方程，则 cut 是一条暴露的非零线涡，也不是完整
lifting-domain 解。

## ② 学科机理

NASA TP-2995、Hess lifting-body panel formulation 与 PAN AIR 的共同结构是：

1. 实际物面 source 负责无穿透；
2. body doublet/potential trace 负责外部切向势；
3. wake doublet 是势跃，其上游边必须等于 lifting surface 的上下势差；
4. steady load-free wake 沿流向保持该势跃，而不是新增一组可调 circulation。

Krebs–Bramesfeld–Cole 的 DDE 表示把第 3–4 点提升为连续 P2 material
trace。由此，本阶段的可动空间只有：

> 把 wake 的 double-layer influence 作为 body cut jump 的线性映射消元进
> actual-boundary Galerkin 矩阵。

不允许增加规定环量、Kutta 常数、LESP 幅值、正则项或目标力。

## ③ 缺件还是错件

| 命题 | 判定 |
|---|---|
| body circulation 与 wake circulation 是两个独立未知量 | 错件，重复记账 |
| 只要 body cut 存在，wake 可在求解后附加 | 错件，方程漏掉 wake influence |
| wake strength 由同一 body jump 映射并进入一套方程 | 缺件 |
| steady 方程通过即可声称 unsteady Kelvin/LEV 完成 | 错件 |

## ④ 预登记方案

继续使用 S3a 的闭合 full-wing diamond body 和分类 TE cut。对
`alpha=-5/0/+5°`、wake downstream edge `x=2/4/8` 及 paired singular
quadrature `q=3/4/5`，组装：

```text
[body source] + [body double layer]·mu_body
              + [wake double layer]·J_cut·mu_body = 0
```

其中 `J_cut` 是 upper-minus-lower trace；未知向量中没有 wake amplitude。
对矩阵秩、条件数、弱残差、wake attachment、零攻角、攻角反对称、展向镜像、
翼尖零跃变、非零 circulation、far-wake Cauchy 和 quadrature Cauchy 分别设硬门。

完整输入、阈值、GO/NO-GO 和禁止项已在实现前冻结于
`actual_boundary_body_wake_coupled_cases.yaml`。本门即使通过，也只授权下一步
unsteady material-history/Kelvin gate，不授权 pressure、force 或 production。

## 执行结果：NO-GO

冻结门执行后，代数层面看似良好：

- 系统为 `81×81`、rank 81，没有独立 wake/circulation 未知量；
- 最大条件数 `270.96`，归一弱残差 `4.08e-16`；
- body jump 到 wake、翼尖零 jump、三行 material identity 与 wake 内部
  P2 continuity 均严格通过。

但物理/离散门失败：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| zero-alpha 假 jump | `5.76e-5` | `1e-10` |
| ±alpha 反对称误差 | `1.15e-4` | `1e-10` |
| span mirror | `5.44e-5` | `1e-10` |
| q4→q5 root Cauchy | `0.269` | `0.02` |
| wake x4→x8 root Cauchy | `0.315` | `0.10` |

因此“小残差＋满秩”不能支持 claim，`N3.1j3b6d16` 必须
`falsified/frozen`。

## NO-GO 后的病因拆分

只读高阶指纹显示，在固定 `x=8` 时 root jump 随
`q=3/4/5/6/8/10/12` 逐步趋向
`−0.03446/−0.07059/−0.09660/−0.11247/−0.12489/−0.12631/−0.12651`；
这说明 finite-angle body–wake 共边 pair 并非代数上无解，而是冻结的低阶
quadrature 尚未进入收敛域。

更关键的是：把**一条** P2 band 从 TE 直接拉到远场会失去 shape regularity。
固定 `q=10` 时，cutoff `x=4/8/16/32` 的 root jump 为
`−0.12557/−0.12631/−0.11972/−0.08157`，同时 mirror residual 重新增大。
因此“加长同一面元等于 far-wake 收敛”是错组件，不是应该提高 q 或放宽门的
常数问题。

NASA 的 wake-network 方法使用多个 wake panels/strips；Krebs DDE 的
材料时间步也逐带生成 wake，而不是持续拉长第一带。下一 live claim
`N3.1j3b6d17` 必须建立：

> shape-regular chronological material bands ＋ 显式 P2 interface ＋
> 仅第一带参与 body-wake 共边 paired singular block。

该方向仍是 equation-only，不得接 pressure/force。 
