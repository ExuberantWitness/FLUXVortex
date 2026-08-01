# N3 S3f：三维 material-wake 几何进入 actual-boundary 方程

日期：2026-07-28  
Claim：`N3.1j3b6d18c1`  
当前状态：**EXECUTED / NO-GO**  
角色：无压力、无力的几何接口与客观性 oracle。

## ① 病因定位

S3e 已证明固定 body、规定均匀 `x` 对流下的 material history 时间收敛。
同时，`N3.1j4c5/c6` 已证明任意 P2 material sheet 的无核几何推进、Kelvin
身份和显式 patch seam。

两条证据链目前没有接上：`solve_actual_boundary_body_wake_p2()` 只接收
`wake_edge_x_nodes`，再从 body cut 复制 `y,z` 来重建所有直带。一个已经卷曲的
三维 wake 无法进入下一次实际边界方程。若此时直接编写“步内 equilibrium”，
迭代器看到的仍是被压平的 wake，属于空壳。

病因挂在 `N3.1j3b6d18c`，判定为**缺少显式三维 wake geometry 接口**。可动
空间只限输入表示和 affine assembly adapter；body/cut、P2 空间算子、符号、
已验证 old-known/current-unknown 分区以及 pressure/force 路径全部冻结。

## ② 学科机理

- Krebs 2021 §3.2–3.3：relaxed wake 是 gapless material sheet，顶点按
  local velocity 移动；移动后的几何继续参与 surface/new-wake 解。
- Krebs–Bramesfeld–Cole 2022：surface 与 wake strength 相互影响，已脱落
  element 的强度保持在其移动几何上。
- Bramesfeld–Maughmer 2008：wake-sensitive 载荷依赖 wake shape；连续
  distributed-vorticity wake 的意义正在于可无经验 core 地卷起。
- Pate 2017：bound body 与三角化 deforming wake 应进入同一一致的
  boundary-element 几何和 circulation ledger。
- NASA TP-2995：panel network 的几何与 abutment 是显式拓扑数据，不能由
  距离猜测替代。

因此正确顺序是：

```text
explicit moved material history
  -> geometry-general actual-boundary affine solve
  -> 再组合 wake velocity / Heun / within-stage equilibrium
```

而不是先做迭代器，再在每次迭代中把 wake 压回直带。

## ③ 缺件还是错件

| 组成部分 | 判定 |
|---|---|
| `wake_edge_x_nodes` 直带路径 | S3b–e canonical 仍有效，冻结保留 |
| 用直带重建替代 relaxed 三维 geometry | 对 d18c 是错件 |
| typed `MaterialWakeHistory` 几何输入 | 缺件 |
| proximity welding / 重新平均 old rows | 禁止 |
| 几何接口通过前做 pressure/force | 禁止 |

## ④ 预登记

新增 typed `prescribed_wake_history` 候选，但不删除旧入口。三组冻结门：

1. **straight equivalence**：同一两带 history 分别走旧 `x-node` 路径和新
   explicit 路径，比较 matrix、RHS、body potential 与 cut jump；
2. **curved consumption**：声明非平面的 far edge 与 seam edge，检查输出几何
   未被压平、old rows 未变，并要求当前 cut jump 对几何变化有可分辨响应；
3. **rigid-frame objectivity**：body、wake、incident/wall vectors 一起作
   任意刚体旋转平移，标量方程和 cut jump 必须不变。

同时检查 material interfaces、body-wake singular topology、rank、条件数、
弱残差、attachment 与 tip identity。阈值已冻结于
`actual_boundary_explicit_wake_geometry_cases.yaml`。

即使 GO，也只授权下一步组合已有 Heun wake advection；不授权 moving body、
步内 equilibrium、LEV、压力、力或生产。

## 执行结果：NO-GO，缺的是 material orientation

直尾迹、曲尾迹和代数门均通过：

- 新旧直带路径的 matrix、RHS、body potential、cut jump 差均为 `0`；
- 输入/输出 geometry、old/active-known strength 与全部 history interface
  mutation 均为 `0`；
- 曲尾迹使当前 cut jump 改变 `1.516e-3`，证明几何确实被方程消费；
- body-wake common-edge pair 始终仅最新带 `8` 对；
- rank deficiency `0`，最大弱残差 `3.51e-16`。

但预登记的通用刚体旋转使 classified cut 的世界坐标字典序反转。随后 wake
span 参数方向、面法向与 doublet jump 的方向身份一起丢失：

- rigid matrix error `1.889e-3`；
- rigid RHS error `3.422e-4`；
- rigid cut-jump error `3.842e-2`。

两个反事实把病因从底层积分中分离出来：

1. 绕 `x` 轴旋转、保持 cut 字典序不变时，matrix/RHS/jump 客观性误差仅
   `4.86e-17 / 9.38e-18 / 3.83e-15`；
2. 对原通用旋转恢复同一 material cut 顺序后，误差恢复为
   `5.55e-17 / 1.78e-17 / 6.63e-15`。

因此被证伪的不是“三维 wake geometry 应进入方程”，而是：

> 仅靠 vertices/faces/rows、没有有向 body-cut ↔ wake material trace 映射，
> 足以保持物理身份。

`N3.1j3b6d18c1` 按该充分性命题 `falsified/frozen`。下一候选
`N3.1j3b6d18c1a` 必须显式携带 ordered body material vertex IDs、
P2 trace permutation 与 `wake_mu = s_attach * body_jump` 的有向符号。
不得用世界坐标排序、距离匹配或事后翻转数组修补。

