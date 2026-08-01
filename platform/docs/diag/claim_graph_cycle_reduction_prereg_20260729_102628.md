# R0 图级周期归约契约预登记

时间：2026-07-29 10:26（Asia/Shanghai）  
范围：只修正 ClaimGraph 力账归属与守卫，不改变 V4.1 的任何气动公式、常数、网格、运动学或报告的 L/T。

## ① 数据指纹、唯一 claim 与可动空间

- 条件性 66 点运行在 `6_2.6_27.5_5` 两次稳定停止；两次均为
  `force_ledger=PASS`，但旧 `unclassified_force=0.18909270180913929 N`。
- 单独冷/暖重复该工况时旧 remainder 约为机器精度；严格按
  `anchor → anchor → probe` 顺序时稳定复现失败。
- 复现序列的 probe：
  - reported：`L=5.7494141888987595 N`，`T=0.7533914662914185 N`；
  - instantaneous arithmetic mean：`L=5.559600113876473 N`，
    `T=0.7534020983934296 N`；
  - reported−arithmetic 的 body-frame 向量：
    `[-0.016532795048874012, 0, 0.1890927018091382] N`；
  - 旧 N1 `uvlm_remainder`：
    `[-0.01653279504887406, 0, 0.18909270180913929] N`；
  - 两者最大差：`1.0824674490095276e-15 N`。
- 唯一病因：
  \[
  r=R\!\left(\sum_j f_j\right)-\sum_jM(f_j)=R(F)-M(F),
  \]
  其中报告总力使用 winsorized robust reducer \(R\)，节点通道使用普通
  周期均值 \(M\)。这不是遗漏气动力，而是图级时间归约组件缺失。
- 唯一 claim：新增图级 `R0 CycleReduction`；不得挂到 validated/frozen N1，
  也不得改写 N2/N3/N4/N5/N6 的物理命题。
- 可动空间仅限 L/T 已计算完成后的运行时账本归属、manifest 和守卫。

## ② 机理证据

这是确定性的数值归约恒等式，不是待选择的气动学机理。证据来自冻结源码
`_v2_robo.py` 的 total robust mean、channel arithmetic mean，以及上述三调用
可执行复现。开放式气动文献不能改变非线性算子
\(R(\sum f_j)\ne\sum R(f_j)\)；因此本案不以文献替代代码代数。

## ③ 裁决

裁决为“缺组成部分”：ClaimGraph 缺少 graph-level cycle reducer，而不是
N1–N6 中某个气动力组件错误。当前把该差值归给 N1 `uvlm_remainder` 是错误归属。

## ④ 单一候选

只实现一个候选：

1. 保持逐时步物理总账 \(F_t=\sum_j f_{j,t}\)；
2. 用 arithmetic-cycle ledger 检查物理通道闭合；
3. 只对合计总账执行一次既有 robust reducer；
4. 定义
   \[
   q_{\mathrm{num}}=R(F)-M(F)
   \]
   并由独立 `R0` diagnostic 节点记账；
5. N1 的 `uvlm_remainder` 固定为零；
6. `unclassified_physical_force` 检查
   \(M(F)-\sum_jM(f_j)\)，不得用 \(q_{\mathrm{num}}\) 冒充物理力。

禁止分别对每个物理通道执行 robust mean，禁止放宽力账容差，禁止修改报告 L/T。

## Go / No-Go

只有同时满足以下条件才 GO：

1. 上述 `anchor → anchor → probe` 三次调用的 L/T 与改动前逐 bit 相同；
2. probe 的 `R0 q_num` 与旧 remainder 逐轴差 `≤1e-12 N`；
3. probe 与无裁剪 anchor 的 `unclassified_physical_force ≤1e-9 N`；
4. N1 implementation hash 不变，R0 有独立实现指纹和 manifest 身份；
5. graph reported ledger、物理 arithmetic ledger、runner 双 guard 全通过；
6. 现有 benchmark/runner 测试和新增正负对照全部通过；
7. 重新从冻结 118 seed 开始干净运行 66 点，不静默复用旧 28 点。

任一条件失败即 NO-GO：R0 候选记为 falsified，回查真实通道遗漏或数值不稳定。

