# A15 — joint 全长发散的定位：新生 LEV 自由未知量的正反馈，不是结构

日期：2026-11-24　数据源：
`artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current/ROJ11_A16_FULL_20261124_LEPIN_joint.partial.json`
（782 条 record 完整保存；终止原因 `Failed to allocate 796262400 bytes`，即我并发起了第二个
GPU 任务造成的 OOM，**发散在此之前已开始**）

## 结论

发散不是结构失稳，不是粒子容量，不是 OOM 之前才出现的：

| 观测量 | 步 750 | 步 782 | 判读 |
|---|---|---|---|
| `instantaneous_zmax_over_c` | 0.03470 | 0.03359 | 全程平稳，膜振幅没炸 |
| `particle_count` | 27371 | 26810 | 几乎不变（−2%），`particle_cull_count` 恒为 271（稳态老化率） |
| `bound_circulation_total` | 40.47 | **−0.54** | 单调塌陷 −45% 后翻负 |
| `newborn_lev_circulation_max_abs` | 0.196 | **16.19** | 32 步涨 83× |
| `lesp_pre_max_abs` | 0.144 | 1.006 | 涨 7× |
| `coupling_residual` | 3.7e-14 | 2.6e-9 | 涨 10⁴ |
| `cn` | +0.66 | **−90.66** | 步 776 起正负翻转 |

逐窗口读出的事件顺序：

```
步 750–755  newton 反复打到 100/120 上限（结构求解已在挣扎），cn 正常 0.5–0.68
步 756–760  newborn 0.275 → 0.395 开始加速
步 764–771  newborn 0.589 → 1.370；lesp_pre 0.162 → 0.203
步 774      newborn 2.968；newton 120
步 776      cn 首次翻负 (−0.109)，coupling_residual 2.7e-11
步 777–782  newborn 3.63 → 16.19；cn −0.37 → 31.8 → −90.7；|G|_tot 已塌到 22 → 翻负
```

**增长律**：`newborn` 每步约 ×1.5，`lesp_pre` 与 `coupling_residual` 同步上升，
`bound_circulation_total` 同步下降。即

```
新生 LEV 强度 ↑ → 前缘诱导 ↑ → LESP ↑ → 分离判据更易满足 → 新生 LEV 强度 ↑
```

这是 joint 路径把新生 LEV 环量当作**耦合求解里的自由未知量**（`lambda`）所引入的正反馈。
生产路径没有这个未知量：两条收敛全长跑（`CNCONV` legacy、`LEPIN_signed`）的
`newborn_lev_circulation_max_abs` **2100 步恒为 0**。

## 与生产路径的另一处差异（量级提示，不是发散原因）

同一时刻（步 750）joint 跑的 `bound_circulation_total` = 40.47，而生产全长跑在
t\* 中点之后稳定在 70–72（步 525 时 69.07），差 1.75×；`lesp_pre_max_abs` 生产在
**0.42–0.44** 一带，joint 跑却从 0.144 起步。也就是说 joint 不只是在发散的晚期不同，
它在稳态段就已经处在与生产完全不同的解分支上。这一点此前（A12）只看到 cn 的差别，
现在有了环量与 LESP 两个独立的量。

## 对 LEV 几何修复（A14）的含义

* 修复方向必须是**几何**（涡片弧长/落位），不能通过把新生强度放进耦合求解来"补强"
  —— 那正是这条发散的机制。
* `routeA_100_open/notev` 在 joint 路径上 cn 冲到 5.83 也是同一类：改变前缘涡的强度
  与落位会立刻通过 LESP 反馈回到分离判据上。
* A14 的 `FLUXV_LE_SHEET_ARC` 只改涡片**落位**、不改强度，且默认 `off`
  （`pytest tests/test_roj_unified_case_runner_parity_gpu.py` 6/6 通过，生产逐位不变）。

## 仍未定位

`structural_newton_iterations` 从步 754 起就持续打满 100/128 上限，早于 newborn 的可见
加速。它可能是发散的前兆（结构残差没有降到容差，导致耦合不动点失去压缩性），也可能是
发散开始后载荷变乱的结果。要区分需要记录每步的 Newton 残差序列（当前 payload 只存
迭代次数），这需要改 runner 的观测字段，**本轮未做**。
