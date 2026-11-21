# S1：M1 的"105 → 10"是推断且错了 —— 实测 104.87 → 29.00 → 10.33 次/步（2026-11-22）

> 提交：`7f52ead`（`q16_structural_solver.py`、`q16_flux_v5m_native_fsi.py`、`case_runner.py`、新 `profile_a16_phases.py`）
> 验收：30 步 `cn` 逐位、80 步 digest 链 on/off 全等、80 步跨版本 cache=0 全等、parity 6/6

## 1. 我之前写错了什么

`327d65f`（M1）的提交信息写"105 factorizations per outer step → 10 (one per load fraction beta)"，`q16_flux_v5m_native_fsi.py` 里也写着"turns ~58 factorizations per outer step into ~10"。**这两个数字都是推断，没有计数器支撑，而且错了。** 本轮先加计数器（此前仓库里没有任何分解计数，`direct_solve_count` 只数 solve 调用），实测得到 **29.00 次/步**。

## 2. 两个各自独立的单槽行为

用同一 30 步窗口、同一棵树、只改一处，实测序列：

| 配置 | 分解/步 | 命中/步 | 30 步 `step_wall` |
|---|---|---|---|
| `FLUXV_LU_CACHE=0`（`solve_ex` 原路径） | **104.87** | — | **4.675 s** |
| `=1`，单槽（干预前） | **29.00** | 75.87 | **2.966 s** |
| `=1`，按 β 的桶，但 `am_slope` 未修 | **29.00** | 75.87 | 2.863 s |
| `=1`，按 β 的桶 + `am_slope` memo | **10.33** | 94.53 | **2.406 s** |

**Bug 1：缓存是单槽。** `_EFFECTIVE_LU_CACHE` 只用 `id(self)` 一个键，一条记录。β 遍历是 `β1,β1,β2,β2,…,β10,β10`（每个 β 下先 predictor 再 corrector，内部还有 Newton 迭代），下一趟回到 β1 时槽里是 β10 的键 ⇒ **每趟 10 次重分解**。`29.00 = 2.9 趟 × 10`，而 `proposal/步 = 87/30 = 2.9`。

**Bug 2：`am_slope` 每次调用重建。** 只修 Bug 1 之后仍然是 29.00 —— 这条是被测量抓出来的，光看代码看不出来。`am_slope = anchor_am − previous_am` 是 `_integrate_structure` 每次调用新建的张量，而 M1b 的载荷作用 memo 以 `id(slope)` 为键 ⇒ 新 slope 对象 ⇒ 每趟重建 10 个 action ⇒ 10 个新 `data_ptr` ⇒ LU 缓存（以矩阵为键）**无论有多少槽都全部 miss**。修法是把 slope 本身 memo 住（`_am_slope_memo`，键 `(id(anchor), id(previous))`，持有两个操作数防 id 回收）。

**一处关键细节：`inverse_beta_dt2` 必须在内键。** 它是子步长的函数，而子步长是 β 的函数，每趟取 10 个不同值。我第一版把它放进外键，结果每个 β 分到一个只有一条记录的桶，**完全复现单槽**（实测仍是 29.00）。这条也是测出来的，不是想出来的。

## 3. 查表上移省掉的东西

`_acceleration_load_matrix`（`q16_structural_solver.py:964-967`）对 2976² 矩阵做 `isfinite().all().item()`，而且它在缓存键**之前**被调用 ⇒ 即使命中也要付这次 71 MB 扫描。现在键直接从 `acceleration_load_action.generalized_matrix` 取，命中路径跳过：

- `effective` 的 2976² 三次张量运算（`:1513-1520` 原位置）与 `[free][:,free]` gather；
- `_acceleration_load_matrix` 的形状/类型校验与 71 MB `isfinite` 全矩阵扫描。

安全性：命中意味着 `entry[3] is raw_added_mass`（`is` 判等），即同一个已校验过的张量对象；缓存持有该对象，所以被释放再复用的 `data_ptr` 不可能撞上活条目。线性残差门（`:1572`）仍在每次调用上跑，用的是缓存的 `effective_free`。

## 4. 稳态计数（150 步，插桩运行内的计数器）

- 回代 **159.89 次/步**，分解 **10.07 次/步**，命中 149.83 次/步。
- 即稳态下 **10.07 次分解服务 159.89 次求解**，桶在 150 步内不失效。

## 5. 验收（全部逐位，不是容差）

| 检查 | 结果 |
|---|---|
| 30 步 `cn` 序列 vs 改前基线 | `torch.equal` 通过 |
| 80 步 committed-digest 链 `cache=1` vs `cache=0`（同树） | 80/80 全等 |
| 80 步 `cache=0` @HEAD vs `cache=0` @改动前树 | 80/80 全等 |
| parity `test_roj_unified_case_runner_parity_gpu.py` | **6/6** |

默认值已翻为 `FLUXV_LU_CACHE=1`，`=0` 保留为逃生开关（回到 `solve_ex` 原路径）。

## 6. 顺带发现的既有缺陷（与本次改动无关）

A16 在 **8 ≤ steps < 100** 时会在 `_finalize` 崩：`statistics_min_time_star = 20` ⇒ 统计窗口起点是第 2000 步，而 `window_selection` 给的窗口是 `[100, T]`，T=80 时为空，`platform/.../rojratsirikul2011_q16.py:490` 抛 `z_history must be (T, nc, ns) with T >= 8`。所以 80 步验收只能用 `.partial.json` 读 digest 链（本轮就是这么做的），或改跑 ≥100 步；< 8 步则走 `window_selection=null` 分支不会崩。

## 7. 仍未测

- `propose` 内部 ring 求和 vs 粒子 BS 的耗时拆分（第一版剖析脚本因标签重复计数与闭合公式错误未通过自门，已修，正在重跑）。
- 结构侧非 LU 部分的内部拆分。
