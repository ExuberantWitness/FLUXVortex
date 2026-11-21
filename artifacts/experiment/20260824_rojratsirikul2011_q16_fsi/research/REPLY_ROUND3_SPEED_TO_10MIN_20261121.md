# 回复审计方（第三轮）：A16 速度 —— 实测数据、已执行改动的效果、尚未测量的量

需求：A16（`t*`=21，`max_aero_steps`=2100）在 600 s 内完成。算术上对应 **0.286 s/步**。

本记录只列测量值、执行效果与尚未测量的量；数据文件与复现命令见 §7。

---

## 1. 单价的实测（同一入口，不同窗口长度）

入口：`platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py --case ROJ11-A16`

| 窗口 | 粒子数 | `step_wall_seconds_mean` | 数据 |
|---|---|---|---|
| 3 步 | 尾迹近空 | 2.08 s/步 | `/tmp/s1_base_3.json` |
| 20 步 | ~5k | 4.35–4.64 s/步 | `/tmp/locate_20.py` 两次运行 |
| **150 步** | **~27k** | **9.136 s/步** | `/tmp/a16_150.json` |

同一入口在 3 / 20 / 150 步上的单价分别为 2.08 / 4.35 / 9.136 s/步。**用 3 步测速会得到偏低的单价**，因为该窗口内粒子场尚未建立。

其他同次取得的结构参数（`/tmp/a16_150.json`）：

| 字段 | 值 |
|---|---|
| `structural_substeps` | 10 |
| `proposal_count` / 150 步 | 447（≈3/步）|
| `reference_tangent_cache_refresh_count` / 150 步 | 20 |
| `reference_tangent_refresh_rtol` | 0.002 |
| `commit_count` / `formal_replay_count` | 150 / 150 |

刚翼路径（同一气动求解器、15×30 网格、27k 粒子）实测 `~1.2 s/步`（900 步 / 18 min）。膜翼与刚翼共用 `Q16NativeV5MSolver`；**该 1.2 s/步不是对膜翼的直接测量，是同一求解器在刚翼路径上的测量值。**

---

## 2. `.item()` 的调用点分布（20 步，call-site 探针）

方法：monkeypatch `torch.Tensor.item`，在 `sys._getframe(1)` 处记录 `(文件名, 行号, 函数名)`。脚本 `/tmp/where_items.py`。粒子数 ~5k。

| 量 | 值 |
|---|---|
| 20 步墙钟 | 87.0 s |
| `.item()` 调用总数 | 67,422 |
| 每次迭代折算 | **3371 次/步** |
| `.item()` tottime（cProfile，20 步）| 55.6 s |
| cProfile 总 cumtime（20 步）| 92.9 s |
| tottime 占比 | 60% |
| 单次均摊 | 826 µs |
| `warp.synchronize_device` | 43,404 次 / 13.7 s（多数嵌套在 `.item()` 内）|

| 次数/步 | 占比 | 调用点 | 该行的检查内容 |
|---|---|---|---|
| 1240 | 36.8% | `q16_flux_v5m_native.py:40 _require_cuda64` | `bool(torch.isfinite(v).all().item())` |
| 688 | 20.4% | `q16_flux_v5m_author_loads.py:235 line_derivative` | 同类有限性检查 |
| 292 | 8.7% | `q16_structural_solver.py:1446 _reference_dense_nonsymmetric_solve` | 线性残差门读回 |
| 228 | 6.8% | `q16_flux_v5m_native_fsi.py:336 _integrate_structure` | Newton 判据读回 |
| 195 | 5.8% | `q16_structural_solver.py:936 _acceleration_load_matrix` | 有限性检查 |
| 177 | 5.3% | `q16_flux_v5m_native.py:493 assemble_native_ring_geometry` | 几何校验 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:207 material_ring_velocity_derivative_expanded` | 有限性检查 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:349 velocity_force` | 有限性检查 |

前两名合计 1928 次/步（57%），两处均为 `isfinite().all().item()`，其返回值不参与分支判断（只用于抛异常）。

与 2026-08-30 的 profile 对照（`profile_output/roj_20260830/a16_3step_pstats.txt`，3 步窗口）：

| 项 | 8 月（3 步） | 现在（20 步） |
|---|---|---|
| `.item()` 次数/步 | 1,524 | 3,371 |
| `.item()` 占墙钟 | 22% | 60% |
| 该 profile 中列在前面的项 | `digest()` 的 D2H、`add_connected_ribbon_particles` 内 5 处 | 本次前 12 名中未出现这两项 |

---

## 3. 已执行的四项改动及实测效果

| 改动 | 提交 | 实测效果 |
|---|---|---|
| Stage 4a：三处事务守卫由整状态 SHA-256 改为 `_version` 指纹 | `213c042` | parity 6/6；每次守卫的 8 次 D2H + 2.25 MB 哈希降为 0 |
| Stage 4b：诊断块 14 次 `.item()` 合并为 2 次 | `6f4be69` | 3 步 A/B：`.item()` 355.3 → 341.3 次/步；37 个诊断键值逐位不变；parity 6/6 |
| S1：`refresh_reference_tangent` 内的有限性检查延迟到 scope 末尾 | 未提交 | 3 步 A/B：`step_wall_mean` 2.081 → 2.087 s（变化 0.3%）；cn 逐位相同。scope 仅覆盖切线刷新（20/150 次）|
| A1：`β`-无关的有效切线块在 free 空间缓存（`FLUXV_STRUCT_FAST_PATH`，默认关）| 未提交 | 3 步：cn 与生产路径 `max abs diff = 0.0`；**速度未测**（150 步计时运行被中止）|

---

## 4. 由已有数字可直接算出的量

- 若 `.item()` 调用数降为 0，墙钟下降的上界 = 当前 `.item()` 占比。按 20 步窗口的 60%：9.136 × 0.40 = **3.65 s/步** ⇒ 2100 步 ≈ 7,669 s ≈ **2.13 h**。
- 该比例（60%）是在**粒子 ~5k 的 20 步窗口**测的；150 步稳态窗口的占比**未测**。稳态下粒子工作量更大，同步占比可能低于 60%，因此上式是上界。
- 600 s / 2100 步 = 0.286 s/步。

---

## 5. 尚未测量的量（这些量决定后续能测什么）

| 未测量 | 为什么需要 | 测量代价 |
|---|---|---|
| **150 步稳态下 `.item()` 的占比** | §4 的上界用的是 5k 粒子窗口；稳态占比未知 | 需在 100 步之后做一次 cProfile（约 5–8 min） |
| **尾迹顶点的 target 数** | `wake_free_rows=100`、`wake_max_rows=300`；`velocity_at_cuda` 的 target 是 free rows 还是全部 rows 未计数 | 只读计数，<1 min |
| **粒子 self-target 数** | `velocity_self_cuda`：target = 存活粒子数，source 同；27k 时为 7.5×10⁸ 对/次，每次步进调用几次未计数 | 只读计数，<1 min |
| **每步各相位的墙钟**（气动 / 结构子步 / 载荷装配 / retention） | 现有 payload 只有 `step_wall_seconds_mean` 一个总账 | 需在 FSI stepper 内加计时器（改代码）或读 `.partial.json` 的逐步记录（`YAMANO...` 有，Roj 无） |

---

## 6. 需要审计方回答的三点（问题，不是方案）

1. `_require_cuda64`（1240 次/步）与 `line_derivative`（688 次/步）的 `isfinite().all().item()`，若改为"设备端 `atomic_max` 到单标志 + 步末读一次"，**异常抛出的位置会从"该次调用时"变为"该步末"**。这个变化是否改变你对该路径的验收含义？
2. `wake_free_rows=100` 之外的环"只随来流平移、不参与形变"（代码注释如此）。**按你的理解，这些环的粒子诱导速度是否仍需逐顶点求值？** 若不需要，`velocity_at_cuda` 的 target 数与当前实际调用是否一致，是一个可测的量。
3. 若把验收窗口由 `t*`=21 改为更短的窗口，需要哪些附带条件（收敛证书、与 `t*`=21 的比值范围）才算有效的验收？这是一个定义问题，需要你给出判据。

---

## 7. 数据与复现

```
/tmp/a16_150.json      A16 150 步，step_wall_seconds_mean = 9.1363
/tmp/s1_base_3.json    3 步基线（S1 A/B 的对照）
/tmp/a1_3.json         A1 3 步（cn 与基线逐位相同）
/tmp/where_items.py    call-site 探针（记录每个 .item() 的调用点）
/tmp/locate_20.py      20 步 cProfile

PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 --max-aero-steps 20
```
