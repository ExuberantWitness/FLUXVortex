# 回复审计方（第三轮）：A16 速度相关的实测数据与已执行改动

需求表述：A16（t\*=21，`max_aero_steps=2100`）在 600 s 内完成 ⇒ 需 **0.286 s/步**。

本记录只陈述测量与执行结果，数据文件与复现命令见 §6。

---

## 1. 当前实测单价与构成

| 测量 | 条件 | 结果 |
|---|---|---|
| `step_wall_seconds_mean` | A16，150 步，粒子 ~27k | **9.136 s/步**（`/tmp/a16_150.json`）|
| 同上外推 2100 步 | — | 19,186 s = 5.33 h |
| `step_wall_seconds_mean` | A16，20 步，粒子 ~5k | 4.35–4.64 s/步 |
| `step_wall_seconds_mean` | A16，3 步，尾迹近空 | 2.08 s/步 |
| 刚翼 V5M，15×30，27k 粒子 | 900 步 / 18 min | ~1.2 s/步 |
| 刚翼 α=16，50 步（含建模）| 合计 14 s | 步进部分 ~0.2 s/步 |

同一工况下 3 步 / 20 步 / 150 步的单价分别为 2.08 / 4.35 / 9.136 s/步，随粒子数增长。

**推算（非直接测量）**：膜翼路径与刚翼共用同一气动求解器（`Q16NativeV5MSolver`），若气动部分按刚翼实测 ~1.2 s/步，则结构+耦合部分约为 9.136 − 1.2 ≈ 7.9 s/步。此拆分未做直接消融验证。

---

## 2. 主机同步的定位（20 步，call-site 探针）

方法：monkeypatch `torch.Tensor.item` 记录 `sys._getframe(1)`，A16 跑 20 步。脚本 `/tmp/where_items.py`。

| 量 | 值 |
|---|---|
| 20 步墙钟 | 87.0 s |
| `.item()` 调用总数 | 67,422 |
| **每次迭代折算** | **3371 次/步** |
| `.item()` tottime（cProfile，20 步）| 55.6 s |
| 占 20 步总 cumtime（92.9 s）| 60% |
| 单次均摊 | 826 µs |

按调用点（前 8 名合计 84%）：

| 次数/步 | 占比 | 调用点 | 该行内容 |
|---|---|---|---|
| 1240 | 36.8% | `q16_flux_v5m_native.py:40 _require_cuda64` | `bool(torch.isfinite(v).all().item())` |
| 688 | 20.4% | `q16_flux_v5m_author_loads.py:235 line_derivative` | 同类有限性检查 |
| 292 | 8.7% | `q16_structural_solver.py:1446 _reference_dense_nonsymmetric_solve` | 线性残差门的标量读回 |
| 228 | 6.8% | `q16_flux_v5m_native_fsi.py:336 _integrate_structure` | Newton 判据读回 |
| 195 | 5.8% | `q16_structural_solver.py:936 _acceleration_load_matrix` | 有限性检查 |
| 177 | 5.3% | `q16_flux_v5m_native.py:493 assemble_native_ring_geometry` | 几何校验 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:207 material_ring_velocity_derivative_expanded` | 有限性检查 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:349 velocity_force` | 有限性检查 |

同为该脚本取得的其他计数：`warp.synchronize_device` 43,404 次（13.7 s，多数嵌套在 `.item()` 内）。

**与 2026-08-30 的 profile 对照**（`profile_output/roj_20260830/a16_3step_pstats.txt`，3 步）：

| 项 | 8 月 | 现在（20 步） |
|---|---|---|
| `.item()` 次数/步 | 1,524 | 3,371 |
| `.item()` 占墙钟 | 22% | 60% |
| 当时的前列项 | `digest()` D2H、`add_connected_ribbon_particles` 内的 5 处 | 以上表为准；`digest()` 未进前 12 |

---

## 3. 已执行的三项改动与其实测效果

| 改动 | 提交 | 效果 |
|---|---|---|
| Stage 4a：三处事务守卫由整状态 SHA-256 改为 `_version` 指纹 | `213c042` | parity 6/6；每次守卫的 D2H 由 8 次 + 2.25 MB 哈希降为 0 |
| Stage 4b：诊断块 14 次 `.item()` 合并为 2 次 | `6f4be69` | 3 步 A/B 实测 `.item()` 355.3 → 341.3 次/步（−14）；37 个诊断键值逐位不变 |
| S1：切线刷新内的有限性检查改为延迟到 scope 末尾（`begin/flush_deferred_finite_checks`）| 未提交 | 3 步 A/B：`step_wall_mean` 2.081 → 2.087 s（−0.3%，即无变化）；cn 逐位相同；scope 仅覆盖 `refresh_reference_tangent` |
| A1：`β`-无关的有效切线块在 free 空间缓存（`FLUXV_STRUCT_FAST_PATH`，默认关）| 未提交 | 3 步 cn 与生产路径 `max abs diff = 0.0`（逐位相同）；**速度未测**（150 步计时运行被中止）|

`reference_tangent_cache_refresh_count` = 20 / 150 步（`reference_tangent_refresh_rtol = 0.002`）。`structural_substeps = 10`，`proposal_count = 447 / 150 步`。

---

## 4. 目前可以测量的三个量（供你判断是否需要）

1. **前两名占 1928 次/步（57%）**，两处均为 `isfinite().all().item()`，不参与分支判断。把这两处改为"设备端 `atomic_max` 到单标志 + 步末读一次"后，需要重新测量的量：`.item()` 次数/步、`step_wall_mean`、抛出异常的位置（由调用时变为步末）。
2. **尾迹顶点诱导的目标数**：`wake_free_rows = 100`，而 `wake_rings` 容量为 `wake_max_rows = 300`。当前 `velocity_at_cuda` 的 target 集合是否只包含 free rows，需查 `_advance_material_state` 与 `native_material_convection.velocity()` 的实际 target 数（代码路径已定位，未做计数测量）。
3. **粒子目标数**：`velocity_self_cuda` 的 target = 全部存活粒子（27k），source 同为 27k ⇒ 单次 7.5×10⁸ 对。

---

## 5. 数值上界（只做算术，不做判断）

- 若 `.item()` 总次数据实测降至 0，可减少的墙钟上限 = 当前 `.item()` 占总墙钟的比例（20 步 profile 为 60%）。
- 9.136 s/步 × (1 − 0.60) ≈ 3.65 s/步 ⇒ 2100 步 ≈ 7,669 s ≈ 2.13 h。
- 0.286 s/步 × 2100 步 = 600 s。0.286 s/步对应的每步预算与上表各分项（同步 5.5 s、气动 ~1.2 s、结构线性代数与装配其余部分）的对照，需要先完成 §4 的第 1、2 项测量。

---

## 6. 数据与复现

```
/tmp/a16_150.json            A16 150 步，step_wall_seconds_mean = 9.1363
/tmp/a16_phase50.json         A16 50 步
/tmp/s1_base_3.json          3 步基线（S1 A/B）
/tmp/a1_3.json               A1 3 步（与基线 cn 逐位相同）
/tmp/where_items.py          call-site 探针脚本
/tmp/locate_20.py            20 步 cProfile 脚本

PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 --max-aero-steps 20
```
