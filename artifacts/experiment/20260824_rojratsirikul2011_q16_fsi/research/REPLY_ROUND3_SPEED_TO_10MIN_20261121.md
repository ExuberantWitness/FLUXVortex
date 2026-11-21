# 回复审计方（第三轮）：A16 速度 —— 四项待测量的结果、已执行改动的效果

需求：A16（`t*`=21，`max_aero_steps`=2100）在 600 s 内完成。算术对应 **0.286 s/步**。

本记录只列测量值、执行效果与尚未测量的量。数据文件与复现命令见 §8。

---

## 1. 各窗口长度下的 `step_wall_seconds_mean`

| 窗口 | 粒子数 | 单价 | 仪器 |
|---|---|---|---|
| 3 步 | 尾迹近空 | **2.08 s/步** | 无 |
| 15 步 | ~1.6k | 3.945 s/步 | 轻量 monkeypatch（只计数）|
| 20 步 | ~5k | 4.35–4.64 s/步 | cProfile 开启 |
| 40 步 | ~11k | 10.52 s/步 | 采样线程（2 ms 轮询）|
| **150 步** | **~27k** | **9.136 s/步** | **无** |

只有 **3 步**与 **150 步**两行是无插桩测量。40 步一行叠加了采样线程的 GIL 争用，其 10.52 s/步高于 20 步（4.35）与 150 步（9.136）之间的趋势线，不能与无插桩值直接比较。

其他同次取得的结构参数（`/tmp/a16_150.json`）：`structural_substeps=10`、`proposal_count=447/150 步`、`reference_tangent_cache_refresh_count=20/150 步`、`reference_tangent_refresh_rtol=0.002`、`commit_count=formal_replay_count=150`。

---

## 2. 粒子 Biot–Savart 的 target 集合（15 步，按调用者标注）

方法：monkeypatch `velocity_at_cuda` / `velocity_self_cuda` / `velocity_and_jacobian_self_cuda`，记录调用者与 target 数。

| 次数/步 | 调用者 | targets |
|---|---|---|
| 5.5 | `q16_flux_v5m_native.py:propose` | 450（配点）|
| 2.8 | `q16_flux_v5m_native.py:propose` | 1800（涡丝中点，4×450）|
| 1 | `q16_flux_v5m_native.py:wake_velocity` | **120 / 480 / 600 / 720 …** = `n_rows × 30 × 4`（尾迹顶点）|
| 1 | `velocity_self_cuda` | **n**（粒子数；targets = sources）|

`wake_free_rows=100`、`wake_max_rows=300`。上表第 3 行的 target 数随尾迹行数增长，与 `n_rows × ns × 4`（ns=30）一致：120=1 行、480=4 行、600=5 行、720=6 行；行数上限 300 时对应 36,000 个 target。

稳态（n=27,371）下每步的对数：

| target 集合 | targets/步 | 对/步 | 占比 |
|---|---|---|---|
| 尾迹顶点 | 36,000 | 9.85×10⁸ | 51% |
| 粒子自诱导 | 27,371 | 7.49×10⁸ | 39% |
| 配点 | 2,475 | 6.78×10⁷ | 3.5% |
| 涡丝中点 | 5,040 | 1.38×10⁸ | 7% |
| 合计 | ~70,900 | **1.94×10⁹** | — |

---

## 3. `.item()` 的调用点分布（20 步，cProfile + call-site 探针）

| 量 | 值 |
|---|---|
| `.item()` 调用 | 67,422 ⇒ **3371 次/步** |
| `.item()` tottime | 55.6 s / 92.9 s cumtime = **60%** |
| 单次均摊 | 826 µs |
| `warp.synchronize_device` | 43,404 次 / 13.7 s（多数嵌套在 `.item()` 内）|

| 次数/步 | 占比 | 调用点 | 该行内容 |
|---|---|---|---|
| 1240 | 36.8% | `q16_flux_v5m_native.py:40 _require_cuda64` | `bool(torch.isfinite(v).all().item())` |
| 688 | 20.4% | `q16_flux_v5m_author_loads.py:235 line_derivative` | 同类有限性检查 |
| 292 | 8.7% | `q16_structural_solver.py:1446` | 线性残差门读回 |
| 228 | 6.8% | `q16_flux_v5m_native_fsi.py:336` | Newton 判据读回 |
| 195 | 5.8% | `q16_structural_solver.py:936` | 有限性检查 |
| 177 | 5.3% | `q16_flux_v5m_native.py:493` | 几何校验 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:207` | 有限性检查 |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:349` | 有限性检查 |

与 2026-08-30 的 profile（3 步窗口）对照：`.item()` 1,524 次/步、22%；当时列在前面的 `digest()` 与 `add_connected_ribbon_particles` 内 5 处，本次前 12 名中未出现。

---

## 4. 分相墙钟份额（40 步采样，203,003 次样本）

最内层帧：

| 份额 | 帧 |
|---|---|
| **35.3%** | `q16_structural_solver.py:_reference_dense_nonsymmetric_solve` |
| **23.9%** | `warp/_src/context.py:synchronize_device` |
| **10.7%** | `q16_flux_v5m_native.py:_require_cuda64` |
| **8.2%** | `q16_flux_v5m_author_loads.py:line_derivative` |
| 3.3% | `kernels_ring_velocity.py:native_ring_velocity_expanded` |
| 2.3% | `q16_structural_solver.py:_acceleration_load_matrix` |
| 2.1% | `q16_flux_v5m_author_loads.py:velocity_force` |
| 2.1% | `q16_flux_v5m_native_fsi.py:_integrate_structure` |

外层生产帧：

| 份额 | 帧 |
|---|---|
| 36.9% | `q16_structural_solver.py:step` |
| 12.5% | `q16_flux_v5m_native.py:evaluate` |
| 8.2% | `material_ring_velocity_derivative_expanded` |
| 7.4% | `kernels_q16_ans_eas.py:_assert_finite2` |
| 5.1% | `kernels_q16_ancf.py:_assert_finite` |
| 3.9% | `kernels_q16_mesh.py:_require_global_state` |
| 3.2% | `q16_flux_v5m_native.py:_ring_velocity` |
| 2.6% | `kernels_q16_ans_eas.py:_strain` |
| 2.2% | `kernels_q16_ans_eas.py:_assert_finite3` |
| 1.9% | `q16_structural_solver.py:_norms` |

有限性检查四帧合计 18.6%；两处 `.item()` 主因合计 18.9%。
**该 40 步运行的墙钟为 10.52 s/步（含采样开销），与 §1 的无插桩 9.136 s/步不可直接换算**，上表只用于份额。

---

## 5. 已执行的四项改动与实测效果

| 改动 | 提交 | 实测效果 |
|---|---|---|
| Stage 4a：三处事务守卫由整状态 SHA-256 改为 `_version` 指纹 | `213c042` | parity 6/6；每次守卫的 8 次 D2H + 2.25 MB 哈希降为 0 |
| Stage 4b：诊断块 14 次 `.item()` 合并为 2 次 | `6f4be69` | 3 步 A/B：`.item()` 355.3 → 341.3 次/步；37 个诊断键值逐位不变 |
| S1：`refresh_reference_tangent` 内有限性检查延迟到 scope 末尾 | 未提交 | 3 步 A/B：`step_wall_mean` 2.081 → 2.087 s；cn 逐位相同。scope 仅覆盖 20/150 次刷新 |
| A1：`β`-无关的有效切线块在 free 空间缓存（`FLUXV_STRUCT_FAST_PATH`，默认关）| 未提交 | 3 步：cn `max abs diff = 0.0`；**速度未测**（150 步计时运行被中止）|

---

## 6. 由已有数字直接算出的关系

- 若 `.item()` 调用数降为 0：墙钟下降上界 = 其占比。按 20 步窗口 60%：9.136 × 0.40 = **3.65 s/步** ⇒ 2100 步 ≈ 7,669 s ≈ 2.13 h。该 60% 是粒子 ~5k 窗口的值；**稳态窗口的占比未测**，故该式是上界。
- 600 s / 2100 步 = 0.286 s/步。
- 若尾迹顶点的粒子求值按 `wake_free_rows=100` 限制：targets 由 36,000 降为 12,000，该集合的对数由 9.85×10⁸ 降为 3.28×10⁸ ⇒ 总对数由 1.94×10⁹ 降为 1.29×10⁹（−33%）。此计算假定"冻结行不需粒子诱导"，该假定尚未验证。

---

## 7. 尚未测量的量

| 未测量 | 边界 |
|---|---|
| 无插桩的稳态（150 步）`.item()` 份额 | cProfile 需约 23 min；采样法自带 ~50–70% 开销 |
| `_reference_dense_nonsymmetric_solve` 内部各 70.9 MB 张量操作的调用次数与各自耗时 | 需在函数内插桩（改代码）|
| 每步各相位（气动 / 结构子步 / 装配 / retention）的无插桩墙钟拆分 | Roj harness 不输出逐步 `__DS_PROGRESS__`（Yamano harness 有）|
| 冻结行（`n_rows > wake_free_rows`）的顶点诱导是否被下游消费 | 需读 `_advance_material_state` 与 `native_material_convection.velocity()` 的 target 构造（代码已定位，未逐行核对）|

---

## 8. 数据与复现

```
/tmp/a16_150.json       A16 150 步（无插桩），step_wall_seconds_mean = 9.1363
/tmp/s1_base_3.json     3 步基线（无插桩），2.081
/tmp/a1_3.json          A1 3 步，cn 与基线逐位相同
/tmp/where_items.py     .item() call-site 探针
/tmp/count_targets.py   粒子 BS target 集合计数
/tmp/sample_prof.py     采样 profiler
/tmp/locate_20.py       20 步 cProfile

PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_fsi_native.py  # 见上行命令
PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
  --case ROJ11-A16 --max-aero-steps 150
```

---

## 9. 需要审计方回答的三点

1. `_require_cuda64`（1240 次/步）与 `line_derivative`（688 次/步）的 `isfinite().all().item()`，若改为"设备端 `atomic_max` 到单标志 + 步末读一次"，**异常抛出位置会从"该次调用时"变为"该步末"**。这是否改变你对该路径的验收含义？
2. `wake_free_rows=100` 之外的环只随来流平移（代码注释如此）。**按你的理解，这些环顶点的粒子诱导速度是否仍被下游消费？** 如果是，§6 第三行的 −33% 不成立。
3. 若把验收窗口由 `t*`=21 改短，需要哪些附带条件（收敛证书、与 `t*`=21 的比值范围）才算有效验收？
