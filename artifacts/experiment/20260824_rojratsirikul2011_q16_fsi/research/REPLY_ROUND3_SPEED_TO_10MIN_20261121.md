# 回复审计方（第三轮）：A16 速度 —— 全部测量结果（含四项原待测量）

需求：A16（`t*`=21，`max_aero_steps`=2100）在 600 s 内完成 ⇒ **0.286 s/步**。

本记录只列测量值、执行效果与尚未测量的量。数据文件见 §9。

---

## 1. 各窗口的 `step_wall_seconds_mean`（并标注仪器）

| 窗口 | 粒子数 | 单价 | 仪器 |
|---|---|---|---|
| 3 步 | 尾迹近空 | **2.08 s/步** | 无 |
| 15 步 | ~1.6k | 3.945 s/步 | monkeypatch 计数（~µs/次）|
| 20 步 | ~5k | 4.35–4.64 s/步 | cProfile |
| **30 步** | **~8k** | **4.720 s/步** | **计时器插桩（本次主测量）** |
| 40 步 | ~11k | 10.52 s/步 | 采样线程（2 ms 轮询，GIL 争用）|
| 60 步 | ~16k | 6.156 s/步 | 采样线程（同上）|
| **150 步** | **~27k** | **9.136 s/步** | **无** |

计时器插桩（30 步）的 4.720 s/步与同区间无插桩的 4.35–4.64 s/步相差 2–8%；采样线程的 10.52 s/步（40 步）明显高于 20 步（4.35–4.64）与 150 步（9.136）之间的趋势线。

结构参数（`/tmp/a16_150.json`）：`structural_substeps=10`、`proposal_count=447/150 步`、`reference_tangent_cache_refresh_count=20/150 步`、`rtol=0.002`、`commit_count=formal_replay_count=150`。

---

## 2. `.item()`：计时器测量（无采样偏差）

30 步运行，wall 141.6 s：

| 量 | 值 |
|---|---|
| 调用次数 | 103,510 ⇒ **3450 次/步** |
| 累计时间 | **92.20 s = 65% 的 wall** |
| 单次均摊 | **891 µs** |

与另两次测量对照：20 步 cProfile 给 3371 次/步、tottime 60%；40 步采样给 `.item()` 直接帧 18.9% + `synchronize_device` 23.9%。三次的口径不同（cProfile 的 tottime 含其内部 sync；采样按最内层帧归类；计时器直接包住 `.item()` 调用），计时器法开销最低（~1 µs/次）。

调用点分布（20 步 cProfile 的 call-site 探针）：

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

与 2026-08-30 的 profile（3 步）对照：`.item()` 1,524 次/步、22%；当时列在前面的 `digest()` 与 `add_connected_ribbon_particles` 内 5 处，本次前 12 名中未出现。

---

## 3. 每步相位拆分（计时器，绝对秒；30 步，wall 141.6 s）

| 相位 | 绝对 | 占 wall | 次数/步 |
|---|---|---|---|
| `_integrate_structure`（FSI 结构积分）| 128.05 s | **90.4%** | 2.9 |
| `structural_step`（Newmark 子步）| 112.19 s | 79.2% | 58.0 |
| `_reference_dense_nonsymmetric_solve` | 76.75 s | 54.2% | 104.9 |
| `propose`（气动 + 粒子 + 载荷装配）| 13.15 s | **9.3%** | 2.9 |
| `refresh_reference_tangent` | 11.04 s | 7.8% | 1.0 |

嵌套关系：`_integrate_structure` ⊃ `structural_step` ⊃ `_reference_dense_nonsymmetric_solve`。
`propose` 与 `_integrate_structure` 不相交（合计 99.7%）。

---

## 4. `_reference_dense_nonsymmetric_solve` 内部（同一 30 步运行）

| 操作 | 次数/步 | 累计 | 占 wall | 单次 |
|---|---|---|---|---|
| `torch.linalg.solve_ex`（LU）| 104.9 | **1.47 s** | **1.0%** | 0.5 ms |
| ≥10⁶ 元素 `__getitem__`（`effective[free][:,free]`）| 364.0 | **0.11 s** | **0.08%** | 0.3 ms |
| 整个函数 | 104.9 | **76.75 s** | **54.2%** | **24.4 ms** |

⇒ 该函数 24.4 ms/次中，**LU 0.5 ms（2%）、大 gather ≈0.1 ms（0.4%）**，其余 **~23.8 ms/次（97.6%）** 落在该函数内除 LU 与 gather 之外的张量运算与同步上。该函数内的 `.item()` 调用（约 2.8 次/次调用 ⇒ 2.5 ms）只解释其中的 ~10%。

---

## 5. 粒子 Biot–Savart 的 target 集合（15 步，按调用者标注）

| 次数/步 | 调用者 | targets |
|---|---|---|
| 5.5 | `propose` | 450（配点）|
| 2.8 | `propose` | 1800（涡丝中点，4×450）|
| 多次 | `q16_flux_v5m_native.py:995 wake_velocity` | **`n_rows × 30 × 4`**（尾迹顶点；1 行=120，2 行=240 …）|
| 1 | `velocity_self_cuda` | **n**（targets = sources）|

稳态（n=27,371、`wake_max_rows=300`）下每步对数：

| target 集合 | targets/步 | 对/步 | 占比 |
|---|---|---|---|
| 尾迹顶点 | 36,000 | 9.85×10⁸ | 51% |
| 粒子自诱导 | 27,371 | 7.49×10⁸ | 39% |
| 配点 | 2,475 | 6.78×10⁷ | 3.5% |
| 涡丝中点 | 5,040 | 1.38×10⁸ | 7% |
| 合计 | ~70,900 | **1.94×10⁹** | — |

`wake_free_rows=100` 与 `wake_max_rows=300` 见 `NativeV5MConfig`。`wake_velocity`（`q16_flux_v5m_native.py:995-1005`）的 target 参数是 `trial.wake_rings` 的整体，其 RK3 三阶各调用一次。

---

## 6. 已执行的四项改动与实测效果

| 改动 | 提交 | 实测效果 |
|---|---|---|
| Stage 4a：三处事务守卫由整状态 SHA-256 改为 `_version` 指纹 | `213c042` | parity 6/6；每次守卫的 8 次 D2H + 2.25 MB 哈希降为 0 |
| Stage 4b：诊断块 14 次 `.item()` 合并为 2 次 | `6f4be69` | 3 步 A/B：`.item()` 355.3 → 341.3 次/步；37 个诊断键值逐位不变 |
| S1：`refresh_reference_tangent` 内有限性检查延迟到 scope 末尾 | 未提交 | 3 步 A/B：`step_wall_mean` 2.081 → 2.087 s；cn 逐位相同。scope 仅覆盖该函数的 1.0/步 调用 |
| A1：`β`-无关的有效切线块在 free 空间缓存（`FLUXV_STRUCT_FAST_PATH`，默认关）| 未提交 | 3 步：cn `max abs diff = 0.0`；**速度未测**（150 步计时运行被中止）|

---

## 7. 由已有数字直接算出的关系（含各自的假定）

1. `.item()` 归零的墙钟下降上界 = 其占比（30 步计时器法 65%）：9.136 × 0.35 = **3.20 s/步** ⇒ 2100 步 ≈ 6,720 s ≈ 1.87 h。**假定**：150 步稳态下该占比不低于 30 步窗口的 65%（未测）。
2. 600 s / 2100 步 = **0.286 s/步**。
3. 若 `_reference_dense_nonsymmetric_solve` 的 24.4 ms/次降到与 LU 同量级（0.5 ms）⇒ 该相位由 54.2% 降至 ~1%，wall 下降 ~53%：9.136 × 0.47 = **4.29 s/步** ⇒ 2100 步 ≈ 2.5 h。**假定**：其余 23.8 ms 可被消除（来源尚未定位）。
4. 若尾迹顶点的粒子求值按 `wake_free_rows=100` 限制：targets 36,000 → 12,000，对数 9.85×10⁸ → 3.28×10⁸，总对数 −33%。**假定**：冻结行（`n_rows > 100`）的顶点诱导不被下游消费——`propose` 9.3% 的占比意味着该项即使全免也只影响 ~9.3% 的 wall。
5. 上述 1+3 同时成立（互有重叠，不能相加）：设 `.item()` 部分计入 dense_solve 内的比例与整体相同（65%），则两项目标重叠 ≈ 0.542 × 0.65 = 0.35，去重后 wall 下降上界 ≈ 0.65 + 0.54 − 0.35 = 0.84 ⇒ 9.136 × 0.16 = 1.46 s/步 ⇒ 2100 步 ≈ 51 min。**该式为上界，且依赖两项假定。**

---

## 8. 尚未测量

| 未测量 | 边界 |
|---|---|
| 150 步稳态的 `.item()` 占比（计时器法）| 一次 150 步 × 计时器插桩，约 23 min |
| `_reference_dense_nonsymmetric_solve` 内 ~23.8 ms/次的来源分解（除 LU/gather/item 之外）| 需在该函数体内分段插桩（改代码）|
| `propose` 内部细分（气动求解 / 粒子沉积 / 载荷装配 / retention）| 需在 propose 内分段插桩（改代码）|
| 冻结行顶点诱导是否被下游消费 | 需读 `wake_velocity` 三个 RK3 调用的结果消费链（`trial.wake_rings.copy_` 仅用于 free rows 时才算免）|

---

## 9. 数据与复现

```
/tmp/a16_150.json     A16 150 步（无插桩）9.1363 s/步
/tmp/measure4.py      计时器插桩（.item / 相位 / solve_ex / gather / wake targets）—— 30 步结果见 §2–§4
/tmp/where_items.py   .item() call-site 探针
/tmp/count_targets.py 粒子 BS target 集合计数
/tmp/sample_prof.py   采样 profiler（有 GIL 争用，只用于份额）
/tmp/locate_20.py     20 步 cProfile

PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 /tmp/measure4.py
```

---

## 10. 需要审计方回答的三点

1. 两处 `isfinite().all().item()`（合计 1928 次/步，57%）改为"设备端 `atomic_max` + 步末读一次"后，**异常抛出位置由"该次调用时"变为"该步末"**。这是否改变你对该路径的验收含义？
2. `wake_free_rows=100` 之外的环只随来流平移。**按你的理解，这些环顶点的粒子诱导速度是否仍被下游消费？**（`propose` 仅占 9.3%，此项对总 wall 的影响上界为 9.3%。）
3. `_reference_dense_nonsymmetric_solve` 每次 24.4 ms，其中 LU 0.5 ms、gather ≈0.1 ms、函数内 `.item()` ≈2.5 ms。**你认为剩余 ~23.8 ms 应落在哪些具体操作上？** 我们的分段插桩尚未覆盖该函数体内部。
