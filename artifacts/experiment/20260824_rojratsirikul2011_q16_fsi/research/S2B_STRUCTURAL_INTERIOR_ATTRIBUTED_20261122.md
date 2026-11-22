# S2b：结构侧内部拆分 —— `linearization` 962.7 ms/步 是最大叶子，`tangent_refresh` 每次 10.5 s 是真实计算而非积压（2026-11-22）

> 工具：`platform/warp_vpm/profile_a16_phases.py`（新增入口同步 split、内部算子标签）
> 原始：`/tmp/s2b_phase150.json`、`/tmp/s2b_phase150.log`
> 前置：`S2_STEADY_PHASE_TABLE_150STEPS_20261122.md`

## 1. 自门

| 检查 | 结果 |
|---|---|
| 闭合：三最外层相位 vs 墙钟 | **831.2 / 835.9 s = 99.4%** |
| 无插桩同窗口对照 | 5.573（插桩）vs 5.683（无插桩）s/步 = **−1.9%**，在运行间波动内 |
| **backlog（入口同步法）** | `propose` 0.0、`integrate_structure` 0.0、`tangent_refresh` 0.0 ms/步 |

backlog 那一行是本轮最重要的方法学结果：在每个最外层相位入口插一次 `torch.cuda.synchronize()` 并把 `start→mid` 记为"此前排队的工作"，测得 **三个相位入口处队列都已排空**。因此前一轮相位表里的时间不是被同步点吸收的积压，是各相位自己的 GPU 工作。

## 2. 150 步稳态相位表（扩展版）

| 相位 | calls/步 | ms/步 | exclusive ms/步 |
|---|---|---|---|
| `integrate_structure` | 2.98 | 2369.4 | 28.8 |
| ├ `struct_step` | 59.60 | **1849.6** | (见 §3) |
| ├ `velocity_force` | 179.82 | 491.0 | 491.0 |
| `propose`（气动） | 2.98 | 1801.3 | 0.0 |
| ├ `advance_wrk3` | 2.99 | 1218.8 | (重叠) |
| ├ `particle_BS` | 26.95 | 937.5 | (重叠) |
| ├ `ring_BS` | 47.95 | 777.4 | (重叠) |
| └ `surface_evaluate` | 184.83 | 145.6 | 145.6 |
| `tangent_refresh` | 0.13 | **1370.6** | 1370.6 |
| 合计（闭合） | | **5541.3** | ≈100% |

内部算子：

| 标签 | calls/步 | ms/步 | 每次调用 |
|---|---|---|---|
| `linearization`（`_current_fields`） | **379.51** | **963.0** | **2.54 ms** |
| `tangent_action` | 5.95 | 903.5 | 151.8 ms |
| `residual` | 219.49 | 304.0 | 1.38 ms |
| `dense_solve` | 159.89 | 365.7 | 2.29 ms |
| `accel_matrix_validate`（71 MB `isfinite`） | 169.96 | 57.0 | 0.34 ms |
| `norms`（内含 `wp.synchronize_device`） | 279.09 | 57.4 | 0.21 ms |
| `dot` | 279.09 | 42.2 | 0.15 ms |
| `predict_kinematics` | 60.60 | 20.8 | 0.34 ms |
| `kinematics` | 219.49 | 10.9 | 0.05 ms |
| `project_free` | 339.69 | 5.3 | 0.02 ms |

## 3. `struct_step` 的 1849.6 ms/步 拆完

`struct_step` 1849.6 = `dense_solve` 365.7 + `residual` 304.0 + **`linearization` 962.7** + `norms` 57.4 + `predict_kinematics` 20.8 + 其他 ≈139。（`linearization` 总计 963.0，其中落在 `tangent_refresh` 体内的只有 0.13 次/步 ≈ 0.3 ms。）

⇒ **单一最大叶子是 `linearization` 962.7 ms/步 = 整步的 17.3%**，379.51 次/步、2.54 ms/次。分配关系：每牛顿迭代在 `:1886` 对 `current` 算 1 次（219.49 次/步，与 `residual` 次数相等），另有约 160 次是 `:2196` 对**试探状态** `trial` 算的。

`:2196` 那次的性质需要说清：它被包在 `try/except` 里，用途是**几何有效性闸门**（`Q16 current geometry is orientation reversing or singular`）；在 `globalize_newton == False` 的常规路径上，`accepted_trial = trial; break` 在用到它的数值之前就跳出了（`:2205-2207`），**即算了 2.54 ms 只用它的异常、不用它的值**。

## 4. `tangent_refresh` 的 1370.6 ms/步 拆完

- 0.13 次/步（150 步内约 19.5 次）⇒ **每次约 10.5 s**，且 backlog=0，是自身算力。
- 体内是 47 批列探测（dof 2976 / 批宽 64）：`tangent_action` 5.95 次/步 × 151.8 ms = 903.5 ms/步，即每次刷新 47 × 151.8 ms ≈ 7.1 s。
- 余下 **≈467 ms/步（推导值，= 1370.6 − 903.5 − 0.3）** 是批循环的非 kernel 开销，每次刷新约 3.6 s、每批约 78 ms。该桶对应 `:713-727` 每批重建 `repeated_linearization`：`wp.from_torch(wp.to_torch(value).expand((count,)+shape).contiguous(), ...)`。
- **该值可被位级消除**：`repeated_linearization` 只依赖 `count` 与 `linearization`，而 47 批里只有 **2 个不同的 `count`**（前 46 批 64、最后一批 32），现在却被重建 47 次。改成按 `count` 缓存后语义逐位相同。

## 5. 本轮产出的两个 S3 候选（都还没实施）

| 候选 | 上界（推导） | 逐位性 | 风险 |
|---|---|---|---|
| ① `repeated_linearization` 按 `count` 缓存（2 次而不是 47 次） | **≈458 ms/步**（467 × 46/47） | **逐位相同**（同样的值、同样的 `.contiguous()` 结果，只是不重复拷） | 峰值显存 +1 份 (64, …) 线性化缓冲 |
| ② 试探状态线性化只做几何有效性判定（`:2196`），换成更便宜的判定或复用 | ≈406 ms/步（≈160 × 2.54 ms） | 非逐位（判定语义必须重建） | 需 NaN/翻转单元注入测试保证同样的检出 |

## 6. 边界

- 单工况（A16）、单窗口（步 1–150）、fp64、4090 D。
- `advance_wrk3` / `particle_BS` / `ring_BS` / `tangent_action` / `linearization` 相互重叠（例如 `tangent_action` 出现在刷新体内，`linearization` 同时出现在 `struct_step` 与刷新体内），表里标 `(overlaps)`，只可作 inclusive 读。
- §4 的 467 ms/步 是残差推导值，不是直接测量；直接测量需要在批循环内对 `expand().contiguous()` 单独打点。
- `Q16NativeAuthorEndpointLoad.assemble` 仍未插桩（类属性名不匹配），载荷组装未分离。
