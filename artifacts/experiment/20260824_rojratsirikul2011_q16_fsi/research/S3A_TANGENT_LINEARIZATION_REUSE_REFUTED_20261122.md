# S3a：`repeated_linearization` 缓存 —— 假设否证，只值 0.6%；刷新成本全在切向作用本身（2026-11-22）

> 提交：`_TANGENT_LINEARIZATION_REUSE`（默认 1）+ `tests/test_q16_tangent_refresh_linearization_reuse_gpu.py`
> 原始：`/tmp/s3a_phase150.json`、`/tmp/s3a_microbench.py`、`/tmp/s3a_batchwidth.py`
> 前置：`S2B_STRUCTURAL_INTERIOR_ATTRIBUTED_20261122.md`

## 1. 假设与结果

S2b 把 `tangent_refresh` 的 1370.6 ms/步 拆成 "`tangent_action` 903.5 + 余下 ≈467（推导）"，并推测余下主要是每批重建 `repeated_linearization`（`wp.to_torch(v).expand(...).contiguous()`，47 批里只有 2 个不同 `count`）。

**改成按 `count` 缓存（47 次 → 2 次）后实测：**

| | `tangent_refresh` ms/步 | 墙钟 s/步 |
|---|---|---|
| 改前 | 1370.6 | 5.573 |
| 改后 | **1362.3** | 5.656 |

**只降 8.3 ms/步（0.6%）**，墙钟在运行间波动内。假设否证。

## 2. 微基准把余下的钱找出来了

`/tmp/s3a_microbench.py`（真实算子、冻结线性化、逐语句 CUDA Event，冷/热两趟各 47 批）：

| 语句 | ms/次 | 说明 |
|---|---|---|
| `directions`（zeros + arange + 单位散布） | 0.15 | |
| `repeated`（`expand().contiguous()`） | **1.58** | 冷趟 47 次、热趟 2 次合计 94 次 |
| `action`（`_tangent_action_linearized_prechecked`） | **153.89** | **占总时间的 99%** |
| `scatter`（`wp.to_torch` + `tangent_matrix[:,s:e] = action.T`） | 0.12 | 我原以为这个跨步幅转置写会贵，实测不是 |

即：**一次刷新 10.8 s，其中 47 × 153.89 ms ≈ 7.2 s 全在切向作用里，批循环记账是零头。** S3a 省下的 45 次重建 × ≈3 ms ≈ 135 ms/次刷新 ≈ 17.5 ms/步，与实测 8.3 ms/步 同量级（剩下的差在缓存本身也要建 2 次）。

## 3. 批宽杠杆也不存在

代码注释写"列项与批宽无关，切线对任何批宽逐位相同"，且 2 GB 工作区预算（每方向 50×108×576×8 = 24.9 MB）允许 86 方向，而硬上限是 `min(64, dof)`。于是测 64 / 86 / 108：

| 批宽 | 批数 | 墙钟 | 与 64 逐位相同 |
|---|---|---|---|
| 64 | 47 | **10.80 s** | — |
| 86 | 35 | **10.83 s** | **是**（`max|Δ| = 0`） |
| 108 | 28 | **10.81 s** | **是** |

**批数从 47 降到 28，时间一点没变** ⇒ 刷新成本与批数无关，是每列算子作用的**吞吐**（10.8 s / 2976 列 = 3.63 ms/列），不是启动开销。批宽这条路关闭。

## 4. 结论与边界

- `tangent_refresh`（1362 ms/步 = 步价 24%）在当前实现下**没有位级安全杠杆**：批宽无效、记账无效，唯一剩的是"更少刷新"（`reference_tangent_refresh_rtol` 是数值参数，属改物理）或"换切线装配方式"（解析装配 / 低秩更新替代 2976 列探测，属大改）。
- 保留 S3a 的改动：逐位相同（`tests/test_q16_tangent_refresh_linearization_reuse_gpu.py` 对真实算子比较两条路径的 2976² 切线，`torch.equal` 通过），省掉每次刷新 45 次 24.9 MB 的 `contiguous()` 拷贝，实测 0.6%。`FLUXV_TANGENT_LINEARIZATION_REUSE=0` 可回到逐批重建。
- 微基准的语句级拆分有一个已知缺口：`action` 的 153.89 ms/次与本轮 pass 总墙钟之间有约 3.5 s 未归因，可能是 Warp 自用流与 torch 事件流的边界（`wp.from_torch`/`wp.to_torch` 处的同步）导致事件对没有覆盖全部 GPU 时间。这不影响上面的结论（结论只用到"`action` 占绝对多数"与"批宽无效"两条，两者都由独立测量支持）。
