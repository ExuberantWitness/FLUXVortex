# 粒子 Biot–Savart 的 target 集合实测（#2/#3）

> 方法：monkeypatch `CudaParticleField.velocity_at_cuda` / `velocity_self_cuda` /
> `velocity_and_jacobian_self_cuda`，按**调用者函数**与 target 数记录；A16 跑 15 步。
> 脚本 `/tmp/count_targets.py`。**未修改任何生产代码。**

## 1. 每步的调用构成（15 步，`step_wall_mean = 3.945 s`）

| 次数/步 | 调用者 | targets | 说明 |
|---|---|---|---|
| **5.5** | `q16_flux_v5m_native.py:propose` | **450** | 配点（nc×ns）|
| **2.8** | `q16_flux_v5m_native.py:propose` | **1800** | `mids_flat`（4×450 涡丝中点）|
| **1** | `q16_flux_v5m_native.py:wake_velocity` | **120 → 480 → 600 → 720 …** | **尾迹顶点**，随尾迹行数线性增长 |
| **1** | `velocity_self_cuda` → `velocity` | **n（粒子数）** | 粒子自诱导 |
| 1 | `velocity_self_cuda` 外层 | n | 同上（wrapper 转调 `velocity_at_cuda`）|

尾迹顶点 target 数的观测序列 120 / 480 / 600 / 720 … = `n_rows × ns × 4`（ns=30）：
120 = 1 行，480 = 4 行，600 = 5 行，720 = 6 行。

## 2. 两个结论（可核对的事实）

### 2.1 尾迹顶点的粒子诱导对**全部行**求值

`wake_free_rows = 100`、`wake_max_rows = 300`。实测 target 数 = `n_rows × 30 × 4`，每步 **1 次**，行数随时间增长。
⇒ **`wake_free_rows` 的门控没有作用到"粒子诱导求值"这一步**；即使某行已冻结（只随来流平移、不参与形变），它的顶点仍被当作 target 参与粒子互诱导计算。

### 2.2 粒子自诱导每步 1 次，target = source = n

稳态 n = 27,371 ⇒ **7.49×10⁸ 对/次**，每步 1 次。

## 3. 由此得到的稳态每步 BS 对数（按 target 数 × source 数）

source 数 = n = 27,371（稳态）：

| target 集合 | targets/步 | 对/步 | 占比 |
|---|---|---|---|
| 尾迹顶点（300 行上限）| 36,000 | **9.85×10⁸** | **51%** |
| 粒子自诱导 | 27,371 | **7.49×10⁸** | **39%** |
| 配点（5.5×450）| 2,475 | 6.78×10⁷ | 3.5% |
| 涡丝中点（2.8×1800）| 5,040 | 1.38×10⁸ | 7% |
| **合计** | ~70,900 | **1.94×10⁹** | — |

## 4. 尚未测量的两格

- 稳态（150 步窗口）下 `.item()` 的占比：已知 20 步窗口（粒子 ~5k）为 **60%**；稳态未测（采样 profiler 因 GIL 争用过慢，运行被拉长，见下条）；
- 每步各相位的墙钟拆分：payload 只有 `step_wall_seconds_mean` 一个总账；Roj harness 不输出逐步 `__DS_PROGRESS__`（Yamano harness 有）。

## 5. 已取得但需说明来源的单价点

| 窗口 | 粒子数 | `step_wall_seconds_mean` |
|---|---|---|
| 3 步 | 尾迹近空 | 2.08 |
| 15 步 | ~1.6k | 3.945 |
| 20 步 | ~5k | 4.35–4.64 |
| 40 步 | ~11k | （采样 profiler 运行中，进程耗时被采样线程拉长，未取）|
| 60 步 | ~16k | 6.156 |
| **150 步** | **~27k** | **9.136** |

**注意**：40/60 步的两次运行都叠加了采样线程或 cProfile 的开销，其 `step_wall_mean` 不能与无插桩的运行直接比较；只有 150 步（`/tmp/a16_150.json`）是无插桩测量。

## 6. 复现
```
PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \
  python3 /tmp/count_targets.py        # #2/#3 target 计数
```
