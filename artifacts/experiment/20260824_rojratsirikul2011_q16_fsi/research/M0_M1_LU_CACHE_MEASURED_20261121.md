# M0/M1：LU 分解缓存的实测（审计方 C3 成立；1.57× 提速且逐位相同）

## M0 —— 用 CUDA Event 核实真实 LU 成本

脚本 `/tmp/m0_lu_events.py`，2436² fp64，`torch.cuda.Event(enable_timing=True)`（非 `perf_counter`）：

| 操作 | 实测 |
|---|---|
| `lu_factor_ex(A)` | **21.65 ms** |
| `lu_solve(lu, piv, b)` | **0.600 ms** |
| 比值 | **36.1×** |
| `solve_ex(A, b)`（异步外壳，事件计时）| 21.75 ms |
| `solve_ex + info.item()` | 21.90 ms |
| 隐含吞吐（`2/3 n³` = 9.64 GFLOP）| 0.445 TFLOP/s |

⇒ 我此前 `perf_counter` 测得的 **0.47 ms/次是入队时间**；真实 GPU 时间是 **21.8 ms/次**。以 104.9 次/步计 ⇒ **2.29 s/步**，与 `phase.dense_solve` 实测 2.56 s/步（含有效矩阵装配）一致。
**`SPEED_DIAGNOSIS...md` 与 `REPLY_ROUND3...md` 中"LU 只占 1.0%"的归因作废。**

## M1 —— 缓存分解（`FLUXV_LU_CACHE=1`，默认关）

实现：`q16_structural_solver.py::_reference_dense_nonsymmetric_solve` 由
`solve_ex(effective_free, right)` 改为 `lu_factor_ex` 缓存 + `lu_solve`。
**逐位等价**：torch 的 `solve_ex` 内部即 `_linalg_solve_ex_out` → `linalg_lu_factor_ex_out` + `linalg_lu_solve_out`。

| 检查 | 结果 |
|---|---|
| 3 步 cn | 与基线 `repr` 完全一致 ⇒ **逐位相同** |
| 30 步 `step_wall_mean` | **3.002 s/步**（基线 4.720）⇒ **1.57×** |
| 30 步 wall | 107 s（基线 141.6 s）|
| 实测省时 | **1.72 s/步**（M0 预测上限 2.0 s/步）|

## 一个必须记录的失败与根因

第一版键用 `id(acceleration_load_action)`，30 步运行在**线性残差门**失败。根因：`Q16NativeAddedMassAction` 每子步新建，**对象被 GC 后 Python 复用 `id()`**，于是命中了一个内容不同的旧分解。
修正：键改为值指纹 `(id(tangent), tangent._version, inverse_beta_dt2, added_mass.data_ptr(), added_mass._version, shape)`，并把 `added_mass` 张量**存入缓存条目**以防止 `data_ptr` 被回收后重用。

## 未达到预测上限的原因（未进一步优化）

M0 预测 105→10 次分解可省 2.0 s/步；实测省 1.72 s/步。差额来自：`added_mass` 每子步新建 ⇒ `data_ptr` 变化 ⇒ 跨子步不命中，命中主要发生在同一子步内的 1.8 次 Newton 迭代与同一耦合迭代内。
要逼近 10 次/步，需在 `Q16NativeV5MFSIStepper`（普通类）内按 `(id(anchor_am_matrix), id(am_slope), beta)` memoize 该 action 并持有引用（同时消除 `id` 复用风险）。**尚未实施。**

## 验收状态
- 逐位相同（3 步）；
- 30 步 1.57×；
- parity 6/6：见提交信息；
- 默认关，生产路径逐位不变。
