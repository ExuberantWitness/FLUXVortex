# S3b：`velocity_force` 复用 —— 无命中（前提错），真实冗余上限只 73 ms/步（1.3%）⇒ 撤回（2026-11-22）

> 已撤回代码改动（`git checkout src/fluxvortex/warp_fsi/q16_flux_v5m_author_loads.py`）
> 数据：`/tmp/s3b_30.json`、剖析器 `--steps 30` 输出

## 1. 我原来的前提与实测

**前提**：`_integrate_structure` 每子步对 `velocity_force` 调 4 次（anchor/endpoint × predictor/corrector），后两次与前两次用**同一个** `velocity` 对象，所以按 `(velocity 对象, include_mf21)` 记忆整个调用可命中 50%。

**实现后实测（30 步，逐位正确但因无命中而无收益）**：

| 量 | 改前 | 改后 |
|---|---|---|
| digest 链 | 基线 | **全等** |
| `step_wall` | 2.406 s/步 | 2.406 s/步 |
| `surface_evaluate` 调用/步 | 179.97 | **179.00**（≈0 命中） |

**前提错在哪里**：`:578`
```python
average_velocity_t = 0.5 * (wp.to_torch(velocity) + wp.to_torch(predictor.velocity))
average_velocity = wp.from_torch(average_velocity_t, ...)
```
第二对调用（`:583/586`）用的是**新造的 `average_velocity`**，不是 `velocity`。所以每子步 4 次调用的输入两两不同 ⇒ 记忆永远不命中。已撤回该改动（空转代码不留）。

## 2. 真正剩下的冗余只有状态侧，上限 73 ms/步

`surface.evaluate` 在 30 步窗口的构成（剖析器新增标签）：

| 项 | 调用/步 | ms/步 |
|---|---|---|
| `surface_evaluate` 总 | 179.97 | **145.6** |
| ├ `assemble_native_ring_geometry` | 179.97 | **75.9** |
| └ 6 次 transfer 插值（未单独打点的那条路径） | — | ≈69.7 |
| `velocity_force` 总 | 175.03 | 500.6 |

载荷对象的 `structural_state` 是**冻结**的，所以 `evaluate` 里的状态侧结果是每次调用都一样的：
- 3 次 state 插值 ≈ 69.7 × 3/6 ≈ **35 ms/步**；
- `assemble` 里由 state 决定的 `rings / collocation / normals / areas` ≈ 75.9 的一半 ≈ **38 ms/步**。

合计 **≤73 ms/步 = 步价的 1.3%**，而且要做到必须**把 `assemble_native_ring_geometry` 拆成"状态侧"和"速度侧"两半**（它目前一次调用同时产出两侧的张量），属于有逐位风险的重构。

## 3. 判定

- 收益 1.3%，改动是有风险的中间层重构 ⇒ 性价比低于已识别的其他项（试探态线性化闸门 ≈406 ms = 7.2%、1B 每外步单次分解 ≈196 ms = 3.5%）。
- 已把 S3b 从待办中移出，代码零改动（仅剖析器新增两个标签与"模块级函数"包装支持，保留）。

## 4. 边界

- 30 步窗口（约 8k 粒子）；150 步稳态下 `surface_evaluate` 是 156.6 ms/步（S2/S2b 表），比例相近，所以 1.3% 的结论在稳态也成立。
- 6 次 transfer 插值的那条路径没有被单独打点（我包装的 `Q16CudaSurfaceTransfer.interpolate` 只被调 1.1 次/步，说明 `evaluate` 走的是另一条），69.7 ms 是 `evaluate − assemble` 的残差，不是直接测量。
