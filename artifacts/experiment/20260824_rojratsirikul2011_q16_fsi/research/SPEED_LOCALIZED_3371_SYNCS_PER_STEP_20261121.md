# 定位完成：**3371 次 `.item()`/步 = 60% 墙钟**，前两个来源占 57%（均为纯校验）

> 方法：monkeypatch `torch.Tensor.item` 记录 call site；A16 跑 20 步（只读，无修改）。
> **注意：3 步测速会严重低估**（尾迹空）；本次用 20 步。

## 1. 总量

| 量 | 值 |
|---|---|
| 20 步墙钟 | 87.0 s ⇒ **4.35 s/步**（粒子 ~5k）|
| `.item()` 调用 | 67,422 ⇒ **3371 次/步** |
| `.item()` tottime | 55.6 s / 92.9 s = **60%**（20 步 profile）|
| 单次代价 | ~826 µs ⇒ **主要是在排空 GPU 流水线，不是 D2H** |

与 8 月 profile 对比：当时 1524 次/步、22%；**现在 3371 次/步、60%**——增加了 2 倍，说明有路径新增了逐迭代读回。

## 2. 按调用点分组（前 8 名占 84%）

| 次数/步 | 占比 | 调用点 |
|---|---|---|
| **1240** | **36.8%** | `q16_flux_v5m_native.py:40 _require_cuda64` ← `isfinite().all().item()` |
| **688** | **20.4%** | `q16_flux_v5m_author_loads.py:235 line_derivative` |
| 292 | 8.7% | `q16_structural_solver.py:1446 _reference_dense_nonsymmetric_solve`（线性残差门）|
| 228 | 6.8% | `q16_flux_v5m_native_fsi.py:336 _integrate_structure` |
| 195 | 5.8% | `q16_structural_solver.py:936 _acceleration_load_matrix` |
| 177 | 5.3% | `q16_flux_v5m_native.py:493 assemble_native_ring_geometry` |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:207 material_ring_velocity_derivative_expanded` |
| 172 | 5.1% | `q16_flux_v5m_author_loads.py:349 velocity_force` |

## 3. 判读

**前两名（1928 次/步，57%）都是纯有限性校验**，不参与任何数值决策：
- `_require_cuda64`：`bool(torch.isfinite(v).all().item())`
- `line_derivative`：同类模式

它们是**可以直接换成设备端 `atomic_max` 标志 + 每步读一次**的（范式已在 `kernels_q16_ans_eas.py:693-709`；我在 S1 里已经用它包了切线刷新，但 S1 只覆盖那一处，所以在 3 步测速里看不出收益）。

其余 6 项（~1150 次/步）混合：`_acceleration_load_matrix`、`assemble_native_ring_geometry`、`velocity_force` 中的校验可一并去掉；但 `_reference_dense_nonsymmetric_solve` 的线性残差门与 `_integrate_structure` 的 Newton 判据是**控制流决策**，需要"设备端累积 + 每子步读一次"而不是删除。

## 4. 预期与边界

- 只做前两名 ⇒ 墙钟 **~1.5–2×**（5.33 h → 2.7–3.5 h）；
- 全部 3371 次压缩到每步 ≤50 次 ⇒ 理论上 **~2.5–3×**；
- **不足以到 20 分钟**，也与 A1（张量搬运）无关——**A1 打偏了，A2（Woodbury）在同步占 60% 的前提下没有意义**。若要把全窗压到 ~50 min，必须再动结构侧的真实计算量（子步求解次数或矩阵规模），那是改数值路径，需要单独的等价性论证。

## 5. 结论（给决策）
当前 A16 全窗 5.33 h 的构成：**~50–60% 是主机同步等待**、~15% 是气动+粒子、其余是结构线性代数与装配。**先把同步砍掉是唯一无损且立刻见效的一步**；它之后才知道剩余部分是否值得继续投入。
