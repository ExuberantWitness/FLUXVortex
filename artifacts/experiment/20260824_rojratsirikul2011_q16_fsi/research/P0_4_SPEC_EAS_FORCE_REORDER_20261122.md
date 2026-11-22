# P0-4（S3-v2B）实现规格：内力核换序免物化 `projected_b` —— 代数已算清，含一处必须先承认的后果

> 状态：**规格 + 代数推导完成，未实现**
> 依据：`kernels_q16_ans_eas.py:204-236`（`_b_entry`）、`:375-437`（`q16_projected_b_kernel`）、`:520-555`（`q16_ans_eas_force_kernel`）

## 1. 前提（已核实）

`q16_ans_eas_force_kernel` 只在最后一行读 `projected_b`：

```
for point in range(32):
    stress = _stress(strain + alpha*mode, λ, μ)
    for component in range(6):
        dual = Σ_row transform[element,point,row,component] * stress[row]
        result += reference_weight * projected_b[batch,element,point, component*96+dof] * dual
```

`dual` 只依赖 `strain`/`alpha`/`transform`，**与 B 无关** ⇒ 换序可行。

## 2. `_b_entry` 的秩结构（这是换序的钥匙）

`_b_entry(first, second, third, component c, coordinate k, coef1, coef2, coef3)` 对系数是**线性的**，且每个分量只用 1–2 项：

| c | 表达式 |
|---|---|
| 0 | `coef1 · first[k]` |
| 1 | `coef2 · second[k]` |
| 2 | `coef3 · third[k]` |
| 3 | `coef1 · second[k] + coef2 · first[k]` |
| 4 | `coef2 · third[k] + coef3 · second[k]` |
| 5 | `coef1 · third[k] + coef3 · first[k]` |

而 `projected_b[e,p,c,(node,k)] = Σ_sample w[p,s] · _b_entry(basis[e,p,s], c, k, coef1(node,s), coef2(node,s), coef3(node,s))`。

## 3. 换序后的形式

把 `projected_b` 的定义代入内力式并对 `c` 先求和：

```
M_1(e,p,s,k) = dual[0]·first[k] + dual[3]·second[k] + dual[5]·third[k]
M_2(e,p,s,k) = dual[1]·second[k] + dual[3]·first[k] + dual[4]·third[k]
M_3(e,p,s,k) = dual[2]·third[k] + dual[4]·second[k] + dual[5]·first[k]

force[e,(node,k)] = Σ_p w_p Σ_s w[p,s] · ( coef1(node,s)·M_1 + coef2(node,s)·M_2 + coef3(node,s)·M_3 )
```

`M_i` 只依赖 (element, point, sample, k) 与 `dual`；与 `node` 无关 ⇒ **16 个节点的贡献共用同一组 M**。

## 4. 工作量对比（每 element·point，手算）

| | 现路径 | 换序 |
|---|---|---|
| 物化 `projected_b` | 576 项 × 12 sample × ~2 乘 ≈ **13.8k 乘** + 921,600 个 double 的写/读（每次调用 7.4 MB） | **0** |
| 内力缩并 | 96 dof × 6 comp = 576 乘 | 12 sample × 3 k × (3 M ≈ 6 乘 + 16 节点 × 3 coef ≈ 48 乘) ≈ **0.8k 乘** |
| 合计 | ≈ **14.4k 乘** | ≈ **0.8k 乘** |

⇒ 算术量约 **18×** 下降（你估的 8–10× 偏保守），并且完全消除 7.4 MB/次的 `projected_b` 物化（S2b 实测 `lin_projected_b` 占一次 linearization 的 **75%**、546.1 ms/步中的绝大部分）。

## 5. 必须先承认的后果：**这一项不可能是逐位的**

与 P0-3 不同（散加顺序恰好等于 node-occurrence 求和 ⇒ 实测逐位相同），P0-4 改变了求和次序与因子化方式（把 `Σ_c dual·B` 拆成 `Σ_i coef_i·(Σ_c dual·∂B/∂coef_i)`），浮点加法/乘法不可交换结合 ⇒ **结果与现路径差 ~1e-15 相对**，而且它作用在**残差**上（Newton 接受判据），所以整条轨迹会从第 1 步起与现有 digest 基线分叉。

两个选项：

| 选项 | 影响 |
|---|---|
| **A（建议）** | flag 默认 **0**：生产路径保持逐位，P0-4 只在测量/后续切换时开启。代价是平时的速度不升。 |
| B | flag 默认 1：拿到 ~540 ms/步（150 步窗口约 −13%），但**整条链从此不再位级等价**，所有既有 digest 基线作废，后续验证改用"相对差 + 残差门 + 观测量容差"。 |

## 6. 落地步骤（若选 A 或 B）

1. 新增 `q16_ans_eas_force_reordered_kernel`（输入 `first/second/third`、`strain`、`alpha`、`transform`、`enhanced_mode`、`reference_weight`、`λ/μ`；输出 force），实现 §3 的 M 形式；
2. 新增一等价测试：同一 `(state, linearization)` 上，现核与换序核的 `force` 相对差 ≤ 1e-14，且逐 element/dof 比较；
3. 在 `_internal_force_prechecked` 加 flag 分支；换序路径需要 `first/second/third`（`_sample_fields` 已产出），**不需要 `_projected_b`** ⇒ 可让 `_current_fields` 在"仅残差"用途下跳过 `projected_b`（这正是省下 546 ms/步的地方）；
4. 150 步 A/B：步价、`max|ΔCn|`、`max|Δ|zmax|/c|`；
5. 若走选项 B，需同时重写验收口径文档（把"逐位"改成"相对差 + 残差门"）。

## 7. 边界

- §4 的算术量是手算的相对量级，未在 GPU 上实测；真实收益须按 §6.4 的 A/B 测。
- `M_i` 的推导假设 `sample_weight` 与 `reference_weight` 的乘积可交换（纯标量，成立），以及 `_b_entry` 对系数严格线性（已逐分支核对）。
- 若 `MAX_PROJECTION_SAMPLES` 或 `PROJECTION_COMPONENT_COUNT` 在别的网格上不同，§3 的形式不变（M 的个数恒为 3）。
