# A16 法向力分解（#1）的实现方案：rhs 线性可分 ⇒ 三项贡献可精确拆开

> 状态：**方案就绪，未实现**（需要改 `propose` 并逐项验证；本轮上下文不足以完成验证）
> 同时启动的长跑：`ROJ11_A16_FULL_20261123_CNCONV.json`（2100 步，带 `h9_mean_cn_normal` 门）

## 1. 为什么这项能精确做（不是近似）

`Q16NativeV5MSolver.propose` 里的配点右端是四项的**线性组合**：

```
rhs = Σ_panels ( collocation_velocity − v_inf − wake_velocity − particle_velocity ) · normals
γ   = A⁻¹ rhs                      (A = 冻结点涡 AIC，可复用其 LU)
load = L(γ)                        (L 对 γ 线性：KJ 点力或 5P 求积)
```

`L ∘ A⁻¹` 是线性的 ⇒ **载荷可以按 rhs 的分项精确分解**：

| 分项 | rhs 项 | 物理含义 |
|---|---|---|
| 运动学/束缚 | `collocation_velocity − v_inf` | 刚体运动 + 迎角 |
| 尾迹 | `−wake_velocity`（环诱导，含束缚环 + 尾迹环） | 尾迹下洗 |
| 粒子 | `−particle_velocity`（`add_connected_ribbon` 沉积的 LEV 云） | **LEV 表示**（1/25 × 10× 相消对所在项） |

做法：对每项各算一次 `load_i = L(A⁻¹ rhs_i)`，取 `load_i · n̂ / qS` 得到法向力分量。因为 `A⁻¹` 的 LU 已经缓存（`_aic_lu`），每多算一项只多一次 450×450 回代 + 一次载荷缩并，**代价可忽略**。

## 2. 实现（约 30 行，flag 门控）

1. 在 `propose` 里把 rhs 的组装改成分项保留：
   ```
   rhs_kin = (collocation_velocity − v_inf) · normals
   rhs_wake = (−wake_velocity) · normals
   rhs_part = (−particle_velocity) · normals
   rhs = rhs_kin + rhs_wake + rhs_part          # 逐位等于原来的求和
   ```
   **注意**：改成"先分项再相加"会改变求和次序 ⇒ 与现路径差 ~1e-16。若要保留逐位，就额外再算一遍分解用的 rhs（不动主路径）。按你"逐位不重要"的口径，直接分项即可。
2. 在**同一个 γ = A⁻¹ rhs 的 LU** 上，对三项各做一次回代得到 `γ_i`，再各做一次载荷缩并得到 `Fn_i`。
3. 结果写进 `retention`（⇒ payload 的 `aerodynamic.load_split_normal_*`），并记录 `Fn_kin + Fn_wake + Fn_part` 与总 `Fn` 的闭合误差（应 ≤1e-14 相对）。
4. 一次 150 步的运行即可读出三项占比（法向力在 `t*∈[1,1.5]` 已建立）；若要收敛窗的值，跑 2100 步。

## 3. 这一步要回答的问题

- **`Fn_part / Fn_total` 是多少？** 如果 LEV 粒子项占总法向力的绝大部分（在刚翼上是 +0.878 → −0.912 的符号翻转级别的权重），则膜翼的 +12~22% 残差很可能就在这一项上。
- **`Fn_wake` 与 `Fn_kin` 的比例**是否与刚翼不同？膜翼在 16° 已深度分离，尾迹下洗项会显著增大——若 `Fn_wake` 偏大，则是尾迹/环量输运的问题，不是 LEV 沉积。
- 与已确立的"近场效力 1/25"对照：若 `Fn_part` 只占很小份额却仍需 10× 闭合放大才能对上总量，则相消对在膜翼上的配比与刚翼不同，这就是 L3 的直接证据。

## 4. 边界

- 分解只对**载荷**成立（`L∘A⁻¹` 线性）；对**结构响应**不线性（几何非线性、大变形），所以不能把 `zmax` 按同样比例拆开。
- 分离判据（LESP/释放）依赖的是 rhs 的**总**值，不是分项 ⇒ 分解不会改变时间推进（除非改主路径的求和次序）。
- `particle_velocity` 一项包含了 `advance_wrk3` 之后的**全部**粒子（含新沉积的与旧尾迹粒子），如果要进一步分成"新沉积 LEV"与"旧粒子"，需要按 `birth_step` 或 `source_strip` 再筛一次（`ptype`/`source_strip` 数组已存在）。
