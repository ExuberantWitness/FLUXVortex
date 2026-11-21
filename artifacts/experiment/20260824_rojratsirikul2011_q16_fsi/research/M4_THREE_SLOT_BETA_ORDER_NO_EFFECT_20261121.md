# M4：三槽式的 β 顺序修正 —— 已实现，实测无差异（受限于 owner 不保留 per-endpoint 加速度）

## 1. 改动
`three_slot_mf1_cancel` 的 `Mf1 q̈` 扣减由"β 插值**之后**、用子步初值加速度"改为"在**外步端点**扣减、再插值/外推"：
`anchor_const_eff = anchor_const − anchor_am @ q̈`、`prev_const_eff = prev_const − prev_am @ q̈`（首步用 `end_const/end_am`）。

## 2. 实测（Yamano，`bound_rate` + 三槽式）

| 步 | 旧顺序 | M4 |
|---|---|---|
| 1 | 2.029% | 2.029% |
| 2 | 2.550% | 2.550% |
| 3 | 8.095% | 8.095% |
| 4 | 13.281% | 13.281% |

**4 步全部相同**（步 1 相同属必然：首步结构加速度为零，所有 `Mf1 q̈` 项消失）。

## 3. 为什么没有差异（代数原因）
我没有各端点自己的加速度：`Q16NativeV5MFSI­Owner` 只保留 `previous_aerodynamic_load`（`q16_flux_v5m_native_fsi.py:200`），没有 `previous_acceleration`。因此对**所有**端点都用同一个 `q̈_anchor`。

设 `A, P` 为锚/前端点常值、`M_a, M_p` 为其 Mf1，同一加速度 `a`：
- 插值前扣减：`A + β(A−P) − M_a a − β(M_a−M_p) a`
- 插值后扣减（旧）：`A + β(A−P) − M_a a`

两者之差 = `−β (M_a − M_p) a`。实测该量低于 3 位小数的可分辨范围（`a` 小、`am_slope` 小）。

## 4. 要真正实现审计方的公式，需要
在 owner 上保存 `previous_acceleration`（以及首步的 endpoint 加速度），使每个端点用**自己的** `q̈` 扣减。这是**状态字段的新增**（会影响 `digest()`/`clone()`/checkpoint），应作为独立一项，而不是混在本次顺序修正里。

## 5. 边界
- 改动只作用于实验 flag `three_slot_mf1_cancel`（默认关）；生产路径不受影响；
- 4 步不足以暴露高阶差异；若将来加入 per-endpoint 加速度，需 8 步对照；
- 本项**未带来精度收益也未带来回退**。
