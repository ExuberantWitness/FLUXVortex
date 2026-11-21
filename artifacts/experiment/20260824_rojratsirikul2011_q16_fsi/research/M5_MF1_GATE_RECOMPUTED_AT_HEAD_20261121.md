# M5：Mf1 gate 在 HEAD 上重算 —— 272.7%，payload 里的 37.35% 是陈旧硬编码值

## 1. 重算结果（`verify_yamano2020_q16_native_load_gates.py`，HEAD）

| 指标 | 值 |
|---|---|
| `aic_relative_error` | 3.007e-07 |
| `mf2_max_abs_error` | 3.816e-17 |
| `qf_resultant_max_relative_error` | 2.894e-03 |
| `mf1_author_common_field_work` | **-0.04127802486316853**（作者 MATLAB 局部算子）|
| `mf1_q16_common_field_work` | **-0.1538472887713444** |
| `mf1_q16_full_global_common_field_work` | **-0.1538472887713444**（与上一行相同 ⇒ HEAD 的算子就是全局的）|
| **`mf1_common_field_relative_error`** | **2.7270990867738667 = 272.7%** |
| `status` | `passed_with_warning`，warning = `mf1_common_field` |

阈值是 5%，实测 272.7% ⇒ **54 倍超阈**。

## 2. 为什么此前所有 payload 都写 37.35%

`platform/warp_vpm/reproduce_yamano2020_q16_flux_v5m_native.py:55-58` 把 gate 文件路径写死为
`artifacts/experiment/20260823_q16_fluxv5m_native_fsi/YAMANO_Q16_NATIVE_LOAD_GATES.json`，
`:368` 直接 `json.loads` 该文件并把 `metrics` 原样拷进 payload。
因此 **2026-08-23 之后的所有 Yamano payload（含本轮的 `/tmp/yam_*.json`）里的
`mf1_common_field_relative_error = 0.37353` 描述的都是旧的"局部支撑 Mf1"算子**，
不是 HEAD 的算子。

## 3. 对审计方 A5 溯源的影响

- 审计方给的两个数字（3.727 倍、`-0.153847` vs `-0.041278`）**在本轮重算中逐位复现**；
- 但"37.35% 是当前值"这一读法**不成立**——当前值是 272.7%；
- 该 gate 作为"被测项"应从既有清单中移除，或标注其来源为 2026-08-23 的局部算子。

## 4. 边界
- 本记录只重算 gate，未改生产路径；
- 272.7% 与 3.727 倍是同一事实的两种表达（前者是相对误差，后者是共同场功之比）；
- 未判断"全局 vs 局部 Mf1 哪个正确"——那需要受控 A/B（同 commit、同 mode、同 ρ），仓库中不存在。
