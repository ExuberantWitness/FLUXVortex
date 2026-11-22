# A1 + A2：门禁真值化与端点专属加速度（2026-11-22）

> 提交：`reproduce_yamano2020_q16_flux_v5m_native.py`（A1）、`q16_flux_v5m_native_fsi.py`（A2）
> 钩子：`FLUXV_LOAD_GATE_STRICT`（A1，默认 0）、`FLUXV_THREE_SLOT_ENDPOINT_ACCELERATION`（A2，默认 1）
> 验收：默认路径 30 步 digest 链逐位不变；parity 6/6

## A1：载荷门禁从"读文件"改成"运行时重算"

**原来**：`_require_load_gates()` 只做 `json.loads` 一个固定路径 `artifacts/experiment/20260823_q16_fluxv5m_native_fsi/YAMANO_Q16_NATIVE_LOAD_GATES.json`，把它的 `metrics` 原样抄进 payload。所以无论代码与算子怎么变，payload 里的 `mf1_common_field_relative_error` 永远是 **0.3735346675462884**。

**现在**：调用同一个生产者 `verify_yamano2020_q16_native_load_gates.run_gates()`（自包含：它自己建 Yamano mesh/surface/solver）在进程内重算，payload 增加三个字段：`native_load_gate_source = "runtime_recompute"`、`native_load_gate_is_stale = false`、`native_load_gate_stale_artifact`（把旧文件的值作为**对照**记录）。

**实测一次 Yamano 正式 STEP1**：

| 字段 | 值 |
|---|---|
| `mf1_common_field_relative_error`（HEAD 重算） | **2.7270990867738667**（272.7%） |
| 陈旧工件里的值（仅作对照） | 0.3735346675462884（37.35%） |
| `status` | `passed_with_warning` |
| `warnings` | `['mf1_common_field']` |

**阻断行为不变**：`mf1_common_field` 在生产者里属于 `diagnostic_gates`，硬门是 `aic / mf2 / qf_resultant / mf1_physical_sign`。所以重算既不会新阻断正式 CASE，也不会放过硬门。`FLUXV_LOAD_GATE_STRICT=1` 可把诊断门失败也升级为阻断。

代价：每次正式 CASE 启动多一次 Yamano 小 case 的构建与门计算。

## A2：`previous_acceleration` 与端点专属加速度

**原来**（M4 的实现）：三槽式扣减在 β 插值前的端点做，但**两个端点共用锚加速度**——代码注释里写明了这是与审计方公式的偏差，因为 owner 只保留 `previous_aerodynamic_load`，没有 `previous_acceleration`。

**现在**：
- `Q16NativeV5MFSIOwner` 新增 `previous_acceleration: wp.array | None`；
- commit 时在替换 `owner.acceleration` **之前**把旧值记下（`:941`）；
- 三槽式的 previous 端点用 `previous_acceleration`，锚端点用 `acceleration`，首外步（无 previous）回退到锚——与审计方 A9-1 的 `prev_const_eff = prev_const − prev_am @ q̈_prev` 一致；
- `FLUXV_THREE_SLOT_ENDPOINT_ACCELERATION=0` 可退回旧行为以便 A/B。

**为什么在 A16 上验证而不是 Yamano**：三槽式只在 `wake_history_mode='bound_rate'` 下允许，Yamano 正式 CASE 是 `material`（打开 flag 会直接 `ValueError`）。A16 的 `bound_rate` 是唯一跑得到该分支的工况。

**实测（A16，`three_slot_mf1_cancel=1`，30 步）**：

| 形式 | 30 步平均 Cn | step_wall |
|---|---|---|
| 端点专属加速度（新，默认） | **0.285620778** | 2.187 |
| 锚加速度（旧） | 0.286640391 | 2.188 |

`max |ΔCn| = 2.732e-03`（步 3），均值相对差 **0.36%**。所以这一项**有实际数值效果**，与 M4 的 β 顺序（实测零效果）不同。

## 验收

| 检查 | 结果 |
|---|---|
| 默认路径 30 步 digest 链 | **逐位不变** |
| parity `test_roj_unified_case_runner_parity_gpu.py` | **6/6** |
| Yamano 正式 STEP1 运行 | 完成，门禁字段为运行中重算值 |

## 边界

- A2 的效果只在 `bound_rate` + 三槽式下可见；A16 单工况、30 步窗口、fp64。
- A1 的重算只覆盖该生产者的 4 项门（aic / mf2 / qf_resultant / mf1_common_field + mf1 物理符号），不覆盖其他工件（`YAMANO_Q16_5X3_MF1FIX_STEP4.json` 等）里的历史数字。
- A2 没有改变"首外步用锚加速度"这一处回退（那里确实没有 previous 加速度可用），它与审计方公式的差异仍然存在，只是范围从"所有端点"缩小到"仅首外步"。
- A1 的重算成本（一次额外 Yamano 小 case 构建）未计入任何速度表。
