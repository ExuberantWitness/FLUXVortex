# 三论文退化修复报告

日期：2026-08-24

## 问题

修改 LESPcrit（物理化公式）后，用户要求验证不干扰已验证成果。
发现 Yang/Izra 大幅退化（lift MAE 68.4 vs 4.10; CT MAE 0.465 vs 0.0175）。

## 根因

Q16/FSI 开发修改了 `bing_joint_ptera_gpu.py` 的 `_calculate_loads`，
将 Pterra 验证过的 GP1→W 力变换替换为新的 "V5M scientific frame"
变换（`_transform_wrench_cuda`），导致力符号/量级错误。

**不是 LESPcrit 修改**——代码路径完全隔离（三篇论文不导入
`q16_flux_v5m_native.py`）。

## 修复

`7b46905`：恢复 Pterra GP1→W 变换（冻结路径的精确公式）。

## 验证结果（修复后，当前树）

### Baik（bit-exact，与冻结值完全一致）
| Case | CL RMSE | CD RMSE |
|---|---|---|
| W1 | 0.38142 | 0.17629 |
| W2 | 0.42245 | 0.35308 |
| W3 | 0.38540 | 0.21723 |
| W4 | 0.49699 | 0.41224 |
| **宏** | **0.42156** | **0.28971** |

### Yang（6 攻角全匹配冻结参考）
| 指标 | 修复后 | 冻结值 | V4B |
|---|---|---|---|
| lift MAE | 4.1048 | 4.1048 ✓ | 4.55 |
| drag MAE | 1.5186 | 1.5186 ✓ | 2.64 |

### Izraelevitz（12 条件全匹配）
| 指标 | 修复后 | 冻结值 | V4B |
|---|---|---|---|
| CT MAE | 0.01745 | 0.01745 ✓ | 0.0198 |

## LESPcrit 修改影响评估

`compute_lesp_crit()` 在 `q16_flux_v5m_native.py` 中，只在
`NativeV5MConfig` 的 `effective_lesp_crit` 属性中使用。
三篇论文的 runner 不导入该文件 → **零影响**（已通过 Baik bit-exact 实证）。

## 提交

- `11e0215`: LESPcrit 物理化公式
- `7b46905`: force transform 修复（本次退化的根因修复）
