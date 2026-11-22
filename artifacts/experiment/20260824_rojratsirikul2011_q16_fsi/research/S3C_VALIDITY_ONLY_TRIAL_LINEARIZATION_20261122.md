# S3c：试探态线性化只算"会抛异常的部分" —— 逐位相同，150 步 5.630 → 5.346 s/步（1.053×）（2026-11-22）

> 钩子：`FLUXV_TRIAL_LINEARIZATION_VALIDITY_ONLY`（默认 **1**，逐位等价）
> 数据：`/tmp/s3c_30.json`、`/tmp/s3c_80.json`、`/tmp/s3c_150.json`（对照 `/tmp/s1_80_cache1.partial.json`、`/tmp/s4_off150.json`）
> 测试：`tests/test_q16_validity_only_linearization_gpu.py`（4/4）

## 1. 浪费在哪里

牛顿线搜索里 `trial_linearization = self._operator._linearization_prechecked(trial)` 的结果，在常规路径上被**丢弃**：

```python
if not globalize_newton:
    accepted_trial = trial
    break          # 值没被用过；只有罕见的 globalize_newton 分支才会用它
```

它的唯一用途是**异常**（几何方向反转/奇异）。而 `_current_fields` 能抛异常的位置只有两处：

| 位置 | 异常 |
|---|---|
| `_strain` 内的 `q16_geometry_gate_kernel` | `Q16 current geometry is orientation reversing or singular` |
| `_alpha` 内的 `_assert_finite2` | `Q16 EAS parameter contains non-finite values` |

`_projected_b` **一处都不抛**，而它正是最贵的一步（每次物化 1×50×108×576 ≈ 310 万个 float64）。

## 2. 实测的线性化内部拆分（30 步窗口）

| 项 | 调用/步 | ms/步 | 占比 |
|---|---|---|---|
| `linearization` 总 | 267.73 | **724.4** | 100% |
| ├ `lin_projected_b` | 267.73 | **546.1** | **75%** |
| ├ `lin_sample_fields` | 267.73 | 91.2 | 13% |
| ├ `lin_alpha` | 267.73 | 53.9 | 7% |
| └ `lin_strain` | 267.73 | 25.3 | 3% |
| 合计 | | 716.5（闭合 1%） | |

## 3. 实现

`Q16CudaMITC16EASOperator.validity_only_prechecked(state)` = `_sample_fields` + `_strain` + `_alpha`（同样的三次调用，去掉 `_projected_b`），并由 `Q16CudaMITC16EASMeshOperator` 转发（外层算子只是 gather 后转给元素算子）。

**为什么逐位**：异常面按构造完全相同（同两个抛出点、同样的消息），`_projected_b` 不抛 ⇒ 接受/拒绝的试探序列、`factor` 折半次数、`accepted_trial` 全部不变。

## 4. 验收（全部做过）

| 检查 | 结果 |
|---|---|
| 30 步 digest 链 vs 基线 | **全等** |
| **80 步** digest 链 vs 改动前树 | **80/80 全等** |
| **150 步** digest 链 vs 改动前基线 | **全等** |
| parity `test_roj_unified_case_runner_parity_gpu.py` | **6/6** |
| 注入测试 | **4/4**：坍缩态两条路径抛出**同一消息**；压缩扫描 1.0/0.5/0.1/0.02/0.0 五个点**判断完全一致** |

## 5. 步价

| 窗口 | 改前 | 改后 | 加速 | 省 |
|---|---|---|---|---|
| 30 步 | 2.406 s/步 | **2.152** | 1.118× | 0.254 s（10.6%） |
| **150 步（稳态）** | **5.630 s/步** | **5.346** | **1.053×** | **0.284 s（5.0%）** |

提议数 447 = 447、耦合迭代均值 1.980 = 1.980，全部不变。

稳态增益（5.0%）小于 30 步窗口（10.6%）的原因：30 步窗口结构侧占比更高，而稳态下气动侧已占 32.7%、切线刷新占 24%。

## 6. 边界

- 省掉了 `_projected_b`（约 75%），但有效性路径仍要付 `sample_fields + strain + alpha`（30 步窗口约 170 ms/步）。
- `linearization` 剩下的部分是约 219 次/步对 `current` 态的调用——那些**真的需要** `projected_b`（残差/内力用它），不在本项范围内。
- 单工况（A16）、fp64、4090 D；150 步窗口 runner 自报 `stationary_window_found: false`。
- 注入测试覆盖的是"坍缩/压缩"这一类几何失效，没有覆盖"有限但极端"的状态（那类由 `_alpha` 的有限性检查兜底，两条路径共用同一检查）。
