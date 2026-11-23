# S5-0 核查：`edge`/`inv_e` 外提**不是逐位等价**，且 FLOP 占比只有 ~5%（2026-11-23）

> 依据：`src/fluxvortex/warp_fsi/kernels_ring_velocity.py:33-91`（`_ring_bs_partials`）
> 状态：**核查完成，未实施**

## 1. 核的实际结构

核按 `(target, block)` 并行，块内循环源：

```python
for s in range(start, stop):
    ax = tx - starts[s,0];  bx = tx - ends[s,0]      # target 相对
    ex = ax - bx                                      # ← 数学上 = ends - starts
    inv_e = 1 / max(sqrt(ex²+ey²+ez²), eps)           # ← 数学上只依赖源
    ... 其余全部 target×source
```

所以 `ex/ey/ez` 与 `inv_e` 在**数学上**只依赖源，跨目标重复计算确实是浪费。

## 2. 但外提不是逐位等价

`ex` 现在算的是 `(tx − starts) − (tx − ends)`，而外提只能算 `ends − starts`。两者在浮点上**不相等**：

- 绝对误差 ~ `eps · |tx|`（~1e-16 × 弦长量级）
- `|ex|` ≈ 源段长度（0.003–0.03 量级）
- ⇒ `ex` 的相对差 ~ **1e-14**，经 `sqrt`、除法后进入整个环诱导场

这与 P0-3（散加顺序恰好等价、实测逐位）不同：这里是**表达式改写本身改变了舍入**。所以 S5-0 一旦落地，整条气动链不再逐位，所有既有 digest 基线作废。

## 3. 且收益比 v2 估的小

现核每对（target, source）的主要算术：3 次减法（a） + 3 次（b） + 3 次（ex） + 6 乘（cross） + 3 次 sqrt + 3 次除法（inv_a/b/e） + 3 乘 + 3 减（u） + 3 乘 2（scalar） + 1 除（inv_den） + 2 乘（h2） + 1 sqrt + 1 除（kv） + 3 乘（w·c）≈ **~45 个算子**，其中 `ex/ey/ez`（3 减）+ `inv_e`（1 sqrt + 1 max + 1 div）≈ **6 个 ⇒ ~13%**，而 `inv_e` 涉及的 sqrt+div 是最贵的两个指令，按延迟折算大约 **15–25%** 的核时间 —— 与 v2 的"~25% FLOPs"同量级。

但**只有 `inv_e` 那部分是真正可省**（`ex/ey/ez` 三个减法即使外提也要读数组，收益近零）。按 `ring_BS` 800 ms/步 计，**上限 ≈ 120–200 ms/步（2–3.5%）**，且要以放弃逐位为代价。

## 4. 真正剩下的气动大块不在这一项

| 项 | 实测 | 性质 |
|---|---|---|
| `ring_BS` | 800 ms/步（47.95 次重启动/步） | 主体是 target×source 的逐对工作，无重复 |
| `particle_BS` | 937–965 ms/步 | 同上；`advance_wrk3` 的 3 阶 RK3 各一次 |
| 上述两项的**远场 fp32** | —— | **非逐位**，需新核 + 源的 fp32 影子（S5-1） |
| S5-0 外提 | ≈120–200 ms/步 | 非逐位（1e-14） |

⇒ 气动侧要再快，只有两条：**(a) 远场混合精度（大改、非逐位）**，或 **(b) 减少对流次数（S4a/S4b，已实现，flag 默认 0，且 cull 账本待修）**。S5-0 是这两者之间唯一"小改但非逐位、收益 2–3.5%"的选项。

## 5. 边界

- 上述 FLOP 分解是手工计数，未做 GPU 实测；`ring_BS` 800 ms/步 是 S2b 的实测值。
- 未试"shared-memory 装载 `filament_gamma`/`filament_core4`"这类**逐位安全**的访存优化——它不改算术，只改访存，收益待测（可能在带宽受限时有 5–15%）。
