# V4.1 旧种子 W2 序列复现审计预登记

**日期**：2026-07-29  
**触发证据**：主推力 witness 的 W2 超过冻结 `0.15 N` 复现门  
**状态**：PRE-REGISTERED；不修改 solver，不放宽原门

## 1. 审计问题

区分以下三种情况：

1. **旧缓存漂移**：当前 W2 重复稳定，但系统性偏离冻结值；
2. **运行间/门限分叉**：相同 W2 在同进程重复本身超过 `0.15 N`；
3. **调用序列依赖**：W2 在相邻 tw20/tw25 调用前后发生超过 `0.15 N` 的变化。

本审计只分类数值身份，不选择气动 claim。

## 2. 冻结执行序列

所有调用沿用 full V4.1 生产配置。

1. `A0 = 8_2.6_0_5`：cold preconditioner，保存但不计正式重复；
2. `A1 = 8_2.6_0_5`：warm formal anchor；
3. `X1 = 10_2.6_22.5_5`；
4. `X2 = 10_2.6_22.5_5`，立即重复；
5. `B20 = 10_2.6_20_5`；
6. `B25 = 10_2.6_25_5`；
7. `X3 = 10_2.6_22.5_5`，在邻点后重复。

每次调用无论是否超过 0.15 N 都必须保存 raw L/T、节点贡献、guards、
graph identity、调用参数和 wall time；只有 guard/source/非有限值失败才中止。

## 3. 固定统计

- `cache_delta(Xi)`：相对冻结 W2 的逐通道有符号与绝对差；
- `repeat_spread`：X1/X2/X3 的 max-min；
- `sequence_shift`：X3 与 mean(X1,X2) 的绝对差；
- `neighbor_midpoint`：`0.5*(B20+B25)`；
- `midpoint_delta`：mean(X1,X2,X3) 与 neighbor midpoint 的差。

阈值沿用既有 `0.15 N`，不新增更宽容差。

## 4. 预登记分类

- `CACHE_DRIFT`：W2 repeat spread 和 sequence shift 均 ≤0.15 N，但至少一个
  通道的 mean cache delta >0.15 N；
- `RUN_VARIABILITY`：W2 repeat spread >0.15 N；
- `SEQUENCE_DEPENDENCE`：sequence shift >0.15 N；
- `CACHE_COMPATIBLE_BUT_PRIOR_OUTLIER_UNRESOLVED`：三次均在 cache band，
  但保留前一失败，不能自动恢复 authoritative 身份。

分类可同时成立，优先级为
`RUN_VARIABILITY > SEQUENCE_DEPENDENCE > CACHE_DRIFT > CACHE_COMPATIBLE...`。

## 5. 后续门

- 出现前三类任一项：禁止继续混合旧 seed；建立完全 fresh 的 confirmed151
  baseline，且每点绑定 source/graph/guard。
- 仅出现最后一类：扩大到预登记的 seed 代表点复现审计；仍不得直接恢复旧 seed。
- 禁止通过增大容差或用 tw20/tw25 插值覆盖 W2。
