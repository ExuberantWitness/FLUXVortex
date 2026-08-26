# Q16 DVM node-ribbon 4/8 步 FSI 清单

## Identity

- run id：`20260822_q16_dvm_node_ribbon_fsi_8step`
- idea：原 `Lcrit=0.11` 的 DVM-Q16 FSI 长时事务
- stage：`auxiliary/dev`
- 状态：`PASS (bounded auxiliary/dev gate)`

## Planning

- [x] 冻结用户要求与条件释放语义
- [x] 确认两步 DVM 基线和历史 Q16 长时结构夹具
- [x] 冻结 4 步 pilot、4+4 main 和停止条件
- [x] 冻结 GPU-only、载荷唯一性和事务指标

## Implementation

- [x] 增加 DVM 4+4 prefix/resume 长时回归门
- [x] 增加每步释放、node 拓扑、Ptera pin 和 impulse 审计
- [x] pilot 未暴露生产缺陷，无需修改生产代码

## Pilot / Smoke

- [x] 4 步 prefix 通过
- [x] 输出有限且残差满足原门
- [x] 每步 source/solver/wake/owner 单次递增
- [x] 条件释放序列未被强制改写

## Main Run

- [x] 从 4 步 prefix 恢复至 8 步
- [x] 8/8 步全部接受
- [x] 第 9 个耗尽坐标失败且 parent 不变
- [x] GPU-only 和粒子容量检查通过（1917/4096）

## Validation

- [x] 受影响回归通过（74/74）
- [x] metrics、日志、环境与哈希完整
- [x] 声明按支持/不支持分类

## Closeout

- [x] 写出 1--2 句结论
- [x] 明确周期/论文 CASE 前的下一验证节点

结论：原 `Lcrit=0.11` 下，DVM-Q16 FSI 的 4 步 pilot 和 8 步
prefix/resume 均通过，生产代码无需修改。该工况持续 active；下一节点是
保持原阈值的自然 release off/on/restart，再进入论文复现 CASE。
