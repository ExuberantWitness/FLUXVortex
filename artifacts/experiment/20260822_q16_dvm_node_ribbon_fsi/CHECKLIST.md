# Q16 DVM node-ribbon FSI 执行清单

状态：`PASS (bounded dev gate)`

- [x] 冻结 Q16/GPU-only/separated-LEV/joint-TEV/free-wake 边界
- [x] 确认历史 8 步 FSI 夹具仍默认 `hirato_ring`
- [x] 确认当前 Mancini 生产气动 owner 为 `dvm_node_ribbon`
- [x] 冻结 owner/time-layer/load 验收指标
- [x] 纠正验收语义：机制强制集成不等于每步强制释放
- [x] 默认 `Lcrit` 未越门 DVM-FSI 一步通过，零粒子判为合法
- [x] 补 release event / separated state 独立反例
- [x] 用原 `Lcrit=0.11` 的自然越门工况跑 active DVM 一步 Q16 FSI pilot
- [x] 根据 pilot 修复 cell-owned/node-shared 释放拓扑边界
- [x] 跑 DVM 两步轨迹
- [x] 跑受影响 GPU-only 回归（73/73）
- [x] 生成 metrics/结果/运行日志/声明边界
- [x] 明确下一 FSI 验证节点

下一节点：在不改变 `Lcrit` 的前提下运行 4--8 步 DVM Q16 FSI，要求
条件释放可自然开/关、每个接受步只提交一次 source/wake/structure owner，
并持续满足载荷唯一性、虚功与事务门；随后才进入周期或论文 CASE。
