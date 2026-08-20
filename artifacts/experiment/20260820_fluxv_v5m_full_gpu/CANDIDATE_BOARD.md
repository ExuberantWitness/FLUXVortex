# FLUX-V5M GPU 优化候选板

| ID | 层级 | 策略 | 状态 | 预期收益 | 当前结论 |
|---|---|---|---|---|---|
| C1 | implementation | exploit | completed | 环涡核心约 3×；载荷速度重复计算约 4→1 | 回归与论文矩阵通过 |
| C2 | implementation | exploit | completed | 动态形状融合降低 Python launch 开销 | warm 迭代加速；冷编译单列 |
| C3 | brief | fusion | queued | 合并 particle LESP/impulse/ledger 微内核 | 需 Triton/Warp 2D reduction 设计 |
| C4 | brief | explore | hold | 多工况 batched AIC/solve/load 显著提高占用 | 架构收益大，改动面也最大 |
| C5 | implementation | explore | archived | 多线程 + 多 CUDA stream | 实测 active 1.56→1.66 s、attached 4.36→4.84 s，负收益 |
| C6 | implementation | exploit | completed | 结构 CG 融合 dot/max/axpy | kernel −48.6%，同步 −83.3% |

选择结论：当前主线是 C1+C2+C6。C5 已实测为负收益，不会为了“看起来更并行”而
强行并发多个单卡求解器；后续 C4 必须实现真正的 case 维张量批处理，才值得重新评估。
