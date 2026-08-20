# FLUX-V5M GPU 吞吐优化检查表

日期：2026-08-20

- [x] 用 Nsight 建立 kernel-launch 基线
- [x] 明确主瓶颈为小张量/归约微内核碎片与重复环涡求值
- [x] 建立候选优化板
- [x] 选择首个主线：四涡腿批量化 + 四组载荷点批量化
- [x] 复跑数值一致性与吞吐基准
- [x] 复跑 Nsight，比较 launches/step 与 GPU kernel time
- [x] 评估形状动态 `torch.compile` 的冷/热收益
- [x] 评估多 stream 线程并发并归档负结果
- [x] 更新四论文端到端 wall-time、显存与加速比
- [x] 独立性能/完整性审计（PASS with WARNINGS）

当前路线：`exploit / batching-and-fusion`

基线：

- active/joint/free 三个小工况：75,886 CUDA launches / 约 30 个时间步
- 环涡 96 targets × 96 rings：0.736 ms/call
- 四个 active 20-step 工况顺序执行：1.560 s
- 四个 attached 64-step 工况顺序执行：4.357 s

最终候选：

- 四涡腿与四组载荷点批量化；总诱导速度在融合图内直接归约；
- active-LEV 20-step warm：约 0.128 s（基线单工况约 0.390 s）；
- 38-step 纯生产 warm profile：35,575 → 21,725 kernels，GPU kernel
  time 57.16 → 35.97 ms；
- Warp-FSI：78,758 → 40,517 kernels，host synchronize 7,201 → 1,204，
  GPU kernel time 1.760 → 0.414 s；
- 三篇 3D 论文全矩阵：582.98 → 443.01 s（1.316×），峰值显存
  11,104 → 6,494 MiB；
- 首次融合编译约 2–3.4 s，后续同进程参数扫描复用缓存；短小一次性诊断可用
  `FLUXV_V5M_FUSE=0`，正式多工况迭代保持默认开启。
