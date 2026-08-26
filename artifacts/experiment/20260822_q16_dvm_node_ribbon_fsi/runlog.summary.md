# 运行日志摘要

## 运行序列

1. 审计发现现有 Q16 FSI 夹具默认使用历史 `hirato_ring`，不能证明最新
   DVM node-ribbon owner 已集成。
2. 5°、原 `Lcrit=0.11` 一步 pilot 收敛且零释放；按条件释放契约判定
   为合法亚临界结果。
3. `Lcrit=0.001` active 探针通过，但因改变阈值被排除出验收。
4. 20°、原 `Lcrit=0.11` 一步 active pilot 通过。
5. 首次两步 active 轨迹在第二步触发
   `node-local DVM activity differs from adjacent cell union`。
6. 将物质释放 owner 统一为 ribbon cell，并把 node 有效历史投影为相邻
   cell 事件并集；保留 node 原始阈值作诊断。
7. 两步 active 轨迹通过；结果 SHA-256 为
   `6bc39686b1ce03e2c2b3f76347cfaecb12ed9ee0db8b621b0424f5176f51b36a`，
   trajectory chain SHA-256 为
   `b5444d3e399ef2cd022c126a4faba66d7768f2fb5804699a8237754ae9fec519`。
8. 受影响 GPU-only 回归最终为 73/73 通过，63.99 s；Ruff 和
   `py_compile` 通过。

## 最终执行条件

- GPU：NVIDIA GeForce RTX 4090 D（driver 580.173.02）
- 数值路径：CUDA float64，无 CPU 数值回退
- active fixture：20°，`Lcrit=0.11`，`dt=0.04`，两个接受步
- 计时：证据轨迹 9.9754 s；受影响回归 63.99 s
- 未访问论文评分器或 GT；未运行完整论文 CASE/正式验证矩阵
