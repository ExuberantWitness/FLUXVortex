# Q16 DVM node-ribbon 4/8 步 FSI 计划

## 1. Objective

- run id：`20260822_q16_dvm_node_ribbon_fsi_8step`
- 实验层级：`auxiliary/dev`
- 选定思路：用已经通过两步门的 `dvm_node_ribbon` owner，在同一个
  Q16 predictor/corrector owner 上先推进 4 步，再从完全相同的前缀恢复并
  推进至 8 步。
- 用户核心要求：原始 `Lcrit` 下推进 4--8 步 DVM-Q16 FSI。
- 不可妥协条件：Q16 only；`Lcrit=0.11`；separated LEV 始终集成但只在
  条件越门时释放；joint TEV、free wake 和 predictor/corrector 始终启用；
  CUDA float64；无 CPU 数值回退；不以降低阈值制造释放。
- 研究问题：DVM node-ribbon 的条件释放、物质历史、自由尾迹和唯一
  Ptera 载荷 owner 能否在 8 个连续接受步内保持事务和数值不变量？
- 零假设：两步证据掩盖了更晚的 source/wake 拓扑分叉、重复载荷、
  owner 多次提交、结构耦合失败或虚功漂移。
- 备择假设：4 步前缀和恢复后的 8 步轨迹都通过，且每步只提交一次。

## 2. Baseline And Comparability

- 直接基线：`20260822_q16_dvm_node_ribbon_fsi`，20°、`Lcrit=0.11`、
  `dt=0.04` 的两步 active DVM 轨迹。
- 长时结构控制：历史 `20260822_q16_real_fsi_long_horizon` 的已声明
  Q16 长时夹具（`E=1e9 Pa`、质量比例阻尼 `20 s^-1`、结构门
  `5e-8`），但把气动 owner 从 `hirato_ring` 替换为当前 DVM。
- 气动离散：2 弦向 x 3 展向 Ptera panel，`dvm_ndiv=20`、
  `dvm_naterm=8`、`dvm_max_wake=32`、粒子容量 4096。
- 时间与耦合：`dt=0.04`，耦合门 `2e-7`，最多 64 次外迭代。
- 主指标：8/8 接受步，同一 prefix/resume 哈希链，owner/aero/source/wake
  代数每步恰增一。
- 必需指标：每步 cell 释放数、node 原始/有效拓扑、Ptera separated pin、
  粒子数、source/solver/wake step、载荷 owner、生产/诊断 impulse、耦合/
  结构/虚功残差、trajectory/result 哈希。
- 可比性边界：结构长时参数相对两步 active 基线是明确改变，因此比较
  的是数据通路和长时事务，不比较位移或载荷数值优劣。

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/test_q16_dvm_node_ribbon_fsi.py` | 两步 DVM FSI 门 | 增加 4+4 prefix/resume 长时门 | 复用真实生产数据通路 | 运行时间、粒子增长 |
| 本实验目录 | 证据包 | 保存计划、日志、metrics、manifest 和声明边界 | 可复核 | 不得把 dev gate 写成论文精度 |

只有运行暴露独立失败契约时才修改生产代码；禁止预先改阈值、物理参数或
评分门。

## 4. Execution Design

- 最小实验：同一 9 坐标 DVM solver 推进 4 个接受步。
- smoke/pilot：运行新增长时 pytest 到 4 步前缀，验证结构/气动代数、
  source/solver/wake 计数、粒子容量和全部残差。
- full run：从该前缀恢复 4 步到总计 8 步，再请求耗尽的第 9 步，验证
  失败坐标不改变已提交 parent。
- stop condition：CPU fallback、NaN/Inf、强制气动模式漂移、
  `Lcrit != 0.11`、重复 impulse、拓扑不一致、残差越门、粒子容量耗尽、
  或失败 trial 污染 parent。
- abandonment condition：只有关闭/削弱 separated flow、降低阈值、
  减少 Q16/Ptera/DVM 拓扑或放宽门限才能达到 8 步。
- 最强替代假设：物理/事务实现正确，但 8 步所需成本或结构固定点鲁棒性
  超出当前有界开发夹具。

## 5. Runtime Strategy

- pilot：4 步前缀；预计 1--3 分钟。
- main：4+4 prefix/resume；预计 2--6 分钟。
- 设备：唯一 `cuda:0`，RTX 4090 D；不使用额外 GPU。
- 安全效率手段：复用已编译 Warp/Torch kernel、同一 owner 前缀恢复、
  保持现有 GPU-PCG/GMRES；不跳过 corrector 或尾迹推进。
- 监控：约每 30--60 秒检查一步进度、GPU 错误、残差和粒子容量；若
  失败先保存最后有效前缀，再运行最小判别测试。
- 日志/输出：本目录中的 `runlog.summary.md`、`metrics.json`、
  `metrics.md`、`summary.md`、`claim_validation.md` 和 manifests。

专用 `bash_exec`、artifact/memory 接口不可用；使用本地非交互终端和
持久化文件回退，不放宽验收门。

## 6. Fallbacks And Recovery

- 4 步失败：不启动 8 步；定位第一个失败时层并验证 parent 未污染。
- 4 步通过、8 步失败：保留 4 步作为 partial，不反复放宽迭代门。
- OOM：先审计是否为泄漏或无界诊断缓存；不裁剪物理粒子来伪造通过。
- 条件释放模式改变：保留真实开/关序列，不要求每步非零释放。

## 7. Checklist Link

- `artifacts/experiment/20260822_q16_dvm_node_ribbon_fsi_8step/CHECKLIST.md`
- 下一个未完成项：实现并运行 4 步 pilot。

## 8. Revision Log

| 时间 | 变化 | 原因 | 影响 |
|---|---|---|---|
| 2026-08-22 | 冻结 4+4 prefix/resume 路线 | 用户要求推进原阈值 4--8 步 | 不改气动阈值或离散 |
| 2026-08-22 | 复用声明过的阻尼长时结构夹具 | 隔离 DVM 气动迁移与已知无阻尼结构失稳 | 不与两步基线比较载荷/位移大小 |
| 2026-08-22 | 4 步 pilot 通过，无生产修复 | 5 个气动坐标均为 3 个 cell 条件释放；第 3 坐标有 1 个 node 原始阈值与共享拓扑不同 | 保留 raw/effective 差异作为正确的三维连接诊断 |
| 2026-08-22 | 8 步主门通过 | 9 个气动坐标均自然越门；第 9 个耗尽坐标被事务性拒绝 | 支持持续 active 长时通路，不支持同轨迹 release off/on 声明 |
