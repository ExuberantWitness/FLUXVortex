# Q16 DVM node-ribbon FSI 迁移计划

## 1. 选定思路与强制边界

- run id：`20260822_q16_dvm_node_ribbon_fsi`
- 实验层级：`auxiliary/dev`
- 选定思路：将已通过 Mancini fast/slow 的 CUDA DVM node-owned
  connected-ribbon 气动 owner 接入现有 Q16 predictor/corrector 强耦合事务，
  不再以历史 `hirato_ring` 夹具作为当前 FSI 生产证据。
- 用户强制条件：Q16 only；CUDA float64 科学数值路径；无 CPU
  fallback；separated LEV 从 step zero 开启；joint TEV 和 free wake 强制开启；
  predictor 必须推进真实 DVM source、粒子前沿和自由尾迹。
- 禁止项：不关闭 LEV；不用 Q9/toy 结构；不将 vortex impulse 重复
  叠加到 DVM/Ptera 表面载荷；不调论文参数或评分门。

## 2. 问题、假设和基线

- 研究问题：最新 DVM node-ribbon 气动状态能否在每个 Q16 强耦合
  trial 中真实推进，通过唯一 `KJ+dGamma` 载荷 owner 向结构传递，
  并在接受时只提交一次？
- 零假设：旧 FSI 转移仍隐式依赖 Hirato 冲量载荷或不能序列化/
  推进 DVM source bank，因而 DVM 模式会在载荷、几何或事务门停止。
- 备择假设：零冲量通路可作为合法零长度贡献，Ptera resolved
  point loads 单独闭合总力/力矩/虚功，DVM 及两个 owner 每个接受步各前进
  一次，失败 trial 不污染 parent。
- 基线 A：Mancini node-owned mandatory GPU，fast/slow full RMSE
  `1.04886310/0.22526573`；证明气动 owner 可用。
- 基线 B：历史 Q16 Hirato 8 步事务门；只证明 FSI 框架能运行，
  不作为当前 DVM 气动科学基线。

## 3. 验收指标

1. 最小 DVM FSI 一步接受，结构/气动 generation 各增加一；
2. `separated_source == dvm_node_ribbon`、`dvm_source_bank.it == _steps_done`；
3. LEV 释放必须严格受 `abs(LESP)>Lcrit` 及 DVM source event 控制：
   未越门时允许零粒子且不得伪造释放；越门时必须生成 connected
   ribbon、进入当步 RHS 并随真实尾迹推进；
4. 每步 `load_owner == ptera_kj_plus_dgamma`；生产 impulse 为零、诊断
   impulse 保持可观测；
5. Ptera resolved point load 闭合源总力/力矩，Q16 转移保持虚功共轭；
6. coupling residual `<=2e-7`，结构 residual 有限并满足原门，功平衡
   relative residual `<=1e-6`；
7. 强制失败后 parent solver/state 哈希不变；
8. 分别验证“未越门零释放”和“自然越门激活释放”；激活工况使用
   原 `Lcrit=0.11`，通过有限翼攻角改变越门，不人为降低阈值；
9. 一步通过后运行有界两步轨迹；两步失败则保留一步证据并定位。

## 4. 预期修改

| 文件 | 目标 |
|---|---|
| `platform/warp_vpm/test_q16_dvm_node_ribbon_transaction.py` | 补 release event / separated state 独立反例 |
| `platform/warp_vpm/test_q16_incremental_trial_geometry.py` | 让共享气动夹具显式选择 source，不静默默认 Hirato |
| `platform/warp_vpm/test_q16_dvm_node_ribbon_fsi.py` | 新增最新气动 owner 的真实 Q16 FSI 门 |
| `src/fluxvortex/warp_fsi/q16_mandatory_aero_mode.py` | 如必要，增加显式生产 source 身份，不破坏历史夹具隔离 |
| `platform/warp_vpm/q16_real_fsi_coupling.py` | 仅修复 DVM owner 实测暴露的真实边界错误 |

## 5. 执行顺序与停止条件

1. 只读审计 owner/load/transaction 数据流；
2. 先写精确失败的回归门；
3. 运行 DVM 一步 FSI pilot：先证明未越门零释放合法，再用原
   `Lcrit=0.11` 的高攻角工况证明条件释放真实进入 FSI；
4. 如失败，只修证据指向的单一边界；
5. 一步通过后运行两步轨迹和受影响回归；
6. 生成 `RESULTS.md`、`metrics.json`、`runlog.summary.md` 和证据哈希。

立即停止：CPU fallback，LEV/joint TEV/free wake 任一丢失，重复 impulse
载荷复活，失败 trial 污染 parent，非有限状态，或需要改气动评分/
论文参数才能通过。

## 6. 工具偏差

`experiment` 技能要求的 `bash_exec`、artifact 和 memory 接口当前不可用；
使用本地非交互命令与本目录持久化证据作为回退，不因此放宽科学门。

## 7. 执行修订记录

- 初始 5°/`Lcrit=0.11` pilot 没有生成 LEV 粒子。根据“机制始终集成、
  释放严格有条件”的科学契约，该结果被重新分类为合法的亚临界证据，
  而不是失败。
- 曾用 `Lcrit=0.001` 检查 active 数据通路；该运行仅是诊断性探针，
  因改变释放阈值，不进入验收证据。
- 正式 active 夹具改为 20° 有限翼、原始 `Lcrit=0.11`。它在初始化和
  两个接受步均自然越门，证明条件释放进入同一步 DVM/Ptera/FSI 通路。
- 第一版两步 active 轨迹在第二个 FSI 步暴露 node/cell 活性不一致。
  根因是 node lane 与 cell lane 分别用 LESP 阈值决定物质释放，违反了
  三维 connected-ribbon 的共享拓扑。
- 修复后由展向 cell lane 唯一拥有释放事件；node lane 保留原始阈值
  结果用于诊断，但其物质历史和连接拓扑投影为相邻 cell 事件并集。
  这消除了虚假节点释放，同时没有把亚临界状态强制改成 active。
