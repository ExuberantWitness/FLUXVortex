# Q16 DVM node-ribbon 8 步 FSI 运行记录

## 1. 研究问题

原 `Lcrit=0.11` 下，当前 DVM node-ribbon 气动 owner 能否在同一 Q16
predictor/corrector 强耦合 owner 上连续提交 8 步，并保持条件释放、自由
尾迹、唯一载荷和事务不变量？

## 2. 研究类型

`auxiliary/dev` 长时集成与系统契约验证。不是论文精度试验。

## 3. 目标与成功标准

先完成 4 步前缀，再从同一哈希链恢复 4 步；要求 8/8 步接受，
source/solver/wake/owner 每步只前进一次，耦合残差不超过 `2e-7`，结构
残差不超过 `5e-8`，虚功相对残差不超过 `1e-6`。生产 impulse 必须为
零，第 9 个耗尽坐标必须失败且不污染 parent。

## 4. 实验设置

- Q16 MITC16/EAS 单宏单元，20° 初始俯仰；
- Ptera 2 弦向 x 3 展向 panel；
- `dvm_node_ribbon`，`Lcrit=0.11`，`ndiv=20`，`naterm=8`，
  `max_wake=32`，粒子容量 4096；
- separated LEV、joint TEV、free wake 从 step zero 开启；
- `dt=0.04 s`，`E=1e9 Pa`，质量比例阻尼 `20 s^-1`；
- CUDA float64，RTX 4090 D，无 CPU 数值回退；
- 耦合/结构门 `2e-7/5e-8`，外迭代上限 64。

结构刚度和阻尼取自已声明的历史 Q16 八步开发夹具，目的是隔离最新 DVM
气动 owner 的长时迁移；因此不把本次位移或载荷与两步无阻尼夹具作数值
优劣比较。

## 5. 实验结果

4 步 pilot 和独立 8 步主门都通过。主证据运行耗时 57.8235 s，8 个接受
步均为 4 次耦合迭代和 5 次气动评估。LEV 粒子从初始化的 206 增至 1917；
DVM source/solver step 均为 9，自由尾迹和 frontier 均推进 8 次。

9 个气动坐标中三个 cell 均自然满足 `|LESP|>0.11`。第 3 个气动坐标
出现一次 node raw/effective 差异，证明 cell-owned/node-shared 投影在
长轨迹中实际生效。所有时层载荷 owner 均为 `ptera_kj_plus_dgamma`；
生产 impulse 始终为零，诊断 impulse 保持非零。

第 9 个耗尽坐标以 `Q16IncrementalAeroLifecycleError` 停止，已提交八步
parent 的结构哈希、solver 哈希、对象身份及 owner 代数均不变。

## 6. 分析

结果支持“持续 active 条件下，DVM node-ribbon 已真实进入 Q16 长时强耦合
事务”。它没有要求每步强制释放；本工况恰好在全部 9 个气动坐标自然
越门。由于没有出现 cell release off/on，本次不能证明同一轨迹内的释放
关闭、重启和 frontier restart 逻辑。

结构在第 4 步后使用了声明过的非定切线 GMRES fallback，但结构残差始终
低于原门；这不是气动失败。粒子容量占用 46.8%，当前八步没有容量风险。

## 7. 结论与决策

八步 DVM-Q16 FSI 开发门通过，结论从“两步可执行”提升到“八步持续
active 事务稳定”。下一步不应继续堆叠静态 active 步数；应在保持
`Lcrit=0.11` 的情况下采用已声明的时间变化运动，使 cell release 自然
关闭并重启，然后再把同一通路放入论文复现 CASE。
