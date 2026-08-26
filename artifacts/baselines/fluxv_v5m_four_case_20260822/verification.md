# 当前复现基线核验

日期：2026-08-22

## 判定

当前状态为 `baik_and_mancini_mandatory_pass_yang_izraelevitz_pending`，不能沿用
“当前四篇已验证”的说法；Baik W1--W4 与 Mancini fast/slow 已形成本次 fresh、
可重复的 mandatory GPU 结果。

原因不是旧指标丢失，而是模型合同后来收紧：历史三维四论文 runner 的 Ptera
底盘使用 `enable_lev=False` 和 `prescribed_wake=True`；当前生产模型要求
separated LEV、joint TEV、free wake 同时开启。因此历史指标只能作为迁移前
reference，必须由新生产入口重新计算后才能形成当前 CASE 结果。

## 已验证

- 生产入口对 LEV-off、joint-TEV-off、prescribed-wake 均在首步前 fail-close；
- 6 步 CUDA float64 冒烟完成：`steps_done=6`、LEV 粒子 `60`、粒子推进
  `4` 次、free-wake convection `5` 次、joint TEV 非空；
- 没有 CPU 气动 oracle 参与该冒烟；CPU 仅构造 Ptera 几何/控制对象；
- 三周期 Q16 长测已暂停并改为显式 opt-in，不进入默认回归；
- 周期探索时临时加入的 LEV 粒子 core-spreading 默认改动已撤出，避免在
  未经四 CASE 复验前改变气动数值。

## 历史 reference（不可冒充当前结果）

| CASE | 历史指标 |
|---|---:|
| Yang lift / drag MAE | `4.1048266 / 1.5186256 gf` |
| Izraelevitz–Scherer CT MAE | `0.01745211` |
| Baik CL / CD macro RMSE | `0.42156277 / 0.28970979` |
| Mancini fast / slow CL RMSE | `1.25531975 / 0.29509478` |

## 下一锚点

保持 GT、几何、运动和评分代码不变，将同一 node-owned mandatory 生产合同迁移
到 Yang 与 Izraelevitz；先执行代表性 smoke，过门后才启动完整工况。Baik 继续
使用自身 CUDA LDVM 的 LEV/TEV wake 生产路径，不转换成三维 Ptera surrogate。

## 2026-08-22 mandatory Mancini 实测

| 工况/网格 | RMSE(CL) | 冻结 reference | 判定 |
|---|---:|---:|---|
| fast smoke，2×6、24 step/c | `6.18864881` | `1.25531975` | FAIL |
| fast full，4×12、64 step/c | `316.16452568` | `1.25531975` | FAIL |
| slow smoke，2×6、24 step/c | `4.25876768` | `0.29509478` | FAIL |

fast full 完成 `449/449` 步、自由尾迹推进 `448` 次、最终 LEV 粒子 `30072`，
CUDA 长时执行本身有效；但预测峰值 `CL=3043.80@t*=4.70`，实验峰值仅
`4.83@t*=0.96`。`smoke -> full` 的恶化不是随机失败：既有
`V5F_M5_REFINEMENT_STOP_REPORT.md` 已证明相同常强度闭合环表示的材料释放随
`1/dt` 增长。本轮不再运行 slow full，也不以调阈值/core/网格修补。

后续验收改为：先把已有 v5h node-owned 连通 ribbon 源接入 CUDA Ptera 数据面，
通过 `Gamma_birth=O(dt)`、共享展向节点、Kelvin、单一载荷 owner 和 fast smoke
非退化门；这些机械门通过后才恢复论文 full 队列。

## 2026-08-22 mandatory Baik fresh 复算

| 工况 | CL RMSE | CD RMSE | LEV 释放次数 | 尾迹推进步 | 判定 |
|---|---:|---:|---:|---:|---|
| W1 | `0.38142176` | `0.17628873` | `517` | `1536` | PASS |
| W2 | `0.42244782` | `0.35308252` | `524` | `1536` | PASS |
| W3 | `0.38539542` | `0.21722954` | `479` | `1536` | PASS |
| W4 | `0.49698608` | `0.41223839` | `484` | `1536` | PASS |

宏平均为 CL `0.4215627678`、CD `0.2897097942`，与冻结 GPU reference 位级
一致。四个 CASE 均保留 `256` 个 TEV 历史点；最终证据运行监测到 GPU 峰值
利用率 `100%`、进程级峰值显存 `4669 MiB`；summary 总门为 `PASS`。证据位于
`results/baik/summary.json`，生产入口为
`platform/warp_vpm/reproduce_baik_v5m_mandatory.py`。

## 三维 node-owned 迁移门进度

- CUDA source-only：2/2 测试通过，首步 TEV 求解值非零但持久化值严格为零，
  第二步恢复求解值持久化；LESP pin 残差小于 `2e-15`。
- CUDA connected ribbon：3/3 测试通过；相邻同强度 cell 的重复内边在沉积前
  消去，非均匀强度只保留一条有符号内边，`seam_count=0`，全局向量矩闭合。
- 展向批处理 CUDA source bank、节点前沿、connected ribbon、Ptera RHS/load、
  Eq. 9 TE row、free wake 和 predictor fork 已接通；monolithic/incremental 与
  predictor 分支保持一致，受影响回归 `41 passed`。
- DVM 物理模式由 Ptera `KJ+dGamma` 唯一拥有载荷；诊断 vortex impulse 不再
  二次叠加。释放事件与分离边界状态分开，避免 source-off 单步把 Ptera 从
  `LESP=0.11` 错切到约 `0.65` 后再切回。
- Mancini 已通过；该结论不自动外推到 Yang、Izraelevitz。

## 2026-08-22 node-owned mandatory Mancini 最终结果

| 工况/网格 | RMSE(CL) | 冻结门 | 粒子数 | free-wake 步 | 判定 |
|---|---:|---:|---:|---:|---|
| fast smoke，2×6、24 step/c | `0.84303058` | `1.25531975` | `25911` | `168` | PASS |
| fast full，4×12、64 step/c | `1.04886310` | `1.25531975` | `76312` | `448` | PASS |
| slow smoke，2×6、24 step/c | `0.21476298` | `0.29509478` | `21792` | `168` | PASS |
| slow full，4×12、64 step/c | `0.22526573` | `0.29509478` | `62521` | `448` | PASS |

两个 full 均使用 CUDA float64、separated LEV、joint TEV、free wake、DVM
node-owned connected ribbon 和唯一 Ptera surface-load owner；没有 attached/
prescribed fallback，也没有 post-hoc LDVM load delta。fast/slow full 的最大
LESP pin 残差为 `3.153e-14/1.955e-14`，保留 Neumann 行最大残差为
`4.663e-15/2.887e-15`。墙钟分别 `2810.92 s` 与 `1123.51 s`；精度门通过，
但直接粒子自诱导仍是明确性能瓶颈。
