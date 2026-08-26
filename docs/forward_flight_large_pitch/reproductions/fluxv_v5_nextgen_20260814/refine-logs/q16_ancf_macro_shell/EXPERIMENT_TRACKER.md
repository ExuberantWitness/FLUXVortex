# FLUX-V5M Q16 ANCF 宏壳实验跟踪表

**冻结选择**：唯一新生产单元为 Q16；Q9 不实现；separated LEV 与真实自由尾迹在全部正式 FSI/co-design 中强制开启；科学数据面为 CUDA float64。

| Run ID | 里程碑 | 目的 | 系统/变体 | 指标 | 优先级 | 状态 | STOP/GO 备注 |
|---|---|---|---|---|---|---|---|
| Q16-H0-001 | H0 | 冻结 Q16 节点序、96 DOF、积分、ANS/EAS 与设计 schema | Q16 合同 | schema/hash/exact API | MUST | PARTIAL | 节点序/96 DOF/6×6×3、MITC16 tying、E33 nodal ANS + thickness-linear EAS 已冻结并实现；宏材料设计 schema 尚未冻结 |
| Q16-H0-002 | H0 | 冻结 GPU/LEV/wake 生产边界 | Q16 FSI config | CUDA float64、LEV/wake 无关闭入口 | MUST | PARTIAL | 新 CUDA 核与事务边界已锁；真实联合 solver 入口尚未建立 |
| Q16-H1-001 | H1 | 形函数与导数恒等式 | 单 Q16 | unity/Kronecker/derivative ≤64eps | MUST | PASS | Q16-only focused + joint 已通过 |
| Q16-H1-002 | H1 | 三次场和刚体运动再现 | 单 Q16 | displacement/strain/energy | MUST | PARTIAL | 三次场、刚体 strain/energy/force 与 2×1 shared-node 装配已过；仍缺 2×2 场再现 |
| Q16-H1-003 | H1 | 膜、弯曲、剪切、厚度压缩 patch | 单 Q16/2×2 Q16 | patch residual | MUST | PARTIAL | MITC16 tying/刚体门通过；寄生剪切抑制 96.54×；E33 ANS+EAS 局部凝聚与四厚度比有限性通过，仍缺网格/厚度收敛证据 |
| Q16-H1-004 | H1 | energy-force 与 tangent 一致性 | 随机畸变 Q16 | directional derivative、Jv | MUST | PASS | energy 相对误差 2.11e-10；Jv 相对误差 1.41e-10 |
| Q16-H1-005 | H1 | 质量、质心、刚体惯量 | 单 Q16/2×2 Q16 | exact mass/moment | MUST | PARTIAL | 一致质量/平移总质量已过；质心/惯量/2×2 尚未运行 |
| Q16-H1-006 | H1 | CUDA 与独立 Q16 oracle 一致 | small mesh | force/mass/Jv ≤1e-10 | MUST | PASS | 2×1/2×2 shared-node mesh 已实现；MITC16+ANS/EAS 凝聚路径已接入确定性 CSR gather；projected force/mass/Jv 与 CPU oracle 通过 |
| Q16-H1-007 | H1 | 唯一边界约束 owner | clamped 2×1 Q16 | state/projector/reaction/CUDA | MUST | PASS | 7 个唯一根部节点/42 DOF；CPU/CUDA 互补投影、只读状态、host/float32/nonfinite 负门通过 |
| Q16-H2-001 | H2 | 薄/中厚方板法向压力 | h/L 四档 | displacement/energy/shear split | MUST | TODO | 锁死趋势 → STOP |
| Q16-H2-002 | H2 | 悬臂板法向载荷与横向剪切 | Q16 h-refinement | tip deflection/rotation | MUST | TODO | final error ≤2% |
| Q16-H2-003 | H2 | 大转动扭曲壳 | Q16 | energy/reaction/finite | MUST | TODO | 非有限或 path-dependent fail-open → STOP |
| Q16-H2-004 | H2 | 模态验证 | Q16 | first 6 frequencies ≤3% | MUST | TODO | 质量/刚度 owner 单独诊断 |
| Q16-H2-005 | H2 | 过积分敏感性 | 6×6×3 vs 8×8×4 | headline change <0.5% | MUST | TODO | 差异过大则修积分/锁死控制 |
| Q16-H2-006 | H2 | 结构非线性求解失败闭合 | Newton–Krylov | residual/iteration/rollback | MUST | PARTIAL_PASS | GPU Newmark–Newton–CG 单步、小载荷、max-iter failure/clean retry 已过；仍缺长时、多网格、预条件与线搜索门 |
| Q16-H3-001 | H3 | 任意 aero point 的 Q16 几何映射 | distorted Q16 | position/velocity/Jacobian | MUST | PARTIAL_PASS | 任意点 CPU/CUDA 映射已过；尚未接真实 VLM 网格 owner |
| Q16-H3-002 | H3 | 功共轭载荷传递 | random q/f/δq | virtual work ≤1e-11 | MUST | PASS | exact transpose；equal-four-node 负控失败；CUDA 对 oracle 1e-12 |
| Q16-H3-003 | H3 | 总力/总矩守恒 | pressure fields | normalized residual ≤1e-11 | MUST | PASS | director lever arm 已计入，注册 force/moment gate 通过 |
| Q16-H3-004 | H3 | separated LEV 强制启用 | separated maneuver | nonzero LEV release/feedback | MUST | TODO | off/zero-owner 入口首步前拒绝 |
| Q16-H3-005 | H3 | predictor 真实尾迹事务 | injected failure/retry | commit=0/1、hash、fresh parity | MUST | PARTIAL | 通用 LEV/TEV/wake oracle 事务已过；尚未绑定真实 CudaJointLEVTEVSolver |
| Q16-H3-006 | H3 | CUDA-only 运行时门 | representative FSI | host calls/device/dtype | MUST | PARTIAL | Q16 continuum/transfer 严格 CUDA float64；完整联合时间循环未建立 |
| Q16-H3-007 | H3 | GPU scaling | batch 1/8/32/128 | throughput/memory/Nsight | MUST | TODO | 未达标先 profile，不降阶/关 LEV |
| Q16-H4-001 | H4 | 刚性极限 | stiff Q16 + V5M | CL/CD/CM parity | MUST | TODO | 同气动网格比较 |
| Q16-H4-002 | H4 | 结构-only 法向压力系统回归 | Q16 wing | displacement/reaction | MUST | TODO | 隔离气动误差 |
| Q16-H4-003 | H4 | 柔性阵风/机动收敛 | Q16+LEV+free wake | loads/deflection/PC residual | MUST | TODO | 两级加密变化 <3% |
| Q16-H4-004 | H4 | 经典柔性翼实验验证 | frozen case | aero/struct metrics | MUST | BLOCKED | 先冻结公开几何/材料/运动/GT |
| Q16-H5-001 | H5 | 宏材料映射正定与质量闭合 | random Q16 design fields | SPD/mass/inertia | MUST | TODO | 任一非物理组合首核前拒绝 |
| Q16-H5-002 | H5 | 端到端设计梯度 | 8 directions | median ≤1e-4, max ≤5e-4 | MUST | TODO | 未过不得优化 |
| Q16-H5-003 | H5 | optimizer failure transaction | injected failure | design/archive/state rollback | MUST | TODO | 坏候选不得污染 Pareto archive |
| Q16-H6-001 | H6 | 均匀 Q16 基线 | 3 initializations | equal-mass objectives | MUST | BLOCKED | H1–H5 全 PASS 后运行 |
| Q16-H6-002 | H6 | 仅展向 Q16 基线 | 3 initializations | Pareto/hypervolume | MUST | BLOCKED | 同单元、同物理、同预算 |
| Q16-H6-003 | H6 | 完整二维 Q16 宏场 | 3 initializations | Pareto/hypervolume | MUST | BLOCKED | ≥5%/≤2% 预注册门 |
| Q16-H6-004 | H6 | fresh forward 复算 Pareto 候选 | detached evaluator | objective/constraint exactness | MUST | BLOCKED | 禁止复用 optimizer cache 自证 |
| Q16-H7-001 | H7 | fresh integrity audit | all artifacts | hashes/schema/counts/claims | MUST | BLOCKED | reviewer PASS 才解锁声明 |

## 执行纪律

1. 严格按 H0→H7；任一 MUST 项失败即停止下游。
2. 每个修复后都必须运行本行 focused test、所有既有 Q16 tests、Black/Ruff/py_compile/diff-check。
3. 正式 FSI/co-design 输出必须记录 CUDA device、dtype、Q16 mesh、aero mesh、LEV/wake 状态、PC/CG ledger、transfer residual 和输入文件 SHA-256。
4. Q9 相关文件、配置、测试或结果一旦出现，视为合同漂移并停止晋级。
5. attached-only、wake-off 和 CPU 结果不得用于替代任何 MUST 项。
