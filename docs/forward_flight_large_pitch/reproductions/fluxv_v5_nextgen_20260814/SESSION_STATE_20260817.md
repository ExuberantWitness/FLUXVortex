# 会话状态快照（2026-08-17 收盘）— 新会话从这里接续

## 所有者现行指令（最高优先级）
**用 V5H15 最新算法直接复现三论文并报结果**（Baik 2012 W1–W4 / Izraelevitz
Fig.14-Scherer / Yang 2025；Ramesh、Hirato、Izraelevitz Fig.11 不入评分）。
执行计划：refine-logs/v5h15_birth_sigma_scheduling/PAPER_CASE_EXECUTION_PLAN_20260817.md
（含三论文几何/运动学/量纲/GT/指标全清单与顺序 W2→Izraelevitz→Yang）。

## 算法基线（已冻结）
- V5H15 四叶（graded k=5/r=4 格栅 + κ=1.75 出生 σ=0.00266 + invariant-
  reconstructed WRK3 rVPM + 冻结 Ptera 父场），token:
  /tmp/fluxv-v5h15-audit-20260817-ZHUMgt/（209 modules verified）
- 全周期 runner 骨架已验证：/tmp/v5h14-probe/driverC.py（捕获机制）

## 谱系成果链（全部已归档）
- V5H12：执行协议修复 ✅（refine-logs/v5h12_execution_repair/）
- V5H13：分级格栅机理 ✅（预测 0.1243 精确命中；k=12 被证伪）
- V5H15：**首个完整矩阵** ✅（2421 stages、9/9 预测命中、载荷门全过、
  3 阶干净收敛；formal A 工件 /tmp/fluxv-v5h15-formal-A-20260817）
- 遗留（非阻塞论文诊断）：inner 状态通道容差 5/27 未过（真实 6.3e-5@三阶，
  归一化 amendment 已被预检证伪；载荷通道级收敛 5–6 位）；B4 outer gate
  无实现。选项 a/b/c 待所有者定（refine-logs/v5h15_.../
  AMENDMENT_PROPOSAL_CONVERGENCE_NORMALIZATION_20260817.md §8）。

## 治理红线（DiGT-1，必须遵守）
论文 GT 已被所有者诊断性解封（OWNER_AUTHORIZED_DIAGNOSTIC_UNLOCK_20260817.md）：
仅 DIAGNOSTIC 用途；此后任何方法/参数/容差决策不得依论文结果选择；
正式 formal 复现声明仍需原门链。

## 首批诊断数字（W2 启动瞬态，3 release）
CL/CD：L1 (+0.294/−0.073)、L2 (−1.550/+0.367)、L3 (−7.231/+0.448)；
N47/79/143 载荷一致 5–6 位。周期态需全周期 runner（下一步工程）。

## 下一会话动作
1. 写 Baik W2 全周期 runner（复用 driverC 骨架；32 Ptera 步/周期 × 2–3 周期
   @N143；1 Hz 滤波；对 GT 400 相位点 RMSE/MAE）；
2. 报 W2 结果 → Izraelevitz 12 条件 → Yang 6 条件；
3. 全程 DIAGNOSTIC 标注。

## 追加（2026-08-17 深夜，文献架构定案）
- W2 诊断修正后基线：CL RMSE 1.26/corr 0.97（符号修正）；
  零 release 对照证明现架构粒子对载荷贡献=0（单向耦合）。
- 完整架构设计已定稿：HYBRID_UVLM_VPM_DESIGN_20260817.md（buffer 面元
  +ΔΓ 转化+RHS 反馈，Proulx-Cabana 2022/2024 + arXiv 2511.11430）。
- 下一动作：按设计文档实现（开放项 1-3 先核对），验收线 CL RMSE < 1.26。

## 追加 2（实现蓝图定稿）
开放项 1 已解决（注入点/框架/求解式全核实，见设计文档"开放项核对结果"）；
wake 结构已核实（gridWrvp 点格 + strengths + ages）。下一会话按设计文档
"实现蓝图"直接编码 HybridSolver scratch（会话 1），参数 N_buf∈{2,4}。

## 追加 3（新任务，所有者 2026-08-17 深夜）
所有者质疑 DVM（LDVM）实现正确性，要求跨工况性能曲线。资源：
- 原始 Fortran：AUTORESEARCH/FLUXV/docs/.../literature/ramesh_ldvm_v2_5_source.zip
  （SHA fd574792…；ldvm.f95 审计 SHA eada8df2…；含作者可执行文件/参考输出
  force_pr_amp45_k0.2_le.dat/input/motion/sd7003 坐标）
- 审计报告：.../unified_fluxv_v4_ldvm_stevens_20260812/SOURCE_AUDIT_RAMESH_LDVM.md
- Python clean-room：platform/ldvm_fourier.py；行为测试 test_ramesh_ldvm_reference.py
任务：(1) 用作者输入(motion_pr_amp45_k0.2.dat)运行 Fortran 可执行文件取参考
输出，与 force_pr_amp45_k0.2_le.dat 对齐；(2) Python 实现同工况对比（CL/CD/
CM 相位曲线 + 环量/LEV/TEV 事件时刻），定位偏差；(3) 跨工况扫描
（amp/k 网格）绘制 Python-vs-Fortran 性能曲线；(4) 若 DVM 有 bug，其影响
经 v5h_dvm_source（birth 事件）传导至整个 V5H 链——修复后重跑 W2 诊断。
执行顺序：先 (1)(2) 对单点工况，再扩展。

## Warp VPM Session 1 完成（2026-08-18）
- Warp 1.14.0 + RTX 4090D；pfield.py（ParticleField 超级字典 SoA + 容量缓冲
  + ID 类型生命周期）+ blob_velocity_kernel（全 float64 显式常量）。
- **数值验证：与冻结核 rel_diff = 5.5e-16（机器精度 PASS）**
- **性能：10k=0.07s / 50k=0.69s / 100k=2.75s（~3.6 GFLOP-pairs/s）**
- 架构继承用户手写代码设计思想（超级字典/ID 类型/容量缓冲），Warp 替代 CuPy。
- 下一步（Session 2）：RK3 时间推进 + LEV/TEV 脱落 + RHS 反馈 → W2 全周期首跑。

## Warp VPM Session 2 完成（2026-08-18）
- WarpHybridSolver（RK3 + TEV 面元转化 + 守恒合并 + RHS 反馈 + Warp 内核）
- **W2 全周期求解：52.8 秒**（vs CPU 2.7h → **180×加速**；vs torch GPU 10min → **11×加速**）
- 最终粒子 57,223；CL RMSE=1.570/corr 0.898；CD RMSE=1.549/corr 0.927
- LEV 脱落已框架但未激活（panel API 不匹配，后续接 LDVM 事件）
- **速度革命性**：全三论文 22 case 现在仅需 ~22 分钟
- 下一步：LEV 正确接入 + 三论文全量复现

## Session 2-3 总结（2026-08-18）

### 已完成
1. **Warp VPM 内核**（Session 1）：与冻结核 5.5e-16 一致，100k=2.75s
2. **WarpHybridSolver**（Session 2）：52.8s/case（180×加速），CL 1.57/CD 1.55
3. **LEV 尝试**（Session 3）：简化运动学 LEV 已接入但改善微弱

### 核心发现
- **面元→粒子转化引入的数值扩散 > 粒子物理收益**：
  裸核（面元 prescribed wake）CL 1.259 > 混合（粒子尾迹）CL 1.570
- **正确架构应为加法**：保留面元 TEV（精确）+ 仅添加粒子 LEV（面元做不到的）
- LEV 简化模型太弱（~150 粒子，环量小）——需接入完整 LDVM LESP 事件

### 三论文复现路径（已清晰）
| 论文 | 就绪度 | 方法 |
|---|---|---|
| Baik W1-W4 | 运动学全通用 | 裸核 prescribed wake（最优精度）|
| Izraelevitz | movement+GT 就绪 | 同上 |
| Yang | 四连杆+GT 就绪 | 同上 |

### 下一会话行动
1. **修正加法架构**：禁用面元转化（`pass` 替换确认生效），仅添加 LEV 粒子
2. **接入完整 LDVM LESP**：用 DVMSourceEvent 的实际 gamma_lev 值
3. **跑三论文**：22 case 用裸核（~20min each）或修正后混合（52.8s each）
