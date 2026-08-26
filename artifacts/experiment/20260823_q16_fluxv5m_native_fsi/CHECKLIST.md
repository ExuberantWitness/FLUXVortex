# 执行检查表

- [x] 冻结正式 CASE、Q16/V5M 网格和禁止依赖范围
- [x] N0：生产依赖纯度门
- [x] N1：作者 AIC oracle 方向/顺序门
- [x] N2：CUDA float64 状态所有权门
- [x] N3：原始 Lcrit=0.11 LEV 释放门
- [x] N4：LEV/TEV/free-wake trial/commit 事务门
- [x] N5a：V5M 面板载荷直接 Q16 力、矩、虚功传递门
- [x] N5b：`Mf2_vec1` step1–4 逐面板 CUDA oracle 门
- [x] N5c：`Qf_p_global` step1–4 合力 CUDA oracle 门
- [x] N5d-hard：直接 Q16 `Mf1` 有限、非零且厚度向加速度功为负
- [x] N5d-warning：记录跨 9-DOF/6-DOF 离散共同场差异 37.35%，不作为轨迹硬阻断
- [x] N6a：正式网格 1 步 CASE（端点误差 1.425%，残差 1.60e-8，4 次耦合）
- [x] N6b：正式网格 2 步 CASE（端点误差 1.757%，残差 1.85e-8）
- [x] N6c：正式网格 4 步 CASE（端点误差 7.633%，残差 9.31e-9）
- [x] N6d-execution：正式网格 8 步 CASE 完整执行（退出码 0，764.80 s）
- [ ] N6d-accuracy：8 步端点误差降至 5% 以内（当前 27.918%，未通过）
- [ ] N7a：生成作者正式 `t*<=0.55` 完整状态轨迹并校验端点位移
- [ ] N7b：作者 9-DOF 曲面直接采样到 Q16 5×3 状态（无 Q4/Q9 运行时）
- [ ] N7c：CUDA float64 比较作者状态与当前FSI状态的 Q16 恢复力/能量
- [ ] N7d：按独立 oracle 修复首个根因并重跑正式 1/4/8 步 CASE

## 当前证据

- 旧 Ptera/Q4 一步/八步结果：仅作迁移前参考，不计入本检查表。
- MATLAB Mf1/Mf2 与 AIC fixture：允许作为只读离线 oracle。
- 正式载荷门：`YAMANO_Q16_NATIVE_LOAD_GATES.json`；硬门与诊断警告已分离，`mf1_common_field` 只保留为 warning。
- 正式 CASE 将输出每次气动提案、34 个结构子步和耦合残差，真实错误出现时可直接定位。
- 旧 `YAMANO_Q16_5X3_V5M_15X10_STEP1.json` 耗时 655.96 秒、端点误差 15.81%，因缺少载荷门而判定为无效旧证据。
- 当前作者载荷重跑：`YAMANO_Q16_5X3_V5M_15X10_STEP1_AUTHOR_LOAD_RERUN.json`，退出码 0；下一步直接运行 8 步并从同一轨迹读取 2/4/8 检查点。
- 正式八步结果：`YAMANO_Q16_5X3_V5M_15X10_STEP8_AUTHOR_LOAD.json`，状态 `completed`。1/2/3 步误差分别为 1.425%/1.757%/3.685%，脉冲结束后误差持续放大，8 步为 27.918%。
- 8 步所有耦合残差均低于 2e-8，结构实时残差门保持 3e-7；因此长时程偏差不是通过放宽数值门换来的。
- LESP 最大值 0.01787，始终低于固定 `Lcrit=0.11`；LEV 模块始终集成，但该工况前 8 步物理释放次数为 0。
