# 运行摘要

本轮把 Yamano `single_sheet` 正式网格推进到 8 个外步。修复论文 Q4
局部 Mf1 和 `dp_lift1+p_interp` 常量压力装配后，前 4 个端点均进入 5%
误差门；第 5–8 点误差单调增至 27.385%，故当前结论是“短时 CASE 已
复现，8 点轨迹尚未复现”。

所有 8 步均保持 Q16-only、CUDA float64、mandatory separated LEV、joint
TEV/free wake 和原子 predictor/corrector 事务。LEV 零释放是因为 LESP
始终低于原始 `Lcrit=0.11`。下一步只修论文 `Mf2_vec1` 尾迹运动历史和
重叠流体窗口，不继续扩时或调物理参数。
