# Claim validation

## 可声明

- 正式 5×3 Q16 / 15×10 UVLM 的前 4 个 Yamano 端点 tip-z 误差均 ≤5%。
- Q16、论文 Mf1 投影、常量环量面压力、mandatory separated LEV、TEV、
  free wake 和 predictor/corrector 已在同一 CUDA FSI 通路中运行。
- 8 个外步均数值收敛且每步只提交一次真实尾迹。
- 8 步内 LEV 没有释放，因为 `|LESP|` 未达到 0.11。

## 不可声明

- 不可声明 8 点轨迹已达到 5%：末点误差为 27.385%。
- 不可声明 `t*=1.0`、多周期或整篇论文已经复现。
- 不可把零 LEV 粒子解释为 LEV 模块被关闭。
- 不可把长时误差归因于 Q16 网格或通过调 `Lcrit/E/rho` 修正。

## 下一 claim 门

实现并独立验证 `Mf2_vec1` 尾迹运动历史后，8 个端点最大误差必须 ≤5%，
同时保留现有残差、GPU-only、LEV 和尾迹事务硬门。
