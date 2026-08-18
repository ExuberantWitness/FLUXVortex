# W2 大误差归因修正（对照实验 + 相关性验尸）

## 结论链条
1. 与 fluxv_old 预测的相位相关 = **−0.992/−0.993（完美反相关）**→ 大误差
   主因是**诊断提取的符号约定错误**（pterasoftware 汇报约定 lift=−forces_W[2]，
   我的收割直接用了 forces_W）。
2. 符号修正后：**W2 CL RMSE 1.26 / corr +0.97，CD RMSE 0.80** ——与旧基线
   （fluxv_old CL 1.08 / CD 0.78）同量级。V5H15 链没有"越改越差"。
3. **零 release 对照**：CL RMSE 5.55（未翻转口径）与 3-release 跑 5.56
   几乎逐位一致 → 冻结粒子云对载荷的贡献 ≈ 0（释放的 rVPM 云在当前
   配置下对载荷几乎无影响）；当前数字 = Ptera 核（prescribed wake UVLM）
   的固有水平，含与 fluxv_old 相同的 −0.92 CL 系统偏置（两者共享同一
   Ptera 核）。

## 含义
- 之前"冻结云毒药"和"粒子数"两个归因都不成立（所有者判断正确）；
- 剩余偏差（CL −0.92）是 Ptera/prescribed-wake 核的已知系统特征；
- rVPM 粒子尾迹要体现价值，必须每步 release + 连续对流，并证明能
  **主动改进** Ptera 核的诱导场（对照基线现已干净建立）。

## 已修正的正式诊断分数（W2，符号修正，含 3 release 冻结云）
CL: RMSE 1.259 / MAE 1.028 / bias −0.924 / corr +0.971
CD: RMSE 0.795 / MAE 0.641 / bias −0.495 / corr +0.838
（旧基线对照：fluxv_old CL RMSE 1.079/bias −0.929；v4b 1.036/−0.875）
