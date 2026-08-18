# Baik W2 全周期诊断结果（DiGT-1，第一个三论文数字）

管线：V5H15 链（κ=1.75/graded k=5）× 64 步（2 周期）× 逐步原生载荷 →
末周期相位 CL/CD → 1 Hz sharp Fourier low-pass → 对 GT 400 唯一相位点。
runner：/tmp/v5h15-paper/w2_runner.py；曲线：w2_curves.npz。

## 结果（vs Baik 数字化 GT；旧基线 macro MAE: v4b CL=0.549/CD=0.298）

| 量 | RMSE | MAE | bias | pred 范围 | GT 范围 |
|---|---|---|---|---|---|
| CL | 5.564 | 4.638 | −3.306 | [−4.50, 2.55] | [−0.54, 4.65] |
| CD | 2.191 | 1.714 | +0.745 | [−1.11, 2.39] | [−2.31, 1.70] |

## 判读（诚实）

1. **管线价值**：全周期→滤波→评分链路首次打通，可复用于任何后续版本；
2. **结果差的原因已精确定位**：rVPM 粒子尾迹只在 step 3–5（前 0.6s，
   周期前 17%）被输送，之后 parent 场冻结——第二周期载荷实为
   "Ptera prescribed wake + 冻结 parent"，物理不成立；
3. **缺失部件被隔离**：step 6–63 的粒子连续输送（transport 事务目前
   绑定 release 事件，无法脱离 release 独立运行）——这正是 outer
   机制（B4）的核心工程，也是三论文复现的真正门槛；
4. scratch 解除的有界截断仅为诊断（冻结文件未动）。

## 对三论文路线的结论

Izraelevitz/Yang 无需重复此实验——同样的连续输送缺口会先出现。
下一步唯一关键工程：**无 release 的持续 transport 事务**（每 Ptera 步
推进既有粒子云 + 更新 parent 场），实现后三个论文共用。
