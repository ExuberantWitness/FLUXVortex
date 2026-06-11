# 文献坐标系:分区 FSI 预测-校正的高阶化与梯度加速(2026-06)

## 五个子方向的代表作与我们的相对位置

### 1. 分区耦合的高阶时间表示 —— waveform iteration 家族
- Rüth, Uekermann, Mehl, "Quasi-Newton Waveform Iteration for Partitioned FSI" (IJNME 2021; arXiv:2001.02654):块(时间窗)内耦合变量的**连续多项式表示**(经子步采样插值)+ IQN 加速,达成时间高阶 + 多速率。
- 后续:Time-adaptive multirate QN waveform iteration;"A waveform iteration implementation for black-box multi-rate higher-order coupling"(arXiv:2511.07616, 2025)→ 已进 preCICE 生态。
- **我们的位置**:两遍方案 = **1 阶 waveform(线性窗内插值)+ 单次 Picard**。我们的"主导误差=线性力插值"诊断与该家族动机完全一致。
- **缺口**:waveform 家族的高阶靠**多采样点**(更多子步交换/流体解);**用端点导数(Hermite)替代多采样、且导数来自可微分求解器 JVP——未见先例**。

### 2. 拟牛顿界面耦合(IQN-ILS 一族)
- Degroote/Vierendeels IQN-ILS;Generalized Broyden 综述 (Arch.Comput.Methods Eng. 2023);多级/多向量变体。
- 共识:不可压+质量比 O(1) 时弱耦合无条件不稳定(added-mass);IQN 基本克服。
- **近邻热点**:ML-enhanced predictors for accelerated convergence of partitioned FSI (CPC 2025, arXiv:2405.09941)——**用学习的预测器减少耦合迭代数**。
- **我们的位置**:Picard-2 实验=朴素版;M*=1 弱载下单次已近收敛(0.022%),与文献"重载才难"一致 → 假设 3(M*<0.5 探针)有文献支撑。

### 3. 可微分物理用于求解器内部(非设计优化)
- Warp/JAX 生态(JAX-FEM、DiFVM、Diff-FlowFSI arXiv:2505.23940=可微 FSI 平台);implicit-diff/固定点 custom VJP 常用于**训练**。
- **JVP 构造耦合预测器/加速物理耦合本身:未见直接先例**(最近的是 ML 学习预测器)。

### 4. 预测器质量理论(经典)
- Piperno (IJNMF 1997) 结构预测器+流体子循环;Piperno & Farhat (CMAME 2001 Part II) 能量传输分析。
- 已知:松耦合下预测器阶次改善精度/稳定性。我们的两遍方案中预测器只决定流体求值状态——Phase A 实测其影响为二阶效应,与"校正遍插值主导"一致。

### 5. UVLM 尾迹-时间步绑定
- 混合 UVLM-VPM 自适应尾迹转换(arXiv:2511.11430 旋翼, 2025;MDPI Fluids 2022)→ 我们 P0-2 的混合尾迹在已知版图内。
- **"子脱落使耦合窗与尾迹分辨率解耦"作为研究对象:未见专门工作**(常规默认 wake panel = U·dt)。

## 两个关键问题的明确回答
1. **Hermite 界面力插值(带导数)有没有人做过?** 协同仿真/FMI 领域有(线性系统四阶 Hermite, Energy Reports 2022;IFOSMONDI C1 Hermite, arXiv:2101.04485)——但 (a) 不在 FSI/气弹 waveform 线,(b) 导数来自线性系统 Jacobian/有限差,非可微求解器精确 JVP,(c) 无"近收敛方案中是否值得"的诚实边界研究。
2. **JVP/可微分预测器?** 无直接先例;最接近的是 2025 的 ML 预测器(学习而非解析梯度)。

## 对 idea 生成的输入
- 新颖性楔子:**导数信息(autodiff-JVP)驱动的 Hermite-waveform 耦合**用于分区气弹 + **子脱落解耦**(让大块假设复活)+ **1e-6 级验证基准上的诚实边界刻画**(质量比/块长适用域)。
- 风险提示(来自我们 Phase A 数据):1× 块下残差仅 0.022%,任何"精度提升"主张必须在 (a) 大块(脱钩后)或 (b) 重载 M*<0.5 或 (c) 成本维度(迭代数/流体解数)上立足。
