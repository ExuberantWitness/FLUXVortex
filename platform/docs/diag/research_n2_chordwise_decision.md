# N2.5 弦向力闭合证据表与 Phase A 裁决（2026-07-27）

## 1. 病因输入（纠正后）

Fig18 推力 U6/U10 曲线身份纠正后，f=1.4 Hz、tw=22.5°、AoA=5°：

| U (m/s) | 实测 T (N) | V4.1 T (N) | 模型−实测 (N) |
|---:|---:|---:|---:|
| 6 | -1.140 | -0.030 | +1.110 |
| 8 | -2.275 | -0.470 | +1.805 |
| 10 | -3.171 | -1.133 | +2.038 |

实测 `dT/dU≈-0.508 N/(m/s)`，V4.1 `≈-0.276 N/(m/s)`。两者同号，
模型缺少的是随 U 增长的阻力量级。旧“低 U 缺 3 N、高 U 不缺”的指纹来自
曲线反标，作废。

同口径三点通道重跑显示，U6→U10 的风轴推力斜率中：

- UVLM Bernoulli/压力通道约 `-0.253 N/(m/s)`；
- 前缘吸力约 `+0.304 N/(m/s)`；
- rig `d_para=0.5` 严格贡献 `-0.125 N/(m/s)`；
- UIUC 截面阻力的 body-x 增量约 `-0.187 N/(m/s)`；
- N2 风升力向砍量对风轴推力严格为零。

因此 N2.2 在完整 L-B 力分解意义上不完整，但不再是“反号主犯”。

## 2. 文献机理

| 路线 | 弦向力表达/语义 | 参数与适用性 | 对本案含义 |
|---|---|---|---|
| 原始 BL | `C_C^f = eta C_C^pot sqrt(f'')`；只含弦向压力力，黏性贡献在最终 `C_D` 中另加 | `eta` 来自静态翼型数据；大规模 LE 分离时 Kirchhoff 式失效 | 证明 normal/chordwise 必须分账；不能把风升力砍量当完整闭合 |
| Sheng 2008 | `C_C = eta C_Nα(α-α0)^2(sqrt(f')-E0)`；再由 `C_D=C_N sinα-C_C cosα+C_D0` 变换 | `E0≈0.2` 只是多数 NACA 系列经验值，`eta` 可调；二维低马赫俯仰 | 能给全分离负弦向力，但直接移植 `E0=0.2` 无 NACA2406/本 Re 锚，禁止 |
| IAG/Bangga 2020 | 指出 BL 的 `eta` 使阻力高度敏感；从滞后攻角处的静态极曲线取得切向力 | 仍有 drag limiting factor，且论文验证主要为 Re≈7.5e5、较厚风机翼型 | “静态 CT@lagged alpha”是最少拟合路线，但必须有可比静态 CT 极曲线 |
| BL 2024 实现审查 | 静态 `CL/CD → CN/CC`；大规模 LE 分离时原 Kirchhoff CC 不再有效，各改型使用经验相关 | 明确区分压力弦向力与后加黏性阻力 | 不支持把任一经验相关当跨翼型普适定律 |

主要来源：

- Sheng, Galbraith & Coton, *A Modified Dynamic Stall Model for Low Mach
  Numbers*, J. Solar Energy Engineering 130, 031013 (2008),
  DOI 10.1115/1.2931509。
- Bangga et al., *An improved second-order dynamic stall model for wind
  turbine airfoils*, Wind Energy Science 5, 1037–1058 (2020),
  https://doi.org/10.5194/wes-5-1037-2020 。
- Melani et al., *The Beddoes-Leishman dynamic stall model: Critical aspects
  in implementation and calibration*, Renewable and Sustainable Energy
  Reviews 202, 114677 (2024)。

## 3. 组成部分判定

判定为“**组成部分不完整，但不是当前 D1 的已证实主犯**”：

1. N2.2 只改风升力，缺少独立 chordwise port，架构上不完整。
2. 当前已有 `leading_edge_suction`、UIUC `profile_drag` 和 UVLM 压力投影。
   直接添加 Sheng/IAG `C_C` 会重复计算吸力或压力阻力。
3. 旧 Kirchhoff/LESP 分离阻诊断只有约 `-0.009/+0.06 N/(m/s)` 的 U 斜率潜力，
   远小于纠正后约 `-0.23 N/(m/s)` 的剩余斜率，不能凭名称宣称闭合 D1。
4. 可用静态数据是 SD7003 代理，不是本翼 NACA2406/圆管前缘组合的静态 `C_T`
   极曲线；Sheng 的 `E0` 和 IAG 限幅参数均无本案锚。

## 4. Go/No-Go 预登记

### 当前裁决：NO-GO（禁止进入生产力）

不启用 `lb_ct`、不加入 `E0`、不把静态 `C_T` 与现有 UIUC drag 叠加。理由是
唯一簿记和翼型证据均未满足；这样做会成为用文献常数包装的残差吸收。

### 允许的下一步：诊断端口

N2.5 保持 `open`，仅在以下条件同时满足后才允许做替代式候选：

1. 每时步导出 N1 压力、前缘吸力、现有 UIUC 阻力和候选 `C_C`，证明目标项唯一；
2. 候选必须**替代**现有 `profile_drag`/相应吸力份额，不允许相加；
3. 单条带静态回放能重构输入 `CL/CD`，误差和外推域显式报告；
4. 无 RoboEagle 动态数据拟合，Sheng `E0`/IAG limiter 无本翼锚时保持禁用；
5. 三点门要求高 U 阻力增量大于低 U，且不得破坏 `dT/df` Pearson、N1/N4
   frozen guard 和 Fig17/19 扭转趋势。

在上述条件前，Phase A 不改气动力；纠正后优先病灶转向 N3.1/N5 的升力—扭转
反向和角区频率过供。
