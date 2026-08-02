# 战役 2 — 文献聚焦：flapping wing modeling（2026-08-03，用户方向裁定）

日期：2026-08-03
方向：用户裁定"聚焦 flapping wing modeling"——此前文献散布在鸽翼 CFD/feathering 幅值
研究/机翼扭转等外围；本档聚焦**扑翼建模方法**（解析/半经验力模型），战役 2 缺的
"扭转×涡升力的既定形式"在这里。

---

## 1. 一手锚定文献（flapping wing modeling 体系）

| ID | 文献 | 形式 | 对战役 2 的意义 |
|---|---|---|---|
| M1 | **DeLaurier, Aeronaut. J. 97(964):125-130 (1993)**，"An aerodynamic model for flapping-wing flight" | 修正条带理论（caseA C3 已有） | **扭转进入力项的既定路径**：总扭转 θ=θ̃+θ̄a+θ̄wash；有效攻角 α = [ẋ·cos(θ+θwash)+(0.75c−y_ea)θ̇]/U + θ（几何+俯仰率）；经 Jones 修正 Theodorsen α' = C_Jones·α − w₀/U，downwash w₀/U = 2(α'+θ)/(2+AR)；法向力 C_n = 2π(α'+α₀+θ)；失速后 cross-flow drag (C_d)_cf≈1.98（战役 1 锚源） |
| M2 | **Reichert & DeLaurier 2007 / Larijani & DeLaurier**（DeLaurier 体系扩展） | feathering 参数 **χ = θ₀/(α_a)max = θ₁/arctan(π·St)** | **A3 解决**：feathering 参数一手解析定义（St=fA/U）；χ=1 恒攻角零推力 |
| M3 | **Polhamus 涡升力定律**（J. Aircraft 8(4):193-199 (1971)；NASA TN D-3767；caseA A2-C7 已有） | **C_Lv = K_v·cosα·sin²α**（K_v=(K_p−K_p²K_i)/cosΛ，K_i=1/(πAR) 椭圆翼解析） | **涡升力×攻角的文献闭式**：非单调（峰值 α≈54.7° 后回落）——"深扭转卸载"的解析形状 |
| M4 | **Han, Chang & Han, Bioinspir. Biomim. 12:036004 (2017)** doi:10.1088/1748-3190/aa640d | Polhamus 类比 + K_P(α,J)/K_V(α,J) 修正（147 实验案例，J=0~∞，α=−5~95°） | **涡力/势流力分量的攻角+前进比修正函数**（论文给定形式）；全文在付费墙，函数细节待取 |
| M5 | **2D 非定常扑翼理论**（PhysRevE 66:051907, 2002） | bound Γ 与 **LEV 环量 Γ_I 的闭式方程**（涡位置参数 b/a 拟合于实验） | LEV 环量的攻角/速度依赖解析形式 |
| M6 | **Gehrke & Mulleners 2020**（arXiv:2007.15729；Bioinspir. Biomim. 16:016002） | **涡龄标度**：剪切层速度 u_s = R₂·φ̇·cosβ + 0.25c·β̇，advective time σ=∫u_s dτ；Γ 随 σ 增长至饱和/离体 | **N3.1"事件年龄"命题的直接锚**：LEV 供给=剪切层速度积分，升力峰=环量饱和 |

## 2. 建模路径（组合洞察，非自造）

```
扭转 θ ──(M1: α = [ẋcos(θ)+...]/U + θ)──► 有效攻角 α'
   │                                        │
   └──(M2: χ = θ₁/arctan(πSt))────► feathering 程度
                                            ▼
                    涡升力 C_Lv = K_v(α',J)·cosα'·sin²α'   (M3/M4)
                                            ▼
                    N3 供给形状 = 文献闭式（非单调, 峰值 55° 后卸载）
```

**关键**：战役 2 证伪的"阈值形状无文献"结论**部分过时**——Polhamus 的
cosα·sin²α 就是文献给出的"涡升力随攻角非单调"闭式（峰值后回落 = 深扭转卸载）。
但需注意：α 峰值区（模型 α_kin 45-66°）与 Polhamus 峰值（54.7°）的交互需数值验证
（§3）。

## 3. 数值验证计划（D3 数据直接算，不重跑求解器）

- 用 D3 探针（tw22.5/45、U6/8/10）的 α_kin 分布 × Polhamus 形状 cosα·sin²α，
  对比当前 |A0| 直供的 twist 响应：
  - 若 Polhamus 形状的 N3 供给随 tw 增率先降（翻负）→ 候选替换方向可行
  - 若仍增 → Polhamus 单形状不足以翻斜率，需 K_v(α,J)（M4 付费墙细节）或
    攻角口径（M1 α' 含 downwash）修正
- 验证结果决定是否新预登记（候选：N3 供给形状替换为 Polhamus 文献形式）

## 4. 与树内纪律的关系

- N3.1.0 falsified（|A0| 直供作持续幅值）——Polhamus 替换**正是 N3.1 claim 框架
  要求的"供给律文献形式"**（onset/幅值/年龄拆分的替代品）
- 不触碰 aeff 映射（N5 域）：α' 口径若引入需显式簿记（M1 的 α' 是模型级概念，
  与现有 aeff_sep 的簿记关系要理清——新预登记内容）
- M6 涡龄标度 = N3.1"事件年龄"的直接实现路径（B1 形成窗 falsified 的替代：
  涡龄是连续积分非窗函数）

## 5. 诚实缺口

1. M4（Han 2017）K_V(α,J) 具体函数在付费墙内——需全文或 Sci-Hub 通道；若不可得，
   用 M3 经典 K_v(AR) 解析形式（椭圆翼 K_i=1/(πAR)）
2. M1 的 α' 含 downwash/AR 修正，与 UVLM 的 aeff_sep 是两套攻角口径——引入需
   簿记论证（候选命题的一部分，非默认）
3. M5 的 Γ_I 闭式含拟合参数 b/a——若采用需明确"文献拟合参数"边界（用论文给的值，
   不重拟合）
