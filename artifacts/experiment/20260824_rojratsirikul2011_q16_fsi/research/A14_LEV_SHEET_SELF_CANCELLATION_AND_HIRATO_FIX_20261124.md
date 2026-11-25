# A14 — 前缘涡缺陷定位到"涡片自抵消"；并把它逐项对到 Hirato 的涡片表示（= 计划 L3）

日期：2026-11-24　分支：`run/q16-lev-tev-pc-fsi-20260821`
触发：用户要求参考 `FLUXV/researchpaper/` 中的文献（重点 Hirato et al.,
*Vortex-Sheet Representation of Leading-Edge Vortex Shedding from Finite Wings*）
修复 LEV 缺陷。

> **本笔记在同日更正过一次。** 首次版本把标志 `FLUXV_LE_SHEET_ARC` 说成"Hirato §II.D 的
> 弧落位"，这是读错原文；期刊也写错了。更正见 §4b、§6 与文末"更正记录"。§1–§3 的测量不受影响。

---

## 1. 缺陷的量化：不是"没有涡"，是"涡自己抵消掉了"

A13 在 100 步平均上发现中展面时均流场在**刚性平板**上也看不到回流区；A16 已把窗口
拉到 5 个脱落周期确认。机理在**已存的测量文件**里，不需要新跑：

`artifacts/baselines/fluxv_v5m_rojratsirikul2011_p1_joint_validation/near_le_cohorts_a15_700step.json`
（A15，700 步，probe 在步 100）：

| cohort | 粒子数 | 前缘法向速度 RMS (m/s) | 距前缘/弦长 |
|---|---|---|---|
| newest | 271 | **0.067** | 0.502 |
| second | 271 | 0.152 | 0.502 |
| third | 271 | 0.245 | 0.503 |
| 4–5 | 542 | 0.728 | 0.506 |
| 6–10 | 1355 | 0.937 | 0.514 |
| older | 22994 | 0.978 | 0.838 |
| **合计** | 25704 | **1.153** | — |

两条读数：

1. **最新投放的一层只贡献前缘法向速度的 0.067/1.153 = 5.8%。**
2. 越老的 cohort 距前缘越远（0.502 → 0.838），即涡量被抹成一片云，**没有贴在前缘的紧凑涡核**。

另一条零成本读数：全长 A16（2100 步）的 `particle_circulation_sum` 恒为机器零
（1.7e-21 → −1.1e-16）。这**不能**单独证明 LEV 无效——任何闭合涡环在该测度下都是零——
它只标示了投放的拓扑：生产路径走的是 `add_connected_ribbon_particles`（闭合回路）。

## 2. 机理条文：闭合投放的抵消率随厚度变化，且强方向性

纯几何量测 `platform/warp_vpm/diagnose_lev_ribbon_cancellation.py`
（不时间推进；31 节点展向线、真实几何 c=0.0688 m / b=0.1375 m、钉扎强度 −0.0342 m²/s、
生产投放参数 `dvm_smoothing_radius_chord=0.04`、`dvm_target_spacing_chord=0.018`），
在首排配置点读诱导法向速度，单极子（开放丝）参考 **0.67594 m/s**：

| d/c | 0.001 | 0.005 | 0.01 | 0.02 | 1/30 | 0.05 |
|---|---|---|---|---|---|---|
| 沿**法向** | 0.0% | 0.4% | 1.5% | 5.7% | 14.8% | 29.4% |
| 沿**弦向** | 2.1% | 11.5% | 24.8% | 56.0% | **103.6%** | 162.2% |

落盘：`research/lev_ribbon_cancellation.json`。

（首版探针把弦长写成 0.25 m，绝对速度因此小 3.63 倍（参考 0.186 m/s）；
**比值不变**——换成真实几何后 14 格的比值逐格相同到三位有效数字，参考变为 0.67594。）

* 法向加厚几乎无用（0.05c 才回到 29%）——首排配置点在**弦向下游** 0.033c 处。
* **弦向延展是控制参数**。
* 实测的 5.8% 落在"弦向厚度 ≈ 1e-3 c"这一格，与下面实测的 0.0054c 一致。

注意这个量测只回答"首排配置点的法向诱导"，即 LEV 对束缚解的反馈；
它不回答"翼面上方有没有回流"。

## 3. 生产涡片的实际深度、每步注入量

`platform/warp_vpm/diagnose_roj_lev_feed.py`（`sys.setprofile` 读 `_deposit_dvm_ribbon`
与 `_lev_sheet_frontier` 的返回值，不改求解器），A16：

| | 弦向延展 | 折合面板（nc=15） |
|---|---|---|
| 生产臂 | **0.00536 c** | **0.08** |
| 弦向放宽 2 面板 | 0.13333 c | 2.00 |

每步注入量（`lev_max_abs`，步 39）：**0.00115 m²/s / 条带**。
钉扎值是 0.0342 m²/s ⇒ 每步只注入钉扎值的 **3.4%**；
相对于该步（斜升中）束缚前缘行的均值 0.0118 则是 9.7%。落盘：`research/lev_feed_a16_40step.json`。

即：生产涡片既是**细缝**（几何），又只带**增量**（强度）。两个因子相乘，
才解释得了"30/30 分离却无回流"。

**几何与场内的对账**：按 §2 的表，0.0054c 的细缝保留约 12%（0.005c 格：11.5%，
即 0.0776 m/s）；§1 实测"最新 cohort"在前缘行的法向速度 RMS 是 0.067 m/s。
两者相差 15%。这只是量级对账（§1 的数据来自 A15 的 joint 路径，强度不同），
但方向一致：**几何上的细缝**与**场内最新一层只有 5.8% 份额**是同一件事。

## 4. 已排除的修法

`routeA_100_{closed,open,notev}.json`（joint 路径，A15/A19，100 步，`experiment_cn` 0.7 / 0.8）：

| 臂 | cn 尾部 50 步均值 (A15) | 首步 cn |
|---|---|---|
| closed（现状） | 0.612 | 57.8 |
| open（单符号丝） | **5.83** | 43.2 |
| open + drop_tev_compensation | 5.97 | 43.2 |

裸单符号丝过冲约 10 倍。A15 进一步定位：把新生 LEV 强度当作耦合求解的自由未知量会形成
正反馈（×1.5/步）。所以修复不能是"把闭合去掉"，也不能是"把强度放进耦合求解"。

`le_open_deposit` 默认 `False` 且没有 env 入口，只被一个 artifact 脚本和一个测试调用：
生产 A16/A17/A10/A23 全长全部走闭合片。

## 4b. Hirato et al. 的机制（原文，Sec. II.D）——首次版本在此读错

出处：Hirato et al., *Journal of Aircraft*（Article in Advance, 2019），doi:10.2514/1.C035124。
（首次版本写成 "J. Fluid Mech. 862"，错。）
算例：AR=6 等弦长翼，弦长 Re=20,000，`LESPcrit = 0.27`（取自 SD7003 翼型的 2D CFD），
绕 1/4 弦的 0–45° 俯仰斜坡。

逐条（引号内为原文）：

1. **脱落条件与强度**："an LEV ring for any strip of the UVLM is shed only when
   LESP(t) > LESPcrit for that strip. When LESP > LESPcrit, the strength of the LEV
   ring is determined so that the LESP is brought back to the predetermined LESPcrit value."
2. **落位**（Ansari et al. 的方法）："places the new LEV at one-third of the distance from
   the leading edge to the location of the last LEV." 式 (7)：
   `x_{L,n} = (2/3) x_LE + (1/3) x_{L,n−2}`。
3. **劈开而不是加宽**："the placement of a new LEV ring is implemented by splitting a
   previously shed LEV ring into two LEV rings at the point determined by Eq. (7)." 此后
   "this and all other LEV rings convect with the local velocity … with no change in strength."
4. **前缘处的那根丝**："The aft end of the new nascent vortex ring lies on the geometric
   leading edge … In the UVLM formulation, however, there is no bound-vortex filament at
   the geometric leading edge. The frontmost bound-vortex filament is located at one-quarter
   of the lattice width behind the leading edge."——与我们的 `[0.25dx, c + 0.25dx]` 一致。
5. **Pseudovortex ring**："of strength Γ_{L,n}, has the front edge at the geometric leading
   edge and the aft edge at the aft edge of the rearmost bound-vortex ring"，使
   "the geometric leading edge ends up having a zero-strength vortex filament"，
   紧挨前缘前方那根丝的强度是 `(Γ^n_L − Γ^{n−1}_L)`。
6. **强度由 LESP 定**："At every time step when LEV shedding is active …, the strength of
   the pseudovortex ring Γ^n_L is set so as to make the instantaneous LESP equal to the LESPcrit."
7. **Kelvin**（式 8 → 9）：`Γ^n_W = Γ^{n−1}_{b,i max} + Γ^{n−1}_L`。
   涡片的展向丝，从自由边数起：`Γ^{n−2}_L, (Γ^{n−1}_L − Γ^{n−2}_L), (Γ^n_L − Γ^{n−1}_L)`。

**稳态推论**：Γ^n_L ≈ Γ^{n−1}_L 时，内部各丝消失，整张涡片等价于**一个**从前缘绕到
卷起涡核的环，强度是**完整的** Γ_L；每步真正新增的只是 `(Γ^n_L − Γ^{n−1}_L)` 那一根丝。

### 与我们的实现逐项对照

| Hirato | 我们（生产路径） |
|---|---|
| 环链，每个环保持完整的脱落时刻强度 | 每步一个锚在前缘的闭合回路，只带增量 0.00115（钉扎值的 3.4%） |
| 新环放在"到上一个 LEV 环距离的 1/3"，上一个环是被对流走的 | 前沿取 2D source bank 的出生点（0.0054c 后、0.002c 上），与上一个 LEV 去了哪无关 |
| 跨弦长的 pseudovortex ring 抵消前缘处那根丝 | 没有：锚点腿（前缘线上的 −γ_lev）留在场里 |
| LESP 约束加在最前一根**丝**上：Γ_{b,1} + Γ_L = crit | 前缘**行**的环量钉在 ±crit·scale（legacy 臂符号反） |
| Kelvin 经 TEV：Γ_W = Γ^{n−1}_{b,max} + Γ^{n−1}_L | TEV 行加 γ_lev（`tev_compensation`） |

### 这正是 v2 计划的 L3

计划 L3 的三项——① 共享边连续前缘涡环链 `le_wake_rings`，② 净前缘丝 LESP 闭合 /
Kelvin 账本，③ 尾端净差量剥离为单极粒子——与上表第 1、5+6+7、"每步新增的那根丝"逐项对应。
其验收门 `w_dip/w_mono ≥ 0.80` 就是 §2 这类量测；今天量到的是 0.058（cohort）与
0.02–0.06（几何）。**L3 没有执行。**

### 工况差异（必须随结论一起写）

* 论文是俯仰斜坡（非定常、LEV 持续增长并被带走），我们是定常前飞 α=16°；
  定常分离下 LEV 的判据与形态仍是 open 问题。论文的"1/3"是斜坡上的经验落位。
* 论文翼型 SD7003、AR=6、`LESPcrit=0.27`；我们 AR=2、平板/膜、`lesp_crit=0.11`。
* **论文自述只适用于 AR ≥ 4**（2026-11-25 更正：本条首次版本写"原文没有这句话"，错）。原文
  （引言，第 2 页）："Implementation of such separated-tip-vortex sheets in conjunction with LEV
  sheets resulted in numerical problems when the sheets intersected with each other. For this
  reason, the current method is only applicable to wings with an aspect ratio (AR) of four and
  higher"——因为翼尖涡按附着涡丝处理。论文结果部分另有 "Although constant-chord wings have
  been used for illustration in this work, the method can be used for analysis of arbitrary
  wing planforms"，说的是平面形状，不是展弦比。我们的 Roj 翼 AR=2、扑翼 AR=3，
  都在论文的适用范围之外，采用该方法属于外推。
* 论文的 `F_S = πρcU∞²A₀²` 只进 `C_L = C_N cosα + C_S sinα` 的弦向项；
  我们对标的论文靶值是**垂直弦线**的 Cn，所以补它不改变 Cn 误差。

## 5. 更正后的定位

* **现象层面**（无回流区）：根因是 LEV 表示——细缝 + 增量 + 无 pseudovortex ring。
  对应计划 L3。
* **数值层面**（Cn 高 35–48%，见 A17）：**不能由 LEV 解释**。刚性平板比附着薄翼理论
  还高 27%；分离模型只会把力压到附着值以下。主嫌疑在共有的气动环节
  （有限翼展下洗 / AIC / 翼尖）。判别实验（刚性 α=5/10/15°）已排队。
* 两件事是两条线，不要合并成一个"修 LEV 就能提精度"的结论。

## 6. 代码现状

* `_lev_sheet_frontier`：生产投放与 joint 求解此前各自内联计算前沿，已抽成一个共用方法
  （本项目吃过"两条路径各算各的、差一个符号"的教训）。
* `FLUXV_LE_SHEET_EXTENT` ∈ {`off`, `chord`}，默认 `off`（生产路径逐位不变，
  `tests/test_roj_unified_case_runner_parity_gpu.py` 6/6）。它只是"把投放的弦向延展
  换成固定面板数"的受控旋钮，**不是任何文献的落位**。首次版本把它叫作 `FLUXV_LE_SHEET_ARC`
  并称其为 Hirato 的弧，还有一个按错误规则写的 `arc_third`；后者已删除。
* **量纲陷阱**：`trial.reference_node_chord` 是全弦长（`|TE − LE|`），不是面板弦长。
  以"面板"给出的长度必须先除以 `chordwise_panels`；第一版没除，把片送到了 2 倍弦长外。
* `chord` 臂（2 面板）在冻结容量下越界：失败时每步投放 432 个粒子、场上 32400、
  容量 32768（A16 §3）。用户已授权放宽容量与间距；但该臂不再作为"Hirato 修复"来跑。

## 更正记录（2026-11-24）

1. `X_{n−1}` / `X_{n−2}` 是**上两个 LEV 环的位置**，不是翼面节点；"2 个面板不是自由参数"撤回。
2. 期刊：Journal of Aircraft（首次版本误写 AIAA Journal）（doi:10.2514/1.C035124），不是 J. Fluid Mech.。
3. "论文自述仅适用于 AR ≥ 4"：首次版本写了这句，它是对的；上面的第 3 条"更正"（"原文无此话，撤回"）
   才是错的，2026-11-25 对照原文第 2 页恢复。
4. §6 原称"已实现 Hirato §II.D 弧落位"：撤回，改为受控旋钮并改名。
5. "每步注入约 10% Γ₁"：改为"钉扎值的 3.4%"（10% 是相对斜升中该步的前缘行均值）。

## 未解决

* `near_le_cohorts` 里"最新 cohort 距前缘 0.502c"不可能由一步对流达到。该脚本用
  `solver.surface.evaluate(None, None)` 取参考几何，是 artifact 期的 A15 自建设置，
  口径待核；本文只用它的**相对增长**与各 cohort 的速度份额，不用这个绝对距离。
* 贴着翼面加宽的闭合环是束缚涡量，不产生回流——这是推理，没有实测。
* L3 的实现尚未开始。
