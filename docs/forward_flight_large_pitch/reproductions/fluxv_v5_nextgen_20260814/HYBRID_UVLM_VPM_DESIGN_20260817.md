# 混合 UVLM-rVPM 完整架构设计（文献背书版，供实现）

## 文献基础
- Proulx-Cabana et al. 2022 (Fluids 7(2):81)：NL-UVLM-VPM 核心方法（含
  "Conversion of Straight-Line Vortex Elements to Vortex Particles" 节）
- Proulx-Cabana 2024 (Fluids 9(1):24)：参数敏感性（buffer 面元行数、核半径）
- NL-UVLM-VPM 2025 (arXiv:2511.11430)：转化公式 Γᵢ = ΔΓᵢ·dl/nᵢ
- Willis 2006 (MIT)、Wang 2018：panel+particle 唤醒混合的边界条件惯例

## 架构（五要素）
1. **近尾迹 buffer**：尾缘后保留 N_buf 步的 wake 面元（承担 Kutta 条件/
   尾缘压力均衡）。2024 敏感性研究：buffer 行数与对流距离挂钩而非固定值，
   行数越多转化误差离翼越远（推荐 1–4 步起步做诊断扫描）。
2. **面元→粒子转化（替代语义）**：buffer 之外的 wake 面元转为粒子：
   每段直涡元 dl 上放 nᵢ 个粒子，单个强度 Γᵢ = ΔΓᵢ·dl/nᵢ（ΔΓᵢ = 相邻
   面元环量差，无邻居则 Γᵢ）；粒子位置沿段按弧长比例分布；核半径 σ =
   相邻粒子最大距离（常数保持），σ > 间距。
3. **RHS 反馈**：无穿透切向条件于 3/4 弦 collocation 点：
   (V_∞ + V_bound + V_wake_panels + **V_particles**)·n = 0
   粒子诱导速度与 wake 面元同槽进入影响系数（我们代码的
   `_currentStackWakeWingInfluences__E`），符号/框架（E↔GP1）需在实现时
   用代码核对（开放项 1）。
4. **防双重计数**：转化 = 面元从尾迹表示中移除，其环量由粒子承载
   （替代非叠加）。Kelvin 一致性由 ΔΓ 规则隐式保证（arXiv Eq.7）。
5. **时间步进序**：solve（含粒子下洗）→ 对流粒子（我们的 WRK3 流）→
   老化 buffer → 转化新一批（arXiv 未明说顺序，开放项 2）。

## 映射到本仓库
- buffer：Ptera 保留 wake 面元 N_buf 步后"转化"——实现为：把超过 N_buf
  步的 wake 环量按 ΔΓ 规则生成粒子并从 parent 表示中移除（具体挂点：
  `_calculate_wake_wing_influences` 重写 + wake 老化钩子）；
- 反馈：粒子场（direct kernel，P×N_coll 廉价）加到 collocation 下洗；
- 粒子对流：复用 `ir_wrk3_stream_macro`（已验证）；
- 现有 V5H "release/LDVM 源"机制与面元转化机制的关系是开放项 3：
  文献架构以转化为主；LDVM 源脱落是我们的既有机制，两者可能统一
  （LDVM 释放强度应满足 ΔΓ 规则）。

## 开放项（实现前核对）
1. E 框架 vs GP1 框架的变换（读 pterasoftware stackCpp 的框架注释）；
2. 转化在步内的确切时序（读 Proulx-Cabana 博士论文 PDF，
   depozit.isae.fr/theses/2024/2024_Proulx-Cabana_Vincent.pdf，本地下载）;
3. LDVM release 与 ΔΓ 转化的统一（对比两者环量账本）。

## 验收
干净对照基线（Ptera 裸核 CL RMSE 1.26/corr 0.97）：完整架构 W2 必须
显著 < 1.26 才有存在价值；目标 v4b 以下（<1.0）。

## 工程量估计：2–3 个会话（buffer+转化 1 个、RHS 反馈+联调 1 个、
诊断扫描+验收 1 个）。

## 开放项核对结果（2026-08-17 深夜补）

**开放项 1 已解决**：`_currentStackWakeWingInfluences__E` = 逐面板标量
（wake 诱导速度(GP1 分量)·面板法向 stackUnitNormals_GP1 的 einsum 点积）；
环量求解为 `_current_bound_vortex_strengths = solve(GridInfluences,
−WakeInfluences − FreestreamInfluences)`。粒子反馈注入 = 在
`_calculate_wake_wing_influences` 重写中：super() 之后把
`direct_kernel(particles, target=stackCpp_GP1_CgP1).velocity · 法向`
加进该数组。框架一致（GP1），无需 E↔GP1 变换。

**wake 结构已核实**：wake 为逐翼的 ring vortex 点格
（`wing.gridWrvp_GP1_CgP1`，每步 shed 一行），伴随
`_current_wake_vortex_strengths` 与 ages。转化 = 提取超过 N_buf 行的
ring 的 4 条线段 → ΔΓ 规则生成粒子 → 将该行 strengths 置零（移除）。

## 实现蓝图（会话 1 直接照此编码）

scratch solver 类（继承 V5H15NativeBaikCouplingSolver）：
```python
class HybridSolver(V5H15NativeBaikCouplingSolver):
    # 持久粒子态（诊断 tier，绕过 owner 账本，文档披露）
    _px: np.ndarray; _pg: np.ndarray; _ps: np.ndarray

    def _calculate_wake_wing_influences(self) -> None:
        self._convert_aged_wake_rows()   # >N_buf 行 → ΔΓ 粒子；strengths 置零
        super()._calculate_wake_wing_influences()  # 仅 buffer 行贡献
        self._inject_particle_downwash() # direct kernel @ stackCpp, ·法向, 相加

    def _convect_particles(self):        # 每步 RK2：自由流 + 自诱导 + Ptera parent
        ...
```
步序：convert → super() → inject → solve → loads；粒子对流在步末。
诊断简化（v1）：粒子对流用显式 RK2（WRK3 流接入留会话 2）；
LDVM release 停用（文献架构以面元转化替代，开放项 3 的统一后续再议）。
参数：N_buf ∈ {2,4} 扫描；σ=相邻粒子最大间距（文献规则）。
验收：W2 CL RMSE < 1.26（Ptera 裸核）；目标 < 1.0（v4b）。

## 会话 1 结果（2026-08-17，HybridSolver v1，N_BUF=4）

运行 ~118 min，粒子累积至 42k（每步约 750，wake 行转化正常）。

| | CL RMSE | CL corr | CL bias | CD RMSE | CD corr | CD bias |
|---|---|---|---|---|---|---|
| Ptera 裸核对照 | 1.259 | 0.971 | −0.924 | 0.795 | 0.838 | −0.495 |
| **Hybrid v1** | 2.073 | 0.920 | −0.967 | 1.007 | 0.718 | **+0.060** |
| v4b 基线 | 1.036 | 0.981 | −0.875 | 0.723 | 0.860 | −0.408 |

**判定**：
1. 反馈通路工作正常（载荷显著改变，粒子真正参与载荷——架构成立）；
2. **CD 系统偏置被消除**（−0.50→+0.06）——粒子尾迹修正了 prescribed wake
   的阻力偏差，物理方向正确；
3. 但 RMSE 劣化、CL corr 下降——v1 简化的代价，候选原因（会话 2 修）：
   a) σ 规则：用了段长（展向 82mm，过度平滑，尖端卷起被抹掉）——文献
      规则是"相邻粒子最大距离"，应远小于此；
   b) 粒子对流缺 Ptera parent 速度（buffer 尾迹对转化粒子的诱导）；
   c) N_BUF=4 未扫描；
   d) 每步 ~750 粒子提示每步转化了不止一行（chordwise 多行 wake），
      需核对 wake 行拓扑。

## 会话 2 清单（按优先级）
1. σ 改为行间距尺度（~7.5mm）或按文献"最大邻距"计算；
2. 对流加 parent（buffer wake 诱导速度）项；
3. N_BUF∈{2,8} 扫描；
4. 核对每步转化的行数与 ΔΓ 消解。

## 会话 2 v2 结果（σ=30mm 细分 + 20 步记忆截断）
CL: RMSE 1.688/corr 0.893/bias +0.434（v1: 2.073/−0.967；裸核 1.259/−0.924）
CD: RMSE 1.476/bias +0.656（v1: 1.007/+0.06）
判读：σ 锐化使 CL bias 减半且翻正、CL RMSE 改善——方向正确；
CD 退化与 20 步截断强相关（v1 无截断 CD bias +0.06）→ 远尾迹对 CD 关键。
v3 已启动（pid 4089776，log /tmp/v5h15-paper/hybridv3.log）：σ=30mm 细分
+ 无截断（预计 ~4h，粒子 ~75k）。

## v5 结果（GPU + 守恒合并，2026-08-17）
- GPU 核（torch FP64 分块）：与冻结核一致 1.1e-15；40k 自评估 1.2s（CPU ~60s）；
  **case 时间 2.7–4h → ~10 min**。
- 寿命控制按文献用**守恒合并**（Winckelmans/Siemaszko/FLOWVPM）：>16 步
  粒子按 0.12m 网格合并，γ 求和精确守恒、强度加权质心近似保冲量。
- W2：CL RMSE 1.616/bias +0.452/corr 0.887；CD RMSE 1.484/bias +0.692。
- 版本对比：v1(粗σ全尾迹) CD bias 最好(+0.06) 但 CL 最差；v2/v5(细σ+寿命
  控制) CL 改善 CD 变差——σ 与寿命参数在 CL/CD 间交换。均未超裸核 RMSE。
- **下一步（每 case 仅 ~10 min，可廉价扫描）**：N_BUF∈{2,8} × σ∈{20,30,45mm}
  × 合并格 {0.08,0.12,0.20m} 网格扫描，找 CL+CD 综合最优点；然后才谈
  是否超越 prescribed wake。
