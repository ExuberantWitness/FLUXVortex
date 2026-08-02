# Phase 3 预登记 — 候选 N2.6-QSSEP（准定常分离压差阻）

日期：2026-08-01
状态：`PREREGISTERED`（判决规则先于结果）
节点：N2.6（partial，可动空间）
基线：V4.1 fixed-name 184（confirmed 42 曲线/151 条件）
病因文档：`phase3_lesion_T2_quasisteady_sep_drag.md`（①②③ 完整链条）

---

## 1. 候选命题（唯一，禁多补丁并行）

> 对同一周期平均双侧力账，增加**准定常分离压差阻**（cross-flow form drag）：
>
> ```
> dN_sep[j] = C_D,sep · ½ρU² · c[j] · dy[j] · sin²(α_eff,local[j])     [N, 逐条带法向]
> ```
>
> - `α_eff,local[j]` = 条带 j 的局部有效攻角（既有 L-B aeff 或 A0 反演，不新增量）
> - `C_D,sep` = 文献平板法向阻系数 **1.8–1.98** 的有限翼修正（取 1.8 下限，
>   零新拟合——该值本身就是文献常数，非对实测数据拟合）
> - 施加方向 = 条带局部法向 → 输出面板法向载荷（co-design 适配）
> - 与 N3.3（面板法向 ds 涡诱导阻）在深失速区唯一簿记（禁双计）

**机理**：扑翼推进净推力 = 前缘吸力 − 分离压差阻 − 附着阻。分离压差阻随 U²
增长（q=½ρU²）、f 无关（准定常）、随 sin²(α_eff) 随攻角增长；aoa15 深失速区
已由 N3.3 覆盖（该处 bias 小）。这正是 184 基线 T bias 指纹的完整签名。

## 2. 实现边界（shadow/minimal，不覆盖 V4.1）

- 新 closure：`n26_qssep_v0_shadow`（在 `_v2_robo.py` _CLOSURES 注册，与 v41
  逐键相等 + 仅新增 N2.6 分离压差阻通道）
- 生产 `closure='v41'` 数值逐位不动（_resolved_call 契约 sha256 冻结）
- 新通道：`sep_drag_qs`（separated pressure drag，物理角色）
- 力账：`_v41_booked` 增加该通道，ForceLedger 闭合校验

## 3. go/no-go 判决规则（2026-08-01 用户裁定修正）

**唯一门槛 = 完整 184 基准相对 V4.1 的全指标对比**（用户裁定：中间过程比较是
内部诊断，最终判据是与 V4.1 的精度变化，在尽可能多的指标上优于 V4.1）。

**晋升判据（唯一）**：`s6_sweep_v41_full184`（fixed-name V4.1 基线）vs
候选 184 全扫，confirmed 域逐指标对比：

- **必须不恶化**：任一 L 曲线 MAE 恶化 ≤0.15N；dT/df 形状（Pearson）不降
- **目标（越多越好）**：T MAE / T bias / T RMSE / T 趋势捕获 / L MAE / L bias /
  L 趋势捕获——在尽可能多的指标上优于 V4.1

**中间探针（内部诊断，非门槛）**：
- U6/8/10 三点：确认阻力方向正确（dT<0）、升力隔离（|dL|≤0.15）、趋势不变
- representative32：确认三图关键峰值/转折/边界无异常

**执行序列**：三点探针（已完成，方向正确）→ representative32 →
完整 184（conditional184 scope）→ 与 fixed-name V4.1 全指标对照 →
晋升或证伪回写 claim 树。

**证伪条件**：完整 184 相对 V4.1 在 confirmed 域无可计指标改善且任一 L 恶化
>0.15N → falsified/frozen，禁调参重跑；不反向证伪父命题 N2.6。

## 4. 执行序列

1. 快环 3 点（U6/8/10 × f2.3 × tw22.5 × aoa5）→ G1/G2 初判
2. 代表工况 12-16 点（覆盖三图关键峰值/转折/边界：tw15 峰、tw45 滚落、
   U6/U8/U10、aoa0/5/10/15 角区）→ G3/G4 初判
3. 完整 184（`lb_sweep_candidate.py --scope full184`）→ 与 fixed-name V4.1 对照
4. 全部 G1-G5 过 → 晋升候选；任一不过 → falsified + 禁重走注册

## 5. 已知风险（诚实）

1. C_D,sep 1.8 是平板值，RoboEagle 有限翼/扭转翼的实际分离阻可能偏低
   （外推方向：Hoerner 1.98×有限AR修正，禁拟合）
2. α_eff,local 的选取（aeff vs A0 反演）影响 sin² 幅值——用既有 L-B aeff
   （已 validated N2.1），不新增量
3. 与 N3.3 深失速重叠的簿记边界是 G3 的验证对象，禁预设
4. 若 G1 显示量级不足（文献值偏保守），**不改常数**——按 ③ 裁决升级为
   SVI-DW 深度实现（Riziotis 路线），本候选按"量级不足"留档
