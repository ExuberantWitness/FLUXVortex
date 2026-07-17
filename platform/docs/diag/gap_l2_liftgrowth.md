# 案卷 L2:升力频率增长缺失(v4 主案,2026-07-17 立案)

基线:v4(H16+plateau_fn+fsep_lag+geo_stall_vec+le+d_para0.5+uiuc)。
姊妹案:gap_lift.md(v1 时代,教训与文献仍引用),research_lift_deficit.md(34 题录)。

## 1. step-1 指纹(v4 刚性 118 扫描)

- **dL/df 缺失**:aoa0 = 0(−0.04);aoa5/10/15 = +1.79/+1.90/+1.76 N/Hz(α 饱和);
- **U 标度**:U6 +0.19 / U8 +1.8 / U10 +2.9(超线性,~onset 型);
- **扭转**:tw0 已有 0.38,增至 tw22.5(1.84)后持平;tw15 尖峰缺 1.45N(f2.3);
- Fig17b 频率展宽:实测 +0.3~+1.5N,模型 −0.1~−0.4N(反号)。

## 2. 机理判别(step-1 内)

要求:α≠0 才开(上下冲程不对称)、α 饱和、∝U 以上、随 tw 增强。
**首嫌:LEV 周期平均升力**(KJ 路由 L∝ρ·U·Γ_lev,Γ_lev∝f·超临界载荷):
- ∝U ✓(KJ 力线性 U);aoa0 对消 ✓;tw 增强(外侧超临界更多)✓;α 饱和
  (Γ_lev 由 LESP 门定,超阈后不再随 α 增)✓——四项全中。
- 与 v1 案卷合流:H15"DSV lift did NOT emerge"、vimp 反相审计(下冲程后半负峰,
  与物理 DSV 升力反相)——**LEV 升力供给通道从未正确入账**。
现闭合 LEV 升力路由:仅物质冲量 vimp(已知反相/近零);Vlev_a 表面压力路由被
lev_impulse 跳过(防双计);vnf 仅 hirato 模式。

## 3. 通道量化(探针进行中)

目标:vimp/KJ-LEV/bern 各通道现供 dL/df vs 需求 +1.8;文献实现候选
(Polhamus vnf / KJ-on-LEV / 冲量修正)按第二轮文献菜单(research_lift_deficit
M1/M3 已核实)选型。验收:aoa0 零变(对消保持)、tw0 推力律不破、U8 f 线
gapL 斜率 |<0.4|、全 MAE 双通道不回退。
