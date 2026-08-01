# Fig17/18/19 父 Claim 归因协议

**日期**：2026-07-29  
**状态**：PROTOCOL-ONLY；fresh residual fingerprint 尚未生成  
**允许输出**：active parent hypothesis 或明确 NO_DECISION  
**禁止输出**：因果证明、子节点选择、YAML 状态修改、候选参数

## 1. 输入硬门

只有同时满足以下条件才允许实例化本协议：

- fresh manifest `status=complete`；
- result、contribution cases、case guards 和 expected keys 精确为同一
  confirmed151；
- scorecard/residual/fingerprint 精确为 42 curves、434 samples、
  151 conditions、34 physical families、8 alias groups；
- Fig19(c,d) 8 条曲线、96 点和 33 conditional-only conditions 零泄漏；
- result/contributions 哈希、graph identity、runtime/call contract、guards 和
  body/wind force ledger 全过；
- 节点与角色严格为 N1/N2/N3/N4/N6/R0 的当前生产身份；
- N1/N4/R0 frozen，N5 observer，N6 necessary-physics/dead-end，只有 N2/N3
  partial 父节点可成为 active hypothesis。

partial/running、150/151、哈希不符或账本不闭合为 `INVALID_EVIDENCE`，不是
`NO_DECISION`。

## 2. 两阶段隔离

### Prepare

只允许读取 fresh residual fingerprint，禁止读取 contribution。固定：

- 一个 active disease；
- 支持它的 physical-family IDs；
- 每族零和 contrast 的类型、测点索引和系数；
- witness conditions；
- 别名一致性规则。

### Evaluate

只有 prepare artifact 已原子落档并绑定 fingerprint SHA 后，才允许读取
contribution。Evaluate 不得新增/删除 PF、修改 contrast 或改变阈值。

## 3. 零和趋势 Contrast

所有系数之和必须为零，确保常数补偿无法改变裁决：

- 单调族：`END = y[-1] - y[0]`；
- 稳健内峰 k：`RISE = y[k] - y[0]` 与
  `ROLLOFF = y[k] - y[-1]`，两者构成不可拆 bundle；
- 稳健内谷 k：`FALL = y[0] - y[k]` 与
  `RECOVERY = y[-1] - y[k]`，两者构成不可拆 bundle。

沿用已冻结单点复现容差：

\[
\tau_F=0.15\ {\rm N},\qquad
\tau_c=\tau_F\lVert c\rVert_1=0.30\ {\rm N}.
\]

低于可辨识门的实验 contrast 不能成为节点票。

## 4. Frozen-state Leave-one-parent-force-out

只评估：

- N2 父贡献 = `separation + profile_drag`；
- N3 父贡献 = `ds_vortex + vortex_normal`。

子通道可以报告但不能单独挑选。模型及节点贡献都以相同的
model-to-raw-measurement-x 插值；实验力值不得插值。

对预登记 contrast：

\[
E=c^Ty_{\rm exp},\quad B=c^Ty_{\rm V4.1},\quad
G_n=c^Ty_n,\quad D_n=B-G_n.
\]

一次恢复必须同时满足：

1. 实验趋势可辨识；
2. V4.1 原趋势失败；
3. 删除父贡献后方向恢复；
4. \(|D_n-E|\le |B-E|-\tau_c\)；
5. \((B-E)G_n>0\)，即该父节点直接贡献确实把误差推向错误方向；
6. 整条曲线 MAE 改善超过 `0.15 N`；
7. 不翻转该曲线原本正确的其他稳健 contrast；
8. 峰/谷 bundle 的两侧必须同时恢复。

该操作冻结 N2/N3 状态轨迹，只擦除已报告加性力，因此最高因果状态固定为
`HYPOTHESIS_ONLY`。特别是删除 N2 力不会重新计算依赖它的 N3 状态。

## 5. Physical-family 投票

- 8 个 alias group 先做一致性检查；
- 同一 PF 的所有官方 alias 必须给出相同恢复裁决，否则为
  `ALIAS_SENSITIVE` 且不投票；
- 一个 PF 只计一票；
- 至少两个不同 PF、且使用不同 condition/contrast support，才算复现；
- active disease 的 family-equal MAE 改善还必须超过 `0.15 N`。

固定输出：

| N2 通过独立 PF | N3 通过独立 PF | 裁决 |
|---:|---:|---|
| ≥2 | <2 | `ACTIVE_N2_WRONG_COMPONENT_HYPOTHESIS` |
| <2 | ≥2 | `ACTIVE_N3_WRONG_COMPONENT_HYPOTHESIS` |
| ≥2 | ≥2 | `NO_DECISION_MULTIPLE_PARENTS` |
| <2 | <2 | `NO_DECISION_NO_REPLICATED_RESTORATION` |

若只有禁用节点看似解释误差，输出 `NO_DECISION_FORBIDDEN_PATH`。若父贡献擦除
都不解释，输出 `NO_DECISION_MISSING_OR_STATE_MEDIATED`；不能由此直接断言缺组件。

## 6. 输出边界

报告必须保存全部输入/源码哈希、active disease、contrast、逐曲线和逐 PF 结果、
alias consensus、N2/N3 票数以及：

```yaml
causal_status: HYPOTHESIS_ONLY
claim_writeback_allowed: false
spatial_panel_load_required: true
force_and_moment_ledger_required: true
posthoc_total_force_redistribution_forbidden: true
```

唯一父 hypothesis 成立后，才授权以该病因为锚点运行 research-pipeline 的开放式
一手文献扩展；必须同时覆盖主要竞争解释、相邻可迁移机理和反证来源，但不得脱离
已冻结病因漫游。达到机理饱和后，才能裁决“错组件/缺组件”并预登记一个机理候选。
