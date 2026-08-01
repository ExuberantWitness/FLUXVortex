# Fig17/18/19 fresh V4.1 Active Disease 冻结协议

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取任何 fresh
精度统计  
**阶段**：G1 数据病灶；不授权 claim 选择、文献方案或模型修改

## 1. 输入硬门

只接受 `fig171819_confirmed_compare.py` 从同一 complete fresh151 bundle 原子生成
并由 execution receipt 绑定的 scorecard、residual artifact 和 fingerprint。
输入必须通过：

- 42 confirmed curves、434 raw measurements、151 solver conditions；
- 34 physical families、8 alias groups；
- Fig19(c,d) 零泄漏；
- measurement force 不插值；
- complete-resume、固定 V4.1 graph/call/runtime/guard/source closure 全过；
- scorer、benchmark、measurement、fingerprint 和 authorization 的冻结 SHA 全过。

任一失败输出 `INVALID_EVIDENCE`。不得用旧 mixed/full184、旧118或 partial sweep
补齐。

## 2. Disease 的统计单位

Active disease 不是“误差最大的点”或“最差曲线”，而是以下零和趋势命题：

```text
(channel, abscissa, contrast_bundle, failure_signature)
```

其中：

- `channel ∈ {L,T}`；
- `abscissa ∈ {twist_deg,frequency_Hz}`；
- 单调族只允许 `END = y[-1]-y[0]`；
- 单峰族必须把 `RISE = y[k]-y[0]` 与
  `ROLLOFF = y[k]-y[-1]` 作为不可拆 bundle；
- 单谷族必须把 `FALL = y[0]-y[k]` 与
  `RECOVERY = y[-1]-y[k]` 作为不可拆 bundle；
- 所有 contrast 系数之和必须严格为零；
- `failure_signature` 按 bundle 中固定顺序保存每个 contrast 的
  `sign(B_j-E_j)`；`END` 为一个符号，`PEAK/TROUGH` 为两个符号。signature
  不同的曲线不得合并。

沿用预先冻结的可辨识门：

\[
\tau_F=0.15\ {\rm N},\qquad
\tau_c=\tau_F\lVert c\rVert_1=0.30\ {\rm N}.
\]

`END` 只有在 `|E| >= tau_c` 且 `sign(B) != sign(E)` 时才是 baseline
趋势失败。`PEAK/TROUGH` 的两个分量必须分别满足
`|E_j| >= tau_c` 与 `sign(B_j) != sign(E_j)`；只错一侧不是完整峰/谷 disease。
contrast excess error 固定为各分量
`max(0, |B_j-E_j|-tau_c)` 的 bundle 内算术平均。

实验 contrast 小于可辨识门、端点退化、多个噪声转折、或峰/谷任一侧不满足
可辨识门的曲线不能支持 disease。

## 3. Alias 与独立复现

1. 先按 benchmark 的精确 physical-family contract 合并官方别名；
2. 同一 PF 的所有 alias 必须得到相同 contrast 类型、失败方向和 baseline
   失败判定，否则该 PF 标记 `ALIAS_SENSITIVE` 并撤票；
3. 每个 PF 最多一票；
4. disease 至少由两个不共享 condition/contrast support 的 PF 支持；
5. 曲线名、figure panel 或重复数字化观测不得冒充独立复现。

## 4. 确定性唯一选择

只对满足第 2、3 节的 disease 候选排序。每个候选同时计算：

- `PF-equal`：先在 PF 内合并 alias，再等权平均 contrast excess error；
- `curve-equal`：每条官方曲线等权；
- `point-weighted`：只在该候选预先定义的 contrast support 点上按测点计权；
- 独立支持 PF 数；
- 覆盖的不同 solver-condition support 数。

三个权重视角分别按以下固定优先级排序：

1. 独立支持 PF 数降序；
2. 平均 contrast excess error 降序；
3. 不同 solver-condition support 数降序；
4. disease ID 字典序，仅用于完全数值相等时的确定性序列化，不用于打破科学并列。

只有三个权重视角具有同一个严格第一名，且第一名与第二名的 contrast excess
差超过 `tau_c`，才冻结一个 active disease。否则固定输出：

| 情形 | 输出 |
|---|---|
| 无候选达到两个独立 PF | `NO_DECISION_NO_REPLICATED_DISEASE` |
| 三种权重第一名不同 | `NO_DECISION_WEIGHTING_SENSITIVE` |
| 第一、第二名不可辨识 | `NO_DECISION_DISEASE_TIE` |
| alias 撤票后失去复现 | `NO_DECISION_ALIAS_SENSITIVE` |
| 只有幅值偏置、无可辨识零和趋势失败 | `NO_DECISION_OFFSET_ONLY` |

不得为了得到唯一 disease 而放宽阈值、拆开峰/谷 bundle、改分组或选择单个最大
残差点。

## 5. Disease Spec 必备字段

冻结产物必须原子写入，并至少包含：

```yaml
status: ACTIVE_DISEASE_FROZEN | NO_DECISION_*
input_bundle_id: ...
scorecard_sha256: ...
fingerprint_sha256: ...
disease_id: ...
channel: L | T
abscissa: twist_deg | frequency_Hz
contrast_bundle: END | PEAK | TROUGH
failure_signature: [-1 | 1, ...]
support_physical_family_ids: [...]
support_official_curve_keys: [...]
alias_consensus: [...]
contrasts:
  - id: ...
    coefficients: [...]
    sum: 0.0
witness_condition_keys: [...]
rankings:
  physical_family_equal: [...]
  official_curve_equal: [...]
  point_weighted: [...]
claim_decision: NO_DECISION
reason: NODE_ATTRIBUTION_REQUIRED
```

`disease_spec` 不得包含 contribution 数值、N2/N3 名称、候选公式、参数或文献结论。
Prepare 阶段也不得打开、stat 或 hash contribution 文件。

## 6. 后续边界

只有 `ACTIVE_DISEASE_FROZEN` 才允许进入 G2，并严格按父 claim attribution
协议比较 N2 与 N3。若输出任何 `NO_DECISION_*`，下一步是补充不改变模型的诊断
证据或报告不可辨识性，不得跳到 LESP、空间涡、rVPM、结构、参数扫或任意候选。
