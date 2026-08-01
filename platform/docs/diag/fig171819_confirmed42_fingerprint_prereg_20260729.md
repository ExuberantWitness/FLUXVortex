# V4.1 confirmed42 残差指纹预登记

**日期**：2026-07-29  
**阶段**：研究流程①——数据规律；先于 claim 归因与文献方案  
**输入**：`baseline_residual_fingerprint_v41_confirmed42_20260729_125402.json`  
**输入 SHA-256**：`5a51f5f6460fea8e39c8c87df008a2760c23afbf81331fa4cc992af14fcdecbe`

## 1. 本轮只回答什么

本轮只回答 V4.1 在来源已确认的 Fig17/18/19(a,b) 上“错在哪里、错误如何随
figure/channel/U/f/AoA/twist 和曲线形状变化”。它不允许由总 L/T 残差直接猜测
N2、N3 或任何子候选。

在节点逐通道归因完成前，输出必须为：

```text
claim_decision = NO_DECISION
reason = NODE_ATTRIBUTION_REQUIRED
```

## 2. 冻结统计单位

- 官方精度终点单位：42 条 confirmed 曲线、434 个原始测点、151 个 solver 条件。
- Claim 复现单位：按 `(channel, ordered solver-condition tuple)` 去重后的
  34 个独立 physical curve family。
- 8 组跨图重复物理工况只作重复观测与数字化一致性检查，不得冒充独立复现。
- Fig19(c,d) 8 条曲线及其 96 个测点不得进入任何排名或 witness。

主排序必须同时给出：

1. 434 点 point-weighted；
2. 42 曲线 equal-weighted；
3. 34 physical-family equal-weighted。

三种排序明显冲突时，只能输出不稳定性，不得唯一挂接 claim。

## 3. 预登记逐点与逐曲线指纹

每个测点保存 raw x、实际模型求值 x、实验/模型/误差、solver 插值左右条件及权重。
实验 force 值不得插值。

每条曲线固定计算：

- MAE、RMSE、bias、median/q90/max absolute error、SSE 与绝对误差份额；
- raw-x 梯形归一化 signed/absolute residual；
- 实验/模型 range、standard deviation、Pearson、Spearman、中心化 cosine；
- OLS slope、首末端增量、逐段斜率符号一致率；
- argmax/argmin 的 x 和高度、内部转折与峰后下降；
- 既有 `captured` 只作兼容指标，不单独裁决病因。

固定分层为 figure×channel、abscissa×channel，以及从 solver condition
身份提取的 U、f、AoA、twist 常值组。边际统计只作描述，不单独支持因果归因。

## 4. Witness 选择规则

不看节点力之前，只允许由以下确定性规则选取诊断 witness：

1. 绝对误差份额最高的曲线；
2. slope-sign / turn topology 未捕获的曲线；
3. 每条入选曲线的两端、实验/模型极值和最大绝对残差点；
4. 将测点映射为其 solver 插值括号，去重得到最小 condition set。

Witness 只授权逐节点力通道 observer 重放，不是候选参数扫。

## 5. Claim 唯一归因门

后续对预先固定的 contrast \(c\) 使用：

\[
E=c^T y_{\mathrm{exp}},\quad M=c^T y_{\mathrm{model}},\quad
M_{-n}=M-c^T y_n .
\]

只有恰一个可动父节点被移除后能恢复实验 contrast 的符号/转折，并在至少两个
不共享 physical family 的证据单元复现，而其他节点均不能恢复，才允许挂接该
父 claim。否则为 `AMBIGUOUS` 或 `NO_DECISION`。

树边界：

- N1/N4 validated-frozen 只能触发完整性复审；
- N2/N3 只有 partial/open 且唯一 owner 成立时可动；
- N5 是观察节点，不是力修复节点；
- N6、falsified/dead-end 路径禁止重走；
- 归因到 N3 父节点也不能直接跳到 N3.1h/i，子节点必须等待定向一手文献与有效
  机理 witness。

## 6. NO-GO

输入哈希或 42/434/151 合同不符、Fig19(c,d) 混入、实验值被插值、重复物理
曲线被当作独立复现、任意阈值在看结果后设置、或在节点力归因前输出具体 claim，
均判本轮无效。
