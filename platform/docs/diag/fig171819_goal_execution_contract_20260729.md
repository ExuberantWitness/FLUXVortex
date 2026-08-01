# FLUXV Fig17/18/19 单一 Claim 改进执行合同

**日期**：2026-07-29  
**状态**：ACTIVE  
**当前阶段**：G0 fresh V4.1 基线生成与完整性验收  
**当前 run id**：`20260729_135128`

## 唯一结果目标

在不修改 V4.1 validated/frozen claim、不重启 falsified/dead-end 路径、且不使用
常数补偿或总力后分配的前提下，只改写一个有唯一归因证据的 partial/open claim，
使完整 Fig17/18/19 的 50 条曲线、530 个测量点、184 个唯一 solver conditions
上的预登记 physical-family-equal 精度相对 fresh V4.1 可复现提升。42 条已确认
曲线只能作为阶段门，不能替代最终 50/184 终点。

候选必须输出有作用位置的逐面板时变载荷，并同时闭合全翼力、关于固定参考点的
力矩和功共轭传力，以支持后续质量—刚柔—运动 co-design。本轮不做结构求解、
材料优化或 FSI 性能优化。

## 证据范围

- 原始数据合同：50 条曲线、530 个原始测点。
- 当前无歧义主证据：42 条曲线、434 点、151 solver conditions、
  34 independent physical families、8 alias groups。
- Fig19(c,d)：8 条曲线、96 点；固定频率身份 unresolved，禁止进入病因选择、
  候选评分或当前视觉验收。
- 用户已把最终 solver scope 锁定为 184 个唯一 conditions。这要求 Fig19(c,d)
  两通道共用一个、且属于 confirmed 频率集合的固定频率，但当前公开一手来源没有
  给出该频率值。旧代码中的 `2.6 Hz` 只能标为 conditional inference，不能冒充
  实验元数据或用于最终晋升。取得可追溯的共同固定频率后，才允许冻结 exact184
  condition keys；若权威证据与 184 前提冲突，必须暂停最终门并请用户裁决，不能
  静默把终点改成 217/199/其他条件数。

## 不可交换的阶段门

### G0 — 冻结可信 V4.1

必须完成 fresh confirmed151，取得 complete manifest、151 个 result/evidence/
guard、result/contribution 哈希、42/434/151 coverage 和零 GPU complete-resume
重验。旧 mixed/full184 和旧118图不得替代。

### G1 — 数据病灶

在不读取节点 contribution 排名的条件下，生成 confirmed-only scorecard 和
42/34-family residual fingerprint；冻结一个 active disease、支持 PF、零和趋势
contrast 和 witness。此阶段最高结论固定为
`NO_DECISION / NODE_ATTRIBUTION_REQUIRED`。

### G2 — 唯一父节点假设

只比较可动父节点 N2 与 N3。对预登记 contrast 做 frozen-state
leave-one-parent-force-out；alias group 只计一票，至少两个独立 PF 复现。
该结果只能是 `HYPOTHESIS_ONLY`，不能直接改 YAML 或宣称因果。

### G3 — 病因锚定后的开放式一手文献扩展

以 G1/G2 已冻结病因和唯一父节点为检索锚点，运行 research-pipeline 的开放式
文献扩展：覆盖高质量一手论文、相互竞争的学术解释、相邻但可迁移的机理框架及
反证来源。不得在归因前先选方案，也不得把开放式检索变成与病因无关的技术漫游。
文献停止条件是达到机理饱和，能够用一手证据裁决“现组件错误/缺组成部分”，排除
主要竞争解释，并定义一个可证伪的单一候选。

### G4 — 唯一候选预登记

一次只允许一个候选；固定公式、状态、空间载荷生成途径、守恒门、代表工况、
失败预测和 GO/NO-GO。禁止参数扫、常数吸收、多补丁并行。

### G5 — Shadow/minimal 实验

先运行关键峰值、转折、边界代表工况。未过门即回写 falsified 并停止；过门后
才运行完整 confirmed42，并与同一 immutable fresh V4.1 bundle 成对评分和视觉核对。

### G6 — 最终范围与 Claim 回写

取得 Fig19(c,d) 共同固定频率的可追溯身份后，冻结 exact184 keys 并完成
50 curves / 530 measurements / 184 solver conditions 最终门。
全部精度、趋势、空间载荷、力/矩/功守恒和视觉门通过才可晋升；否则证伪并禁止
换名重走。

## 当前唯一允许动作

1. 监控并完成 fresh151；
2. 在不读取 partial 精度结论的前提下完成评分、bundle 和可视化工具；
3. 查明 Fig19(c,d) 的共同固定频率身份，以便冻结 exact184；
4. fresh complete 后执行 G1。

## 当前明确禁止

- 从 partial sweep 报告 Fig17/18/19 精度；
- 用 `trend_metrics.py`、`plot_3way.py`、旧 mixed/full184 fingerprint 作本轮证据；
- 在 G1/G2 前选择 N2/N3 子节点或回到 LESP 参数调整；
- 先实现空间涡、rVPM、结构或 FSI；
- 修改 N1/N4/R0，重启 N5/N6 或任何已 falsified 子命题；
- 把仅有合力闭合称为 co-design 空间载荷完成。

## 当前权威资产

- fresh result：
  `platform/docs/s6_sweep_v41_confirmed151_fresh_20260729_135128.json`
- fresh manifest：
  `platform/docs/diag/fig171819_v41_confirmed151_fresh_manifest_20260729_135128.json`
- fresh contributions：
  `platform/docs/diag/fig171819_v41_confirmed151_contributions_20260729_135128.json`
- baseline runner prereg/authorization：
  `v41_confirmed151_fresh_prereg_20260729.md`、
  `v41_confirmed151_resume_integrity_addendum_20260729.md`、
  `v41_confirmed151_launch_authorization_20260729.md`
- postprocess prereg：
  `v41_fresh151_postprocess_prereg_20260729.md`
- active disease prereg：
  `fig171819_active_disease_prereg_20260729.md`
- parent attribution protocol：
  `fig171819_parent_attribution_protocol_20260729.md`

修改本合同只能用于纠正新的一手身份或完整性证据，不能根据候选结果放宽验收门。

## G1 执行证据冻结（fresh 完成前）

病灶选择、Prepare 与父节点 Evaluate 的科学实现及运输层已在不读取 fresh/live
result、manifest、contributions 的条件下冻结。生产执行只允许使用以下 literal
身份，禁止按“最新 ref”发现：

- evidence root commit：
  `cfe4fb155a549ef5b2178226578f6848a0a4c66e`；
- attestation commit：
  `14d6f72fabcd8c194162b0388a92f4e4de32c6fa`；
- attestation payload exact-byte SHA-256：
  `a4373300e0553101ce69d6c417ce5ef069424c1cfb8a065f6423af4e06ffb12b`；
- execution authorization exact-byte SHA-256：
  `22b667811a88d704590bd004bb4e0fb7e5c4865516d6941691ee786ab2e77149`；
- inner launcher exact-byte SHA-256：
  `bac3d098a5e886c185d66020865a12afd49ff403df7eac0bee285b1235e27bc9`；
- outer bootstrap exact-byte SHA-256：
  `f0a57b088f3a17e24eaac22a4f704bf2a94e3e151c23a0bb37d5cd18ffabaaef`。

Evidence root 为无 parent 的 32-path exact allowlist；attestation 的唯一 parent
为该 evidence，tree delta 只有固定 attestation payload。Runtime closure 为
24 paths，并绑定 CPython 3.12.13、`-I -S -B` 与 NumPy 2.4.6 的 1325 个
regular distribution files。

repo-local raw-only detached dry-run 已由外层 `ls-tree + cat-file blob` 物化后
通过。权威 receipt 为：

- H0 `f2cf77df3b526efccea3732fa6fe272ab04af5bfc6e2eaef2710e485bb66ad98`；
- H1 `ea605e904a6e5ff3335a77451ba4cf5dfa841e5ec6b4d46e08a0fdd53250b3c8`；
- H2 `0708e80f4e64d6e671c0ace955cda778ed58e380a835a3d05a58b33f4c72543a`；
- final `5c1e49936db8717cd191ce9234ab35c5c1729df7efa627be3ca977fc8117b7e5`。

状态严格为 `PASS/PASS/PASS/DRY_RUN_PASS`；真实 import probe 后 closure SHA
仍为 `51c2f6d2ea108e0e964e9227dc53954b7772790dd6cf30666d6db5d3dd84f7dc`，
且未生成 `.pyc`。独立零数据复审裁决 scientific protocol、TCB、minimal CLI
fix 与 v4 raw detached dry-run 均为 GO。

较早的 v3 evidence/attestation
`cdc968aee6eb8934ec01491e53d4dfa4ab041578` /
`6ebc96c6dd146900e6435e8b70732a246aecb25d`，以及 archive-based v4
dry-run，固定标为 `SUPERSEDED/NONAUTHORITATIVE`；它们未读取科学输入、未运行
selector，保留仅用于失败先例审计。

本节只授权在 fresh151 完成并通过 G0 后按三阶段链执行 G1/G2，不构成精度、
病因或 claim 裁决。
