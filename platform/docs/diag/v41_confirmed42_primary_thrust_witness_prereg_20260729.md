# V4.1 confirmed42 主推力病灶：节点通道 witness 预登记

**日期**：2026-07-29  
**阶段**：研究流程①——唯一 claim 归因前的节点力 witness  
**状态**：PRE-REGISTERED；未运行 witness，未指定 N2/N3，未修改模型

## 1. 数据指纹裁决

来源：

- `fig171819_v41_confirmed42_fingerprint_20260729_125657.json`
- 42 条 confirmed 曲线、434 点、34 个独立物理曲线族

主病灶不是由旧 D4–D8 或旧 trend score 选出，而是 fresh confirmed42 的三种
统计单位一致给出：

- 推力占全部 absolute residual 的 `61.6529%`；
- 21/21 条推力曲线 bias 均为正；
- 推力 point-weighted `MAE=1.66243 N`、`bias=+1.66059 N`，说明主误差几乎
  完全是模型推力过正/阻力不足，而不是正负抵消；
- 独立 physical-family MAE 排名前 10 位全部是推力族；
- 最差三族均为 Fig18(c) `U=10 m/s`，在 `f=2.0/2.3/2.6 Hz` 下
  MAE 分别为 `2.501/2.568/2.647 N`；
- Fig18(c) 的 `U10-U6` 推力 contrast 方向正确但幅值严重不足：

| f | 实验 mean contrast | 模型 mean contrast |
|---:|---:|---:|
| 2.0 Hz | −2.437 N | −0.971 N |
| 2.3 Hz | −2.668 N | −0.817 N |
| 2.6 Hz | −2.780 N | −0.623 N |

Lift 扭转峰后回落 14/14 未被模型捕获，是独立的第二病灶；本轮按单一 claim
纪律不同时处理它。

## 2. 本轮问题

本轮只判：

> 主导推力正偏差由某个现有可动节点的错误力贡献造成，还是现有图中缺少一个
> 随局部来流/分离状态增长的负推力（阻力）组成部分？

在 witness 前禁止把答案预写为 N2、N3、LESP、LEV 或 rig drag。

## 3. 精确 witness 条件

所有条件采用完整 V4.1 生产调用：`nc=12, ns=16, n_cycle=4`，
`steps_per_cycle=wake_rows=spc_of(U,f)`，其余参数与冻结 184 runner 完全相同。

| ID | U | f | nominal twist | AoA | 对比角色 |
|---|---:|---:|---:|---:|---|
| W1 | 6 | 2.6 | 22.5 | 5 | U contrast 低端 |
| W2 | 10 | 2.6 | 22.5 | 5 | U contrast 高端；共享中心 |
| W3 | 10 | 2.6 | 0 | 5 | 高 U 扭转低端 |
| W4 | 10 | 2.6 | 45 | 5 | 高 U 扭转高端 |
| W5 | 10 | 1.4 | 22.5 | 5 | 频率低端 |
| W6 | 8 | 2.6 | 22.5 | 5 | 中 U 与跨图重复物理族负对照 |

预先固定 contrasts：

- \(C_U = W2-W1\)；
- \(C_f = W2-W5\)；
- \(C_{tw,0\to22.5}=W2-W3\)；
- \(C_{tw,22.5\to45}=W4-W2\)。

## 4. 必须保存的通道

每个 witness 保存：

- 报告的 wind-axis L/T 与冻结 baseline 差；
- N1/N2/N3/N4/N6 以及 graph numerical reduction 的 body/wind force；
- 每个 `ForceContribution` 的 role 与 channel；
- claim manifest、implementation hash、claim graph identity；
- force ledger、physical remainder、cycle reduction 与 output invariance guards；
- 完整解析后的调用参数和源码哈希。

N5 是 observer，不得伪造力；N6 的必要 rig drag 可以观察，但 dead-end 身份禁止
将其系数用于吸收 T3b。

## 5. GO / NO-GO

### 数值 GO

- 六点 L/T 均与冻结 baseline 相差不超过 `0.15 N`；
- 所有 guards 通过；
- 节点贡献与总力在现有 ledger 容差内闭合；
- 六点 claim graph identity 完全相同。

### Claim 裁决

对每个 contrast \(c\) 计算：

\[
E=c^T y_{\rm exp},\quad M=c^T y_{\rm model},\quad
M_n=c^T y_n,\quad M_{-n}=M-M_n .
\]

- 若恰一个 partial/open 节点的现有贡献导致结构错误，且移除它在至少两个独立
  contrast 中改善，则判“组成部分错”并挂该父节点。
- 若现有节点均不能产生实验要求的额外负推力方向，而图的物理责任合同中只有一个
  partial/open 父节点拥有该缺失端口，则判“缺组成部分”并挂该父节点。
- 若 N2/N3 多个节点均可解释、仅常数平移可解释、或需要 N6 吸收，输出
  `AMBIGUOUS/NO_DECISION`，不得进入文献候选。

## 6. 禁止

- 不运行参数扫、不开多个候选；
- 不调整 `d_para/cds/Tv/LESPcrit`；
- 不用 Fig19(c,d)；
- 不因本轮推力 witness 顺带修改 lift 峰后回落；
- 不以总力 MAE 直接宣布唯一 claim。
