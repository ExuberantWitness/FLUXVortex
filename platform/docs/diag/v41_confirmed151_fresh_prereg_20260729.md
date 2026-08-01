# V4.1 fresh confirmed151 基线预登记

**日期**：2026-07-29  
**触发证据**：
`v41_seed_w2_sequence_repro_20260729_131106.json`
(`sha256:56cd350a8085668384994394943a63c87d36688ea1e71c23ce932c225347a59d`)  
**分类**：`CACHE_DRIFT`  
**状态**：PRE-REGISTERED；尚未启动 151 点

## 1. 必要性

旧合并 184 使用 85 个 confirmed 冻结种子与 66 个 fresh Fig18 点。W2 的三次
受控重复：

```text
L = 10.682763 / 10.654610 / 10.654610 N
T =  0.423502 /  0.415309 /  0.415309 N
```

重复 spread 为 L `0.028153 N`、T `0.008193 N`，无运行分叉或序列依赖；但
mean 相对旧缓存 L 偏 `+0.169179 N`，超过冻结 `0.15 N` 门。因此旧 seed 不再
拥有当前执行身份下的 authoritative baseline 资格。

## 2. 运行合同

- 条件集合：`EVIDENCE_CONFIRMED` 的全部 151 个唯一条件；
- 构成：Fig17 全部、Fig18 全部、Fig19(a,b) 全部；
- Fig19(c,d) 及其 33 个 conditional-only 条件不运行、不评分；
- **禁止复用任何旧 L/T seed**；
- 每点采用与 full184 相同的 V4.1 生产调用：
  `nc=12, ns=16, n_cycle=4`,
  `steps_per_cycle=wake_rows=spc_of(U,f)`；
- 首次 anchor 只作 cold preconditioner；第二次 warm anchor 作为 151 点中的
  正式结果；
- 每个正式条件调用一次。W2 的同进程稳定性已由独立三重复审计证明。

## 3. 每点证据

每次调用必须原子保存：

- `L_wind/T_wind`；
- N1/N2/N3/N4/N6/R0 的 channel、role、body/wind force；
- claim manifest hash 与共同 graph identity；
- force ledger、physical remainder、cycle reduction、output invariance guards；
- 相对旧合并 baseline 的差只作诊断，不作 fresh 运行失败门；
- wall time 与执行顺序。

每次 GPU 调用前后验证 aerodynamic solver、claim runtime/YAML 和基础输入的
源码哈希。运行期间禁止修改这些文件。

## 4. 输出

使用同一 timestamp 的三个 versioned 文件：

1. `s6_sweep_v41_confirmed151_fresh_<ts>.json`：151 个 L/T；
2. `fig171819_v41_confirmed151_fresh_manifest_<ts>.json`：运行身份、guards；
3. `fig171819_v41_confirmed151_contributions_<ts>.json`：逐点节点通道。

禁止覆盖旧 118、旧 184、旧 scorecard 或 fixed-name production 文件。

## 5. GO / NO-GO

### 数值 GO

- 151/151 条件恰好完成，无额外 key；
- 所有 L/T 有限；
- 所有 guards 通过；
- 所有点 graph identity 相同；
- 节点力重算与报告总力在 `1e-9 N` 内闭合；
- 运行前后 solver source hash 集完全相同；
- confirmed coverage 为 42/42 曲线、151/151 条件。

### 后续授权

GO 后使用已冻结 scope-v3 scorer 重新生成 fresh confirmed42 的 434 点基线和
残差指纹。旧 provisional fingerprint 不得继续用于定量 claim 归因。

### NO-GO

任一点失败即停止并保留 checkpoint；只允许在同一源码与运行合同下显式 resume。
禁止用旧 seed 填洞、插值失败条件或放宽 guard。
