# V4.1 fresh confirmed151 断点证据完整性附录

**日期**：2026-07-29  
**上位预登记**：
`platform/docs/diag/v41_confirmed151_fresh_prereg_20260729.md`  
**触发审计对象**：
`platform/lb_sweep151_fresh.py`
(`sha256:d5bd099c8463ca2ac760d22bd37bf3d093cd8821cc09b293d95e0bd837e439ba`)  
**状态**：PRE-REGISTERED FIX；正式 151 点尚未启动

## 1. 触发反例

独立只读审计证明旧 `_valid_saved_case()` 会错误接受以下 checkpoint：

1. 将结果 `L/T` 改为 `+999999/-999999 N`；
2. 清空 `claim_contributions`；
3. 将另一工况的 evidence 接到当前 result。

因此旧 runner 为 **NO-GO**。本附录只修复运行证据身份，不改变 V4.1
气动公式、常数、网格、运动学、claim 状态或实验数据。

## 2. 必修合同

每个已保存工况在 resume 和最终完成前都必须由原始证据重新验证：

- result 必须恰含有限的 `L/T`；
- 外层 key、evidence condition、`condition_key` 和完整 `resolved_call`
  必须对应同一预登记工况；
- old-baseline 值及 signed delta 必须由权威旧 baseline 和当前 result
  重新得到；
- guards 必须全部通过，并与 manifest 的 per-case guards 一致；
- graph identity 与共同 claim manifest 的 identity 必须一致；
- claim manifest hash 必须由共同 manifest 重新计算；
- raw contributions 的节点集合必须恰为
  `N1/N2/N3/N4/N6/R0`，每个节点均有非空、有限、三维 body force；
- contribution summary、body ledger 和 wind force 必须从 raw contributions
  重新生成，禁止信任保存的 `max_*_error_N`；
- 重算总力必须在 `1e-9 N` 内闭合到当前 result 的 `L/T`。

三份 checkpoint 必须具有相同 run id、合法 schema/path/status，且在任何 GPU
调用前满足：

```text
set(results) == set(case_evidence) == set(case_guards)
```

unexpected key、orphan 和验证失败 case 必须同步移除并立即原子 checkpoint；
不允许保留跨文件错配。

## 3. 执行身份与耐久性

- 冻结完整 solver/control 路径集合和哈希；文件新增、删除或内容改变均失败；
- 将被 `run_claim_witnesses.py` 直接导入的 `lb_sweep184.py` 纳入 control
  source；
- 冻结 Python executable/version、NumPy、Warp、关键环境变量、运行 device/dtype
  和 151 工况完整 resolved-call contract hash；
- 每次 GPU 调用前后复核 source 集；
- atomic JSON 使用同目录唯一临时文件、file fsync、replace、directory fsync；
- warm anchor 失败时先保存 cold/warm 数值、guards、source 检查和 failed
  manifest，再抛出 NO-GO。

## 4. 预登记反例测试

无需 GPU 的测试必须至少证明：

1. 合法 case 通过；
2. 篡改 `L/T`、raw contribution、summary、ledger、guards、condition、
   resolved call、graph identity、manifest hash 均失败；
3. 跨工况 evidence 接线失败；
4. result/evidence/guard orphan 在 GPU 前被拒绝或同步清理；
5. solver/control 文件新增、删除或哈希变化失败；
6. runtime identity 或 call-contract 变化失败；
7. 最终 151 点在写 complete 前全部重新验证，三文件精确同键；
8. cold/warm anchor 失败会留下可审计 NO-GO 记录。

只有上述测试通过并获得独立复审 GO，才授权启动 fresh151 GPU campaign。
