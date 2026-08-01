# R0 第二次执行门预登记：同调用内气动输出不变量

时间：2026-07-29（Asia/Shanghai）  
前序结果：`r0_gate1_nogo_20260729.md`

## 为什么需要第二个门

Gate 1 证明历史 CUDA anchor 不能满足未经验证的跨进程逐 bit 控制，但 probe
精确重现了旧输出和旧 reducer 差值。R0 的可动空间严格位于
`result` 气动力字段已生成之后。因此有效的因果控制必须在同一次 solver 调用
内比较 ClaimGraph 执行前后的既有输出，而不是比较两个不同 CUDA 轨迹。

这不是把 Gate 1 改判为通过；Gate 1 永久保留 NO-GO。Gate 2 是在再次执行前
注册的新实验。

## 唯一候选保持不变

- `q_num = reported robust total - arithmetic total`
- `physical remainder = arithmetic total - classified physical channels`
- N1 `uvlm_remainder = 0`
- R0 从唯一 canonical `numerical_cycle_reduction` 同时发布状态和记账
- 不改变任何气动公式、常数、网格、运动学或已有结果字段

## 新的可辨识 go/no-go

对 `anchor → anchor → probe` 每次调用：

1. 在构建旧兼容 `result` 后、执行 ClaimGraph 前，冻结
   `L, Fx, T, P, Fx_body, Fz_body, L_body, T_body_f, L_wind, T_wind`
   的 IEEE-754 二进制值；
2. ClaimGraph 和 R0 执行后，上述字段必须全部逐 bit 相同；
3. 跨进程 anchor 相对冻结 118 基准仍必须 `<=0.15 N`；
4. probe R0 向量与 Gate 1/旧 remainder 最大差 `<=1e-12 N`；
5. 每次 `unclassified_physical_force <=1e-9 N`；
6. R0 发布状态与 ForceLedger 必须来自同一个 canonical 数组；
7. N1/N4 指纹不变，R0 最终指纹独立固定；
8. runner 四项账本守卫、静态测试和 frozen-source 审计全部通过。

任一失败即 Gate 2 NO-GO；不得继续 66 点。
