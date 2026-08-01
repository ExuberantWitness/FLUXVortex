# N2.6e1bc WPJ 公式 oracle 结果

日期：2026-07-30  
范围：执行前审计 `M1` 与输入 lineage 守卫  
裁决：`FORMULA-ORACLE GO / A0 PHYSICS NOT EXECUTED / PRODUCTION OFF`

## 1. 实现身份

- `claim_runtime/svi_dw_weak_uk_2d.py`：
  `8aebe04fbcf50d14276dcaf0efdf224c3f7e9f7fa744a566f6fe48763d3f87f2`
- `tests/test_svi_dw_weak_uk_2d.py`：
  `a5a32c22d6656b3cb80bc982a66584b6d24a0a1332344921b0c4343484e5bfa8`
- 执行前独立审计：
  `406c1f65451305647889ce0f70f4d2a2152dcaaf7ae9bd34a17c22e8f1fded31`

模块不求解 outer/IBL/wake 状态、不输出压力或力，也未接入 V4.1 或
ClaimGraph。

## 2. 已通过

定向单测 `7/7`：

1. wake circulation 使用有向 edge 差，不使用 raw node sum；
2. `mu -> mu+C` 规范平移下 Kelvin 账不变；
3. edge 方向反转时环量符号反转；
4. 远尾迹点涡环量独立记账；
5. 旧 material transport 只检查保留 ID，新生 ID 不被冻结；
6. missing history、非闭合 CV、stage/density 不一致和非法输入
   fail closed；
7. `Gamma_birth/dt` 冒充 vorticity flux、weak-UK backsolve 冒充
   pressure provenance 被拒绝。

全部 `test_svi_dw*.py` 共 `103/103` 通过；`py_compile` 与
`git diff --check` 通过。

## 3. 未通过且不得推断

制造数据上的

\[
\dot\Gamma_b+J_\omega-\Delta p/\rho=0
\]

只检查代码符号和单位，不证明该缩约是物理 moving-CV 守恒式。当前尚未
冻结局部涡量库存、壁面扩散/生成、interface transport、body--wake
compatibility 或可观测 primitive provider。因此：

- 不把本结果称为 A0 equation/limit GO；
- 不授权 A1/A2、solver、Fig17/18/19 或生产接入；
- `N2.6e1bc` 保持 `open`；
- 后继必须先完成
  `n26e1bc_prereg_independent_audit_20260730.md` 的 `M2--M10`。

## 4. Claim 回写

唯一可以冻结的窄结论是：

> 若 `mu_w` 是 material wake 的节点势跃，则 Kelvin circulation 必须由
> 有向 edge 差及另列的远尾迹环量组成；直接求和节点势跃是规范依赖的
> 错误表示。

这是一条表示/账本结论，不是候选气动机理验证。
