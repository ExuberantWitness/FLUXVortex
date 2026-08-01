# Actual-body h/p → forming → VES 预注册审计

**审计时间**：2026-07-28T17:47:43+08:00  
**内容裁决**：`PASS`  
**assurance**：`PROVISIONAL / REUSED REVIEWER CONTEXT / NOT FRESH`  
**执行裁决**：`NO-GO`  
**审计方式**：只读；未修改文件，未启动正式或生产计算  

## 1. 结论

`actual_body_hp_forming_ves_preregistration_20260728_173708.md/json` 的研究顺序、
13 点矩阵、15 个上游 checks、13/39 history 账、boundary-specific forming、
VES necessity、ForceLedger 身份、claim effect 和 hard nonclaims 相互一致。
审计后补冻 result-interpretation contract JSON；post-revision 复核没有发现新
blocker。

本 PASS 只证明预注册内容自洽。reviewer 复用了此前 claim/hp tree 审计上下文，
因此不能充当文档所要求的 fresh independent **result** review，也不能解锁
任何计算、claim 改写或 production。

## 2. 确定性检查

| 检查 | 结果 |
|---|---|
| timestamp/latest Markdown `cmp` | 0 |
| timestamp/latest JSON `cmp` | 0 |
| Markdown SHA256 | `03efdce13d0aba57a9412f06416d51e664b74b88926b140a2abe3be502e8314e` |
| JSON SHA256 | `6042083f22d8d67d7d0b330f81be23ad468b2ffcf2a053cd5c8bf3a317163513` |
| JSON parse | PASS |
| 冻结 source hashes | 11/11 match |
| aggregate checks | 15/15，唯一且与合同顺序一致 |
| h/p points | 13 unique IDs / 13 unique `(h,p,dt,q)` tuples |
| mixed cube | `H1/H2 × p2/p3 × dt{0.125,0.0625}`，`q=12` |
| axis/tail points | `AH0/AP1/ADT/AQ8/AQ10` 完整 |
| zeroth-order 账 | `13 × zero = 13` |
| first-order 账 | `13 × {zero,+epsilon,-epsilon} = 39` |
| authorization flags | execution=false / production=false / claim-state-change=false |

## 3. 科学边界复核

- \(p\) 明确定义为 body/cut/wake/transport/pressure 的一致多项式阶次，
  没有与 quadrature order \(q\) 混淆。
- formal 终止、15 checks、确定性复算、fresh review、resolved branch、
  P1/P3 同弱式 oracle 和守恒 common-space projection 都是执行前硬门。
- sharp branch 只使用二维 finite-angle Xia oracle；finite base 保留双 front
  与 base/confluence；smooth LE 先要求 profile/edge、material spike 和相对
  IBL flux。
- VES 只在 massless geometry 后仍有 h/p-stable nonzero cokernel 且存在独立
  mass/momentum/entrainment 证据时打开。
- raw 10 components 没有被写成全局状态维数；零质量极限明确要求
  \(\rho_s\to0\Rightarrow q\to0,\Delta p\to0\)。
- `ForceLedger` 正确属于 ClaimGraph 图级聚合，不属于 N4。
- 当前阻塞是气动状态/压力 provider，不是结构模型。
- 不使用 L/T/Fig17/18/19 选择网格、阶次、核、forming 或 VES。

## 4. 仍然关闭的三把执行锁

1. 正式 canonical result 尚未生成并完成 15-check 确定性复核。
2. fresh independent result review 尚未发生；本审计不能替代它。
3. P1/P3 同阶 body/cut/wake/transport/pressure provider 与 oracle 尚不存在。

冻结时 PID 922252 仍在运行；canonical 与 latest result 均不存在。该状态只
说明 formal 任务尚未终止，不提供物理判决。

## 5. 非阻断、后续必须预注册的内容

- M0 executable cases YAML 中冻结 H0/H1/H2 mesh hashes、faces/DOFs 和构造器；
- 冻结 continuum uncertainty ball、rank/cokernel 阈值、repeat allowance；
- 在相应子门前冻结 held-out field histories；
- M0 persistent witness 后才可独立冻结 M1 的 full-wing NACA-2406 网格；
- 不得事后用 Fig17/18/19 选择这些定义。

## 6. Post-revision 证据

- contract JSON SHA256：
  `6e95b9ec4c51f47f432ce0e3e96c7647e5bf5e9949d877e4be4750ac5ef643cd`
- 该值已同时出现在预注册 Markdown 与 JSON；
- 11/11 source hashes 复算一致；
- revision 后没有新增 blocker。

