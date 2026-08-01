# Actual-body 类型化求积证书案卷：独立综合审计

**时间**：2026-07-28T18:29:51+08:00  
**审计对象**：
`research_actual_body_typed_quadrature_certificate_20260728_181846.md/json`  
**锁定 SHA256**：

- Markdown：`2d98c1f26d796b8f40118b2bd5feee3e30ccfd1306bcc6cd69964618594001f0`
- JSON：`978ce2248dc60817c980e7f822ffe1d3258af31b7e2d06d5414bc83873f66e75`

**最终裁决**：`PASS`  
**blocker**：0  
**review_independence**：`fresh-context / same-family`  
**acceptance_status**：`provisional`

## 审计范围与结果

1. **代码一致性：PASS**
   - 与 `actual_boundary_p2_galerkin.py`、
     `actual_boundary_body_wake.py`、
     `actual_wake_reachable_pressure.py` 一致；
   - 正确识别 touching subregion 的当前 6/6/2 实现、浮点 solve/backward
     residual、共享 quadrature bias 和 mesh 未检查几何自交等边界。
2. **数学闭合：PASS**
   - \(K_A/e/r_\phi\) verified-solve 半径、induced norm 与 verified
     spectral-radius upper bound 的角色正确；
   - velocity operator \(G_v\) 与 contraction matrix \(K_A\) 已分型；
   - \(M_{\rm dyn}\) 与 \(M_{\rm metric}\) 已分离；
   - 固定 verified-SPD \(Q\) 下的 \(2^7\) 顶点上界、凸盒 QP 下界正确；
   - interval \(Q\) 使用联合 validated optimization 或
     \(Q_c\pm\epsilon I\) 修正，且 lower-side 必须 verified PSD/SPD。
3. **文献适用域：PASS**
   - 文献只授权候选积分规则、误差分析和 verified arithmetic；
   - 没有把 flat/constant-density/静态结果外推成当前 moving P2
     body-wake/pressure 的现成证书。
4. **Claim/governance：PASS**
   - d19a–d19d 的 run/campaign 分工合理；
   - 正确指出深节点未递归治理、missing guard fail-open 和 campaign phase
     缺失；
   - 没有把 campaign authorization 当 `StepContext` 或 `ForceLedger` 物理量。
5. **非主张边界：PASS**
   - 没有声称正式 fixed-P2 result 已通过；
   - 没有声称证书已实现；
   - 没有声称 Fig17/18/19 已完成或精度已提升。

## 非阻断 WARN

- 实现时应显式对 quadratic bound 和单调平方根做 outward rounding；
- JSON 将来可展开 31 histories、22 fields 和 15 checks，而不只保留压缩 guard
  名称；
- 历史 `/proc` 快照不是不可变原始证据，但已在审计时现场复核，且主案卷只将其
  作为历史状态陈述。

## 授权边界

该 PASS 只认可研究案卷的代码映射、数学方向和非主张边界。它不代表：

- 类型化积分 enclosure 已实现；
- verified solve 已实现；
- d19 claim 可晋升；
- 正式 one-shot 已完成；
- h/p continuum gate 已通过；
- forming/VES 或 Fig17/18/19 获得授权。

