# Review Summary

**Problem**：在 Yang 2025、Scherer/Izraelevitz Figure 14 和 Baik W1–W4 三篇开发论文上同时改进 FluxV v4b。  
**Date**：2026-08-14  
**Rounds**：4 / 5  
**Final Score**：9.21 / 10  
**Final Verdict**：READY

## Round-by-Round Resolution

| Round | Main concern | Resolution | Status |
|---:|---|---|---|
| 1 | shared wake、owner、apparatus一次全做，三重双计风险 | 首版收缩为 v5a；v5b 与 apparatus 移出 | resolved |
| 2 | duplicate activity gate；时变时间常数；2D/3D baseline混账 | 删除 gate；对流坐标状态；同截面 residual | resolved |
| 3 | equilibrium residual误用LDVM投影；Kelvin声明过强 | 分离两条空间映射；Kelvin只作底层各自回归 | resolved |
| 4 | final formula and claim audit | 无blocking issue，进入实现 | READY |

## Final Status

- Anchor：preserved
- Focus：tight
- Modernity：intentionally deterministic; no forced ML component
- Strongest feature：一个 UVLM 骨架、两个互斥 residual、一个新增对流状态
- Remaining risk：`lambda_tau` 是 development hypothesis；截面 `Lcrit` 仍是 source-conditioned transfer；真正泛化需第四篇盲测。
