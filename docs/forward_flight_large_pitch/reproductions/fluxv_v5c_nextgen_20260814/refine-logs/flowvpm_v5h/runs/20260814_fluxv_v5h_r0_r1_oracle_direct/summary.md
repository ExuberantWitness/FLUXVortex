# FluxV v5h R0–R1 总结

## 结论

R0–R1 通过。隔离的 Python direct 后端在冻结的 Julia FLOWVPM oracle 上复现了 Gaussian-erf 速度/Jacobian、`J^T Gamma` reformulated VPM、三阶段低存储 RK3 和 corrected Pedrizzetti 松弛。所有预注册数值门通过，未使用 clip、NaN 清零、ridge、SFS、黏性或目标数据调参。

这项结论严格限于“出生之后的三维自由涡量输运算子”。它没有证明 LEV/LESP 出生强度是稳定的，也没有接入 Ptera 或计算 Yang、Figure 14、Baik 的气动力成绩。

## 主要结果

- FLOWVPM 官方固定环境测试：14/14 testsets 通过。
- Python 定向与跨语言 parity：24/24 tests 通过。
- 速度/Jacobian relative L2：`1.315e-16 / 8.136e-17`。
- RK 各状态和 RHS 的最坏 relative L2：`2.4557e-16`。
- corrected Pedrizzetti 强度与模长差：均为 `0`。
- clip/nonfinite：`0/0`。
- 当前源码重新评估的 `metrics.json` 与正式工件逐字节一致。
- Julia same-basename JSON 重放逐字节一致；HDF5 因容器字节布局不完全确定，以 `h5diff` 验证语义零差异。

## 资格边界

- `GO`：进入 R2/B2，研究 TE 有向边到向量粒子的保守 shadow bridge。
- `NO-GO`：现在就把该后端用于 LEV 出生、Ptera 受力或三篇论文评分。
- `NO-GO`：把 FLOWVPM 的核、relaxation 或 SFS 当成 v5f `q~1/dt` 出生病态的修复器。
- `DEFER`：FMM、GPU、SFS、黏性、删除/合并及 restart，直到 direct bridge 和制造源收敛通过。

下一步是 B2：先做全局共享边 incidence 和 TE ring→particle shadow 场连续性；只有 owner=1、Kelvin/vector moment 和 probe-field 门都通过，才考虑 exclusive replacement。
