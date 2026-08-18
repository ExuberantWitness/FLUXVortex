# FluxV v5h R0–R1 schema-v2 总结

## 结论

R0–R1 在修正证据 schema 后仍通过。Python direct 后端复现了固定 FLOWVPM 的 Gaussian-erf U/J、reformulated VPM、低存储 RK3 与 corrected Pedrizzetti；近场独立 probe、随 step 时间变化的来流和全部配置契约也通过。

schema-v2 消除了旧工件的歧义：`pre/post` 只保存状态，只有 `rhs` 保存并声明有效的 U/J。自动测试会重新计算整份 `metrics.json`，并在 RK 系数、转置、formulation 或 SFS/黏性开关被改变时失败。

## 数值证据

- FLOWVPM 固定环境官方测试：14/14。
- Python unit/parity/config 测试：29/29。
- full/probe U relative L2：`1.315e-16 / 2.042e-16`。
- full/probe J relative L2：`8.136e-17 / 8.781e-17`。
- `r/sigma=1e-4...2` 近场最坏逐点 U/J relative L2：`1.901e-15 / 2.943e-15`。
- fixed-Uinf 六个 RK stage 最坏 relative L2：`2.456e-16`；affine step-time Uinf 为 `0`。
- corrected relaxation、clock、clip、nonfinite：全部为 `0`。
- fresh JSON 与 metrics 均逐字节重现；fresh HDF5 与正式 HDF5 的 `h5diff` 为零。

## 裁决

- `GO`：R2/B2 的全局有向边 TE shadow bridge。
- `BLOCKED`：exclusive ring→particle ownership、Ptera 双向耦合、LEV source 和气动力评分。
- `NO-GO`：用 rVPM 核/SFS/relaxation 掩盖 v5f 的 `q~1/dt` 出生病态。

该结果只支持“冻结 fixtures 上的 direct 三维涡量输运数值 parity”，不支持长期稳定、三维翼耦合或三篇论文精度声明。
