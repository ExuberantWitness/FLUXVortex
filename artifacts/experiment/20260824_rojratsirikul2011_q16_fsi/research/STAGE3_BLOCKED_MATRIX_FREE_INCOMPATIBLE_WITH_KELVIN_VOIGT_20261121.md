# Stage 3 被结构性阻塞：Q16 的矩阵无关路径与 Kelvin-Voigt 阻尼互斥（2026-11-21）

> 起因：外部审计 §三.3 建议"用稀疏结构切线 + `Mf1` 低秩 Woodbury 替代 47 批列探测"。
> 我核实的结论曾是"这条路只需选一个已存在的开关"。**这一条在构造期就失败了。**

## 1. 实测

```
FLUXV_Q16_NONSYMMETRIC_SOLVER=gmres \
  python3 platform/warp_vpm/profile_roj_q16_v5m.py --case ROJ11-A16 --steps 3 --skip-profiler
→ ValueError: stiffness_damping_coefficient requires the reference-dense solver
  at src/fluxvortex/warp_fsi/q16_structural_solver.py:515
```

对照：默认路径（`reference_dense`）同一命令 **exit 0**，`last_record.cn = 0.05395471777078108`（两次切片逐位相同，重复性 1e-12 通过）。

## 2. 原因（代码级）

- `src/fluxvortex/runtime/case_runner.py:385` 给 Roj 统一配置设了**正的** `stiffness_damping_coefficient`（Kelvin-Voigt θ·K_ref，对应作者 MATLAB 的 `theta_a*(Qd_eps + Qd_k)`）；
- `q16_structural_solver.py:505-517`：阻尼力与 Newmark 有效贡献**复用缓存的参考切线**，因此当 `θ > 0` 且 `nonsymmetric_solver != "reference_dense"` 时直接 `raise`；
- 于是 `_gmres_solve`（`:1051`）虽已实现、`_cg_solve`（`:936`）虽可用、`_effective_action` 虽接受任意 `AccelerationLoadAction`，**对 Roj 工况都不可达**。

## 3. 两条解锁路径（都不许在本轮做）

1. **把 θ·K_ref 阻尼项实现进矩阵无关动作**：物理承载项，必须与稠密路径逐位等价，需要单独的推导 + 红测（不是开关改动）；
2. **设 θ = 0**：**改物理**，禁止。

⇒ 审计的"8~15× 结构求解提速"在 Roj 工况上**既未被证实也未被否证**，它需要先做 (1)。这不是"选择开关"级别的工程项。

## 4. 由此产生的优先级变化

Stage 3 从"~5 分钟的 A/B"变成"需要先实现阻尼项的独立立项"。因此**下一步改为 Stage 4**（digest 穿参 + `.item()` 合批），它没有这类结构耦合，且判据干净（parity 6/6 + digest 序列逐字节相同）。

一个**仍未做**的廉价诊断（下次可做，约 40 行 + 1 分钟）：绕开 `case_runner`，直接用 `Q16CudaNewmarkStepper(θ=0)` 对同一个模型/载荷跑 3 个结构子步，两种求解器对比增量与单步耗时。它只回答"矩阵无关路径数值上是否等价、是否更快"，**不触碰生产配置**（θ 仍为生产值）。这样能在决定是否投入 (1) 之前，先知道值不值得。

## 5. 边界
- 本记录只证明"生产配置下不可达"，**没有**证明矩阵无关路径数值上不等价或更慢；
- 未修改任何生产数值；`FLUXV_Q16_NONSYMMETRIC_SOLVER` 默认仍是 `reference_dense`，parity 6/6 未被影响（本次未跑 parity，因为**没有**代码改动影响生产路径——只加了一个未被选中的 env 读取）。
