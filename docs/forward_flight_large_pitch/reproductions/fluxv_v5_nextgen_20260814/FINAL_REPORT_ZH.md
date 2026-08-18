# FluxV v5a / v5b 联合实现与验证报告

日期：2026-08-14  
最终状态：**v5a 被冻结试验否决；v5b 已实现共享尾迹与单一压力载荷原型，但在进入三论文精度比较前触发 NO-GO。FluxV v4b 仍是当前经过三论文验证的推荐版本。**

## 1. 执行范围

本轮严格按“先 5a、再 5b、最后联合汇报”的顺序执行：

1. 实现 v5a 的 equilibrium residual、成对 LDVM transient residual、两状态对流高通及单一 `qS` 账本；
2. 先跑冻结代表工况，执行预先规定的 stop condition；
3. v5a 停止后，独立实现 v5b 的 UVLM bound AIC、TE/LE material wake、LESP 约束、Hirato Eq. 9/17/24、唯一 surface-pressure force ledger；
4. 在任何 Yang / Izraelevitz Figure 14 / Baik 精度打分前，执行 no-LEV 退化、力账本和出生极限门。

没有根据三篇论文的观测曲线调 `LESPcrit`、核半径、相位、幅值或偏置；失败结果没有隐藏。

## 2. v5a：实现完成，但假设被数据否决

v5a 的公式、极限与账本单测通过；冻结 cache smoke 使用
`kinematic_proxy + projected_integrated_proxy`，因此是 development proxy，
不是尚缺 UVLM-induced strip velocity 的 canonical 条带实现。

| 任务 / 通道 | FluxV v4b | FluxV v5a | v5a/v4b | 结果 |
|---|---:|---:|---:|---|
| Yang lift MAE [gf] | 4.554510 | 3.951385 | 0.868 | 改善 |
| Yang drag MAE [gf] | 2.643997 | 2.062567 | 0.780 | 改善 |
| Izraelevitz Fig. 14 CT RMSE | 0.025949 | 0.094696 | 3.649 | 严重退化 |
| Baik filtered macro CL RMSE | 0.657542 | 0.801649 | 1.219 | 退化 |
| Baik filtered macro CD RMSE | 0.345152 | 0.404409 | 1.172 | 退化 |

18 个 numeric/audit gate 仅通过 6 个，promotion gate 为 0/18。Figure 14
和 Baik 两篇同时失败，触发计划中的 stop condition。因此没有继续 v5a full，
也没有围绕目标曲线调 `lambda_tau`。

结论：**“完整 equilibrium residual + 只高通 integrated LDVM discrepancy”
不是三论文统一改进方案。**

## 3. v5b：代码完成到唯一力账本，但晋级门失败

### 3.1 已实现内容

v5b 当前包含以下隔离模块：

- shared TE/LE material-wake 状态与 UVLM bound solve；
- Hirato Eq. 6、7、9、17、24；
- chronological previous/current time-layer；
- Ptera structured GP1 geometry、材料速度、完整对称翼网格及风轴转换适配器；
- 唯一 surface-potential pressure ledger；
- 逐面板压力、力、力矩和全局闭合守卫；
- no-LEV N1 内部基线、active LEV sequence 和物质环量不可变性测试。

联合 v5/Hirato/Ramesh 相关回归均通过；源码冻结后的最终目标测试集合为
`75 passed, 6 subtests passed`。这些测试证明接口、守恒与 fail-closed
契约成立，不等价于三论文精度验证。

### 3.2 晋级门

| 门 | 测量值 | 判定 |
|---|---:|---|
| G1：LEV 关闭时逐点退化到**当前 FluxV** | `max(ΔCL,ΔCD)=0.556435`，要求 `≤1e-12` | **FAIL** |
| G4：内部压力/力账本重组残差 | `7.105×10⁻15`，要求 `≤1e-12` | INTERNAL PASS |
| G5：四级光滑跨阈值细化 | 全局拟合 `p=1.193736` | DEV PASS（post-hoc） |
| G6：高 AR Ramesh 力一致性 | 因 G1 失败未运行 | NOT RUN |

G1 使用 Yang 2025、AoA 15°、20 step/cycle 的同一 Ptera movement 输入，
并把 `LESPcrit=10` 以保证全程无 LEV。当前 FluxV 与 standalone v5b 的末周期结果为：

| 模型 | mean CL | mean CD |
|---|---:|---:|
| 当前 FluxV | 0.720634 | -0.023772 |
| standalone v5b（LEV disabled） | 0.486298 | 0.144652 |

相位最大绝对差为 `ΔCL=0.556435`、`ΔCD=0.528629`。这不是小的数值误差，
而是因为 standalone v5b 同时换掉了 Ptera prescribed TE wake、尾迹时间语义
和 Ptera Kutta–Joukowski/unsteady-ring load owner。它不是“在原 FluxV 上只增加
LEV”的隔离改动。

因此即使 v5b 内部守恒账本闭合，也不能把它作为下一代 FluxV 在三篇论文上
打分。按合同，G1 失败后停止 G6 和 cross-paper 运行，三论文 v5b 精度保持
`blocked_not_scored`。

当前 force-gate runner 中的 G6 只是 fail-closed 的 `not_run` 占位项，并没有
实现可调用的高 AR Ramesh 力校核。因此即使后续修复 G1，也必须先实现 G6，
不能直接把该 runner 的状态改成 promotion。

### 3.3 出生极限口径修正

最初无力 smoke 用“在 `t=0` 突然放到 15°”的非光滑运动检查出生极限，得到
`p=0.000164`，接近常数。该试验混入了有限初始跃变，不能验证光滑 DVM 出生极限。
随后用解析速度的 quintic 0→20°光滑启动、四级
`Δt={0.01,0.005,0.0025,0.00125}s` 重新检查，得到全局拟合
`p=1.193736`。但四级局部阶次为 `0.216/1.749/1.431`，首个越阈值事件的
物理时刻也不完全相同，因此这只是 post-hoc development diagnostic，**不是**
渐近 `O(Δt)` 证明或预注册盲验。

同样，G4 的 `7.105×10⁻15` 是同一压力通道的内部重组闭合；它能发现账本
装配错误，但不是独立的力守恒验证，也不能单独证明全局不存在第二力源。本轮只
声称“runner 内每个提交步调用一次该账本”。

## 4. 联合结论

本轮没有得到一套可以诚实宣称“在三篇论文上均优于 v4b”的新模型：

- v5a：能改善 Yang，但明显破坏 Figure 14 和 Baik；
- v5b：共享尾迹和唯一压力力账本已经实现，机械内部闭合，但尚未保持当前
  FluxV 的 no-LEV 基线，因此没有资格进行三论文精度比较；
- FluxV v4b：仍是当前三论文证据中性能最好、可继续使用的版本，但其 Baik
  尾迹敏感性与几何替代限制仍需保留。

这次结果排除了两个看似自然、实际不泛化的捷径，也明确了下一步唯一合理的路径：

1. 在 **Ptera 原生 solver/time layer** 内实现 shared TE/LE wake，而不是外置 N1
   load solver；
2. LEV 关闭时必须逐步、逐面板、逐载荷通道精确返回当前 FluxV；
3. 只在同一个 AIC / TE state / pressure owner 中做 LEV off/on 消融；
4. 先通过高 AR Ramesh onset/force gate，再冻结 Yang、Figure 14、Baik 的完整矩阵；
5. 任何改动都不得重新叠加 v4b LDVM force discrepancy、v5a polar residual、impulse
   或其他第二力源。

## 5. 主要产物

结果根目录：

`/tmp/fluxv-v5-nextgen/docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/`

冻结结果：

- v5a：`runs/20260814_fluxv_v5a_cache_smoke_frozen/`
- v5b 无力拓扑：`runs/20260814_fluxv_v5b_no_force_smoke_frozen/`
- v5b 力晋级门（源码冻结后重跑）：
  `runs/20260814_fluxv_v5b_force_gate_reproducible/`
- 最终联合图（引用上述可复现力门）：
  `runs/20260814_fluxv_v5_joint_final_verified/`

旧的 `force_gate_frozen` 与 `v5_joint_final` 目录保留为开发历史，不作为最终
证据入口。

完整性口径：所有 manifest **已声明**的 48 个哈希均匹配，但 v5a/v5b manifests
尚不是传递依赖闭包；G4 原始逐步账本和 v5a 前一周期状态也未全部归档。因此本轮
审计等级为 `WARN / provisional`，而不是“完整可重现性 PASS”。

关键代码：

- `platform/forward_flight_benchmarks/fluxv_v5a.py`
- `platform/forward_flight_benchmarks/fluxv_v5b_shared_wake.py`
- `platform/forward_flight_benchmarks/fluxv_v5b_force.py`
- `platform/forward_flight_benchmarks/fluxv_v5b_sequence.py`
- `platform/forward_flight_benchmarks/fluxv_v5b_ptera_adapter.py`
- `platform/forward_flight_benchmarks/run_fluxv_v5b_force_gate.py`

独立审查：

- `EXPERIMENT_AUDIT.md` / `EXPERIMENT_AUDIT.json`
- `CLAIMS_FROM_RESULTS.md`

本报告中的“不打分”是主动的实验完整性约束，不代表缺少执行；它表示 v5b 在
进入三论文精度声称前已经被更基础、也更严格的退化测试阻断。
