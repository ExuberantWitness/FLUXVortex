# FluxV v5a / v5b Experiment Integrity Audit

日期：2026-08-14  
审查独立性：`same-family`  
接受状态：`provisional`  
总体完整性：**WARN**

## 审查结论

本轮支持保守结论：v5a development proxy 被跨论文门否决；standalone v5b
在无 LEV 时不能退化到当前 FluxV，因此三论文精度必须保持
`blocked_not_scored`。它不支持“v5a/v5b 已经形成一套在三篇论文上优于 v4b
的新算法”。

## A–F 审查

| 维度 | 状态 | 结论 |
|---|---|---|
| A. Ground truth / input identity | PASS | v5a 使用冻结的 Yang、Scherer Figure 14、Baik 实验数据和既有 v4b 对照；v5b 没有伪造论文精度行，而是显式不评分。 |
| B. Metrics / normalization | PASS | v5a 所有 headline 由 CSV 独立重算一致；无相位、幅值或偏置拟合。v5b G1 的 CL/CD 均值与相位最大差由原始 20 点曲线重算一致。 |
| C. Artifact/hash integrity | WARN | 最终四组工件中 manifest 已声明的 48 个哈希全部匹配；最终图 manifest 使用相对路径并覆盖 6 输入、6 图件和 LaTeX。v5a/v5b 数值 manifests 仍不是完整传递依赖闭包。 |
| D. Pipeline/model identity | WARN | G1 是公平且决定性的外部对照。G4 只是内部通道重组到舍入误差，不是独立力守恒证明；`unique_force_owner` 也不是仓库级静态证明。 |
| E. Scope/robustness | WARN | G5 是看到旧诊断失败后加入的四级 post-hoc 细化；局部阶次不单调，不能称渐近 `O(dt)` 证明。G6 只是 `not_run` 占位项。 |
| F. Data type / phantom-result check | PASS | v5a 是 simulation prediction 对真实实验真值；v5b 仅报告机械门，没有把缺失的论文结果填成零或代理结果。 |

## 独立重算

- v5a Yang lift/drag MAE：`3.951385438 / 2.062566960 gf`，相对 v4b
  为 `0.867576 / 0.780094`；只在该任务改善。
- v5a Figure 14 RMSE：`0.094695507`，相对 v4b 为 `3.649270×`。
- v5a Baik filtered macro CL/CD RMSE：`0.801648991 / 0.404409064`，
  相对 v4b 为 `1.219160× / 1.171683×`；8 个 case/channel 无一改善。
- v5a gate：`6/18` numeric、`0/18` promotion。
- v5b G1：最大相位差 `CL=0.556435291`、`CD=0.528628967`，要求
  `≤1e-12`，FAIL。
- v5b G4：内部账本重组最大残差 `7.105427358e-15`；只能作内部算术检查。
- v5b G5：全局拟合 `p=1.193736011`，局部阶次
  `0.216/1.749/1.431`；只能作 post-hoc development diagnostic。
- v5b G6：NOT RUN；当前 runner 中还是 fail-closed 占位项。

## 可复现性

- 最终目标测试：`75 passed, 6 subtests passed`。
- v5a、no-force v5b、force-gate v5b 与最终图共 48 个已声明哈希全部匹配。
- 最终三份 PDF 由第二个独立输出目录重生成后逐字节一致。
- 数值缺口：v5a 前一周期状态、no-force 四级原始出生样本、G4 原始逐步
  力账本未全部归档；完整 import graph、完整包版本与 dirty diff 也未全部冻结。

## Verdict

`WARN / provisional`。没有数值或哈希不一致；WARN 来自证据语义与传递
provenance 不完整。当前 no-go 结论可信，但任何“v5b 已验证”“G4 独立守恒通过”
或“G5 已证明渐近收敛”的更强表述均不成立。

