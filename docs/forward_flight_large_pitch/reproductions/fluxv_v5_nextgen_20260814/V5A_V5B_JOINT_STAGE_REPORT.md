# FluxV v5a / v5b 联合阶段报告（证据骨架）

> 历史阶段快照。本文件早于 v5b 力晋级门，最终结论以
> `FINAL_REPORT_ZH.md` 为准；最终证据目录为
> `20260814_fluxv_v5b_force_gate_reproducible`。

日期：2026-08-14  
状态：**v5a 已按 stop condition 停止；v5b 仅通过无力耦合机械门，三论文精度未评分。**

## 1. 当前可支持的结论

1. v5a 的冻结 cache smoke 不是可晋级模型。它使用
   `incidence_source=kinematic_proxy` 与
   `aggregation_scope=projected_integrated_proxy`，因此不满足 canonical
   strip-local、UVLM-induced-incidence 路径。
2. v5a 在 Yang 均值任务上相对 v4b 改善，但在 Izraelevitz Figure 14
   与 Baik W1--W4 上明显退化。按预注册 stop condition，停止 v5a full 与调参；
   18 个 promotion gate 中通过 0 个。
3. v5b shared-wake 的 G0--G2 无力 smoke 共通过 18/18 个机械门。这只证明
   Ramesh 参考保护、关闭态恒等、Hirato/LESP 拓扑与环量账本在该 smoke
   范围内闭合。
4. 当前冻结的 v5b **no-force shared-wake smoke** 返回
   `force_coupling=not_implemented`。因此这份 run 中 Yang、Figure 14、Baik
   的载荷精度状态必须保持 `blocked_not_scored`；不能用它宣称 v5b 的升力、
   阻力或推力优于任何模型。后续独立 force-coupled sequence run 必须另行评分。

## 2. v5a 冻结结果与停止理由

下表直接来自冻结 `case_metrics.csv`。误差越低越好；Baik 数值为 W1--W4
的 1 Hz filtered waveform RMSE 宏平均。

| 任务 / 通道 | FluxV v4b | FluxV v5a dev proxy | v5a / v4b | 判定 |
|---|---:|---:|---:|---|
| Yang lift MAE (gf) | 4.554510 | 3.951385 | 0.868 | 改善，但不足以晋级 |
| Yang drag MAE (gf) | 2.643997 | 2.062567 | 0.780 | 改善，但不足以晋级 |
| Izraelevitz Fig. 14 all-14 CT RMSE | 0.025949 | 0.094696 | 3.649 | 明显退化，触发 stop |
| Baik filtered macro CL RMSE | 0.657542 | 0.801649 | 1.219 | 退化，触发 stop |
| Baik filtered macro CD RMSE | 0.345152 | 0.404409 | 1.172 | 退化，触发 stop |

该结果只能描述冻结 integrated-cache 适配器，不能代表正式 v5a canonical
空间条带模型的精度。由于两个论文方向失败，按计划不继续跑 v5a full，也不在
这些任务上调参。

## 3. v5b G0--G2 机械门

| 等级 | 通过 / 总数 | 证据范围 | 是否为力证据 |
|---|---:|---|---|
| G0 | 1 / 1 | 既有 Ramesh LDVM golden 回归保护 | 否 |
| G1 | 7 / 7 | 关闭态恒等、synthetic no-LEV、Eq. 9 / Kelvin / convection 账本 | 否 |
| G2 | 10 / 10 | Hirato live shadow、LESP birth/remesh、material invariance、birth-limit diagnostic | 否 |

G2 的 birth-limit 拟合斜率虽为正，但约为 `1.64e-4`，接近零，只能作为很弱的
拓扑渐近诊断；它不提供压力或载荷证据。

阻断链如下：

```text
shared-wake circulation/topology passes G0--G2
                    |
                    v
pressure / force coupling = not implemented
                    |
                    v
Yang / Figure 14 / Baik accuracy = blocked_not_scored
```

## 4. v5b sequence cross-paper 预留区（不得提前填数）

| 任务 | v5b 力耦合序列结果 | 精度指标 | 当前状态 |
|---|---|---|---|
| Yang | — | — | 等待 force-coupled sequence 输出 |
| Izraelevitz Figure 14 | — | — | 等待 force-coupled sequence 输出 |
| Baik W1--W4 | — | — | 等待 force-coupled sequence 输出 |

只有同时满足以下条件，才允许补入该表并开展 v4b / v5a / v5b 精度对比：

- sequence 输出包含可审计的 pressure/force ledger；
- `force_coupling` 不再是 `not_implemented`；
- 三论文评分使用冻结观测、过滤与聚合口径；
- 关闭 v5b 时逐点还原 parent/v4b，且所有 source/result hash 被记录；
- 结果明确区分 smoke、full 与 canonical eligibility。

## 5. 联合图与复现命令

绘图脚本读取两个冻结 run，不硬编码实验数值；遇到无力 summary 中混入
cross-paper accuracy 字段时会 fail closed。

```bash
cd /tmp/fluxv-v5-nextgen/platform
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  -m forward_flight_benchmarks.plot_fluxv_v5_joint_report \
  --v5a-run ../docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5a_cache_smoke_frozen \
  --v5b-run ../docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5b_no_force_smoke_frozen \
  --output-dir ../docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5_joint_report_skeleton
```

联合图的两个 panel 分别表示：

- (a) v5a 冻结 dev-proxy 误差相对 v4b 的比值；虚线 1.0 表示与 v4b 持平；
- (b) v5b G0--G2 机械门计数，并显式标注三论文载荷 `BLOCKED / NOT SCORED`。

输出目录：

`/tmp/fluxv-v5-nextgen/docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/20260814_fluxv_v5_joint_report_skeleton/`

图文件：

- [PNG](runs/20260814_fluxv_v5_joint_report_skeleton/fluxv_v5_joint_stage_summary.png)
- [PDF](runs/20260814_fluxv_v5_joint_report_skeleton/fluxv_v5_joint_stage_summary.pdf)

其中 `figure_manifest.json` 记录输入 hash、图片 hash、所画数据行，以及
`v5b_sequence_crosspaper.metrics=null`。后续 sequence 数据到达时，应新增独立 run
目录与新报告版本，不覆盖本次骨架或两个冻结 smoke。

## 6. 冻结证据位置

- v5a：`runs/20260814_fluxv_v5a_cache_smoke_frozen/`
- v5b no-force：`runs/20260814_fluxv_v5b_no_force_smoke_frozen/`
- v5a smoke 报告：`V5A_SMOKE_STOP_REPORT.md`
- v5b 合同：`V5B_SMOKE_GATE_CONTRACT.md`

本报告不包含 v5b 三论文载荷数值，也不将 shadow circulation 等同于气动力。
