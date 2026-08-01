# N3.1j0 空间 P2—统一薄面压力 v0 结果

日期：2026-07-29

## 裁决

当前 quick v0 可执行包为 **NO-GO / falsified for promotion**，不进入
确认域 151 点生产扫。这只证伪该执行包的晋升资格，不证伪父命题 N3.1j
“空间涡态—统一面板压力”，也不证伪同机理的收敛实现。冻结 V4.1 未重跑、
未修改，只作为只读叠图和同曲线探索比较。

## 做了什么

- 新闭合 `n3_spatial_pressure_v0` 把 LESP 限定为释放/供给触发，不再直接
  充当持续正涡力幅值。
- LEV 由连续材料 P2 近场面带保存强度、位置和运动；其诱导速度、束缚反应
  和势跳率只通过一次统一面板 Bernoulli 压力成力。
- 旧 `dCN_ds`、冲量涡力、vortex-normal-force、粒子力和独立前缘吸力均
  不参与该闭合；N3 是同一步耦合压力与 N1 反事实压力的逐面板精确差。
- 三点 smoke、16→24 阶求积、`dt→dt/2` 和 32 条件代表集均已实际运行。

## 数值门

| 门 | 结果 |
|---|---:|
| P2/束缚/LESP/压力/力账 | 通过，代表运行 32/32 有限且守卫通过 |
| 附着极限 | 通过；无释放时 N3 位级为零 |
| 16→24 阶求积 | 通过；三哨兵最大力相对差 0.318% < 0.5% |
| `dt→dt/2` | 不通过晋升；无 `1/dt` 爆炸，但 45° 哨兵升力仍改变 7.118% |
| P2 自诱导 | v0 未包含，manifest 明确标为 promotion blocker |

## Fig. 17/18/19 同曲线结果

32 条件 quick 代表集形成 14 条完整曲线，其中确认域 12 条、Fig.19(c,d)
条件域 2 条。下表只使用候选与冻结 V4.1 都完整的同一组 12 条确认曲线
（104 个相同原始测量采样点）。必须注意，候选是 `nc=4, ns=8`、两周期的
quick 网格，而 V4.1 scorecard 来自生产网格；因此这是同曲线探索比较，
不是离散化匹配的 closure-only 比较：

| 指标 | N3 spatial P2 v0 | V4.1 同曲线 | 变化 |
|---|---:|---:|---:|
| 全部 MAE | 1.9818 N | 1.1026 N | +79.7% |
| 全部 RMSE | 2.2884 N | 1.3156 N | +73.9% |
| 全部 bias | -1.9096 N | +0.8145 N | 符号翻转为系统性低估 |
| 趋势捕获 | 0.417 | 0.667 | -0.250 |
| 推力 MAE | 2.6523 N | 1.3663 N | +94.1% |
| 推力趋势捕获 | 0.167 | 1.000 | 明显退化 |
| 升力 MAE | 1.3113 N | 0.8388 N | +56.3% |
| 升力趋势捕获 | 0.667 | 0.333 | 有局部机理信号，但不足以晋升 |

视觉和数据指纹一致：

- Fig.17 的 `f=2.6 Hz` 扭转线中，候选推力由 `-2.307` 降到
  `-5.885 N`，实测两端约为 `-1.634` 与 `-1.754 N`；扭转越大，
  负推力过供越严重。
- Fig.18 的 `U=6 m/s` 频率线中，候选推力由 `-1.757` 降到
  `-2.851 N`，实测却由约 `-1.140` 升到 `+0.742 N`，斜率反号。
- Fig.19 高迎角线仍有部分升力趋势信息，但推力随频率进一步变负。
  守恒账和有限值已排除，指纹指向当前空间状态到薄面弦向压力闭合；不过
  时间步未收敛且比较网格未匹配，物理闭合与时间/空间离散误差仍有混杂。

## Claim tree 改写

- `N3.1j0`: 当前 quick v0 可执行包 `open → falsified`, `freeze: true`；
  不得外推为同机理收敛实现已被证伪。
- `N3.1j`: 保持 `partial`。空间涡态、统一面板压力和守恒传力仍是父方向。
- 可动空间不是调 `A0crit`、衰减、压力裁剪或总力重分配。若继续父方向，
  必须先补足独立的移动分离通量/表面库存（N2.6）、P2 自诱导材料推进，
  以及实际厚度双侧压力；否则会重复该 v0 的弦向力失败。

## 证据

- 代表运行：
  `runs/20260729_191134/candidate_results.json`
- 分数：
  `runs/20260729_191134/fig171819_scorecard.json`
- 同曲线比较：
  `matched_v41_comparison.json`
- 图：
  `runs/20260729_191134/fig17_candidate_overlay_01.png`、
  `fig18_candidate_overlay_01.png`、`fig19_candidate_overlay_01.png`
- 求积：
  `../n3_spatial_pressure_v0_q24/runs/20260729_190426/`
- 半时间步：
  `../n3_spatial_pressure_v0_dt2/runs/20260729_190618/`

主运行命令：

```bash
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_pressure_v0 \
  --closure n3_spatial_pressure_v0 \
  --scope representative32 \
  --quick \
  --seed-run platform/docs/candidates/n3_spatial_pressure_v0/runs/20260729_190240 \
  --timestamp 20260729_191134
```

叠图命令：

```bash
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  platform/plot_candidate_overlay.py \
  platform/docs/candidates/n3_spatial_pressure_v0/runs/20260729_191134/candidate_results.json \
  --data-md platform/docs/data.md \
  --candidate-label "N3 spatial P2 v0 (quick)" \
  --baseline-json platform/docs/s6_sweep_v41_full184_20260729_105013.json \
  --baseline-label "V4.1 frozen"
```
