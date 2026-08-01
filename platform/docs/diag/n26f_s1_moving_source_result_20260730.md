# N2.6f1 S1 moving-NACA0015 来源响应结果

日期：2026-07-30  
节点：`N2.6f1`  
裁决：`S1 FORMAL RESPONSE PASS / N2.6f1 REMAINS OPEN`

## 1. 运行身份

本轮使用未经修改的 official
`naca0015-pitching.c`：

- source SHA：
  `8ff0282a4bfa67473a46f67aea768f27b6db91a44d24296c036f5c701c0acb86`；
- executable SHA：
  `7730989e5bcd006bbe5f1f9b02b58114894e2d4c0a79f64750f7fb94b895dc37`；
- Basilisk tarball SHA：
  `fe6b4b5821517d792c58f0413ea6de4b5bd6b1d337578bb5ceb2fa6f07f8f193`；
- formal settings：`lmax=15`，`512 points/c`，`DT=0.01`，
  `Re=10000`，`L0=64c`；
- process return code：`0`；
- runtime：`33:00.96`，maximum RSS `188,268 kB`；
- official log：`5,179` 个严格 16 列有限行；
- log SHA：
  `91c44bc8993e1e1c07297915431b363a526856529461ec04e3c651bee2c84309`；
- 末端：normalized time `1.84994`，pitch `56.1244 deg`。

资产 manifest、编译命令、九个自定义 header、参考 CSV/图片、qcc 和全部
本轮输出的 path/size/SHA 已绑定在被忽略的：

- `platform/data_external/n26f_runs/source_s1_naca0015/asset_manifest.json`；
- `platform/data_external/n26f_runs/source_s1_naca0015/run_receipt.json`；
- `platform/data_external/n26f_runs/source_s1_naca0015/s1_formal_response.json`。

scorer 明确输出
`scope=N2.6f1.S1_RESPONSE_ONLY` 和
`claim_promotion_authorized=false`；单轮 S1 不能晋升整个节点。

## 2. 冻结定量门

未平滑逐步振荡；按评分前冻结的各自完整参考域和分段线性精确积分：

| field | range-normalized RMSE | limit | peak angle model/ref | error | limit | result |
|---|---:|---:|---:|---:|---:|---|
| CD | `4.671552%` | `10%` | `53.7876/54.9854 deg` | `1.1978 deg` | `3 deg` | PASS |
| CL | `6.639071%` | `10%` | `48.4783/46.1181 deg` | `2.36025 deg` | `3 deg` | PASS |

post-start actual `dt` 为
`0.000189465--0.000791459`，中位数 `0.000313716`。这只说明原 DT
上限轴可能退化，不能据此宣称时间收敛；处理已由
`n26f_source_gate_protocol_amendment_20260730.md` 预先规定。

CL/CD 对照图：

- `s1_cl_cd_comparison.png` SHA
  `c5b3097e7e861db625f4be5c2b71cd44285c529715e5a14cbda17499893719d9`。

图中 embedded-boundary 的逐步高频振荡完整保留，没有根据参考曲线平滑或
挑点。总体增长、CL 峰值区和高角 CD 增长与数字化参考一致。

## 3. 44/约55 度视觉门

来源代码的第二个文件名因整数截断为 `54`。

| checkpoint | Cp SHA | vorticity SHA | side-by-side SHA |
|---|---|---|---|
| `44 deg` | `f6888e3b...58d0e3` | `3b2b88ae...716bc4` | `e0dedb3b...d62454` |
| `54 deg` | `ca4a242b...c4fdd` | `8a8b4277...4002fc` | `fbf821a0...14972` |

主线程对原始图和并排图逐张核对：

1. 两个角度的前缘负涡、吸力面负涡列、正号剪切层和尾缘正涡列均未反号；
2. 44 度的前缘卷起、弦中负涡和尾缘正涡处于与 Schneiders 相同的上下游
   次序；来源解的弦后段负涡更弥散，但没有缺失或移到压力面；
3. 54 度的前缘帽状剪切层、三个主要下游负涡区、中间正涡和尾缘正涡列与
   参考数目/次序一致；涡核尺度有差异，但没有拓扑反相或翼面穿透。

所以预登记的“方向、主 LEV/TEV 数目和位置不得反相”视觉条件通过。该判断
不是图像像素拟合，也不外推到 RoboEagle。

## 4. 裁决边界

S1 回答的是：

> 固定 Basilisk moving embedded-boundary 来源路径，在未经修改的正式分辨率
> 下能够复现 Schneiders 快速俯仰翼型的 CL/CD 响应与主要涡拓扑。

因此 `S1 FORMAL RESPONSE PASS`。

仍未回答：

- `128/256/512` 空间 Cauchy；
- 有效而非退化的时间 Cauchy；
- pressure/full-viscous fragment exporter 与 `embed_force()` 的一次牵引账；
- N2.6 缺失分离压力病因；
- Fig17/18/19 精度。

故 `N2.6f1` 继续保持 `open`。下一步只能执行预登记的 S2 派生只读
instrumentation 和数值独立性门，不能越级运行目标条带。
