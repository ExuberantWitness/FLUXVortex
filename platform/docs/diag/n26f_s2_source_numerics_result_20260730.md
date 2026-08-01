# N2.6f1 S2 来源数值独立性与牵引账结果

日期：2026-07-30  
节点：`N2.6f1`  
裁决：`SOURCE-NUMERICS NO-GO / TARGET NOT AUTHORIZED`

## 1. 执行边界与证据身份

本轮只运行固定源码的 rapidly pitching NACA0015 来源算例，没有读取或运行
RoboEagle 目标条带，没有修改 V4.1、N2/N3 力或 Fig17/18/19 缓存。

完整机器结果：

`platform/data_external/n26f_runs/s2_naca0015/s2_family_result.json`

SHA-256：
`a4bebb7d417c4ecbcf7d5c11b027b45b9ae99178d942d1f49dc7f1a0d0d8b008`。
对应 manifest SHA-256：
`b5633481359d76f1285adf14c3c1540e2f6ca44c9923902acd0a25383dc5b156`。

机器复验确认：

- S1 actual receipt、official log、asset manifest 和重算指标全部绑定；
- 时间轴 freeze 从 raw `d_phys` 重新派生并逐字段一致；
- 九个 S2 角色中的八个独立运行及一个共享角色身份均逐行通过
  `i/t/dt/theta/CD/CL` 绑定；
- 最细 instrumented log 与 official formal log 字节 SHA 完全相同，
  CL/CD 最大差和相对 L2 均为零；
- 八个可复现构建二进制与实际运行二进制逐 SHA 一致。

因此以下失败不是 instrumentation 改变了流场，也不是日志混源。

## 2. 决定性时间轴指纹

原预登记最后两档 `DTcap={0.01,0.005}` 的 cap active fraction 均为零，
median actual-dt ratio 为 `1.0`，故按执行前修订判为
`AXIS-DEGENERATE`，两条相同曲线不能解释为时间收敛。

只从 `lmax=14, DTcap=0.01` 的 post-start `d_phys` 得到：

\[
d50=6.948415270944626\times10^{-4}.
\]

冻结替代最后两档为：

\[
\{d50/2,d50/4\}
=\{3.474207635472313\times10^{-4},
1.7371038177361564\times10^{-4}\}.
\]

替代轴确实有效：

- cap active fraction：`0.9992501 / 0.9998123`；
- median actual-dt ratio：`2.0000026`。

但最后两级响应为：

| channel | relative L2 | peak relative change | frozen limit |
|---|---:|---:|---:|
| CD | 8.619578% | 8.768767% | 3% |
| CL | 4.885594% | 13.453583% | 3% |

四项全部至少有一项越门，且 CD/CL 峰值变化并非小幅舍入误差。
因此有效替代时间轴的结论是 `NO-GO`，不得退回已退化原轴宣称收敛。

## 3. 空间轴与移动边界牵引账

空间最后两级 `256 -> 512 points/c` 同样未过：

| channel | relative L2 | peak relative change | frozen limit |
|---|---:|---:|---:|
| CD | 9.658540% | 4.589975% | 3% |
| CL | 3.550171% | 2.063703% | 3% |

其中 CL peak 单项通过不改变 CL waveform L2 和 CD 两项失败。

逐层移动 embedded-boundary 有向轮廓闭合为：

| points/c | 44 deg | 54 deg | frozen limit |
|---:|---:|---:|---:|
| 128 | `8.36e-16` | `2.2766e-3` | `1e-8` |
| 256 | `1.7798e-3` | `1.9134e-3` | `1e-8` |
| 512 | `9.1531e-5` | `8.2577e-4` | `1e-8` |

128 层 44 度的偶然闭合没有跨角度或跨网格保持；最细层两相位仍分别高于
门槛约 `9.15e3` 和 `8.26e4` 倍。

与此同时，每个已有 fragment 的 pressure、完整 viscous traction 及其和，
与同一步 `embed_force()` 的相对残差均为机器零；逐 fragment primitive
重算也逐位一致。这证明 exporter/求和账正确，但不能把**缺口尚存的离散
轮廓**变成水密物面。故该账不能作为 co-design 材料载荷 oracle。

## 4. 执行偏差与保守解释

第一次 lmax=15 前台运行被外部 `SIGTERM` 中断在
`tau=1.10603, angle=30.5953 deg`，空 `runtime.txt` 和部分日志完整保存在：

`platform/data_external/n26f_runs/s2_naca0015/l15_dt0p01_interrupted_20260730/`。

它没有进入 manifest。随后只用同一二进制 SHA
`bed11d80fb0c84631a3732e55c3aa8d99472f7be6d3e227eb6fcb442d1568adf`
和冻结命令完整重跑；return code 为零、5179 行日志完成，未改变源码、
网格、DT、阈值或指标。

另一个须披露的弱点是：操作监视期间曾看到 official log 行，之后才写入
替代时间轴 freeze。机器选择器和两次复验只读取
`d_phys=min(d_CFL,d_embed)`，没有读取 CL/CD、参考曲线或目标力；九个
decision 字段均从 raw sidecar 唯一重算。该程序性暴露不被包装成盲法，
但本轮结论是保守 NO-GO，不产生错误晋升。

S1 的 44/54 度涡拓扑仍是人工视觉证据，现有资产没有预登记的机器分类器。
本地 qcc、直接依赖、source diff 和运行二进制已冻结并可字节复现，但没有
把整个系统编译器/系统库传递闭包宣称为跨机器认证。

## 5. 裁决

机器总门为：

```text
all_runs_complete = true
instrumentation_neutrality = true
space_pass = false
time_pass = false
traction_pass = false
s2_pass = false
n26f2_target_observation_authorized = false
```

因此禁止运行 `N2.6f2` 的 U10 目标条带。S1 来源响应通过只说明该实现能在
一个 formal 网格上得到量级相符的 CL/CD 与主涡图，不能覆盖独立数值轴和
水密牵引门的失败。
