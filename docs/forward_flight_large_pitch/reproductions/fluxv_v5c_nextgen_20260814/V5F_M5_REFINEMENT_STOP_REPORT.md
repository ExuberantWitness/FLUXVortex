# FluxV v5f M5 时间步/core 停止报告

## 结论

v5f 的 Ptera 原生材料 LEV 多步账本在代数上闭合，但没有通过预注册的 M5
时间步/core 门，因此状态为 **NO-GO / `blocked_not_run`**。按照冻结计划，未读取
Yang、Izraelevitz Figure 14 或 Baik 的实验载荷，也未运行任何论文精度评分。

失败不是通过挑选 core 可以解决的：当 `80 -> 160` steps/cycle、时间步减半时，
三个预注册 core 的半周期峰值材料释放环量分别增长 `2.192x`、`8.260x` 和
`4.681x`，全部达到或超过 `1/dt` 级增长。

## 冻结工况

- Yang 2025 名义四连杆运动，安装角 `15 deg`；
- `Lcrit=sin(5 deg)=0.08715574274765817`，来自已声明的 Yang 分离角来源；
- 固定空间网格 `2 x 4`，构造 1 周期、尾迹保留 1 周期；
- 每组只推进到同一物理半周期；
- `steps/cycle={20,40,80,160}`；
- Hirato Eq.25 core `rc/d_min={0.10,0.25,0.49}`；
- `d_min=min(global minimum span-panel width, U_inf Lcrit dt/sqrt(2))`；
- 无 cap、ridge、最小二乘 fallback、目标载荷选根或实验拟合。

## 结果

下表为相同半周期内的 `max |Gamma_LEV,new|/(U c)`：

| core ratio | 20 SPC | 40 SPC | 80 SPC | 160 SPC | 80→160 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 1.899410 | 3.349582 | 4.087538 | 8.960179 | 2.192072x |
| 0.25 | 2.661609 | 2.695061 | 2.179608 | 18.003188 | 8.259828x |
| 0.49 | 10.386688 | 2.468564 | 4.504423 | 21.084023 | 4.680738x |

所有 12 组计算仍满足：

- 最大无穿透残差 `5.684341886080802e-14`；
- 最大 LESP 约束残差 `1.582067810090848e-15`；
- Eq.9 后缘 Kelvin 账本最大残差 `0.0`；
- 材料几何与环量均为有限数；
- solver 未进入 poisoned 状态。

这说明“小残差”只证明线性代数和记账自洽，不能证明材料尾迹时间推进收敛。
M5 的时间/core 子门已经失败，继续做弦向/展向敏感性或论文评分不能挽救该门，
反而会违反停止规则。

## 失败机制取证

停止后进行了不读取实验载荷的只读取证。当前证据没有发现 Ptera wake、历史材料
LEV、newborn、Eq.9 或 Eq.24 被重复加入；冻结几何下分别删除历史材料 RHS 或
Ptera wake RHS 也不能消除增长。因此，不能把本次失败归因于一个可直接删掉的
重复力项或错时间层。

直接的增益机制是：Eq.7/Eq.24 卷起持续改变 newborn 与 pseudovortex 的相对几何，
两者对 LESP 约束的 Schur 响应逐步接近相消。代表性 `20 SPC, rc/d_min=0.25`
轨迹中，Schur 条件数从约 `1.15` 增至 `17.13`，同时
`max |Gamma_LEV|/(Uc)` 从约 `0.065` 增至 `1.903`。冻结材料几何时条件数保持在
约 `1.16` 以下，说明几何反馈是必要诱因；删除 Eq.9 修补或历史材料诱导反而会让
峰值进一步增大。

另发现一个确定的空间拓扑缺件：每条 strip 独立保存 aft 节点，相邻活动 strip
并不共享同一个展向节点，裂缝会从毫米级增长到分米级。只读内存实验把连续活动区
节点缝合后可消除裂缝，但 `20/40/80/160 SPC` 的峰值环量仍不形成共同极限。因此，
该缺件必须修，但它不是本次发散的充分原因。

据此，当前候选分类为“展向连通实现缺件 + 常强度闭合环/Eq.7/pseudovortex
表示在卷起几何下的结构性 NO-GO”。这里的结论只针对当前离散表示，不否定材料
LEV 或共享尾迹的一般思路。

本候选的首生/重启位置还必须限定为 **Ramesh et al. AIAA 2012-3027 Eq.(12)**
版本：`U_inf*A0*dt/sqrt(2)`。Ramesh 2014 JFM 修订版与 LDVM v2.5 源码采用的
则是 `0.5*v_edge,local*dt`，连续脱落才使用 1/3 规则。两者都有一手来源，但不是
同一离散模型；因此本报告的 NO-GO 不能自动外推到后一个 local-edge-velocity
出生律。后者若测试，必须作为隔离候选重新跑同一 M5，不得改写本产物。

## 可复现证据

正式产物目录：

`runs/20260814_fluxv_v5f_m5_refinement_final/`

- `per_step_refinement.csv`：12 组、共 450 个已提交时间步；
- `aggregate_refinement.csv`：12 组同物理时刻汇总；
- `summary.json`：M5 判定和停止原因；
- `run_manifest.json`：15 个直接源码/依赖哈希和 3 个结果哈希。

18/18 个声明哈希已重新核对，无不匹配；manifest 使用仓库相对路径，不含本机
`/home` 或 `/tmp` 路径。关键 SHA256：

- runner：`a22b62a80e042dde348c63614367363a1d23287e33b67ef74ef3b7899a2a4c66`
- solver：`a444e5ec0624cba4560b1e0e6deaa6c654bc8bf86a12cb2705a0cda4c76f4563`
- per-step CSV：`6f7ad12e33aaa531d18607327c06980bdc74d4357bd9f768c2c287387cc5f595`
- aggregate CSV：`30bad71f08ebfc370997de9dde2c3785de746976191fd4b4b5e19ade553e0ecf`
- summary：`485abaf7d8f3ca010cbd6e52caed8272343bbd78ace9f0bd9fe1eaa7938d0d05`
- manifest：`f8011568d2a89a29c3236d63fa9f8cd793da87acf561c50e8ff4216976ded508`

独立 same-family 只读复算得到相同 12 组峰值、细化倍率和 NO-GO 判定。该证据只
支持“v5f 当前连续材料反馈未通过机械收敛门”，不支持对材料 LEV 思路作一般性否定。

## 后续决策

1. 归档 v5f 当前 constant-LESP/shared-wake continuation，不对三篇论文评分；
2. 不通过调整 core、`Lcrit` 或实验权重修补；
3. v4b 继续作为目前三个开发数据集上的 qualified reference；
4. 下一候选不能用 ridge、环量 cap、挑选 core 或删除 newborn 伪装修复；
5. 若继续 native-material 路线，必须改为 node-owned、展向连通且至少线性/高阶
   涡量的自由涡片，显式处理间歇活动边界的自由边/拓扑重连，并先在独立 AR6
   canonical 工况通过时间细化、seam、Kelvin 与 Schur 下界门；
6. 任何下一候选都必须重新从 M0–M5 开始，不能继承本次候选的性能晋级资格。
