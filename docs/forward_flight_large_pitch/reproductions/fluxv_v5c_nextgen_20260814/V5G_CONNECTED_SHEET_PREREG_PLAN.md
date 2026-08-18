# FluxV v5g 连通高阶材料涡片：预注册机械计划

状态：`preregistered_before_v5f2_result`  
日期：2026-08-14（Asia/Shanghai）

本计划在 v5f2 的时间步/core 结果揭晓前冻结。制定过程中不读取 Yang 2025、
Izraelevitz Figure 14 或 Baik W1--W4 的实验载荷，也不授权论文评分。

## 决策顺序

1. 先运行 v5f2 单因素门：只把 v5f 的 2012 `U A0 dt/sqrt(2)` 首次/重启
   出生几何，替换为 revised Ramesh/LDVM-v2.5 的局部前缘速度半步律；连续
   脱落仍用 Eq.7 的 1/3 规则。其余 AIC、LESP、TE Eq.9、载荷、网格、阈值和
   core 完全冻结。
2. 若 v5f2 通过时间步/core 门，停止 v5g 重写，先独立审计 v5f2 的空间收敛与
   所有权；仍不得立即评分。
3. 若 v5f2 失败，才进入 v5g0。v5g0 只构造无载荷的连通 P2 材料涡片底座。
4. v5g1 只有在独立的三维前缘环量出生通量律和 moving-curved P2 ALE 输运均
   通过机械门后，才允许引入力所有者并另行预注册三论文评分。

## 来源边界

- Ramesh 2012 的 `U A0 dt/sqrt(2)` 是早期二维出生几何，不等于 LDVM v2.5。
- revised Ramesh/JFM 与 LDVM v2.5 使用首次/间歇重启 `0.5 q_LE dt`、连续
  1/3 几何。Fortran 的 `q_LE` 在本步新 TEV 已求解后计算；Ptera 原生求解时层
  没有同一对象。因此 v5f2 只能称为“v2.5 几何律到 Ptera 时层的迁移”，除非
  另行闭合本步 TEV 时层。
- 该几何律不提供三维连续 P2 环量通量；不得重新使用逐条带 `A0=Acrit` 反演
  来冒充通量律。

## v5g0 拓扑和输运合同

- 使用全局 node-owned P2 sheet：几何节点、边中点自由度和势跳只有一个 owner；
  条带视图只读。
- 跨展向 seam 和跨时间 band seam 的几何及 P2 trace jump 必须严格为 `0.0`。
- 活动连通分量显式维护 root/tip/front/free-edge 角色；真正自由边势跳为零，镜像
  根部不是自由边。
- Eq.7 只能做 P2 保守重参数化：保持旧材料 ID、参考 P2 场、共享 trace、Kelvin
  和 Helmholtz 账本；禁止逐条带独立移动共享节点。
- 旧材料涡片以同阶 ALE/Heun 推进；clamp-after-update、ridge、SVD 截断、删出生
  column、挑 core 或按目标误差选时间窗全部禁止。
- v5g0 使用 manufactured finite-rate source；`ForceLedger` 必须为空。

## Schur 可观测性门

对每个活动连通分量，以 P2 mass matrix 白化后的 newborn source--LESP 响应
`S_hat` 必须同时满足：

- `sigma_min >= 1e-3`；
- `sigma_min/sigma_max >= 1e-3`；
- source nullity 为零；
- phase-matched `80 -> 160` 的 `sigma_min` 比位于 `[0.5, 2.0]`，且不得趋零。

Schur 只作守卫，不用于解出生幅值。

## v5g0 硬门

### M0 exact-off

- off 时 parent/v4b state、wake、pressure、force bitwise 不变；
- 新模块不得导入三篇论文的 digitization 或评分模块。

### M1 node/P2/Eq.7

- 全局 node 和 P2 DOF owner 唯一；seam jump 严格 `0.0`；
- 常数/线性/二次制造场达到机器精度，线性涡量精确；
- off-sheet velocity 相对 oracle `<=1e-10`；
- Eq.7 pull-back、shared trace、boundary circulation 误差 `<=1e-12`；
- 未分类边、重复 ID、退化面元立即 fail closed。

### M2 单 stage AIC/event/Eq.9

- 相对无穿透 `<=1e-10`，无 solver fallback；
- LESP 只验证 causal event/front，不强制 post-solve `A0=Acrit`；
- Eq.9、全局 Kelvin、old material immutability `<=1e-12`；
- inactive component 的 source 和 DDE RHS bitwise zero。

### M3 ALE/self-advection

- 非零二面角短时门：128/192 位移差 `<=2%`、Heun 误差比 `>=3.5`；
- moving-curved typed-inflow P2 ALE：Heun 比 `>=3.5`、finest relative L2
  `<=1%`、shared trace `<=2e-12`、跨 gauge 物理场差 `<=1e-10`；
- Kelvin/Helmholtz/Cauchy normalized residual `<=1e-12`；
- self-field channel sum `<=1e-12 U`；ForceLedger 条目严格为零。

### M4 时间收敛

在独立 AR=6 canonical、固定空间网格和完整尾迹上运行 20/40/80/160 SPC。
对 probe velocity、centroid、LE/TE circulation、birth rate、sheet vorticity、
Schur `sigma_min`：

- `E40/E80 >=1.5`；
- `E80/max(||Q160||,1e-12) <=0.05`；
- `max|delta_g|` 的 log--log slope 在 `[0.75,1.25]`，且 160/80 `<=0.65`；
- `max|delta_g|/(U^2 dt)`、material speed/U、总变差的 160/80 比均 `<=1.25`；
- 任一约 2 倍增长、seam 非零、拓扑序列不收敛或 Schur 趋零立即 STOP。

### M5 空间收敛

固定 160 SPC，在 `ns={4,8,16,32}` 上要求 trace 和 field 同时收敛：

- `E4/E8 >=1.2`、`E8/E16 >=1.2`；finest relative difference `<=5%`；
- trace total variation 与 peak 的 32/16 比 `<=1.25`；
- free/root/tip/front 角色在全部网格一致；
- 出现逐条带交替或随细化增长的 tip/front boundary layer 即 NO-GO。

## 载荷资格

只有 source-level M4/M5 均通过后，才可执行 M6：以单一统一面压 owner 生成力，
parent KJ/dGamma、旧 LE suction、LEV impulse、VNF、v4b correction 均不得并存。
逐面元 channel closure `<=1e-12`，飞机级闭合 `<=1e-9 N`，未分类力严格为零；
no-release 增量 bitwise zero。M6 仍只是机械资格，不是实验精度结论。

## 当前裁决

- v5f2 前置单因素门：GO。
- v5g0 无载荷机械底座：conditional GO，仅在 v5f2 失败后启动。
- v5g1 载荷和论文候选：NO-GO，直到独立三维出生通量律与 P2 ALE 输运闭合。

