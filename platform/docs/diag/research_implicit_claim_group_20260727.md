# N1–N2.6–N3 隐式 ClaimGroup：循环物理不能伪装成 DAG 顺序

## 1. 运行时病因

外层 `ClaimGraph` 的拓扑排序适合单向数据流，但同一时间 stage 内存在闭环：

```text
N1 外流速度/压力梯度
  -> N2.6 IBL/分离/位移亏损
  -> N3 新生与既有自由涡
  -> 反诱导速度
  -> N1 束缚解
```

若按 DAG 只执行一次，结果依赖人为节点顺序；若在节点外手写循环，claim tree
又不再是实际运行图。这是运行架构缺组成部分，不是气动常数问题。

## 2. 学科机理

Drela–Giles 类相互作用边界层方法、Drela IBL3 及 Zhang 等非参数 IBL 都把
无黏、黏性和辅助方程组成同一个全局非线性残差，并以 Newton 类方法强耦合。
弱顺序耦合在分离和强相互作用区缺乏稳定性/一致性保证。

## 3. 方案裁决

外层仍保持无环 DAG；把物理上必须同步一致的节点组成一个原子
`ImplicitClaimGroup`。组内：

- 状态和残差均按 claim node/物理方程命名；
- 每个残差块使用预先固定的物理尺度，只作量纲归一；
- Jacobian 的每个子块保留“哪个残差对哪个状态”的身份；
- Newton 更新只允许残差下降的回溯线搜索；
- 每次迭代报告逐块物理/归一残差、秩和条件数；
- 不允许正则项、任意块权重、目标 L/T、压力残差拟合或 LESP 幅值方程。

该组最终至少包含：

```text
state:
  N1.Gamma_bound
  N2.6.IBL_state
  N2.6.separation/release_state
  N3.DDE_new + DDE_geometry_stage

residual:
  N1.no_penetration
  N1/N2.6.displacement_interaction
  N2.6.IBL_momentum_energy
  N2.6.wall_generation_inventory_release
  N3.trace_vorticity_kelvin_free_edge
```

LESP/BEF 只改变活动拓扑，不添加幅值残差。

## 4. 首个实现的证据边界

首个 CPU 实现只验证通用联立求解语义：

- 制造线性和非线性三块系统恢复已知状态；
- 状态/残差块重排不改变具名解；
- 每块残差均达到预登记阈值；
- 秩亏、非有限 Jacobian、残差不下降和禁用块名明确失败。

它不组装 N1/N2.6/N3 物理 Jacobian，不输出压力或力，也不使任何开放物理
claim 晋升。只有 N2.6b4/c 闭合后，才可建立物理 ClaimGroup profile。

## 5. 来源

- Drela & Giles, AIAA Journal 25(10), 1987, viscous–inviscid analysis.
- Drela, AIAA 2013-2437, IBL3 strong coupling.
- Zhang et al., AIAA 2019-1154, global Newton strong coupling and free
  transition.
- Bempedelis et al., Eqs. (41)–(48), conservative surface IBL residual.

