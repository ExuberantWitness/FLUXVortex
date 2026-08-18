# FluxV v5f：Ptera 原生材料 LEV 预注册计划

## 目的与边界

v5f 只检验一个假设：材料 LEV 必须进入现有 Ptera/FluxV 的同一个 UVLM
AIC、TE 尾迹和载荷账本，才能避免 v4b 的二维差量投影与 v5b 的双重求解链。

本候选不复用 v5b 的自定义 bound AIC 或 unified pressure，也不叠加 LDVM、
polar、impulse、Polhamus 或 profile-drag 力。Ptera 的四腿 Kutta--Joukowski
与 `dGamma/dt` 仍是唯一基础载荷所有者。

## 冻结算法

1. `enabled=False` 时工厂直接返回原 `UVPMHybridSolver`。
2. `enabled=True` 但材料 LEV 历史为空且 LESP 未越阈值时，所有 hook 只调用
   同一个 parent 方法，不加零数组、不重算、不改 TE wake。
3. 触发时使用 Ptera 当步原生矩阵和右端：

   ```text
   A = current Ptera bound-ring AIC
   b = -wake normal influence - freestream/motion normal influence
   ```

   历史 LEV 只进入固定 RHS；当步 newborn 只作为增广列出现一次：

   ```text
   [A  B] [Gamma_b] = [b]
   [H  0] [Gamma_L]   [sign(A0_pre) A0crit]
   ```

   不允许 ridge、cap 或根据实验载荷选根。奇异、非有限或残差超限时事务回滚，
   不提交材料状态。
4. 载荷后由 parent 生成下一步 TE row，再按同一时间层实施 Kelvin 关系
   `Gamma_TE,new = Gamma_b,rear + Gamma_L`。旧材料涡强度不可修改，只更新位置和年龄。
5. 在 active 阶段仍先运行 Ptera 原生 KJ + `dGamma_b/dt`；以后只允许加入
   同一 `Gamma_L` 状态导出的局部线强度修正与 Hirato Eq.17 修正。共享前缘的
   newborn/pseudovortex 丝线相消，所以 front 修正恒为零；span 腿由本步表面释放
   的展向差分给出；TE 腿使用“本步表面释放减上一步尾迹释放”；Eq.17 则只在
   本步 active 条带上使用 `(Gamma_L^n-Gamma_L^{n-1})/dt`。TE 历史与 Eq.17
   活动门必须分账，二者进入同一个逐面元载荷账本。
6. 材料涡核不沿用 Ptera bound core。Hirato Eq.25 只约束固定 cutoff 为“预期
   最小涡环尺寸的小比例、通常小于 0.5”；Ramesh 首涡适配给出活动阈值处的最小
   弦向出生尺度 `d_birth=U_inf*Lcrit*dt/sqrt(2)`。因此每个离散配置预先冻结
   `d_min=min(min_span_panel_width,d_birth)`，并运行无默认优胜者的
   `rc/d_min={0.10,0.25,0.49}` 三点族。`0.25` 只可作为 development smoke；
   三点对结论不一致时不得晋级，也不得依据论文载荷选择其中一点。

   此处“Ptera 原生”只表示 bound AIC、坐标/环量约定、TE wake 和载荷所有者均为
   Ptera；材料 LEV 与当步 pseudovortex 的诱导速度必须使用同一个 Hirato Eq.25
   Lamb--Oseen cutoff。Stage 2A 曾使用的 Ptera Nguyen/Ramasamy 核仅是排序/符号
   机械探针，不得进入材料时间推进或冒充 Eq.25。

## 机械门与停止规则

- **M0 hard-off identity**：工厂返回 parent 类型；输出逐位一致。
- **M1 pristine identity**：Yang、Figure 14、Baik 代表 movement 上，AIC、RHS、
  bound Gamma、TE 顶点/强度/年龄、逐面元力矩、飞机载荷与 VPM 粒子逐位一致。
- **M2 augmented algebra**：无穿透和 LESP 约束残差均不大于 `1e-10`；
  `Gamma_L=0` 严格退化为 parent solve。
- **M3 time/Kelvin**：Eq.9 残差不大于 `1e-12`；newborn 在 solve、load、
  convection 各出现一次；旧材料强度逐位不变。
- **M4 force ledger**：逐面元和飞机级载荷闭合；`Gamma_L -> 0` 连续退化；
  runner 中不存在第二载荷提供者。
- **M5 refinement**：光滑启动下 newborn circulation 与位移不出现 `1/dt`
  发散；时间、弦向和展向敏感性分别报告；上述三个预注册 cutoff ratio 必须给出
  同方向结论。Hirato cutoff 与 Ramesh Vatistas `rc=1.3 U dt` 不得混称同一模型。

任一机械门失败即停止，不运行论文精度评分。全部通过后，先跑 Yang 15 deg、
Figure 14 `(theta, psi)=(15,60)`、Baik W2 三个代表工况；三者均不劣于冻结
v4b/v5c0 后才运行全部 22 工况及全工况图。

## 已知失败路径

v5b 的 no-LEV 链并非当前 FluxV 的 reduction：其 custom AIC 与 Ptera 的相对
Frobenius 差约 2.05、环量相关系数约 -0.995，TE 出生几何和 wake kernel 不同；
即使输入完整 Ptera 状态，替代面压载荷仍与 Ptera 显著不等价。因此不得通过
翻转最终符号、调 core 或拟合 0.556 相位误差来修补 v5b。

## 证据等级

本计划在 v5f 论文结果产生前冻结。所有 Yang、Figure 14、Baik 工况均已用于
开发，不是 held-out；即使全 22 工况通过，也只能称同一开发集合上的跨工况
一致改善，后续仍需新增独立实验作泛化验证。
