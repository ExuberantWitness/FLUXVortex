# N3 空间涡态—统一面板压力—守恒传力：文献补强裁决

日期：2026-07-28  
阶段：Phase ②学科机理 → Phase ③方向裁决  
状态：**literature-only；不改变 claim state，不授权 production、h/p、D4–D8 replay、118 或 Fig17/18/19**

## 0. 本轮结论

最终生产方向应是：

```text
LESP/IBL 释放条件
  → 近场物质涡片/空间自由涡态
  → 与实际翼面 bound source-doublet / potential-jump 在同一时间层联立
  → 唯一 unsteady-Bernoulli 面板压力
  → 一次表面积分
  → 当前构形功共轭传力
```

而不是：

```text
A0/LESP → 持续正涡力
空间涡量矩 → 总力闭合 → 按面积/刚度/质量事后分配
纯涡粒子 → 独立第二套 LEV 力
```

**VPM 裁决**：纯 VPM 不是当前第一生产方案。近场需要保留与实际翼面压力
强耦合的连续涡片/势跳历史；涡粒子只可作为远场运输或守恒压缩后端，并且在任一
位置、任一时刻，同一段环量只能由涡片或粒子之一拥有。既有
`N3.1f=dead_end/frozen` 不复活；若以后需要粒子化，应在 `N3.1j2` 下建立
“近场连续片 → 远场粒子”的新命题，而不是恢复旧 rVPM 总力路线。

本轮回答了架构充分性，**没有回答**实际 RoboEagle 的缺态究竟是无质量形成片几何，
还是还需要 VES 的质量/卷吸状态。该判定仍服从正在运行的 S3ai-v2.2 正式门及后续
actual-body h/p 顺序。

## 1. 检索与去重审计

使用三个互不排名的文献分片：

| 分片 | 原始条目 |
|---|---:|
| spatial-vortex | 8 |
| panel-pressure | 13 |
| conservative-transfer | 18 |
| 合计 | 39 |

机械按 DOI 或稳定报告标识符去重：

- `10.3390/fluids7020081` 同时出现在 spatial-vortex 与 panel-pressure；
- `10.3390/app11073149` 同时出现在 spatial-vortex 与 panel-pressure。

故分片去重后为 **37 条唯一记录**。主代理又针对剩余逻辑缺口核验 3 篇
Cambridge/JFM 一手全文：

- Mavroyiakoumou & Alben 2022，三维移动膜面涡片到压力跳；
- Gordillo 2025，束缚环量/涡片到非定常压力分布及冲量等价；
- Fang et al. 2026，材料自由片、束缚片、Kutta、分布压力跳与气动功同账。

最终证据库存为 **40 条唯一记录**。本文件只展开与裁决直接相关的核心条目；
机器可读文件保存 40 个去重键和取舍理由。三个分片的逐条原始清单未另存为独立
artifact，因此 `39 → 37` 的来源计数保持 **provisional**，不能仅凭本文件独立
重算；40 个最终去重键本身可机器复核。综述只用于定位，不作为关键机理的唯一证据。

## 2. ① 病因：数据指纹、挂树与可动空间

### 2.1 现有数据能证明什么

V4.1 的生产 N3 仍是：

```text
|A0|-crit → cds → Tv memory → dCN_ds → 独立法向力
```

现有证据分两层：

1. `D3` 的既有生产通道分解把深扭转趋势翻转定位到 N3；
2. D4–D8 完整性审计发现缺 raw、`dt`、before/after-cds、双翼 ledger、
   filter/phase 等证据，因此 D4/D5 的绝对积分值和 D6–D8 的实验映射目前只能
   视为 code-internal/provisional，不能继续抬高为完整实验定律。

所以本轮不使用 D4–D8 的旧数值来“证明”新模型，只保留更稳健的病因：

- `A0/LESP` 的学科角色是临界/释放条件，不是已形成 LEV 的持续力幅值；
- 有限个总涡量矩或合力/合矩不能唯一反演逐面板压力；
- 结构 co-design 需要物理位置上的压力/牵引，而不是总力后的经验重分配。

### 2.2 挂到 claim tree

| 节点 | 当前证据含义 | 本轮影响 |
|---|---|---|
| `N3.1d2` | LESP 只作 onset/feed proxy | 文献独立补强；不改 state |
| `N3.1h1` | Γ+位置/运动可作总力影子态 | 保持 diagnostic-only |
| `N3.1h2` | 有限矩唯一重构压力 | 数学上仍 falsified/frozen |
| `N3.1j2` | 连续自由涡面的材料推进 | 是缺失的生产状态提供者 |
| `N3.1j3` | 连续涡面进入唯一面板压力 | 公式身份正确，真实 provider 缺失 |
| `N3.1j3b6d` | 实际边界 source-doublet 重表示 | 是实际厚度统一压力的近场主线 |
| `N3.1i2a` | `Q=J^T f` 功共轭身份 | 代数已 validated；不等于实际接口完成 |
| `N3.1i2b` | FLUXV 当前构形耦合 | 保持 open；不在本轮建设结构模型 |
| `N3.1f` | 旧 rVPM 生产路线 | 保持 dead-end/frozen，不复活 |

### 2.3 可动空间

允许研究：

- `N3.1j2` 的近场连续物质涡片及守恒远场压缩；
- `N3.1j3b6d` 的 actual-body 同时间层 bound/free 联立；
- `N3.1j3` 的唯一压力账及外部 `ΔCp` 验证；
- `N3.1i2b` 的**接口契约**，但不先选梁/壳或材料模型。

禁止：

- 调 `cds/Tv/crit` 继续重排旧标量；
- 由目标总力反演涡粒子强度、片面质量、卷吸或压力；
- 把 vortex-force/impulse 总力作为第二本 production force ledger；
- 用质量、刚度或面积权重事后分配总力；
- 在正式压力门和 actual-body h/p 前跳到 VES。

## 3. ② 学科机理：关键一手证据

### 3.1 空间涡态必须比“Γ+涡心”更完整

Ramesh et al. 2014 将 LESP 用于触发 LEV shedding；随后需要显式 LEV/TEV
位置、环量、诱导势/速度与时间历史才能形成弦向压力差
([doi:10.1017/jfm.2014.297](https://doi.org/10.1017/jfm.2014.297))。
Joshi et al. 2025 将各展向 LDVM 条带通过 lifting-line downwash 耦合，明确保存
每条带的 bound、LEV、TEV 和位置状态；但其 `LESPcrit=0.21` 由一个工况选择，
且没有统一三维面板压力
([doi:10.1017/jfm.2025.10495](https://doi.org/10.1017/jfm.2025.10495))。

Xia & Mohseni 2017 表明，新生自由片还需要释放方向、强度和相对速度，并由
unsteady Kutta、环量、质量和动量共同约束
([doi:10.1017/jfm.2017.513](https://doi.org/10.1017/jfm.2017.513))。
因此最低近场状态不是一个力系数，而是：

```text
S_free = {
  material geometry X_f,
  circulation/potential-jump density,
  sheet connectivity and side identity,
  local velocity,
  birth/release flux,
  time history
}
```

### 3.2 自由涡态本身不唯一决定翼面压力

Fritz & Long 2004 的三维 UVLM 面板压力需要：

```text
bound circulation 的弦向/展向差分
+ circulation time derivative
+ 全尾迹诱导速度
+ 面板几何与运动
```

而不只需要自由涡位置和强度
([doi:10.2514/1.7357](https://doi.org/10.2514/1.7357))。

Mavroyiakoumou & Alben 2022 在三维移动膜面上把压力跳写为

```text
∂[p]/∂α1 = G(r, γ1, γ2, μ1, μ2, τ1, τ2, νv)
```

其中除了两方向束缚涡片强度，还显式需要两侧平均切向速度、膜面切/法向运动和
几何；随后沿弦积分并以尾缘 `[p]=0` 固定边界条件
([doi:10.1017/jfm.2022.957](https://doi.org/10.1017/jfm.2022.957))。

Fang et al. 2026 在二维材料涡片上给出更直接的核查式：

```text
[p](s,t)
= ∫_TE^s ∂t γ̄(s',t) ds'
  + [Re(w(s,t))-U(t)] γ̄(s,t)
  + Γ̇(t).
```

同一文献用该分布压力计算输入功率，而自由片的材料环量另由
`D_t(γ̄ s_α)=0` 保存
([doi:10.1017/jfm.2026.11521](https://doi.org/10.1017/jfm.2026.11521))。
这不是 RoboEagle 的三维生产公式，但它构造性说明：

> 自由涡的 Γ、位置和运动只能通过同一边界求解改变束缚状态与平均速度；
> 分布压力还需要束缚涡片/势跳及其材料时间历史和压力边界条件。

Gordillo 2025 的线性薄翼式

```text
Δp = ρ(∂t Γ + U∞ γ)
```

又证明压力积分与 impulse 力只有在同一假设、同一 wake 运动和完整起始涡贡献下
才等价
([doi:10.1017/jfm.2025.10177](https://doi.org/10.1017/jfm.2025.10177))。
因此 impulse/vortex-force 可以做独立观察器，不能成为并联第二力账。

### 3.3 薄面压力跳与实际厚度双侧压力必须分层

Morino 的任意运动厚体势流理论要求在**实际表面**求解 source/doublet 或表面势，
再由表面速度、势率和 Bernoulli 得到侧别压力
([NASA CR-2464](https://ntrs.nasa.gov/citations/19750004821))。

因此：

- `N3.1j3a` 的薄面 `Δp` 恒等式可以保持 validated；
- 它不能唯一恢复 `Cp_upper/Cp_lower` 或厚翼弦向压力力；
- 实际厚度生产需要 `N3.1j3b6d` 的 bound source-doublet、势切面/wake jump、
  mean-potential history 和远场 pressure gauge；
- 黏性分离压力仍属于 N2.6 的独立物理输入，但必须在压力级合成后只成力一次。

### 3.4 VES 是有独立质量/卷吸物理的另一种片，不是压力残差名称

DeVoria & Mohseni 2019 的 VES 显式拥有片面质量、片内动量、法向速度跳和压力跳；
零卷吸极限退化为无质量普通涡片，普通自由片不能任意携带非零压力跳
([doi:10.1017/jfm.2019.134](https://doi.org/10.1017/jfm.2019.134))。

所以 VES 只有在以下证据同时存在时才可打开：

1. actual-body 无质量形成几何已正确；
2. h/p 后仍有收敛非零动力余核；
3. 独立场级数据支持片面质量、卷吸或片内动量；
4. 状态不是由目标力或压力 residual 反演。

### 3.5 粒子适合作为 wake 后端，不是压力闭合捷径

Proulx-Cabana et al. 2022 以 UVLM 表示翼面和近尾迹，较老 wake 才转换为
VPM；粒子携带位置、矢量强度和核尺度，并以 stretching/PSE/SGS 演化
([doi:10.3390/fluids7020081](https://doi.org/10.3390/fluids7020081))。
Zhu et al. 2021 则用实际叶片 source-doublet 面板和近尾迹面元产生唯一
unsteady-Bernoulli 表面压力，远尾迹才粒子化；转换保存零阶和一阶涡量矩
([doi:10.3390/app11073149](https://doi.org/10.3390/app11073149))。

这两篇共同给出正确的 ownership 边界：

```text
near body / recent wake: panel or continuous sheet
far wake: optional particles
same circulation: exactly one owner
surface load: exactly one pressure evaluation
```

它们不支持“粒子自己再产生一份 LEV 总力”，也不提供 RoboEagle 光滑前缘的
分离 birth law。

### 3.6 面板压力到结构自由度必须守恒功，但守恒总量不等于局部准确

Farhat, Lesoinne & Le Tallec 1998 对非匹配流固界面给出运动/载荷的共轭关系：

```text
u_a = H u_s,
f_s = H^T f_a,
```

使离散虚功相等
([doi:10.1016/S0045-7825(97)00216-8](https://doi.org/10.1016/S0045-7825(97)00216-8))。
对当前构形非线性映射 `x_a=g(q_s,d_g)`，同一原理是：

```text
δx_a = J_as δq_s,
Q_s  = J_as^T f_a.
```

这与已冻结 `N3.1i2a` 一致。若 `J_as` 精确含三平移和三转动刚体列，同一操作
同时恢复合力和关于指定原点的合矩。

但 Jaiman et al. 2005/2006 证明，名义全局守恒的点投影仍会在曲面非匹配网格上
产生局部过冲；common refinement 能在交叠子面上更准确地传递离散牵引
([doi:10.1002/nme.1434](https://doi.org/10.1002/nme.1434),
[doi:10.1016/j.jcp.2006.02.016](https://doi.org/10.1016/j.jcp.2006.02.016))。
Jiao & Heath 2004 给出严格积分守恒的 common-refinement 数据传递
([doi:10.1002/nme.1147](https://doi.org/10.1002/nme.1147))。

因此传力验收必须有两层：

1. `J^T` 的功、合力、合矩恒等；
2. 非匹配曲面上的局部牵引、弯矩密度和网格收敛。

本轮只定义气动载荷接口，不选择结构材料、梁/壳离散或耦合时间积分。

## 4. ③ 缺组成部分，还是组成部分错误

### 4.1 错组件

1. **`A0/LESP → sustained dCN` 角色错误**：LESP 是 release/onset 条件，
   不是已形成空间 LEV 的持续压力幅值。
2. **独立 N3 总力加法错误**：空间涡态若已进入同一 bound/pressure 解，再加
   `dCN_ds` 会双计。
3. **总力后经验分配错误**：质量、刚度或面积权重不能定义气动压力。
4. **纯粒子即完整载荷模型错误**：粒子运输不提供 bound pressure state、birth
   condition、actual-thickness mean potential 或唯一 panel traction。

### 4.2 缺组件

生产链仍缺四个显式 provider：

```text
P_birth:
  光滑前缘/分离流形上的 circulation flux、方向、相对速度与 side identity

P_free:
  近场连续物质涡片的 X, chi/gamma, topology, velocity, history

P_bound:
  实际翼面 source-doublet / bound potential jump、cut/wake jump、
  mean potential、same-stage time history

P_pressure:
  在同一几何/时间层上由总势与总速度生成唯一 panel pressure/traction
```

`N3.1i2a` 的功共轭传力代数已经存在，所以**不是先缺结构模型**。当前真正阻塞
仍是上游气动状态与压力 provider。

## 5. ④ 有证据的候选方案

### 5.1 首选：near-sheet / optional far-particle 双表示

```text
N2.6 / N3.1d2
  LESP + IBL inventory
  只产生 release event / circulation flux
        │
        ▼
N3.1j2
  actual-body near-field continuous material sheet
  {X, chi/gamma, topology, side, velocity, history}
        │
        ├── sufficiently far from pressure-sensitive body region
        │       └── optional conservative sheet→particle conversion
        │           {X_p, vector Gamma_p, core, ownership}
        ▼
N3.1j3b6d
  same-stage actual-surface source-doublet + bound/free coupled solve
        │
        ▼
N3.1j3
  one unsteady Bernoulli pressure ledger
        │
        ▼
ClaimGraph ForceLedger（图级聚合器，不属于 N4）
  one surface integration only
        │
        ▼
N3.1i2
  current-configuration J_as^T / common-refinement transfer
```

这里的 `ForceLedger` 是 `claim_runtime/core.py` 中的图级唯一力账聚合器。它不
是 N4：N4 仍是 validated/frozen、默认关闭的 CT 吸力一致性诊断节点，只提供
`n4_force`，本方向不修改、复用或重新解释 N4。

近场保持连续片的原因不是“高阶一定更准”，而是：

- release junction、material history、Kutta/Kelvin 和近壁压力高度敏感；
- 过早粒子化会丢失片侧、连接和 potential-jump history；
- 已有 formal gate 正在检验的就是 fixed-space 可达压力相容性，不能在结果前
  换成另一个未取证自由度。

远场粒子化只在下列 guard 后允许：

- circulation ownership 无重叠；
- 零阶环量与一阶冲量守恒；
- 转换前后 body induced velocity / potential-rate 在压力误差球内；
- 转换没有瞬时力、力矩或功率跳；
- 粒子 core、conversion distance/age 先由网格收敛而非 Fig 目标选取。

### 5.2 备选：全连续涡面

若远场成本在 Fig17/18/19 规模内可接受，第一生产版可以完全不粒子化。它减少
ownership 和 remapping 风险，但需要更强的长时 roll-up、近奇异积分和重网格守卫。
该选择属于实施成本裁决，不改变压力物理。

### 5.3 暂不进入：VES companion

VES 只在无质量形成片通过后仍有独立场证据时作为 companion，不替换 bound pressure
solve，也不通过目标力初始化。当前为 **conditional NO-RUN**。

### 5.4 明确 NO-GO

| 候选 | 裁决 |
|---|---|
| 调 `cds/Tv/crit` 修 Fig 曲线 | NO-GO |
| `Γ+centroid` 直接重构全翼压力 | NO-GO |
| 纯 rVPM 替换 N1/N3 | NO-GO |
| vortex-force/impulse 直接作为 production 总力 | production NO-GO；diagnostic GO |
| 总力后按面积/刚度/质量分摊 | NO-GO |
| 压力 residual 直接命名为 VES 质量/卷吸 | NO-GO |
| 近场连续片 + 同一 bound pressure + 可选远场粒子 | 条件 GO，先 shadow |

## 6. 预登记的验证顺序

### G0：正在运行的 S3ai-v2.2

先按已冻结 result-interpretation contract 判为四类之一：

```text
PROTOCOL-NO-GO
ZEROTH-ORDER NAMED-LAW OBSTRUCTION
FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS
NO RESOLVED WITNESS
```

该结果最多改写 named-law obstruction，不直接授权本候选。

### G1：actual-body pressure h/p

- 实际翼体/近尾迹同一空间的 h/p 收敛；
- source-doublet、wake cut、Kutta、material history 和 pressure residual
  分开报告；
- attached limit 逐面板退化到冻结 N1 压力/力账；
- 禁止由总力残差选择阶次或核。

### G2：forming geometry

- 先测试 Xia–Mohseni 类无质量 forming geometry；
- 检查 junction 有限速度、环量通量、质量/动量、方向和 side identity；
- 光滑三维前缘适配必须有自己的场级门，不用二维尖边成功代替。

### G3：统一压力

- 每个自由涡态只通过 total induced velocity / potential history 进入同一 pressure；
- pressure channels 先求和、只成力一次；
- body-frame panel force、总力、合矩和气动功逐步守恒；
- 使用公开 `ΔCp` 数据时不比较不可辨识的绝对双侧压力或厚翼 `Ct`。

### G4：传力接口

- 任意虚位移的 `δq^TQ=(Jδq)^Tf`；
- 三平移/三转动列恢复合力与合矩；
- nonmatching surface 用 common refinement/mortar 或等价弱积分；
- 局部弯矩密度、扭矩和气动功随气动/结构网格分别收敛；
- 质量与刚度参数不进入压力或载荷投影权重。

### G5：研究数据恢复

执行已冻结 D4–D8 replay contract，恢复 raw、`dt`、cds 前后、时钟、phase、
filter 和 wing-scope 证据。失败只表示数据无效，不用于调整模型。

### G6：生产晋升

顺序不可交换：

```text
三点 E1/E2 复现门
→ 代表点逐时步压力/力账
→ 118 工况
→ 趋势记分卡
→ Fig17/18/19 数值与人工视觉核对
```

只有全链通过才允许让新压力路径替代旧 `dCN_ds`；替代必须是一次 claim-tree
有证据改写，不能让两条力路径并存。

## 7. 对 claim tree 的本轮允许改写

本文件是 literature-only，**现在不改变任何 state/freeze**。正式门结束并经独立
结果审计后，允许的最大改写是：

- 给 `N3.1j2` 增加“近场连续片、远场可选粒子、唯一 ownership”子命题；
- 给 `N3.1j3` 增加“free state 不直接产力，必须由 bound pressure provider
  观察”的证据；
- 给 `N3.1i2` 增加 Farhat/Jiao/Jaiman/Kiviaho 文献，但 `N3.1i2b` 仍 open；
- 不复活 `N3.1f`；
- 不因文献相似性把 `N3.1j2/j3`、forming geometry 或 VES 设为 validated。

另需在正式 one-shot 终止后处理一项治理债务：D4–D8 审计已证明旧 raw 证据不完整，
claim tree 中依赖 D4/D5 绝对积分的表述应明确标注为 code-internal/provisional，
直至 research-grade replay 恢复其证据资格。该修正不得在正式计算中途改写
冻结输入或偷换物理结论。

## 8. 本轮非声明

- 没有声称涡粒子方案已验证；
- 没有声称连续涡片一定提高 Fig17/18/19；
- 没有声称缺失状态就是 VES；
- 没有声称结构模型已经或需要现在实现；
- 没有修改 V4.1 公式、常数、网格、运动学、closure profile 或 ForceLedger；
- 没有执行 D4–D8、三点、118 或 Fig17/18/19；
- 没有用任何实验目标选择粒子核、转换距离、pressure offset 或载荷权重。
