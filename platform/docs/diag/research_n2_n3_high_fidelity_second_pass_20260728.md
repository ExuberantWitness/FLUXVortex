# N2/N3 第二轮高保真数据审计：空间状态门与压力输出门拆分

日期：2026-07-28  
作用域：只裁决气动证据和下一研究门；不修改 V4.1 公式、常数、网格、运动学
或结构模型。

## 0. 结论先行

第二轮检索没有找到可直接下载、同时满足 `N2.6b4f3` 联合契约的目标域三维
连续近壁速度场。这个结论只对本轮明确审计的公开记录和 2026-07-28 的注册表
快照成立，不声称“世界上不存在数据”。

但本轮发现了此前没有进入案卷的独立动态表面压力资产：

1. 4TU NACA643418 reverse-flow v2，`Re=250k`，150 cycles；
2. 4TU DU-97-W-300/VG v2，`Re=1e6`，完整时序及 phase mean/std；
3. Princeton NACA0021 ramp-pitch database，`Re=0.5m--5.5m`。

因此原来把“取得空间场”和“验证统一压力”串成一个等待链是不必要的。研究链
必须拆为：

```text
V-gate: 隐藏空间涡态身份
    必须用坐标化连续速度/近壁场独立检验

P-gate: 空间状态 → 逐面板压力输出
    可以立刻用独立表面压力作外验
```

压力输出可以证伪模型，但压力单独不能唯一识别隐藏涡态。两门不得互相冒充。

## 1. ① 病因定位

### 1.1 数据规律

此前的六资产审计已经发现，搜索结果中的“dynamic PIV/LES”经常在文件层变成：

- 相位平均图；
- 只有表面压力；
- 孤立快照；
- 无移动壁面/edge 身份的外流 PIV；
- 论文可申请，但没有可下载场。

本轮出现了同一失败模式的更强反例。4TU NACA643418 v2 的官方摘要写明实验
使用“PIV and surface pressure measurements”，但远程 ZIP 中央目录的实际
`1389` 项全部属于压力采集和处理链：

- `Pressure/unsteady` 分支占 `1281` 项；
- 文件名中 `PIV/velocity/flowfield` 命中数为 `0`；
- README 只给出 `process_unsteady_pressure.m` 的使用说明；
- 抽取的原始 MAT 只有 `ChanNames/ConvertVer/ConvertedData`；
- `253` 个非空采集通道各有 `24000` samples，不含 `x,y,z,u,v,w`。

异常按文件而非摘要裁决：公开附件没有 PIV 数组。

### 1.2 挂树

- 合格速度场缺口：`N2.6b4f3b`，保持 **open**；
- metadata→payload 身份审计：新增 `N2.6b4f3b2`；
- 完整空间场→逐面板压力：`N3.1i1d`，保持 **open**；
- 压力-only 只提供输出外验、不能识别隐藏涡态：新增 `N3.1i1d1`。

### 1.3 可动空间

允许：

- 新增独立压力数据 loader/observation contract；
- 对模型预测的 `Cp(s,t)` 做完全未调参的外验；
- 继续请求/生成坐标化速度场，推进 V-gate。

禁止：

- 用压力残差反调 LESP、`cds/Tv/Tvl` 或涡片幅值；
- 从压力传感器反演一个“唯一涡心”并宣称空间态已验证；
- 把 reverse-flow、VG、`Re=1e6` 资产当作 RoboEagle 目标域训练集；
- 由电影、JPG 或论文等值图数字化制造速度场。

## 2. ② 学科机理与一手来源

### 2.1 A0/LESP 是事件量，不是持续载荷幅值

Ramesh 等的 LESP-DVM 用临界 LESP 触发前缘放涡，已形成涡的强度与位置随后
独立演化。Narsipur 等表明分离后黏性 LESP 会明显回落。Sudharsan & Sharma
进一步区分了 LESP 的前缘事件意义和局部壁面涡量通量事件。

Rezapour & Mulleners 2026 的新结果把这个边界扩展到恢复过程：

- `A0` 回穿可以作为重附着启动的事件指标；
- 后续仍有向下游传播的 shear-layer wave；
- wave propagation 与 boundary-layer relaxation 具有独立时间尺度；
- 论文明确指出临界值随 Reynolds 数和翼型变化。

所以 `A0` 在“起涡/重附着事件边界”上有物理作用，但不能替代涡强度、位置、
运动和压力形成状态。

### 2.2 压力是必要输出，但不是隐藏状态的唯一观测

Li & Wu 的 vortex-force 分解、Wang & Eldredge 的 impulse matching，以及
Hirato 的空间涡片/非定常压力链共同支持：持续涡载荷取决于涡量分布、位置/
运动和时间率，而不仅是一个临界标量。

反方向不成立。有限表面压力观测即使足以积分出总力，也通常不唯一确定离翼
涡量分布。`N3.1h2` 已用矩问题反例冻结了“有限矩唯一重构面压力”；本轮同样
不允许把 pressure-only 数据变成隐藏涡态的伪标签。

### 2.3 本轮官方数据事实

#### 4TU NACA643418 reverse-flow v2

- DOI `10.4121/8cec2ee4-9dfa-47ba-a71d-69e73b625d38.v2`；
- CC BY 4.0；
- 归档 `3,261,847,090` bytes，MD5
  `7ceb734a1835b88e2df73018ac215413`；
- README：`Re=250k`、150 cycles、mean AoA `155--170°`；
- 文件级结论：压力-only，未发布摘要提到的 PIV。

#### 4TU DU-97-W-300/VG v2

- DOI `10.4121/374c2baa-aca1-487e-8463-6ef167569be7.v2`；
- 关联论文 DOI `10.5194/wes-11-1971-2026`；
- CC BY-SA 4.0；
- 归档 `1,706,315,457` bytes，MD5
  `171d4e6b76f82cb3c464916a21aeddb0`；
- ZIP `120` 项，只含 TSV/TXT；
- 包含稳态/非稳态 `Cp`、积分力、全时序、phase mean/std；
- 无 PIV/velocity/flowfield 项，且 `Re=1e6`。

#### Princeton ramp-pitch

- DOI `10.34770/b3vq-sw14`；
- CC BY 4.0；
- NACA0021，`Re=0.5m--5.5m`；
- 32 个压力孔，150 个 half-cycles 的 phase average；
- 是压力外验资产，不是速度场。

#### NDSU swept finite wing

- DOI `10.3390/fluids6120457`；
- NACA0012，AR=4，`Re=200k`， sweep `0/15/30°`；
- 论文报告四个展向站的 2D PIV；
- 方法部分明确数据由 selected snapshots/phase averages 构成；
- Data Availability 声明为向通讯作者申请。

其有限翼/sweep 身份有机制价值，但公开状态和时间表示均不足以驱动物质 flow
map。

#### 注册表反证边界

对十项高保真 LES/DDES/finite-wing 论文 DOI：

```text
10.2514/6.2024-1352
10.1063/5.0239448
10.2514/1.J056108
10.2514/1.C035427
10.2514/1.J062109
10.2514/1.J063039
10.2514/1.J058681
10.1017/jfm.2018.939
10.1115/FEDSM2014-21588
10.3390/fluids6120457
```

Crossref `relation={}`，DataCite 以论文 DOI 查
`relatedIdentifiers.relatedIdentifier` 均返回 `0`。这只说明注册表没有
正式关联的数据 DOI，不能证明作者没有原始数据，因此下一步仍可走请求路线。

另外两项“公开 dataset”经 payload 审计：

- AIP 2026 plunging/plasma collection 只有四个 MP4；
- SciELO 2019 OpenFOAM 记录只有 12 个 JPG 和两个 XLS。

它们不是数值场。

## 3. ③ 缺组件还是组件错误

### 3.1 组件错误

把“论文/摘要声称使用 PIV”解释成“公开附件含 PIV”，是数据身份组件错误。
NACA643418 v2 给出了直接文件级反例。

把 pressure-only 数据用于反标定隐藏涡态，也是逆问题身份错误。它会把
`Cp` 残差吸收到涡强度/位置，产生恰好被研究纪律禁止的常数吸收或状态补丁。

### 3.2 缺组件

V-gate 仍缺：

- 同一流动事件的连续三分量速度；
- 网格/坐标、移动壁面、双侧和 edge 身份；
- 目标 Reynolds/转捩/分离/三维横流联合覆盖。

P-gate 不再缺数据。现在缺的是一个严格的压力 observation contract，把模型
产生的逐面板压力投影到传感器位置，并把 `Cp` 积分、力/矩和不确定度放在同一
守恒账中。

### 3.3 方向裁决

```text
空间状态证据：缺组件，继续 acquisition / independent generation
压力输出验证：数据已具备，推进独立 observation gate
现有 V4.1 力公式：本轮不改
```

这不是把最终方向退化为 pressure surrogate。生产链仍是：

```text
空间涡态 → 统一面板压力 → 守恒气动载荷
```

只是把隐藏状态验证与输出验证并行，而不再错误串联。

## 4. ④ 有证据的方案与预登记

### 4.1 P0：pressure-only observation contract

实现前冻结以下规则：

1. 数据侧输入仅为几何、传感器弧长/侧别、时间/相位、运动学、来流和发布的
   uncertainty/std；
2. 模型侧输入为未经该资产调参产生的 panel pressure；
3. 用保守曲面投影把 panel `Cp` 投到传感器或把传感器积分到公开力；
4. loader 自洽门要求压力积分与发布力/矩的差落入发布不确定度和离散积分误差
   的传播区间，不另设经验百分比；
5. 模型门至少分别报告：
   - chordwise `Cp(s,t)`；
   - suction-peak/pressure-wave 的位置与相位；
   - pressure-integrated force/moment；
   - cycle mean、phase mean 和 cycle variability；
6. 三套 out-of-domain 资产只能做 falsification/stress test；不得训练生产
   系数，也不得使 N3 空间态晋升；
7. pressure gate 通过也必须保持
   `spatial_state_physical_promotion=false`。

### 4.2 V0：继续取得隐藏状态证据

请求顺序扩为：

1. Baldan/Guardone `Re=135k, span/c>=1` WRLES；
2. Lee/Chanez/Gross `Re=100k/200k/400k` 三维 LES 连续窗口；
3. Batther/Lee `Re=200k` DDES 连续场；
4. NDSU finite-wing raw phase-locked PIV，用于 sweep/crossflow 拓扑而非
   连续材料流；
5. 4TU NACA643418 未随 v2 发布的 PIV，用于 pressure-vortex paired
   out-of-domain stress test。

未经张明昊授权不发送请求。

### 4.3 Go/no-go

- 第二轮公开空间场：**NO-GO**；
- 独立统一压力 observation gate：**GO**；
- 任何生产气动力改写：**NO-GO，尚未取得 V-gate 证据**；
- claim 影响：
  - `N2.6b4f3b` 保持 open；
  - `N3.1i1d` 保持 open；
  - 新增两个窄义 validated/frozen 身份命题；
  - 不改 N1/N4，不复活 N3.1f，不改 `cds/Tv/Tvl/LESPcrit`。

## 5. 可复查资产

- `high_fidelity_field_candidate_inventory_20260728.yaml`
- `audit_high_fidelity_field_candidates.py`
- `high_fidelity_field_candidate_audit.json`

正式 deep audit 结果：

```text
all checks=true
spatial_state_gate=NO-GO
independent_pressure_output_gate=GO
physical_promotion=false
```

## 6. 一手来源

- 4TU NACA643418 v2:
  https://doi.org/10.4121/8cec2ee4-9dfa-47ba-a71d-69e73b625d38.v2
- 4TU DU-97-W-300/VG v2:
  https://doi.org/10.4121/374c2baa-aca1-487e-8463-6ef167569be7.v2
- Princeton Dynamic Stall Database:
  https://doi.org/10.34770/b3vq-sw14
- Ullah et al., *Fluids* 6 (2021) 457:
  https://doi.org/10.3390/fluids6120457
- Lee, Chanez & Gross, *Physics of Fluids* 36 (2024) 117169:
  https://doi.org/10.1063/5.0239448
- Batther & Lee, *Computers & Fluids* 249 (2022) 105691:
  https://doi.org/10.1016/j.compfluid.2022.105691
- Rezapour & Mulleners, *Journal of Fluid Mechanics* 1029 (2026) A52:
  https://doi.org/10.1017/jfm.2026.11228
- Ramesh et al., *Journal of Fluid Mechanics* 751 (2014):
  https://doi.org/10.1017/jfm.2014.297
- Narsipur et al., *Journal of Fluid Mechanics* 900 (2020) A25:
  https://doi.org/10.1017/jfm.2020.467
- Sudharsan & Sharma, *Journal of Fluid Mechanics* 996 (2024), A11:
  https://doi.org/10.1017/jfm.2024.753
