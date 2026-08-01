# N3 空间涡态生产方向：机制证据与候选边界

日期：2026-07-28  
阶段：Phase ②文献机理 → Phase ③方向预裁决  
状态：**literature-only；不改 claim 状态，不授权 production、h/p、118 或 Fig17/18/19**

## 1. 当前问题与可动空间

现有 V4.1 的生产 N3 将 `A0/LESP → Tv → dCN_ds` 作为持续正涡力幅值链。
D4/D5 已经证伪无符号/有符号旧标量重排足以修复角区趋势；N3.1h2 又证伪
“有限总力矩唯一恢复面板压力”。因此当前可动空间不是继续调 `Tv`、`crit`
或 `cds`，而是 N3.1j 下的：

1. `N3.1j...b3e3b`：物理可达兼容路径上的动力残差横截门；
2. `N3.1j...b3c`：actual body-newborn physical-Kutta formation solve；
3. `N2.6c2c`：只有无质量形成片不足时，才可能打开的光滑前缘
   IBL→VES 物理 junction。

N1/N4 已 validated/frozen；任何方案都必须复用 N1 的 body/wake 势流状态和
N4 的唯一力账，不得并联第二套总力闭合。

## 2. 文献证据矩阵

| 来源 | 文献直接支持 | 对 FLUXV 的许可 | 不许可的外推 |
|---|---|---|---|
| Ramesh et al., JFM 751 (2014), [doi:10.1017/jfm.2014.297](https://doi.org/10.1017/jfm.2014.297) | `A0` 与 LESP 对应；超过临界 LESP 时触发 LEV 离散放涡，随后由离散涡演化预测流动与力 | `A0/LESP` 作为起涡/供给边界条件；LEV 必须拥有独立涡态 | 不支持把 `A0` 本身持续映射为正法向力幅值 |
| Ramesh, JFM 886 (2020), [doi:10.1017/jfm.2019.1070](https://doi.org/10.1017/jfm.2019.1070) | `A0` 控制薄翼前缘吸力奇性、局部前缘速度和驻点位置 | 把 `A0` 解释为局部前缘临界量 | 不支持由单个 `A0` 唯一确定已经形成的 LEV 强度、位置和载荷 |
| Sudharsan & Sharma, JFM 996 (2024), [doi:10.1017/jfm.2024.753](https://doi.org/10.1017/jfm.2024.753) | 低 Reynolds 数下，LESP 只捕获显著影响前缘 `Cp` 的事件；远离前缘的多涡事件需要空间定位的 vorticity/BEF 信息；临界 LESP 依赖几何和工况 | N3 必须保留事件的空间身份；LESP 门不能承担全流场幅值闭合 | 不支持以单一、跨工况固定 LESP 常数替代空间演化 |
| Pitt Ford & Babinsky, JFM 720 (2013), [doi:10.1017/jfm.2013.28](https://doi.org/10.1017/jfm.2013.28) | PIV 同时测得 LEV/TEV 的环量与位置；起动升力由惯性项和分布在脱落涡中的环量共同形成 | 至少需要涡强度、位置及非环量项，才能解释载荷历史 | 总力吻合不等价于已获得面板压力分布 |
| Graham, Pitt Ford & Babinsky, JFM 815 (2017), [doi:10.1017/jfm.2017.45](https://doi.org/10.1017/jfm.2017.45) | 冲量法若缺少近壁/物面涡片会产生显著力误差；补齐物面涡片后才与测力吻合 | 空间自由涡态必须与物面 bound-sheet 状态统一计算 | 冲量总力不能反推出唯一表面压力，不能作为第二力账 |
| Li & Wu, JFM 836 (2018), [doi:10.1017/jfm.2017.783](https://doi.org/10.1017/jfm.2017.783) | vortex-force map 的力贡献取决于涡结构相对翼型的空间位置和运动方向 | 可作为“强度＋位置/速度影响载荷”的独立 guard | force map 是力归因工具，不是预测性 LEV 释放或面板压力闭合 |
| Li, Zhao & Graham, JFM 900 (2020), [doi:10.1017/jfm.2020.515](https://doi.org/10.1017/jfm.2020.515) | 三维涡力图表明力可能主要来自锥形涡片而非涡核；有限域外涡还需修正项 | N3 的生产态应优先保存空间涡片/分布，而非只保存单个“核心涡矩” | 单点/单核涡态不能被宣称为三维充分状态 |
| Wang et al., JFM 953 (2022), [doi:10.1017/jfm.2022.956](https://doi.org/10.1017/jfm.2022.956) | 多体 vortex-pressure force 取决于涡的强度、位置和局部速度；几何通过辅助势改变映射 | 支持几何相关、空间相关的载荷观察器 | 仍主要给总力/区域归因，不提供结构所需的唯一面板压力 |
| Darakananda & Eldredge, JFM 858 (2019), [doi:10.1017/jfm.2018.792](https://doi.org/10.1017/jfm.2018.792) | 以自由涡片表示形成剪切层、点涡表示卷起核心；可守恒地在二者间转移环量而避免伪力 | 支持“形成片＋卷起涡”的分层空间状态，以及守恒的降阶/合并 | 三个降阶调节量和二维尖边验证不能直接晋升到 RoboEagle 光滑三维前缘 |
| Dumoulin, Eldredge & Chatelain, JFM 977 (2023), [doi:10.1017/jfm.2023.997](https://doi.org/10.1017/jfm.2023.997) | bound vortex sheet、自由点涡片、unsteady Kutta 与 impulse-preserving circulation transfer 可形成轻量模型 | 支持复用面板势解并将尾迹压缩为守恒空间态 | 论文只处理单一尾缘释放，并含三个复杂度/精度参数；不是 LEV 光滑前缘闭合 |
| Joshi, Soto & Bhattacharya, JFM 1017 (2025), [doi:10.1017/jfm.2025.10495](https://doi.org/10.1017/jfm.2025.10495) | AR=3 动态扭转翼的 3-D LDVM 显式推进 LEV/TEV，逐展向条带耦合 lifting-line；可捕获总升力趋势和部分空间涡结构 | 证明“空间涡态＋展向耦合”能服务扭转翼，而非只能做二维总力 | `LESPcrit=0.21` 由一个测力工况遍历选择，主要验证总升力；论文也明确后续仍需完善 3-D 效应，因此不能原样作为无拟合生产层 |
| Jones, JFM 496 (2003), [doi:10.1017/S0022112003006645](https://doi.org/10.1017/S0022112003006645) | moving plate 的 bound sheet 与两条 free sheets 联立；unsteady Kutta 同时约束释放率和切向释放，并给出力/矩 | 支持物面—自由片统一求解和形成几何的必要性 | 高有效攻角下仍有使时间积分终止的边缘事件，不是普适稳定闭合 |
| Xia & Mohseni, JFM 830 (2017), [doi:10.1017/jfm.2017.513](https://doi.org/10.1017/jfm.2017.513) | 对有限角尖边，unsteady Kutta、环量、质量、动量联立决定新生片方向、强度和相对速度 | 正 first-order obstruction 通过 h/p 后，优先测试无质量 finite-forming geometry | 二维尖边控制体不能直接证明三维光滑前缘 junction，也不引入片面质量 |
| DeVoria & Mohseni, JFM 866 (2019), [doi:10.1017/jfm.2019.134](https://doi.org/10.1017/jfm.2019.134) | VES 具有独立 `gamma`、法向速度跳 `q`、片面质量 `rho_s`、片内动量和压力跳；零卷吸时退化为无质量普通涡片 | 只在正确无质量形成几何后仍有收敛的非零动力余核时，授权 VES companion 候选 | 压力残差本身不能被改名为 `q/rho_s`；尖边示例不闭合 RoboEagle 光滑前缘释放 |
| Proulx-Cabana et al., Fluids 7 (2022), [doi:10.3390/fluids7020081](https://doi.org/10.3390/fluids7020081) | NL-UVLM 负责翼面/近尾迹，VPM 负责远尾迹；VPM 还需要 PSE、SGS 和 FMM | VPM 可作为后续可扩展的自由尾迹运输后端 | VPM 是离散与运输技术，不自动给出光滑前缘释放、unsteady pressure 或唯一结构载荷；全量替换 N1 会越过 frozen 节点 |

## 3. 缺组成部分还是组成部分错误

当前证据支持两个分层判断，但正式数值门尚未返回，因此不晋升：

1. **已足够判定的错件**：`A0/LESP → 持续正 dCN` 是组成部分角色错误。
   文献把 LESP 放在起涡/释放边界；D4/D5 又排除了仅重排该标量。
2. **仍需正式门判定的缺件**：空间形成片几何是否已经足以闭合统一压力，
   还是必须增加 VES 的质量/卷吸状态。这个问题必须由
   `N3.1j...b3e3b` 的 actual-reachable residual、随后独立 h/p 和
   forming-geometry 覆盖率回答，不能由文献相似性直接决定。

## 4. 候选方向裁决

| 候选 | 当前裁决 | 原因 |
|---|---|---|
| 继续调 `A0/Tv/cds/crit` | **NO-GO** | 已证伪标量重排；与 LESP 的文献角色冲突 |
| 用 vortex-force/impulse map 直接给总力 | **诊断 GO，production NO-GO** | 可验证涡态—力关系，但不能给唯一面板压力，会形成第二力账 |
| 直接用完整 3-D VPM 替换 N1/N3 | **首轮 production NO-GO** | 越过 frozen N1；仍缺释放与压力闭合；计算复杂度显著扩大 |
| 条带 LDVM/点涡覆盖层 | **shadow GO** | 可检验强度＋位置＋运动是否解释 D4–D8 指纹；在统一压力和光滑前缘门前不进生产 |
| actual body + 无质量 finite-forming sheet + unified panel pressure | **条件优先候选** | 若正式门给 first-order obstruction 且 h/p 持续，先测试最小新增物理自由度 |
| DDE + VES companion | **后备候选** | 只有正确 forming geometry 后仍存在收敛非零余核，并有独立场级质量/卷吸证据时才可打开 |

## 5. 面向结构设计的最终生产形态

生产目标不是“总力闭合模型”，而是：

```text
A0/LESP 与近壁库存
        │  仅负责事件/释放门
        ▼
形成片 birth state: (位置, 方向, circulation flux, relative velocity)
        ▼
空间自由涡片/守恒压缩涡态: (X, Gamma, material identity, velocity)
        ▼
与 N1 bound sheet / wake 在同一 BIE 与同一时间层联立
        ▼
同一 unsteady Bernoulli / doublet jump 生成每个面板 Delta-p
        ▼
唯一 ForceLedger 积分 + work-conjugate 面板牵引传给结构
```

关键不变量：

- `A0/LESP` 不再直接生成力，只能生成具名 birth flux/事件；
- 空间涡态通过诱导速度/势改变同一个面板压力场，禁止额外叠加 `dCN_ds`；
- 面板压力积分必须与总 body-frame force、风轴转换和结构虚功一致；
- 涡片合并/粒子化只能在环量、冲量及其造成的瞬时力不变的 guard 下进行；
- VES 的 `q/rho_s/rho_s v/Delta p` 不得由目标力、压力残差或旧标量反演。

## 6. 预登记的下一裁决链

1. 等待并独立复核正式 31-history v2.2 结果；任何 guard 失败均为
   `PROTOCOL-NO-GO`，不改物理树。
2. 若是 zeroth-order obstruction：先审查 pressure observation、
   finite-base topology 和 material transfer；禁止跳到 geometry/VES。
3. 若是 first-order obstruction：执行 actual-body 独立 h/p；只有 continuum
   lower bound 保持正值，才打开 `N3.1j...b3c`。
4. 在同一 midpoint stage 联立 Xia–Mohseni 类无质量形成几何；以 junction
   有限速度、Kelvin、BIE、pressure 与 h/p 覆盖率作 GO/NO-GO。
5. 只有第 4 步正确且仍有稳定非零 cokernel，才为 `N2.6c2c` 建立
   独立 IBL→VES 场级预登记。
6. 生产晋升后依次执行：三点复现门 → 118 工况 → 趋势记分卡 →
   Fig17/18/19 数值和人工视觉核对。

## 7. 本轮证据边界

- Semantic Scholar API 本轮返回 HTTP 429；未把缺失的 S2 结果伪装成检索完成。
- arXiv 查询噪声较大，只保留了可由 DOI/出版社页交叉核实的条目。
- 该文献裁决没有运行任何 FLUXV 气动工况，也没有修改 claim YAML、冻结 runner、
  V4.1 参数或生产力路径。
- Joshi et al. (2025) 是当前最接近“有限翼＋动态扭转＋显式 LEV 空间态”的直接证据，
  但其 `LESPcrit` 数据选择与总力验证边界必须保留，不能被写成生产充分性。
