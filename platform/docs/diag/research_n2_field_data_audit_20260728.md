# N2.6b4f / N2.6c1 独立近壁场资产审计

## 0. 裁决问题

当前不是“有没有一篇动态失速论文”，而是：

```text
有没有可合法取得、可审计、带时空坐标和移动壁面身份的独立速度场，
足以检验 four-plus-two profile H0 并推进近壁流体材料面？
```

论文中的图片、数据可申请声明、总力曲线和只有外流区域的 PIV 均不能回答该
问题。审计标准已经冻结在 `near_wall_field_contract_cases.yaml`。

## 1. 目标数据的最低身份

目标测试集必须同时包含：

- `Re=1.1e5--1.9e5`，并覆盖局部相对速度扩域；
- 移动壁面、动态、分离、转捩；
- 双侧近壁法向速度剖面及 edge 命名规则；
- 三维曲面拓扑和横流；
- 时间连续的速度场，而不是互不关联的相位图片；
- 独立 test split 和可复查来源；
- 不以 L/T、压力残差、LESP 或结构响应为输入标签。

这是一项联合覆盖要求。某个数据集满足其中一两项，不等于可以把缺失项从
其他不相容数据集拼接补齐。

## 2. 一手/官方资产分层

| 资产 | 公开状态 | 可支持的命题 | 不能支持的命题 |
|---|---|---|---|
| Baldan & Guardone, NACA0012 WRLES, `Re=135k` | 论文公开；原始数据为 reasonable request | 目标 Re、三维周期翼段、转捩/分离/完整俯仰周期；论文证明完整 DSV 需 `span/c≥1` | 文件未取得/审计；无有限翼翼尖横流，当前不可执行 H0、flow map 或 ridge |
| Sudharsan/Sharma BEF wall-resolved LES，`Re=200k--500k` | 论文公开；未发现原始场公开仓 | onset/BEF 与涡量形成事件 | 目标场训练、三维横流及 RoboEagle 释放 |
| Deep Blue pitching flat plate DNS，DOI `10.7302/cwsp-7q09` | CC BY 4.0；单动态文件 1.61/4.02 GB | 2D moving-field 文件/算法参考 | `Re=100`，无三维横流和目标转捩 |
| Deep Blue NACA0012 LES，DOI `10.7302/0e3g-6j84` | CC BY 4.0；动态包 35.6 GB 起 | 静态分离/转捩的二维切片 | `Re=23k`、静态、无移动壁面、无完整 3D 近壁场 |
| NREL/OEDI airfoil high-fidelity archive | 公开力时序/极曲线 | 后续独立总力外验 | profile decoder 或材料面训练 |
| 4TU NACA0021 surging PIV，DOI `10.4121/0B01A240-559F-4F4B-802D-7B1D36AA0024` | CC BY 4.0；60.5 MB，已取得并校验 | 独立实验归档和失败模式审计 | 见第 3 节，不能进入 b4f/b2b |

这里“不能支持”不是对数据质量的否定，而是对 claim 类型和适用域的限制。

## 3. 4TU 独立 PIV 的实际文件审计

### 3.1 获取身份

持久外部缓存：

```text
platform/data_external/4tu_surging_naca0021/data.zip
```

目录已被 Git 忽略，不使用 `/tmp`。归档：

- 大小：`60,546,364` bytes；
- SHA-256：
  `10ff83472157e28ea05bb168caa15713aac6e104211cc2275a4ae538e0be9156`;
- DOI：`10.4121/0B01A240-559F-4F4B-802D-7B1D36AA0024`;
- 许可：CC BY 4.0；
- 论文报告：NACA0021、90°、surging、`Re=1.5e4`、phase-averaged PIV。

### 3.2 数据指纹

`field_asset_4tu_surging_audit.json` 对全部 MAT 文件逐一读取，发现：

1. 两个频率各有 12 个相位，动态文件共 24 个；
2. 23 个文件为 `165x326`，`fre5/Phase90` 为 `165x325`；
3. 动态文件含 `U_final/V_final/W_final` 和 Reynolds-stress 通道，但全部
   缺少 `x/y` 坐标；
4. 坐标只存在于静态文件，其网格为 `166x219`，没有发布映射，禁止移植；
5. 动态速度场最低有限值比例为 `0.95817`，反映遮挡/无测量区；
6. 数据是 phase average，不是用于任意 `[t0,t]` 流映射的连续瞬时序列。

README 声称各文件含坐标，但实际动态 MAT 变量与该描述不一致。按“异常先信
数据”纪律，以文件内容为权威，不能用 README 替代数组。

### 3.3 裁决

该资产对 `N2.6b4f3` **NO-GO**，缺失：

```text
dynamic coordinates
continuous time snapshots
wall position/velocity
surface basis and normal rays
edge position/velocity/convention
double-side surface topology
target Reynolds coverage
3D surface crossflow
transitional near-wall profiles
```

允许用途只有：

- 独立归档 checksum / MAT schema 审计；
- 验证缺坐标、错网格、NaN 遮挡必须被拒绝；
- 在发布元数据范围内做相位流场视觉检查。

禁止猜测动态坐标、从 GIF 反演网格或以静态坐标强行配准。它不进入
four-plus-two H0、材料 flow map、spike ridge 或 V4.1 载荷晋升。

## 4. 扩展公开资产审计

`external_field_candidate_inventory.yaml` 把后续取得的五类资产与 KTH
静态剖面库逐项放回同一个冻结契约。`audit_external_field_candidates.py`
不看载荷，不把不同数据集取并集，并对实际抽取的 Zenodo MAT 和 Edinburgh
CSV 做 checksum、变量和维度核验。

| 资产 | 数据指纹 | 可保留角色 | 生产裁决 |
|---|---|---|---|
| Cambridge translating plate, DOI `10.17863/CAM.7816` | `Re=10k`；`x,y,u(x,y,t),v(x,y,t)`；100 Hz；二维平移平板 | 连续二维场读取/积分器压力测试 | NO-GO：论文明确翼下激光阴影无可靠近壁数据；无双侧壁面/edge；Re 不符 |
| Glasgow dynamic-stall database, DOI `10.5525/gla.researchdata.464` | `Re≈1.5e6`；6510 个动态压力试验；表面 Cp/角度/积分载荷 | 后续独立面板压力与载荷外验 | NO-GO：没有速度场 |
| Zenodo feather NACA2414, DOI `10.5281/zenodo.13831092` | `Re=200k`；平面 PIV；平均场加少数 `τ` 快照 | 后失速剪切层位置与单帧 schema 审计 | NO-GO：发布数据不是连续 1000 帧速度序列；静止翼、二维、无壁面/edge |
| Edinburgh Ōtomo pitching PIV, DOI `10.7488/ds/7677` | `Re=32k`；四种运动各 1666 帧；`dt=0.006 s`；`126×126` x/y/u/v | 连续真实二维场插值、低 Re 负迁移试验 | NO-GO：无壁面拓扑/法向/edge，且 Re/三维横流不符 |
| KTH NACA4412 LES profiles | `Re=100k,200k,400k,1M`；静态双侧局部切/法向平均剖面 | 静态剖面坐标与矩一致性单测 | NO-GO：无时间、移动壁面、动态分离或横流 |

### 4.1 Zenodo 文件级事实

远程 ZIP 中央目录含 146 项。`PIV Guide.docx` 明确每个工况只有一个
`Avg.mat`，部分工况另有 `Tau_0.25/0.50/0.75.mat`。实际抽取：

- `00000/Avg.mat`：`147×288` 的 `P,omega,u,v,x,y`；
- `00000/Tau_0.50.mat`：同样只有一个 `147×288` 的
  `omega,u,v,x,y` 场；
- 该瞬时 MAT 的约 54.5 MB 主要来自 `1760×3456` 的 `I_image/x_image/
  y_image`，不是连续速度帧。

因此论文“1 kHz time-resolved acquisition”不能被替换成“公开归档提供了
1 kHz 连续速度数组”。这是发布内容与原始采集能力的身份差异。

### 4.2 Edinburgh 文件级事实

`32_sym.zip` 的远程中央目录有 1667 项：一个目录加 `LER_00001.csv` 至
`LER_01666.csv`。每帧四列 `x,y,u,v`。抽取首、中、末三帧后：

- 每帧 `15,876×4`，对应固定 `126×126` 网格；
- 三帧均有限，坐标范围相同；
- README 给出 `dt=0.006 s`。

这证明它是真正的连续二维 Eulerian 场资产，但不是契约要求的壁面随体剖面
资产。文件没有曲面节点、侧别、法向射线、壁面速度和 edge 约定，不能仅凭
NACA0018 几何和已发表运动学自行补成“测得的近壁场”。

### 4.3 联合裁决

六项资产的 `production_eligible_assets=[]`。该结论不是“没有可用数据”，
而是：

1. Cambridge/Edinburgh 可降低真实二维时空插值实现风险；
2. KTH 可降低静态 profile/edge 坐标实现风险；
3. Glasgow 可在统一面板压力形成后提供外部压力/载荷验证；
4. 它们不能拼成一个从同一流场同时观测到的材料演化事件。

在材料 flow map 问题上拼接不同 Re、几何、运动和测量平面的数据，会破坏
同一流体材料粒子的身份，因而不是 multi-fidelity，而是命题偷换。

### 4.4 Edinburgh 真实场时序插值预门

在抽取相邻帧前，`external_piv_temporal_interpolation_cases.yaml` 冻结
中心帧 833、stride `1/2/4`、尺度无关中点/persistence 比和误差单调门。
结果：

- 七帧坐标残差为 0，数据全部有限；
- stride 1 的线性中点对 `u/v` 都优于 persistence，误差比分别为
  `0.719/0.861`；
- `v` 的中点误差随 stride 增大单调增加；
- `u` 的误差为 `0.01464/0.02213/0.01691 m/s`，不单调，预门 NO-GO。

因此 `N2.6c1b2b0`“原始二维 PIV + 朴素线性时间插值可直接作为材料轨迹
驱动”被证伪并冻结。完整机理与后续可动空间见
`research_n2_real_field_interpolation_decision_20260728.md`。不允许通过
删除 stride 4、只看 `v` 或无锚平滑把结果改成通过。

## 5. 缺组件还是组件错误

### 组件错误

把公开的“动态 PIV/LES”元数据直接视为可训练近壁场，是组件身份错误。数据
必须先经过坐标、材料时间、壁面、edge、Re 和流态联合审计。

把实验曾以 1 kHz 采集，等同于归档已发布 1 kHz 连续速度场，也是组件身份
错误。Zenodo 文件级审计已经直接证伪这一替换。

### 缺组件

当前真正缺的是至少一个：

1. 目标域独立 wall-resolved CFD/PIV 的可下载近壁时序场；或
2. 按冻结协议独立生成的 RoboEagle/代表有限翼验证场。

四加二 decoder、field interpolation 和 ridge 算法都不能制造这份证据。

### 方向判定

当前裁决是“**缺物理证据组件**”，不是“已有气动公式再调一个常数”，也
不是“曲率/积分器公式错误”。因此：

- 不改 N1/N4；
- 不把 Cambridge/Edinburgh 低 Re 二维结果晋升成生产分离律；
- 不开始 `N2.6b4f4` 的生产 decoder 训练；
- 先完成可发送的数据请求规范，同时允许 Edinburgh 进入一个明确标为
  non-promotion 的二维真实场插值预门。

## 6. 下一步 acquisition ladder

按成本和证据价值排序：

1. 使用 `n2_target_near_wall_field_request_20260728.md` 优先向
   Baldan/Guardone 申请 `span/c≥1, Re=135k` 原始 WRLES 近壁场；论文已
   明确数据可 reasonable request，但未经用户授权不代发邮件；
2. 若获得该三维周期翼段场，先做 b4f H0、b2b 和三维物质-spike 门；其
   spanwise 三维性有价值，但不能替代有限翼翼尖横流；
3. 再独立生成/取得有限翼、横流、移动壁面场，执行最终目标三维门；
4. 只有两层都通过，才训练 `N2.6b4f4` 并推进 `N2.6c1b3`。

二维/周期翼段数据不能使有限翼生产节点晋升，但可以提前证伪 four-plus-two
H0、场插值或分离 backbone。

## 7. 来源

- Baldan, M. & Guardone, A., *Wall-resolved large eddy simulations of a
  pitching airfoil incurring in deep dynamic stall*, arXiv:2405.12036.
- Sudharsan, M., Ganapathysubramanian, B. & Sharma, A.,
  *A vorticity-based criterion to characterise leading-edge dynamic stall
  onset*, Journal of Fluid Mechanics.
- Towne, A. & Dawson, S., *Low-Reynolds-number pitching airfoil direct
  numerical simulations*, DOI:10.7302/cwsp-7q09.
- Towne, A. et al., *Turbulent airfoil wake large eddy simulation*,
  DOI:10.7302/0e3g-6j84.
- Xu, G. et al., *PIV measurement flow field of the airfoil NACA0021 under
  surging motion*, DOI:10.4121/0B01A240-559F-4F4B-802D-7B1D36AA0024.
- Xu, G. et al., *On the unsteady aerodynamics of a surging airfoil at
  90° incidence*, Experiments in Fluids 66 (2025) 85,
  DOI:10.1007/s00348-025-04011-2.
- Graham, W. R., Pitt Ford, C. W. & Babinsky, H., *An impulse-based
  approach to estimating forces in unsteady flow*, JFM 815 (2017),
  DOI:10.1017/jfm.2017.45; data DOI:10.17863/CAM.7816.
- Green, R. B. & Giuni, M., *Dynamic stall database R & D 1570-AM-01:
  Final Report*, data DOI:10.5525/gla.researchdata.464.
- Sedky, G. et al., *Distributed feather-inspired flow control mitigates
  stall and expands flight envelope*, PNAS 121 (2024),
  DOI:10.1073/pnas.2409268121; data DOI:10.5281/zenodo.13831092.
- Ōtomo, S. et al., *Unsteady lift on a high-amplitude pitching aerofoil*,
  Experiments in Fluids 62 (2021); data DOI:10.7488/ds/7677.
- Vinuesa, R. et al., *Turbulent boundary layers around wing sections up
  to Rec = 1,000,000*, Int. J. Heat Fluid Flow 72 (2018), KTH WingData.
