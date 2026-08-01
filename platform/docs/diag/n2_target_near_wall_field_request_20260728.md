# N2.6 目标近壁时序场数据请求规范

日期：2026-07-28  
用途：只用于无载荷的边界层状态、材料 flow map、物质-spike 和守恒释放
验证；未经张明昊明确授权，不发送邮件或外部消息。

## 1. 为什么请求

FLUXV 正在研究 `Re=1.1e5--1.9e5` 扑翼/俯仰动态失速中的：

```text
近壁剖面 → 流体材料面 → 分离backbone → 涡量/卷吸释放
```

公开论文和归档通常只给总力、表面压力、相位平均图或外流区 PIV，无法判断
有限 IBL 状态是否足以重构材料分离事件。请求的数据不会以 lift、drag、
pressure residual 或 LESP 作为 decoder 输入。

## 2. Tier A：目标 Re 周期翼段提前证伪包

首选对象是 Baldan/Guardone `Re=135k` pitching NACA0012 wall-resolved
LES。该研究是三维 spanwise-periodic 翼段，并指出完整周期 DSV 物理要求
`span/chord≥1`；数据可向通讯作者 `alberto.guardone@polimi.it` reasonable
request。它可提前检验目标 Re 的三维局部材料分离/释放，但没有有限翼翼尖
横流，不能单独使 RoboEagle 生产节点晋升。若只能取得二维场，则角色进一步
降为二维退化证伪。

最低请求：

- 一个完整俯仰周期，至少覆盖 attached → onset → LEV growth →
  detachment → reattachment；
- 每个保存时刻的 `t, alpha, alpha_dot` 及刚体壁面位置/速度；
- 双侧近壁三分量速度 `u,v,w` 和密度；若只能给 spanwise 平均场，需同时
  说明 averaging 方式并保留至少一个三维子窗；
- 网格坐标、cell/node 拓扑、无效值掩码和壁面节点编号；
- 解析后的双侧 wall-normal rays，或足以无歧义重建它们的网格；
- wall shear/skin friction 如已有可一并提供，但不作为唯一分离标签；
- edge 的定义、edge 位置/速度和局部外流状态；
- 原始时间步、输出间隔、空间单位、速度单位、参考系和边界条件；
- Reynolds/Mach、转捩处理、湍流/亚格子模型和数值耗散说明。
- 若为 PIV：body/wall mask、无效矢量标志、相关峰/信噪比或作者采用的
  a-posteriori velocity uncertainty；若已做插值/滤波，请给原始场和处理
  参数；
- 若为 CFD：至少一个时间保存频率或时间步加密子窗，用于区分真实动力学与
  时间插值截断误差。

建议交换格式：

```text
HDF5/Zarr
  /time                         [nt]
  /mesh/coordinates             [nnode,2 or 3]
  /mesh/connectivity            [ncell,nvertex]
  /wall/node_ids_upper/lower
  /kinematics/wall_position     [nt,nwall,3]
  /kinematics/wall_velocity     [nt,nwall,3]
  /field/velocity               [nt,nnode,3]
  /field/density                [nt,nnode]
  /field/valid_mask             [nt,nnode]
  /edge/...                     documented convention
```

若完整场过大，可先给连续的 onset 前后时间窗；不能用互不关联的相位平均帧
代替连续时间窗。

### 2.1 Tier A0 备选：PBFM 上游二维 RANS 原生体场

Baldan 2026 博士论文 Appendix F.B.3 确认，公开 PBFM 表面 HDF5 的上游
是 Fluent 2024R2 生成的 `512 surface × 128 wall-normal` O-grid 二维
可压缩 URANS，每周期 2048 步。公开 HDF5 仅保存表面位置×时间场，但作者
可能仍持有原生 case/data。

若 WRLES 暂时无法共享，可先请求一个未参与任何模型选择的 test condition：

- 一个完整周期的 native Fluent case/data，或
  `x,y,u,v,rho,p,T,valid_mask` 导出；
- 512×128 原始 O-grid 或不降低法向近壁分辨率的子域；
- 2048 个原始保存时刻；若存储过大，给 onset 前后连续子窗并说明输出间隔；
- wall node、上/下表面、壁面刚体运动和参考系；
- SST+intermittency 状态、edge convention 与任何已有 wall-distance 字段。

该包只用于二维数据管线、profile/edge 与 material-flow-map 提前证伪，
不能使三维横流、自由 LEV 或 RoboEagle 生产节点晋升。

### 2.2 二级获取路线及证据上限

若首选 WRLES 暂不可得，按下列顺序请求。每条路线的作用域在请求前冻结；
取得文件后不得因数据稀缺而抬高证据等级。

1. **Lee, Chanez & Gross 2024 三维 LES**：
   `Re=100k/200k/400k`，但计算展宽约 `0.1c`。请求连续三分量体速度、网格、
   壁面运动和保存频率。即使取得，也只用于 span-limited 三维提前证伪；
   未覆盖 `span/c≥1` 时不得证明完整 DSV 横向尺度或有限翼生产身份。
2. **Batther & Lee 2022 DDES**：
   `Re=200k`，官方数据声明为向作者申请。请求一个未参与闭合选择的连续场
   窗口，字段遵循 Tier A。它可作独立数值方法交叉证伪；DDES 模型误差和
   几何/运动学差异必须单独标注。
3. **NDSU swept finite-wing PIV**：
   NACA0012、AR=4、`Re=200k`、sweep `0/15/30°`，公开论文只报告四个
   展向站的 selected snapshots/phase averages。若能取得 raw phase-locked
   PIV，允许检验 sweep/crossflow 拓扑及展向一致性；非连续三维序列不得
   推进 material flow map 或 V-gate。
4. **4TU NACA643418 reverse-flow PIV**：
   v2 摘要称实验包含 PIV，但 `3.26 GB` 公开归档的文件级审计只发现压力
   采集/处理链。请求摘要对应的坐标化 PIV、掩码、相位/时间身份，并与已
   公开压力配对。其 `Re=250k` 和 reverse-flow 身份只允许
   pressure-vortex paired out-of-domain stress test。

这些路线均不改变 Tier A/Tier B 的 GO 条件。未经张明昊明确授权，不发送
任何请求。

## 3. Tier B：三维生产资格包

用于最终 RoboEagle/代表有限翼生产资格。除 Tier A 全部字段外还必须含：

- 有限翼曲面三角/四边形拓扑、双侧身份和展向坐标；
- 三分量速度及展向横流；
- 可动/扭转壁面逐时刻几何和速度；
- 多个 wall-normal 层，覆盖黏性层到已声明 edge；
- 至少一组未参与任何闭合选择的 independent test case；
- `Re=1.1e5--1.9e5`，并记录局部相对速度导致的 span/time Re 范围；
- 动态、分离、转捩、移动壁面和三维横流联合出现于同一场数据；
- 数据 DOI、版本、checksum、许可和可复查生成脚本/输入文件。

## 4. 不需要和禁止使用的字段

作者可以附带力/压力用于最终外验，但以下量不会进入 profile decoder、
材料面推进或分离流形定位：

```text
lift / thrust / total force / moment target
LESP
pressure residual
V4.1 prediction residual
structural response
```

不请求作者为 FLUXV 预先计算某个经验阈值；希望获得尽可能原始且带身份的场。

## 5. 收到数据后的预注册 go/no-go

GO 必须同时满足：

1. checksum、许可、坐标/单位/参考系可审计；
2. 壁面无滑移残差在作者声明的离散误差内；
3. 双侧、法向、edge 和时间身份无歧义；
4. 对 proper Euclidean 观察者变换，标量诊断保持不变；
5. flow map 对时间步和法向层距收敛；
6. backbone 对网格、层距和积分时间窗收敛；
7. Tier A 只允许二维退化裁决；Tier B 才允许三维生产晋升。

任一关键身份缺失即 NO-GO。不得从图片反演坐标、把平均场复制成时间序列，
或将不同数据集拼接成同一材料事件。

## 6. 可发送的英文请求正文

Subject: Request for near-wall time-resolved velocity fields for an
independent dynamic-stall model audit

Dear [Author],

We are developing a physics-traceable low-order aerodynamic model for a
flapping wing at chord Reynolds numbers of approximately
`1.1e5--1.9e5`. We would like to test, independently of force data, whether
near-wall integral states can reconstruct material-surface separation and
vorticity release.

Would it be possible to obtain one continuous time window of the Re=135,000
wall-resolved LES velocity field, ideally covering attached flow, onset, LEV
growth/detachment, and reattachment? The most useful package would include
mesh coordinates/connectivity, upper/lower wall node identities, wall
position and velocity, three-component velocity (and density) snapshots,
validity masks, time/kinematic metadata, and the convention used to define the
boundary-layer edge. HDF5, Zarr, or the native solver format would all be
acceptable.

If sharing the WRLES field is currently impractical, a smaller alternative
would already be very useful: one held-out native Fluent volume-field case
from the 512×128 O-grid, 2048-step dynamic-stall simulations used upstream of
the public PBFM surface dataset. We would use this explicitly as a
two-dimensional RANS pre-falsification case, not as three-dimensional LES
evidence.

The field would be used only for a no-load validation of profile
reconstruction, material flow maps, and separation backbones. Lift, drag,
pressure residuals, LESP, and structural response will not be used as model
inputs. We will cite the original work and follow any license, storage, or
co-authorship conditions you specify.

For either option, if the full cycle is too large, a continuous onset-centred
subset with the same metadata would already be valuable.

Best regards,  
Zhang Minghao  
Xi'an Technological University
