# N2.6g 贴体 body-fixed MRF observer：研究裁决

日期：2026-07-30  
阶段：病因定位 → 一手文献 → 缺件/错件裁决 → 本轮唯一预登记方向  
冻结边界：V4.1、N1、N2/N3 物理、Fig17/18/19 数据和 N2.6f1 失败结果均不改。

## 1. 病因定位与唯一可动节点

`N2.6f1` 的来源门给出三个共同指纹：

1. 有效替代时间轴最后两级 CD/CL 不收敛；
2. 空间 `256 -> 512 points/c` 的 CD/CL waveform 不收敛；
3. 最细 moving PLIC 物面在旧资产标记 44/54 度（来源 nominal
   44/55 度）仍不闭合，但对已有 fragments 的 traction 求和精确。

所以当前病因不是“总力少一个常数”，也不是 LESP 或 \(f_2\) 幅值不足。
唯一能被数据支持的定位是：

> `N2.6f1` 的移动 embedded-boundary 数值表示不能同时提供独立时间/空间
> 响应和水密材料牵引；该实现组件错，不授权观察目标。

可动空间只在 `N2.6 full-domain NS observer` 的数值后端。V4.1 生产力、
validated N1/N4、N3 涡力和目标实验均不可动。

## 2. 学科机理与一手来源

### 2.1 为什么换成机体固连贴体网格

Jee & Moser, *JCP* 231 (2012) 6268--6289,
DOI `10.1016/j.jcp.2012.04.014`，给出 rapidly pitching rigid body 的
守恒积分形式：网格固定于刚体，但求解绝对速度；其目的正是避免每步移动
几何拓扑，同时保持惯性系动量解释。

Chandar & Sitaraman, *CPC* 273 (2022) 108279,
DOI `10.1016/j.cpc.2021.108279`，表明普通 overset 插值并非守恒，并会降低
压力阶或在移动界面产生高幅压力振荡；采用 overset 会新加一个待闭合接口，
不是本轮最小替代。

Schneiders et al., *JCP* 235 (2013) 786--809,
DOI `10.1016/j.jcp.2012.09.038`，明确讨论 moving cut-cell 拓扑改变导致
离散算子/截断误差跳变和非物理力振荡，需要专门的保守平滑处理。这与
N2.6f1 的跨网格/跨相位指纹同类，但不能反向证明任何特定贴体求解器正确。

因此方向裁决为：**缺的不是又一个气动力标量，而是拓扑固定、水密且保留
材料边界身份的贴体离散组件。**

### 2.2 为什么本轮只预登记 Nektar++ v5.9.0

Nektar++ 的 spectral/\(hp\) 方法和软件框架由：

- Cantwell et al., *CPC* 192 (2015) 205--219,
  DOI `10.1016/j.cpc.2015.02.008`；
- Moxey et al., *CPC* 249 (2020) 107110,
  DOI `10.1016/j.cpc.2019.107110`

给出。这些来源支持高阶贴体离散能力，不等于支持本工况准确性。

唯一固定版本为：

```text
tag: v5.9.0
tag object: 40b8298a53804ab0bb3b71e620e95dcee24bf31d
dereferenced commit: f729cda85b6a206e008fd705af8001cfe6e0d6fb
```

官方 tag：
`https://gitlab.nektar.info/nektar/nektar/-/tags/v5.9.0`。

v5.9.0 自带
`MovingRefFrame_Rot_naca0012.{xml,tst,rst}`，其运行身份是：

- `VelocityCorrectionScheme`；
- Galerkin；
- IMEX order 2；
- `MovingFrameWall/MovingFrameFar`；
- 解析 `Theta_z/Omega_z/DOmega_z`；
- quarter-chord pivot；
- 每步 `AeroForces`。

这只是 Re=100、restart-based、10-step field regression，不能冒充
rapid-pitch 或 dynamic-stall 物理验证。

Nektar issue #335 明确旧 MRF pressure boundary condition 遗漏 viscous
term 且 restart 未恢复完整运动初态：
`https://gitlab.nektar.info/nektar/nektar/-/work_items/335`。
MR !1692 于 2024-05-17 合入修复，v5.9 CHANGELOG 明列该项；其
`MovingFrameWall` 先调用 `AddVisPressureBCs`，再加入 rigid-body
acceleration。故 `<v5.7` 明确禁止。

不选择当前 v5.10.0，因为 MR !2040 已把同一回归改入
`VCSFSI + MRFWall/MRFFar + RIGIDSOLVER` 架构；该 MR 的动机是低质量
刚体耦合的 added-mass/pressure-decomposition 稳定化，涉及 60 个文件和
大规模改动。它对 prescribed-motion observer 同时替换后端和 FSI/
pressure-time 架构，违反单一最小候选。精确地说，该回归使用
`VCSFSI`，不是 `PressDecompVCSFSI`。

### 2.3 来源物理和载荷输出边界

Ghosh Choudhuri, Knight & Visbal, *AIAA Journal* 32 (1994) 673--681,
DOI `10.2514/3.12040`，以独立二阶时空算法研究 Re=10000 快速俯仰的
主/次/三级分离结构；Visbal & Shang, *AIAA Journal* 27 (1989)
1044--1051, DOI `10.2514/3.10219`，是当前 source case 的快速俯仰
Navier--Stokes 基础。它们规定需要同时检查力响应和涡拓扑，而不是只看
一个峰值。

Nektar v5.9 `AeroForces` 分开输出 pressure/viscous/total，并能把 MRF 力
转换回惯性系；`FieldConvert extract` 和
`wss:bnd=...:addnormals=1` 能导出边界压力、剪切和法向。此证据只授权
预登记 traction ledger，不预先保证其闭合或材料排序正确。

## 3. 缺件/错件裁决

| 层级 | 裁决 | 理由 |
|---|---|---|
| 全域二维黏性 NS 方程 | 未证伪 | N2.6f1 失败来自离散数值门，不是方程反例 |
| moving PLIC/cut-cell 物面 | 错件并冻结 | 时空不独立且最细物面不水密 |
| overset 接口 | 本轮 NO-GO | 新增非守恒插值与压力振荡组成部分 |
| 贴体 body-fixed MRF | 缺件，允许新 claim | 固定拓扑、绝对速度、无几何穿越接口 |
| 高阶谱元本身 | 未验证 | 官方回归仅 smoke；必须重过 source physics |
| LESP/\(A_0\)/\(f_2\) 持续幅值 | 禁止 | 与本次数值病因无关，既有 claim 边界不变 |

## 4. 本轮唯一预登记候选

本轮唯一预登记 successor 为：

> `N2.6g1`: Nektar++ v5.9.0 / commit `f729cda...`，
> body-fitted、body-fixed MovingReferenceFrame，
> `VelocityCorrectionScheme + Galerkin + IMEX2`，
> 在 closed NACA0015 上复现同一来源运动，并从同一场导出有序双侧
> pressure/full viscous traction。

角色仅为 source-gated observer。它不替换 V4.1，不追加总力，不观察目标，
也不因为“高阶”而自动晋升。

## 5. Go/no-go 方向

执行顺序固定为：

1. 版本、构建、官方 MRF regression、几何和运动学；
2. 单一空间/时间轴；
3. AeroForces 与独立边界 traction 的力矩账；
4. 同一 Schneiders CL/CD、峰值相位和来源 nominal 44/55 度涡拓扑；
5. 全部通过后才允许观察原冻结 U10 单条带。

任一 source gate 失败，只证伪并冻结 `N2.6g1` 实现命题；不得改成 v5.10、
overset、ALE、RANS、不同 TE 或目标拟合后继续同轮。
