# N2.6g1 Nektar++ body-fixed MRF 来源门预登记

日期：2026-07-30  
状态：`PREREGISTERED / SOURCE-GATED IMPLEMENTATION AUTHORIZED / TARGET OFF`  
活动节点：`N2.6g1`  
前置裁决：`research_n26g_body_fitted_mrf_decision_20260730.md`

## 1. 单一候选与禁止项

只实施：

```text
Nektar++ v5.9.0
commit f729cda85b6a206e008fd705af8001cfe6e0d6fb
IncNavierStokesSolver
VelocityCorrectionScheme / Galerkin / IMEX order 2
body-fitted, body-fixed MovingReferenceFrame
```

不并行实施或试跑 v5.10、SU2、OpenFOAM、overset、ALE、RANS、LES、
cut-cell rescue 或 VPM。V4.1 保持冻结；任何 observer 输出不得进入
ForceLedger。持久资产只写仓库的 `platform/external/` 和
`platform/data_external/`，禁止 `/tmp`。

## 2. 不可变源码与构建门 G0a

只从官方 GitLab 的解引用 commit 获取 archive：

`https://gitlab.nektar.info/nektar/nektar/-/archive/f729cda85b6a206e008fd705af8001cfe6e0d6fb/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb.tar.gz`

下载后、解压前记录 URL、HTTP identity、bytes 和 SHA-256。解压树必须由
archive 唯一生成；不得后拉 master。冻结：

- source archive 和 commit；
- CMake cache、编译器/链接器、系统依赖版本；
- configure/build 命令与 stdout/stderr；
- `IncNavierStokesSolver`、`NekMesh`、`FieldConvert`、`Tester` SHA；
- official regression XML/TST/RST SHA。

构建只使用本机已存在依赖；若 CMake 必须下载第三方，则每个 URL/SHA 先
进入 build manifest，不能静默抓 latest。源码 patch 数必须为零。

原样运行官方
`MovingRefFrame_Rot_naca0012.tst`。六个冻结指标必须各自在官方 tolerance：

| metric | u | v | p |
|---|---:|---:|---:|
| L2 reference | 29.9834 | 0.548735 | 0.857798 |
| L2 tolerance | 5e-4 | 5e-5 | 5e-3 |
| Linf reference | 1.33746 | 0.947604 | 1.98367 |
| Linf tolerance | 5e-4 | 5e-4 | 5e-3 |

该门只验证版本实现身份。构建、SHA、原回归任一失败：
`N2.6g1 IMPLEMENTATION-NO-GO / FROZEN`，禁止进入自建 NACA0015。

## 3. 冻结 NACA0015 几何与网格族 G0b

### 3.1 解析几何

弦长 \(c=1\)，quarter-chord pivot `(0.25,0)`，闭合 NACA0015：

\[
y_t=5(0.15)\left(
0.2969\sqrt{x}-0.1260x-0.3516x^2
+0.2843x^3-0.1036x^4\right).
\]

禁止换成开尾缘 `-0.1015`、圆钝/截断 TE 或改变厚度。上表面从 TE 到 LE，
下表面从 LE 回 TE，保留 `upper/lower/LE/TE` material IDs。

外域固定为 `[-32,32]c x [-32,32]c`。Gmsh 固定本机
`4.8.4`，2-D algorithm 6；解析曲线采用两侧 cosine-spaced material
nodes。唯一组合空间梯级为：

| level | intervals/side | first wall-normal size | near/wake bulk size | velocity modes | pressure modes |
|---|---:|---:|---:|---:|---:|
| H0 | 64 | `4e-4 c` | `8e-2 c` | 4 | 3 |
| H1 | 128 | `2e-4 c` | `4e-2 c` | 5 | 4 |
| H2 | 256 | `1e-4 c` | `2e-2 c` | 6 | 5 |

boundary-layer thickness 固定 `0.08c`、growth ratio `1.15`；wake refinement
box 固定 `x/c=[-0.5,8]`, `|y|/c<=4`；远场最大 size `4c`。所有值在任何
流场计算前写入 geometry manifest。若该唯一网格处方不能生成正 Jacobian
网格，则是 `IMPLEMENTATION-NO-GO`，禁止在看到流场后改 mesh 参数。

每层运行前必须满足：

- analytic-node residual `<=5e-13 c`；
- TE gap、相邻有向端点 gap `<=1e-12 c`；
- `||sum(n ds)||/sum(ds) <=1e-12`；
- 无重复/缺失 boundary edge，upper/lower 各一条连续 material chain；
- 全部高阶几何 Jacobian `>0`；
- source boundary perimeter 对 analytic quadrature 的相对误差逐层下降，
  H2 `<=1e-5`。

不满足时不得通过人工补边、traction 归一化或载荷重分配。

## 4. 运动学和参考系门 G0c

为与现有 source 坐标保持一致，正的日志攻角对应物理 nose-up 的顺时针
body rotation，故 Nektar MRF 使用：

\[
\Theta_z(t)=-\alpha(t),\quad
\Omega_z(t)=-\dot\alpha(t),\quad
\dot\Omega_z(t)=-\ddot\alpha(t),
\]

其中

\[
\alpha(t)=
\begin{cases}
0,&t<0.2,\\
0.6[(t-0.2)+(e^{-4.6(t-0.2)}-1)/4.6],&t\ge0.2,
\end{cases}
\]

\[
\dot\alpha=0.6[1-e^{-4.6(t-0.2)}],\qquad
\ddot\alpha=2.76e^{-4.6(t-0.2)} .
\]

XML 只使用 v5.9 expression evaluator，不 patch solver。对所有计划 solver
times 和 `{0,0.2,1.6971,2.0172,2.05}`，解析式与独立 Python oracle 的
normalized max error 必须 `<=1e-12`；`Theta/Omega/DOmega` 微分恒等式
同门。

另做：

1. zero-motion MRF 与同网格 inertial solver 的 field/force relative L2
   `<=1e-10`；
2. uniform free-stream 在任意冻结 frame motion 下的 divergence 和
   momentum residual `<=1e-10`；
3. AeroForces 的 inertial/body transform 与解析刚体旋转
   relative error `<=1e-12`。

任一反号或 frame-invariance 失败禁止看 source CL/CD。

## 5. 独立数值轴 G1

来源参数完全沿用现有 S1：

- `Re=10000`、`U=c=rho=1`；
- stationary 至 `t=0.2`，随后上述 rapid pitch；
- square domain `64c`；
- 无 turbulence/transition/SVV 调参；
- pressure/velocity solver tolerance 固定在运行 manifest；
- 所有层运行至 `t=2.05`，不得按响应提前停止。

### 5.1 空间轴

固定 `dt=2.5e-4 c/U`，只运行 `H0/H1/H2`。正式来源层始终是 H2，不按
reference closeness 选层。

### 5.2 时间轴

固定 H2，只运行：

```text
dt = {1.0e-3, 5.0e-4, 2.5e-4} c/U
```

IMEX2 在冻结 smooth pre-separation window
`theta in [5,20] deg` 的 CL/CD 三层 Richardson observed order 必须
`>=1.8`。不得换窗口。

空间和时间最后两级都沿用 N2.6f1 的同一 Cauchy 定义和阈值：

- CL/CD full-support relative L2 `<=3%`；
- CL/CD peak relative change `<=3%`；
- 峰值角漂移只报告，不选择层级。

所有日志必须有限、完整、时间单调。任一末级越门：
`N2.6g1 SOURCE-NUMERICS NO-GO / FROZEN`。

## 6. 统一牵引账 G2

在 44 度、54 度和正式 H2/dt 最细层的 CL/CD extrema 保存：

- ordered material arclength、`upper/lower/LE/TE` ID；
- body/inertial coordinates 和 unit normal；
- pressure；
- full viscous traction vector与 WSS；
- native boundary quadrature weights；
- pressure/viscous/total force 和关于 quarter chord 的 moment。

独立 postprocessor 必须从 primitive `p, grad(u), n, ds` 重算：

\[
\mathbf t=-(p-p_\infty)\mathbf n+
\mu(\nabla\mathbf u+\nabla\mathbf u^T)\mathbf n .
\]

与 `AeroForces` 分别比较 pressure、viscous、total force 和 moment；
body/inertial 两系 relative residual 均 `<=1e-8`。有向 boundary closure
仍须 `<=1e-12`。禁止先积分总力再按面积分配，禁止残差归一化和 TE 补力。

## 7. 来源响应门 G3

使用与 N2.6f1 完全相同、SHA 已冻结的 Schneiders CL/CD digitization、
支持域、分段线性 union-knot 积分和峰值定义：

- CD/CL range-normalized RMSE 各 `<=10%`；
- CD/CL peak-angle error 各 `<=3 deg`；
- 44/54 度 vorticity 图由主线程视觉核对：旋向、主 LEV/TEV 数目和相对
  位置不得反相；
- G0、G1、G2 必须同时通过。

视觉证据是明确的人审门，不伪装成机器分类器。

G3 通过只晋升：

> 在该 Re=10000 closed-NACA0015 rapid-pitch 来源域，固定版本的贴体
> body-fixed MRF observer 具有预登记范围内的数值独立性、来源响应和
> 材料边界 traction 一致性。

它不证明 RoboEagle、三维横流、转捩或生产精度。

## 8. 目标授权 G4

只有 G0--G3 全部通过，才允许另行生成目标 run manifest，并观察既有冻结点：

```text
U=10 m/s, f=1.4 Hz, AoA=5 deg, nominal twist=22.5 deg
y=0.55 m, band=[0.5,0.6] m
```

目标阶段仍是 observer ledger，不注入 V4.1。单条带 go/no-go、八条带扩展、
Fig17/18/19 代表点和完整 184 的顺序沿用
`n26f_full_domain_ns_observer_prereg_20260730.md §§5--9`，但不得复用
Basilisk 数值结果或放宽门。

## 9. 状态机与不可后验修改项

- source/build/geometry/kinematics失败：`IMPLEMENTATION-NO-GO`；
- 数值、traction或来源响应失败：`N2.6g1 falsified/frozen`；
- G0--G3 全过：`SOURCE-GO`，只授权 G4；
- G4 未执行前：`Fig17/18/19 improvement = NOT TESTED`。

看到任何 CL/CD 或目标后均不得改变版本、solver、mesh family、dt、窗口、
Re、运动律、TE、SVV、阈值、reference normalization 或 traction map。

