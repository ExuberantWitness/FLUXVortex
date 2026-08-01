# N2.6g1 Nektar++ body-fixed MRF 来源门预登记 v2

日期：2026-07-30  
状态：`PREREGISTERED-v2 / AUDIT REQUIRED / SOURCE FLOW OFF / TARGET OFF`  
活动节点：`N2.6g1`  
被取代但保留的 v1：
`n26g1_nektar_mrf_source_prereg_20260730.md`,
SHA-256
`8cbfcc6e4d3cf3d070537d2a7f1ee407d8b417fe32067e4a0f8d869401cf6d33`  
v1 审计：`n26g1_nektar_mrf_prereg_audit_20260730.md`  
前置裁决：`research_n26g_body_fitted_mrf_decision_20260730.md`

## 0. 本次预登记回答的唯一问题

`N2.6f1` 已经证伪的是 moving PLIC/cut-cell 数值实现，不是全域二维黏性
Navier--Stokes observer 父命题。v2 只检验一个 successor：

> 固定版本 Nektar++ 的贴体、body-fixed MovingReferenceFrame 能否在
> Re=10000 rapid-pitch closed-NACA0015 来源域，同时给出数值独立的
> CL/CD、正确的来源响应，以及能保守传给非刚性结构自由度的材料面
> pressure/full-viscous traction？

这不是生产闭合，不改 V4.1，不调 LESP、\(A_0\)、\(f_2\) 或任何经验常数。
observer 输出不得进入 `ForceLedger`。只有 G0--G3 全过，另一个已经冻结的
目标域角色才可解锁。任何 target 输出在这之前均为非法证据。

## 1. 单一实现、身份和持久化边界

唯一实现为：

```text
Nektar++ version 5.9.0
official GitLab commit f729cda85b6a206e008fd705af8001cfe6e0d6fb
IncNavierStokesSolver
VelocityCorrectionScheme / Galerkin / IMEX2
body-fitted / body-fixed MovingReferenceFrame
absolute-velocity formulation
```

不并行试跑 v5.10、SU2、OpenFOAM、overset、ALE、RANS、LES、VPM、cut-cell
rescue 或另一个 TE。`Nektar++` 只是本轮唯一预登记 successor，不是学术上
唯一可行方法。

持久资产只能进入：

```text
platform/external/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb/
platform/data_external/n26g1_nektar_mrf/
platform/docs/diag/
```

禁止 `/tmp` 持久化，禁止覆盖任何 N2.6f1 或 V4.1 资产。

已经在解包前取得的官方 archive 身份为：

| field | value |
|---|---|
| URL | `https://gitlab.nektar.info/nektar/nektar/-/archive/f729cda85b6a206e008fd705af8001cfe6e0d6fb/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb.tar.gz` |
| bytes | `80377680` |
| SHA-256 | `59918ce766b89d544550b2c8fba438f865c411a14c132cf53f4cf358cca7553c` |
| response headers SHA-256 | `0e3aa1447dcfd6b329fc0943ca6469c153856296dd7fb8f3864e6c18d47f06d6` |
| ETag | `"ed5e9d219ddc28a11f02da3cb3673960"` |
| request ID | `01KYR3A913A5JF7TJGCQJRYBX6` |
| archive `VERSION` | `5.9.0` |
| receipt manifest SHA-256 at acquisition | `95e1ab50134b2ae1db7823dfd29fe31a31b58bfc0c51bd6e64674b9cbdda8fee` |

protected tag 只提供仓库身份，不冒充密码学签名。

## 2. G0a：封闭构建与官方组件身份

### 2.1 编译器和系统依赖

固定：

| executable/package | identity |
|---|---|
| `/usr/bin/gcc-11` | 11.4.0; SHA-256 `821af3c74506283c179ca413bb33e6b528805a4dd8a5c09df125e5ad560a9e89` |
| `/usr/bin/g++-11` | 11.4.0; SHA-256 `2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0` |
| `/usr/bin/cmake` | 3.22.1; SHA-256 `fd22547781b64bb2db04370970b93db1f3fada1e41e60873b015ee0747009fc0` |
| Boost dev | Ubuntu `1.74.0.3ubuntu7` |
| TinyXML dev | Ubuntu `2.6.2-6ubuntu0.22.04.1` |
| zlib dev | Ubuntu `1:1.2.11.dfsg-2ubuntu9.2` |
| OpenBLAS dev | Ubuntu `0.3.20+ds-1` |

正式构建保留官方 serial Scotch 能力。因为宿主当前没有 Scotch，执行者只能
从当前 Ubuntu Jammy apt 索引获取以下两个精确版本的 `.deb`，保存原始
`.deb`、URL/HTTP headers/SHA-256，并用 `dpkg-deb -x` 解到候选自己的
`dependency_sysroot/`，不得 `sudo apt install`：

```text
libscotch-6.1=6.1.3-1
libscotch-dev=6.1.3-1
```

`SCOTCH_INCLUDE_DIR`、`SCOTCH_LIBRARY`、`SCOTCHERR_LIBRARY` 必须显式指向
该只读 sysroot 的绝对路径。若精确版本不可得、ELF/headers 不完整或 CMake
未使用这些绝对路径，判 `G0a IMPLEMENTATION-NO-GO`，不得改成 Scotch OFF。

### 2.2 唯一 CMake 配置

构建目录是源码树外的 `build-release-serial`；源码树在解包前后分别生成
逐文件 size/SHA-256 manifest。配置和编译都运行在
`unshare --user --map-root-user --net` 无网络命名空间中。固定主要选项：

```text
CMAKE_BUILD_TYPE=Release
CMAKE_C_COMPILER=/usr/bin/gcc-11
CMAKE_CXX_COMPILER=/usr/bin/g++-11
CMAKE_EXPORT_COMPILE_COMMANDS=ON
NEKTAR_BUILD_LIBRARY=ON
NEKTAR_BUILD_DEMOS=OFF
NEKTAR_BUILD_SOLVERS=ON
NEKTAR_BUILD_SOLVER_LIBS=OFF
NEKTAR_SOLVER_INCNAVIERSTOKES=ON
all other NEKTAR_SOLVER_*=OFF
NEKTAR_BUILD_UTILITIES=ON
NEKTAR_UTILITY_FIELDCONVERT=ON
NEKTAR_UTILITY_NEKMESH=ON
NEKTAR_UTILITY_EXTRAS=OFF
NEKTAR_BUILD_TESTS=ON
NEKTAR_BUILD_UNIT_TESTS=OFF
NEKTAR_BUILD_PERFORMANCE_TESTS=OFF
NEKTAR_TEST_ALL=OFF
NEKTAR_BUILD_PYTHON=OFF
NEKTAR_USE_MPI=OFF
NEKTAR_USE_SCOTCH=ON
NEKTAR_USE_SYSTEM_BLAS_LAPACK=ON
NEKTAR_USE_HDF5=OFF
NEKTAR_USE_FFTW=OFF
NEKTAR_USE_ARPACK=OFF
NEKTAR_USE_PETSC=OFF
NEKTAR_USE_METIS=OFF
NEKTAR_USE_VTK=OFF
NEKTAR_USE_MESHGEN=OFF
NEKTAR_USE_CCM=OFF
NEKTAR_USE_CGNS=OFF
NEKTAR_USE_CWIPI=OFF
NEKTAR_USE_LST=OFF
NEKTAR_USE_LIKWID=OFF
THIRDPARTY_BUILD_BOOST=OFF
THIRDPARTY_BUILD_TINYXML=OFF
THIRDPARTY_BUILD_ZLIB=OFF
THIRDPARTY_BUILD_BLAS_LAPACK=OFF
THIRDPARTY_BUILD_SCOTCH=OFF
THIRDPARTY_USE_SSL=ON
```

只构建 `IncNavierStokesSolver FieldConvert NekMesh Tester`。配置后必须满足：

- `CMakeCache.txt` 中没有任何 `THIRDPARTY_BUILD_*=ON`；
- 没有 `NOTFOUND`、关键 option 未消费或源码树 `ThirdParty/` 下载物；
- source tree post-build manifest 与 post-extract manifest 完全相同；
- 保存 configure/build stdout/stderr、cache、compile commands、动态链接库
  解析、四个 binary SHA-256 和峰值 RAM/disk。

源码 patch 数必须为零。任何静默下载、源码变化、未知依赖或磁盘可用量低于
`15 GiB` 都立即 fail closed。

### 2.3 官方 MRF regression

原样运行 commit 内：

```text
solvers/IncNavierStokesSolver/Tests/
  MovingRefFrame_Rot_naca0012.{xml,tst,rst}
```

CTest 名称固定为
`IncNavierStokesSolver_MovingRefFrame_Rot_naca0012`。六个官方指标必须逐项
进入其原 tolerance：

| metric | u | v | p |
|---|---:|---:|---:|
| L2 reference | 29.9834 | 0.548735 | 0.857798 |
| L2 tolerance | 5e-4 | 5e-5 | 5e-3 |
| Linf reference | 1.33746 | 0.947604 | 1.98367 |
| Linf tolerance | 5e-4 | 5e-4 | 5e-3 |

这是固定组件 identity regression，不称作 Nektar 完整回归。失败即
`IMPLEMENTATION-NO-GO / FROZEN`。

## 3. G0b：唯一 source 几何、网格和资源处方

### 3.1 source 几何

\(c=1\)，pivot `(0.25,0)`，closed NACA0015：

\[
y_t(x)=5(0.15)\left(0.2969\sqrt{x}-0.1260x-0.3516x^2
+0.2843x^3-0.1036x^4\right),\quad 0\le x\le1 .
\]

`-0.1036` 强制零尾缘；source 内禁止换成 open TE、圆钝 TE 或截断 TE。
上表面材料坐标 \(\xi\in[0,1]\) 从 LE 到 TE，下表面也从 LE 到 TE，二者
分别保留 ID；积分边界方向另由 fluid-domain orientation 决定。LE、TE 是
共享端点，不复制为有限 gap。外边界固定
`[-32,32] x [-32,32]`。

Gmsh 可执行文件固定 `/usr/bin/gmsh` 4.8.4，SHA-256
`21ee9676ee08261883d578a7c50707bf1bc9b51bbd6adbfab8a7d30e5d4d31`。
固定：

```text
Mesh.MshFileVersion = 4.1
Mesh.Binary = 0
Mesh.Algorithm = 6
Mesh.RecombineAll = 0
Mesh.ElementOrder = 3
Mesh.HighOrderOptimize = 2
Mesh.RandomFactor = 1e-9
General.NumThreads = 1
```

NACA 每侧使用 cosine material nodes
\(x_i=(1-\cos(i\pi/N))/2\)，每个相邻节点间是一条三次
OpenCASCADE BSpline segment；相同的 analytic formula 产生其 Gmsh
控制点和独立 geometry oracle。所有体单元为三角形，近壁 BoundaryLayer
field 允许产生四边形层；物面、outer-left/right/top/bottom 都是显式
Physical Curve，流体是单一 Physical Surface。

### 3.2 唯一三层网格

网格尺寸 field 用 `min(BoundaryLayer, inner Box, wake Box, transition Box,
far)`，相同区域重叠时只取最小值。除表中三层外不得新增层：

| item | H0 | H1 | H2 |
|---|---:|---:|---:|
| intervals per surface side | 64 | 128 | 256 |
| first wall-normal size | `4e-4` | `2e-4` | `1e-4` |
| BL growth ratio | 1.15 | 1.15 | 1.15 |
| BL total thickness | 0.08 | 0.08 | 0.08 |
| inner box `[-0.5,2] x [-1.25,1.25]` | 0.12 | 0.08 | 0.05 |
| wake box `[1.5,8] x [-1.25,1.25]` | 0.25 | 0.16 | 0.10 |
| transition box `[-1,10] x [-4,4]` | 0.75 | 0.50 | 0.25 |
| far maximum | 4.0 | 4.0 | 4.0 |
| velocity NUMMODES | 4 | 5 | 6 |
| pressure NUMMODES | 3 | 4 | 5 |

Nektar expansion 对全部体 composite 固定为：

```text
u,v: TYPE=MODIFIED, NUMMODES=P
p:   TYPE=MODIFIEDQUADPLUS1, NUMMODES=P-1
```

`.geo`、`.msh`、NekMesh 转换命令、最终 `.xml`、session overlay 及其 SHA
必须在第一次 solver output 前冻结。实现者只能把上述处方机械编码，不能
看到流场后改 field、曲线或质量优化。Gmsh 和 NekMesh stdout/stderr 均保存。

### 3.3 几何、质量和资源硬门

每层在求解前必须全部满足：

- analytic control-node residual `<=5e-13 c`；
- TE gap 和有向相邻端点 gap `<=1e-12 c`；
- `||sum(n ds)||/sum(ds) <=1e-12`；
- upper/lower 各是一条从 LE 到 TE 的连续材料链，无重复或缺失边；
- 所有线性及高阶 Jacobian 严格为正，尺度化最小 Jacobian `>=1e-3`；
- perimeter 对独立 16384-panel analytic quadrature 的相对误差逐层下降，
  H2 `<=1e-5`；
- H2 体单元总数 `<=60000`；
- H2 `u+v+p` expansion coefficients 总数 `<=3,000,000`；
- 静态预估峰值 RAM `<=20 GiB`，build+mesh+所有计划输出的磁盘预算
  `<=12 GiB`，且根分区保留 `>=15 GiB`。

先允许一个 H0、`t<=0.05`、不评分且不保存物理快照的 performance pilot。
按实际 walltime/RAM 线性外推五个正式 source runs；若总 walltime 预测
超过 `96 h`、任一 H2 run 超过 `36 h` 或 RAM 超门，结果是
`RESOURCE-NO-GO`，不得缩小域、降 mesh、改 dt 或换 solver 后续跑。

## 4. G0c：运动、参考系和无零分母规范

### 4.1 完整分段运动

正日志攻角是物理 nose-up 的顺时针 body rotation。定义
\(\tau=t-0.2\)：

\[
\alpha(t)=
\begin{cases}
0,&t<0.2,\\
0.6[\tau+(e^{-4.6\tau}-1)/4.6],&t\ge0.2,
\end{cases}
\]

\[
\dot\alpha(t)=
\begin{cases}
0,&t<0.2,\\
0.6[1-e^{-4.6\tau}],&t\ge0.2,
\end{cases}
\quad
\ddot\alpha(t)=
\begin{cases}
0,&t<0.2,\\
2.76e^{-4.6\tau},&t\ge0.2.
\end{cases}
\]

在切换点 \(t=0.2\) 采用右支，因此
\(\alpha=0,\dot\alpha=0,\ddot\alpha=2.76\)。MRF 表达式为：

\[
\Theta_z=-\alpha,\quad\Omega_z=-\dot\alpha,\quad
\mathrm{DOmega}_z=-\ddot\alpha .
\]

XML 只用 v5.9 expression evaluator，不 patch solver。所有计划 time 及
`{0,0.2,1.6971,2.0172,2.05}` 对独立 Python oracle 校验。微分恒等式只在
两个开区间检查，明确排除切换点的 \(\ddot\alpha\) 跳跃。

### 4.2 固定的尺度化误差

禁止裸 relative error。标量统一为

\[
E(a,b;s)=|a-b|/\max(s,|b|).
\]

固定尺度：角 `1 rad`、角速度 `1 s^-1`、角加速度 `1 s^-2`、速度 `U`、
压力 `rho U^2`、长度 `c`、力 `q_inf c`、力矩 `q_inf c^2`，其中
\(q_\infty=\rho U^2/2=0.5\)。场 L2 用对应尺度乘
\(\sqrt{A}\)，分母是该固定尺度与 reference L2 的较大者。系数差直接取
绝对差。所有 norm 使用完整相同 support 和固定 quadrature。

### 4.3 三个 reference-frame manufactured gates

1. **K0 static identity**：同一 H0 方形无翼网格、`u=1,v=p=0`，
   zero-motion MRF 与 inertial solver 运行 10 个 `dt=1e-3` 步；每步
   `u/v/p` 场尺度化 L2、边界通量和力的 max error `<=1e-10`。
2. **K1 moving-frame uniform field**：同一无翼网格，使用正式
   `Theta/Omega/DOmega` 的解析惯性均匀流坐标变换，outer 全部
   `MovingFrameFar`；离散 divergence 与完整 MRF momentum residual 的
   尺度化 L2/Linf 各 `<=1e-10`。输入、解析场和 residual kernel 在运行
   前冻结 SHA。
3. **K2 rotation/force transform**：对
   `theta={-0.6,-0.3,0,0.3,0.6}` rad 的五个非零解析 traction 向量及
   quarter-chord moments，Nektar native body/inertial transform 对解析
   rotation 的尺度化误差 `<=1e-12`。

K0--K2 任何失败，禁止观察 source CL/CD。

## 5. 唯一正式 source session

### 5.1 方程、离散和初边值条件

所有 source runs 固定：

```text
rho=1, U=1, c=1, Re=10000, Kinvis=1e-4, mu=1e-4, p_inf=0
EQTYPE=UnsteadyNavierStokes
SolverType=VelocityCorrectionScheme
EvolutionOperator=Nonlinear
AdvectionForm=Convective
Projection=Galerkin
SpectralVanishingViscosity=DGKernel
SVVDiffCoeff=1.0
SpectralHPDealiasing=True
TIMEINTEGRATIONSCHEME=IMEX, ORDER=2
```

不设置可变 `SVVCutoffRatio`；DGKernel 使用 v5.9 该 commit 的实现默认，
其源码 SHA 与 build manifest 一起冻结。初值在全域固定
`u=1,v=0,p=0`，不使用 restart。

body：

```text
u=v=0, USERDEFINEDTYPE=MovingFrameWall
p Neumann, USERDEFINEDTYPE=MovingFrameWall
```

left/right/top/bottom 全部采用官方 v5.9 MRF regression 同类条件：

```text
u=1, v=0, USERDEFINEDTYPE=MovingFrameFar
p=0 Dirichlet
```

这与 Basilisk source 的右侧 pressure-outlet 不同，必须在结果中报告为
solver-template implementation difference；不得在看到响应后切换 BC。
`p=0` 同时固定 pressure gauge。

`GLOBALSYSSOLNINFO` 对 `u,v` 固定
`IterativeStaticCond / LowEnergyBlock / IterativeSolverTolerance=1e-10 /
AbsoluteTolerance=False`；对 `p` 固定
`IterativeStaticCond / Diagonal / IterativeSolverTolerance=1e-10 /
AbsoluteTolerance=False`。任何未收敛线性步、NaN、CFL/solver fatal 都是
该层失败，不能重跑为另一 preconditioner。

`AeroForces` 每步输出；checkpoint 每 `0.01 c/U` 及 nominal 44/55 deg
最近步输出。禁止从 checkpoint restart 拼接正式曲线，除非只是完全相同
session 的机器中断恢复，且 receipt 记录中断前后 SHA 与连续 time index。

### 5.2 独立空间和时间轴

全部从 \(t=0\) 运行至 \(t=2.05\)，不得按响应提前停止。

空间轴：

```text
H0/H1/H2, dt=2.5e-4
```

时间轴：

```text
H2, dt={1.0e-3, 5.0e-4, 2.5e-4}
```

H2/`2.5e-4` 是两轴共享的一次运行，故总共五个不同正式 flow runs，不是
六个；资源外推以五个计。正式 source 输出固定取 H2/`2.5e-4`，不按与
reference 的接近程度选层。

所有曲线先按解析 \(\alpha(t)\) 转为单调角度坐标。共同 support 用相邻两层
原始 knot 的 union，只有线性插值，不平滑。对 \(g=CL,CD\)：

\[
D_{01}=\|g_0-g_1\|_{L2(I)},\quad
D_{12}=\|g_1-g_2\|_{L2(I)},\quad
p_{\rm obs}=\log_2(D_{01}/D_{12}),
\]

其中 \(I=[5,20]\) deg，L2 是 union-knot 上分段线性差的精确平方积分。
若 `D01` 或 `D12 <= 1e-12` coefficient-deg\(^{1/2}\)，标记
`AXIS-DEGENERATE`，不把无穷阶当 pass。时间轴 CL/CD 各要求
`p_obs>=1.8`。

空间轴和时间轴的最后两级还分别要求：

- CL/CD 在各自冻结 full reference support 的 range-normalized L2
  change `<=3%`；
- CL/CD first-argmax peak coefficient change / reference range `<=3%`；
- 峰值角漂移只报告，不用于选择层。

任一正式日志非有限、不完整、time/angle 非单调、最后两级越门或 Richardson
失败：`SOURCE-NUMERICS NO-GO / FROZEN`。

## 6. G2：统一材料 traction、分布收敛和结构虚功

### 6.1 符号与原生量映射

惯性轴 \(+x\) 是来流下游、\(+y\) 向上；正日志攻角是 nose-up 顺时针。
\(\mathbf n_{sf}\) 从 solid 指向 fluid。fluid-on-body traction 定义为：

\[
\mathbf t_{fb}=-(p-p_\infty)\mathbf n_{sf}
+\mu(\nabla\mathbf u+\nabla\mathbf u^T)\mathbf n_{sf}.
\]

Nektar trace normal \(\mathbf n_f=-\mathbf n_{sf}\) 指向 fluid domain 外部，
所以其 native pressure `rho*p*n_f` 和 viscous
`-mu*(grad(u)+grad(u)^T)*n_f` 必须与上式相同。`AeroForces` 在 MRF 下输出
旋回初始惯性方向的原始 force；不把原始数当系数：

\[
C_D=F_x/(q_\infty c),\quad C_L=F_y/(q_\infty c),\quad
C_M=M_{c/4}/(q_\infty c^2).
\]

adapter 源码和 SHA 在第一个 solver output 前冻结。

### 6.2 保存时刻和共同材料网格

在 nominal `44 deg`、`55 deg` 和正式细层 CL/CD extrema 保存完整 primitive：
ordered material \(\xi\)、surface ID、body/inertial coordinates、
\(\mathbf n_{sf}\)、`p`、全部 `grad(u)`、WSS、native quadrature 和
pressure/viscous/total traction。44/55 的样本是最近 solver time，时间相同
时取较早者；必须报告 actual time/angle。旧参考图片文件名中的 `54` 只是
历史资产名，不能改写 nominal 55 deg。

共同材料网格固定为每侧 512 个 cosine intervals，upper/lower 分开，不跨
LE/TE 平滑。每个 source element 的压力和 full viscous traction 通过其
原生高阶边界 expansion 在共同点求值；合力/合矩用每个原生 element 的
Gauss--Lobatto quadrature，不做 residual redistribution 或 TE 补力。

### 6.3 必须同时通过的账

1. primitive 重积分与 `AeroForces` 的 pressure、viscous、total
   \(F_x,F_y,M_{c/4}\)，在 body/inertial 两系的尺度化 residual 均
   `<=1e-8`；
2. boundary closure `||sum(n ds)||/sum(ds)<=1e-12`；
3. H1/H2 以及 H2 中/细时间层，在完整 upper+lower support 上分别比较
   pressure traction 和 full viscous traction：
   \[
   \|\Delta\mathbf t\|_{L2(ds)}/
   \max(q_\infty\sqrt{2s_{\rm side}},\|\mathbf t_{\rm fine}\|_{L2(ds)})
   \le3\%.
   \]
   同时对 \(d\mathbf t/d\xi\) 的弱形式投影比较 `<=5%`，不以点值噪声
   冒充物理失败；
4. 保守投影到冻结结构测试基
   \[
   \phi_k(\xi)\in\{1,\xi,\xi^2,\xi^3,\xi^4\}
   \]
   的 \(x/y\) 两方向虚位移。材料面原生 traction 虚功与共同网格投影后的
   虚功，对每个非刚体模式 `k=2,3,4` 的固定力尺度 residual `<=1e-8`；
   `k=0,1` 同时恢复合力和一阶矩账。

G2 只证明数值一致、守恒且可用于 co-design 的材料载荷传递，不声称逐点
traction 已由实验验证。禁止先积分总力再按面积重分配。

## 7. G3：冻结来源响应与视觉门

reference/scorer 身份：

| asset | SHA-256 |
|---|---|
| CD CSV | `bb068e3880b2f142527739cd8a425df84dbabeb7f33f789b7217eeda1ff0b022` |
| CL CSV | `77da2650d58ded6f049b0cc879e2d2e8069ab3029698f8f5a8759e5564c1ff6a` |
| nominal-44 image (`fig18.png`) | `319d83ae36b27439e98fae69dbb33e67f88eaa4dffb2ae0d2b3e3db969541888` |
| nominal-55 image (`fig19.png`, legacy name) | `1b2acafd06fa35f00f65673d3d1ada9fcdbd66124a788a718de463a537c01363` |
| asset manifest | `3ed734c76c37d1873ac8584c9fc27add69cc1214e3f2c176883144b416a74546` |
| inherited scorer | `platform/n26f_source_gate.py`, `35a9292ee6eeb4a4098f2f167f8d2c0007de021a58df469a421592ac56c491d9` |
| scoring amendment | `b119f8e230c1dc2a367bc3da6825b3d06a09e10f3a50683d6d55e04346bc8138` |

冻结 support：

```text
CD [0.4730174926739976, 54.98539927034809] deg
CL [0.020766621629590087, 54.500844765657654] deg
```

沿用 SHA-frozen scorer 的 unsmoothed piecewise-linear union-knot exact
integral、reference-range normalization 和 first argmax：

- CD/CL range-normalized RMSE 各 `<=10%`；
- CD/CL peak-angle error 各 `<=3 deg`；
- G0、G1、G2 必须同时通过。

视觉渲染固定 body frame、`x/c=[-1,3]`、`y/c=[-2,2]`、等长坐标；vorticity
无插值平滑，色标固定对称 `[-20,20] U/c`，离散 levels
`{-20,-10,-5,-2,2,5,10,20}`，翼面黑色、正涡红、负涡蓝。每个 nominal
时刻同时保存无色标原场和带色标 PNG，SHA 入 manifest。

主线程在不知道数值 RMSE 结论的单独表中逐项核对：

1. 主 LEV 旋向；
2. 主 LEV/TEV 数目；
3. 主涡核相对 LE/TE、上/下表面及弦长的位置象限；
4. 是否出现参考图不存在的跨翼反相；
5. actual angle 与 nominal 差值。

任何旋向反相、主涡数不一致或主核落到错误表面/弦向象限为视觉 no-go。
大小与精确像素不作拟合目标。人审表、两图和实际取样角必须持久化。

G3 通过只晋升：

> 在该 Re=10000 closed-NACA0015 rapid-pitch 来源域，固定版本的贴体
> body-fixed MRF observer 具有预登记范围内的数值独立性、来源响应、
> 材料面 traction 一致性和非刚体虚功守恒。

它不证明 RoboEagle、三维横流、转捩或 Fig17/18/19 精度提升。

## 8. source/target 角色转换与 G4 锁

source 和 target 是预先声明的两个不同验证角色：

- source：本文件唯一 closed-NACA0015，禁止在 G0--G3 内改变 TE；
- target：继承的固定 open-TE NACA2406-like 数值代理，仅在 source 全过后
  新建独立 mesh/session/manifest，不能复用 source mesh 或数值结果。

继承 target prereg：
`n26f_full_domain_ns_observer_prereg_20260730.md`,
SHA-256
`9c24e072ca3023d5b4aa33ec5259c0e1f1c4a92c1a2629a8675ff5b9a31fcc24`。
只有 G0--G3 全 PASS，才允许 target adapter/geometry 在任何 target output
前冻结 SHA，并观察：

```text
U=10 m/s, f=1.4 Hz, AoA=5 deg, nominal twist=22.5 deg
y=0.55 m, band=[0.5,0.6] m
```

G4 仍是 shadow observer ledger，不向 V4.1 注力。单条带 go/no-go、八条带、
代表工况、完整 184 的顺序沿用继承 prereg，但不能复用 N2.6f1 Basilisk
数值或放宽门。

## 9. 状态机与不可后验修改项

```text
v2 未获 PASS TO EXECUTE:
    SOURCE FLOW OFF; TARGET OFF

build/regression/geometry/kinematics/resource gate 失败:
    N2.6g1 IMPLEMENTATION-NO-GO / FROZEN

数值独立、traction、虚功或来源响应失败:
    N2.6g1 SOURCE-NUMERICS/PHYSICS NO-GO / FALSIFIED+FROZEN

G0--G3 全通过:
    N2.6g1 SOURCE-GO; 只授权 N2.6g2 G4

G4 未执行:
    Fig17/18/19 improvement = NOT TESTED
```

看到任何 source CL/CD 或 target 输出后，不得改变版本、依赖、compiler、
solver、preconditioner、BC/IC、mesh family、dt、domain、Re、运动律、TE、
SVV、dealiasing、阈值、reference normalization、traction map、共同材料
网格、虚功基或视觉 rubric。失败只回写 claim；禁止同轮换一个补丁继续。
