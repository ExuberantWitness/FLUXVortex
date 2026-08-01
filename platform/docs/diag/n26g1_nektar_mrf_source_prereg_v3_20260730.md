# N2.6g1 Nektar++ body-fixed MRF 来源门预登记 v3

日期：2026-07-30  
状态：`PREREGISTERED-v3 / INDEPENDENT RE-AUDIT REQUIRED / SOURCE FLOW OFF / TARGET OFF`  
活动节点：`N2.6g1`

## 0. 复合规范和优先级

本文件不是另一个候选，而是 v2 在任何 `.msh`、build 或 flow output 之前的
规范修订。完整预登记对象严格定义为：

1. base：
   `n26g1_nektar_mrf_source_prereg_v2_20260730.md`,
   SHA-256
   `14d29b5b6523ed523efae255b345edab61a1a5b2f31e54d265f986963d3c9b0b`；
2. v2 audit：
   `n26g1_nektar_mrf_prereg_v2_audit_20260730.md`,
   SHA-256
   `95c6f42a75c73fefb5e6e04eccff15f85690b4021deef08baf1e0e89e6f73527`；
3. 本文件的规范覆盖。

本文件明确替换的文字以本文件为准；没有替换的 v2 条款继续有效。禁止把
二者任选其一解释。v1、v1 audit、v2 和 v2 audit 均保留，不覆盖历史。

候选、物理问题、Nektar commit、source 数据、阈值、V4.1 冻结和
`TARGET OFF` 均不改变。本次只消除构建、mesh、manufactured gate、输出留存、
Cauchy 和结构传力的实现自由度。

## 1. 替换 v2 §2.1--2.2：可执行的封闭构建

### 1.1 固定路径与依赖

以下绝对路径是预登记身份的一部分：

```text
ROOT=/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV
NEK_ROOT=$ROOT/platform/external/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb
NEK_SRC=$NEK_ROOT
NEK_BUILD=$NEK_ROOT/build-release-serial
SCOTCH_SYSROOT=$NEK_ROOT/dependency_sysroot
SCOTCH_INC=$SCOTCH_SYSROOT/usr/include/scotch
SCOTCH_LIBDIR=$SCOTCH_SYSROOT/usr/lib/x86_64-linux-gnu
SCOTCH_LIB=$SCOTCH_LIBDIR/libscotch.so
SCOTCHERR_LIB=$SCOTCH_LIBDIR/libscotcherr.so
PY=/home/exuber/anaconda3/envs/fluxvortex/bin/python
RUNROOT=$ROOT/platform/data_external/n26g1_nektar_mrf/source_gate_v3
NEK_WRAPPER=$NEK_ROOT/run_nektar_serial.sh
```

仍只取得并保存 Ubuntu Jammy 精确包
`libscotch-6.1=6.1.3-1`、`libscotch-dev=6.1.3-1`，原 `.deb`、headers、
URL、bytes、SHA-256 全部进入 receipt；只允许 `dpkg-deb -x`，不安装系统。
解包后，所有上述路径先做 `realpath`，必须仍在 `SCOTCH_SYSROOT` 内。

OpenBLAS 唯一绑定为：

```text
package libopenblas0-pthread/libopenblas-pthread-dev = 0.3.20+ds-1
realpath /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.20.so
SHA-256 1dedc9fee9ca46eb73e1abc9d989093acbb5bf1bb474fe673af7f0591fa4b2d9
OPENBLAS_HOME=/usr
```

不得回退到 generic system BLAS。

### 1.2 完整 configure argv

`NEK_SRC` 必须来自 v2 固定 archive 的一次解包；在解包树生成 file
size/SHA manifest 后，完整 configure argv 固定如下：

```bash
env -i \
  PATH=/usr/bin:/bin \
  LC_ALL=C \
  SCOTCH_DIR="$SCOTCH_SYSROOT/usr" \
  SCOTCH_INCDIR="$SCOTCH_INC" \
  OPENBLAS_HOME=/usr \
  unshare --user --map-root-user --net \
  /usr/bin/cmake -S "$NEK_SRC" -B "$NEK_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSCOTCH_LIBRARY:FILEPATH="$SCOTCH_LIB" \
  -DSCOTCHERR_LIBRARY:FILEPATH="$SCOTCHERR_LIB" \
  -DNEKTAR_BUILD_LIBRARY=ON \
  -DNEKTAR_BUILD_DEMOS=OFF \
  -DNEKTAR_BUILD_SOLVERS=ON \
  -DNEKTAR_BUILD_SOLVER_LIBS=OFF \
  -DNEKTAR_SOLVER_ADR=OFF \
  -DNEKTAR_SOLVER_ACOUSTIC=OFF \
  -DNEKTAR_SOLVER_CARDIAC_EP=OFF \
  -DNEKTAR_SOLVER_COMPRESSIBLE_FLOW=OFF \
  -DNEKTAR_SOLVER_DIFFUSION=OFF \
  -DNEKTAR_SOLVER_DUMMY=OFF \
  -DNEKTAR_SOLVER_IMAGE_WARPING=OFF \
  -DNEKTAR_SOLVER_ELASTICITY=OFF \
  -DNEKTAR_SOLVER_INCNAVIERSTOKES=ON \
  -DNEKTAR_SOLVER_MMF=OFF \
  -DNEKTAR_SOLVER_PULSEWAVE=OFF \
  -DNEKTAR_SOLVER_REVIEWSOLUTION=OFF \
  -DNEKTAR_SOLVER_SHALLOW_WATER=OFF \
  -DNEKTAR_SOLVER_VORTEXWAVE=OFF \
  -DNEKTAR_BUILD_UTILITIES=ON \
  -DNEKTAR_UTILITY_FIELDCONVERT=ON \
  -DNEKTAR_UTILITY_NEKMESH=ON \
  -DNEKTAR_UTILITY_EXTRAS=OFF \
  -DNEKTAR_BUILD_TESTS=ON \
  -DNEKTAR_BUILD_UNIT_TESTS=OFF \
  -DNEKTAR_BUILD_PERFORMANCE_TESTS=OFF \
  -DNEKTAR_TEST_ALL=OFF \
  -DNEKTAR_BUILD_PYTHON=OFF \
  -DNEKTAR_USE_MPI=OFF \
  -DNEKTAR_USE_SCOTCH=ON \
  -DNEKTAR_USE_SYSTEM_BLAS_LAPACK=OFF \
  -DNEKTAR_USE_OPENBLAS=ON \
  -DNEKTAR_USE_HDF5=OFF \
  -DNEKTAR_USE_FFTW=OFF \
  -DNEKTAR_USE_ARPACK=OFF \
  -DNEKTAR_USE_PETSC=OFF \
  -DNEKTAR_USE_METIS=OFF \
  -DNEKTAR_USE_VTK=OFF \
  -DNEKTAR_USE_MESHGEN=OFF \
  -DNEKTAR_USE_CCM=OFF \
  -DNEKTAR_USE_CGNS=OFF \
  -DNEKTAR_USE_CWIPI=OFF \
  -DNEKTAR_USE_LST=OFF \
  -DNEKTAR_USE_LIKWID=OFF \
  -DTHIRDPARTY_BUILD_BOOST=OFF \
  -DTHIRDPARTY_BUILD_TINYXML=OFF \
  -DTHIRDPARTY_BUILD_ZLIB=OFF \
  -DTHIRDPARTY_BUILD_BLAS_LAPACK=OFF \
  -DTHIRDPARTY_BUILD_SCOTCH=OFF \
  -DTHIRDPARTY_USE_SSL=ON
```

引号展开后的逐 token JSON 也必须保存；未知/未消费 option 直接失败。
配置后新增硬断言：

- `SCOTCH_HEADERS_DIRS`、`SCOTCH_INCLUDE_DIR`、`SCOTCH_LIBRARY`、
  `SCOTCHERR_LIBRARY` 的 canonical realpath 全在 `SCOTCH_SYSROOT`；
- `SCOTCH_FOUND=TRUE`、`NEKTAR_USE_SCOTCH=ON`、
  `THIRDPARTY_BUILD_SCOTCH=OFF`；
- `NEKTAR_USE_OPENBLAS=ON`、`NEKTAR_USE_SYSTEM_BLAS_LAPACK=OFF`；
- `BLAS_LAPACK`/`LAPACK_LIBRARIES` 的每个 realpath、SHA 入 manifest，
  且 `libopenblas` 解析到上述固定文件；
- 没有任何 `THIRDPARTY_BUILD_*=ON`、`NOTFOUND` 或网络下载资产。

### 1.3 build 与运行时 wrapper

build 命令固定：

```bash
env -i PATH=/usr/bin:/bin LC_ALL=C \
  SCOTCH_DIR="$SCOTCH_SYSROOT/usr" SCOTCH_INCDIR="$SCOTCH_INC" \
  OPENBLAS_HOME=/usr \
  unshare --user --map-root-user --net \
  /usr/bin/cmake --build "$NEK_BUILD" --parallel 6 \
  --target IncNavierStokesSolver FieldConvert NekMesh Tester
```

所有 CTest、NekMesh、FieldConvert 和 solver 调用必须通过唯一 wrapper；其
规范内容现在固定为：

```sh
#!/bin/sh
set -eu
export LC_ALL=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/platform/external/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb/dependency_sysroot/usr/lib/x86_64-linux-gnu"
exec "$@"
```

wrapper 必须用 `apply_patch` 创建，首次执行前冻结 SHA。四个 binary 的
`readelf -d` 和 wrapper 下 `ldd` 都入 manifest；任何 `not found`、Scotch
不来自 sysroot 或 OpenBLAS 不来自固定 realpath均为 G0a no-go。

## 2. 替换 v2 §3.1--3.2：唯一可机械生成的 hybrid mesh

### 2.1 analytic curve 与 cubic Hermite--Bezier 规则

保留 v2 的 closed NACA0015 公式、domain、H0/H1/H2 数值和 material
coordinate。删除“每相邻节点一条未定义 cubic BSpline”和“全部体单元为
triangle”的文字。唯一 topology 是：

> 物面 BoundaryLayer 内为 quadrilateral；fan 与其余外场为 triangle。

对每层 \(N=\{64,128,256\}\)，令
\(x_i=(1-\cos(i\pi/N))/2\)、\(s_i=\sqrt{x_i}\)。材料坐标固定为
\(\xi=s=\sqrt{x}\in[0,1]\)，上/下表面
\(\mathbf r_\pm(s)=(s^2,\pm y_t(s^2))\)。每个
\([s_i,s_{i+1}]\) 用唯一 cubic Hermite--Bezier：

\[
\begin{aligned}
\mathbf P_0&=\mathbf r(\theta_i),\\
\mathbf P_1&=\mathbf P_0+\frac{\Delta s}{3}\mathbf r'(s_i),\\
\mathbf P_2&=\mathbf P_3-\frac{\Delta s}{3}\mathbf r'(s_{i+1}),\\
\mathbf P_3&=\mathbf r(\theta_{i+1}).
\end{aligned}
\]

\[
x'(s)=2s,\qquad y'_\pm(s)=\pm 2s\,y_t'(s^2).
\]

在 \(s=0\) 使用解析极限
\(y'_+=+0.222675,\ y'_-=-0.222675\)；在 \(s=1\) 使用
\(y'_+=-0.363375,\ y'_-=+0.363375\)。这样 LE/TE 均没有零切向量。
浮点常数以 Python binary64 `repr` 写入 `.geo`，禁止格式截断。

使用 Gmsh Built-in kernel 的 `Bezier`，而非 OCC/自动 spline：

- upper curve tags `1..N`，全部按 LE→TE；
- lower curve tags `N+1..2N`，全部按 LE→TE；
- shared endpoint point tags `LE=1, TE=2`；upper interior endpoint
  `i=1..N-1` 的 tag 是 `1000+i`，lower 是 `2000+i`；
- upper interval `i=0..N-1` 的两个 control-point tags 是
  `300000+2*i`、`300000+2*i+1`，lower 对应
  `400000+2*i`、`400000+2*i+1`；
- inner loop 有向序列
  `{1,2,...,N,-2N,-(2N-1),...,-(N+1)}`；
- outer point tags `900001..900004` 为
  `(-32,-32),(32,-32),(32,32),(-32,32)`；
- outer curve tags `100001..100004` 按 CCW；
- outer loop `100010={100001,100002,100003,100004}`；
- inner loop tag `100011`，fluid surface tag `100020`。

固定 Physical identities：

```text
Physical Curve("body_upper",101) = upper 1..N
Physical Curve("body_lower",102) = lower N+1..2N
Physical Curve("outer_bottom",201) = 100001
Physical Curve("outer_right",202) = 100002
Physical Curve("outer_top",203) = 100003
Physical Curve("outer_left",204) = 100004
Physical Surface("fluid",301) = 100020
```

### 2.2 完整 Gmsh field graph

每层把 v2 表内的 `hwall/hinner/hwake/htransition` 机械代入：

```text
Field[1] = BoundaryLayer
Field[1].CurvesList = generator literal-expands 1,2,...,2N
Field[1].Size = hwall
Field[1].SizeFar = hinner
Field[1].Ratio = 1.15
Field[1].Thickness = 0.08
Field[1].Quads = 1
Field[1].FanPointsList = shared TE point tag
Field[1].FanPointsSizesList = {8}
Field[1].AnisoMax = 10
Field[1].IntersectMetrics = 0
BoundaryLayer Field = 1

Field[2] = Box
Field[2].XMin=-0.5; Field[2].XMax=2
Field[2].YMin=-1.25; Field[2].YMax=1.25
Field[2].ZMin=-1; Field[2].ZMax=1
Field[2].VIn=hinner; Field[2].VOut=4; Field[2].Thickness=0.25

Field[3] = Box
Field[3].XMin=1.5; Field[3].XMax=8
Field[3].YMin=-1.25; Field[3].YMax=1.25
Field[3].ZMin=-1; Field[3].ZMax=1
Field[3].VIn=hwake; Field[3].VOut=4; Field[3].Thickness=0.50

Field[4] = Box
Field[4].XMin=-1; Field[4].XMax=10
Field[4].YMin=-4; Field[4].YMax=4
Field[4].ZMin=-1; Field[4].ZMax=1
Field[4].VIn=htransition; Field[4].VOut=4; Field[4].Thickness=1.00

Field[5] = Min
Field[5].FieldsList = {2,3,4}
Background Field = 5
```

同时固定：

```text
Mesh.MeshSizeFromPoints = 0
Mesh.MeshSizeFromCurvature = 0
Mesh.MeshSizeExtendFromBoundary = 0
Mesh.MeshSizeMin = hwall
Mesh.MeshSizeMax = 4
Mesh.Algorithm = 6
Mesh.RecombineAll = 0
Mesh.Smoothing = 10
Mesh.ElementOrder = 3
Mesh.HighOrderOptimize = 2
Mesh.MshFileVersion = 4.1
Mesh.Binary = 0
Mesh.RandomFactor = 1e-9
General.NumThreads = 1
```

`Quads=1` 只重组 BL；outer 保持 triangle。必须报告
`line/triangle/quadrilateral` 数量、每个 Physical ID 到 Nektar composite
的双射和每个 composite element count。出现其他 2-D element family、
空 physical group 或多对一含混即 no-go。

### 2.3 generator 和 exact argv

canonical generator 固定为待实现的
`platform/n26g1_nektar_mrf_gate.py mesh-spec`；它只能实现上述公式，不接受
除 `--level` 和输出路径外的数值参数。执行顺序和 argv 固定：

```bash
$PY platform/n26g1_nektar_mrf_gate.py mesh-spec \
  --level H0 --geo "$RUNROOT/mesh/H0.geo"
$PY platform/n26g1_nektar_mrf_gate.py mesh-spec \
  --level H1 --geo "$RUNROOT/mesh/H1.geo"
$PY platform/n26g1_nektar_mrf_gate.py mesh-spec \
  --level H2 --geo "$RUNROOT/mesh/H2.geo"

/usr/bin/gmsh -2 -format msh41 -o "$RUNROOT/mesh/H0.msh" "$RUNROOT/mesh/H0.geo"
/usr/bin/gmsh -2 -format msh41 -o "$RUNROOT/mesh/H1.msh" "$RUNROOT/mesh/H1.geo"
/usr/bin/gmsh -2 -format msh41 -o "$RUNROOT/mesh/H2.msh" "$RUNROOT/mesh/H2.geo"

$NEK_WRAPPER "$NEK_BUILD/dist/bin/NekMesh" \
  "$RUNROOT/mesh/H0.msh" "$RUNROOT/mesh/H0.xml"
$NEK_WRAPPER "$NEK_BUILD/dist/bin/NekMesh" \
  "$RUNROOT/mesh/H1.msh" "$RUNROOT/mesh/H1.xml"
$NEK_WRAPPER "$NEK_BUILD/dist/bin/NekMesh" \
  "$RUNROOT/mesh/H2.msh" "$RUNROOT/mesh/H2.xml"
```

没有 NekMesh process module；只是 fixed Gmsh input→Nektar XML conversion。
在任何 `.msh` 产生前，generator、三份 `.geo` 和逐 token argv 必须冻结
SHA 并由审计脚本确认三份 `.geo` 只在预登记 level 常数处不同。若任一输出
路径已存在，generator/launcher 必须拒绝覆盖；禁止生成多个 mesh 后挑选。

v2 几何门中的 `analytic control-node residual` 改为：

- 每个 material endpoint 对 analytic NACA residual `<=5e-13 c`；
- 每个两个 Hermite control point 对上述固定导数公式 residual
  `<=5e-13 c`。

其他 Jacobian、closure、perimeter、element/DOF/resource 门不变。

## 3. 替换 v2 §4.1 和 §4.3：唯一 MRF XML 与可执行 K0--K2

### 3.1 source MRF XML

Nektar v5.9 evaluator 没有任意 `if`，故右连续 gate 唯一编码为：

\[
H_+(t)=\mathrm{floor}\{[\mathrm{sign}(t-0.2)+2]/2\},
\quad \tau=\max(t-0.2,0).
\]

它在 `t<0.2` 为 0，在 `t=0.2` 和之后为 1。source session 必须逐字生成：

```xml
<FUNCTION NAME="VelMRF">
  <E VAR="Theta_z" VALUE="-floor((sign(t-0.2)+2)/2)*0.6*(max(t-0.2,0)+(exp(-4.6*max(t-0.2,0))-1)/4.6)"/>
  <E VAR="Omega_z" VALUE="-floor((sign(t-0.2)+2)/2)*0.6*(1-exp(-4.6*max(t-0.2,0)))"/>
  <E VAR="DOmega_z" VALUE="-floor((sign(t-0.2)+2)/2)*2.76*exp(-4.6*max(t-0.2,0))"/>
</FUNCTION>
<FORCING>
  <FORCE TYPE="MovingReferenceFrame">
    <FRAMEVELOCITY> VelMRF </FRAMEVELOCITY>
    <PIVOTPOINT> 0.25, 0.0, 0.0 </PIVOTPOINT>
    <OutputFile>mrf_motion</OutputFile>
    <OutputFrequency>1</OutputFrequency>
  </FORCE>
</FORCING>
```

XML generator 在所有正式 solver knots，以及
`0,0.2-dt,0.2,0.2+dt,1.6971,2.0172,2.05` 上同时计算 evaluator-compatible
expression 和独立 piecewise oracle；使用 v2 固定尺度，max error
`<=1e-12`。XML/manifest SHA 必须在第一个 flow output 前冻结。

### 3.2 K0 static identity

固定无翼网格为 `[-1,1]^2` 的 `16x16` transfinite quadrilateral mesh，
geometry order 1，`u/v` NUMMODES 4、`p` NUMMODES 3；四边全部
`u=1,v=0,p=0` Dirichlet。`dt=1e-3`、10 步、`Kinvis=1e-4`。

只比较 zero-motion MRF 与无 MRF 的每步 `u/v/p` field、四边 mass flux 和
discrete divergence；无物面所以不比较 force。v2 固定尺度下 L2/Linf max
error `<=1e-10`，mass-flux/divergence scale error `<=1e-12`。

### 3.3 K1 moving-frame uniform absolute flow

使用同一无翼网格和正式 source MRF XML，但在表达式中机械替换
`t -> t+0.4`；初值和 exact body-basis velocity为：

\[
\theta=\Theta_z(t+0.4),\quad
\mathbf u_B=(\cos\theta,-\sin\theta),\quad p=0.
\]

`MovingFrameFar` 在四边接收 inertial `(1,0)`；`dt=1e-6`、10 步、
IMEX2。固定 MRF 绝对速度方程 residual：

\[
\mathbf R_m=\partial_t\mathbf u_B+
[\mathbf u_B-\boldsymbol\Omega\times(\mathbf x-\mathbf x_p)]
\cdot\nabla\mathbf u_B+
\boldsymbol\Omega\times\mathbf u_B+\nabla p-\nu\nabla^2\mathbf u_B ,
\]
\[
R_d=\nabla\cdot\mathbf u_B .
\]

独立 oracle 使用解析
\(\partial_t\mathbf u_B=-\boldsymbol\Omega\times\mathbf u_B\)，固定
`32x32` Cartesian sample 和完整 Nektar quadrature。analytic
`R_m/R_d` 尺度化 L2/Linf `<=1e-12`；第 10 步 Nektar field 对 exact 的
尺度化 L2/Linf `<=1e-9`。该门不包含物面 DOmega pressure BC；那由官方
MRF regression 和 G2 实体表面账共同覆盖。

### 3.4 K2 adapter sign/rotation unit

K2 只称 adapter unit，不冒充 native filter test。固定映射：

\[
\mathbf F_I=R(\theta)\mathbf F_B,\qquad M_{z,I}=M_{z,B}.
\]

五个 binary64 输入/期望为：

| theta | `(Fxb,Fyb,Mzb)` | `(Fxi,Fyi,Mzi)` |
|---:|---|---|
| -0.6 | `(1,2,0.3)` | `(1.9546205616997492,1.0860287564243212,0.3)` |
| -0.3 | `(-2,1,-0.2)` | `(-1.6151527715898724,1.546376902448285,-0.2)` |
| 0.0 | `(0.5,-0.25,0.4)` | `(0.5,-0.25,0.4)` |
| 0.3 | `(3,-1,-0.5)` | `(3.1615296740381575,-0.068775869141587398,-0.5)` |
| 0.6 | `(-0.75,-1.5,0.7)` | `(0.22796199891029434,-1.6614852774107942,0.7)` |

固定尺度 max error `<=1e-12`。native `AeroForces` 不在 K2 宣称；它必须在
G2 用 source primitive/body/inertial 三账实测。

K0--K2 的唯一入口为：

```bash
$PY platform/n26g1_nektar_mrf_gate.py manufactured \
  --build-manifest "$RUNROOT/build/build_manifest.json" \
  --output "$RUNROOT/gates/manufactured_result.json"
```

脚本、K0/K1 mesh/XML 和 expected arrays 在执行前一并冻结 SHA；命令不接受
阈值或数值 override。

## 4. 替换 v2 §5 输出和 Cauchy 定义

### 4.1 formal curve 与 evidence-extraction replay

五个 formal runs 全部从 `t=0` 独立运行到 `2.05`；`AeroForces` 每步输出，
正式 CL/CD 只来自这五条连续曲线，绝不由 restart 拼接。

不再每 `0.01` checkpoint。只对 H1/`2.5e-4`、H2/`5e-4` 和
H2/`2.5e-4` 固定加 checkpoint；`OutputFrequency` 分别是
`800,400,800`，XML 中写整数，不调用 evaluator 不支持的 `round`：

```xml
<FILTER TYPE="Checkpoint">
  <PARAM NAME="OutputFile">restart_0p2</PARAM>
  <PARAM NAME="OutputFrequency">LEVEL_SPECIFIC_INTEGER</PARAM>
  <PARAM NAME="OutputStartTime">0.0</PARAM>
</FILTER>
```

H0 不保留周期场。formal curve 结束后按固定 first-argmax 得到 H2-fine 的
CL/CD extrema step；nominal 44/55 deg 按最近 step、tie earlier。对
H1/H2 spatial-final-two 和 H2 time-final-two，在每个 44/55 step；另对
H2-fine 每个不重复 extrema step：

1. 取不晚于目标 step 的最近 `0.2` checkpoint；
2. 用完全相同 mesh/session/binary 从该 checkpoint replay 到目标 step；
3. replay 只用于导出该 step primitive/field，不进入 formal curve；
4. replay 与 formal curve 的重叠每步 `CL/CD/CM` absolute difference
   `<=1e-10`，time、Theta/Omega/DOmega bitwise-identical；
5. 不满足则 `RESTART-EVIDENCE NO-GO`，不能从别的 checkpoint 重试。

field retention 只有：0.2 checkpoints、44/55、H2-fine extrema 和每个 run
最终场。performance pilot 必须按“五个 formal + 最坏八个不超过 0.2 的
replay”估算，合计仍须 `<=96 h`、磁盘 `<=12 GiB`、保留空间 `>=15 GiB`。

### 4.2 Cauchy 指标

删除 v2 未定义的“range-normalized L2 change”文字。对任意 coarse/fine
曲线，在各字段冻结 support 上，以 union knots 的分段线性误差精确积分：

\[
I_{cf}=\sum_i\frac{\Delta\theta_i}{3}
(e_{0,i}^2+e_{0,i}e_{1,i}+e_{1,i}^2),
\]
\[
E_{L2}=
\frac{\sqrt{I_{cf}/(\theta_{\max}-\theta_{\min})}}{R_{\rm ref}},
\qquad
E_{\rm peak}=\frac{|\max g_c-\max g_f|}{R_{\rm ref}}.
\]

固定：

```text
R_ref(CD)=3.7916000008521866
R_ref(CL)=3.908274940826322
```

两轴最后两级的 CL/CD `E_L2`、`E_peak` 各 `<=3%`。reference ranges 非零，
无可选分母。v2 的 time observed-order 公式、`[5,20] deg` 和退化判据不变。

## 5. 替换 v2 §6.2--6.3：cross-mass 一致载荷与虚功

### 5.1 两个不同但相容的输出

G2 不再把“在共同点求值”称作保守传力。每个 upper/lower side 单独定义：

1. **分布诊断场**：投影到每侧 512 cosine-\(x\) intervals（节点
   \(x_i=(1-\cos(i\pi/512))/2,\ \xi_i=\sqrt{x_i}\)）的 continuous P1
   common material space \(V_c=\mathrm{span}\{N_i(\xi)\}\)；
2. **结构一致载荷**：对任意已冻结结构位移基
   \(\boldsymbol\psi_j(\xi)\)，直接从 native aerodynamic traction 形成
   \(f_j=\int_\Gamma\boldsymbol\psi_j\cdot\mathbf t_a\,ds\)，不经过总力重分配。

### 5.2 cross-mass/L2 operator

分布诊断使用同一个 analytic closed-NACA0015 reference surface
\(\Gamma_{\rm ref}\) 和 \(ds_{\rm ref}\)，因此所有 level 的 mass matrix
完全相同。对每一侧、每个 traction component、pressure 和 viscous 分开：

\[
M_{ij}=\int_{\Gamma_{\rm ref}} N_iN_j\,ds_{\rm ref},\qquad
b_i=\int_{\Gamma_{\rm ref}} N_i t_a(\xi)\,ds_{\rm ref},\qquad
M\mathbf a=\mathbf b .
\]

分布投影的积分 partition 是 native aerodynamic boundary elements 与 common
cosine-\(x\) intervals 的全部 material-coordinate intersections。每个
intersection 固定用 32-point Gauss--Legendre；32→64 点的每个
`M,b,force,moment,virtual-work` fixed-scale change 必须 `<=1e-12`。
`M` 用 float64 symmetric Cholesky，condition number 必须 `<=1e10`，
solve relative residual `<=1e-12`；失败不允许正则化。

分布比较使用 mass norm：

\[
E_t=\frac{\sqrt{(\mathbf a_c-\mathbf a_f)^TM
(\mathbf a_c-\mathbf a_f)}}
{\max(q_\infty\sqrt{s_{\rm side}},
\sqrt{\mathbf a_f^TM\mathbf a_f})}.
\]

在 nominal 44/55 deg，H1/H2 和 H2 medium/fine 的每个 side、`x/y`
component、pressure/viscous channel 分别要求 `E_t<=3%`。删除 v2 中未定义的
\(d\mathbf t/d\xi\) 弱导数门。

### 5.3 刚体与非刚体虚功基

\(\xi\in[0,1]\) 对 upper/lower 都按 LE→TE；\(\chi_+\)、\(\chi_-\) 是
只在对应 side 非零的 indicator。固定测试位移：

```text
R_x       = e_x
R_y       = e_y
R_theta   = e_z × (r(xi)-r_p), r_p=(0.25,0)

B_sigma   = chi_sigma sin(pi xi)^2 e_y
C_sigma   = chi_sigma sin(pi xi)^2 e_x
T_sigma   = chi_sigma sin(pi xi)^2 [e_z × (r(xi)-r_p)]
S_sigma   = chi_sigma sin(pi xi)^2 n_sf(xi)
sigma     = upper, lower
```

`R_x/R_y/R_theta` 分别对应物理合力和 quarter-chord moment；后八个模式
分别覆盖 side-resolved bending、chordwise deformation、twist-like 和
skin-normal deformation。

结构 consistent load 不使用上述 diagnostic \(ds_{\rm ref}\)，而是在每个
level 的 native aerodynamic material surface 上用同一 32/64 点交叉积分，
把结构位移基直接评价在 native \(\xi\) 上。对每个模式，其离散虚功与
native traction 直接积分：

\[
E_W=\frac{|W_{\rm consistent}-W_{\rm native}|}
{\max(q_\infty c,|W_{\rm native}|)}\le10^{-10}.
\]

H1/H2 和 H2 medium/fine 的非刚体虚功 Cauchy：

\[
\frac{|W_c-W_f|}{\max(q_\infty c,|W_f|)}\le3\%.
\]

upper/lower pressure/viscous/total 分开保存，再求和。合力/合矩与
`AeroForces` 的 v2 `<=1e-8` 门继续有效；native body→inertial force
transform 也只在此实体 source 账中获得证据。

这里的可推广 claim 只到：

> 对任意随后冻结的结构 shape functions，可用同一 native-traction
> cross-integral 形成 consistent nodal load，并逐自由度保持虚功；
> 本轮预登记模式用于验证实现，不把总力按面积回填。

它不声称这八个模式穷尽未来结构设计空间。

## 6. v3 执行锁

在复合规范获得新的 `PASS TO EXECUTE` 前：

```text
archive extraction = OFF
dependency acquisition = OFF
build = OFF
mesh generation = OFF
manufactured/source flow = OFF
target = OFF
Fig17/18/19 improvement = NOT TESTED
```

复审必须同时核验 base v2、v2 audit 与本覆盖文件。任何剩余 P0 只允许在
产生 `.msh`/build/flow 前形成新的版本化规范；一旦这些输出开始，版本、
dependency、mesh、session、gates、transfer、threshold 和 retention 均锁定，
失败只能回写 `N2.6g1`。
