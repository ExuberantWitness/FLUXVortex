# N2.6g1 Nektar++ body-fixed MRF 来源门预登记 v4

日期：2026-07-30  
状态：`PREREGISTERED-v4 / MECHANICAL DIFF AUDIT REQUIRED / SOURCE FLOW OFF / TARGET OFF`  
活动节点：`N2.6g1`

## 0. 复合规范、身份和不变项

本文件是 v3 独立审计后、任何 extraction/build/mesh/flow 之前的单一
mechanical errata。完整预登记对象依次为：

1. base v2：
   `n26g1_nektar_mrf_source_prereg_v2_20260730.md`,
   SHA-256
   `14d29b5b6523ed523efae255b345edab61a1a5b2f31e54d265f986963d3c9b0b`；
2. v2 audit：
   `n26g1_nektar_mrf_prereg_v2_audit_20260730.md`,
   SHA-256
   `95c6f42a75c73fefb5e6e04eccff15f85690b4021deef08baf1e0e89e6f73527`；
3. v3：
   `n26g1_nektar_mrf_source_prereg_v3_20260730.md`,
   SHA-256
   `35703a0b55a2aaff1ea507eb23dcc460d35491d5324f9c885e857e8f8f04a782`；
4. v3 audit：
   `n26g1_nektar_mrf_prereg_v3_audit_20260730.md`,
   SHA-256
   `9456705d5a134580e999648dda7dd14c39eb34e1fefb0f7e798e35b3e868a1c8`；
5. 本文件的覆盖条款。

本文件只覆盖下列四个 P0 和列出的 P2 澄清；其余 v3/v2 条款继续有效。
候选、claim、Nektar commit、source 物理、正式五条数值轴、网格尺寸、
方程、BC/IC、阈值、reference、V4.1、target lock 和失败语义均不改变。
禁止把不同版本任选其一解释。

## 1. 替换 v3 §1 的路径和 source immutability scope

### 1.1 五个互不重叠的持久化 scope

固定：

```text
ROOT=/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV
NEK_CONTAINER=$ROOT/platform/external/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb
NEK_ARCHIVE=$NEK_CONTAINER/source_receipt/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb.tar.gz
NEK_PAYLOAD_PARENT=$NEK_CONTAINER/source_payload
NEK_SRC=$NEK_PAYLOAD_PARENT/nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb
NEK_BUILD=$NEK_CONTAINER/build-release-serial
SCOTCH_SYSROOT=$NEK_CONTAINER/dependency_sysroot
SCOTCH_INC=$SCOTCH_SYSROOT/usr/include/scotch
SCOTCH_LIBDIR=$SCOTCH_SYSROOT/usr/lib/x86_64-linux-gnu
SCOTCH_LIB=$SCOTCH_LIBDIR/libscotch.so
SCOTCHERR_LIB=$SCOTCH_LIBDIR/libscotcherr.so
NEK_WRAPPER=$NEK_CONTAINER/run_nektar_serial.sh
BUILD_RECEIPT=$NEK_CONTAINER/build_receipt
PY=/home/exuber/anaconda3/envs/fluxvortex/bin/python
RUNROOT=$ROOT/platform/data_external/n26g1_nektar_mrf/source_gate_v4
```

互斥 scope 为：

1. immutable payload：`NEK_SRC`；
2. source receipt/archive：`NEK_CONTAINER/source_receipt`；
3. build：`NEK_BUILD`；
4. dependency sysroot：`SCOTCH_SYSROOT`；
5. wrapper/build receipt：`NEK_WRAPPER`、`BUILD_RECEIPT`。

任何路径 realpath 重叠、symlink 逃逸或 build file 落入 `NEK_SRC` 都直接
`G0a NO-GO`。

### 1.2 唯一 extraction 和 immutable manifest

extraction 前必须：

- archive bytes/SHA/headers 与 v2 receipt 一致；
- tar member 全为相对路径，不含 `..` component；
- 所有 member 都在唯一顶层目录
  `nektar-f729cda85b6a206e008fd705af8001cfe6e0d6fb/`；
- `NEK_SRC` 不存在，`NEK_PAYLOAD_PARENT` 不存在或为空；拒绝覆盖。

唯一 extraction argv：

```text
["/usr/bin/tar","--extract","--gzip",
 "--file",NEK_ARCHIVE,
 "--directory",NEK_PAYLOAD_PARENT,
 "--no-same-owner"]
```

只允许 launcher 创建空的 `NEK_PAYLOAD_PARENT` 后执行该 argv。对
`NEK_SRC` 生成 canonical manifest，逐项包含：

```text
relative POSIX path, type(file/dir/symlink), permission mode,
regular-file bytes and SHA-256, symlink literal target
```

条目按 `LC_ALL=C` 的 UTF-8 byte order 排序；manifest JSON 使用 sorted
keys、UTF-8、无 NaN 和单一换行，保存在 scope 外：

```text
$NEK_CONTAINER/source_receipt/source_payload_manifest_post_extract.json
```

configure 前和四 target build/official CTest 后各重新计算同一
`NEK_SRC` scope，后者保存为：

```text
$BUILD_RECEIPT/source_payload_manifest_post_build.json
```

两份 canonical manifest 必须 byte-identical；新增、删除、内容、mode 或
symlink target 变化计数都必须为 0。没有 `.git` 不允许把 “git diff 空”
冒充此门；不允许 patch、代码生成或测试输出写进 source payload。

### 1.3 对 v3 build argv 的唯一路径替换

v3 §1.2--1.3 的所有 CMake/build options、Scotch/OpenBLAS 绑定、网络隔离、
target、wrapper 内容和 binary linkage 门保持不变，只把：

```text
NEK_ROOT used as source  -> NEK_SRC
NEK_ROOT used as container/build parent -> NEK_CONTAINER
RUNROOT/source_gate_v3 -> RUNROOT/source_gate_v4
```

机械替换。完整展开后的 configure/build argv JSON 仍在执行前冻结；CMake
source 参数必须恰为 `-S "$NEK_SRC"`，build 参数必须恰为
`-B "$NEK_BUILD"`。wrapper 位于 `NEK_WRAPPER`，不在 `NEK_SRC`。

## 2. 替换 v3 §2.1 的 Hermite 端点

保留全部 NACA、material coordinate、tag 和导数条款，唯一公式修正为：

\[
\begin{aligned}
\mathbf P_0&=\mathbf r(s_i),\\
\mathbf P_1&=\mathbf P_0+\frac{\Delta s}{3}\mathbf r'(s_i),\\
\mathbf P_2&=\mathbf P_3-\frac{\Delta s}{3}\mathbf r'(s_{i+1}),\\
\mathbf P_3&=\mathbf r(s_{i+1}).
\end{aligned}
\]

generator、geometry oracle 和审计脚本中不得出现未定义的
`theta_i/theta_{i+1}`。

## 3. 替换 v3 §2.2--2.3 的 composite 双射门

### 3.1 固定 shape split

Gmsh 仍只有一个 volume identity：

```text
Physical Surface("fluid",301) = 100020
```

保留 quad-BL + triangular-outer topology。NekMesh 固定 commit 的
`InputGmsh` 按 shape 对 physical 301 作唯一机械 split。转换后必须恰有：

```text
volume composite ID set = {301,302}
one and only one Triangle composite = C_tri
one and only one Quadrilateral composite = C_quad
{C_tri,C_quad} = {301,302}
```

不预选哪一种 shape 保留 301；它由固定 importer 的 encounter order 决定，
而不是由响应选择。NekMesh stdout 的 `Tag 301 => ...` remap 行、最终 XML
的 composite shape/count 及解析后的：

```json
{"physical_surface":301,
 "triangle_composite":C_tri,
 "quadrilateral_composite":C_quad}
```

必须一致并冻结 SHA。缺 shape、多于两个 volume composites、ID set 不是
`{301,302}`、混合 shape composite 或 remap/log/XML 不一致均 no-go。

### 3.2 DOMAIN、EXPANSIONS 和 boundary identities

最终 geometry 的 `DOMAIN` 必须恰好覆盖：

```text
C[C_tri,C_quad]
```

每层 session generator 把解析出的两个 literal ID 以升序同时写入全部
expansions：

```xml
<E COMPOSITE="C[C_tri,C_quad]" NUMMODES="P"
   TYPE="MODIFIED" FIELDS="u,v" />
<E COMPOSITE="C[C_tri,C_quad]" NUMMODES="P-1"
   TYPE="MODIFIEDQUADPLUS1" FIELDS="p" />
```

其中最终 XML 必须写 literal integers，不保留 `C_tri/C_quad/P`
placeholder。官方 MRF regression 本身使用 mixed volume composites 的
同一 expansion family；本条不改变离散阶次。

boundary physical IDs `101,102,201,202,203,204` 各自仍须映射到唯一、
非空、纯 Segment boundary composite，且 element count 与 `.msh` 相同。
删除 v3 的“每个 Physical ID 到 Nektar composite 双射”总括句，替换为：

> boundary IDs 一对一；volume Physical Surface 301 按预登记 shape split
> 一对二，除此之外禁止任何 remap。

## 4. 完全替换 v3 §4.1 的 checkpoint/replay

### 4.1 五个 formal runs 内直接保存 nominal 44/55

删除全部周期 `0.2` checkpoint、`LEVEL_SPECIFIC_INTEGER`、checkpoint
restart 和“最多八个 0.2 replay”。五个 formal runs 仍全部从 `t=0`
独立连续运行到 `2.05`，正式曲线身份不变。

独立解析运动 oracle 给出：

```text
alpha=44 deg continuous root: t=1.6970773734196007
alpha=55 deg continuous root: t=2.0172255339938934
```

按“最近 solver angle；绝对差相等时 earlier step”的冻结规则，literal
step 表为：

| dt | step44 | t44 | alpha44(deg) | step55 | t55 | alpha55(deg) |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-3` | 1697 | 1.697 | 43.997342815142176 | 2017 | 2.017 | 54.99224852934752 |
| `5e-4` | 3394 | 1.697 | 43.997342815142176 | 4034 | 2.017 | 54.99224852934752 |
| `2.5e-4` | 6788 | 1.697 | 43.997342815142176 | 8069 | 2.01725 | 55.000840882345955 |

每个 formal session 包含两个独立 `Checkpoint` filters。对每行和每个
nominal \(k\)，固定：

```text
OutputFile = nominal_44 或 nominal_55
OutputFrequency = literal step44 或 step55
OutputStartTime = literal (t_k - dt/2)
```

故每个 filter 在 `t_k` 第一次输出；由于 `2*t_k>2.05`，每个 formal run
恰好保留一个 44 和一个 55 snapshot。冻结 XML 中不得出现公式、
placeholder 或旧 `restart_0p2` token。运行前 parser 必须从 filter
源码规则独立预测 filename/index/time；实际数量、step、time 任一不符即
`EVIDENCE-RETENTION NO-GO`，不得改 frequency 重跑。

### 4.2 H2-fine extrema 的 deterministic prefix reruns

formal H2/`dt=2.5e-4` 完成后，只从其未平滑 CL/CD 曲线按冻结
first-argmax 分别得到 `step_CLmax`、`step_CDmax`。二者去重；若与已保存
nominal step 相同则直接复用。其余每个唯一 step 只允许一次 prefix rerun，
总数最多 2。

每个 prefix：

- 从 `t=0` 和正式解析 `u=1,v=0,p=0` IC 开始，禁止 restart；
- 使用同一 H2 mesh、同一 binary、`dt=2.5e-4`、IMEXOrder2、MRF XML、
  solver/preconditioner/SVV/dealiasing/BC 和全部正式 filters；
- session 与 formal 的唯一数值差异是
  `NumSteps = literal extrema step`；
- 所有 `OutputFile` basename 可相同，但 cwd 固定为独立
  `RUNROOT/prefix/<quantity>_step_<integer>/`，不得触碰 formal cwd；
- final `.fld` 是该 extrema primitive 的唯一来源。

每个正式/prefix 调用的 exact cwd 和 argv JSON 为：

```text
cwd = unique formal-or-prefix directory
argv = [NEK_WRAPPER,
        NEK_BUILD + "/dist/bin/IncNavierStokesSolver",
        absolute level mesh XML,
        absolute frozen session XML]
```

不传 `--set-start-time`。launcher 只按该 JSON 调用，不经 shell 拼接。
prefix session、argv 和 formal→prefix XML semantic diff 在执行前冻结；
diff whitelist 只允许 `NumSteps` 和非数值输出隔离信息。

prefix 的每个 overlapping step 必须满足：

```text
abs(CL_prefix-CL_formal) <= 1e-10
abs(CD_prefix-CD_formal) <= 1e-10
abs(CM_prefix-CM_formal) <= 1e-10
time/Theta/Omega/DOmega = bitwise-identical
```

prefix response 不进入 G3 RMSE/peak 或 Fig17/18/19 评分，只提供 frozen
extrema field。任何前缀差异、缺步、额外重跑或从另一个 extrema 选择
snapshot 都是 `DETERMINISTIC-PREFIX NO-GO`。

### 4.3 更新 retention/resource 门

持久 field 只有：

- 五个 formal 各自的 nominal 44、nominal 55 和 final；
- H2-fine 至多两个去重 extrema prefix final fields；
- manufactured gates 明示的小场输出。

performance pilot 资源外推改为：

```text
five complete formal runs
+ at most two H2-fine prefix runs from t=0 to their registered extrema steps
```

合计仍须 `<=96 h`，任一 H2 run 仍须 `<=36 h`，全部 build/mesh/output
仍须 `<=12 GiB`，根分区始终保留 `>=15 GiB`。失败只能回写
`RESOURCE-NO-GO`，不得恢复 restart、删 field、改 mesh/dt 或放宽门。

## 5. 替换 v3 §5 中的 P2 含混

### 5.1 reference 与 native 几何角色

common diagnostic mass matrix 和跨层非刚体 mode 使用 level-independent
analytic closed-NACA0015：

```text
r_ref(xi), n_sf,ref(xi), ds_ref
```

刚体 moment identity 使用每层 native geometry：

```text
R_theta,h = e_z × (r_h(xi)-r_p)
```

从而精确恢复该层 native `AeroForces` moment。非刚体
`T_sigma/S_sigma` 分别使用 `r_ref/n_sf,ref`，避免把 mesh geometry 误差
混入结构 mode identity。consistent load 仍在 native aerodynamic surface
上积分并按 material `xi` 评价这些函数。

### 5.2 独立 accumulator、范数和量纲

对每个测试 mode 取 unit generalized displacement \(q_j=1\)，固定：

\[
W_{\rm consistent}=\mathbf q^T\mathbf f,
\qquad
W_{\rm native}=\int_{\Gamma_h}\boldsymbol\psi_j\cdot\mathbf t_a\,ds_h .
\]

两者必须由独立 accumulators 计算：consistent path 先形成 load vector 再
点乘；native path 直接积分，禁止复用同一返回标量。

32→64 quadrature 门用所有 matrix/vector/channel entries 的 fixed-scale
elementwise maximum：

```text
M entry scale = c
b/force/translational-mode work scale = q_inf*c
moment/rotational-or-twist-mode work scale = q_inf*c^2
```

condition number 固定为 spectral 2-norm
`kappa_2(M)=sigma_max/sigma_min`。solve residual 固定：

\[
\frac{\|M\mathbf a-\mathbf b\|_2}
{\max(\|\mathbf b\|_2,q_\infty c)}\le10^{-12}.
\]

`R_theta` 和 `T_sigma` 的虚功误差分母使用 \(q_\infty c^2\)；其余位移
mode 使用 \(q_\infty c\)。pressure/viscous/total、upper/lower 继续分账。

### 5.3 Gmsh 随机身份

在 v3 固定 Gmsh options 中增加并唯一冻结：

```text
Mesh.RandomSeed = 1
```

这等于 Gmsh 4.8.4 默认值，只把默认变成显式身份，不改变 mesh family。

## 6. v4 执行锁和差分复核

本文件生成后仍保持：

```text
archive extraction = OFF
dependency acquisition = OFF
build = OFF
mesh generation = OFF
manufactured/source flow = OFF
target = OFF
Fig17/18/19 improvement = NOT TESTED
```

差分复核只回答：

1. 四个 v3 P0 是否由本文件机械关闭；
2. 本文件是否意外改变候选、物理、数值轴、阈值或 target lock；
3. source/path/composite/prefix/structure 处方是否仍有未绑定自由度。

若差分复核 `PASS TO EXECUTE`，只授权 G0a：

```text
archive extraction + dependency acquisition + isolated build
+ source immutability comparison + official CTest
```

不因此授权 source CL/CD，更不授权 target/Fig17/18/19。后续严格按
G0a→G0b→G0c/K0--K2→performance pilot→G1/G2/G3→N2.6g2 G4 顺序推进；
任一门失败只回写 claim，禁止同轮换实现。
