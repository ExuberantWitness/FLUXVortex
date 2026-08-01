# N2.6f 全域二维黏性 NS observer 预登记

日期：2026-07-30  
状态：`PREREGISTERED / IMPLEMENTATION AUTHORIZED / PHYSICS NOT YET RUN`  
活动节点：`N2.6f`  
冻结基线：V4.1；本 observer 不修改其代码、参数、力或 184 工况缓存。

## 1. 病因、证据和可动空间

corrected Fig18 在
`f=1.4 Hz, nominal twist=22.5 deg, AoA=5 deg` 的冻结残差
`model - experiment` 为：

| U (m/s) | V4.1 thrust residual (N) |
|---:|---:|
| 6 | +1.110 |
| 8 | +1.805 |
| 10 | +2.037758 |

模型随来流增大的压力阻力量级不足，但 Meng 公开资料没有同步双侧压力、
壁面剪切、分离线或 wing-off tare，故总力本身不能唯一证明 N2.6。
`N2.6e1bc0` 又已经证明 regular outer corner mode 不足以承担 generic
\(O(\Delta t)\) 出生；后续审计进一步证明：

1. 平板 \(B_v\)、LESP、\(f_2\) 或当前有限 IBL 亏损矩不能直接成为
   finite-angle \(B_{TE}\)；
2. 当前 Riziotis strong-VI 实现依赖的最近尾缘控制点在加密时仍有
   `7.33--9.36%` 非 Cauchy 变化；
3. 完整 strong coupling 不会自动修复未定义的 finite-angle 点迹，整翼
   remesh 后的 IBL/势历史转移也没有来源规定；
4. 局部 NS--potential patch 还会新增未闭合的接口位置、入流剖面和
   traction trace。

所以本轮唯一可动空间不是再造一个标量闭合，而是：

> 用一个不含人工 NS--outer 接口的全域二维、移动无滑移壁面 Navier--Stokes
> observer，直接观察双侧 \(p(s,t)\)、\(\tau_w(s,t)\)、分离拓扑和有限控制体
> 涡量账，判断 N2.6 的“缺失分离压力”方向是否成立。

它是证据生成器，不是第三套生产气动力，不向 V4.1 追加总力。

## 2. 学科机理与方向裁决

求解无量纲二维不可压缩黏性方程：

\[
\nabla\cdot\mathbf u=0,\qquad
\partial_t\mathbf u+\mathbf u\cdot\nabla\mathbf u
=-\nabla p+Re^{-1}\nabla^2\mathbf u ,
\]

移动实体面满足完整 no-slip/no-penetration。压力、壁面剪切和涡量必须来自
同一收敛流场；总 traction 只积分一次：

\[
\mathbf t=-(p-p_\infty)\mathbf n+\tau_w\mathbf t_s .
\]

采用全域而非局部 NS 的原因是删除当前不可辨识的 artificial interface。
采用 laminar 2D NS 是最小状态选择：不引入 RANS/transition 常数，也不把
LESP 当持续载荷幅值。它只能观察代理二维截面；目标低 Re 转捩、三维横流和
真实试件截面仍是明确适用域缺口。

最终冻结求解器为 **Basilisk** moving embedded-boundary 路径。SU2 只做过
只读资产比较，未安装、未运行，也不是并行候选。

主要一手/原始来源：

- Schneiders et al., *JCP* 235 (2013) 786--809,
  DOI `10.1016/j.jcp.2012.09.038`；
- Visbal & Shang, *AIAA Journal* 27 (1989) 1044--1051,
  DOI `10.2514/3.10219`；
- Basilisk 官方 NACA2414 starting-vortex example：
  `https://basilisk.fr/src/examples/naca2414-starting.c`；
- Basilisk moving NACA0015 source case：
  `https://basilisk.fr/sandbox/ghigo/src/test-navier-stokes/naca0015-pitching.c`。

## 3. 冻结源码资产与许可边界

Basilisk 是滚动快照，不得以后重新抓取 latest：

| asset | frozen identity |
|---|---|
| tarball | `https://basilisk.fr/basilisk/basilisk.tar.gz` |
| size | `9,610,797 bytes` |
| Last-Modified | `2026-07-27 14:51:42 GMT` |
| ETag | `"92a62d-65798dc4d56f2"` |
| SHA-256 | `fe6b4b5821517d792c58f0413ea6de4b5bd6b1d337578bb5ceb2fa6f07f8f193` |
| moving case SHA-256 | `8ff0282a4bfa67473a46f67aea768f27b6db91a44d24296c036f5c701c0acb86` |
| starting case SHA-256 | `5566e55f643ca0f0a0f4e26a79beb8bd17afc9bbb84faf0a5dd9ea4df6a2fe32` |

moving case 的九个公开依赖同样冻结：

| file | SHA-256 |
|---|---|
| `myembed.h` | `e8aa667e61c097a66ae1ba4bc291da8505f6c48da2be6c00a7db26e6ec922a29` |
| `mycentered.h` | `96a17c3a508522962232e81c342a05d8a8071b7de3e5f4da4483ba9101f6dd60` |
| `myembed-moving.h` | `df4c4e16fd8dfea6d61e4a6a34cbed37ec05647d64d8ebc134e262372844a006` |
| `myperfs.h` | `48f4bbd2567aa7c11575fce8b8b5ca202af542ac0f5c5739c66409d56fcc07ec` |
| `myembed-tree-moving.h` | `8710a71ba3af8949415dfaeea3cf15bcc1dfb511c33fd9a4cfcdfb4da44f4e71` |
| `myquadratic.h` | `c9680c2c664c19dff4236d75cc80d771cfa62d7a935bd975f1ee435f2ccb911b` |
| `mytimestep.h` | `dc5896f61405846e09e317fcb53aab33a46358445deb5aac570b461e285c9ce8` |
| `myviscosity-embed.h` | `7e2a3eddc11d9b4d9518d39d9d5e330ad5c48c96f3167ee8b1d35ca965313c19` |
| `mypoisson.h` | `3a8c1b982fad94a0b01fad6335acc5e113529fd52a6f5767d9ce321365062138` |

Schneiders Fig.17 参考曲线：

- `CD` SHA `bb068e3880b2f142527739cd8a425df84dbabeb7f33f789b7217eeda1ff0b022`；
- `CL` SHA `77da2650d58ded6f049b0cc879e2d2e8069ab3029698f8f5a8759e5564c1ff6a`。

许可隔离：

- Basilisk、修改后的 C observer、构建物和 GPLv3 文本保存在
  `platform/external/basilisk-fe6b4b58/`；
- 它以独立进程运行；FLUXV 只消费带 schema 的 CSV/NPZ；
- Basilisk 头或派生 C 不编入 Python/Warp 生产模块；
- 对外分发 observer 时必须随附 GPLv3、固定源码和修改记录。

持久证据只能写入仓库目录，禁止使用 `/tmp`。

## 4. 来源验证门 N2.6f1

任何 RoboEagle 工况之前按固定顺序执行。

### S0：构建和静止 embedded-boundary smoke

运行官方 NACA2414 starting-vortex case：

- `Re=10000`、AoA `6 deg`、域 `16c`；
- 最高 `256 points/c`；
- 输出有限的 `CL/CD`、压力力、黏性力和涡量场。

S0 只验证 qcc、AMR、embedded geometry、压力/黏性力路径可运行，不是
moving-boundary physics GO。

### S1：moving NACA0015 source identity

原样保持来源参数：

- NACA0015、`Re=10000`、quarter-chord pivot；
- `p_w0=0.6`、`p_ts=0.2`、`p_t0=1`；
- 域 `64c`、`DT=0.01 c/U`；
- source formal resolution `512 points/c`；
- Poisson/viscous tolerance `1e-4`；
- 输出 `CL/CD`、44/55 度表面 \(C_p\)、表面涡量和涡结构。

先验证未改源码 formal run，再允许复制为 N2.6f observer。

### S2：独立数值轴

不读取 Fig17/18/19：

- 空间：`128/256/512 points/c`，固定 `DT=0.01 c/U`；
- 时间：`DT=0.02/0.01/0.005 c/U`，固定 `256 points/c`；
- 正式来源结果始终使用预登记最细层，不按目标表现选层级。

通过条件：

1. 无 NaN、嵌入面闭合、压力/黏性力均有限；
2. 最后两级 CL/CD 曲线 relative \(L_2\)、峰值变化均 `<=3%`；
3. 表面 quadrature 的压力加剪切与 `embed_force()` 相对差 `<=1%`；
4. 对 Schneiders Fig.17 的 CL/CD range-normalized RMSE 各 `<=10%`，
   峰值角差 `<=3 deg`；
5. 44/55 度涡结构完成主线程视觉核对，方向、主 LEV/TEV 数目和位置不得反相；
6. 失败不得通过调 `Re`、pitch law、数值耗散或参考数据归一化补救。

S0/S1 编译或资产失败是 `PROTOCOL-NO-GO`；S2 收敛或来源响应失败则
`N2.6f1 -> falsified/frozen`。两者都不反向证伪 Navier--Stokes 方程。

## 5. 唯一目标代表点 N2.6f2

代表点按冻结残差最大值选择，不用来拟合参数：

\[
U=10\ {\rm m/s},\quad f=1.4\ {\rm Hz},\quad
\alpha=5^\circ,\quad tw_{\rm nominal}=22.5^\circ .
\]

冻结 V4.1/实验 thrust 为
`-1.133224843/-3.170982766 N`，残差 `+2.037757923 N`。

先只观察一个由几何--运动学压力容量选择的材料条带，选择器不读取实验或
V4.1 气动力：

\[
S_j=\left\langle q_{2D,j}\Delta y_j
\oint_{\partial A_j}|\mathbf n\cdot\mathbf e_D|\,ds\right\rangle_T .
\]

固定 `ns=8` 时最大者：

- zero-based `j=5`；
- span band `[0.5,0.6] m`，中心 `y=0.55 m`，`\eta=0.6875`；
- `c=0.282037532 m`，`\Delta y=0.1 m`；
- NACA2406-like 标准开尾缘数值代理；
- 固定非扫掠转轴 `x_e=0.25*0.287=0.07175 m`；
- flap 半幅 `22.5 deg`；
- 局部 twist 半幅 `7.734375 deg`，相位 `+90 deg`；
- `rho=1.225 kg/m3`，`nu=1.5e-5 m2/s`。

该截面约覆盖
`U_2D=10.000--10.378 m/s`、`alpha_sec=-13.59--23.24 deg`、
`Re_mean≈1.90e5`、`k≈0.124`；被二维化丢弃的运动学 crossflow RMS 比约
`2.35%`，必须随结果报告。

## 6. 目标运动学和数值协议

复制冻结 V4.1 的运动身份：

\[
\theta(t)=22.5^\circ\sin\Omega t,\qquad
\psi_j(t)=11.25^\circ\eta_j\sin(\Omega t+\pi/2).
\]

二维 observer 使用明确的 planar strip projection：

- pivot 垂向位置 \(h_j(t)=y_j\sin\theta(t)\)；
- pitch 为 \(\psi_j(t)\)；
- 来流为 V4.1 同一 \((U,U\tan\alpha)\)；
- NACA2406-like 双侧壳随 \(h_j,\psi_j\) 完整移动并满足 no-slip；
- 完整三维基向量和被丢弃的 spanwise relative velocity只作为适用域诊断。

这是目标独立的二维降阶假设，不得写成真实三维截面运动。

目标数值轴：

- 空间：`256/512/1024 points/c`；
- 最大时间步：`0.02/0.01/0.005 c/U`，同时保留实际 CFL 限制后的 dt 历史；
- 域：`16/32/64 c`，比较时保持相同 `points/c`；
- 至少运行到相邻周期的压力/剪切/力周期均变化 `<=1%`；
- 正式结果固定使用 `1024 points/c`、`0.005 c/U`、`64c`，不得按目标误差
  选择较好层级。

若最细层因资源失败，状态是 `RESOURCE/PROTOCOL-NO-GO`，不得用粗层结果参与
物理裁决。

## 7. 空间牵引与 co-design 账

observer 必须保存逐时步、逐侧：

- \(p(s,t)\)、\(\tau_w(s,t)\)、surface vorticity；
- \(\tau_w=0\) 的分离/再附着位置和侧别；
- 压力中心；
- finite TE control-volume circulation inventory/flux；
- 源网格、法向、切向、quadrature 权重和不确定度。

通过 `(side, xi)` common-refinement/L2 projection 把牵引映射到
`[0.5,0.6] m` 的双侧材料面板；不得先积分为总力后按面积分配。目标面板只
积分一次，并用现有 `Q=J^T f` 检查虚功。

必须分别闭合：

1. source surface quadrature 与目标面板合力；
2. 关于固定转轴的合矩；
3. panel work 与 generalized work；
4. pressure 与 wall-shear 两个具名账。

结果进入独立 observer ledger，不进入生产 ForceLedger。其余七条带标为
`unobserved`，禁止填零冒充全翼载荷。

## 8. 目标诊断 go/no-go

现有 `claim_raw_out` 不能给出完整 V4.1 局部力：N2 chop、N1 suction 和
vortex-impulse remainder 仍有全局未分配通道。因此本门只允许比较同一 band：

\[
\Delta T_p =
T_{p,\mathrm{NS},[0.5,0.6]}
-T_{p,\mathrm{N1\ Bernoulli},[0.5,0.6]} .
\]

禁止称其为 `NS - full V4.1 strip correction`，也禁止用它直接填补
`2.037758 N` 全翼残差。

`N2.6f2 DIAGNOSTIC-GO` 必须同时满足：

1. 空间、时间、域三轴末级变化分别 `<=5%`；
2. \(\Delta T_p\) 的最不利数值不确定度上界仍 `<0`；
3. 压力造成的 streamwise 差异绝对值至少为 wall-shear 差异的两倍；
4. 负推力差异与具名侧别 \(C_p\) 平台、壁剪切过零/反向和分离区同相出现；
5. source→panel force/moment relative residual `<=1%`；
6. 虚功 relative residual `<=1e-10`；
7. 不读取实验值选择网格、dt、域、黏度、几何、涡核或比例。

反号、落入数值不确定度、仅由摩擦主导或拓扑不一致：
`N2.6f2 -> falsified/frozen`，禁止重跑调参。

即使通过，也只晋升：

> 在 NACA2406-like 代理几何的该条带/工况，V4.1 N1 Bernoulli 缺少可观测的
> 双侧黏性分离压力差。

它只授权在**同一候选、参数不变**的情况下扩展到冻结八条带，形成一个完整
Fig18 U10 shadow。只有八条带一次牵引积分同时改善 thrust 绝对误差、且 lift
不比 V4.1 恶化 `0.15 N`，才开放 U6/U8 三点；随后才是 Fig17/19 代表点和
完整 50 曲线/184 工况。

## 9. 明确禁止

- 不以 LESP/\(A_0\)/\(f_2\) 决定持续涡力；
- 不把 NS 总力作为 V4.1 后加常数；
- 不从 Fig17/18/19 反演压力、黏度、转捩、截面或网格；
- 不把 assumed NACA2406-like 写成 Meng 实测试件截面；
- 不把单截面二维 GO 写成三维生产或全翼精度改善；
- 不在来源验证未通过时运行目标；
- 不在单条带诊断未通过时启动八条带或 184。
