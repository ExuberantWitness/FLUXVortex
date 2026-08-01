# N2.6f1 来源门指标与时间轴协议修订

日期：2026-07-30  
状态：`PREREGISTERED AMENDMENT / SOURCE RESPONSE NOT YET SCORED`  
适用节点：`N2.6f1`

## 1. 修订触发与边界

本修订发生在 formal moving-NACA0015 运行尚未结束、尚未计算 CL/CD 参考误差、
尚未打开任何 RoboEagle 目标工况时。触发因素只有两个协议事实：

1. 原预登记没有唯一规定离散参考点与 CFD 密集曲线之间的 RMSE/L2 积分；
2. Basilisk 的 `DT` 是步长上限，formal 运行中可见实际 `dt` 远小于
   `0.005`，故原 `{0.02,0.01,0.005}` 可能不是有效时间扰动轴。

本修订不改变候选、方程、参考数据、网格轴、误差阈值或 go/no-go；只消除
指标自由度，并预先规定时间轴退化时的处理。禁止根据稍后的 CL/CD 表现再次
选择积分、平滑、角度域或时间层级。

唯一实现 scorer：

`platform/n26f_source_gate.py`

## 2. 日志身份与完成性

official `log` 的每个非空行必须恰好有 16 个有限数值。固定校验：

- iteration 从 `0` 开始逐行加一；
- multigrid 整数列为非负整数；
- normalized time 严格递增，`dt>0`；
- 外层运行 receipt 的 return code 必须为 `0`；
- 末行满足
  `tau_last >= 1.85 - max(dt_last,5e-6)`；
- 删除全部 `tau<0` 后，才能处理预俯仰的 `theta=0` 重复点；
- post-start 打印值完全相同的角度组取 CL/CD 算术平均，之后角度必须严格
  递增；
- 完整覆盖 CL/CD 各自的全部参考支持域；
- 必须存在非空的 `cp-angle-44-pid-*`、`cp-angle-54-pid-*`、
  `omega-zoom-angle-44.png`、`omega-zoom-angle-54.png`。

第二个输出文件名按来源代码的整数截断确为 `54`，不得后验改名为 `55`。

## 3. 来源响应指标

分别保留两条数字化曲线的完整支持域：

| field | support (deg) | points | reference range | peak angle (deg) |
|---|---:|---:|---:|---:|
| CD | `[0.4730174926739976,54.98539927034809]` | 53 | `3.7916000008521866` | `54.98539927034809` |
| CL | `[0.020766621629590087,54.500844765657654]` | 57 | `3.908274940826322` | `46.1180518345131` |

模型和参考都定义为无平滑的分段线性曲线。对两者节点和域端点的并集，
若一个小区间两端误差为 \(e_0,e_1\)，精确积分：

\[
I_e=\sum_i\frac{\Delta\theta_i}{3}
(e_{0,i}^2+e_{0,i}e_{1,i}+e_{1,i}^2).
\]

range-normalized RMSE 定义为：

\[
NRMSE_{\rm range}=
\frac{\sqrt{I_e/(\theta_{\max}-\theta_{\min})}}
{\max y_{\rm ref}-\min y_{\rm ref}}.
\]

CL、CD 分别要求 `<=10%`。峰值只从各自分段线性节点和域端点选取；相同
峰值取最小角度，不允许平滑、样条或抛物线拟合。模型与参考峰值角差分别
要求 `<=3 deg`。

## 4. 空间与时间 Cauchy 指标

coarse/fine 曲线在上述各自固定域，以两条曲线节点并集精确积分：

\[
L_{2,\rm rel}=
\sqrt{\frac{\int(y_c-y_f)^2\,d\theta}{\int y_f^2\,d\theta}},
\qquad
r_{\rm peak}=
\frac{|\max y_c-\max y_f|}{|\max y_f|}.
\]

分母为 fine；零分母直接失败。CL、CD 的两个指标均须 `<=3%`。

- 空间最后两级固定为 `256 vs 512 points/c`，`DT=0.01`；
- 原时间最后两级仍固定为 `DT=0.01 vs 0.005`，`256 points/c`；
- 峰值角漂移只报告，不新增事后门。

## 5. 时间轴退化与一次性替代轴

原 `{0.02,0.01,0.005}` 三档仍全部执行并保留，不因当前 actual `dt` 较小
而省略。派生诊断源码必须另外记录不含 `DT` cap 的：

\[
d_{\rm CFL}=CFL\min_{\rm faces}\frac{\Delta fm}{|u_f|},\quad
d_{\rm embed}=\min_{\rm cut\ cells}\frac{\Delta}{|u_b\cdot n|},
\quad d_{\rm phys}=\min(d_{\rm CFL},d_{\rm embed}).
\]

这里采用来源代码的 `SEPS=0` 极限，不新增 epsilon。若某 fragment 的
`|u_b·n|=0`，只在诊断文件中记为有限哨兵 `1e30`，表示该约束不活跃；
该值不进入推进、力或任何可调公式。

若原最后两档任一满足以下任一条件，则记为
`AXIS-DEGENERATE / PROTOCOL-AMEND`，不得将曲线相同解释为时间收敛：

- `mean(DTcap <= d_phys) < 0.5`；
- 两档 median actual `dt` 比 `<1.5`。

此状态不证伪 N2.6f1。此时在读取任何 CL/CD、参考曲线或目标力之前，只按
`lmax=14, DT=0.01` 的 post-start `d_phys` 计算全精度中位数 `d50`，并一次
冻结新 cap：

\[
\{d50,\ d50/2,\ d50/4\}.
\]

不允许根据力曲线再选 cap。最后两档固定为 `d50/2 vs d50/4`，且二者均须：

- `mean(DTcap <= d_phys) >=0.5`；
- `median(dt_{medium})/median(dt_{fine}) >=1.5`；
- CL/CD 的 \(L_{2,\rm rel}\) 和 \(r_{\rm peak}\) 均 `<=3%`。

否则为 `PROTOCOL-NO-GO`，不进入目标工况。

## 6. 最小只读 instrumentation

official formal source保持逐字不改。数值轴使用注明 parent SHA 和白名单
diff 的派生副本，只读相同流场，额外导出：

- 每步 `Fp_x,Fp_y,Fmu_x,Fmu_y,d_CFL,d_embed,d_phys`；
- 在既有 44/约55 度事件，逐 cut-cell fragment 输出
  `pid,xe,ye,nx,ny,ds,p_surface,mua,dudn_x,dudn_y`,
  `dFp_x,dFp_y,dFmu_x,dFmu_y,tau_w,sigma_mu_n`；
- 浮点格式固定为 `%.17g`。

为避免改变 Basilisk 同名事件链次序，派生副本不得新增 event：

- 在用户已有 `advection_term` 更新 `p_w/p_aw` 之前只读计算
  `d_CFL/d_embed/d_phys`；
- 在已有 `logfile` 复用其唯一一次 `embed_force()` 结果写 step sidecar；
- 在已有 `surface_profile` 的 44/54 度事件追加 fragment sidecar。

instrumented 与 official run 的原 16 列日志必须具有相同行数及完全相同的
`i/t/dt`；CL/CD/Fp/Fmu 只能有机器舍入量级差异。

离线按所有 PID 汇总。Basilisk 的法向身份必须记录为 `fluid -> solid`；
映射到材料外法向时显式反号。完整黏性力不能偷换成只有切向的 wall shear。

冻结几何闭合：

\[
\epsilon_\Gamma=
\frac{\|\sum n\,ds\|_2}{\sum ds}\le10^{-8},
\quad ds>0,\quad ds\ {\rm finite}.
\]

冻结一次牵引积分闭合：

\[
r_F=
\frac{\|\sum(dF_p+dF_\mu)-(F_p+F_\mu)_{\rm embed}\|_2}
{\max(\|(F_p+F_\mu)_{\rm embed}\|_2,10^{-14})}\le1\%.
\]

44 和约55 度均必须通过；压力、完整黏性、切向剪切和法向黏性余量分别报告。
该 instrumentation 只验证 exporter/traction 账，不改变推进、AMR 或 force。
