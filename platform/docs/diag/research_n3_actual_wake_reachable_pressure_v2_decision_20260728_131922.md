# N3 S3ai-v2：可达压力残差门的角色修正与冻结定义

时间：2026-07-28 13:19:22 +08:00  
节点：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
状态：**定义冻结，尚未正式执行；不授权 force/production**

## ① 病因：数据指纹、树节点与可动空间

S3ai-v1 没有产生数值物理结果。它在执行前被否决，原因不是“精度
不够”，而是观察器定义自证：

1. pressure observer 用 `W_direct` 重新求解体面势，再用同一个解检查
   BIE，残差必然接近零；
2. 所谓 Kelvin ledger 直接累加
   `(g_mid-g_prev)+(g_cur-g_mid)`，其闭合是望远镜恒等式；
3. body 与 direct-W 使用不同求积阶，不能把差异归为物理动力残差；
4. pressure dual-mass residual 与 BIE/Kelvin residual 量纲不同，却被
   混入同一 uncertainty；
5. S3e 的外部初始零 `g` 没有对应的 stored `phi`，不能冒充 compatible
   initial state。

这把病灶挂到开放节点
`N3.1j3b6d18c2b3b3b2c2b2b3e3b`：未知的是“具名 unsteady-pressure
law 在 S3e 可达路径上是否横截”，不是 N1/N4，也不是 force
聚合。可动空间只允许增加只读观察器、误差球和负对照；冻结 S3e
方程、材料历史、几何、时间积分与 pressure 公式均不可修改。

## ② 学科机理：一手来源给出的方程角色

- Dumoulin、Eldredge 与 Chatelain（JFM 977, 2023,
  DOI `10.1017/jfm.2023.997`）把 no-through-flow、Kelvin circulation
  transfer 与 unsteady Kutta 分为不同方程。由此可知：材料附件的拓扑
  恒等式不能再被计作第二个独立物理 closure。
- Xia 与 Mohseni（JFM 830, 2017,
  DOI `10.1017/jfm.2017.513`）的高 Reynolds 数尖缘模型把两侧速度、
  环量变化率、相对速度和形成方向联立。可达 pressure residual 即使
  非零，也不能直接改名为自由“涡状态”。
- Ramesh 等（arXiv:2205.08647）说明 unsteady Kutta 不只要求有限速度
  或平滑离开，还要求尾缘两侧非定常压力/涡量释放相容。因此本门必须
  显式观察
  `M(g_{n+1}-g_n)+dt*P_mid`，不能只看几何或稳态速度。
- DeVoria 与 Mohseni（JFM 866, 2019,
  DOI `10.1017/jfm.2019.134`）的有限 VES 具有独立的面质量、动量与
  entrainment 库存。没有这些守恒证据，pressure residual 不能授权
  finite-VES 状态。

## ③ 方向裁决：错件与缺件

本轮首先确认两个**观察器错件**，而不是先宣布缺状态：

1. **stored-state pressure 错件**：主 pressure 必须直接使用 S3e 已存
   `global_body_potential`。独立 direct solve 只用于交叉审计，绝不回写
   material trace/history。
2. **Kelvin 角色错件**：当前 S3e 没有独立跟踪随流闭合曲线，因此不能
   构造真正的全局 Kelvin functional。可实现的是从
   `band.surface.face_mu` 独立恢复两条材料边界，形成完整九节点的
   attachment/orientation inventory。它是表示硬门，不是物理 closure。

所以 S3ai-v2 的裁决边界是：

- 只判定固定 81-DOF diamond 空间中，具名 pressure law 的零阶或一阶
  可达 obstruction；
- 即使 obstruction 非零，也必须先补 actual-body 空间 `h/p` 收敛门；
- 空间见证复现后，先测试 Xia–Mohseni 无质量 forming geometry；
- 只有该参数自由几何组成仍不足，且独立质量/动量/entrainment 库存成立，
  才允许进入 finite VES。

## ④ 方案：有证据的树改写与预登记

冻结预登记：
`actual_wake_reachable_pressure_obstruction_cases_20260728_131922.yaml`。

核心定义如下。

### compatible pre-step

每条路径用未修改的 S3e 从 `[-dt,1]` 运行。`[-dt,0]` 内
`alpha=0`，step-0 full stage 生成 t=0 的 stored `phi`、零强度 band
和 material history。测量只取 `[0,1]`；外部 `g(-dt)=0` 永不进入
pressure window。

### surface-only material inventory

对每条 band，只从 `surface.face_mu` 的 P2 boundary edge coefficients
恢复

\[
r_k=\mu_{k,\mathrm{current}}-\mu_{k,\mathrm{previous}},\qquad
R_w^{mat}=\sum_k r_k.
\]

显式 attachment 为 \(w=sPc\)，所以
\(R_w^b=sP^\mathsf T R_w^{mat}\)。存储体 trace 必须重新由
\(c=J\phi_s\) 计算。表示库存

\[
I_\beta=-c+\beta R_w^b
\]

只检查 previous-full→half→full 的完整九节点增量。正确
\(\beta=+1\) 必须闭合；错误 \(\beta=-1\) 必须在非零 stage 排除零。
它不叫“独立 Kelvin 方程”。

### stored-pressure residual

\[
R_n=M_a(g_{n+1}^{mat}-g_n^{mat})
       +\Delta t\,P(\phi_{s,n+1/2}),\qquad
\Omega=\frac{R_W(+\varepsilon)-R_W(-\varepsilon)}{2\varepsilon}.
\]

`g` 来自 surface material trace；`phi_s` 是 stored stage state。
独立解

\[
\phi_d=B^{-1}(b-W_{direct}g)
\]

仅形成 BIE/compatibility 审计和同 residual-space 的 pressure observer
差 \(u_{alg}\)。material derivative 项保持不变，且 `phi_d` 不传播到
下一步。

### 同空间误差球

共同锚点为
\((\varepsilon,\Delta t,q)=(0.0025,0.0625,12)\)。

- epsilon：`[0.01,0.005,0.0025]`，中心差二阶 Richardson；
- timestep：`[0.25,0.125,0.0625]`，midpoint 二阶 Richardson；
- body/direct-W quadrature：`[8,10,12]`，只在收缩时使用几何 tail；
- 相邻 \(2\times2\times2\) cube 完整测量三组二阶和一组三阶
  Möbius mixed difference；
- fresh repeat 与 stored/direct-pressure cross-observer 分别形成
  \(u_{repeat}\) 和 \(u_{alg}\)；
- `Z0=sum R_n(0)` 与
  `V0=sum ||R_n(0)||` 有独立误差球，绝不作为 tangent offset。

最终

\[
U_\Omega=u_\varepsilon+u_{\Delta t}+u_q+u_{mix}
         +u_{repeat}+u_{alg}+u_{round},
\]

\[
L_\Omega=\max(0,\|\Omega_A\|_{M_a^{-1}}-U_\Omega).
\]

`LΩ>0` 只给 fixed-space reachable witness；`LΩ=0` 只写
`NO RESOLVED WITNESS`，不得写“零态 closure validated”。

### 执行前负对照

正式 runner 前必须证明：

- 单 DOF 污染 stored `phi` 会让 stored-BIE/pressure 失败，而独立 direct
  solve 仍正常；
- wrong attachment sign 与 wrong birth sign 均被非零 stage 检出；
- 非对称九节点 trace 的 reverse/sign round-trip 只在精确 typed map 下
  保持不变；
- 修改 row cache 不改变 face_mu functional，但 row/surface guard 失败；
- 改 surface boundary `face_mu` 会改变 functional 或 fail closed；
- 局部反对称 defect 即使展向积分为零，完整 trace 门仍失败。

## Go / No-Go

- 任一表示、负对照、BIE、对称性或收敛门失败：
  `PROTOCOL-NO-GO`；
- zero reference 的误差区间排除零：固定空间
  `ZEROTH-ORDER NAMED-LAW OBSTRUCTION`；
- zero reference 通过且 `LΩ>0`：
  `FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS`；
- 全门通过但 `LΩ=0`：`NO RESOLVED WITNESS`。

本文件和冻结 YAML 都不产生 force、production、118 或
Fig17/18/19 结果。
