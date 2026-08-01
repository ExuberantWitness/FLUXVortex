# S3ai-v1 可达压力阻碍门执行前审计（2026-07-28 13:03 +08:00）

## 裁决

**PROTOCOL-AUDIT-NO-GO / ABORTED BEFORE FORMAL EXECUTION**

时间戳冻结预登记
`actual_wake_reachable_pressure_obstruction_cases_20260728_125228.yaml`
保持原样；未调用正式 guard，未生成结果 JSON，未形成 pressure、force、
forming-state、118 工况或 Fig17/18/19 结论。

开放 claim
`N3.1j3b6d18c2b3b3b2c2b2b3e3b`
仍为 open。否决对象只是 v1 证据协议，不是具名物理 pressure law。

## ① 数据指纹与病因

实现后、执行前的独立审计发现五个阻断问题：

1. `direct full-BIE` 先用同一 `B/W/g/b` 重解 `phi`，再检查该解自己的
   residual，不能独立验证 S3e 存储的 `global_body_potential`。
2. Kelvin 账把
   `(g_mid-g_previous)+(g_current-g_mid)` 直接望远镜求和，因此恒等为零；
   它没有从 material bands 独立计算 wake circulation，也没有反号负对照。
3. q5/q8/q10/q12 body family 全部用 q10 direct-W，导致 direct-W equality
   与 spatial quadrature 收敛混账。
4. pressure observation 直接使用 body-cut trace，没有从 newest material
   band 通过显式 attachment permutation/sign 独立提取并核对 material trace。
5. 对称零解要求 quadrature residual “严格下降”；精确零或浮点平台期会被
   错判为 NO-GO，且严格单调本身不是可靠的空间收敛证据。

另外，v1 将 BIE、compatibility、Kelvin residual 映射进 pressure
dual-mass uncertainty，但没有定义映射。量纲审计进一步证明这些量不在同一
残差空间，不能进入同一个 `max`。

## ② 机理与离散约束

冻结的弱式观察量为

\[
R_{P,n}=M_a(g_{n+1}-g_n)+\Delta t\,P_m .
\]

若 \(g\) 是 doublet potential jump，则
\(M_a[g]=L^3/T\)，\(\Delta t P_m=L^3/T\)，且

\[
\|R_P\|_{M_a^{-1}}\sim L^{5/2}/T .
\]

中心可达切向

\[
\Omega=\frac{R_W(+\varepsilon)-R_W(-\varepsilon)}
{2\varepsilon}
\]

也在该空间。BIE、compatibility 与 circulation/Kelvin residual 不同量纲，
必须作为各自尺度化 backward-error hard guards，而不是 pressure uncertainty
的组成部分。

## ③ 缺件还是错件

本次只定位为 **证据协议错件**：

- 尚无数据说明 massless forming geometry 缺失或 finite VES 必需；
- 尚无数据证伪具名 weak unsteady-pressure law；
- 更不允许由七个离散 residual 分量宣布“缺七维状态”。

因此不能跳到自由 \(\zeta\)、VES inventory、pressure force 或生产闭合。

## ④ v2 必须满足的有证据改写

1. 从 material history/attachment 独立提取完整九节点 trace，显式检查
   permutation、sign、active 七节点、zero tips 及与 body-cut trace 的一致性。
2. 用存储的 S3e `phi` 检查
   \(B\phi_{\mathrm{S3e}}+W_{\mathrm{direct}}g-b\)，并另报
   \(\|\phi_{\mathrm{direct}}-\phi_{\mathrm{S3e}}\|\)；禁止重解自证。
3. direct-W 与 body quadrature 同阶，或把 q10 equality audit 与 spatial
   family 完全分离。
4. Kelvin 必须从 material bands 的 signed wake functional 与独立 bound
   functional 计算，逐半步检查，并要求错误 birth sign 的负对照失败。
5. 所有误差族共享 finest anchor
   \((\varepsilon_f,\Delta t_f,q_f)=(0.0025,0.0625,12)\)；
   epsilon/dt 用二阶 Richardson，quadrature 用收缩尾界，repeat/algebraic
   refinement 只在同一 anchor 比较。
6. 构造同量纲 \(U_\Omega\)，以
   \[
   [L_\Omega,H_\Omega]=
   [\max(0,\|\Omega_f\|-U_\Omega),\ \|\Omega_f\|+U_\Omega]
   \]
   判定。删除无机理锚的 `100x/10x` 阈值。
7. epsilon=0 必须跑完整窗口
   \(Z_0=\sum_nR_{P,n}(0)\)，同时报告
   \(V_0=\sum_n\|R_{P,n}(0)\|_{M^{-1}}\) 防止正负抵消；其独立
   \(U_0\) 不得从 \(\Omega\) 中作 offset。
8. 增加真实 S3e step integration regression、material-trace/attachment
   负对照、Kelvin 反号负对照与 initial compatible stage。

在 v2 再次于任何正式执行前冻结且独立审计通过之前，S3ai 不产生物理裁决。
