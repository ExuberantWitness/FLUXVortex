# N3 S3ah：正式执行前的 Schur-rank 逻辑审计

日期：2026-07-28 12:44:37 +08:00  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3e3`  
裁决：**ABORTED BEFORE FORMAL EXECUTION / CIRCULAR-RANK-NO-GO**

## ① 病因定位

S3ah 原预登记拟从相容参考态计算

\[
G_A=L_M^\mathsf T
\left(I-C\Phi_g\right)
J_P^{-1}M_aQ,
\qquad
\Phi_g=-B^{-1}W,
\qquad
Q=L_M^{-\mathsf T},
\]

并把 `rank(G_A)=7` 解释为“七维分布式 forming state 的最低代数维数”。
正式数值执行前的独立公式审计发现，这个 rank 在预登记的其他通过条件下已被
代数恒等地确定，因而不是一个开放的物理假设。

令

\[
H=I-C\Phi_g=I+CB^{-1}W.
\]

S3af 已冻结：

- \(B\) 非奇异；
- Morino block
  \[
  K_M=
  \begin{bmatrix}
  B&W\\
  -C&I
  \end{bmatrix}
  \]
  为满秩 \(88\times88\) 方阵。

由 Schur 补，

\[
\det K_M=\det(B)\det(H),
\]

所以 \(H\) 必为满秩 \(r=7\)。S3ah 又预先要求 \(J_P\) 满秩、\(M_a\) 正定，
而 \(L_M^\mathsf T\)、\(M_a\)、\(Q\) 均可逆。因此

\[
\operatorname{rank}(G_A)
=\operatorname{rank}(H)
=7.
\]

这与 pressure closure、\(\Delta t\) 及 history basis 无关。原 YAML 同时把
`compatibility_tangent_rank_min: 7` 写进总通过阈值，使结果在执行前已被前提
锁定。

## ② 学科机理与定义审计

原 closure 的符号本身是自洽的：

\[
F_P=M_a(C\phi-c^-)+\Delta t\,P(\phi,g)=0.
\]

但

\[
c^-_0=c_0+\Delta t M_a^{-1}P_0
\]

只是一个**代数上使 closure 为零的制造参考历史**。它尚未被证明来自前一时刻
同时满足 BIE、compatibility、Kelvin 与 material transport 的可达状态。七个
任意 \(c^-\) P2 扰动也没有被证明属于物理可达 history tangent。

质量度量的正确 Cholesky 写法是

\[
M_a=L_ML_M^\mathsf T,\qquad
Q=L_M^{-\mathsf T},\qquad
\|A\|_{M_a}=\|L_M^\mathsf T A\|_2.
\]

原 Markdown 使用这一写法；原 YAML 的 `M_a^(1/2)` 表述不够精确，但这不是
本次 NO-GO 的主因。

## ③ 缺件还是错件

| 命题/组成 | 裁决 |
|---|---|
| 用任意 previous-trace 输入下的 `rank(G_A)` 发现物理缺态维数 | **错判据，falsified** |
| `rank(G_A)=7` 的数值实现一致性 | 可作代码回归，但不是新物理证据 |
| 一个固定、可分离、rank-one 修正覆盖任意七维输入 | 代数上不够，但不能外推证伪所有“一个物理标量”模型 |
| 制造的 \(c^-_0\) | closure-consistent，但 dynamically unverified |
| 真正可判别的对象 | 兼容流形上物理可达历史/几何扰动对未强加动力残差的 transversality/cokernel |

因此本轮不能授权 \(\zeta\)、VES、七维 P2 inventory、pressure force 或生产
closure。也不能把本来必为 7 的 rank 数值重新包装为“实验发现”。

## ④ 下一方案与 go/no-go

下一门必须先构造连续的、物理可达的相邻状态：

\[
R_B(x,p)=0,\qquad A(x,p)=0,
\]

其中参数路径 \(p(t)\) 来自已通过的 body motion、material history、
Kelvin/transport 与 actual geometry stage，而不是任意注入 \(c^-\)。
随后只观察未强加的 pressure/birth 动力残差

\[
R_D\!\left(x^{n+1},x^n,p^{n+1},p^n\right).
\]

需要检验：

1. \(R_D\) 是否随 \(\Delta t,h,p\) 收敛到零；
2. 若不收敛，先按 pressure observation、finite-base topology、
   forming geometry 或 material transport 定位组成错误；
3. 只有在物理可达切空间上，现有变量对 \(R_D\) 的切向像存在稳定非零
   cokernel，且有限 forming-zone/VES 守恒律提供相应的新状态、记忆和耦合
   后，才能判为缺状态；
4. 禁止 least squares、删除 body rows、把 residual 改名为 state、用
   Fig17/18/19 力数据选择秩或状态。

S3ah 原正式数值执行因此取消。冻结的
`actual_wake_unsteady_pressure_tangent_cases_20260728_123758.yaml` 保留为
审计证据，不回写“通过”结果；latest 文件只记录本次 pre-execution abort。

