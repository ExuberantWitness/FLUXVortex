# N2.6e1bc moving-interface circulation 守恒推导

日期：2026-07-30  
范围：二维、常密度、moving trailing-edge control area  
裁决：`THREE-TERM GLOBAL-BOUND R_UK FALSIFIED / FULL LOCAL-CV UNOBSERVABLE`

## 1. 为什么重推导

原 shadow 草稿使用

\[
R_{UK}=\dot\Gamma_b+J_{\omega,out}
-\frac{p_L-p_U}{\rho}.
\]

执行前审计要求回答三件事：

1. `dot Gamma_b` 是全局束缚环量还是尾缘局部库存；
2. `J_omega` 是否包含 moving boundary、材料片穿越和黏性扩散；
3. 压力差在什么边界条件下才可替换壁面涡量通量。

以下推导不读取任何目标载荷，也不授权 solver。

## 2. 定向与局部库存

二维涡量

\[
\omega=(\nabla\times u)\cdot k
\]

取逆时针为正。尾缘流体控制域是小邻域扣除实体楔角。令有向边界始终
“流体在左侧”：

\[
W_U:P_U\to TE,\qquad
W_L:TE\to P_L,\qquad
C_o:P_L\to P_U.
\]

控制域外法向为 `n`；物理壁指向流体的法向为 `n_B=-n`。冻结远端压力差

\[
\Delta p_r=p_L(P_L)-p_U(P_U).
\]

局部总环量库存必须包含：

\[
\Gamma_A^{loc}
=\int_{A_f}\omega\,dA
+\sum_m\int_{S_m\cap A}\gamma_m\,ds
+\sum_q\Gamma_q^{atom}.
\]

三项分别是 bulk vorticity、域内材料片涡量和可能的角点原子。它不是全局
bound circulation `Gamma_b`。

若材料片穿过 moving outer boundary，其分布式出流为

\[
J_{\rm sheet}
=\sum_{\rm crossings}
\gamma\,
\frac{(v_s-v_C)\cdot n_C}
{|t_s\cdot n_C|}.
\]

因此不能以普通多边形边中点的有限 `omega` 采样替代片穿越；切触还需要
显式拓扑处理。

## 3. 精确 moving-CV 恒等式

常密度二维 Newtonian 流满足

\[
\partial_t\omega+
\nabla\cdot(\omega u-\nu\nabla\omega)=q_\omega .
\]

Reynolds 输运给出

\[
\frac{d}{dt}\int_A\omega\,dA+
\oint_{\partial A}
\left[\omega(u-v_{\partial A})-\nu\nabla\omega\right]\cdot n\,ds
-Q_A=0,
\]

其中 `Q_A=int_A q_omega dA`。对真实 no-slip、impermeable moving wall，
壁面对流项为零。拆成外边界和实体壁：

\[
R_A=
\dot\Gamma_A^{loc}
+J_o^{adv}+J_o^\nu+J_W^\nu-Q_A=0,
\]

\[
J_o^{adv}
=\int_{C_o}\omega(u-v_C)\cdot n\,ds+J_{\rm sheet},
\]

\[
J_o^\nu=-\nu\int_{C_o}\partial_n\omega\,ds,\qquad
J_W^\nu=\nu\int_W\partial_{n_B}\omega\,ds.
\]

物理壁切向动量给出

\[
\nu\partial_{n_B}\omega
=-\partial_s(p/\rho)+f_t-a_{B,t}.
\]

沿两段有向壁面望远镜化：

\[
J_W^\nu=
-\frac{p_L-p_U}{\rho}
+\frac{p_{TE,L}-p_{TE,U}}{\rho}
+\int_W(f-a_B)\cdot t\,ds.
\]

只有在真实 TE 内端压力单值时，才得到

\[
\boxed{
R_A=
\dot\Gamma_A^{loc}
+J_o^{adv}+J_o^\nu
-\frac{\Delta p_r}{\rho}
+\int_W(f-a_B)\cdot t\,ds
-Q_A=0
}.
\]

全部项单位均为 `m^2/s^2`。

## 4. 原三项式为什么不成立

原式只有在以下条件全部满足并重新命名后才可能成为上述恒等式的特例：

- `dotGamma_b` 实际是局部总库存率 `dotGamma_A_loc`；
- `J` 独立包含 bulk 与 distributional sheet 的 moving-boundary 出流；
- outer viscous diffusion 可忽略；
- 壁面是真实 no-slip 物理壁，不是 transpiration/displacement interface；
- 壁静止且无切向体力，或这些项已显式加入；
- TE 内端压力单值；
- 无 VES 表面质量、动量、entrainment 或 wake pressure jump。

当前 e1bc 状态和代码均不满足这些条件：

- 使用的是全局 bound circulation；
- `TEControlVolume2D` 只是普通闭合多边形，没有实体扣除、局部库存、
  扩散、壁分支或材料片 crossing provider；
- IBL transpiration boundary 不是物理 no-slip wall；
- `delta*/theta/xi` 不能唯一恢复二维 `omega/dn omega/tau`。

所以普通 `"control_volume_quadrature"` provenance 标签不构成物理
moving-CV 实现。

## 5. 压力与 Kelvin 的角色

压力不直接出现在 bulk curl equation；它只在物理壁切向动量替换壁面
黏性通量后出现。同一 trial 的 Bernoulli pressure head 必须使用同一势、
速度、gauge 和移动节点历史：

\[
\frac{p_L-p_U}{\rho}
=-\partial_t(\phi_L-\phi_U)
-\frac12(|u_L|^2-|u_U|^2),
\]

\[
\partial_t\phi
=\frac{d\phi_{\rm node}}{dt}
-v_{\rm node}\cdot\nabla\phi.
\]

若压力参与 Newton，它必须在每个 trial 更新；“统一压力只计算一次”只能
表示收敛后不再追加第二套 pressure/force。

材料 wake 的规范不变 Kelvin 账为

\[
R_K=
\Gamma_b+
\sum_e(\mu_{w,j}-\mu_{w,i})
+\sum_q\Gamma_{v,q}
-\Gamma_{\rm total,0}=0.
\]

若把 `J_omega` 定义为 `Gamma_birth/dt`，Kelvin 时间差分已经给出
`\dot Gamma_b+J_omega=0`；再强加原三项式只会把 `Delta p` 恒等压回零，
不是独立 closure。

Zhu 等引用的

\[
\dot\Gamma=-U\gamma_{TE}+\Delta p_{TE}/\rho
\]

是特定薄翼端点关系；`U gamma_TE` 不能无证明改名为完整 moving-CV 出流。
Xia--Mohseni 的局部关系采用 shrinking inviscid、pressure-continuous、
massless-sheet 极限，也不授权非零 pressure-jump 的黏性 CV。若 wake
维持非零 pressure jump，需要 DeVoria--Mohseni 型 VES 库存。

## 6. 当前可观测性裁决

完整 observer 至少需要：

- 实体扣除的有向域、边界速度和 GCL；
- bulk `omega`、wake `gamma=d_s mu`、角点原子；
- sheet/CV 交点、片速度和交角 Jacobian；
- outer `d_n omega` 或验证过的黏性/湍流闭合；
- physical-wall `d_nB omega`，或 `d_s p/a_B/f_t`；
- same-trial pressure/history/gauge；
- body--wake finite-part / `g-Cphi` compatibility；
- 非零 wake pressure jump 时的 VES 质量、动量和 entrainment。

现有状态不足以唯一提供这些量。因此 full moving-CV 当前只可登记为未来
inner-state observer，不可参与求根、选分支、kill guard 或生产。

## 7. 与 finite-corner 缩放的交叉结论

有界 regular mode

\[
u_g=A r^\beta,\qquad
\gamma_g=B r^\beta,\qquad
\dot r=A r^\beta
\]

给出

\[
\Gamma_{\rm birth}=O(\Delta t^{(1+\beta)/(1-\beta)})
=O(\Delta t^{1.1292006}),
\]

不能一般性匹配 `Delta Gamma_b=O(dt)`。这证明至少缺少一类非均匀状态：
viscous inner layer、受控 singular amplitude、finite forming-zone/VES、
bulk-vorticity inventory 或初始时间层。它不唯一证明其中哪一类正确。

决定性 outer-reference 门已单独冻结在
`n26e1bc0_corner_kelvin_scaling_prereg_20260730.md`。

## 8. Claim 裁决

- `全局 dotGamma_b + 普通 CV flux - Delta p/rho` 作为一般
  moving-CV closure：`falsified/frozen`；
- gauge-invariant Kelvin edge ledger：窄义 formula oracle 保留；
- full local moving-interface circulation theory：机理成立，但当前状态
  不可观测，保持 future/open；
- weak-CV 从当前 solver 候选降级为 observer/future viscous-inner claim；
- 不得用 endpoint gamma、`Gamma_birth/dt`、steady pressure root 或目标力
  回填缺失 primitive。

## 9. 一手来源

- Terrington, Hourigan & Thompson, *JFM* 890 (2020) A5,
  DOI `10.1017/jfm.2020.128`，二维 deforming-interface circulation
  conservation，尤其 Eqs. (2.6), (2.21), (2.22)。
- Morton, *Geophysical & Astrophysical Fluid Dynamics* 28 (1984),
  DOI `10.1080/03091928408230368`，moving-boundary vorticity generation。
- Xia & Mohseni, *JFM* 830 (2017), DOI `10.1017/jfm.2017.513`，
  shrinking inviscid finite-angle mass/momentum/Kutta 极限。
- Zhu et al., *JFM* 893 (2020) R2,
  DOI `10.1017/jfm.2020.254`，特定薄翼 pressure-jump relation 与实验
  适用域。
- DeVoria & Mohseni, *JFM* 866 (2019),
  DOI `10.1017/jfm.2019.134`，带表面质量/动量/entrainment 的 VES。
- Taha & Rezaei, *JFM* 868 (2019),
  DOI `10.1017/jfm.2019.159`，triple-deck 选择非零 viscous TE singular
  amplitude 的薄翼证据。
