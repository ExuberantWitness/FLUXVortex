# N3：finite forming-zone / VES 状态的一手文献裁决

日期：2026-07-28  
作用：S3ah 循环 rank 判据终止后的 Phase ②  
状态：**PRIMARY-SOURCE REVIEW COMPLETE / MODEL SELECTION OPEN**

## ① 当前病因与可动空间

S3ag 证明任意 birth-rate 或 steady pressure residual 小值不能保证
body–wake compatibility。S3ah 又在正式执行前被证明是循环判据：
`rank(G_A)=7` 已由 Morino Schur 满秩前提代数锁定，不能识别物理缺态。

当前只允许回答：

1. 已有兼容 material-wake 路径是否满足未强加的 unsteady pressure/birth
   动力残差；
2. 若不满足，是 forming geometry/出射运动这个组成错误，还是确实缺少具有
   质量、entrainment 和动量记忆的有限形成区状态。

禁止先写自由 \(\zeta\)、把 \(g-C\phi\) 改名为状态，或从 Fig17/18/19 力数据
选择状态维数。

## ② 一手学科机理

### DeVoria–Mohseni 2019：三维 VES 是有守恒内容的状态本体

DeVoria & Mohseni, *Journal of Fluid Mechanics* 866 (2019), 660–688,
DOI `10.1017/jfm.2019.134`。

论文直接在三维曲面上定义：

- sheet 位置 \(\boldsymbol x_s(s,b,t)\)；
- 面质量 \(\rho_s\)；
- 内禀 sheet 速度
  \(\boldsymbol v=\boldsymbol w+v_n\boldsymbol n\)；
- 切向涡片强度
  \(\boldsymbol\gamma=\boldsymbol n\times[\![\boldsymbol u]\!]\)；
- entrainment 强度
  \(q=-\boldsymbol n\cdot[\![\boldsymbol u]\!]\)。

关键方程是：

- 面质量守恒，Eq. (2.6)；
- 面动量守恒并显式包含压差 \([\![p]\!]\)，Eq. (2.7)；
- \(\boldsymbol\gamma\) 与 \(q\) 的定义，Eqs. (2.13)–(2.14)；
- generalized Birkhoff–Rott 外场耦合，Eqs. (2.15)–(2.16)；
- 复合强度
  \(\boldsymbol\alpha=\boldsymbol\gamma+q\boldsymbol n\) 的演化，
  Eq. (2.24)。

sharp-edge 合并条件

\[
\chi_v=\chi_1+\chi_2
\]

是 Eq. (5.4)，论文证明它消除边缘诱导速度奇异；出射角和进入自由 sheet 的
质量/动量边界由 Eqs. (5.6)–(5.7) 决定。

这个来源的关键反事实也很严格：若自由 sheet 的 \(\rho_s=0\)，Eq. (2.6)
给出 \(q=0\)，Eq. (2.7) 给出 \([\![p]\!]=0\)。所以一个不携带
\(\rho_s,q,\boldsymbol v\) 守恒的 residual 修正量不能自称 VES。

### DeVoria–Mohseni 2020：空间位置与有限 sheet 段可计算，但不是完整压力闭合

DeVoria & Mohseni, *Journal of Fluid Mechanics* 903 (2020), A24,
DOI `10.1017/jfm.2020.663`。

论文以

\[
\gamma-iq,\qquad z_s(\Gamma,t),\qquad \Gamma_s(t)
\]

显式表示起动分离 sheet 的强度、位置和总环量；Eq. (1.1) 是 generalized
Birkhoff–Rott 积分，Eq. (2.5) 的复 Kutta 条件同时决定总环量与净
entrainment。附录 B 用有限段端点 \(\omega_k\) 离散卷起 sheet。

它直接支持“空间强度＋位置/运动”，但论文为求自相似外场明确绕过了 2019
VES 的 surface mass/momentum equations，并假定压差足以使 sheet 随两侧
主值速度运动。因此它是有限段可计算性的证据，不能单独授权 FLUXV 的完整
面压力或 VES 动量闭合。

### Xia–Mohseni 2017：先验上更窄的无质量 forming-geometry 分支

Xia & Mohseni, *Journal of Fluid Mechanics* 830 (2017), 439–478,
DOI `10.1017/jfm.2017.513`。

其有限角 sharp-edge 方程联立：

- 形成片方向；
- 强度 \(\gamma_g\)；
- 相对速度 \(u_g\)；
- 环量生成率
  \[
  \dot\Gamma_g=u_g\gamma_g
  =\tfrac12(u_{2-}^2-u_{1+}^2)
  \]
  （Eq. (4.17)）；
- 控制体质量/动量与最终出射角（Eqs. (5.16)–(5.19)、(6.3)）。

这个来源授权无质量 sharp-edge 极限中的 forming geometry、强度和运动联合
求解。它没有授权带 \(\rho_s,q,[\![p]\!]\) 的有限质量 forming zone；后者
必须回到 DeVoria–Mohseni 2019。

## ③ 缺件/错件的分层判定

| 候选 | 允许的物理含义 | 当前证据状态 |
|---|---|---|
| \(H_G\)：无质量 forming geometry | 新出生 sheet 的空间位置、出射方向、\(\gamma_g,u_g,\dot\Gamma_g\) 联立；\([\![p]\!]=0\) sharp limit | 文献授权，尚未通过 actual reachable-path 门 |
| \(H_V\)：finite VES | \(\boldsymbol x_s,\rho_s,\boldsymbol v,\boldsymbol\gamma,q\) 由面质量/动量及外场共同演化，可支持压差 | 文献授权为物理模型，尚未证明 FLUXV 必需 |
| 自由 P2 \(\zeta\) 或 residual slack | 无独立质量、动量、几何或材料记忆 | 禁止 |
| 只引入涡粒子位置与强度 | 可表示卷起外场，但若无 birth/entrainment/压力守恒，不能闭合形成区 | 不充分 |

模型选择顺序固定：

1. 先沿已验证的 actual compatible material-history 路径观察未强加
   pressure/birth 残差；
2. 再测物理 forming-geometry 变量对该残差的 transversality；
3. 只有 geometry 切向像存在稳定非零 cokernel，且 VES 质量/动量离散能
   覆盖该 cokernel 并通过零质量退化极限时，才允许进入 \(H_V\)。

## ④ 下一方案的预注册边界

第一门只做 observation：

\[
R_P^n
=M_a(g^{n+1}-g^n)+\Delta t\,P^{n+1/2},
\]

其中 \(g^n,g^{n+1/2},g^{n+1}\) 必须来自前一时刻 BIE、compatibility、
material transport 与 actual geometry stage 都通过的同一路径。禁止制造
previous trace。

若该残差不随 \(\Delta t,h,p\) 收敛，第二门才比较：

\[
T_G
=
\frac{dR_P}{d\xi_G}
\]

在 Xia 型可达 forming-geometry/strength/relative-speed 切空间上的像与
cokernel。只有 \(T_G\) 的稳定 cokernel 不能由无质量形成变量覆盖，才研究
VES 离散：

\[
\left(
\boldsymbol x_s,\rho_s,\boldsymbol v,
\boldsymbol\gamma,q
\right)_h,
\]

并逐项保留 Eq. (2.6)、Eq. (2.7)、Eq. (2.24) 的守恒来源。

本文件不授权压力成力、LESP、生产 closure、118 工况或 Fig17/18/19。

