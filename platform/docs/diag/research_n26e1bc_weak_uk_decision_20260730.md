# N2.6e1bc：强相互作用—弱式非定常 Kutta 联立裁决

日期：2026-07-30  
父命题：`N2.6e1`  
状态：`REFRAMED / THREE-TERM WUK NO-GO / INNER-STATE QUESTION OPEN`

> 2026-07-30 执行前独立审计保留“强相互作用 + 材料势跃 + 弱式
> unsteady-Kutta”的研究方向，但判定当前方程与预登记尚不可执行。
> 本文的机理裁决不等于实现授权；后继必须先通过
> `n26e1bc_prereg_independent_audit_20260730.md` 列出的修复门。
>
> 随后的严格 moving-interface circulation 推导进一步证伪了
> `global dotGamma_b + ordinary J_omega - Delta p/rho` 作为一般
> solver row：正确库存必须是尾缘局部 bulk+sheet+atom circulation，
> 并显式包含 moving-boundary sheet crossing、outer diffusion、物理壁/
> interface 项。详见
> `research_n26e1bc_moving_cv_derivation_20260730.md`。因此 weak-CV
> 已降级为 future viscous-inner observer；当前只先执行
> `N2.6e1bc0` 的 regular-corner/Kelvin 缩放判别。

## 1. 病因指纹、claim 节点和可动空间

`N2.6e1b1` 在闭合 NACA0015 的余弦网格上得到：

| `128 -> 256 panels/side` | 变化 |
|---|---:|
| lower TE trace | 7.33% |
| upper TE trace | 9.16% |
| mean TE trace | 8.20% |
| trace jump / newborn sheet strength | 9.36% |
| newborn integrated circulation | 0.38% frozen physical-scale score; 1.92% direct relative change |

这不是“网格还不够细”。闭合 NACA0015 的尾缘实体夹角为

\[
\tau=2\tan^{-1}(0.1816875)=20.5952^\circ .
\]

有限角 Laplace corner 在消去首个 Kutta 奇异模态后，下一速度模态为

\[
u_\tau(r)\sim A r^\beta,\qquad
\beta=\frac{2\pi}{2\pi-\tau}-1=0.0606803 .
\]

余弦网格加倍使最近控制点距离精确缩小约四倍，因此理论预言

\[
\frac{|u(h)-u(h/4)|}{|u(h/4)|}
=4^\beta-1=8.776\%.
\]

实测 lower/upper/mean 的区间幂次分别为
`0.0511/0.0632/0.0569`，与几何给出的 `0.06068` 同阶；mean 的
`8.20%` 也与 `8.776%` 直接对应。与此同时，Kelvin 配对的出生环量已经
收敛而 point trace 没有收敛。这把病灶唯一定位到：

> `N2.6e1b` 把有限角角点的幂律迹和点值 sheet strength 当成了可独立
> 时间推进的出生状态。

`N2.6e1b2` 又表明，把这一点值换成裁切的 Xia--Mohseni 端点节点并不能
解决问题：积分环量进入约 1% 区间，而两个端点仍变化
`8.57%--108.97%`，且 current state 越出 no-backflow 适用域。

可动空间仅为尚未冻结的 `N2.6e1b/e1c` **组成关系**。N1、N4、
`N2.6e1a`、V4.1、旧 LESP 力、最近控制点、epsilon 裁切端点和已证伪
pressure-residual-only 路线均不可改。

## 2. 一手机理

### 2.1 Riziotis--Voutsinas：outer、IBL 和 wake 必须同时求解

Riziotis & Voutsinas (2008), DOI `10.1002/fld.1525`，把双侧 IBL 的
质量亏损通过

\[
u_{e,n}|_{\rm wall}
=\rho_e^{-1}\partial_s(\rho_e u_e\delta^*)
\]

反馈到同一个实际翼面外流，并明确把 outer、IBL、转捩/应力方程组成
Newton--Raphson 强相互作用系统。分离位置也只在边界层收敛后更新。

但其离散出生量仍取最近控制点的 `mean/jump`。因此该来源授权
“同一步强耦合”的拓扑，不证明这个点值离散具有网格极限。

### 2.2 有限角空间表示

S. Y. Sun & G. X. Wu (2022),
DOI `10.1016/j.enganabound.2021.12.012`，在**稳态** lifting-body
HOBEM 中显式使用由实体角决定的非整数幂局部基。它支持“角点幂次必须进入
空间表示”，但不提供非定常 forming-sheet 出生律。

Xia & Mohseni (2017), DOI `10.1017/jfm.2017.513`，先从收缩控制体的
质量、环量和动量通量出发，再在额外符号域内退化到端点关系。其一手算法
要求 actual previous state，且 point strengths 在 junction 本身并非
普遍良定义。因此可保留弱控制体守恒，不能继续消费独立端点点值。

### 2.3 黏性尾缘不是零压差 Kutta

Taha & Rezaei (2019), DOI `10.1017/jfm.2019.159`，用 triple-deck
匹配决定一个非零尾缘奇异幅值；这证明非定常环量生成是由黏性尾缘内区
选择，而不是纯势流端点自动给出。

Zhu et al. (2020), DOI `10.1017/jfm.2020.254`，使用广义关系

\[
\dot\Gamma=-U\gamma_{TE}+\frac{\Delta p_{TE}}{\rho}
\]

并实验观察到尾缘停滞迹线随相位运动。该工作授权非零
`\Delta p_TE` 的物理可能性，但明确没有给出闭合算法。

仓内 `N3...b3e2` 已证伪“weak/collocation `Delta p=0` 的 Newton
残差足够小就选为物理解”。所以新候选必须从一致上一时刻连续延拓，并把
压力、IBL 和弱通量作为交叉守卫，不能再做稳态 pressure-root 选根。

## 3. 缺件还是错件

裁决为两层：

1. **组成关系错**：把 `e1b` 无黏出生 provider 先单独验证、再把结果交给
   `e1c` 的顺序不受来源机理支持。强相互作用的尾缘状态必须与 IBL、
   transpiration 和压力历史同一步联立。
2. **组成部分缺**：现有代码没有以积分 wake potential jump、弱式尾缘
   涡量通量和有限角形成几何为状态的联立 DAE；也没有尾缘横向动量的独立
   kill guard。

因此不再开第三个 endpoint 算子，也不把已收敛总环量直接冒充完整
forming state。

## 4. 唯一候选

唯一候选命名为：

`N2.6e1bc-SVI-WUK-WPJ`

> 在二维尖尾缘、附着或首次分离前的 actual-surface shadow 中，以双侧
> IBL、材料 wake potential jump 和形成几何为同一步状态；以完整 body
> BIE、transpiration、IBL 动量/能量、Kelvin 和弱式 unsteady-Kutta
> 控制体残差联立，压力只由收敛后的同一总势/总速度通过一次非定常
> Bernoulli 得到。

最小状态为

\[
z=\{\phi_b^\pm,\delta^{*\pm},\theta^\pm,\xi^\pm,
\mu_{w,k},X_{w,k},\theta_{w,0},\phi_b^{n-1}\},
\]

其中 `xi` 是来源规定的互斥 `n/C_tau` 记忆。新生积分环量由 Kelvin
决定；有限角方向由弱式动力残差的参考连通分支决定。旧 material wake
严格满足 `D mu_w/Dt=0`。

第一阶段只含一条 TE wake，不引入分离第二尾迹、`Delta h`、LEV、VES、
目标载荷或 Taha `B_v`。Taha 模型只作薄翼小扰动极限 oracle，不能与
full IBL 重复记账。

该候选是 Riziotis 强相互作用、Sun--Wu 角点空间表示、Xia 弱控制体和
Zhu 广义非定常 Kutta 的**新综合**；不存在一篇文献已证明其充分性。
因此必须以强证伪 shadow 处理，不能写成“来源方法已经验证”。

## 5. 明确禁止

- 不再增加最近控制点网格或选择 epsilon；
- 不把 `Delta p_TE` 直接按比例变成 `mu_w`；
- 不用 steady `Delta p=0` residual 单独选根；
- 不同时强加 Morino wake closure 和另一套同维 Kutta closure；
- 不删除满秩 body BIE 行，不用 least-squares、core、clamp 或阻尼补闭合；
- 不从 Fig12、Fig17/18/19、总力或 V4.1 残差选择状态、方向或尺度；
- 不输出独立 vortex-force/impulse production 力；
- 不在该 attached gate 中加入第二尾迹或 VES。

下一文件 `n26e1bc_strong_vi_weak_uk_prereg_20260730.md` 冻结最小实现和
go/no-go。只有该门通过，才允许恢复 `N2.6e1d` 的分离第二尾迹与
Figure 12；失败则该具体综合 `falsified/frozen`，不得追加修补方程。
