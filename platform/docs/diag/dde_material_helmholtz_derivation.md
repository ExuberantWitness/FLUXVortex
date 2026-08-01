# Derivation Package

## Target

判断 P2 DDE 自由片在顶点随材料运动、材料重心坐标中的势跃值保持不变时，是否已经
满足单个仿射三角面元的 Kelvin–Helmholtz 拉伸恒等式，还是必须再引入经验“拉伸修正”。

立即目标是一个**精确恒等式**，不是完整自由尾迹算法，也不是压力或载荷近似。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

成立范围必须重述为“光滑、无黏、无新生涡量的旧自由片上，一个非退化、
piecewise-affine 材料三角面元”。它不能自动证明弯曲公共边、重网格或涡片重联。

## Invariant Object

组织推导的量不是面元局部坐标中的涡量分量，而是材料势跃
\(\chi(a^1,a^2)\)。对与 N1 法向同向的 DDE，代码标量满足
\(\mu_{\mathrm{DDE}}=-\chi=-\Gamma_{\mathrm{N1}}\)。

旧自由片的 Kelvin 身份是

\[
\frac{D\chi}{Dt}=0.
\]

## Assumptions

- \(\mathbf{x}(a^1,a^2,t)\) 是一个非退化材料面映射；
- 一个三角面内该映射为仿射，P2 势跃用材料重心坐标表达；
- 流体无黏、无 baroclinic source，且该面元不是当步生成边界；
- 不发生重网格、切割、合并、重联或拓扑改变；
- 面元节点顺序保存材料取向；
- 本推导只处理单元内部；跨非共面公共边的兼容条件另行研究。

## Notation

- \(\mathbf{g}_1=\partial\mathbf{x}/\partial a^1\)，
  \(\mathbf{g}_2=\partial\mathbf{x}/\partial a^2\)：协变切向基；
- \(J=\lvert\mathbf{g}_1\times\mathbf{g}_2\rvert\)；
- \(\mathbf{n}=(\mathbf{g}_1\times\mathbf{g}_2)/J\)；
- \(\nabla_s\)：曲面梯度；
- \(\boldsymbol{\gamma}=\mathbf{n}\times\nabla_s\chi\)：物理涡片强度；
- \(\mathbf{u}=D\mathbf{x}/Dt\)：材料点速度。

## Derivation Strategy

从材料势跃这个守恒标量出发，先把物理涡片强度写成协变基上的向量密度，再对材料
时间求导。这样可以区分“势跃保持”这一 Kelvin 命题与“涡量因拉伸而改变”这一
Helmholtz 命题；二者不是两个可独立调参的状态。

## Derivation Map

1. 曲面梯度与协变/逆变基给出 \(J\boldsymbol{\gamma}\) 的精确表达；
2. \(D\chi/Dt=0\) 使材料偏导 \(\chi_{,1},\chi_{,2}\) 不随时间改变；
3. 对协变基求导得到 Cauchy 拉伸式；
4. P2 重心节点值保持使上述材料标量身份在单元内离散精确；
5. 非共面公共边、重网格和黏性项不由该单元恒等式覆盖。

## Main Derivation

Step 1（identity）。逆变基满足
\(\nabla_s\chi=\chi_{,1}\mathbf{g}^1+\chi_{,2}\mathbf{g}^2\)，并且

\[
\mathbf{n}\times\mathbf{g}^1=\frac{\mathbf{g}_2}{J},\qquad
\mathbf{n}\times\mathbf{g}^2=-\frac{\mathbf{g}_1}{J}.
\]

因此

\[
J\boldsymbol{\gamma}
=J\mathbf{n}\times\nabla_s\chi
=\chi_{,1}\mathbf{g}_2-\chi_{,2}\mathbf{g}_1. \tag{1}
\]

Step 2（proposition under assumptions）。由 \(D\chi/Dt=0\)，且
\(a^1,a^2\) 是材料坐标，

\[
\frac{D\chi_{,1}}{Dt}=0,\qquad
\frac{D\chi_{,2}}{Dt}=0. \tag{2}
\]

Step 3（identity）。材料基的时间导数为

\[
\frac{D\mathbf{g}_i}{Dt}
=\frac{\partial\mathbf{u}}{\partial a^i}.
\]

对式 (1) 求材料导数：

\[
\frac{D(J\boldsymbol{\gamma})}{Dt}
=\chi_{,1}\frac{\partial\mathbf{u}}{\partial a^2}
-\chi_{,2}\frac{\partial\mathbf{u}}{\partial a^1}
=(J\boldsymbol{\gamma}\cdot\nabla_s)\mathbf{u}. \tag{3}
\]

式 (3) 是面涡量向量密度的 Cauchy/Helmholtz 拉伸形式。故旧自由片无需第二个经验
“stretch coefficient”：保持材料势跃并更新几何，物理涡片强度会通过
\(J,\mathbf{g}_1,\mathbf{g}_2\) 自动变化。

Step 4（discrete identity）。仿射三角面内，重心坐标本身是材料坐标。保持六个 P2
节点的 \(\mu_{\mathrm{DDE}}\) 值不变，等价于保持整个 P2
\(\chi=-\mu_{\mathrm{DDE}}\) 多项式不变。因此式 (1)–(3) 在每个非退化面元内可按
机器精度检查，不需要拟合或平滑。

## Remarks and Interpretation

- Krebs 在 wake relaxation 后“重新解系数”不等价于给材料势跃添加物理修正。
  其算法还要重新满足全局无穿透、Kutta、势跃连续及梯度连续；如果系数定义在当前
  局部坐标中，几何改变后也必须重表达。
- FLUXV 当前把六个 P2 值绑定到材料重心节点，因此局部 Kelvin/Helmholtz 身份可以
  直接由式 (1) 检查。
- 这条结果支持“空间涡态”而非总力闭合：涡片强度随实际几何变形改变，后续诱导场
  和统一压力自然感知涡位置与拉伸。

## Boundaries and Non-Claims

- 不证明非共面相邻面元的涡量兼容；直接比较两个三维切向量通常没有正确的平行运输
  语义；
- 不证明重网格后的 circulation transfer；
- 不包含黏性扩散、涡量生成、涡片碰撞、重联或耗散；
- 不验证 N1–DDE 全局无穿透解、压力、力、118 工况或柔性翼生产耦合。

## Open Risks

- 弯曲公共边需要守恒的弱式 trace/flux compatibility，而不是沿用平面向量相等；
- 新生 TEV/LEV 的 \(\chi\) 仍须由全局 Kelvin/Kutta/no-penetration 系统确定；
- 高变形网格可能失去局部一一映射，必须设 Jacobian/质量门；
- 结构运动带来的 moving-surface pressure rate 仍须在统一压力推导中单独审计。

