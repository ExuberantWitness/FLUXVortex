# N2.6b/c 机理裁决：张量型非定常三维 IBL，而非标量 LEV 供给律

## 1. 病因指纹、树节点与可动空间

现有证据已经把误差源定位到 `N2.6b/c`：

- `N3.1j4b3/b4` 已证伪从 LESP 临界值逐条带生成整张 LEV 片幅值；
- `N2.6b2` 已验证矢量环量库存的代数守恒账，但该账本没有输运、壁面剪切、
  耗散、转捩或分离闭合；
- `N2.6d1` 已建立与冻结 N1 中弧面共置的 NACA-2406 双侧壁面；
- N1 AIC、运动学、N4 力簿记和已经验证的双侧 Bernoulli 恒等式均不在可动
  空间内。

因此病因不是一个待调的 LEV 力系数，而是：**表面附着黏性状态缺少能表示
三维横流和能量耗散的闭合方程，分离释放也没有由该状态导出的守恒边界。**

## 2. 一手文献给出的最小方程骨架

Bempedelis et al. 的三维有限体积积分边界层推导从非定常不可压缩边界层方程
出发，沿壁面法向积分，得到

\[
\frac{\partial\mathbf M}{\partial t}
+\widetilde{\nabla}\!\cdot\mathbf T
=-\widetilde{\nabla}\mathbf q_e\cdot\mathbf M
+\frac{\boldsymbol\tau_w}{\rho},
\]

\[
\frac{\partial\,{\rm tr}(\mathbf T)}{\partial t}
+\widetilde{\nabla}\!\cdot(\mathbf E-\mathbf T\mathbf q_e)
=2D-\mathbf T:\widetilde{\nabla}\mathbf q_e
+\mathbf q_e\!\cdot\!
\left(\widetilde{\nabla}\mathbf q_e\cdot\mathbf M
-\frac{\boldsymbol\tau_w}{\rho}\right).
\]

其中

\[
\mathbf M=\int_0^\infty(\mathbf q_e-\mathbf q)\,dz,\qquad
\mathbf T=\int_0^\infty[(\mathbf q_e-\mathbf q)\otimes\mathbf q]\,dz,
\]

\(\mathbf E\) 是动能亏损通量，\(D\) 是耗散积分。方程以张量形式写成，适合
任意曲面有限体积；论文明确指出，法向积分丢失的剖面信息必须由闭合关系补回。

### 必须纠正的状态层级

基础三方程系统的独立守恒储存量是：

1. 两个切向自由度的 \(\mathbf M\)；
2. 一个标量 \({\rm tr}(\mathbf T)\)。

\(\mathbf T\)、\(\mathbf E\)、\(D\) 和 \(\boldsymbol\tau_w\) 是通量/源项所需的
剖面矩和闭合量，不能全部当作互不受约束的自由状态。更高阶的 Drela 类四/六
方程模型可以增加横流或雷诺应力输运状态，但必须以独立闭合证据晋升。

Mager 的 NACA TR-1067 表明三维动量积分方程可以用于旋转壁面、螺旋桨和
直升机叶片；Lokatt & Eller 则验证了嵌入曲面上的有限体积形式可做到曲面守恒、
坐标旋转不变和网格收敛。它们支持“曲面守恒方程”这个架构，不证明任何特定
层流/湍流闭合可直接移植到 RoboEagle。

## 3. 缺件还是错件

### N2.6b：缺组成部分

`N2.6b2` 的矢量环量账本正确但不充分。生产状态至少还缺：

- 非定常三维 IBL 的守恒储存量 \((\mathbf M,{\rm tr}\mathbf T)\)；
- 张量动量通量、动能亏损通量、壁面剪切和耗散的物理闭合；
- 横流、逆压梯度、转捩/间歇性和非平衡湍流记忆；
- 随柔性双侧壳运动的面积 Jacobian、切空间和客观输运。

逐条带二维 L-B 是错层级件：可作为二维对照或局部闭合先验，不能作为三维
生产输运方程。

### N2.6c：缺组成部分

分离不应由 LESP、BEF 或一个普适形状因子阈值直接给幅值。分离流形应从已解
壁面剪切矢量场及其拓扑/附着性丧失导出；穿过该流形的附着库存、质量卷吸与
动量通量才可守恒地转成 DDE 新生面带。

Bempedelis et al. 层流剖面闭合中的特定临界形状因子只属于该闭合和验证范围；
其湍流简化甚至不能表示湍流分离。因此任何 `H_crit` 数值均不得直接移植为
RoboEagle 常数。

## 4. 有证据的方向裁决

### GO：方程骨架与守恒离散

先建立移动曲面上的三方程 IBL 守恒骨架。首个 CPU oracle 只接收具名的
\(\mathbf T,\mathbf E,D,\boldsymbol\tau_w\) 作为制造输入，验证：

- 曲面有限体积内部边严格抵消；
- 正交坐标旋转下向量/张量客观；
- 退化为二维时恢复标量动量—能量账；
- 移动面积下使用广延量 \(A\mathbf M,A\,{\rm tr}\mathbf T\)；
- 缺失通量或源项必须显式留下残差。

这一步不声称已有边界层预测能力，也不输出压力或力。

### OPEN：物理剖面、转捩与分离闭合

RoboEagle 的闭合必须用独立的剖面/PIV/高保真黏性场验证横流角、动量厚度、
形状因子、壁面剪切方向、转捩和分离线。候选至少要比较：

1. 三方程剖面积分闭合；
2. 增加横流/非平衡应力输运的四至六方程闭合；
3. 少量全黏性 VPM/CFD 作为场级 oracle。

### NO-GO

- 从总升力、推力或面板压力残差反演 \(\mathbf M,\mathbf T,\mathbf E\)；
- 把 LESP、BEF、`f2` 或固定 `H_crit` 当持续释放幅值；
- 在二维条带上独立推进后再用展向平滑伪装三维输运；
- 在闭合和分离线未验证前把 oracle 接入生产力链。

## 5. 对最终生产架构的约束

```text
柔性结构构形
  -> 双侧黏性壳的客观运动与曲率
  -> N1 外流 + N2.6a 移动壁面源
  -> N2.6b 三维 IBL 守恒状态与物理闭合
  -> N2.6c 剪切拓扑分离流形及守恒释放
  -> N3 连续 DDE 空间涡态
  -> N1/N2.6/N3 隐式一致解
  -> 唯一双侧 Bernoulli 面板压力
  -> 功共轭结构广义载荷
```

总力只能是统一面板压力积分的结果，不能反过来闭合表面状态或重分配结构载荷。

## 6. 来源

- Bempedelis, Bayeux, Blanchard, Radenac & Villedieu, *A 3D
  Finite-Volume Integral Boundary Layer method for icing applications*,
  Eqs. (6)–(13), (41)–(48).
- Mager, *Generalization of Boundary-Layer Momentum-Integral Equations to
  Three-Dimensional Flows Including Those of Rotating System*, NACA TR-1067,
  1952.
- Lokatt & Eller, *Finite-volume scheme for the solution of integral boundary
  layer equations*, Computers & Fluids 132 (2016) 62–71,
  doi:10.1016/j.compfluid.2016.04.002.
- Drela & Merchant, MIT 16.13 lecture notes 32–35, three-dimensional integral
  boundary-layer equations, closure and characteristics.

