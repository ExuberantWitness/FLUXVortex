# N2.6b4f 裁决：四加二 IBL 状态是可证伪零假设，不是完整近壁场

## 0. 范围

本裁决只处理气动载荷链中的

```text
N1 外流
  -> N2.6b 曲面 IBL 库存
  -> N2.6c 分离流形/守恒释放
  -> N3 空间涡态
  -> 统一面板压力。
```

结构动力学、载荷传递和柔性设计不在本阶段。结构侧唯一留下的未来接口约束是：
气动结果必须是带空间位置和方向的面板压力，而不是只闭合总力。

## 1. 病因、数据指纹和可动空间

### 1.1 已有反例

`ibl_to_ves_profile_results.json` 已证明：

- 在剖面外增加一段已经达到外流速度的平台，IBL 速度亏损矩不变，
  VES 面质量和面动量却改变；
- 两条单调剖面可以具有相同的 `M` 和 `tr(T)`，同时具有不同的层厚、
  面质量和面动量。

因此

```text
有限 IBL 矩 -> 完整近壁剖面/edge/VES state
```

不是单值映射。`N2.6b4d` 已据此证伪冻结，`N2.6b4e` 只冻结“显式
profile-edge 同源投影”的状态身份，不代表动力闭合已经完成。

### 1.2 RoboEagle 目标域

本项目的目标域同时具有：

- 约 `1.1e5--1.9e5` 的自由来流弦长 Reynolds 数，局部相对速度继续扩域；
- 移动壁面、大幅扑动和扭转；
- 双侧三维横流、逆压梯度、转捩和首次回流；
- 分离后由 N2.6c 向空间涡态交接。

目标不是复原一个稳态附着剖面，而是给出足以计算物质 spike、层边缘运动和
守恒释涡的时空近壁场。

### 1.3 可动与禁止空间

可动节点只有：

```text
N2.6b4f -> N2.6c1c -> N2.6c2c -> N3.1i。
```

禁止：

- 修改冻结的 N1、N2.6b3、N2.6b4e、N2.6c2b、N4；
- 复活 `N3.1f` 的旧 rVPM 生产支路；
- 用 L/T、面板压力残差、LESP 或结构响应选择剖面、edge 或潜变量；
- 把制造剖面测试称为动态失速物理验证。

## 2. 一手文献给出的信息边界

### 2.1 Drela IBL3：状态拓扑有物理意义，但公开剖面族不是本案证据

Drela 的三维 IBL 以 `delta, U_tau, W_tau, Psi` 描述流向/横流剖面，
并以两个附加应力/转捩状态描述非平衡记忆。该拓扑保留旋转客观性、横流和
强黏性—无黏耦合，是比当前 L-B 标量状态更合适的参考上限。

但这只支持“需要独立剖面和记忆状态”。它不证明公开的假定剖面族能覆盖
RoboEagle 的扑翼动态失速域。

### 2.2 Zhang 2022：四个 IBL 主变量是矩，不是剖面坐标的充分证明

Zhang 的四个主变量为

```text
Q_IBL = {delta*_1, delta*_2, theta_11, theta_12}，
```

并另加转捩包络和湍流应力 lag 状态。论文明确说明：

1. Drela 的 profile-based closure 采用少量参数化的假定剖面族；
2. 既有剖面族多由启发式多项式构造，可能遗漏真实剖面模态并引入非物理模态；
3. 用 DeepONet/Neural Implicit Flow 做算子学习更系统，但训练和推理成本可能
   过高；
4. 论文实际选择的是直接回归 IBL 所需的闭合积分量，而非重构完整剖面；
5. 数据约为 1.5 万条 DBL3 层流剖面和 7.8 万条 RANS-SA 湍流剖面；LES、
   DNS 和实验数据因成本/数量没有进入该研究。

所以 Zhang 闭合可为 `T/E/D/tau_w` 等 IBL 通量提供候选函数，但其输出类型
缺少 N2.6c 所需的剖面、edge 几何/运动、法向梯度和历史。把该回归直接接到
VES release 是“组件类型错误”，不是参数没调好。

### 2.3 公开资产审计

2026-07-28 审计：

- MIT DSpace 条目 `hdl:1721.1/147502` 的 ORIGINAL bundle 只有论文 PDF，
  TEXT bundle 只有抽取文本；
- MIT Drela 公共下载目录未发现 IBL3 源码或剖面数据；
- 以 `IBL3`、论文题名、`AIAA 2022-1078` 和作者/主题检索 GitHub，未找到
  可复用的官方数据或实现。

因此不能把论文中的样本数量当作本仓可训练资产，也不能转录未公开的权重或
剖面族。当前只能先冻结数据契约与可辨识性门。

## 3. 缺组件还是组件错误

### 3.1 错组件：直接积分量回归充当完整近壁状态

判定：**NO-GO，证伪。**

映射

```text
(H1, H2, H12, Re_theta, history) -> IBL closure integrals
```

没有输出 `u(n), rho(n), delta_e, n_e, v_e`，因而不能计算物质面曲率演化，
也不能唯一给出 VES 的实际质量、动量和法向卷吸。给它增加一个由总力选择的
edge 常数仍然只是常数吸收。

### 3.2 缺组件：带 edge 的、受矩约束的时序剖面解码器

允许研究的最小结构是

```text
conserved state:
    Qc = {delta*_1, delta*_2, theta_11, theta_12}
memory state:
    h  = {transition envelope, stress-lag state}
edge state:
    E  = {delta_e, n_e, xdot_e, u_e^+, u_e^-,
          named edge convention}
optional profile state:
    z  = field-identified transportable residual coordinates

profile decoder:
    D(Qc, h, E, z, local history) -> {rho(n), u(n)}
```

`z` 不能预先按方便程度指定维数。先把 `z=empty` 的四加二模型作为零假设；
只有独立场数据证明条件剖面离散度超过数据/离散误差，才增加最少的残差模态。
新增模态必须具有可预测的时序输运，不能只是逐帧拟合坐标。

解码结果必须精确或在数据离散误差内满足：

- 移动壁面无滑移和给定 edge 速度；
- 曲面切向性、刚体旋转客观性和横流镜像对称；
- 二维极限；
- 重新积分后返回同一 `Qc`；
- 与 N2.6b4e 相同 edge convention 下同时返回 IBL 与 bound-VES 状态。

## 4. 可辨识性实验预登记

### 4.1 数据契约

每个独立场数据集必须提供：

- 时间、曲面节点/连接关系、双侧身份和曲面坐标；
- 壁面位置/速度、法向和正交切向基；
- 沿物理法线的 `rho(n), u(n)`；
- edge 位置、速度和命名的检测规则；
- Reynolds 数、运动学、三维横流、转捩和分离覆盖标签；
- train/validation/test 角色与独立审计标识。

禁止字段包括力/力矩目标、L/T、LESP、压力残差和结构响应。

### 4.2 H0：四加二状态足以条件重构

在不使用积分载荷的 test 工况上：

1. 将剖面统一到数据集声明的 edge 坐标；
2. 由场剖面计算 `Qc,h,E`；
3. 搜索状态空间近邻，测量其剖面和法向梯度的条件离散度；
4. 训练/拟合只受场量监督的解码器；
5. leave-one-kinematics-out 检查剖面、壁面剪切、edge 运动和重积分矩；
6. 用解码场推进 N2.6c1b oracle，检查 spike 流形误差。

若状态近邻的条件剖面差异显著高于场数据和离散误差，或 spike 位置/时刻不能
进入 oracle 误差带，则拒绝 H0，进入 H1。

### 4.3 H1：增加最少的可输运残差状态

只对 H0 解码残差做场级 POD/受约束自动编码，不接收任何载荷目标。按以下顺序
增加 `z`：

1. 每增加一维，先验证其可由局部 IBL 状态及历史推进；
2. 再验证 moment/null-space 中的剖面误差是否下降；
3. 最后验证 N2.6c1 spike，而不是先看总 L/T；
4. 跨运动学、Reynolds 数、扭转和侧别留出验证。

不能稳定输运、只在训练轨迹上降低重构误差的模态不得进入生产树。

## 5. Go / No-Go

### 本阶段 GO

- 冻结独立场数据 schema、客观性和禁止标签；
- 实现 manufactured identity guard，证明数据接口和坐标变换无歧义；
- 将四加二状态标为“可证伪零假设”；
- 将直接积分量回归标为 IBL flux closure，而非 profile/VES closure。

### 生产晋升 NO-GO

在获得并通过代表性近壁时序场之前：

- 不实现任意多项式/神经网络剖面作为 V4.1 生产闭合；
- 不声称 `N2.6b4f` 已完成；
- 不接入自由 LEV 力或统一面板压力；
- 不用三点或 118 总力结果选择剖面架构。

## 6. 原始来源

- Drela, M., *Three-Dimensional Integral Boundary Layer Formulation for
  General Configurations*, AIAA 2013-2437,
  https://doi.org/10.2514/6.2013-2437.
- Drela, M., *Fast 3D Viscous Calculation Methods*, MIT/Boeing presentation,
  https://web.mit.edu/drela/Public/Drela_Boeing_6May14.pdf.
- Zhang, S., *Three-dimensional Integral Boundary Layer Method for Viscous
  Aerodynamic Analysis*, MIT PhD thesis, 2022,
  https://hdl.handle.net/1721.1/147502.
- Zhang, S. et al., *Closure Modeling for Three-dimensional Integral Boundary
  Layer using Physics-constrained Neural Network and Model Inversion*,
  AIAA 2022-1078, https://doi.org/10.2514/6.2022-1078.
- DeVoria, A. C. & Mohseni, K., *The vortex-entrainment sheet in an inviscid
  fluid: theory and separation at a sharp edge*, JFM 866 (2019),
  https://doi.org/10.1017/jfm.2019.134.

