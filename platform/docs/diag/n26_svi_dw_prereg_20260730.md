# N2.6-SVI-DW 单一候选预登记

日期：2026-07-30  
状态：`PREREGISTERED / N2.6e1 SOURCE-METHOD IMPLEMENTATION AUTHORIZED /
N2.6e2 TARGET INTEGRATION PENDING`  
基线：冻结的 V4.1；候选只以新 closure shadow 运行。

## 1. 步骤①：病因指纹与唯一可动空间

修正 Fig18 曲线身份后，在 `f=1.4 Hz, twist=22.5 deg, AoA=5 deg`：

| U (m/s) | V4.1 推力残差 model - experiment (N) |
|---:|---:|
| 6 | +1.110 |
| 8 | +1.805 |
| 10 | +2.038 |

模型随 U 的趋势方向正确，但高 U 压力/阻力缺口增大。该指纹与缺失的黏性
位移效应和分离压力阻力相容；它也与未分离的实验机构/tare 相容。G0e 先以
Fig15 当前 V4.1 静态滑翔锚判别“额外静态阻力”解释。

唯一可动节点为 `N2.6`。冻结：

- N1 AIC、网格、运动学和尾迹内核；
- N4 ForceLedger；
- V4.1、旧 N2 chop、旧 N3 直接涡力和 LESP 持续幅值；
- `d_para=0.5` 物理基线，不以新常数吸收残差。

## 2. 步骤②：一手机理与路线裁决

唯一候选采用 Riziotis & Voutsinas (2008) 的 strong
viscous--inviscid interaction 与 double-wake：

1. 双侧积分边界层保存质量/动量亏损和非平衡记忆；
2. Riziotis 的体积亏损通量
   \(M_B=u_e\delta^*\) 与代码中另行命名的质量亏损通量
   \(\mathcal M_B=\rho_e u_e\delta^*\) 不得混写；后者通过
   \(w_{e,n}=\rho_e^{-1}\partial_s\mathcal M_B\) 作为同一外流解的
   Neumann/RHS；
3. 内外流非线性联立收敛，不在势流求解后追加总力；
4. 分离点由收敛的边界层状态提供，从该点释放第二条剪切尾迹；
5. 尾迹反诱导进入同一外流解，唯一双侧 Bernoulli 压力产生
   \(p^\pm(s,t)\)、压力中心、力和力矩。

对照路线裁决：

- Ramanathan--Gopalarathnam：仅首次回流/LEV initiation，生产 NO-GO；
- MFOIL：稳态单尾迹，只允许作单元参考，生产 NO-GO；
- 静态极曲线/L-B 弦向力：不能唯一给出面板压力，生产 NO-GO；
- LESP/BEF/f2 持续幅值：已证伪，禁止重走。

来源：

- Riziotis & Voutsinas, *Int. J. Numer. Meth. Fluids* 56 (2008),
  DOI `10.1002/fld.1525`;
- Yu et al., *Wind Energy* 27 (2024), DOI `10.1002/we.2889`，作为
  double-wake 系谱与适用域限制的开放复核；
- Drela & Giles, *AIAA Journal* 25 (1987)，强相互作用 IBL；
- Ramesh et al., *JFM* 751 (2014)，仅用于锁定 LESP 的起涡角色。

## 3. 步骤③：缺件/错件判定

- 缺件：N2.6 尚无可执行的物理 IBL 闭合、transpiration 强耦合和 separation
  double wake；
- 错件：N2.2 的全局风升力向 chop 及任何独立总力增量不能代表统一压力；
- N3 的角色只保留已经释放的空间涡态；shadow 中不得与 double wake 再记
  第二份涡力。

## 4. 步骤④：候选命题

> 对同一实际双侧翼面外流表示，双侧非定常 IBL 亏损通过守恒
> transpiration 强耦合回同一外流，并由边界层分离位置释放第二尾迹，可以
> 在不使用目标总力拟合的情况下产生随 U 和局部分离面积增长的压力阻力，
> 同时输出可用于 co-design 的面板压力。

候选按三个互不越权的阶段推进：

1. `N2.6e1` 是来源论文二维 actual-surface 方法复现，只验证 strong VI +
   double-wake 方程、压力拓扑和分离滞回；
2. `N2.6e2` 才是目标域二维条带降阶 shadow；它必须声明自己的实际双侧外流
   表示，不能把双侧厚翼压力作为冻结零厚 N1 压力的增量；
3. `N2.6e3` 是三维曲面 IBL、连续分离流形和统一全翼压力的生产命题。

`e1` 通过不自动授权 `e2/e3`。条带 shadow 只检验该机理能否跨过 Fig18
三点门，不构成三维横流生产晋升。

## 5. 步骤⑤：最小忠实实现边界

必须实现：

- 来源方法阶段的真实上下表面面板表示；
- 两侧 \(\delta^*,\theta\) 及转捩/应力记忆状态；
- transpiration 对外流 RHS 的反馈和同一步非线性收敛；
- Riziotis 2003 上游一手论文直接规定收敛的 `Cf=0` 为二维来源方法的
  分离事件；它只用于 source-response gate，不复活已经证伪的
  `N2.6c1a` 三维生产命题。分离位置仍须在边界层收敛后才更新并重网格；
- TE wake 与 separation wake 的唯一 Kelvin ledger；
- 唯一双侧非定常 Bernoulli 压力和壁面剪切牵引；
- 条带压力映射到全翼材料面板，输出 panel force/moment；
- N2.6e 作为一个 solver-loop 内部强耦合 stage 或独立 private solver 执行；
  求解完成后读取 `solver_channels` 的普通 `ClaimComponent` 不具备该能力；
- 零 IBL 亏损严格退化到候选自身的 actual-surface 无黏基线；
- 厚度趋零且网格收敛时才应收敛到 N1 thin-sheet 极限。有限厚度候选不得
  同时被要求逐位等于零厚 N1 压力。

禁止：

- 用 Fig17/18/19 选择 `Ncrit`、分离阈值、wake core、压力 offset 或比例；
- 用稳态 MFOIL/静态极曲线加时间滞后冒充 Riziotis 候选；
- 旧 N2 chop、旧 N3 涡力、Garrick suction 与双侧鼻部压力重复记账；
- 由总力残差反演 \(C_p\)、\(\delta^*\)、\(C_f\) 或分离位置。
- 在冻结 N1 已经求解后追加 transpiration 力或“双侧压力增量”；
- 用来源论文二维复现通过，替代目标域表示门或三维 co-design 晋升门。

## 6. 先验 go/no-go

### A. 数值与原方法门

“原程序逐位身份复现”与“论文公开响应复现”必须分开。论文未公开面板数、
时间步、涡核、整翼重网格状态/旧势转移和完整运行身份，前者 `NO-GO`；
Riziotis 2003 上游论文已补齐闭包公式、BDF2、转捩、`Cf=0` 和 Bernoulli
压力，后者按 `n26e1_source_response_contract_20260730.md` 的 Fig.12
矢量曲线和分离历史执行。

1. 零亏损严格返回候选自身 actual-surface 无黏基线；厚度趋零与面板加密时
   才检查对 N1 thin-sheet 极限的收敛，不规定不可能的逐位恒等；
2. 收敛方程的两侧质量亏损、代数 Kelvin、TE/分离释放和 ForceLedger
   相对残差 `<1e-8`；常强度 collocation 的积分 source-flux 和表面迹环量
   属空间截断量，必须独立报告网格收敛，禁止伪称代数 ledger；
3. 面板/时间步加倍后周期均值 L/T/M 变化 `<=2%`，
   \(C_p\) 相对 L2 变化 `<=5%`，`s_sep/c <=0.02`，事件相位 `<=2 deg`；
4. 不看 RoboEagle 力，先在 Fig.12 八个相位逐面、逐侧达到公开
   double-wake 曲线 nRMSE `<=5%`，同时复现该论文模型自身的分离延迟、
   再附着方向和双侧压力拓扑。

任何一项失败：`N2.6e1` 不允许进入目标域表示裁决。即使全部通过，也只说明
来源方法复现成立；必须另行冻结 `N2.6e2` 的 actual-surface→全翼材料面板
降阶和适用域门，才允许进入 Fig 工况。

### B. Fig18 三点门

仅在 `N2.6e1` 来源门和 `N2.6e2` 目标表示门通过后，冻结全部参数运行
U=6/8/10 三点。必须同时满足：

- \(\Delta T(10)<\Delta T(8)<\Delta T(6)\le0\)；
- 三点 thrust RMSE 和 `dT/dU` 斜率误差均优于 V4.1；
- 任一 lift 不得比 V4.1 恶化超过 0.15 N；
- 修正可追溯到双侧 \(C_p\) 平台、压力中心和分离区，而非显式 \(U^2\) 项。

失败：`N2.6e2-SVI-DW-stripwise -> falsified/frozen`，禁止调参重跑；
它不反向证伪已经独立通过的来源方法，也不证明三维 `N2.6e3` 必然失败。

### C. 后续门

三点通过后，才运行 Fig17/18/19 的峰值、转折和边界代表点；代表点全部通过
后，才运行完整 50 曲线/184 工况。最终比较保留 V4.1 不变。

## 7. 已知目标域风险

Riziotis 的验证主要位于
\(Re=1.5\text{--}6.3\times10^6,\ k=0.05\text{--}0.1\)；
RoboEagle 约为
\(Re=1.1\text{--}1.9\times10^5,\ k=0.13\text{--}0.39\)。
低 Re 转捩、强三维横流和高 k 均属外推。若 `k>0.15` 不收敛、对网格/尾迹核
敏感、条带分离线不连续，或只能改善总力而不能稳定输出全翼 \(p^\pm\)，
立即杀死该具体 shadow。

## 8. 运行时架构审计

现有 `claim_runtime.components` 仅对低层求解器已经计算好的
`solver_channels` 做来源记账，不能让 IBL 位移厚度、分离位置或第二尾迹在
同一步反馈 AIC/Kutta/Kelvin 系统。因此：

- 普通后处理组件路线 `NO-GO`；
- `N2.6e1` 可以使用独立的小规模二维隐式求解器；
- `N2.6e2/e3` 必须成为拥有 outer/IBL/wake/pressure 联立残差的原子内部
  stage；生产规模需 block-sparse/JFNK/GMRES 或等价求解，现有 dense
  `ImplicitClaimGroup` 只适合二维 canonical oracle；
- 压力只由收敛后的总势、总速度通过一次非定常 Bernoulli 得到，壁面牵引
  只积分一次。任何旧 N2/N3/LESP 力增量均与该候选互斥。

## 9. G0e 授权结果

`g0e_fig15_static_discriminator_20260730.md` 区分了两个此前混写的身份：
冻结 Fig17/18/19 V4.1 实际为 `visc=False`，其 U=8 静态
`L/D=8.12979`，没有通过 Fig15；历史 `visc=True` 物理锚为
`L/D=7.09841` 并通过。已有附着摩擦在 U=8 只提供 `0.134665 N` 阻力，
为 corrected Fig18 `1.805 N` 缺口的 `7.46%`。

裁决为：

- 禁止增加静态阻力常数或以 `d_para`/摩擦倍率吸收剩余缺口；
- N2 分离压力/强 VI 候选仍与指纹相容，但与未知动态 tare 尚不可唯一分离；
- 授权 `N2.6e1` 来源方法复现；
- `N2.6e2` 目标 Fig18 运行仍以 e1 来源门和 actual-surface 目标表示门为
  前置条件。
