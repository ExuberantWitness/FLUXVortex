# N3.1j4b5 全局守恒 predictor–corrector 预登记骨架

状态：**FORMULATION-ONLY / NOT ELIGIBLE TO ASSEMBLE A PHYSICAL SYSTEM**。
N1 环量—P2 势跳方向/量纲、半翼镜像、单元内材料 Kelvin–Helmholtz 恒等式和
无力代数账已经通过窄门；但光滑前缘的黏性涡量供给律与曲面跨单元兼容仍未闭合，
因此当前制造系统不得被解释成物理 LEV predictor–corrector。

## 1. 由 b3/b4 隔离出的病因

两个不同自由边界拓扑都已证伪“逐条带 `A0=LESPcrit` 可反演连续高阶片幅值”：

- b3 全半翼 active：源强 trace Cauchy 比 `0.739`；
- b4 在固定临界展向边终止：trace/field Cauchy 比 `0.282/0.763`。

两案的 LESP 线性残差均约 `1e-16`，条件数不超过 `28.3`，所以失败不是求解器误差，
而是把 onset criterion 当作空间幅值闭合的组成部分错误。

## 2. 一手方程边界

- Ramesh et al. 2018 将 LESP 假设定义为给定翼型/Re 下的 **LEV initiation
  criticality**；它决定何时开放前缘生成拓扑，不给出光滑前缘的局部涡量通量。
- Kandil, Chu & Tureaud 1982 使用一阶面涡量（等价二阶 doublet），单元内先满足
  `div(omega)=0`；相邻面元节点施加涡量连续。无穿透、连续性、Kutta 和对称条件组成
  过约束最小二乘系统。
- Kandil 的旧自由片按当地速度对流；材料环量由 Kelvin 保持，拉伸后的涡量由
  Helmholtz 更新。新自由条带与束缚面在同一时步求解。
- Krebs 2021 Fig.3.8 同样采用“最小二乘预测—消除强度不连续—重新评价 flow
  tangency”的迭代，并在 wake relaxation 前后各执行一次。
- Xia & Mohseni 2017 表明涡片形成问题需要把 Kutta、环量守恒、质量和动量共同闭合；
  单一临界标量不能同时确定片的方向、强度和相对速度。其完整闭合针对平板/尖点
  边缘；论文把光滑表面分离点预测明确留作后续，不能原样移植到 SD7003 型厚翼前缘。
- DeVoria & Mohseni 2018 的 vortex-entrainment sheet 表明纯涡片只携带涡量，
  若分离界面存在卷吸，还需面质量、内禀切向流和法向速度跳来闭合质量/动量。
- Sudharsan & Sharma 2024 表明 LESP 只能捕获显著影响前缘压力的事件；基于壁面
  涡量通量的 BEF 才能在壁面上定位局部涡脱落。这支持新增 `q_sep` 生成端口，
  不支持把 `A0` 或 `f2` 改名为通量。

## 3. 边界状态与下一系统的未知量

每个显式 stage `t_n, t_{n+1/2}, t_{n+1}`：

```text
x_boundary = [
  q_sep,                    # 新生片的有符号环量通量/卷吸/相对运动边界状态
  separation_location,      # 光滑前缘分离线，不等同于几何尖边
  sheet_direction           # 新生剪切层方向
]

x = [
  Gamma_b,                 # 冻结 N1 束缚环量
  mu_new,                  # 新生 DDE P2 势跳自由度
  lambda_interface         # N1–DDE 接口守恒乘子（若公式审计证明需要）
]
```

`x_boundary` 不是让势流全局系统自由吸收残差的额外旋钮。它必须由独立的 N2.6
黏性/边界层方程给出；若采用联立形式，则必须把对应的壁面涡量、质量和动量方程
逐项加入残差账。只有 `q_sep*Delta t` 这样的时间积分量才与 `mu_new/Gamma_b`
同属环量量纲。

旧 DDE 材料片的 `mu_old` 不是未知幅值；其 Kelvin 身份冻结，只更新几何和等价涡量
表达。新生片几何和供给来自独立、已预登记的生成边界模型，不能由总力选择。

## 4. 方程族与残差账

候选系统必须分别输出以下残差，禁止只报合并目标函数：

```text
r_np       = N1 no-penetration at every bound collocation point
r_trace    = P2 potential-jump continuity at every internal/interface edge
r_vort     = Kandil/Krebs edge vorticity/gradient compatibility
r_kelvin   = material circulation balance, including TEV and LEV families
r_kutta_TE = sharp/cusped trailing-edge compatibility only
r_sep_flux = smooth-leading-edge viscous generation/feed balance
r_sep_mass = entrainment/mass balance when the chosen sheet model carries it
r_sep_mom  = tangential/normal momentum balance needed by that feed model
r_sym      = mirror-root compatibility
r_free     = every physical free edge has zero jump or an explicit mate
r_lesp_evt = LESP threshold event identity (boolean/topology only, no amplitude row)
```

LESP 不进入 `mu_new` 的逐条带等式右端。它只决定生成拓扑是否开放。新生幅值必须
先由 `r_sep_flux`（以及所选模型要求的质量/动量项）提供，然后与
`r_np/r_kelvin/r_trace/r_vort` 全局相容。无 `r_sep_*` 的势流守恒系统不能声称
已经决定了光滑前缘新涡量。

## 5. predictor–corrector 顺序

```text
读取 t-stage 几何与旧材料片
  -> LESP observer 判定 event/topology
  -> N2.6 求 separation line / q_sep / direction / entrainment state
  -> 创建零占位的新生 DDE 拓扑
  -> 全局最小二乘预测 Gamma_b / mu_new，并消费有量纲的 q_sep*Delta t
  -> 只作有证据的 trace/vorticity/Kelvin 投影校正
  -> 重算 N1/DDE 诱导与 no-penetration residual
  -> 迭代至预登记残差门
  -> stage 状态入账
三 stage 完成
  -> 形成一个显式 P2 时间材料带
  -> Heun/等价二阶几何推进
  -> 重新求解并计算统一面板压力
```

## 6. 公式审计状态

1. **GO（窄门）**：N1 `Gamma_b` 与 DDE `mu` 的符号/单位；
   对齐法向时 `mu_DDE=-Gamma_N1`，诱导场相对误差 `1.07e-10`；
2. **GO（窄门）**：半翼镜像根部的势跳/涡量方向；
   镜像 N1/DDE 场相对误差 `1.04e-10`；
3. TEV Eq.9 与 LEV Kelvin 在同一 `r_kelvin` 中的时间索引；
4. **GO（单元内窄门）**：材料 P2 势跳自动满足仿射单元
   `D(J gamma)/Dt=(J gamma·grad_s)u`；曲面跨单元/重网格仍 open；
5. **GO（代数账窄门）**：过约束系统只按 `U_ref` 或 `U_ref L_ref`
   做量纲归一，禁止可调权重；
6. **OPEN / 当前主阻塞**：光滑前缘 `q_sep` 的壁面涡量通量/卷吸方程、分离线和
   新生片方向；
7. **OPEN**：曲面跨单元 vorticity compatibility 与物理矩阵的网格收敛；
8. **OPEN**：上述物理系统通过后，才允许进入统一压力账。

## 7. 预定 GO / NO-GO

GO 至少同时要求：

- 每个具名残差随迭代下降且达到运行前冻结阈值；
- `ns=4/8/16` 的 `mu` trace 与固定物理探针场均具有正 Cauchy 收敛；
- Kelvin/Helmholtz、自由边界和三 stage 中点身份逐项通过；
- `q_sep` 有独立壁面/边界层证据，且其时间积分与新生片环量逐步闭合；
- 无 pressure/force target 参与求解。

以下任一项直接 NO-GO：

- Tikhonov、span smoothing、滤波或人工耗散使结果“变顺”；
- 把 b3/b4 的逐条带 exact LESP cap 重新作为幅值方程；
- 把 `A0`、`f2`、`CV` 或目标总力改名/缩放为 `q_sep`，但没有壁面涡量通量、
  质量/动量或独立空间场证据；
- 把尖锐尾缘 Kutta 条件直接当作光滑厚翼前缘分离供给律；
- 只看诱导场/总力收敛而源强不收敛；
- 修改 `LESPcrit`、P-R 几何或网格来吸收残差；
- 在空间场门前打开统一压力或 ForceLedger。
