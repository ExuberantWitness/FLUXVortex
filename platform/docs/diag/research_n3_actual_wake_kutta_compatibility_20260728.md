# N3 S3ag：Kutta closure 的兼容性与正则根裁决

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3e`  
执行状态：**COUNTEREXAMPLE-GO / PHYSICAL-COMPATIBILITY-NO-GO / production off**

## ① 病因定位

S3af 已证明正确的代数槽为

\[
B\phi+Wg=b,\qquad F_K(\phi,g,\text{history},X)=0,
\]

其中 \(n=81\)、\(r=7\)，全部 body BIE 必须保留。但“存在一个 7 维
closure 槽”不等于任意 7 条残差都给出物理解。

实施前的只读可行性诊断出现两个不同但同源的指纹：

1. **prescribed birth branch**：用 S3ae 的局部二维
   \(\dot\Gamma_g\) 乘翼展 envelope \(1-y^2\)，完整 BIE 残差约
   \(3\times10^{-16}\)，但
   \(A=g-C\phi\) 在四个 \(\Delta t\) 上保持约 `0.013–0.016`，没有随
   newborn 厚度消失。
2. **pressure branch**：weak-P2 与 midpoint/vertex-average 两种合理
   pressure observation 都可被 Newton 降到接近机器精度，却分别产生
   `|g|max≈14.6` 与 `≈4.3`，同时 `|g-Cφ|` 达 `O(3–10)`；零残差并未
   识别同一个正则根。

这些数值已经被观察，故本门明确是反例复现，不冒充盲验收。可动空间只在
`...b3e`；S3ae 局部恒等式、S3af 完整 BIE/closure 槽、冻结 N1/N4 与生产
力均不改。

## ② 学科机理

Xia–Mohseni 的

\[
\dot\Gamma_g=u_g\gamma_g
=\tfrac12(u_{2-}^2-u_{1+}^2)
\]

是局部二维 circulation-rate/product 关系。形成片还需 unsteady Kutta
强度、形成方向和局部动量共同闭合。Dumoulin–Eldredge–Chatelain 的非定常
panel 系统也把 no-through-flow、Kelvin 与 Kutta 作为不同约束联立。

三种物理角色不能混写：

- pressure Kutta 选择当前 inviscid circulation/wake strength；
- Kelvin/material conservation 转移已脱落环量并保存物质历史；
- attachment/geometry 规定片从边缘发出和运动，不自动确定强度。

Chouliaras 与 Wang 支持“独立 wake jump＋完整 body BIE＋pressure Kutta”，
但前者是稳态、已知平面 wake。其 Morino 关系只作 Newton 初值。Ramesh
进一步说明，仅满足 Kelvin 仍可能留下尾缘涡量不连续和非零压差。
Zhu 等的强非定常实验还表明真实边界层可支持非零尾缘压差，因此
zero-pressure Kutta 只能作为当前 inviscid 候选，不能直接冻结为扑翼普适律。

Leroy–Devinant 的三维非定常薄涡片理论在共同脱落边要求 body 与 wake
potential jump 及其适用方向导数相容；Morino–Bernardini 把 jump
不相容关联到 junction 的 vortex-line singularity。因此在连续 sharp
junction 上 \(A=g-C\phi=0\) 是正则相容身份；离散时 \(A\) 是必须随
\(h/p/\Delta t\) 收敛的 guard，不能被当作可忽略误差。若 \(A\) 在接缝
极限保持 \(O(1)\)，分布上相当于未解析的 edge doublet/line-vorticity
defect；即便 newborn 内部 \(\Delta g=O(\Delta t)\)，接缝近场仍不会因
时间细化自动正则。

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| S3ae 局部二维 birth-rate/P2 表示 | validated/frozen，不改 |
| 独立 \(g\)＋全部 body BIE＋一套 \(r\) 维 closure 槽 | validated/frozen，不改 |
| 逐展向条带搬用一个 prescribed 二维 rate 就足够 | 待本门复现的错组件 |
| pressure residual 为零即可晋升物理解 | 待本门复现的错组件 |
| \(g-C\phi\)、有界幅值、连续分支和交叉残差 | 缺少的正则根守卫 |
| actual 3D incident-side velocity/formation state | 物理缺件 |
| 调 metric、平滑、clamp 或目标力选根 | 禁止进入 |

## ④ 方案与 go/no-go

本门不改模型，只冻结两个 counterexample：

1. 从同一 q5/x2 Morino state 取旧物质迹；
2. 对四个 \(\Delta t\) 以 \(q(y)=\dot\Gamma_g(1-y^2)\) 构造 newborn，
   解完整独立-wake BIE，测量 trace 阶、完整 BIE 与
   \(A=g-C\phi\) 的时间阶；
3. 在同一 steady actual system 上，从 Morino 初值分别求 weak-P2 与
   collocation pressure roots；
4. Newton 只使用解析 Jacobian 与确定性回溯；同时报告完整 BIE、
   pressure residual、dense pressure jump、wake 幅值、Schur condition、
   \(A\) 和两种 observation 的 root 差；
5. 任何“小 residual 但不正则”的结果均判 physical-compatibility
   `NO-GO`。

若反例复现，下一节点必须先建立 actual 3D side-velocity trace、edge
compatibility 和 continuation/regular-root 条件，再研究真正的
birth/pressure closure。禁止把 Newton convergence、某个测试 metric 或
某条 lift 曲线当作选根依据。

本门无 pressure force、无 LESP、无 Fig17/18/19、无 118 工况、无结构。

## 定义审计

第一次正式执行前，独立只读审计冻结了三个实现歧义，没有读取正式结果：

1. `birth edge defect order` 取绝对值，禁止负阶发散因
   `order <= 0.25` 被误判为通过；
2. weak map 是逐 segment 先算上下 incident-face 压差，再作 consistent
   P2 line assembly；collocation map 在内部 span vertex 平均左右两个
   **pressure values**，不跨楔角先平均 velocity；
3. Newton 先提 full step，只允许固定序列 \(2^{-k}\)、\(k=0,\ldots,23\)
   的 residual-decrease line search；禁止的是 physical/state damping、
   regularization 和 amplitude clamp。

当前 absolute defect/amplification 阈值只属于 chord=1、\(U=1\) 的固定
q5/x2 witness fingerprint，不是可移植的普适正则性阈值。moving-wall 或
非定常 pressure closure 尚缺两侧势时间率，本门会 fail closed。

## 正式结果

`actual_wake_kutta_compatibility_results.json` 的 12 项检查全部通过。

### Prescribed birth branch

| \(\Delta t\) | newborn length | \(\max|\Delta g|\) | full-BIE residual | \(\max|g-C\phi|\) |
|---:|---:|---:|---:|---:|
| 0.0040 | 0.0056795 | 0.006000 | \(3.48\times10^{-16}\) | 0.0137915 |
| 0.0020 | 0.0028397 | 0.003000 | \(2.93\times10^{-16}\) | 0.0160313 |
| 0.0010 | 0.0014199 | 0.001500 | \(3.02\times10^{-16}\) | 0.0151095 |
| 0.0005 | 0.0007099 | 0.000750 | \(3.42\times10^{-16}\) | 0.0133451 |

trace increment 的时间阶为 `1.000000`，但 compatibility defect 的拟合阶
仅 `0.022786`；circulation-flux 残差最大
\(2.60\times10^{-18}\)。所以 local birth identity 与全部 body BIE 都被
准确执行，仍没有得到 glued-sheet compatibility。裁决只证伪
“arbitrary prescribed stripwise \(q\) 足够”，不证伪由 actual side
velocity、Kelvin、方向和动量联立的 birth law。

### Steady zero-pressure observations

| observation | pressure residual | \(\max|g|\) | 相对 Morino | \(\max|g-C\phi|\) | q8 dense \(\max|\Delta p/\rho|\) |
|---|---:|---:|---:|---:|---:|
| weak active P2 | \(3.52\times10^{-15}\) | 14.6473 | 117.73× | 10.2950 | 0.30440 |
| midpoint/vertex-average | \(1.29\times10^{-12}\) | 4.29335 | 34.51× | 2.94398 | 0.14354 |

两个 reduced Jacobian 均为 rank 7；解析 Jacobian 的方向中心差分误差最大
\(2.23\times10^{-9}\)，全部 Newton 更新接受 full step。两种 observation
得到的 \(g\) 最大差为 `10.3539`。这证明它们是健康的代数根，但不是同一
正则物理分支；也不能把两个不同 observation-map 的根误写成“同一非线性
方程的多根证明”。

## Claim 裁决与下一门

- `...b3e1`：任意 prescribed birth-flux 的充分性，`falsified/frozen`；
- `...b3e2`：pressure residual/ Newton convergence 的充分性，
  `falsified/frozen`；
- `...b3e3`：一致初态上的 regular-branch compatibility 门，`open`。

下一门从同时满足完整 BIE 与 compatibility 的参考态出发，选择一套明确
命名的 closure，做 mass-whitened Schur、解析 Jacobian、forcing
continuation、\(h/p/\Delta t\) 相容收敛、Kelvin 与未强加 residual
交叉检查。若当前 \((\phi,g)\) 已耗尽自由度而无法同时满足 compatibility
和 birth/pressure 动力条件，裁决必须是“缺 forming-sheet/junction state
及 DAE 结构”，不能删除 \(A\)、body rows 或增加 least-squares 配平。

## 一手来源

- Xia & Mohseni, *JFM* 830 (2017),
  <https://doi.org/10.1017/jfm.2017.513>.
- Dumoulin, Eldredge & Chatelain, *JFM* 977 (2023),
  <https://doi.org/10.1017/jfm.2023.997>.
- Chouliaras et al., *CMAME* 373 (2021),
  <https://doi.org/10.1016/j.cma.2020.113556>.
- Wang, Abdel-Maksoud & Song, *Ocean Engineering* 130 (2017),
  <https://doi.org/10.1016/j.oceaneng.2016.12.009>.
- Ramesh, *On Satisfying the Kutta Condition in Unsteady Thin Aerofoil
  Theory*, <https://arxiv.org/abs/2205.08647>.
- Zhu et al., *JFM* 893 R2 (2020),
  <https://doi.org/10.1017/jfm.2020.254>.
- Leroy & Devinant, *International Journal for Numerical Methods in Fluids*
  29(1) (1999),
  <https://doi.org/10.1002/(SICI)1097-0363(19990115)29:1%3C75::AID-FLD773%3E3.0.CO;2-7>.
- Morino & Bernardini, *Finite Elements in Analysis and Design* 38 (2001),
  <https://doi.org/10.1016/S0955-7997(01)00063-7>.
