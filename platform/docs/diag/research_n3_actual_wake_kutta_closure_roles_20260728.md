# N3 S3af：独立 wake 状态与 Kutta 闭合的方程角色

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b3b`  
状态：**EXECUTED / ALGEBRAIC-ROLE GO / QUOTIENT NO-GO / production off**

## ① 病因定位

S3ae 已证明局部 sharp-junction 的物质 birth flux 能使
\(\Delta g=O(\Delta t)\) 且 P2 片涡量为 \(O(1)\)。但原公式包随后把
完整 cut 的节点数 \(p\) 与独立 jump 秩 \(r\) 混为一谈，并提出从 body
Galerkin 方程中删除同样数量的 test modes。

对冻结的 diamond full-wing canonical 做只读审计得到：

- base continuous P2 自由度 \(n_0=74\)；
- classified body 自由度 \(n=81\)；
- 完整 cut P2 节点数 \(p=9\)；
- 两个翼尖的 upper/lower DOF 相同，对应 jump 行严格为零；
- \(\operatorname{rank}C_{\rm full}=r=7\)，恰等于复制 DOF 数；
- body-only Galerkin 矩阵 \(B\) 的秩为 81。

所以 9 个节点不是 9 个独立 Kutta 方程。更重要的是，81 条 body 方程
没有可由拓扑自动删除的 7 维代数冗余。可动空间只在开放的 `...b3b`
及其后续闭合节点；冻结的 actual-boundary assembly、N1/N4 和生产力不动。

探索性诊断还发现，若分别用 Euclidean 与全表面一致质量内积定义待删除
子空间，两个投影算子及其解明显不同，而各自投影残差都很小。该指纹已在
预登记 YAML 中明示，因此 S3af 的 quotient 部分是**反例复现门**，不是
事后伪装成盲测的验收门。

## ② 学科机理

一手 lifting-BEM 文献支持两种明确架构，但不支持抽象的
“antisymmetric quotient 必须删除”：

1. Chouliaras 等保留全部 body BIE 系数，并把 wake potential jump 作为
   独立未知量，再用同数量的 zero-pressure-jump Kutta 方程闭合；
   Morino 关系只作为线性 Kutta 近似或 Newton 初值。
2. Wang 等同样通过尾缘压差对 wake dipole strengths 的 Jacobian 迭代
   独立 wake strength。
3. Cattarossi 等的 sharp-TE triple-DOF collocation 方案确实用 pressure
   Kutta 替换一侧 collocation row，但它具有显式 leeward/windward/wake
   三重变量；不能外推为当前 Galerkin cut 的 basis-invariant quotient。

因此，当前离散的正确方程角色应写成

\[
B\phi+Wg=b,\qquad B\in\mathbb R^{n\times n},
\quad W\in\mathbb R^{n\times r},
\]

再且只再加一套

\[
F_{\rm Kutta}(\phi,g,g^n,X)=0\in\mathbb R^r .
\]

旧基线采用 Morino closure

\[
g-C\phi=0 .
\]

消去 \(g\) 后恰得当前方阵

\[
(B+WC)\phi=b.
\]

这表明 `g=Cφ` 在这里是旧 Kutta closure，而不是必须与新的 birth law
同时强加的第二个拓扑公理。新的 birth-flux 或 unsteady pressure Kutta
应研究为它的物理替代。

文献边界必须保持：

- pressure Kutta 是压力连续的非线性闭合，不自动等于三维 LEV 形成律；
- 二维 unsteady Bernoulli 的势跳时间导数只锚定方程角色，不能直接外推
  到展向流、扑翼 LEV 或有限厚 base；
- triple-DOF row replacement 不证明当前 Galerkin quotient。

## ③ 缺件还是错件

| 命题 | 裁决 |
|---|---|
| \(p=9\) 个 cut 节点给出 9 个独立 Kutta 模态 | 错件；实际 \(r=7\) |
| body BIE 存在 7 个拓扑冗余 test modes | 与满秩指纹冲突 |
| 删除 test modes 后再同时强加 Morino 与 birth law | 错件；人为制造方程配平 |
| 独立 \(g\) + 全部 body BIE + 一套 Kutta closure | 文献支持的缺组件 |
| S3ae birth flux 足以替代 Morino 并满足全局有限速度 | 尚未验证，保持 open |
| finite NACA-2406 base 等价于单 sharp junction | 已证伪，冻结不动 |

## ④ 方案与 go/no-go

S3af 先做无压力、无力的 equation-role oracle：

1. 从 \(C_{\rm full}\) 提取 7 个非零独立行 \(C\)，验证两个端点零行；
2. 验证连续 prolongation \(P\)、paired injection \(K=\tfrac12C^T\) 的
   窄义拓扑恒等式，但不把它解释成删 PDE 方程；
3. 从现有 eliminated wake operator 做精确右逆分解
   \(A_w=WC\)，只用于只读代数审计；
4. 检查 \([B\ W]\) 为 81 行满秩、右零度为 7；
5. 解
   \[
   \begin{bmatrix}B&W\\-C&I\end{bmatrix}
   \begin{bmatrix}\phi\\g\end{bmatrix}
   =
   \begin{bmatrix}b\\0\end{bmatrix}
   \]
   并与冻结的 eliminated solve 逐项对照；
6. 对四种可逆 wake 坐标基验证物理解协变；
7. 用 Euclidean 与 surface-\(L^2\) 两种 quotient 重做同一制造问题，
   同时报告各自 projected residual、attachment residual 和**完整**
   body-BIE residual。

只有 `algebraic-role GO` 与 `quotient NO-GO` 同时复现，才允许改写树：

- 原 `...b3b` 的“必须删除 test modes”部分判
  `falsified/frozen`；
- 新增“独立 wake jump + 一套 Kutta closure”开放/局部验证节点；
- 下一物理门研究 birth-flux 是否能**替代** Morino，而不是同时追加。

该门不授权 pressure、force、Fig17/18/19、118 工况、finite base 或生产
激活。若失败，只能检查 cut rank、取向或矩阵装配；禁止通过挑行、质量矩阵
选择、least squares、阻尼、核半径或载荷目标修补。

## S3af 执行结果

预登记门原样执行，所有 algebraic-role checks 与 quotient 反例复现 checks
均通过：

| 指标 | 结果 |
|---|---:|
| 完整 cut 节点 / 独立 jump 秩 | `9 / 7` |
| base / classified body DOF | `74 / 81` |
| `rank(B)` | `81` |
| `[B,W]` rank / nullity | `81 / 7` |
| wake factorization 最大误差 | `8.47e-22` |
| Morino block rank | `88` |
| block 与旧消元解最大差 | `2.96e-15` |
| 四种 wake 坐标基协变误差 | `9.02e-16` |
| 完整 BIE 归一化残差最大值 | `4.30e-16` |
| quotient 投影算子差（operator norm） | `0.609106` |
| 两 quotient 解最大差 | `2.20429e-3` |
| 两 quotient projected residual 最大值 | `3.08e-18` |
| Euclidean / surface-L2 完整 BIE 相对残差 | `7.365e-3 / 9.284e-3` |
| 输入 mutation | `0` |

这给出两个不同层级的裁决：

1. **validated/frozen（窄义代数）**：active wake trace 有 7 个独立
   unknown；保留全部 81 条 body BIE 后恰缺一套 7 维 Kutta closure。
   Morino block 是当前消元方阵的严格等价展开。
2. **falsified/frozen**：classified cut 拓扑并不唯一要求删除 7 个 body
   test modes。小 projected residual 掩盖了被删除的完整 BIE residual，
   且结果依赖所选 metric。

因此下一可动节点不是另造 \(Q\)，而是
`N3.1j3b6d18c2b3b3b2c2b2b3e`：在独立 \(g\) 表示中验证守恒
birth-flux 或 weak unsteady-pressure Kutta 是否能作为**唯一闭合**
替代 Morino，并同时通过 Schur rank、junction 有限速度和压力一致性门。

## 一手来源

- Chouliaras et al., *CMAME* 373 (2021) 113556,
  <https://doi.org/10.1016/j.cma.2020.113556>.
- Wang, Abdel-Maksoud & Song, *Ocean Engineering* 130 (2017) 398–406,
  <https://doi.org/10.1016/j.oceaneng.2016.12.009>.
- Cattarossi et al., *Computers & Mathematics with Applications* 206
  (2026) 257–279,
  <https://doi.org/10.1016/j.camwa.2026.01.021>.
- Erickson, *Panel Methods—An Introduction*, NASA TP-2995,
  <https://ntrs.nasa.gov/citations/19910009745>.
