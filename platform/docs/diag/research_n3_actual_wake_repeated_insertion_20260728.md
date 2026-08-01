# N3 S3ac：actual old-state/newborn repeated insertion

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2b2b1`  
状态：**EXECUTED / NO-GO**

## ① 病因定位

S3ab 已排除“63-DOF 初始向量”这个错误问题：newborn 带是 characteristic
space-time inflow，不拥有人工初值。actual 门仍缺一个组成：

- old 45-DOF material state 要在释放后继续 ALE 输运；
- 旧 body row 一旦释放，就从 algebraic body trace 转为 material seam；
- newborn current row 成为新的 algebraic body trace；
- old-state transport、newborn inflow 与 actual body solve 必须处于同一个
  midpoint/full composition，不能各自推进后再 copy seam。

可动空间只限于 `c2b2`。S3x、S3z、S3aa、S3ab 都已 frozen，不修改其
离散公式、核、求积或阈值。

## ② 学科机理

Krebs (2021/2022) 的 relaxed-wake 顺序是：创建新生 wake、联立 surface/
newborn strength、移动既有 material wake、再重新满足 surface/newborn
约束；已释放 strength 保持 material identity。Dziuk–Elliott 的 ALE
transport theorem 则要求 tangential mesh gauge 与 material scalar 通过
relative tangential advection配对。S3ab进一步证明 newborn 部分必须由
release-time inflow trace生成。

因此 actual step 必须做一个 typed domain decomposition：

1. **old material subdomain**：用 validated consistent-P2 ALE 形式输运；
2. **birth space-time subdomain**：用 `[released seam, g_mid, g_end]`
   的 inflow identity形成 newborn band；
3. **current body boundary**：在 half/full geometry 上用 validated S3aa
   九维 actual boundary residual 解 `g_mid/g_end`；
4. 三者只通过具名 seam/trace 耦合，不通过全局数组补值。

## ③ 缺件还是错件

| 组成 | 裁决 |
|---|---|
| fixed-topology old-state midpoint transport | S3x validated/frozen |
| weak normal release/geometry gauge | S3z validated/frozen |
| half/full actual algebraic trace solve | S3aa validated/frozen |
| zero-area birth representation | S3ab validated/frozen |
| old body row 继续当 algebraic trace | 角色错误，释放后必须 materialize |
| actual domain-decomposed composition | **缺件** |

## ④ S3ac 预登记

对每个 step：

1. 在 initial old topology 上用 actual four-channel field做 global weak P1
   normal projection，并组装 full old-state P2 ALE operator。
2. 用 explicit midpoint predictor把所有 old P1/P2 identities推进到 half；
   原 body row在这一步作为 released material row，不再 clamp。
3. 插入 half newborn predictor `[seam_half, seam_half, g_half]`，只用 S3aa
   affine actual residual求 `g_half`。
4. 在完整 half actual solution 的物理场中，仅对 old material subdomain
   重组 weak geometry/P2 transport operator。
5. 用 midpoint operator把 old P1/P2 identities推进到 endpoint。
6. 用 S3ab identity组成 full newborn inflow
   `[seam_end, g_half, g_end]`，并用 S3aa求 `g_end`。
7. 对固定物理时间窗运行 `3/6/12` 次插入，比较共同的 original-material
   P1/P2 identities和最终 body trace。

GO 需要：

- 每步 band count 严格 `B→B+1`，无 anonymous scalar；
- half/full old-state方程残差、actual boundary residual、seam、P2 roundtrip
  全部闭合；
- S3ab endpoint identity逐 step闭合；
- 原始材料身份不丢失，输入不变；
- 三个步长族至少在 original-material geometry、P2 state、body trace 中
  显示二阶 Cauchy；
- 无 geometry iteration、remap、copy/clamp、epsilon、pressure、force、
  LESP、target 或 structure。

NO-GO 时保留最先失败的 named residual。不得降阈值或回到 63-DOF 初值法。

## 执行结果与病因改写

完整门在形成 Cauchy 结果前异常终止。持久化 12 步族重放证明异常不是首因：

| stage | wake \(|\mu|_\infty\) | wake \(|x|_\infty\) | body doublet | wake sheet | total velocity |
|---|---:|---:|---:|---:|---:|
| step 1 initial | 0.0732 | 1.50 | 68.70 | 68.94 | 0.990 |
| step 1 midpoint | 0.0926 | 1.50 | 80.48 | 67.80 | 13.92 |
| step 2 initial | 0.0926 | 1.50 | 1.3171e7 | 1.3168e7 | 1.4003e4 |
| step 2 midpoint | 663.0 | 1.50 | 9.1983e5 | 8.9402e5 | 3.8302e4 |

step 2 initial 的 actual matrix condition 仍只有 `55.06`，weak residual 为
`3.47e-16`，势和几何尚未爆炸；首个失效是 newborn 带贴近 body–wake
junction 后，两个近奇异大速度通道不再保持有限的联合极限。随后 endpoint
势跳变为 `1.108e4`，mass condition 变为 `8.388e5`，到 step 4 midpoint
才出现 off-plane 浮点异常。

因此：

- S3x/S3z/S3aa/S3ab 的冻结窄义身份不改；
- “值迹连续＋分通道求值＋explicit midpoint 足以连续 birth”被证伪；
- 不能以 implicit time integrator、wake panel length、epsilon、core、clamp
  或 damping 修复；
- 下一门必须先检验 newborn 尺度趋零时的涡量/速度有界性，再区分
  unsteady-Kutta/vorticity-continuity 缺件与 close-evaluation 缺件。

新裁决见
`research_n3_actual_wake_birth_junction_decision_20260728.md`。
