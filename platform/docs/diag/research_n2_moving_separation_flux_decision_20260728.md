# N2.6c2c：移动分离流形的相对 IBL 通量不是释放闭合

## 1. 病因、节点与可动空间

现有 `N2.6b3` 已冻结移动曲面上的三维 IBL 广延库存和物理边界通量：

```text
U_M = M
F_M · nu = T · nu

U_E = tr(T)
F_E · nu = (E - T U_e) · nu .
```

现有 `N2.6c2b` 也已冻结新生 vortex-entrainment sheet（VES）必须独立
保存 `gamma`、`q`、`rho_s`、`rho_s v` 和压力跳的状态身份。但
`N2.6b3` 的边界通量仅针对固定在材料曲面坐标中的控制边；`N2.6c1` 的
分离流形却会相对翼面材料点迁移。若直接把 `F · nu` 登记为释放率，就遗漏
了移动边界扫过库存的 `-c_rel U` 项。

这个病因是一个守恒接口缺件，不是载荷误差常数，也不是 VES 状态幅值律。
可动空间只限于开放的 `N2.6c2c`。冻结的 N1、`N2.6b3`、`N2.6c2b`、
统一面板压力和 ForceLedger 均不改。

## 2. 学科机理

Cermelli、Fried 与 Gurtin（JFM 544, 2005）对带边界的演化曲面建立了
surface transport relation。对随翼面材料速度运动的曲面区域
`A(t)`，其内部边界再以有符号切向共法向速度 `c_rel` 相对翼面迁移，
表面 Reynolds 输运定理给出

```text
d/dt ∫_A U dA
  = ∫_A (D_w U/Dt + U div_s(v_w)) dA
    + ∮_(∂A) c_rel U ds .
```

若材料曲面上的局部守恒律为

```text
D_w U/Dt + U div_s(v_w) + div_s(F) = S ,
```

则

```text
d/dt ∫_A U dA
  = ∫_A S dA - ∮_(∂A) (F·nu - c_rel U) ds .
```

因此从保留的附着区向外的相对通量唯一是

```text
F_relative,out = F·nu - c_rel U .
```

这里 `nu` 是附着区的外向单位表面共法向；`c_rel>0` 表示边界沿 `nu`
扩张附着区，因而边界扫入库存并减少净出流。这个符号同时由平面移动区间
和球面移动纬线的解析面积率确定。

DeVoria 与 Mohseni（JFM 866, 2019）只完成尖锐边 VES 合流。其论文明确把
“光滑曲面上可移动分离点”留作计划中的第二部分；截至本次检索没有找到
可用于生产晋升的公开第二部分。因此尖锐边的角度、强度或内禀冲量条件
不得移植到 RoboEagle 光滑前缘。

## 3. 缺件还是错件

### 错件

```text
固定边界物理通量 F·nu
    == 穿过移动分离流形的释放通量
```

是错误组成部分。只要 `c_rel != 0`，它就违反移动控制区的广延库存守恒。

### 缺件

缺少一个只读、无闭合的相对通量投影：

```text
R_M = [T·nu - c_rel M] ds
R_E = [(E - T U_e)·nu - c_rel tr(T)] ds .
```

该投影需要调用者显式提供：

1. 完整 IBL 状态和物理通量闭合；
2. 有侧别的外向共法向 `nu`；
3. 分离流形相对材料翼面的有符号速度 `c_rel`；
4. 曲线积分测度 `ds`。

它不生成 `gamma`、`q`、`rho_s`、`rho_s v`、压力跳、自由片方向或位置。
那些量仍必须由 `N2.6b4f + N2.6c1c` 的真实近壁场和 `N2.6c2c` 物理
junction 闭合。

## 4. 预登记方案与 go/no-go

在实现前冻结 `moving_separation_flux_cases.yaml`：

1. 固定边界严格退化为既有 `surface_ibl_physical_flux`；
2. 平面扩张/收缩控制区满足解析广延库存率；
3. 球面移动纬线满足解析球冠面积率；
4. 物理输运和边界扫掠严格线性叠加；
5. proper rotation 下动量率随坐标旋转，能量率不变；
6. 共法向和相对速度同时反向时，通量严格反号；
7. 非切向共法向、非有限速度、非正测度和维度错误必须失败。

全部通过只允许冻结：

> 给定完整 IBL 状态、定向移动流形及相对速度时，相对库存通量的
> Reynolds 输运恒等式已正确实现。

即使通过，`physical_promotion=false`；不得声称已预测分离流形、LEV
供给或气动力，也不得把 IBL 亏损动量/能量率改名为 VES 的实际质量、
动量、环量或卷吸率。

## 5. 原始来源

- Cermelli, P., Fried, E. & Gurtin, M. E., *Transport relations for
  surface integrals arising in the formulation of balance laws for evolving
  fluid interfaces*, Journal of Fluid Mechanics 544 (2005), 339–351,
  doi:10.1017/S0022112005006695.
- Gurtin, M. E., Struthers, A. & Williams, W. O., *A transport theorem for
  moving interfaces*, Quarterly of Applied Mathematics 47 (1989), 773–777,
  doi:10.1090/qam/1031691.
- DeVoria, A. C. & Mohseni, K., *The vortex-entrainment sheet in an
  inviscid fluid: theory and separation at a sharp edge*, Journal of Fluid
  Mechanics 866 (2019), 660–688, doi:10.1017/jfm.2019.134.

## 6. 预登记结果与 claim 裁决

`moving_separation_flux_guard.py --write` 在不改变预登记阈值的情况下
通过全部门：

```text
平面扩张/收缩库存残差       2.78e-17 / 0
球冠动量/能量库存残差       6.77e-15 / 1.11e-16
物理通量+扫掠叠加误差       2.78e-17 / 0
旋转动量/能量残差           1.91e-17 / 1.67e-16
方向反转反对称误差           0 / 0
非法输入拒绝                 4 / 4
```

因此 `N2.6c2c1` 晋升为 **validated/frozen**。父节点 `N2.6c2c` 保持
**open**：新算子只提供 IBL 动量亏损与能量亏损的相对出流率，尚未提供
VES 的实际质量、动量、环量、卷吸、压力跳和新生片几何。结果文件中的
`physical_promotion.eligible=false` 是硬边界。
