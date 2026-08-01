# N2.6c1b2b 独立真实场时序插值裁决

日期：2026-07-28  
对象：Edinburgh Ōtomo `32_sym` time-resolved PIV，DOI `10.7488/ds/7677`  
边界：只审计 Eulerian 帧序列的时间插值；不定位分离、不推进生产材料面、
不接入载荷。

## 1. 病因定位

`N2.6c1b2a` 已证明 RK4 对给定光滑解析速度场四阶收敛，但这不证明实验 PIV
帧之间存在足够精确、平滑且带不确定性边界的速度插值。该缺口挂在
`N2.6c1b2b`，不在 N1 UVLM、N4 簿记或 RK4 公式本身。

预登记文件 `external_piv_temporal_interpolation_cases.yaml` 在抽取相邻帧
前冻结：

- 中心帧 833；
- 对称 stride `1,2,4`，对应半窗 `0.006,0.012,0.024 s`；
- 线性中点误差与单边 persistence 误差之比；
- 细窗中点必须优于 persistence；
- 窗口增大时中点误差必须不减；
- 坐标必须不漂移，所有值必须有限；
- 即使 GO 也禁止物理晋升。

## 2. 数据指纹

七帧均为 `15,876×4` 的 `x,y,u,v`，固定 `126×126` 网格。坐标最大残差为
0，所有值有限。文件 checksum 和逐帧指标记录在
`external_piv_temporal_interpolation_results.json`。

| 分量 | stride 1 中点/persistence | 中点误差，stride 1/2/4 (m/s) | 预门 |
|---|---:|---:|---|
| `u` | 0.7190 | 0.01464 / 0.02213 / 0.01691 | NO-GO：不单调 |
| `v` | 0.8612 | 0.005618 / 0.005925 / 0.006279 | GO |

总裁决为 **NO-GO**。最细时间间隔的线性中点对两个分量都优于单边保持，
但 `u` 通道没有表现出随时间窗收缩而一致降低的插值误差。因此不能声称
“原始发布帧 + 朴素线性时间插值”形成了收敛的材料轨迹驱动场。

## 3. 学科机理

该结果与三类原始学术结论一致：

1. Sciacchitano (2019) 将 PIV 不确定性分为测量链中的随机和系统误差，并
   强调必须把速度不确定性传播到派生量。只看到有限值不等于获得了轨迹
   误差带。
2. Mancho et al. (2006) 对离散速度场的粒子轨迹插值比较表明，空间插值
   表现不能预测加入时间插值后的表现；其光滑混沌基准中，bicubic 空间加
   三阶 Lagrange 时间插值优于低阶组合。这支持“时间重构是独立组件”，但
   不能直接把其方案无验证移植到含测量噪声的 PIV。
3. Vocke et al. (2021) 指出 PIV 的遮挡、反光、低光强和错误向量会形成
   mask；其 Lagrangian 补全是在有真值/遮挡基准的条件下比较，而不是把
   不可见近壁区任意平滑出来。

Weldon et al. (2008) 的实验 material-spike 工作也使用实验观测与独立数值
剪切/压力预测对照，而不是把无壁面身份的 PIV 插值结果自身当作分离真值。

## 4. 缺组成部分还是组成部分错误

### 已证伪的组成方式

```text
原始二维PIV帧
  → 逐点线性时间插值
  → 无误差状态的材料轨迹
```

该方式在预登记中心窗上没有通过收敛预门，登记为
`N2.6c1b2b0` falsified/frozen。禁止通过删除 stride=4、只看通过的 `v`
通道或改门槛重启。

### 缺少的组成部分

真正缺少的是一个“**带观测身份和误差传播的场重构组件**”，至少需要：

- 作者提供的 wall/body mask、无效矢量和 PIV a-posteriori uncertainty；
- 明确的壁面位置/速度和网格参考系；
- 在 withheld 连续帧上的空间—时间插值误差；
- 将速度场不确定性传播到 flow map、Weingarten change 和 backbone 位置；
- 数值场则需输出步长/保存频率加密序列，以区分插值截断误差与真实动力学。

这仍不能修复该数据集的低 Re、二维和无 edge 身份，所以它只影响数值实现
风险，不改变 `N2.6b4f3b` 的目标数据缺口。

## 5. 方案与 go/no-go

### NO-GO

- 不对 Edinburgh 帧加任意平滑后重跑同一门；
- 不把三阶时间插值当作必然正确的替换；
- 不从 PIV 图像人工描出壁面后宣称满足数据契约；
- 不以更好的总力拟合作为插值方案选择依据。

### 后续 GO 前提

只有取得带 mask/UQ/壁面身份的数据后，才预登记比较：

1. 线性、三阶时间和物理约束重构；
2. withheld 帧误差与时间步收敛；
3. 无滑移和质量守恒残差；
4. flow-map/曲率/backbone 的不确定性传播；
5. 方案选择完全不读取力、压力残差或 V4.1 误差。

## 6. 参考文献

- Sciacchitano, A. (2019), *Uncertainty quantification in particle image
  velocimetry*, Measurement Science and Technology 30, 092001,
  DOI:10.1088/1361-6501/ab1db8.
- Mancho, A. M. et al. (2006), *A comparison of methods for interpolating
  chaotic flows from discrete velocity data*, Computers & Fluids 35,
  416–428, DOI:10.1016/j.compfluid.2005.02.003.
- Vocke, M. et al. (2021), *Lagrangian interpolation algorithm for PIV
  data*, International Journal of Heat and Fluid Flow 89, 108733,
  DOI:10.1016/j.ijheatfluidflow.2020.108733.
- Weldon, M. et al. (2008), *Experimental and numerical investigation of
  the kinematic theory of unsteady separation*, JFM 611, 1–11,
  DOI:10.1017/S0022112008002395.
