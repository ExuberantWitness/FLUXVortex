# N3.1j3b3 弯曲双层势与材料时间率裁决

日期：2026-07-28

## 1. 病因：压力公式已闭合，压力输入尚未闭合

`N3.1j3a` 已冻结统一压力账，

```text
delta_p = rho [D_wall(chi)/Dt + (u_bar-v_wall)·grad_s(chi)]。
```

`N3.1j3b1/b2` 又分别冻结了双侧 Bernoulli 代数和单平面 DDE 势跳。但这
仍不足以说明生产空间涡面能够提供压力输入：

1. 单平面 owner 主值为零，不能检验弯曲面上其他面元产生的非零主值；
2. 单体面求和不能检验显式 patch 分割是否改变势；
3. 瞬时势不能提供自由片相对壁面运动造成的
   `D_wall(phi_bar)/Dt`。

因此本节点挂在可动的 `N3.1j3b`，不修改已冻结的 `N3.1j3a/b1/b2`。

## 2. 学科机理

连续 doublet sheet 是双层势。Plemelj 极限说明两侧势由片上 Cauchy 主值
加减半个势跳得到；弯曲面主值包含整个曲面的非局部贡献，不能以 owner
平面自项代表。

Hirato 的非定常压力式要求势跳/环量的真实时间率，而不是 LESP 或经验滞后
时间常数。对随材料运动的连续涡面，势跳保持 Kelvin 身份，但观测壁面上的
平均势仍会因源面和壁面的相对运动改变。三个真实 stage 的二次 Lagrange
导数是时间离散身份；它不能补造缺失中点。

相关证据：

- Krebs (2021) dissertation，P2 distributed-doublet 势跳、全局连续和
  严格面内配点；
- Hirato et al. (2019), DOI `10.2514/1.C035124`，Eq.17 的统一非定常
  压力时间项；
- Terrington, Hourigan & Thompson, JFM 936 A44 (2022),
  DOI `10.1017/jfm.2022.91`，移动壁面侧别压力梯度和环量源。

## 3. 缺组件还是组件错误

裁决是“**缺一个显式材料势历史组件**”，不是现有 Bernoulli 公式错误：

- `doublet_potential.py` 的瞬时算子保持冻结；
- 新组件只组合三个真实 `surface + wall points + time` stage；
- P2 material `mu` 或拓扑若跨 stage 改变，立即失败；
- 不接受压力、力、LESP、结构量、平滑系数或时间常数。

实现位于 `claim_runtime/material_potential_history.py`，与压力成力模块隔离。

## 4. 预注册门与结果

`curved_material_potential_cases.yaml` 在命名 guard 实现/执行前冻结三组门。

### 4.1 单位球 degree-1 双层势

外法向单位球取 `mu=cos(theta)`。按当前符号，解析解为：

```text
phi_inside  = (2/3) r cos(theta)
phi_outside = -(1/3) r^(-2) cos(theta)
PV_surface  = mu/6。
```

octasphere 面数 `8/32/128/512` 的最大误差：

| 面数 | 离面势误差 | 片上主值误差 |
|---:|---:|---:|
| 8 | 0.113902 | 0.061425 |
| 32 | 0.048920 | 0.034254 |
| 128 | 0.014551 | 0.012226 |
| 512 | 0.003799 | 0.003441 |

两列均严格单调；最后 Cauchy 比分别为 `3.8302/3.5527`，通过预登记
`>=3.0`，最细误差均通过 `<=0.005`。

### 4.2 八 patch 表示不变性

把 128 面 octasphere 按八个原始八面体面显式拆分，再对各 patch 的 owner
主值和非 owner 势求和，与单体曲面最大差
`1.53e-16 <= 2e-12`。patch 表示没有改变标量势。

### 4.3 三时刻材料势率

令 level-2 源面以 `c_dot=0.25 e_x` 刚性平移，固定壁面点
`P=(2,0,0.6)`。解析离散参考来自独立速度 oracle：

```text
d phi(P-c)/dt = -c_dot · grad(phi)。
```

对半窗 `0.16/0.08/0.04/0.02`，端点三时刻 Lagrange 导数误差为：

```text
1.4933e-5, 3.6833e-6, 9.1479e-7, 2.2796e-7
```

相邻误差比分别为 `4.054/4.026/4.013`，通过二阶门 `>=3.8`；最细误差通过
`<=5e-7`，材料 `mu` 残差严格为零。

## 5. Claim 裁决

- `N3.1j3b3`：`validated/frozen`，仅限 CPU 弯曲势/patch/三时刻材料率
  数值身份；
- `N3.1j3b`：仍为 `partial`；
- `N3.1j3`：仍为 `open`；
- `N3.1j4b5b` 与 `N2.6c`：仍为 `open`；
- ForceLedger 和 V4.1 生产力：仍阻断。

该 GO 消除了统一压力路径中的一个数学/时间离散缺件，但没有制造 LEV
释放强度、目标域近壁场或真实面板压力证据。

