# N3 P1：压力跳可观测域与双侧厚翼压力缺件

日期：2026-07-28  
作用域：裁决统一面板压力能被公开数据验证到哪一级，以及弦向压力力还缺什么；
不修改 V4.1、N1/N4、网格、常数或结构模型。

## 0. 结论

```text
gauge-free ΔCp(x,phase) / Cn observation：GO
由 ΔCp 唯一恢复 Cp_upper/Cp_lower：FALSIFIED
由 ΔCp 唯一恢复厚翼 profile Ct：FALSIFIED
完整双侧压力缺件：actual-thickness mean-potential/source + viscous separation
FLUXV external pressure comparison：NOT EXECUTED
```

这修正了 P0 后过宽的“公开压力已经可以验证整个统一压力”表述。公开数据当前
能直接验证的是薄升力面压力跳及其法向积分；不能仅凭压力跳验证厚翼弦向压力
力或两个绝对侧压。

## 1. ① 病因定位

### 1.1 数据规律

4TU DU-97-W-300/VG v2 的 244 个坐标严格分成：

```text
xup1..xup122：LE→TE
xlo122..xlo1：TE→LE
```

反转 lower 分支后，上下 `122` 个 `x/c` 的最大差为 `0`。定义

```text
ΔCp(x,phase) = Cp_lower - Cp_upper
Cn = ∫_LE^TE ΔCp dx .
```

15 个工况的 `Cn` 最大轮回误差为
`4.983908841182938e-10`。

但以下构造给出直接非唯一性：

```text
g(x,phase) = 0.2 sin(phase) (x-0.5)
Cp_upper* = Cp_upper + g
Cp_lower* = Cp_lower + g .
```

它使 `ΔCp` 最大变化仅 `8.88e-16`、`Cn` 最大变化仅 `8.88e-16`，却使同一
厚翼几何的 `Ct` 最大变化 `0.0335124`。因此任何“由压力跳选一个两侧分摊，
再声称得到 profile drag”的方案都不可辨识。

### 1.2 挂树

- `N3.1j3b4`：压力跳唯一决定双侧 Cp 与厚翼 Ct，**falsified/frozen**；
- `N3.1j3b5`：公开双侧 Cp 到 gauge-free ΔCp/Cn 的观测身份，
  **validated/frozen**；
- `N3.1j3b6`：实际厚度上的均势/源面与黏性侧压组件，**open**；
- `N3.1j3b` 父节点保持 partial；
- `N3.1j3a` 已验证的统一压力跳恒等式不变。

### 1.3 可动空间

允许：

- 用 `ΔCp/Cn` 外验薄升力面压力跳；
- 在 shadow 路径研究 actual-thickness 双侧势/压力；
- 将 N2.6 的黏性库存/分离释放通过同一双侧压力账进入最终载荷。

禁止：

- 任意把 `ΔCp` 按 1/2 分到上下表面；
- 用公共 Bernoulli gauge 拟合 `Ct`；
- 把静态极曲线或 `d_para` 加到同一 profile-pressure 账上重复成力；
- 用 DU-97 的 `Cp/Ct` 调整 RoboEagle 的 `LESPcrit/cds/Tv/Tvl`。

## 2. ② 学科机理与一手来源

### 2.1 薄升力面只规范压力跳

当前 `N3.1j3a` 的

```text
Δp = rho [D_wall(chi)/Dt + (ubar-v_wall)·grad_s(chi)]
```

由势跳 `chi=phi_plus-phi_minus` 决定，并且公共 Bernoulli gauge 严格消失。
它是薄升力面上规范的法向载荷变量。

`N3.1j3b1` 也只证明：**在给定**均势率、主值速度和两侧极限时，两侧
Bernoulli 之差回到同一个压力跳。它没有证明均势率或每侧绝对压力已由生产
算子提供。

### 2.2 真实壁面侧压需要额外状态

Morino 的任意运动升力体 Green-function 理论把真实物面上的势与由无穿透
边界给出的法向导数联系起来，并特别指出厚度趋零时算子会退化/奇异。Dusto
与 Epton 的非定常面板法进一步把 source 与 doublet 分布放在**实际厚体
表面**上。两者共同说明：厚体的 mean potential/thickness flow 不是薄面
势跳可以随意补出的公共模态。

Terrington、Hourigan 与 Thompson 的三维壁面涡量生成式又要求每个真实壁面
侧的切向压力梯度与壁面加速度。仅知道两侧压力差，不能替代两个真实侧面的
压力梯度。

一手来源：

- Morino, *A General Theory of Unsteady Compressible Potential
  Aerodynamics*, NASA CR-2464:
  https://ntrs.nasa.gov/citations/19750004821
- Dusto & Epton, *An Advanced Panel Method for Analysis of Arbitrary
  Configurations in Unsteady Subsonic Flow*, NASA CR-152323:
  https://ntrs.nasa.gov/citations/19800017771
- Terrington, Hourigan & Thompson, *Journal of Fluid Mechanics* 936
  (2022), A44:
  https://doi.org/10.1017/jfm.2022.91

### 2.3 势流厚度组件仍不足以闭合动态失速

source/doublet 厚体面可以给出无黏 attached mean-pressure 基线，但不能单独
生成分离后的黏性压力亏损、转捩和卷吸释放。生产方向必须是：

```text
frozen N1 circulation / spatial free-vortex state
        + actual-thickness no-penetration mean potential
        + N2.6 viscous inventory and conservative release
        ↓
one dual-side unsteady pressure ledger
        ↓
one surface-force integration
```

它不是在 N1 后面再加一个经验 drag。

## 3. ③ 缺组件还是组件错误

### 3.1 错组件

把 `N3.1j3a` 的压力跳称作完整双侧 `Cp`，或者由它唯一计算厚翼 `Ct`，是表示
组件错误。公共压力模态反例已经构造性证伪，不依赖任何实验噪声或模型参数。

### 3.2 缺组件

完整空间载荷仍缺：

1. actual-thickness 上/下表面气动几何；
2. 移动物面的 source/no-penetration 解与 mean potential history；
3. 由远场条件固定的公共 Bernoulli gauge；
4. 与 N2.6 黏性/分离状态一致的侧压修正；
5. 双侧压力只积分一次的唯一簿记。

这也是现有 `N2.2` 全局升力向砍量无法产生正确弦向分离压力力、`N2.5` 只能
作为过渡极曲线替代件而非最终生产组件的根本原因。

## 4. ④ 方案与预登记

### 4.1 P1a：立即可做的压力跳外验

冻结输出：

- `ΔCp(x,phase)`；
- suction-side/pressure-side 差值峰的位置、幅值和相位；
- 同一 `ΔCp` 的 `Cn`；
- 不比较 `Cp_upper/Cp_lower/Ct`；
- 首个 case 必须是 no-VG；VG case 在模型没有显式 VG 物理时只作数据账，
  不作模型优劣裁决。

但当前空间态尚未通过 V-gate，且没有 DU-97 外部几何求解适配器，所以
`model_comparison_executed=false`。

### 4.2 P1b：actual-thickness 双侧压力 shadow

实现前 GO 条件：

1. source 强度只由逐时刻实际表面无穿透/壁面速度决定；
2. doublet/势跳只来自 N1 与通过空间门的空间涡态，不从压力目标反演；
3. 远场 pressure/potential gauge 显式固定且 gauge 变换不改变力；
4. 静止圆柱/已知厚翼 attached Cp、刚体平移/俯仰非定常势通过收敛门；
5. 双侧压力差逐面板退化到 `N3.1j3a`；
6. source/thickness、bound、free-sheet 与 viscous 通道先在压力级合成，只成力
   一次；
7. 没有 N2.6 证据时，只能标为 inviscid shadow，不能吸收分离 drag。

任一项失败即 NO-GO。不得通过增加 pressure offset、Ct 常数或目标载荷
normalization 修复。

## 5. 可复查资产

- `pressure_jump_observation_cases.yaml`
- `pressure_jump_observation_guard.py`
- `pressure_jump_observation_results.json`
- `test_pressure_jump_observation.py`

