# N2.6a / N3.1j3b：移动壁面环量源与双侧统一压力

## 1. 问题与可动空间

冻结 N1 只给出中弧面束缚势跳及压力跳。它足够一次性形成薄翼法向力，却不能区分
吸力面和压力面的切向压力梯度，因此不能直接给出真实移动壁面每一侧的环量生成。
本案只增加 N1 的只读观察器；不修改 AIC、环量、网格、运动学或既有力。

## 2. 移动面 Bernoulli

在惯性系中，每侧无旋外流满足

```text
partial_t(phi_s) + 0.5 |u_s|^2 + p_s/rho = C(t),  s in {plus, minus}.
```

对速度为 `v_w` 的壁面物质点，

```text
D_w(phi_s)/Dt = partial_t(phi_s) + v_w dot u_s,
B_s = D_w(phi_s)/Dt - v_w dot u_s + 0.5 |u_s|^2,
p_s/rho = C(t) - B_s.
```

令

```text
chi = phi_plus - phi_minus,
u_bar = 0.5 (u_plus + u_minus),
u_plus/minus = u_bar +/- 0.5 grad_s(chi).
```

直接相减得到

```text
(p_minus-p_plus)/rho
  = B_plus-B_minus
  = D_w(chi)/Dt + (u_bar-v_w) dot grad_s(chi).
```

右端正是 `unified_panel_pressure.py` 已验证的压力跳，其中其
`local_velocity = u_bar-v_w`。所以双侧扩展不增加第二条压力定律，也不允许再次成力。
公共规范 `C(t)` 同时平移两侧压力，对压力跳和表面压力梯度均无影响。

## 3. 移动无滑移固壁的环量生成

Terrington、Hourigan 与 Thompson（JFM 936 A44, 2022）Eq. 4.30 给出每一物理
壁面侧的边界涡量通量/界面环量源

```text
sigma_s = n_s cross [a_w + grad_s(p_s/rho + Phi_g)].
```

这里 `n_s` 必须逐侧定义为约定的流体侧法向。压力梯度和壁面加速度造成相对无黏
加速度并生成界面环量；黏性使无滑移成立并把该环量转移进流体，而不是创造净环量。

对零厚度配对面，取 `n_minus=-n_plus`，并定义
`delta_p=p_minus-p_plus`，则

```text
sigma_plus + sigma_minus
  = -n_plus cross grad_s(delta_p/rho).
```

壁面加速度严格抵消。这只是配对代数守卫；要区分哪一侧向自由剪切层释放，仍必须
恢复 `p_plus/p_minus` 并求解 N2.6b 曲面边界层库存与 N2.6c 分离流形。

## 4. 不能越过的边界

- `sigma_s` 不是 LEV 自由片的直接幅值；
- LESP/BEF 是拓扑或事件指标，不是持续供给率；
- 双侧压力只能驱动 N2.6a，气动力仍由唯一压力跳一次性形成；
- 当前中弧面观察器不等同于厚翼双侧几何；曲率、侧别迁移和闭合翼面积分必须另设门；
- 在双层势主值、Plemelj 跳跃、材料时间导数和表面梯度收敛通过前，不接生产。

## 5. 文献锚

- Terrington, Hourigan & Thompson, *J. Fluid Mech.* 936 (2022), A44,
  doi:10.1017/jfm.2022.91，尤其 Eq. 2.36、4.18–4.20、4.30–4.31。
- Gehlert & Babinsky, *J. Fluid Mech.* 915 (2021), A50,
  doi:10.1017/jfm.2021.121。
