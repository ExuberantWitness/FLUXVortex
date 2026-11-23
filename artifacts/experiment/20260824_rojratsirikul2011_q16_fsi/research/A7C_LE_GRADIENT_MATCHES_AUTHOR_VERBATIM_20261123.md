# A7c 仲裁：前缘 ∂Γ/∂x 的写法与作者**逐字一致**（并溯源到 Ghommem 2011 p.138）—— A7b 的"实现错误"嫌疑解除（2026-11-23）

> 源：`FSI_by_FEM_and_UVLM/single_sheet/cores/solver/fluid/calc_fluid_force.m:32-43`（作者 MATLAB 原码，在仓库内）

## 1. 作者原式

```matlab
%%[*] 微分による先頭要素は，dx_Γ1 = ( (Γ_i − Γ_{i−1})/Δx )|_{i=1} = Γ_1/Δx とする．
%%% M. Ghommem, Modeling and Analysis for Optimization of Unsteady Aeroelastic
%%% Systems, Doctoral dissertation of Virginia Polytechnic Institute and
%%% State University, p. 138, 2011.
Gamma_mat  = reshape( Gamma, Ny, []).';
dx_Gamma   = [ Gamma_mat(1,:);  diff( Gamma_mat, [], 1)]./d_x_mat;

Gamma_mat2 = [ zeros(Nx,1)  Gamma_mat   zeros(Nx,1)];
dy_Gamma   = (Gamma_mat2(:,3:end) − Gamma_mat2(:,1:end−2))./(2*d_y_mat);
dy_Gamma(:,1)  =  Gamma_mat(:,1)./d_y_mat(:,1);
dy_Gamma(:,end) = −Gamma_mat(:,end)./d_y_mat(:,end);
```

## 2. 与我们的实现逐项对照

| 项 | 作者 | 我们（`q16_flux_v5m_native.py:1830-1835`） | 一致 |
|---|---|---|---|
| 前缘 `∂Γ/∂x` | `Γ(1)/Δx` | `dx_gamma[0] = grid[0] / dx[0]` | ✅ |
| 内部 | `diff(Γ)./Δx`（后向） | `(grid[1:] − grid[:-1]) / dx[1:]` | ✅ |
| 展向内部 | 中心差分 `/(2Δy)`，两侧零填充 | 同 | ✅ |
| 展向两端 | `Γ/Δy`、`−Γ/Δy` | 同 | ✅ |
| 方向余弦 | `tau_x./d_x_vec` | `tau_x / ‖tau_x‖` | ✅ |
| `dp_lift` | `Σ V_surf·(τx·∂xΓ + τy·∂yΓ)` | `ρ·Σ external_flow·gradient` | ✅（ρ 在我们这边显式） |

## 3. 结论

1. **前缘梯度不是"实现与作者不一致"。** 它是作者沿用 Ghommem (2011, p.138) 的**既定离散**，我们的移植逐字一致。
2. 因此 A7b 里"4 种约定相差 −54%…+92%"应读作：**该方法对前缘离散敏感**，而不是"我们写错了"。按"禁止逐工况调参"的纪律，**不得**为了让 Cn 对上论文而改这一行。
3. A7b 的价值不变——它证明了 `dp_lift1`（占载荷 76.7%）由前缘面板的等效梯度主导；只是修改它的正当性不来自"修 bug"。

## 4. 对 +22% 的含义（更新后的候选表）

| 候选 | 状态 |
|---|---|
| 口径（升力/法向） | ✅ 已修（−9.76%） |
| 诱导速度项（尾迹/LEV/Mf2_1） | ✅ 二阶量，已排除 |
| **前缘 ∂Γ/∂x 约定** | ✅ **与作者一致，非 bug** |
| `mf2 = ρΔΓ/Δt` 非定常项（占 23.3%） | ⚠️ 与作者一致（`dp_add = (Γ − Γ_old)/Δt`，`:52`），但**在 `bound_rate` 下它是否应与 Mf1 并存**仍未定（A4-3slot 显示去掉 Mf1 反而更高） |
| AIC 算子 | ⚠️ 仍未被独立验证（A7 两次校验失败） |
| **物理残差**（低 Re、深度分离、拱后二次分离） | ⚠️ 成为最可能的剩余解释 |

## 5. 边界

- 只对照了**强格式** `calc_fluid_force.m`；`calc_fluid_force_weak.m`（作者弱式）未对照，但弱式改变的是 `dp_add` 与 Mf1/Mf2_1 的取舍，不改变 ∇Γ。
- 作者代码的注释是日文（编码显示为乱码），"先頭要素 = Γ₁/Δx"是我按公式与引文读出的。
- 本文**没有**核对 Ghommem 2011 原文 p.138 的推导，只确认了作者引用了它。
