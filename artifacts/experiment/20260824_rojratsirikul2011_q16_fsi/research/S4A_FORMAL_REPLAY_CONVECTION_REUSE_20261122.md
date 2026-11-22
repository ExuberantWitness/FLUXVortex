# S4a：形式重放复用预对流场 —— 1.105×，载荷差 3.6e-10（2026-11-22）

> 钩子：`FLUXV_REUSE_CONVECTED_FIELD`（`case_runner._reuse_convected_field_choice`，**默认 0**）
> 数据：`/tmp/s4_off150.json`、`/tmp/s4_on150.json`（各 150 步，同窗口、同一棵树、只差该 flag）
> 前置：`S2B_STRUCTURAL_INTERIOR_ATTRIBUTED_20261122.md`（propose 1827 ms/步 = 32.7%）

## 1. 结果

| | 步价 | 提议数 | 耦合迭代均值 | `formal_replay_count` |
|---|---|---|---|---|
| 关 | **5.630 s/步** | 447 | 1.980 | 150 |
| 开 | **5.094 s/步** | 447 | 1.980 | 150 |
| | **1.105×**（−0.536 s/步） | 相同 | 相同 | 相同 |

载荷一致性（150 步全程）：

| 量 | 差 |
|---|---|
| `max abs(ΔCn)` / Cn | **3.59e-10**（步 138 取到） |
| `max abs(Δ\|zmax\|/c)` | **1.20e-12** |
| 窗口 101–150 平均 Cn | 相对差 **1.58e-12** |
| 窗口 101–150 平均 `\|zmax\|/c` | 相对差 **7.15e-12** |

粒子数与提议/耦合迭代结构完全一致 ⇒ 不是"少算了一步物理"，而是滞后一个耦合残差量级的对流。

## 2. 实现要点（两次失败先记下）

**① 不能复用 trial 状态本身。** 第一次实现把 `proposal.trial_state` 传回去，跑到容量墙：
`OverflowError: particle capacity exceeded by connected ribbon: 32689+270>32768`。
原因：`trial_state` 里**已经含本次 propose 的沉积**，重放再沉一次就是双份。正确做法是复用"**对流之后、沉积之前**"的状态，因此 `propose` 增加 `capture_preconvected`，在 `_advance_material_state`（+ 可选拉伸）之后、`native_aic`/沉积之前克隆一份，并随 `NativeV5MProposal.preconvected_state` 返回。

**② 必须复位 `trial.step`。** `propose` 末尾 `trial.step += 1`（`q16_flux_v5m_native.py:2048`），而复用源本身已是 `committed.step + 1` ⇒ 再加一被 `commit` 闸门拒绝（`native V5M commit must advance exactly one step`）。复用分支里把 `trial.step` 复位成 `committed.step`，这样 propose 内部所有对 `trial.step` 的用法（粒子 `birth_step`、LE 窗口 cutoff、龄期 horizon）与默认路径语义完全一致。

**③ 只复用于 `formal_replay`。** 重放的结构状态是 `(structural.state, structural.velocity)`，与最后那次试探的 `(q_guess, v_guess)` 只差收敛残差（≤5e-7 相对），这正是滞后量级。耦合迭代内部暂不复用（那是 S4b）。

## 3. 由测得量反推的单次对流成本

省下 1 次对流（2.98 次 propose 里的 1 次）= 0.536 s/步 ⇒ **单次对流 ≈ 0.536 s**，与 S2b 相位表吻合（`advance_wrk3` 1218.8 ms + 尾迹 RK3 的粒子/环部分 ≈ 1.5–1.8 s/步 ≈ 2.98 × 0.54）。

## 4. 默认值

保持**默认 0**：该路径不是逐位（3.6e-10），按仓库纪律"默认生产路径逐位不变"必须门控。打开方式：`FLUXV_REUSE_CONVECTED_FIELD=1`。
若接受 3.6e-10 的差异作为生产默认，需要一次显式确认（这不是本记录能替代的决定）。

## 5. S4b 的算术上界（未实施）

还可以再复用 2 次/步（耦合迭代 1.98 次里的其余次数），上界约再省 **2 × 0.536 ≈ 1.07 s/步** ⇒ 约 4.0 s/步。代价是滞后从"一个耦合残差"变成"整步内第一次猜测"，且可能在附加质量界面上改变不动点（耦合迭代次数可能上升），必须按 S2b 的相位表重测 `propose` 与 `coupling_iteration_mean` 才能判定净收益，并给出与默认路径同窗口的 `|ΔCn|`、`|Δzmax/c|`、`ΔSt` 三项对照。

## 6. 边界

- 单工况（A16）、单窗口（步 1–150）、fp64、4090 D；150 步窗口的 runner 自报 `stationary_window_found: false`，属早期瞬态窗。
- 只测了一个 flag 组合；`three_slot_mf1_cancel`、`bound_rate_author_weak` 与它的交互未测。
- 复用路径下 `retention` 里的 `particle_cull_*` 记 0（龄期剔除已在被复用的那次 propose 里发生过），并新增 `convected_reuse` 字段标识该分支。
