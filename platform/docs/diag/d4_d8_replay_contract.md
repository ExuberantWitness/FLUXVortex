# D4–D8 Research-Grade Replay Contract

**冻结时间**：2026-07-28T17:01:00+08:00  
**状态**：D3/N3.1 raw replay authorized; full D4–D8 remains preregistered only  
**execution_allowed**：`D3_ONLY`  
**production_change_allowed**：`false`  
**适用范围**：D4、D5、D6、D7、D8，以及 N5.1 的输入身份守卫  

本合同修复观测、指标、单位、数据处理、证据归档和执行守卫，不改变任何气动
公式。2026-07-29 主线复位后，原 S3ai-v2.2 one-shot 前置条件因其物理裁决为
`UNKNOWN` 且协议工程已停止扩展而撤销；只授权 D3 的
`U8/AoA5/f2.6/+90°/tw0,15,22.5,45°` 原始证据重放。D4–D8 其余工况仍禁止
执行。即使 D3 replay 全部通过，也只恢复局部机理判别的证据资格，不验证
N3.1h 空间涡态，不授权 118 或 Fig17/18/19 晋升。

### 0.1 数值运行时预条件

2026-07-29 的 fresh-process `off→on→on` 及其独立复现都得到首调用到
第二调用的 `L_inst` 最大差为 `0.4709081776444961 N`，第二与第三调用
逐字段 bitwise 一致。反转为 `on→off→on` 后，完全相同的差异跟随第一
个调用，而第二个 `observer-off` 与第三个 `observer-on` 逐字段差为零。
因此异常属于调用顺序相关的数值运行时瞬态，不属于 `claim_raw_out`
观测器效应。限制 `OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1` 未消除
该瞬态。

正式 D3 runner 必须先执行一次固定 tw0 完整生产网格预条件：

- 预条件结果不得进入任何物理指标或 shape gate；
- manifest 必须保存其调用、数值结果 bundle 哈希、L/T 与耗时；
- 预条件到正式 tw0 `observer-off` 的逐字段差异必须归档为被排除的
  cold→formal 数值瞬态，不得作为物理差异；
- 正式 `observer-off→observer-on→observer-on repeat` 仍必须逐字段
  bundle SHA256 一致，不能只用浮点最大差为零代替 bitwise 门。

这只是数值执行身份条件，不改变公式、参数、网格、运动学或 claim 状态。
由于预条件状态属于当前进程，正式 runner 禁止跨进程 resume；失败或中断后
必须使用新输出目录，或以 `--force` 完整重跑四点。正式执行必须包含并按
`0/15/22.5/45°` 顺序运行全部四点。Python、NumPy、Warp、device/GPU 和线程
环境均进入 `run_identity_sha256`。

## 1. 冻结工况

| 项目 | U | AoA | f | nominal twist | `twist_amp_deg` | phase | SPC |
|---|---:|---:|---:|---:|---:|---:|---:|
| D3/D4/D5 | 8 m/s | 5° | 2.6 Hz | 0/15/22.5/45° | 0/7.5/11.25/22.5° | +90° | 540 |
| D6 | 8 m/s | 5° | 2.0 Hz | 0/22.5/45° | 0/11.25/22.5° | +90° | 720 |
| D7 | 8 m/s | 5° | 2.0 Hz | 0/22.5/45° | 0/11.25/22.5° | −90° | 720 |
| D8 | 8 m/s | 5° | 2.0 Hz | 15° + D7 三点 | 7.5° + D7 三点 | −90° | 720 |

共同配置：

```text
closure=v41
nc=12
ns=16
n_cycle=4
wake_rows=steps_per_cycle
flap_amp_deg=22.5
real_geom=true
sym=true
```

正式 runner 必须保存 requested config、closure profile、YAML default、函数
default 和最终 resolved config；禁止只保存调用时显式写出的参数。

## 2. Raw channel 合同

每个 case 保存最后一个完整周期的逐时步、逐条带、未 clipping、未滤波、未
对齐数据。

### 2.1 时间、坐标与运动学

```text
step
time_s
dt_s
cycle_index
phase_solver_rad
phase_paper_rad
theta_rad
theta_dot_rad_s
psi_rad[strip]
psi_dot_rad_s[strip]
y_ref_edge_m
y_ref_center_m
eta_ref
chord_m
dy_single_m
panel_normal_body[strip,3]
```

定义：

```text
phase_solver = Omega*t mod 2*pi
phase_paper = (Omega*t - pi/2) mod 2*pi
eta_ref = (y_ref_center - root_offset)/half_span
dt = 1/(freq*SPC)
```

展向坐标使用未扑动的翼局部参考构型，禁止使用运动后的世界坐标。

### 2.2 N2/N3 状态分层

```text
A0_signed
lb_lesp_crit
a0_crit
A0_excess_pre_cds
dCN_drive_after_cds
dCN_drive_after_sign
dCN_drive_after_f2gate
dCN_state_after_memory
u_le_normal_signed_m_s
Urel_le_m_s
q_dyn_Pa
alpha_kin_rad
alpha_eff_lb_rad
event_active
event_onset
event_sign
formation_T_hat
f_qs
f2
K
CNc
CV_signed
CNv_signed
lb_ds_step
tau_v_pre
tau_v_post
tau_v_reset
```

精确定义：

```text
A0_excess_pre_cds = max(abs(A0_signed)-lb_lesp_crit, 0)
dCN_drive_after_cds = lb_cds*A0_excess_pre_cds
lb_ds_step = 2*Urel_le*dt/chord
```

当前 `_dCN_drive_raw` 已乘 `lb_cds`，不得再命名为 `A0_excess`。replay 必须
显式保存 before-cds、after-cds、after-sign、after-f2gate、after-memory 五层。

### 2.3 力、翼数与坐标系

每个力数组必须同时带 `wing_scope` 和 `frame`：

```text
qcdy_single_N
qcdy_mirror_pair_N
qcdy_solver_legacy_N
F_N2_body_one_mesh_N[strip,3]
F_N3_body_one_mesh_N[strip,3]
F_total_body_one_mesh_N[3]
F_N2_body_reported_pair_N[strip,3]
F_N3_body_reported_pair_N[strip,3]
F_total_body_reported_pair_N[3]
F_rig_body_pair_N[3]
L_N2_wind_pair_N
T_N2_wind_pair_N
L_N3_wind_pair_N
T_N3_wind_pair_N
L_total_wind_pair_N
T_total_wind_pair_N
```

```text
qcdy_single = q*c*dy_single
qcdy_mirror_pair = 2*qcdy_single
F_reported_pair = 2*F_one_mesh
L_wind = Fz*cos(alpha)-Fx*sin(alpha)
T_wind = -(Fx*cos(alpha)+Fz*sin(alpha))
```

当前实现用一侧几何，同时 `dy_lb=2*half_span/ns`，最终报告又乘 2。replay
必须把 single/legacy/pair 三种 ledger 并列输出，以判断是否存在额外倍数；
不得在 observer 或 replay runner 中偷偷除 2。若不闭合，只登记新的
force-scope 病灶并令 validity gate 失败。

## 3. 指标与物理单位

最后一周期含 SPC 个样本，不重复终点：

\[
dt=\frac{1}{f\,SPC},\qquad J_X=dt\sum_{n,j}X_{n,j}.
\]

正式名称：

- `E_A0_Ns`：`q*c*dy*A0_excess_pre_cds` 的 force-scaled LESP exposure；
  不是物理 N3 冲量。
- `J_CV_normal_proxy_Ns`
- `J_CNv_normal_proxy_Ns`
- `J_N3_body_xyz_Ns`
- `J_N3_wind_L_Ns`
- `J_N3_wind_T_Ns`

真正的 production N3 impulse 必须对实际 N3 力向量积分后投影，不能以
`qcdy*dCN` 标量替代。

## 4. 两个不可混用的时钟

```text
formation_T_hat += abs(u_LE dot n)*dt/chord
lb_ds_step = 2*Urel*dt/chord
tau_v += 0.45*lb_ds_step = 0.9*Urel*dt/chord
```

`tau_v` 在 `not lev_active and alpha_eff<0` 时复位。D4 的 `tau_v>4.24`
必须严格使用 `tau_v_post > 4.24`。replay 不得把它替换成 `formation_T_hat`，
也不得把实现的 `0.9` 改为文档曾误写的 `0.45`。改变时钟方程属于气动模型
改写，不是指标修复。

## 5. 展向指标

对 \(X\in\{CV,CNv,N3_L\}\)：

\[
J_j^+=dt\sum_n\max(X_{nj},0),\quad
J_j^-=dt\sum_n\max(-X_{nj},0),\quad
J_j^{abs}=J_j^++J_j^-.
\]

分别输出：

```text
eta_centroid_positive
eta_centroid_negative
eta_centroid_absolute
outboard_share_positive
outboard_share_negative
outboard_share_absolute
```

外翼定义固定为 `eta_ref>=0.5`；跨阈值条带按参考单元与外翼区的重叠比例
拆分。禁止以 signed 总量作质心分母，避免正负抵消导致发散。

## 6. 半周期与 phase identity

半周期由解析运动学固定，不允许从力结果重新命名：

```text
flap_rate_positive: phase_solver in [-pi/2, pi/2)
flap_rate_negative: phase_solver in [pi/2, 3pi/2)
```

`lift_dominant_half` 只在 tw0 上由预先冻结的处理链选择一次，随后对所有
twist/phase 固定。

D7 输入身份拆成两个独立守卫：

1. `code_internal_phase_identity`：验证解析式、左右翼镜像和 body/wind 轴；
2. `external_Figure10_identity`：要求 Figure 10 数字化运动学、机构装配角到
   代码轴系的旋转矩阵及左右翼守卫。

第一个通过不能代替第二个。当前证据只支持 conditional code-internal identity。

## 7. Figure 16 数据处理合同

仓库中的 Figure 16 数据是论文中已经做过 5 阶 8 Hz Butterworth 处理的曲线
数字化，不是实验仪器 raw。论文/仓内资料没有完整给出采样率、因果/零相位
方式和滤波初值。因此：

- 实验资产必须标为 `published_filtered_gt`；
- 禁止再次对该数字化曲线做 8 Hz 滤波；
- 模型处理只能标为 `filter_emulation`；
- 当前三周期铺展后的 `filtfilt` 只能叫
  `zero_phase_emulation_v1`，不能叫实验同处理；
- 真正的 common-filter gate 只有取得实验仪器 raw、采样率和处理方式后才能
  PASS。

现有 `fig16_compare.py` 的：

1. median/MAD clipping；
2. `filtfilt`；
3. 以首个模型 tw0 lift 优化全局相移；

均不得进入 raw 层。processed 层必须同时保存：

```text
no_clip
legacy_clip8
zero_phase_emulation_v1
```

若 clipping 改变任何科学门，或 no-clip 与 legacy-clip 分支裁决不同，结论为
`INCONCLUSIVE`。禁止让模型力信号决定实验相移；只有实测运动学能够提供
跨域 phase alignment。

## 8. 三层不可变 artifact

### `raw/`

```text
model_<case>.npz
model_<case>.schema.json
experiment_fig16_digitized.csv
experiment_fig17_digitized.csv
run_manifest.json
```

只含未 clipping、未滤波、未对齐的模型输出，以及明确标成 digitized/
published-filtered 的实验资产。

### `processed/`

```text
processed_<case>.npz
processing_manifest.json
```

记录 phase masks、wing-scope 换算、body→wind 投影、滤波器 SOS、采样率、
padding、clip 索引、alignment 来源及 parent raw SHA256。每个 processed
artifact 必须是 raw 的纯函数。

### `summary/`

```text
d4_d8_replay_results.json
d4_d8_replay_report.md
guard_report.json
```

summary 只保存可由 raw/processed 独立重算的指标、门结果和 claim 影响。

每次运行必须绑定：

- Git commit、dirty diff SHA256 和全部 untracked 依赖 SHA256；
- runner、contract、`_v2_robo.py`、`lb_dyn.py`、static polar、geometry、
  closure profile 和 claim YAML SHA256；
- `data.md`、`datav2.md`、原 PDF/图像和 digitized CSV SHA256；
- requested/resolved config canonical-JSON SHA256；
- Python/NumPy/SciPy/Warp/GPU/dtype/device；
- command、cwd、start/end time、exit code；
- raw/processed/summary SHA256 及 parent hash。

缓存只有完整 identity hash 完全相同时才允许复用。

## 9. 先于科学结论的 validity gates

任一失败，结果为 `INVALID/NO_DECISION`，不能写成机理 NO-GO：

1. case/config/hash 完整；
2. raw shape、有限值、时间步正确；
3. observer on/off 的 production L/T 和逐时步力在冻结容差内完全一致；
4. node/strip/one-mesh/reported-pair ledger 闭合；
5. 全部 summary 可由 raw 独立重算；
6. filter/alignment provenance 合格；
7. 确定性重复运行在预登记容差内一致。

## 10. D4–D8 科学门

### D4：晚龄与 separation source

\[
J_{\rm sep}=\int\sum q c\,dy_{\rm pair}|CV|dt,
\quad
S_\tau=
\frac{\int 1_{\tau_v>4.24}|F_{N3,L_w}|dt}
{\int |F_{N3,L_w}|dt}.
\]

- H2 GO：实际 N3 从 22.5→45 增长，且
  `J_sep(45)<=J_sep(22.5)`。
- H2 NO-GO：`J_sep(0)<J_sep(22.5)<J_sep(45)`，且实际 N3 同向单调。
- H3 GO：`S_tau(45)>=0.30` 且
  `S_tau(45)/S_tau(22.5)>=2`。
- H3 NO-GO：任一条件不满足。
- 阈值落入不确定度区间：`INCONCLUSIVE`。

### D5：signed source 与展向位置

对 CV、CNv 分别计算 signed/positive/negative N·s。

- H11 GO：至少一个量满足
  `J22.5>0`、`J22.5>J0`、`J22.5>J45`，且
  `(J22.5-J45)/J22.5>=0.10`。
- 两个量都单调、符号错误或下降不足 10%：NO-GO。
- centroid/outboard share 只作定位，不单独触发 GO。

### D6：+90° alpha/peak 机制

令 tw0 冻结的 lift-dominant half 为 \(H^*\)：

\[
A_H(w)=
\frac{\int_H\sum q c\,dy\,|\alpha_{\rm kin}|dt}
{\int_H\sum q c\,dy\,dt}.
\]

- H12 GO：`A_H(45)<=0.9*A_H(22.5)`，且同半周期实际
  `N3 wind-lift mean` 增长。
- 必要条件失败：NO-GO。
- Figure-16 指纹：处理后 tw45 正峰与负峰幅值都小于 tw22.5。
- no-clip 与 frozen emulation 分支裁决不同：INCONCLUSIVE。

### D7：−90° 相位敏感性

必须同时满足：

1. lift-dominant half 的 `|alpha_kin|` 从 22.5→45 下降；
2. tw45 正、负升力峰幅值都小于 tw22.5；
3. `mean L(45)<mean L(22.5)`。

全部通过才是 H13 GO；任一失败为 NO-GO。即使 GO，也只证明 phase 是一阶
敏感量，不证明 external Figure 10 identity。

### D8：phase-only 局部充分性

使用双翼风轴周期均值：

```text
Lift:   L15 > L22.5 > L45 and L15 > L0
Thrust: T22.5 > T0 and T45 < T22.5
```

全部通过才能称 `phase-only locally sufficient`；任一失败为 NO-GO。实验
参考固定使用 data.md 中 0/15/22.5/45 的已数字化点，禁止看完结果后改用
“约 25°”。

## 11. 允许与禁止的修改边界

属于“修指标合同”，不改气动模型：

- 新增只读 raw observer；
- 补 `dt`；
- 拆分 before/after `cds`；
- 区分 `formation_T_hat` 与 `tau_v`；
- 显式 single/pair、body/wind；
- 实现 centroid/outboard diagnostics；
- 固定 half-cycle/phase identity；
- raw/processed/summary 分层；
- filter/hash/manifest/executable guards；
- observer on/off force-isolation regression。

禁止混入 replay：

- 修改 `lb_cds`、critical LESP、`Tv/Tvl`；
- 修改 `tau_v` 演化或复位；
- 修改 `dy_lb` 的生产力路径；
- 修改 panel normal 或 body→wind 公式；
- 修改默认 twist phase；
- 用滤波后信号反馈气动力；
- 根据 D4–D8 结果调常数。

后一组均属于气动 claim 改写，必须重新走：
数据病因 → 文献机理 → 缺件/错件裁决 → 预登记方案。

## 12. 绑定的当前输入

| 文件 | SHA256 |
|---|---|
| `platform/docs/diag/d4_d8_experiment_audit.md` | `b987a77628f99f0fe5961774553b4cfa1bd68d8d9cabfec7fbb808262592a124` |
| `platform/docs/diag/research_n3_landscape_20260727.md` | `d0489993ae1cc10ee50d5e235c2aa4415f98314ce32102e212ebe431f504a704` |
| `platform/_v2_robo.py` | `b5bf4c33da55e86b606b8c7a9f5909d6ba6a068332db01066db8f2d5bdcf8918` |
| `platform/lb_dyn.py` | `11b7e81acc7b8b43a4df74954d44653334156a16bc507485423aad6e6e8445b1` |
| `platform/fig16_compare.py` | `5a3b0df100b45edec4e1775dec6e7347876279598b58a64ec2a8746ed32985ea` |
| `platform/docs/datav2.md` | `15fa067119743efee1c509aeb1657fb16393fb74b9db905f8d7a09dcc8fe9072` |
| `platform/docs/data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` |
| `platform/lb_sweep118.py` | `059add02d0d3d448c632956ead2a0a83a6307ea0673ae83c9a2c20937200f325` |
| `platform/_v2_repro_nc12.py` | `880cacb1e7844341255e06d8e464274932aa9fcfa7fbd13679d6983d216548ba` |
| `platform/_v2_robogeom.py` | `2de57d9062e61cdeafcf4bced2647917e50e7adcfa114f1be8c71eeef6b69e98` |
| `platform/claim_nodes/n3_ds_vortex.yaml` | `991752d97566a843bd854e4de159096e7ccfe28d0afd7ba42d57b5e30d20f9a3` |
| `platform/claim_nodes/n5_twist_coupling.yaml` | `780dc2e937de8ea40ba4e89d616be52d347c5f0915ac131e0ee70be1511eb1e5` |
