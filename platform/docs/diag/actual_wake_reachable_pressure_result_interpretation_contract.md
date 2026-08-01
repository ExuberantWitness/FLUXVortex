# S3ai-v2.2 可达压力正式结果判读合同

**冻结时间**：2026-07-28T16:59:09+08:00  
**性质**：result interpretation contract；不是结果，也不改变 claim state  
**适用结果**：
`platform/docs/diag/actual_wake_reachable_pressure_obstruction_results_20260728_134229.json`  
**latest 规则**：
`actual_wake_reachable_pressure_obstruction_results.json` 必须保持不存在  
**直接 claim 节点**：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  

本合同在正式结果出现之前从冻结 v2/v2.1/v2.2 定义、one-shot wrapper、
formal authorization 和当前 claim tree 中逐项抽取。目的不是预判结果，而是
阻止结果出来后的解释漂移。

## 1. 原子结果身份

结果必须同时满足：

- `protocol_version == "S3ai-v2.2"`；
- `production_activation_allowed == false`；
- `fixed_space_only == true`；
- 63 个 `code_fingerprints`；
- `execution_accounting` 精确为：
  - `history_count=31`
  - `measurement_step_count=380`
  - `prestep_count=31`
  - `marcher_step_count=411`
  - `half_full_solve_count=822`
  - `observed_stage_count=791`
- 31 个 case 严格按冻结 registry 顺序出现；
- `ordered_case_names`、case payload hash、observation payload hash、
  scientific payload hash 和 one-shot provenance 均可重算；
- canonical result 与永久 marker 都是 no-follow regular file；
- latest result 不得写入。

最终结果包含 22 个顶层字段：

```text
stage
protocol_version
contract_file
definition_chain
generated_at_utc
code_fingerprints
stage_decision
production_activation_allowed
fixed_space_only
checks
execution_accounting
negative_controls
v22_span_parity
omega
zero_sum
zero_noncancellation
cases
nonclaims
ordered_case_names
observation_payload_sha256
one_shot_provenance
scientific_payload_sha256
```

每个 `cases.<history>` 必须包含：

```text
configuration
role
epsilon_signed
timestep
quadrature_order
case_identity
measurement_steps
observed_stages
stage_times
stage_roles
measurement_times
stored_window_residual
direct_window_residual
stored_step_residuals
direct_step_residuals
typed_stage_arrays
value_hashes
stored_noncancellation
direct_noncancellation
diagnostics
checks
mass_active
extended_value_hashes
observation_payload_sha256
```

形状门：

| 字段 | 形状 |
|---|---:|
| `mass_active` | `(7,7)` |
| `stage_times` | `(S,)` |
| `measurement_times` | `(N,3)` |
| stored/direct window residual | `(7,)` |
| stored/direct step residual | `(N,7)` |
| 四个 complete-trace 数组 | `(S,9)` |
| 两个 weak-pressure 数组 | `(S,7)` |

其中 `N∈{4,8,16}`，`S=1+2N`。

## 2. 进入物理判读前的 15 个 aggregate checks

以下十五项必须全部为 `true`：

```text
registry_accounting
all_history_hard_guards
wrong_birth_negative_control
wrong_attachment_negative_control
projected_omega_even_convergence
projected_omega_odd_convergence
projected_zero_sum_even_convergence
projected_zero_sum_odd_convergence
projected_zero_noncancellation_even_convergence
projected_zero_noncancellation_odd_convergence
matched_stage_projected_q_families
manufactured_parity_controls
omega_odd_interval_includes_zero
zero_sum_odd_interval_includes_zero
zero_stagewise_odd_interval_includes_zero
```

任意一项为 false，只能得到 `PROTOCOL-NO-GO`，不得继续解释压力机理。

### 每条 history 的硬守卫

31 条 history 均须通过：

- stored/direct BIE backward error ≤ `2e-11`；
- conditioned backward error ≤ `1e-2`；
- direct-W factorization residual ≤ `5e-11` 且 rank deficiency=0；
- stored material/body trace ≤ `2e-11`；
- direct body compatibility ≤ `2e-8`；
- zero-tip、surface trace/history/row-cache/boundary duplicate ≤ `2e-12`；
- material inventory increment ≤ `2e-11`；
- old-state mutation=0；
- history time/geometry、midpoint identity、current attachment ≤ `2e-12`；
- entrance prestep inventory ≤ `2e-11`；
- typed body/direct-W quadrature mismatch count=0；
- input mutation=0。

### 负对照

wrong-birth 与 wrong-attachment 都必须：

```text
value > 0
separation_ratio >= 1e6
```

分母为 `max(correct residual, same-trace round floor)`。definition/unit-test
负对照不能伪装成这两条正式 history 结果。

### 收敛与偶奇守卫

- Richardson PASS：floating plateau，或
  `coarse-medium > medium-fine` 且比值 ≥ `3.2`。
- q-tail PASS：floating plateau，或
  `delta1 > round_floor` 且 `rho=delta2/delta1 < 1`。
- round floor：
  `4096*eps64*max(unprojected typed operand norms)`，只计一次。
- v2.2 matched-stage q-family 共 393 项。
- 任意 odd interval 下界 `L_minus>0` 为 `PROTOCOL-NO-GO`。
- `L_minus=0` 只能写作 `NO RESOLVED SYMMETRY VIOLATION`，不是精确对称证明。

## 3. 唯一合法的四分支决策

严格按以下优先级重推：

```text
if any(aggregate check is not true):
    PROTOCOL-NO-GO
elif L_zero_sum_even > 0 or L_zero_stagewise_noncancellation_even > 0:
    ZEROTH-ORDER NAMED-LAW OBSTRUCTION
elif L_omega_even > 0:
    FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS
else:
    NO RESOLVED WITNESS
```

只能使用 `v22_span_parity.*.even.interval.lower` 判定，不能以未投影的顶层
`omega/zero_*` 诊断替代。

### `PROTOCOL-NO-GO`

含义：协议、数值、对称性或守卫失败。  
不含义：压力机理被证伪、无 VES、存在 VES、现有 closure 正确。

### `ZEROTH-ORDER NAMED-LAW OBSTRUCTION`

含义：在 α=0 基态，具名零额外状态压力律已有可分辨 residual。  
禁止：形成 centered tangent、cokernel、state identity 或 VES 结论。

### `FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS`

含义：零参考区间含零后，冻结 incidence path 的 even tangent residual
在固定空间误差球之外。  
禁止：称为状态维数、VES 身份、生产载荷验证或 Fig17/18/19 改善。

### `NO RESOLVED WITNESS`

含义：在本固定空间、本路径和当前误差球内没有分辨出 obstruction。  
这是科学上的未决结果，不是 `zero-extra-state closure validated`，也不能把节点
设为 falsified/validated。

## 4. 证据上限与 nonclaims

即使得到 obstruction，最多证明：

> 在固定 81-body-DOF、7 active-P2 DOF、symmetric-diamond canonical 上，
> 由同一 S3e actual-body/material-wake march 产生的相邻状态不满足具名
> 零额外状态压力律
> `R_n=M(g_cur-g_prev)+dt*P_mid`，且 residual 超过冻结同空间误差球。

绝不能外推为：

- 缺失状态就是 VES；
- 状态维数为 7 或任何确定值；
- obstruction rank/cokernel 已识别；
- forming geometry 或 pressure observer 已正确；
- prescribed uniform-x convection 等于真实 roll-up；
- face_mu inventory 是 closed Kelvin loop；
- actual-body h/p 已收敛；
- panel pressure、force、LEV amplitude 或 production 已验证；
- 118 或 Fig17/18/19 已改善。

## 5. 对 claim tree 的唯一允许写法

目标节点当前必须保持 `freeze:false`。

| 正式结果 | 最大允许改写 |
|---|---|
| 文件缺失、进程崩溃、原子/授权/哈希失败 | scientific claim state 不变；只登记执行完整性事件 |
| `PROTOCOL-NO-GO` | 保持 `open`；记录失败 check、artifact SHA 和病灶节点 |
| `ZEROTH-ORDER...` | 最多 `open→partial`；限定为 fixed-space zeroth-order named-law obstruction |
| `FIRST-ORDER...WITNESS` | 最多 `open→partial`；记录 even interval、所有 checks 和 artifact SHA |
| `NO RESOLVED WITNESS` | 保持 `open`；记录当前路径/空间/误差球内未解析 witness |

无论哪个分支都不得：

- 设为 `validated/frozen`；
- 改写 N2.6c2b 的 VES 身份节点；
- 激活 N3.1j4b5b；
- 修改 production closure、force、118 或 Fig17/18/19；
- 直接建立 `residual = VES state` 的子节点。

## 6. 正式结果出现后的确定性复核顺序

1. 校验 canonical result、permanent marker 和 latest-absent 规则，计算 SHA256。
2. 核对 stdout receipt 的 result SHA、decision、authorization/token SHA。
3. 严格核对 22 个顶层字段。
4. 调用只读 `_validate_serialized_result`。
5. 每条 history 重算：
   - `S=1+2N` 和 stage-role 顺序；
   - window residual 等于 step residual 逐项和；
   - `mass_active` 为 exact symmetric/SPD/persymmetric；
   - 5 个 frozen hash、14 个 extended hash、case payload hash。
6. 重算 global observation/scientific payload SHA。
7. 核对 one-shot provenance 的 source/runtime/execution-input start=end、
   registry manifest、marker receipt 和 `latest_pointer_written=false`。
8. 输出十五个 aggregate checks 原值；任一 false 即停止物理解释。
9. 独立按第 3 节优先级重推 decision。
10. obstruction 分支记录 lower/upper、uncertainty components、收敛报告；
    no-witness 分支同时记录 upper bound，禁止把 lower=0 写成 residual=0。
11. claim YAML 改写前执行 fresh independent result review。
12. 下一实验只能先预登记 actual-body h/p；当前授权禁止直接执行 state/VES、
    force、118 或 Fig17/18/19。

## 7. 后续机理顺序

若 witness 被解析：

1. actual-body 空间 h/p 重现 witness；
2. 若重现，先测试 parameter-free Xia–Mohseni massless forming geometry；
3. 只有 obstruction 仍存在，且独立 mass/momentum/entrainment inventory 能解释，
   才允许讨论 finite VES。

## 8. 绑定的冻结输入

| 文件 | SHA256 |
|---|---|
| `platform/actual_wake_reachable_pressure_obstruction_v22_one_shot.py` | `ddcb2dccfe315c4dfd978cc04f17fdfbdf99dcc8cf8172f6b3c8d9cea76b428c` |
| `platform/actual_wake_reachable_pressure_obstruction_v2_guard.py` | `d2f05dd9a4951c082ed3949f59d95dec10f9a052885394d24a6621ec1b295b73` |
| `platform/claim_runtime/reachable_pressure_symmetry.py` | `595df927549158801f27af26bd5e8049a260d3d689bc77d4637b9590b75581f6` |
| `platform/claim_runtime/actual_wake_reachable_pressure.py` | `1bd97ae35c5ec76b400ace4e42ea3fb3286af56d101cf1263fd0f996821946f9` |
| `platform/docs/diag/actual_wake_reachable_pressure_obstruction_cases_20260728_131922.yaml` | `8345035356d300d4154ac276fc54b8ab27b6e83704cb4444436dbc0c2c59c75b` |
| `platform/docs/diag/actual_wake_reachable_pressure_obstruction_cases_20260728_133218.yaml` | `e751381942ec7c0cac8ea055c6aad1a5756643b85992bc04f4bee88339b563b4` |
| `platform/docs/diag/actual_wake_reachable_pressure_obstruction_cases_20260728_134229.yaml` | `3d662a69c1da80a1452b6b05c67107b188070871bb1b594977467ab7384e0b27` |
| `platform/docs/diag/actual_wake_reachable_pressure_execution_authorization_20260728_143034.yaml` | `39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e` |
| `platform/claim_nodes/n3_ds_vortex.yaml` | `991752d97566a843bd854e4de159096e7ccfe28d0afd7ba42d57b5e30d20f9a3` |
| `platform/docs/diag/claim_tree.md` | `eff4947a9d4ee0ca29d2b99e5288c04f74b8a845168d80e0e4403bc9b0ff812c` |

