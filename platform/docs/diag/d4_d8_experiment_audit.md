# D4–D8 / Figure 10 / N5.1 / N3.1h 实验完整性审计

**生成时间**：2026-07-28T16:48:55+08:00  
**审计技能**：`experiment-audit`  
**审计员**：`/root/d4_d8_integrity_review`，GPT-5.6-Sol ultra  
**独立性**：same-family  
**接受等级**：provisional  
**模式**：只读；审计员未编辑文件、未运行新的气动实验  

## Overall Verdict: FAIL

D4–D8 的若干定性 NO-GO 方向可能成立，但精确数值及据此写入的
claim states 不能独立复核。这里的 `FAIL` 不等于证明结果被伪造；它表示：

1. 缺少能够绑定运行配置、代码身份和逐步输出的原始结果；
2. D4/D5 的“冲量”指标与实现的物理语义不一致；
3. D6 的模型/实验信号处理不一致；
4. N5.1 的声明树与实际可执行守卫不一致；
5. 当前 dirty worktree 无法证明叙事数值对应哪一个生产实现。

本审计不改写 claim YAML，不授权生产修改，也不授权从叙事表格反向制造
NPZ/JSON。它只限定后续可引用证据的上限。

## A–G 检查

| 检查 | 状态 | 证据与裁决 |
|---|---|---|
| A. Ground-truth provenance | WARN | Figure 17 实验均值可追溯到 `platform/docs/data.md` 的论文曲线数字化；Figure 16 条件和 8 Hz 滤波可追溯到 `platform/docs/datav2.md`。但没有点级数字化来源、不确定度或机器可读 Figure 10 trace。未发现以模型输出伪造实验 GT。 |
| B. Metric integrity | **FAIL** | `research_n3_landscape_20260727.md` 将 D4/D5 定义为时间积分，但 `_v2_robo.py:2553-2581` 只做样本求和，未乘 `dt`；单位是 force-sample sum，不是 N·s。文档的 `I_A0` 不含 `cds`，运行通道却已乘 `lb_cds=2.5`。文档写 `tau_v += 0.45 Vdt/c`，`lb_dyn.py` 实际为 `0.45*(2Vdt/c)`。相同离散下的排序/比例可作线索，绝对数及物理单位不可引用。 |
| C. Result existence | **FAIL** | D4/D5 诊断、D6 half-cycle/peak、D7/D8 的 −90° 与 tw15 精确输出只存在于 prose/YAML。`s6_sweep_v41.json` 仅间接保存 +90° 的 L/T mean，不能复核这些诊断。旧 `fig16_series*.npz` 不含 D4–D8 新通道。 |
| D. Runtime/dead-code alignment | **FAIL** | D5 预登记的展向质心和外翼占比没有实现/输出。`TwistResponseObserver.channel_names=()`，且 solver 未提供 `n5_observation`，因此 N5.1c 的相位/形状 guard 是声明性的，不是运行时守卫。 |
| E. Scope | WARN | D4/D5、D6/D7 和 D8 都是单一操作点附近的三至四点 deterministic pilot。它们最多支持局部证伪，不能支持跨 Fig17/18/19 域的充分性声明。 |
| F. Evaluation type | WARN | D4/D5=`simulation_only`；D6=`simulation_only` 加 Figure-16 定性 real-GT；D7/D8=`simulation_only` 对数字化 real-GT；D9=静态代码/代数加人工视觉；N3.1h=文献/数学命题。D6 使用未滤波模型 extrema 对比已做五阶 8 Hz 滤波的实验曲线，处理链不匹配。 |
| G. Reproducibility/versioning | **FAIL** | 核心代码、claim YAML、claim tree 和数据均处于 dirty/untracked 混合状态；N3/N5 的实现哈希为空；runtime manifest 缺完整工况、网格、时间步、代码/数据哈希和运行时间。当前工作树还含 v41 `lb_hybrid=0`、`lb_cla3d=True` 的生产力变化，不能把整个工作树表述为“仅增加 observer”。 |

## 逐项 claim 影响

### D4

- 三配置下的 H2/H3 排序和比值在共同 `dt` 下可能仍支持局部 NO-GO。
- `I_A0`、`I_|CV|` 和“production N3 impulse”的绝对数不是物理冲量。
- 合理状态：`provisional local NO-GO; raw replay required`。

### D5

- signed scalar 的三点趋势若原表真实，可作为 H11 的局部证伪线索。
- 预登记的 spanwise centroid/outboard share 没有实现，且无 raw。
- 合理状态：`narrative-only; unsupported for promotion`。

### D6

- 没有 half-cycle、alpha、N3 或 peak raw；旧 Figure-16 cache 不覆盖它。
- 模型 extrema 未应用实验同一 8 Hz 滤波。
- 合理状态：H12 NO-GO 必须 metadata-bearing replay 后重新裁决。

### D7 / D8

- Figure-17 实验 GT 可追溯，但 −90° 模型三点和 tw15 没有 raw。
- 若叙事表数值成立，phase-only 不充分的逻辑合理；证据等级仍只能是 provisional。
- 不授权把默认相位改为 −90°，也不授权 118 晋升。

### D9 / Figure 10

- 代码内部 `psi=A_t(y/span)sin(Ωt+φ)`、镜像和 body→wind 变换可复核；
  在给定论文正 twist 人工判读的条件下，代码内部 −90° 映射一致。
- 没有 Figure 7/10 装配角到代码轴系的完整机器可执行映射。
- 合理状态：`conditional code-internal identity`，不能据此冻结外部机制身份。

### N5.1

- 顶层 `partial` 合理；N5.1a/b 的局部 falsification 只能 provisional。
- N5.1c 的 `validated/frozen` 强于现有归档证据；空 observer 不能保护该命题。

### N3.1h

- 它是文献/数学方向，不是运行时已验证组件；`partial` 合理。
- N3.1h2 的 `w=B f_a`、`3N_a >> 6` 非唯一性论证有显式数学依据。
- N3.1h1 仍是 proposed shadow diagnostic，不能称为生产验证。

## Figure-16 旧资产边界

`platform/docs/diag/fig16_series*.npz` 为 2026-07-04/05 的旧缓存，只含
旧模型、+90°、tw0/22.5/45 的 T/L。它们内部能够复现现有
`fig16_stats.json`，但不含 `n3_event_diag`、−90° 或 tw15，不能证明
D4–D8。缓存只按 key 复用，不校验配置/代码/数据哈希；模拟 trace 还经过
模型自身 median/MAD clipping，并以首个模型优化全局时移，因此不能称为
未经处理的 raw。

## 生产影响边界

D4–D6 observer 的复制与汇总位于既有 N3 力计算之后，observer 本身不接入
气动力；D7–D9 也没有直接修改 +90° 默认值。但是当前 worktree 相对 HEAD
还存在 `lb_hybrid: 1→0` 和 `lb_cla3d: False→True`，两者进入实际力路径。
它们可能属于同期 E1/E2 身份修复，而不是 D4–D9 observer 导致；因此只能说
“observer force-isolated”，不能说“整个工作树生产输出未变”。

## 必须采取的动作

1. 不得由 prose/YAML 的表格反向生成 raw NPZ/JSON；缺失必须保持缺失。
2. 正式可达压力 one-shot 结束后，使用专用 D4–D8 runner 重跑；运行身份必须
   绑定代码/数据 SHA256、完整配置、命令、环境、`dt` 和时间戳。
3. 重跑前修订指标合同：加入 `dt`；明确 `cds`、单翼/双翼、法向/风轴投影；
   统一 `tau_v` 定义；模型与实验采用一致的 causal filtering。
4. 保存未经 clipping/对齐的逐时步 raw、处理后 trace 和 summary，三者分离。
5. 将 N5.1c 实现为真正的 kinematics/mirror/axis executable guard。
6. 在新证据生成前，D4–D8 精确数值和 N5.1c `validated/frozen` 不得作为
   已审计事实进入生产 claim promotion。

## 审计输入身份

| 文件 | SHA256 |
|---|---|
| `platform/_v2_robo.py` | `b5bf4c33da55e86b606b8c7a9f5909d6ba6a068332db01066db8f2d5bdcf8918` |
| `platform/lb_dyn.py` | `11b7e81acc7b8b43a4df74954d44653334156a16bc507485423aad6e6e8445b1` |
| `platform/fig16_compare.py` | `5a3b0df100b45edec4e1775dec6e7347876279598b58a64ec2a8746ed32985ea` |
| `platform/claim_runtime/components.py` | `a3461670c079b6f2ddd6640174263d5a5b6f793ba2f4a29f9734c43be2b81742` |
| `platform/claim_runtime/core.py` | `9f497fd951d4774231856e1e59b06de3ed677d737b21a3e5dbb596540737953d` |
| `platform/claim_nodes/n3_ds_vortex.yaml` | `991752d97566a843bd854e4de159096e7ccfe28d0afd7ba42d57b5e30d20f9a3` |
| `platform/claim_nodes/n5_twist_coupling.yaml` | `780dc2e937de8ea40ba4e89d616be52d347c5f0915ac131e0ee70be1511eb1e5` |
| `platform/docs/diag/research_n3_landscape_20260727.md` | `d0489993ae1cc10ee50d5e235c2aa4415f98314ce32102e212ebe431f504a704` |
| `platform/docs/diag/research_n3_spatial_loads_20260727.md` | `9b09e3a98f098faf94e668c5fff593c8ab292b3185fce4f1e4ca3d80e1aca51e` |
| `platform/docs/diag/research_n3_twist_gate.md` | `88fe8246bc29a1d2770b12a8cc7462d583046c62eb471925a4fee2ffffd7754b` |
| `platform/docs/diag/research_bern_twist.md` | `a0414f20fec65308cec3aec01afb37e7afaecd35b90512296ddd5736c1f36387` |
| `platform/docs/diag/claim_tree.md` | `eff4947a9d4ee0ca29d2b99e5288c04f74b8a845168d80e0e4403bc9b0ff812c` |
| `platform/docs/data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` |
| `platform/docs/datav2.md` | `15fa067119743efee1c509aeb1657fb16393fb74b9db905f8d7a09dcc8fe9072` |
| `platform/docs/s6_sweep_v41.json` | `965da388863dc57b390d58b49fe3b8978bdc77c3603b4a3276c97d4d17f94c73` |
| `platform/docs/diag/fig16_series.npz` | `d29878dd19738cad3fabe186e6d8275a6aa78eb16ab89c0b7189bd273c5052e6` |
| `platform/docs/diag/fig16_series_nc4.npz` | `446674a16fd9b82c15fda316cd4fea0705a5e22e7f6a2e11f1513e4d8223d735` |
| `platform/docs/diag/fig16_series_nc8.npz` | `46b3f31e43605550848fe6547c6e96ab945b681b32712cb48ef544b7dcff6e88` |
| `platform/docs/diag/fig16_stats.json` | `dc7cc368b30a298fa819417d570deabfbdeebca6e150d42a98645d910e91cf7b` |
| `platform/docs/diag/fig16_compare.png` | `9dc5849890992e56f2d0f917d9cc29c1557decba38e874fdfcaf01d3c54a33fb` |
| `platform/docs/repro_data.json` | `808ffeed36be0071850e954231417fa7007167c59eb5730cd8cb6829ff18101c` |

## 审计追溯限制

reviewer 最终回复的完整实质内容转录已保存在
`.aris/traces/experiment-audit/2026-07-28_run01/001-d4-d8-integrity.response.md`。
该转录保留全部裁决和证据，但不是工具消息的 byte-verbatim 导出；
trace metadata 因而标为 `substantive_transcription_not_byte_verbatim`。
由于上游上下文压缩，reviewer 的初始 task prompt 无法逐字恢复；request trace
显式记录 `prompt_capture_status=unavailable_after_context_compaction`，不以摘要
反向伪造原 prompt。第二次只读追溯请求及其 `UNAVAILABLE` 回复也单独存档。
