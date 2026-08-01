# Phase 2.2 资产分类清单（Codex 期间 07-27→07-30 产出，2026-08-01 评估）

> 评估原则：① 先验证（数值/测试/幂等）再提交；② 闭环资产必提交；③ 证据资产必提交；
> ④ 探索半成品评估后定；⑤ 临时/污染文件不提交。
> 验证状态：v41 三点复现 0.100N ✓ / claim_runtime 可运行 ✓ / repro_data 幂等 ✓ /
> attribution 47 tests ✓ / claim_runtime CPU tests 33 ✓。GPU 端到端测试待 184 发布后跑。

## A 类 — 闭环资产（必提交，单 commit）

| 文件 | 内容 | 验证 |
|---|---|---|
| `platform/_v2_robo.py` | v41 预设修正（lb_hybrid=0 + lb_cla3d=True）+ hirato/P2 只读探针 + claim 账本校验 | 三点复现 0.100N < 0.15N ✓；哈希 e13feca0 已入 FROZEN |
| `platform/fig171819_benchmark.py` | 50 曲线/184 条件契约 + 双 scope（本轮改：fig19_cd=conditional_scope） | import ✓；promotion_eligible=True ✓ |
| `platform/lb_sweep184.py` | 184 发布 runner（本轮改：data_gate 双 scope + FROZEN 哈希更新 + dry-run 文案） | dry-run ✓；发布运行中 |
| `platform/lb_sweep118.py` | 118 条件 + BASE 定义（H16+kelvin+plateau_fn） | dry-run 引用 ✓ |
| `platform/lb_sweep_candidate.py` | 隔离候选 runner（shadow 用） | 结构 ✓ |
| `platform/fig171819_claim_attribution.py` | frozen attribution 协议（本轮同步哈希链） | 47 tests ✓ |
| `platform/fig171819_confirmed_compare.py` | confirmed42 scorer（本轮同步哈希链） | 47 tests ✓ |
| `platform/docs/repro_data.json` | Fig18 U6/U10 推力身份反标修正 | 幂等 ✓（808ffeed） |
| `platform/correct_fig18_curve_identity.py` | 身份修正脚本（幂等） | no-op ✓ |
| `platform/claim_nodes/*.yaml`（9 个） | M0/N1-N6/R0/LEGACY DevReady 资产（本轮改 M0 证据） | claim_dag 加载 ✓ |
| `platform/claim_dag.py` | claim 树代码本体 | summary/validate ✓ |
| `platform/claim_runtime/`（81 py） | 组件注册表 + oracle 系（SVI-DW/P2/hirato/压力） | import ✓；18 个相关测试 ✓ |
| `platform/docs/diag/claim_tree.md` | 树人类可读视图（本轮回写 08-01 条目） | — |
| `platform/trend_metrics.py` | 记分卡指针（v41 基线） | — |
| `platform/docs/diag/d1_fig18_curve_identity_audit.md` | Fig18 身份审计 | — |
| `platform/docs/diag/fig19_cd_frequency_identity_exhaustion_20260729.md` | fig19_cd 一手资产穷尽 | — |
| `platform/docs/diag/g0c/g0d/g0e` 系列 | 归因审计（NO_DECISION/PUBLIC_NO_GO/Fig15 判别） | — |

## B 类 — 证据资产（必提交，按主题 2-3 commit）

- **S3 空间态门系列**：`research_n3_*.md`（60+ 文件，07-27/28 门链 S3a→S3m + S3ah/ai 审计）
- **N2.6 SVI-DW 系列**：`n26_svi_dw_prereg*`、`n26e1*`（S0 基础 validated + e1b1/e1b2/e1bc0/e1bc1 结果）、
  `research_n26e1*.md`、`digitize_riziotis_fig12.py`、`score_riziotis_fig12.py`
- **R0 周期归约**：`r0_gate1_nogo_20260729.md`、`r0_gate2_result_20260729.md`、`verify_cycle_reduction_r0.py`
- **confirmed151 系列**：`v41_confirmed151*`（87 点 manifest + contributions + resume 日志）
- **N1/N2/N3 归因 witness**：`run_n1_n2_ledger_phase_witnesses.py`、`run_n1_n2_n3_aoa_ladder_witnesses.py`、
  `n1_n2_n3_aoa_ladder_prereg_20260730.md` + `n1_n2_n3_aoa_ladder_runs/`
- **guard 系**（60+ py）：`actual_*_guard.py`、`dde_*_guard.py`、`*_audit*.py` —— 只读 oracle 守卫

## C 类 — 探索半成品（评估后定）

| 文件 | 状态 | 处置 |
|---|---|---|
| `platform/claim_runtime/p2_spatial_candidate.py` 等 P2 候选 | N3.1j0 NO-GO（dt 未收敛） | 提交为 oracle 资产（有测试），不入生产 |
| `platform/claim_runtime/svi_dw_*.py` | N2.6e1 来源复现（e1b1 等 NO-GO） | 提交为 oracle 资产（有测试），标注来源复现阻塞 |
| `platform/docs/candidates/n3_spatial_*` | v0/v1_shadow FALSIFIED 结果 | 提交留档（证伪证据） |
| `platform/lb_sweep151_fresh.py` | fresh151 runner（87 点中断） | 提交（runner 本身完整，数据 87/151） |
| `platform/tests/`（120 tests） | claim_runtime 相关 18 个绿 | 全量提交（测试是证据） |

## D 类 — 临时/污染文件（不提交）

- `.aris/`、`.pagecheck.QDvxoS/`、`hirato_page.html`、`1.npy`（已删）、`findings.md`
- `DERIVATION_PACKAGE*.md`（若为 scratch，确认后决定）
- `platform/docs/_v2_*.gif/png/html`（可视化 scratch）
- `*.npz.ckpt`（录音 checkpoint，gitignore 已有 *.npz）

## 提交策略

按 A→B→C 顺序 commit（每类一个 commit，message 带结论），D 类清理；
`HTTPS_PROXY=http://127.0.0.1:6789` + TLS 重试 push 到 `aero-rvpm-lev`。
**提交前 184 发布必须完成**（fixed-name 产物一起提交）。
