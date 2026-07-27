# FLUXV v4.2 战役 — Codex 交接文档（2026-07-27）

> 复制此文档到 Codex 即可接手。所有命令默认在 `/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV` 下执行。
> Python 环境：`/home/exuber/anaconda3/envs/fluxvortex/bin/python`（Warp GPU 栈）。
> 仓库：`https://github.com/ExuberantWitness/FLUXVortex.git`，分支 `aero-rvpm-lev`。

---

## 0. 项目一句话

FLUXV RoboEagle 扑翼数字孪生（刚性翼，Meng et al. Drones 2025, 9, 535），用 GPU UVLM + L-B 动态失速修正层复现实测 Fig17/18/19 的升力/推力。代码在 `platform/`（前缀 `_v2_`），主入口 `gpu_run_twist()`。

---

## 1. 方法论纪律（最重要，违反即返工）

用户裁定（2026-07-26/27，已入 memory `feedback_claim_chain_research.md`）：**这是 research，不是"有 bug 就修、work 就行"的改代码。**

模型 = 一棵 **claim chain 树**，每个组成部分是带证据状态的命题。每次改进严格四步，跳步即违规：

1. **定位病因**：数据分析出误差规律（指纹），把病灶挂到树的某个节点；明确可动空间（validated 节点禁碰）。
2. **学科机理**：`research-pipeline` / `/research-lit` 查文献，先于方案。机理未经文献锚不定方案。
3. **改进方向裁决**：显式写"缺组成部分（新节点）"还是"组成部分本身错（旧 claim 证伪需改写）"。
4. **针对机理提方案**：方案 = 树的有证据改写（节点状态转移 + 证据链接）；go/no-go 判决规则预登记。

**红线**：禁常数吸收（d_para 这类）；禁补丁式试错；禁碰 validated 节点；禁 `/tmp` 存持久数据。

---

## 2. claim chain 树代码本体（2026-07-27 重构完成）

仿真结构已从"文档树"升级为"代码本体树"（commit `521c523`）：

- **`platform/claim_nodes/*.yaml`** = DevReady 资产，每个模块一个文件：
  - `n1_uvlm.yaml`（N1 UVLM 附着流主引擎）✓validated ❄freeze
  - `n2_kirchhoff.yaml`（N2 L-B 双时滞分离砍量）~partial，含子节点 N2.1/N2.2/N2.3/N2.4
  - `n3_ds_vortex.yaml`（N3 ds 涡升增强）~partial，含 N3.1/N3.2/N3.3
  - `n4_ct_consistency.yaml`（N4 CT 吸力一致簿记）✓validated ❄freeze
  - `n5_twist_coupling.yaml`（N5 扭转耦合响应）✗falsified
  - `n6_d_para.yaml`（N6 d_para 钝体阻）∎dead_end ❄freeze
- **`platform/claim_dag.py`** = `ClaimNode`（状态机）+ `ClaimDAG`（遍历/验证/修改检查）。
  - 修改规则：validated 节点 freeze 禁改（N1/N4）；已证伪/死路节点禁重走（N5/N6）；partial/open 节点可动但需证据+归因。
  - `python platform/claim_dag.py` 查看树状态 + 修改前检查。
- 每个 YAML 字段：`id/title/claim/state/freeze/evidence/refs/memory/guard/depends_on/provides_to/children/version/changed/notes`。

**接手新节点时**：先 `claim_dag.validate_modification(node_id, change_desc)` 检查是否可动。

---

## 3. 当前 git 状态

```
分支: aero-rvpm-lev (已 push 到 origin)
最近 commits:
  0e2f384 docs(tree): claim tree code ontology synced
  521c523 feat(claim-dag): claim chain tree code ontology (DevReady + DAG)
  adbf503 docs(tree): E1/E2 lesions registered
  de715bf feat(closure): v4.1 production preset + v4_legacy switch group
  bf9f7fc fix(plot)+docs: L/T variable-swap bug (fixed)
```

`gpu_run_twist()` 的新参数：`closure='v41'`（默认，生产预设）或 `closure='v4_legacy'`（旧基线）。lb_* 签名默认 `None` = 从预设取，显式传值则覆盖。

**官方数值基线**（07-25 冻结，gitignored 数据）：`platform/docs/s6_sweep_v41.json`、`s6_sweep_v4_legacy.json`（118 点，key 格式 `U_f_tw_aoa`）。

---

## 4. 当前阻塞：E1/E2（v4.1 晋升基线不可复现）

**这是接手后第一件事**。

**事实**：07-25 候选 F 的 118 扫调用字典有个未落档的要素，导致当前代码 `closure='v41'` 在角区复现不出官方缓存：

| 探针 | 代码 closure='v41' | 07-25 缓存 | 实测 |
|---|---|---|---|
| aoa15/f2.6/tw22.5 L | +25.06 | +15.59 | ~15.1 |
| aoa0/f2.6/tw22.5 L | +1.70 | +4.77 | ~2.9 |
| 验证区（aoa5 全 f、U 族、tw 族）| 基本一致 | — | — |

**LB_DIAG2 通道归因（aoa15，单翼周期均）**：
- chop（砍量）−2.68
- **hybrid 回补 +4.64** ← 主犯
- **ds 幅值驱动 +2.55** ← 次犯
- 净 +4.51 ×2 ≈ +9N ≈ 观察到的 +9.8N 过冲

**关键归因发现（强假设，数值支持但未完整验证）**：候选 F 真身 = `lb_hybrid=0`（纯砍，无准定常截面力回补）。claim 树证伪注册表早有"杂交：hybrid=1（量级 18.9 过冲）"——是已证伪节点，但我在写 `closure='v41'` 预设时错把 `lb_hybrid=1.0` 复活了。数值推导：E0（hybrid=0, cds=0）aoa15 L=10.89；+ ds 的 L 贡献 +4.87 → 15.76 ≈ 缓存 15.59 ✓。

**主嫌**：`lb_lesp_crit=0.23`（当前默认 0.18）。方向全对：aoa15 25.1→~19、aoa0 1.7→0.4、aoa5 不动；但 origin 无记录，且 0.23 无文献锚（0.18 才是当初反演值）。

### 接手后的裁决路径（用户已选"走 1 = 先归因"）

1. **验证 candidate F 真身 = hybrid=0**：跑 `closure='v41', lb_hybrid=0` 的 aoa15/aoa0/aoa5 三点，对照缓存。若 aoa15 落到 ~15.8、aoa5 不动 → 确认 hybrid=0 是真身，预设 bug 修正即闭环。
2. **若 hybrid=0 仍不够**（aoa15 还差 ~3N）：`lb_lesp_crit=0.23` 是第二嫌疑。但 0.23 无锚 → 走 claim 树流程：N2.4 当真病灶（hybrid 回补应过 K(f2) 衰减，不是满额准定常截面力），入 Phase A 扩战场。
3. **用户红线**：禁用无文献锚的常数换复现。0.23 若无锚，宁可走 (ii) E1 当模型病灶。

---

## 5. 后续战役（v4.2，顺序 A→B→C，计划在 `~/.claude/plans/co-design-*.md`）

| Phase | 病灶 | 节点 | 方向预判（待文献确认） |
|---|---|---|---|
| **A** | D1 推力缺阻 + dT/dU 反号（质的差别） | N2.2 错件 | 局部来流系分解（分离压差阻）；`research-pipeline` 查 L-B/Bangga/Sheng 分离阻簿记 |
| **B** | D2 ds 角区过供（U6/aoa0 回归） | N3.1 缺门控 | Gharib 形成门 + 相干性门（非驱动本身错） |
| **C** | D3 扭转扫升力形状（tw15-22 峰后滚落） | N5 归因未定 | 先通道归因（独立节点 vs N2/N3 派生），禁先修 |

**顺序不可换**：Phase A 改 N2.2 簿记会改变扭转响应基线 → C 必须在 A/B 定稿后重基线。

**用户验收门**：drag 绝对误差 ≤20%；轨迹相似性 v3 八族（趋势捕获为主，MAE 门可放宽，memory `feedback_rom_law_capture.md`）。

---

## 6. 环境与命令

```bash
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV
PY=/home/exuber/anaconda3/envs/fluxvortex/bin/python

# 单点探针（验证 closure 预设）
$PY -c "
import sys; sys.path.insert(0,'platform')
import warp as wp; wp.init()
from _v2_robo import gpu_run_twist
r = gpu_run_twist(U=8.0, aoa_deg=15.0, freq=2.6, flap_amp_deg=22.5, twist_amp_deg=11.25,
                  twist_phase_deg=90., nc=4, ns=8, n_cycle=2, steps_per_cycle=60, wake_rows=60,
                  real_geom=True, sym=True, les_suction=True, visc=True, d_para=0.5, a0_crit=0.23,
                  closure='v41')   # 或 closure='v4_legacy'
print('L=%.3f T=%.3f'%(r['L_wind'], r['T_wind']))
"

# LB_DIAG2 通道分解（病灶归因）
LB_DIAG2=/tmp/lbd.npy $PY -c "...读 /tmp/lbd.npy..."

# 118 全扫（已备，阻塞于 E1 裁决）
$PY platform/lb_sweep118.py both          # v41 + v4_legacy
$PY platform/lb_sweep118.py both --quick  # 12 点快环

# 趋势记分卡（轨迹相似性）
$PY platform/trend_metrics.py platform/docs/s6_sweep_v41.json

# 三者对比图（实测/v4/v4.1）
$PY platform/plot_3way.py   # 输出 fig17/18/19_en.png

# claim 树状态
$PY platform/claim_dag.py
```

**运动学口径（铁律，memory `project_fluxv_robowing_aero_calib.md` ㊷）**：标称幅值 = 峰-峰半幅。`flap_amp_deg=22.5`（标称 45°）、`twist_amp_deg=tw/2`（标称 tw=22.5° → ±11.25°）。Fig18/19 频率线实测在 tw=22.5° 条件测。

**网格**：生产 `nc=4, ns=8, n_cycle=2, spc=60·nc, wake_rows=spc`（le 网格，cosine 布点是伪影已换 le/均匀）。

**代理**：活的代理 `127.0.0.1:6789`；`git push` 走 `HTTPS_PROXY=http://127.0.0.1:6789` + TLS 重试（首次常失败，重试 2-3 次）。

---

## 7. 已证伪候选总账（禁重走）

- chop 方向：body-z（sin(aoa) 泄漏 +1.5N@aoa15）、全矢量（两通道耦合不清）
- ds：signed 驱动（候选 G，aoa0 过抵消）、f2gate、全矢量、瞬时式（无 Tv 记忆）
- CT：|CT| 前向记账（双计 +3.6N）、完整 (1−√f2)（过拖阻 5×）
- sep_drag f² 型（dT/df 形状 0.995→−0.03）
- Prandtl cla3d 修 chop（自相似，近零效）
- d_para 常数吸收 T3b（用户红线）
- **杂交 hybrid=1（量级过冲，候选 F 真身是 hybrid=0）**

---

## 8. 接手第一步清单

1. 读 `platform/claim_dag.py` 跑一遍 `summary()`，确认树状态。
2. 读 `platform/docs/diag/claim_tree.md`（人类可读视图 + E1/E2 病灶登记）。
3. 读 `platform/docs/diag/research_symmetry_break.md`（候选 C→D→E→E2→F 演化链，通道分解证据）。
4. **执行第 4 节裁决路径**：验证 `closure='v41', lb_hybrid=0` 在 aoa15/aoa0/aoa5 三点。
5. 闭环 E1/E2 后，启动 118 全扫（`lb_sweep118.py`），生成自洽基线。
6. 进 Phase A-②：`research-pipeline` 查分离压差阻簿记文献。

**用户身份**：张明昊，西安工业大学交叉创新研究院讲师，RL 控制 + 机电建模仿真。沟通用中文。坚持异常先信用户观测、交付图必须视觉核对、不跑通不算交付。
