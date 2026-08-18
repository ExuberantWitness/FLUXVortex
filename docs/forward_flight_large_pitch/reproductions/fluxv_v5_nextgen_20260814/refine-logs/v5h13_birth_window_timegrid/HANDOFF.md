# V5H13 birth-window graded time grid — handoff

状态：治理包已预注册；实现未开始。批准链：V5H12 formal-A 数值门 STOP →
research-pipeline Stage 1（文献+hostile review+σ-resize 证伪器）→ Gate 1
批准 Idea B（r=4, k=5）。

接手者必读（顺序）：PLAN.md → CHECKLIST.md → EXPERIMENT_TRACKER.md →
FREEZE_INPUTS.json → 本文件 → V5H12 HANDOFF（/tmp/fluxv-v5-nextgen/docs/
forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/
refine-logs/v5h12_execution_repair/HANDOFF.md）。

关键事实：
1. 冻结控制 = FREEZE_INPUTS.json 中 8 个文件（4 个 V5H12 终态叶 + 4 个
   V5H11 科学叶）；任何漂移即 STOP。
2. 改动地图 = PLAN 第 "Allowed change map" 表（8 个机械 fork + 各自唯一
   许可 delta）。两事务拼接不可行（stage chain 会从 genesis 重启）；
   必须在 fork 的 stream macro 加逐子步 Δt 表。
3. (r=4, k=5) 与门计账 amendment 已冻结；预测 0.4972/r=0.1243 ±20%。
4. 禁止 σ 演化（Idea A 已被证伪器否决：median Z=+0.28 s⁻¹）。
5. A/B 继承 V5H12 全部协议；GT/scorer 保持 sealed。
