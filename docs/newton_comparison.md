# Newton #2848 耦合框架 vs 两遍窗口预测-校正:源码级能力对比

**问题**:Newton(PR #2848 实验性耦合框架)是否已有类似"预测-校正"的耦合求解实现?
**答案(源码验证):没有。我们的两遍窗口预测-校正正是其框架缺失的能力,且每项差异都有量化证据。**

## Newton #2848 实际拥有的(读 pr2848 分支源码确认)

- `SolverCoupled`:模型分区(Entry/ModelView)、状态分发/回收、**per-entry substeps**(solver_coupled.py:2146-2169)
- `SolverProxyCoupled`:lagged/staggered 代理耦合,**支持同区间重解迭代**(`iterations` 参数,`iteration_restart=True` 时从 state_in 重分发状态,solver_coupled_proxy.py:675-685)——这是 Picard 型每步迭代,默认 iterations=1
- `SolverAdmmCoupled`:线性化 ADMM(约束型耦合)
- 钩子:有效质量(块形式)、虚拟惯量代理、速度级 rewind、反馈力收割

## 决定性差异(源码行号 + 我们的量化证据)

| 能力 | Newton #2848 | 两遍窗口 PC(我们) | 证据 |
|---|---|---|---|
| **子步间耦合力时间插值** | ❌ ZOH——同一 control 重复用于全部 substeps(solver_coupled.py:2152,2169) | ✅ 力族线性插值(含矩阵算子) | **ZOH=42% 误差,插值=0.0005%**(500步,1e-6 基准) |
| **预测遍** | ❌ 全包 grep 无 predictor/extrapolation | ✅ 外推力推进到窗尾,在预测态求耦合量 | 预测遍初值达 1% → 窗口迭代只需 1-2 次 |
| **窗口级多速率** | substeps 存在但耦合交换在顶层步 | ✅ 34 子步/耦合解 + 窗内插值 | 子脱落实验:窗长与流体离散可解耦(5.2%→3.7%@2×窗) |
| **力算子传递** | 部分等价(虚拟惯量/有效质量钩子) | ✅ 附加质量矩阵时间插值进 M_eff | M*=0.3 下窗口迭代仍 10×/次收缩 |
| **外部基准验证** | 示例级演示 | ✅ 对 MATLAB 全状态 1e-6,GPU 0.0005% | fixtures+链式验收体系 |

## 定位结论

1. **两遍窗口预测-校正本身就是相对 Newton 生态的创新**——无需额外硬造:它填补的是 #2848 明确缺失的两件事(子步力插值、预测遍),且我们有同一算例上 42%→0.0005% 的端到端证据链证明这两件事是 make-or-break。
2. 求解器无关性:方案只要求耦合对象提供 (a) 状态快照/恢复(Newton 已有 `iteration_restart` 雏形)、(b) 子步力插值入口(需新增,可作为对 #2848 的具体建议)、(c) 可选的有效质量注入(钩子已有)。
3. 三轮可微分研究的边界结论(docs/grad_pc_study.md)同样适用于 Newton 场景:该方案不需要梯度,普通 Picard 重解(他们已有 `iterations`)加上述两件结构性能力即可。

## 对 #2848 的具体可贡献点(按 Newton 团队"外部包/示例先行"建议)
1. 子步耦合力插值钩子(force ramp across substeps)——最小 API 改动,我们有量化收益证据
2. 预测遍模式(`mode="predictor-corrector"`:先推后解再回退重推,基于现有 iteration_restart 机制扩展)
3. 气弹外部包示例(UVLM 力源 + 任意 Newton 结构求解器),复用 warp_fsi 已验证核
