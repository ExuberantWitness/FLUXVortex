# Research Idea Report

**Direction**: 可微分梯度信息能否实质性增强强耦合分区 FSI 的预测-校正窗口推进(负结果复查与深化)
**Generated**: 2026-06-11 | **Pipeline**: research-lit → idea-creator(外部 LLM 401,Claude 生成,已注明)→ pilots → novelty-check
**Ideas**: 10 生成 → 3 组 pilot(6 runs)→ 1 推荐 + 1 备选

## Landscape(摘要,详 research/LIT_LANDSCAPE.md)
我们的两遍方案 = waveform iteration 家族(preCICE/Rüth-Uekermann-Mehl 线)的 1 阶 1-Picard 特例。高阶 waveform 靠多采样;**带导数 Hermite 在 FSI 线无先例**(co-sim/FMI 有线性系统版);**JVP 耦合预测器无先例**(最近邻 2025 ML 预测器);**子脱落解耦无专门研究**;质量比不稳定性已证(任意 staggered 方案都存在失稳质量比),重载 = 公认难区。

## Pilot 结果(t*=0.546,对各自 picard3 收敛参考)
| Pilot | 配置 | 结果 | 信号 |
|---|---|---|---|
| P3 重载探针 | M*=0.3, std vs picard3 | **分裂误差 3.8%(M*=1 时 0.022%,放大 ~170×)** | ✅ 强阳性 |
| P2 子脱落 | 2× 窗 + S=2 | 5.2% → 3.7%(关 28% 缺口;尾迹分辨率与力采样双机制确认) | ✅ 部分阳性 |
| P1 插值阶 | M*=1, quad/hermite | 无增益(2.25e-4 基线已近收敛,与诊断一致);quad(陈旧历史点)反而 1.3e-3 更差 | ⚪ 中性(机制按预期) |
| P1 参考 | dense4 | 实现缺陷:样本间最近邻=零阶保持 → 3.4% 偏差,**不可作参考**;修正=样本间线性插值 | 🔧 修正点 |

## 🏆 推荐 Idea:重载分区气弹耦合的"梯度增强适用域边界图"
- **假设**:窗口预测-校正的分裂误差随质量比 M* 急剧放大(pilot:170×@M*=0.3);在该区域,导数信息(JVP-Hermite 窗内力插值 + 梯度加速迭代)以低于 Picard 重迭代的成本恢复精度;在 M*≈1 区域增强无收益(已证负)。产出 = (M* × 窗长) 平面上"哪种方案最经济"的边界图 + 机制解释。
- **最小完整实验**(全本地,~2-3 h 批量):M* ∈ {1.0, 0.5, 0.4, 0.3} × 方案 {std, picard2, hermite-JVP} × 窗 {1×, 2×+子脱落},8 块,指标 = 距各自 picard3 + 流体解数 + 墙钟;插值阶研究在 M*=0.3 用修正后的 dense 参考重做。
- **新颖性**:CONFIRMED——质量比稳定性分析(Causin/Förster 一族、AMP)关注稳定/迭代数,无"窗口方案精度边界图";JVP/可微分预测-校正无先例;最近邻 = 2025 ML-学习预测器(机制不同:学习 vs 解析导数)。
- **风险**:LOW-MEDIUM(双向都有价值:若 hermite-JVP 在重载也输给 picard2,边界图+机制仍是干净的 empirical 贡献)
- **审稿人最强反对**:"M* 缩放 = 改质量矩阵,非物理一致的重载算例" → 应对:M* 是该无量纲家族的标准参数,且结论以同-M* 自洽收敛参考衡量;补充敏感性说明。
- **Pilot**: POSITIVE(P3)

## 备选 Idea:子脱落解耦(P2)
作为推荐 idea 的窗长维度组件并入;单独成文价值中等。

## Eliminated
| Idea | 原因 |
|---|---|
| I4 quad-history 单独成文 | pilot 显示更差(陈旧点);作为消融保留 |
| I7 能量指标 | 工具性,并入指标体系 |
| I10 精确 JVP vs FD | 待 hermite 在重载有信号后才有意义 |
