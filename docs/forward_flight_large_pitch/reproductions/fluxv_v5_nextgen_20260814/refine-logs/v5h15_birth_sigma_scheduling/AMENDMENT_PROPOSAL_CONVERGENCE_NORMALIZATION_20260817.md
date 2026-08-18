# Amendment 提案：inner 收敛门的归一化基数（供 V5H16 预注册）

## 1. 合同出处（追溯结果）

V5H11 EXPERIMENT_PLAN.md L151-152："ID-aligned X/Gamma/sigma/material/
frontier N64->128 relative-L2 各 ≤1e-6"。**原文未规定归一化基数**；
实现选择了"对通道自身 N143 末态范数归一"（run_*_w2.py stable_norm）。
该合同写于谱系从未有过完整矩阵的时期（预期大通道早尾迹形态）。

## 2. 证据（V5H15 formal A + Probe C）

- 27 项指标 ratio 6.2–15.6（干净 2.6–3.2 阶收敛）；
- 失败 5 项全部为绝对差 ~1e-9 级、但通道末态量级小 → 相对值被放大
  （L3/γ：d=5e-9，rel=6.3e-5 → 归一基数仅 ~8e-5 量级）；
- κ 1.75→1.45（σ −17%）相对误差不降反杂动 → 与分辨率无关，
  纯归一化效应。

## 3. 文献依据

- [Oberkampf & Roy, Verification and Validation in Scientific Computing（Cambridge, 2010）](https://www.cambridge.org/core/books/verification-and-validation-in-scientific-computing/05CA1F8F3CCB5AE5445FDF55239A0183)；[Roy & Oberkampf 2011, CMAME](https://www.sciencedirect.com/science/article/abs/pii/S0045782511001290)：收敛验证的误差范数框架；
- [P1 Error（arXiv:2403.07492）](https://arxiv.org/html/2403.07492v1)：**纯相对误差在参考量级小时病态放大**，标准修法为混合判据 `error/max(|ref|, atol) ≤ tol`；
- 实践惯例（[FEniCSx 收敛教程](https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html)等）：归一基数取良态参考尺度。

## 4. 提案的修正（最小改动）

`relative_64_128 := d64_128 / max(reference_norm, birth_norm)`，其中
`birth_norm` = 该层**出生态**同通道范数（trajectory 记录的 start 通道，
物理上即释放涡量的自然尺度，良态非小）。阈值 1e-6、ratio ≥1.5、
roundoff 豁免及其余 22 项全部不变。

## 5. 零成本预检（预注册前置）

用**已归档**的 V5H15 formal A 工件（trajectory start/end 通道）按新公式
重算 27 项——若 27/27 PASS，amendment 在既有证据上验证成立；V5H16 分支
只需 fork + 新公式 + 重跑 formal A/B。若仍有失败项，则问题不止归一化，
提案作废并回到诊断。

## 6. 边界

- 仅为归一化基数定义的显式化（填补合同未规定处），非放宽阈值；
- ratio 门与绝对 roundoff 豁免原样保留；
- 需所有者批准后以 V5H16 名义预注册执行。

## 7. 预检结果（2026-08-17）：**证伪，提案作废**

按第 5 节零成本预检在已归档 V5H15 工件上重算：L2/γ 仍 1.08e-6、
L3/γ 仍 6.29e-5、L3/σ 6.5e-6、L3 两 tracer ~1.7–2.0e-6——出生尺度分母
不改善任何失败项。**反向发现**：L2/γ 的参考范数（~1.2e-7）比 L3/γ
（~8.8e-5）小 700 倍却相对误差更小 → 失败与通道量级无关；
layer-3 状态通道存在真实的、干净三阶收敛的相对误差 6.3e-5，达到 1e-6
需 N 再细化 ~4×（矩阵冻结不可行）。同时确认：**物理载荷门
（force/moment ≤2e-3）27 项中全部通过**——失败仅限状态通道的严格容差。

## 8. 修正后的真实选项

- (a) **状态通道容差重标定**：1e-6 对晚尾迹状态通道从未可达（谱系无
  先例数据支持该值）；物理意义门（载荷 2e-3）已全过。需更强文献/合同
  考古论证 state-channel 容差的合理量级（如 1e-4），amendment 级；
- (b) **细化矩阵**（新增 N=286 级）：改动合同矩阵，重成本低收益存疑；
- (c) **接受 inner-gate negative result**：以"载荷门全过 + 状态通道
  6.3e-5@三阶"作为结论写入论文边界声明。
