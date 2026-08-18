# V5H15 tracker

| ID | Gate | Status |
|---|---|---|
| V5H15-G0 | preregistration (KAPPA frozen, predictions frozen) | PASS_DOC |
| V5H15-G1 | fork + constant + tests | TODO |
| V5H15-G2-001 | dependency re-sign | BLOCKED_BY_G1 |
| V5H15-G2-002 | disposable smoke | BLOCKED_BY_G2_001 |
| V5H15-G3-001A | formal A | BLOCKED_BY_G2_002 |
| V5H15-G3-001B | formal B + parity + audit | BLOCKED_BY_G3_001A |

## 终局记录（2026-08-17）

- **G1 PASS**：8 文件 fork + κ=1.75（σ=0.00266 精确冻结）+ 127 测试全绿；
  hostile review 的 1 个 must-fix（freeze 路径）已修复并复验。
- **G2-001 PASS**：token `/tmp/fluxv-v5h15-audit-20260817-ZHUMgt/`（41
  leaves + 209 modules）verified；V5H13/V5H12 旧 token fail closed。
- **G2-002 smoke PASS**：σ≈0.00266（κ 生效）；layer-1 max conv **0.0325
  与探针逐位一致**；全部门/账本/库存通过。
- **G3-001A formal A = 收敛门 STOP，但矩阵首次完整跑完**：
  `/tmp/fluxv-v5h15-formal-A-20260817`（verify_artifact 只读复验 PASS；
  2421 stages、9/9 层、153 载荷行、运行 ~85 min CPU）。
  - **9/9 预注册预测全部命中**（N32 三层与探针逐位一致
    0.0325/0.2018/0.2355；79/143 按 47/N 缩放 ±20% 内）；
  - 稳定性门全部通过（全矩阵 max conv 0.2355@N32 → 0.0703@N143，全部
    < 0.35 预注册硬上限）；
  - **收敛阶全部干净**：27 项指标 ratio 6.25–9.65（≈2.6–3.2 阶），
    22/27 通过；5 项失败全部是**继承的绝对容差** relative_64_128 ≤ 1e-6：
    layer2/γ 1.08e-6（超 8%）、layer3/γ 6.29e-5、layer3/σ 6.53e-6、
    layer3/material_tracer 1.98e-6、layer3/frontier_tracer 1.69e-6；
  - 解读：κ=1.75 消除了稳定性障碍，但 σ 平滑引入的误差底在晚尾迹
    （layer 3）小幅抬升了相对误差（γ 绝对差仅 5.5e-9，但通道量级小，
    相对归一放大）。layer3 ratio 6.3（低于渐近 8）提示接近 σ 误差底。
- formal B 依治理禁止（A 非 PASS）。κ 冻结不动；容差为 V5H11 冻结合同
  不可放宽。
- **谱系意义**：V5H12（协议）→V5H13（格栅+预测命中）→V5H15（稳定性
  解决+首个完整矩阵+干净 3 阶收敛）。剩余阻塞 = 晚尾迹绝对精度 vs 继承
  容差。
- **下一步选项（需新预注册）**：(a) κ 回调探针（1.4–1.5：稳定性预测
  0.2355×1.75/1.4≈0.29<0.5 仍有余量，σ 底按 σ^n 下降）；(b) 检查
  relative_64_128 的归一化基数（若为小量级通道的相对归一，可能测量的是
  噪声底而非分辨率）；(c) outer B4 gate 语境下重评该容差的合同来源。
