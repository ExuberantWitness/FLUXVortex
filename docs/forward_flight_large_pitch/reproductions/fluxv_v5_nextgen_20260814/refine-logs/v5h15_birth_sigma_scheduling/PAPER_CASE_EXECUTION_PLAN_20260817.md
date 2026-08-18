# 三论文直接复现执行计划（DiGT-1 诊断性，所有者指令 2026-08-17）

指令：用当前最新算法（V5H15 链：graded k=5/r=4 格栅 + κ=1.75 出生 σ +
invariant-reconstructed WRK3 rVPM + 冻结 Ptera 父场）直接复现三论文并报告
结果。算法已冻结（V5H15 叶）；本任务为 DiGT-1 诊断性运行，输出不得作为
formal 复现成功，且此后不得依结果回调方法参数（污染规则见
OWNER_AUTHORIZED_DIAGNOSTIC_UNLOCK_20260817.md）。

## 共同工程：全周期 runner（三论文共用骨架）

现状缺口：V5H15 formal 链只跑 3 个 release（周期前 0.33s 启动瞬态）。
论文比较需要整周期（或多周期）相位分辨载荷：
1. 复用 executor 的 committer/solver 机制，把 Ptera 步进扩展到全周期
   （W2：T=3.56s / Δt=0.11125s → 32 步/周期；release 策略按各论文工况）；
2. 每 Ptera 步聚合力/力矩 → 相位 CL/CD（qS 用各工况量纲）；
3. 周期判据：相邻周期 CL/CD 相位差 < 收敛阈 → 取末周期；
4. Baik 合同：1 Hz sharp Fourier low-pass，与数字化 GT 400 相位点比
   RMSE/MAE（GT 在 docs/.../baik2012_w1_w4/runs/）。

## 论文一：Baik 2012 W2（最短路径，先做）
- 输入全部就绪：W2_CASE（executor 内冻结）、GT、滤波合同。
- 运行 2–3 周期 N=143（κ/格栅继承）；预计 CPU ~2–4 h。
- 产出：W2 CL/CD 相位曲线 + RMSE/MAE。

## 论文二：Izraelevitz Fig.14（第二优先）
- 几何/运动：NACA 63A015，AR=3（c=0.1016, b=0.3048），俯仰轴 0.75c，
  z=h cos(ωt)、θ=θmax cos(ωt+ψ)，h/c=0.6、St=0.2、k=π/6、J'=6；
  15°（ψ=15..105°，7 条件）+25°（ψ=45..105°，5 条件）。
- 工程：新 movement（正弦，比 W2 简单）+ 网格/粒子生成按 AR=3 平板；
  CT = 周期平均 −Fx/(qS)；Cd0=0.057 主口径。14 marker MAE/RMSE。
- 运行：12 条件 × 2–3 周期，CPU ~4–8 h。

## 论文三：Yang 2025（第三优先）
- 几何/运动：c=0.130、b=0.250、U=5.5、f=2.5 Hz、安装攻角 0–25° 六条件，
  nominal four-bar 扑动（docs/.../plev2025/source_data/ 数字化）。
- 工程：four-bar 运动学最大（非简谐+扑动）；周期均值 lift/thrust MAE
  （gf），D=−T 口径。
- 运行：6 条件，CPU ~2–6 h。

## 执行顺序与汇报
W2 全周期 → Izraelevitz 12 条件 → Yang 6 条件；每个 case 完成即报
RMSE/MAE + 曲线数据落盘（DIAGNOSTIC 标注）。

## 下一会话开工入口
- 算法基线：V5H15 四叶 + token（/tmp/fluxv-v5h15-audit-20260817-ZHUMgt/）
- 全周期 runner 从 _run_formal_level 骨架派生（probe 驱动
  /tmp/v5h14-probe/driverC.py 已验证捕获机制）
- 首批诊断数字（3-release 瞬态，载荷级收敛 1e-6）：见
  tracker 终局记录
