# FLUX-V5M 完全 GPU 化检查表

日期：2026-08-20
状态：完成（PASS with WARNINGS）

## H0 — 基线与隔离

- [x] 四论文 GPU 验证固化为本地提交 `f6251cd`
- [x] 新建隔离分支 `run/v5m-full-gpu-20260820`
- [x] 记录基线不完整项：active-LEV、joint-TEV、多翼、image、free-wake、Warp-FSI host 数值
- [x] 保存所有目标文件初始 SHA-256

## H1 — 统一生产合同

- [x] 新增统一 V5M GPU 入口
- [x] exact CUDA device + float64 + single-device 检查
- [x] 禁止隐式 CPU→CUDA 上传
- [x] 生产能力矩阵可查询且 immutable
- [x] 所有未授权模式首步前 fail-close
- [x] 更新 `HANDOFF_MODEL_USAGE.md`，移除 CPU fallback 叙述

效果检验节点 G1：入口/设备/能力矩阵负门全部通过；legacy CPU solver 无法由生产入口到达。

## H2 — GPU 粒子与 active-LEV

- [x] CUDA 常驻粒子状态
- [x] CUDA Biot-Savart self/target
- [x] CUDA 多级粒子推进
- [x] CUDA LESP/active/cap/birth
- [x] CUDA impulse/ledger
- [x] 非零粒子动态回归

效果检验节点 G2：monkeypatch NumPy 粒子/LESP/载荷算子后 active-LEV 仍完成；Nsight 观察到粒子 kernel。

## H3 — joint-TEV 与尾迹拓扑

- [x] CUDA 增广 A/b 组装和 solve
- [x] CUDA Kelvin/Neumann/LESP residual
- [x] prescribed wake 全 CUDA
- [x] free wake 全 CUDA
- [x] 多翼/多机/image 明确不属于 V5M 生产合同并 fail-close
- [x] host `_finalize_loads` 永不执行

效果检验节点 G3：active 和 joint 两种非退化工况通过；CPU solve/wake/finalize 攻击全部未命中。

## H4 — Warp-FSI

- [x] `GPU_ONLY` 为不可降级生产合同
- [x] V5M facade 排除 `ml_fluid/ml_chain` CPU reference
- [x] 结构残差/收敛归约 GPU 化
- [x] 小/空批量和非有限失败闭合
- [x] MATLAB fixture 的 STRUCT_CG/COUPLING/TRAJ/NEWMARK_AM CUDA 回归

效果检验节点 G4：host 数学攻击未命中；Warp profile 显示流体、结构、耦合和归约 kernel。

## H5 — 数值与性能复验

- [x] attached CUDA 回归
- [x] active-LEV CUDA/CPU reference 差分
- [x] joint-TEV CUDA/CPU reference 差分
- [x] 四论文 fresh GPU 回归
- [x] runtime monitor 计数闭合
- [x] Nsight 完整工况 profile
- [x] 无 CPU science kernel/BLAS 调用

效果检验节点 G5：所有预注册数值门通过，且运行证据与冻结源码哈希一致。

## H6 — 独立审计与交付

- [x] fresh-process 实验审计
- [x] 结果、源码、日志、profile SHA-256
- [x] A–F 完整性结论
- [x] 明确支持矩阵与剩余限制
- [x] 仅对授权生产模式声明 CUDA float64 科学数据面；保留 CPU 控制面与范围警告

效果检验节点 G6：独立审计 PASS；若有未闭合能力，则降级声明并继续实施。
