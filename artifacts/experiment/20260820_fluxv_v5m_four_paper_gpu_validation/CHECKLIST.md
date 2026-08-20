# FLUX-V5M 四论文精度与 GPU 验证清单

## Identity

- run id: `20260820_fluxv_v5m_four_paper_gpu_validation`
- idea id: `fa8eaca-fix-replay-plus-four-paper-gpu`
- stage: GPU-only V2 complete; second fresh audit PASS with non-blocking warnings

> 首轮独立审计为 **FAIL**，其记录保留在 `EXPERIMENT_AUDIT.md`。
> V2 已完成修复、全矩阵 fresh rerun 和第二次独立审计；最终限定结论为
> **PASS_WITH_WARNINGS**，见 `REPORT_GPU_ONLY_V2_20260820.md` 与
> `EXPERIMENT_AUDIT_V2.md`。

## Planning

- [x] 研究问题与零/备择假设冻结
- [x] baseline 与比较合同冻结
- [x] smoke、full run、停止条件已写；CPU fallback 已禁止
- [x] 四篇 runner 与 GT/输出逐项核对
- [x] GPU 覆盖边界逐函数核对：冻结论文入口均 `enable_lev=False`；2D/修正已迁移 CUDA

## P0/P1 Fix Replay

- [x] P0-1 ledger clip closure：`0.00e+00`
- [x] P0-2 auto CUDA selection：RTX 4090 D / Warp CUDA
- [x] P0-2 forced CPU 旧诊断：finite，和 GPU 仅末位差；不计入 GPU-only 结论
- [x] P0-2 invalid/unavailable device fail-fast：CPU/bogus/no-CUDA 均 fail-fast
- [x] P1-1 G0 production chassis and frozen bound：CL `0.4850` in `[0.4700,0.4882]`
- [x] P1-1 failure exits nonzero negative control：exit 1
- [x] P1-2 G0b failure exits nonzero negative control：exit 1
- [x] P1-2 G0c failure exits nonzero negative control：exit 1
- [x] legacy G0/G0b/G0c positive paths：此前诊断 all exit 0；因含 CPU Ptera，不计 GPU-only 科学证据

## GPU Evidence

- [x] GPU/driver/compute capability recorded：RTX 4090 D, 24 GiB, sm_89
- [x] actual Warp CUDA device recorded
- [x] CUDA kernel path exercised
- [x] GPU utilization and memory observed during G0c：SM peak约34%，+388 MiB
- [x] CPU fallback 曾做数值诊断；自 GPU-only 冻结后禁止再次使用
- [x] each paper classified：当前冻结四论文评分均 CPU/hybrid-host，非 active-LEV GPU
- [x] GPU-only hard gate：`cpu`/非法设备/无 CUDA 均非零退出
- [x] GPU UVLM 诱导速度、影响矩阵、线性求解与载荷回归
- [x] GPU LDVM 诱导速度、方程求解与时间推进回归
- [x] GPU polar/ledger 修正回归
- [x] 每篇运行证明无 CPU 数值 fallback

## Four Papers

- [x] Baik W1–W4 GPU-only fresh（CPU fresh 仍仅诊断）
- [x] Yang full frozen matrix GPU-only fresh（CPU fresh 仍仅诊断）
- [x] Izra Fig.14 full frozen matrix GPU-only fresh（CPU partial 不计）
- [x] Mancini fast/slow GPU-only fresh
- [x] all metrics compared to experiment and V4B
- [x] historical outputs excluded from fresh evidence

## Validation

- [x] outputs and hashes complete
- [x] metric keys finite
- [x] intentional CUDA source changes and runtime SHA bindings recorded
- [x] claim validation completed：3 papers improve; Mancini partial
- [x] summary/report completed
- [x] next action explicit：fresh read-only review, then isolated commit

## Independent Audit Remediation

- [x] 首轮 `EXPERIMENT_AUDIT.md/.json` 落盘，overall=`FAIL`
- [x] Ptera prescribed-wake 坐标推进迁移到 CUDA，并以父类方法攻击回归封锁
- [x] LDVM LEV shedding 改为 CUDA masked circular buffer；step 内无 `.item()`/host bool
- [x] strip-area、finite-wing gain、polar slope、Baik macro score 改由 CUDA 计算
- [x] GT SHA、原始预测曲线、metric contract 与 GPU runtime monitor 接入 runner
- [x] 四论文 GPU-only v2 full fresh rerun
- [x] 完整 metric contract/曲线/哈希离线重算
- [x] 第二次 fresh independent audit：PASS_WITH_WARNINGS，无 blocker

## Final Scope Warnings

- [x] Mancini fast/slow 保持 PARTIAL，禁止写成优于 V4B
- [x] G0/G0b/G0c/P0 旧字面量仅作提交级诊断，不计入 fresh GPU claim
- [x] CPU 仅允许几何/配置对象、调度、搬运、I/O、序列化与遥测
- [x] 混合 CPU/CUDA 投影输入会上传 GPU，而非 API fail-close；正式入口全 CUDA
- [x] 不声明 active-LEV、多翼、镜像面、自由尾迹或通用 Ptera 已覆盖
- [x] 每工况一次运行；无鲁棒性、收敛性或不确定度声明
