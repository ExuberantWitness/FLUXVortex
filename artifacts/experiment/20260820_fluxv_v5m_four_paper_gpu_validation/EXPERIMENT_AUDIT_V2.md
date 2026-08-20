# EXPERIMENT AUDIT V2

- run id: `20260820_fluxv_v5m_four_paper_gpu_validation`
- date: 2026-08-20
- reviewer independence: `same-family`
- acceptance status: `provisional`
- overall verdict: **PASS_WITH_WARNINGS**
- blockers: none

## A–F verdict

| Dimension | Verdict | Summary |
|---|---|---|
| A. Ground-truth provenance | PASS | 四篇均为数字化实验观测；Baik scored 去重 GT 与纯 GT 公共 3200 点逐位相等 |
| B. Score normalization | PASS | 无预测自身归一化；全部论文指标可从持久化 raw curve 独立重算 |
| C. Result integrity | PASS | summary、canonical result、NPZ payload、源码与 GT SHA 全部闭合 |
| D. Executed path | PASS | 15/15；CUDA 计数、攻击回归、遥测与 current-source Nsight 均闭合 |
| E. Scope | PASS（限定） | 仅支持 24 个冻结确定性工况；无重复/收敛/不确定度证据 |
| F. Evaluation type | real_gt | 模型预测对数字化实验观测 |

## Frozen evidence

- Baik summary SHA-256:
  `f2fba13edd70f089f8e3d7da027e7d98c3ba178f228db06720919b9a94104592`
- Three-paper summary SHA-256:
  `6ecd1a046f4d31391485c38ff92e585e7803cd03a815e975f55f90b19cd9abcb`
- Three-paper canonical result SHA-256:
  `c4cc9bcc635717351982f611747dcd069293463902e0721167fff049fd5ac437`
- GPU metric replay SHA-256:
  `5479580ebf8fd0b86c357e3c006db75056aaef262f407ed92e0ff5a25d75bbf8`
- Final Ptera CUDA backend SHA-256:
  `9199e87a7d3023f330ba4e62376fe51f0220a09f1e290a7ca4abe7b1838f3a62`
- Final CUDA corrections SHA-256:
  `177cf0f9e4cb6c9c666ee01f6cc9767f86e873c873c618af97c52bdce16d87e3`
- Final LDVM CUDA SHA-256:
  `31f19712679efa5af92d9bf1b9b9d3c998d6a11a421cf67c4f975ced5bb7b6d1`
- Nsight report SHA-256:
  `6001d5900a0ff943cccacc1e54dd8dfe183baba2d303a9223b6629f01f636fa9`

## GPU path findings

- 三篇三维累计 AIC/wake/solve/load/ledger 各 11,022；wake-convection
  11,002。
- Baik 遥测峰值 56% / 1181 MiB / 57 samples。
- 三篇遥测峰值 75% / 11,146 MiB / 2,032 samples。
- Nsight 记录 16,516 次 CUDA kernel launch，并捕获 reduction、cross、
  CUTLASS GEMM、GPU LU 与 TRSM。
- 允许的 host 边界仅为几何/配置对象、调度、数据搬运、I/O、序列化和
  遥测；正式结果的气动科学数值与评分在 CUDA float64 上执行。

## Warnings

1. 有限翼投影的混合 CPU/CUDA 输入会上传到 CUDA 后计算，并非所有 CPU
   输入均 API fail-close；正式 runner 五个输入均原生 CUDA。
2. G0/G0b/G0c/P0 字面量属于 `fa8eaca` 提交级旧诊断，不是本轮四论文
   GPU fresh 结果。
3. 每工况只有一次运行；Nsight 是 8 步机制 smoke。
4. 不覆盖三维 active-LEV、多翼、镜像面、自由尾迹或通用 Ptera。
5. 旧 `REPORT_GPU_ONLY_20260820.md` 保持撤回状态，不得引用其旧数字。

## Supported claims

- 支持四篇冻结工况的最终指标。
- 支持 Baik、Yang、Izraelevitz 优于 V4B；Mancini 仅 PARTIAL。
- 支持冻结入口的科学数值链使用 CUDA GPU。
- 不支持整个 FLUX-V5M/Ptera 全路径 GPU-only、四篇全部优于 V4B、鲁棒性
  或泛化性声明。
