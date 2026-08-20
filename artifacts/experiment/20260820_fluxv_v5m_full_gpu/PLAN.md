# FLUX-V5M 完全 GPU 化实施计划

日期：2026-08-20
分支：`run/v5m-full-gpu-20260820`
基线提交：`f6251cd`
状态：实施中

## 1. 目标与术语

本任务把“FLUX-V5M 完全 GPU 化”定义为一个可机械验证的生产合同，而不是“程序中出现了 CUDA kernel”。在一次求解的时间推进区间内，下列科学数值计算必须全部在同一 CUDA 设备上完成：

1. Ptera 环量影响矩阵、尾迹诱导、稠密求解和面板载荷；
2. LEV/TEV 判定、联合系统、粒子诱导、粒子推进、冲量载荷及账本归并；
3. LDVM 状态推进及有限翼修正；
4. Warp-FSI 流体/结构推进、耦合和线性求解；
5. 论文验证指标、归约与边界门。

CPU 仅允许负责配置解析、几何/第三方对象构造、控制流、显式结果序列化、文件 I/O 和 GPU 遥测。禁止 NumPy/SciPy/Numba/BLAS 在时间推进区间执行科学数值计算，禁止 CPU fallback，禁止混合 CPU/CUDA 输入被静默上传。

“生产支持”只授予经过本计划正门与负门的模式。通用 Ptera 中尚未迁移的任意组合不被冒充为 V5M：生产入口必须在首步前明确拒绝。

## 2. 冻结输入

- 四论文 GPU 验证基线提交：`f6251cd`
- 已审计报告：`REPORT_GPU_ONLY_V2_20260820.md`
- 当前 CUDA Ptera 后端：`platform/warp_vpm/bing_joint_ptera_gpu.py`
- 当前 CPU 对照实现：`platform/warp_vpm/bing_joint_ptera.py`
- 当前粒子实现：`platform/warp_vpm/pfield.py`
- 当前 Warp-FSI：`src/fluxvortex/warp_fsi/`

这些输入只作对照；任何精度变化都必须由 CUDA/CPU 同工况差分测试显式记录。

## 3. 能力矩阵

| 能力 | 基线状态 | 本任务目标 | 验收方式 |
|---|---|---|---|
| 单翼 attached、prescribed wake | CUDA | 保持 | 位级/容差回归 + Nsight |
| 单翼 attached、free wake | 部分 CUDA，未授权 | CUDA | wake 坐标/age/速度无 host 数值路径 |
| active LEV、post-hoc TEV | CPU 主路径 | CUDA | active/inactive/出生/推进/载荷门 |
| joint LEV+TEV | CPU `np.linalg.solve` | CUDA | 增广矩阵、残差和 Kelvin/LESP 门 |
| 多翼/多机 | 未支持 | CUDA 或首步 fail-close | 独立能力门，不静默降级 |
| image surface | 未支持 | CUDA 或首步 fail-close | 镜像符号/诱导差分门 |
| LDVM + corrections | CUDA | 保持并收紧混合设备门 | 所有输入 exact CUDA device |
| Warp-FSI 主求解 | GPU kernel + 部分 host 数值 | CUDA | monkeypatch host 数学攻击 + profile |

## 4. 实施顺序

### M0：统一生产入口与 fail-close

- 新增唯一公开 V5M GPU 工厂/能力描述。
- 强制 CUDA、float64、单设备、无自动 CPU 上传。
- legacy CPU solver 明确标为参考/诊断，不再出现在生产用法文档。
- 为 unsupported 模式提供首步前非零退出和可读原因。

### M1：GPU 粒子容器

- 粒子位置、Gamma、sigma、circulation、类型和出生步常驻 CUDA。
- 添加、裁剪、归约、自诱导、目标诱导和 WRK/RK 推进均在 CUDA。
- 只在最终序列化或显式审计快照时复制到 host。
- CPU/混合设备输入必须拒绝，不能自动迁移。

### M2：active-LEV 与 joint-TEV

- 将 LESP、active mask、Gamma cap、粒子出生和增广系统移至 CUDA。
- 将 Kelvin、Neumann、LESP、几何、有限性与账本闭合门移至 CUDA 归约。
- CUDA 结果与冻结 CPU reference 做受控差分，不使用 CPU reference 参与生产计算。

### M3：尾迹、镜像与拓扑

- prescribed/free wake 的坐标、age 与诱导全 CUDA。
- 对多翼/多机/image surface 逐项迁移；未通过的组合保持 fail-close。
- 拒绝 `only_final_results=True` 触发第三方 host finalize，或提供 CUDA finalize。

### M4：Warp-FSI

- `GPU_ONLY` 成为 V5M 生产入口不可覆盖的硬合同。
- 结构残差、迭代门、流体/链模型中的运行时 NumPy/SciPy 运算迁移到 Warp/Torch CUDA。
- 小批量、空批量、收敛失败、设备漂移均 fail-close。

### M5：端到端审计

- 运行单元/集成/负攻击、四论文回归、运行时监控和 Nsight。
- fresh-process 独立复算摘要与源码 SHA。
- 只有所有声明模式均有 profile 和负门时，才允许写“完全 GPU 化”。

## 5. 必过正门

1. `torch.cuda.is_available()` 且所有科学 tensor 均为 CUDA float64；
2. attached、active-LEV、joint-TEV 各至少一个非退化动态工况；
3. 粒子数非零时完成多步推进，账本闭合且状态有限；
4. GPU 与冻结 CPU reference 在预注册容差内一致；
5. 四论文指标不退化超过既定容差；
6. Nsight 显示 solve、Biot-Savart、粒子、载荷、归约与修正 kernel；
7. 生产运行中 host 数学攻击未被触发。

## 6. 必过负门

- CUDA 不可用、设备不是 CUDA、任一科学输入在 CPU：首步前拒绝；
- monkeypatch `numpy.linalg.solve/einsum/norm/sum`, SciPy solve 和 Ptera host finalize/wake 更新：生产运行不得命中；
- unsupported 多翼/image/free-wake 组合：若未授权，首步前明确拒绝；
- GPU OOM、非有限输出、账本漂移、Kelvin/LESP/Neumann 超门：非零退出且不发布 PASS；
- 运行中设备、dtype、kernel/callable 绑定漂移：拒绝并允许 fresh clean retry。

## 7. 声明边界

本计划不会把 Python 控制平面称为 GPU，也不要求第三方几何对象驻留 GPU。只有“科学数值数据面完全 CUDA 化”可被声明。若能力矩阵仍有一个 V5M 生产模式为“未支持”，最终报告必须写“严格 GPU 子集”而不是“完全 GPU 化”。
