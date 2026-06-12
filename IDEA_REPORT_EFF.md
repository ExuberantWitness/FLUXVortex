# IDEA_REPORT:FSI 链计算效率(research-pipeline Stage 1)

## 调研结论(逐点回应用户四条修正)

### 1. DPFMM 在低速扑翼是否合适?——保留意见成立,降级
FMM/DPFMM 的收益前提是**大 N + 远场主导**。低速扑翼恰好相反:尾迹对流慢、
贴着翼面(wake capture),诱导速度由**近场主导**;且我们当前 150 面板的
热点根本在结构装配(P0 实测 91%)。结论:FMM 移出近期路线,留作大网格
(真机翼/多翼)时与种群控制组合使用。
[DPFMM 文献](https://www.sciencedirect.com/science/article/pii/S1000936120303204)的
100min→2min 是 N~1e4 多旋翼场景,不可外推到本场景。

### 2. FLOWVPM.jl 调研(源码级)
[FLOWVPM_particlefield.jl](https://github.com/byuflowlab/FLOWVPM.jl) 实测:
**无合并/无重网格/无强度阈值删除**——只有 `maxparticles` 上限(满了报错)
与 static 粒子分类。其稳定性来自 rVPM 公式本身(可变滤波宽度),不解决
种群增长。结论:可借鉴其 rVPM 公式与 GPU kernel 结构,**种群控制需自研**。

### 3. 粒子无界增长(用户指出的核心机制)——文献地图
- [NVLM/VPM 自适应尾迹转换(2025/11,架构同型!)](https://arxiv.org/abs/2511.11430):
  面板→粒子转换守恒环量、按弧长自适应密度——**转换侧可直接借鉴**;
  但**无种群控制**(靠 CPU 集群 FMM 硬扛)。
- [Siemaszko 2025 "Downsampling the Vorticity Field"](https://doi.org/10.1002/fld.70002):
  正面攻击此问题(付费墙,待取);
- 经典答案 = **VPM 重网格化**(Koumoutsakos 谱系,M4' 插值到规则网格:
  既控数量又维持重叠/精度)+ 旋翼码的**远尾迹聚并**(守恒 0/1 阶涡量矩)。
- **我们当前实现的 drop-oldest 上限是静默丢环量——必须替换。**

### 4. 不做小批量测试——试点设计改为全尺度
种群机制的"全尺度" = 粒子维度上百倍周期(刚体扑翼场 0.02s/窗可跑
100+ 扑动周期、~30 万粒子),而非块数缩样。

## 已完成试点(P0/P1,真实链上)
| 试点 | 结果 |
|---|---|
| P0 组件基准 | kmem 切线 62%、弹性力 29%、LU 7.5%、流体 1%、粒子 0.3%(当前 2400 粒子时) |
| P1 批量 einsum 弹性 | **kmem 139→35ms(3.9×)、力 32→16ms(2×)**,K rel 9e-16、Qk rel 4e-16 ✓ |

## 排序的优化路线(预期×成本×风险)

| # | 路线 | 预期 | 成本 | 风险 | 定位 |
|---|---|---|---|---|---|
| 1 | **粒子种群控制**:距离分级 moment-conserving 合并(0/1 阶矩守恒,影响阈值给误差界)+ 转换侧借鉴 2511.11430 | 长时程粒子数 **有界**(O(影响阈值) vs 线性增长);解锁物理小时级的前提 | 中(~2-3 天:机制+全尺度验证) | 中(精度界需实测) | **核心**(用户 #3) |
| 2 | CPU 弹性矢量化(P1 已验证原型) | 链 ~2.5-3×(91% 热点的 ~3.5×) | 低(1 天落地+红线) | 低(数值 1e-15 级) | 立即 |
| 3 | GPU 延迟(CUDA graph/批量 LLT/device 簿记) | 单环境 ~50-200× | 高(3-5 天) | 中 | 物理小时级最终解 |
| 4 | FMM/treecode | 大 N 才有收益 | 高 | 低 | 缓,与 #1 组合时再上 |

## 下一步(Gate 1)
建议:**路线 1(种群控制)为本轮研究主体**,全尺度试点 = 刚体扑翼场
100 周期三方案对决(无界 / drop-oldest(现状) / 矩守恒分级合并),
判据 = 粒子数曲线、每窗成本曲线、升力史 vs 无界参考的误差;
路线 2 并行落地(独立、低风险、有红线)。

---

# 深调研补遗:VPM 种群控制专项(用户指令:排除重网格化,公里级作业域)

## 重网格化排除的确认
作业域数公里 → 规则网格不可行 ✓;且重网格本质是欧拉化,丢 Lagrangian 长尾迹优势。
**无网格合并是唯一适配形态。**

## 核心机制(有直接文献先例)
**[Morphing-wing UAV VPM(arXiv 2307.02371,MIT/Tedrake 组)](https://arxiv.org/abs/2307.02371)**
——与我们完全同场景(扑翼/变形翼 + VPM + 控制),原文:
> "replace pairs of vortices with one vortex when doing so changes the
> induced velocity on the wing less than a threshold amount. The new vortex
> has strength equal to the sum, located at strength-weighted-average position."

要素:①成对贪婪合并;②判据 = **对翼面诱导速度的改变 < 阈值**(物理误差界,
不是几何启发式);③合并公式守恒 0 阶矩(Σα 精确)+ 1 阶矩(加权质心)。
性质:1/r² 衰减 → 远处自动粗化、近场保细(**天然距离分级**);
脱落率=合并率 → **种群有界**;无网格 → 公里域无碍。
他们的坦诚短板:非实时(1.5s 机动规划 5s-1min,NumPy/GPU 可选)——
正是我们 GPU 线的空间。

## 支撑机理文献
- **[矩守恒 NNLS 合并(arXiv 2604.00668,2026)](https://arxiv.org/abs/2604.00668)**:
  任意阶矩守恒的严格版(octree 分箱 + NNLS),若 1 阶矩守恒不够时的升级路径
- **[SPH Vacondio 合并/分裂谱系](https://www.sciencedirect.com/science/article/abs/pii/S0045782512003842)**:
  质量+动量守恒合并的成熟机械,含 [GPU 加速版](https://www.researchgate.net/publication/283017978_GPU-accelerated_adaptive_particle_splitting_and_merging_in_SPH)
- **[NVLM/VPM 转换(arXiv 2511.11430)](https://arxiv.org/abs/2511.11430)**:
  面板→粒子转换的矩守恒 + 按弧长自适应密度(转换侧借鉴)
- 3D 矢量强度注意点:单粒子无法精确守恒 1 阶矩张量 → 误差判据(②)承担实责,
  优先合并方向相近的粒子对(α₁·α₂>0 阈值)

## 全尺度试点设计(Gate 1 提案,无小批量)
战场:刚体扑翼场(0.02s/窗),**100+ 扑动周期**(粒子维度全尺度):
| 方案 | 机制 |
|---|---|
| A 无界(参考) | 不控制,跑到内存/时间允许的上限 |
| B drop-oldest(现状) | 暴露静默丢环量的危害 |
| C **成对矩守恒合并**(新) | 翼面误差阈值 ε 扫描(3 档) |
判据:粒子数曲线(C 应饱和)、每窗墙钟曲线、升力史 vs A 的多维误差
(compare_metrics)、总环量守恒账本。随后柔性场(chain_hybrid)复测一档。
