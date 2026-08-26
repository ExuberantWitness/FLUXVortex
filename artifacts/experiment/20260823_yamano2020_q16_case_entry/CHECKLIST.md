# Yamano 2020 Q16–DVM FSI 复现清单

状态：`EIGHT_ENDPOINT_PARTIAL_ACCURACY`

## Provenance

- [x] 冻结论文、single-sheet 参数、轨迹与首步 fluid fixture 哈希
- [x] 使用九自由度布局的正确 tip-z oracle
- [x] 固定 `0.002 × 34 = 0.068` 时间谱系
- [x] 排除旧三自由度误索引参考

## 实现

- [x] Q16-only、CUDA float64、无 CPU 数值 fallback
- [x] separated LEV mandatory，`Lcrit=0.11` 条件释放
- [x] joint TEV/free wake/predictor-corrector 原子事务
- [x] Ptera `dGamma` 与显式 `Mf1` 去重
- [x] 论文 Q4 局部 Mf1 → Q16 运动学/虚功投影
- [x] 论文 `dp_lift1` + `p_interp` 常量压力 → Q16 投影
- [x] `lift2`/`Mf21` 分布式压力传递
- [x] 作者 corrector `beta=0,...,33/34`
- [x] 外步起点切线缓存 + 困难子步实时 GPU 切线刷新
- [x] 逐外步原子进度证据

## 数值与事务门

- [x] 5×3 Q16 首步加速度比 MATLAB：1.000049
- [x] 5×3 Q16 前五模态 RMS 误差：0.832%
- [x] 3 个端点到 `t*=0.204` 最大误差：4.906%
- [x] 4 个端点到 `t*=0.272` 最大误差：4.906%
- [x] 8 个外步全部完成，wake 计数严格为 1…8
- [x] 8 步结构残差 ≤3e-7、耦合残差 ≤5e-7、功平衡通过
- [x] 8 步 separated LEV 始终启用；LESP 未越过 0.11，故合法零释放
- [ ] 8 个端点最大位移误差 ≤5%（当前 27.385%）
- [ ] `t*=1.0` 长轨迹门

## Mf2 独立契约

- [x] 冻结 MATLAB step2/3/4 fixture 哈希与 `Mf2_vec1` 幅值审计
- [x] 冻结 step1–4 `Mf2_vec1` 逐面板输入/输出 oracle
- [x] CUDA `dt_generate_q1234_mat` 逐面板回归通过
- [x] free-wake 顶点速度进入 predictor/corrector 分支状态与事务哈希
- [x] Q4 `Mf2` 压力投影逐自由度与 MATLAB 闭合
- [x] 诊断直接接入失败已保留：4 步误差 23.877%，不得作为正式改进

## FLUX-V5M 科学坐标门

- [ ] Q16/作者坐标直接作为 V5M 科学坐标，不经过 Ptera 展示旋转
- [ ] 弦向与来流同向门通过
- [ ] 面板法向与 Q16 有向曲面法向一致
- [ ] 新生和历史 free-wake 下游方向门通过
- [ ] AIC/环量方向契约有独立断言
- [ ] 科学坐标载荷保持力、矩和虚功闭合
- [ ] 坐标门通过后才允许 `Mf2` 进入正式广义力
- [ ] 保持 `Mf21` 与 `Mf2_vec1` 独立且 separated-LEV owner 只加一次
- [ ] 2 步与 4 步 pilot 通过后重跑正式 8 点
- [ ] 8 点 ≤5% 后再扩展到 `t*=1.0`
