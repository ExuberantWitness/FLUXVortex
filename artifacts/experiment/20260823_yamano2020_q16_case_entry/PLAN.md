# Yamano 2020 single-sheet Q16–DVM FSI 复现计划

## 目标与硬约束

复现 Yamano et al., *Influence of boundary conditions on a flutter-mill*,
JSV 478 (2020) 115359 的 `single_sheet` 工况：前缘固支、AR=1、
`U*=25`、`M*=1`、`h/L=10^-3`、`nu=0.3`。

- 结构仅用 Q16；Q4 只作为论文气动积分中间空间，不是结构求解器。
- 科学数值路径为 CUDA float64，不允许 CPU 数值 fallback。
- separated LEV 从第 0 步启用，释放判据固定为
  `|LESP| > Lcrit=0.11`。
- joint TEV、free wake、predictor/corrector 共用可回滚事务；每个提交外步
  只允许一次真实尾迹推进。
- 不通过修改 `E/rho/Lcrit` 或参考数据追分。

## 冻结证据

- 正确 tip-z 自由度：MATLAB 九自由度节点布局中的
  `h_X_vec(9*(node-1)+3,:)`。
- 轨迹：`traj_long_t1.0.mat`，SHA-256
  `1dc0c0ab71e8aac72f08fce2b1b73ab3f6646ef4b19d19c4de3da5e5d52dd621`。
- 首步流体 fixture：`fixture_step1_t0.0680.mat`，SHA-256
  `1f9bf07638620b701a85c7a0ad816e74cd781ba8dc6c69f929e6b982d4f6b047`。
- 参考 CSV：`yamano2020_matlab_tip_dt002.csv`，SHA-256
  `b7105d5a382b4597b72a982d073762b39bf7263416c8c0e4651090963cdd414a`。
- 时间谱系：`dt_struct*=0.002`，34 个结构子步，`dt_aero*=0.068`。

## 已完成节点

1. 证明 2×2 Q16 不是正式结构网格：5×3 Q16 的首步 tip 加速度与
   MATLAB 相差 0.0049%，前五阶频率 RMS 误差 0.832%。
2. 把论文 Q4 局部 `Mf1` 经 `q_q4=T q_q16`、`T^T` 虚功投影到 Q16；
   前五模态 Mf1 RMS 误差降至 4.433%。
3. 修复论文常量环量压力：用 `dp_lift1`、弦向 `p_interp` 和 Q4 面压力
   装配替代 Ptera 四条线涡点力直接转置；首端点误差由 41.69% 降至
   4.906%。
4. 修复作者 corrector 子步时钟为 `beta=0,...,33/34`。
5. 正式 5×3 Q16 / 15×10 UVLM 已运行 8 个外步到 `t*=0.544`。前 4 点
   误差均不超过 5%；第 5–8 点出现累计偏差，末点误差 27.38%。
6. 结构侧采用外步起点 GPU 切线缓存；困难子步在 24 次准 Newton 后
   单次刷新实时切线。所有接受态仍由实时非线性残差验收。

## 当前判断

短时 CASE 已复现到 4 个论文端点，但 8 点轨迹尚未满足 5% 全程门。
纯结构、脉冲、Mf1 和瞬时环量压力已分别闭合。`Mf2_vec1` 的 CUDA
材料导数、AIC 反解和 Q4 全自由度投影现已分别与 MATLAB oracle 闭合到
浮点舍入量级；但是直接接入整机后，第 2–4 步 `Mf2` 合力变成
`+3.90/+18.05/+30.95 N`，而作者 fixture 对应约为
`-0.331/-0.894/-1.006 N`，并使第 4 点误差从 4.632% 恶化到 23.877%。

因此当前根因不再是 `Mf2` 公式，而是 FLUX-V5M 科学坐标被 Ptera 展示
坐标污染：作者/Q16 坐标要求弦向 `+x`、展向 `+y`、法向 `+z`、来流和
尾迹下游均为 `+x`；现有 `author` 路径把 Q16 点先当作 Ptera world，
经其 180 度展示旋转后安装为 GP `-x/-z`，但 `vInf_GP` 仍为 `+x`。
错误的 `Mf2` 整机直连必须视为诊断试验，不得成为正式结果。

## 下一开发与检验节点

1. 建立 FLUX-V5M 自有科学坐标层：Ptera 只保留 Panel/wake 对象容器和
   生命周期，不允许其 `GP->W` 展示矩阵进入 Q16 插值、AIC、自由尾迹、
   `Mf2`、压力或结构载荷计算。
2. 增加 fail-closed 科学门：`dot(chord,+vInf)>0`、气动法向与 Q16 曲面
   有向法向一致、首个自由尾迹行位于尾缘下游、所有载荷和运动在同一
   科学坐标中保持功共轭。任一门失败时禁止 `Mf2` 进入正式广义力。
3. 保留已验证的 CUDA `dt_generate_q1234_mat` 和逐面板 oracle；在新坐标
   层内保存真实 free-wake CUDA 顶点速度，并计算
   `Mf2_vec1=A^-1(-Gamma_wake_dt_q1234_n)`。未提交 trial 不得污染父状态。
4. 仅在坐标门全部通过后，把 `rho*Mf2_vec1` 与 `dp_lift1` 一起送入作者
   Q4 `p_interp` 压力装配，再以 `T^T` 投影到 Q16；`Mf21` 与 separated
   LEV owner 保持独立，任何一项不得重复。
5. 先跑单元/事务回归和 2 步 pilot，再跑 4 步辨别累计趋势，最后重跑
   正式 8 点。主门为第 8 点
   相对误差 ≤5%；辅助门为第 5–8 点不再单调发散、wake 计数严格 1…8、
   结构/耦合/功平衡门不退化。
6. 只有 8 点门通过，才扩展到 `t*=1.0`；不继续盲目延长错误轨迹。

## 本轮运行契约

- `run_id`: `yamano2020_q16_v5m_scientific_frame_mf2_20260823`
- 类型：先 `auxiliary/dev` oracle 与 2 步 pilot，再 `main/test` 8 点复现。
- 基线：`FULL_FSI_Q16_5X3_AERO_15X10_STEP8_AUTHOR_PRESSURE_SOURCE_BETA.json`。
- 唯一主要机制变化：把全部 Q16/FLUX-V5M 科学量统一到作者坐标，并在
  该坐标中恢复尾迹运动势压 `Mf2_vec1`；不改网格、时间步、材料、外载、
  `Lcrit`、LEV/TEV/free-wake 模式或误差定义。
- 预计改动：GPU 气动后端、增量尾迹事务、Q16 FSI 压力组合、Yamano
  oracle/测试与本目录证据文件。
- 计算预算：oracle/回归小于 10 分钟，2 步 pilot 约 3–5 分钟；通过后
  8 步正式运行约 12–18 分钟。
- 放弃条件：逐面板 oracle 不闭合、科学坐标门不闭合、出现 CPU 数值
  fallback、尾迹速度未进入事务哈希、LEV 被关闭或重复加力、2 步 pilot
  反向恶化且无法由独立分量解释。

## 停止条件

出现 CPU 数值 fallback、Q16 被替代、LEV 被关闭、尾迹 trial 被重复
提交、参考哈希漂移，或通过改物理参数追分时立即判该次运行无效。
