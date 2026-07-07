# P2-S5 强耦合移植:施工记录(2026-07-07,进行中)

> 设计依据:docs/p2_s5_coupling_research.md(00cacda)。目标:梁-膜翼耦合 FSI 稳定,
> 收 A2/A3 锚点,通往 S5b(闭合移植)→ S6(柔性 Fig17-19 vs 实测,终局)。

## T1 — madd 全 n⊗n 修复(用户拍板范围)✅

`newton_pc/adapters/flap.py` added_mass_operator 分支重写:
- L_v 只取 z → **三分量法向投影 Ñ(P,3nn)**,输出侧同构:M₃ = Lfᵀ·[ρ·area·n_c]·A⁻¹·Ñ;
- **对称化 ½(M+Mᵀ) + 定号投影**(eigh 截去正特征值——表观质量必须反抗加速度,
  去稳方向不允许上结构 LHS);截量占比记录:**31%**(AIC 非互易+单边度量的谱质量)。

**门禁 G1 PASS**(platform/p2_s5_gates.py):刚性静止翼上 madd 对均匀 a_z 的合反作用
55.2 g vs 解析平板 ρπ/4∫c²dy = 55.9 g(**ratio 0.989**,门禁 ±30%);展向分布跟随 c²
(中段带 [0.92, 1.08]);−Ms 严格 PSD(投影后 max eig 1.7e-19)。

## T2 — 窗口级强耦合迭代 ✅(Aitken → IQN-ILS 升级,触发条件记录在案)

- coupler(newton_pc/coupler.py):新增 `min_iterations`(SHARPy ≥3)与
  `adaptive_tol_rel`(KW:加速度/力量纲的绝对容差不可移植)——默认值全向后兼容。
- `platform/wing_aitken.py`:KW Aitken(ω 跨窗继承保号+限幅 0.5)。
  **实测局限**:强 added-mass 窗上残差平台/极限环(1.5e-5 平 50+ 次;5.6e-3 二环)
  ——标量松弛对多主模不动点无效,恰为 Degroote 记录的 IQN-ILS 触发条件。
- `platform/wing_iqn.py`:**IQN-ILS**(Degroote 2009)+ **跨窗历史复用 reuse=4**
  (二次学习界面 Jacobian;秩过滤 lstsq)。

## T3 — 耦合调试链(五个真问题,逐一确诊修复)

1. **幅值 ramp 冷启动 → 尾迹堆环自激**(S4 堵点②复现):前 8 窗近静止 commit,
   f 4→26→50→782→2.8e4 N。处方:全幅起步。
2. **全幅起步 θ̇ 阶跃 → 膜 kHz 振铃**:根排速度跳 11.35 rad/s,w3 死
   (带/不带 madd 皆然 → 排除 madd 机械)。处方:运动学一致 IC。
3. **madd 模式清零 dΓ/dt 丢环量非定常阻尼**:窗口收敛而轨迹发散(11→39→141 m/s)。
   处方:**BNV 广义 Robin 形式**——完整 dΓ/dt 保留在力里,LHS madd 配 RHS
   −madd·a_lag(a_lag = provider 解算时刻加速度,随力集插值);收敛时补偿→0,
   不动点=真耦合解,无双计。NodalForceSet 增加 a_lag 通道。
4. **钉线结构网格 → 锥度区畸形气动面元**:γ@TE 锥度面元逐窗倍增(4→1246),
   AIC 病态自激。处方:**气动格/结构格解耦**——气动侧回到弦向分数网格
   (S1 出口验证的 lattice),结构↔气动经静止构型预计算的重心插值 W
   (虚功一致 f_s=Wᵀf_a;madd 变换 W₃ᵀM W₃)。WingModel.W_a2s/W3。
5. **冲程中段满速起动 → 强排斥不动点**(L −15→−125→−413 即使 IQN):处方:
   **冲程顶点起动**(θ=+45°, θ̇=0,试验台真实释放点)——界面速度严格为零,
   气动载荷从静态连续生长;IC = 预平衡形整体刚转 + a=θ̈×r(与 prescribed 逐点一致)。

(短跑验证进行中;G2/G3/锚点表/kelvin 冒烟待回填。)
