# P2-S1: Warp-native 几何精确杆(Simo-Reissner)— 实现记录(2026-07-06)

规格:2 节点几何精确杆,1 高斯点缩减积分(抗剪切锁死),Crisfield-Jelenić 相对旋转插值
(常应变单元,客观性由构造保证),节点 DOF = [u(3), ψ(3)],ψ = 全局总旋转向量。
本构:N=diag(EA,κₛGA,κₛGA)Γ,M=diag(GJ,EI₂,EI₃)κ。零拟合:全部常数=管几何+材料物性。

## 文件

- `src/fluxvortex/warp_fsi/kernels_beam3d.py` — SO(3) 工具(exp/log/Jr/Jr⁻¹,级数守护)、
  闭式内力(解析变分)、能量 kernel、FD 一致切线、TubeSection、Beam3DConstants、
  beam_newmark_step(稠密批量解)。
- `platform/beam3d_solver.py` — WarpBeam3DEntry(StructuralEntry 协议 + 预定运动三环节)。
- `tests/test_beam3d_warp.py` — 门禁 1-5;`platform/p2_s1_beam3d.py` — 出口门禁 7。

## 门禁结果(fp64, RTX 4090)

| # | 门禁 | 结果 |
|---|---|---|
| 1a | 内力 == dE/dq(能量 FD,非循环) | rel 8.4e-9 PASS |
| 1b | K(0) vs 独立手推线性 1GP-Timoshenko Ke | rel 7.5e-10 PASS |
| 2 | 刚体平移+旋转 30/90/150° 零应变 | max‖Q‖/EA 1.6e-16 PASS |
| 3a | 悬臂端挠度 vs Timoshenko 解析(16 单元) | rel 1.0e-3 PASS |
| 3b | 纯弯卷弧 1/4 圆(几何精确经典基准,32 单元) | 位置 9.0e-5·L,转角 7 位 PASS |
| 4 | 模态 vs 解析(32 单元):弯 27.2Hz/扭 553.8Hz/轴 3033Hz | rel 2.1e-3 / 1.0e-4 / 1.0e-4 PASS |
| 5 | 稠密步 vs gpu_newmark_step(PCG)交叉 + ring-down 2000 步 | 幅值比 0.999,频率 rel 2.3e-3 PASS |
| 6 | ANCF 回归(nblk 参数化后) | q rel 1.4e-13 红线 PASS 不变 |
| 7 | 出口:±45°/2.3Hz 扑动 2 周期 + 重放 | 见 p2_s1_beam3d.py 输出 |

Entry 级:substep == 直接 beam_newmark_step 逐位 0.0;快照重放逐位 0.0;
预定慢转刚体跟随 2.5e-5·L。

## 与计划的两处偏差(如实)

1. **一致切线 = 对闭式内力的逐列中心差分**(fp64, h=1e-7 → ~1e-9 一致精度,自动含
   材料+几何+参数化全项,对称化后进 PCG/阻尼算子),非手推 Crisfield-Jelenić 附录闭式。
   门禁改为非循环锚:内力 vs 能量 FD(1a)+ 零应力切线 vs 独立手推线性 Ke(1b)。
   本库 Newmark 中切线只进阻尼算子(内力走 stage 平均精确进 RHS),1e-9 一致切线与
   闭式在求解器语义上不可区分。P4 可微需要的是**内力**的伴随(闭式,可微),不是切线。
2. **梁 Newmark 内层解 = 批量稠密 LU**(beam_newmark_step,两阶段算法逐行同
   gpu_newmark_step/numerical_solver,仅 S 解法不同)。原因(实测):梁切线跨
   EA/L~1e8 .. GJ/L~1e2(κ~1e6),Jacobi-PCG 不收敛(876 ms/步 @ ndof=102,CG 打满);
   稠密 LU 用已验证的 batched_dense_solve(AIC 同款)。门禁 5 对 PCG 参考单步交叉验证。
   注:gpu_newmark_step 的 nblk 参数化(36→edofs.shape[1])对 ANCF 数值恒等(回归 PASS)。

## 文档化近似(S1 已知,后续阶段升级路径)

1. **ψ 加性 Newmark 更新**:ψ̇≠ω(差 T(ψ) 因子),每步增量 ~1e-3 rad 为二阶效应;
   图卡限制 |ψ|<0.9π,Entry 每子步守卫(扑动 ±45°+弹性 ≪ 界)。纯弯卷弧只能到 <π
   (计划原文"full circle"在总旋转向量参数化下不可达,改 1/4 圆精确基准,结果 9e-5·L)。
2. **常参考系一致质量**:线性形函数平动块 + 参考系转动惯量块;大预定旋转下转动惯量
   不随框架旋转(薄壁管 ρI~9e-7,影响极小)。升级路径:per-env Me(S2/S3)。
3. **剪切模量 G12=5GPa**(拉挤碳管基体主导 4-6GPa;勿用各向同性 58GPa)。论文无扭转
   实测(eiDATA 只有弦向弯曲 k;且论文定义翼扭转抗性=弦向 EI,已被 K_MEAS 锚定,
   管自身 GJ 是次级参数)。S6 加 GJ 敏感性扫描(4/5/6GPa)。
4. **外加集中力矩**需经 F_ψ = Jr(ψ)ᵀRᵀm 映射(测试已含);气动载荷以节点力进入
   (panel_to_beam 式投影),不受此影响。

## S3 预留(已内建)

edofs 间接寻址(kernel 不假设 ndof=6nn;Beam3DConstants 接受注入 dof_map/ndof)→
梁节点平动 DOF 可别名到 ANCF 膜节点位置 DOF,ψ 独立编号 ⇒ 膜天然不传弯矩。
新增设计要求(Step-0 探针 D1 发现):柔性 FSI 冷启动需载荷 ramp-in 或静态 Newton
首平衡(见 docs/p2_step0_3mat_probe.md)。
