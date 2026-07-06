# P2-S2: 张力场 CST 膜单元 — 实现记录(2026-07-06)

规格来源:文献锚定调研 docs/p2_s2_membrane_research.md(松弛能三分支定案)。
单元:3 节点 CST,TL,3 平动 DOF/节点,无弯曲/无转动;各向同性预张力 N0 以
2nd-PK 初应力线性能量项进入;面外刚度 100% 来自 K_G(N0)(FD 切线自动携带)。

## 本构(闭式,单位参考面积)

主 Green 应变 e1≥e2(2×2 解析谱分解),β=E/(1−ν²),κ=N0/(hβ):
- taut:W = N0(e1+e2) + ½hβ(e1²+2νe1e2+e2²);n_i = N0+hβ(e_i+νe_j)
- wrinkled(min_e2 包络,C¹):W = N0(1−ν)e1 + ½hE·e1² − ½N0²/(hβ);n1 单轴,n2=0
- slack:W = W_wr(e1_sl) 常数,e1_sl = −N0(1−ν)/(hE);n=0
- 混合判据内生于边界:taut↔wrinkled 用总力 nf2=0;wrinkled↔slack 用 n1_uni=0
  (三区交汇点解析闭合:e1=e2=e1_sl 处 nf2 恰为 0)
- **η 混合:W_η = (1−η)W_relaxed + η·W_full(η=1e-4)** — 一行同时给出:
  全局切线 ≥ η·K_taut > 0(稠密 LU 永不奇异)+ 分支边界 C¹ 保持 + slack 区
  残余压刚度(Zhang & Kiendl 2024 标定协议)

已知非光滑点(如实):e1=e2 简并(各向同性张力场的褶皱方向未定义,测度零,
max-特征值 kink)——P4 用 sqrt(D+ε²) 正则;分支边界 C¹(力连续,仅切线跳)。

## 文件

- `src/fluxvortex/warp_fsi/kernels_membrane.py` — 能量/内力(解析变分:
  H̃=Bm·N·Bmᵀ,f_g1=A0(H̃₁₁g1+H̃₁₂g2))/FD 一致切线/诊断(e1,e2,n1,n2)kernel
  + MembraneConstants(dof_map 注入=S3 梁-膜共享钩子)。
- `beam_newmark_step` 泛化为块尺寸无关(12→edofs.shape[1]),梁/膜共用稠密驱动。
- `tests/test_membrane_warp.py` — 门禁 M1-M7。

## 门禁结果(fp64, RTX 4090;Mylar h=0.05mm E=4GPa ν=0.3 ρ=1390;N0=30 N/m)

| # | 门禁 | 结果 |
|---|---|---|
| M1 | 内力==dE/dq(能量 FD,taut/wrinkled/slack 三分支受控态+显式余量) | rel-of-scale 1.3e-7 PASS |
| M2 | K(0) vs 独立解析(CST 平面应力材料阵 + K_G(N0) 几何刚度) | rel 2.1e-11 PASS |
| M3 | 刚体运动(能量不变/力随框架旋转) | 1.6e-16 / 3.8e-12 PASS |
| M4 | 预张力方膜模态 vs 解析 f_mn=½√(N0/ρh)√((m/a)²+(n/a)²) | f11 rel 4.8e-3;**f(4N0)/f(N0)=2.00000**(√N0 签名,旧壳路径失败的判别测试)PASS |
| M5 | 压力鼓起 vs 解析级数(N0∇²w=−p) | rel 4.2e-3 PASS(注:p=4Pa 时膜自张紧 ΔN≈3.6%N0,几何刚化属物理,门禁取 p=1Pa 线性区) |
| M6 | 褶皱 kink(n2 斜率 hβ→η·hβ;wrinkled n1 单轴支) | 斜率机器精度;n1 rel 3e-5 PASS |
| M7 | Newmark ring-down(模态 IC,稠密驱动) | 幅值比 0.999,频率 vs 网格特征值 rel 5.2e-4 PASS |

## 实现注记(调试中确认的事实)

1. FD 能量门禁的三个坑(已内建到测试):①近零力分量按自身归一会把 FD 噪声放大
   (改按力场标度归一);②每节点随机噪声产生的应变≫偏置应变,把单元推到分支边界/
   简并点(改受控光滑场+显式余量断言);③能量对 q 是四次的(Green²),h_fd 取 1e-8
   平衡截断/舍入。
2. 静鼓 IC 直接 release 会带入欠解析的快速面内模态(~8kHz)产生 ~7% 有界首摆过冲
   ——ring-down 门禁用纯模态 IC;FSI 冷启动同理支持 Stein 三段式预平衡的必要性。
3. 大 p 鼓起的"误差"实为几何自张紧(ΔN/N0≈3.6%@p=4Pa)——线性解析只在小载荷区
   有效,验证时先核 ΔN/N0。

## S3 待接

MembraneConstants 接受 dof_map/ndof 注入 ⇒ 膜边缘平动 DOF 别名到梁节点平动;
装配 assertion×2 与预应力模态体检见调研报告 §决策点4;冷启动序列见 §决策点3。
