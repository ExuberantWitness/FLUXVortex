# Case 4: Goland Wing 颤振 — 验证状态

## 现有实现(原 FLUXVortex,已复现 ✓)

`tests/benchmark_goland.py` + `src/fluxvortex/beam_fe.py`(Euler-Bernoulli
弯曲-扭转梁,3 DOF/节点)+ PteraSoftware UVLM,**滞后(lagged)staggered 耦合**。
本次重跑完整复现:

| V (m/s) | σ_w (1/s) | 状态 |
|---|---|---|
| 130 | −0.101 | stable |
| 140 | −0.037 | stable |
| 144 | +0.564 | FLUTTER |
| 160 | +0.794 | FLUTTER |

**颤振速度 140.2 m/s**(σ_w=0 过零插值),vs Goland & Luke (1948) 解析参考
~137 m/s,**误差 2.4%**。证据:`flap_arena/out/goland_legacy_sweep.log`。

参考值注记(诚实):137 m/s 是 Goland-Luke 经典条带理论解析值(教科书标准引用);
部分 3D UVLM 分析(SHARPy 等)给 ~163–169 m/s。差异来自气动模型维度 +
I_α 取值(本例 I_α=mc²/24=4.98,低于某些文献的 8.64)。140.2 对 137 的吻合
是在经典条带理论参考系下成立的。

## newton_pc 两遍预测-校正移植(进行中,未通过 ⚠)

为给我们这轮验证的 two-pass 耦合器补一个颤振基准,建了 `newton_pc/adapters/beam.py`
(GolandBeamEntry + BeamUVLMProvider,复用验证过的环-UVLM 核)+
`flap_arena/goland_newtonpc.py`。**当前未通过**:

- 隔离诊断:结构速度馈入气动(state 的 vels)时纯沉浮扰动**发散**
  (max|w| 0.05→3.2m);vels=0 时**衰减**(→0.015)。即运动诱导气动项产生
  反阻尼而非阻尼。
- 正负号都发散(−1 较轻),排除单纯符号翻转;疑似非定常 Bernoulli dγ/dt 项
  在移动结构子步上的处理 / 力传递与速度耦合的交互,需进一步审计。
- 这与 flap.py 的拍动场(大速度、已对 Ptera 验证)不同:Goland 是小扰动
  气弹稳定性问题,对运动诱导气动阻尼的符号/量级极敏感——颤振耦合的典型难点。

**结论**:Case 4 的物理结果(140.2 m/s)由现有 lagged 实现立住;newton_pc
two-pass 在该 case 的移植尚未验证,运动-气动阻尼耦合 bug 已隔离待修。

## 复现
```bash
python tests/benchmark_goland.py                       # 现有 lagged: 140.2 m/s
python flap_arena/goland_newtonpc.py --single 160      # WIP 移植(当前发散)
```
