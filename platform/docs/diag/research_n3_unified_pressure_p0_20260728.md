# N3 统一面板压力 P0：独立动态压力观测账

日期：2026-07-28  
作用域：只建立公开压力数据的几何、相位和守恒积分身份；不运行或修改
V4.1，不拟合空间涡态，不涉及结构求解。

## 0. 裁决

```text
公开动态压力数据账：PASS
进入统一压力外部证伪：GO
FLUXV 模型比较：NOT EXECUTED
空间涡态晋升：NO-GO
生产公式修改：NO
```

P0 解决的是“压力观测是否能被无歧义地放进同一守恒账”，不是“模型是否已经
产生正确压力”。父命题 `N3.1i1d` 继续保持 open。

## 1. ① 病因定位

### 1.1 初始异常

4TU DU-97-W-300/VG v2 同时发布：

- 244 个顺序翼面坐标；
- 15 个工况的 300 相位 `Cp` mean/std；
- 同相位的 pressure-derived `Cn_unc/Ct_unc/Cm_LE_unc`。

直接把 244 点当作闭合轮廓积分时，`Cn` 到机器精度一致，但 `Ct` 最大相差
约 `0.0148`。如果把它当成“压力数据不守恒”或给模型增加弦向修正，会形成
错误病因。

### 1.2 数据指纹

坐标顺序是：

```text
upper LE(index 0) → upper TE(index 121)
lower TE(index 122) → lower LE(index 243)
```

上下尾缘之间有一条高 `0.0174c` 的竖直基底段。数据是沿上下翼面压力孔插值
的 profile pressure；README 没有声称测量尾缘基底压力。

同时计算两本账：

1. 把 `121→122` 当成有压力的闭合基底面；
2. 只积分公开 profile 的上下表面，排除该段。

结果在所有 `5 configurations × 3 frequencies = 15` 个工况上一致：第二本
账的 `Ct` RMS 均更小。典型 no-VG/f=2 工况从 `0.00709` 降到
`9.18e-5`。所以异常属于**观测几何/簿记身份**，不是气动闭合幅值。

### 1.3 挂树与可动空间

- 挂到 `N3.1i1d2`：P0 pressure observation ledger；
- 父节点 `N3.1i1d` 保持 open；
- 允许增加模型压力到该 profile 的保守投影和未调参比较；
- 禁止用 `Cp/Cn/Ct/Cm` 残差修改 `LESPcrit/cds/Tv/Tvl` 或涡片强度；
- 禁止由 pressure-only 反演唯一涡态；
- 不修改 N1/N4 或生产气动力。

## 2. ② 学科机理与数据来源

压力作用在已知表面上时，二维无量纲压力合力由轮廓线积分唯一给出。按归档
顺时针坐标与其符号约定：

```text
Cn    = -Σ Cp_mid Δx
Ct    = +Σ Cp_mid Δy
Cm_LE = +Σ Cp_mid (x_mid Δx + y_mid Δy)
```

求和只覆盖发布的上下 profile segments，不给没有压力观测的厚尾缘基底虚构
压力。这个排除是几何域身份，不是针对载荷残差选择的系数。

Sahoo、Yu 与 Ragni 的论文说明：非定常测量为 300 Hz、150 pitching cycles；
压力管路的相位延迟和幅值衰减经过修正，但非定常数据没有常规 blockage/
streamline-curvature/buoyancy 修正。论文也明确提醒，压力孔给出的是
quasi-2D chordwise pressure，不能覆盖三维分离的全部展向变化。因此该资产
适合压力输出 stress test，不适合空间涡态身份晋升。

一手来源：

- 数据 DOI：
  https://doi.org/10.4121/374c2baa-aca1-487e-8463-6ef167569be7.v2
- Sahoo, Yu & Ragni, *Wind Energy Science* 11 (2026), 1971–1988：
  https://doi.org/10.5194/wes-11-1971-2026

## 3. ③ 缺组件还是组件错误

### 3.1 已修正的错件

把上下表面 profile 坐标之间的厚尾缘连接段自动解释成“有相同 Cp 的真实
测压面”是 observation ledger 错件。它主要污染 `Ct`，而不改变竖直段
`Δx=0` 下的 `Cn`，正好解释初始指纹。

### 3.2 仍缺的组件

P0 数据侧已经可用，但完整 P-gate 仍缺：

1. 模型侧 side-resolved panel `Cp(s,t)`；
2. 从模型面板到公开 244 点 profile 的保守、无拟合投影；
3. 同一预测压力产生的 `Cn/Ct/Cm_LE`；
4. 独立于该数据的空间涡态。

此外，归档提供 cycle standard deviation，却没有完整的 Cp→force 测量与
插值不确定度传播。因此 `Ct/Cm` 的约 `1e-4/4.4e-4` 离散残差只记录，不
事后制造一个“物理通过百分比”。

## 4. ④ 有证据方案与下一门

### 4.1 已冻结 P0

`unified_pressure_observation_cases.yaml` 冻结：

- 资产 DOI、archive 身份、15 个工况和允许角色；
- 坐标顺序、厚尾缘段和压力积分公式；
- 数据输入白名单与模型调参禁令；
- `Cn` 确定性轮回门；
- `Ct/Cm` 只报告、不事后设物理容差；
- `physical_promotion=false`。

`unified_pressure_observation_guard.py` 使用 HTTP Range，只读取所需 ZIP
成员，不下载 `1.7 GB` 整包。正式结果
`unified_pressure_observation_results.json`：

```text
case_count = 15
all deterministic checks = true
max phase/AoA mismatch = 4.982700474442936e-09
max Cn round-trip error = 4.983908841182938e-10
five malformed trailing rows = deterministically discarded
base exclusion reduces Ct RMS in 15/15 cases
model_comparison_executed = false
physical_promotion = false
```

### 4.2 下一步 P1 预登记边界

下一步不是把 DU-97 数据喂给 V4.1 调参，而是建立**外部翼型压力基准适配器**：

1. 输入只含 DU-97 几何、刚性俯仰运动、来流和模型自身空间态；
2. 模型在自己的面板上先形成唯一压力账；
3. 用守恒投影输出到公开 profile；投影前后合力/力矩必须闭合；
4. 冻结 no-VG 作为首个 stress case，VG cases 只在模型显式拥有 VG 物理
   后才允许比较，不能把 VG 效应吸收到分离常数；
5. 首次比较必须报告完整 `Cp(s,phase)`、波峰位置/相位及积分载荷，不以
   总力单项选模；
6. P1 通过仍不能替代 V-gate。

在模型能产生符合该接口的 side-resolved panel pressure 之前，P1 保持
NO-RUN；不得用当前总力闭合结果伪造面板压力。

### 4.3 P1 表示审计后的范围修正

随后完成的 `research_n3_pressure_jump_observation_p1_20260728.md` 证明，上述
P1 必须再拆成：

```text
P1a：公开双侧Cp → gauge-free ΔCp/Cn，当前薄升力面可对接
P1b：Cp_upper/Cp_lower/Ct，须先有actual-thickness mean-potential/source
     与N2.6黏性双侧压力组件
```

所以“side-resolved panel pressure”仍是最终门要求，但不是当前薄面压力跳
可以直接宣称拥有的量。P0 的数据账结论不变；即时可执行的模型外验范围收窄为
`ΔCp/Cn`，完整双侧压力门保持 NO-RUN。
