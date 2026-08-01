# N2.6e1b2 结果回写

日期：2026-07-30  
实验：`n26e1b2_xm_coupled_junction_gate_20260730`  
Claim：`N2.6e1b2`  
裁决：`FALSIFIED / FROZEN`

## 实验完整性

- 正式运行严格使用执行前冻结的 v2 预登记：实际 NACA0015，
  `32/64/128 panels-per-side`，`epsilon/h=1/4,1/8,1/16`，
  `Delta t U/c=0.01`，正/负镜像和显式 no-birth 共 27 个工况。
- 27/27 工况完成；所有 Cauchy 分数有限。最大无穿透、Kelvin 和
  finite-angle Kutta 代数残差分别不超过
  `5.30e-15`、`1.46e-15` 和 `1.11e-16`；最大矩阵条件数
  `5.85e4`，远低于冻结的 `1e12` 门。
- 九个对称工况均返回无 forming segment 的显式 no-birth。
- 镜像门仅在 `p128, epsilon/h=1/4` 和 `1/8` 的束缚节点数组出现
  `2.85e-10`、`2.08e-10` 的浮点误差，超过冻结的 `1e-10`，
  因而按预登记保持 FAIL；这不是本轮 NO-GO 的主因。
- runner 不读取压力、力、Figure 12、Fig17/18/19 或任何目标响应，
  也未修改 V4.1。
- 正式 JSON 与 Markdown 的 SHA256 分别为
  `c47f99a4b68e5417ff33697880bf1c8d9fe52fccc2808ffcbd2ca30d5fcc830d`
  和
  `9e1c51a88a51ce76e53a8e1b4ef63e25c585ed973158dd98969da284ad62fd79`。

## 数据指纹

末级 `epsilon/h: 1/8 -> 1/16`，在 `p128` 的 side-1 canonical：

| 量 | 冻结 score |
|---|---:|
| upper endpoint `gamma1` | 25.70% |
| lower endpoint `gamma2` | 108.97% |
| forming strength `gamma_g` | 1.14% |
| forming circulation `Gamma_g` | 1.02% |
| bound circulation `Gamma_bound` | 1.02% |

末级 panel `64 -> 128`，固定 `epsilon/h=1/16`：

| 量 | 冻结 score |
|---|---:|
| upper endpoint `gamma1` | 8.57% |
| lower endpoint `gamma2` | 38.88% |
| forming strength `gamma_g` | 0.885% |
| forming circulation `Gamma_g` | 0.797% |
| bound circulation `Gamma_bound` | 0.797% |

因此，Kelvin 配对的积分环量和 forming-panel 强度已经进入冻结的
`2%` 区间，但两个有限角端点节点值仍强烈依赖网格和 epsilon。该局部
非收敛不能由全局环量的抵消隐藏，因为后续时间推进、IBL 和统一面板压力
需要一个定义良好的局部 junction trace。

另一个独立阻断是连续推进适用域：在每个非退化 canonical 中，当前解
恰有一侧离开 Xia--Mohseni 推导所需的 no-backflow 符号域。例如正
`6 deg` 分支的 `gamma1>0`，故固定壁面下 `u1+=gamma1>0`；镜像分支
则在另一侧违反。这不影响本轮“冻结 previous state 的空间矩阵”代数
审查，却禁止把其 current endpoints 冒充下一时步的 previous provider。

## Claim 裁决

被证伪的是：

> 在当前实际 NACA0015、线性节点束缚涡面板、epsilon 裁切端点和一个
> 常强度 forming panel 的固定预算离散下，把两个裁切端点节点强度作为
> 独立、逐点收敛的 junction 状态，即可形成可连续推进的
> Xia--Mohseni 空间 provider。

本轮判定为“**组成部分错**”，而不是“预算略小”：失败集中在有限角
端点的点值表象；同一解的积分环量已收敛，而端点值随
panel/epsilon 轴保持 `8.6%--109%` 变化。继续加密、选择某个 epsilon、
外推端点值或放宽 `2%` 门，均会把正则化选择变成隐藏模型参数。

未被证伪的是 Xia--Mohseni 的连续守恒关系、有限角 Kutta 关系或父节点
`N2.6e1b`。父节点保持 `open`；只有经一手文献授权、以弱式/奇异兼容
junction 变量取代独立端点点值的全新空间组成部分，才可另立 claim 和
预登记。

## 冻结后果

- 禁止重跑同族 `epsilon` 裁切线性节点算子，包括追加更细网格；
- 禁止用已收敛的总环量掩盖未收敛端点，再把端点送入时间推进；
- 禁止挑选 epsilon、滤波、外推、clamp、松弛或修改 Cauchy 门；
- 禁止将本空间矩阵解释为 Figure 12、Fig17/18/19 或载荷模型的验证；
- V4.1 生产路径保持不变；
- 下一步回到 Phase ②：只根据上述数据指纹检索角点奇异/弱式环量通量
  和可恢复统一面板压力的一手机理，随后只预登记一个新候选。

